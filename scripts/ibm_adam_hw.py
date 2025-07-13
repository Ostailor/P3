#!/usr/bin/env python3
"""
Classical‐shadow energy of a 2‐term ADAPT‐VQE state using PennyLane‐Qiskit.
Supports local simulation (Aer) and real QPU runs via Fire Opal’s direct SDK,
in true batch mode (6 circuits per Fire Opal job), without qiskit-ibm-catalog.
"""

import os
import pickle
import argparse
import time
import numpy as np
import pennylane as qml

from collections import Counter, defaultdict
from dotenv import load_dotenv

from qiskit import QuantumCircuit, transpile
from qiskit.qasm2 import dumps                                                        # Qiskit 1.0+ exporter
from qiskit_aer import AerSimulator                                                  # Aer noise‐model mirroring

import fireopal as fo                                                                # Fire Opal direct SDK
from fireopal.credentials import make_credentials_for_ibmq                            # IBMQ creds helper

# ─── Load env vars & parse CLI ──────────────────────────────────
load_dotenv()
parser = argparse.ArgumentParser()
parser.add_argument('--shots',     type=int,      default=10000, help='total shots')
parser.add_argument('--batch',     type=int,      default=1000,  help='shots per circuit')
parser.add_argument('--uniform',   action='store_true',        help='use uniform layers')
parser.add_argument('--simulator', action='store_true',        help='use AerSimulator')
args = parser.parse_args()

# ─── Read API keys & backend name ───────────────────────────────
QCTRL_API_KEY = os.getenv("QCTRL_API_KEY")
IBM_TOKEN     = os.getenv("QISKIT_IBM_TOKEN")
BACKEND_NAME  = os.getenv("QISKIT_IBM_BACKEND", "ibm_brisbane")  # ensure defined
if not (QCTRL_API_KEY and IBM_TOKEN):
    raise RuntimeError("Please set QCTRL_API_KEY & QISKIT_IBM_TOKEN")

# ─── Authenticate Fire Opal & build IBMQ credentials ───────────
fo.authenticate_qctrl_account(api_key=QCTRL_API_KEY)
credentials = make_credentials_for_ibmq(
    token   = IBM_TOKEN,
    hub     = "ibm-q",
    group   = "open",
    project = "main"
)

# ─── Load ADAPT‐VQE artifacts ────────────────────────────────────
ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ADAPT_DIR = os.path.join(ROOT, 'results/advanced_benchmarking/adapt_vqe/adam')
raw_ops   = [
    ln.split(':',1)[1].strip()
    for ln in open(os.path.join(ADAPT_DIR,'selected_ops.txt'))
    if ':' in ln
]
params = np.load(os.path.join(ADAPT_DIR, 'params.npy'))
ham_of = pickle.load(open(os.path.join(ROOT,'inputs','bk_symm_tapered.pkl'),'rb'))
n_qubits = max(q for term in ham_of.terms for q,_ in term) + 1

# ─── Map λ‐strings → PennyLane excitations ──────────────────────
def ex_0(t): qml.SingleExcitation(t, wires=[6,5])
def ex_1(t): qml.DoubleExcitation(t, wires=[9,8,7,6])
LAM = {
    "<function <lambda> at 0x10cdd0540>": ex_0,
    "<function <lambda> at 0x10cdcbd80>": ex_1,
}
exc = [LAM[s] for s in raw_ops]

# ─── Build base QASM circuit via PennyLane ──────────────────────
dev = qml.device('default.qubit', wires=n_qubits)
@qml.qnode(dev)
def state_qnode(t):
    for θ, g in zip(t, exc): g(θ)
    for w in range(n_qubits): qml.Identity(w)
    return qml.state()

qasm    = qml.workflow.construct_tape(state_qnode)(params).to_openqasm(False)
base_qc = QuantumCircuit.from_qasm_str(qasm)
base_qc.measure_all()
print(f"RAW adaptive depth (measure-all): {base_qc.depth()}")

# ─── Prepare Pauli‐bias for classical shadows ────────────────────
weights = defaultdict(float)
for term, c in ham_of.terms.items():
    if term:
        weights[tuple(sorted(term))] += abs(c)**2
tot_w = sum(weights.values())
pZ    = sum(w for t,w in weights.items() if all(op=='Z' for _,op in t)) / tot_w
bias  = {'Z': pZ, 'X': (1-pZ)/2, 'Y': (1-pZ)/2}

def sample_basis():
    return ''.join(
        np.random.choice(list('XYZ'), p=[bias[b] for b in 'XYZ'])
        for _ in range(n_qubits)
    )

def build_shadow_layer(bases: str):
    qc = base_qc.copy()
    qc.remove_final_measurements(inplace=True)
    for q,b in enumerate(bases):
        if b=='X':
            qc.h(q)
        elif b=='Y':
            qc.sdg(q); qc.h(q)
    qc.measure_all()
    return qc

# ─── Batch executor for 6 circuits ──────────────────────────────
def run_batch(qcs, shots):
    if args.simulator:
        sim = AerSimulator()
        out = []
        for qc in qcs:
            qt = transpile(qc, sim)
            out.append(sim.run(qt, shots=shots).result().get_counts(qt))
        return out

    # hardware via Fire Opal
    qasm_list = [dumps(qc) for qc in qcs]
    job       = fo.iterate(
        circuits     = qasm_list,
        shot_count   = shots,
        credentials  = credentials,
        backend_name = BACKEND_NAME
    )  # no 'channel' arg :contentReference[oaicite:2]{index=2}

    while True:  # poll until done
        st = job.status()
        print(f"[FireOpal] {st['action_status']}: {st['status_message']}")
        if st['action_status'] in ("SUCCESS","FAILURE","REVOKED"):
            break
        time.sleep(0.5)

    res = job.result()
    job.stop_iterate()  # tear down the session immediately :contentReference[oaicite:3]{index=3}
    return [dict(e) for e in res.get("results", [])]

# ─── Main loop & energy reconstruction ──────────────────────────
counts_tot  = Counter()
shot_budget = args.shots

while shot_budget > 0:
    if args.uniform:
        tags = ['Z','X','Y','XZ','YZ','XY']
        qcs  = [build_shadow_layer(t if len(t)==1 else t[0]*n_qubits) for t in tags]
    else:
        bases_list = [sample_basis() for _ in range(6)]
        qcs        = [build_shadow_layer(b) for b in bases_list]

    m   = len(qcs)
    per = min(args.batch, shot_budget // m)
    if per == 0:
        break

    print(f"\nBatch of {m} circuits × {per} shots each …")
    for counts in run_batch(qcs, per):
        counts_tot.update(counts)
    shot_budget -= per * m

# ─── Inversion & final energy calc ──────────────────────────────
def inv(bit, basis, op):
    if op=='I': return 1
    if op==basis: return 3*(1-2*int(bit))
    return 0

exp_vals = defaultdict(float)
total    = sum(counts_tot.values())
for key,cnt in counts_tot.items():
    bitstr = "".join(key) if isinstance(key,(tuple,list)) else str(key)
    bits   = bitstr.zfill(n_qubits)[::-1]
    for term, coeff in ham_of.terms.items():
        if not term:
            exp_vals[term] += cnt/total
        else:
            val = 1
            for w,op in term:
                val *= inv(bits[w], bits[w] if not args.uniform else 'Z', op)
            exp_vals[term] += val * (cnt/total)

energy = sum(float(c.real)*exp_vals[t] for t,c in ham_of.terms.items())
print(f"\n🔬 Classical-shadow energy ≈ {energy:.8f} Ha ({total} shots)")
