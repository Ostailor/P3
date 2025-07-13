#!/usr/bin/env python3
"""
locally-biased-shadow energy of a 2-term ADAPT-VQE state on IonQ via qBraid + Fire Opal
======================================================================================
• Hadfield-style bias (weights ~ |c_P|²)  • fall-back to uniform 6-layer
• live depth and job status               • adaptive batching ≤30-min   
• runs on qBraid’s IonQ qpu.forte-enterprise-1 with Q-CTRL Fire Opal + fallback debiasing
"""

import os
import pickle
import argparse
import threading
import time
import numpy as np
import pennylane as qml
from collections import Counter, defaultdict
from qiskit import QuantumCircuit
from qiskit.qasm2 import dumps as qasm_dumps

# ─── IMPORTS FOR qBRAID + FIRE OPAL ────────────────────────────────────
from qbraid.runtime import IonQProvider
import fireopal as fo
from qiskit_ionq.exceptions import IonQJobFailureError

# ─── CLI ──────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument('--shots',    type=int, default=500,   help='total physical shots')
ap.add_argument('--batch',    type=int, default=100,   help='initial batch size')
ap.add_argument('--uniform',  action='store_true',     help='force uniform 6-layer')
ap.add_argument('--qctrl-key', required=True,          help='your Q-CTRL Fire Opal API key')
args = ap.parse_args()

print("▶ Authenticating with Fire Opal…")
fo.authenticate_qctrl_account(api_key=args.qctrl_key)
print("✅ Fire Opal authentication successful\n")

print("▶ Initializing qBraid IonQProvider (using $IONQ_API_KEY)…")
provider = IonQProvider()

print("▶ Listing available IonQ devices…")
devices = provider.get_devices()
print(f"   → found {len(devices)} devices:")
for d in devices:
    print(f"     • {d.id}")

print("\n▶ Selecting qpu.forte-enterprise-1")
device = provider.get_device("qpu.forte-enterprise-1")
print(f"✅ Selected device: {device.id}")
print(f"   • Status: {device.status}\n")   # <-- fixed: print status attribute, not call

# ─── artefacts ────────────────────────────────────────────────────────
ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ADAPT    = os.path.join(ROOT, 'results/advanced_benchmarking/adapt_vqe/adam')
raw_ops  = [ln.split(':',1)[1].strip()
            for ln in open(os.path.join(ADAPT,'selected_ops.txt'))
            if ':' in ln]
params   = np.load(os.path.join(ADAPT, 'params.npy'))
ham_of   = pickle.load(open(os.path.join(ROOT,'inputs','bk_symm_tapered.pkl'),'rb'))
n_qubits = max(q for term in ham_of.terms for q,_ in term) + 1

assert len(raw_ops)==len(params), "params length mismatch"

# ─── λ-string → excitation ────────────────────────────────────────────
def ex_0(t): qml.SingleExcitation(t, wires=[6,5])
def ex_1(t): qml.DoubleExcitation(t, wires=[9,8,7,6])
LAM = {
    "<function <lambda> at 0x10cdd0540>": ex_0,
    "<function <lambda> at 0x10cdcbd80>": ex_1,
}
exc = [LAM[s] for s in raw_ops]

# ─── PennyLane → Qiskit (no measurement) ─────────────────────────────
print("▶ Building base QASM circuit via PennyLane…")
dev = qml.device('default.qubit', wires=n_qubits)
@qml.qnode(dev)
def state(t):
    for θ,g in zip(t,exc): g(θ)
    for w in range(n_qubits): qml.Identity(w)
    return qml.state()

qasm = qml.workflow.construct_tape(state)(params).to_openqasm(False)
base = QuantumCircuit.from_qasm_str(qasm)
base.measure_all()
print(f"✅ RAW adaptive depth (with measure-all): {base.depth()}\n")

# ─── locally-biased Clifford generator ────────────────────────────────
print("▶ Calculating locally-biased weights…")
weights = defaultdict(float)
for term, c in ham_of.terms.items():
    if term:
        weights[tuple(sorted(term))] += abs(c)**2
tot_w = sum(weights.values())
pZ    = sum(w for t,w in weights.items() if all(op=='Z' for _,op in t)) / tot_w
bias  = {'Z': pZ, 'X': (1-pZ)/2, 'Y': (1-pZ)/2}
print(f"✅ Bias computed: Z={bias['Z']:.3f}, X/Y={(1-bias['Z'])/2:.3f}\n")

def sample_basis():
    return ''.join(np.random.choice(list('XYZ'), p=[bias[b] for b in 'XYZ'])
                   for _ in range(n_qubits))

def build_shadow_layer(bases: str) -> QuantumCircuit:
    qc = base.copy()
    qc.remove_final_measurements(inplace=True)
    for q,b in enumerate(bases):
        if b=='X': qc.h(q)
        elif b=='Y': qc.sdg(q); qc.h(q)
    qc.measure_all()
    return qc

# ─── test if device accepts flexible bases ────────────────────────────
print("▶ Testing custom basis support…")
try:
    job = device.run(build_shadow_layer('X'*n_qubits), shots=1)
    job.result()
    HAS_FLEX = True
    print("✅ Device accepts custom bases\n")
except Exception as e:
    HAS_FLEX = False
    print("⚠️ Device rejects custom bases — using uniform layers")
    print(f"   • Error: {str(e)}\n")

# ─── execution helper ─────────────────────────────────────────────────
def watch(job):
    last = None
    while True:
        s = job.status()
        if s != last:
            print(f"[qBraid] {job.id()[:8]} ▶ {s}")
            last = s
        if s in ('COMPLETED','CANCELLED','FAILED'):
            break
        time.sleep(0.5)

def run_circ(qc: QuantumCircuit, shots: int) -> Counter:
    rem, size = shots, args.batch
    cnt = Counter()
    while rem > 0:
        print(f"▶ Submitting {size} shots via Fire Opal…")
        qasm_str = qasm_dumps(qc)
        try:
            fo_job = fo.execute(
                circuits=[qasm_str],
                shot_count=size,
                credentials=provider.credentials(),
                backend_name="qpu.forte-enterprise-1"
            )
            res = fo_job.result()
            counts = res["results"][0]["counts"]
            print("✅ Fire Opal job complete")
            cnt.update(counts)
        except Exception as e:
            print(f"⚠️ Fire Opal failed ({str(e)}) — falling back to direct run")
            job = device.run(qc, shots=size)
            threading.Thread(target=watch, args=(job,), daemon=True).start()
            result = job.result()
            counts = result.data.get_counts(decimal=False)
            print("✅ Direct qBraid run complete")
            cnt.update(counts)
        rem -= size
        size = min(size, rem)
        print(f"   → Shots remaining: {rem}")
    return cnt

# ─── main loop ────────────────────────────────────────────────────────
counts_tot = Counter()
shot_budget = args.shots
print(f"▶ Starting measurement loop: total shots={shot_budget}, batch={args.batch}\n")

while shot_budget > 0:
    if not HAS_FLEX or args.uniform:
        for tag in ['Z','X','Y','XZ','YZ','XY']:
            if shot_budget <= 0:
                break
            bases = {'Z':'Z'*n_qubits,'X':'X'*n_qubits,
                     'Y':'Y'*n_qubits,'XZ':'X'*n_qubits,
                     'YZ':'Y'*n_qubits,'XY':'X'*n_qubits}[tag]
            qc = build_shadow_layer(bases)
            layer_shots = min(args.batch, shot_budget)
            print(f"\n▶ Layer {tag:<2} | depth={qc.depth()} | shots={layer_shots}")
            counts_tot.update(run_circ(qc, layer_shots))
            shot_budget -= layer_shots
    else:
        bases = sample_basis()
        qc    = build_shadow_layer(bases)
        layer_shots = min(args.batch, shot_budget)
        print(f"\n▶ Random bases {bases[:8]}… | depth={qc.depth()} | shots={layer_shots}")
        counts_tot.update(run_circ(qc, layer_shots))
        shot_budget -= layer_shots

print(f"\n✅ Total physical shots collected: {sum(counts_tot.values())}\n")

# ─── shadow inversion & energy ───────────────────────────────────────
print("▶ Computing expectation values…")
def inv(bit, basis, op):
    if op == 'I': return 1
    if op == basis: return 3*(1-2*int(bit))
    return 0

exp = defaultdict(float)
tot = sum(counts_tot.values())
for key, m in counts_tot.items():
    bits = ''.join(ch for ch in str(key) if ch in '01').zfill(n_qubits)[::-1]
    for term in ham_of.terms:
        if not term:
            exp[term] += m/tot
            continue
        val = 1
        for w,op in term:
            curr_basis = bits[w] if (HAS_FLEX and not args.uniform) else 'Z'
            val *= inv(bits[w], curr_basis, op)
        exp[term] += val*m/tot

energy = sum(float(c.real)*exp[t] for t,c in ham_of.terms.items())
print(f"\n🔬 Classical-shadow energy ≈ {energy:.8f} Ha "
      f"({sum(counts_tot.values())} shots)\n")
