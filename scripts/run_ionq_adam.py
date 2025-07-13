#!/usr/bin/env python3
"""
locally-biased-shadow energy of a 2-term ADAPT-VQE state on IonQ
================================================================
• Hadfield-style bias (weights ~ |c_P|²)  • fall-back to uniform 6-layer
• live depth and job status               • adaptive batching ≤30-min   
"""

import os, pickle, argparse, threading, time, math, numpy as np, pennylane as qml
from collections import Counter, defaultdict
from qiskit import QuantumCircuit
from qiskit_ionq import IonQProvider
from qiskit_ionq.exceptions import IonQJobFailureError

# ─── CLI ──────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument('--shots',  type=int, default=1_000, help='total physical shots')
ap.add_argument('--batch',  type=int, default=1_000,  help='initial batch size')
ap.add_argument('--uniform',action='store_true',      help='force 6-layer uniform shadow')
args = ap.parse_args()

# ─── artefacts ────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ADAPT = os.path.join(ROOT, 'results/advanced_benchmarking/adapt_vqe/adam')
raw_ops   = [ln.split(':',1)[1].strip()
             for ln in open(os.path.join(ADAPT,'selected_ops.txt'))
             if ':' in ln]
params    = np.load(os.path.join(ADAPT, 'params.npy'))
ham_of    = pickle.load(open(os.path.join(ROOT, 'inputs','bk_symm_tapered.pkl'),'rb'))
n_qubits  = max(q for term in ham_of.terms for q,_ in term)+1

assert len(raw_ops)==len(params), "params length mismatch"

# λ-string → excitation (expand if needed)
def ex_0(t): qml.SingleExcitation(t, wires=[6,5])
def ex_1(t): qml.DoubleExcitation(t, wires=[9,8,7,6])
LAM = {"<function <lambda> at 0x10cdd0540>": ex_0,
       "<function <lambda> at 0x10cdcbd80>": ex_1}
exc = [LAM[s] for s in raw_ops]

# ─── PennyLane → Qiskit (no measurement) ─────────────────────────────
dev = qml.device('default.qubit', wires=n_qubits)
@qml.qnode(dev)
def state(t):
    for θ,g in zip(t,exc): g(θ)
    for w in range(n_qubits): qml.Identity(w)
    return qml.state()

qasm = qml.workflow.construct_tape(state)(params).to_openqasm(False)
base = QuantumCircuit.from_qasm_str(qasm); base.measure_all()
print(f"RAW adaptive depth (with measure-all) : {base.depth()}")

# ─── locally-biased Clifford generator ───────────────────────────────
PAULI = {'I':0,'X':1,'Y':2,'Z':3}
weights = defaultdict(float)
for term, c in ham_of.terms.items():
    if not term: continue
    weights[tuple(sorted(term))] += float(abs(c)**2)
tot_w = sum(weights.values())
pZ = sum(w for t,w in weights.items()
         if all(op=='Z' for _,op in t))/tot_w     # bias towards Z
bias = {'Z': pZ, 'X': (1-pZ)/2, 'Y': (1-pZ)/2}    # Hadfield 2020 ①③

def sample_basis():
    return ''.join(np.random.choice(list('XYZ'), p=[bias[b] for b in 'XYZ'])
                   for _ in range(n_qubits))

def build_shadow_layer(bases:str)->QuantumCircuit:
    qc = base.copy()
    qc.remove_final_measurements(inplace=True)
    for q,b in enumerate(bases):
        if b=='X': qc.h(q)
        elif b=='Y': qc.sdg(q); qc.h(q)
    qc.measure_all()
    return qc

# test if backend accepts an X-rotation on *all* qubits
provider = IonQProvider()
backend  = provider.get_backend('simulator')
try:
    backend.run(build_shadow_layer('X'*n_qubits), shots=1).result()
    HAS_FLEX = True
except IonQJobFailureError:
    HAS_FLEX = False
    print("⚠️  backend rejects custom bases — falling back to 6-layer uniform")

# ─── execution helper ────────────────────────────────────────────────
def watch(job):
    last=None
    while True:
        s=job.status().name
        if s!=last: print(f"[IonQ] {job.job_id()[:8]} ▶ {s}"); last=s
        if s in ('DONE','CANCELLED','ERROR'): break
        time.sleep(0.4)

def run_circ(qc, shots):
    rem,b,size=shots,args.batch,args.batch
    cnt=Counter()
    while rem:
        try:
            job=backend.run(qc,shots=size)
            threading.Thread(target=watch,args=(job,),daemon=True).start()
            cnt.update(job.result().get_counts()); rem-=size
            size=min(size,rem)
        except IonQJobFailureError as e:
            if "TooLongPredictedExecutionTime" in str(e) and size>250:
                size//=2
            else: raise
    return cnt

# ─── main loop ────────────────────────────────────────────────────────
counts_tot=Counter(); est_energy=0
shot_budget=args.shots
while shot_budget>0:
    if not HAS_FLEX or args.uniform:
        # cycle through 6 deterministic layers
        for tag in ['Z','X','Y','XZ','YZ','XY']:
            if shot_budget<=0: break
            qc=build_shadow_layer({'Z':'Z'*n_qubits,
                                   'X':'X'*n_qubits,
                                   'Y':'Y'*n_qubits,
                                   'XZ':'X'*n_qubits,
                                   'YZ':'Y'*n_qubits,
                                   'XY':'X'*n_qubits}[tag])
            layer_shots=min(args.batch,shot_budget)
            print(f"\nLayer {tag:<2}  circuit depth : {qc.depth()}")
            counts_tot.update(run_circ(qc,layer_shots))
            shot_budget-=layer_shots
    else:
        bases=sample_basis()
        qc=build_shadow_layer(bases)
        print(f"\nRandom bases {bases[:8]}…  circuit depth : {qc.depth()}")
        layer_shots=min(args.batch,shot_budget)
        counts_tot.update(run_circ(qc,layer_shots))
        shot_budget-=layer_shots

print("\nTotal physical shots :", sum(counts_tot.values()))

# ─── shadow inversion & energy ────────────────────────────────────────
def inv(bit,basis,op):
    if op=='I': return 1
    if op==basis: return 3*(1-2*int(bit))
    return 0

exp=defaultdict(float); tot=sum(counts_tot.values())
for key,m in counts_tot.items():
    bits=''.join(ch for ch in str(key) if ch in '01').zfill(n_qubits)[::-1]
    for term,_ in ham_of.terms.items():
        if not term: exp[term]+=m/tot; continue
        val=1
        for w,op in term: val*=inv(bits[w], bits[w] if HAS_FLEX and not args.uniform else 'Z', op)
        exp[term]+=val*m/tot

energy=sum(float(c.real)*exp[t] for t,c in ham_of.terms.items())
print(f"\n🔬  Classical-shadow energy ≈ {energy:.8f} Ha "
      f"({sum(counts_tot.values())} shots)")