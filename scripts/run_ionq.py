#!/usr/bin/env python3
"""
Evaluate k-UpCCGSD (k = 2, 14 qubits) on IonQ *ideal* simulator
and print circuit depths at every stage.
"""

import os, pickle, numpy as np, pennylane as qml
from pennylane import qchem, workflow
from openfermion.utils import count_qubits
from qiskit_ionq import IonQProvider
from qiskit_ionq.exceptions import IonQJobFailureError
from qiskit import QuantumCircuit
from collections import Counter, defaultdict
import threading, time

# ─── Load artefacts ───────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ham_of = pickle.load(open(os.path.join(ROOT, "inputs", "bk_symm_tapered.pkl"), "rb"))
weights = np.load(os.path.join(
        ROOT, "results", "advanced_benchmarking", "adapt_vqe", "adam",
        "params.npy"))
n_qubits = count_qubits(ham_of)
hf_state = qchem.hf_state(8, n_qubits)

# ─── Ansatz (no measurement) ──────────────────────────────────────────
dev = qml.device("default.qubit", wires=n_qubits)
@qml.qnode(dev)
def ansatz(w):
    qml.kUpCCGSD(w, wires=range(n_qubits), k=w.shape[0],
                 delta_sz=0, init_state=hf_state)

base_qasm = workflow.construct_tape(ansatz)(weights).to_openqasm(False)
raw_depth = QuantumCircuit.from_qasm_str(base_qasm).depth()
print(f"RAW ansatz depth (no meas) : {raw_depth}")

# ─── Helper functions ────────────────────────────────────────────────
def simplify(term):
    parity, order = defaultdict(int), []
    for wire, op in term:
        key = (wire, op); parity[key] ^= 1
        if key not in order: order.append(key)
    return tuple(k for k in order if parity[k])

def qwc_commute(t1, t2):
    d = {w: g for w, g in t1}
    return all((d.get(w,"I") in (g,"I") or g=="I") for w,g in t2)

def group_terms_qwc(terms):
    groups=[]
    for t in terms:
        for g in groups:
            if all(qwc_commute(t, o) for o in g): g.append(t); break
        else: groups.append([t])
    return groups

def rotate_circuit(pauli_term):
    qc = QuantumCircuit.from_qasm_str(base_qasm)
    seen=set()
    for wire, op in pauli_term:
        if wire in seen: continue
        if op=="X": qc.h(wire)
        elif op=="Y": qc.sdg(wire); qc.h(wire)
        seen.add(wire)
    depth = qc.depth()
    print(f"   ↳ rotated circuit depth   : {depth}")
    return qc

def watch(job, poll=0.4):
    last=None
    while True:
        s=job.status().name
        if s!=last: print(f"[IonQ] {job.job_id()[:8]} ▶ {s}"); last=s
        if s in ("DONE","CANCELLED"): break
        if s=="ERROR":
            try: job.result()
            except IonQJobFailureError as e: print("   ↳",e); break
        time.sleep(poll)

def submit_with_adaptive_shots(qc, shots, backend, start=1000, min_batch=250):
    depth = qc.depth()
    print(f"   ↳ submitted circuit depth : {depth}")
    rem, batch, cnts = shots, start, Counter()
    while rem:
        try:
            print(f"   • submit {batch} shots (rem {rem})")
            job=backend.run(qc,shots=batch)
            threading.Thread(target=watch,args=(job,),daemon=True).start()
            cnts.update(job.result().get_counts())
            rem -= batch; batch=min(batch, rem)
        except IonQJobFailureError as e:
            if "TooLongPredictedExecutionTime" in str(e) and batch>min_batch:
                batch//=2; continue
            raise
    return cnts

# ─── Group Pauli terms ───────────────────────────────────────────────
all_terms=[simplify(tuple(k)) for k in ham_of.terms if k]
groups   = group_terms_qwc(all_terms)
print(f"Found {len(groups)} QWC groups.")

# Quick-test override (two terms, 1 000 shots) ────────────────────────
TEST_FIRST_ONLY=True
TOTAL_SHOTS=1000 if TEST_FIRST_ONLY else 10000
if TEST_FIRST_ONLY:
    groups=[[all_terms[0]],[all_terms[1]]]

# ─── Execute ─────────────────────────────────────────────────────────
backend = IonQProvider().get_backend("simulator")
global_counts={t:Counter() for t in all_terms}

for gi,grp in enumerate(groups,1):
    print(f"\n🔹 Group {gi}/{len(groups)} — {len(grp)} term(s)")
    qc_grp = rotate_circuit(grp[0])
    cnts   = submit_with_adaptive_shots(qc_grp, TOTAL_SHOTS, backend)
    for t in grp: global_counts[t].update(cnts)

# ─── Energy estimator ────────────────────────────────────────────────
def parity(bits, term):
    return (-1)**sum(int(bits[::-1][w]) for w,_ in term)

energy=0.0
for term, coeff in ham_of.terms.items():
    if not term: energy+=coeff.real; continue
    t=simplify(tuple(term)); cnts=global_counts[t]
    if not cnts: continue
    shots=sum(cnts.values())
    exp=sum(parity(b,t)*c for b,c in cnts.items())/shots
    energy+=coeff.real*exp

print(f"\n✅  Estimated energy (sampled) = {energy:.8f} Ha")
