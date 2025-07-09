#!/usr/bin/env python3
"""
Phase 5.2 – H₂ on ibm_sherbrooke
• RL-2 (TREX + ZNE)            • PennyLane auto-grouping
• Verbose prints for inspection & debugging
"""

from __future__ import annotations
import os, json, time, datetime
from pathlib import Path
import importlib.metadata as im
import numpy as np
import pennylane as qml
from qiskit_ibm_runtime import QiskitRuntimeService

# ───────────────────── 1. environment info ───────────────────────────
print("🔌  PennyLane plugins:")
for ep in im.entry_points(group="pennylane.plugins"):
    print(f"   • {ep.name}")
print()

# ───────────────────── 2. IBM backend ────────────────────────────────
service  = QiskitRuntimeService()                       # ~/.qiskit/qiskit-ibm.json
backend  = service.backend(os.getenv("IBM_QPU_DEVICE", "ibm_sherbrooke"))
print(f"🔭  Backend: {backend.name}   pending-jobs: {backend.status().pending_jobs}\n")

# ───────────────────── 3. mitigated device ───────────────────────────
SHOTS = int(os.getenv("IBM_QPU_SHOTS", "8192"))
dev   = qml.device(
    "qiskit.remote",
    wires=4,
    backend=backend,
    shots=SHOTS,
    resilience_level=2,         # RL-2 = TREX + ZNE
    optimization_level=1,
    seed_transpiler=42,
    session=backend,
)
print(f"🛠️  Device configured →  shots={SHOTS}  RL=2  opt_lvl=1\n")

# ───────────────────── 4. Hamiltonian build ──────────────────────────
symbols, coords = ["H", "H"], np.array([0, 0, 0, 0, 0, 0.74])
H_full, n_q = qml.qchem.molecular_hamiltonian(symbols, coords)
coeffs, ops = H_full.terms()
print(f"📏  Hamiltonian terms (total): {len(ops)}")

const_shift, c_noI, o_noI = 0.0, [], []
for c, o in zip(coeffs, ops):
    if isinstance(o, qml.Identity) and len(o.wires) == 0:
        const_shift += c                       # save nuclear-repulsion + core
    else:
        c_noI.append(c); o_noI.append(o)

print(f"📏  Identity-free terms      : {len(o_noI)}")
print(f"⚖️  Constant shift            : {const_shift:+.6f} Ha\n")

print("🔍  First three (coeff, op) after stripping:")
for c, o in list(zip(c_noI, o_noI))[:3]:
    print(f"     {c:+.6f}   {o}")
print()

H_noI   = qml.sum(*(c * o for c, o in zip(c_noI, o_noI)))
hf_state = qml.qchem.hf_state(2, n_q)

# ───────────────────── 5. QNode definition ───────────────────────────
@qml.qnode(dev)
def energy(theta: float = 0.0):
    qml.BasisState(hf_state, wires=range(n_q))
    qml.DoubleExcitation(theta, wires=[0, 1, 2, 3])
    return qml.expval(H_noI)            # PennyLane auto-groups internally

# ───────────────────── 6. reference energy (ideal) ───────────────────
sim = qml.device("default.qubit", wires=4)
@qml.qnode(sim)
def hf_reference():
    qml.BasisState(hf_state, wires=range(n_q))
    qml.DoubleExcitation(0.0, wires=[0, 1, 2, 3])
    return qml.expval(H_full)

print("📚  Reference HF (ideal)    :", f"{hf_reference():+.6f} Ha\n")

# ───────────────────── 7. execute on hardware ────────────────────────
theta = 0.0
print("🚀  Submitting job …")
t0 = datetime.datetime.now(datetime.UTC)
prim = energy(theta)                        # PrimitiveResult
wall = (datetime.datetime.now(datetime.UTC) - t0).total_seconds()

E_meas = prim.evs[0]
σ_meas = prim.stds[0] / np.sqrt(prim.shots)
E_total = E_meas + const_shift

print("\n🎯  Results")
print(f"     Measured ⟨Pauli⟩      : {E_meas:+.6f} ± {σ_meas:.6f} Ha")
print(f"     Constant shift        : {const_shift:+.6f} Ha")
print("     -----")
print(f"     TOTAL energy          : {E_total:+.6f} ± {σ_meas:.6f} Ha\n")
print(f"📑  Primitive shots        : {prim.shots}")
print(f"🔖  Job ID                 : {prim.job_id}")
print(f"⏱️   Wall-time              : {wall:.1f} s\n")

# ───────────────────── 8. save raw result ────────────────────────────
Path("results").mkdir(exist_ok=True)
fname = f"results/h2_{backend.name}_{int(time.time())}.json"
with open(fname, "w") as f:
    json.dump(prim.to_dict(), f, indent=2)
print("💾  PrimitiveResult saved →", fname)
