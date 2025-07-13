#!/usr/bin/env python3
"""ADAPT-VQE for dibenzothiophene (DBT).

Usage: python adapt_vqe.py

Inputs
------
* ../inputs/bk_symm_tapered.pkl: pickled OpenFermion QubitOperator

Outputs
-------
* results/advanced_benchmarking/adapt_vqe/<optimizer>/:
    - energy_history.npy
    - params.npy
    - selected_ops.txt
    - log.json
    - convergence.png (if matplotlib is installed)

Procedure (for each optimizer: adam, spsa, cobyla):
1. Build singles & doubles pool.
2. Iteratively pick operator with largest gradient (above threshold).
3. Re-optimize all parameters.
4. Log everything.
"""

import os
import json
import time
import pickle
import pennylane as qml
import pennylane.numpy as pnp
from pennylane import qchem
from pennylane import FermionicSingleExcitation, FermionicDoubleExcitation
from openfermion.utils import count_qubits
from scipy.optimize import minimize

# ─── Paths ─────────────────────────────────────────────────────────
HERE         = os.path.abspath(os.path.dirname(__file__))
ROOT         = os.path.abspath(os.path.join(HERE, ".."))
INPUT_DIR    = os.path.join(ROOT, "inputs")
BASE_RESULTS = os.path.join(ROOT, "results", "advanced_benchmarking", "adapt_vqe")
os.makedirs(BASE_RESULTS, exist_ok=True)

# ─── Load Hamiltonian ──────────────────────────────────────────────
with open(os.path.join(INPUT_DIR, "bk_symm_tapered.pkl"), "rb") as f:
    of_ham = pickle.load(f)
num_qubits = count_qubits(of_ham)

# ─── Optimizer wrappers ───────────────────────────────────────────
class COBYLAOptimizer:
    def __init__(self, maxiter=100, rhobeg=1.0, tol=None, **options):
        self.options = {"maxiter": maxiter, "rhobeg": rhobeg, "tol": tol}
        self.options.update(options)

    def step_and_cost(self, cost, params):
        res = minimize(lambda x: float(cost(x)), params,
                       method="COBYLA", options=self.options)
        print(f"COBYLA step: params={res.x}, energy={res.fun}")
        return res.x, res.fun

# ─── Convert to PennyLane Hamiltonian ─────────────────────────────
def of_to_pl(op):
    coeffs = []
    obs    = []
    for term, coeff in op.terms.items():
        coeffs.append(float(coeff.real))
        if term == ():
            obs.append(qml.Identity(0))
        else:
            paulis = [getattr(qml, f"Pauli{g}")(w) for w, g in term]
            obs.append(paulis[0] if len(paulis) == 1 else qml.prod(*paulis))
    return qml.Hamiltonian(coeffs, obs)

hamiltonian = of_to_pl(of_ham)

# ─── Reference state & master pool ───────────────────────────────
ELECTRONS = 8
hf_state  = qchem.hf_state(ELECTRONS, num_qubits)

# build pool of PennyLane‐callable excitations
singles, doubles             = qchem.excitations(ELECTRONS, num_qubits)
singles_wires, doubles_wires = qchem.excitations_to_wires(singles, doubles, wires=range(num_qubits))

MASTER_POOL   = []
MASTER_LABELS = []

for wire_seq in singles_wires:
    MASTER_POOL.append(
        lambda theta, wires=wire_seq: FermionicSingleExcitation(theta, wires=wires)
    )
    MASTER_LABELS.append(f"FermionicSingle{wire_seq}")

for w1, w2 in doubles_wires:
    MASTER_POOL.append(
        lambda theta, a=w1, b=w2: FermionicDoubleExcitation(
            theta, wires1=a, wires2=b
        )
    )
    MASTER_LABELS.append(f"FermionicDouble{w1}+{w2}")

# ─── Device & helper funcs ────────────────────────────────────────
DEV = qml.device("lightning.qubit", wires=num_qubits)

def make_ansatz(ops):
    def ansatz(params):
        qml.BasisState(hf_state, wires=range(num_qubits))
        for t, op in zip(params, ops):
            op(t)
    return ansatz

def energy(params, ops):
    """Builds a QNode that returns ⟨hamiltonian⟩ for the given ansatz."""
    @qml.qnode(DEV)
    def circuit(p):
        # prepare and excite
        qml.BasisState(hf_state, wires=range(num_qubits))
        for angle, op in zip(p, ops):
            op(angle)
        # measure energy
        return qml.expval(hamiltonian)

    return circuit(params)

# ─── Core ADAPT-VQE routine ────────────────────────────────────────
def adapt_vqe(opt_name, optimizer, max_pool=10, grad_thresh=1e-10, opt_steps=200):
    print(f"\n=== ADAPT-VQE with {opt_name} ===")
    res_dir = os.path.join(BASE_RESULTS, opt_name)
    os.makedirs(res_dir, exist_ok=True)

    # fresh local copies of the pool
    local_pool   = MASTER_POOL.copy()
    local_labels = MASTER_LABELS.copy()

    selected_ops    = []
    selected_labels = []
    params          = pnp.array([], requires_grad=True)
    energies        = []
    depths          = []

    start = time.perf_counter()
    for it in range(max_pool):
        print(f"\n-- Iteration {it+1}/{max_pool}")
        # 1) compute gradients
        grads = []
        for op in local_pool:
            trial_ops    = selected_ops + [op]
            trial_params = pnp.append(params, 0.0)
            grad_val     = qml.grad(lambda p: energy(p, trial_ops))(trial_params)[-1]
            grads.append(pnp.abs(grad_val))

        max_grad = float(pnp.max(grads))
        best_idx = int(pnp.argmax(grads))
        print(f"  max gradient = {max_grad:.3e} at idx {best_idx}")

        if max_grad < grad_thresh:
            print("  gradient below threshold; stopping selection.")
            break

        # 2) select and remove from pool
        selected_ops.append(local_pool.pop(best_idx))
        selected_labels.append(local_labels.pop(best_idx))
        params = pnp.append(params, 0.0)
        print(f"  selected {selected_labels[-1]} (total ops={len(selected_ops)})")

        # 3) re-instantiate only Adam (to reset its state), reuse SPSA/COBYLA as-is
        if opt_name == "adam":
            optimizer = qml.AdamOptimizer(stepsize=optimizer.stepsize)

        # 4) optimize all parameters
        cost_real = lambda p: pnp.real(energy(p, selected_ops))
        for step in (1, opt_steps//2, opt_steps):
            params, e = optimizer.step_and_cost(cost_real, params)
            energies.append(float(e))
            print(f"    step {step}/{opt_steps}: E = {e:.6e}")

        # 5) record circuit depth
        specs = qml.specs(qml.QNode(make_ansatz(selected_ops), DEV))(params)
        depth = specs["resources"].depth
        depths.append(depth)
        print(f"  circuit depth = {depth}")

    runtime = time.perf_counter() - start
    final_e = energies[-1] if energies else None
    print(f"\n{opt_name} done in {runtime:.1f}s; final E = {final_e:.6e}")

    # ─── save logs ───────────────────────────────────────────────
    pnp.save(os.path.join(res_dir, "energy_history.npy"), pnp.array(energies))
    pnp.save(os.path.join(res_dir, "params.npy"), pnp.array(params))

    with open(os.path.join(res_dir, "selected_ops.txt"), "w") as f:
        for i, lbl in enumerate(selected_labels):
            f.write(f"{i}: {lbl}\n")

    with open(os.path.join(res_dir, "log.json"), "w") as f:
        json.dump({
            "optimizer":    opt_name,
            "iterations":   len(energies),
            "final_energy": final_e,
            "circuit_depth": depths[-1] if depths else None,
            "runtime_sec":  runtime,
        }, f, indent=2)

    # optional convergence plot
    try:
        import matplotlib.pyplot as plt
        plt.plot(energies, marker="o", ms=3)
        plt.xlabel("Optimization step")
        plt.ylabel("Energy (Ha)")
        plt.title(f"Convergence ({opt_name})")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(res_dir, "convergence.png"))
        plt.close()
    except ImportError:
        pass

# ─── Entry point ─────────────────────────────────────────────────
if __name__ == "__main__":
    optimizers = {
        "adam":   qml.AdamOptimizer(stepsize=0.05),
        "spsa":   qml.SPSAOptimizer(maxiter=50, a=0.05, c=0.02),
        "cobyla": COBYLAOptimizer(maxiter=200, rhobeg=0.5, tol=1e-6),
    }

    for name, opt in optimizers.items():
        adapt_vqe(name, opt)
