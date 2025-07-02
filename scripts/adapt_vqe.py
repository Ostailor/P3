#!/usr/bin/env python3
"""ADAPT-VQE for dibenzothiophene (DBT).

Usage: ``python adapt_vqe.py``

Inputs
------
* ``../inputs/bk_symm_tapered.pkl``: pickled OpenFermion QubitOperator

Outputs
-------
* ``../results/advanced_benchmarking/adapt_vqe/<optimizer>/`` directory
  containing energy and parameter logs, selected operators, and a convergence
  plot for each optimizer run.

This script implements ADAPT-VQE using PennyLane only. For each optimizer
(Adam, SPSA, COBYLA) the procedure is:

1. Build the operator pool (singles and doubles).
2. Iteratively select operators with the largest gradient until the gradient
   threshold or maximum pool iterations is reached.
3. After each operator selection, optimize all parameters with the chosen
   optimizer.
4. Log energies, selected operators, circuit depth, and runtime.

Author: open collaboration
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


# ---------------------- Paths and directories ----------------------
HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
INPUT_DIR = os.path.join(ROOT, 'inputs')
BASE_RESULTS = os.path.join(ROOT, 'results', 'advanced_benchmarking', 'adapt_vqe')
os.makedirs(BASE_RESULTS, exist_ok=True)

# ---------------------- Load Hamiltonian ---------------------------
with open(os.path.join(INPUT_DIR, 'bk_symm_tapered.pkl'), 'rb') as f:
    of_ham = pickle.load(f)
num_qubits = count_qubits(of_ham)


class COBYLAOptimizer:
    def __init__(self, maxiter=100, rhobeg=1.0, tol=None, **options):
        self.options = {"maxiter": maxiter, "rhobeg": rhobeg, "tol": tol}
        self.options.update(options)

    def step_and_cost(self, cost, params):
        # cost: function R^n -> scalar
        res = minimize(lambda x: float(cost(x)), params, method="COBYLA", options=self.options)
        print(f"COBYLA step: params={res.x}, energy={res.fun}")
        return res.x, res.fun

def of_to_pl(op):
    """Convert an OpenFermion QubitOperator into a PennyLane Hamiltonian."""
    coeffs = []
    obs = []

    for term, coeff in op.terms.items():
        coeffs.append(float(coeff.real))
        if term == ():
            # Identity term (no Pauli string)
            obs.append(qml.Identity(0))
        else:
            # Build single-qubit Pauli ops
            paulis = [getattr(qml, f"Pauli{g}")(w) for w, g in term]
            if len(paulis) == 1:
                obs.append(paulis[0])
            else:
                # Option A: use qml.prod
                obs.append(qml.prod(*paulis))
                # Option B: use repeated @
                # composite = reduce(operator.matmul, paulis)
                # obs.append(composite)

    return qml.Hamiltonian(coeffs, obs)

hamiltonian = of_to_pl(of_ham)

# ---------------------- Reference state & pool ---------------------
ELECTRONS = 8
hf_state = qchem.hf_state(ELECTRONS, num_qubits)

# generate excitation lists
singles, doubles = qchem.excitations(ELECTRONS, num_qubits)
singles_wires, doubles_wires = qchem.excitations_to_wires(singles, doubles, wires=range(num_qubits))

print("Example singles_wires:", singles_wires[:5])
print("Example doubles_wires:", doubles_wires[:2])


# build operator pool and labels
operator_pool = []
operator_labels = []

# Single excitations
for wire_seq in singles_wires:
    operator_pool.append(
        lambda theta, wires=wire_seq: FermionicSingleExcitation(theta, wires=wires)
    )
    operator_labels.append(f"FermionicSingle{wire_seq}")

# Double excitations
for wires1, wires2 in doubles_wires:
    operator_pool.append(
        lambda theta, w1=wires1, w2=wires2: FermionicDoubleExcitation(
            theta, wires1=w1, wires2=w2
        )
    )
    operator_labels.append(f"FermionicDouble{wires1}+{wires2}")


# ---------------------- Device and helper functions ----------------
DEV = qml.device('lightning.qubit', wires=num_qubits)


def make_ansatz(ops):
    def ansatz(params):
        qml.BasisState(hf_state, wires=range(num_qubits))
        for t, op in zip(params, ops):
            op(t)
    return ansatz


def energy(params, ops):
    ans = make_ansatz(ops)

    @qml.qnode(DEV)
    def circuit(p):
        ans(p)
        return qml.expval(hamiltonian)

    return circuit(params)


# ---------------------- ADAPT-VQE Loop -----------------------------

def adapt_vqe(opt_name, optimizer, max_pool=10, grad_thresh=1e-10, opt_steps=200):
    print(f"\nStarting ADAPT-VQE with optimizer: {opt_name}")
    res_dir = os.path.join(BASE_RESULTS, opt_name)
    os.makedirs(res_dir, exist_ok=True)

    selected_ops = []
    params = pnp.array([], requires_grad=True)
    energies = []
    depths = []
    selected_labels = []
    start = time.perf_counter()

    for it in range(max_pool):
        print(f"\nIteration {it+1}/{max_pool}")
        # Compute gradients for all candidate operators
        grads = []
        for op in operator_pool:
            trial_ops = selected_ops + [op]
            trial_params = pnp.append(params, 0.0)
            cost = lambda p: energy(p, trial_ops)
            grad = qml.grad(cost)(trial_params)[-1]
            grads.append(pnp.abs(grad))
        #print("  all grads:", [float(g) for g in grads])

        max_grad = float(pnp.max(pnp.array(grads)))
        best_idx = int(pnp.argmax(pnp.array(grads)))
        print(f"  Max gradient = {max_grad:.6e} at index {best_idx}")
        if max_grad < grad_thresh:
            print("  Gradient threshold reached; stopping.")
            break

        # Select operator with largest gradient
        selected_ops.append(operator_pool.pop(best_idx))
        selected_labels.append(str(selected_ops[-1]))
        params = pnp.append(params, 0.0)
        if opt_name == 'adam':
            optimizer = qml.AdamOptimizer(stepsize=optimizer.stepsize)
        elif opt_name == 'spsa':
            optimizer = qml.SPSAOptimizer(maxiter=optimizer.maxiter,
                                        a=optimizer.a, c=optimizer.c)
        print(f"  Selected operator: index {best_idx}, total ops = {len(selected_ops)}")

        # Optimize all parameters with chosen optimizer
        cost = lambda p: pnp.real(energy(p, selected_ops))
        for step in range(opt_steps):
            params, e = optimizer.step_and_cost(cost, params)
            energies.append(float(e))
            print(f"    Opt step {step+1}/{opt_steps}: Energy = {e:.6e}")

        # Build a QNode for the current ansatz
        qnode = qml.QNode(make_ansatz(selected_ops), DEV)

        # Create a specs function that returns resource info (including depth)
        specs_fn = qml.specs(qnode)

        # Call it with your optimized params to get the 'depth' field
        specs = specs_fn(params)
        depth = specs['resources'].depth
        depths.append(depth)
        print(f"  Circuit depth after optimization = {depth}")

    runtime = time.perf_counter() - start
    if energies:
        print(f"Finished {opt_name} in {runtime:.2f}s, final energy = {energies[-1]:.6e}")
    else:
        print(f"Finished {opt_name} in {runtime:.2f}s, no energies recorded as gradient less than threshold (1e-10) on first iteration.")

    pnp.save(os.path.join(res_dir, 'energy_history.npy'), pnp.array(energies))
    pnp.save(os.path.join(res_dir, 'params.npy'), pnp.array(params))

    with open(os.path.join(res_dir, 'selected_ops.txt'), 'w') as f:
        for idx, lbl in enumerate(selected_labels):
            f.write(f'{idx}: {lbl}\n')

    with open(os.path.join(res_dir, 'log.json'), 'w') as f:
        json.dump({
            'optimizer': opt_name,
            'iterations': len(energies),
            'final_energy': energies[-1] if energies else None,
            'circuit_depth': depths[-1] if depths else None,
            'runtime_sec': runtime,
        }, f, indent=2)

    try:
        import matplotlib.pyplot as plt

        plt.plot(range(len(energies)), energies, marker='o', ms=3)
        plt.xlabel('Iteration')
        plt.ylabel('Energy (Ha)')
        plt.title(f'ADAPT-VQE convergence ({opt_name})')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(res_dir, 'convergence.png'))
        plt.close()
    except Exception:
        print("Failed to plot convergence graph. Ensure matplotlib is installed.")


if __name__ == '__main__':
    optimizers = {
        'adam': qml.AdamOptimizer(stepsize=0.05),
        'spsa':  qml.SPSAOptimizer(maxiter=50, a=0.05, c=0.02),
        'cobyla': COBYLAOptimizer(maxiter=200, rhobeg=0.5, tol=1e-6),
    }

    for name, opt in optimizers.items():
        adapt_vqe(name, opt)