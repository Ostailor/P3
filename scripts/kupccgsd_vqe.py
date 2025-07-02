#!/usr/bin/env python3
"""k-UpCCGSD VQE for dibenzothiophene (DBT).

Usage: ``python kupccgsd_vqe.py``

Inputs
------
* ``../inputs/bk_symm_tapered.pkl``: pickled OpenFermion QubitOperator

Outputs
-------
* ``../results/advanced_benchmarking/kupccgsd_vqe/<optimizer>/`` directory
  containing energy histories, optimized parameters, and convergence plots for
  each optimizer run.

The ansatz uses PennyLane's ``kUpCCGSD`` template. Here ``k`` denotes the
number of repetitions; we set ``k = 2`` in this script. For each optimizer
(Adam, SPSA, COBYLA) we record the energy at each iteration, circuit depth, and
runtime.
"""

import os
import json
import time
import pickle
import pennylane as qml
import pennylane.numpy as pnp
from pennylane import qchem
from openfermion.utils import count_qubits
from scipy.optimize import minimize

# ---------------------- Paths and directories ----------------------
HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
INPUT_DIR = os.path.join(ROOT, 'inputs')
BASE_RESULTS = os.path.join(ROOT, 'results', 'advanced_benchmarking', 'kupccgsd_vqe')
os.makedirs(BASE_RESULTS, exist_ok=True)
IMPROVEMENT_THRESHOLD = 1e-6

# ---------------------- Load Hamiltonian ---------------------------
with open(os.path.join(INPUT_DIR, 'bk_symm_tapered.pkl'), 'rb') as f:
    of_ham = pickle.load(f)
num_qubits = count_qubits(of_ham)

class COBYLAOptimizer:
    def __init__(self, maxiter=100, rhobeg=1.0, tol=None, **options):
        self.options = {"maxiter": maxiter, "rhobeg": rhobeg, "tol": tol}
        self.options.update(options)

    def step_and_cost(self, cost, params):
        # Remember original shape
        orig_shape = params.shape

        # Flatten initial guess
        x0 = params.reshape(-1)

        # Wrapper: takes 1D vector, reshapes, calls cost
        def flat_cost(x_flat):
            weights = x_flat.reshape(orig_shape)
            return float(cost(weights))

        # Run SciPy minimize on the flattened vector
        res = minimize(flat_cost, x0, method="COBYLA", options=self.options)

        # Reshape optimized vector back into param_shape
        new_params = res.x.reshape(orig_shape)
        new_energy = res.fun
        print(f"COBYLA: Optimized energy = {new_energy:.6e}, ")

        return new_params, new_energy



def of_to_pl(op):
    coeffs, obs = [], []
    for term, coeff in op.terms.items():
        coeffs.append(float(coeff.real))
        if term == ():
            obs.append(qml.Identity(0))
        else:
            paulis = [getattr(qml, f"Pauli{g}")(w) for w, g in term]
            obs.append(qml.prod(*paulis) if len(paulis) > 1 else paulis[0])
    return qml.Hamiltonian(coeffs, obs)

hamiltonian = of_to_pl(of_ham)

# ---------------------- Ansatz parameters -------------------------
ELECTRONS = 8
K_REPS = 2  # number of UpCCGSD repetitions
hf_state = qchem.hf_state(ELECTRONS, num_qubits)
param_shape = qml.kUpCCGSD.shape(
    k=K_REPS,
    n_wires=num_qubits,
    delta_sz=0
)

# ---------------------- Device and circuit ------------------------
DEV = qml.device('lightning.qubit', wires=num_qubits)


@qml.qnode(DEV)
def circuit(weights):
    qml.kUpCCGSD(
        weights,
        wires=range(num_qubits),
        k=K_REPS,
        delta_sz=0,
        init_state=hf_state
    )
    return qml.expval(hamiltonian)


def cost_fn(weights):
    return pnp.real(circuit(weights))


# ---------------------- Optimization helper -----------------------

def run_vqe(opt_name, optimizer, max_steps=200, min_improvement=1e-6):
    res_dir = os.path.join(BASE_RESULTS, opt_name)
    os.makedirs(res_dir, exist_ok=True)

    params = pnp.random.normal(0, 1e-2, param_shape, requires_grad=True)
    energy_hist = []
    start = time.perf_counter()
    prev_energy = float('inf')

    for step in range(max_steps):
        params, energy = optimizer.step_and_cost(cost_fn, params)
        energy_hist.append(float(energy))

        improvement = prev_energy - energy
        if improvement < IMPROVEMENT_THRESHOLD:
            print(f"Stopping at step {step} — improvement {improvement:.2e} < {IMPROVEMENT_THRESHOLD:.2e}")
            break
            
        prev_energy = energy

    runtime = time.perf_counter() - start
    specs_fn = qml.specs(circuit)
    info = specs_fn(params)
    resources = info["resources"]
    depth = resources.depth
    print(f"Finished {opt_name} in {runtime:.2f}s, final energy = {energy_hist[-1]:.6e}, circuit depth = {depth}")


    pnp.save(os.path.join(res_dir, 'energy_history.npy'), pnp.array(energy_hist))
    pnp.save(os.path.join(res_dir, 'final_params.npy'), pnp.array(params))

    with open(os.path.join(res_dir, 'log.json'), 'w') as f:
        json.dump({
            'optimizer': opt_name,
            'k': K_REPS,
            'num_parameters': int(pnp.prod(param_shape)),
            'iterations': len(energy_hist),
            'final_energy': energy_hist[-1],
            'circuit_depth': int(depth),
            'runtime_sec': runtime,
        }, f, indent=2)

    try:
        import matplotlib.pyplot as plt

        plt.plot(range(len(energy_hist)), energy_hist, marker='o', ms=3)
        plt.xlabel('Iteration')
        plt.ylabel('Energy (Ha)')
        plt.title(f'k-UpCCGSD convergence ({opt_name})')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(res_dir, 'convergence.png'))
        plt.close()
    except Exception:
        pass


if __name__ == '__main__':
    n_vars = int(pnp.prod(param_shape))
    min_maxiter = n_vars + 2
    optimizers = {
        'cobyla': COBYLAOptimizer(maxiter=min_maxiter, rhobeg=0.5, tol=1e-6),
    }

    for name, opt in optimizers.items():
        run_vqe(name, opt)