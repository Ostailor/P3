#!/usr/bin/env python3
"""Baseline VQE pipeline using PennyLane.

This script implements steps 3.1–3.4 of the project checklist.

Molecule: dibenzothiophene (DBT)
Basis: STO-3G
Active space: 8 electrons in 8 orbitals
Mapping: Bravyi–Kitaev with symmetry tapering
Device: lightning.qubit
Ansatz: UCCSD
Optimizer: Adam
"""

import os
import json
import pickle
import numpy as np
import pennylane as qml
from pennylane import qchem
from openfermion.utils import count_qubits
from pennylane import from_openfermion

# -------------------------- 3.1 Project setup --------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
INPUT_DIR = os.path.join(PROJECT_ROOT, 'inputs')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 123
np.random.seed(SEED)

setup_log = os.path.join(RESULTS_DIR, 'setup_log.json')
with open(setup_log, 'w') as f:
    json.dump({'seed': SEED, 'pennylane_version': qml.__version__}, f, indent=2)

# ---------------------- 3.2 Load qubit Hamiltonian ----------------------
ham_path = os.path.join(INPUT_DIR, 'bk_symm_tapered.pkl')
with open(ham_path, 'rb') as f:
    hamiltonian = pickle.load(f)

num_qubits = count_qubits(hamiltonian)
num_terms = len(hamiltonian.terms)

ham_log = os.path.join(RESULTS_DIR, 'hamiltonian_log.txt')
with open(ham_log, 'w') as f:
    f.write(f"Qubits: {num_qubits}\n")
    f.write(f"Pauli terms: {num_terms}\n")
    f.write("First few terms:\n")
    for term in list(hamiltonian.terms.items())[:5]:
        f.write(f"  {term}\n")

# ------------------ 3.3 Initial state and UCCSD ansatz ------------------

n_electrons = 8
hf_state = qchem.hf_state(n_electrons, num_qubits).astype(int)
singles, doubles = qchem.excitations(n_electrons, num_qubits)
s_wires, d_wires = qchem.excitations_to_wires(singles, doubles)

num_params = len(s_wires) + len(d_wires)

def uccsd_theta(params):
    return qml.UCCSD(
        params,
        wires=range(num_qubits),
        s_wires=s_wires,
        d_wires=d_wires,
        init_state=hf_state,  # <-- Explicit!
    )

ansatz_log = os.path.join(RESULTS_DIR, 'ansatz_log.txt')
with open(ansatz_log, 'w') as f:
    f.write(f"Number of variational parameters: {num_params}\n")

# ----------------------- 3.4 Device and cost fn ------------------------

device = qml.device('lightning.qubit', wires=num_qubits)
obs = from_openfermion(hamiltonian, wires=list(range(num_qubits)))


@qml.qnode(device)
def circuit(params):
    qml.BasisState(hf_state, wires=range(num_qubits))
    uccsd_theta(params)
    return qml.expval(obs)

cost_fn = circuit

init_params = np.zeros(num_params)
initial_energy = cost_fn(init_params)

with open(os.path.join(RESULTS_DIR, 'initial_energy.txt'), 'w') as f:
    f.write(f"Initial energy: {initial_energy}\n")

# --------------------- (Optional) Optimizer Loop ------------------------
# Uncomment this section if you want a simple optimization run

# from pennylane.optimize import AdamOptimizer
# max_steps = 200
# params = np.zeros(num_params, dtype=float)
# opt = AdamOptimizer(stepsize=0.05)
# energy_progress = []

# for n in range(max_steps):
#     params, energy = opt.step_and_cost(cost_fn, params)
#     energy_progress.append(energy)
#     if n % 10 == 0:
#         print(f"Step {n:03d} | Energy = {energy:.8f} Ha")

# # Save optimizer trace
# with open(os.path.join(RESULTS_DIR, 'energy_progress.npy'), 'wb') as f:
#     np.save(f, np.array(energy_progress))
# with open(os.path.join(RESULTS_DIR, 'final_energy.txt'), 'w') as f:
#     f.write(f"Final energy: {energy_progress[-1]}\n")
