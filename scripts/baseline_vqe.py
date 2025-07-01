#!/usr/bin/env python3
"""Baseline VQE setup (Steps 3.1-3.5).

This script prepares the inputs for a VQE calculation using PennyLane.
It performs the following actions:
    * Loads the tapered Bravyi–Kitaev Hamiltonian produced in phase 2.
    * Prints and logs basic information about the Hamiltonian.
    * Constructs the Hartree–Fock state and the UCCSD ansatz.
    * Sets up the device and cost function.
    * Configures the classical optimizer (Adam).

All inputs are expected in the ``inputs`` directory and results/logs are
written to ``results``.
"""
import os
import pickle
from datetime import datetime, UTC

import numpy as np
import pennylane as qml
from openfermion.utils import count_qubits

# ----- Step 3.1: Environment information and paths -------------------------
SEED = 123
np.random.seed(SEED)

HERE = os.path.abspath(os.path.dirname(__file__))
INPUT_DIR = os.path.join(HERE, '..', 'inputs')
RES_DIR = os.path.join(HERE, '..', 'results')
os.makedirs(RES_DIR, exist_ok=True)

LOG_PATH = os.path.join(RES_DIR, 'baseline_setup.log')
with open(LOG_PATH, 'a') as log:
    log.write(f"\nRun at {datetime.now(UTC).isoformat()} UTC\n")
    log.write(f"PennyLane version: {qml.__version__}\n")
    log.write(f"Random seed: {SEED}\n")

# ----- Step 3.2: Load and validate qubit Hamiltonian ----------------------
HAM_PATH = os.path.join(INPUT_DIR, 'bk_symm_tapered.pkl')
with open(HAM_PATH, 'rb') as f:
    qubit_op = pickle.load(f)

num_qubits = count_qubits(qubit_op)
num_terms = len(qubit_op.terms)
terms = list(qubit_op.terms.items())

with open(LOG_PATH, 'a') as log:
    log.write(f"Loaded Hamiltonian from {HAM_PATH}\n")
    log.write(f"Number of qubits: {num_qubits}\n")
    log.write(f"Number of terms: {num_terms}\n")
    log.write("First few terms:\n")
    for term, coeff in terms[:3]:
        log.write(f"  {term}: {coeff}\n")
    log.write("...\n")

print(f"Hamiltonian qubits: {num_qubits}, terms: {num_terms}")
print("First term:", terms[0])
print("Last term:", terms[-1])

# Helper to convert OpenFermion operator to PennyLane Hamiltonian ----------
def openfermion_to_pennylane(op):
    coeffs = []
    obs = []
    for term, coeff in op.terms.items():
        if term == ():
            coeffs.append(coeff)
            obs.append(qml.Identity(0))
            continue
        pauli_ops = []
        for wire, gate in term:
            if gate == 'X':
                pauli_ops.append(qml.PauliX(wire))
            elif gate == 'Y':
                pauli_ops.append(qml.PauliY(wire))
            elif gate == 'Z':
                pauli_ops.append(qml.PauliZ(wire))
        # Use qml.prod for tensor products
        obs.append(qml.prod(*pauli_ops) if len(pauli_ops) > 1 else pauli_ops[0])
        coeffs.append(coeff)
    return qml.Hamiltonian(coeffs, obs)

hamiltonian = openfermion_to_pennylane(qubit_op)

# ----- Step 3.3: HF reference and UCCSD ansatz ----------------------------
ELECTRONS = 8
hf_state = qml.qchem.hf_state(ELECTRONS, num_qubits)
singles, doubles = qml.qchem.excitations(ELECTRONS, num_qubits)
s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)

num_params = len(singles) + len(doubles)
with open(LOG_PATH, 'a') as log:
    log.write(f"HF reference electrons: {ELECTRONS}\n")
    log.write(f"Singles: {len(singles)}, Doubles: {len(doubles)}\n")
    log.write(f"Total variational parameters: {num_params}\n")

# ----- Step 3.4: Device and cost function --------------------------------
DEV = qml.device('lightning.qubit', wires=num_qubits)

def uccsd_ansatz(params):
    qml.UCCSD(
        params,
        wires=range(num_qubits),
        s_wires=s_wires,
        d_wires=d_wires,
        init_state=hf_state  # optional but recommended
    )

@qml.qnode(DEV)
def circuit(params):
    qml.BasisState(hf_state, wires=range(num_qubits))
    uccsd_ansatz(params)
    return qml.expval(hamiltonian)

initial_params = np.zeros(num_params)
initial_energy = circuit(initial_params)
with open(LOG_PATH, 'a') as log:
    log.write(f"Initial energy: {initial_energy}\n")

# ----- Step 3.5: Classical optimizer configuration -----------------------
LEARNING_RATE = 0.2
MAX_STEPS = 200
optimizer = qml.optimize.AdamOptimizer(stepsize=LEARNING_RATE)

with open(LOG_PATH, 'a') as log:
    log.write("Optimizer: Adam\n")
    log.write(f"Learning rate: {LEARNING_RATE}\n")
    log.write(f"Max iterations: {MAX_STEPS}\n")

print("Optimizer configured: Adam, lr=", LEARNING_RATE, ", max steps=", MAX_STEPS)