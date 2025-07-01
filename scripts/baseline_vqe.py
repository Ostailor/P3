#!/usr/bin/env python3
import os, pickle
from datetime import datetime, UTC

import pennylane as qml
import pennylane.numpy as pnp
from openfermion.utils import count_qubits

# -- Paths & Logging --
SEED = 123
pnp.random.seed(SEED)
HERE = os.path.abspath(os.path.dirname(__file__))
INPUT_DIR = os.path.join(HERE, '..', 'inputs')
RES_DIR = os.path.join(HERE, '..', 'results')
os.makedirs(RES_DIR, exist_ok=True)
LOG_PATH = os.path.join(RES_DIR, 'baseline_setup.log')

def log(msg):
    with open(LOG_PATH, 'a') as f:
        f.write(msg + "\n")

log(f"Run at {datetime.now(UTC).isoformat()} UTC")
log(f"PennyLane version: {qml.__version__}")
log(f"Random seed: {SEED}")

# -- Load Hamiltonian --
with open(os.path.join(INPUT_DIR, 'bk_symm_tapered.pkl'), 'rb') as f:
    qubit_op = pickle.load(f)
num_qubits = count_qubits(qubit_op)
num_terms = len(qubit_op.terms)
log(f"Hamiltonian loaded: {num_qubits} qubits, {num_terms} terms")

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

hamiltonian = of_to_pl(qubit_op)

# -- UCCSD Setup --
ELECTRONS = 8
hf_state = pnp.array(qml.qchem.hf_state(ELECTRONS, num_qubits), dtype=int)
singles, doubles = qml.qchem.excitations(ELECTRONS, num_qubits)
s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)
num_params = len(singles) + len(doubles)
log(f"UCCSD setup: electrons={ELECTRONS}, params={num_params}")

# -- QNode circuit (raw expectation) --
dev = qml.device("lightning.qubit", wires=num_qubits)

@qml.qnode(dev)
def circuit(params):
    qml.UCCSD(
        params,
        wires=range(num_qubits),
        s_wires=s_wires,
        d_wires=d_wires,
        init_state=hf_state,
    )
    return qml.expval(hamiltonian)

# -- Cost wrapper to ensure real scalar output --
def cost_fn(params):
    return pnp.real(circuit(params))

# -- Adam + plateau LR scheduler --
init_lr = 0.05
min_lr = 0.005
lr = init_lr
patience = 5        # steps to wait for improvement
factor = 0.5         # LR reduction factor
counter = 0
best_energy = float('inf')

opt = qml.AdamOptimizer(stepsize=lr)
params = pnp.random.normal(0, 1e-2, num_params, requires_grad=True)
energy_hist = []

for n in range(200):
    # gradient and update
    grad = qml.grad(cost_fn)(params)
    params = opt.step(cost_fn, params)
    energy = float(cost_fn(params))
    energy_hist.append(energy)

    # logging
    if n % 10 == 0 or n == 199:
        print(f"Step {n:03d} | lr={lr:.4f} | E={energy:.8f} | ‖grad‖={pnp.linalg.norm(grad):.2e}")
    log(f"Step {n:03d} | lr={lr:.4f} | E={energy:.8f}")

    # check improvement
    if energy < best_energy - 1e-6:
        best_energy = energy
        counter = 0
    else:
        counter += 1
        if counter >= patience and lr > min_lr:
            lr = max(lr * factor, min_lr)
            opt.stepsize = lr
            print(f"No improvement for {patience} steps. Reducing lr to {lr:.4f}")
            log(f"Reduced lr to {lr:.4f} after {patience} non-improving steps")
            counter = 0

# -- Save results --
pnp.save(os.path.join(RES_DIR, "energy_history.npy"), pnp.array(energy_hist))
pnp.save(os.path.join(RES_DIR, "param_history.npy"), params)
log(f"Final energy: {energy_hist[-1]:.8f}")
print("VQE complete! Final energy:", energy_hist[-1])
