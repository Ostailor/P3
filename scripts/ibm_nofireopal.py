#!/usr/bin/env python3
"""
Full Hamiltonian expectation via Qiskit Runtime Estimator in job mode (no Batch),
with optional local statevector simulation, full trace logging,
big‑endian mapping, and correct full‑device observable embedding.
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import logging
from logging.handlers import RotatingFileHandler

import pennylane as qml
from dotenv import load_dotenv
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit_ibm_runtime import (
    QiskitRuntimeService,
    Estimator,
    EstimatorOptions,
)
from qiskit.transpiler import generate_preset_pass_manager
from pennylane import qchem

# ────────────────────────────────────────────────────────────────────────────
# Logger setup
logger = logging.getLogger("quantum_run")
logger.setLevel(logging.INFO)
console = logging.StreamHandler(sys.stdout)
console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(console)
file_h = RotatingFileHandler("ibm_qpu_and_simulator.log", maxBytes=5e6, backupCount=2)
file_h.setFormatter(console.formatter)
logger.addHandler(file_h)
# ────────────────────────────────────────────────────────────────────────────

def load_adapt_state(params, raw_ops, n_qubits):
    """Rebuild the ADAPT‑VQE state prep from labels via PennyLane."""
    ELECTRONS = 8
    singles, doubles = qchem.excitations(ELECTRONS, n_qubits)
    sw, dw = qchem.excitations_to_wires(singles, doubles, wires=range(n_qubits))

    pool_labels, pool_ops = [], []
    for w in sw:
        lbl = f"FermionicSingle{list(w)}"
        pool_labels.append(lbl)
        pool_ops.append(lambda theta, w=w: qml.SingleExcitation(theta, wires=w))
    for w1, w2 in dw:
        lbl = f"FermionicDouble{list(w1)}+{list(w2)}"
        pool_labels.append(lbl)
        pool_ops.append(lambda theta, w1=w1, w2=w2: qml.DoubleExcitation(theta, wires=[*w1, *w2]))

    exc = [pool_ops[pool_labels.index(lbl)] for lbl in raw_ops]
    logger.info("Loaded %d excitation operators", len(exc))

    dev = qml.device("default.qubit", wires=n_qubits)
    @qml.qnode(dev)
    def state_qnode(t):
        hf = qchem.hf_state(ELECTRONS, n_qubits)
        qml.BasisState(hf, wires=range(n_qubits))
        for θ, g in zip(t, exc):
            g(θ)
        return qml.state()

    qasm = qml.workflow.construct_tape(state_qnode)(params).to_openqasm(False)
    qc = QuantumCircuit.from_qasm_str(qasm)
    qc.barrier()
    return qc, state_qnode

def pauli_list_from_ham(ham, n_qubits):
    """Convert OpenFermion Hamiltonian to list of (coeff, Pauli) using big‑endian mapping."""
    paulis = []
    for term, coeff in ham.terms.items():
        if not term:
            label = "I" * n_qubits
        else:
            z = ["I"] * n_qubits
            for wire, op in term:
                # big‑endian: wire i → position (n_qubits‑1‑i)
                z[n_qubits - 1 - wire] = op
            label = "".join(z)
        paulis.append((float(coeff.real), Pauli(label)))
    return paulis

def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=10000)
    parser.add_argument("--simulator", action="store_true", help="use local statevector")
    parser.add_argument("--instance", default=os.getenv("QISKIT_IBM_INSTANCE"))
    parser.add_argument("--token", default=os.getenv("QISKIT_IBM_TOKEN"))
    parser.add_argument("--channel", default=os.getenv("QISKIT_IBM_CHANNEL", "ibm_cloud"))
    parser.add_argument("--backend", default=os.getenv("QISKIT_IBM_BACKEND", "ibm_brisbane"))
    parser.add_argument(
        "--resilience_level", type=int, choices=[0, 1, 2], default=1,
        help="Error mitigation level for hardware: 0=no, 1=readout, 2=readout+ZNE"
    )
    args = parser.parse_args()

    # Force Qiskit Runtime logs to INFO so your file captures runtime-level details
    os.environ.setdefault("QISKIT_IBM_RUNTIME_LOG_LEVEL", "INFO")

    logger.info("Run config: %s", json.dumps({
        "shots": args.shots,
        "simulator": args.simulator,
        "backend": args.backend,
        "resilience_level": args.resilience_level
    }))
    print(f"Simulation mode: {args.simulator}")

    if not (args.instance and args.token):
        raise RuntimeError("Please set QISKIT_IBM_INSTANCE & QISKIT_IBM_TOKEN")

    # ── Load ADAPT‑VQE data ──────────────────────────────────────────────
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    adapt_dir = os.path.join(root, "results/advanced_benchmarking/adapt_vqe/adam")
    raw_ops = [
        l.split(":", 1)[1].strip()
        for l in open(os.path.join(adapt_dir, "selected_ops.txt"))
        if ":" in l
    ]
    params = np.load(os.path.join(adapt_dir, "params.npy"))
    with open(os.path.join(root, "inputs", "bk_symm_tapered.pkl"), "rb") as f:
        ham = pickle.load(f)
    n_qubits = max(q for term in ham.terms for q, _ in term) + 1

    logger.info("Loaded ADAPT-VQE: %d excitations, %d qubits", len(raw_ops), n_qubits)
    state_circ, state_qnode = load_adapt_state(params, raw_ops, n_qubits)
    logger.info("State prep depth: %d", state_circ.depth())

    pauli_terms = pauli_list_from_ham(ham, n_qubits)
    total = len(pauli_terms)
    logger.info("Hamiltonian terms: %d", total)

    # ── Set up hardware vs simulator ────────────────────────────────
    if not args.simulator:
        svc = QiskitRuntimeService(channel=args.channel, token=args.token, instance=args.instance)
        backend = svc.backend(args.backend)
        opts = EstimatorOptions(default_shots=args.shots, resilience_level=args.resilience_level)
        pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
        estimator = Estimator(mode=backend, options=opts)
        logger.info("Initialized hardware backend %s with resilience_level=%d", args.backend, args.resilience_level)
    else:
        logger.info("Running simulator mode")

    energy = 0.0
    for i, (coeff, pauli) in enumerate(pauli_terms, start=1):
        label = pauli.to_label()
        logger.info("Term %d/%d: %s", i, total, label)
        print(f"Term {i}/{total}: {label}")

        if args.simulator:
            psi = state_qnode(params)
            expv = float(np.real(np.vdot(psi, pauli.to_matrix() @ psi)))
        else:
            qc_t = pm.run(state_circ)
            little_pauli = Pauli(pauli.to_label()[::-1]).apply_layout(qc_t.layout)
            job = estimator.run([(qc_t, little_pauli)])
            evs = job.result()[0].data.evs
            expv = float(evs) if np.ndim(evs) == 0 else float(evs[0])

        sigma = np.sqrt((1 - expv**2) / args.shots)
        contrib = coeff * expv
        out = f"coeff={coeff:+.6f} exp={expv:+.6f} σ={sigma:.6f} → {contrib:+.6f}"
        logger.info(out)
        print(out)
        energy += contrib

    print()
    logger.info("Total ADAPT-VQE energy = %.8f Ha", energy)
    print(f"Total ADAPT-VQE energy = {energy:.8f} Ha")


if __name__ == "__main__":
    main()
