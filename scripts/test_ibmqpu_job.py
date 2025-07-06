#!/usr/bin/env python3
"""Submit a minimal H2 test circuit to an IBM QPU.

Implements Phase 5.2 from the checklist. Requires environment
variables:
  - ``IBM_QUANTUM_TOKEN``: API token for IBM Quantum cloud.
  - ``IBM_QPU_DEVICE``: target backend (ibm_brisbane, ibm_sherbrooke, or
    ibm_torino). Defaults to ``ibm_torino``.
  - ``IBM_QPU_SHOTS``: optional shot count (default 100).

Example usage::

    export IBM_QUANTUM_TOKEN='YOUR_TOKEN'
    python scripts/test_ibmqpu_job.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import pennylane as qml
    from pennylane_qiskit import IBMQDevice
    from qiskit_ibm_provider import IBMProvider
except Exception as exc:  # pragma: no cover - missing deps
    sys.exit(
        "Required packages not installed. Run `pip install pennylane "
        "pennylane-qiskit qiskit-ibm-provider`. (error: %s)" % exc
    )

TOKEN = os.getenv("IBM_QUANTUM_TOKEN")
if TOKEN is None:
    sys.exit(
        "IBM_QUANTUM_TOKEN environment variable not set. Obtain a token from "
        "https://quantum.ibm.com/account and set it with `export IBM_QUANTUM_TOKEN='TOKEN'`."
    )

DEVICE_NAME = os.getenv("IBM_QPU_DEVICE", "ibm_torino")
SHOTS = int(os.getenv("IBM_QPU_SHOTS", "100"))

provider = IBMProvider(token=TOKEN)
backend = provider.get_backend(DEVICE_NAME)

# Minimal UCCSD-type circuit for H2 (very small)
symbols = ["H", "H"]
coords = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.74])
hamiltonian, qubits = qml.qchem.molecular_hamiltonian(symbols, coords)

hf_state = qml.qchem.hf_state(2, len(qubits))

@qml.qnode(qml.device("qiskit.ibmq", wires=len(qubits), backend=backend, shots=SHOTS, provider=provider))
def circuit(angle):
    qml.BasisState(hf_state, wires=range(len(qubits)))
    qml.DoubleExcitation(angle, wires=[0, 1, 2, 3])
    return qml.expval(hamiltonian)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "phase5_ibmqpu"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

angle = 0.0
submission_time = datetime.utcnow().isoformat() + "Z"
job = circuit.device.run(circuit.qtape, args=[angle])

info = {
    "device": DEVICE_NAME,
    "shots": SHOTS,
    "job_id": job.job_id(),
    "submitted": submission_time,
}

info_path = RESULTS_DIR / "test_job_info.json"
with info_path.open("w") as f:
    json.dump(info, f, indent=2)

print(f"Job {job.job_id()} submitted to {DEVICE_NAME} at {submission_time}")
print("Waiting for job to complete...")

while not job.in_final_state():
    time.sleep(5)

result = job.result()
counts = result.get_counts()

(Path(RESULTS_DIR) / "test_job_counts.json").write_text(json.dumps(counts, indent=2))

cal_data = backend.properties().to_dict()
(Path(RESULTS_DIR) / "device_calibration.json").write_text(json.dumps(cal_data, indent=2))

print("Results saved in", RESULTS_DIR)