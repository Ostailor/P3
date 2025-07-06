#!/usr/bin/env python3
"""Check IBM Quantum IAM credentials and log QPU status (Phase 5.1)."""

from __future__ import annotations
import json, sys
from pathlib import Path

# ---------- 3rd-party imports ----------
try:
    from qiskit_ibm_runtime import QiskitRuntimeService
except Exception as exc:
    sys.exit("Install with `pip install qiskit-ibm-runtime` – " f"{exc}")

# ---------- where to store results ----------
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "phase5_ibmqpu"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- load service (reads ~/.qiskit/qiskit-ibm.json) ----------
try:
    service = QiskitRuntimeService()           # uses saved token + CRN
except Exception as exc:
    sys.exit(f"Could not authenticate to IBM Quantum – {exc}")

# ---------- dump account metadata ----------
acct_path = RESULTS_DIR / "account_info.json"
with acct_path.open("w") as f:
    json.dump(service.active_account(), f, indent=2)

# ---------- check a few backends ----------
DEVICES = [
    "ibm_brisbane",
    "ibm_fez",
    "ibm_kingston",
    "ibm_marrakesh",
    "ibm_sherbrooke",
    "ibm_torino",
]

status_log = RESULTS_DIR / "device_status.txt"
with status_log.open("w") as f:
    for name in DEVICES:
        try:
            backend = service.backend(name)     # runtime syntax
            status  = backend.status()
            f.write(
               f"{name}: operational={status.operational}, "
               f"pending_jobs={status.pending_jobs}, "
               f"qubits={backend.num_qubits}\n")
        except Exception as exc:
            f.write(f"{name}: error ({exc})\n")

print(f"✔ Account info → {acct_path}")
print(f"✔ Device status → {status_log}")
