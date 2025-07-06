Phase 5: IBM QPU Testing
========================

Overview
--------
This document records preparation steps and the short hardware test run
performed in Phase 5.1 and 5.2. The goal is to verify IBM Quantum account
access and run a minimal job on real hardware.

5.1 QPU Access Preparation
-------------------------
1. Install dependencies::

       pip install -r requirements.txt

   ``pennylane-qiskit`` and ``qiskit-ibm-provider`` are required for hardware
   access. Ensure the environment variables below are exported before running
   the setup script.

2. Export IBM Quantum credentials (replace values with your own)::

       export IBM_QUANTUM_TOKEN="<token>"
       export IBM_QUANTUM_HUB="<hub>"       # optional
       export IBM_QUANTUM_GROUP="<group>"   # optional
       export IBM_QUANTUM_PROJECT="<project>"  # optional

3. Run ``scripts/setup_ibmq_account.py``. The script writes ``account_info.json``
   and ``device_status.txt`` under ``results/phase5_ibmqpu/``. Review these
   files to confirm the account and device statuses.

5.2 Test Job Execution
----------------------
1. Choose a backend with light queue (``ibm_brisbane``, ``ibm_sherbrooke`` or
   ``ibm_torino``)::

       export IBM_QPU_DEVICE=ibm_torino
       export IBM_QPU_SHOTS=100
       python scripts/test_ibmqpu_job.py

   The job submits a tiny H₂ circuit with a single ``DoubleExcitation`` gate
   and waits until completion. Job information and results are saved to
   ``results/phase5_ibmqpu/``.

2. The output files include:
   - ``test_job_info.json`` – device name, shot count, job ID and submission time.
   - ``test_job_counts.json`` – raw measurement counts.
   - ``device_calibration.json`` – calibration data for the chosen backend.

Lessons Learned
---------------
- Always verify the token via ``setup_ibmq_account.py`` before submitting jobs.
- Queue time varies widely; ``ibm_torino`` typically has the shortest wait.
- Saving the calibration and job ID allows for reproducibility and later
  troubleshooting.