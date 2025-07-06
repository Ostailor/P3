Phase 5: IBM QPU Execution & Real-Hardware Benchmarking — Checklist
5.1. QPU Access Preparation
 Verify IBM Quantum account credentials and ensure API token access is functional.

 Install and verify PennyLane (with Qiskit plugin) and all dependencies are up to date.

 Check real-time calibration/status for available QPUs (ibm_brisbane, ibm_fez, ibm_kingston, ibm_marrakesh, ibm_sherbrooke, ibm_torino).

 Prioritize QPUs based on queue, error rates, and connectivity for both test and main runs.

 Document all account/device setup details for reproducibility.

5.2. Test QPU Job (Debugging & Familiarization) — (10 Free Minutes)
 Create a script (scripts/test_ibmqpu_job.py) for a minimal test run:

Use a very small molecule (e.g., H₂, 2 qubits, simple circuit).

Keep circuit depth and shot count minimal for speed.

Set up for one of: ibm_brisbane, ibm_sherbrooke, or ibm_torino.

 Submit the test job to the chosen QPU.

 Monitor the job: track queue time, execution, retrieval of results.

 Document all settings: device, number of shots, job ID, submission time.

 Save and log all outputs, including device calibration data and raw measurement results.

 Note any errors, bottlenecks, or unexpected issues encountered during submission or execution.

 Prepare a short “QPU job run guide” for the team, summarizing lessons learned.

5.3. Main QPU Scientific Runs (90-Minute Budget)
 Select your best-performing ansatz/circuit and parameter initialization from Phase 4.

Confirm it fits within available qubits and circuit depth limits.

 Create a dedicated script for the main QPU run (scripts/run_ibmqpu_vqe.py):

Accept QPU name, Hamiltonian, ansatz, and parameters as arguments.

Include robust error handling and clear logging.

 (If possible) Pre-test the main circuit on IBM’s noiseless simulator for timing and resource estimation.

 Submit the main VQE job to the best available QPU (based on calibration and queue status).

 Monitor job status in real time; document queue time, job ID, start/end time.

 If job is at risk of timing out, break into smaller sub-tasks or batch executions.

 Save all results (raw measurement data, output energies, parameters) to /results/phase5_ibmqpu/ with clear filenames.

 Repeat on additional QPUs if time/resources permit, to cross-benchmark hardware.

5.4. Data Collection, Logging, and Analysis
 For each QPU run, log:

QPU name, calibration data, queue/wait time, runtime, job ID.

Number of shots and circuit depth.

Final measured energies and statistical error bars.

All raw measurement counts and output files.

 Compare QPU results with simulator and classical benchmarks for the same ansatz and parameters.

 Save all scripts, logs, and results for reproducibility.

5.5. Documentation & Reporting
 Document all job submission workflows and troubleshooting steps in a README or similar.

 Include a summary of QPU performance vs simulator (accuracy, noise, run time, limitations).

 Prepare plots and tables showing QPU results, queue times, and energy comparisons.

 Write up a “lessons learned from real hardware” section for your final report, including:

Any errors encountered and solutions/workarounds

Impact of hardware noise and queue time

Recommendations for future hardware runs

Parallelization & Best Practices
Assign different QPU test/main jobs to multiple team members for maximum use of limited QPU time.

Always save and back up QPU job logs, outputs, and job IDs.

Double check all inputs and parameters before job submission—hardware time is precious!

Prepare to resume or re-run any interrupted or failed jobs quickly.
