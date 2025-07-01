3. Baseline VQE Pipeline (PennyLane Only, Script-Based Checklist)
3.1. Project & Environment Preparation
 Set up a project folder with clear subdirectories:

/inputs/ for all input files (integrals, Hamiltonian)

/scripts/ for all Python scripts

/results/ for outputs (energies, logs, plots)

 Install and verify PennyLane, PennyLane-QChem, NumPy, and matplotlib in your Python environment.

 Record the random seed and PennyLane version for reproducibility.

3.2. Load and Validate Qubit Hamiltonian
 Write a script to load your DBT qubit Hamiltonian (mapped via Jordan–Wigner, from Step 2).

 Print/verify:

Number of qubits

Number of Hamiltonian terms

Sanity check on Hamiltonian structure (e.g., first/last few terms)

 Log these details in a results or setup log file.

3.3. Define Initial State and UCCSD Ansatz
 Document and set up the number of electrons and qubits according to your chosen active space.

 Specify the Hartree–Fock reference state as your quantum circuit’s starting point.

 Programmatically determine all single and double excitations for the active space.

 Set up the UCCSD ansatz:

Ensure all required parameters and mappings are correct

Log the total number of variational parameters

3.4. Device and Cost Function Setup
 Specify your PennyLane device:

Use the recommended high-performance simulator (lightning.qubit)

Choose analytic or sampling mode and log the choice

 Set up the cost function as the expectation value of the Hamiltonian with respect to the UCCSD circuit.

 Log the initial energy (with initial parameters, usually zeros or small random values).

3.5. Classical Optimizer Configuration
 Choose and configure your optimizer (start with Adam for the baseline).

 Decide on:

Learning rate/stepsize

Maximum number of iterations

Convergence/stopping criteria (e.g., energy threshold, parameter change threshold)

 Document all optimizer settings and random seed.

3.6. Baseline VQE Optimization Run
 Run the VQE optimization loop in your script:

Record energy at each iteration

Record parameter values at each iteration (or at checkpoints)

Monitor for convergence issues or failed runs (save logs)

 Save the full history of energies and parameters to results files for later analysis.

3.7. Results Analysis and Visualization
 Plot the VQE energy convergence curve and save as a figure in /results/.

 Record and report:

Final optimized energy

Number of iterations to converge

Optimized parameter values

Total runtime (using the time module, if possible)

 Compare final VQE energy to your reference (e.g., Hartree–Fock, CCSD, etc.)

 Document and discuss convergence behavior (smooth, noisy, stuck, etc.).

3.8. Documentation and Traceability
 Clearly document all choices in your script header:

Molecule name, basis, active space, mapping, device, ansatz, optimizer, random seed, parameter count

 Comment each major step for clarity (inputs, setup, optimization, output)

 Save all key printouts and logs to a text file for traceability

 Maintain a requirements.txt with all used package versions

3.9. Reproducibility Check
 Delete all output files, rerun the entire script, and ensure all outputs are identical or consistent

 If possible, test the script on another machine/environment to confirm reproducibility

 Push or package scripts, inputs, and results with a README explaining how to rerun everything from scratch

Optional, for Robustness
 Try different initializations for parameters (all zeros, small random)

 Repeat the run with different optimizer settings and document effects

 If convergence fails, save a summary of the issue and steps taken to debug
