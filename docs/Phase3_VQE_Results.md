Phase 3: Baseline VQE Results
=============================

This phase performs a UCCSD VQE optimization for dibenzothiophene using the
Bravyi–Kitaev tapered Hamiltonian. The optimizer is Adam with an adaptive
learning rate.

Summary
-------
- **Device:** lightning.qubit
- **Ansatz:** UCCSD
- **Parameters:** 204
- **Iterations:** 200
- **Final energy:** -864.689 Ha

A convergence plot is saved as `results/vqe_convergence.png`. A JSON summary of
final parameters and runtime is saved as `results/vqe_summary.json`.

Re-running `python scripts/analyze_vqe_results.py` regenerates these files.