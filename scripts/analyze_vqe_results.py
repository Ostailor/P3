#!/usr/bin/env python3
"""Analyze baseline VQE results and create convergence plot."""
import os
import json
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.abspath(os.path.dirname(__file__))
RES_DIR = os.path.join(HERE, '..', 'results')

energy = np.load(os.path.join(RES_DIR, 'energy_history.npy'))
params = np.load(os.path.join(RES_DIR, 'param_history.npy'))
summary = {
    'final_energy': float(energy[-1]),
    'iterations': int(len(energy)),
    'optimized_parameters': params[-1].tolist() if params.ndim > 1 else params.tolist(),
}

log_path = os.path.join(RES_DIR, 'baseline_setup.log')
if os.path.exists(log_path):
    with open(log_path) as f:
        for line in f:
            if line.startswith('Runtime seconds'):
                summary['runtime_sec'] = float(line.split(':')[1].strip())

with open(os.path.join(RES_DIR, 'vqe_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

plt.plot(range(len(energy)), energy, marker='o', ms=3)
plt.xlabel('Iteration')
plt.ylabel('Energy (Ha)')
plt.title('VQE Convergence')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RES_DIR, 'vqe_convergence.png'))