#!/usr/bin/env python3
"""
Compare quantum VQE results with classical benchmarks
"""

import json
import numpy as np

def compare_results():
    # Load benchmark results
    with open('benchmark_results/method_comparison.json', 'r') as f:
        benchmarks = json.load(f)
    
    # Load quantum VQE results
    with open('results/advanced_benchmarking/kupccgsd_vqe/adam/log.json', 'r') as f:
        vqe_results = json.load(f)
    
    # The VQE energy includes a constant shift from the Hamiltonian mapping
    # From your phase2_mapping_results.md: constant = -855.6109991074851
    HAMILTONIAN_CONSTANT = -855.6109991074851
    
    vqe_energy_raw = vqe_results['final_energy']
    vqe_energy_corrected = vqe_energy_raw - HAMILTONIAN_CONSTANT  # Remove the constant shift
    
    hf_energy = benchmarks['hf_energy']
    
    print("QUANTUM vs CLASSICAL COMPARISON")
    print("="*50)
    print("Classical Methods (Full System):")
    print(f"HF energy:              {hf_energy:.6f} Ha")
    print(f"DFT (B3LYP) energy:     {benchmarks['dft_b3lyp_energy']:.6f} Ha")
    print(f"CASSCF energy:          {benchmarks['casscf_energy']:.6f} Ha")
    if benchmarks['ccsd_estimated']:
        print(f"CCSD (estimated):       {benchmarks['ccsd_estimated']:.6f} Ha")
    
    print(f"\nQuantum VQE (Active Space + Constant):")
    print(f"VQE raw energy:         {vqe_energy_raw:.6f} Ha")
    print(f"Hamiltonian constant:   {HAMILTONIAN_CONSTANT:.6f} Ha")
    print(f"VQE correlation energy: {vqe_energy_corrected:.6f} Ha")
    print(f"VQE total estimate:     {vqe_energy_raw:.6f} Ha")
    
    print(f"\nCorrelation Energy Analysis:")
    hf_correlation = 0.0  # Reference point
    dft_correlation = benchmarks['dft_b3lyp_energy'] - hf_energy
    casscf_correlation = benchmarks['casscf_energy'] - hf_energy
    vqe_correlation = vqe_energy_corrected  # This is the correlation energy in active space
    
    print(f"HF correlation:         {hf_correlation:.6f} Ha (reference)")
    print(f"DFT correlation:        {dft_correlation:.6f} Ha")
    print(f"CASSCF correlation:     {casscf_correlation:.6f} Ha")
    print(f"VQE correlation:        {vqe_correlation:.6f} Ha (active space)")
    
    print(f"\nQuantum Advantage Analysis:")
    print(f"VQE vs HF improvement:  {vqe_energy_corrected:.6f} Ha")
    print(f"VQE vs CASSCF:          {vqe_energy_corrected - casscf_correlation:.6f} Ha better")
    print(f"VQE vs DFT:             {vqe_energy_corrected - dft_correlation:.6f} Ha difference")
    
    # The correct comparison should be the correlation energy recovered
    print(f"\nNote: VQE captures correlation effects in 8e,8o active space")
    print(f"      Classical methods use full molecular calculation")
    print(f"      Direct energy comparison requires same reference state")

if __name__ == "__main__":
    compare_results()