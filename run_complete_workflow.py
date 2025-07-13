#!/usr/bin/env python3
"""
Complete workflow execution script for qBraid
Runs the entire Dibenzothiophene VQE analysis pipeline in the correct order.
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def run_command(cmd, description, log_file=None):
    """Execute a command and handle output."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        if log_file:
            # Redirect output to log file
            with open(log_file, 'w') as f:
                result = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, 
                                      text=True, timeout=3600)  # 1 hour timeout
            print(f"Output saved to: {log_file}")
        else:
            # Show output in console
            result = subprocess.run(cmd, shell=True, text=True, timeout=3600)
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ SUCCESS - Completed in {elapsed:.1f}s")
            return True
        else:
            print(f"❌ FAILED - Exit code: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ TIMEOUT - Command exceeded 1 hour limit")
        return False
    except Exception as e:
        print(f"❌ ERROR - {str(e)}")
        return False

def main():
    """Execute the complete workflow."""
    print("🚀 Starting Complete Dibenzothiophene VQE Workflow")
    print("=" * 80)
    
    workflow_start = time.time()
    failed_steps = []
    
    # Phase 0: Setup and Geometry Preparation
    steps = [
        {
            'cmd': 'python scripts/setup_qbraid.py',
            'desc': 'Environment Setup and Validation',
            'log': None,
            'required': True
        },
        {
            'cmd': 'python scripts/geom_opt_pyscf.py',
            'desc': 'Geometry Optimization',
            'log': 'pyscf_logs/dbt_geoopt.log',
            'required': True
        },
        
        # Phase 1: Molecular Setup
        {
            'cmd': 'python scripts/analyze_orbitals.py',
            'desc': 'Orbital Analysis',
            'log': 'pyscf_logs/orbital_energies.log',
            'required': True
        },
        {
            'cmd': 'python scripts/run_pyscf_dbt.py --basis sto-3g',
            'desc': 'Molecular Integrals (STO-3G)',
            'log': 'pyscf_logs/dbt_STO3G.log',
            'required': True
        },
        
        # Phase 2: Quantum Hamiltonian
        {
            'cmd': 'python scripts/map_to_qubit_hamiltonian.py',
            'desc': 'Qubit Hamiltonian Mapping',
            'log': 'logs/mapping_stats.log',
            'required': True
        },
        
        # Phase 3: VQE Simulations
        {
            'cmd': 'python scripts/kupccgsd_vqe.py',
            'desc': 'k-UpCCGSD VQE (High Accuracy)',
            'log': None,
            'required': True
        },
        {
            'cmd': 'python scripts/adapt_vqe.py',
            'desc': 'ADAPT-VQE (Hardware Compatible)',
            'log': None,
            'required': True
        },
        
        # Phase 4: Benchmarking
        {
            'cmd': 'python scripts/benchmark_methods.py',
            'desc': 'Classical Method Benchmarking',
            'log': 'logs/classical.log',
            'required': True
        },
        {
            'cmd': 'python scripts/compare_with_benchmarks.py',
            'desc': 'Quantum vs Classical Comparison',
            'log': None,
            'required': True
        },
        
        # Phase 5: Circuit Analysis
        {
            'cmd': 'python scripts/calculate_kupccgsd_depth.py',
            'desc': 'k-UpCCGSD Circuit Depth Analysis',
            'log': None,
            'required': True
        },
        {
            'cmd': 'python scripts/calculate_adapt_depth.py',
            'desc': 'ADAPT-VQE Circuit Depth Analysis',
            'log': None,
            'required': True
        },
        
        # Phase 6: Optional Hardware (only if IBM credentials available)
        {
            'cmd': 'python scripts/ibm_nofireopal.py --shots 1000 --backend ibm_brisbane --resilience_level 1',
            'desc': 'IBM Quantum Hardware Execution (Optional)',
            'log': 'ibm_qpu_and_simulator.log',
            'required': False
        }
    ]
    
    print(f"Total steps: {len(steps)}")
    print(f"Required steps: {sum(1 for s in steps if s['required'])}")
    print(f"Optional steps: {sum(1 for s in steps if not s['required'])}")
    
    for i, step in enumerate(steps, 1):
        step_start = time.time()
        
        success = run_command(step['cmd'], f"{i}/{len(steps)} - {step['desc']}", step['log'])
        
        if not success:
            if step['required']:
                failed_steps.append(f"Step {i}: {step['desc']}")
                print(f"\n❌ CRITICAL FAILURE - Required step failed!")
                print(f"Failed command: {step['cmd']}")
                break
            else:
                print(f"\n⚠️  OPTIONAL STEP FAILED - Continuing workflow...")
                print(f"Failed command: {step['cmd']}")
                failed_steps.append(f"Step {i}: {step['desc']} (optional)")
    
    # Workflow Summary
    total_time = time.time() - workflow_start
    
    print("\n" + "=" * 80)
    print("🏁 WORKFLOW SUMMARY")
    print("=" * 80)
    print(f"Total runtime: {total_time/60:.1f} minutes")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not failed_steps:
        print("✅ ALL STEPS COMPLETED SUCCESSFULLY!")
        print("\n📁 Key Output Files:")
        print("• results/advanced_benchmarking/kupccgsd_vqe/adam/log.json")
        print("• results/advanced_benchmarking/adapt_vqe/adam/log.json")
        print("• benchmark_results/method_comparison.json")
        print("• inputs/bk_symm_tapered.pkl")
        
    else:
        print(f"⚠️  WORKFLOW COMPLETED WITH {len(failed_steps)} ISSUES:")
        for failure in failed_steps:
            print(f"• {failure}")
    
    print("\n📖 For detailed analysis, see:")
    print("• README.md - Complete documentation")
    print("• docs/ - Phase-specific workflow guides")
    print("• pyscf_logs/ - Classical calculation logs")
    
    return len(failed_steps) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)