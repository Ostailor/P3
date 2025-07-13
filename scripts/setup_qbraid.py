#!/usr/bin/env python3
"""
Setup script for qBraid execution - ensures all directories exist
and validates the environment before running the main workflow.
"""

import os
import sys
import subprocess
import importlib.util

def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def create_directories():
    """Create all necessary directories for the project."""
    directories = [
        'dbt_geometry',
        'integrals', 
        'inputs',
        'outputs',
        'pyscf_logs',
        'logs',
        'results',
        'results/advanced_benchmarking',
        'results/advanced_benchmarking/kupccgsd_vqe',
        'results/advanced_benchmarking/adapt_vqe',
        'results/phase5_ibmqpu',
        'benchmark_results',
        'docs'
    ]
    
    print("Creating necessary directories...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✓ {directory}")

def validate_files():
    """Check that essential input files exist."""
    required_files = [
        'dbt_geometry/dbt_raw.sdf',
        'dbt_geometry/dbt_opt.xyz',
        'requirements.txt'
    ]
    
    print("\nValidating required files...")
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} (missing)")
            missing_files.append(file_path)
    
    return missing_files

def check_dependencies():
    """Check if key dependencies are installed."""
    key_packages = [
        ('pennylane', 'pennylane'),
        ('qiskit', 'qiskit'),
        ('pyscf', 'pyscf'),
        ('openfermion', 'openfermion'),
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('matplotlib', 'matplotlib')
    ]
    
    print("\nChecking dependencies...")
    missing_packages = []
    for package, import_name in key_packages:
        if check_package(package, import_name):
            print(f"  ✓ {package}")
        else:
            print(f"  ✗ {package} (not found)")
            missing_packages.append(package)
    
    return missing_packages

def main():
    """Main setup function."""
    print("=" * 60)
    print("qBraid Setup Script for Dibenzothiophene VQE Project")
    print("=" * 60)
    
    # Create directories
    create_directories()
    
    # Validate files
    missing_files = validate_files()
    
    # Check dependencies
    missing_packages = check_dependencies()
    
    # Summary
    print("\n" + "=" * 60)
    print("SETUP SUMMARY")
    print("=" * 60)
    
    if not missing_files and not missing_packages:
        print("✅ Setup complete! All requirements satisfied.")
        print("\nYou can now run the workflow:")
        print("1. python scripts/geom_opt_pyscf.py")
        print("2. python scripts/run_pyscf_dbt.py --basis sto-3g")
        print("3. python scripts/map_to_qubit_hamiltonian.py")
        print("4. python scripts/kupccgsd_vqe.py")
        print("5. python scripts/adapt_vqe.py")
        return True
    else:
        print("⚠️  Setup incomplete. Please address the following:")
        
        if missing_files:
            print("\nMissing files:")
            for f in missing_files:
                print(f"  - {f}")
        
        if missing_packages:
            print("\nMissing packages (install with: pip install -r requirements.txt):")
            for p in missing_packages:
                print(f"  - {p}")
        
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)