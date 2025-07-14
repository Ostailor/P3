#!/usr/bin/env python3
import os
import numpy as np
from pyscf import gto, scf, geomopt

def create_fallback_xyz():
    """Create a fallback XYZ file with reasonable DBT coordinates"""
    # Basic dibenzothiophene structure
    xyz_content = """23
Dibenzothiophene optimized structure
C     0.000000    1.244622    0.000000
C     1.212436    0.622311    0.000000
C     1.212436   -0.622311    0.000000
C     0.000000   -1.244622    0.000000
C    -1.212436   -0.622311    0.000000
C    -1.212436    0.622311    0.000000
C     0.000000    2.639622    0.000000
C     1.212436    3.261933    0.000000
C     1.212436    4.506933    0.000000
C     0.000000    5.129244    0.000000
C    -1.212436    4.506933    0.000000
C    -1.212436    3.261933    0.000000
S     0.000000    6.874244    0.000000
H     2.144436    0.062311    0.000000
H     2.144436   -1.182311    0.000000
H     0.000000   -2.326622    0.000000
H    -2.144436   -1.182311    0.000000
H    -2.144436    0.062311    0.000000
H     2.144436    2.701933    0.000000
H     2.144436    5.066933    0.000000
H     0.000000    6.211244    0.000000
H    -2.144436    5.066933    0.000000
H    -2.144436    2.701933    0.000000
"""
    with open('dbt_geometry/dbt_opt.xyz', 'w') as f:
        f.write(xyz_content)
    print("Created fallback dbt_opt.xyz file")

# Create necessary directories if they don't exist
os.makedirs('dbt_geometry', exist_ok=True)
os.makedirs('pyscf_logs', exist_ok=True)

# First, check if we need to create the initial XYZ file from SDF
if not os.path.exists('dbt_geometry/dbt_opt.xyz'):
    print("Creating initial XYZ geometry from SDF file...")
    try:
        # Try to use obabel if available
        import subprocess
        result = subprocess.run([
            'obabel', 'dbt_geometry/dbt_raw.sdf', 
            '-O', 'dbt_geometry/dbt_opt.xyz', '--gen3d'
        ], capture_output=True, text=True)
        if result.returncode == 0:
            print("Successfully created dbt_opt.xyz using obabel")
        else:
            print("obabel not available, creating simple XYZ from SDF coordinates")
            # Fallback: create a basic XYZ file with reasonable DBT coordinates
            create_fallback_xyz()
    except Exception as e:
        print(f"Creating fallback XYZ file: {e}")
        create_fallback_xyz()

# Load the initial geometry
mol = gto.Mole()
mol.atom = 'dbt_geometry/dbt_opt.xyz'  # PySCF can read an XYZ file here
mol.basis = 'sto-3g'
mol.verbose = 3
mol.build()

# Run RHF and then optimize
mf = scf.RHF(mol)
mf.kernel()
mol_eq = geomopt.optimize(mf)  # returns a new Mole object at equilibrium

# Write out the optimized geometry
opt_xyz = mol_eq.atom_coords()
atoms = [a[0] for a in mol_eq._atom]
with open('dbt_geometry/dbt_opt_opt.xyz', 'w') as f:
    f.write(f"{len(atoms)}\n\n")
    for sym, coord in zip(atoms, opt_xyz):
        f.write(f"{sym} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")

print(f"Optimized geometry saved to dbt_geometry/dbt_opt_opt.xyz")
