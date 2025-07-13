#!/usr/bin/env python3
import os
import numpy as np
from pyscf import gto, scf, geomopt

# Create necessary directories if they don't exist
os.makedirs('dbt_geometry', exist_ok=True)
os.makedirs('pyscf_logs', exist_ok=True)

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
