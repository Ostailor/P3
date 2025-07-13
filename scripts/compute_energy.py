#!/usr/bin/env python3
import os
from pyscf import gto, scf, mcscf

# 1) Build molecule
xyz_path = os.path.join('dbt_geometry','dbt_opt_opt.xyz')
with open(xyz_path) as f:
    atom_block = "\n".join(f.read().splitlines()[2:])
mol = gto.Mole(atom=atom_block, unit='Angstrom',
               charge=0, spin=0, basis='cc-pVTZ')
mol.build()

# 2) RHF with extended SCF cycles
mf = scf.RHF(mol)
mf.max_cycle   = 100
mf.conv_tol    = 1e-8
mf.level_shift = 0.2
ehf = mf.kernel()
print(f"RHF energy = {ehf:.8f} Ha")

# 3) CASCI(8,8) with tightened tolerances
ncas, nelecas = 8, 8
casci = mcscf.CASCI(mf, ncas, nelecas)
casci.conv_tol        = 1e-8
casci.conv_tol_grad   = 1e-6
casci.fcisolver.max_cycle = 200

# Unpack all outputs
e_tot, e_casci, ci_vector, mo1, mo2 = casci.kernel()
print(f"CASCI total energy = {e_tot:.8f} Ha")
print(f"CASCI(8,8)    energy = {e_casci:.8f} Ha")
