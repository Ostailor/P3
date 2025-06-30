#!/usr/bin/env python3
import numpy as np
from pyscf import gto, scf

# Load optimized geometry
mol = gto.Mole()
mol.atom = 'dbt_geometry/dbt_opt_opt.xyz'
mol.basis = 'sto-3g'
mol.build()

# Run RHF
mf = scf.RHF(mol)
mf.verbose = 0
mf.kernel()

# Print orbital energies
mo_energies = mf.mo_energy
for idx, energy in enumerate(mo_energies):
    print(f"MO {idx+1:>2d}: {energy:.6f} Ha")

# Suggest HOMO index
homo = mol.nelectron // 2
print(f"\nHOMO is MO {homo}, LUMO is MO {homo+1}")


from pyscf import lo

# Get the AO orthogonalization matrix
orth_coeff = lo.orth_ao(mol, 'lowdin')

# Build density matrix for the first 44 MOs
mo_occ = np.array([2]*44 + [0]*(mf.mo_coeff.shape[1]-44))
dm = mf.make_rdm1(mo_coeff=mf.mo_coeff, mo_occ=mo_occ)

# Mulliken population analysis in the orthogonalized AO basis
popl = mf.mulliken_pop(mol, orth_coeff, dm)
print("Core orbital Mulliken populations:", popl)