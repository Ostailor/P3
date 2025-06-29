import os
import numpy as np
from pyscf import gto, scf

# Determine project root and paths
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
coord = os.path.join(root, 'data', 'coords', 'DBT_pubchem.xyz')
outdir = os.path.join(root, 'pyscf', 'outputs')
os.makedirs(outdir, exist_ok=True)

# 1. Build molecule and run RHF
mol = gto.M(atom=coord, basis='6-31g', unit='Angstrom', verbose=0)
mf = scf.RHF(mol).run()

# 2. Extract one‐electron integrals (h_core) and two‐electron AO integrals
h1 = mf.get_hcore()               # shape (nbasis, nbasis)
eri = mol.intor('int2e')          # shape (nbasis, nbasis, nbasis, nbasis)

# 3. Save to .npy files
np.save(os.path.join(outdir, 'h1_6-31g.npy'), h1)
np.save(os.path.join(outdir, 'eri_6-31g.npy'), eri)

print("Saved h1 and eri to pyscf/outputs/")
