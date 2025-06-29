import numpy as np
from pyscf import gto, scf

# 1. Load molecule
mol = gto.M(atom='data/coords/DBT_pubchem.xyz', basis='6-31g', unit='Angstrom', verbose=0)
# 2. RHF
mf = scf.RHF(mol).run()

# 3. MO energies & print with indices
print("Index   Energy (Ha)")
for i, e in enumerate(mf.mo_energy):
    print(f"{i:>5}   {e:>10.6f}")
