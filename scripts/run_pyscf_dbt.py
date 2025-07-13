#!/usr/bin/env python3
import argparse
import numpy as np
import os
from pyscf import gto, scf, mcscf

def main(basis):
    # Create necessary directories if they don't exist
    os.makedirs('integrals', exist_ok=True)
    os.makedirs('pyscf_logs', exist_ok=True)
    
    # 1) Build the molecule
    mol = gto.Mole()
    mol.atom = 'dbt_geometry/dbt_opt_opt.xyz'
    mol.basis = basis
    mol.verbose = 4
    mol.build()

    # 2) RHF reference
    mf = scf.RHF(mol)
    rhf_energy = mf.kernel()
    print(f"RHF/{basis} energy = {rhf_energy:.6f} Ha")

    # 3) Active space: 8e, 8o (orbitals 45–52 → Python indices 44–51)
    nocc = mol.nelectron // 2
    cas_start = 44
    ncas = 8
    mc = mcscf.CASSCF(mf, ncas, ncas)
    mc.frozen = cas_start  # freeze MOs 0–43
    mc.kernel()

    # 4) Extract integrals in active space
    h1, ecore = mc.get_h1cas()          # one-electron integrals and core energy
    eri = mc.get_h2cas()                # two-electron integrals (chemist's notation)

    # 5) Save to disk
    np.save('integrals/h1_act.npy', h1)
    np.save('integrals/eri_act.npy', eri)
    np.save('integrals/ecore_act.npy', np.array([ecore]))
    print("Saved h1_act.npy, eri_act.npy, and ecore_act.npy in integrals/")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--basis', choices=['sto-3g','6-31g'], default='sto-3g')
    args = p.parse_args()
    main(args.basis)
