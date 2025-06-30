#!/usr/bin/env python3
import os
import pickle
import numpy as np
from pyscf import ao2mo

from openfermion import InteractionOperator, get_fermion_operator
from openfermion.transforms import jordan_wigner, bravyi_kitaev
from openfermion.transforms.opconversions.remove_symmetry_qubits import (
    symmetry_conserving_bravyi_kitaev,
)
from openfermion.utils import count_qubits

# ─── Paths ─────────────────────────────────────────────────────────────
HERE    = os.path.abspath(os.path.dirname(__file__))
INT_DIR = os.path.join(HERE, '..', 'integrals')
OUT_DIR = os.path.join(HERE, '..', 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Helper: spatial → spin expansion ───────────────────────────────────
def spatial_to_spin(h1s, eris):
    n = h1s.shape[0]; m = 2 * n
    h1 = np.zeros((m, m)); eri = np.zeros((m, m, m, m))
    # one-body
    for p in range(n):
        for q in range(n):
            h1[p, q] = h1s[p, q]
            h1[p+n, q+n] = h1s[p, q]
    # two-body
    for p in range(n):
        for q in range(n):
            for r in range(n):
                for s in range(n):
                    v = eris[p, q, r, s]
                    eri[p,   q,   r,   s]   = v
                    eri[p+n, q+n, r+n, s+n] = v
                    eri[p+n, q+n, r,   s]   = v
                    eri[p,   q,   r+n, s+n] = v
    return h1, eri

# ─── Load & restore integrals ──────────────────────────────────────────
h1_spatial = np.load(os.path.join(INT_DIR, 'h1_act.npy'))
eri_packed  = np.load(os.path.join(INT_DIR, 'eri_act.npy'))
ecore       = float(np.load(os.path.join(INT_DIR, 'ecore_act.npy')).item())

n_orb       = h1_spatial.shape[0]
eri_spatial = ao2mo.restore(1, eri_packed, n_orb)
h1_spin, eri_spin = spatial_to_spin(h1_spatial, eri_spatial)

# ─── Build Fermionic Hamiltonian ───────────────────────────────────────
fermion_ham = InteractionOperator(ecore, h1_spin, eri_spin)
ferm_op     = get_fermion_operator(fermion_ham)

# ─── Untapered JW & BK for comparison ──────────────────────────────────
jw_op = jordan_wigner(ferm_op)
print("Untapered JW → qubits:", count_qubits(jw_op),
      ", terms:", len(jw_op.terms),
      ", shift:", jw_op.terms.get((), 0.0).real)
with open(os.path.join(OUT_DIR, 'jw_hamiltonian.pkl'), 'wb') as f:
    pickle.dump(jw_op, f)

bk_op = bravyi_kitaev(ferm_op)
print("Untapered BK → qubits:", count_qubits(bk_op),
      ", terms:", len(bk_op.terms),
      ", shift:", bk_op.terms.get((), 0.0).real)
with open(os.path.join(OUT_DIR, 'bk_hamiltonian.pkl'), 'wb') as f:
    pickle.dump(bk_op, f)

# ─── BK mapping + Z₂ taper in one call ────────────────────────────────
# This will map via BK and remove symmetry qubits
# symmetry_conserving_bravyi_kitaev(op, n_qubits, n_fermions)
tapered_bk = symmetry_conserving_bravyi_kitaev(
    ferm_op,
    2 * n_orb,   # total spin-orbitals (16)
    8            # number of active electrons
)
print("Tapered BK → qubits:", count_qubits(tapered_bk),
      ", terms:", len(tapered_bk.terms),
      ", shift:", tapered_bk.terms.get((), 0.0).real)
with open(os.path.join(OUT_DIR, 'bk_symm_tapered.pkl'), 'wb') as f:
    pickle.dump(tapered_bk, f)

print("Done.")
