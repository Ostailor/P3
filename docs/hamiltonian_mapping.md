# Hamiltonian Mapping & Qubit Encoding

This file summarizes the qubit-encoding of the 8e, 8o spin‐orbital active‐space Hamiltonian for DBT.

## Active Space

- **Electrons**: 8 (4 α, 4 β)  
- **Spatial orbitals**: 8 (MOs 44–51)  
- **Spin-orbitals**: 16  

## Mapping Results

| Mapping Method             | Qubits | Pauli Terms | Notes                                 |
|----------------------------|-------:|------------:|:--------------------------------------|
| Jordan–Wigner (JW)         |     16 |        5 413 | Standard JW on 16 spin‐orbitals       |
| Bravyi–Kitaev (BK)         |     16 |        5 413 | Standard BK on 16 spin‐orbitals       |
| Parity + 2-qubit reduction |     14 |        5 413 | Parity mapping with two‐qubit tapering |

## Files

- `es_problem.pkl`                    — Qiskit Nature `ElectronicStructureProblem`  
- `spinorb_fermionic_op.pkl`          — spin‐orbital `FermionicOp`  
- `active_spinorb_jw_op.pkl`          — JW‐mapped `SparsePauliOp`  
- `active_spinorb_bk_op.pkl`          — BK‐mapped `SparsePauliOp`  
- `active_spinorb_pr_op.pkl`          — Parity (+2q) `SparsePauliOp`  

