Phase 1 Workflow

Overview

Phase 1 establishes the molecular Hamiltonian for dibenzothiophene (DBT) by preparing a reliable 3D geometry, validating an active space of 8 electrons in 8 orbitals (8e,8o), and generating one- and two-electron integrals using PySCF.

Environment Setup

cd phase1_hamiltonian
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

(Dependencies are pinned in requirements.txt.)

Directory Structure

phase1_hamiltonian/
├── dbt_geometry/        # Raw and optimized geometry files
├── pyscf_inputs/        # (unused—scripts in `scripts/` folder)
├── pyscf_logs/          # Log files for SCF, geometry opt, orbital analysis
├── integrals/           # h1_act.npy, eri_act.npy
├── docs/                # This workflow document
├── visualization/       # Orbital isosurface images
└── scripts/             # Helper and driver scripts

Geometry Preparation

Download raw SDF from PubChem (CID 7841) as dbt_geometry/dbt_raw.sdf.

Convert to XYZ:

obabel dbt_geometry/dbt_raw.sdf -O dbt_geometry/dbt_opt.xyz --gen3d

Optimize geometry using PySCF HF/STO-3G:

python scripts/geom_opt_pyscf.py > pyscf_logs/dbt_geoopt.log 2>&1

Output: dbt_geometry/dbt_opt_opt.xyz (
Final SCF energy: –846.71385112 Ha
)

Orbital Analysis

Run orbital energies:

python scripts/analyze_orbitals.py > pyscf_logs/orbital_energies.log

Key output:

MO 48 (HOMO): –0.072929 Ha
MO 49 (LUMO):  +0.076416 Ha

Mulliken populations confirm core MOs (1–44) each have ~2 electrons and remain chemically inert.

Integral Generation

Run CAS(8,8) driver for both basis sets:

python scripts/run_pyscf_dbt.py --basis sto-3g   > pyscf_logs/dbt_STO3G.log 2>&1
python scripts/run_pyscf_dbt.py --basis 6-31g    > pyscf_logs/dbt_6-31G.log 2>&1

Outputs saved in integrals/:

h1_act.npy (one-electron integrals)

eri_act.npy (two-electron integrals)