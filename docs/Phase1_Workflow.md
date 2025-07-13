Phase 0 & 1 Workflow

Overview

Phase 0 prepares the molecular geometry for dibenzothiophene (DBT) from raw structural data.
Phase 1 establishes the molecular Hamiltonian by validating an active space of 8 electrons in 8 orbitals (8e,8o), and generating one- and two-electron integrals using PySCF.

Environment Setup

cd /Users/omtailor/P3
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

(Dependencies are pinned in requirements.txt.)

Directory Structure

P3/
├── dbt_geometry/        # Raw and optimized geometry files
├── pyscf_logs/          # Log files for SCF, geometry opt, orbital analysis
├── integrals/           # h1_act.npy, eri_act.npy, ecore_act.npy
├── docs/                # This workflow document
├── inputs/              # Processed molecular data files
└── scripts/             # Helper and driver scripts

### Phase 0: Geometry Preparation

**1. Download Raw Structure Data**
Download raw SDF from PubChem (CID 7841) as `dbt_geometry/dbt_raw.sdf`.
(This file is already present in the repository)

**2. Convert SDF to XYZ Format**
```bash
obabel dbt_geometry/dbt_raw.sdf -O dbt_geometry/dbt_opt.xyz --gen3d
```
- **Purpose**: Converts SDF format to XYZ with 3D coordinates
- **Runtime**: ~1 second
- **Output**: `dbt_geometry/dbt_opt.xyz` - Initial 3D geometry

**3. Optimize Geometry using PySCF**
```bash
python scripts/geom_opt_pyscf.py > pyscf_logs/dbt_geoopt.log 2>&1
```
- **Purpose**: Optimizes geometry using HF/STO-3G level of theory
- **Runtime**: ~30 seconds
- **Output**: `dbt_geometry/dbt_opt_opt.xyz` - Optimized geometry
- **Expected Final SCF energy**: –846.71385112 Ha

### Phase 1: Molecular Setup and Classical Calculations

**1. Analyze Orbital Structure**
```bash
python scripts/analyze_orbitals.py > pyscf_logs/orbital_energies.log
```
- **Purpose**: Identifies HOMO/LUMO and validates active space selection
- **Runtime**: ~10 seconds
- **Expected Output:**
```
MO 48 (HOMO): –0.072929 Ha
MO 49 (LUMO):  +0.076416 Ha
```

**2. Generate Molecular Integrals**
```bash
python scripts/run_pyscf_dbt.py --basis sto-3g > pyscf_logs/dbt_STO3G.log 2>&1
python scripts/run_pyscf_dbt.py --basis 6-31g > pyscf_logs/dbt_6-31G.log 2>&1
```
- **Purpose**: Runs CASSCF(8e,8o) calculations and extracts active space integrals
- **Runtime**: ~2-3 minutes per basis set
- **Outputs**: 
  - `integrals/h1_act.npy` - One-electron integrals
  - `integrals/eri_act.npy` - Two-electron integrals  
  - `integrals/ecore_act.npy` - Core energy
- **Expected Output:**
```
HF/sto-3g energy = -845.832329 Ha
CASSCF (8e,8o) converged
Saved h1_act.npy, eri_act.npy, and ecore_act.npy in integrals/
```