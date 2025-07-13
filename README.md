# Quantum Chemistry Simulation of Dibenzothiophene for Asphalt Aging Analysis

This repository contains a complete quantum chemistry pipeline for analyzing dibenzothiophene (DBT), a key sulfur-containing compound in asphalt binders. The implementation uses Variational Quantum Eigensolver (VQE) algorithms to understand oxidation resistance mechanisms critical to pavement durability.

## Overview

The project demonstrates quantum advantage in quantum chemistry through:
- **k-UpCCGSD VQE**: Superior accuracy (-864.69 Ha) with deep circuits (9,398 layers)
- **ADAPT-VQE**: Hardware-compatible shallow circuits (41 layers) with good accuracy (-857.89 Ha)
- **Classical benchmarking**: HF, DFT, and CASSCF comparisons
- **Hardware validation**: IBM Quantum execution with error mitigation

## Prerequisites and Installation

### System Requirements
- **Operating System**: macOS, Linux, or Windows
- **Python**: 3.8 or higher
- **Memory**: 8+ GB RAM recommended for classical simulations
- **Storage**: 2+ GB for results and intermediate files

### Environment Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd P3
```

2. **Create a virtual environment (recommended):**
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

The requirements.txt includes all necessary packages:
- **Quantum Computing**: PennyLane (0.41.1), Qiskit (1.2.4), qiskit-ibm-runtime (0.29.0)
- **Classical Chemistry**: PySCF (2.9.0), OpenFermion (1.7.1)
- **Scientific Computing**: NumPy (2.3.1), SciPy (1.16.0), Matplotlib (3.10.3)
- **Hardware Access**: IBM Quantum providers and Fire Opal integration

### Environment Variables

**For IBM Quantum Hardware Access (Optional):**

Set up IBM Quantum credentials for hardware execution:

```bash
# Option 1: Environment variables
export QISKIT_IBM_TOKEN="your_ibm_quantum_token"
export QISKIT_IBM_INSTANCE="hub/group/project"  # e.g., "ibm-q/open/main"

# Option 2: Save credentials permanently
python scripts/setup_ibmq_account.py
```

**Runtime Logging (Optional):**
```bash
export QISKIT_IBM_RUNTIME_LOG_LEVEL="INFO"  # DEBUG, INFO, WARNING, ERROR
```

## Project Structure

```
P3/
├── scripts/                    # Main execution scripts
│   ├── run_pyscf_dbt.py       # Classical chemistry setup
│   ├── kupccgsd_vqe.py        # High-accuracy quantum VQE
│   ├── adapt_vqe.py           # Hardware-compatible VQE
│   ├── benchmark_methods.py   # Classical comparisons
│   └── ibm_nofireopal.py      # Hardware execution
├── inputs/                     # Molecular data and Hamiltonians
├── results/                    # VQE simulation outputs
├── benchmark_results/          # Classical method comparisons
├── docs/                       # Analysis documentation
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Step-by-Step Execution Guide

### Phase 0: Geometry Preparation

**1. Download Raw Structure Data**
The raw SDF file from PubChem (CID 7841) is already present as `dbt_geometry/dbt_raw.sdf`.

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

**3. Analyze Molecular Orbitals (Alternative)**
```bash
python scripts/analyze_orbitals.py
```
- **Purpose**: Alternative orbital analysis for validation
- **Runtime**: ~5 seconds
- **Outputs**: Orbital energy analysis confirming MOs 45-52 for active space

### Phase 2: Quantum Hamiltonian Preparation

**3. Map to Quantum Hamiltonian**
```bash
python scripts/map_to_qubit_hamiltonian.py
```
- **Purpose**: Converts molecular Hamiltonian to qubit operators using Bravyi-Kitaev mapping
- **Runtime**: ~45 seconds
- **Outputs**: 
  - `inputs/bk_symm_tapered.pkl` - Z₂-tapered Hamiltonian (14 qubits, 129 terms)
  - `docs/phase2_mapping_results.md` - Mapping analysis

**Expected Output:**
```
| Mapping Method  | Qubits | Pauli Terms | Constant Energy    |
|-----------------|--------|-------------|-------------------|
| BK (Z₂-tapered) | 14     | 129         | –855.6109991074851|
```

### Phase 3: Quantum VQE Simulations

**4. Run k-UpCCGSD VQE (High Accuracy)**
```bash
python scripts/kupccgsd_vqe.py
```
- **Purpose**: Executes structured VQE with k-UpCCGSD ansatz
- **Runtime**: ~2 minutes
- **Memory**: ~2 GB peak usage
- **Outputs**: 
  - `results/advanced_benchmarking/kupccgsd_vqe/adam/log.json` - Final energy (-864.69 Ha)
  - `energy_history.npy` - Convergence data (114 iterations)
  - `final_params.npy` - Optimized parameters (252 total)

**Expected Output:**
```
Final energy: -864.690626 Ha
Parameters: 252
Runtime: 112.43 seconds
Convergence: 114 iterations
```

**5. Run ADAPT-VQE (Hardware Compatible)**
```bash
python scripts/adapt_vqe.py
```
- **Purpose**: Executes adaptive VQE with operator selection
- **Runtime**: ~30 seconds
- **Outputs**: 
  - `results/advanced_benchmarking/adapt_vqe/adam/` - Results directory
  - `selected_ops.txt` - Selected excitation operators
  - `params.npy` - Optimized parameters (2 total)
  - Final energy: -857.89 Ha

**Expected Output:**
```
Selected operators: 2
Final energy: -857.892352 Ha
Selected operators:
  FermionicDouble[6, 7]+[11, 12]
  FermionicDouble[5, 6]+[11, 12]
```

### Phase 4: Classical Benchmarking

**6. Run Classical Method Comparisons**
```bash
python scripts/benchmark_methods.py
```
- **Purpose**: Executes HF, DFT (B3LYP), and CASSCF for comparison
- **Runtime**: ~1 minute
- **Outputs**: 
  - `benchmark_results/method_comparison.json` - Energy comparisons
  - Console output with correlation energy analysis

**7. Compare Quantum vs Classical Results**
```bash
python scripts/compare_with_benchmarks.py
```
- **Purpose**: Analyzes quantum advantage and correlation recovery
- **Runtime**: ~5 seconds
- **Outputs**: Quantitative comparison showing VQE's advantages

**Expected Output:**
```
QUANTUM vs CLASSICAL COMPARISON
==================================================
Classical Methods (Full System):
HF energy:              -845.832329 Ha
DFT (B3LYP) energy:     -849.454736 Ha
CASSCF energy:          -845.934396 Ha

Quantum VQE (Active Space + Constant):
VQE correlation energy: -9.079627 Ha
VQE vs DFT:             2.5x better correlation recovery
VQE vs CASSCF:          89x better correlation recovery
```

### Phase 5: Circuit Depth Analysis

**8. Analyze k-UpCCGSD Circuit Depth**
```bash
python scripts/calculate_kupccgsd_depth.py
```
- **Purpose**: Determines actual circuit complexity for hardware assessment
- **Runtime**: ~30 seconds
- **Outputs**: 
  - `results/advanced_benchmarking/kupccgsd_vqe/adam/circuit_analysis/depth_analysis.json`
  - Reveals 9,398 layers, 15,375 gates (too deep for hardware)

**Expected Output:**
```
k-UpCCGSD Circuit Analysis:
Total gates: 15375
Actual circuit depth: 9398
Circuit width: 14
```

**9. Analyze ADAPT-VQE Circuit Depth**
```bash
python scripts/calculate_adapt_depth.py
```
- **Purpose**: Confirms hardware-compatible circuit complexity
- **Runtime**: ~10 seconds
- **Outputs**: 
  - `results/advanced_benchmarking/adapt_vqe/adam/circuit_analysis/depth_analysis.json`
  - Shows 41 layers (229x shallower than k-UpCCGSD)

**Expected Output:**
```
ADAPT-VQE Circuit Analysis:
Selected operators: 2
Hardware log depth: 41 layers
Depth ratio: 229.2x deeper (k-UpCCGSD)
Hardware advantage: ADAPT-VQE is 229x shallower!
```

### Phase 6: Hardware Execution (Optional - Requires IBM Quantum Access)

**10. Setup IBM Quantum Credentials (First Time Only)**
```bash
python scripts/setup_ibmq_account.py
```
- **Purpose**: Saves IBM Quantum credentials for hardware access
- **Requirements**: IBM Quantum Network account and token

**11. IBM Quantum Hardware Testing**
```bash
python scripts/ibm_nofireopal.py --shots 10000 --backend ibm_brisbane --resilience_level 2
```
- **Purpose**: Validates ADAPT-VQE on real quantum hardware
- **Runtime**: ~10-30 minutes (depending on queue)
- **Requirements**: IBM Quantum Network access credentials
- **Options**:
  - `--shots`: Number of measurements per Pauli term (default: 10000)
  - `--backend`: IBM backend name (ibm_brisbane, ibm_torino, etc.)
  - `--resilience_level`: Error mitigation level (0=none, 1=basic, 2=advanced)
  - `--channel`: 'ibm_quantum' or 'ibm_cloud' (default: ibm_quantum)
- **Outputs**: 
  - `ibm_qpu_and_simulator.log` - Hardware execution logs
  - Expectation values with error bars across 129 Hamiltonian terms

**Expected Output:**
```
Backend: ibm_brisbane (133 qubits)
Circuit depth: 41 layers
Total measurements: 1,290,000 shots
Statistical uncertainties: σ=0.001-0.010 Ha
Resilience level 2 results show improved consistency
```

## Key Results Summary

| Method | Energy (Ha) | Correlation (Ha) | Circuit Depth | Hardware Compatible |
|--------|-------------|------------------|---------------|-------------------|
| HF | -845.83 | 0.00 (ref) | N/A | N/A |
| DFT (B3LYP) | -849.45 | -3.62 | N/A | N/A |
| CASSCF | -845.93 | -0.10 | N/A | N/A |
| k-UpCCGSD VQE | -864.69 | -9.08 | 9,398 layers | ❌ |
| ADAPT-VQE | -857.89 | -3.28 | 41 layers | ✅ |

## Key Findings

1. **Quantum Advantage**: VQE recovers 9.08 Ha correlation energy (2.5x better than DFT)
2. **Hardware Trade-off**: ADAPT-VQE enables hardware execution with 229x shallower circuits
3. **Industrial Relevance**: Results guide oxidation-resistant asphalt formulation design
4. **Algorithm Selection**: Circuit depth, not parameter count, limits NISQ implementation

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# Check PennyLane installation
python -c "import pennylane as qml; print(qml.__version__)"
```

**2. Memory Issues**
```bash
# Reduce k-repetitions for k-UpCCGSD if memory limited
# Edit kupccgsd_vqe.py: K_REPS = 1  # instead of 2
```

**3. IBM Quantum Access Issues**
```bash
# Verify credentials
python -c "from qiskit_ibm_runtime import QiskitRuntimeService; QiskitRuntimeService().backends()"

# Re-setup account if needed
python scripts/setup_ibmq_account.py
```

**4. PySCF Convergence Issues**
```bash
# Check molecular geometry in dbt_geometry/
# Ensure basis set compatibility (STO-3G is minimal but stable)
```

### Performance Optimization

**For Faster Classical Simulations:**
- Use smaller basis sets (STO-3G instead of 6-31G)
- Reduce active space size (6e,6o instead of 8e,8o)
- Use fewer k-repetitions in k-UpCCGSD

**For Hardware Execution:**
- Start with fewer shots (1000-5000) for testing
- Use resilience_level=1 for faster execution
- Monitor IBM backend queue times

## File Descriptions

### Core Scripts
- `run_pyscf_dbt.py` - Molecular setup and classical chemistry calculations
- `kupccgsd_vqe.py` - High-accuracy quantum VQE implementation  
- `adapt_vqe.py` - Hardware-compatible adaptive quantum VQE
- `map_to_qubit_hamiltonian.py` - Quantum Hamiltonian preparation
- `benchmark_methods.py` - Classical method comparisons

### Analysis Scripts
- `analyze_orbitals.py` - Frontier orbital identification
- `compare_with_benchmarks.py` - Quantum vs classical analysis
- `calculate_kupccgsd_depth.py` - Circuit complexity analysis
- `calculate_adapt_depth.py` - Hardware compatibility assessment

### Hardware Scripts
- `ibm_nofireopal.py` - IBM Quantum hardware execution
- `setup_ibmq_account.py` - IBM Quantum credential setup
- `baseline_vqe.py` - Alternative VQE implementation

### Utility Scripts
- `analyze_vqe_results.py` - Result visualization and analysis
- `geom_opt_pyscf.py` - Molecular geometry optimization

## Output Directories

- `inputs/` - Molecular data and prepared Hamiltonians
- `results/advanced_benchmarking/` - VQE simulation results
- `benchmark_results/` - Classical method comparisons
- `docs/` - Analysis documentation and workflow guides
- `pyscf_logs/` - Classical calculation logs
- `logs/` - General execution logs

## Hardware Requirements

- **Classical Simulation**: 
  - Standard laptop (8+ GB RAM recommended)
  - Runtime: 5-10 minutes total for all classical steps
- **Quantum Hardware**: 
  - IBM Quantum Network access for hardware validation
  - Queue times: 5-60 minutes depending on backend
- **Circuit Limits**: 
  - ADAPT-VQE (41 layers) executable on current NISQ devices
  - k-UpCCGSD (9,398 layers) requires fault-tolerant devices

## Research Applications

This codebase enables research in:
- **Quantum Chemistry**: Correlation energy recovery and molecular simulation
- **Materials Science**: Asphalt aging mechanisms and additive design
- **Quantum Computing**: NISQ algorithm development and hardware benchmarking
- **Industrial Applications**: Infrastructure durability and sustainability

## Citation

If you use this code, please cite the accompanying research paper:
```
Om Tailor. "Phase 3: Quantum Chemistry Simulation of Dibenzothiophene for Asphalt Aging Analysis." 
University of Maryland, Department of Computer Science, 2025.
```

## License

MIT License - See LICENSE file for details

## Contact

For questions or issues:
- **Author**: Om Tailor
- **Email**: otailor@terpmail.umd.edu
- **Institution**: University of Maryland, Department of Computer Science
- **LinkedIn**: https://www.linkedin.com/in/om-tailor-02b793226/

## Acknowledgments

- IBM Quantum Network for hardware access
- PennyLane and Qiskit development teams
- OpenFermion and PySCF communities
- QBraid and Aqora
- 
