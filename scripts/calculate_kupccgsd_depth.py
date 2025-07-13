#!/usr/bin/env python3
"""
Calculate the actual circuit depth for k-UpCCGSD VQE
"""

import os
import numpy as np
import pennylane as qml
from qiskit import QuantumCircuit
from pennylane import qchem

def load_kupccgsd_circuit(params, n_qubits, k_reps=2):
    """Create k-UpCCGSD circuit similar to ibm_nofireopal approach."""
    ELECTRONS = 8
    
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev)
    def kupccgsd_qnode(weights):
        # Hartree-Fock state preparation
        hf = qchem.hf_state(ELECTRONS, n_qubits)
        qml.BasisState(hf, wires=range(n_qubits))
        
        # k-UpCCGSD ansatz
        qml.kUpCCGSD(
            weights,
            wires=range(n_qubits),
            k=k_reps,
            delta_sz=0,
            init_state=hf
        )
        return qml.state()
    
    # Get the quantum circuit as QASM
    qasm = qml.workflow.construct_tape(kupccgsd_qnode)(params).to_openqasm(False)
    qc = QuantumCircuit.from_qasm_str(qasm)
    qc.barrier()
    
    return qc, kupccgsd_qnode

def main():
    # Load k-UpCCGSD parameters
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    kupccgsd_dir = os.path.join(root, "results/advanced_benchmarking/kupccgsd_vqe/adam")
    
    params = np.load(os.path.join(kupccgsd_dir, "final_params.npy"))
    n_qubits = 14  # From your logs
    k_reps = 2     # From your log.json
    
    print(f"Loaded k-UpCCGSD parameters: {len(params)} parameters")
    print(f"Qubits: {n_qubits}, k repetitions: {k_reps}")
    
    # Create the circuit
    circuit, qnode = load_kupccgsd_circuit(params, n_qubits, k_reps)
    
    # Calculate actual circuit depth
    actual_depth = circuit.depth()
    
    print(f"\nk-UpCCGSD Circuit Analysis:")
    print(f"Total gates: {len(circuit.data)}")
    print(f"Actual circuit depth: {actual_depth}")
    print(f"Circuit width: {circuit.num_qubits}")
    
    # Compare with ADAPT-VQE
    print(f"\nComparison:")
    print(f"k-UpCCGSD depth:  {actual_depth} layers")
    print(f"ADAPT-VQE depth:  41 layers")
    print(f"Depth ratio:      {41/actual_depth:.1f}x shallower (k-UpCCGSD)")
    
    # Save circuit info
    circuit_info = {
        "algorithm": "k-UpCCGSD",
        "parameters": len(params),
        "qubits": n_qubits,
        "k_repetitions": k_reps,
        "total_gates": len(circuit.data),
        "circuit_depth": actual_depth,
        "circuit_width": circuit.num_qubits
    }
    
    os.makedirs(os.path.join(kupccgsd_dir, "circuit_analysis"), exist_ok=True)
    with open(os.path.join(kupccgsd_dir, "circuit_analysis", "depth_analysis.json"), "w") as f:
        import json
        json.dump(circuit_info, f, indent=2)
    
    print(f"\nCircuit analysis saved to: {kupccgsd_dir}/circuit_analysis/")
    
    # Optional: Save circuit diagram
    try:
        with open(os.path.join(kupccgsd_dir, "circuit_analysis", "circuit.qasm"), "w") as f:
            f.write(circuit.qasm())
        print("Circuit QASM saved for inspection")
    except:
        print("Could not save circuit QASM")

if __name__ == "__main__":
    main()