#!/usr/bin/env python3
"""
h2_energy_from_job.py – Post-processor for H₂ (0.74 Å, STO-3G)

• Online mode:   --job-id JOB_ID [--instance HUB/GROUP/PROJECT]
• Offline mode:  --json  path/to/PrimitiveResult.json
• Handles 14-value, 10-value, 1-value outputs + ZNE at resilience_level 2.
• Very verbose debug prints: every nested key in metadata["resilience"].

Author: 2025  MIT-licensed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np


###############################################################################
# 1. Build Hamiltonian & constant shift                                       #
###############################################################################
def h2_coeffs_noI(bond_len: float = 0.74) -> Tuple[np.ndarray, float]:
    import pennylane as qml

    symbols = ["H", "H"]
    coords = np.array([0.0, 0.0, 0.0, 0.0, 0.0, bond_len])
    H_full, _ = qml.qchem.molecular_hamiltonian(symbols, coords)

    coeffs, ops = H_full.terms()
    shift, coeff_noI = 0.0, []
    for c, o in zip(coeffs, ops):
        if isinstance(o, qml.Identity):
            shift += c
        else:
            coeff_noI.append(c)

    print(f"[debug]  Constant shift          : {shift:+.6f} Ha")
    print(f"[debug]  Non-identity term count : {len(coeff_noI)}")
    return np.array(coeff_noI, float), float(shift)


###############################################################################
# 2. Runtime helper                                                            #
###############################################################################
def _service(channel: str, instance: Optional[str]):
    from qiskit_ibm_runtime import QiskitRuntimeService

    order = (
        ["ibm_quantum_platform", "ibm_quantum", "ibm_cloud"]
        if channel == "auto"
        else [channel]
    )
    for ch in order:
        try:
            return QiskitRuntimeService(channel=ch, instance=instance)
        except Exception:
            print(f"[debug]  Channel '{ch}' not usable")
    raise RuntimeError("No valid channel found")


###############################################################################
# 3. Pretty-print nested dicts / lists                                         #
###############################################################################
def _dump_nested(prefix: str, node: Any, indent: int = 4, max_items: int = 6):
    spacer = " " * indent
    if isinstance(node, dict):
        for k, v in node.items():
            _dump_nested(f"{prefix}{k}/", v, indent + 2)
    elif isinstance(node, (list, np.ndarray)):
        head = node[:max_items]
        tail = " …" if len(node) > max_items else ""
        print(f"{spacer}{prefix}[list] len={len(node)} sample={head}{tail}")
    else:
        print(f"{spacer}{prefix}{node}")


###############################################################################
# 4. Collect PUB data, look for ZNE                                            #
###############################################################################
def _collect_pub(pubs):
    evs: List[float] = []
    stds: List[float] = []
    zne_val: Optional[float] = None
    zne_std: Optional[float] = None

    for idx, pub in enumerate(pubs):
        print(f"[debug]  PUB {idx}  evs={pub.data.evs}  stds={pub.data.stds}")
        evs.extend(pub.data.evs)
        stds.extend(pub.data.stds)

        meta: dict = getattr(pub, "metadata", {})
        print(f"[debug]  PUB {idx}  metadata keys → {list(meta.keys())}")

        # full resilience subtree
        if "resilience" in meta:
            print(f"[debug]  ----- resilience subtree (PUB {idx}) -----")
            _dump_nested("resilience/", meta["resilience"], indent=6)
            print(f"[debug]  -----------------------------------------")

        if "shots" in meta:
            print(f"[debug]  ----- circuit_metadata subtree (PUB {idx}) -----")
            _dump_nested("circuit_metadata/", meta["circuit_metadata"], indent=6)
            print(f"[debug]  -----------------------------------------")

        # legacy top-level zne block
        if "zne" in meta:
            print(f"[debug]  ----- top-level zne block (PUB {idx}) -----")
            _dump_nested("zne/", meta["zne"], indent=6)
            print(f"[debug]  -------------------------------------------")

        # Search both possible locations
        for block in (meta.get("zne"), meta.get("resilience", {}).get("zne")):
            if not block:
                continue
            val = block.get("extrapolated_expvals") or block.get("extrapolated_value")
            std = block.get("extrapolated_stddevs") or block.get("extrapolated_stddev")
            if val is not None:
                zne_val = float(val[0] if isinstance(val, list) else val)
                zne_std = float(
                    std[0] if isinstance(std, list) else (std or 0.0)
                )
                print("[debug]  ZNE energy found →", zne_val)
    return np.array(evs, float), np.array(stds, float), zne_val, zne_std


###############################################################################
# 5. Load from job or JSON                                                    #
###############################################################################
def load_from_job(job_id: str, channel: str, instance: Optional[str]):
    svc = _service(channel, instance)
    job = svc.job(job_id)
    print("[debug]  Job status:", job.status())
    return _collect_pub(job.result())


def load_from_json(path: Path):
    prim = json.loads(path.read_text())

    class _FakePub:
        def __init__(self, d):
            self.data = type("Data", (), d["data"])
            self.metadata = d.get("metadata", {})

    return _collect_pub([_FakePub(p) for p in prim["pub_results"]])


###############################################################################
# 6. Assemble best energy                                                     #
###############################################################################
def assemble(
    coeffs: np.ndarray,
    evs: np.ndarray,
    stds: np.ndarray,
    shift: float,
    zne_val: Optional[float],
    zne_std: Optional[float],
):
    if zne_val is not None:
        label, e_elec, sigma = "ZNE", zne_val, zne_std or 0.0
    elif evs.size == coeffs.size:
        label = "raw-14"
        e_elec = float(np.dot(coeffs, evs))
        sigma = float(np.linalg.norm(coeffs * stds))
    elif evs.size > 1:
        label = f"group-{evs.size}"
        e_elec = float(evs.sum())
        sigma = float(np.linalg.norm(stds))
    else:
        label = "single-val"
        e_elec = float(evs[0]) - shift
        sigma = float(stds[0])
    return e_elec, e_elec + shift, sigma, label


###############################################################################
# 7. CLI                                                                      #
###############################################################################
def main():
    p = argparse.ArgumentParser(
        description="Post-process IBM Runtime Estimator results for H₂ (0.74 Å, STO-3G)."
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--job-id", help="Runtime job ID")
    src.add_argument("--json", help="PrimitiveResult JSON")
    p.add_argument("--instance", help="hub/group/project string")
    p.add_argument("--channel", default="ibm_cloud", help="'auto', 'ibm_cloud', …")
    args = p.parse_args()

    coeffs, shift = h2_coeffs_noI()

    if args.job_id:
        evs, stds, zne_val, zne_std = load_from_job(
            args.job_id, args.channel, args.instance
        )
    else:
        evs, stds, zne_val, zne_std = load_from_json(Path(args.json))

    e_elec, e_tot, sigma, tag = assemble(
        coeffs, evs, stds, shift, zne_val, zne_std
    )

    print("\n── H₂ (0.74 Å, STO-3G) energy ──")
    print(f"Best estimate ({tag}) : {e_tot:+.6f}  ± {sigma:.6f}  Ha (BO)")
    print(f"Electronic part       : {e_elec:+.6f}  Ha")
    print(f"Constant shift (R_n)  : {shift:+.6f}  Ha")
    print(f"Observable count      : {evs.size}")
    if zne_val is not None:
        print("Mitigation            : Zero-Noise Extrapolation (RL 2)")
    src_line = f"online job {args.job_id}" if args.job_id else f"file {args.json}"
    print("Data source          :", src_line)


if __name__ == "__main__":
    main()
