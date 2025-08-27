
from __future__ import annotations
from typing import Dict, Any
import os
from Bio.PDB import PDBParser

def basic_pose_checks(pdb_path: str, min_atoms_per_chain: int = 2) -> Dict[str, Any]:
    """Lightweight physical sanity checks.
    Returns dict with boolean flags and a final 'pass' field.
    - file_exists
    - parsed_ok
    - n_chains
    - atoms_per_chain_ok
    - two_chain_interface_ok (needs >=2 chains)
    """
    out = {
        "file_exists": os.path.exists(pdb_path),
        "parsed_ok": False,
        "n_chains": 0,
        "atoms_per_chain_ok": False,
        "two_chain_interface_ok": False,
        "pass": False,
    }
    if not out["file_exists"]:
        return out
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("pose", pdb_path)
    except Exception:
        return out
    out["parsed_ok"] = True
    models = list(structure.get_models())
    if not models:
        return out
    model = models[0]
    chains = list(model.get_chains())
    out["n_chains"] = len(chains)
    if len(chains) < 2:
        return out
    atoms_ok = True
    for ch in chains:
        n = sum(1 for _ in ch.get_atoms())
        if n < min_atoms_per_chain:
            atoms_ok = False
            break
    out["atoms_per_chain_ok"] = atoms_ok
    out["two_chain_interface_ok"] = len(chains) >= 2
    out["pass"] = out["parsed_ok"] and out["two_chain_interface_ok"] and out["atoms_per_chain_ok"]
    return out
