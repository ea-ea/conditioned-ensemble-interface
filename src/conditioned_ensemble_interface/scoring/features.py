from __future__ import annotations
from typing import Dict, Tuple, List
import os
from Bio.PDB import PDBParser
import numpy as np

POSITIVE = {"LYS","ARG","HIS"}
NEGATIVE = {"ASP","GLU"}
HYDROPHOBIC = {"ALA","VAL","LEU","ILE","PRO","PHE","MET","TRP","TYR"}

def _load_structure(pdb_path: str):
    parser = PDBParser(QUIET=True)
    return parser.get_structure("pose", pdb_path)

def _atom_coords(atom):
    v = atom.get_vector()
    return np.array([float(v[0]), float(v[1]), float(v[2])], dtype=float)

def _first_model(structure):
    models = list(structure.get_models())
    return models[0] if models else None

def _inter_chain_atom_pairs(model) -> List[Tuple]:
    chains = list(model.get_chains())
    if len(chains) < 2:
        return []
    atoms_by_chain = []
    for ch in chains:
        atoms_by_chain.append([a for a in ch.get_atoms() if a.element != "H"])
    pairs = []
    for i in range(len(atoms_by_chain)):
        for j in range(i+1, len(atoms_by_chain)):
            for a in atoms_by_chain[i]:
                for b in atoms_by_chain[j]:
                    pairs.append((a, b))
    return pairs

def compute_interface_features(pose_path: str) -> dict:
    # Gracefully handle missing files
    if not os.path.exists(pose_path):
        return {"pose_path": pose_path, "missing_file": 1.0}

    try:
        structure = _load_structure(pose_path)
    except Exception:
        return {"pose_path": pose_path, "parse_error": 1.0}

    model = _first_model(structure)
    if model is None:
        return {"pose_path": pose_path, "no_models": 1.0}

    chains = list(model.get_chains())
    if len(chains) < 2:
        return {"pose_path": pose_path, "single_chain_or_no_atoms": 1.0}

    # Compute simple interface features
    pairs = _inter_chain_atom_pairs(model)
    if not pairs:
        return {"pose_path": pose_path, "no_interface_pairs": 1.0}

    contact_count = 0
    hydrophobic_contacts = 0
    salt_bridges = 0
    clashes = 0

    # centroid per chain
    chain_centroids = []
    for ch in chains:
        coords = np.array([_atom_coords(a) for a in ch.get_atoms() if a.element != "H"])
        if len(coords):
            chain_centroids.append(coords.mean(axis=0))
    centroid_distance = 0.0
    if len(chain_centroids) >= 2:
        centroid_distance = float(np.linalg.norm(chain_centroids[0] - chain_centroids[1]))

    for a, b in pairs:
        d = float(np.linalg.norm(_atom_coords(a) - _atom_coords(b)))
        if d < 2.0:
            clashes += 1
        if d <= 4.0:
            contact_count += 1
            ra = a.get_parent(); rb = b.get_parent()
            if ra.id[0] == " " and rb.id[0] == " ":
                if ra.get_resname() in HYDROPHOBIC and rb.get_resname() in HYDROPHOBIC:
                    hydrophobic_contacts += 1
                if (ra.get_resname() in POSITIVE and rb.get_resname() in NEGATIVE) or \
                   (rb.get_resname() in POSITIVE and ra.get_resname() in NEGATIVE):
                    salt_bridges += 1

    approx_buried_score = float(contact_count - 5.0 * clashes)

    return {
        "pose_path": pose_path,
        "contact_count_4A": float(contact_count),
        "hydrophobic_contacts": float(hydrophobic_contacts),
        "salt_bridges": float(salt_bridges),
        "clashes": float(clashes),
        "centroid_distance": float(centroid_distance),
        "approx_buried_score": approx_buried_score
    }

def condition_features(conditions: dict) -> dict:
    ph = float(conditions.get("pH", 7.4))
    ionic = float(conditions.get("ionic_strength", 0.15))
    has_cofactor = 1.0 if conditions.get("cofactor") else 0.0
    sulfation = float(conditions.get("glycosaminoglycan_sulfation_level", 0.0))
    return {
        "pH": ph,
        "ionic_strength": ionic,
        "has_cofactor": has_cofactor,
        "glycosaminoglycan_sulfation_level": sulfation
    }