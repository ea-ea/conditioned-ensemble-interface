#!/usr/bin/env python3
"""
Compute a docking box from ligand coordinates.
Tries: data/3ptb/ligand_BEN.pdb (HETATM) then data/3ptb/ligand_from_sdf.pdb (ATOM/HETATM).
If needed, it will generate ligand_from_sdf.pdb from ligand.sdf using Open Babel via conda run.
"""

import os, subprocess, sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
d = ROOT / "data" / "3ptb"
pdb_ben = d / "ligand_BEN.pdb"
sdf = d / "ligand.sdf"
pdb_from_sdf = d / "ligand_from_sdf.pdb"

def read_coords_pdb(path: Path, het_only: bool):
    coords = []
    if not path.exists():
        return coords
    with path.open() as fh:
        for L in fh:
            if het_only:
                if L.startswith("HETATM"):
                    coords.append([float(L[30:38]), float(L[38:46]), float(L[46:54])])
            else:
                if L.startswith(("ATOM", "HETATM")):
                    coords.append([float(L[30:38]), float(L[38:46]), float(L[46:54])])
    return coords

def ensure_pdb_from_sdf():
    if pdb_from_sdf.exists() and pdb_from_sdf.stat().st_size > 0:
        return
    if not sdf.exists() or sdf.stat().st_size == 0:
        print("[box] ligand.sdf missing; downloading BEN_ideal.sdf from RCSB...", file=sys.stderr)
        subprocess.run(
            ["curl", "-L", "https://files.rcsb.org/ligands/download/BEN_ideal.sdf", "-o", str(sdf)],
            check=True
        )
    print("[box] converting ligand.sdf → ligand_from_sdf.pdb with Open Babel...", file=sys.stderr)
    # call Open Babel via conda run -n dock
    subprocess.run(
        ["conda", "run", "-n", "dock", "obabel", str(sdf), "-O", str(pdb_from_sdf)],
        check=True
    )

def main():
    # 1) Try BEN-extracted PDB first
    coords = read_coords_pdb(pdb_ben, het_only=True)
    if coords:
        print("[box] using coordinates from ligand_BEN.pdb", file=sys.stderr)
    else:
        # 2) Ensure we have a PDB derived from SDF
        ensure_pdb_from_sdf()
        coords = read_coords_pdb(pdb_from_sdf, het_only=False)
        if coords:
            print("[box] using coordinates from ligand_from_sdf.pdb", file=sys.stderr)
    if not coords:
        print("[error] no ligand coordinates found in either file.", file=sys.stderr)
        sys.exit(1)

    c = np.array(coords, dtype=float)
    ctr = c.mean(axis=0)
    span = c.max(axis=0) - c.min(axis=0) + 8.0  # add 8 Å padding

    print(f"center_x={ctr[0]:.2f}")
    print(f"center_y={ctr[1]:.2f}")
    print(f"center_z={ctr[2]:.2f}")
    print(f"size_x={span[0]:.2f}")
    print(f"size_y={span[1]:.2f}")
    print(f"size_z={span[2]:.2f}")

if __name__ == "__main__":
    main()
