#!/usr/bin/env python3
import argparse, glob
from rdkit import Chem
from rdkit.Chem import AllChem

def load_any(path):
    if path.lower().endswith((".sdf",".sd")):
        return Chem.SDMolSupplier(path, removeHs=False)[0]
    return Chem.MolFromPDBFile(path, removeHs=False)

def best_rmsd(native_path, poses_glob, topk=5):
    nat = load_any(native_path)
    if nat is None:
        raise RuntimeError(f"Failed to load native: {native_path}")
    res=[]
    for p in sorted(glob.glob(poses_glob)):
        mol = load_any(p)
        if mol is None: 
            continue
        if mol.GetNumAtoms()==nat.GetNumAtoms():
            r = AllChem.GetBestRMS(nat, mol)
            res.append((p, r))
    res.sort(key=lambda x: x[1])
    return res[:topk], res[0][1] if res else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", required=True)
    ap.add_argument("--poses_glob", required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=2.0, help="success Å")
    args = ap.parse_args()
    top, best = best_rmsd(args.native, args.poses_glob, args.topk)
    if not top:
        print("No comparable poses (atom count mismatch or load failure)."); return
    print("Best pose:", top[0])
    print("Top-k list:", top)
    successes = sum(1 for _,r in top if r <= args.threshold)
    print(f"Top-{args.topk} success @ {args.threshold} Å: {successes}/{len(top)}")
if __name__ == "__main__":
    main()
