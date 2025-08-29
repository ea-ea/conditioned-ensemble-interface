#!/usr/bin/env python3
import argparse, glob, json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Create JSONL dataset from complex PDBs")
    ap.add_argument("--id", default="example", help="dataset item id")
    ap.add_argument("--native", required=True, help="path to native complex PDB")
    ap.add_argument("--glob", required=True, help="glob for docked complex PDBs")
    ap.add_argument("--out", required=True, help="output JSONL file")
    ap.add_argument("--pH", type=float, default=7.4)
    ap.add_argument("--ionic_strength", type=float, default=0.15)
    args = ap.parse_args()

    poses = [args.native] + sorted(glob.glob(args.glob))
    if not poses:
        raise SystemExit("no poses found")

    item = {
        "id": args.id,
        "poses": poses,
        "conditions": {"pH": args.pH, "ionic_strength": args.ionic_strength},
        "label": {"native_pose": args.native}
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write(json.dumps(item) + "\n")
    print(f"[ok] wrote {args.out} with {len(poses)} poses")

if __name__ == "__main__":
    main()
