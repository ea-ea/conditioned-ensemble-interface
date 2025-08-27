#!/usr/bin/env python3
"""
Convert a simple manifest CSV into the JSONL dataset used by CEI.
Usage:
  python scripts/build_dataset_from_manifest.py \
    --manifest benchmarks/minipep/manifest.csv \
    --out datasets/minipep.jsonl
"""
import argparse, csv, json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(args.manifest, newline="")))
    with open(outp, "w") as f:
        for r in rows:
            poses = [p.strip() for p in r["poses"].split(";") if p.strip()]
            item = {
                "id": r["id"],
                "poses": poses,
                "conditions": {
                    "pH": float(r["pH"]),
                    "ionic_strength": float(r["ionic_strength"])
                },
                "label": {"native_pose": r["native_pose"]} if r.get("native_pose") else {}
            }
            f.write(json.dumps(item) + "\n")
    print(f"[build] wrote {outp} with {len(rows)} items")

if __name__ == "__main__":
    main()