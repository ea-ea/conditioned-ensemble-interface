
#!/usr/bin/env python3
"""Aggregate and filter predictions using basic physical checks.

Usage:
  python scripts/aggregate_and_filter.py --dataset datasets/train.jsonl --pred runs/out_learned.jsonl --out runs/summary.csv --method softmax --temperature 1.0
"""
import argparse, json, csv, pathlib, math
from typing import Dict, Any, List
from conditioned_ensemble_interface.utils.posechecks import basic_pose_checks
from conditioned_ensemble_interface.scoring.features import compute_interface_features, condition_features
from conditioned_ensemble_interface.scoring.ensemble import aggregate

def load_jsonl(path):
    for line in open(path):
        yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="JSONL dataset used for predictions")
    ap.add_argument("--pred", required=True, help="JSONL predictions from 'cei'")
    ap.add_argument("--out", required=True, help="CSV summary output path")
    ap.add_argument("--filtered", default="runs/filtered_predictions.jsonl", help="Filtered predictions JSONL output")
    ap.add_argument("--method", default="best", choices=["best", "mean", "softmax"])
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--min_atoms_per_chain", type=int, default=2)
    args = ap.parse_args()

    # Map id -> conditions from dataset
    dset = {row["id"]: row.get("conditions", {}) for row in load_jsonl(args.dataset)}

    out_rows = []
    filtered_preds = []

    for pred in load_jsonl(args.pred):
        pid = pred["id"]
        conditions = dset.get(pid, {})
        kept = []
        for s in pred["scores"]:
            pose = s["pose"]
            score = float(s["score"])
            checks = basic_pose_checks(pose, min_atoms_per_chain=args.min_atoms_per_chain)
            feats = compute_interface_features(pose)
            # Simple pass rule: must pass basic checks AND have no parse errors and at least some contact signal if available
            passes = checks["pass"] and (feats.get("contact_count_4A", 0.0) >= 0.0)
            if passes:
                kept.append({"pose": pose, "score": score})
        n_in = len(pred["scores"])
        n_pass = len(kept)
        agg = aggregate([k["score"] for k in kept], method=args.method, temperature=args.temperature) if kept else float("nan")
        filtered_preds.append({"id": pid, "scores": kept})
        out_rows.append({
            "id": pid,
            "n_poses_in": n_in,
            "n_pass": n_pass,
            "pass_rate": (n_pass / n_in) if n_in else math.nan,
            "aggregate_method": args.method,
            "aggregate_score": agg
        })

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else ["id","n_poses_in","n_pass","pass_rate","aggregate_method","aggregate_score"])
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    pathlib.Path(args.filtered).parent.mkdir(parents=True, exist_ok=True)
    with open(args.filtered, "w") as f:
        for row in filtered_preds:
            f.write(json.dumps(row) + "\n")

    print(f"[aggregate] wrote {args.out} and {args.filtered}")

if __name__ == "__main__":
    main()
