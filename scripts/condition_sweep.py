#!/usr/bin/env python3
"""
Sweep pH and ionic strength, rescore poses, and write a CSV + PNG.
Usage:
  python scripts/condition_sweep.py --in runs/out_learned.jsonl --out runs/sweep.csv
"""
import argparse, json, csv
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from conditioned_ensemble_interface.scoring.model import load_model
from conditioned_ensemble_interface.scoring.features import compute_interface_features, condition_features
from conditioned_ensemble_interface.scoring.ensemble import aggregate

def load_preds(path):
    for line in open(path):
        yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="pred_path", required=True)
    ap.add_argument("--model", default="artifacts/model.joblib")
    ap.add_argument("--out", default="runs/sweep.csv")
    args = ap.parse_args()

    model = load_model({"path": args.model})

    ph_vals = np.linspace(6.5, 8.0, 7)          # 6.5, 6.75, ..., 8.0
    ionic_vals = [0.05, 0.15, 0.30]              # low / physio / high
    rows = []

    for pred in load_preds(args.pred_path):
        pid = pred["id"]
        poses = [s["pose"] for s in pred["scores"]]
        for ph in ph_vals:
            for ionic in ionic_vals:
                cond = {"pH": float(ph), "ionic_strength": float(ionic)}
                scores = []
                for p in poses:
                    feats = compute_interface_features(p)
                    feats.update(condition_features(cond))
                    scores.append(model.score(feats))
                agg = float(aggregate(scores, method="softmax", temperature=1.0))
                rows.append({"id": pid, "pH": float(ph), "ionic_strength": float(ionic), "aggregate": agg})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"[sweep] wrote {args.out} with {len(rows)} rows")

    # simple heatmap per id (one figure per complex)
    for pid in sorted(set(r["id"] for r in rows)):
        dfp = pd.DataFrame([r for r in rows if r["id"] == pid])
        pt = dfp.pivot(index="pH", columns="ionic_strength", values="aggregate")
        ax = plt.imshow(pt.values, aspect="auto", origin="lower")
        plt.xticks(range(len(pt.columns)), pt.columns)
        plt.yticks(range(len(pt.index)), [f"{v:.2f}" for v in pt.index])
        plt.xlabel("ionic strength (M)"); plt.ylabel("pH")
        plt.title(f"aggregate score — {pid}")
        plt.colorbar(ax)
        out_png = Path(f"runs/sweep_{pid}.png")
        plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
        print(f"[sweep] wrote {out_png}")

if __name__ == "__main__":
    main()