
#!/usr/bin/env python3
"""Train a simple learned re-scoring model on pose features.

Usage:
  python scripts/train_gbt.py --dataset datasets/train.jsonl --out artifacts/model.joblib
"""
import argparse, json, pathlib, os
from typing import List, Dict, Any
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from joblib import dump

from conditioned_ensemble_interface.scoring.features import compute_interface_features, condition_features

def load_items(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))
    return items

def build_table(items) -> (np.ndarray, np.ndarray, List[str]):
    X, y = [], []
    feature_keys = None
    for ex in items:
        cond = condition_features(ex.get("conditions", {}))
        label_native = ex.get("label", {}).get("native_pose")
        poses = ex.get("poses", [])
        for p in poses:
            feats = compute_interface_features(p)
            feats.update(cond)
            # set label: 1 if this pose equals native else 0 (skip if unknown)
            if label_native is None:
                continue
            label = 1 if p == label_native else 0
            # choose numeric keys and fix order
            if feature_keys is None:
                feature_keys = sorted([k for k,v in feats.items() if isinstance(v, (int, float))])
            X.append([float(feats.get(k, 0.0)) for k in feature_keys])
            y.append(label)
    if not X:
        raise RuntimeError("No labeled poses found in dataset")
    return np.array(X, dtype=float), np.array(y, dtype=int), feature_keys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="JSONL with poses and native labels")
    ap.add_argument("--out", required=True, help="Path to write model.joblib")
    args = ap.parse_args()

    items = load_items(args.dataset)
    X, y, keys = build_table(items)

    # simple train/val split
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(X))
    ntr = int(0.8 * len(idx))
    tr, va = idx[:ntr], idx[ntr:]

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X[tr], y[tr])
    if len(va) > 0:
        prob = model.predict_proba(X[va])[:,1]
        auc = roc_auc_score(y[va], prob)
        print(f"[train] validation AUC = {auc:.3f}")
    else:
        print("[train] small dataset; skipping validation")

    # stash feature order for inference
    model.feature_order_ = keys
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    dump(model, args.out)
    print(f"[train] wrote {args.out} with {len(keys)} features")

if __name__ == "__main__":
    main()
