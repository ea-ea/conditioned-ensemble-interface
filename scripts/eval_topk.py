#!/usr/bin/env python3
"""
Compute Top-1 and Top-2 success given a labeled dataset and predictions.
Usage:
  python scripts/eval_topk.py --dataset datasets/minipep.jsonl --pred runs/minipep_preds.jsonl
"""
import argparse, json

def load_jsonl(p):
    for line in open(p):
        yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pred", required=True)
    args = ap.parse_args()

    labels = {r["id"]: r.get("label", {}).get("native_pose") for r in load_jsonl(args.dataset)}
    n = 0; top1 = 0; top2 = 0
    for pr in load_jsonl(args.pred):
        nid = pr["id"]; gold = labels.get(nid)
        if not gold: continue
        n += 1
        poses = sorted(pr["scores"], key=lambda s: s["score"], reverse=True)
        ordered = [p["pose"] for p in poses]
        if len(ordered) >= 1 and ordered[0] == gold: top1 += 1
        if gold in ordered[:2]: top2 += 1

    if n == 0:
        print("No labeled items.")
        return
    print(f"Top-1: {top1}/{n} = {top1/n:.3f}")
    print(f"Top-2: {top2}/{n} = {top2/n:.3f}")

if __name__ == "__main__":
    main()