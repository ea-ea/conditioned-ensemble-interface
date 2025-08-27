
import json, sys, pathlib
from collections import defaultdict

def load_jsonl(path):
    for line in open(path):
        yield json.loads(line)

def main():
    if len(sys.argv) != 3:
        print("Usage: python eval_baseline.py <dataset.jsonl> <predictions.jsonl>")
        sys.exit(1)
    ds_path, pred_path = sys.argv[1], sys.argv[2]

    labels = {}
    for ex in load_jsonl(ds_path):
        if "label" in ex and "native_pose" in ex["label"]:
            labels[ex["id"]] = ex["label"]["native_pose"]

    total = 0
    correct = 0
    for pred in load_jsonl(pred_path):
        id_ = pred["id"]
        if id_ not in labels:
            continue
        total += 1
        scores = pred["scores"]
        # pick top-1 by score
        top_pose = sorted(scores, key=lambda x: x["score"], reverse=True)[0]["pose"]
        if top_pose == labels[id_]:
            correct += 1

    if total == 0:
        print("No labeled items found.")
        sys.exit(1)

    acc = correct / total
    print(f"Top-1 success: {correct}/{total} = {acc:.3f}")

if __name__ == "__main__":
    main()
