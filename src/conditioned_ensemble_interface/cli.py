from __future__ import annotations
import argparse, yaml, json, pathlib
from .scoring.model import load_model, score_ensemble
from .data.loaders import load_dataset

def main():
    p = argparse.ArgumentParser(prog="cei", description="Condition-aware ensemble interface scoring")
    p.add_argument("--config", type=str, help="Path to a YAML config")
    p.add_argument("--dataset", type=str, help="Path to dataset config or folder")
    p.add_argument("--out", type=str, default="runs/out.jsonl")
    args = p.parse_args()

    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    ds = load_dataset(args.dataset or cfg.get("dataset", {}))
    model = load_model(cfg.get("model", {}))

    out_path = pathlib.Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for item in ds:
            scores = score_ensemble(model, item)
            f.write(json.dumps({"id": item.get("id"), "scores": scores}) + "\n")
    print(f"[cei] wrote {out_path}")
