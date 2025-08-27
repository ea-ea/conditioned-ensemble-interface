#!/usr/bin/env bash
# Run the full demo pipeline end-to-end.
# Usage: bash scripts/run_all.sh
set -euo pipefail

# 1) Ensure venv is active
if [ -z "${VIRTUAL_ENV:-}" ]; then
  echo "[run_all] Please activate your virtualenv first: source .venv/bin/activate"
  exit 1
fi

echo "[run_all] Installing required extras..."
python -m pip install --upgrade pip
python -m pip install scikit-learn joblib pandas matplotlib

# 2) Ensure example data exists
mkdir -p datasets runs artifacts

if [ ! -f datasets/train.jsonl ]; then
  cat > datasets/train.jsonl <<'JSONL'
{"id":"toy-1","poses":["examples/pose1.pdb","examples/pose2.pdb"],"conditions":{"pH":7.4,"ionic_strength":0.15},"label":{"native_pose":"examples/pose1.pdb"}}
{"id":"toy-2","poses":["examples/pose1.pdb","examples/pose2.pdb"],"conditions":{"pH":6.8,"ionic_strength":0.20},"label":{"native_pose":"examples/pose2.pdb"}}
JSONL
fi

if [ ! -f configs/example_with_model.yaml ]; then
  mkdir -p configs
  cat > configs/example_with_model.yaml <<'YAML'
dataset:
  items:
    - id: "toy-1"
      poses: ["examples/pose1.pdb", "examples/pose2.pdb"]
      conditions:
        pH: 7.0
        ionic_strength: 0.15
        cofactor: false
        glycosaminoglycan_sulfation_level: 0.0
model:
  path: "artifacts/model.joblib"
YAML
fi

# 3) Train
echo "[run_all] Training model..."
python scripts/train_gbt.py --dataset datasets/train.jsonl --out artifacts/model.joblib

# 4) Predict
echo "[run_all] Running inference..."
cei --config configs/example_with_model.yaml --out runs/out_learned.jsonl

# 5) Aggregate
echo "[run_all] Aggregating & filtering..."
python scripts/aggregate_and_filter.py --dataset datasets/train.jsonl --pred runs/out_learned.jsonl --out runs/summary.csv --method softmax --temperature 1.0

# 6) Report (PNG + HTML)
echo "[run_all] Building quick report..."
python - <<'PY'
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
df = pd.read_csv('runs/summary.csv')
ax = df.plot(kind='bar', x='id', y='aggregate_score', legend=False)
ax.set_xlabel('complex id'); ax.set_ylabel('aggregate score')
plt.tight_layout(); Path('runs').mkdir(exist_ok=True, parents=True)
plt.savefig('runs/summary.png', dpi=150); plt.close()
html = df.to_html(index=False)
Path('runs/summary.html').write_text(f"<h1>Summary</h1>{html}")
print('[run_all] wrote runs/summary.png and runs/summary.html')
PY

echo "[run_all] DONE. Artifacts:"
ls -l runs/summary.csv runs/summary.png runs/summary.html runs/out_learned.jsonl artifacts/model.joblib