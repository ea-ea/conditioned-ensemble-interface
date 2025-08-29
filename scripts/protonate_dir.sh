#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "usage: protonate_dir.sh runs/3ptb_smina/pose_*.pdb"
  exit 1
fi
for f in "$@"; do
  conda run -n dock obabel "$f" -h -O "$f"
  echo "[ok] protonated $f"
done
