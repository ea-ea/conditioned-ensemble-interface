# Project narrative — Conditioned Ensemble Interface

**What it is.** A condition-aware, multi-state re-scoring and design layer that upgrades structure-prediction pipelines for drug discovery. It improves robustness by ranking ensembles of poses with physical checks and environment features (pH, ionic strength, cofactors, glycosaminoglycan sulfation).

**Why it matters.** Modern predictors often output one “best” structure without accounting for microstate diversity or lab conditions. Medicinal chemistry decisions need ranking that survives real buffers and avoids impossible geometry.

**What’s novel.**
- Ensemble-first interface scoring (best/mean/softmax aggregation).
- Condition-aware features that change ranking with environment.
- Built-in physical sanity gates so broken poses are never rewarded.
- Clean adapters so users can plug in outputs from various tools.

**What’s included today.**
- Reproducible demo on toy data; ready hooks for public benchmarks.
- A simple learned scorer (gradient-boosted trees) to prove the concept.
- Scripts to train, predict, aggregate, and report.

**What’s next (roadmap).**
- Dataset adapters: CrossDocked 2020 (small molecules) and a peptide–protein benchmark.
- Rich physical checks (bond lengths/angles; clash energy; rotamer sanity).
- Design loop demo: generate backbones (diffusion) → sequence design → predict complexes → re-score.
- Paper-style evaluation tables and plots in `docs/`.

**How a team could use this.**
- Drop your predicted poses into the dataset format.
- Run `bash scripts/run_all.sh` to get ranked results and a quick report.
- Integrate the scorer as a step in your lead triage.