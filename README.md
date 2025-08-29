conditioned-ensemble-interface

Condition-aware re-scoring for docking and structure ensembles
Elevate the right pose under the conditions you’ll actually assay (pH, ionic strength, cofactors), while filtering physically broken structures.

***Why this matters (problem → outcome)***

Modern docking and structure prediction give you an ensemble of candidate poses for each complex (protein–ligand or peptide–protein). Picking the right one depends on assay conditions:

- pH shifts protonation → salt bridges and hydrogen-bond networks turn on/off
- Ionic strength screens long-range electrostatics → some poses fall apart
- Cofactors / glycosaminoglycan sulfation bias which surface patches are truly compatible

Ignoring these leads to fragile rankings and wasted wet-lab cycles.

***What this project achieves***

A condition-aware, physics-gated re-scoring layer you run after docking/structure prediction
Robust aggregation across poses (best / mean / softmax) instead of betting on one noisy guess
Simple, reproducible CLI + scripts that anyone can run on a clean machine

***A real-data demo (trypsin–benzamidine; PDB: 3PTB) with Top-5 redock success = 5/5 and best RMSD ≈ 0.08 Å***

***How it works***
[pose generator]  ->  many poses / complex
(docking, AF, etc.)
           |  (for each pose)
           v
  [Feature extractor]
  - interface contacts, hydrophobics
  - simple clash metrics, chain separation
  - + explicit CONDITION FEATURES (pH, ionic strength, cofactors)
           |
           v
  [Learned scorer (tiny)]
  - baseline Gradient Boosting (swap to XGBoost/MLP/GNN later)
           |
  [Physical sanity gates]
  - block obviously broken geometries
           |
           v
  [Ensemble aggregation]
  - best / mean / softmax → one robust score per complex


**Why it helps drug discovery: you send fewer artifacts to the lab, and your top-N reflects the actual buffer you’ll test in.**

**Key features**

Condition-aware inputs: pH, ionic strength, optional cofactors (e.g., GAG sulfation)
Physics sanity filters: simple PoseBusters-style gates to reject impossible structures
Ensemble aggregation: best / mean / softmax (temperature-weighted)
Tiny, swappable model: starts with Gradient Boosting; keep the interface, swap the learner
Reproducible reports: CSV/JSONL + plots you can drop into a README or slide
Real-data mini demo (3PTB, redocking)
Reproduces the result in minutes on macOS/Linux.

***What you’ll see at the end***

Best RMSD to crystal ≈ 0.08 Å
Top-5 success @ 2.0 Å = 5/5

A CSV summary with your aggregated score; initial pass-rate may be low before protonation

0) Prereqs

Project Python env (from repo root):

python3 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev]' || true


Tools env via conda (Intel macOS/Linux):

# if conda not installed, install Miniconda; then:
conda create -n dock -c conda-forge -c bioconda --override-channels smina openbabel rdkit numpy -y


I use smina (Vina-compatible) + Open Babel + RDKit in dock, separate from your .venv.

1) Get the system and make poses
# from repo root
mkdir -p data/3ptb runs/3ptb_smina
curl -L https://files.rcsb.org/download/3PTB.pdb -o data/3ptb/3ptb.pdb

# extract benzamidine and convert to SDF (warnings are OK)
awk '$1=="HETATM" && $4=="BEN" || $1=="CONECT" {print}' data/3ptb/3ptb.pdb > data/3ptb/ligand_BEN.pdb
conda run -n dock obabel data/3ptb/ligand_BEN.pdb -O data/3ptb/ligand.sdf --addtotitle "BEN"

# receptor/ligand → PDBQT
grep -E '^ATOM' data/3ptb/3ptb.pdb > data/3ptb/receptor_clean.pdb
conda run -n dock obabel data/3ptb/receptor_clean.pdb -xr -O data/3ptb/receptor.pdbqt
conda run -n dock obabel data/3ptb/ligand.sdf -O data/3ptb/ligand.pdbqt

# compute a docking box from the ligand
conda run -n dock python scripts/box_from_ligand.py
# copy the printed center_x/y/z and size_x/y/z

# dock 10 poses with smina
conda run -n dock smina \
  --receptor data/3ptb/receptor.pdbqt \
  --ligand   data/3ptb/ligand.pdbqt \
  --center_x <cx> --center_y <cy> --center_z <cz> \
  --size_x   <sx> --size_y   <sy> --size_z   <sz> \
  --num_modes 10 --exhaustiveness 8 \
  --out runs/3ptb_smina/poses.pdbqt --log runs/3ptb_smina/smina.log

# split to PDBs
conda run -n dock obabel runs/3ptb_smina/poses.pdbqt -O runs/3ptb_smina/pose_.pdb -m

2) Build complexes and run the condition-aware re-scorer
# native ligand pdb + chain tag
conda run -n dock obabel data/3ptb/ligand.sdf -O runs/3ptb_smina/native_ligand.pdb
python scripts/tag_chain_L.py runs/3ptb_smina/native_ligand.pdb runs/3ptb_smina/native_ligand_L.pdb

# tag each docked pose to chain L
python scripts/tag_chain_L_dir.py runs/3ptb_smina/pose_*.pdb

# merge protein + ligand to make complexes
mkdir -p runs/3ptb_complexes
cat data/3ptb/receptor_clean.pdb runs/3ptb_smina/native_ligand_L.pdb > runs/3ptb_complexes/complex_native.pdb; echo "END" >> runs/3ptb_complexes/complex_native.pdb
for i in {1..10}; do cat data/3ptb/receptor_clean.pdb "runs/3ptb_smina/pose_${i}_L.pdb" > "runs/3ptb_complexes/complex_pose_${i}.pdb"; echo "END" >> "runs/3ptb_complexes/complex_pose_${i}.pdb"; done

# dataset jsonl
python scripts/make_jsonl_from_complexes.py --native runs/3ptb_complexes/complex_native.pdb --glob "runs/3ptb_complexes/complex_pose_*.pdb" --out datasets/3ptb_complexes.jsonl

# train tiny scorer + predict + aggregate
python scripts/train_gbt.py --dataset datasets/3ptb_complexes.jsonl --out artifacts/real_3ptb_complex.joblib
cat > configs/real_3ptb_complex.yaml <<'YAML'
dataset:
  items:
    - id: "3ptb"
      poses:
        - "runs/3ptb_complexes/complex_native.pdb"
        - "runs/3ptb_complexes/complex_pose_1.pdb"
        - "runs/3ptb_complexes/complex_pose_2.pdb"
        - "runs/3ptb_complexes/complex_pose_3.pdb"
        - "runs/3ptb_complexes/complex_pose_4.pdb"
        - "runs/3ptb_complexes/complex_pose_5.pdb"
        - "runs/3ptb_complexes/complex_pose_6.pdb"
        - "runs/3ptb_complexes/complex_pose_7.pdb"
        - "runs/3ptb_complexes/complex_pose_8.pdb"
        - "runs/3ptb_complexes/complex_pose_9.pdb"
        - "runs/3ptb_complexes/complex_pose_10.pdb"
      conditions: { pH: 7.4, ionic_strength: 0.15 }
model:
  path: "artifacts/real_3ptb_complex.joblib"
YAML
cei --config configs/real_3ptb_complex.yaml --out runs/real_3ptb_complex_preds.jsonl \
  || PYTHONPATH=src python -m conditioned_ensemble_interface.cli --config configs/real_3ptb_complex.yaml --out runs/real_3ptb_complex_preds.jsonl
python scripts/aggregate_and_filter.py --dataset datasets/3ptb_complexes.jsonl --pred runs/real_3ptb_complex_preds.jsonl --out runs/real_3ptb_complex_summary.csv --method softmax --temperature 1.0

3) RMSD to crystal (Top-k)
conda activate dock
python scripts/eval_pose_rmsd_pdb.py --native runs/3ptb_smina/native_ligand.pdb --poses_glob "runs/3ptb_smina/pose_*.pdb" --topk 5 --threshold 2.0

*** Demo result (example):***

Best pose: … RMSD ≈ 0.078 Å

Top-5 success @ 2.0 Å: 5/5

runs/real_3ptb_complex_summary.csv → id=3ptb, n_poses_in=11, n_pass=1, pass_rate≈0.09, aggregate_score≈0.9999

Note: initial pass_rate can be low on raw docked PDBs. Protonating docked ligands (one-line Open Babel -h) usually increases pass-rate; see Troubleshooting.

***What’s in this repo (map)***

src/conditioned_ensemble_interface/
    scoring/features.py — interface + condition features
    scoring/model.py — tiny scorer (GB); easy to swap
    scoring/ensemble.py — best / mean / softmax aggregation
    utils/posechecks.py — physical sanity gates
    cli — cei command entrypoint
scripts/
    train_gbt.py — trains the baseline learner
    aggregate_and_filter.py — applies sanity gates + aggregates
    eval_pose_rmsd_pdb.py — RDKit RMSD / Top-k evaluator
    box_from_ligand.py — quick docking box calculator
    tag_chain_L.py, tag_chain_L_dir.py, make_jsonl_from_complexes.py — tiny helpers

configs/ — YAMLs for inference
datasets/ — JSONL manifests
runs/ — predictions, summaries, logs
.github/workflows/ci.yml — smoke test CI (imports from src/)

***Troubleshooting***

NumPy / RDKit errors: always run RDKit scripts in conda env (conda activate dock), not your .venv.
All poses filtered (pass_rate=0): you fed ligand-only files. Build protein+ligand complexes and re-run.
Low pass_rate: protonate docked ligands:

`conda activate dock
for p in runs/3ptb_smina/pose_*.pdb; do obabel "$p" -h -O "$p"; done`


then re-tag chain L, rebuild complexes, and re-aggregate.

***Citation***
Please cite PDB if you use 3PTB and any datasets you integrate (PDBbind, CrossDocked, etc.).

CI
This repo ships with a simple smoke-test CI (imports from src/, computes features, runs the tiny model, and aggregates). It avoids editable-install quirks and ensures the core pipeline works on a clean runner.

Author
Elif Arslan - peptide nanofibers, ECM-mimetic ligands, condition-aware modeling.