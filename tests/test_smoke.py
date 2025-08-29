# Minimal smoke test: import from src, compute features, score, and aggregate.
import os, math

# Make sure Python can find our package directly from the source tree
SRC = os.path.join(os.getcwd(), "src")
if SRC not in os.sys.path:
    os.sys.path.insert(0, SRC)

from conditioned_ensemble_interface.scoring.features import compute_interface_features, condition_features
from conditioned_ensemble_interface.scoring.model import load_model
from conditioned_ensemble_interface.scoring.ensemble import aggregate

def test_features_and_model_scoring():
    # example poses shipped in the repo
    p1 = "examples/pose1.pdb"
    p2 = "examples/pose2.pdb"

    f1 = compute_interface_features(p1)
    f2 = compute_interface_features(p2)

    # we just require that we returned some numeric features (not parse_error flags)
    assert isinstance(f1, dict) and len(f1) > 0
    assert isinstance(f2, dict) and len(f2) > 0

    # add simple condition features (these should always parse)
    cond = condition_features({"pH": 7.4, "ionic_strength": 0.15})
    assert "pH" in cond and "ionic_strength" in cond

    # load model (falls back to dummy if no artifact exists)
    m = load_model({"path": "artifacts/model.joblib"})
    s1 = m.score({**f1, **cond})
    s2 = m.score({**f2, **cond})

    assert isinstance(s1, float) and isinstance(s2, float)

    # sanity: aggregation should return a finite float
    agg = aggregate([s1, s2], method="softmax", temperature=1.0)
    assert isinstance(agg, float) and math.isfinite(agg)
