
from __future__ import annotations
from typing import Dict, Any, List
import json, os
from .features import compute_interface_features, condition_features

try:
    import joblib  # scikit-learn compatible
except Exception:
    joblib = None

class DummyModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    def score(self, feats: Dict[str, float]) -> float:
        # simple linear proxy using approx_buried_score if present
        s = float(feats.get("approx_buried_score", 0.0))
        # squash to 0..1
        return 1.0 / (1.0 + pow(2.71828, -0.01 * s))

class SklearnModel:
    def __init__(self, path: str):
        if joblib is None:
            raise RuntimeError("joblib not available; install scikit-learn")
        self.model = joblib.load(path)
        self.feature_order = getattr(self.model, "feature_order_", None)
    def score(self, feats: Dict[str, float]) -> float:
        if self.feature_order is None:
            # heuristic: use all numeric features sorted by key
            keys = sorted([k for k,v in feats.items() if isinstance(v, (int,float))])
        else:
            keys = self.feature_order
        x = [[float(feats.get(k, 0.0)) for k in keys]]
        y = float(self.model.predict_proba(x)[0][1]) if hasattr(self.model, "predict_proba") else float(self.model.predict(x)[0])
        return y

def load_model(cfg: Dict[str, Any] = None):
    cfg = cfg or {}
    model_path = cfg.get("path")
    if model_path and os.path.exists(model_path):
        return SklearnModel(model_path)
    return DummyModel(**cfg)

def score_ensemble(model, item: Dict[str, Any]) -> List[Dict[str, float]]:
    cond = condition_features(item.get("conditions", {}))
    out = []
    for pose in item.get("poses", []):
        feats = compute_interface_features(pose)
        feats.update(cond)
        score = model.score(feats)
        out.append({"pose": pose, "score": float(score)})
    return out
