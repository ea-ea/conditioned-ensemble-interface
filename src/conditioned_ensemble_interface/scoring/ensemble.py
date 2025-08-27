
from __future__ import annotations
from typing import List, Dict
import math

def aggregate(scores: List[float], method: str = "best", temperature: float = 1.0) -> float:
    if not scores:
        return float("nan")
    if method == "best":
        return float(max(scores))
    if method == "mean":
        return float(sum(scores) / len(scores))
    if method == "softmax":
        # temperature > 0; larger -> flatter
        t = max(1e-6, float(temperature))
        ws = [math.exp(s / t) for s in scores]
        denom = sum(ws)
        return float(sum(w*s for w, s in zip(ws, scores)) / (denom if denom else 1.0))
    raise ValueError(f"Unknown method: {method}")
