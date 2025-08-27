from __future__ import annotations
from typing import Iterator, Dict, Any
import pathlib, json, yaml

def load_dataset(spec) -> Iterator[Dict[str, Any]]:
    if isinstance(spec, str) and pathlib.Path(spec).exists():
        path = pathlib.Path(spec)
        if path.suffix.lower() in [".yaml",".yml"]:
            cfg = yaml.safe_load(path.read_text())
            for row in cfg.get("items", []):
                yield row
        else:
            for line in open(spec):
                yield json.loads(line)
    elif isinstance(spec, dict):
        for row in spec.get("items", []):
            yield row
    else:
        raise ValueError("Unsupported dataset spec")
