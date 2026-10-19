"""Load DepthGraspConfig from YAML/JSON."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from so101_ml.hybrid.depth_grasp import DepthGraspConfig


def load_grasp_config(path: str | Path | None = None) -> DepthGraspConfig:
    if not path:
        return DepthGraspConfig()
    p = Path(path)
    text = p.read_text()
    if p.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Grasp config {p} must be a mapping")
    return DepthGraspConfig.from_dict(data)
