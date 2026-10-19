from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from so101_ml.dataset.reader import SO101Dataset


def compute_stats(
    dataset: SO101Dataset, episode_indices: list[int] | None = None
) -> dict[str, list[float]]:
    if dataset.meta.episode_count < 1:
        raise ValueError("Cannot compute stats on an empty dataset")
    if episode_indices is None:
        episode_indices = list(range(dataset.meta.episode_count))
    if len(episode_indices) < 1:
        raise ValueError("Cannot compute stats on an empty episode list")
    states = [dataset.load_episode(i).state for i in episode_indices]
    actions = [dataset.load_episode(i).action for i in episode_indices]
    state = np.concatenate(states, axis=0)
    action = np.concatenate(actions, axis=0)
    state_std = state.std(axis=0)
    action_std = action.std(axis=0)
    state_std = np.maximum(state_std, 1e-6)
    action_std = np.maximum(action_std, 1e-6)
    return {
        "state_mean": state.mean(axis=0).astype(float).tolist(),
        "state_std": state_std.astype(float).tolist(),
        "action_mean": action.mean(axis=0).astype(float).tolist(),
        "action_std": action_std.astype(float).tolist(),
    }


def write_stats(root: Path, stats: dict[str, Any]) -> Path:
    path = Path(root) / "stats.json"
    path.write_text(json.dumps(stats, indent=2) + "\n")
    return path


def load_stats(root: Path) -> dict[str, Any]:
    path = Path(root) / "stats.json"
    if not path.is_file():
        raise FileNotFoundError(f"No stats at {path}")
    return json.loads(path.read_text())
