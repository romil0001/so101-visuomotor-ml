"""Protocol schema and offline scoring for visuomotor eval.

Robot rollouts live in the full-stack SO-101 repo. This module loads protocol
YAML, stores trial results, and aggregates pc_success / per-condition rates.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

SCORE_SUCCESS = frozenset({"y", "s"})
SCORE_FAIL = frozenset({"n", "f"})

@dataclass(frozen=True)
class Condition:
    id: str
    pen: str
    lighting: str = "overhead"
    distractors: str = "none"
    split: str = "id"


@dataclass(frozen=True)
class Protocol:
    task: str
    timeout_s: float
    n_trials_per_condition: int
    success_criterion: str
    conditions: list[Condition]
    home_qpos: tuple[float, ...] | None = None
    home_steps: int = 80
    home_dt: float = 0.04

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "timeout_s": self.timeout_s,
            "n_trials_per_condition": self.n_trials_per_condition,
            "success_criterion": self.success_criterion,
            "home_qpos": list(self.home_qpos) if self.home_qpos is not None else None,
            "home_steps": self.home_steps,
            "home_dt": self.home_dt,
            "conditions": [asdict(c) for c in self.conditions],
        }


@dataclass(frozen=True)
class TrialResult:
    trial_index: int
    condition_id: str
    success: bool
    steps: int
    early_stop: bool
    timeout_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_protocol(path: str | Path) -> Protocol:
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Protocol {path} must be a mapping")
    conditions = [Condition(**_condition_fields(raw)) for raw in data.get("conditions") or []]
    if not conditions:
        raise ValueError(f"Protocol {path} needs at least one condition")
    criterion = str(data.get("success_criterion") or "").strip()
    raw_home = data.get("home_qpos")
    home_qpos = tuple(float(x) for x in raw_home) if raw_home else None
    return Protocol(
        task=str(data["task"]),
        timeout_s=float(data["timeout_s"]),
        n_trials_per_condition=int(data["n_trials_per_condition"]),
        success_criterion=criterion,
        conditions=conditions,
        home_qpos=home_qpos,
        home_steps=int(data.get("home_steps", 80)),
        home_dt=float(data.get("home_dt", 0.04)),
    )


def _condition_fields(raw: dict[str, Any]) -> dict[str, Any]:
    cid = str(raw.get("id") or f"pen={raw.get('pen', 'unknown')}")
    return {
        "id": cid,
        "pen": str(raw.get("pen", "")),
        "lighting": str(raw.get("lighting", "overhead")),
        "distractors": str(raw.get("distractors", "none")),
        "split": str(raw.get("split", "id")),
    }


def score_from_key(ch: str | None) -> bool | None:
    if ch is None:
        return None
    c = ch.strip().lower()
    if c in SCORE_SUCCESS:
        return True
    if c in SCORE_FAIL:
        return False
    return None


def _pc_success(n_success: int, n: int) -> float:
    if n <= 0:
        return 0.0
    return 100.0 * n_success / n


def summarize(results: list[TrialResult], condition_ids: list[str]) -> dict[str, Any]:
    n_trials = len(results)
    n_success = sum(1 for r in results if r.success)
    per_condition: dict[str, dict[str, float | int]] = {}
    for cid in condition_ids:
        rows = [r for r in results if r.condition_id == cid]
        n = len(rows)
        ns = sum(1 for r in rows if r.success)
        per_condition[cid] = {"n": n, "n_success": ns, "pc_success": _pc_success(ns, n)}
    return {
        "pc_success": _pc_success(n_success, n_trials),
        "n_trials": n_trials,
        "n_success": n_success,
        "per_condition": per_condition,
    }
