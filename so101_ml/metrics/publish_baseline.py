"""Rebuild protocol summaries from trial logs and publish the real baseline graphs.

Fixture rates in assets/eval are replaced by the fullest scored run of each
policy. Conditions with zero trials are omitted so an interrupted grid is not
drawn as 0% on cells that were never run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from so101_ml.metrics.eval_protocol import TrialResult, summarize

PROTOCOL_ORDER = ("keys=left", "keys=center", "keys=right")


def policy_family(run_dir: Path) -> str:
    text = str(run_dir).lower()
    if "hybrid" in text:
        return "hybrid"
    if "smolvla" in text:
        return "smolvla"
    if "act" in text:
        return "act"
    return "other"


def load_trial_results(run_dir: Path) -> list[TrialResult]:
    trials = run_dir / "trials"
    if not trials.is_dir():
        return []
    results: list[TrialResult] = []
    for path in sorted(trials.glob("*/result.json")):
        raw = json.loads(path.read_text())
        results.append(
            TrialResult(
                trial_index=int(raw["trial_index"]),
                condition_id=str(raw["condition_id"]),
                success=bool(raw["success"]),
                steps=int(raw["steps"]),
                early_stop=bool(raw.get("early_stop", False)),
                timeout_s=float(raw.get("timeout_s", 0.0)),
            )
        )
    return results


def summary_from_trials(results: list[TrialResult]) -> dict[str, Any]:
    seen: list[str] = []
    for result in results:
        if result.condition_id not in seen:
            seen.append(result.condition_id)
    ordered = [cid for cid in PROTOCOL_ORDER if cid in seen]
    ordered.extend(cid for cid in seen if cid not in ordered)
    summary = summarize(results, ordered)
    summary["per_condition"] = {
        cid: row for cid, row in summary["per_condition"].items() if int(row["n"]) > 0
    }
    return summary


def backfill_summaries(eval_root: Path) -> list[Path]:
    """Write summary.json for every run that has trial results and no summary yet."""
    written: list[Path] = []
    eval_root = Path(eval_root)
    if not eval_root.is_dir():
        return written
    for trials in eval_root.rglob("trials"):
        if not trials.is_dir():
            continue
        run_dir = trials.parent
        if (run_dir / "summary.json").is_file():
            continue
        results = load_trial_results(run_dir)
        if not results:
            continue
        summary = summary_from_trials(results)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        written.append(run_dir / "summary.json")
    return written


def _expected_trials(run_dir: Path) -> int | None:
    protocol_path = run_dir / "protocol.json"
    if not protocol_path.is_file():
        return None
    data = json.loads(protocol_path.read_text())
    n = data.get("n_trials_per_condition")
    conditions = data.get("conditions") or []
    if n is None or not conditions:
        return None
    return int(n) * len(conditions)


def select_baselines(eval_root: Path) -> dict[str, dict[str, Any]]:
    """Fullest keys=* run per policy family. Complete 15-trial grids win ties on count."""
    best: dict[str, tuple[tuple[int, int, str], dict[str, Any]]] = {}
    for summary_path in Path(eval_root).rglob("summary.json"):
        run_dir = summary_path.parent
        if not (run_dir / "trials").is_dir() and "per_condition" not in json.loads(summary_path.read_text()):
            continue
        data = json.loads(summary_path.read_text())
        if "pc_success" not in data:
            continue
        family = policy_family(run_dir)
        if family == "other":
            continue
        per = data.get("per_condition") or {}
        if not any(str(cid).startswith("keys=") for cid in per):
            continue
        n_trials = int(data.get("n_trials") or 0)
        if n_trials < 1:
            continue
        expected = _expected_trials(run_dir)
        complete = int(expected is not None and n_trials >= expected)
        rank = (complete, n_trials, str(run_dir))
        current = best.get(family)
        if current is None or rank > current[0]:
            payload = dict(data)
            payload["method"] = _method_label(family, n_trials, expected)
            payload["source_run"] = str(run_dir)
            payload["complete_protocol"] = bool(complete)
            best[family] = (rank, payload)
    order = ("act", "smolvla", "hybrid")
    ranked = {family: payload for family, (_rank, payload) in best.items()}
    return {family: ranked[family] for family in order if family in ranked}


def _method_label(family: str, n_trials: int, expected: int | None) -> str:
    name = {"act": "ACT", "smolvla": "SmolVLA", "hybrid": "Hybrid"}.get(family, family)
    if expected is not None and n_trials < expected:
        return f"{name} (partial {n_trials}/{expected})"
    return f"{name} ({n_trials} trials)"


def publish_baselines(eval_root: Path, assets_dir: Path) -> dict[str, Any]:
    """Backfill summaries, write assets/eval/summary_*.json, and redraw the graphs."""
    eval_root = Path(eval_root)
    assets_dir = Path(assets_dir)
    written = backfill_summaries(eval_root)
    selected = select_baselines(eval_root)
    if not selected:
        raise FileNotFoundError(f"No keys=* protocol summaries under {eval_root}")
    eval_assets = assets_dir / "eval"
    eval_assets.mkdir(parents=True, exist_ok=True)
    for stale in eval_assets.glob("summary_*.json"):
        stale.unlink()
    index_runs: dict[str, Any] = {}
    for family, summary in selected.items():
        out = eval_assets / f"summary_{family}.json"
        out.write_text(json.dumps(summary, indent=2) + "\n")
        index_runs[family] = {
            "summary": str(out),
            "source_run": summary.get("source_run"),
            "pc_success": summary["pc_success"],
            "n_trials": summary["n_trials"],
            "n_success": summary["n_success"],
            "complete_protocol": summary.get("complete_protocol", False),
        }
    index = {
        "protocol": "config/eval/pen_to_keys.yaml",
        "fixture": False,
        "backfilled_summaries": [str(p) for p in written],
        "runs": index_runs,
    }
    (eval_assets / "baseline.json").write_text(json.dumps(index, indent=2) + "\n")
    return {"index": index, "summaries": selected}
