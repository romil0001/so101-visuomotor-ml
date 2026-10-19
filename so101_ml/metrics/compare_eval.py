"""Compare two protocol summary.json files (baseline vs a later policy)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_summary(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_dir():
        path = path / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"No summary.json at {path}")
    data = json.loads(path.read_text())
    if "pc_success" not in data:
        raise ValueError(f"{path} has no pc_success")
    return data


def _conditions(summary: dict[str, Any]) -> dict[str, Any]:
    raw = summary.get("per_condition")
    if raw is None:
        raw = summary.get("by_condition") or {}
    return dict(raw)


def compare_summaries(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    b_pc = float(baseline["pc_success"])
    c_pc = float(candidate["pc_success"])
    b_cond = _conditions(baseline)
    c_cond = _conditions(candidate)
    ids = list(dict.fromkeys([*b_cond.keys(), *c_cond.keys()]))
    per: dict[str, dict[str, float | int]] = {}
    for cid in ids:
        br = b_cond.get(cid) or {}
        cr = c_cond.get(cid) or {}
        bp = float(br.get("pc_success", 0.0))
        cp = float(cr.get("pc_success", 0.0))
        per[cid] = {
            "baseline_pc_success": bp,
            "candidate_pc_success": cp,
            "delta_pc_success": cp - bp,
            "baseline_n": int(br.get("n", br.get("trials", 0)) or 0),
            "candidate_n": int(cr.get("n", cr.get("trials", 0)) or 0),
        }
    return {
        "baseline_pc_success": b_pc,
        "candidate_pc_success": c_pc,
        "delta_pc_success": c_pc - b_pc,
        "baseline_n_trials": int(baseline.get("n_trials", 0) or 0),
        "candidate_n_trials": int(candidate.get("n_trials", 0) or 0),
        "per_condition": per,
    }


def format_comparison(report: dict[str, Any]) -> str:
    lines = [
        f"pc_success  baseline={report['baseline_pc_success']:.1f}  "
        f"candidate={report['candidate_pc_success']:.1f}  "
        f"delta={report['delta_pc_success']:+.1f}",
        f"trials      baseline={report['baseline_n_trials']}  candidate={report['candidate_n_trials']}",
    ]
    for cid, row in report["per_condition"].items():
        lines.append(
            f"  {cid}: {row['baseline_pc_success']:.1f} -> {row['candidate_pc_success']:.1f}  "
            f"delta={row['delta_pc_success']:+.1f}  "
            f"n={row['baseline_n']}/{row['candidate_n']}"
        )
    return "\n".join(lines) + "\n"
