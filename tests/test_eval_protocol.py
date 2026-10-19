"""Protocol load + summarize (no robot)."""

from __future__ import annotations

from pathlib import Path

from so101_ml.metrics.eval_protocol import TrialResult, load_protocol, summarize
from so101_ml.paths import project_root


def test_load_bundled_protocol() -> None:
    path = project_root() / "config" / "pen_to_keys.yaml"
    proto = load_protocol(path)
    assert "keys" in proto.task.lower() or "cup" in proto.task.lower()
    assert len(proto.conditions) == 3
    assert proto.n_trials_per_condition >= 1


def test_summarize_pc_success() -> None:
    results = [
        TrialResult(1, "keys=left", True, 10, False, 60.0),
        TrialResult(2, "keys=left", False, 10, False, 60.0),
        TrialResult(3, "keys=center", True, 10, False, 60.0),
    ]
    summary = summarize(results, ["keys=left", "keys=center"])
    assert summary["n_trials"] == 3
    assert summary["n_success"] == 2
    assert abs(summary["pc_success"] - 200.0 / 3) < 1e-6
    assert summary["per_condition"]["keys=left"]["pc_success"] == 50.0
