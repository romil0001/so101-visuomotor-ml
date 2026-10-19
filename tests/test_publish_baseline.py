"""Baseline graphs come from scored trial logs, not the illustrative fixtures."""

from __future__ import annotations

import json
from pathlib import Path

from so101_ml.metrics.publish_baseline import backfill_summaries, publish_baselines, select_baselines


def _trial(run: Path, index: int, condition: str, success: bool) -> None:
    folder = run / "trials" / f"{index:04d}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "result.json").write_text(
        json.dumps(
            {
                "trial_index": index,
                "condition_id": condition,
                "success": success,
                "steps": 10,
                "early_stop": False,
                "timeout_s": 60.0,
            }
        )
    )


def test_backfill_and_select_prefers_complete_grid(tmp_path: Path) -> None:
    act = tmp_path / "eval" / "act_so101" / "run_partial"
    act.mkdir(parents=True)
    (act / "protocol.json").write_text(
        json.dumps(
            {
                "n_trials_per_condition": 5,
                "conditions": [{}, {}, {}],
            }
        )
    )
    for i, cond in enumerate(["keys=left"] * 5 + ["keys=center"] * 5, start=1):
        _trial(act, i, cond, success=(i == 1))

    full = tmp_path / "eval" / "smolvla_so101" / "run_full"
    full.mkdir(parents=True)
    (full / "protocol.json").write_text(
        json.dumps({"n_trials_per_condition": 1, "conditions": [{}, {}, {}]})
    )
    for i, cond in enumerate(["keys=left", "keys=center", "keys=right"], start=1):
        _trial(full, i, cond, success=False)

    written = backfill_summaries(tmp_path / "eval")
    assert len(written) == 2
    selected = select_baselines(tmp_path / "eval")
    assert selected["act"]["n_trials"] == 10
    assert selected["act"]["n_success"] == 1
    assert selected["act"]["complete_protocol"] is False
    assert "keys=right" not in selected["act"]["per_condition"]
    assert selected["smolvla"]["complete_protocol"] is True
    assert selected["smolvla"]["pc_success"] == 0.0

    assets = tmp_path / "assets"
    (assets / "graphs").mkdir(parents=True)
    published = publish_baselines(tmp_path / "eval", assets)
    assert published["index"]["fixture"] is False
    assert (assets / "eval" / "summary_act.json").is_file()
    assert not (assets / "eval" / "summary_hybrid.json").exists()
