"""Combine teleop + DAgger datasets and diff protocol summaries."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from so101_ml.metrics.compare_eval import compare_summaries, format_comparison
from so101_ml.dataset import JOINT_NAMES, DatasetMeta, EpisodeWriter, SO101Dataset
from so101_ml.dataset.combine import combine_datasets


def _write_episode(root: Path, index: int, *, source: str = "teleop") -> None:
    ds = SO101Dataset(root)
    writer = EpisodeWriter(ds, index, source=source, intervention=source == "dagger")
    img = np.zeros((24, 32, 3), dtype=np.uint8)
    writer.add_frame(np.full(6, index, np.float32), np.full(6, index + 1, np.float32), 0.0, img)
    writer.save()


def _make(tmp: Path, name: str, n: int, source: str) -> Path:
    root = tmp / name
    SO101Dataset.create(
        root,
        DatasetMeta(
            name=name,
            fps=15,
            joint_names=list(JOINT_NAMES),
            camera_name="front",
            camera_width=32,
            camera_height=24,
            task="task",
            episode_count=0,
        ),
    )
    for i in range(n):
        _write_episode(root, i, source=source)
    return root


def test_combine_puts_corrections_after_demos(tmp_path: Path) -> None:
    base = _make(tmp_path, "base", 2, "teleop")
    corr = _make(tmp_path, "corr", 1, "dagger")
    result = combine_datasets(base, corr, tmp_path / "merged")
    assert result.n_base == 2
    assert result.n_corrections == 1
    assert result.correction_indices == (2,)
    merged = SO101Dataset(result.root)
    assert merged.meta.episode_count == 3
    assert merged.load_episode(0).source == "teleop"
    assert merged.load_episode(2).source == "dagger"
    assert merged.load_episode(2).intervention is True
    np.testing.assert_allclose(merged.load_episode(2).action[0], np.full(6, 1.0))


def test_compare_summaries_reports_delta() -> None:
    baseline = {
        "pc_success": 40.0,
        "n_trials": 10,
        "n_success": 4,
        "per_condition": {"keys=left": {"n": 5, "n_success": 1, "pc_success": 20.0}},
    }
    candidate = {
        "pc_success": 60.0,
        "n_trials": 10,
        "n_success": 6,
        "per_condition": {"keys=left": {"n": 5, "n_success": 3, "pc_success": 60.0}},
    }
    report = compare_summaries(baseline, candidate)
    assert report["delta_pc_success"] == 20.0
    assert report["per_condition"]["keys=left"]["delta_pc_success"] == 40.0
    text = format_comparison(report)
    assert "delta=+20.0" in text


def test_compare_accepts_by_condition_alias() -> None:
    report = compare_summaries(
        {"pc_success": 10, "by_condition": {"a": {"trials": 2, "pc_success": 0}}},
        {"pc_success": 50, "per_condition": {"a": {"n": 2, "pc_success": 50}}},
    )
    assert report["per_condition"]["a"]["baseline_n"] == 2
    assert report["per_condition"]["a"]["delta_pc_success"] == 50.0
