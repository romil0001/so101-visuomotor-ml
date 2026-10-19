from __future__ import annotations

from pathlib import Path

import numpy as np

from so101_ml.dagger_offline import combine_demos_and_corrections
from so101_ml.dataset import JOINT_NAMES, DatasetMeta, EpisodeWriter, SO101Dataset


def _ds(root: Path, name: str, n: int, source: str) -> Path:
    path = root / name
    ds = SO101Dataset.create(
        path,
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
        w = EpisodeWriter(ds, i, source=source, intervention=source == "dagger")
        img = np.zeros((24, 32, 3), dtype=np.uint8)
        w.add_frame(np.full(6, i, np.float32), np.full(6, i + 1, np.float32), 0.0, img)
        w.save()
    return path


def test_combine_demos_and_corrections(tmp_path: Path) -> None:
    demos = _ds(tmp_path, "demos", 2, "teleop")
    corr = _ds(tmp_path, "corr", 1, "dagger")
    merged = combine_demos_and_corrections(demos, corr, tmp_path / "out")
    assert merged.n_base == 2
    assert merged.n_corrections == 1
    assert SO101Dataset(merged.root).load_episode(2).source == "dagger"
