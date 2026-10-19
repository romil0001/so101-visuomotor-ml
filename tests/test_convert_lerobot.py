"""Convert local SO101Dataset → LeRobot dataset (RGB + language, no depth)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from so101_ml.convert_lerobot import convert_so101_to_lerobot, so101_lerobot_features
from so101_ml.dataset import JOINT_NAMES, DatasetMeta, EpisodeWriter, SO101Dataset


def _tiny_dataset(root: Path, *, n_eps: int = 1, frames: int = 3) -> SO101Dataset:
    ds = SO101Dataset.create(
        root,
        DatasetMeta(
            name="unit_demo",
            fps=15,
            joint_names=list(JOINT_NAMES),
            camera_name="wrist",
            camera_width=32,
            camera_height=24,
            task="Place keys into metal cup",
            episode_count=0,
            camera_names=("wrist", "overhead"),
            use_depth=True,
            depth_camera_name="wrist",
        ),
    )
    for ep_i in range(n_eps):
        w = EpisodeWriter(ds, ep_i)
        for t in range(frames):
            img_w = np.full((24, 32, 3), t + 10, dtype=np.uint8)
            img_o = np.full((24, 32, 3), t + 40, dtype=np.uint8)
            st = np.arange(6, dtype=np.float32) + t
            act = st + 0.5
            depth = np.full((24, 32), 100 + t, dtype=np.uint16)
            w.add_frame(st, act, float(t), images={"wrist": img_w, "overhead": img_o}, depth_u16=depth)
        w.save()
    return SO101Dataset(root)


def test_features_have_wrist_overhead_no_depth() -> None:
    feats = so101_lerobot_features(height=24, width=32, camera_names=("wrist", "overhead"))
    assert "observation.images.wrist" in feats
    assert "observation.images.overhead" in feats
    assert "observation.state" in feats
    assert "action" in feats
    assert not any("depth" in k for k in feats)


def test_convert_writes_lerobot_episodes(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "lerobot_out"
    _tiny_dataset(src, n_eps=2, frames=3)
    meta = convert_so101_to_lerobot(
        src,
        dst,
        repo_id="local/so101_unit",
        max_episodes=2,
    )
    assert meta["episode_count"] == 2
    assert meta["fps"] == 15
    assert (dst / "meta").is_dir()
    # Reload with LeRobot API
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id="local/so101_unit", root=dst)
    assert ds.num_episodes == 2
    assert ds.num_frames == 6
    sample = ds[0]
    assert "task" in sample or "task_index" in sample or any("task" in str(k) for k in sample)
    assert sample["observation.state"].shape[-1] == 6
    assert sample["action"].shape[-1] == 6
