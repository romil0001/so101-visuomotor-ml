"""Tests for the local SO-101 episode dataset (no robot required)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from so101_ml.dataset import JOINT_NAMES, DatasetMeta, EpisodeWriter, SO101Dataset, compute_stats, write_stats


def _rgb_frame(h: int, w: int, value: int) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = (value, 40, 80)
    return frame


def test_joint_names_are_six_so101_joints() -> None:
    assert JOINT_NAMES == (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    )


def test_episode_writer_roundtrip(tmp_path: Path) -> None:
    meta = DatasetMeta(
        name="unit_demo",
        fps=15,
        joint_names=list(JOINT_NAMES),
        camera_name="front",
        camera_width=64,
        camera_height=48,
        task="Pick pen",
        episode_count=0,
    )
    ds = SO101Dataset.create(tmp_path, meta)
    writer = EpisodeWriter(ds, index=0)
    n = 8
    for t in range(n):
        state = np.linspace(t, t + 5, 6, dtype=np.float32)
        action = state + 1.0
        writer.add_frame(
            state=state,
            action=action,
            timestamp=t / 15.0,
            image_rgb=_rgb_frame(48, 64, t * 10),
        )
    path = writer.save()
    assert path.name == "000000"

    loaded = SO101Dataset(tmp_path)
    assert loaded.meta.episode_count == 1
    assert loaded.meta.task == "Pick pen"
    assert loaded.meta.fps == 15
    ep = loaded.load_episode(0)
    assert ep.num_frames == n
    assert ep.state.shape == (n, 6)
    assert ep.action.shape == (n, 6)
    assert ep.timestamps.shape == (n,)
    np.testing.assert_allclose(ep.state[3], np.linspace(3, 8, 6), rtol=1e-5)
    frames = ep.load_frames()
    assert frames.shape == (n, 48, 64, 3)
    assert frames.dtype == np.uint8


def test_episode_writer_two_cameras(tmp_path: Path) -> None:
    meta = DatasetMeta(
        name="unit_demo",
        fps=15,
        joint_names=list(JOINT_NAMES),
        camera_name="wrist",
        camera_names=("wrist", "overhead"),
        camera_width=32,
        camera_height=24,
        task="Pick pen",
        episode_count=0,
    )
    ds = SO101Dataset.create(tmp_path, meta)
    writer = EpisodeWriter(ds, index=0)
    writer.add_frame(
        np.zeros(6, np.float32),
        np.ones(6, np.float32),
        0.0,
        images={"wrist": _rgb_frame(24, 32, 1), "overhead": _rgb_frame(24, 32, 9)},
    )
    writer.save()
    loaded = SO101Dataset(tmp_path)
    assert loaded.meta.names() == ("wrist", "overhead")
    ep = loaded.load_episode(0)
    assert ep.load_frames("wrist").shape == (1, 24, 32, 3)
    assert ep.load_frames("overhead").shape == (1, 24, 32, 3)
    assert (tmp_path / "episodes" / "000000" / "wrist.mp4").is_file()
    assert (tmp_path / "episodes" / "000000" / "overhead.mp4").is_file()


def test_episode_writer_depth(tmp_path: Path) -> None:
    meta = DatasetMeta(
        name="unit_demo",
        fps=15,
        joint_names=list(JOINT_NAMES),
        camera_name="wrist",
        camera_names=("wrist", "overhead"),
        camera_width=32,
        camera_height=24,
        task="Pick pen",
        episode_count=0,
        use_depth=True,
        depth_camera_name="wrist",
    )
    ds = SO101Dataset.create(tmp_path, meta)
    writer = EpisodeWriter(ds, index=0)
    depth = np.arange(32 * 24, dtype=np.uint16).reshape(24, 32)
    writer.add_frame(
        np.zeros(6, np.float32),
        np.ones(6, np.float32),
        0.0,
        images={"wrist": _rgb_frame(24, 32, 1), "overhead": _rgb_frame(24, 32, 9)},
        depth_u16=depth,
    )
    writer.save()
    ep = SO101Dataset(tmp_path).load_episode(0)
    assert ep.depth_path is not None
    loaded = ep.load_depth()
    assert loaded.shape == (1, 24, 32)
    np.testing.assert_array_equal(loaded[0], depth)


def test_resume_appends_next_episode_index(tmp_path: Path) -> None:
    meta = DatasetMeta(
        name="unit_demo",
        fps=15,
        joint_names=list(JOINT_NAMES),
        camera_name="front",
        camera_width=32,
        camera_height=24,
        task="task",
        episode_count=0,
    )
    ds = SO101Dataset.create(tmp_path, meta)
    w0 = EpisodeWriter(ds, index=ds.next_episode_index())
    w0.add_frame(
        np.zeros(6, np.float32),
        np.ones(6, np.float32),
        0.0,
        _rgb_frame(24, 32, 1),
    )
    w0.save()
    ds = SO101Dataset(tmp_path)
    assert ds.next_episode_index() == 1
    w1 = EpisodeWriter(ds, index=ds.next_episode_index())
    w1.add_frame(
        np.ones(6, np.float32),
        np.zeros(6, np.float32),
        0.0,
        _rgb_frame(24, 32, 2),
    )
    w1.save()
    ds = SO101Dataset(tmp_path)
    assert ds.meta.episode_count == 2
    assert ds.load_episode(1).action[0, 0] == 0.0


def test_abort_discards_episode(tmp_path: Path) -> None:
    meta = DatasetMeta(
        name="unit_demo",
        fps=15,
        joint_names=list(JOINT_NAMES),
        camera_name="front",
        camera_width=32,
        camera_height=24,
        task="task",
        episode_count=0,
    )
    ds = SO101Dataset.create(tmp_path, meta)
    writer = EpisodeWriter(ds, index=0)
    writer.add_frame(
        np.zeros(6, np.float32),
        np.ones(6, np.float32),
        0.0,
        _rgb_frame(24, 32, 9),
    )
    writer.abort()
    ds = SO101Dataset(tmp_path)
    assert ds.meta.episode_count == 0
    assert not (tmp_path / "episodes" / "000000").exists()


def test_compute_stats_mean_std(tmp_path: Path) -> None:
    meta = DatasetMeta(
        name="unit_demo",
        fps=15,
        joint_names=list(JOINT_NAMES),
        camera_name="front",
        camera_width=32,
        camera_height=24,
        task="task",
        episode_count=0,
    )
    ds = SO101Dataset.create(tmp_path, meta)
    writer = EpisodeWriter(ds, index=0)
    states = np.array([[0, 2, 4, 6, 8, 10], [2, 4, 6, 8, 10, 12]], dtype=np.float32)
    actions = states + 10
    for t in range(2):
        writer.add_frame(states[t], actions[t], float(t), _rgb_frame(24, 32, 0))
    writer.save()
    ds = SO101Dataset(tmp_path)
    stats = compute_stats(ds)
    write_stats(tmp_path, stats)
    saved = json.loads((tmp_path / "stats.json").read_text())
    np.testing.assert_allclose(saved["state_mean"], states.mean(axis=0), rtol=1e-5)
    np.testing.assert_allclose(saved["state_std"], states.std(axis=0), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(saved["action_mean"], actions.mean(axis=0), rtol=1e-5)
    np.testing.assert_allclose(saved["action_std"], actions.std(axis=0), rtol=1e-5, atol=1e-6)
    assert all(s > 0 for s in saved["state_std"])


def test_compute_stats_episode_indices_ignore_held_out(tmp_path: Path) -> None:
    meta = DatasetMeta(
        name="unit_demo",
        fps=15,
        joint_names=list(JOINT_NAMES),
        camera_name="front",
        camera_width=32,
        camera_height=24,
        task="task",
        episode_count=0,
    )
    ds = SO101Dataset.create(tmp_path, meta)
    img = _rgb_frame(24, 32, 0)
    w0 = EpisodeWriter(ds, 0)
    for t in range(2):
        w0.add_frame(np.zeros(6, np.float32), np.ones(6, np.float32), float(t), img)
    w0.save()
    w1 = EpisodeWriter(SO101Dataset(tmp_path), 1)
    for t in range(2):
        w1.add_frame(np.full(6, 100.0, np.float32), np.full(6, 101.0, np.float32), float(t), img)
    w1.save()
    ds = SO101Dataset(tmp_path)
    train_stats = compute_stats(ds, episode_indices=[0])
    all_stats = compute_stats(ds)
    np.testing.assert_allclose(train_stats["state_mean"], np.zeros(6), atol=1e-6)
    np.testing.assert_allclose(all_stats["state_mean"], np.full(6, 50.0), rtol=1e-5)
    assert train_stats["state_mean"][0] != all_stats["state_mean"][0]


def test_rejects_wrong_qpos_length(tmp_path: Path) -> None:
    meta = DatasetMeta(
        name="unit_demo",
        fps=15,
        joint_names=list(JOINT_NAMES),
        camera_name="front",
        camera_width=32,
        camera_height=24,
        task="task",
        episode_count=0,
    )
    ds = SO101Dataset.create(tmp_path, meta)
    writer = EpisodeWriter(ds, index=0)
    with pytest.raises(ValueError, match="6"):
        writer.add_frame(
            np.zeros(5, np.float32),
            np.zeros(6, np.float32),
            0.0,
            _rgb_frame(24, 32, 0),
        )
