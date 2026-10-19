"""ACT forward, train two steps, reload select_action (CPU, no robot)."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import torch

from so101_ml.dataset import JOINT_NAMES, DatasetMeta, EpisodeWriter, SO101Dataset, compute_stats, write_stats
from so101_ml.models.act import ACT, ACTConfig, ACTPolicy, find_policy_weights
from so101_ml.train import split_episodes, train


def _tiny_cfg() -> ACTConfig:
    return ACTConfig(
        n_joints=6,
        chunk_size=8,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        latent_dim=8,
        kl_weight=10.0,
        backbone="tiny",
        image_size=32,
        dropout=0.0,
        n_action_steps=8,
    )


def _synthetic_dataset(tmp_path: Path, episodes: int = 2, t: int = 16) -> Path:
    ds = SO101Dataset.create(
        tmp_path,
        DatasetMeta(
            name="synth",
            fps=15,
            joint_names=list(JOINT_NAMES),
            camera_name="front",
            camera_width=32,
            camera_height=32,
            task="synth",
            episode_count=0,
        ),
    )
    rng = np.random.default_rng(0)
    for i in range(episodes):
        w = EpisodeWriter(ds, i)
        for k in range(t):
            state = rng.normal(size=6).astype(np.float32)
            action = state + 0.1
            img = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
            w.add_frame(state, action, k / 15.0, img)
        w.save()
    write_stats(tmp_path, compute_stats(SO101Dataset(tmp_path)))
    return tmp_path


def test_act_forward_with_depth() -> None:
    cfg = _tiny_cfg()
    cfg = ACTConfig(**{**cfg.to_dict(), "use_depth": True})
    model = ACT(cfg)
    b, c = 2, cfg.chunk_size
    images = torch.rand(b, 3, cfg.image_size, cfg.image_size)
    depth = torch.rand(b, cfg.image_size, cfg.image_size)
    state = torch.randn(b, 6)
    actions = torch.randn(b, c, 6)
    mask = torch.ones(b, c)
    out = model(images=images, state=state, actions=actions, action_mask=mask, depth=depth)
    assert out["pred_actions"].shape == (b, c, 6)
    out["loss"].backward()


def test_train_with_depth_then_select_action(tmp_path: Path) -> None:
    ds = SO101Dataset.create(
        tmp_path / "data",
        DatasetMeta(
            name="depth",
            fps=15,
            joint_names=list(JOINT_NAMES),
            camera_name="wrist",
            camera_names=("wrist", "overhead"),
            camera_width=32,
            camera_height=32,
            task="synth",
            episode_count=0,
            use_depth=True,
            depth_camera_name="wrist",
        ),
    )
    rng = np.random.default_rng(0)
    for i in range(2):
        w = EpisodeWriter(ds, i)
        for k in range(8):
            w.add_frame(
                rng.normal(size=6).astype(np.float32),
                rng.normal(size=6).astype(np.float32),
                k / 15.0,
                images={
                    "wrist": rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8),
                    "overhead": rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8),
                },
                depth_u16=rng.integers(100, 1500, size=(32, 32), dtype=np.uint16),
            )
        w.save()
    cfg = ACTConfig(**{**_tiny_cfg().to_dict(), "use_depth": True})
    out_dir = tmp_path / "ckpt"
    train(
        dataset_root=tmp_path / "data",
        output_dir=out_dir,
        steps=2,
        batch_size=2,
        config=cfg,
        device="cpu",
    )
    policy = ACTPolicy.load(out_dir, device="cpu")
    assert policy.cfg.use_depth
    a = policy.select_action(
        [np.zeros((32, 32, 3), dtype=np.uint8), np.zeros((32, 32, 3), dtype=np.uint8)],
        np.zeros(6, dtype=np.float32),
        depth_u16=np.ones((32, 32), dtype=np.uint16) * 500,
    )
    assert a.shape == (6,)


def test_jax_train_two_steps(tmp_path: Path) -> None:
    pytest = __import__("pytest")
    jax = pytest.importorskip("jax")
    del jax
    root = _synthetic_dataset(tmp_path / "data")
    out_dir = tmp_path / "ckpt_jax"
    cfg = ACTConfig(**{**_tiny_cfg().to_dict(), "backend": "jax", "backbone": "tiny"})
    train(
        dataset_root=root,
        output_dir=out_dir,
        steps=2,
        batch_size=2,
        config=cfg,
        backend="jax",
    )
    policy = ACTPolicy.load(out_dir, device="cpu")
    a = policy.select_action(np.zeros((32, 32, 3), dtype=np.uint8), np.zeros(6, dtype=np.float32))
    assert a.shape == (6,)


def test_act_forward_backward() -> None:
    cfg = _tiny_cfg()
    model = ACT(cfg)
    b, c = 2, cfg.chunk_size
    images = torch.rand(b, 3, cfg.image_size, cfg.image_size)
    state = torch.randn(b, 6)
    actions = torch.randn(b, c, 6)
    mask = torch.ones(b, c)
    mask[:, -2:] = 0
    out = model(images=images, state=state, actions=actions, action_mask=mask)
    assert out["pred_actions"].shape == (b, c, 6)
    assert out["loss"].ndim == 0
    out["loss"].backward()
    grads = [p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad]
    assert any(grads)


def test_act_forward_two_cameras() -> None:
    cfg = _tiny_cfg()
    model = ACT(cfg)
    images = torch.rand(2, 2, 3, cfg.image_size, cfg.image_size)
    state = torch.randn(2, 6)
    actions = torch.randn(2, cfg.chunk_size, 6)
    mask = torch.ones(2, cfg.chunk_size)
    out = model(images=images, state=state, actions=actions, action_mask=mask)
    assert out["pred_actions"].shape == (2, cfg.chunk_size, 6)
    out["loss"].backward()


def test_train_two_cameras_then_select_action(tmp_path: Path) -> None:
    ds = SO101Dataset.create(
        tmp_path / "data",
        DatasetMeta(
            name="dual",
            fps=15,
            joint_names=list(JOINT_NAMES),
            camera_name="wrist",
            camera_names=("wrist", "overhead"),
            camera_width=32,
            camera_height=32,
            task="synth",
            episode_count=0,
        ),
    )
    rng = np.random.default_rng(0)
    for i in range(2):
        w = EpisodeWriter(ds, i)
        for k in range(8):
            w.add_frame(
                rng.normal(size=6).astype(np.float32),
                rng.normal(size=6).astype(np.float32),
                k / 15.0,
                images={
                    "wrist": rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8),
                    "overhead": rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8),
                },
            )
        w.save()
    out_dir = tmp_path / "ckpt"
    train(dataset_root=tmp_path / "data", output_dir=out_dir, steps=2, batch_size=2, config=_tiny_cfg(), device="cpu")
    policy = ACTPolicy.load(out_dir, device="cpu")
    a = policy.select_action(
        [np.zeros((32, 32, 3), dtype=np.uint8), np.zeros((32, 32, 3), dtype=np.uint8)],
        np.zeros(6, dtype=np.float32),
    )
    assert a.shape == (6,)


def test_split_episodes_disjoint_and_deterministic() -> None:
    train_a, val_a = split_episodes(10, val_frac=0.2, seed=0)
    train_b, val_b = split_episodes(10, val_frac=0.2, seed=0)
    train_c, val_c = split_episodes(10, val_frac=0.2, seed=1)
    assert set(train_a).isdisjoint(val_a)
    assert sorted(train_a + val_a) == list(range(10))
    assert len(val_a) >= 1
    assert len(train_a) >= 1
    assert train_a == train_b and val_a == val_b
    assert (train_a, val_a) != (train_c, val_c)


def test_split_episodes_two_keeps_one_val() -> None:
    train_ep, val_ep = split_episodes(2, val_frac=0.2, seed=0)
    assert len(train_ep) == 1
    assert len(val_ep) == 1
    assert set(train_ep + val_ep) == {0, 1}


def test_split_episodes_single_is_train_only() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        train_ep, val_ep = split_episodes(1, val_frac=0.2, seed=0)
    assert train_ep == [0]
    assert val_ep == []
    assert any("train-only" in str(w.message).lower() or "1 episode" in str(w.message).lower() for w in caught)


def test_find_policy_weights_prefers_best_val(tmp_path: Path) -> None:
    (tmp_path / "last").mkdir()
    (tmp_path / "best_val").mkdir()
    (tmp_path / "last" / "model.pt").write_bytes(b"last")
    (tmp_path / "best_val" / "model.pt").write_bytes(b"best")
    (tmp_path / "model.pt").write_bytes(b"legacy")
    assert find_policy_weights(tmp_path) == tmp_path / "best_val" / "model.pt"


def test_find_policy_weights_legacy_root(tmp_path: Path) -> None:
    (tmp_path / "model.pt").write_bytes(b"legacy")
    assert find_policy_weights(tmp_path) == tmp_path / "model.pt"


def _biased_dataset(tmp_path: Path) -> Path:
    """Two episodes whose means differ so pooling val would change stats."""
    ds = SO101Dataset.create(
        tmp_path,
        DatasetMeta(
            name="biased",
            fps=15,
            joint_names=list(JOINT_NAMES),
            camera_name="front",
            camera_width=32,
            camera_height=32,
            task="synth",
            episode_count=0,
        ),
    )
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    for i, fill in enumerate((0.0, 100.0)):
        w = EpisodeWriter(ds, i)
        for k in range(4):
            state = np.full(6, fill, dtype=np.float32)
            w.add_frame(state, state + 1.0, k / 15.0, img)
        w.save()
    return tmp_path


def test_train_two_steps_then_select_action(tmp_path: Path) -> None:
    root = _synthetic_dataset(tmp_path / "data")
    out_dir = tmp_path / "ckpt"
    cfg = _tiny_cfg()
    train(dataset_root=root, output_dir=out_dir, steps=2, batch_size=2, config=cfg, device="cpu")
    assert (out_dir / "last" / "model.pt").is_file() or (out_dir / "model.pt").is_file()
    assert (out_dir / "config.json").is_file()
    assert (out_dir / "stats.json").is_file()
    metrics = (out_dir / "metrics.jsonl").read_text().strip().splitlines()
    assert len(metrics) == 2
    policy = ACTPolicy.load(out_dir, device="cpu")
    policy.reset()
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    state = np.zeros(6, dtype=np.float32)
    a = policy.select_action(img, state)
    assert a.shape == (6,)
    assert np.isfinite(a).all()
    a2 = policy.select_action(img, state)
    assert a2.shape == (6,)


def test_select_action_replans_only_when_queue_empty() -> None:
    cfg = _tiny_cfg()
    cfg.n_action_steps = 4
    model = ACT(cfg)
    stats = {
        "state_mean": [0.0] * 6,
        "state_std": [1.0] * 6,
        "action_mean": [0.0] * 6,
        "action_std": [1.0] * 6,
    }
    policy = ACTPolicy(model, stats, device="cpu")
    policy.reset()
    calls = {"n": 0}
    orig = policy._predict_chunk

    def counted(image_rgb, state, depth_u16=None):
        calls["n"] += 1
        return orig(image_rgb, state, depth_u16=depth_u16)

    policy._predict_chunk = counted  # type: ignore[method-assign]
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    st = np.zeros(6, dtype=np.float32)
    for _ in range(4):
        policy.select_action(img, st)
    assert calls["n"] == 1
    policy.select_action(img, st)
    assert calls["n"] == 2


def test_train_stats_from_train_split_only(tmp_path: Path) -> None:
    root = _biased_dataset(tmp_path / "data")
    out_dir = tmp_path / "ckpt"
    train(
        dataset_root=root,
        output_dir=out_dir,
        steps=2,
        batch_size=2,
        config=_tiny_cfg(),
        device="cpu",
        seed=0,
        val_frac=0.5,
        eval_every=2,
    )
    split = json.loads((out_dir / "split.json").read_text())
    assert set(split["train_episodes"]).isdisjoint(split["val_episodes"])
    expected = compute_stats(SO101Dataset(root), episode_indices=split["train_episodes"])
    saved = json.loads((out_dir / "stats.json").read_text())
    np.testing.assert_allclose(saved["state_mean"], expected["state_mean"], rtol=1e-5)
    all_stats = compute_stats(SO101Dataset(root))
    assert saved["state_mean"][0] != all_stats["state_mean"][0]


def test_train_writes_val_l1_and_last_best_val(tmp_path: Path) -> None:
    root = _synthetic_dataset(tmp_path / "data", episodes=4, t=8)
    out_dir = tmp_path / "ckpt"
    train(
        dataset_root=root,
        output_dir=out_dir,
        steps=4,
        batch_size=2,
        config=_tiny_cfg(),
        device="cpu",
        seed=0,
        val_frac=0.25,
        eval_every=2,
    )
    split = json.loads((out_dir / "split.json").read_text())
    assert split["seed"] == 0
    assert set(split["train_episodes"]).isdisjoint(split["val_episodes"])
    assert split["val_episodes"]
    cfg = json.loads((out_dir / "config.json").read_text())
    assert cfg["seed"] == 0
    assert cfg["eval_every"] == 2
    assert cfg["val_frac"] == 0.25
    rows = [json.loads(line) for line in (out_dir / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    assert all({"step", "loss", "l1", "kl"} <= row.keys() for row in rows)
    val_rows = [row for row in rows if "val_l1" in row]
    assert len(val_rows) >= 2
    assert (out_dir / "last" / "model.pt").is_file()
    assert (out_dir / "best_val" / "model.pt").is_file()
    policy = ACTPolicy.load(out_dir, device="cpu")
    a = policy.select_action(np.zeros((32, 32, 3), dtype=np.uint8), np.zeros(6, dtype=np.float32))
    assert a.shape == (6,)
