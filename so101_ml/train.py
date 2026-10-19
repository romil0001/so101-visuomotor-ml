"""Train ACT on a local SO101Dataset. Mac MPS / CUDA / CPU. No Hub."""

from __future__ import annotations

import json
import random
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from so101_ml.dataset import SO101Dataset, compute_stats, write_stats
from so101_ml.models.act import ACT, ACTConfig, preprocess_depth, preprocess_image


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_episodes(
    n_episodes: int,
    val_frac: float = 0.2,
    seed: int = 0,
    force_train: list[int] | tuple[int, ...] | None = None,
) -> tuple[list[int], list[int]]:
    """Hold out episodes for val. Indices in force_train always stay in train.

    DAgger corrections are forced into train so the val score stays a holdout
    of the original demonstrations.
    """
    if n_episodes < 1:
        raise ValueError("Need at least one episode to split")
    forced = sorted({int(i) for i in (force_train or [])})
    if any(i < 0 or i >= n_episodes for i in forced):
        raise ValueError(f"force_train indices {forced} outside 0..{n_episodes - 1}")
    pool = [i for i in range(n_episodes) if i not in set(forced)]
    if len(pool) <= 1:
        if len(pool) == 1 or n_episodes == 1:
            warnings.warn("Too few episodes for a val split; train-only", stacklevel=2)
        return sorted(pool + forced), []
    rng = np.random.default_rng(seed)
    indices = np.array(pool, dtype=np.int64)
    rng.shuffle(indices)
    n_val = int(round(len(pool) * val_frac))
    n_val = max(1, n_val)
    n_val = min(n_val, len(pool) - 1)
    val = sorted(int(i) for i in indices[:n_val])
    train = sorted(int(i) for i in list(indices[n_val:]) + forced)
    return train, val


class ACTChunkDataset(Dataset):
    def __init__(
        self,
        root: Path,
        cfg: ACTConfig,
        stats: dict,
        episode_indices: list[int] | None = None,
    ) -> None:
        self.ds = SO101Dataset(root)
        self.cfg = cfg
        self.state_mean = np.asarray(stats["state_mean"], dtype=np.float32)
        self.state_std = np.asarray(stats["state_std"], dtype=np.float32)
        self.action_mean = np.asarray(stats["action_mean"], dtype=np.float32)
        self.action_std = np.asarray(stats["action_std"], dtype=np.float32)
        self.episodes: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]] = []
        self.index: list[tuple[int, int]] = []
        self.camera_names = list(self.ds.meta.names())
        self.use_depth = bool(self.ds.meta.use_depth)
        if episode_indices is None:
            episode_indices = list(range(self.ds.meta.episode_count))
        for local_i, ep_i in enumerate(episode_indices):
            ep = self.ds.load_episode(ep_i)
            per_cam = []
            for name in self.camera_names:
                frames = ep.load_frames(name)
                resized = np.stack(
                    [
                        cv2.resize(f, (cfg.image_size, cfg.image_size), interpolation=cv2.INTER_AREA)
                        for f in frames
                    ]
                )
                per_cam.append(resized)
            stacked = np.stack(per_cam, axis=1)
            depth_stack: np.ndarray | None = None
            if self.use_depth:
                depth_frames = ep.load_depth()
                depth_stack = np.stack(
                    [
                        cv2.resize(z, (cfg.image_size, cfg.image_size), interpolation=cv2.INTER_NEAREST)
                        for z in depth_frames
                    ]
                )
            self.episodes.append((ep.state, ep.action, stacked, depth_stack))
            for t in range(ep.num_frames):
                self.index.append((local_i, t))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        ep_i, t = self.index[i]
        state, action, frames, depth_stack = self.episodes[ep_i]
        t_len = state.shape[0]
        cams = frames[t]
        imgs = [
            preprocess_image(cams[i], self.cfg.image_size, self.cfg.backbone)
            for i in range(cams.shape[0])
        ]
        img = torch.stack(imgs, dim=0)
        st = (state[t] - self.state_mean) / self.state_std
        chunk = np.zeros((self.cfg.chunk_size, self.cfg.n_joints), dtype=np.float32)
        mask = np.zeros((self.cfg.chunk_size,), dtype=np.float32)
        end = min(t + self.cfg.chunk_size, t_len)
        n = end - t
        chunk[:n] = (action[t:end] - self.action_mean) / self.action_std
        mask[:n] = 1.0
        out: dict[str, torch.Tensor] = {
            "images": img,
            "state": torch.from_numpy(st.astype(np.float32)),
            "actions": torch.from_numpy(chunk),
            "action_mask": torch.from_numpy(mask),
        }
        if self.use_depth:
            assert depth_stack is not None
            out["depth"] = preprocess_depth(depth_stack[t], self.cfg.image_size, self.cfg.depth_max_mm)
        return out


def resolve_act_weights(path: Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    for cand in (path / "best_val" / "model.pt", path / "last" / "model.pt", path / "model.pt"):
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"No ACT weights under {path} (expected best_val/model.pt)")


def _load_init_weights(model: ACT, path: Path) -> Path:
    ckpt = resolve_act_weights(path)
    try:
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    return ckpt


def _save_weights(model: ACT, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def _batch_depth(batch: dict, device: str) -> torch.Tensor | None:
    depth = batch.get("depth")
    if depth is None:
        return None
    return depth.to(device)


def _validate(model: ACT, loader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    l1s: list[float] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            state = batch["state"].to(device)
            actions = batch["actions"].to(device)
            mask = batch["action_mask"].to(device)
            out = model(
                images=images, state=state, actions=actions, action_mask=mask, depth=_batch_depth(batch, device)
            )
            losses.append(float(out["loss"].detach().cpu()))
            l1s.append(float(out["l1"].detach().cpu()))
    model.train()
    return float(np.mean(losses)), float(np.mean(l1s))


def train(
    *,
    dataset_root: Path,
    output_dir: Path,
    steps: int = 40_000,
    batch_size: int = 2,
    config: ACTConfig | None = None,
    device: str | None = None,
    lr_backbone: float = 1e-5,
    lr_rest: float = 1e-4,
    val_frac: float = 0.2,
    seed: int = 0,
    eval_every: int = 1000,
    grad_clip: float = 1.0,
    backend: str = "torch",
    force_train: list[int] | tuple[int, ...] | None = None,
    init_checkpoint: str | Path | None = None,
) -> Path:
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = config or ACTConfig()
    cfg = ACTConfig(**{**cfg.to_dict(), "backend": backend})
    if backend == "jax":
        if force_train or init_checkpoint:
            raise ValueError(
                "DAgger fine-tune (--corrections / --init-checkpoint) is implemented for --backend torch"
            )
        from so101_ml.jax_train import train_jax

        return train_jax(
            dataset_root=dataset_root,
            output_dir=output_dir,
            steps=steps,
            batch_size=batch_size,
            config=cfg,
            val_frac=val_frac,
            seed=seed,
            eval_every=eval_every,
            grad_clip=grad_clip,
            lr_backbone=lr_backbone,
            lr_rest=lr_rest,
        )
    device = device or detect_device()
    set_seed(seed)

    ds = SO101Dataset(dataset_root)
    train_episodes, val_episodes = split_episodes(
        ds.meta.episode_count, val_frac=val_frac, seed=seed, force_train=force_train
    )
    if cfg.use_depth != ds.meta.use_depth:
        cfg = ACTConfig(**{**cfg.to_dict(), "use_depth": ds.meta.use_depth, "depth_camera_name": ds.meta.depth_camera_name})
    stats = compute_stats(ds, episode_indices=train_episodes)
    write_stats(output_dir, stats)
    (output_dir / "split.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "train_episodes": train_episodes,
                "val_episodes": val_episodes,
                "force_train": list(force_train or []),
            },
            indent=2,
        )
        + "\n"
    )

    data = ACTChunkDataset(dataset_root, cfg, stats, episode_indices=train_episodes)
    if len(data) < 1:
        raise ValueError(f"No frames in train split of {dataset_root}")
    bs = min(batch_size, len(data))
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(
        data, batch_size=bs, shuffle=True, num_workers=0, drop_last=len(data) > bs, generator=g
    )
    val_loader: DataLoader | None = None
    if val_episodes:
        val_data = ACTChunkDataset(dataset_root, cfg, stats, episode_indices=val_episodes)
        if len(val_data) >= 1:
            val_bs = min(batch_size, len(val_data))
            val_loader = DataLoader(val_data, batch_size=val_bs, shuffle=False, num_workers=0)

    run_config = {
        **cfg.to_dict(),
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "val_frac": val_frac,
        "eval_every": eval_every,
        "task": ds.meta.task,
        "camera_names": list(ds.meta.names()),
        "lr_backbone": lr_backbone,
        "lr_rest": lr_rest,
        "grad_clip": grad_clip,
        "backend": backend,
        "init_checkpoint": str(init_checkpoint or ""),
        "force_train": list(force_train or []),
    }
    (output_dir / "config.json").write_text(json.dumps(run_config, indent=2) + "\n")

    model = ACT(cfg).to(device)
    if init_checkpoint:
        loaded = _load_init_weights(model, Path(init_checkpoint))
        lr_backbone *= 0.1
        lr_rest *= 0.1
        print(f"Initialized from {loaded}  lr_backbone={lr_backbone}  lr_rest={lr_rest}", flush=True)
        model = model.to(device)
    backbone_params = list(model.backbone.parameters()) + list(model.proj.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    rest = [p for p in model.parameters() if id(p) not in backbone_ids]
    opt = torch.optim.AdamW(
        [
            {"params": [p for p in backbone_params if p.requires_grad], "lr": lr_backbone},
            {"params": rest, "lr": lr_rest},
        ]
    )
    metrics_path = output_dir / "metrics.jsonl"
    model.train()
    it = iter(loader)
    t0 = time.time()
    best_val_l1 = float("inf")
    for step in range(1, steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        images = batch["images"].to(device)
        state = batch["state"].to(device)
        actions = batch["actions"].to(device)
        mask = batch["action_mask"].to(device)
        opt.zero_grad(set_to_none=True)
        out = model(
            images=images, state=state, actions=actions, action_mask=mask, depth=_batch_depth(batch, device)
        )
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        row = {
            "step": step,
            "loss": float(out["loss"].detach().cpu()),
            "l1": float(out["l1"].detach().cpu()),
            "kl": float(out["kl"].detach().cpu()),
        }
        do_val = val_loader is not None and (step % eval_every == 0 or step == steps)
        if do_val:
            assert val_loader is not None
            val_loss, val_l1 = _validate(model, val_loader, device)
            row["val_l1"] = val_l1
            row["val_loss"] = val_loss
            _save_weights(model, output_dir / "last" / "model.pt")
            if val_l1 < best_val_l1:
                best_val_l1 = val_l1
                _save_weights(model, output_dir / "best_val" / "model.pt")
        with metrics_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        if step == 1 or step % 50 == 0 or step == steps or do_val:
            dt = time.time() - t0
            extra = f"  val_l1={row['val_l1']:.4f}" if "val_l1" in row else ""
            print(
                f"step {step}/{steps}  loss={row['loss']:.4f}  l1={row['l1']:.4f}  "
                f"kl={row['kl']:.4f}{extra}  {dt:.1f}s  device={device}",
                flush=True,
            )
    _save_weights(model, output_dir / "last" / "model.pt")
    return output_dir


def train_from_env(
    *,
    dataset_name: str = "so101_d405_demo",
    steps: int = 40_000,
    batch_size: int = 0,
    output_dir: str = "",
    val_frac: float = 0.2,
    seed: int = 0,
    eval_every: int = 1000,
    backend: str = "torch",
    corrections_name: str = "",
    init_checkpoint: str = "",
) -> Path:
    from so101_ml.paths import project_root
    from so101_ml.dataset.combine import combine_datasets

    root = project_root()
    dataset_root = root / "data" / "datasets" / dataset_name
    if not (dataset_root / "meta.json").is_file():
        raise FileNotFoundError(f"No local dataset at {dataset_root}. Provide a local dataset under data/datasets/")
    force_train: tuple[int, ...] | None = None
    init_path = init_checkpoint.strip()
    if corrections_name.strip():
        corr_root = root / "data" / "datasets" / corrections_name.strip()
        if not (corr_root / "meta.json").is_file():
            raise FileNotFoundError(
                f"No corrections at {corr_root}. Collect corrections in the full-stack repo (python -m so101 dagger)"
            )
        n_corr = SO101Dataset(corr_root).meta.episode_count
        if n_corr < 1:
            raise ValueError(f"Corrections dataset {corr_root} has no episodes")
        combined = root / "data" / "datasets" / f"{dataset_name}__with_dagger"
        merged = combine_datasets(dataset_root, corr_root, combined)
        print(
            f"Combined {merged.n_base} demos + {merged.n_corrections} corrections → {combined}",
            flush=True,
        )
        dataset_root = merged.root
        force_train = merged.correction_indices
        if not init_path:
            guess = root / "outputs" / "train" / f"act_{dataset_name}"
            try:
                init_path = str(resolve_act_weights(guess))
                print(f"Fine-tune init: {init_path}", flush=True)
            except FileNotFoundError:
                print(f"No baseline checkpoint at {guess}; training the combined set from scratch", flush=True)
        if not output_dir:
            output_dir = str(root / "outputs" / "train" / f"act_{dataset_name}_dagger")
    meta = SO101Dataset(dataset_root).meta
    device = detect_device() if backend == "torch" else "cpu"
    if batch_size <= 0:
        if backend == "jax":
            batch_size = 4
        else:
            batch_size = 8 if device == "cuda" else 2
    out = Path(output_dir) if output_dir else root / "outputs" / "train" / f"act_{dataset_name}"
    cfg = ACTConfig(
        use_depth=meta.use_depth,
        depth_camera_name=meta.depth_camera_name,
        backend=backend,
    )
    print(
        f"ACT train  dataset={dataset_root}  backend={backend}  device={device}  "
        f"batch={batch_size}  steps={steps}  depth={meta.use_depth}"
    )
    print(f"  seed={seed}  val_frac={val_frac}  eval_every={eval_every}")
    print(f"Checkpoints: {out}/last and {out}/best_val (pick best_val for protocol eval)")
    return train(
        dataset_root=dataset_root,
        output_dir=out,
        steps=steps,
        batch_size=batch_size,
        config=cfg,
        device=device,
        val_frac=val_frac,
        seed=seed,
        eval_every=eval_every,
        backend=backend,
        force_train=force_train,
        init_checkpoint=init_path or None,
    )
