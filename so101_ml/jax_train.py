"""Train ACT with Flax/JAX (tiny backbone). Depth-aware when dataset has depth."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from torch.utils.data import DataLoader

from so101_ml.dataset import SO101Dataset, compute_stats, write_stats
from so101_ml.models.act import ACTConfig
from so101_ml.models.act_jax import ACTModule, init_act_params
from so101_ml.train import ACTChunkDataset, split_episodes


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _to_jax_batch(batch: dict) -> dict:
    out = {
        "images": jnp.array(batch["images"].numpy()),
        "state": jnp.array(batch["state"].numpy()),
        "actions": jnp.array(batch["actions"].numpy()),
        "action_mask": jnp.array(batch["action_mask"].numpy()),
    }
    if "depth" in batch:
        out["depth"] = jnp.array(batch["depth"].numpy())
    return out


def _save_weights(params, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(params))


def train_jax(
    *,
    dataset_root: Path,
    output_dir: Path,
    steps: int = 40_000,
    batch_size: int = 4,
    config: ACTConfig | None = None,
    val_frac: float = 0.2,
    seed: int = 0,
    eval_every: int = 1000,
    grad_clip: float = 1.0,
    lr_backbone: float = 1e-5,
    lr_rest: float = 1e-4,
) -> Path:
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = config or ACTConfig(backend="jax", backbone="tiny")
    cfg = ACTConfig(**{**cfg.to_dict(), "backend": "jax"})
    if cfg.backbone != "tiny":
        print("Warning: JAX backend uses backbone='tiny' (resnet18 is torch-only)")
        cfg = ACTConfig(**{**cfg.to_dict(), "backbone": "tiny"})
    set_seed(seed)

    ds = SO101Dataset(dataset_root)
    if cfg.use_depth != ds.meta.use_depth:
        cfg = ACTConfig(
            **{
                **cfg.to_dict(),
                "use_depth": ds.meta.use_depth,
                "depth_camera_name": ds.meta.depth_camera_name,
            }
        )
    train_episodes, val_episodes = split_episodes(ds.meta.episode_count, val_frac=val_frac, seed=seed)
    stats = compute_stats(ds, episode_indices=train_episodes)
    write_stats(output_dir, stats)
    (output_dir / "split.json").write_text(
        json.dumps(
            {"seed": seed, "train_episodes": train_episodes, "val_episodes": val_episodes},
            indent=2,
        )
        + "\n"
    )

    data = ACTChunkDataset(dataset_root, cfg, stats, episode_indices=train_episodes)
    if len(data) < 1:
        raise ValueError(f"No frames in train split of {dataset_root}")
    bs = min(batch_size, len(data))
    loader = DataLoader(data, batch_size=bs, shuffle=True, num_workers=0, drop_last=len(data) > bs)
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
        "backend": "jax",
    }
    (output_dir / "config.json").write_text(json.dumps(run_config, indent=2) + "\n")

    n_cams = len(ds.meta.names())
    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    params = init_act_params(cfg, init_rng, n_cams=n_cams)
    module = ACTModule(cfg)
    tx = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(lr_rest))
    opt_state = tx.init(params)

    @jax.jit
    def loss_fn(params, batch, dropout_key):
        out = module.apply(
            {"params": params},
            batch["images"],
            batch["state"],
            batch["actions"],
            batch["action_mask"],
            batch.get("depth"),
            train=True,
            rngs={"dropout": dropout_key},
        )
        return out["loss"], out

    @jax.jit
    def train_step(params, opt_state, batch, dropout_key):
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch, dropout_key)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, metrics

    @jax.jit
    def eval_step(params, batch):
        out = module.apply(
            {"params": params},
            batch["images"],
            batch["state"],
            batch["actions"],
            batch["action_mask"],
            batch.get("depth"),
            train=False,
        )
        return out["loss"], out["l1"]

    metrics_path = output_dir / "metrics.jsonl"
    it = iter(loader)
    t0 = time.time()
    best_val_l1 = float("inf")
    for step in range(1, steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        jbatch = _to_jax_batch(batch)
        rng, dropout_key = jax.random.split(rng)
        params, opt_state, loss, metrics = train_step(params, opt_state, jbatch, dropout_key)
        row = {
            "step": step,
            "loss": float(loss),
            "l1": float(metrics["l1"]),
            "kl": float(metrics["kl"]),
        }
        do_val = val_loader is not None and (step % eval_every == 0 or step == steps)
        if do_val:
            assert val_loader is not None
            val_losses: list[float] = []
            val_l1s: list[float] = []
            for vb in val_loader:
                vj = _to_jax_batch(vb)
                vloss, vl1 = eval_step(params, vj)
                val_losses.append(float(vloss))
                val_l1s.append(float(vl1))
            val_loss = float(np.mean(val_losses))
            val_l1 = float(np.mean(val_l1s))
            row["val_l1"] = val_l1
            row["val_loss"] = val_loss
            _save_weights(params, output_dir / "last" / "model.msgpack")
            if val_l1 < best_val_l1:
                best_val_l1 = val_l1
                _save_weights(params, output_dir / "best_val" / "model.msgpack")
        with metrics_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        if step == 1 or step % 50 == 0 or step == steps or do_val:
            dt = time.time() - t0
            extra = f"  val_l1={row['val_l1']:.4f}" if "val_l1" in row else ""
            print(
                f"step {step}/{steps}  loss={row['loss']:.4f}  l1={row['l1']:.4f}  "
                f"kl={row['kl']:.4f}{extra}  {dt:.1f}s  backend=jax",
                flush=True,
            )
    _save_weights(params, output_dir / "last" / "model.msgpack")
    return output_dir
