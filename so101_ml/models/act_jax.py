"""Flax/JAX ACT — same API as PyTorch ACTPolicy for train/eval."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax import serialization
from flax.core import freeze, unfreeze

from so101_ml.models.act import ACTConfig, find_policy_weights, preprocess_depth, preprocess_image


@dataclass
class TrainState:
    params: Any
    opt_state: Any


class TinyBackbone(nn.Module):
    out_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Conv(32, (3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(32, (3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(self.out_dim, (3, 3), strides=(2, 2), padding="SAME")(x)
        return x


class DepthEncoder(nn.Module):
    dim_model: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Conv(32, (3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(self.dim_model, (3, 3), strides=(2, 2), padding="SAME")(x)
        return x


class TransformerBlock(nn.Module):
    dim_model: int
    n_heads: int
    dim_feedforward: int
    dropout: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads, qkv_features=self.dim_model, dropout_rate=self.dropout
        )
        y = nn.LayerNorm(name="ln1")(x)
        y = attn(y, y, deterministic=not train)
        x = x + y
        y = nn.LayerNorm(name="ln2")(x)
        y = nn.Dense(self.dim_feedforward, name="ff1")(y)
        y = nn.relu(y)
        y = nn.Dense(self.dim_model, name="ff2")(y)
        return x + y


class ACTModule(nn.Module):
    cfg: ACTConfig

    @nn.compact
    def __call__(
        self,
        images: jnp.ndarray,
        state: jnp.ndarray,
        actions: jnp.ndarray | None = None,
        action_mask: jnp.ndarray | None = None,
        depth: jnp.ndarray | None = None,
        *,
        train: bool = True,
        rngs: dict[str, jax.Array] | None = None,
    ) -> dict[str, jnp.ndarray]:
        cfg = self.cfg
        if cfg.backbone == "tiny":
            backbone = TinyBackbone(cfg.dim_model)
            proj = lambda x: x
            _backbone_dim = cfg.dim_model
        else:
            raise ValueError("JAX ACT supports backbone='tiny' only (use torch for resnet18)")

        if images.ndim == 4:
            images = images[:, None, ...]
        b, n, _c, _h, _w = images.shape
        flat = images.reshape(b * n, _c, _h, _w)
        feat = backbone(flat)
        feat = proj(feat)
        tokens = feat.reshape(b * n, -1, cfg.dim_model)
        hw = tokens.shape[1]
        img_tokens = tokens.reshape(b, n * hw, cfg.dim_model)

        if depth is not None:
            if not cfg.use_depth:
                raise ValueError("depth provided but use_depth is false")
            d_feat = DepthEncoder(cfg.dim_model)(depth[:, None, ...])
            d_tokens = d_feat.reshape(b, -1, cfg.dim_model)
            img_tokens = jnp.concatenate([img_tokens, d_tokens], axis=1)

        state_tok = nn.Dense(cfg.dim_model, name="state_proj")(state)[:, None, :]
        enc_pos = self.param("enc_pos", nn.initializers.normal(stddev=0.02), (1, 256, cfg.dim_model))
        dec_queries = self.param(
            "dec_queries", nn.initializers.normal(stddev=0.02), (1, cfg.chunk_size, cfg.dim_model)
        )
        latent_pos = self.param(
            "latent_pos",
            nn.initializers.normal(stddev=0.02),
            (1, 1 + cfg.chunk_size, cfg.dim_model),
        )

        if actions is not None:
            act_tok = nn.Dense(cfg.dim_model, name="action_proj")(actions)
            lat = jnp.concatenate([state_tok, act_tok], axis=1) + latent_pos[:, : 1 + cfg.chunk_size]
            for i in range(max(1, cfg.n_encoder_layers // 2)):
                lat = TransformerBlock(cfg.dim_model, cfg.n_heads, cfg.dim_feedforward, cfg.dropout, name=f"lat_{i}")(
                    lat, train=train
                )
            mu = nn.Dense(cfg.latent_dim, name="mu")(lat[:, 0])
            logvar = jnp.clip(nn.Dense(cfg.latent_dim, name="logvar")(lat[:, 0]), -10, 10)
            key = rngs["dropout"] if rngs else jax.random.PRNGKey(0)
            std = jnp.exp(0.5 * logvar)
            z = mu + std * jax.random.normal(key, mu.shape)
            kl = -0.5 * jnp.mean(jnp.sum(1 + logvar - mu**2 - jnp.exp(logvar), axis=-1))
        else:
            z = jnp.zeros((b, cfg.latent_dim))
            kl = jnp.array(0.0, dtype=jnp.float32)

        z_tok = nn.Dense(cfg.dim_model, name="z_proj")(z)[:, None, :]
        enc = jnp.concatenate([z_tok, state_tok, img_tokens], axis=1)
        enc = enc + enc_pos[:, : enc.shape[1]]
        for i in range(cfg.n_encoder_layers):
            enc = TransformerBlock(cfg.dim_model, cfg.n_heads, cfg.dim_feedforward, cfg.dropout, name=f"enc_{i}")(
                enc, train=train
            )
        memory = enc
        queries = jnp.broadcast_to(dec_queries, (b, cfg.chunk_size, cfg.dim_model))
        for i in range(cfg.n_decoder_layers):
            attn = nn.MultiHeadDotProductAttention(
                num_heads=cfg.n_heads,
                qkv_features=cfg.dim_model,
                dropout_rate=cfg.dropout,
                name=f"dec_attn_{i}",
            )
            y = nn.LayerNorm(name=f"dec_ln1_{i}")(queries)
            y = attn(y, memory, deterministic=not train)
            queries = queries + y
            y = nn.LayerNorm(name=f"dec_ln2_{i}")(queries)
            y = nn.Dense(cfg.dim_feedforward, name=f"dec_ff1_{i}")(y)
            y = nn.relu(y)
            y = nn.Dense(cfg.dim_model, name=f"dec_ff2_{i}")(y)
            queries = queries + y
        pred = nn.Dense(cfg.n_joints, name="action_head")(queries)

        if actions is not None:
            if action_mask is None:
                action_mask = jnp.ones((b, cfg.chunk_size))
            mask = action_mask[..., None]
            l1 = jnp.sum(jnp.abs(pred - actions) * mask) / jnp.maximum(jnp.sum(mask), 1.0)
            loss = l1 + cfg.kl_weight * kl
        else:
            l1 = jnp.array(0.0, dtype=jnp.float32)
            loss = l1
        return {"pred_actions": pred, "loss": loss, "l1": l1, "kl": kl}


def init_act_params(cfg: ACTConfig, rng: jax.Array, *, n_cams: int = 1) -> Any:
    module = ACTModule(cfg)
    dummy_img = jnp.zeros((1, n_cams, 3, cfg.image_size, cfg.image_size))
    dummy_state = jnp.zeros((1, cfg.n_joints))
    dummy_actions = jnp.zeros((1, cfg.chunk_size, cfg.n_joints))
    dummy_mask = jnp.ones((1, cfg.chunk_size))
    kwargs: dict = {
        "images": dummy_img,
        "state": dummy_state,
        "actions": dummy_actions,
        "action_mask": dummy_mask,
        "train": False,
    }
    if cfg.use_depth:
        kwargs["depth"] = jnp.zeros((1, cfg.image_size, cfg.image_size))
    variables = module.init(rng, **kwargs)
    return variables["params"]


class ACTPolicyJAX:
    def __init__(self, params: Any, cfg: ACTConfig, stats: dict[str, Any]) -> None:
        self.params = params
        self.cfg = cfg
        self.module = ACTModule(cfg)
        self.state_mean = np.asarray(stats["state_mean"], dtype=np.float32)
        self.state_std = np.asarray(stats["state_std"], dtype=np.float32)
        self.action_mean = np.asarray(stats["action_mean"], dtype=np.float32)
        self.action_std = np.asarray(stats["action_std"], dtype=np.float32)
        self._action_queue: deque[np.ndarray] = deque()

    def reset(self) -> None:
        self._action_queue = deque()

    def _predict_chunk(
        self,
        image_rgb: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...],
        state: np.ndarray,
        depth_u16: np.ndarray | None = None,
    ) -> np.ndarray:
        if isinstance(image_rgb, (list, tuple)):
            imgs = [preprocess_image(im, self.cfg.image_size, self.cfg.backbone) for im in image_rgb]
            img = jnp.array(np.stack([im.numpy() for im in imgs])[None, ...])
        else:
            t = preprocess_image(image_rgb, self.cfg.image_size, self.cfg.backbone)
            img = jnp.array(t.numpy()[None, None, ...])
        st = (np.asarray(state, dtype=np.float32) - self.state_mean) / self.state_std
        st_j = jnp.array(st[None, ...])
        depth_j = None
        if self.cfg.use_depth:
            if depth_u16 is None:
                raise ValueError("depth_u16 required when use_depth is true")
            depth_j = jnp.array(
                preprocess_depth(depth_u16, self.cfg.image_size, self.cfg.depth_max_mm).numpy()[None, ...]
            )
        out = self.module.apply(
            {"params": self.params},
            img,
            st_j,
            actions=None,
            action_mask=None,
            depth=depth_j,
            train=False,
        )
        pred = np.asarray(out["pred_actions"][0])
        return pred * self.action_std + self.action_mean

    def select_action(
        self,
        image_rgb: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...],
        state: np.ndarray,
        depth_u16: np.ndarray | None = None,
    ) -> np.ndarray:
        if not self._action_queue:
            chunk = self._predict_chunk(image_rgb, state, depth_u16=depth_u16)
            n = min(int(self.cfg.n_action_steps), int(self.cfg.chunk_size), chunk.shape[0])
            self._action_queue.extend(np.asarray(row, dtype=np.float32) for row in chunk[:n])
        return np.asarray(self._action_queue.popleft(), dtype=np.float32)

    @classmethod
    def load(cls, directory: Path, device: str = "cpu") -> ACTPolicyJAX:
        del device  # JAX picks default backend (CPU/GPU)
        directory = Path(directory)
        cfg_dir = directory if (directory / "config.json").is_file() else directory.parent
        cfg = ACTConfig.from_dict(json.loads((cfg_dir / "config.json").read_text()))
        stats = json.loads((cfg_dir / "stats.json").read_text())
        weights = find_policy_weights(directory)
        cfg_data = json.loads((cfg_dir / "config.json").read_text())
        n_cams = len(cfg_data.get("camera_names") or ["front"])
        template = init_act_params(cfg, jax.random.PRNGKey(0), n_cams=n_cams)
        with open(weights, "rb") as f:
            params = serialization.from_bytes(template, f.read())
        return cls(params, cfg, stats)

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text(json.dumps(self.cfg.to_dict(), indent=2) + "\n")
        stats = {
            "state_mean": self.state_mean.tolist(),
            "state_std": self.state_std.tolist(),
            "action_mean": self.action_mean.tolist(),
            "action_std": self.action_std.tolist(),
        }
        (directory / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
        with open(directory / "model.msgpack", "wb") as f:
            f.write(serialization.to_bytes(self.params))
