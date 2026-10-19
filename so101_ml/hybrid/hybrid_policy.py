"""Hybrid policy: base policy (ACT/SmolVLA) + depth grasp override."""

from __future__ import annotations

from typing import Any

import numpy as np

from so101_ml.hybrid.depth_grasp import DepthGraspConfig, DepthGraspController


class HybridGraspPolicy:
    """Wraps any policy with ``select_action(images, state, depth_u16=...)``.

    Depth is ignored by SmolVLA; the grasp controller uses it near contact.
    """

    def __init__(
        self,
        base: Any,
        grasp: DepthGraspController | None = None,
        *,
        grasp_cfg: DepthGraspConfig | None = None,
    ) -> None:
        self.base = base
        self.grasp = grasp or DepthGraspController(grasp_cfg or DepthGraspConfig())
        # Eval loop enables depth when policy.cfg.use_depth is True.
        base_cfg = getattr(base, "cfg", None)
        self.cfg = type("Cfg", (), {"use_depth": True, "depth_camera_name": "wrist"})()
        if base_cfg is not None and hasattr(base_cfg, "depth_camera_name"):
            self.cfg.depth_camera_name = base_cfg.depth_camera_name

    def reset(self) -> None:
        if hasattr(self.base, "reset"):
            self.base.reset()
        self.grasp.reset()

    def select_action(
        self,
        image_rgb: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...],
        state: np.ndarray,
        depth_u16: np.ndarray | None = None,
    ) -> np.ndarray:
        # Base policies may not accept depth kwargs (SmolVLA wrapper ignores it).
        try:
            action = self.base.select_action(image_rgb, state, depth_u16=depth_u16)
        except TypeError:
            action = self.base.select_action(image_rgb, state)
        return self.grasp.override(np.asarray(action, dtype=np.float32), state, depth_u16)
