"""Tests for HybridGraspPolicy (base + depth grasp)."""

from __future__ import annotations

import numpy as np

from so101_ml.dataset.schema import JOINT_NAMES
from so101_ml.hybrid.depth_grasp import DepthGraspConfig
from so101_ml.hybrid.hybrid_policy import HybridGraspPolicy


class _StubBase:
    def __init__(self) -> None:
        self.resets = 0
        self.calls = 0

    def reset(self) -> None:
        self.resets += 1

    def select_action(self, image_rgb, state, depth_u16=None):
        self.calls += 1
        out = np.asarray(state, dtype=np.float32).copy()
        out[JOINT_NAMES.index("gripper")] = 80.0
        return out


def test_hybrid_uses_depth_near_object() -> None:
    base = _StubBase()
    cfg = DepthGraspConfig(
        enter_mm=200.0,
        target_mm=100.0,
        tol_mm=10.0,
        close_gripper=True,
        gripper_closed=11.0,
        hold_close_frames=2,
        gain_deg_per_mm=0.0,
    )
    policy = HybridGraspPolicy(base, grasp_cfg=cfg)
    policy.reset()
    assert base.resets == 1
    img = np.zeros((24, 32, 3), dtype=np.uint8)
    state = np.zeros(6, dtype=np.float32)
    depth = np.full((48, 64), 100, dtype=np.uint16)
    out = policy.select_action(img, state, depth_u16=depth)
    assert out[JOINT_NAMES.index("gripper")] == 11.0
    assert policy.cfg.use_depth is True


def test_hybrid_passthrough_when_far() -> None:
    base = _StubBase()
    cfg = DepthGraspConfig(enter_mm=120.0, target_mm=90.0)
    policy = HybridGraspPolicy(base, grasp_cfg=cfg)
    img = np.zeros((24, 32, 3), dtype=np.uint8)
    state = np.arange(6, dtype=np.float32)
    depth = np.full((48, 64), 500, dtype=np.uint16)
    out = policy.select_action([img, img], state, depth_u16=depth)
    assert out[JOINT_NAMES.index("gripper")] == 80.0
    assert base.calls == 1
