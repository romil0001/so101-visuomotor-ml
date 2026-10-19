"""Depth grasp servo: metric wrist depth overrides vertical joints near contact."""

from __future__ import annotations

import numpy as np

from so101_ml.dataset.schema import JOINT_NAMES
from so101_ml.hybrid.depth_grasp import DepthGraspConfig, DepthGraspController, median_roi_depth_mm


def _depth(h: int, w: int, fill_mm: int, *, invalid: int = 0) -> np.ndarray:
    z = np.full((h, w), fill_mm, dtype=np.uint16)
    z[:2, :] = invalid
    return z


def test_median_roi_depth_center_patch() -> None:
    z = np.zeros((100, 100), dtype=np.uint16)
    z[40:60, 40:60] = 120
    med = median_roi_depth_mm(z, roi_frac=0.2, min_mm=1.0, max_mm=2000.0)
    assert med is not None
    assert abs(med - 120.0) < 1.0


def test_median_roi_ignores_invalid() -> None:
    z = np.zeros((40, 40), dtype=np.uint16)
    assert median_roi_depth_mm(z, roi_frac=0.5, min_mm=1.0, max_mm=2000.0) is None


def test_idle_passthrough_when_far() -> None:
    cfg = DepthGraspConfig(enter_mm=150.0, target_mm=90.0)
    ctrl = DepthGraspController(cfg)
    action = np.array([1, 2, 3, 4, 5, 50], dtype=np.float32)
    state = action.copy()
    out = ctrl.override(action, state, _depth(48, 64, 400))
    np.testing.assert_array_equal(out, action)
    assert ctrl.phase == "idle"


def test_enters_grasp_and_pulls_closer_when_too_far() -> None:
    cfg = DepthGraspConfig(
        enter_mm=200.0,
        target_mm=100.0,
        tol_mm=10.0,
        gain_deg_per_mm=0.1,
        vertical_joints=("elbow_flex",),
        joint_signs={"elbow_flex": 1.0},
        close_gripper=False,
    )
    ctrl = DepthGraspController(cfg)
    state = np.zeros(6, dtype=np.float32)
    action = np.zeros(6, dtype=np.float32)
    # 150 mm > target 100 → positive error → +elbow_flex with sign +1
    out = ctrl.override(action, state, _depth(48, 64, 150))
    assert ctrl.phase == "grasping"
    ei = JOINT_NAMES.index("elbow_flex")
    assert out[ei] > action[ei]


def test_closes_gripper_when_in_tolerance() -> None:
    cfg = DepthGraspConfig(
        enter_mm=200.0,
        target_mm=100.0,
        tol_mm=15.0,
        close_gripper=True,
        gripper_closed=12.0,
        hold_close_frames=3,
    )
    ctrl = DepthGraspController(cfg)
    state = np.zeros(6, dtype=np.float32)
    action = np.array([0, 0, 0, 0, 0, 80], dtype=np.float32)
    out = ctrl.override(action, state, _depth(48, 64, 105))
    gi = JOINT_NAMES.index("gripper")
    assert out[gi] == 12.0
    assert ctrl.phase in {"closing", "done"}


def test_done_passthrough_after_hold() -> None:
    cfg = DepthGraspConfig(
        enter_mm=200.0,
        target_mm=100.0,
        tol_mm=20.0,
        close_gripper=True,
        gripper_closed=10.0,
        hold_close_frames=2,
        gain_deg_per_mm=0.0,
    )
    ctrl = DepthGraspController(cfg)
    state = np.zeros(6, dtype=np.float32)
    action = np.array([1, 2, 3, 4, 5, 90], dtype=np.float32)
    ctrl.override(action, state, _depth(48, 64, 100))
    ctrl.override(action, state, _depth(48, 64, 100))
    out = ctrl.override(action, state, _depth(48, 64, 100))
    assert ctrl.phase == "done"
    np.testing.assert_array_equal(out, action)


def test_reset_returns_to_idle() -> None:
    cfg = DepthGraspConfig(enter_mm=200.0, target_mm=100.0, close_gripper=False)
    ctrl = DepthGraspController(cfg)
    ctrl.override(np.zeros(6, np.float32), np.zeros(6, np.float32), _depth(48, 64, 120))
    assert ctrl.phase == "grasping"
    ctrl.reset()
    assert ctrl.phase == "idle"


def test_none_depth_passthrough() -> None:
    ctrl = DepthGraspController(DepthGraspConfig())
    action = np.arange(6, dtype=np.float32)
    out = ctrl.override(action, action, None)
    np.testing.assert_array_equal(out, action)
