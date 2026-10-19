"""Metric wrist-depth grasp assist for the last centimeters before contact."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from so101_ml.dataset.schema import JOINT_NAMES


@dataclass
class DepthGraspConfig:
    """Tune for keys-on-table with D405 wrist cam (mm)."""

    enabled: bool = True
    roi_frac: float = 0.3
    enter_mm: float = 180.0
    target_mm: float = 95.0
    exit_mm: float = 280.0
    tol_mm: float = 15.0
    min_mm: float = 40.0
    max_mm: float = 2000.0
    gain_deg_per_mm: float = 0.08
    vertical_joints: tuple[str, ...] = ("shoulder_lift", "elbow_flex")
    # Positive sign: increase joint when object is farther than target (depth > target).
    joint_signs: dict[str, float] = field(
        default_factory=lambda: {"shoulder_lift": 1.0, "elbow_flex": 1.0}
    )
    max_delta_deg: float = 4.0
    close_gripper: bool = True
    gripper_closed: float = 15.0
    hold_close_frames: int = 8

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DepthGraspConfig:
        fields = cls.__dataclass_fields__
        kwargs: dict[str, Any] = {}
        for k, v in data.items():
            if k not in fields:
                continue
            if k == "vertical_joints":
                kwargs[k] = tuple(str(x) for x in v)
            elif k == "joint_signs":
                kwargs[k] = {str(a): float(b) for a, b in dict(v).items()}
            else:
                kwargs[k] = v
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "roi_frac": self.roi_frac,
            "enter_mm": self.enter_mm,
            "target_mm": self.target_mm,
            "exit_mm": self.exit_mm,
            "tol_mm": self.tol_mm,
            "min_mm": self.min_mm,
            "max_mm": self.max_mm,
            "gain_deg_per_mm": self.gain_deg_per_mm,
            "vertical_joints": list(self.vertical_joints),
            "joint_signs": dict(self.joint_signs),
            "max_delta_deg": self.max_delta_deg,
            "close_gripper": self.close_gripper,
            "gripper_closed": self.gripper_closed,
            "hold_close_frames": self.hold_close_frames,
        }


def median_roi_depth_mm(
    depth_u16: np.ndarray,
    *,
    roi_frac: float = 0.3,
    min_mm: float = 40.0,
    max_mm: float = 2000.0,
) -> float | None:
    """Median depth (mm) in a centered ROI; None if no valid pixels."""
    z = np.asarray(depth_u16)
    if z.ndim != 2:
        raise ValueError(f"depth must be HxW, got {z.shape}")
    h, w = z.shape
    rf = float(np.clip(roi_frac, 0.05, 1.0))
    rh, rw = max(1, int(h * rf)), max(1, int(w * rf))
    y0 = (h - rh) // 2
    x0 = (w - rw) // 2
    patch = z[y0 : y0 + rh, x0 : x0 + rw].astype(np.float32)
    valid = patch[(patch >= min_mm) & (patch <= max_mm)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


class DepthGraspController:
    """State machine: idle → grasping → closing → done (until reset)."""

    def __init__(self, cfg: DepthGraspConfig | None = None) -> None:
        self.cfg = cfg or DepthGraspConfig()
        self.phase: str = "idle"
        self._close_left: int = 0
        self.last_depth_mm: float | None = None

    def reset(self) -> None:
        self.phase = "idle"
        self._close_left = 0
        self.last_depth_mm = None

    def override(
        self,
        action: np.ndarray,
        state: np.ndarray,
        depth_u16: np.ndarray | None,
    ) -> np.ndarray:
        out = np.asarray(action, dtype=np.float32).copy()
        if not self.cfg.enabled or depth_u16 is None or self.phase == "done":
            return out

        med = median_roi_depth_mm(
            depth_u16,
            roi_frac=self.cfg.roi_frac,
            min_mm=self.cfg.min_mm,
            max_mm=self.cfg.max_mm,
        )
        self.last_depth_mm = med

        if self.phase == "closing":
            return self._force_close(out)

        if med is None:
            return out

        if self.phase == "idle":
            if med <= self.cfg.enter_mm:
                self.phase = "grasping"
            else:
                return out

        if self.phase == "grasping" and med >= self.cfg.exit_mm:
            self.phase = "idle"
            return out

        if self.phase != "grasping":
            return out

        error = med - self.cfg.target_mm  # >0 → too far
        for name in self.cfg.vertical_joints:
            if name not in JOINT_NAMES:
                continue
            i = JOINT_NAMES.index(name)
            sign = float(self.cfg.joint_signs.get(name, 1.0))
            delta = sign * self.cfg.gain_deg_per_mm * error
            delta = float(np.clip(delta, -self.cfg.max_delta_deg, self.cfg.max_delta_deg))
            # Correct around current state so we don't fight stale chunk targets too hard.
            out[i] = float(state[i]) + delta

        if abs(error) <= self.cfg.tol_mm and self.cfg.close_gripper:
            self.phase = "closing"
            self._close_left = max(1, int(self.cfg.hold_close_frames))
            return self._force_close(out)
        return out

    def _force_close(self, action: np.ndarray) -> np.ndarray:
        gi = JOINT_NAMES.index("gripper")
        action[gi] = float(self.cfg.gripper_closed)
        self._close_left -= 1
        if self._close_left <= 0:
            self.phase = "done"
        return action
