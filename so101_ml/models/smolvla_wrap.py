"""SmolVLA adapter with the same select_action(images, state) interface as ACT."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from so101_ml.dataset.schema import JOINT_NAMES


@dataclass
class SmolVLAWrapConfig:
    use_depth: bool = False
    depth_camera_name: str = "wrist"
    camera_names: tuple[str, ...] = ("wrist", "overhead")
    # Policy/checkpoint camera keys (SmolVLA base uses camera1/camera2/…).
    policy_camera_names: tuple[str, ...] = ("camera1", "camera2")
    robot_type: str = "so101_follower"
    task: str = "Place keys into metal cup"
    image_height: int = 480
    image_width: int = 640


class SmolVLAPolicyWrap:
    """Thin wrapper: numpy images/state → LeRobot SmolVLA → joint qpos."""

    def __init__(
        self,
        policy: Any,
        preprocess: Any,
        postprocess: Any,
        ds_features: dict[str, Any],
        *,
        device: str = "cpu",
        cfg: SmolVLAWrapConfig | None = None,
    ) -> None:
        self.policy = policy
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.ds_features = ds_features
        self.device = torch.device(device)
        self.cfg = cfg or SmolVLAWrapConfig()

    @classmethod
    def load(
        cls,
        checkpoint: str | Path,
        *,
        device: str | None = None,
        task: str = "Place keys into metal cup",
        camera_names: tuple[str, ...] | list[str] = ("wrist", "overhead"),
        policy_camera_names: tuple[str, ...] | list[str] | None = None,
        image_height: int = 480,
        image_width: int = 640,
    ) -> SmolVLAPolicyWrap:
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.smolvla import SmolVLAPolicy
        from lerobot.utils.feature_utils import hw_to_dataset_features

        from so101_ml.train import detect_device

        device = device or detect_device()
        path = str(checkpoint)
        model = SmolVLAPolicy.from_pretrained(path)
        cam_names = tuple(str(n) for n in camera_names)
        if policy_camera_names is None:
            # Prefer checkpoint image keys; fall back to camera1.. for smolvla_base.
            img_keys = []
            feats = getattr(model.config, "input_features", None) or {}
            for k in feats:
                if "images" in str(k):
                    img_keys.append(str(k).split(".")[-1])
            policy_cams = tuple(img_keys) if img_keys else tuple(f"camera{i+1}" for i in range(len(cam_names)))
        else:
            policy_cams = tuple(str(n) for n in policy_camera_names)
        if len(policy_cams) < len(cam_names):
            raise ValueError(f"policy_camera_names {policy_cams} shorter than camera_names {cam_names}")
        policy_cams = policy_cams[: len(cam_names)]

        rename_map = {
            f"observation.images.{src}": f"observation.images.{dst}"
            for src, dst in zip(cam_names, policy_cams, strict=True)
            if src != dst
        }
        preprocess, postprocess = make_pre_post_processors(
            model.config,
            path,
            preprocessor_overrides={
                "device_processor": {"device": str(device)},
                **({"rename_observations_processor": {"rename_map": rename_map}} if rename_map else {}),
            },
        )
        joint = {f"{name}.pos": float for name in JOINT_NAMES}
        obs: dict[str, Any] = dict(joint)
        for name in policy_cams:
            obs[str(name)] = (int(image_height), int(image_width), 3)
        ds_features = {
            **hw_to_dataset_features(joint, "action"),
            **hw_to_dataset_features(obs, "observation"),
        }
        cfg = SmolVLAWrapConfig(
            camera_names=cam_names,
            policy_camera_names=policy_cams,
            task=task,
            image_height=image_height,
            image_width=image_width,
        )
        wrap = cls(model, preprocess, postprocess, ds_features, device=device, cfg=cfg)
        wrap.reset()
        return wrap

    def reset(self) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def select_action(
        self,
        image_rgb: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...],
        state: np.ndarray,
        depth_u16: np.ndarray | None = None,  # unused — depth is hybrid-only
    ) -> np.ndarray:
        del depth_u16
        from lerobot.policies.utils import build_inference_frame, make_robot_action

        if isinstance(image_rgb, (list, tuple)):
            images = list(image_rgb)
        else:
            images = [image_rgb]
        if len(images) != len(self.cfg.camera_names):
            raise ValueError(
                f"Got {len(images)} images but camera_names={self.cfg.camera_names}"
            )
        observation: dict[str, Any] = {
            f"{name}.pos": float(state[i]) for i, name in enumerate(JOINT_NAMES)
        }
        for pol_name, img in zip(self.cfg.policy_camera_names, images, strict=True):
            observation[pol_name] = np.asarray(img)
        frame = build_inference_frame(
            observation=observation,
            device=self.device,
            ds_features=self.ds_features,
            task=self.cfg.task,
            robot_type=self.cfg.robot_type,
        )
        batch = self.preprocess(frame)
        with torch.inference_mode():
            action_t = self.policy.select_action(batch)
        action_t = self.postprocess(action_t)
        robot_action = make_robot_action(action_t, self.ds_features)
        return np.array(
            [float(robot_action[f"{name}.pos"]) for name in JOINT_NAMES],
            dtype=np.float32,
        )
