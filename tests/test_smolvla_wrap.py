"""SmolVLA wrap unit test with a stubbed LeRobot policy (no Hub download)."""

from __future__ import annotations

import numpy as np
import torch

from so101_ml.dataset.schema import JOINT_NAMES
from so101_ml.models.smolvla_wrap import SmolVLAPolicyWrap, SmolVLAWrapConfig


class _StubPolicy:
    def __init__(self) -> None:
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def select_action(self, batch):
        return torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def test_smolvla_wrap_select_action_numpy() -> None:
    from lerobot.utils.feature_utils import hw_to_dataset_features

    joint = {f"{n}.pos": float for n in JOINT_NAMES}
    obs = dict(joint)
    obs["wrist"] = (24, 32, 3)
    obs["overhead"] = (24, 32, 3)
    ds_features = {
        **hw_to_dataset_features(joint, "action"),
        **hw_to_dataset_features(obs, "observation"),
    }

    def identity(x):
        return x

    wrap = SmolVLAPolicyWrap(
        _StubPolicy(),
        identity,
        identity,
        ds_features,
        device="cpu",
        cfg=SmolVLAWrapConfig(
            camera_names=("wrist", "overhead"),
            policy_camera_names=("wrist", "overhead"),
            image_height=24,
            image_width=32,
        ),
    )
    wrap.reset()
    imgs = [np.zeros((24, 32, 3), dtype=np.uint8), np.zeros((24, 32, 3), dtype=np.uint8)]
    out = wrap.select_action(imgs, np.zeros(6, dtype=np.float32))
    assert out.shape == (6,)
    np.testing.assert_allclose(out, [1, 2, 3, 4, 5, 6])
