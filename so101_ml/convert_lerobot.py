"""Convert local SO101 episodes to a LeRobot dataset for SmolVLA fine-tuning.

Depth is intentionally dropped — SmolVLA is RGB + language; depth is used at
eval time by ``DepthGraspController``, not in the LeRobot export.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from so101_ml.dataset import JOINT_NAMES, SO101Dataset


def so101_lerobot_features(
    *,
    height: int,
    width: int,
    camera_names: tuple[str, ...] | list[str],
) -> dict[str, dict[str, Any]]:
    from lerobot.utils.feature_utils import hw_to_dataset_features

    joint = {f"{name}.pos": float for name in JOINT_NAMES}
    obs: dict[str, Any] = dict(joint)
    for name in camera_names:
        obs[name] = (int(height), int(width), 3)
    return {
        **hw_to_dataset_features(joint, "action"),
        **hw_to_dataset_features(obs, "observation"),
    }


def convert_so101_to_lerobot(
    source: str | Path,
    dest: str | Path,
    *,
    repo_id: str = "local/so101_d405_demo",
    robot_type: str = "so101_follower",
    max_episodes: int | None = None,
    image_writer_threads: int = 1,
    resume: bool = True,
) -> dict[str, Any]:
    """Write a LeRobot dataset under ``dest`` from a local SO101Dataset.

    If ``dest`` already has episodes and ``resume`` is True, append remaining
    source episodes (useful after an encoder crash mid-convert).
    """
    import gc

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source = Path(source)
    dest = Path(dest)
    src = SO101Dataset(source)
    names = src.meta.names()
    n_eps = src.meta.episode_count
    if max_episodes is not None:
        n_eps = min(n_eps, int(max_episodes))

    start_ep = 0
    if dest.exists() and (dest / "meta" / "info.json").is_file() and resume:
        ds = LeRobotDataset.resume(
            repo_id=repo_id,
            root=dest,
            image_writer_processes=0,
            image_writer_threads=image_writer_threads,
        )
        start_ep = int(ds.meta.total_episodes)
        print(f"Resuming convert at episode {start_ep:06d} (already have {start_ep})", flush=True)
    elif dest.exists() and any(dest.iterdir()):
        raise FileExistsError(
            f"Destination exists and is not empty: {dest}. Pass resume=True or delete it."
        )
    else:
        features = so101_lerobot_features(
            height=src.meta.camera_height,
            width=src.meta.camera_width,
            camera_names=names,
        )
        ds = LeRobotDataset.create(
            repo_id=repo_id,
            fps=int(src.meta.fps),
            features=features,
            root=dest,
            robot_type=robot_type,
            use_videos=True,
            image_writer_processes=0,
            image_writer_threads=image_writer_threads,
        )

    for ep_i in range(start_ep, n_eps):
        ep = src.load_episode(ep_i)
        frames_by_cam = {name: ep.load_frames(name) for name in names}
        task = str(ep.task or src.meta.task)
        for t in range(ep.num_frames):
            frame: dict[str, Any] = {
                "observation.state": np.asarray(ep.state[t], dtype=np.float32),
                "action": np.asarray(ep.action[t], dtype=np.float32),
                "task": task,
            }
            for name in names:
                frame[f"observation.images.{name}"] = np.asarray(frames_by_cam[name][t])
            ds.add_frame(frame)
        ds.save_episode(parallel_encoding=False)
        del frames_by_cam
        gc.collect()
        print(f"Converted episode {ep_i:06d}  frames={ep.num_frames}  task={task!r}", flush=True)

    ds.finalize()
    return {
        "repo_id": repo_id,
        "root": str(dest),
        "episode_count": n_eps,
        "fps": int(src.meta.fps),
        "task": src.meta.task,
        "camera_names": list(names),
        "use_depth": False,
        "resumed_from": start_ep,
    }
