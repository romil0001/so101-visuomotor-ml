from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

N_JOINTS = len(JOINT_NAMES)


@dataclass(frozen=True)
class DatasetMeta:
    name: str
    fps: int
    joint_names: list[str]
    camera_name: str
    camera_width: int
    camera_height: int
    task: str
    episode_count: int
    camera_names: tuple[str, ...] = ()
    use_depth: bool = False
    depth_camera_name: str = "wrist"

    def names(self) -> tuple[str, ...]:
        if self.camera_names:
            return tuple(self.camera_names)
        return (self.camera_name,)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["camera_names"] = list(self.names())
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetMeta:
        raw_names = data.get("camera_names")
        camera_names = tuple(str(x) for x in raw_names) if raw_names else ()
        camera_name = str(data["camera_name"]) if data.get("camera_name") else camera_names[0]
        if not camera_names:
            camera_names = (camera_name,)
        return cls(
            name=str(data["name"]),
            fps=int(data["fps"]),
            joint_names=list(data["joint_names"]),
            camera_name=camera_name,
            camera_width=int(data["camera_width"]),
            camera_height=int(data["camera_height"]),
            task=str(data["task"]),
            episode_count=int(data["episode_count"]),
            camera_names=camera_names,
            use_depth=bool(data.get("use_depth", False)),
            depth_camera_name=str(data.get("depth_camera_name", "wrist")),
        )


@dataclass(frozen=True)
class EpisodeMeta:
    index: int
    num_frames: int
    task: str
    source: str = "teleop"
    intervention: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpisodeMeta:
        return cls(
            index=int(data["index"]),
            num_frames=int(data["num_frames"]),
            task=str(data["task"]),
            source=str(data.get("source", "teleop")),
            intervention=bool(data.get("intervention", False)),
        )


@dataclass
class Episode:
    index: int
    num_frames: int
    task: str
    state: Any
    action: Any
    timestamps: Any
    video_path: Path
    camera_name: str
    video_paths: dict[str, Path] | None = None
    depth_path: Path | None = None
    source: str = "teleop"
    intervention: bool = False

    def load_frames(self, camera_name: str | None = None) -> Any:
        from so101_ml.dataset.video import read_mp4

        name = camera_name or self.camera_name
        paths = self.video_paths or {self.camera_name: self.video_path}
        return read_mp4(paths[name])

    def load_depth(self) -> Any:
        import numpy as np

        if self.depth_path is None or not self.depth_path.is_file():
            raise FileNotFoundError(f"No depth file for episode {self.index}")
        return np.load(self.depth_path)
