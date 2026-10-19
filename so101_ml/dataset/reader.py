from __future__ import annotations

from pathlib import Path

import numpy as np

from so101_ml.dataset.schema import DatasetMeta, Episode, EpisodeMeta
from so101_ml.dataset.video import read_mp4


def episode_dir(root: Path, index: int) -> Path:
    return root / "episodes" / f"{index:06d}"


class SO101Dataset:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        meta_path = self.root / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"No dataset meta at {meta_path}")
        import json

        self.meta = DatasetMeta.from_dict(json.loads(meta_path.read_text()))

    @classmethod
    def create(cls, root: Path, meta: DatasetMeta) -> SO101Dataset:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        (root / "episodes").mkdir(exist_ok=True)
        import json

        (root / "meta.json").write_text(json.dumps(meta.to_dict(), indent=2) + "\n")
        return cls(root)

    def save_meta(self) -> None:
        import json

        (self.root / "meta.json").write_text(json.dumps(self.meta.to_dict(), indent=2) + "\n")

    def next_episode_index(self) -> int:
        return self.meta.episode_count

    def __len__(self) -> int:
        return self.meta.episode_count

    def load_episode(self, index: int) -> Episode:
        if index < 0 or index >= self.meta.episode_count:
            raise IndexError(f"Episode {index} out of range (count={self.meta.episode_count})")
        import json

        folder = episode_dir(self.root, index)
        emeta = EpisodeMeta.from_dict(json.loads((folder / "episode.json").read_text()))
        state = np.load(folder / "state.npy")
        action = np.load(folder / "action.npy")
        timestamps = np.load(folder / "timestamps.npy")
        video_paths = {name: folder / f"{name}.mp4" for name in self.meta.names()}
        first = self.meta.camera_name
        depth_path = None
        if self.meta.use_depth:
            candidate = folder / f"{self.meta.depth_camera_name}_depth.npy"
            if candidate.is_file():
                depth_path = candidate
        return Episode(
            index=emeta.index,
            num_frames=emeta.num_frames,
            task=emeta.task,
            state=state,
            action=action,
            timestamps=timestamps,
            video_path=video_paths[first],
            camera_name=first,
            video_paths=video_paths,
            depth_path=depth_path,
            source=emeta.source,
            intervention=emeta.intervention,
        )


def read_episode_frames(episode: Episode) -> np.ndarray:
    return read_mp4(episode.video_path)
