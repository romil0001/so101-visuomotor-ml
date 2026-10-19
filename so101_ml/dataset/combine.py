"""Stack a teleop dataset and a DAgger correction dataset into one training root."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path

from so101_ml.dataset.reader import SO101Dataset, episode_dir
from so101_ml.dataset.schema import DatasetMeta


@dataclass(frozen=True)
class CombineResult:
    root: Path
    n_base: int
    n_corrections: int
    correction_indices: tuple[int, ...]

    @property
    def episode_count(self) -> int:
        return self.n_base + self.n_corrections


def _same_layout(a: DatasetMeta, b: DatasetMeta) -> None:
    if a.names() != b.names():
        raise ValueError(f"Camera mismatch: base {list(a.names())} vs corrections {list(b.names())}")
    if (a.camera_width, a.camera_height, a.fps) != (b.camera_width, b.camera_height, b.fps):
        raise ValueError("fps or image size differs between base and corrections")
    if bool(a.use_depth) != bool(b.use_depth):
        raise ValueError(f"use_depth base={a.use_depth} corrections={b.use_depth}")
    if list(a.joint_names) != list(b.joint_names):
        raise ValueError("joint_names differ between base and corrections")


def combine_datasets(base: Path, corrections: Path, dest: Path) -> CombineResult:
    """Copy base episodes, then correction episodes. Rewrites episode indices.

    Correction episodes are marked source=dagger so training can force them
    into the train split. dest is replaced.
    """
    base_ds = SO101Dataset(base)
    corr_ds = SO101Dataset(corrections)
    _same_layout(base_ds.meta, corr_ds.meta)
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    out = SO101Dataset.create(
        dest,
        replace(base_ds.meta, name=dest.name, episode_count=0),
    )
    mapping: list[dict[str, int | str]] = []
    index = 0

    def _copy(src_ds: SO101Dataset, kind: str, n: int) -> None:
        nonlocal index
        for src_i in range(n):
            src = episode_dir(src_ds.root, src_i)
            dst = episode_dir(out.root, index)
            shutil.copytree(src, dst)
            raw = json.loads((dst / "episode.json").read_text())
            raw["index"] = index
            if kind == "dagger":
                raw["source"] = "dagger"
                raw["intervention"] = True
            (dst / "episode.json").write_text(json.dumps(raw, indent=2) + "\n")
            mapping.append({"index": index, "from": kind, "source_index": src_i})
            index += 1

    _copy(base_ds, "base", base_ds.meta.episode_count)
    n_base = index
    _copy(corr_ds, "dagger", corr_ds.meta.episode_count)
    n_corr = index - n_base
    out.meta = replace(out.meta, episode_count=index)
    out.save_meta()
    correction_indices = tuple(range(n_base, index))
    (dest / "combine.json").write_text(
        json.dumps(
            {
                "base": str(Path(base).resolve()),
                "corrections": str(Path(corrections).resolve()),
                "n_base": n_base,
                "n_corrections": n_corr,
                "correction_indices": list(correction_indices),
                "episodes": mapping,
            },
            indent=2,
        )
        + "\n"
    )
    return CombineResult(
        root=dest,
        n_base=n_base,
        n_corrections=n_corr,
        correction_indices=correction_indices,
    )
