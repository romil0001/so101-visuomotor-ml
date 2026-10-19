"""Offline human-gated DAgger: merge corrections and fine-tune ACT.

Physical pause / leader-sync / takeover runs in the full-stack repo
(https://github.com/romil0001/SO101_Visuomotor_Manipulation). This module
trains on demos + correction episodes already on disk.
"""

from __future__ import annotations

from pathlib import Path

from so101_ml.dataset.combine import CombineResult, combine_datasets
from so101_ml.train import resolve_act_weights, train


def combine_demos_and_corrections(
    demos: str | Path,
    corrections: str | Path,
    dest: str | Path,
) -> CombineResult:
    """Stack teleop demos then DAgger correction episodes (forced into train later)."""
    return combine_datasets(Path(demos), Path(corrections), Path(dest))


def finetune_act_with_corrections(
    *,
    demos: str | Path,
    corrections: str | Path,
    output_dir: str | Path,
    init_checkpoint: str | Path,
    steps: int = 8_000,
    batch_size: int = 2,
    val_frac: float = 0.2,
    seed: int = 0,
    eval_every: int = 500,
    combined_dir: str | Path | None = None,
) -> Path:
    """Combine datasets, force corrections into the train split, fine-tune ACT."""
    demos = Path(demos)
    corrections = Path(corrections)
    output_dir = Path(output_dir)
    combined = Path(combined_dir) if combined_dir else demos.parent / f"{demos.name}__with_dagger"
    merged = combine_datasets(demos, corrections, combined)
    init = resolve_act_weights(Path(init_checkpoint))
    return train(
        dataset_root=merged.root,
        output_dir=output_dir,
        steps=steps,
        batch_size=batch_size,
        val_frac=val_frac,
        seed=seed,
        eval_every=eval_every,
        force_train=merged.correction_indices,
        init_checkpoint=init,
        backend="torch",
    )
