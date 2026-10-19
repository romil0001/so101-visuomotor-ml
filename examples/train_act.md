# Training ACT (offline)

Requires a local dataset in the episode schema of [`docs/methods.md`](../docs/methods.md).

```bash
pip install -e ".[dev]"
python -m so101_ml train \
  --dataset so101_d405_demo \
  --steps 40000 \
  --val-frac 0.2 \
  --seed 0
```

Checkpoints: `outputs/train/act_<dataset>/best_val/model.pt` (preferred for protocol comparison) and `last/model.pt`. Validation L1 is used only for checkpoint selection, not as a task score.
