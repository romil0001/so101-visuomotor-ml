# HG-DAgger fine-tuning (offline stage)

On-robot intervention collection is performed in the companion repository (`python -m so101 dagger`). This package aggregates those corrections with the demonstration corpus and fine-tunes ACT.

```bash
python -m so101_ml dagger-finetune \
  --dataset so101_d405_demo \
  --corrections so101_d405_demo_dagger \
  --init-checkpoint outputs/train/act_so101_d405_demo \
  --steps 8000
```

Correction episodes are forced into the training split; the validation holdout remains drawn from original demonstrations. The baseline checkpoint directory is left unchanged for fair `compare-eval`.
