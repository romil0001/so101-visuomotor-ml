# Offline DAgger fine-tune

Record corrections on the robot with the full-stack `python -m so101 dagger`,
then:

```bash
python -m so101_ml dagger-finetune \
  --dataset so101_d405_demo \
  --corrections so101_d405_demo_dagger \
  --init-checkpoint outputs/train/act_so101_d405_demo \
  --steps 8000
```

Correction episodes are forced into the train split; val stays a holdout of demos.
