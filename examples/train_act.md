# Train ACT

```bash
# Dataset layout: data/datasets/so101_d405_demo/{meta.json,episodes/...}
pip install -e ".[dev]"
python -m so101_ml train --dataset so101_d405_demo --steps 40000 --val-frac 0.2 --seed 0
# Checkpoints: outputs/train/act_so101_d405_demo/{best_val,last}/model.pt
```
