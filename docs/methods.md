# Methods

Offline visuomotor learning for SO-101 pick-and-place (*Place keys into metal cup*).
Hardware apparatus and live robot loops: [full stack repo](https://github.com/romil0001/SO101_Visuomotor_Manipulation).

## Research question

Can imitation learning (ACT / SmolVLA), interactive corrections (HG-DAgger), and
depth-assisted contact control improve **protocol-measured** success on a
low-cost arm—reporting `pc_success`, not train loss?

## Dataset

Local episode folders (no Hub required for ACT):

| Modality | File | Format |
|----------|------|--------|
| State | `state.npy` | float32 [T, 6] |
| Action | `action.npy` | float32 [T, 6] |
| RGB | `wrist.mp4`, `overhead.mp4` | H×W×3 |
| Depth | `wrist_depth.npy` | uint16 mm (optional) |

DAgger corrections set `source=dagger` and `intervention=true` in `episode.json`.

## ACT

Zhao et al. (2023): ResNet18 + CVAE transformer, chunk size 100, L1 + KL.
Episode holdout (`val_frac=0.2`); checkpoint on offline `val_l1` (`best_val/`).
Optional metric depth in the vision stack; optional JAX/Flax tiny backbone.

## SmolVLA

Convert demos with `python -m so101_ml convert`, then fine-tune with LeRobot.
Depth is dropped at export. Language task string matches recording.

## Human-gated DAgger

Ross et al. (2011) DAgger; Kelly et al. (2019) HG-DAgger.

1. Policy rolls out on the robot (full stack).
2. Operator pauses; leader moves to follower pose; human corrects.
3. Only correction frames are saved.
4. This repo merges demos + corrections and fine-tunes ACT (`dagger-finetune`),
   forcing correction episodes into the train split.

## Hybrid depth grasp

Base policy handles approach/transport; near contact, an analytic controller
adjusts vertical joints from wrist depth ROI, then releases for place.

## Protocol evaluation

YAML conditions (keys left / center / right), repeated trials, binary success.
Primary metric: `pc_success` and `per_condition`. Compare runs with
`compare-eval`.

## Reinforcement learning (roadmap)

After a measured DAgger improvement, the next stage is **human-in-the-loop RL**
(HIL-SERL / Luo et al. 2024): sparse success reward (classifier or manual),
demo buffer + online SAC/RLPD, human interventions for safety. Not shipped in
v1 of this showcase—implementation belongs with the full-stack robot loop.

## References

- Zhao et al. (2023). ACT. *CoRL*.
- Ross, Gordon, Bagnell (2011). DAgger. *AISTATS*.
- Kelly et al. (2019). HG-DAgger. *ICRA*.
- Luo et al. (2024). HIL-SERL. Human-in-the-loop RL for manipulation.
- Hugging Face LeRobot / SmolVLA documentation.
