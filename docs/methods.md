# Methods

**Platform plant:** SO-101 (leader–follower, STS3215).  
**Task:** *Place keys into metal cup.*  
**Author:** Romil Modi.  
**Hardware and live control loops:** [SO101_Visuomotor_Manipulation](https://github.com/romil0001/SO101_Visuomotor_Manipulation).

This document describes the **learning and evaluation methodology** implemented in `so101_ml`. Apparatus construction, power, and camera networking are out of scope here.

## Research question

Can a low-cost visuomotor stack support (i) reproducible offline imitation (ACT; SmolVLA), (ii) interactive imitation via human-gated DAgger, and (iii) depth-assisted contact control, when success is measured by a **fixed protocol** with binary labels and condition strata—rather than by training loss?

## Apparatus (summary)

| Element | Specification |
|---------|----------------|
| Plant | SO-101 leader + follower, 6 DoF each |
| Sensing | Wrist RGB + metric depth (Intel RealSense D405); overhead RGB |
| Compute (learning) | Offline training on workstation (e.g. Apple MPS / CUDA) |
| Driver boundary | Hugging Face LeRobot for I/O in the companion repo; learning logic in `so101_ml` |

## Data collection and representation

Leader joint commands are recorded as **actions**; follower joint states as **states**. Cameras are fixed between record and eval via shared configuration in the companion stack.

Episode directory layout (`data/datasets/<name>/episodes/NNNNNN/`):

| Modality | File | Shape / format |
|----------|------|----------------|
| Proprioception | `state.npy` | float32 `[T, 6]` |
| Command | `action.npy` | float32 `[T, 6]` |
| Wrist / overhead RGB | `*.mp4` | native resolution, dataset fps |
| Wrist depth (optional) | `wrist_depth.npy` | uint16 millimetres, colour-aligned |

Joint order: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`.

DAgger correction episodes set `source=dagger` and `intervention=true` in `episode.json`. Only expert intervention frames are treated as supervised targets for fine-tuning; autonomous policy frames are not labelled as expert actions.

## Policies

### Action Chunking Transformer (ACT)

Implementation follows Zhao et al. (2023): ResNet-18 visual backbone, CVAE latent, transformer encoder–decoder, action chunks of length 100, L1 reconstruction plus KL regularisation. Training uses an episode-level holdout (`val_frac=0.2`, fixed seed). Checkpoints are selected on offline validation L1 (`best_val/`). Optional RGB + normalised wrist depth in the vision stack; optional JAX/Flax `tiny` backbone via `--backend jax`.

### SmolVLA

Demonstrations are converted to LeRobot format (`python -m so101_ml convert`) and fine-tuned with language task conditioning. **Depth is dropped at export.** Inference uses the same camera names and task string as recording, via `SmolVLAPolicyWrap`.

### Hybrid depth-assisted grasp

A base policy (ACT or SmolVLA) controls approach and transport. When wrist depth in a configured ROI indicates proximity to the target, `DepthGraspController` overrides vertical joints using metric depth, then returns control after contact / lift criteria. Depth is an intervention signal at contact, not an additional SmolVLA input.

### Human-gated DAgger (interactive imitation)

Behavioural cloning suffers from covariate shift when the learner visits states absent from demonstrations (Ross, Gordon, & Bagnell, 2011). Human-gated DAgger (Kelly et al., 2019) lets an expert intervene on demand.

**On-robot (companion repository):** policy rollout → pause → synchronise leader to follower pose → record correction window → resume or save.

**Offline (this repository):** `combine_datasets` concatenates demos and corrections; `dagger-finetune` / `train --corrections` forces correction indices into the train split, initialises from a prior ACT checkpoint (reduced learning rate), and writes a new run directory so the baseline checkpoint is preserved.

## Evaluation protocol

Scored evaluation uses a frozen YAML protocol (default: keys left / center / right). Each trial: home pose, scene reset, closed-loop rollout under timeout, binary human judgment against a written success criterion. Aggregate metrics:

- `pc_success` ∈ [0, 100] (percent in published summaries)
- `per_condition` rates
- Pairwise `delta_pc_success` via `compare-eval`

Train loss is explicitly not used as a manipulation score.

### Reported baseline (committed figures)

| Method | `n_trials` | `n_success` | `pc_success` | Completeness |
|--------|------------|-------------|--------------|--------------|
| ACT | 10 | 1 | 10.0 | Incomplete condition grid |
| SmolVLA | 15 | 0 | 0.0 | Full 5×3 grid |

See [`assets/eval/baseline.json`](../assets/eval/baseline.json). Low offline rates motivate DAgger and hybrid contact assistance.

## Reinforcement learning (next stage)

After a statistically and operationally meaningful DAgger improvement under the same protocol, we plan **human-in-the-loop sample-efficient RL** in the style of HIL-SERL (Luo, Xu, Wu, & Levine, 2024): offline demos, optional vision-based success classifier, online actor–learner with human interventions, and joint/workspace limits. That loop depends on real-robot I/O and is **not** part of this offline package.

## Reproducibility

```bash
python -m so101_ml train --dataset <name> --val-frac 0.2 --seed 0
python -m so101_ml dagger-finetune --dataset <name> --steps 8000
python -m so101_ml compare-eval --baseline <a>/summary.json --candidate <b>/summary.json
```

## References

1. Zhao, T. Z., Kumar, V., Levine, S., & Finn, C. (2023). Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware. *Conference on Robot Learning (CoRL)*.
2. Ross, S., Gordon, G., & Bagnell, J. A. (2011). A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. *AISTATS*.
3. Kelly, M., Sidrane, C., Driggs-Campbell, K., & Kochenderfer, M. J. (2019). HG-DAgger: Interactive Imitation Learning with Human Experts. *IEEE ICRA*.
4. Luo, J., Xu, C., Wu, J., & Levine, S. (2024). Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning. *arXiv:2410.21845*.
5. Hugging Face LeRobot. SO-101 documentation and SmolVLA. https://huggingface.co/docs/lerobot
