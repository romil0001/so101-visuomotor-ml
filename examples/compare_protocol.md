# Comparing protocol runs

Given two scored evaluation directories (each with `summary.json` from the companion stack):

```bash
python -m so101_ml compare-eval \
  --baseline path/to/baseline/summary.json \
  --candidate path/to/post_dagger/summary.json
```

The report includes overall and per-condition `delta_pc_success` (candidate − baseline). Prefer the same YAML protocol, camera configuration, and task string across runs.
