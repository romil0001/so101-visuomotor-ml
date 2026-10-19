# Compare protocol summaries

```bash
python -m so101_ml compare-eval \
  --baseline path/to/baseline/summary.json \
  --candidate path/to/dagger_eval/summary.json
```

`delta_pc_success` is candidate minus baseline (overall and per condition).
