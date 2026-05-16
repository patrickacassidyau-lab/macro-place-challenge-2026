# Ablation flags (`MACRO_PLACER_ABLATION`)

Run quick cohort comparisons for the innovation narrative (multi-pole flagship):

```bash
# Full placer
MACRO_PLACER_PROFILE=comp uv run backtest submissions/mobo_surrogate/placer.py --quick --comp

# Without multi-pole evacuation
MACRO_PLACER_ABLATION=no_multipole MACRO_PLACER_PROFILE=comp \
  uv run backtest submissions/mobo_surrogate/placer.py --quick --comp

# Without analytical / ePlace global
MACRO_PLACER_ABLATION=no_eplace MACRO_PLACER_PROFILE=comp \
  uv run backtest submissions/mobo_surrogate/placer.py --quick --comp

# Surrogate-only pool (no oracle refinement after SA modes)
MACRO_PLACER_ABLATION=no_oracle MACRO_PLACER_PROFILE=comp \
  uv run backtest submissions/mobo_surrogate/placer.py --quick --comp
```

## Results table (fill after runs)

| Ablation | geomean proxy | vs full | notes |
|----------|---------------|---------|-------|
| none | | | |
| no_multipole | | | |
| no_eplace | | | |
| no_oracle | | | |

Log experiments with `scripts/placement_experiment.py` for ledger-backed keep/drop.
