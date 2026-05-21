# Ablation flags (`MACRO_PLACER_ABLATION`)

Default **`analytical`**: Nesterov global placement before pool selection; heuristic soup off.

**`legacy`** (or **`none`**): re-enable multi-pole, cool-bin routing, axis-greed, heavy sequence-pair SA.

```bash
# Analytical-first (default)
MACRO_PLACER_PROFILE=comp uv run backtest submissions/mobo_surrogate/placer.py --quick --comp

# Legacy heuristic stack
MACRO_PLACER_ABLATION=legacy MACRO_PLACER_PROFILE=comp \
  uv run backtest submissions/mobo_surrogate/placer.py --quick --comp

# Disable individual legacy phases (with legacy base)
MACRO_PLACER_ABLATION=legacy,no_multipole MACRO_PLACER_PROFILE=comp \
  uv run backtest submissions/mobo_surrogate/placer.py --quick --comp

MACRO_PLACER_ABLATION=no_eplace MACRO_PLACER_PROFILE=comp \
  uv run backtest submissions/mobo_surrogate/placer.py --quick --comp
```

## Results table (fill after runs)

| Ablation | geomean proxy | vs analytical | notes |
|----------|---------------|---------------|-------|
| analytical | | | default |
| legacy | | | |
| legacy,no_multipole | | | |
| no_eplace | | | |

Log experiments with `scripts/placement_experiment.py` for ledger-backed keep/drop.
