# Competition profile (`MACRO_PLACER_PROFILE`)

## Diagnosis baseline (fast profile, May 2026)

| Benchmark | proxy | WL | congestion | cong/wl | ρ (pool) |
|-----------|-------|-----|------------|---------|----------|
| ibm01 | 1.209 | 0.074 | 1.304 | **17.6** | 0.93 |
| ibm07 | 1.492 | 0.066 | 2.002 | **30.5** | 0.61 |
| ibm18 | 1.791 | 0.054 | 2.430 | **45.0** | 0.32 |

Priority fixes applied: WL-aligned surrogate weights, analytical global warm-start, larger SA budget, post-survivor ePlace + WL refinement, NG45 halo (12 µm).

## Local validation (≤15 min/bench — use this for tuning loops)

```bash
uv run backtest submissions/mobo_surrogate/placer.py --quick --validate
# Or single bench:
MACRO_PLACER_DIAGNOSE=1 uv run backtest submissions/mobo_surrogate/placer.py -b ibm01 --validate
```

Do **not** use bare `MACRO_PLACER_PROFILE=comp` for local diagnosis — default `BUDGET_SECS=3600` can exceed 30+ minutes per design.

## Judge submission (recommended)

```bash
export MACRO_PLACER_PROFILE=comp
export MACRO_PLACER_BUDGET_SECS=3600

uv run backtest submissions/mobo_surrogate/placer.py --all --comp --jobs 0
```

Or use the preset flag:

```bash
uv run backtest submissions/mobo_surrogate/placer.py --all --comp
```

## Overnight / max quality

```bash
export MACRO_PLACER_PROFILE=full
export MACRO_PLACER_BUDGET_SECS=7200
uv run backtest submissions/mobo_surrogate/placer.py --all --full
```

## Key env vars (comp defaults)

| Variable | comp default | Purpose |
|----------|--------------|---------|
| `MACRO_PLACER_BUDGET_SECS` | 3600 | Wall-clock cap per `place()` |
| `MACRO_PLACER_ITER_CAP` | 60000 | SA iteration ceiling |
| `MACRO_PLACER_ROUTE_BAL` | 0.006 | Route-imbalance surrogate (was 0.015) |
| `MACRO_PLACER_ROUTE_PRESSURE_W` | 0.0018 | RUDY pressure (was 0.0042) |
| `MACRO_PLACER_ANALYTICAL_ITERS` | 320 | Momentum analytical global |
| `MACRO_PLACER_EPLACE_ITERS` | 200 | ePlace warm-start |
| `MACRO_PLACER_HALO` | 0.5 (IBM) / 12.0 (NG45 auto) | Tier-2 clearance |
| `MACRO_PLACER_ORIENT_STEPS` | 48 | Klein-4 orientation search |
| `MACRO_PLACER_POLISH` | 1 | stdcell `optimize_stdcells` |
| `MACRO_PLACER_LEDGER_SURR` | 1 | Fit surrogate from oracle pool |
| `MACRO_PLACER_WL_FINAL_STEPS` | 80 | Timing-critical WL axis pass |

Explicit env exports always override profile defaults.
