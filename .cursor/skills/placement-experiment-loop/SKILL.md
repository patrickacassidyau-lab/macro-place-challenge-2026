---
name: placement-experiment-loop
description: >-
  Two-agent loop for macro placer iteration: run labeled backtests into
  results/experiments/runs.jsonl, classify keep/drop vs cohort best, and
  dispatch a progress-checker subagent for long-term what-works reporting.
---

# Placement experiment loop

Use this skill when improving `submissions/mobo_surrogate/placer.py` (or other placers) and you need durable evidence of what helps proxy quality.

## Roles

### 1) Implementation agent (you)

1. State a **hypothesis** in one sentence.
2. Apply the smallest code change that tests it.
3. Run a logged experiment (never raw `backtest` alone for comparisons):

```bash
uv run python scripts/placement_experiment.py run \
  --label cool_bin_v2 \
  --hypothesis "cool-bin burst lowers congestion on ibm01 without hurting ibm04" \
  --preset smoke -b ibm01 -b ibm04 \
  --env MACRO_PLACER_COOL_BIN_STEPS=24
```

4. Read the JSON summary (`verdict`, `geomean_proxy`, `total_overlaps`).
5. **Hard gate**
   - `verdict=drop` or overlaps > 0 → revert the code change.
   - `verdict=keep` → keep and note env defaults worth promoting.
   - `verdict=neutral` → keep only if it unlocks a follow-up; otherwise revert or tighten.
   - Run `uv run python scripts/placement_claim_verifier.py check --label <label>`; **not improved** → backtrack or do not land the change.
6. After **3+ logged runs** in the same session, dispatch the progress checker (below).

### 2) Progress checker agent (subagent)

Launch a **readonly** `explore` or `generalPurpose` subagent with this prompt:

- Read `results/experiments/runs.jsonl` and `results/experiments/latest_report.md`.
- Group by cohort (`preset`, benchmark set, env keys).
- List **keep** vs **drop** experiments with geomean deltas and runtime.
- Call out env knobs that repeatedly help vs hurt.
- Recommend the next **two** experiments (label + hypothesis + exact command).
- Do **not** edit code; report only.

Parent agent must apply checker recommendations or explain why not.

## Presets

| Preset | Use |
|--------|-----|
| `smoke` | Fast iteration; proxy not comparable across competition baselines |
| `fast` | Medium budget |
| `quick` | Five-benchmark subset |
| `none` | Full local env (slow) |

Compare runs only within the **same cohort** (same preset, benchmarks, env key set).

## Ledger

- Append-only log: `results/experiments/runs.jsonl`
- Artifacts: `results/experiments/artifacts/<run_id>.json`
- Markdown rollup: `results/experiments/latest_report.md` (refreshed by hook after logged runs)

Manual report:

```bash
uv run python scripts/placement_progress_report.py --hours 168 \
  --out results/experiments/latest_report.md
```

## Innovation bar

Treat **<0.5%** geomean gain on smoke as noise unless it repeats on `fast`/`quick`. Target **≥1%** on the active cohort before claiming progress.
