"""
Shared MACRO_PLACER_* environment presets for ``backtest`` / ``evaluate``.

``--fast`` / ``--smoke`` apply condensed surrogate + oracle **budgets** for quicker wall-clock.
Phases keep the same ordering (multi-mode SA → oracle gates → refine passes); step counts and
SA length shrink — proxy is not comparable to full-budget competition runs.
"""

from __future__ import annotations

import os
from typing import Dict

# Condensed surrogate/oracle budgets (same keys as legacy backtest ``_FAST_BACKTEST_ENV``).
FAST_MACRO_PLACER_ENV: Dict[str, str] = {
    "MACRO_PLACER_TIME_SCALE": "0.04",
    "MACRO_PLACER_ITER_FLOOR": "160",
    "MACRO_PLACER_ITER_CAP": "300",
    # More surrogate hypotheses → oracle picks lower proxy (still SA‑budget‑limited).
    "MACRO_PLACER_MODES": "3",
    "MACRO_PLACER_TOPK": "2",
    "MACRO_PLACER_CONGEST_BOOST": "0",
    "MACRO_PLACER_TORCH_REFINE": "0",
    "MACRO_PLACER_POLISH": "0",
    # Tier‑1 proxy is congestion‑heavy — prioritize oracle coordinate / congestion escapes vs trimming steps.
    "MACRO_PLACER_AXIS_GREED": "26",
    "MACRO_PLACER_AXIS_GREED_PASS2": "22",
    "MACRO_PLACER_AXIS_PASS2_ENABLE": "1",
    "MACRO_PLACER_AXIS_GREED_PASS3": "18",
    "MACRO_PLACER_AXIS_PASS3_SCALE": "0.34",
    "MACRO_PLACER_AXIS_PASS3_ENABLE": "1",
    "MACRO_PLACER_AXIS_DELTA": "0.00935",
    "MACRO_PLACER_HOT_ESC_STEPS": "16",
    "MACRO_PLACER_HOT_ESC_GUARD": "0.71",
    "MACRO_PLACER_MULTIPOLE_STEPS": "16",
    "MACRO_PLACER_MULTIPOLE_GUARD": "0.695",
    "MACRO_PLACER_POLE_LS_ROUNDS": "10",
    "MACRO_PLACER_PAIR_POLE_STEPS": "10",
    "MACRO_PLACER_POST_LEGAL": "4",
    "MACRO_PLACER_REPEL_ROUNDS": "3",
    "MACRO_PLACER_INCR_SCALE": "0.52",
    "MACRO_PLACER_PC_STEPS_CAP": "58",
}

SMOKE_MACRO_PLACER_ENV: Dict[str, str] = {
    **FAST_MACRO_PLACER_ENV,
    "MACRO_PLACER_TIME_SCALE": "0.03",
    "MACRO_PLACER_ITER_FLOOR": "90",
    "MACRO_PLACER_ITER_CAP": "110",
    "MACRO_PLACER_MODES": "1",
    "MACRO_PLACER_TOPK": "1",
    "MACRO_PLACER_ADAPT_MODES": "0",
    "MACRO_PLACER_FD_ITERS": "6",
    "MACRO_PLACER_AXIS_GREED": "0",
    "MACRO_PLACER_AXIS_GREED_PASS2": "0",
    "MACRO_PLACER_AXIS_GREED_PASS3": "0",
    "MACRO_PLACER_AXIS_PASS3_ENABLE": "0",
    "MACRO_PLACER_NET_HOT_BLEND": "0",
    "MACRO_PLACER_NET_HOT_HOTESC_BLEND": "0",
    "MACRO_PLACER_INCR_MACRO_BIAS": "0",
    "MACRO_PLACER_AXIS_NET_HOT_REFRESH_EVERY": "0",
    "MACRO_PLACER_HOT_ESC_STEPS": "0",
    "MACRO_PLACER_INCREMACRO_ENABLE": "0",
    "MACRO_PLACER_MULTIPOLE_STEPS": "0",
    "MACRO_PLACER_MULTIPOLE_ENABLE": "0",
    "MACRO_PLACER_POLE_LS_ROUNDS": "0",
    "MACRO_PLACER_POLE_LS_ENABLE": "0",
    "MACRO_PLACER_PAIR_POLE_STEPS": "0",
    "MACRO_PLACER_PAIR_POLE_ENABLE": "0",
    "MACRO_PLACER_INCR_SCALE": "0.32",
    "MACRO_PLACER_PC_STEPS_CAP": "22",
    "MACRO_PLACER_REPEL_ROUNDS": "0",
    "MACRO_PLACER_POST_LEGAL": "1",
}


def apply_preset_env(preset: Dict[str, str]) -> None:
    """Set ``MACRO_PLACER_*`` keys not already exported (explicit env overrides preset)."""
    for key, val in preset.items():
        if key not in os.environ:
            os.environ[key] = val
