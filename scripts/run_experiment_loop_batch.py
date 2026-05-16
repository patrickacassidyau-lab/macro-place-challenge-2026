#!/usr/bin/env python3
"""Run N labeled placement experiments in one cohort."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent

Experiment = Tuple[str, str, Dict[str, str]]

DEFAULT_BATCH: List[Experiment] = [
    ("innov01_dpp_topk3", "DPP survivor batch (k=3), LTR off", {
        "MACRO_PLACER_DPP_TOPK": "1",
        "MACRO_PLACER_LTR_RANK": "0",
        "MACRO_PLACER_TOPK": "3",
    }),
    ("innov02_ltr_dpp", "LTR rerank + DPP survivor batch", {
        "MACRO_PLACER_DPP_TOPK": "1",
        "MACRO_PLACER_LTR_RANK": "1",
        "MACRO_PLACER_TOPK": "3",
    }),
    ("innov03_tuned_dpp", "tuned cool-swap + LTR + DPP", {
        "MACRO_PLACER_DPP_TOPK": "1",
        "MACRO_PLACER_LTR_RANK": "1",
        "MACRO_PLACER_TOPK": "3",
        "MACRO_PLACER_COOL_BIN_STEPS": "24",
        "MACRO_PLACER_COOL_BIN_K": "24",
        "MACRO_PLACER_COOL_BIN_TRIALS": "7",
        "MACRO_PLACER_COOL_BIN_HOT_BIAS": "0.82",
        "MACRO_PLACER_COOL_BIN_PATIENCE": "3",
        "MACRO_PLACER_SWAP_STEPS": "20",
        "MACRO_PLACER_SWAP_PATIENCE": "4",
    }),
    ("innov04_dpp_off", "control: no DPP, top-2 proxy survivors", {
        "MACRO_PLACER_DPP_TOPK": "0",
        "MACRO_PLACER_TOPK": "2",
    }),
    ("innov05_ltr_heavy", "stronger LTR + DPP", {
        "MACRO_PLACER_DPP_TOPK": "1",
        "MACRO_PLACER_LTR_BLEND": "0.34",
        "MACRO_PLACER_LTR_STEPS": "260",
        "MACRO_PLACER_TOPK": "3",
    }),
    ("innov06_dpp_wide", "wider DPP kernel (more diversity)", {
        "MACRO_PLACER_DPP_TOPK": "1",
        "MACRO_PLACER_DPP_SIGMA": "0.34",
        "MACRO_PLACER_TOPK": "3",
    }),
    ("innov07_dpp_tight", "tighter DPP kernel", {
        "MACRO_PLACER_DPP_TOPK": "1",
        "MACRO_PLACER_DPP_SIGMA": "0.14",
        "MACRO_PLACER_TOPK": "3",
    }),
    ("innov08_topk4", "DPP with k=4 survivors", {
        "MACRO_PLACER_DPP_TOPK": "1",
        "MACRO_PLACER_TOPK": "4",
    }),
    ("innov09_ltr_only", "LTR only, no DPP", {
        "MACRO_PLACER_DPP_TOPK": "0",
        "MACRO_PLACER_LTR_RANK": "1",
        "MACRO_PLACER_TOPK": "3",
    }),
    ("innov10_tuned_no_dpp", "tuned cool-swap without DPP", {
        "MACRO_PLACER_DPP_TOPK": "0",
        "MACRO_PLACER_COOL_BIN_STEPS": "24",
        "MACRO_PLACER_COOL_BIN_K": "24",
        "MACRO_PLACER_COOL_BIN_TRIALS": "7",
        "MACRO_PLACER_COOL_BIN_HOT_BIAS": "0.82",
        "MACRO_PLACER_COOL_BIN_PATIENCE": "3",
        "MACRO_PLACER_SWAP_STEPS": "20",
        "MACRO_PLACER_SWAP_PATIENCE": "4",
    }),
    ("innov11_modes5_dpp", "five surrogate modes + DPP", {
        "MACRO_PLACER_MODES": "5",
        "MACRO_PLACER_DPP_TOPK": "1",
        "MACRO_PLACER_TOPK": "3",
    }),
    ("innov12_axis_dpp", "axis-greed + DPP", {
        "MACRO_PLACER_AXIS_GREED": "32",
        "MACRO_PLACER_DPP_TOPK": "1",
        "MACRO_PLACER_TOPK": "3",
    }),
    ("innov13_combo_best", "tuned cool-swap + heavy LTR + wide DPP", {
        "MACRO_PLACER_DPP_TOPK": "1",
        "MACRO_PLACER_DPP_SIGMA": "0.30",
        "MACRO_PLACER_LTR_BLEND": "0.34",
        "MACRO_PLACER_LTR_STEPS": "260",
        "MACRO_PLACER_TOPK": "3",
        "MACRO_PLACER_COOL_BIN_STEPS": "24",
        "MACRO_PLACER_COOL_BIN_K": "24",
        "MACRO_PLACER_COOL_BIN_TRIALS": "7",
        "MACRO_PLACER_COOL_BIN_HOT_BIAS": "0.82",
        "MACRO_PLACER_COOL_BIN_PATIENCE": "3",
        "MACRO_PLACER_SWAP_STEPS": "20",
        "MACRO_PLACER_SWAP_PATIENCE": "4",
    }),
    ("innov14_smoke_topk2", "minimal DPP smoke (k=2)", {
        "MACRO_PLACER_DPP_TOPK": "1",
        "MACRO_PLACER_TOPK": "2",
    }),
    ("innov15_baseline_code", "post-DPP code defaults (no extra env)", {}),
]


def run_one(label: str, hypothesis: str, env: Dict[str, str], benchmarks: List[str]) -> dict:
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/placement_experiment.py",
        "run",
        "--label",
        label,
        "--hypothesis",
        hypothesis,
        "--preset",
        "smoke",
    ]
    for b in benchmarks:
        cmd.extend(["-b", b])
    for k, v in env.items():
        cmd.extend(["--env", f"{k}={v}"])
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed:\n{proc.stdout}\n{proc.stderr}")
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"{label}: no JSON summary in stdout\n{proc.stdout}\n{proc.stderr}")


def _existing_labels() -> set[str]:
    ledger = ROOT / "results/experiments/runs.jsonl"
    if not ledger.exists():
        return set()
    labels: set[str] = set()
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        labels.add(str(json.loads(line).get("label", "")))
    return labels


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=15)
    p.add_argument("-b", "--benchmark", action="append", default=["ibm01", "ibm04"])
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()
    batch = DEFAULT_BATCH[: max(1, args.count)]
    if args.skip_existing:
        done = _existing_labels()
        batch = [(l, h, e) for l, h, e in batch if l not in done]
    summaries = []
    for label, hyp, env in batch:
        print(f"RUN {label}", flush=True)
        summaries.append(run_one(label, hyp, env, args.benchmark))
    print(json.dumps({"completed": len(summaries), "runs": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
