#!/usr/bin/env python3
"""Rigorous cohort verdict for placer claims and logged experiments."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from macro_place.experiment_tracker import (  # noqa: E402
    ExperimentRecord,
    RUNS_JSONL,
    best_in_cohort,
    load_records,
)

IMPROVE_FRAC = 0.05

# Env keys whose prior logged runs in smoke|ibm01,ibm04 never beat cohort best by 5%.
INEFFECTIVE_ENV_SIGNATURES: Tuple[Tuple[str, str], ...] = (
    (("MACRO_PLACER_SOFT_FOLLOW", "1"),),
    (("MACRO_PLACER_ADAPT_ROUTE_RAT", "1.5"),),
    (("MACRO_PLACER_LTR_RANK", "0"),),
    (("MACRO_PLACER_CONGEST_BOOST", "1"),),
    (("MACRO_PLACER_FD_WARM", "0"),),
    (("MACRO_PLACER_INCREMACRO_ENABLE", "0"),),
    (("MACRO_PLACER_DPP_TOPK", "1"), ("MACRO_PLACER_TOPK", "3")),
    (("MACRO_PLACER_DPP_TOPK", "1"), ("MACRO_PLACER_TOPK", "4")),
)


def _cohort_key(preset: str, benchmarks: Sequence[str]) -> str:
    return f"{preset}|{','.join(sorted(benchmarks))}"


def _match_env(env: Dict[str, str], signature: Tuple[Tuple[str, str], ...]) -> bool:
    return all(env.get(k) == v for k, v in signature)


def ineffective_env_warning(env: Dict[str, str]) -> Optional[str]:
  for sig in INEFFECTIVE_ENV_SIGNATURES:
      if _match_env(env, sig):
          keys = ", ".join(f"{k}={v}" for k, v in sig)
          return f"prior ledger runs with {keys} never reached 5% cohort proxy gain"
  return None


def geomean_gain_pct(baseline: float, candidate: float) -> float:
    if baseline <= 0 or not math.isfinite(baseline) or not math.isfinite(candidate):
        return float("nan")
    return (baseline - candidate) / baseline * 100.0


def improved_vs_baseline(baseline: float, candidate: float, *, frac: float = IMPROVE_FRAC) -> bool:
    if candidate <= 0 or baseline <= 0:
        return False
    return candidate <= baseline * (1.0 - frac)


def find_record(records: Sequence[ExperimentRecord], *, run_id: Optional[str], label: Optional[str]) -> ExperimentRecord:
    if run_id:
        for r in records:
            if r.run_id == run_id:
                return r
        raise SystemExit(f"Unknown run_id {run_id!r}")
    if label:
        matches = [r for r in records if r.label == label]
        if not matches:
            raise SystemExit(f"Unknown label {label!r}")
        return matches[-1]
    raise SystemExit("Provide --run-id or --label")


def verdict_for_record(
    record: ExperimentRecord,
    *,
    prior: Sequence[ExperimentRecord],
    frac: float = IMPROVE_FRAC,
) -> Tuple[str, Dict[str, Any]]:
    if record.total_overlaps > 0:
        return "not improved", {"reason": "overlaps > 0", "geomean_proxy": record.geomean_proxy}

    cohort = record.cohort_key()
    best = best_in_cohort(prior, cohort)
    if best is None:
        return "not improved", {
            "reason": "no prior cohort baseline; 5% gain not established",
            "geomean_proxy": record.geomean_proxy,
            "cohort": cohort,
        }

    gain = geomean_gain_pct(best.geomean_proxy, record.geomean_proxy)
    ok = improved_vs_baseline(best.geomean_proxy, record.geomean_proxy, frac=frac)
    payload: Dict[str, Any] = {
        "cohort": cohort,
        "baseline_label": best.label,
        "baseline_geomean": best.geomean_proxy,
        "candidate_geomean": record.geomean_proxy,
        "geomean_gain_pct": gain,
        "required_gain_pct": frac * 100.0,
    }
    return ("improved" if ok else "not improved"), payload


def cmd_check(args: argparse.Namespace) -> int:
    records = load_records()
    record = find_record(records, run_id=args.run_id, label=args.label)
    prior = [r for r in records if r.run_id != record.run_id]
    verdict, detail = verdict_for_record(record, prior=prior, frac=args.improve_frac)
    warn = ineffective_env_warning(record.env)
    out = {"verdict": verdict, "label": record.label, "run_id": record.run_id, **detail}
    if warn:
        out["ineffective_env_warning"] = warn
    print(json.dumps(out, indent=2, sort_keys=True))
    print(verdict)
    return 0 if verdict == "improved" else 1


def cmd_propose(args: argparse.Namespace) -> int:
    env = {}
    for item in args.env:
        k, v = item.split("=", 1)
        env[k] = v
    warn = ineffective_env_warning(env)
    if warn:
        print("not improved")
        print(json.dumps({"reason": warn, "proposed_env": env}, indent=2))
        return 1
    print("no prior ineffective signature matched")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="5% cohort proxy claim verifier for placement experiments.")
    sub = p.add_subparsers(dest="cmd", required=True)

    chk = sub.add_parser("check", help="Verdict for a logged run vs cohort best before it.")
    chk.add_argument("--run-id", default=None)
    chk.add_argument("--label", default=None)
    chk.add_argument("--improve-frac", type=float, default=IMPROVE_FRAC)
    chk.set_defaults(func=cmd_check)

    prop = sub.add_parser("propose", help="Warn if proposed env matches historically ineffective moves.")
    prop.add_argument("--env", action="append", default=[], help="KEY=VALUE")
    prop.set_defaults(func=cmd_propose)

    args = p.parse_args(argv)
    if not RUNS_JSONL.exists() and args.cmd == "check":
        print(json.dumps({"verdict": "not improved", "reason": f"missing ledger {RUNS_JSONL}"}, indent=2))
        print("not improved")
        return 1
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
