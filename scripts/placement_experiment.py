#!/usr/bin/env python3
"""Record labeled placer backtests into results/experiments/runs.jsonl."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from macro_place.experiment_tracker import (  # noqa: E402
    ARTIFACTS_DIR,
    ExperimentRecord,
    _utc_now,
    append_record,
    classify_vs_baseline,
    git_snapshot,
    load_records,
    summarize_payload,
)


def _parse_env(pairs: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            raise SystemExit(f"Invalid --env {item!r}; expected KEY=VALUE")
        k, v = item.split("=", 1)
        out[k] = v
    return out


def _benchmark_list(args: argparse.Namespace) -> List[str]:
    if args.quick:
        from macro_place.backtest import QUICK_BENCHMARKS

        return list(QUICK_BENCHMARKS)
    if args.all:
        from macro_place.evaluate import IBM_BENCHMARKS

        return list(IBM_BENCHMARKS)
    if args.benchmark:
        return list(args.benchmark)
    return ["ibm01"]


def _artifact_relpath(artifact: Path) -> str:
    try:
        return str(artifact.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(artifact.resolve())


def run_logged_backtest(
    placer: Path,
    *,
    preset: str,
    benchmarks: List[str],
    env: Dict[str, str],
    jobs: int,
    trials: int,
    artifact: Path,
) -> None:
    from macro_place.backtest import _load_placer, run_backtest as backtest_run
    from macro_place.placer_presets import (
        FAST_MACRO_PLACER_ENV,
        SMOKE_MACRO_PLACER_ENV,
        apply_preset_env,
    )

    merged = os.environ.copy()
    merged.update(env)
    for key, val in merged.items():
        os.environ[key] = val

    if preset == "smoke":
        apply_preset_env(SMOKE_MACRO_PLACER_ENV)
    elif preset == "fast":
        apply_preset_env(FAST_MACRO_PLACER_ENV)
    elif preset not in ("quick", "none"):
        raise SystemExit(f"Unknown preset {preset!r}")

    for key, val in env.items():
        os.environ[key] = val

    placer_path = placer.resolve()
    rows, wall_clock = backtest_run(
        placer_path,
        benchmarks,
        trials=trials,
        json_out=artifact.resolve(),
        placer_factory=lambda: _load_placer(placer_path),
        jobs=jobs,
    )
    if not rows:
        raise SystemExit("Backtest returned no rows.")
    if wall_clock is None:
        wall_clock = 0.0
    payload = {
        "placer": str(placer_path),
        "parallel_jobs": max(1, jobs),
        "wall_clock_s": wall_clock,
        "sum_placement_compute_s": sum(r.runtime_s for r in rows),
        "runs": [r.to_dict() for r in rows],
    }
    artifact.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def cmd_run(args: argparse.Namespace) -> int:
    placer = (ROOT / args.placer).resolve() if not Path(args.placer).is_absolute() else Path(args.placer)
    env = _parse_env(args.env)
    benchmarks = _benchmark_list(args)
    preset = args.preset

    run_id = args.run_id or os.environ.get("PLACEMENT_EXPERIMENT_ID")
    if not run_id:
        from uuid import uuid4

        run_id = uuid4().hex[:12]

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = ARTIFACTS_DIR / f"{run_id}.json"
    run_logged_backtest(
        placer,
        preset=preset,
        benchmarks=benchmarks,
        env=env,
        jobs=args.jobs,
        trials=args.trials,
        artifact=artifact,
    )

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    gm, overlaps, per_bench, per_vs = summarize_payload(payload)
    head, dirty = git_snapshot()
    prior = load_records()
    record = ExperimentRecord(
        run_id=run_id,
        ts=_utc_now(),
        label=args.label,
        hypothesis=args.hypothesis,
        placer=str(placer.relative_to(ROOT)) if placer.is_relative_to(ROOT) else str(placer),
        preset=preset,
        benchmarks=benchmarks,
        env=env,
        git_head=head,
        git_dirty=dirty,
        geomean_proxy=gm,
        total_overlaps=overlaps,
        wall_clock_s=float(payload.get("wall_clock_s") or 0.0),
        sum_placement_compute_s=float(payload.get("sum_placement_compute_s") or 0.0),
        per_bench=per_bench,
        per_bench_vs_replace_pct=per_vs,
        artifact_json=_artifact_relpath(artifact),
    )
    record.verdict = classify_vs_baseline(record, prior=prior)
    append_record(record)

    print(
        json.dumps(
            {
                "run_id": record.run_id,
                "label": record.label,
                "preset": record.preset,
                "benchmarks": record.benchmarks,
                "geomean_proxy": record.geomean_proxy,
                "total_overlaps": record.total_overlaps,
                "verdict": record.verdict,
                "artifact_json": record.artifact_json,
            },
            separators=(",", ":"),
        )
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run and log placer backtest experiments.")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run backtest and append ledger row.")
    run.add_argument("--label", required=True, help="Short experiment name.")
    run.add_argument("--hypothesis", default="", help="What this run is testing.")
    run.add_argument("--placer", default="submissions/mobo_surrogate/placer.py")
    run.add_argument(
        "--preset",
        choices=("smoke", "fast", "quick", "none"),
        default="smoke",
        help="Backtest preset (smoke=fast iteration).",
    )
    run.add_argument("-b", "--benchmark", action="append", default=None)
    run.add_argument("--quick", action="store_true")
    run.add_argument("--all", action="store_true")
    run.add_argument("--env", action="append", default=[], help="MACRO_PLACER_* override KEY=VALUE")
    run.add_argument("--trials", type=int, default=1)
    run.add_argument("--jobs", type=int, default=1)
    run.add_argument("--run-id", default=None)
    run.set_defaults(func=cmd_run)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
