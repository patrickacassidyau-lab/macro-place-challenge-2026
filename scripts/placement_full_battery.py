#!/usr/bin/env python3
"""Run the full IBM benchmark battery after a smoke geomean gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from macro_place.experiment_tracker import load_records  # noqa: E402


def _smoke_pair_geomean() -> float | None:
    want = {"ibm01", "ibm04"}
    best = None
    for rec in load_records():
        if rec.preset != "smoke":
            continue
        if set(rec.benchmarks) != want:
            continue
        if rec.total_overlaps > 0:
            continue
        if best is None or rec.geomean_proxy < best:
            best = rec.geomean_proxy
    return best


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Full IBM battery after smoke gate.")
    p.add_argument("--gate", type=float, default=1.15, help="Max smoke geomean to proceed.")
    p.add_argument("--label", default="full_ibm_battery")
    p.add_argument("--hypothesis", default="Full IBM proxy validation after smoke gate")
    p.add_argument("--placer", default="submissions/mobo_surrogate/placer.py")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    gm = _smoke_pair_geomean()
    if gm is None:
        print(json.dumps({"status": "blocked", "reason": "no smoke ibm01+ibm04 ledger row"}))
        return 2
    if gm > float(args.gate):
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "reason": "smoke geomean above gate",
                    "smoke_geomean": gm,
                    "gate": float(args.gate),
                }
            )
        )
        return 2

    cmd = [
        "uv",
        "run",
        "python",
        "scripts/placement_experiment.py",
        "run",
        "--label",
        args.label,
        "--hypothesis",
        args.hypothesis,
        "--placer",
        args.placer,
        "--preset",
        "none",
        "--all",
    ]
    if args.dry_run:
        print(json.dumps({"status": "ready", "smoke_geomean": gm, "command": cmd}))
        return 0

    import subprocess

    proc = subprocess.run(cmd, cwd=ROOT)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
