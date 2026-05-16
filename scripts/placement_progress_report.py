#!/usr/bin/env python3
"""Summarize placement experiment ledger for long-running agent loops."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from macro_place.experiment_tracker import (  # noqa: E402
    RUNS_JSONL,
    ExperimentRecord,
    best_in_cohort,
    load_records,
)


def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _fmt_delta(cur: float, ref: float) -> str:
    if ref <= 0:
        return "n/a"
    pct = (cur - ref) / ref * 100.0
    return f"{pct:+.2f}%"


def build_report(records: Sequence[ExperimentRecord], *, since: Optional[datetime]) -> str:
    if not records:
        return f"No experiments logged yet. Run `uv run python scripts/placement_experiment.py run ...` first.\nLedger: {RUNS_JSONL}"

    filtered = list(records)
    if since is not None:
        filtered = [r for r in filtered if _parse_ts(r.ts) >= since]
    if not filtered:
        return "No experiments in the selected time window."

    by_cohort: Dict[str, List[ExperimentRecord]] = defaultdict(list)
    for rec in filtered:
        by_cohort[rec.cohort_key()].append(rec)

    lines: List[str] = [
        "# Placement experiment progress",
        "",
        f"- Ledger: `{RUNS_JSONL}`",
        f"- Runs in window: {len(filtered)}",
        "",
    ]

    for cohort, group in sorted(by_cohort.items()):
        valid = [r for r in group if r.total_overlaps == 0]
        if not valid:
            lines.append(f"## Cohort `{cohort}`")
            lines.append("- No overlap-free runs.")
            lines.append("")
            continue
        best = min(valid, key=lambda r: r.geomean_proxy)
        lines.append(f"## Cohort `{cohort}`")
        lines.append(
            f"- Best: `{best.label}` ({best.run_id}) geomean **{best.geomean_proxy:.4f}** @ {best.ts}"
        )
        lines.append(f"- Hypothesis: {best.hypothesis or '(none)'}")
        if best.env:
            env_s = ", ".join(f"{k}={v}" for k, v in sorted(best.env.items()))
            lines.append(f"- Env: {env_s}")
        lines.append("")
        lines.append("| verdict | label | geomean | Δ vs best | wall s | overlaps |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for rec in sorted(group, key=lambda r: r.ts):
            delta = _fmt_delta(rec.geomean_proxy, best.geomean_proxy)
            lines.append(
                f"| {rec.verdict} | {rec.label} | {rec.geomean_proxy:.4f} | {delta} | "
                f"{rec.wall_clock_s:.1f} | {rec.total_overlaps} |"
            )
        lines.append("")

        keep = [r for r in group if r.verdict == "keep"]
        drop = [r for r in group if r.verdict == "drop"]
        if keep:
            lines.append("### Likely keep")
            for rec in keep[-5:]:
                lines.append(f"- `{rec.label}` ({rec.run_id}): {rec.hypothesis or rec.label}")
        if drop:
            lines.append("### Likely drop")
            for rec in drop[-5:]:
                lines.append(f"- `{rec.label}` ({rec.run_id}): {rec.hypothesis or rec.label}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Report on placement experiment ledger.")
    p.add_argument("--hours", type=float, default=None, help="Only include runs from the last N hours.")
    p.add_argument("--out", type=str, default=None, help="Write markdown report to this path.")
    args = p.parse_args(argv)

    since = None
    if args.hours is not None:
        since = datetime.now(timezone.utc) - timedelta(hours=args.hours)

    report = build_report(load_records(), since=since)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
