"""Append-only experiment ledger for placer backtests."""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4

EXPERIMENTS_DIR = Path("results/experiments")
RUNS_JSONL = EXPERIMENTS_DIR / "runs.jsonl"
ARTIFACTS_DIR = EXPERIMENTS_DIR / "artifacts"


@dataclass
class ExperimentRecord:
    run_id: str
    ts: str
    label: str
    hypothesis: str
    placer: str
    preset: str
    benchmarks: List[str]
    env: Dict[str, str]
    git_head: Optional[str]
    git_dirty: Optional[bool]
    geomean_proxy: float
    total_overlaps: int
    wall_clock_s: float
    sum_placement_compute_s: float
    per_bench: Dict[str, float]
    per_bench_vs_replace_pct: Dict[str, Optional[float]]
    artifact_json: str
    verdict: str = "neutral"
    notes: str = ""

    def cohort_key(self) -> str:
        b = ",".join(sorted(self.benchmarks))
        return f"{self.preset}|{b}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def geomean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(math.exp(sum(math.log(max(v, 1e-12)) for v in values) / len(values)))


def git_snapshot() -> tuple[Optional[str], Optional[bool]]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return head or None, bool(dirty)
    except (OSError, subprocess.CalledProcessError):
        return None, None


def append_record(record: ExperimentRecord) -> Path:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    with RUNS_JSONL.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
    return RUNS_JSONL


def load_records() -> List[ExperimentRecord]:
    if not RUNS_JSONL.exists():
        return []
    out: List[ExperimentRecord] = []
    for line in RUNS_JSONL.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        out.append(ExperimentRecord(**raw))
    return out


def summarize_payload(payload: Dict[str, Any]) -> tuple[float, int, Dict[str, float], Dict[str, Optional[float]]]:
    runs = payload.get("runs") or []
    per_bench: Dict[str, float] = {}
    per_vs: Dict[str, Optional[float]] = {}
    overlaps = 0
    for row in runs:
        name = str(row["name"])
        overlaps += int(row.get("overlaps") or 0)
        cur = float(row["proxy_cost"])
        prev = per_bench.get(name)
        if prev is None or cur < prev:
            per_bench[name] = cur
            per_vs[name] = row.get("vs_replace_pct")
    gm = geomean(list(per_bench.values()))
    return gm, overlaps, per_bench, per_vs


def best_in_cohort(records: Sequence[ExperimentRecord], cohort_key: str) -> Optional[ExperimentRecord]:
    cohort = [r for r in records if r.cohort_key() == cohort_key and r.total_overlaps == 0]
    if not cohort:
        return None
    return min(cohort, key=lambda r: r.geomean_proxy)


def classify_vs_baseline(
    record: ExperimentRecord,
    *,
    prior: Sequence[ExperimentRecord],
    rtol: float = 1e-4,
) -> str:
    if record.total_overlaps > 0:
        return "drop"
    best = best_in_cohort(prior, record.cohort_key())
    if best is None:
        return "neutral"
    if record.geomean_proxy < best.geomean_proxy - rtol * max(abs(best.geomean_proxy), 1.0):
        return "keep"
    if record.geomean_proxy > best.geomean_proxy + rtol * max(abs(best.geomean_proxy), 1.0):
        return "drop"
    return "neutral"
