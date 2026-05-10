"""
Fast local evaluation loop for macro placers with diagnostics.

Use this to iterate on models without waiting for the full OpenROAD tier-2 flow.
Reports proxy decomposition, gap vs published SA/RePlAce baselines, and whether
density or congestion dominates the weighted proxy relative to wirelength.
Also prints **routing-capacity imbalance** and a coarse **pin-splat RUDY** peak plus
PlacementCost route-congestion means — use these columns to correlate cheap search
signals with Tier-1 ``congestion_cost``.

Top submissions on the public leaderboard often use tens of minutes to hours
across benchmarks (within the official 1h/benchmark cap); this harness runs the
*same number* of placements (benchmarks × trials) as before, but can execute
them in parallel with ``--jobs`` to cut wall-clock time.

Usage:
    uv run backtest submissions/mobo_surrogate/placer.py
    uv run backtest submissions/mobo_surrogate/placer.py --fast -b ibm01   # ~3–8 min (tier‑1, reduced SA/oracle)
    uv run backtest submissions/mobo_surrogate/placer.py --smoke -b ibm01 # ~30–90s wiring check (proxy not comparable)
    uv run backtest submissions/mobo_surrogate/placer.py --quick --json results.json
    uv run backtest path/to/placer.py -a --trials 3 --jobs 0    # auto parallelism
    uv run backtest path/to/placer.py -a --jobs 8               # explicit pool size

``--fast`` only sets ``MACRO_PLACER_*`` env vars that are **not** already exported (your env wins).
"""

from __future__ import annotations

# Before transitive matplotlib imports — avoids slow fresh font-cache builds (tmp dirs / sandbox).
from pathlib import Path

_MPLCACHE = Path(__file__).resolve().parent.parent / ".mplconfig"
_MPLCACHE.mkdir(parents=True, exist_ok=True)

import argparse
import importlib.util
import math
import json
import multiprocessing as mp
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(_MPLCACHE))

import numpy as np
import torch

from macro_place.evaluate import (
    IBM_BENCHMARKS as BENCHMARKS,
    NG45_BENCHMARKS,
    REPLACE_BASELINES,
    SA_BASELINES,
)
from macro_place.loader import load_benchmark, load_benchmark_from_dir
from macro_place.objective import _set_placement, compute_proxy_cost
from macro_place.placer_presets import (
    FAST_MACRO_PLACER_ENV as _FAST_BACKTEST_ENV,
    SMOKE_MACRO_PLACER_ENV as _SMOKE_BACKTEST_ENV,
    apply_preset_env,
)
from macro_place.routing_surrogate import (
    compute_total_routing_imbalance,
    multipole_congestion_sites_um,
    pin_splat_rudy_peak,
    plc_congestion_headroom,
)


def _load_placer(path: Path):
    """Same convention as macro_place.evaluate._load_placer."""
    path = path.resolve()
    if spec := importlib.util.spec_from_file_location(path.stem, str(path)):
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        raise RuntimeError(f"Failed to load placer from {path}")

    for attr in vars(mod).values():
        if (
            isinstance(attr, type)
            and attr.__module__ == path.stem
            and callable(getattr(attr, "place", None))
        ):
            return attr()

    raise RuntimeError(
        f"No placer class found in {path}.\n"
        "Expected a class with place(self, benchmark) -> Tensor."
    )


QUICK_BENCHMARKS = ["ibm01", "ibm04", "ibm09", "ibm12", "ibm17"]

TESTCASE_ICCAD04 = Path("external/MacroPlacement/Testcases/ICCAD04")


@dataclass
class BenchDiagnostics:
    name: str
    proxy_cost: float
    wirelength: float
    density: float
    congestion: float
    overlaps: int
    runtime_s: float
    valid: bool
    sa_baseline: Optional[float]
    replace_baseline: Optional[float]
    initial_proxy: float
    initial_wirelength: float
    initial_density: float
    initial_congestion: float
    wl_share_of_weighted_terms: float
    density_pressure: float
    congestion_pressure: float
    vs_replace_pct: Optional[float]
    # Innovation / surrogate diagnostics (tier-2 research signal — correlate with PlacementCost congestion)
    routing_capacity_imbalance_proxy: float = 0.0
    pin_splat_rudy_peak: float = 0.0
    plc_mean_route_congestion_hv_max: float = 0.0
    #: After placement: spatially-separated high max(H,V) congestion poles counted (tier‑1 grid)
    multipole_hotspot_peaks_k: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def _weighted_term_shares(wl: float, den: float, cong: float) -> tuple[float, float, float]:
    """Normalized contributions matching proxy = wl + 0.5*den + 0.5*cong."""
    denom = wl + 0.5 * den + 0.5 * cong
    if denom <= 0:
        return 0.33, 0.33, 0.34
    return wl / denom, (0.5 * den) / denom, (0.5 * cong) / denom


def diagnose_single(
    name: str,
    testcase_root: Path,
    placer_path: Path,
    placer_factory=None,
    ng45_dir: Optional[str] = None,
    seed_shuffle: Optional[int] = None,
) -> BenchDiagnostics:
    if ng45_dir:
        nf = Path(ng45_dir) / "netlist.pb.txt"
        pf = Path(ng45_dir) / "initial.plc"
        benchmark, plc = load_benchmark(str(nf), str(pf) if pf.exists() else None, name=name)
    else:
        benchmark, plc = load_benchmark_from_dir(str(testcase_root / name))

    if seed_shuffle is not None:
        torch.manual_seed(seed_shuffle)
        random.seed(seed_shuffle)
        np.random.seed(seed_shuffle)

    init_placement = benchmark.macro_positions.clone()
    init_costs = compute_proxy_cost(init_placement, benchmark, plc)

    inst = placer_factory() if placer_factory is not None else _load_placer(placer_path)
    t0 = time.time()
    placement = inst.place(benchmark)
    runtime = time.time() - t0

    costs = compute_proxy_cost(placement, benchmark, plc)
    from macro_place.utils import validate_placement

    valid, _ = validate_placement(placement, benchmark)

    wl_s, den_s, cong_s = _weighted_term_shares(
        costs["wirelength_cost"],
        costs["density_cost"],
        costs["congestion_cost"],
    )

    rp = REPLACE_BASELINES.get(name)
    vs_rp = (
        ((rp - costs["proxy_cost"]) / rp * 100.0)
        if rp is not None and rp > 0
        else None
    )

    # "Pressure": how intense each axis is vs wirelength alone (normalized ratios)
    wl = max(costs["wirelength_cost"], 1e-9)
    den_p = costs["density_cost"] / wl
    cong_p = costs["congestion_cost"] / wl

    pos_hard = placement[: benchmark.num_hard_macros].numpy().astype(np.float64)
    imb = compute_total_routing_imbalance(benchmark, pos_hard)
    rsplat = pin_splat_rudy_peak(benchmark, pos_hard, grid_g=40)
    _, _, mcon = plc_congestion_headroom(plc, benchmark)

    multipole_cnt = 0
    try:
        _set_placement(plc, placement, benchmark)
        mp_s = multipole_congestion_sites_um(plc, benchmark, k=14, min_sep_cells=2)
        multipole_cnt = int(mp_s.shape[0])
    except Exception:
        multipole_cnt = 0

    return BenchDiagnostics(
        name=name,
        proxy_cost=float(costs["proxy_cost"]),
        wirelength=float(costs["wirelength_cost"]),
        density=float(costs["density_cost"]),
        congestion=float(costs["congestion_cost"]),
        overlaps=int(costs["overlap_count"]),
        runtime_s=runtime,
        valid=valid,
        sa_baseline=SA_BASELINES.get(name),
        replace_baseline=rp,
        initial_proxy=float(init_costs["proxy_cost"]),
        initial_wirelength=float(init_costs["wirelength_cost"]),
        initial_density=float(init_costs["density_cost"]),
        initial_congestion=float(init_costs["congestion_cost"]),
        wl_share_of_weighted_terms=float(wl_s),
        density_pressure=float(den_p),
        congestion_pressure=float(cong_p),
        vs_replace_pct=float(vs_rp) if vs_rp is not None else None,
        routing_capacity_imbalance_proxy=float(imb),
        pin_splat_rudy_peak=float(rsplat),
        plc_mean_route_congestion_hv_max=float(mcon),
        multipole_hotspot_peaks_k=multipole_cnt,
    )


def _worker_diagnose(task: Tuple[str, str, str, Optional[str], Optional[int]]) -> BenchDiagnostics:
    """
    Process-pool worker: (benchmark_name, testcase_root_str, placer_path_str, ng45_dir, seed_shuffle).
    """
    name, root_s, placer_s, ng45_dir, seed_shuffle = task
    testcase_root = Path(root_s).resolve()
    placer_path = Path(placer_s).resolve()
    return diagnose_single(
        name,
        testcase_root,
        placer_path,
        placer_factory=None,
        ng45_dir=ng45_dir,
        seed_shuffle=seed_shuffle,
    )


def format_report(rows: List[BenchDiagnostics], wall_clock_s: Optional[float] = None) -> str:
    lines = []
    lines.append("─" * 122)
    lines.append(
        f"{'Bench':>8}  {'Proxy':>8}  {'vsInit':>8}  {'vsRep%':>8}  "
        f"{'wl_shr':>6}  {'den/wl':>7}  {'cong/wl':>8}  "
        f"{'rtImb':>7} {'splat':>6} {'plcC':>5} {'mpK':>3} "
        f"{'Ovlp':>5}  {'t(s)':>6}  note"
    )
    lines.append("─" * 122)

    for r in rows:
        vs_init = (r.initial_proxy - r.proxy_cost) / max(r.initial_proxy, 1e-9) * 100
        note = ""
        rt_imb = r.routing_capacity_imbalance_proxy
        spl = r.pin_splat_rudy_peak
        pcon = r.plc_mean_route_congestion_hv_max
        if r.overlaps > 0:
            note = "INVALID overlaps"
        elif not r.valid:
            note = "INVALID bounds/fix"
        elif r.congestion_pressure > 2.5 and r.density_pressure <= r.congestion_pressure:
            note = "proxy driven by congestion"
        elif r.density_pressure > 2.5:
            note = "proxy driven by density"
        elif r.wl_share_of_weighted_terms > 0.65:
            note = "WL-heavy proxy mix"
        else:
            note = "balanced mix"
        vrep = f"{r.vs_replace_pct:+.1f}" if r.vs_replace_pct is not None else "   —"

        lines.append(
            f"{r.name:>8}  {r.proxy_cost:>8.4f}  {vs_init:>+7.1f}%  {vrep:>8}  "
            f"{r.wl_share_of_weighted_terms:>6.2f}  {r.density_pressure:>7.3f}  "
            f"{r.congestion_pressure:>8.3f}  "
            f"{rt_imb:>7.4f} {spl:>6.3f} {pcon:>5.3f} {r.multipole_hotspot_peaks_k:>3} "
            f"{r.overlaps:>5}  {r.runtime_s:>6.2f}  {note}"
        )

    lines.append("─" * 122)
    avg_proxy = sum(x.proxy_cost for x in rows) / len(rows)
    avg_rp = [x.replace_baseline for x in rows if x.replace_baseline is not None]
    avg_rp_v = sum(avg_rp) / len(avg_rp) if avg_rp else None
    vs_avg_rp = (
        ((avg_rp_v - avg_proxy) / avg_rp_v * 100.0)
        if avg_rp_v is not None and avg_rp_v > 0
        else None
    )

    rp_str = f"{avg_rp_v:.4f}" if avg_rp_v is not None else "—"
    vs_str = f"{vs_avg_rp:+.1f}%" if vs_avg_rp is not None else "—"
    tot_ov = sum(x.overlaps for x in rows)
    sum_place = sum(x.runtime_s for x in rows)
    lines.append(f"{'AVG':>8}  {avg_proxy:>8.4f}           {vs_str:>8}  RePlAce avg proxy {rp_str}")
    notes = []
    if wall_clock_s is not None:
        notes.append(f"wall-clock {wall_clock_s:.2f}s")
    notes.append(f"sum placement compute {sum_place:.2f}s")
    lines.append(f"Total overlaps {tot_ov}  |  " + "  |  ".join(notes))
    lines.append("")

    lines.append("Verdict:")
    if tot_ov > 0:
        lines.append("  • DISQUALIFIED: overlaps — fix legalization / halo rejection before tuning quality.")
        return "\n".join(lines)

    lines.append(
        f"  • Tier‑1 overlap audit: PASSED — {sum(1 for x in rows if x.overlaps == 0)} / {len(rows)} "
        "designs report zero pairwise hard-macro overlaps (proxy scorer overlap_count)."
    )

    gm_proxy = math.exp(sum(math.log(max(r.proxy_cost, 1e-12)) for r in rows) / len(rows))
    lines.append(f"  • Geometric mean proxy on this battery: {gm_proxy:.4f}")

    if vs_avg_rp is not None:
        if vs_avg_rp > 5:
            lines.append(
                "  • Strong vs RePlAce average on this battery — iterate on NG45 "
                "(Tier-2 proxy may differ)."
            )
        elif vs_avg_rp > 0:
            lines.append("  • Beating average RePlAce proxy — tighten worst benchmarks next.")
        else:
            lines.append(
                "  • Below average RePlAce — read den/wl vs cong/wl; "
                "if congestion_pressure is high, add spreading / halo / periphery bias."
            )

    wl_avg = sum(x.wl_share_of_weighted_terms for x in rows) / len(rows)
    lines.append(f"  • Mean WL share of weighted proxy composition: {wl_avg:.3f}")
    lines.append(
        "  • Columns — **rtImb**: Σ | pin-BB WLx / H-capacity − WLy / V-capacity | · net_weight "
        "(ICCAD µm/track supply); **splat**: coarse directional pin-splat load (NeurIPS'22 MaskPlace-style "
        "routing-demand cartoon); **plcC**: mean max(H,V) PlacementCost congestion after evaluation."
    )
    lines.append(
        "  • If proxy stalls: SA mostly optimizes **cheap surrogates**; Tier‑1 congestion comes from "
        "PlacementCost routing — use oracle hotspot / axis-greed phases (in placer) and raise "
        "``MACRO_PLACER_ADAPT_*`` / ``MACRO_PLACER_AXIS_GREED`` before chasing more surrogate terms."
    )
    lines.append(
        "  • **mpK**: count of separated high max(H,V) grid peaks (multi‑pole landscape). "
        "Large mpK with flat proxy suggests several competing hotspots — try ``MULTIPOLE_*`` refine."
    )

    risky = [x.name for x in rows if x.vs_replace_pct is not None and x.vs_replace_pct < -5]
    if risky:
        lines.append(f"  • Worst vs RePlAce (>{5}% gap): focus on " + ", ".join(risky))

    return "\n".join(lines)


def run_backtest(
    placer_path: Path,
    benchmarks: List[str],
    trials: int = 1,
    json_out: Optional[Path] = None,
    placer_factory=None,
    testcase_root: Optional[Path] = None,
    jobs: int = 1,
) -> Tuple[List[BenchDiagnostics], Optional[float]]:
    """
    Runs ``len(benchmarks) * trials`` placement evaluations.

    ``jobs`` > 1 uses a process pool (same workloads, shorter wall-clock on
    multi-core machines). Returned ``wall_clock_s`` is ``None`` if not measured.


    Sequential path preserves the old ``placer_factory`` hook; workers always
    call ``_load_placer`` per task (required for clean process isolation).
    """
    rows: List[BenchDiagnostics] = []
    root = testcase_root or TESTCASE_ICCAD04
    root_s = str(root.resolve())
    placer_s = str(placer_path.resolve())

    tasks: List[Tuple[str, str, str, Optional[str], Optional[int]]] = []
    for name in benchmarks:
        ng45_dir = NG45_BENCHMARKS.get(name)
        if trials <= 1:
            tasks.append((name, root_s, placer_s, ng45_dir, None))
        else:
            for _ in range(trials):
                seed_shuffle = random.randint(0, 10_000_000)
                tasks.append((name, root_s, placer_s, ng45_dir, seed_shuffle))

    started = time.time()
    eff_workers = 1
    if jobs <= 1:
        for t in tasks:
            name, _, _, ng45_dir, seed_shuffle = t
            rows.append(
                diagnose_single(
                    name,
                    root,
                    placer_path,
                    placer_factory,
                    ng45_dir=ng45_dir,
                    seed_shuffle=seed_shuffle,
                )
            )
    else:
        n_workers = max(1, min(jobs, len(tasks), mp.cpu_count() or 1))
        eff_workers = n_workers
        futures = []
        with ProcessPoolExecutor(max_workers=n_workers) as exe:
            for t in tasks:
                futures.append(exe.submit(_worker_diagnose, t))
            for fut in as_completed(futures):
                rows.append(fut.result())

    wall_clock_s = time.time() - started

    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "placer": str(placer_path),
            "parallel_jobs": eff_workers,
            "wall_clock_s": wall_clock_s,
            "sum_placement_compute_s": sum(r.runtime_s for r in rows),
            "runs": [r.to_dict() for r in rows],
        }
        json_out.write_text(json.dumps(payload, indent=2))

    return rows, wall_clock_s


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Backtest macro placer vs proxy baselines.")
    p.add_argument("placer", type=str, help="Path to submissions/.../placer.py")
    p.add_argument(
        "--benchmark",
        "-b",
        type=str,
        default=None,
        help="Single benchmark.",
    )
    p.add_argument("--all", "-a", action="store_true", help="All IBM benchmarks.")
    p.add_argument("--quick", "-q", action="store_true", help=f"Subset: {QUICK_BENCHMARKS}")
    p.add_argument("--ng45", action="store_true", help="NG45 designs (needs submodule flows).")
    p.add_argument("--trials", type=int, default=1, help="Monte Carlo over placer RNG (trial seeds).")
    p.add_argument("--json", type=str, default=None, help="Write JSON results.")
    p.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        help="Process pool size for benchmark×trial tasks. "
        "1 = serial (default; best for trivial placers — pool spawn+PyTorch import dominates). "
        "0 = auto: min(CPU, tasks, 16) for slow placers (~multi-minute runs) to cut wall-clock. "
        ">1 = fixed worker count.",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Apply condensed MACRO_PLACER_* budgets (~3–8 min/design on surrogate placers via ITER_CAP). "
        "Does not override variables you already exported.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Ultra‑condensed preset (~30–90s/design locally; skips oracle-heavy refinement). "
        "Proxy not comparable — wiring check only; "
        "dominates --fast if both are set.",
    )

    args = p.parse_args(argv)

    testcase_root = TESTCASE_ICCAD04
    if not args.ng45 and not testcase_root.exists():
        print(f"Missing {testcase_root}; run: git submodule update --init external/MacroPlacement", file=sys.stderr)
        sys.exit(1)

    placer_path = Path(args.placer)

    def factory():
        return _load_placer(placer_path)

    if args.ng45:
        benchmarks = list(NG45_BENCHMARKS.keys())
    elif args.all:
        benchmarks = list(BENCHMARKS)
    elif args.quick:
        benchmarks = QUICK_BENCHMARKS
    elif args.benchmark:
        benchmarks = [args.benchmark]
    else:
        benchmarks = ["ibm01"]

    n_tasks = len(benchmarks) * max(1, args.trials)
    if args.jobs == 0:
        # Auto-parallel: worthwhile when each placement ≫ process/torch import overhead
        # (competition-grade placers are often minutes → wall-clock wins).
        auto_jobs = min(mp.cpu_count() or 8, max(1, n_tasks), 16)
    elif args.jobs < 0:
        print("--jobs must be >= 0", file=sys.stderr)
        sys.exit(2)
    else:
        auto_jobs = max(1, args.jobs)

    if getattr(args, "smoke", False):
        apply_preset_env(_SMOKE_BACKTEST_ENV)
    elif args.fast:
        apply_preset_env(_FAST_BACKTEST_ENV)

    tags = ""
    if getattr(args, "smoke", False):
        tags = "  [SMOKE preset: very coarse — validate wiring only; proxy not comparable]"
    elif args.fast:
        tags = (
            "  [FAST preset: ITER_CAP/oracle trims — unset MACRO_PLACER_ITER_CAP etc. for full quality]"
        )

    print(
        f"backtest · {placer_path}  benchmarks={benchmarks}  trials={args.trials} "
        f"tasks={n_tasks}  jobs={auto_jobs}{tags}"
    )
    rows, wall_clock = run_backtest(
        placer_path,
        benchmarks,
        trials=args.trials,
        json_out=Path(args.json) if args.json else None,
        placer_factory=factory,
        testcase_root=testcase_root,
        jobs=auto_jobs,
    )

    aggregated: Dict[str, List[BenchDiagnostics]] = {}
    for r in rows:
        aggregated.setdefault(r.name, []).append(r)

    summarized: List[BenchDiagnostics] = []
    for name, group in aggregated.items():
        best = min(group, key=lambda x: x.proxy_cost + (1e9 if x.overlaps > 0 else 0))
        summarized.append(best)

    summarized.sort(key=lambda x: x.name)

    print(format_report(summarized, wall_clock_s=wall_clock))


if __name__ == "__main__":
    main()
