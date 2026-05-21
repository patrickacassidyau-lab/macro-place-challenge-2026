#!/usr/bin/env python3
"""Rank-correlation diagnostic: fast_proxy vs oracle on ibm07 random legal layouts."""

from __future__ import annotations

import math
import sys
import time

import numpy as np
import torch

from macro_place.fast_proxy import build_fast_proxy_state, fast_proxy_from_benchmark
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import _set_placement, compute_proxy_cost


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx.astype(np.float64), ry.astype(np.float64))[0, 1])


def _oracle_proxy(benchmark, plc, pos: np.ndarray) -> float:
    nh = benchmark.num_hard_macros
    ft = benchmark.macro_positions.clone()
    ft[:nh] = torch.tensor(pos, dtype=torch.float32)
    _set_placement(plc, ft, benchmark)
    return float(compute_proxy_cost(ft, benchmark, plc)["proxy_cost"])


def _clip_legal(pos: np.ndarray, benchmark, rng: np.random.Generator) -> np.ndarray:
    """Jitter + canvas clip (same distribution as calibration)."""
    nh = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    half_w = benchmark.macro_sizes[:nh, 0].numpy() * 0.5
    half_h = benchmark.macro_sizes[:nh, 1].numpy() * 0.5
    init = benchmark.macro_positions[:nh].numpy().astype(np.float64)
    p = init.copy()
    j = 0.12 * max(cw, ch)
    p[:, 0] += rng.uniform(-j, j, nh)
    p[:, 1] += rng.uniform(-j, j, nh)
    p[:, 0] = np.clip(p[:, 0], half_w, cw - half_w)
    p[:, 1] = np.clip(p[:, 1], half_h, ch - half_h)
    return p


def _pairwise_rank_disagreement(fast: np.ndarray, oracle: np.ndarray) -> None:
    n = len(fast)
    bins = [(0, 0.01), (0.01, 0.05), (0.05, 0.10), (0.10, 0.25), (0.25, 1e9)]
    print("\n--- Pairwise ranking disagreement (oracle delta bins) ---")
    for lo, hi in bins:
        disagree = 0
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                d_o = oracle[j] - oracle[i]
                if abs(d_o) < lo or abs(d_o) >= hi:
                    continue
                d_f = fast[j] - fast[i]
                total += 1
                if d_o * d_f < 0:
                    disagree += 1
        if total > 0:
            print(
                f"  |Δoracle| in [{lo:.2f}, {hi:.2f}): "
                f"disagree {disagree}/{total} = {100.0 * disagree / total:.1f}%"
            )


def main() -> None:
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    root = "external/MacroPlacement/Testcases/ICCAD04/ibm07"
    print(f"Loading {root} ...", flush=True)
    benchmark, plc = load_benchmark_from_dir(root)
    state = build_fast_proxy_state(benchmark, plc)
    nh = benchmark.num_hard_macros
    rng = np.random.default_rng(42)

    fast_vals: list[float] = []
    oracle_vals: list[float] = []
    positions: list[np.ndarray] = []

    print(f"Scoring {n_samples} random clipped layouts ...", flush=True)
    t0 = time.perf_counter()
    for k in range(n_samples):
        pos = _clip_legal(benchmark.macro_positions[:nh].numpy().astype(np.float64), benchmark, rng)
        fp, _ = fast_proxy_from_benchmark(benchmark, pos, plc=plc, state=state, fd_iters=0)
        oc = _oracle_proxy(benchmark, plc, pos)
        fast_vals.append(fp)
        oracle_vals.append(oc)
        positions.append(pos)
        if (k + 1) % 10 == 0:
            print(f"  {k + 1}/{n_samples} done ({time.perf_counter() - t0:.0f}s)", flush=True)

    fa = np.asarray(fast_vals)
    oa = np.asarray(oracle_vals)
    pearson = float(np.corrcoef(fa, oa)[0, 1])
    spearman = _spearman(fa, oa)
    mae = float(np.mean(np.abs(fa - oa)))
    rmse = float(np.sqrt(np.mean((fa - oa) ** 2)))
    scale = float(np.mean(oa / np.maximum(fa, 1e-9)))

    print("\n=== ibm07 fast_proxy vs oracle (random layouts) ===")
    print(f"  n = {n_samples}")
    print(f"  Pearson r  = {pearson:.4f}")
    print(f"  Spearman r = {spearman:.4f}  {'PASS' if spearman >= 0.75 else 'FAIL (<0.75 → SA ranking unreliable)'}")
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  mean(oracle/fast) scale = {scale:.4f}")
    print(f"  fast  mean={fa.mean():.4f} std={fa.std():.4f}")
    print(f"  oracle mean={oa.mean():.4f} std={oa.std():.4f}")

    _pairwise_rank_disagreement(fa, oa)

    # Start position (benchmark initial)
    init = benchmark.macro_positions[:nh].numpy().astype(np.float64)
    f_init, _ = fast_proxy_from_benchmark(benchmark, init, plc=plc, state=state, fd_iters=0)
    o_init = _oracle_proxy(benchmark, plc, init)

  # Best oracle among random sample (reference)
    best_i = int(np.argmin(oa))
    f_best, o_best = fa[best_i], oa[best_i]

    print("\n--- Start vs random-sample best ---")
    print(f"  INITIAL  fast={f_init:.4f}  oracle={o_init:.4f}")
    print(f"  BEST@50  fast={f_best:.4f}  oracle={o_best:.4f}  (oracle min in sample)")
    print(
        f"  fast thinks init better than sample-best? {f_init < f_best} "
        f"(oracle: {o_init < o_best})"
    )

    # Proxy-SA end position (short run to approximate recent ~1.79 result)
    print("\n--- Proxy-SA end position (short gwtw run, ~2–4 min) ---", flush=True)
    try:
        from macro_place.proxy_sa import gwtw_proxy_sa

        # Minimal legalize: clip + placer-style not available standalone
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        half_w = benchmark.macro_sizes[:nh, 0].numpy() * 0.5
        half_h = benchmark.macro_sizes[:nh, 1].numpy() * 0.5
        fixed = init.copy()
        movable_mask = (~benchmark.macro_fixed[:nh].numpy()).astype(bool)
        movable_idx = np.where(movable_mask)[0]

        def legalize(p: np.ndarray) -> np.ndarray:
            out = p.copy()
            for i in range(nh):
                if not movable_mask[i]:
                    out[i] = fixed[i]
                else:
                    out[i, 0] = np.clip(out[i, 0], half_w[i], cw - half_w[i])
                    out[i, 1] = np.clip(out[i, 1], half_h[i], ch - half_h[i])
            return out

        def oracle_fn(pos: np.ndarray):
            ft = benchmark.macro_positions.clone()
            ft[:nh] = torch.tensor(pos, dtype=torch.float32)
            _set_placement(plc, ft, benchmark)
            c = compute_proxy_cost(ft, benchmark, plc)
            return ft, c

        end = gwtw_proxy_sa(
            benchmark,
            plc,
            state,
            legalize_fn=legalize,
            oracle_proxy_fn=oracle_fn,
            movable_idx=movable_idx,
            fixed_init=fixed,
            movable_mask=movable_mask,
            sizes_np=benchmark.macro_sizes[:nh].numpy(),
            half_w=half_w,
            half_h=half_h,
            cw=cw,
            ch=ch,
            n_workers=2,
            n_iters=80,
            sync_freq=0.25,
            top_k=1,
            budget_secs=180.0,
            seed_base=42,
        )
        f_end, _ = fast_proxy_from_benchmark(benchmark, end, plc=plc, state=state, fd_iters=0)
        o_end = _oracle_proxy(benchmark, plc, end)
        print(f"  END (short SA)  fast={f_end:.4f}  oracle={o_end:.4f}")
        print(f"  INITIAL         fast={f_init:.4f}  oracle={o_init:.4f}")
        print(f"  Δfast  (end-init) = {f_end - f_init:+.4f}")
        print(f"  Δoracle(end-init) = {o_end - o_init:+.4f}")
        if f_end < f_init and o_end > o_init:
            print("  >>> WRONG DIRECTION: fast_proxy improved but oracle worsened")
        elif f_end > f_init and o_end < o_init:
            print("  >>> fast_proxy worsened but oracle improved (SA rejected good basin)")
        else:
            print("  >>> same direction on init vs end")
    except Exception as e:
        print(f"  (skipped short SA: {e})", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
