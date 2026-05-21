"""GWTW simulated annealing: fast_proxy acceptance, oracle at sync + final pick."""

from __future__ import annotations

import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from macro_place.benchmark import Benchmark
from macro_place.fast_proxy import FastProxyState, fast_proxy_from_benchmark
T0_DEFAULT = 0.005
TMIN_DEFAULT = 1e-8
N_MOVES_PER_ITER = 20
MOVE_PROBS = (0.24, 0.48, 0.72, 0.96, 1.0)


@dataclass
class _Worker:
    pos: np.ndarray
    current_fast: float
    best_pos: np.ndarray
    best_fast: float
    T: float
    rng: np.random.Generator = field(repr=False)


def _clip_hard(
    pos: np.ndarray,
    *,
    n_hard: int,
    movable_mask: np.ndarray,
    fixed_init: np.ndarray,
    half_w: np.ndarray,
    half_h: np.ndarray,
    cw: float,
    ch: float,
) -> np.ndarray:
    out = pos.copy()
    for i in range(n_hard):
        if not movable_mask[i]:
            out[i] = fixed_init[i]
        else:
            out[i, 0] = np.clip(out[i, 0], half_w[i], cw - half_w[i])
            out[i, 1] = np.clip(out[i, 1], half_h[i], ch - half_h[i])
    return out


def _movable_list(movable_idx: np.ndarray, movable_mask: np.ndarray) -> List[int]:
    return [int(i) for i in movable_idx if movable_mask[i]]


def _apply_paper_move(
    pos: np.ndarray,
    rng: np.random.Generator,
    T: float,
    movable: List[int],
    *,
    cw: float,
    ch: float,
) -> np.ndarray:
    """Paper move types; caller legalizes and clips after."""
    if not movable:
        return pos.copy()
    p = pos.copy()
    u = float(rng.random())
    nm = max(cw, ch)
    step_shift = T * nm * 0.3
    step_move = T * nm * 2.0

    if u < MOVE_PROBS[0] and len(movable) >= 2:
        i, j = (int(x) for x in rng.choice(movable, size=2, replace=False))
        p[i], p[j] = p[j].copy(), p[i].copy()
    elif u < MOVE_PROBS[1]:
        i = int(rng.choice(movable))
        p[i] += rng.normal(0.0, step_shift, size=2)
    elif u < MOVE_PROBS[2]:
        i = int(rng.choice(movable))
        p[i] += rng.normal(0.0, step_move, size=2)
    elif u < MOVE_PROBS[3] and len(movable) >= 3:
        k = int(rng.integers(3, min(6, len(movable) + 1)))
        ids = list(rng.choice(movable, size=k, replace=False))
        coords = p[ids].copy()
        rng.shuffle(coords)
        p[ids] = coords
    else:
        i = int(rng.choice(movable))
        if rng.random() < 0.5:
            p[i, 0] = cw - p[i, 0]
        else:
            p[i, 1] = ch - p[i, 1]
    return p


def _eval_fast(
    benchmark: Benchmark,
    state: FastProxyState,
    pos: np.ndarray,
    *,
    fd_iters: int,
) -> float:
    cost, _ = fast_proxy_from_benchmark(
        benchmark, pos, state=state, fd_iters=fd_iters
    )
    return float(cost)


def _init_worker_pos(
    wi: int,
    *,
    benchmark: Benchmark,
    movable_idx: np.ndarray,
    movable_mask: np.ndarray,
    fixed_init: np.ndarray,
    half_w: np.ndarray,
    half_h: np.ndarray,
    cw: float,
    ch: float,
    legalize_fn: Callable[[np.ndarray], np.ndarray],
    n_hard: int,
    seed_base: int,
) -> np.ndarray:
    """Benchmark initial + increasing jitter (worker 0 = exact initial, legalized)."""
    benchmark_initial = benchmark.macro_positions[:n_hard].numpy().astype(np.float64).copy()
    rng_w = np.random.default_rng(seed_base + wi)
    jitter_scale = 0.0 if wi == 0 else 0.01 * (1.0 + wi * 0.5)

    pos = benchmark_initial.copy()
    for mi in movable_idx:
        mi = int(mi)
        if not movable_mask[mi]:
            continue
        if jitter_scale > 0.0:
            pos[mi, 0] += float(rng_w.normal(0.0, jitter_scale * cw))
            pos[mi, 1] += float(rng_w.normal(0.0, jitter_scale * ch))
        pos[mi, 0] = np.clip(pos[mi, 0], half_w[mi], cw - half_w[mi])
        pos[mi, 1] = np.clip(pos[mi, 1], half_h[mi], ch - half_h[mi])

    return legalize_fn(
        _clip_hard(
            pos,
            n_hard=n_hard,
            movable_mask=movable_mask,
            fixed_init=fixed_init,
            half_w=half_w,
            half_h=half_h,
            cw=cw,
            ch=ch,
        )
    )


def _run_sa_chunk(
    worker: _Worker,
    chunk_iters: int,
    *,
    benchmark: Benchmark,
    state: FastProxyState,
    fd_iters: int,
    legalize_fn: Callable[[np.ndarray], np.ndarray],
    movable: List[int],
    n_hard: int,
    movable_mask: np.ndarray,
    fixed_init: np.ndarray,
    half_w: np.ndarray,
    half_h: np.ndarray,
    cw: float,
    ch: float,
    alpha: float,
    Tmin: float,
) -> None:
    """Run ``chunk_iters`` SA steps with fast_proxy Metropolis (20 moves per iter)."""
    rng = worker.rng
    pos = worker.pos
    current = worker.current_fast
    best_pos = worker.best_pos
    best_fast = worker.best_fast
    T = worker.T

    for _ in range(chunk_iters):
        trial = pos.copy()
        for _m in range(N_MOVES_PER_ITER):
            trial = _apply_paper_move(trial, rng, T, movable, cw=cw, ch=ch)
            trial = legalize_fn(
                _clip_hard(
                    trial,
                    n_hard=n_hard,
                    movable_mask=movable_mask,
                    fixed_init=fixed_init,
                    half_w=half_w,
                    half_h=half_h,
                    cw=cw,
                    ch=ch,
                )
            )
            new_fast = _eval_fast(benchmark, state, trial, fd_iters=fd_iters)
            delta = new_fast - current
            if delta < 0.0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
                current = new_fast
                pos = trial
                if current < best_fast:
                    best_fast = current
                    best_pos = trial.copy()
        T = max(T * alpha, Tmin)

    worker.pos = pos
    worker.current_fast = current
    worker.best_pos = best_pos
    worker.best_fast = best_fast
    worker.T = T


def _gwtw_sync(
    workers: List[_Worker],
    *,
    top_k: int,
    oracle_proxy_fn: Callable[[np.ndarray], Tuple],
    oracle_lock: threading.Lock,
    legalize_fn: Callable[[np.ndarray], np.ndarray],
    benchmark: Benchmark,
    state: FastProxyState,
    fd_iters: int,
    movable_idx: np.ndarray,
    movable_mask: np.ndarray,
    n_hard: int,
    fixed_init: np.ndarray,
    half_w: np.ndarray,
    half_h: np.ndarray,
    cw: float,
    ch: float,
    seed_base: int,
) -> None:
    """Rank workers by real oracle on local bests; clone top_k with jitter + reheat."""
    n_workers = len(workers)
    oracle_costs: List[float] = []
    oracle_positions: List[np.ndarray] = []

    for w in workers:
        with oracle_lock:
            _ft, costs = oracle_proxy_fn(w.best_pos)
        oc = float(costs.get("proxy_cost", float("inf")))
        if int(costs.get("overlap_count", 0)) > 0:
            oc = float("inf")
        oracle_costs.append(oc)
        oracle_positions.append(w.best_pos.copy())

    ranked = sorted(range(n_workers), key=lambda i: oracle_costs[i])
    top_idx = ranked[: max(1, min(top_k, n_workers))]
    top_positions = [oracle_positions[i] for i in top_idx]

    jitter = 0.02 * max(cw, ch)
    new_workers: List[_Worker] = []
    for wi in range(n_workers):
        src = top_positions[wi % len(top_positions)]
        pos = src.copy()
        rng = np.random.default_rng(seed_base + wi * 7919 + len(new_workers) * 104729)
        for mi in movable_idx:
            mi = int(mi)
            if movable_mask[mi]:
                pos[mi] += rng.normal(0.0, jitter, size=2)
        pos = legalize_fn(
            _clip_hard(
                pos,
                n_hard=n_hard,
                movable_mask=movable_mask,
                fixed_init=fixed_init,
                half_w=half_w,
                half_h=half_h,
                cw=cw,
                ch=ch,
            )
        )
        fast_c = _eval_fast(benchmark, state, pos, fd_iters=fd_iters)
        src_worker = workers[top_idx[wi % len(top_idx)]]
        new_workers.append(
            _Worker(
                pos=pos,
                current_fast=fast_c,
                best_pos=pos.copy(),
                best_fast=fast_c,
                T=min(T0_DEFAULT, src_worker.T * 1.2),
                rng=np.random.default_rng(seed_base + wi + 10007 * (wi + 1)),
            )
        )
    workers[:] = new_workers


def _budget_n_iters(
    n_workers: int,
    sync_freq: float,
    *,
    budget_secs: float,
    fast_proxy_sec_per_iter: float = 0.205 * N_MOVES_PER_ITER,
    oracle_sec: float = 2.0,
) -> int:
    """Derive per-worker iteration count from wall-clock budget."""
    sync_freq = max(0.05, min(sync_freq, 0.5))
    n_syncs = max(0, int(1.0 / sync_freq) - 1)
    oracle_budget = (n_syncs * n_workers + n_workers) * oracle_sec
    sa_budget = max(60.0, budget_secs - oracle_budget)
    denom = fast_proxy_sec_per_iter * max(n_workers, 1)
    return max(200, int(sa_budget / denom))


def gwtw_proxy_sa(
    benchmark: Benchmark,
    combo_plc,
    fast_proxy_state: FastProxyState,
    legalize_fn: Callable[[np.ndarray], np.ndarray],
    oracle_proxy_fn: Callable[[np.ndarray], Tuple],
    movable_idx: np.ndarray,
    fixed_init: np.ndarray,
    movable_mask: np.ndarray,
    sizes_np: np.ndarray,
    half_w: np.ndarray,
    half_h: np.ndarray,
    cw: float,
    ch: float,
    *,
    n_workers: int = 8,
    n_iters: int | None = None,
    sync_freq: float = 0.1,
    top_k: int = 2,
    T0: float = T0_DEFAULT,
    Tmin: float = TMIN_DEFAULT,
    seed_base: int = 42,
    budget_secs: float | None = None,
    budget_ok: Callable[[float], bool] | None = None,
) -> np.ndarray:
    """
    GWTW pool SA: ``fast_proxy`` acceptance; real ``oracle_proxy`` at sync + final.

    Returns best hard-macro positions ``[n_hard, 2]`` by lowest **oracle** proxy.
    """
    del combo_plc  # oracle closure holds plc reference

    n_hard = int(benchmark.num_hard_macros)
    n_workers = max(1, int(n_workers))
    sync_freq = float(sync_freq)
    top_k = max(1, int(top_k))
    T0 = float(T0)
    Tmin = max(float(Tmin), 1e-15)

    if budget_secs is None:
        budget_secs = float(os.environ.get("MACRO_PLACER_BUDGET_SECS", "720"))
    if n_iters is None:
        n_iters = _budget_n_iters(n_workers, sync_freq, budget_secs=budget_secs)

    n_iters = max(200, int(n_iters))
    sync_iter = max(1, int(n_iters * sync_freq))
    fd_iters = int(os.environ.get("MACRO_PLACER_FAST_PROXY_FD_ITERS", "0"))

    if os.environ.get("MACRO_PLACER_DIAGNOSE", "0").lower() in ("1", "true", "yes"):
        import sys

        print(
            f"[DIAGNOSE] gwtw_proxy_sa workers={n_workers} n_iters={n_iters} "
            f"sync_iter={sync_iter} sync_freq={sync_freq} top_k={top_k} "
            f"budget_secs={budget_secs:.0f} fd_iters={fd_iters}",
            file=sys.stderr,
        )

    alpha = math.exp(math.log(Tmin / T0) / max(n_iters, 1))
    movable = _movable_list(movable_idx, movable_mask)
    oracle_lock = threading.Lock()
    state = fast_proxy_state

    workers: List[_Worker] = []
    for wi in range(n_workers):
        pos = _init_worker_pos(
            wi,
            benchmark=benchmark,
            movable_idx=movable_idx,
            movable_mask=movable_mask,
            fixed_init=fixed_init,
            half_w=half_w,
            half_h=half_h,
            cw=cw,
            ch=ch,
            legalize_fn=legalize_fn,
            n_hard=n_hard,
            seed_base=seed_base,
        )
        fast_c = _eval_fast(benchmark, state, pos, fd_iters=fd_iters)
        if os.environ.get("MACRO_PLACER_DIAGNOSE", "0").lower() in ("1", "true", "yes"):
            import sys

            with oracle_lock:
                _ft, costs = oracle_proxy_fn(pos)
            print(
                f"[DIAGNOSE] worker{wi} init jitter_scale="
                f"{0.0 if wi == 0 else 0.01 * (1.0 + wi * 0.5):.4f} "
                f"fast={fast_c:.4f} oracle={float(costs.get('proxy_cost', 0)):.4f} "
                f"overlap={int(costs.get('overlap_count', 0))}",
                file=sys.stderr,
            )
        workers.append(
            _Worker(
                pos=pos,
                current_fast=fast_c,
                best_pos=pos.copy(),
                best_fast=fast_c,
                T=T0,
                rng=np.random.default_rng(seed_base + wi),
            )
        )

    iters_done = 0
    while iters_done < n_iters:
        if budget_ok is not None and not budget_ok(5.0):
            break
        chunk = min(sync_iter, n_iters - iters_done)

        def _chunk(w: _Worker) -> None:
            _run_sa_chunk(
                w,
                chunk,
                benchmark=benchmark,
                state=state,
                fd_iters=fd_iters,
                legalize_fn=legalize_fn,
                movable=movable,
                n_hard=n_hard,
                movable_mask=movable_mask,
                fixed_init=fixed_init,
                half_w=half_w,
                half_h=half_h,
                cw=cw,
                ch=ch,
                alpha=alpha,
                Tmin=Tmin,
            )

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            list(pool.map(_chunk, workers))

        iters_done += chunk
        if iters_done < n_iters:
            _gwtw_sync(
                workers,
                top_k=top_k,
                oracle_proxy_fn=oracle_proxy_fn,
                oracle_lock=oracle_lock,
                legalize_fn=legalize_fn,
                benchmark=benchmark,
                state=state,
                fd_iters=fd_iters,
                movable_idx=movable_idx,
                movable_mask=movable_mask,
                n_hard=n_hard,
                fixed_init=fixed_init,
                half_w=half_w,
                half_h=half_h,
                cw=cw,
                ch=ch,
                seed_base=seed_base + iters_done,
            )

    best_pos = workers[0].best_pos.copy()
    best_oracle = float("inf")
    for w in workers:
        with oracle_lock:
            _ft, costs = oracle_proxy_fn(w.best_pos)
        oc = float(costs.get("proxy_cost", float("inf")))
        if int(costs.get("overlap_count", 0)) > 0:
            continue
        if oc < best_oracle:
            best_oracle = oc
            best_pos = w.best_pos.copy()

    return best_pos


# Legacy oracle-every-iter path (kept for reference / A-B)
@dataclass
class ProxySAContext:
    n_hard: int
    movable_idx: np.ndarray
    movable_mask: np.ndarray
    fixed_init: np.ndarray
    sizes_np: np.ndarray
    half_w: np.ndarray
    half_h: np.ndarray
    cw: float
    ch: float
    legalize_fn: Callable[[np.ndarray], np.ndarray]
    oracle_fn: Callable[[np.ndarray], dict]
    oracle_lock: threading.Lock


def run_gwtw_proxy_sa(ctx: ProxySAContext, *, budget_secs: float, budget_ok: Callable, seed_base: int):
    """Deprecated: use :func:`gwtw_proxy_sa` with ``fast_proxy_state``."""
    raise NotImplementedError(
        "run_gwtw_proxy_sa is replaced by gwtw_proxy_sa(fast_proxy acceptance)"
    )
