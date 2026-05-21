#!/usr/bin/env python3
"""SA-scale move deltas: fast_proxy vs oracle ranking at |Δoracle| < 0.01."""

from __future__ import annotations

import numpy as np
import torch

from macro_place.fast_proxy import build_fast_proxy_state, fast_proxy_from_benchmark
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import _set_placement, compute_proxy_cost

T0 = 0.005


def main() -> None:
    b, plc = load_benchmark_from_dir("external/MacroPlacement/Testcases/ICCAD04/ibm07")
    st = build_fast_proxy_state(b, plc)
    nh = b.num_hard_macros
    rng = np.random.default_rng(7)
    init = b.macro_positions[:nh].numpy().astype(np.float64)
    cw, ch = float(b.canvas_width), float(b.canvas_height)
    hw = b.macro_sizes[:nh, 0].numpy() * 0.5
    hh = b.macro_sizes[:nh, 1].numpy() * 0.5
    nm = max(cw, ch)

    def oracle(pos: np.ndarray) -> float:
        ft = b.macro_positions.clone()
        ft[:nh] = torch.tensor(pos, dtype=torch.float32)
        _set_placement(plc, ft, b)
        return float(compute_proxy_cost(ft, b, plc)["proxy_cost"])

    def fast(pos: np.ndarray) -> float:
        c, _ = fast_proxy_from_benchmark(b, pos, plc=plc, state=st, fd_iters=0)
        return float(c)

    base_o = oracle(init)
    base_f = fast(init)
    print(f"Base: fast={base_f:.4f} oracle={base_o:.4f}  T0={T0}")
    print("\n--- Single-shift moves from initial (shift scale = T0 * nm * 0.3) ---")
    step = T0 * nm * 0.3
    movable = [i for i in range(nh) if not bool(b.macro_fixed[i].item())]
    n_try = min(80, len(movable) * 2)

    move_d_o: list[float] = []
    move_d_f: list[float] = []
    disagree = 0
    for _ in range(n_try):
        pos = init.copy()
        i = int(rng.choice(movable))
        pos[i] += rng.normal(0, step, 2)
        pos[i, 0] = np.clip(pos[i, 0], hw[i], cw - hw[i])
        pos[i, 1] = np.clip(pos[i, 1], hh[i], ch - hh[i])
        o = oracle(pos)
        f = fast(pos)
        d_o = o - base_o
        d_f = f - base_f
        move_d_o.append(d_o)
        move_d_f.append(d_f)
        if d_o * d_f < 0:
            disagree += 1

    do = np.asarray(move_d_o)
    df = np.asarray(move_d_f)
    rx = np.argsort(np.argsort(do))
    ry = np.argsort(np.argsort(df))
    sp = float(np.corrcoef(rx.astype(float), ry.astype(float))[0, 1])
    print(f"  n_moves={n_try}  disagree_sign={disagree} ({100*disagree/n_try:.1f}%)")
    print(f"  Spearman on move deltas = {sp:.4f}")
    print(f"  |Δoracle| mean={np.abs(do).mean():.5f} max={np.abs(do).max():.5f}")
    print(f"  |Δfast|   mean={np.abs(df).mean():.5f} max={np.abs(df).max():.5f}")
    small = np.abs(do) < 0.01
    if small.sum() > 0:
        d2 = int(np.sum((do[small] * df[small]) < 0))
        print(f"  |Δoracle|<0.01: n={small.sum()} disagree={d2} ({100*d2/small.sum():.1f}%)")


if __name__ == "__main__":
    main()
