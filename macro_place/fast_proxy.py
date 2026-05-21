"""
Fast numpy approximation of the Tier-1 oracle proxy (PlacementCost).

proxy = wirelength_cost + density_cost + 0.5 * congestion_cost

where ``density_cost`` and ``wirelength_cost`` follow ``plc_client_os`` conventions
(``get_density_cost`` already includes a 0.5 factor on the top-10% bin mean;
``get_cost`` normalizes HPWL by ``(canvas_w + canvas_h) * net_cnt``).

Congestion matches ``get_congestion_cost``: ABU top 5% of ``H_routing_cong + V_routing_cong``
(Python **list** concatenation, not elementwise sum) after pin L-routing and smoothing.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from macro_place.benchmark import Benchmark


def _net_weights_from_plc(plc, benchmark: Benchmark) -> np.ndarray:
    """Driver-pin weights aligned with ``benchmark.net_pin_nodes`` (same filter as loader)."""
    nh = int(benchmark.num_hard_macros)
    num_macros = int(benchmark.num_macros)
    plc_idx_to_bench: Dict[int, int] = {}
    for bench_idx, plc_idx in enumerate(benchmark.hard_macro_indices):
        plc_idx_to_bench[plc_idx] = bench_idx
    for off, plc_idx in enumerate(benchmark.soft_macro_indices):
        plc_idx_to_bench[plc_idx] = nh + off
    # ports mapped at end of macro index range
    for port_off, plc_idx in enumerate(plc.port_indices):
        plc_idx_to_bench[plc_idx] = num_macros + port_off

    name_to_bench: Dict[str, int] = {}
    for plc_idx, bench_idx in plc_idx_to_bench.items():
        if plc_idx < len(plc.modules_w_pins):
            name_to_bench[plc.modules_w_pins[plc_idx].get_name()] = bench_idx

    pin_slot: Dict[str, Tuple[str, int]] = {}
    for idx in plc.hard_macro_pin_indices:
        pin = plc.modules_w_pins[idx]
        macro_name = pin.get_macro_name() if hasattr(pin, "get_macro_name") else None
        if macro_name and hasattr(pin, "get_name"):
            pin_slot[pin.get_name()] = (macro_name, len(pin_slot))

    weights: List[float] = []
    for driver, sinks in plc.nets.items():
        nodes_in_net = set()
        for pin_name in [driver] + list(sinks):
            if pin_name in pin_slot:
                macro_name, _slot = pin_slot[pin_name]
                if macro_name in name_to_bench:
                    nodes_in_net.add(name_to_bench[macro_name])
            else:
                parent = pin_name.split("/")[0]
                if parent in name_to_bench:
                    nodes_in_net.add(name_to_bench[parent])
        if nodes_in_net:
            driver_idx = plc.mod_name_to_indices[driver]
            weights.append(float(plc.modules_w_pins[driver_idx].get_weight()))
    return np.asarray(weights, dtype=np.float64)


@dataclass
class FastProxyState:
    """Precomputed structures for repeated ``fast_proxy`` calls."""

    net_pin_nodes: List[np.ndarray]
    net_weights: np.ndarray
    hard_pin_offsets: List[np.ndarray]
    hard_sizes: np.ndarray
    soft_sizes: np.ndarray
    soft_half: np.ndarray
    nh: int
    n_soft: int
    n_ports: int
    cw: float
    ch: float
    grid_rows: int
    grid_cols: int
    hroutes_per_micron: float
    vroutes_per_micron: float
    net_cnt: float
    smooth_range: int
    cell_w: float
    cell_h: float
    grid_area: float
    grid_h_cap: float
    grid_v_cap: float
    # Flattened pin tables for vectorized WL / congestion
    pin_owners: np.ndarray
    pin_slots: np.ndarray
    pin_ox: np.ndarray
    pin_oy: np.ndarray
    net_start: np.ndarray


def build_fast_proxy_state(
    benchmark: Benchmark,
    plc=None,
) -> FastProxyState:
    """Build cached state from a ``Benchmark`` (optional ``plc`` for net_cnt / smooth_range)."""
    nh = int(benchmark.num_hard_macros)
    n_soft = int(benchmark.num_macros - nh)
    n_ports = int(benchmark.port_positions.shape[0])
    nrow = int(benchmark.grid_rows)
    ncol = int(benchmark.grid_cols)
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    cell_w = cw / max(ncol, 1)
    cell_h = ch / max(nrow, 1)

    net_pin_nodes = [pn.numpy().astype(np.int64) for pn in benchmark.net_pin_nodes]
    net_weights = _net_weights_from_plc(plc, benchmark) if plc is not None else benchmark.net_weights.numpy().astype(np.float64)

    offsets: List[np.ndarray] = []
    for i in range(nh):
        o = benchmark.macro_pin_offsets[i]
        offsets.append(
            o.numpy().astype(np.float64) if o.numel() else np.zeros((0, 2), dtype=np.float64)
        )

    sizes = benchmark.macro_sizes.numpy().astype(np.float64)
    hard_sizes = sizes[:nh]
    soft_sizes = sizes[nh : nh + n_soft] if n_soft > 0 else np.zeros((0, 2), dtype=np.float64)
    soft_half = 0.5 * soft_sizes if n_soft else np.zeros((0, 2), dtype=np.float64)

    net_cnt = float(benchmark.num_nets)
    smooth_range = 0
    if plc is not None:
        net_cnt = max(float(getattr(plc, "net_cnt", net_cnt) or net_cnt), 1.0)
        smooth_range = int(getattr(plc, "smooth_range", 0) or 0)

    hr = float(benchmark.hroutes_per_micron)
    vr = float(benchmark.vroutes_per_micron)
    grid_h_cap = cell_w * hr
    grid_v_cap = cell_h * vr

    pin_owners: List[int] = []
    pin_slots: List[int] = []
    pin_ox: List[float] = []
    pin_oy: List[float] = []
    net_start: List[int] = [0]
    for pn in net_pin_nodes:
        for row in pn:
            o, s = int(row[0]), int(row[1])
            pin_owners.append(o)
            pin_slots.append(s)
            if o < nh:
                ofs = offsets[o]
                if ofs.shape[0] > s:
                    pin_ox.append(float(ofs[s, 0]))
                    pin_oy.append(float(ofs[s, 1]))
                else:
                    pin_ox.append(0.0)
                    pin_oy.append(0.0)
            else:
                pin_ox.append(0.0)
                pin_oy.append(0.0)
        net_start.append(len(pin_owners))

    return FastProxyState(
        net_pin_nodes=net_pin_nodes,
        net_weights=net_weights,
        hard_pin_offsets=offsets,
        hard_sizes=hard_sizes,
        soft_sizes=soft_sizes,
        soft_half=soft_half,
        nh=nh,
        n_soft=n_soft,
        n_ports=n_ports,
        cw=cw,
        ch=ch,
        grid_rows=nrow,
        grid_cols=ncol,
        hroutes_per_micron=hr,
        vroutes_per_micron=vr,
        net_cnt=net_cnt,
        smooth_range=smooth_range,
        cell_w=cell_w,
        cell_h=cell_h,
        grid_area=cell_w * cell_h,
        grid_h_cap=grid_h_cap,
        grid_v_cap=grid_v_cap,
        pin_owners=np.asarray(pin_owners, dtype=np.int32),
        pin_slots=np.asarray(pin_slots, dtype=np.int32),
        pin_ox=np.asarray(pin_ox, dtype=np.float64),
        pin_oy=np.asarray(pin_oy, dtype=np.float64),
        net_start=np.asarray(net_start, dtype=np.int32),
    )


def _pos_to_cell(x: float, y: float, st: FastProxyState) -> Tuple[int, int]:
    col = int(math.floor(x / st.cell_w)) if st.cell_w > 0 else 0
    row = int(math.floor(y / st.cell_h)) if st.cell_h > 0 else 0
    col = max(0, min(col, st.grid_cols - 1))
    row = max(0, min(row, st.grid_rows - 1))
    return row, col


def _pin_xy(
    owner: int,
    slot: int,
    hard_pos: np.ndarray,
    soft_pos: np.ndarray,
    port_pos: np.ndarray,
    st: FastProxyState,
) -> Tuple[float, float]:
    nh = st.nh
    if owner < nh:
        ox, oy = float(hard_pos[owner, 0]), float(hard_pos[owner, 1])
        ofs = st.hard_pin_offsets[owner]
        if ofs.shape[0] > 0 and slot < ofs.shape[0]:
            return float(ox + ofs[slot, 0]), float(oy + ofs[slot, 1])
        return ox, oy
    if owner < nh + st.n_soft:
        j = owner - nh
        return float(soft_pos[j, 0]), float(soft_pos[j, 1])
    pidx = owner - nh - st.n_soft
    return float(port_pos[pidx, 0]), float(port_pos[pidx, 1])


def _fast_fd(
    hard_pos: np.ndarray,
    soft_pos: np.ndarray,
    port_pos: np.ndarray,
    st: FastProxyState,
    *,
    fd_iters: int,
    att: float = 0.35,
) -> np.ndarray:
    """Few spring iterations: pull soft macros toward fixed-pin centroids per net."""
    if st.n_soft == 0 or fd_iters <= 0:
        return soft_pos

    soft = soft_pos.copy()
    nh = st.nh
    for _ in range(fd_iters):
        disp = np.zeros_like(soft)
        counts = np.zeros(st.n_soft, dtype=np.float64)
        for nid, pn in enumerate(st.net_pin_nodes):
            if pn.shape[0] < 2:
                continue
            fx: List[float] = []
            fy: List[float] = []
            soft_idx: List[int] = []
            for row in pn:
                o, s = int(row[0]), int(row[1])
                if o < nh:
                    x, y = _pin_xy(o, s, hard_pos, soft, port_pos, st)
                    fx.append(x)
                    fy.append(y)
                elif o < nh + st.n_soft:
                    soft_idx.append(o - nh)
                else:
                    x, y = _pin_xy(o, s, hard_pos, soft, port_pos, st)
                    fx.append(x)
                    fy.append(y)
            if not fx:
                continue
            cx = float(np.mean(fx))
            cy = float(np.mean(fy))
            w = float(st.net_weights[nid])
            for j in soft_idx:
                disp[j, 0] += w * (cx - soft[j, 0])
                disp[j, 1] += w * (cy - soft[j, 1])
                counts[j] += w
        for j in range(st.n_soft):
            if counts[j] > 0:
                soft[j, 0] += att * disp[j, 0] / counts[j]
                soft[j, 1] += att * disp[j, 1] / counts[j]
                soft[j, 0] = np.clip(soft[j, 0], st.soft_half[j, 0], st.cw - st.soft_half[j, 0])
                soft[j, 1] = np.clip(soft[j, 1], st.soft_half[j, 1], st.ch - st.soft_half[j, 1])
    return soft


def _pin_coords(
    hard_pos: np.ndarray,
    soft_pos: np.ndarray,
    port_pos: np.ndarray,
    st: FastProxyState,
) -> Tuple[np.ndarray, np.ndarray]:
    centers = np.vstack([hard_pos, soft_pos, port_pos])
    px = centers[st.pin_owners, 0] + st.pin_ox
    py = centers[st.pin_owners, 1] + st.pin_oy
    return px, py


def _compute_wl(
    hard_pos: np.ndarray,
    soft_pos: np.ndarray,
    port_pos: np.ndarray,
    st: FastProxyState,
) -> float:
    px, py = _pin_coords(hard_pos, soft_pos, port_pos, st)
    total = 0.0
    n_nets = len(st.net_weights)
    for nid in range(n_nets):
        s, e = int(st.net_start[nid]), int(st.net_start[nid + 1])
        if e <= s:
            continue
        xs = px[s:e]
        ys = py[s:e]
        total += float(st.net_weights[nid]) * (
            float(xs.max() - xs.min()) + float(ys.max() - ys.min())
        )
    denom = (st.cw + st.ch) * max(st.net_cnt, 1.0)
    return float(total / denom)


def _overlap_area_block(
    mx: float,
    my: float,
    mw: float,
    mh: float,
    st: FastProxyState,
) -> np.ndarray:
    """Per-bin overlap area of macro rectangle with grid cells (matches plc grid density)."""
    nrow, ncol = st.grid_rows, st.grid_cols
    occupied = np.zeros(nrow * ncol, dtype=np.float64)
    x0, x1 = mx - mw * 0.5, mx + mw * 0.5
    y0, y1 = my - mh * 0.5, my + mh * 0.5
    c0 = max(0, int(math.floor(x0 / st.cell_w)))
    c1 = min(ncol - 1, int(math.floor(x1 / st.cell_w)))
    r0 = max(0, int(math.floor(y0 / st.cell_h)))
    r1 = min(nrow - 1, int(math.floor(y1 / st.cell_h)))
    if x1 < 0 or y1 < 0 or x0 > st.cw or y0 > st.ch:
        return occupied
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            gx0, gy0 = c * st.cell_w, r * st.cell_h
            gx1, gy1 = gx0 + st.cell_w, gy0 + st.cell_h
            ox = max(0.0, min(x1, gx1) - max(x0, gx0))
            oy = max(0.0, min(y1, gy1) - max(y0, gy0))
            if ox > 0 and oy > 0:
                occupied[r * ncol + c] += ox * oy
    return occupied


def _compute_density(
    hard_pos: np.ndarray,
    soft_pos: np.ndarray,
    st: FastProxyState,
    *,
    top_p: float = 0.10,
) -> float:
    nrow, ncol = st.grid_rows, st.grid_cols
    occupied = np.zeros(nrow * ncol, dtype=np.float64)
    nh = st.nh
    for i in range(nh):
        occupied += _overlap_area_block(
            float(hard_pos[i, 0]),
            float(hard_pos[i, 1]),
            float(st.hard_sizes[i, 0]),
            float(st.hard_sizes[i, 1]),
            st,
        )
    for j in range(st.n_soft):
        occupied += _overlap_area_block(
            float(soft_pos[j, 0]),
            float(soft_pos[j, 1]),
            float(st.soft_sizes[j, 0]),
            float(st.soft_sizes[j, 1]),
            st,
        )
    grid_cells = occupied / max(st.grid_area, 1e-30)
    nonzero = grid_cells[grid_cells > 0.0]
    ncells = grid_cells.size
    if ncells < 10:
        if nonzero.size == 0:
            return 0.0
        return 0.5 * float(nonzero.mean())

    density_cnt = int(math.floor(ncells * top_p))
    if density_cnt < 1:
        density_cnt = 1
    if nonzero.size == 0:
        return 0.0
    top = np.sort(nonzero)[::-1][:density_cnt]
    return 0.5 * float(top.mean())


def _route_two_pin(
    source: Tuple[int, int],
    sink: Tuple[int, int],
    weight: float,
    h_acc: np.ndarray,
    v_acc: np.ndarray,
) -> None:
    sr, sc = source
    tr, tc = sink
    row_min, row_max = min(sr, tr), max(sr, tr)
    col_min, col_max = min(sc, tc), max(sc, tc)
    if col_max > col_min:
        h_acc[sr, col_min:col_max] += weight
    if row_max > row_min:
        v_acc[row_min:row_max, tc] += weight


def _route_three_pin(
    gcells: List[Tuple[int, int]],
    weight: float,
    h_acc: np.ndarray,
    v_acc: np.ndarray,
) -> None:
    gcells = sorted(gcells, key=lambda x: (x[1], x[0]))
    y1, x1 = gcells[0]
    y2, x2 = gcells[1]
    y3, x3 = gcells[2]
    if x2 > x1:
        h_acc[y1, x1:x2] += weight
    if x3 > x2:
        h_acc[y2, x2:x3] += weight
    rlo, rhi = min(y1, y2), max(y1, y2)
    if rhi > rlo:
        v_acc[rlo:rhi, x2] += weight
    rlo, rhi = min(y2, y3), max(y2, y3)
    if rhi > rlo:
        v_acc[rlo:rhi, x3] += weight


def _smooth_routing(h: np.ndarray, v: np.ndarray, smooth_range: int) -> Tuple[np.ndarray, np.ndarray]:
    if smooth_range <= 0:
        return h, v
    nrow, ncol = h.shape
    sr = smooth_range
    v_out = np.zeros_like(v)
    h_out = np.zeros_like(h)
    for row in range(nrow):
        for col in range(ncol):
            lp = max(0, col - sr)
            rp = min(ncol - 1, col + sr)
            cnt = rp - lp + 1
            val = v[row, col] / max(cnt, 1)
            v_out[row, lp : rp + 1] += val
    for col in range(ncol):
        for row in range(nrow):
            lp = max(0, row - sr)
            up = min(nrow - 1, row + sr)
            cnt = up - lp + 1
            val = h[row, col] / max(cnt, 1)
            h_out[lp : up + 1, col] += val
    return h_out, v_out


def _abu(xx: np.ndarray, n: float = 0.05) -> float:
    flat = np.sort(xx.ravel())[::-1]
    cnt = int(math.floor(len(flat) * n))
    if cnt == 0:
        return float(flat[0]) if flat.size else 0.0
    return float(flat[:cnt].mean())


def _compute_congestion(
    hard_pos: np.ndarray,
    soft_pos: np.ndarray,
    port_pos: np.ndarray,
    st: FastProxyState,
    *,
    top_p: float = 0.05,
) -> float:
    nrow, ncol = st.grid_rows, st.grid_cols
    h_raw = np.zeros((nrow, ncol), dtype=np.float64)
    v_raw = np.zeros((nrow, ncol), dtype=np.float64)
    px, py = _pin_coords(hard_pos, soft_pos, port_pos, st)
    inv_cw = ncol / max(st.cw, 1e-30)
    inv_ch = nrow / max(st.ch, 1e-30)

    n_nets = len(st.net_weights)
    for nid in range(n_nets):
        s, e = int(st.net_start[nid]), int(st.net_start[nid + 1])
        if e - s < 2:
            continue
        cols = np.clip((px[s:e] * inv_cw).astype(np.int32), 0, ncol - 1)
        rows = np.clip((py[s:e] * inv_ch).astype(np.int32), 0, nrow - 1)
        w = float(st.net_weights[nid])
        sr, sc = int(rows[0]), int(cols[0])
        if e - s == 2:
            _route_two_pin((sr, sc), (int(rows[1]), int(cols[1])), w, h_raw, v_raw)
            continue
        seen: Dict[Tuple[int, int], None] = {}
        for r_i, c_i in zip(rows.tolist(), cols.tolist()):
            seen[(r_i, c_i)] = None
        gcells = list(seen.keys())
        if len(gcells) < 2:
            continue
        if len(gcells) == 2:
            other = gcells[1] if gcells[0] == (sr, sc) else gcells[0]
            _route_two_pin((sr, sc), other, w, h_raw, v_raw)
        elif len(gcells) == 3:
            _route_three_pin(gcells, w, h_raw, v_raw)
        else:
            for g in gcells:
                if g != (sr, sc):
                    _route_two_pin((sr, sc), g, w, h_raw, v_raw)

    h_norm = h_raw / max(st.grid_h_cap, 1e-30)
    v_norm = v_raw / max(st.grid_v_cap, 1e-30)
    h_norm, v_norm = _smooth_routing(h_norm, v_norm, st.smooth_range)
    combined = np.concatenate([h_norm.ravel(), v_norm.ravel()])
    return _abu(combined, top_p)


def fast_proxy(
    hard_pos: np.ndarray,
    soft_pos: np.ndarray,
    port_pos: np.ndarray,
    net_pin_nodes: list,
    net_weights: np.ndarray,
    sizes_np: np.ndarray,
    half_w: np.ndarray,
    half_h: np.ndarray,
    cw: float,
    ch: float,
    grid_rows: int,
    grid_cols: int,
    h_routes_per_micron: float,
    v_routes_per_micron: float,
    *,
    fd_iters: int = 0,
    top_p: float = 0.10,
    state: Optional[FastProxyState] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Approximate proxy cost without the plc client.

    Returns ``(proxy_cost, components_dict)`` with keys matching ``compute_proxy_cost``.
    """
    if state is None:
        raise ValueError("fast_proxy requires a prebuilt FastProxyState (use build_fast_proxy_state)")

    soft = _fast_fd(hard_pos, soft_pos, port_pos, state, fd_iters=fd_iters)
    wl = _compute_wl(hard_pos, soft, port_pos, state)
    density = _compute_density(hard_pos, soft, state, top_p=top_p)
    congestion = _compute_congestion(hard_pos, soft, port_pos, state, top_p=0.05)
    # density_cost already includes plc's 0.5 factor on top-10% bins
    proxy = wl + density + 0.5 * congestion
    return proxy, {
        "wirelength_cost": wl,
        "density_cost": density,
        "congestion_cost": congestion,
        "proxy_cost": proxy,
        "overlap_count": 0,
    }


def fast_proxy_from_benchmark(
    benchmark: Benchmark,
    hard_pos: np.ndarray,
    *,
    plc=None,
    state: Optional[FastProxyState] = None,
    fd_iters: int = 0,
    top_p: float = 0.10,
) -> Tuple[float, Dict[str, float]]:
    """Convenience wrapper using benchmark tensors."""
    if state is None:
        state = build_fast_proxy_state(benchmark, plc)
    nh = state.nh
    soft_pos = benchmark.macro_positions[nh : nh + state.n_soft].numpy().astype(np.float64)
    port_pos = benchmark.port_positions.numpy().astype(np.float64)
    sizes = benchmark.macro_sizes[:nh].numpy().astype(np.float64)
    return fast_proxy(
        hard_pos.astype(np.float64),
        soft_pos,
        port_pos,
        state.net_pin_nodes,
        state.net_weights,
        sizes,
        sizes[:, 0] * 0.5,
        sizes[:, 1] * 0.5,
        state.cw,
        state.ch,
        state.grid_rows,
        state.grid_cols,
        state.hroutes_per_micron,
        state.vroutes_per_micron,
        fd_iters=fd_iters,
        top_p=top_p,
        state=state,
    )


def _calibrate_ibm07(n_samples: int = 200, fd_iters: int = 0) -> None:
    from pathlib import Path

    import torch

    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import _set_placement, compute_proxy_cost

    root = (
        Path(__file__).resolve().parent.parent
        / "external"
        / "MacroPlacement"
        / "Testcases"
        / "ICCAD04"
        / "ibm07"
    )

    benchmark, plc = load_benchmark_from_dir(str(root))
    state = build_fast_proxy_state(benchmark, plc)
    nh = benchmark.num_hard_macros
    rng = np.random.default_rng(42)
    cw, ch = state.cw, state.ch
    half_w = benchmark.macro_sizes[:nh, 0].numpy() * 0.5
    half_h = benchmark.macro_sizes[:nh, 1].numpy() * 0.5
    init = benchmark.macro_positions[:nh].numpy().astype(np.float64)

    fast_vals: List[float] = []
    oracle_vals: List[float] = []
    t_fast = 0.0
    t_oracle = 0.0

    for _ in range(n_samples):
        pos = init.copy()
        jitter = 0.12 * max(cw, ch)
        pos[:, 0] += rng.uniform(-jitter, jitter, nh)
        pos[:, 1] += rng.uniform(-jitter, jitter, nh)
        pos[:, 0] = np.clip(pos[:, 0], half_w, cw - half_w)
        pos[:, 1] = np.clip(pos[:, 1], half_h, ch - half_h)

        t0 = time.perf_counter()
        fp, _ = fast_proxy_from_benchmark(benchmark, pos, plc=plc, state=state, fd_iters=fd_iters)
        t_fast += time.perf_counter() - t0

        ft = benchmark.macro_positions.clone()
        ft[:nh] = torch.tensor(pos, dtype=torch.float32)
        t0 = time.perf_counter()
        _set_placement(plc, ft, benchmark)
        oc = compute_proxy_cost(ft, benchmark, plc)["proxy_cost"]
        t_oracle += time.perf_counter() - t0

        fast_vals.append(fp)
        oracle_vals.append(float(oc))

    fa = np.asarray(fast_vals)
    oa = np.asarray(oracle_vals)
    r = float(np.corrcoef(fa, oa)[0, 1])
    print(f"ibm07 calibration n={n_samples} fd_iters={fd_iters}")
    print(f"  Pearson r = {r:.4f}")
    print(f"  fast_proxy  mean={fa.mean():.4f}  std={fa.std():.4f}")
    print(f"  oracle      mean={oa.mean():.4f}  std={oa.std():.4f}")
    ms_fast = 1e3 * t_fast / n_samples
    print(f"  fast ms/call = {ms_fast:.2f}")
    print(f"  oracle ms/call = {1e3 * t_oracle / n_samples:.2f}")
    ok_r = r >= 0.85
    ok_ms = ms_fast <= 50.0
    print(f"  target r>=0.85: {'PASS' if ok_r else 'FAIL'}  target fast<=50ms: {'PASS' if ok_ms else 'FAIL (ok for SA if << oracle)'}")


if __name__ == "__main__":
    from pathlib import Path

    n = int(os.environ.get("MACRO_PLACER_FAST_PROXY_SAMPLES", "200"))
    fd = int(os.environ.get("MACRO_PLACER_FAST_PROXY_FD_ITERS", "0"))
    _calibrate_ibm07(n_samples=n, fd_iters=fd)
