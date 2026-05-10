"""
Cheap routing-feasibility surrogates inspired by global-placement literature:
MaskPlace-style dense congestion cues from pin geometry (RUDY family) and
capacity-aware **anisotropic** wire demand (matches ICCAD04 .plc H/V tracks/µm).

**Multi-pole hotspots** approximate a *multi-modal* congestion landscape on the PlacementCost
router grid — analogous in spirit to continuous density potentials in analytic global placers
(ePlace-style Poisson solves), but discretised into **spatially-separated** high ``max(H,V)``
tiles for evacuation forces (tier‑1–aligned novelty vs a single centroid).

Not a replacement for PlacementCost.get_congestion_cost() — a *differentiable-in-spirit*
signal for search (SA) and for backtest diagnostics vs true congestion.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from macro_place.benchmark import Benchmark


def net_hpwl_xy_parts(xs: np.ndarray, ys: np.ndarray) -> Tuple[float, float]:
    if xs.size == 0:
        return 0.0, 0.0
    return float(xs.max() - xs.min()), float(ys.max() - ys.min())


def net_capacity_imbalance_contrib(
    xs: np.ndarray,
    ys: np.ndarray,
    net_w: float,
    canvas_w: float,
    canvas_h: float,
    hroutes_per_um: float,
    vroutes_per_um: float,
) -> float:
    """
    Penalize mismatch between horizontal vs vertical HPWL **demand** relative to
    available routing **capacity** (tracks scale ∝ canvas edge × tracks/µm).

    Minimizing this tends to spread macro pin BB-stretch across H/V routing in
    proportion to asymmetric track supply — analogous to congestion-aware globalization
    in RePlAce / NTUPlace class tools, cheap enough for incremental SA evaluation.
    """
    wl_x, wl_y = net_hpwl_xy_parts(xs, ys)
    h_budget = canvas_w * max(hroutes_per_um, 1e-12)
    v_budget = canvas_h * max(vroutes_per_um, 1e-12)
    dx = wl_x / h_budget
    dy = wl_y / v_budget
    return float(abs(dx - dy) * net_w)


def compute_total_routing_imbalance(
    benchmark: Benchmark,
    pos_hard: np.ndarray,
    net_weights: Optional[np.ndarray] = None,
) -> float:
    """Full O(nets×pins) sum — used in backtest diagnostics only."""
    nh = benchmark.num_hard_macros
    num_macros = benchmark.num_macros
    soft_xy = benchmark.macro_positions[nh:num_macros].numpy().astype(np.float64)
    ports_xy = benchmark.port_positions.numpy().astype(np.float64)
    offsets = [
        benchmark.macro_pin_offsets[i].numpy().astype(np.float64)
        if benchmark.macro_pin_offsets[i].numel()
        else np.zeros((0, 2), dtype=np.float64)
        for i in range(nh)
    ]

    def pin_xy(owner: int, slot: int) -> Tuple[float, float]:
        if owner < nh:
            ox, oy = float(pos_hard[owner, 0]), float(pos_hard[owner, 1])
            ofs = offsets[owner]
            if ofs.shape[0] > 0 and slot < ofs.shape[0]:
                return float(ox + ofs[slot, 0]), float(oy + ofs[slot, 1])
            return ox, oy
        if owner < num_macros:
            j = owner - nh
            return float(soft_xy[j, 0]), float(soft_xy[j, 1])
        pidx = owner - num_macros
        return float(ports_xy[pidx, 0]), float(ports_xy[pidx, 1])

    nw_arr = benchmark.net_weights.numpy().astype(np.float64)
    if net_weights is not None:
        nw_arr = net_weights.astype(np.float64)

    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    hr = float(benchmark.hroutes_per_micron)
    vr = float(benchmark.vroutes_per_micron)

    tot = 0.0
    for nid in range(benchmark.num_nets):
        pn = benchmark.net_pin_nodes[nid].numpy()
        if pn.shape[0] == 0:
            continue
        xs = np.empty(pn.shape[0], dtype=np.float64)
        ys = np.empty(pn.shape[0], dtype=np.float64)
        for r in range(pn.shape[0]):
            o, s = int(pn[r, 0]), int(pn[r, 1])
            xs[r], ys[r] = pin_xy(o, s)
        tot += net_capacity_imbalance_contrib(xs, ys, float(nw_arr[nid]), cw, ch, hr, vr)
    return tot


def congestion_hotspot_centroid_um(
    plc,
    benchmark: Benchmark,
    *,
    percentile: float = 82.0,
) -> Tuple[float, float, float]:
    """
    Weighted geometric centroid of **hot** routing cells (max of H/V congestion),
    in microns. Used to push macros **radially away** from PlacementCost-identified
    hotspots — directly targets the Tier‑1 congestion term (RePlAce-style
    congestion-driven spreading), unlike cheap SA surrogates.
    """
    try:
        plc.get_congestion_cost()
    except Exception:
        return float(benchmark.canvas_width) * 0.5, float(benchmark.canvas_height) * 0.5, 0.0
    nrow = int(getattr(plc, "grid_row", benchmark.grid_rows))
    ncol = int(getattr(plc, "grid_col", benchmark.grid_cols))
    exp = nrow * ncol
    h_flat = np.asarray(plc.H_routing_cong, dtype=np.float64).ravel()
    v_flat = np.asarray(plc.V_routing_cong, dtype=np.float64).ravel()
    if h_flat.size < exp or v_flat.size < exp:
        return float(benchmark.canvas_width) * 0.5, float(benchmark.canvas_height) * 0.5, 0.0
    w = np.maximum(h_flat[:exp], v_flat[:exp]).reshape(nrow, ncol)
    w = np.maximum(w, 0.0)
    inside = w.ravel()
    thr = float(np.percentile(inside, min(99.0, max(0.0, percentile)))) if inside.size else 0.0
    mask = np.maximum(w - thr, 0.0)
    mass = float(mask.sum())
    if mass < 1e-18:
        cx = float(benchmark.canvas_width) * 0.5
        cy = float(benchmark.canvas_height) * 0.5
        return cx, cy, 0.0
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    cell_w = cw / max(ncol, 1)
    cell_h = ch / max(nrow, 1)
    jj, ii = np.meshgrid(np.arange(ncol, dtype=np.float64), np.arange(nrow, dtype=np.float64))
    xc = (jj + 0.5) * cell_w
    yc = (ii + 0.5) * cell_h
    cx = float((mask * xc).sum() / mass)
    cy = float((mask * yc).sum() / mass)
    return cx, cy, mass


def multipole_congestion_sites_um(
    plc,
    benchmark: Benchmark,
    *,
    k: int = 12,
    min_sep_cells: int = 2,
) -> np.ndarray:
    """
    Return ``[Ks, 2]`` µm‑coordinates of spatially-separated high-congestion **poles**
    from ``max(H_routing_cong, V_routing_cong)`` after ``plc.get_congestion_cost()``:
    greedy scan in descending hotspot strength with Chebyshev separation ≥ ``min_sep_cells``.
    Empty array if grids are unavailable.

    Enables **multi-pole repulsion**: superpose escape directions from several congestion
    modes simultaneously (captures dispersed hotspots poorly approximated by one centroid).
    """
    try:
        plc.get_congestion_cost()
    except Exception:
        return np.zeros((0, 2), dtype=np.float64)

    nrow = int(getattr(plc, "grid_row", benchmark.grid_rows))
    ncol = int(getattr(plc, "grid_col", benchmark.grid_cols))
    exp = nrow * ncol
    h_flat = np.asarray(plc.H_routing_cong, dtype=np.float64).ravel()
    v_flat = np.asarray(plc.V_routing_cong, dtype=np.float64).ravel()
    if h_flat.size < exp or v_flat.size < exp:
        return np.zeros((0, 2), dtype=np.float64)

    w = np.maximum(h_flat[:exp], v_flat[:exp])
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    cell_w = cw / max(ncol, 1)
    cell_h = ch / max(nrow, 1)

    order = np.argsort(-w)
    sep = max(1, min_sep_cells)
    picked_r: list[int] = []
    picked_c: list[int] = []
    for ix in order:
        r_i, c_i = divmod(int(ix), ncol)
        ok = True
        for rr, cc in zip(picked_r, picked_c):
            if max(abs(r_i - rr), abs(c_i - cc)) < sep:
                ok = False
                break
        if not ok:
            continue
        picked_r.append(r_i)
        picked_c.append(c_i)
        if len(picked_r) >= max(1, min(k, 64)):
            break

    if not picked_r:
        return np.zeros((0, 2), dtype=np.float64)

    xs = [(cc + 0.5) * cell_w for cc in picked_c]
    ys = [(rr + 0.5) * cell_h for rr in picked_r]
    return np.stack([np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)], axis=1)


def plc_congestion_headroom(plc, benchmark) -> Tuple[float, float, float]:
    """
    Aggregate H/V congestion heatmaps from PlacementCost (after get_congestion_cost).
    Returns (mean_h, mean_v, mean_maxHV) for correlation with surrogates.
    """
    try:
        plc.get_congestion_cost()
        nrow, ncol = int(benchmark.grid_rows), int(benchmark.grid_cols)
        h_flat = np.asarray(plc.H_routing_cong, dtype=np.float64).ravel()
        v_flat = np.asarray(plc.V_routing_cong, dtype=np.float64).ravel()
        exp = nrow * ncol
        if h_flat.size < exp:
            return 0.0, 0.0, 0.0
        h_flat = h_flat[:exp].reshape(nrow, ncol)
        v_flat = v_flat[:exp].reshape(nrow, ncol)
        mx = np.maximum(h_flat, v_flat)
        return float(np.mean(h_flat)), float(np.mean(v_flat)), float(np.mean(mx))
    except Exception:
        return 0.0, 0.0, 0.0


def plc_density_mean(plc, benchmark) -> float:
    try:
        plc.get_density_cost()
        nrow, ncol = int(benchmark.grid_rows), int(benchmark.grid_cols)
        d = np.asarray(plc.grid_cells, dtype=np.float64).ravel()
        exp = nrow * ncol
        if d.size < exp:
            return 0.0
        return float(np.mean(d[:exp]))
    except Exception:
        return 0.0


def pin_splat_rudy_peak(
    benchmark: Benchmark,
    pos_hard: np.ndarray,
    grid_g: int = 40,
    net_weights: Optional[np.ndarray] = None,
) -> float:
    """
    O(total_pins): splat normalized H/V routing pressure at each pin location
    (MaskPlace / FastPlace-style cheap routing demand cartoon; not full global RUDY).
    """
    nh = benchmark.num_hard_macros
    g = max(8, min(72, grid_g))
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    hr = float(benchmark.hroutes_per_micron)
    vr = float(benchmark.vroutes_per_micron)

    num_macros = benchmark.num_macros
    soft_xy = benchmark.macro_positions[nh:num_macros].numpy().astype(np.float64)
    ports_xy = benchmark.port_positions.numpy().astype(np.float64)
    offsets = [
        benchmark.macro_pin_offsets[i].numpy().astype(np.float64)
        if benchmark.macro_pin_offsets[i].numel()
        else np.zeros((0, 2), dtype=np.float64)
        for i in range(nh)
    ]

    def pin_xy(owner: int, slot: int) -> Tuple[float, float]:
        if owner < nh:
            ox, oy = float(pos_hard[owner, 0]), float(pos_hard[owner, 1])
            ofs = offsets[owner]
            if ofs.shape[0] > 0 and slot < ofs.shape[0]:
                return float(ox + ofs[slot, 0]), float(oy + ofs[slot, 1])
            return ox, oy
        if owner < num_macros:
            j = owner - nh
            return float(soft_xy[j, 0]), float(soft_xy[j, 1])
        pidx = owner - num_macros
        return float(ports_xy[pidx, 0]), float(ports_xy[pidx, 1])

    nw_arr = benchmark.net_weights.numpy().astype(np.float64)
    if net_weights is not None:
        nw_arr = net_weights.astype(np.float64)

    H = np.zeros((g, g), dtype=np.float64)
    V = np.zeros((g, g), dtype=np.float64)

    for nid in range(benchmark.num_nets):
        pn = benchmark.net_pin_nodes[nid].numpy()
        if pn.shape[0] == 0:
            continue
        xs = np.empty(pn.shape[0], dtype=np.float64)
        ys = np.empty(pn.shape[0], dtype=np.float64)
        for r in range(pn.shape[0]):
            o, s = int(pn[r, 0]), int(pn[r, 1])
            xs[r], ys[r] = pin_xy(o, s)
        wl_x, wl_y = net_hpwl_xy_parts(xs, ys)
        wn = float(nw_arr[nid])
        if wl_x + wl_y <= 0:
            continue
        ux = wl_x / (wl_x + wl_y)
        uy = wl_y / (wl_x + wl_y)
        cap_den_h = cw * hr + 1e-9
        cap_den_v = ch * vr + 1e-9
        for r in range(pn.shape[0]):
            ix = int(np.clip(np.floor(xs[r] / cw * g), 0, g - 1))
            iy = int(np.clip(np.floor(ys[r] / ch * g), 0, g - 1))
            H[ix, iy] += wn * ux / cap_den_h
            V[ix, iy] += wn * uy / cap_den_v

    peak = np.maximum(H, V)
    k = max(4, (g * g) // 20)
    flat = np.sort(peak.ravel())
    return float(np.mean(flat[-k:]))


def macro_net_bbox_hotspot_scores(
    plc,
    benchmark: Benchmark,
    pos_hard: np.ndarray,
    *,
    hot_percentile: float = 86.0,
    use_excess_mass: bool = True,
) -> np.ndarray:
    """
    **Net–routing × congestion coupling** (generalizable beyond centroid / tile occupancy):

    For each net, form the pin bounding box in µm, map it onto the PlacementCost global-routing
    congestion grid (``max(H,V)`` after ``get_congestion_cost()``), and accumulate congestion
    mass falling inside that bbox — weighted by the net's Tier‑1 weight. Every **hard macro**
    incident on the net receives that accumulation.

    This targets macros whose *connections traverse* congested regions (Executive Summary:
    congestion estimation + rip‑up / reroute intuition), complementing picking macros whose
    **centers** lie in hot bins.

    Args:
        plc: PlacementCost instance (same object used by Tier‑1 proxy).
        pos_hard: ``[num_hard, 2]`` macro centers in µm.
        hot_percentile: Threshold on ``max(H,V)`` for excess-mass mode (ignored if
            ``use_excess_mass`` is False).
        use_excess_mass: If True, integrate ``max(0, wg - τ)`` over the bbox with τ a grid
            percentile; else integrate raw ``wg``.

    Returns:
        ``scores`` shaped ``[num_hard_macros]``, nonnegative (unnormalized weights).
    """
    try:
        plc.get_congestion_cost()
    except Exception:
        return np.zeros(benchmark.num_hard_macros, dtype=np.float64)

    nh = benchmark.num_hard_macros
    scores = np.zeros(nh, dtype=np.float64)
    nrow = int(benchmark.grid_rows)
    ncol = int(benchmark.grid_cols)
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    cell_w = cw / max(ncol, 1)
    cell_h = ch / max(nrow, 1)
    exp = nrow * ncol
    h_flat = np.asarray(plc.H_routing_cong, dtype=np.float64).ravel()
    v_flat = np.asarray(plc.V_routing_cong, dtype=np.float64).ravel()
    if h_flat.size < exp or v_flat.size < exp:
        return scores

    wg = np.maximum(h_flat[:exp], v_flat[:exp]).reshape(nrow, ncol)
    if use_excess_mass:
        thr = float(np.percentile(wg.ravel(), min(99.5, max(0.0, hot_percentile))))
        wg_use = np.maximum(wg - thr, 0.0)
    else:
        wg_use = wg

    num_macros = benchmark.num_macros
    soft_xy = benchmark.macro_positions[nh:num_macros].numpy().astype(np.float64)
    ports_xy = benchmark.port_positions.numpy().astype(np.float64)
    offsets = [
        benchmark.macro_pin_offsets[i].numpy().astype(np.float64)
        if benchmark.macro_pin_offsets[i].numel()
        else np.zeros((0, 2), dtype=np.float64)
        for i in range(nh)
    ]

    def pin_xy(owner: int, slot: int) -> Tuple[float, float]:
        if owner < nh:
            ox, oy = float(pos_hard[owner, 0]), float(pos_hard[owner, 1])
            ofs = offsets[owner]
            if ofs.shape[0] > 0 and slot < ofs.shape[0]:
                return float(ox + ofs[slot, 0]), float(oy + ofs[slot, 1])
            return ox, oy
        if owner < num_macros:
            j = owner - nh
            return float(soft_xy[j, 0]), float(soft_xy[j, 1])
        pidx = owner - num_macros
        return float(ports_xy[pidx, 0]), float(ports_xy[pidx, 1])

    nw_arr = benchmark.net_weights.numpy().astype(np.float64)

    for nid in range(benchmark.num_nets):
        pn = benchmark.net_pin_nodes[nid].numpy()
        if pn.shape[0] == 0:
            continue
        xs = np.empty(pn.shape[0], dtype=np.float64)
        ys = np.empty(pn.shape[0], dtype=np.float64)
        for r in range(pn.shape[0]):
            o, s = int(pn[r, 0]), int(pn[r, 1])
            xs[r], ys[r] = pin_xy(o, s)
        xmin, xmax = float(xs.min()), float(xs.max())
        ymin, ymax = float(ys.min()), float(ys.max())
        # Degenerate bbox (single pin / collapsed): inflate slightly so a tile is covered.
        if xmax - xmin < cell_w * 0.08:
            xmin -= cell_w * 0.35
            xmax += cell_w * 0.35
        if ymax - ymin < cell_h * 0.08:
            ymin -= cell_h * 0.35
            ymax += cell_h * 0.35

        c0 = int(np.clip(np.floor(xmin / cell_w), 0, ncol - 1))
        c1 = int(np.clip(np.floor(xmax / cell_w), 0, ncol - 1))
        r0 = int(np.clip(np.floor(ymin / cell_h), 0, nrow - 1))
        r1 = int(np.clip(np.floor(ymax / cell_h), 0, nrow - 1))
        if r0 > r1:
            r0, r1 = r1, r0
        if c0 > c1:
            c0, c1 = c1, c0

        block = wg_use[r0 : r1 + 1, c0 : c1 + 1]
        if block.size == 0:
            continue
        mass = float(np.mean(block)) * float(nw_arr[nid])

        owners = pn[(pn[:, 0] >= 0) & (pn[:, 0] < nh)][:, 0].astype(np.int64)
        if owners.size == 0:
            continue
        for om in np.unique(owners):
            scores[int(om)] += mass

    return scores
