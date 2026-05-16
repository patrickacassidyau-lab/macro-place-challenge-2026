"""Topology-aware seeds, cluster hierarchy, and orientation helpers."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from macro_place.placement_global import macro_clusters_from_nets


def klein4_transform_offset(dx: float, dy: float, orient: int) -> tuple[float, float]:
    """Klein-4 pin-offset transforms (N, FN, FS, S)."""
    o = int(orient) % 4
    if o == 0:
        return dx, dy
    if o == 1:
        return -dx, dy
    if o == 2:
        return dx, -dy
    return -dx, -dy


def quarter_transform_offset(dx: float, dy: float, orient: int) -> tuple[float, float]:
    """0/90/180/270° pin-offset rotation (research / non-Tier-2)."""
    o = int(orient) % 4
    if o == 0:
        return dx, dy
    if o == 1:
        return -dy, dx
    if o == 2:
        return -dx, -dy
    return dy, -dx


def effective_macro_halves(
    sizes_np: np.ndarray,
    orient_q: np.ndarray,
    macro_idx: int,
    *,
    allow_quarter: bool,
) -> tuple[float, float]:
    w = float(sizes_np[macro_idx, 0])
    h = float(sizes_np[macro_idx, 1])
    if allow_quarter and int(orient_q[macro_idx]) % 2 == 1:
        w, h = h, w
    return w / 2.0, h / 2.0


def pin_offset_at(
    offsets_np: Sequence[np.ndarray],
    owner: int,
    slot: int,
    orient_q: np.ndarray,
    *,
    allow_quarter: bool,
) -> tuple[float, float]:
    ofs = offsets_np[owner]
    if ofs.shape[0] == 0 or slot >= ofs.shape[0]:
        return 0.0, 0.0
    dx, dy = float(ofs[slot, 0]), float(ofs[slot, 1])
    o = int(orient_q[owner])
    if allow_quarter:
        return quarter_transform_offset(dx, dy, o)
    return klein4_transform_offset(dx, dy, o)


def topological_macro_seed(
    pos: np.ndarray,
    *,
    n_hard: int,
    movable_idx: np.ndarray,
    movable_mask: np.ndarray,
    macro_to_nets: Sequence[Sequence[int]],
    net_pin_nodes: Sequence,
    num_macros: int,
    sizes_np: np.ndarray,
    cw: float,
    ch: float,
) -> np.ndarray:
    """
    Place movable macros along upstream→downstream flow from a netlist DAG heuristic.
    """
    out = pos.copy()
    if len(movable_idx) == 0:
        return out

    adj: list[set[int]] = [set() for _ in range(n_hard)]
    indeg = np.zeros(n_hard, dtype=np.int64)
    for nid, pins in enumerate(net_pin_nodes):
        if pins is None or len(pins) == 0:
            continue
        pn = pins.numpy() if hasattr(pins, "numpy") else np.asarray(pins)
        hard = [int(r[0]) for r in pn if int(r[0]) >= 0 and int(r[0]) < n_hard]
        ports = [int(r[0]) for r in pn if int(r[0]) >= num_macros]
        if not hard:
            continue
        driver = max(hard, key=lambda m: len(macro_to_nets[m]))
        for m in hard:
            if m == driver:
                continue
            if m not in adj[driver]:
                adj[driver].add(m)
                indeg[m] += 1
        if ports:
            for m in hard:
                if m != driver:
                    continue
                for p in ports:
                    _ = p

    seeds = [int(i) for i in np.where((indeg == 0) & movable_mask)[0]]
    if not seeds:
        seeds = [int(movable_idx[np.argmax([len(macro_to_nets[i]) for i in movable_idx])])]
    order: list[int] = []
    q = seeds[:]
    seen = set()
    while q:
        u = int(q.pop(0))
        if u in seen or not movable_mask[u]:
            continue
        seen.add(u)
        order.append(u)
        for v in sorted(adj[u]):
            indeg[v] -= 1
            if indeg[v] <= 0:
                q.append(v)
    for mi in movable_idx:
        mi = int(mi)
        if mi not in seen:
            order.append(mi)

    span_x = max(cw * 0.82, 1.0)
    x0 = cw * 0.09
    y_mid = ch * 0.5
    n = max(len(order), 1)
    for rank, mi in enumerate(order):
        hw = float(sizes_np[mi, 0]) / 2.0
        hh = float(sizes_np[mi, 1]) / 2.0
        frac = rank / max(n - 1, 1)
        out[mi, 0] = np.clip(x0 + frac * span_x, hw, cw - hw)
        lane = 0.12 * ch * math.sin(4.0 * math.pi * frac)
        out[mi, 1] = np.clip(y_mid + lane, hh, ch - hh)
    return out


def multilevel_cluster_seed(
    pos: np.ndarray,
    *,
    n_hard: int,
    movable_idx: np.ndarray,
    movable_mask: np.ndarray,
    macro_to_nets: Sequence[Sequence[int]],
    sizes_np: np.ndarray,
    cw: float,
    ch: float,
) -> np.ndarray:
    """Cluster → global cluster sites → local macro offsets."""
    out = pos.copy()
    cluster_id = macro_clusters_from_nets(n_hard, macro_to_nets, movable_mask)
    uniq = np.unique(cluster_id)
    n_clusters = max(int(uniq.size), 1)
    cols = max(1, int(math.ceil(math.sqrt(n_clusters))))
    rows = max(1, int(math.ceil(n_clusters / cols)))
    cell_w = cw / cols
    cell_h = ch / rows
    cluster_rank = {int(c): i for i, c in enumerate(uniq.tolist())}

    for cid in uniq:
        cid = int(cid)
        members = np.where(cluster_id == cid)[0]
        if members.size == 0:
            continue
        rank = cluster_rank[cid]
        gr = rank // cols
        gc = rank % cols
        cx = (gc + 0.5) * cell_w
        cy = (gr + 0.5) * cell_h
        members_m = [int(m) for m in members if movable_mask[m]]
        members_m.sort(key=lambda m: -float(sizes_np[m, 0] * sizes_np[m, 1]))
        for j, mi in enumerate(members_m):
            hw = float(sizes_np[mi, 0]) / 2.0
            hh = float(sizes_np[mi, 1]) / 2.0
            ox = (j % 3 - 1) * cell_w * 0.11
            oy = (j // 3) * cell_h * 0.11
            out[mi, 0] = np.clip(cx + ox, hw, cw - hw)
            out[mi, 1] = np.clip(cy + oy, hh, ch - hh)
    return out


def apply_macro_orientations_to_plc(
    plc,
    benchmark,
    orient_q: np.ndarray,
    offsets_np: Sequence[np.ndarray],
    *,
    allow_quarter: bool,
) -> None:
    """Push Klein-4 / quarter orientations into PlacementCost pin geometry."""
    if plc is None:
        return
    if not hasattr(plc, "_macro_pin_map"):
        pin_map: dict[str, list[int]] = {}
        for idx, mod in enumerate(plc.modules_w_pins):
            if mod.get_type() == "MACRO_PIN" and hasattr(mod, "get_macro_name"):
                name = mod.get_macro_name()
                pin_map.setdefault(name, []).append(idx)
        plc._macro_pin_map = pin_map

    n_hard = benchmark.num_hard_macros
    for i, macro_idx in enumerate(benchmark.hard_macro_indices):
        if i >= n_hard:
            break
        node = plc.modules_w_pins[macro_idx]
        x, y = node.get_pos()
        name = node.get_name()
        for pin_idx in plc._macro_pin_map.get(name, []):
            pin = plc.modules_w_pins[pin_idx]
            dx = float(getattr(pin, "x_offset", 0.0))
            dy = float(getattr(pin, "y_offset", 0.0))
            if allow_quarter:
                tdx, tdy = quarter_transform_offset(dx, dy, int(orient_q[i]))
            else:
                tdx, tdy = klein4_transform_offset(dx, dy, int(orient_q[i]))
            pin.set_pos(x + tdx, y + tdy)
