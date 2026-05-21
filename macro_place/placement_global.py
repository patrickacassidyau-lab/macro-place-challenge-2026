"""Analytical global placement helpers (ePlace/DREAMPlace-style spreading + sequence pairs)."""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np


def macro_clusters_from_nets(
    n_hard: int,
    macro_to_nets: Sequence[Sequence[int]],
    movable_mask: np.ndarray,
    *,
    min_shared_nets: int = 2,
) -> np.ndarray:
    """Union-find clusters from shared nets (heavy-net macro grouping)."""
    parent = np.arange(n_hard, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for mi in range(n_hard):
        nets_i = macro_to_nets[mi]
        if len(nets_i) < min_shared_nets:
            continue
        nets_set = set(nets_i)
        for mj in range(mi + 1, n_hard):
            if not movable_mask[mi] or not movable_mask[mj]:
                continue
            if len(nets_set.intersection(macro_to_nets[mj])) >= min_shared_nets:
                union(mi, mj)

    labels = np.zeros(n_hard, dtype=np.int64)
    for i in range(n_hard):
        labels[i] = find(i)
    _, inv = np.unique(labels, return_inverse=True)
    return inv.astype(np.int64)


def density_violation_frac(hist: np.ndarray, target: float) -> float:
    """Share of bin area above the uniform target (0 = no violation)."""
    if target <= 0:
        return 1.0
    excess = np.maximum(0.0, hist - target)
    return float(excess.sum() / max(float(hist.sum()), 1e-9))


def density_violation_frac_bins(demand: np.ndarray, bin_cap: float) -> float:
    """Share of bin demand above uniform per-bin capacity (RePlAce-style overflow)."""
    if bin_cap <= 0:
        return 1.0
    excess = np.maximum(0.0, demand - bin_cap)
    return float(excess.sum() / max(float(demand.sum()), 1e-9))


def _per_bin_density_force(
    pos: np.ndarray,
    half_w: np.ndarray,
    half_h: np.ndarray,
    cw: float,
    ch: float,
    grid_g: int,
    alpha: float,
    delta_i: np.ndarray,
    *,
    overflow_delta_beta: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    RePlAce §III-B local density: demand grid, nu_j forces, and persistent Delta_i (single pass).
    """
    grid_g = max(1, int(grid_g))
    cell_w = float(cw) / grid_g
    cell_h = float(ch) / grid_g
    bin_cap = cell_w * cell_h
    demand = np.zeros((grid_g, grid_g), dtype=np.float64)
    grad = np.zeros_like(pos)
    total_area = max(float(np.sum(half_w * half_h * 4.0)), 1e-9)
    beta_od = float(overflow_delta_beta)
    n = pos.shape[0]

    for i in range(n):
        xl = float(pos[i, 0]) - float(half_w[i])
        xr = float(pos[i, 0]) + float(half_w[i])
        yb = float(pos[i, 1]) - float(half_h[i])
        yt = float(pos[i, 1]) + float(half_h[i])
        gxl = max(0, int(xl / cell_w))
        gxr = min(grid_g - 1, int(xr / cell_w))
        gyb = max(0, int(yb / cell_h))
        gyt = min(grid_g - 1, int(yt / cell_h))
        for gx in range(gxl, gxr + 1):
            bx0 = gx * cell_w
            bx1 = (gx + 1) * cell_w
            cx = (gx + 0.5) * cell_w
            for gy in range(gyb, gyt + 1):
                by0 = gy * cell_h
                by1 = (gy + 1) * cell_h
                ox = min(xr, bx1) - max(xl, bx0)
                oy = min(yt, by1) - max(yb, by0)
                if ox > 0 and oy > 0:
                    demand[gx, gy] += ox * oy

    overflow = demand - bin_cap
    nu = np.exp(np.clip(alpha * overflow, -20.0, 20.0))

    for i in range(n):
        xl = float(pos[i, 0]) - float(half_w[i])
        xr = float(pos[i, 0]) + float(half_w[i])
        yb = float(pos[i, 1]) - float(half_h[i])
        yt = float(pos[i, 1]) + float(half_h[i])
        gxl = max(0, int(xl / cell_w))
        gxr = min(grid_g - 1, int(xr / cell_w))
        gyb = max(0, int(yb / cell_h))
        gyt = min(grid_g - 1, int(yt / cell_h))
        for gx in range(gxl, gxr + 1):
            cx = (gx + 0.5) * cell_w
            for gy in range(gyb, gyt + 1):
                ov = float(overflow[gx, gy])
                if ov > 0:
                    delta_i[i] += beta_od * ov / total_area
                cy = (gy + 0.5) * cell_h
                scale = (1.0 + delta_i[i]) * float(nu[gx, gy])
                grad[i, 0] += scale * (float(pos[i, 0]) - cx) / max(cell_w, 1e-9)
                grad[i, 1] += scale * (float(pos[i, 1]) - cy) / max(cell_h, 1e-9)

    return grad, delta_i, demand, bin_cap


def _displacement_growth_scale(
    pos: np.ndarray,
    pos_prev: np.ndarray,
    movable_idx: np.ndarray,
    cw: float,
    ch: float,
) -> float:
    """Cheap O(n_hard) proxy for HPWL change (avoids init_hpwl in the growth loop)."""
    if movable_idx.size == 0:
        return 1.0
    disp = float(
        np.mean(np.linalg.norm(pos[movable_idx] - pos_prev[movable_idx], axis=1))
    )
    hpwl_proxy_delta = disp / max(float(cw + ch) * 0.01, 1e-9)
    return 1.0 / (1.0 + 0.8 * min(hpwl_proxy_delta, 5.0))


def eplace_wl_density_relax(
    pos: np.ndarray,
    *,
    movable_idx: np.ndarray,
    movable_mask: np.ndarray,
    sizes_np: np.ndarray,
    half_w: np.ndarray,
    half_h: np.ndarray,
    cw: float,
    ch: float,
    macro_to_nets: Sequence[Sequence[int]],
    net_w: np.ndarray,
    cluster_id: np.ndarray,
    n_iters: int,
    wl_gamma: float,
    dens_lambda: float = 0.052,
    dens_lambda_start: float | None = None,
    dens_lambda_growth: float = 1.0,
    dens_alpha_start: float = 1e-12,
    dens_alpha_cap: float = 0.15,
    overflow_delta_beta: float = 1.0,
    hpwl_growth_stride: int = 4,
    cluster_gamma: float = 0.016,
    grid_g: int,
    wl_fn: Callable[[np.ndarray], float] | None = None,
    handoff_auto: bool = False,
    dens_viol_max: float = 0.05,
    wl_plateau_rel: float = 0.002,
    wl_window: int = 6,
    min_handoff_iters: int = 12,
) -> np.ndarray:
    """Weighted net pulls + per-bin local density spreading (RePlAce §III-B)."""
    out = pos.copy()
    nm = float(max(cw + ch, 1.0))
    dens_scale = 0.08 * nm
    n_hard = out.shape[0]
    grid_g = max(16, grid_g)
    max_iters = max(1, n_iters)
    min_iters = max(1, min(min_handoff_iters, max_iters))
    viol_thr = max(0.0, float(dens_viol_max))
    plateau_rel = max(1e-6, float(wl_plateau_rel))
    plateau_win = max(2, int(wl_window))
    wl_hist: list[float] = []

    if dens_lambda_start is not None:
        alpha = float(dens_lambda_start)
    else:
        alpha = float(dens_alpha_start)
    alpha_cap = float(dens_alpha_cap)
    alpha_growth = float(dens_lambda_growth)
    delta_i = np.zeros(n_hard, dtype=np.float64)
    vel = np.zeros_like(out)
    mom = 0.9
    wl_stride = max(1, int(hpwl_growth_stride))
    growth_scale = 1.0
    pos_prev_snapshot = out.copy()

    for it in range(max_iters):
        disp = np.zeros_like(out)
        for mi in movable_idx:
            mi = int(mi)
            nbrs: set[int] = set()
            for mj in range(n_hard):
                if mj == mi:
                    continue
                shared = set(macro_to_nets[mi]).intersection(macro_to_nets[mj])
                if not shared:
                    continue
                nbrs.add(mj)
            if not nbrs:
                continue
            sg = np.zeros(2, dtype=np.float64)
            wt = 0.0
            for mj in nbrs:
                shared = set(macro_to_nets[mi]).intersection(macro_to_nets[mj])
                w = max(1.0, float(len(shared)))
                for nid in shared:
                    w += 0.25 * float(net_w[nid])
                sg += w * (out[mj] - out[mi])
                wt += w
            if wt > 0:
                disp[mi] += wl_gamma * sg / wt

        if cluster_gamma > 0:
            for mi in movable_idx:
                mi = int(mi)
                cid = int(cluster_id[mi])
                members = np.where(cluster_id == cid)[0]
                if members.size <= 1:
                    continue
                cent = out[members].mean(axis=0)
                disp[mi] += cluster_gamma * (cent - out[mi])

        dens_grad, delta_i, demand, bin_cap = _per_bin_density_force(
            out,
            half_w,
            half_h,
            cw,
            ch,
            grid_g,
            alpha,
            delta_i,
            overflow_delta_beta=overflow_delta_beta,
        )
        for mi in movable_idx:
            mi = int(mi)
            disp[mi] += dens_scale * dens_grad[mi]

        for mi in movable_idx:
            mi = int(mi)
            vel[mi] = mom * vel[mi] + disp[mi]
            out[mi, 0] = np.clip(out[mi, 0] + vel[mi, 0], half_w[mi], cw - half_w[mi])
            out[mi, 1] = np.clip(out[mi, 1] + vel[mi, 1], half_h[mi], ch - half_h[mi])

        if it % wl_stride == 0 or it + 1 == max_iters:
            growth_scale = _displacement_growth_scale(
                out, pos_prev_snapshot, movable_idx, cw, ch
            )
            if handoff_auto and it + 1 >= min_iters and movable_idx.size > 0:
                wl_hist.append(
                    float(
                        np.mean(
                            np.linalg.norm(
                                out[movable_idx] - pos_prev_snapshot[movable_idx], axis=1
                            )
                        )
                    )
                )
            pos_prev_snapshot = out.copy()
        alpha = min(alpha_cap, alpha * (1.0 + (alpha_growth - 1.0) * growth_scale))

        if handoff_auto and it + 1 >= min_iters:
            viol = density_violation_frac_bins(demand, bin_cap)
            if viol <= viol_thr and len(wl_hist) >= plateau_win:
                old_wl = wl_hist[-plateau_win]
                new_wl = wl_hist[-1]
                if old_wl > 1e-12 and (old_wl - new_wl) / old_wl < plateau_rel:
                    break

    if handoff_auto and wl_fn is not None:
        wl_fn(out)

    return out


def _macro_half_extents(
    m: int,
    sizes_np: np.ndarray,
    rotated: np.ndarray,
) -> tuple[float, float]:
    if bool(rotated[m]):
        return float(sizes_np[m, 1]) / 2.0, float(sizes_np[m, 0]) / 2.0
    return float(sizes_np[m, 0]) / 2.0, float(sizes_np[m, 1]) / 2.0


def sequence_pair_pack(
    gamma_plus: Sequence[int],
    gamma_minus: Sequence[int],
    sizes_np: np.ndarray,
    half_w: np.ndarray,
    half_h: np.ndarray,
    cw: float,
    ch: float,
    movable_mask: np.ndarray,
    fixed_init: np.ndarray,
    rotated: np.ndarray | None = None,
) -> np.ndarray:
    """Legal-by-construction skyline pack from a sequence pair (movable macros)."""
    pos = fixed_init.copy()
    if rotated is None:
        rotated = np.zeros(sizes_np.shape[0], dtype=bool)
    rank_minus = {int(m): i for i, m in enumerate(gamma_minus)}

    x_end = 0.0
    for m in gamma_plus:
        m = int(m)
        if not movable_mask[m]:
            continue
        hw, hh = _macro_half_extents(m, sizes_np, rotated)
        x = x_end + hw
        y = hh + rank_minus[m] * (ch / max(len(gamma_minus), 1)) * 0.92
        x = np.clip(x, hw, cw - hw)
        y = np.clip(y, hh, ch - hh)
        pos[m, 0] = x
        pos[m, 1] = y
        x_end = x + hw
    return pos


def _pick_slack_macro(
    movable: Sequence[int],
    macro_deg: np.ndarray,
    rng: np.random.Generator,
) -> int:
    degs = np.array([max(1.0, float(macro_deg[int(m)])) for m in movable], dtype=np.float64)
    wts = np.sqrt(degs)
    wts = np.maximum(wts, 1e-12)
    wts /= np.sum(wts)
    return int(rng.choice(len(movable), p=wts))


def sequence_pair_sa_legalize(
    seed: np.ndarray,
    *,
    movable_idx: np.ndarray,
    movable_mask: np.ndarray,
    sizes_np: np.ndarray,
    half_w: np.ndarray,
    half_h: np.ndarray,
    cw: float,
    ch: float,
    fixed_init: np.ndarray,
    rng: np.random.Generator,
    n_iters: int,
    cost_fn: Callable[[np.ndarray], float],
    macro_deg: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """SA on sequence-pair swaps (legal placements by construction)."""
    movable = [int(i) for i in movable_idx]
    cur_plus = movable.copy()
    cur_minus = movable.copy()
    rng.shuffle(cur_plus)
    rng.shuffle(cur_minus)
    rotated = np.zeros(sizes_np.shape[0], dtype=bool)
    if macro_deg is None:
        macro_deg = np.ones(sizes_np.shape[0], dtype=np.float64)

    def pack() -> np.ndarray:
        return sequence_pair_pack(
            cur_plus,
            cur_minus,
            sizes_np,
            half_w,
            half_h,
            cw,
            ch,
            movable_mask,
            fixed_init,
            rotated,
        )

    best_pos = pack()
    best_cost = float(cost_fn(best_pos))
    cur_cost = best_cost
    for step in range(max(1, n_iters)):
        trial_plus = cur_plus.copy()
        trial_minus = cur_minus.copy()
        trial_rot = rotated.copy()
        mv = rng.random()
        if mv < 0.22:
            i = _pick_slack_macro(trial_plus, macro_deg, rng)
            j = int(rng.integers(0, len(trial_plus)))
            trial_plus[i], trial_plus[j] = trial_plus[j], trial_plus[i]
        elif mv < 0.40:
            i = _pick_slack_macro(trial_minus, macro_deg, rng)
            j = int(rng.integers(0, len(trial_minus)))
            trial_minus[i], trial_minus[j] = trial_minus[j], trial_minus[i]
        elif mv < 0.58:
            k = min(3, len(trial_plus))
            if k >= 2:
                i = int(rng.integers(0, len(trial_plus) - k + 1))
                block = trial_plus[i : i + k]
                rng.shuffle(block)
                trial_plus[i : i + k] = block
        elif mv < 0.74:
            m = int(_pick_slack_macro(movable, macro_deg, rng))
            trial_rot[m] = not trial_rot[m]
        elif mv < 0.87:
            i, j = rng.integers(0, len(trial_plus), size=2)
            trial_plus[i], trial_plus[j] = trial_plus[j], trial_plus[i]
        else:
            i, j = rng.integers(0, len(trial_minus), size=2)
            trial_minus[i], trial_minus[j] = trial_minus[j], trial_minus[i]

        cand = sequence_pair_pack(
            trial_plus,
            trial_minus,
            sizes_np,
            half_w,
            half_h,
            cw,
            ch,
            movable_mask,
            fixed_init,
            trial_rot,
        )
        cand_cost = float(cost_fn(cand))
        frac = step / max(n_iters - 1, 1)
        temp = max(0.02, 1.0 - frac)
        if cand_cost < cur_cost or rng.random() < math.exp(-(cand_cost - cur_cost) / max(temp, 1e-9)):
            cur_plus, cur_minus = trial_plus, trial_minus
            rotated = trial_rot
            cur_cost = cand_cost
            if cand_cost < best_cost:
                best_cost = cand_cost
                best_pos = cand.copy()
    return best_pos, best_cost


def design_class_from_benchmark(
    n_hard: int,
    cw: float,
    ch: float,
    sizes_np: np.ndarray,
    *,
    hr_pm: float,
    vr_pm: float,
) -> str:
    """Classify benchmark for profile-specific tuning (IBM vs NG45)."""
    density = float(np.sum(sizes_np[:, 0] * sizes_np[:, 1])) / max(cw * ch, 1e-9)
    name_hint = hr_pm > 0 and vr_pm > 0
    if n_hard < 20 and density > 0.35:
        return "ng45_sparse"
    if n_hard < 60 and density < 0.55 and name_hint:
        return "ng45_medium"
    if n_hard > 200:
        return "ibm_large"
    if n_hard > 80:
        return "ibm_medium"
    return "ibm_small"


def timing_critical_scores(
    macro_to_nets: Sequence[Sequence[int]],
    net_pin_nodes: Sequence,
    net_w: np.ndarray,
    n_hard: int,
) -> np.ndarray:
    """Cheap timing-criticality proxy: favor two-pin / high-weight nets."""
    scores = np.zeros(n_hard, dtype=np.float64)
    for i in range(n_hard):
        for nid in macro_to_nets[i]:
            pins = net_pin_nodes[nid]
            pin_count = int(pins.shape[0]) if hasattr(pins, "shape") else 0
            if pin_count <= 2:
                scores[i] += float(net_w[nid]) * 2.0
            elif pin_count > 0:
                scores[i] += float(net_w[nid]) / math.sqrt(float(pin_count))
    return scores


def analytical_global_place(
    pos: np.ndarray,
    *,
    movable_idx: np.ndarray,
    movable_mask: np.ndarray,
    sizes_np: np.ndarray,
    half_w: np.ndarray,
    half_h: np.ndarray,
    cw: float,
    ch: float,
    macro_to_nets: Sequence[Sequence[int]],
    net_w: np.ndarray,
    cluster_id: np.ndarray,
    n_iters: int = 300,
    wl_gamma: float = 0.028,
    dens_lambda_start: float = 0.001,
    dens_lambda_growth: float = 1.08,
    dens_alpha_cap: float | None = None,
    overflow_delta_beta: float = 1.0,
    hpwl_growth_stride: int = 2,
    cluster_gamma: float = 0.016,
    grid_g: int,
    n_hard: int | None = None,
    wl_fn: Callable[[np.ndarray], float] | None = None,
    momentum: float = 0.9,
    dens_viol_max: float = 0.05,
    wl_plateau_rel: float = 0.002,
    wl_window: int = 6,
    min_handoff_iters: int = 12,
) -> np.ndarray:
    """
    DREAMPlace-lite analytical global: WL pulls + per-bin local density with momentum.
    Density strength ``alpha`` grows from ``dens_lambda_start`` with HPWL-aware scaling.
    """
    out = pos.copy()
    vel = np.zeros_like(out)
    nm = float(max(cw + ch, 1.0))
    dens_scale = 0.08 * nm
    n_hard_pos = out.shape[0]
    nh = int(n_hard) if n_hard is not None else n_hard_pos
    grid_g = max(16, grid_g)
    if dens_alpha_cap is not None:
        alpha_cap = float(dens_alpha_cap)
    elif nh > 150:
        alpha_cap = 0.12
    elif nh > 60:
        alpha_cap = 0.30
    else:
        alpha_cap = 0.50
    growth = min(float(dens_lambda_growth), 1.04) if nh > 150 else float(dens_lambda_growth)
    wl_hist: list[float] = []
    max_iters = max(1, n_iters)
    min_iters = max(1, min(min_handoff_iters, max_iters))
    viol_thr = max(0.0, float(dens_viol_max))
    plateau_rel = max(1e-6, float(wl_plateau_rel))
    plateau_win = max(2, int(wl_window))
    mom = float(np.clip(momentum, 0.0, 0.99))
    alpha = float(dens_lambda_start)
    delta_i = np.zeros(n_hard_pos, dtype=np.float64)
    wl_stride = max(1, int(hpwl_growth_stride))
    growth_scale = 1.0
    pos_prev_snapshot = out.copy()

    for it in range(max_iters):
        disp = np.zeros_like(out)
        for mi in movable_idx:
            mi = int(mi)
            nbrs: set[int] = set()
            for mj in range(n_hard_pos):
                if mj == mi:
                    continue
                shared = set(macro_to_nets[mi]).intersection(macro_to_nets[mj])
                if not shared:
                    continue
                nbrs.add(mj)
            if not nbrs:
                continue
            sg = np.zeros(2, dtype=np.float64)
            wt = 0.0
            for mj in nbrs:
                shared = set(macro_to_nets[mi]).intersection(macro_to_nets[mj])
                w = max(1.0, float(len(shared)))
                for nid in shared:
                    w += 0.25 * float(net_w[nid])
                sg += w * (out[mj] - out[mi])
                wt += w
            if wt > 0:
                disp[mi] += wl_gamma * sg / wt

        if cluster_gamma > 0:
            for mi in movable_idx:
                mi = int(mi)
                cid = int(cluster_id[mi])
                members = np.where(cluster_id == cid)[0]
                if members.size <= 1:
                    continue
                cent = out[members].mean(axis=0)
                disp[mi] += cluster_gamma * (cent - out[mi])

        dens_grad, delta_i, demand, bin_cap = _per_bin_density_force(
            out,
            half_w,
            half_h,
            cw,
            ch,
            grid_g,
            alpha,
            delta_i,
            overflow_delta_beta=overflow_delta_beta,
        )
        for mi in movable_idx:
            mi = int(mi)
            disp[mi] += dens_scale * dens_grad[mi]

        for mi in movable_idx:
            mi = int(mi)
            vel[mi] = mom * vel[mi] + disp[mi]
            out[mi, 0] = np.clip(out[mi, 0] + vel[mi, 0], half_w[mi], cw - half_w[mi])
            out[mi, 1] = np.clip(out[mi, 1] + vel[mi, 1], half_h[mi], ch - half_h[mi])

        if it % wl_stride == 0 or it + 1 == max_iters:
            growth_scale = _displacement_growth_scale(
                out, pos_prev_snapshot, movable_idx, cw, ch
            )
            if wl_fn is not None and it + 1 >= min_iters and movable_idx.size > 0:
                wl_hist.append(
                    float(
                        np.mean(
                            np.linalg.norm(
                                out[movable_idx] - pos_prev_snapshot[movable_idx], axis=1
                            )
                        )
                    )
                )
            pos_prev_snapshot = out.copy()
        alpha = min(alpha_cap, alpha * (1.0 + (growth - 1.0) * growth_scale))

        if wl_fn is not None and it + 1 >= min_iters:
            viol = density_violation_frac_bins(demand, bin_cap)
            if viol <= viol_thr and len(wl_hist) >= plateau_win:
                old_wl = wl_hist[-plateau_win]
                new_wl = wl_hist[-1]
                if old_wl > 1e-12 and (old_wl - new_wl) / old_wl < plateau_rel:
                    break

    if wl_fn is not None:
        wl_fn(out)

    return out


def enforce_macro_channels(
    pos: np.ndarray,
    *,
    n_hard: int,
    movable_idx: np.ndarray,
    half_w: np.ndarray,
    half_h: np.ndarray,
    cw: float,
    ch: float,
    channel_width_um: float,
    legalize_fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Widen thin horizontal gaps between vertically separated macro pairs."""
    out = pos.copy()
    ch_w = max(channel_width_um, 1e-6)
    for i in movable_idx:
        i = int(i)
        for j in range(n_hard):
            if i == j:
                continue
            dx = abs(out[i, 0] - out[j, 0])
            dy = abs(out[i, 1] - out[j, 1])
            gap_x = dx - float(half_w[i]) - float(half_w[j])
            gap_y = dy - float(half_h[i]) - float(half_h[j])
            if 0 < gap_x < ch_w and gap_y > ch_w * 2.0:
                sign = 1.0 if out[i, 0] > out[j, 0] else -1.0
                push = (ch_w - gap_x) * 0.5
                out[i, 0] = np.clip(out[i, 0] + sign * push, half_w[i], cw - half_w[i])
    return legalize_fn(out)
