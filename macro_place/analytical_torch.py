"""Nesterov-momentum analytical global placement (continuous HPWL + density)."""

from __future__ import annotations

from typing import Sequence, TypedDict

import numpy as np
import torch


class AnalyticalSchedule(TypedDict):
    lambda_start: float
    lambda_growth: float
    n_iters: int
    lambda_cap: float


def analytical_hyperparams_for_design_class(design_class: str) -> AnalyticalSchedule:
    """Design-class λ schedule and iteration budget (coupled to density-grid footprint)."""
    if design_class in ("ibm_small", "ibm_medium"):
        return AnalyticalSchedule(
            lambda_start=0.0003,
            lambda_growth=1.03,
            n_iters=180,
            lambda_cap=0.08,
        )
    if design_class == "ibm_large":
        return AnalyticalSchedule(
            lambda_start=0.0003,
            lambda_growth=1.03,
            n_iters=140,
            lambda_cap=0.12,
        )
    if design_class.startswith("ng45"):
        return AnalyticalSchedule(
            lambda_start=0.0005,
            lambda_growth=1.04,
            n_iters=250,
            lambda_cap=0.20,
        )
    return AnalyticalSchedule(
        lambda_start=0.0003,
        lambda_growth=1.03,
        n_iters=180,
        lambda_cap=0.08,
    )


def _pin_coords_from_pos(
    ph: torch.Tensor,
    pn: torch.Tensor,
    *,
    n_hard: int,
    num_macros: int,
    off_mat: torch.Tensor,
    mxp: int,
    soft_xy_t: torch.Tensor,
    port_xy_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    oid = pn[:, 0].long()
    sk = pn[:, 1].long()
    px = torch.zeros(pn.shape[0], dtype=torch.float32, device=ph.device)
    py = torch.zeros(pn.shape[0], dtype=torch.float32, device=ph.device)

    hm = oid < n_hard
    if hm.any():
        hi = oid[hm]
        ss = torch.clamp(sk[hm], 0, mxp - 1)
        px[hm] = ph[hi, 0] + off_mat[hi, ss, 0]
        py[hm] = ph[hi, 1] + off_mat[hi, ss, 1]

    sm = (~hm) & (oid < num_macros)
    if sm.any():
        si = (oid[sm] - n_hard).long()
        ns = soft_xy_t.shape[0]
        if ns > 0:
            si = si.clamp(0, ns - 1)
            px[sm] = soft_xy_t[si, 0]
            py[sm] = soft_xy_t[si, 1]

    pm = oid >= num_macros
    if pm.any():
        pj = (oid[pm] - num_macros).long()
        npports = port_xy_t.shape[0]
        if npports > 0:
            pj = pj.clamp(0, npports - 1)
            px[pm] = port_xy_t[pj, 0]
            py[pm] = port_xy_t[pj, 1]

    return px, py


def _hpwl_loss(
    ph: torch.Tensor,
    net_pin_nodes: Sequence,
    net_w: torch.Tensor,
    *,
    n_hard: int,
    num_macros: int,
    off_mat: torch.Tensor,
    mxp: int,
    soft_xy_t: torch.Tensor,
    port_xy_t: torch.Tensor,
) -> torch.Tensor:
    loss = torch.tensor(0.0, dtype=torch.float32, device=ph.device)
    for nid, pn_t in enumerate(net_pin_nodes):
        if pn_t.shape[0] < 2:
            continue
        if isinstance(pn_t, np.ndarray):
            pn = torch.tensor(pn_t, dtype=torch.long, device=ph.device)
        else:
            pn = pn_t.to(device=ph.device)
        xs, ys = _pin_coords_from_pos(
            ph,
            pn,
            n_hard=n_hard,
            num_macros=num_macros,
            off_mat=off_mat,
            mxp=mxp,
            soft_xy_t=soft_xy_t,
            port_xy_t=port_xy_t,
        )
        loss = loss + net_w[nid] * (xs.max() - xs.min() + ys.max() - ys.min())
    return loss


def _bin_density_overflow_loss(
    ph: torch.Tensor,
    dens_hw: torch.Tensor,
    dens_hh: torch.Tensor,
    *,
    cw: float,
    ch: float,
    grid_g: int,
    movable_mask: torch.Tensor,
    n_hard: int,
) -> torch.Tensor:
    """Uniform-bin area demand vs capacity (inflated macro footprints for Tier-2 spacing)."""
    gg = max(8, int(grid_g))
    cell_w = float(cw) / gg
    cell_h = float(ch) / gg
    bin_cap = cell_w * cell_h
    acc = torch.zeros(gg * gg, dtype=torch.float32, device=ph.device)
    for i in range(n_hard):
        if not bool(movable_mask[i].item()):
            continue
        area = 4.0 * dens_hw[i] * dens_hh[i]
        gx = torch.clamp((ph[i, 0] / cell_w).long(), 0, gg - 1)
        gy = torch.clamp((ph[i, 1] / cell_h).long(), 0, gg - 1)
        acc[gx * gg + gy] = acc[gx * gg + gy] + area
    overflow = torch.relu(acc - bin_cap)
    return torch.mean(overflow * overflow)


def nesterov_analytical_global_place(
    pos: np.ndarray,
    *,
    movable_mask: np.ndarray,
    half_w: np.ndarray,
    half_h: np.ndarray,
    dens_half_w: np.ndarray,
    dens_half_h: np.ndarray,
    net_pin_nodes: Sequence,
    net_w: np.ndarray,
    macro_pin_offsets: Sequence[np.ndarray],
    soft_xy: np.ndarray,
    ports_xy: np.ndarray,
    num_macros: int,
    cw: float,
    ch: float,
    grid_g: int,
    n_iters: int = 96,
    lambda_start: float = 0.001,
    lambda_growth: float = 1.08,
    lambda_cap: float = 0.50,
    lr_scale: float = 2.15,
    device: torch.device | None = None,
) -> np.ndarray:
    """
    Joint HPWL + λ·density overflow with Nesterov momentum (pre–pool-survivor globalization).
    ``dens_half_*`` carry macro channel spacing into the density grid (e.g. NG45 12 µm halo).
    """
    n_hard = int(pos.shape[0])
    movable = np.asarray(movable_mask[:n_hard], dtype=bool)
    if device is None:
        device = torch.device("cpu")

    pos_h = torch.tensor(pos[:n_hard], dtype=torch.float32, device=device, requires_grad=True)
    nw_t = torch.tensor(net_w, dtype=torch.float32, device=device)
    mov_t = torch.tensor(movable, dtype=torch.bool, device=device)
    hw_t = torch.tensor(half_w[:n_hard], dtype=torch.float32, device=device)
    hh_t = torch.tensor(half_h[:n_hard], dtype=torch.float32, device=device)
    dhw_t = torch.tensor(dens_half_w[:n_hard], dtype=torch.float32, device=device)
    dhh_t = torch.tensor(dens_half_h[:n_hard], dtype=torch.float32, device=device)
    cw_t = torch.tensor(float(cw), dtype=torch.float32, device=device)
    ch_t = torch.tensor(float(ch), dtype=torch.float32, device=device)

    mxp = max(1, max(int(o.shape[0]) for o in macro_pin_offsets[:n_hard]))
    off_mat = torch.zeros(n_hard, mxp, 2, dtype=torch.float32, device=device)
    for i in range(n_hard):
        oi = macro_pin_offsets[i]
        if oi.size:
            oo = torch.tensor(oi, dtype=torch.float32, device=device)
            off_mat[i, : oo.shape[0], :] = oo

    soft_xy_t = torch.tensor(soft_xy, dtype=torch.float32, device=device)
    port_xy_t = torch.tensor(ports_xy, dtype=torch.float32, device=device)

    lr = float(lr_scale) * min(float(cw), float(ch)) / max(n_hard * 75.0, 400.0)
    opt = torch.optim.SGD([pos_h], lr=lr, momentum=0.9, nesterov=True)
    max_iters = max(1, int(n_iters))
    lam_cap = float(lambda_cap)
    lam0 = float(lambda_start)
    lam_g = float(lambda_growth)

    for it in range(max_iters):
        lam = min(lam_cap, lam0 * (lam_g**it))
        opt.zero_grad(set_to_none=True)
        loss_wl = _hpwl_loss(
            pos_h,
            net_pin_nodes,
            nw_t,
            n_hard=n_hard,
            num_macros=int(num_macros),
            off_mat=off_mat,
            mxp=mxp,
            soft_xy_t=soft_xy_t,
            port_xy_t=port_xy_t,
        )
        loss_d = lam * _bin_density_overflow_loss(
            pos_h,
            dhw_t,
            dhh_t,
            cw=cw,
            ch=ch,
            grid_g=grid_g,
            movable_mask=mov_t,
            n_hard=n_hard,
        )
        (loss_wl + loss_d).backward()
        with torch.no_grad():
            if pos_h.grad is not None:
                pos_h.grad[~mov_t] = 0.0
        opt.step()
        with torch.no_grad():
            for i in range(n_hard):
                if movable[i]:
                    pos_h[i, 0] = torch.clamp(pos_h[i, 0], hw_t[i], cw_t - hw_t[i])
                    pos_h[i, 1] = torch.clamp(pos_h[i, 1], hh_t[i], ch_t - hh_t[i])

    return pos_h.detach().cpu().numpy().astype(np.float64)
