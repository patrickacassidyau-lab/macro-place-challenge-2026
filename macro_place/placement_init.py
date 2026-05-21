"""Alternative macro placement seeds for multi-worker SA (spiral / greedy packer)."""

from __future__ import annotations

import math

import numpy as np


def spiral_macro_seed(
    fixed_init: np.ndarray,
    movable_idx: np.ndarray,
    movable_mask: np.ndarray,
    sizes_np: np.ndarray,
    half_w: np.ndarray,
    half_h: np.ndarray,
    cw: float,
    ch: float,
) -> np.ndarray:
    """Counterclockwise boundary spiral (largest macros first)."""
    pos = fixed_init.copy()
    order = sorted(
        (int(i) for i in movable_idx if movable_mask[i]),
        key=lambda i: -(float(sizes_np[i, 0]) * float(sizes_np[i, 1])),
    )
    n = len(order)
    if n == 0:
        return pos
    margin = max(float(np.max(half_w[order])), float(np.max(half_h[order])), 1e-3)
    cx, cy = float(cw) * 0.5, float(ch) * 0.5
    rx = max(float(cw) * 0.5 - margin, margin)
    ry = max(float(ch) * 0.5 - margin, margin)
    for k, m in enumerate(order):
        ang = 2.0 * math.pi * float(k) / float(n)
        pos[m, 0] = np.clip(cx + rx * math.cos(ang), half_w[m], float(cw) - half_w[m])
        pos[m, 1] = np.clip(cy + ry * math.sin(ang), half_h[m], float(ch) - half_h[m])
    return pos


def greedy_packer_macro_seed(
    fixed_init: np.ndarray,
    movable_idx: np.ndarray,
    movable_mask: np.ndarray,
    sizes_np: np.ndarray,
    half_w: np.ndarray,
    half_h: np.ndarray,
    cw: float,
    ch: float,
) -> np.ndarray:
    """Lower-left row packing (largest macros first)."""
    pos = fixed_init.copy()
    order = sorted(
        (int(i) for i in movable_idx if movable_mask[i]),
        key=lambda i: -(float(sizes_np[i, 0]) * float(sizes_np[i, 1])),
    )
    if not order:
        return pos
    margin = max(float(np.max(half_w[order])), float(np.max(half_h[order])), 1e-3)
    x_cursor = margin
    y_cursor = margin
    row_h = 0.0
    for m in order:
        w = float(sizes_np[m, 0])
        h = float(sizes_np[m, 1])
        if x_cursor + w + margin > float(cw):
            x_cursor = margin
            y_cursor += row_h + margin
            row_h = 0.0
        pos[m, 0] = np.clip(x_cursor + half_w[m], half_w[m], float(cw) - half_w[m])
        pos[m, 1] = np.clip(y_cursor + half_h[m], half_h[m], float(ch) - half_h[m])
        x_cursor += w + margin
        row_h = max(row_h, h)
    return pos
