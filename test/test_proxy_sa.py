"""Smoke tests for proxy-cost SA moves."""

import numpy as np

from macro_place.proxy_sa import MOVE_PROBS, N_MOVES_PER_ITER, ProxySAContext, _apply_move


def test_apply_move_clips():
    fixed = np.array([[5.0, 5.0], [20.0, 15.0]])
    movable = np.ones(2, dtype=bool)
    idx = np.arange(2)
    sizes = np.array([[4.0, 3.0], [6.0, 4.0]])
    hw = sizes[:, 0] / 2
    hh = sizes[:, 1] / 2

    def legalize(p):
        return p

    ctx = ProxySAContext(
        n_hard=2,
        movable_idx=idx,
        movable_mask=movable,
        fixed_init=fixed,
        sizes_np=sizes,
        half_w=hw,
        half_h=hh,
        cw=50.0,
        ch=50.0,
        legalize_fn=legalize,
        oracle_fn=lambda p: {"proxy_cost": 1.0, "overlap_count": 0},
        oracle_lock=__import__("threading").Lock(),
    )
    rng = np.random.default_rng(0)
    out = _apply_move(ctx, fixed, rng)
    assert np.all(out[:, 0] >= hw) and np.all(out[:, 0] <= 50.0 - hw)
    assert len(MOVE_PROBS) == 5
    assert N_MOVES_PER_ITER == 20
