"""Tests for spiral / greedy-packer macro seeds."""

import numpy as np

from macro_place.placement_init import greedy_packer_macro_seed, spiral_macro_seed


def test_spiral_and_greedy_in_bounds():
    fixed = np.array([[5.0, 5.0], [20.0, 15.0], [30.0, 25.0]])
    movable = np.ones(3, dtype=bool)
    idx = np.arange(3)
    sizes = np.array([[4.0, 3.0], [6.0, 4.0], [5.0, 5.0]])
    hw = sizes[:, 0] / 2
    hh = sizes[:, 1] / 2
    sp = spiral_macro_seed(fixed, idx, movable, sizes, hw, hh, 50.0, 50.0)
    gp = greedy_packer_macro_seed(fixed, idx, movable, sizes, hw, hh, 50.0, 50.0)
    for pos in (sp, gp):
        assert np.all(pos[:, 0] >= hw) and np.all(pos[:, 0] <= 50.0 - hw)
        assert np.all(pos[:, 1] >= hh) and np.all(pos[:, 1] <= 50.0 - hh)
