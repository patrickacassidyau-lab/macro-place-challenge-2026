"""Tests for Nesterov analytical global placement."""

import numpy as np
import pytest

from macro_place.analytical_torch import (
    analytical_hyperparams_for_design_class,
    nesterov_analytical_global_place,
)


def test_nesterov_analytical_moves_macros_in_bounds():
  pos = np.array([[5.0, 5.0], [15.0, 12.0], [22.0, 18.0]])
  hw = np.array([2.0, 2.5, 2.0])
  hh = np.array([2.0, 1.5, 2.0])
  movable = np.ones(3, dtype=bool)
  nets = [np.array([[0, 0], [1, 0], [2, 0]], dtype=np.int64)]
  offsets = [np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2))]
  out = nesterov_analytical_global_place(
      pos,
      movable_mask=movable,
      half_w=hw,
      half_h=hh,
      dens_half_w=hw + 1.0,
      dens_half_h=hh + 1.0,
      net_pin_nodes=nets,
      net_w=np.array([1.0]),
      macro_pin_offsets=offsets,
      soft_xy=np.zeros((0, 2)),
      ports_xy=np.zeros((0, 2)),
      num_macros=3,
      cw=40.0,
      ch=40.0,
      grid_g=8,
      n_iters=12,
  )
  assert out.shape == (3, 2)
  assert np.all(out[:, 0] >= hw) and np.all(out[:, 0] <= 40.0 - hw)
  assert np.all(out[:, 1] >= hh) and np.all(out[:, 1] <= 40.0 - hh)


def test_analytical_hyperparams_by_class():
  sm = analytical_hyperparams_for_design_class("ibm_small")
  assert sm["lambda_start"] == 0.0003 and sm["lambda_growth"] == 1.03 and sm["n_iters"] == 180
  lg = analytical_hyperparams_for_design_class("ibm_large")
  assert lg["lambda_start"] == 0.0003 and lg["lambda_growth"] == 1.03 and lg["n_iters"] == 140
  ng = analytical_hyperparams_for_design_class("ng45_medium")
  assert ng["lambda_start"] == 0.0005 and ng["lambda_growth"] == 1.04 and ng["n_iters"] == 250
