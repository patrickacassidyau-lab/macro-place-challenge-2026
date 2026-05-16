"""Fit cheap surrogate feature weights from oracle labels (ledger + in-run pool rows)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from macro_place.experiment_tracker import RUNS_JSONL


@dataclass(frozen=True)
class SurrogateFeatureWeights:
    wl: float = 1.0
    dens: float = 1.0
    route: float = 1.0
    press: float = 1.0
    overflow: float = 1.0
    peri: float = 1.0

    def scale_route(self, route_w0: float, press_w0: float) -> tuple[float, float]:
        return route_w0 * self.route, press_w0 * self.press


def _ridge_fit(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    if x.shape[0] < x.shape[1]:
        ridge = max(ridge, 1e-2)
    xtx = x.T @ x + ridge * np.eye(x.shape[1], dtype=np.float64)
    xty = x.T @ y
    try:
        coef = np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(xtx, xty, rcond=None)[0]
    return coef


def _rows_from_artifact(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    out: list[dict[str, float]] = []
    for row in payload.get("runs") or []:
        try:
            proxy = float(row["proxy_cost"])
            wl = float(row["wirelength"])
            dens = float(row.get("density") or row.get("density_cost") or 0.0)
            cong = float(row.get("congestion") or row.get("congestion_cost") or 0.0)
        except (KeyError, TypeError, ValueError):
            continue
        out.append({"proxy": proxy, "wl": wl, "dens": dens, "cong": cong})
    return out


def load_ledger_oracle_rows(root: Path | None = None) -> list[dict[str, float]]:
    """Aggregate Tier-1 proxy breakdown rows from experiment artifacts."""
    base = root or Path.cwd()
    ledger = base / RUNS_JSONL
    if not ledger.exists():
        return []
    rows: list[dict[str, float]] = []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        rel = rec.get("artifact_json")
        if not rel:
            continue
        art = base / str(rel)
        rows.extend(_rows_from_artifact(art))
    return rows


def fit_weights_from_oracle_rows(
    rows: Sequence[Mapping[str, float]],
    *,
    ridge: float = 1e-3,
) -> SurrogateFeatureWeights:
    """
    Map cheap layout cues to PlacementCost proxy via ridge regression.
    Features: log wl, log dens, log cong, cong/wl ratio.
    """
    if len(rows) < 3:
        return SurrogateFeatureWeights()
    feats: list[list[float]] = []
    targets: list[float] = []
    for row in rows:
        proxy = float(row.get("proxy") or row.get("proxy_cost") or 0.0)
        wl = max(float(row.get("wl") or row.get("wirelength_cost") or 0.0), 1e-9)
        dens = max(float(row.get("dens") or row.get("density_cost") or 0.0), 1e-9)
        cong = max(float(row.get("cong") or row.get("congestion_cost") or 0.0), 1e-9)
        feats.append(
            [
                1.0,
                math.log(wl),
                math.log(dens),
                math.log(cong),
                math.log(cong / wl),
            ]
        )
        targets.append(math.log(max(proxy, 1e-9)))
    x = np.asarray(feats, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    coef = _ridge_fit(x, y, ridge)
    wl_scale = float(np.clip(math.exp(coef[1]), 0.35, 3.5))
    if wl_scale <= 0.0:
        return SurrogateFeatureWeights()
    dens_scale = float(np.clip(math.exp(coef[2]), 0.35, 4.0))
    cong_scale = float(np.clip(math.exp(coef[3]), 0.35, 6.0))
    rat_scale = float(np.clip(1.0 + 0.35 * coef[4], 0.5, 3.0))
    route_scale = cong_scale * rat_scale
    return SurrogateFeatureWeights(
        wl=wl_scale,
        dens=dens_scale,
        route=route_scale,
        press=route_scale * 0.65,
        overflow=dens_scale * 0.85,
        peri=1.0,
    )


def fit_weights_from_pool_rows(
    pool_rows: Sequence[tuple[Any, ...]],
    *,
    ridge: float = 1e-3,
) -> SurrogateFeatureWeights:
    """Use in-run oracle-labeled pool candidates (proxy dict in row[2])."""
    oracle_rows: list[dict[str, float]] = []
    for row in pool_rows:
        if len(row) < 3:
            continue
        oracle = row[2]
        if not isinstance(oracle, dict):
            continue
        oracle_rows.append(
            {
                "proxy": float(oracle.get("proxy_cost", 0.0)),
                "wl": float(oracle.get("wirelength_cost", 0.0)),
                "dens": float(oracle.get("density_cost", 0.0)),
                "cong": float(oracle.get("congestion_cost", 0.0)),
            }
        )
    return fit_weights_from_oracle_rows(oracle_rows, ridge=ridge)


def ledger_oracle_row_count(root: Path | None = None) -> int:
    return len(load_ledger_oracle_rows(root))


def load_ledger_surrogate_weights(
    root: Path | None = None,
    *,
    min_rows: int = 100,
) -> SurrogateFeatureWeights:
    rows = load_ledger_oracle_rows(root)
    if len(rows) < max(1, min_rows):
        return SurrogateFeatureWeights()
    return fit_weights_from_oracle_rows(rows)
