from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Iterable

import numpy as np


DIRECTION_BIN_COUNT = 8
DEFAULT_XY_BINS_M = (-4.0, -2.0, 0.0, 2.0, 4.0)
DEFAULT_Z_BINS_M = (0.0, 1.0, 2.0, 3.0)


@dataclass(frozen=True)
class DirectionalResidualObservation:
    x_w_m: float
    y_w_m: float
    z_w_m: float
    direction_rad: float
    lift_residual_m_s: float
    energy_residual_m: float
    dwell_residual_s: float
    observation_weight: float = 1.0


@dataclass(frozen=True)
class DirectionalResidualCell:
    x_bin: int
    y_bin: int
    z_bin: int
    direction_bin: int
    observation_count: int
    lift_residual_mean_m_s: float
    energy_residual_mean_m: float
    dwell_residual_mean_s: float
    uncertainty: float


@dataclass(frozen=True)
class DirectionalResidualLiftBelief:
    x_edges_m: tuple[float, ...] = DEFAULT_XY_BINS_M
    y_edges_m: tuple[float, ...] = DEFAULT_XY_BINS_M
    z_edges_m: tuple[float, ...] = DEFAULT_Z_BINS_M
    direction_bin_count: int = DIRECTION_BIN_COUNT
    cells: tuple[DirectionalResidualCell, ...] = ()
    update_count: int = 0
    belief_version: str = "directional_residual_lift_belief_v411"


def initial_directional_residual_lift_belief(
    *,
    x_edges_m: Iterable[float] = DEFAULT_XY_BINS_M,
    y_edges_m: Iterable[float] = DEFAULT_XY_BINS_M,
    z_edges_m: Iterable[float] = DEFAULT_Z_BINS_M,
    direction_bin_count: int = DIRECTION_BIN_COUNT,
) -> DirectionalResidualLiftBelief:
    return DirectionalResidualLiftBelief(
        x_edges_m=tuple(float(value) for value in x_edges_m),
        y_edges_m=tuple(float(value) for value in y_edges_m),
        z_edges_m=tuple(float(value) for value in z_edges_m),
        direction_bin_count=int(direction_bin_count),
    )


def update_directional_residual_lift_belief(
    belief: DirectionalResidualLiftBelief,
    observation: DirectionalResidualObservation,
) -> DirectionalResidualLiftBelief:
    key = _cell_key_for_observation(belief, observation)
    cells = {_cell_key(cell): cell for cell in belief.cells}
    prior = cells.get(key)
    weight = max(1e-9, float(observation.observation_weight))
    if prior is None:
        count = 1
        cells[key] = DirectionalResidualCell(
            x_bin=key[0],
            y_bin=key[1],
            z_bin=key[2],
            direction_bin=key[3],
            observation_count=count,
            lift_residual_mean_m_s=float(observation.lift_residual_m_s),
            energy_residual_mean_m=float(observation.energy_residual_m),
            dwell_residual_mean_s=float(observation.dwell_residual_s),
            uncertainty=1.0 / math.sqrt(float(count)),
        )
    else:
        count = int(prior.observation_count) + 1
        alpha = weight / (float(prior.observation_count) + weight)
        cells[key] = replace(
            prior,
            observation_count=count,
            lift_residual_mean_m_s=_blend(prior.lift_residual_mean_m_s, observation.lift_residual_m_s, alpha),
            energy_residual_mean_m=_blend(prior.energy_residual_mean_m, observation.energy_residual_m, alpha),
            dwell_residual_mean_s=_blend(prior.dwell_residual_mean_s, observation.dwell_residual_s, alpha),
            uncertainty=1.0 / math.sqrt(float(count)),
        )
    return replace(
        belief,
        cells=tuple(sorted(cells.values(), key=_cell_key)),
        update_count=int(belief.update_count) + 1,
    )


def query_directional_residual_lift_features(
    belief: DirectionalResidualLiftBelief,
    *,
    x_w_m: float,
    y_w_m: float,
    z_w_m: float,
    direction_rad: float,
) -> dict[str, float | int | str]:
    key = _cell_key_for_values(belief, x_w_m=x_w_m, y_w_m=y_w_m, z_w_m=z_w_m, direction_rad=direction_rad)
    cells = {_cell_key(cell): cell for cell in belief.cells}
    cell = cells.get(key)
    if cell is None:
        return {
            "belief_version": belief.belief_version,
            "belief_local_lift_residual_m_s": 0.0,
            "belief_local_energy_residual_m": 0.0,
            "belief_local_dwell_residual_s": 0.0,
            "belief_uncertainty": 1.0,
            "belief_observation_count": 0,
            "belief_direction_bin": int(key[3]),
            "belief_z_bin": int(key[2]),
            "belief_update_count": int(belief.update_count),
        }
    return {
        "belief_version": belief.belief_version,
        "belief_local_lift_residual_m_s": float(cell.lift_residual_mean_m_s),
        "belief_local_energy_residual_m": float(cell.energy_residual_mean_m),
        "belief_local_dwell_residual_s": float(cell.dwell_residual_mean_s),
        "belief_uncertainty": float(cell.uncertainty),
        "belief_observation_count": int(cell.observation_count),
        "belief_direction_bin": int(cell.direction_bin),
        "belief_z_bin": int(cell.z_bin),
        "belief_update_count": int(belief.update_count),
    }


def directional_residual_observation_from_rows(
    *,
    expected_row: dict[str, object],
    observed_row: dict[str, object],
    direction_rad: float,
) -> DirectionalResidualObservation:
    """Build one residual-memory observation from expected and observed rollout rows."""

    return DirectionalResidualObservation(
        x_w_m=_required_float(observed_row, "initial_x_w", "x_w_m"),
        y_w_m=_required_float(observed_row, "initial_y_w", "y_w_m"),
        z_w_m=_required_float(observed_row, "initial_z_w", "z_w_m"),
        direction_rad=float(direction_rad),
        lift_residual_m_s=float(observed_row.get("w_wing_mean_m_s", 0.0))
        - float(expected_row.get("w_wing_mean_m_s", 0.0)),
        energy_residual_m=float(observed_row.get("energy_residual_m", 0.0))
        - float(expected_row.get("expected_energy_residual_m", 0.0)),
        dwell_residual_s=float(observed_row.get("lift_dwell_time_s", 0.0))
        - float(expected_row.get("expected_lift_dwell_time_s", 0.0)),
    )


def belief_snapshot_row(belief: DirectionalResidualLiftBelief, label: str = "") -> dict[str, object]:
    payload = asdict(belief)
    payload["snapshot_label"] = str(label)
    payload["x_edges_m"] = json.dumps(list(belief.x_edges_m), separators=(",", ":"))
    payload["y_edges_m"] = json.dumps(list(belief.y_edges_m), separators=(",", ":"))
    payload["z_edges_m"] = json.dumps(list(belief.z_edges_m), separators=(",", ":"))
    payload["cell_count"] = len(belief.cells)
    payload["cells_json"] = json.dumps([asdict(cell) for cell in belief.cells], separators=(",", ":"))
    payload.pop("cells")
    return payload


def _cell_key_for_observation(
    belief: DirectionalResidualLiftBelief,
    observation: DirectionalResidualObservation,
) -> tuple[int, int, int, int]:
    return _cell_key_for_values(
        belief,
        x_w_m=observation.x_w_m,
        y_w_m=observation.y_w_m,
        z_w_m=observation.z_w_m,
        direction_rad=observation.direction_rad,
    )


def _cell_key_for_values(
    belief: DirectionalResidualLiftBelief,
    *,
    x_w_m: float,
    y_w_m: float,
    z_w_m: float,
    direction_rad: float,
) -> tuple[int, int, int, int]:
    return (
        _bin_index(float(x_w_m), belief.x_edges_m),
        _bin_index(float(y_w_m), belief.y_edges_m),
        _bin_index(float(z_w_m), belief.z_edges_m),
        _direction_bin(float(direction_rad), int(belief.direction_bin_count)),
    )


def _cell_key(cell: DirectionalResidualCell) -> tuple[int, int, int, int]:
    return (int(cell.x_bin), int(cell.y_bin), int(cell.z_bin), int(cell.direction_bin))


def _bin_index(value: float, edges: tuple[float, ...]) -> int:
    if len(edges) < 2:
        raise ValueError("belief bin edges must contain at least two values")
    return int(np.clip(np.searchsorted(np.asarray(edges, dtype=float), value, side="right") - 1, 0, len(edges) - 2))


def _direction_bin(direction_rad: float, count: int) -> int:
    if count <= 0:
        raise ValueError("direction_bin_count must be positive")
    wrapped = float(direction_rad) % (2.0 * math.pi)
    return int(min(count - 1, math.floor(wrapped / (2.0 * math.pi) * count)))


def _blend(old: float, new: float, alpha: float) -> float:
    return float(old) + float(alpha) * (float(new) - float(old))


def _required_float(row: dict[str, object], *names: str) -> float:
    for name in names:
        if name in row and row[name] not in ("", None):
            return float(row[name])
    raise ValueError("directional_residual_memory_missing_canonical_coordinate:" + ",".join(names))
