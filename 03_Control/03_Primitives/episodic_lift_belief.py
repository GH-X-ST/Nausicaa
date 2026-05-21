from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BeliefGrid:
    belief_grid_id: str
    fan_branch: str
    x_edges_m: tuple[float, ...]
    y_edges_m: tuple[float, ...]
    values_m_s: tuple[tuple[float, ...], ...]
    memory_lambda: float
    observation_count: int

    def as_array(self) -> np.ndarray:
        return np.asarray(self.values_m_s, dtype=float)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def initialise_belief_grid(
    fan_branch: str,
    *,
    x_edges_m: tuple[float, ...] = (1.0, 2.5, 4.0, 5.5, 7.0),
    y_edges_m: tuple[float, ...] = (0.0, 1.2, 2.4, 3.6, 4.8),
    initial_value_m_s: float = 0.0,
    memory_lambda: float = 0.5,
) -> BeliefGrid:
    _validate_fan_branch(fan_branch)
    if len(x_edges_m) < 2 or len(y_edges_m) < 2:
        raise ValueError("belief grid requires at least two x and y edges.")
    values = np.full((len(x_edges_m) - 1, len(y_edges_m) - 1), float(initial_value_m_s))
    grid_id = _belief_id(str(fan_branch), values, 0, float(memory_lambda))
    return _grid(str(fan_branch), tuple(x_edges_m), tuple(y_edges_m), values, float(memory_lambda), 0, grid_id)


def observe_lift_from_episode(
    trajectory: pd.DataFrame,
    updraft_model: object | None = None,
    energy_residual: float = 0.0,
    reference_belief: BeliefGrid | None = None,
) -> BeliefGrid:
    """Build a grid observation from an episode trajectory.

    The first implementation deliberately uses explicit table columns when
    available. A model query is only a fallback so measured episode lift can
    later replace simulation fields without changing the update equation.
    """

    belief = initialise_belief_grid("single_fan_branch") if reference_belief is None else reference_belief
    values = np.full_like(belief.as_array(), np.nan, dtype=float)
    counts = np.zeros_like(values, dtype=float)
    if trajectory.empty:
        fill = float(energy_residual)
        values[:, :] = fill
    else:
        for _, row in trajectory.iterrows():
            x_w_m = _float(row.get("x_w_m", row.get("x_w", np.nan)))
            y_w_m = _float(row.get("y_w_m", row.get("y_w", np.nan)))
            if not np.isfinite(x_w_m) or not np.isfinite(y_w_m):
                continue
            lift_m_s = _observation_value(row, updraft_model, energy_residual)
            ix = _bin_index(x_w_m, belief.x_edges_m)
            iy = _bin_index(y_w_m, belief.y_edges_m)
            if ix is None or iy is None:
                continue
            values[ix, iy] = 0.0 if not np.isfinite(values[ix, iy]) else values[ix, iy]
            values[ix, iy] += lift_m_s
            counts[ix, iy] += 1.0
        observed = counts > 0.0
        values[observed] = values[observed] / counts[observed]
        values[~observed] = float(energy_residual)
    return _grid(
        belief.fan_branch,
        belief.x_edges_m,
        belief.y_edges_m,
        values,
        belief.memory_lambda,
        int(belief.observation_count) + 1,
        _belief_id(belief.fan_branch, values, int(belief.observation_count) + 1, belief.memory_lambda),
    )


def update_belief(
    previous_belief: BeliefGrid,
    observation: BeliefGrid,
    memory_lambda: float,
) -> BeliefGrid:
    """Apply b_next = lambda * b_prev + (1 - lambda) * observation."""

    lam = float(memory_lambda)
    if lam < 0.0 or lam > 1.0:
        raise ValueError("memory_lambda must be in [0, 1].")
    if previous_belief.fan_branch != observation.fan_branch:
        raise ValueError("belief and observation fan_branch must match.")
    prev = previous_belief.as_array()
    obs = observation.as_array()
    if prev.shape != obs.shape:
        raise ValueError("belief and observation grids must have matching shape.")
    values = lam * prev + (1.0 - lam) * obs
    count = int(previous_belief.observation_count) + 1
    return _grid(
        previous_belief.fan_branch,
        previous_belief.x_edges_m,
        previous_belief.y_edges_m,
        values,
        lam,
        count,
        _belief_id(previous_belief.fan_branch, values, count, lam),
    )


def score_primitive_objective(
    current_state: np.ndarray,
    belief: BeliefGrid,
    candidate_primitive: dict[str, object],
) -> float:
    """Return a transparent objective score for primitive ranking, not commands."""

    state = np.asarray(current_state, dtype=float).reshape(-1)
    if state.size < 3:
        raise ValueError("current_state must contain at least x/y/z.")
    ix = _bin_index(float(state[0]), belief.x_edges_m)
    iy = _bin_index(float(state[1]), belief.y_edges_m)
    local_lift = 0.0 if ix is None or iy is None else float(belief.as_array()[ix, iy])
    dwell_bonus = _float(candidate_primitive.get("dwell_time_s", 0.0))
    energy_bonus = _float(candidate_primitive.get("energy_residual_m", 0.0))
    margin_bonus = _float(candidate_primitive.get("minimum_margin_m", 0.0))
    # The coefficients are intentionally simple and auditable. They rank
    # candidate objectives before the governor applies the hard safety filter.
    return float(local_lift + 0.05 * dwell_bonus + 0.25 * energy_bonus + 0.10 * margin_bonus)


def write_belief_manifest(
    path: Path,
    *,
    previous_belief: BeliefGrid,
    updated_belief: BeliefGrid,
    source_episode_id: str,
    belief_update_status: str,
) -> dict[str, object]:
    payload = {
        "belief_grid_id": updated_belief.belief_grid_id,
        "fan_branch": updated_belief.fan_branch,
        "belief_before_hash": belief_hash(previous_belief),
        "belief_after_hash": belief_hash(updated_belief),
        "memory_lambda": float(updated_belief.memory_lambda),
        "observation_count": int(updated_belief.observation_count),
        "source_episode_id": str(source_episode_id),
        "belief_update_status": str(belief_update_status),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="ascii")
    return payload


def belief_hash(belief: BeliefGrid) -> str:
    payload = json.dumps(belief.as_dict(), sort_keys=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _grid(
    fan_branch: str,
    x_edges_m: tuple[float, ...],
    y_edges_m: tuple[float, ...],
    values: np.ndarray,
    memory_lambda: float,
    observation_count: int,
    belief_grid_id: str,
) -> BeliefGrid:
    return BeliefGrid(
        belief_grid_id=belief_grid_id,
        fan_branch=fan_branch,
        x_edges_m=tuple(float(item) for item in x_edges_m),
        y_edges_m=tuple(float(item) for item in y_edges_m),
        values_m_s=tuple(tuple(float(v) for v in row) for row in np.asarray(values, dtype=float)),
        memory_lambda=float(memory_lambda),
        observation_count=int(observation_count),
    )


def _belief_id(fan_branch: str, values: np.ndarray, observation_count: int, memory_lambda: float) -> str:
    digest = hashlib.sha256(np.asarray(values, dtype=float).round(9).tobytes()).hexdigest()[:12]
    return f"belief_{fan_branch}_n{int(observation_count):04d}_l{int(round(float(memory_lambda) * 100)):03d}_{digest}"


def _validate_fan_branch(fan_branch: str) -> None:
    if str(fan_branch) not in {"single_fan_branch", "four_fan_branch"}:
        raise ValueError("fan_branch must be single_fan_branch or four_fan_branch.")


def _observation_value(row: pd.Series, updraft_model: object | None, energy_residual: float) -> float:
    for column in ("w_lift_m_s", "w_wing_mean_m_s", "centre_wind_m_s"):
        if column in row and np.isfinite(_float(row[column])):
            return _float(row[column])
    if updraft_model is not None and hasattr(updraft_model, "velocity_w"):
        value = updraft_model.velocity_w(np.array([_float(row["x_w_m"]), _float(row["y_w_m"]), _float(row.get("z_w_m", 0.0))]))
        return float(np.asarray(value, dtype=float).reshape(-1)[2])
    return float(energy_residual)


def _bin_index(value: float, edges: tuple[float, ...]) -> int | None:
    if not np.isfinite(value):
        return None
    index = int(np.searchsorted(np.asarray(edges, dtype=float), float(value), side="right") - 1)
    if index < 0 or index >= len(edges) - 1:
        return None
    return index


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
