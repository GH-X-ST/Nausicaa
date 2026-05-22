from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from state_contract import STATE_INDEX, as_state_vector


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Belief dataclasses
# 2) Public belief update/query helpers
# 3) Serialisation helpers
# =============================================================================


BELIEF_LAMBDA_VALUES = (0.0, 0.5, 0.8, 0.95)


# =============================================================================
# 1) Belief Dataclasses
# =============================================================================
@dataclass(frozen=True)
class LiftBeliefGrid:
    x_edges_m: tuple[float, ...]
    y_edges_m: tuple[float, ...]
    values: tuple[tuple[float, ...], ...]
    lambda_value: float
    update_count: int = 0


@dataclass(frozen=True)
class LiftObservation:
    x_w_m: float
    y_w_m: float
    lift_evidence_m_s: float
    episode_id: str = ""


# =============================================================================
# 2) Public Belief Update/Query Helpers
# =============================================================================
def initial_belief(
    *,
    lambda_value: float,
    x_edges_m: tuple[float, ...] = (1.2, 2.55, 3.9, 5.25, 6.6),
    y_edges_m: tuple[float, ...] = (0.0, 1.1, 2.2, 3.3, 4.4),
) -> LiftBeliefGrid:
    """Return a compact zero-initialised lift belief grid."""

    if float(lambda_value) not in BELIEF_LAMBDA_VALUES:
        raise ValueError("lambda_value must be one of the retained belief values.")
    shape = (len(x_edges_m) - 1, len(y_edges_m) - 1)
    values = tuple(tuple(0.0 for _ in range(shape[1])) for _ in range(shape[0]))
    return LiftBeliefGrid(
        x_edges_m=tuple(float(value) for value in x_edges_m),
        y_edges_m=tuple(float(value) for value in y_edges_m),
        values=values,
        lambda_value=float(lambda_value),
    )


def update_belief(
    belief: LiftBeliefGrid,
    observation: LiftObservation,
    lambda_: float | None = None,
) -> LiftBeliefGrid:
    """Apply exponential retention and one local lift observation."""

    retain = float(belief.lambda_value if lambda_ is None else lambda_)
    if retain not in BELIEF_LAMBDA_VALUES:
        raise ValueError("lambda_ must be one of the retained belief values.")
    values = np.asarray(belief.values, dtype=float)
    values *= retain
    ix = _bin_index(float(observation.x_w_m), belief.x_edges_m)
    iy = _bin_index(float(observation.y_w_m), belief.y_edges_m)
    values[ix, iy] += (1.0 - retain) * float(observation.lift_evidence_m_s)
    return LiftBeliefGrid(
        x_edges_m=belief.x_edges_m,
        y_edges_m=belief.y_edges_m,
        values=tuple(tuple(float(item) for item in row) for row in values),
        lambda_value=retain,
        update_count=int(belief.update_count) + 1,
    )


def query_belief_features(state: np.ndarray, belief: LiftBeliefGrid) -> dict[str, float]:
    """Return compact local belief features for a state."""

    x = as_state_vector(state)
    values = np.asarray(belief.values, dtype=float)
    ix = _bin_index(float(x[STATE_INDEX["x_w"]]), belief.x_edges_m)
    iy = _bin_index(float(x[STATE_INDEX["y_w"]]), belief.y_edges_m)
    return {
        "belief_local_lift_m_s": float(values[ix, iy]),
        "belief_mean_lift_m_s": float(np.mean(values)),
        "belief_max_lift_m_s": float(np.max(values)),
        "belief_update_count": float(belief.update_count),
        "belief_lambda": float(belief.lambda_value),
    }


def lift_observation_from_rollout_row(row: dict[str, object]) -> LiftObservation:
    """Build a belief observation from one rollout or episode row."""

    return LiftObservation(
        x_w_m=float(row.get("initial_x_w", 0.0)),
        y_w_m=float(row.get("initial_y_w", 0.0)),
        lift_evidence_m_s=float(row.get("context_w_wing_mean_m_s", 0.0)),
        episode_id=str(row.get("episode_id", "")),
    )


# =============================================================================
# 3) Serialisation Helpers
# =============================================================================
def belief_snapshot_row(belief: LiftBeliefGrid, *, label: str = "") -> dict[str, object]:
    row = asdict(belief)
    row["label"] = str(label)
    row["values"] = ";".join(
        ",".join(f"{float(value):.6f}" for value in line)
        for line in belief.values
    )
    row["x_edges_m"] = ";".join(f"{float(value):.6f}" for value in belief.x_edges_m)
    row["y_edges_m"] = ";".join(f"{float(value):.6f}" for value in belief.y_edges_m)
    return row


def _bin_index(value: float, edges: tuple[float, ...]) -> int:
    edge_array = np.asarray(edges, dtype=float)
    return int(np.clip(np.searchsorted(edge_array, float(value), side="right") - 1, 0, edge_array.size - 2))
