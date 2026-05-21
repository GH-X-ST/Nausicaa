from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from episodic_lift_belief import BeliefGrid, score_primitive_objective


@dataclass(frozen=True)
class FixedGatePolicy:
    policy_id: str
    memory_lambda: float | None
    uses_online_update: bool
    uses_static_map: bool
    oracle_only: bool
    description: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


POLICIES = (
    FixedGatePolicy("no_memory_baseline", 0.0, False, False, False, "No carry-over between launch episodes."),
    FixedGatePolicy("static_measured_map_baseline", None, False, True, False, "Uses a fixed measured map without online episode updates."),
    FixedGatePolicy("episodic_memory_policy", 0.5, True, False, False, "Uses decaying episode observations to rank primitive objectives."),
    FixedGatePolicy("oracle_simulation_upper_bound_optional", None, False, False, True, "Simulation-only upper bound; not a hardware policy."),
)


def policy_table() -> pd.DataFrame:
    return pd.DataFrame([policy.as_dict() for policy in POLICIES])


def policy_definition(policy_id: str) -> dict[str, object]:
    rows = policy_table()
    match = rows[rows["policy_id"].astype(str) == str(policy_id)]
    if match.empty:
        raise ValueError(f"unknown fixed-gate policy_id: {policy_id!r}.")
    return match.iloc[0].to_dict()


def candidate_objectives(
    *,
    policy_id: str,
    current_state: np.ndarray,
    belief: BeliefGrid | None,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Return objective scores for candidates without producing surface commands."""

    policy = policy_definition(policy_id)
    if candidates.empty:
        return pd.DataFrame(columns=[*candidates.columns, "policy_id", "candidate_objective_score"])
    rows = []
    for _, row in candidates.iterrows():
        candidate = row.to_dict()
        if bool(policy["oracle_only"]):
            score = _float(candidate.get("oracle_score", candidate.get("energy_residual_m", 0.0)))
        elif belief is None or str(policy_id) == "no_memory_baseline":
            score = _float(candidate.get("energy_residual_m", 0.0)) + 0.05 * _float(candidate.get("dwell_time_s", 0.0))
        else:
            score = score_primitive_objective(current_state, belief, candidate)
        rows.append({**candidate, "policy_id": str(policy_id), "candidate_objective_score": float(score)})
    return pd.DataFrame(rows)


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
