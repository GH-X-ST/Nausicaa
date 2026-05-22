from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from env_ctx import EnvironmentContext
from prim_cat import PrimitiveDefinition, active_primitive_catalogue
from prim_model import (
    PrimitiveOutcomeModel,
    PrimitiveOutcomePrediction,
    predict_primitive_outcome,
)


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Selector dataclasses
# 2) Public selector
# 3) Viability and score helpers
# =============================================================================


# =============================================================================
# 1) Selector Dataclasses
# =============================================================================
@dataclass(frozen=True)
class PrimitiveCandidateDecision:
    primitive_id: str
    governor_mode: str
    viable: bool
    rejection_reason: str
    score: float
    probability_accepted: float
    probability_weak: float
    probability_failed: float
    probability_rejected: float
    probability_boundary_terminal: float
    probability_blocked: float
    probability_continuation_success: float
    probability_terminal_useful: float
    predicted_energy_residual_m: float
    predicted_lift_dwell_time_s: float
    predicted_minimum_wall_margin_m: float
    predicted_continuation_margin_m: float
    uncertainty: float


@dataclass(frozen=True)
class PrimitiveSelectionResult:
    selected_primitive_id: str
    governor_mode: str
    decision_status: str
    candidate_count: int
    viable_count: int
    decisions: tuple[PrimitiveCandidateDecision, ...]
    claim_status: str = "simulation_only_model_scaffold"


# =============================================================================
# 2) Public Selector
# =============================================================================
def select_primitive(
    *,
    context: EnvironmentContext,
    model: PrimitiveOutcomeModel,
    catalogue: tuple[PrimitiveDefinition, ...] | None = None,
    current_state: np.ndarray | None = None,
    governor_mode: str = "continuation",
    min_acceptance_probability: float = 0.20,
    max_uncertainty: float = 4.0,
    min_wall_margin_m: float = 0.10,
    min_speed_margin_m_s: float = 0.0,
    min_attitude_margin_rad: float = 0.0,
) -> PrimitiveSelectionResult:
    """Select a primitive through context features, model predictions, and viability."""

    if governor_mode not in {"continuation", "terminal_episode"}:
        raise ValueError("governor_mode must be 'continuation' or 'terminal_episode'.")
    primitives = catalogue or active_primitive_catalogue()
    decisions = tuple(
        _candidate_decision(
            context=context,
            prediction=predict_primitive_outcome(
                model,
                context,
                primitive,
                state=current_state,
                governor_mode=governor_mode,
            ),
            governor_mode=governor_mode,
            min_acceptance_probability=float(min_acceptance_probability),
            max_uncertainty=float(max_uncertainty),
            min_wall_margin_m=float(min_wall_margin_m),
            min_speed_margin_m_s=float(min_speed_margin_m_s),
            min_attitude_margin_rad=float(min_attitude_margin_rad),
        )
        for primitive in primitives
    )
    viable = [decision for decision in decisions if decision.viable]
    if not viable:
        if _hard_context_rejection(
            context=context,
            min_speed_margin_m_s=min_speed_margin_m_s,
            min_attitude_margin_rad=min_attitude_margin_rad,
        ):
            return PrimitiveSelectionResult(
                selected_primitive_id="",
                governor_mode=governor_mode,
                decision_status="blocked_no_viability_checked_fallback",
                candidate_count=len(decisions),
                viable_count=0,
                decisions=decisions,
            )
        return PrimitiveSelectionResult(
            selected_primitive_id="safe_exit_or_recovery_handoff",
            governor_mode=governor_mode,
            decision_status="recovery_handoff_no_viable_primitive",
            candidate_count=len(decisions),
            viable_count=0,
            decisions=decisions,
        )
    selected = max(viable, key=lambda decision: decision.score)
    return PrimitiveSelectionResult(
        selected_primitive_id=selected.primitive_id,
        governor_mode=governor_mode,
        decision_status="selected_viable_primitive",
        candidate_count=len(decisions),
        viable_count=len(viable),
        decisions=decisions,
    )


# =============================================================================
# 3) Viability and Score Helpers
# =============================================================================
def _candidate_decision(
    *,
    context: EnvironmentContext,
    prediction: PrimitiveOutcomePrediction,
    governor_mode: str,
    min_acceptance_probability: float,
    max_uncertainty: float,
    min_wall_margin_m: float,
    min_speed_margin_m_s: float,
    min_attitude_margin_rad: float,
) -> PrimitiveCandidateDecision:
    rejection_reason = _rejection_reason(
        context=context,
        prediction=prediction,
        governor_mode=governor_mode,
        min_acceptance_probability=min_acceptance_probability,
        max_uncertainty=max_uncertainty,
        min_wall_margin_m=min_wall_margin_m,
        min_speed_margin_m_s=min_speed_margin_m_s,
        min_attitude_margin_rad=min_attitude_margin_rad,
    )
    viable = rejection_reason == ""
    score = _selection_score(prediction=prediction, governor_mode=governor_mode)
    return PrimitiveCandidateDecision(
        primitive_id=prediction.primitive_id,
        governor_mode=governor_mode,
        viable=viable,
        rejection_reason=rejection_reason,
        score=float(score if viable else float("-inf")),
        probability_accepted=float(prediction.probability_accepted),
        probability_weak=float(prediction.probability_weak),
        probability_failed=float(prediction.probability_failed),
        probability_rejected=float(prediction.probability_rejected),
        probability_boundary_terminal=float(prediction.probability_boundary_terminal),
        probability_blocked=float(prediction.probability_blocked),
        probability_continuation_success=float(
            prediction.probability_continuation_success
        ),
        probability_terminal_useful=float(prediction.probability_terminal_useful),
        predicted_energy_residual_m=float(prediction.predicted_energy_residual_m),
        predicted_lift_dwell_time_s=float(prediction.predicted_lift_dwell_time_s),
        predicted_minimum_wall_margin_m=float(
            prediction.predicted_minimum_wall_margin_m
        ),
        predicted_continuation_margin_m=float(prediction.predicted_continuation_margin_m),
        uncertainty=float(prediction.uncertainty),
    )


def _rejection_reason(
    *,
    context: EnvironmentContext,
    prediction: PrimitiveOutcomePrediction,
    governor_mode: str,
    min_acceptance_probability: float,
    max_uncertainty: float,
    min_wall_margin_m: float,
    min_speed_margin_m_s: float,
    min_attitude_margin_rad: float,
) -> str:
    if context.wall_margin_m < min_wall_margin_m:
        return "context_wall_margin_low"
    if context.floor_margin_m < 0.0 or context.ceiling_margin_m < 0.0:
        return "context_vertical_safety_violation"
    if context.speed_margin_m_s < min_speed_margin_m_s:
        return "context_speed_margin_low"
    if context.attitude_margin_rad < min_attitude_margin_rad:
        return "context_attitude_margin_low"
    if prediction.uncertainty > max_uncertainty:
        return "model_uncertainty_high"
    if prediction.probability_rejected + prediction.probability_blocked > 0.65:
        return "predicted_rejection_or_block_high"
    if governor_mode == "continuation":
        if prediction.probability_boundary_terminal > 0.35:
            return "predicted_boundary_terminal_not_continuation_valid"
        if prediction.predicted_minimum_wall_margin_m < min_wall_margin_m:
            return "predicted_wall_margin_low"
        if prediction.predicted_continuation_margin_m < min_wall_margin_m:
            return "predicted_continuation_margin_low"
        if prediction.probability_continuation_success < min_acceptance_probability:
            return "continuation_success_probability_low"
        return ""
    if prediction.probability_boundary_terminal > 0.45:
        if prediction.probability_terminal_useful < 0.25:
            return "terminal_utility_low"
        return ""
    if prediction.probability_continuation_success < min_acceptance_probability:
        if prediction.probability_terminal_useful < 0.25:
            return "terminal_or_continuation_utility_low"
    return ""


def _hard_context_rejection(
    *,
    context: EnvironmentContext,
    min_speed_margin_m_s: float,
    min_attitude_margin_rad: float,
) -> bool:
    return bool(
        context.floor_margin_m < 0.0
        or context.ceiling_margin_m < 0.0
        or context.speed_margin_m_s < min(float(min_speed_margin_m_s), -0.5)
        or context.attitude_margin_rad < min(float(min_attitude_margin_rad), -0.1)
    )


def _selection_score(
    *,
    prediction: PrimitiveOutcomePrediction,
    governor_mode: str,
) -> float:
    if governor_mode == "terminal_episode":
        return float(
            0.65 * prediction.probability_terminal_useful
            + 0.25 * prediction.probability_boundary_terminal
            + 0.12 * prediction.predicted_lift_dwell_time_s
            + 0.10 * prediction.predicted_energy_residual_m
            - 0.30 * prediction.probability_rejected
            - 0.40 * prediction.probability_blocked
            - 0.05 * prediction.uncertainty
        )
    return float(
        prediction.probability_continuation_success
        + 0.15 * prediction.predicted_energy_residual_m
        + 0.05 * prediction.predicted_continuation_margin_m
        - 0.40 * prediction.probability_boundary_terminal
        - 0.05 * prediction.uncertainty
    )


def primitive_selection_row(result: PrimitiveSelectionResult) -> dict[str, object]:
    """Return a compact selector audit row."""

    row = asdict(result)
    row["decisions"] = len(result.decisions)
    return row
