from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

from env_ctx import (
    EnvironmentContext,
    context_feature_vector_json,
    environment_context_row,
)
from prim_cat import PrimitiveDefinition, primitive_parameters_json
from state_contract import as_state_vector


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Rollout schema
# 2) Smoke rollout evaluation
# 3) Serialisation helpers
# =============================================================================


# =============================================================================
# 1) Rollout Schema
# =============================================================================
OUTCOME_CLASSES = ("accepted", "weak", "failed", "rejected", "blocked")
ROLLOUT_EVIDENCE_COLUMNS = (
    "rollout_id",
    "environment_id",
    "W_layer",
    "initial_state_vector",
    "context_feature_vector",
    "primitive_id",
    "primitive_family",
    "primitive_parameters",
    "controller_mode",
    "feedback_mode",
    "latency_case",
    "accepted",
    "outcome_class",
    "energy_residual_m",
    "lift_dwell_time_s",
    "minimum_wall_margin_m",
    "minimum_speed_m_s",
    "exit_state_vector",
    "termination_cause",
    "failure_label",
    "claim_status",
)


@dataclass(frozen=True)
class RolloutConfig:
    W_layer: str = "W0"
    dt_s: float = 0.02
    minimum_speed_m_s: float = 3.0
    wall_margin_reserve_m: float = 0.20


@dataclass(frozen=True)
class RolloutEvidence:
    rollout_id: str
    environment_id: str
    W_layer: str
    initial_state_vector: str
    context_feature_vector: str
    primitive_id: str
    primitive_family: str
    primitive_parameters: str
    controller_mode: str
    feedback_mode: str
    latency_case: str
    accepted: bool
    outcome_class: str
    energy_residual_m: float
    lift_dwell_time_s: float
    minimum_wall_margin_m: float
    minimum_speed_m_s: float
    exit_state_vector: str
    termination_cause: str
    failure_label: str
    claim_status: str


# =============================================================================
# 2) Smoke Rollout Evaluation
# =============================================================================
def simulate_primitive_rollout(
    *,
    rollout_id: str,
    initial_state: np.ndarray,
    context: EnvironmentContext,
    primitive: PrimitiveDefinition,
    config: RolloutConfig | None = None,
) -> RolloutEvidence:
    """Return deterministic smoke evidence for one primitive attempt."""

    cfg = config or RolloutConfig()
    state = as_state_vector(initial_state)
    if primitive.claim_status != "simulation_only":
        raise ValueError("R2-R5 rollout smoke only supports simulation_only primitives.")
    if float(primitive.finite_horizon_s) <= 0.0:
        raise ValueError("primitive finite_horizon_s must be positive.")

    speed_m_s = float(np.linalg.norm(state[6:9]))
    energy_residual_m = _smoke_energy_residual(context, primitive)
    lift_dwell_time_s = (
        float(primitive.finite_horizon_s)
        if context.w_wing_mean_m_s > 0.05 and context.wall_margin_m > 0.0
        else 0.0
    )
    minimum_wall_margin_m = float(
        context.wall_margin_m - _primitive_wall_reserve_m(primitive, cfg)
    )
    minimum_speed_m_s = float(speed_m_s + 0.20 * energy_residual_m)
    outcome_class, termination_cause, failure_label = _classify_smoke_outcome(
        context=context,
        energy_residual_m=energy_residual_m,
        minimum_wall_margin_m=minimum_wall_margin_m,
        minimum_speed_m_s=minimum_speed_m_s,
        minimum_speed_required_m_s=float(cfg.minimum_speed_m_s),
    )
    exit_state = state.copy()
    exit_state[0] += max(speed_m_s, 0.0) * float(primitive.finite_horizon_s) * 0.08
    exit_state[2] += energy_residual_m
    accepted = outcome_class == "accepted"
    return RolloutEvidence(
        rollout_id=str(rollout_id),
        environment_id=context.environment_id,
        W_layer=str(cfg.W_layer),
        initial_state_vector=_vector_json(state),
        context_feature_vector=context_feature_vector_json(context),
        primitive_id=primitive.primitive_id,
        primitive_family=primitive.primitive_family,
        primitive_parameters=primitive_parameters_json(primitive),
        controller_mode=primitive.controller_mode,
        feedback_mode=primitive.feedback_mode,
        latency_case=context.latency_case,
        accepted=accepted,
        outcome_class=outcome_class,
        energy_residual_m=energy_residual_m,
        lift_dwell_time_s=lift_dwell_time_s,
        minimum_wall_margin_m=minimum_wall_margin_m,
        minimum_speed_m_s=minimum_speed_m_s,
        exit_state_vector=_vector_json(exit_state),
        termination_cause=termination_cause,
        failure_label=failure_label,
        claim_status=primitive.claim_status,
    )


def _smoke_energy_residual(
    context: EnvironmentContext,
    primitive: PrimitiveDefinition,
) -> float:
    family_bias = {
        "glide": -0.05,
        "recovery": 0.02,
        "lift_entry": 0.04,
        "lift_dwell": 0.06,
        "mild_turn": -0.02,
        "energy_retaining_bank": 0.01,
        "safe_exit": -0.01,
    }.get(primitive.primitive_family, -0.03)
    value = (
        0.35 * float(context.w_wing_mean_m_s)
        + 0.20 * float(context.lift_score)
        - 0.08
        + family_bias
    ) * float(primitive.finite_horizon_s)
    return float(value)


def _primitive_wall_reserve_m(
    primitive: PrimitiveDefinition,
    config: RolloutConfig,
) -> float:
    reserve = float(config.wall_margin_reserve_m)
    if primitive.primitive_family in {"lift_dwell", "mild_turn", "energy_retaining_bank"}:
        reserve += 0.10
    return reserve


def _classify_smoke_outcome(
    *,
    context: EnvironmentContext,
    energy_residual_m: float,
    minimum_wall_margin_m: float,
    minimum_speed_m_s: float,
    minimum_speed_required_m_s: float,
) -> tuple[str, str, str]:
    if context.floor_margin_m < 0.0 or context.ceiling_margin_m < 0.0:
        return "rejected", "safety_volume_exit", "true_safety_violation"
    if minimum_wall_margin_m < 0.0:
        return "rejected", "wall_margin_stop", "wall_violation"
    if minimum_speed_m_s < minimum_speed_required_m_s:
        return "blocked", "low_speed", "speed_low"
    if energy_residual_m >= 0.05:
        return "accepted", "controlled_finish", "success"
    if energy_residual_m >= -0.03:
        return "weak", "weak_energy_result", "model_boundary_only"
    return "failed", "terminal_recovery_limited", "terminal_recovery_limited"


# =============================================================================
# 3) Serialisation Helpers
# =============================================================================
def rollout_evidence_row(evidence: RolloutEvidence) -> dict[str, object]:
    """Return one CSV-ready rollout evidence row."""

    row = asdict(evidence)
    return {name: row[name] for name in ROLLOUT_EVIDENCE_COLUMNS}


def rollout_with_context_row(evidence: RolloutEvidence, context: EnvironmentContext) -> dict[str, object]:
    """Return evidence plus expanded context fields for smoke archive partitions."""

    row = rollout_evidence_row(evidence)
    row.update({f"context_{key}": value for key, value in environment_context_row(context).items()})
    return row


def _vector_json(values: np.ndarray) -> str:
    return json.dumps([float(value) for value in values], separators=(",", ":"))
