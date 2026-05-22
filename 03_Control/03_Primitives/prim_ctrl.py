from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from command_contract import (
    clip_normalised_command,
    normalised_command_to_surface_rad,
)
from env_ctx import EnvironmentContext
from prim_cat import PrimitiveDefinition
from state_contract import STATE_INDEX, as_state_vector


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Control dataclasses
# 2) Public feedback helpers
# 3) Local bounded feedback laws
# =============================================================================


# =============================================================================
# 1) Control Dataclasses
# =============================================================================
@dataclass(frozen=True)
class PrimitiveControlContext:
    state_vector: np.ndarray
    environment_context: EnvironmentContext
    time_in_primitive_s: float = 0.0


@dataclass(frozen=True)
class PrimitiveControlCommand:
    primitive_id: str
    feedback_mode: str
    command_norm: tuple[float, float, float]
    command_rad: tuple[float, float, float]
    saturation_applied: bool
    command_units: str = "normalised_command_and_radian_surface_targets"
    sign_convention: str = (
        "positive aileron rolls right wing down; positive elevator pitches nose up; "
        "positive rudder yaws nose right"
    )


# =============================================================================
# 2) Public Feedback Helpers
# =============================================================================
def feedback_mode_for_primitive(primitive: PrimitiveDefinition) -> str:
    """Return the active local-feedback mode for one primitive."""

    return {
        "glide": "pitch_speed_preserving_feedback",
        "recovery": "attitude_speed_recovery_feedback",
        "lift_entry": "lift_score_entry_feedback",
        "lift_dwell_arc": "bounded_lift_dwell_arc_feedback",
        "mild_turn_left": "bounded_left_turn_feedback",
        "mild_turn_right": "bounded_right_turn_feedback",
        "energy_retaining_bank": "energy_retaining_bank_feedback",
        "safe_exit_or_recovery_handoff": "recovery_handoff_feedback",
    }.get(primitive.primitive_id, "bounded_neutral_feedback")


def primitive_reference(
    primitive: PrimitiveDefinition,
    control_context: PrimitiveControlContext,
) -> dict[str, float | str]:
    """Return physical reference values used by the bounded local controller."""

    context = control_context.environment_context
    target_bank_rad = {
        "lift_dwell_arc": 0.30 * _turn_sign_from_lift(context),
        "mild_turn_left": -0.25,
        "mild_turn_right": 0.25,
        "energy_retaining_bank": 0.18 * _turn_sign_from_lift(context),
    }.get(primitive.primitive_id, 0.0)
    target_pitch_rad = {
        "glide": -0.04,
        "recovery": 0.02,
        "lift_entry": 0.03 + 0.04 * float(context.lift_score),
        "lift_dwell_arc": 0.02 + 0.03 * float(context.lift_score),
        "energy_retaining_bank": -0.02 + 0.03 * float(context.lift_score),
        "safe_exit_or_recovery_handoff": 0.04,
    }.get(primitive.primitive_id, 0.0)
    return {
        "primitive_id": primitive.primitive_id,
        "target_bank_rad": float(np.clip(target_bank_rad, -0.35, 0.35)),
        "target_pitch_rad": float(np.clip(target_pitch_rad, -0.12, 0.12)),
        "target_speed_margin_m_s": 0.20,
        "lift_score": float(context.lift_score),
    }


def primitive_command_norm(
    primitive: PrimitiveDefinition,
    control_context: PrimitiveControlContext,
) -> PrimitiveControlCommand:
    """Return a bounded normalised command from state and local context feedback."""

    state = as_state_vector(control_context.state_vector)
    reference = primitive_reference(primitive, control_context)
    raw_command = _feedback_command_norm(
        primitive=primitive,
        state=state,
        context=control_context.environment_context,
        target_bank_rad=float(reference["target_bank_rad"]),
        target_pitch_rad=float(reference["target_pitch_rad"]),
    )
    clipped = clip_normalised_command(raw_command)
    command_rad = normalised_command_to_surface_rad(clipped)
    return PrimitiveControlCommand(
        primitive_id=primitive.primitive_id,
        feedback_mode=feedback_mode_for_primitive(primitive),
        command_norm=tuple(float(value) for value in clipped),
        command_rad=tuple(float(value) for value in command_rad),
        saturation_applied=bool(np.any(np.abs(raw_command - clipped) > 1e-12)),
    )


def primitive_control_command_row(command: PrimitiveControlCommand) -> dict[str, object]:
    """Return a compact audit row for a feedback command."""

    row = asdict(command)
    row["command_norm"] = ";".join(f"{value:.9f}" for value in command.command_norm)
    row["command_rad"] = ";".join(f"{value:.9f}" for value in command.command_rad)
    return row


# =============================================================================
# 3) Local Bounded Feedback Laws
# =============================================================================
def _feedback_command_norm(
    *,
    primitive: PrimitiveDefinition,
    state: np.ndarray,
    context: EnvironmentContext,
    target_bank_rad: float,
    target_pitch_rad: float,
) -> np.ndarray:
    phi_rad = float(state[STATE_INDEX["phi"]])
    theta_rad = float(state[STATE_INDEX["theta"]])
    p_rad_s = float(state[STATE_INDEX["p"]])
    q_rad_s = float(state[STATE_INDEX["q"]])
    r_rad_s = float(state[STATE_INDEX["r"]])
    speed_margin_m_s = float(context.speed_margin_m_s)
    lift_score = float(context.lift_score)

    bank_error_rad = float(target_bank_rad) - phi_rad
    pitch_error_rad = float(target_pitch_rad) - theta_rad
    speed_term = np.clip(0.08 - 0.05 * speed_margin_m_s, -0.20, 0.20)

    aileron = 1.20 * bank_error_rad - 0.08 * p_rad_s
    elevator = 1.60 * pitch_error_rad - 0.06 * q_rad_s + speed_term
    rudder = -0.04 * r_rad_s

    if primitive.primitive_id == "lift_entry":
        aileron += 0.08 * _turn_sign_from_lift(context)
        elevator += 0.05 * lift_score
    elif primitive.primitive_id == "lift_dwell_arc":
        aileron += 0.05 * _turn_sign_from_lift(context)
        elevator += 0.03 * lift_score
    elif primitive.primitive_id == "energy_retaining_bank":
        elevator -= 0.04 * max(-speed_margin_m_s, 0.0)
    elif primitive.primitive_id == "safe_exit_or_recovery_handoff":
        aileron += -0.15 * phi_rad
        elevator += 0.03
    elif primitive.primitive_id == "recovery":
        aileron += -0.20 * phi_rad
        elevator += -0.10 * theta_rad

    return np.asarray([aileron, elevator, rudder], dtype=float)


def _turn_sign_from_lift(context: EnvironmentContext) -> float:
    if abs(float(context.lift_direction_y)) > 0.0:
        return float(np.sign(context.lift_direction_y))
    return 1.0
