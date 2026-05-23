from __future__ import annotations

import json
from dataclasses import asdict, dataclass


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Primitive dataclasses
# 2) Active catalogue
# 3) Lookup and serialisation helpers
# =============================================================================


# =============================================================================
# 1) Primitive Dataclasses
# =============================================================================
@dataclass(frozen=True)
class PrimitiveParameter:
    name: str
    value: float | str
    unit: str
    description: str


@dataclass(frozen=True)
class PrimitiveDefinition:
    primitive_id: str
    primitive_family: str
    parameters: tuple[PrimitiveParameter, ...]
    entry_conditions: tuple[str, ...]
    controller_family: str
    controller_mode: str
    feedback_mode: str
    controller_id: str
    linearisation_source: str
    Q_weight_json: str
    R_weight_json: str
    finite_horizon_s: float
    exit_checks: tuple[str, ...]
    metrics_to_record: tuple[str, ...]
    failure_labels: tuple[str, ...]
    claim_status: str = "simulation_only"


ACTIVE_PRIMITIVE_IDS = (
    "glide",
    "recovery",
    "lift_entry",
    "lift_dwell_arc",
    "mild_turn_left",
    "mild_turn_right",
    "energy_retaining_bank",
    "safe_exit_or_recovery_handoff",
)

_COMMON_ENTRY = (
    "finite_state",
    "inside_true_safe_bounds",
    "minimum_speed_margin_positive",
    "attitude_margin_positive",
)
_COMMON_EXIT = (
    "finite_exit_state",
    "true_safe_bounds_preserved",
    "minimum_speed_preserved",
)
_COMMON_METRICS = (
    "energy_residual_m",
    "lift_dwell_time_s",
    "minimum_wall_margin_m",
    "minimum_speed_m_s",
    "termination_cause",
)
_COMMON_FAILURES = (
    "entry_set_violation",
    "true_safety_violation",
    "wall_violation",
    "floor_violation",
    "ceiling_violation",
    "speed_low",
    "latency_limited",
    "terminal_recovery_limited",
    "model_boundary_only",
)


# =============================================================================
# 2) Active Catalogue
# =============================================================================
def active_primitive_catalogue() -> tuple[PrimitiveDefinition, ...]:
    """Return the exact eight-label active LQR primitive catalogue."""

    return (
        _primitive(
            primitive_id="glide",
            primitive_family="glide",
            horizon_s=0.60,
            params=(("target_pitch_rad", -0.04, "rad", "small nose-down glide bias"),),
        ),
        _primitive(
            primitive_id="recovery",
            primitive_family="recovery",
            horizon_s=0.50,
            params=(("target_pitch_rad", 0.02, "rad", "level recovery pitch target"),),
        ),
        _primitive(
            primitive_id="lift_entry",
            primitive_family="lift_entry",
            horizon_s=0.70,
            params=(("lift_score_threshold", 0.10, "1", "minimum useful local lift score"),),
        ),
        _primitive(
            primitive_id="lift_dwell_arc",
            primitive_family="lift_dwell",
            horizon_s=0.80,
            params=(("arc_bank_rad", 0.30, "rad", "bounded dwell bank angle"),),
        ),
        _primitive(
            primitive_id="mild_turn_left",
            primitive_family="mild_turn",
            horizon_s=0.65,
            params=(("turn_direction", "left", "label", "left turn command sign"),),
        ),
        _primitive(
            primitive_id="mild_turn_right",
            primitive_family="mild_turn",
            horizon_s=0.65,
            params=(("turn_direction", "right", "label", "right turn command sign"),),
        ),
        _primitive(
            primitive_id="energy_retaining_bank",
            primitive_family="energy_retaining_bank",
            horizon_s=0.70,
            params=(("bank_limit_rad", 0.35, "rad", "energy-preserving bank limit"),),
        ),
        _primitive(
            primitive_id="safe_exit_or_recovery_handoff",
            primitive_family="safe_exit",
            horizon_s=0.45,
            params=(("handoff_margin_m", 0.30, "m", "minimum wall margin for exit handoff"),),
        ),
    )


def _primitive(
    *,
    primitive_id: str,
    primitive_family: str,
    horizon_s: float,
    params: tuple[tuple[str, float | str, str, str], ...],
) -> PrimitiveDefinition:
    return PrimitiveDefinition(
        primitive_id=primitive_id,
        primitive_family=primitive_family,
        parameters=tuple(
            PrimitiveParameter(name=name, value=value, unit=unit, description=description)
            for name, value, unit, description in params
        ),
        entry_conditions=_COMMON_ENTRY,
        controller_family="lqr",
        controller_mode="lqr_local_feedback",
        feedback_mode="lqr_state_feedback",
        controller_id=f"lqr_{primitive_id}_nominal",
        linearisation_source="straight_trim_plus_primitive_reference_bias",
        Q_weight_json=_nominal_q_weight_json(),
        R_weight_json=_nominal_r_weight_json(),
        finite_horizon_s=float(horizon_s),
        exit_checks=_COMMON_EXIT,
        metrics_to_record=_COMMON_METRICS,
        failure_labels=_COMMON_FAILURES,
        claim_status="simulation_only",
    )


# =============================================================================
# 3) Lookup and Serialisation Helpers
# =============================================================================
def primitive_by_id(primitive_id: str) -> PrimitiveDefinition:
    """Return one active primitive by ID."""

    for primitive in active_primitive_catalogue():
        if primitive.primitive_id == primitive_id:
            return primitive
    raise KeyError(f"unknown active primitive_id: {primitive_id}")


def primitive_parameters_json(primitive: PrimitiveDefinition) -> str:
    """Return compact JSON parameter metadata with units and descriptions."""

    payload = {
        parameter.name: {
            "value": parameter.value,
            "unit": parameter.unit,
            "description": parameter.description,
        }
        for parameter in primitive.parameters
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _nominal_q_weight_json() -> str:
    return json.dumps(
        {
            "grouping": "diagonal_grouped_log_scaled",
            "state_mask": [
                "phi",
                "theta",
                "psi",
                "u",
                "v",
                "w",
                "p",
                "q",
                "r",
                "delta_a",
                "delta_e",
                "delta_r",
            ],
            "source": "lqr_controller_default",
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _nominal_r_weight_json() -> str:
    return json.dumps(
        {
            "grouping": "diagonal_grouped_log_scaled",
            "command_names": ["delta_a_cmd", "delta_e_cmd", "delta_r_cmd"],
            "source": "lqr_controller_default",
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def primitive_definition_row(primitive: PrimitiveDefinition) -> dict[str, object]:
    """Return a CSV-ready primitive catalogue row."""

    row = asdict(primitive)
    row["parameters"] = primitive_parameters_json(primitive)
    row["entry_conditions"] = ";".join(primitive.entry_conditions)
    row["exit_checks"] = ";".join(primitive.exit_checks)
    row["metrics_to_record"] = ";".join(primitive.metrics_to_record)
    row["failure_labels"] = ";".join(primitive.failure_labels)
    return row
