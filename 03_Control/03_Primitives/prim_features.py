from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from env_ctx import EnvironmentContext, context_feature_vector
from latency import latency_case_config
from prim_cat import PrimitiveDefinition
from state_contract import STATE_INDEX, STATE_SIZE


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Feature schema
# 2) Public feature builders
# 3) Row parsing helpers
# =============================================================================


# =============================================================================
# 1) Feature Schema
# =============================================================================
PRIMITIVE_FEATURE_SCHEMA_VERSION = "mixed_start_context_primitive_latency_uncertainty_v2"
PRIMITIVE_FEATURE_NAMES = (
    "start_state_family_code",
    "previous_primitive_status_code",
    "synthetic_time_since_launch_norm",
    "phi_norm",
    "theta_norm",
    "psi_norm",
    "speed_norm",
    "p_norm",
    "q_norm",
    "r_norm",
    "delta_a_norm",
    "delta_e_norm",
    "delta_r_norm",
    "primitive_horizon_s",
    "primitive_param_sum",
    "state_feedback_delay_s",
    "command_delay_s",
    "actuator_t50_s",
    "uncertainty_m_s",
    "terminal_mode_flag",
) + tuple(f"context_{index:02d}" for index in range(13))


@dataclass(frozen=True)
class PrimitiveFeatureRecord:
    feature_schema_version: str
    feature_names: tuple[str, ...]
    feature_vector: tuple[float, ...]


# =============================================================================
# 2) Public Feature Builders
# =============================================================================
def primitive_feature_record(
    *,
    state: np.ndarray,
    context: EnvironmentContext,
    primitive: PrimitiveDefinition,
    governor_mode: str = "continuation",
    start_state_family: str = "unknown",
    previous_primitive_status: str = "unknown",
    synthetic_time_since_launch_s: float = 0.0,
) -> PrimitiveFeatureRecord:
    """Return the auditable primitive model feature vector."""

    x = np.asarray(state, dtype=float).reshape(STATE_SIZE)
    speed = float(np.linalg.norm(x[6:9]))
    latency = latency_case_config(context.latency_case)
    command_delay_s = float(latency.command_onset_delay_s + latency.command_transport_delay_s)
    primitive_params = _numeric_primitive_parameter_sum(primitive)
    values = (
        _category_code(start_state_family),
        _category_code(previous_primitive_status),
        float(np.clip(float(synthetic_time_since_launch_s) / 5.0, 0.0, 4.0)),
        float(x[STATE_INDEX["phi"]] / np.deg2rad(45.0)),
        float(x[STATE_INDEX["theta"]] / np.deg2rad(45.0)),
        float(x[STATE_INDEX["psi"]] / np.deg2rad(90.0)),
        float(speed / 8.0),
        float(x[STATE_INDEX["p"]]),
        float(x[STATE_INDEX["q"]]),
        float(x[STATE_INDEX["r"]]),
        float(x[STATE_INDEX["delta_a"]] / 0.5),
        float(x[STATE_INDEX["delta_e"]] / 0.5),
        float(x[STATE_INDEX["delta_r"]] / 0.5),
        float(primitive.finite_horizon_s),
        primitive_params,
        float(latency.state_feedback_delay_s),
        command_delay_s,
        float(latency.actuator_t50_s),
        float(context.w_local_uncertainty_m_s),
        1.0 if governor_mode == "terminal_episode" else 0.0,
    ) + context_feature_vector(context)
    return PrimitiveFeatureRecord(
        feature_schema_version=PRIMITIVE_FEATURE_SCHEMA_VERSION,
        feature_names=PRIMITIVE_FEATURE_NAMES,
        feature_vector=tuple(float(value) for value in values),
    )


def primitive_feature_vector_json(record: PrimitiveFeatureRecord) -> str:
    return json.dumps([float(value) for value in record.feature_vector], separators=(",", ":"))


def primitive_feature_row(record: PrimitiveFeatureRecord) -> dict[str, object]:
    row = {
        "feature_schema_version": record.feature_schema_version,
        "feature_names": ",".join(record.feature_names),
        "primitive_feature_vector": primitive_feature_vector_json(record),
    }
    row.update(
        {
            f"feature_{name}": float(value)
            for name, value in zip(record.feature_names, record.feature_vector, strict=True)
        }
    )
    return row


# =============================================================================
# 3) Row Parsing Helpers
# =============================================================================
def primitive_feature_vector_from_row(row: dict[str, object]) -> tuple[float, ...]:
    """Return model features from a rollout row, falling back to older context vector."""

    if row.get("primitive_feature_vector"):
        return _parse_vector(row["primitive_feature_vector"])
    if row.get("context_feature_vector"):
        return _parse_vector(row["context_feature_vector"])
    return ()


def _parse_vector(value: object) -> tuple[float, ...]:
    parsed = json.loads(value) if isinstance(value, str) else value
    vector = np.asarray(parsed, dtype=float).reshape(-1)
    if not np.all(np.isfinite(vector)):
        return ()
    return tuple(float(item) for item in vector)


def _numeric_primitive_parameter_sum(primitive: PrimitiveDefinition) -> float:
    total = 0.0
    for parameter in primitive.parameters:
        try:
            total += float(parameter.value)
        except (TypeError, ValueError):
            total += float(len(str(parameter.value))) * 0.01
    return float(total)


def _category_code(value: str) -> float:
    text = str(value)
    if not text:
        return 0.0
    total = 0
    for char in text:
        total = (total * 131 + ord(char)) % 1000
    return float(total) / 1000.0
