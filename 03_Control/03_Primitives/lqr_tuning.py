from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

from lqr_controller import LQRWeightSpec, synthesize_lqr_controller
from prim_cat import PrimitiveDefinition, active_primitive_catalogue


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Tuning contracts
# 2) Candidate generation
# 3) Ranking helpers
# =============================================================================


TUNING_METHOD_VERSION = "grouped_log_scaled_diagonal_qr_v1"
PREFERRED_CANDIDATES_PER_PRIMITIVE = (16, 32)
FALLBACK_CANDIDATES_PER_PRIMITIVE = 8
PREFERRED_PAIRED_TESTS_PER_CANDIDATE = (50, 100)
FALLBACK_PAIRED_TESTS_PER_CANDIDATE = 25
HARD_GATE_LABELS = (
    "finite_state",
    "safe_volume",
    "floor_ceiling_margin",
    "minimum_speed",
    "surface_limits",
    "valid_lqr_synthesis",
    "sampled_data_stability",
    "surrogate_ready",
)
SOFT_OBJECTIVE_TERMS = (
    "energy_residual_m",
    "lift_dwell_time_s",
    "wall_margin_m",
    "saturation_fraction",
    "uncertainty_m_s",
    "latency_degradation",
    "terminal_use_label",
)


@dataclass(frozen=True)
class LQRTuningCandidate:
    primitive_id: str
    candidate_index: int
    controller_id: str
    lqr_Q_weights_json: str
    lqr_R_weights_json: str
    lqr_gain_checksum: str
    lqr_synthesis_status: str
    tuning_stage: str
    tuning_method: str = TUNING_METHOD_VERSION
    raw_K_tuning_allowed: bool = False


@dataclass(frozen=True)
class LQRTuningSchedule:
    primitive_count: int
    candidate_count: int
    paired_tests_per_candidate: int
    planned_rows: int
    fallback_mode: bool
    hard_gates_json: str
    soft_objective_terms_json: str
    method_version: str = TUNING_METHOD_VERSION


def candidate_weight_specs(
    *,
    primitive_id: str,
    candidate_count: int = 16,
    tuning_stage: str = "W0_W1",
) -> tuple[LQRWeightSpec, ...]:
    """Return deterministic grouped log-scaled diagonal Q/R candidates."""

    if int(candidate_count) <= 0:
        raise ValueError("candidate_count must be positive.")
    scales = np.asarray([0.5, 1.0, 2.0, 4.0], dtype=float)
    candidates: list[LQRWeightSpec] = []
    cursor = 0
    for q_att in scales:
        for q_vel in scales:
            for q_rate in (1.0, 2.0):
                r_scale = scales[cursor % len(scales)]
                cursor += 1
                candidates.append(
                    LQRWeightSpec(
                        q_attitude=float(4.0 * q_att),
                        q_velocity=float(2.0 * q_vel),
                        q_rates=float(1.6 * q_rate),
                        q_surfaces=0.15,
                        r_aileron=float(1.0 * r_scale),
                        r_elevator=float(0.9 * r_scale),
                        r_rudder=float(1.1 * r_scale),
                        tuning_stage=tuning_stage,
                        weight_label=f"{primitive_id}_qr_{len(candidates):03d}",
                    )
                )
                if len(candidates) >= int(candidate_count):
                    return tuple(candidates)
    return tuple(candidates[: int(candidate_count)])


def lqr_tuning_schedule(
    *,
    candidate_count: int = 16,
    paired_tests_per_candidate: int = 50,
    fallback_mode: bool = False,
) -> LQRTuningSchedule:
    primitive_count = len(active_primitive_catalogue())
    rows = primitive_count * int(candidate_count) * int(paired_tests_per_candidate) * 2
    return LQRTuningSchedule(
        primitive_count=primitive_count,
        candidate_count=int(candidate_count),
        paired_tests_per_candidate=int(paired_tests_per_candidate),
        planned_rows=int(rows),
        fallback_mode=bool(fallback_mode),
        hard_gates_json=json.dumps(HARD_GATE_LABELS, separators=(",", ":")),
        soft_objective_terms_json=json.dumps(SOFT_OBJECTIVE_TERMS, separators=(",", ":")),
    )


def tuning_candidates_for_primitive(
    primitive: PrimitiveDefinition,
    *,
    candidate_count: int = 16,
) -> tuple[LQRTuningCandidate, ...]:
    rows = []
    for index, weight_spec in enumerate(
        candidate_weight_specs(
            primitive_id=primitive.primitive_id,
            candidate_count=candidate_count,
        )
    ):
        controller = synthesize_lqr_controller(primitive, weight_spec=weight_spec)
        rows.append(
            LQRTuningCandidate(
                primitive_id=primitive.primitive_id,
                candidate_index=int(index),
                controller_id=controller.controller_id,
                lqr_Q_weights_json=controller.lqr_Q_weights_json,
                lqr_R_weights_json=controller.lqr_R_weights_json,
                lqr_gain_checksum=controller.lqr_gain_checksum,
                lqr_synthesis_status=controller.lqr_synthesis_status,
                tuning_stage=controller.tuning_stage,
            )
        )
    return tuple(rows)


def tuning_candidate_row(candidate: LQRTuningCandidate) -> dict[str, object]:
    return asdict(candidate)


def tuning_schedule_row(schedule: LQRTuningSchedule) -> dict[str, object]:
    return asdict(schedule)

