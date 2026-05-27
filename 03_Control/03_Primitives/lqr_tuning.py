from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

from lqr_controller import LQR_SYNTHESIS_SOLVED, LQRWeightSpec, default_lqr_weight_spec, synthesize_lqr_controller
from prim_cat import PrimitiveDefinition, active_primitive_catalogue


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) W01 tuning contracts
# 2) Candidate generation
# 3) Summary helpers
# =============================================================================


# =============================================================================
# 1) W01 Tuning Contracts
# =============================================================================
W01_TUNING_METHOD_VERSION = "w01_grouped_qr_primitive_variant_v2_launch_wide"
PREFERRED_CANDIDATES_PER_PRIMITIVE = (16, 32)
FALLBACK_CANDIDATES_PER_PRIMITIVE = 8
PREFERRED_PAIRED_TESTS_PER_CANDIDATE = (50, 100)
FALLBACK_PAIRED_TESTS_PER_CANDIDATE = 25
HARD_GATE_LABELS = (
    "finite_state",
    "entry_role_compatible",
    "safe_volume",
    "floor_ceiling_margin",
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
    "terminal_use_label",
)


@dataclass(frozen=True)
class LQRTuningCandidate:
    primitive_id: str
    candidate_index: int
    candidate_weight_label: str
    controller_id: str
    lqr_Q_weights_json: str
    lqr_R_weights_json: str
    lqr_gain_checksum: str
    lqr_synthesis_status: str
    tuning_stage: str
    tuning_method: str = W01_TUNING_METHOD_VERSION
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
    method_version: str = W01_TUNING_METHOD_VERSION


# =============================================================================
# 2) Candidate Generation
# =============================================================================
def candidate_weight_specs(
    *,
    primitive_id: str,
    candidate_count: int = 16,
    tuning_stage: str = "W01",
) -> tuple[LQRWeightSpec, ...]:
    """Return deterministic grouped Q/R candidates for rich W01 generation."""

    if int(candidate_count) <= 0:
        raise ValueError("candidate_count must be positive.")
    base = default_lqr_weight_spec(primitive_id, tuning_stage=tuning_stage)
    if str(primitive_id).startswith("launch_capture_"):
        q_scales = np.asarray([0.25, 0.40, 0.60, 0.85, 1.20, 1.80, 2.80, 4.20], dtype=float)
        r_scales = np.asarray([0.25, 0.40, 0.65, 1.00, 1.60, 2.50, 3.80, 5.50], dtype=float)
    else:
        q_scales = np.asarray([0.5, 0.75, 1.0, 1.5, 2.0, 3.0], dtype=float)
        r_scales = q_scales
    candidates: list[LQRWeightSpec] = []
    for q_scale in q_scales:
        for r_scale in r_scales:
            candidates.append(
                LQRWeightSpec(
                    q_attitude=float(base.q_attitude * q_scale),
                    q_velocity=float(base.q_velocity * q_scale),
                    q_rates=float(base.q_rates * q_scale),
                    q_surfaces=float(base.q_surfaces),
                    r_aileron=float(base.r_aileron * r_scale),
                    r_elevator=float(base.r_elevator * r_scale),
                    r_rudder=float(base.r_rudder * r_scale),
                    tuning_stage=tuning_stage,
                    weight_label=f"{primitive_id}_w01_q{q_scale:g}_r{r_scale:g}_{len(candidates):03d}",
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
    rows = primitive_count * int(candidate_count) * int(paired_tests_per_candidate) * 3
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
                candidate_weight_label=weight_spec.weight_label,
                controller_id=controller.controller_id,
                lqr_Q_weights_json=controller.lqr_Q_weights_json,
                lqr_R_weights_json=controller.lqr_R_weights_json,
                lqr_gain_checksum=controller.lqr_gain_checksum,
                lqr_synthesis_status=controller.lqr_synthesis_status,
                tuning_stage=controller.tuning_stage,
            )
        )
    return tuple(rows)


# =============================================================================
# 3) Summary Helpers
# =============================================================================
def tuning_candidate_row(candidate: LQRTuningCandidate) -> dict[str, object]:
    return asdict(candidate)


def tuning_schedule_row(schedule: LQRTuningSchedule) -> dict[str, object]:
    return asdict(schedule)


def w01_candidate_is_solved(candidate: LQRTuningCandidate) -> bool:
    return candidate.lqr_synthesis_status == LQR_SYNTHESIS_SOLVED
