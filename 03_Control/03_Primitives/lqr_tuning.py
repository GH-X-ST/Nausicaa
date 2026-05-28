from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

from lqr_controller import LQR_SYNTHESIS_SOLVED, LQRWeightSpec, default_lqr_weight_spec, synthesize_lqr_controller
from lqr_linearisation import LQR_DEFAULT_AUDIT_REFERENCE_SPEED_M_S
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
W01_TUNING_METHOD_VERSION = "w01_transition_robust_qr_reference_v4"
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
    "transition_success_lcb",
    "hard_failure_ucb",
    "trajectory_integrated_updraft_gain_m",
    "lift_dwell_time_s",
    "rollout_duration_s",
    "saturation_fraction",
    "reference_bias_sweep_coverage",
)
LQR_WEIGHT_FIELD_NAMES = (
    "q_attitude",
    "q_velocity",
    "q_rates",
    "q_surfaces",
    "r_aileron",
    "r_elevator",
    "r_rudder",
)
STRUCTURED_ANCHOR_MULTIPLIERS: tuple[tuple[str, tuple[float, float, float, float, float, float, float]], ...] = (
    ("nominal", (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
    ("attitude_heavy", (1.8, 1.0, 1.2, 1.0, 1.1, 1.1, 1.1)),
    ("velocity_energy_heavy", (1.0, 1.8, 1.0, 1.0, 1.0, 1.0, 1.0)),
    ("rate_damped", (1.0, 1.0, 1.8, 1.0, 1.2, 1.2, 1.2)),
    ("surface_conservative", (1.0, 1.0, 1.0, 1.4, 1.8, 1.8, 1.8)),
    ("surface_aggressive", (1.1, 1.1, 1.1, 0.8, 0.6, 0.6, 0.6)),
    ("balanced_robust", (1.4, 1.3, 1.4, 1.2, 1.4, 1.4, 1.4)),
    ("balanced_agile", (0.9, 1.2, 1.0, 0.8, 0.8, 0.8, 0.8)),
)
REFERENCE_BIAS_FIELD_NAMES = (
    "reference_pitch_bias_rad",
    "reference_bank_bias_rad",
    "reference_speed_bias_m_s",
)
STRUCTURED_REFERENCE_BIASES: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("ref_nominal", (0.0, 0.0, 0.0)),
    ("ref_pitch_up", (0.025, 0.0, 0.0)),
    ("ref_energy_seek", (-0.020, 0.0, 0.35)),
    ("ref_damped_level", (0.0, 0.0, -0.25)),
    ("ref_conservative_level", (0.0, 0.0, 0.0)),
    ("ref_agile_pitch_up", (0.018, 0.0, 0.15)),
    ("ref_left_bias", (0.0, -0.055, 0.0)),
    ("ref_right_bias", (0.0, 0.055, 0.0)),
)
REFERENCE_PITCH_BIAS_RANGE_RAD = (-0.040, 0.040)
REFERENCE_BANK_BIAS_RANGE_RAD = (-0.080, 0.080)
REFERENCE_SPEED_BIAS_RANGE_M_S = (-0.500, 0.500)
LHS_LOG10_MULTIPLIER_MIN = -0.75
LHS_LOG10_MULTIPLIER_MAX = 0.75


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
    """Return deterministic structured log-space Q/R candidates for R5 training."""

    if int(candidate_count) <= 0:
        raise ValueError("candidate_count must be positive.")
    base = default_lqr_weight_spec(primitive_id, tuning_stage=tuning_stage)
    candidates: list[LQRWeightSpec] = []
    for (label, multipliers), (reference_label, reference_bias) in zip(
        STRUCTURED_ANCHOR_MULTIPLIERS,
        STRUCTURED_REFERENCE_BIASES,
        strict=True,
    ):
        if len(candidates) >= int(candidate_count):
            break
        candidates.append(
            _weight_spec_from_multipliers(
                base,
                primitive_id=str(primitive_id),
                multipliers=multipliers,
                reference_bias=reference_bias,
                tuning_stage=tuning_stage,
                weight_label=f"{primitive_id}_robust_anchor_{label}_{reference_label}_{len(candidates):03d}",
            )
        )
    lhs_count = max(0, int(candidate_count) - len(candidates))
    lhs_multipliers = _deterministic_lhs_log_multipliers(
        primitive_id=str(primitive_id),
        tuning_stage=str(tuning_stage),
        sample_count=lhs_count,
    )
    lhs_reference_biases = _deterministic_lhs_reference_biases(
        primitive_id=str(primitive_id),
        tuning_stage=str(tuning_stage),
        sample_count=lhs_count,
    )
    for lhs_index, (multipliers, reference_bias) in enumerate(
        zip(lhs_multipliers, lhs_reference_biases, strict=True),
        start=len(candidates),
    ):
        candidates.append(
            _weight_spec_from_multipliers(
                base,
                primitive_id=str(primitive_id),
                multipliers=multipliers,
                reference_bias=reference_bias,
                tuning_stage=tuning_stage,
                weight_label=f"{primitive_id}_robust_lhs_logqr_refbias_{lhs_index:03d}",
            )
        )
    return tuple(candidates[: int(candidate_count)])


def _weight_spec_from_multipliers(
    base: LQRWeightSpec,
    *,
    primitive_id: str,
    multipliers: tuple[float, float, float, float, float, float, float],
    reference_bias: tuple[float, float, float],
    tuning_stage: str,
    weight_label: str,
) -> LQRWeightSpec:
    values = {
        field: float(getattr(base, field) * float(multiplier))
        for field, multiplier in zip(LQR_WEIGHT_FIELD_NAMES, multipliers, strict=True)
    }
    return LQRWeightSpec(
        q_attitude=values["q_attitude"],
        q_velocity=values["q_velocity"],
        q_rates=values["q_rates"],
        q_surfaces=values["q_surfaces"],
        r_aileron=values["r_aileron"],
        r_elevator=values["r_elevator"],
        r_rudder=values["r_rudder"],
        reference_pitch_bias_rad=float(reference_bias[0]),
        reference_bank_bias_rad=float(reference_bias[1]),
        reference_speed_bias_m_s=float(reference_bias[2]),
        tuning_stage=tuning_stage,
        weight_label=weight_label,
    )


def _deterministic_lhs_log_multipliers(
    *,
    primitive_id: str,
    tuning_stage: str,
    sample_count: int,
) -> tuple[tuple[float, float, float, float, float, float, float], ...]:
    if int(sample_count) <= 0:
        return ()
    seed_material = f"{W01_TUNING_METHOD_VERSION}:{primitive_id}:{tuning_stage}:{sample_count}".encode("ascii")
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], byteorder="big", signed=False)
    rng = np.random.default_rng(seed)
    columns: list[np.ndarray] = []
    span = float(LHS_LOG10_MULTIPLIER_MAX - LHS_LOG10_MULTIPLIER_MIN)
    for _ in LQR_WEIGHT_FIELD_NAMES:
        permutation = rng.permutation(int(sample_count))
        u = (permutation.astype(float) + 0.5) / float(sample_count)
        columns.append(np.power(10.0, LHS_LOG10_MULTIPLIER_MIN + span * u))
    return tuple(
        tuple(float(columns[dim][sample_index]) for dim in range(len(LQR_WEIGHT_FIELD_NAMES)))
        for sample_index in range(int(sample_count))
    )


def _deterministic_lhs_reference_biases(
    *,
    primitive_id: str,
    tuning_stage: str,
    sample_count: int,
) -> tuple[tuple[float, float, float], ...]:
    if int(sample_count) <= 0:
        return ()
    seed_material = f"{W01_TUNING_METHOD_VERSION}:reference:{primitive_id}:{tuning_stage}:{sample_count}".encode("ascii")
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], byteorder="big", signed=False)
    rng = np.random.default_rng(seed)
    ranges = (
        REFERENCE_PITCH_BIAS_RANGE_RAD,
        REFERENCE_BANK_BIAS_RANGE_RAD,
        REFERENCE_SPEED_BIAS_RANGE_M_S,
    )
    columns: list[np.ndarray] = []
    for low, high in ranges:
        permutation = rng.permutation(int(sample_count))
        u = (permutation.astype(float) + 0.5) / float(sample_count)
        columns.append(float(low) + (float(high) - float(low)) * u)
    return tuple(
        tuple(float(columns[dim][sample_index]) for dim in range(len(REFERENCE_BIAS_FIELD_NAMES)))
        for sample_index in range(int(sample_count))
    )


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
    local_reference_speed_m_s: float = LQR_DEFAULT_AUDIT_REFERENCE_SPEED_M_S,
) -> tuple[LQRTuningCandidate, ...]:
    rows = []
    for index, weight_spec in enumerate(
        candidate_weight_specs(
            primitive_id=primitive.primitive_id,
            candidate_count=candidate_count,
        )
    ):
        controller = synthesize_lqr_controller(
            primitive,
            weight_spec=weight_spec,
            local_reference_speed_m_s=float(local_reference_speed_m_s),
        )
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
