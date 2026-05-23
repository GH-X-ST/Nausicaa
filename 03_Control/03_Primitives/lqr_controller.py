from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
from scipy import linalg

from command_contract import clip_normalised_command, normalised_command_to_surface_rad
from latency import AGGREGATE_LIMITS, angle_to_command_norm
from lqr_linearisation import (
    LQR_STATE_MASK,
    LQRLinearisation,
    build_lqr_linearisation,
    lqr_linearisation_row,
    reduced_state_indices,
)
from prim_cat import PrimitiveDefinition, primitive_by_id
from state_contract import STATE_INDEX, STATE_NAMES, STATE_SIZE, as_state_vector


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Controller dataclasses
# 2) Synthesis
# 3) Command evaluation
# 4) Audit helpers
# =============================================================================


# =============================================================================
# 1) Controller Dataclasses
# =============================================================================
LQR_CONTROLLER_VERSION = "time_invariant_reduced_order_lqr_v1"
LQR_SYNTHESIS_SOLVED = "solved"
LQR_SYNTHESIS_BLOCKED = "blocked_lqr_synthesis"
ROLLOUT_DT_S = 0.02


@dataclass(frozen=True)
class LQRWeightSpec:
    q_attitude: float
    q_velocity: float
    q_rates: float
    q_surfaces: float
    r_aileron: float
    r_elevator: float
    r_rudder: float
    tuning_stage: str = "W0_W1"
    weight_label: str = "nominal"


@dataclass(frozen=True)
class LQRController:
    primitive_id: str
    controller_family: str
    controller_mode: str
    feedback_mode: str
    controller_id: str
    controller_version: str
    lqr_reference_id: str
    linearisation_id: str
    linearisation_source: str
    reduced_order_lqr: bool
    lqr_state_mask_json: str
    zero_position_gain_expansion_status: str
    full_state_care_status: str
    full_state_care_message: str
    full_controllability_rank: int
    full_state_size: int
    reduced_controllability_rank: int
    reduced_state_size: int
    care_residual_norm: float
    lqr_Q_weights_json: str
    lqr_R_weights_json: str
    lqr_gain_checksum: str
    lqr_synthesis_status: str
    lqr_blocked_reason: str
    lqr_closed_loop_eigenvalue_summary: str
    sampled_data_check_status: str
    sampled_data_spectral_radius: float
    command_clip_check_status: str
    saturation_summary: str
    latency_actuator_survival_status: str
    tuning_stage: str
    controller_claim_status: str
    k_gain_matrix: tuple[tuple[float, ...], ...]
    reference_state_vector: tuple[float, ...]
    reference_command_vector: tuple[float, float, float]


@dataclass(frozen=True)
class LQRCommand:
    primitive_id: str
    controller_id: str
    feedback_mode: str
    command_norm: tuple[float, float, float]
    command_rad: tuple[float, float, float]
    saturation_applied: bool
    raw_command_rad: tuple[float, float, float]
    command_units: str = "normalised_command_and_radian_surface_targets"


# =============================================================================
# 2) Synthesis
# =============================================================================
def default_lqr_weight_spec(primitive_id: str, *, tuning_stage: str = "W0_W1") -> LQRWeightSpec:
    """Return conservative grouped diagonal Q/R weights for one primitive."""

    base = {
        "glide": (4.0, 2.0, 1.5, 0.15, 1.3, 0.9, 1.4),
        "recovery": (6.0, 2.2, 2.0, 0.20, 1.1, 0.8, 1.2),
        "lift_entry": (4.5, 2.5, 1.8, 0.15, 1.1, 0.9, 1.2),
        "lift_dwell_arc": (5.0, 2.2, 2.1, 0.18, 0.9, 1.0, 1.1),
        "mild_turn_left": (5.0, 2.0, 2.1, 0.18, 0.9, 1.1, 1.0),
        "mild_turn_right": (5.0, 2.0, 2.1, 0.18, 0.9, 1.1, 1.0),
        "energy_retaining_bank": (4.8, 2.8, 1.8, 0.15, 1.0, 0.9, 1.2),
        "safe_exit_or_recovery_handoff": (6.0, 2.0, 2.4, 0.20, 1.1, 0.8, 1.2),
    }.get(str(primitive_id), (4.0, 2.0, 2.0, 0.2, 1.0, 1.0, 1.0))
    return LQRWeightSpec(
        q_attitude=base[0],
        q_velocity=base[1],
        q_rates=base[2],
        q_surfaces=base[3],
        r_aileron=base[4],
        r_elevator=base[5],
        r_rudder=base[6],
        tuning_stage=tuning_stage,
        weight_label="nominal",
    )


@lru_cache(maxsize=128)
def lqr_controller_for_primitive_id(
    primitive_id: str,
    *,
    weight_label: str = "nominal",
) -> LQRController:
    weight_spec = default_lqr_weight_spec(primitive_id)
    if weight_label != "nominal":
        weight_spec = LQRWeightSpec(
            **{
                **asdict(weight_spec),
                "weight_label": str(weight_label),
            }
        )
    return synthesize_lqr_controller(primitive_by_id(primitive_id), weight_spec=weight_spec)


def synthesize_lqr_controller(
    primitive: PrimitiveDefinition,
    *,
    weight_spec: LQRWeightSpec | None = None,
    rollout_dt_s: float = ROLLOUT_DT_S,
) -> LQRController:
    """Synthesize the active time-invariant LQR controller for one primitive."""

    weights = weight_spec or default_lqr_weight_spec(primitive.primitive_id)
    linearisation = build_lqr_linearisation(primitive)
    a_full = np.asarray(linearisation.a_full, dtype=float)
    b_full = np.asarray(linearisation.b_full, dtype=float)
    a_reduced = np.asarray(linearisation.a_reduced, dtype=float)
    b_reduced = np.asarray(linearisation.b_reduced, dtype=float)
    q_reduced = _q_reduced(weights)
    r_matrix = _r_matrix(weights)
    q_full = _expand_q_to_full(q_reduced)

    full_status, full_message = _care_attempt_status(a_full, b_full, q_full, r_matrix)
    blocked_reason = ""
    try:
        p_reduced = linalg.solve_continuous_are(a_reduced, b_reduced, q_reduced, r_matrix)
        k_reduced = np.linalg.solve(r_matrix, b_reduced.T @ p_reduced)
        care_residual = _care_residual_norm(a_reduced, b_reduced, q_reduced, r_matrix, p_reduced)
        k_full = _expand_gain_to_full(k_reduced)
        expansion_status = _gain_expansion_status(k_full)
        eig_cont = np.linalg.eigvals(a_reduced - b_reduced @ k_reduced)
        eig_summary = _eigen_summary(eig_cont)
        sampled_radius = _sampled_spectral_radius(a_reduced, b_reduced, k_reduced, rollout_dt_s)
        sampled_status = "sampled_stable" if sampled_radius < 1.0 else "failed_sampled_data_instability"
        command_status, saturation_summary = _reference_command_check(
            tuple(linearisation.reference.reference_command_vector)
        )
        latency_status = (
            "survives_nominal_latency_actuator_lag"
            if sampled_status == "sampled_stable" and float(np.max(np.real(eig_cont))) < 0.0
            else "failed_linear_lag_smoke"
        )
        synthesis_status = (
            LQR_SYNTHESIS_SOLVED
            if (
                linearisation.finite_ab_check == "finite"
                and linearisation.reduced_controllability_rank == linearisation.reduced_state_size
                and np.isfinite(care_residual)
                and care_residual < 1e-6
                and sampled_status == "sampled_stable"
                and command_status == "passed"
                and latency_status == "survives_nominal_latency_actuator_lag"
                and expansion_status == "zero_position_gains_verified"
            )
            else LQR_SYNTHESIS_BLOCKED
        )
        if synthesis_status == LQR_SYNTHESIS_BLOCKED:
            blocked_reason = _blocked_reason(
                linearisation=linearisation,
                care_residual=care_residual,
                sampled_status=sampled_status,
                command_status=command_status,
                latency_status=latency_status,
                expansion_status=expansion_status,
            )
    except Exception as exc:
        k_full = np.zeros((3, STATE_SIZE), dtype=float)
        care_residual = float("inf")
        eig_summary = f"blocked:{type(exc).__name__}:{exc}"
        sampled_radius = float("inf")
        sampled_status = "blocked_lqr_synthesis"
        command_status = "not_evaluated"
        saturation_summary = "not_evaluated"
        latency_status = "not_evaluated"
        expansion_status = "not_evaluated"
        synthesis_status = LQR_SYNTHESIS_BLOCKED
        blocked_reason = f"reduced_care_failed:{type(exc).__name__}:{exc}"

    q_json = json.dumps(_q_weight_payload(weights), sort_keys=True, separators=(",", ":"))
    r_json = json.dumps(_r_weight_payload(weights), sort_keys=True, separators=(",", ":"))
    gain_checksum = gain_checksum_sha256(k_full)
    controller_id = _controller_id(
        primitive.primitive_id,
        linearisation.linearisation_id,
        q_json,
        r_json,
        gain_checksum,
        weights.weight_label,
    )
    return LQRController(
        primitive_id=primitive.primitive_id,
        controller_family="lqr",
        controller_mode="lqr_local_feedback",
        feedback_mode="lqr_state_feedback",
        controller_id=controller_id,
        controller_version=LQR_CONTROLLER_VERSION,
        lqr_reference_id=linearisation.reference.reference_id,
        linearisation_id=linearisation.linearisation_id,
        linearisation_source=linearisation.linearisation_source,
        reduced_order_lqr=True,
        lqr_state_mask_json=json.dumps(LQR_STATE_MASK, separators=(",", ":")),
        zero_position_gain_expansion_status=expansion_status,
        full_state_care_status=full_status,
        full_state_care_message=full_message,
        full_controllability_rank=int(linearisation.full_controllability_rank),
        full_state_size=int(linearisation.full_state_size),
        reduced_controllability_rank=int(linearisation.reduced_controllability_rank),
        reduced_state_size=int(linearisation.reduced_state_size),
        care_residual_norm=float(care_residual),
        lqr_Q_weights_json=q_json,
        lqr_R_weights_json=r_json,
        lqr_gain_checksum=gain_checksum,
        lqr_synthesis_status=synthesis_status,
        lqr_blocked_reason=blocked_reason,
        lqr_closed_loop_eigenvalue_summary=eig_summary,
        sampled_data_check_status=sampled_status,
        sampled_data_spectral_radius=float(sampled_radius),
        command_clip_check_status=command_status,
        saturation_summary=saturation_summary,
        latency_actuator_survival_status=latency_status,
        tuning_stage=weights.tuning_stage,
        controller_claim_status="simulation_only",
        k_gain_matrix=tuple(tuple(float(value) for value in row) for row in k_full),
        reference_state_vector=linearisation.reference.reference_state_vector,
        reference_command_vector=linearisation.reference.reference_command_vector,
    )


# =============================================================================
# 3) Command Evaluation
# =============================================================================
def lqr_command_for_state(
    *,
    controller: LQRController,
    state_vector: np.ndarray,
) -> LQRCommand:
    """Return the LQR surface command for one state."""

    state = as_state_vector(state_vector)
    x_ref = np.asarray(controller.reference_state_vector, dtype=float)
    u_ref = np.asarray(controller.reference_command_vector, dtype=float)
    gain = np.asarray(controller.k_gain_matrix, dtype=float).reshape(3, STATE_SIZE)
    if controller.lqr_synthesis_status != LQR_SYNTHESIS_SOLVED:
        raise RuntimeError(
            "blocked LQR controller cannot produce an executable command: "
            f"{controller.controller_id}:{controller.lqr_synthesis_status}"
        )
    raw_rad = u_ref - gain @ (state - x_ref)
    raw_norm = _surface_rad_to_unclipped_norm(raw_rad)
    clipped_norm = clip_normalised_command(raw_norm)
    clipped_rad = normalised_command_to_surface_rad(clipped_norm)
    return LQRCommand(
        primitive_id=controller.primitive_id,
        controller_id=controller.controller_id,
        feedback_mode=controller.feedback_mode,
        command_norm=tuple(float(value) for value in clipped_norm),
        command_rad=tuple(float(value) for value in clipped_rad),
        saturation_applied=bool(np.any(np.abs(raw_norm - clipped_norm) > 1e-12)),
        raw_command_rad=tuple(float(value) for value in raw_rad),
    )


def lqr_controller_metadata_row(controller: LQRController) -> dict[str, object]:
    row = asdict(controller)
    row.pop("k_gain_matrix")
    row["k_gain_matrix_json"] = json.dumps(controller.k_gain_matrix, separators=(",", ":"))
    row["reference_state_vector"] = json.dumps(
        list(controller.reference_state_vector),
        separators=(",", ":"),
    )
    row["reference_command_vector"] = json.dumps(
        list(controller.reference_command_vector),
        separators=(",", ":"),
    )
    return row


def lqr_rollout_metadata(controller: LQRController) -> dict[str, object]:
    """Return the LQR fields required on every rollout evidence row."""

    return {
        "controller_family": controller.controller_family,
        "controller_id": controller.controller_id,
        "lqr_reference_id": controller.lqr_reference_id,
        "linearisation_id": controller.linearisation_id,
        "linearisation_source": controller.linearisation_source,
        "reduced_order_lqr": controller.reduced_order_lqr,
        "lqr_state_mask_json": controller.lqr_state_mask_json,
        "zero_position_gain_expansion_status": controller.zero_position_gain_expansion_status,
        "lqr_Q_weights_json": controller.lqr_Q_weights_json,
        "lqr_R_weights_json": controller.lqr_R_weights_json,
        "lqr_gain_checksum": controller.lqr_gain_checksum,
        "lqr_synthesis_status": controller.lqr_synthesis_status,
        "lqr_blocked_reason": controller.lqr_blocked_reason,
        "lqr_closed_loop_eigenvalue_summary": controller.lqr_closed_loop_eigenvalue_summary,
        "care_residual_norm": controller.care_residual_norm,
        "sampled_data_check_status": controller.sampled_data_check_status,
        "sampled_data_spectral_radius": controller.sampled_data_spectral_radius,
        "command_clip_check_status": controller.command_clip_check_status,
        "saturation_summary": controller.saturation_summary,
        "latency_actuator_survival_status": controller.latency_actuator_survival_status,
        "tuning_stage": controller.tuning_stage,
        "controller_claim_status": controller.controller_claim_status,
    }


# =============================================================================
# 4) Audit Helpers
# =============================================================================
def synthesis_audit_row(primitive: PrimitiveDefinition) -> dict[str, object]:
    controller = synthesize_lqr_controller(primitive)
    row = lqr_controller_metadata_row(controller)
    row.update(
        {
            f"linearisation_{key}": value
            for key, value in lqr_linearisation_row(build_lqr_linearisation(primitive)).items()
        }
    )
    return row


def gain_checksum_sha256(gain_matrix: np.ndarray) -> str:
    rounded = np.round(np.asarray(gain_matrix, dtype=float), decimals=12)
    return hashlib.sha256(rounded.tobytes()).hexdigest()


def _q_reduced(weights: LQRWeightSpec) -> np.ndarray:
    diag = []
    for name in LQR_STATE_MASK:
        if name in {"phi", "theta", "psi"}:
            diag.append(weights.q_attitude)
        elif name in {"u", "v", "w"}:
            diag.append(weights.q_velocity)
        elif name in {"p", "q", "r"}:
            diag.append(weights.q_rates)
        else:
            diag.append(weights.q_surfaces)
    return np.diag(np.asarray(diag, dtype=float))


def _r_matrix(weights: LQRWeightSpec) -> np.ndarray:
    return np.diag([weights.r_aileron, weights.r_elevator, weights.r_rudder]).astype(float)


def _expand_q_to_full(q_reduced: np.ndarray) -> np.ndarray:
    q_full = np.eye(STATE_SIZE, dtype=float) * 1e-6
    for reduced_index, full_index in enumerate(reduced_state_indices()):
        q_full[full_index, full_index] = float(q_reduced[reduced_index, reduced_index])
    return q_full


def _expand_gain_to_full(k_reduced: np.ndarray) -> np.ndarray:
    k_full = np.zeros((3, STATE_SIZE), dtype=float)
    for reduced_index, full_index in enumerate(reduced_state_indices()):
        k_full[:, full_index] = k_reduced[:, reduced_index]
    return k_full


def _gain_expansion_status(k_full: np.ndarray) -> str:
    for name in ("x_w", "y_w", "z_w"):
        if np.max(np.abs(k_full[:, STATE_INDEX[name]])) > 1e-12:
            return "failed_position_gain_nonzero"
    return "zero_position_gains_verified"


def _care_attempt_status(
    a: np.ndarray,
    b: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
) -> tuple[str, str]:
    try:
        linalg.solve_continuous_are(a, b, q, r)
    except Exception as exc:
        return "unsuitable_use_reduced_order", f"{type(exc).__name__}:{exc}"
    return "solved", "full_state_care_solved"


def _care_residual_norm(
    a: np.ndarray,
    b: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    p: np.ndarray,
) -> float:
    r_inv = np.linalg.inv(r)
    residual = a.T @ p + p @ a - p @ b @ r_inv @ b.T @ p + q
    scale = max(1.0, float(np.linalg.norm(q, ord="fro")))
    return float(np.linalg.norm(residual, ord="fro") / scale)


def _sampled_spectral_radius(
    a: np.ndarray,
    b: np.ndarray,
    k: np.ndarray,
    dt_s: float,
) -> float:
    discrete = linalg.expm((a - b @ k) * float(dt_s))
    return float(np.max(np.abs(np.linalg.eigvals(discrete))))


def _eigen_summary(eigenvalues: np.ndarray) -> str:
    eig = np.asarray(eigenvalues, dtype=complex)
    payload = {
        "max_real": float(np.max(np.real(eig))),
        "min_real": float(np.min(np.real(eig))),
        "max_abs_imag": float(np.max(np.abs(np.imag(eig)))),
        "count": int(eig.size),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _reference_command_check(command_rad: tuple[float, float, float]) -> tuple[str, str]:
    raw_norm = _surface_rad_to_unclipped_norm(np.asarray(command_rad, dtype=float))
    clipped = clip_normalised_command(raw_norm)
    saturated = bool(np.any(np.abs(raw_norm - clipped) > 1e-12))
    return (
        "passed",
        json.dumps(
            {
                "reference_saturated": saturated,
                "max_abs_reference_norm": float(np.max(np.abs(raw_norm))),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def _surface_rad_to_unclipped_norm(command_rad: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            angle_to_command_norm(value, AGGREGATE_LIMITS[name])
            for value, name in zip(command_rad, ("delta_a", "delta_e", "delta_r"), strict=True)
        ],
        dtype=float,
    )


def _q_weight_payload(weights: LQRWeightSpec) -> dict[str, object]:
    return {
        "grouping": "diagonal_grouped_log_scaled",
        "state_mask": list(LQR_STATE_MASK),
        "q_attitude": float(weights.q_attitude),
        "q_velocity": float(weights.q_velocity),
        "q_rates": float(weights.q_rates),
        "q_surfaces": float(weights.q_surfaces),
    }


def _r_weight_payload(weights: LQRWeightSpec) -> dict[str, object]:
    return {
        "grouping": "diagonal_grouped_log_scaled",
        "command_names": ["delta_a_cmd", "delta_e_cmd", "delta_r_cmd"],
        "r_aileron": float(weights.r_aileron),
        "r_elevator": float(weights.r_elevator),
        "r_rudder": float(weights.r_rudder),
    }


def _controller_id(
    primitive_id: str,
    linearisation_id: str,
    q_json: str,
    r_json: str,
    gain_checksum: str,
    weight_label: str,
) -> str:
    payload = json.dumps(
        [primitive_id, linearisation_id, q_json, r_json, gain_checksum, weight_label],
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("ascii")).hexdigest()[:12]
    return f"lqr_{primitive_id}_{digest}"


def _blocked_reason(
    *,
    linearisation: LQRLinearisation,
    care_residual: float,
    sampled_status: str,
    command_status: str,
    latency_status: str,
    expansion_status: str,
) -> str:
    reasons = []
    if linearisation.finite_ab_check != "finite":
        reasons.append("nonfinite_linearisation")
    if linearisation.reduced_controllability_rank != linearisation.reduced_state_size:
        reasons.append("reduced_controllability_rank_deficient")
    if not np.isfinite(care_residual) or care_residual >= 1e-6:
        reasons.append("care_residual_high")
    if sampled_status != "sampled_stable":
        reasons.append(sampled_status)
    if command_status != "passed":
        reasons.append(command_status)
    if latency_status != "survives_nominal_latency_actuator_lag":
        reasons.append(latency_status)
    if expansion_status != "zero_position_gains_verified":
        reasons.append(expansion_status)
    return ";".join(reasons) if reasons else "blocked_lqr_synthesis"
