from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from arena_contract import TRUE_SAFE_BOUNDS, position_margin_m
from bank_primitive import (
    BankCaseSpec,
    BankFeedbackGains,
    BankPrimitiveConfig,
    bank_feedback_command_norm,
    bank_reference_bank_rad,
    build_bank_primitive_spec,
)
from command_contract import clip_normalised_command, normalised_command_to_surface_rad, surface_rad_to_normalised_command
from episode_schema import validate_primitive_rollout_evidence_frame
from fixed_gate_contract import FIXED_LAUNCH_GATE, launch_gate_admission_status
from flight_dynamics import adapt_glider
from glider import build_nausicaa_glider
from glide_primitive import GlideFeedbackGains, GlidePrimitiveConfig, build_glide_primitive_spec, glide_feedback_command_norm
from latency import (
    actuator_tau_for_case,
    latency_adjusted_command_sample,
    latency_audit_fields_from_case,
    latency_case_config,
    latency_mechanism_flags_from_case,
    latency_pass_label_for_single_run,
)
from primitive_library_generators import generate_command_profile
from primitive_library_schema import PrimitiveCandidateSpec
from primitive_interface import evaluate_entry_set, evaluate_exit_checks
from recovery_primitive import (
    RecoveryFeedbackGains,
    RecoveryPrimitiveConfig,
    build_recovery_primitive_spec,
    recovery_feedback_command_norm,
)
from rollout import CommandSchedule, RolloutConfig, RolloutResult, rk4_step, rollout_open_loop_normalised
from state_contract import STATE_SIZE, as_state_vector
from trim_solver import TrimTarget, solve_straight_trim
from updraft_models import load_updraft_model
from wing_wind_descriptors import wing_wind_descriptor_row


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Evidence constants and data containers
# 2) Public rollout workflow
# 3) Row builders and source-state conversion
# 4) Command-template and metric helpers
# 5) Readiness summaries
# =============================================================================


# =============================================================================
# 1) Evidence Constants and Data Containers
# =============================================================================
ROW_COLUMNS = (
    "sample_id",
    "paired_sample_key",
    "fan_branch",
    "W_layer",
    "test_environment_mode",
    "entry_source",
    "launch_gate_id",
    "initial_state_vector",
    "initial_state_admission_status",
    "primitive_id",
    "primitive_family",
    "controller_mode",
    "feedback_mode",
    "latency_case",
    "claim_status",
    "evidence_role",
    "governor_decision_status",
    "primary_rejection_reason",
    "all_rejection_reasons",
    "outcome_class",
    "accepted",
    "failure_label",
    "duration_s",
    "dwell_time_s",
    "energy_initial_m",
    "energy_final_m",
    "energy_residual_m",
    "minimum_margin_m",
    "minimum_speed_m_s",
    "exit_speed_m_s",
    "control_saturation",
    "rollout_integrity_success",
    "entry_check_status",
    "exit_check_status",
    "mission_feedback_path_status",
    "x0_w_m",
    "y0_w_m",
    "z0_w_m",
    "phi0_rad",
    "theta0_rad",
    "psi0_rad",
    "speed0_m_s",
    "wind_mode",
    "updraft_model_id",
    "updraft_model_source",
    "wind_binding_status",
    "x_terminal_w_m",
    "y_terminal_w_m",
    "z_terminal_w_m",
    "phi_terminal_rad",
    "theta_terminal_rad",
    "psi_terminal_rad",
    "u_terminal_m_s",
    "v_terminal_m_s",
    "w_terminal_m_s",
    "p_terminal_rad_s",
    "q_terminal_rad_s",
    "r_terminal_rad_s",
    "w_wing_mean_m_s",
    "delta_w_lr_m_s",
    "spanwise_gradient_m_s_m",
    "spanwise_w_gradient_m_s_per_m",
    "wind_descriptor_status",
    "wind_descriptor_environment_mode",
    "wind_descriptor_model_id",
    "wind_descriptor_model_source",
    "state_feedback_delay_s",
    "command_onset_delay_s",
    "command_transport_delay_s",
    "actuator_tau_s",
    "actuator_t50_s",
    "actuator_t90_s",
    "latency_jitter_s",
    "timing_model_version",
    "latency_pass_label",
    "state_feedback_delay_applied",
    "command_delay_applied",
    "actuator_lag_applied",
)

MISSION_CONTROLLER_MODE = "feedback_stabilised_primitive"
MISSION_FEEDBACK_MODE = "delayed_state_feedback"
PARTIAL_FEEDBACK_MODE = "instant_state_feedback"
DIAGNOSTIC_CONTROLLER_MODE = "open_loop_rollout"
DIAGNOSTIC_FEEDBACK_MODE = "open_loop"
BLOCKED_FEEDBACK_STATUS = "blocked_true_delayed_state_feedback_unavailable"
PARTIAL_FEEDBACK_STATUS = "partial_feedback_instant_state_no_delayed_state_feedback"
WIND_BINDING_BLOCKED_STATUS = "blocked_updraft_model_unavailable"
W1_MEASURED_WIND_STATUS = "measured_updraft_bound"

W1_MODEL_BY_BRANCH = {
    "single_fan_branch": "single_gaussian_var",
    "four_fan_branch": "four_gaussian_var",
}


@dataclass(frozen=True)
class FixedGatePrimitiveRolloutConfig:
    dt_s: float = 0.02
    horizon_s: float = 0.60
    latency_case: str = "nominal"
    random_seed: int = 20260521
    controller_mode: str = DIAGNOSTIC_CONTROLLER_MODE
    feedback_mode: str = DIAGNOSTIC_FEEDBACK_MODE
    allow_open_loop_diagnostic: bool = True


@dataclass(frozen=True)
class WindBinding:
    wind_model: object | None
    wind_mode: str
    updraft_model_id: str
    updraft_model_source: str
    wind_binding_status: str
    dry_air: bool = False
    blocked_reason: str = ""


@dataclass(frozen=True)
class FixedGateFeedbackReplayResult:
    primitive_spec: object
    time_s: np.ndarray
    x: np.ndarray
    u_norm_requested: np.ndarray
    u_norm_applied: np.ndarray
    delta_cmd_rad: np.ndarray
    entry_checks: tuple[object, ...]
    exit_checks: tuple[object, ...]
    metrics: dict[str, object]
    success: bool
    failure_label: str
    notes: str


# =============================================================================
# 2) Public Rollout Workflow
# =============================================================================
def run_fixed_gate_primitive_rollouts(
    candidate_rows: pd.DataFrame,
    config: FixedGatePrimitiveRolloutConfig | None = None,
) -> pd.DataFrame:
    """Execute fixed-gate primitive row evidence with explicit hierarchy labels.

    Command-template/open-loop rows validate sampling, dynamics, descriptors,
    latency labels, and storage, but they remain diagnostic. Feedback rows are
    emitted only when an actual local-feedback primitive path is used. Because
    this adapter does not implement delayed-state feedback, those rows are
    partial-feedback simulation evidence rather than hardware-ready mission
    candidates.
    """

    cfg = FixedGatePrimitiveRolloutConfig() if config is None else config
    if candidate_rows.empty:
        return pd.DataFrame(columns=ROW_COLUMNS)
    latency_cfg = latency_case_config(cfg.latency_case)
    rows: list[dict[str, object]] = []
    for local_index, candidate in enumerate(candidate_rows.to_dict(orient="records")):
        seed = int(cfg.random_seed) + int(local_index)
        for requested_controller, requested_feedback in _requested_execution_modes(candidate, cfg):
            if requested_controller == DIAGNOSTIC_CONTROLLER_MODE:
                if bool(cfg.allow_open_loop_diagnostic):
                    rows.append(_open_loop_diagnostic_row(candidate, cfg, seed, latency_cfg))
                else:
                    rows.append(
                        _blocked_feedback_row(
                            candidate,
                            cfg,
                            seed,
                            reason="blocked_open_loop_diagnostic_disabled",
                        )
                    )
                continue
            if _is_delayed_feedback_request(requested_feedback):
                rows.append(_blocked_feedback_row(candidate, cfg, seed, reason=BLOCKED_FEEDBACK_STATUS))
                continue
            rows.append(_partial_feedback_row(candidate, cfg, seed, latency_cfg))
    frame = pd.DataFrame(rows, columns=ROW_COLUMNS)
    validate_primitive_rollout_evidence_frame(frame)
    return frame


def build_w0_w1_pairing_audit(candidate_rows: pd.DataFrame, rollout_rows: pd.DataFrame) -> pd.DataFrame:
    if candidate_rows.empty:
        return pd.DataFrame(columns=["fan_branch", "primitive_family", "paired_sample_key", "has_W0", "has_W1"])
    grouped = candidate_rows.groupby(["fan_branch", "primitive_family", "paired_sample_key"], dropna=False)
    rows = []
    rollout_keys = set(
        tuple(row)
        for row in rollout_rows[["fan_branch", "primitive_family", "paired_sample_key", "W_layer"]].astype(str).to_numpy()
    ) if not rollout_rows.empty else set()
    for key, group in grouped:
        fan_branch, primitive_family, paired_sample_key = key
        rows.append(
            {
                "fan_branch": str(fan_branch),
                "primitive_family": str(primitive_family),
                "paired_sample_key": str(paired_sample_key),
                "has_W0": (str(fan_branch), str(primitive_family), str(paired_sample_key), "W0") in rollout_keys,
                "has_W1": (str(fan_branch), str(primitive_family), str(paired_sample_key), "W1") in rollout_keys,
                "w1_scheduled_independent_of_w0_success": "W0" in set(group["W_layer"].astype(str)) and "W1" in set(group["W_layer"].astype(str)),
            }
        )
    return pd.DataFrame(rows)


def build_rollout_outcome_summary(rollout_rows: pd.DataFrame) -> pd.DataFrame:
    if rollout_rows.empty:
        return pd.DataFrame(columns=["fan_branch", "W_layer", "evidence_role", "outcome_class", "row_count"])
    return (
        rollout_rows.groupby(["fan_branch", "W_layer", "evidence_role", "outcome_class"], dropna=False)
        .size()
        .reset_index(name="row_count")
    )


def build_archive_move_on_gates(rollout_rows: pd.DataFrame, reachable_rows: pd.DataFrame | None = None) -> dict[str, object]:
    reachable_count = 0 if reachable_rows is None else int(len(reachable_rows))
    evidence = _mission_or_partial_rows(rollout_rows)
    branch_layer_counts = (
        evidence.groupby(["fan_branch", "W_layer"]).size().to_dict()
        if not evidence.empty
        else {}
    )
    all_branch_layer_counts = (
        rollout_rows.groupby(["fan_branch", "W_layer"]).size().to_dict()
        if not rollout_rows.empty and {"fan_branch", "W_layer"}.issubset(rollout_rows.columns)
        else {}
    )
    w1_measured = _w1_measured_updraft_rows(rollout_rows)
    w0_by_branch = _layer_count_by_branch(rollout_rows, "W0")
    w1_by_branch = _layer_count_by_branch(rollout_rows, "W1")
    w1_measured_by_branch = _layer_count_by_branch(w1_measured, "W1")
    branches = {"single_fan_branch", "four_fan_branch"}
    paired_branch_layers_present = all(
        int(all_branch_layer_counts.get((branch, layer), 0)) > 0
        for branch in branches
        for layer in ("W0", "W1")
    )
    measured_w1_present = all(int(w1_measured_by_branch.get(branch, 0)) > 0 for branch in branches)
    archive_prepared = bool(paired_branch_layers_present and measured_w1_present)
    mission_ready = all(
        int(branch_layer_counts.get((branch, layer), 0)) > 0
        for branch in branches
        for layer in ("W0", "W1")
    )
    code_ready = _has_required_evidence_columns(rollout_rows) and not _has_promoted_diagnostic_rows(rollout_rows)
    partial_count = _role_count(rollout_rows, "partial_feedback")
    mission_count = _role_count(rollout_rows, "mission_candidate")
    blocked_count = _role_count(rollout_rows, "blocked_partial")
    return {
        "code_ready": "ready" if code_ready else "blocked_schema_or_promoted_diagnostic_rows",
        "archive_prepared": "ready" if archive_prepared else _archive_prepared_block_reason(paired_branch_layers_present, measured_w1_present),
        "mission_evidence_ready": "ready" if mission_ready else "blocked_no_mission_or_partial_feedback_rows_for_both_branches",
        "code_readiness": "ready" if code_ready else "blocked_schema_or_promoted_diagnostic_rows",
        "archive_readiness": "ready" if archive_prepared else _archive_prepared_block_reason(paired_branch_layers_present, measured_w1_present),
        "w_ladder_readiness": "ready" if _w1_independent(rollout_rows) else "blocked_w1_pairing_missing",
        "reachable_state_readiness": "ready" if reachable_count > 0 else "blocked_no_reachable_downstream_extraction",
        "w0_row_count_by_branch": w0_by_branch,
        "w1_row_count_by_branch": w1_by_branch,
        "w1_measured_updraft_row_count": int(len(w1_measured)),
        "w1_measured_updraft_row_count_by_branch": w1_measured_by_branch,
        "mission_candidate_row_count": int(mission_count),
        "partial_feedback_row_count": int(partial_count),
        "accepted_partial_feedback_row_count": int(len(_accepted_role_rows(rollout_rows, "partial_feedback"))),
        "accepted_w0_partial_feedback_row_count": int(len(_accepted_role_layer_rows(rollout_rows, "partial_feedback", "W0"))),
        "accepted_w1_partial_feedback_row_count": int(len(_accepted_role_layer_rows(rollout_rows, "partial_feedback", "W1"))),
        "blocked_partial_row_count": int(blocked_count),
        "mission_or_partial_evidence_row_count": int(len(evidence)),
        "ablation_diagnostic_row_count": _role_count(rollout_rows, "ablation_diagnostic"),
        "reachable_downstream_row_count": reachable_count,
        "feedback_path_status": PARTIAL_FEEDBACK_STATUS if partial_count > 0 else BLOCKED_FEEDBACK_STATUS,
    }


# =============================================================================
# 3) Row Builders and Source-State Conversion
# =============================================================================
def _open_loop_diagnostic_row(
    candidate: dict[str, object],
    config: FixedGatePrimitiveRolloutConfig,
    seed: int,
    latency_cfg: object,
) -> dict[str, object]:
    state0 = _initial_state_from_candidate(candidate)
    binding = _wind_binding(candidate)
    if binding.blocked_reason:
        return _blocked_feedback_row(candidate, config, seed, reason=binding.blocked_reason, binding=binding)
    admission = str(candidate.get("initial_state_admission_status", launch_gate_admission_status(state0)))
    schedule = _command_schedule(candidate, config)
    result = rollout_open_loop_normalised(
        state0,
        schedule,
        RolloutConfig(
            dt_s=float(config.dt_s),
            t_final_s=float(config.horizon_s),
            wind_mode=str(binding.wind_mode),
            latency_case=str(config.latency_case),
            actuator_tau_s=actuator_tau_for_case(latency_cfg),
        ),
        wind_model=binding.wind_model,
        seed=int(seed),
        scenario_name="fixed_gate_open_loop_diagnostic",
    )
    terminal = result.x[-1]
    metrics = result.metrics
    minimum_margin = _minimum_true_margin(result.x)
    exit_speed = _speed_m_s(terminal)
    rollout_success = bool(metrics.get("rollout_success", False))
    failure_label = str(result.failure_label if result.failure_label != "not_run" else "diagnostic_open_loop_not_mission_evidence")
    outcome_class = "diagnostic_rollout_success" if rollout_success else "diagnostic_rollout_failed"
    latency_fields = _latency_fields(config.latency_case, accepted=False, state_feedback_delay_applied=False)
    descriptor = _wind_descriptor(candidate, state0, binding)
    return {
        **_common_row(candidate, state0),
        **_wind_binding_fields(binding),
        "controller_mode": DIAGNOSTIC_CONTROLLER_MODE,
        "feedback_mode": DIAGNOSTIC_FEEDBACK_MODE,
        "latency_case": str(config.latency_case),
        "claim_status": "simulation_only",
        "evidence_role": "ablation_diagnostic",
        "governor_decision_status": "not_governor_candidate_diagnostic",
        "primary_rejection_reason": "diagnostic_open_loop_not_mission_candidate",
        "all_rejection_reasons": "diagnostic_open_loop_not_mission_candidate",
        "outcome_class": outcome_class,
        "accepted": False,
        "failure_label": failure_label,
        "duration_s": float(result.time_s[-1] - result.time_s[0]) if result.time_s.size else 0.0,
        "dwell_time_s": 0.0,
        "energy_initial_m": _specific_energy_height_m(state0),
        "energy_final_m": _specific_energy_height_m(terminal),
        "energy_residual_m": _specific_energy_height_m(terminal) - _specific_energy_height_m(state0),
        "minimum_margin_m": minimum_margin,
        "minimum_speed_m_s": _minimum_speed_m_s(result.x),
        "exit_speed_m_s": exit_speed,
        "control_saturation": float(metrics.get("saturation_fraction", 0.0)),
        "rollout_integrity_success": rollout_success,
        "entry_check_status": admission,
        "exit_check_status": "diagnostic_exit_pass" if rollout_success else str(result.failure_label),
        "mission_feedback_path_status": BLOCKED_FEEDBACK_STATUS,
        **_terminal_state_fields(terminal),
        **descriptor,
        **latency_fields,
    }


def _partial_feedback_row(
    candidate: dict[str, object],
    config: FixedGatePrimitiveRolloutConfig,
    seed: int,
    latency_cfg: object,
) -> dict[str, object]:
    state0 = _initial_state_from_candidate(candidate)
    binding = _wind_binding(candidate)
    if binding.blocked_reason:
        return _blocked_feedback_row(candidate, config, seed, reason=binding.blocked_reason, binding=binding)
    try:
        result = _run_local_feedback_primitive(candidate, state0, config, seed, latency_cfg, binding)
    except (RuntimeError, ValueError) as exc:
        return _blocked_feedback_row(candidate, config, seed, reason=f"blocked_feedback_adapter_error:{exc}", binding=binding)

    terminal = result.x[-1]
    metrics = dict(result.metrics)
    entry_pass = _checks_pass(getattr(result, "entry_checks", ()))
    exit_checks = getattr(result, "exit_checks", ())
    exit_pass = _checks_pass(exit_checks)
    rollout_success = bool(metrics.get("rollout_success", bool(result.success)))
    primitive_success = bool(result.success)
    failure_label = str(result.failure_label if result.failure_label else "success")
    if primitive_success:
        outcome_class = "accepted"
        governor_status = "accepted_partial_feedback_governor_compatible"
        rejection = "none"
        mission_feedback_status = PARTIAL_FEEDBACK_STATUS
    elif not entry_pass:
        outcome_class = "rejected"
        governor_status = "rejected_entry_check_failed"
        rejection = "primitive_entry_check_failed"
        mission_feedback_status = "partial_feedback_entry_check_failed"
    else:
        outcome_class = "failed"
        governor_status = "rejected_exit_or_rollout_check_failed"
        rejection = failure_label
        mission_feedback_status = "partial_feedback_exit_or_rollout_failed"
    latency_fields = _latency_fields(config.latency_case, accepted=primitive_success, state_feedback_delay_applied=False)
    descriptor = _wind_descriptor(candidate, state0, binding)
    return {
        **_common_row(candidate, state0),
        **_wind_binding_fields(binding),
        "controller_mode": MISSION_CONTROLLER_MODE,
        "feedback_mode": PARTIAL_FEEDBACK_MODE,
        "latency_case": str(config.latency_case),
        "claim_status": "simulation_only",
        "evidence_role": "partial_feedback",
        "governor_decision_status": governor_status,
        "primary_rejection_reason": rejection,
        "all_rejection_reasons": rejection,
        "outcome_class": outcome_class,
        "accepted": primitive_success,
        "failure_label": failure_label,
        "duration_s": float(result.time_s[-1] - result.time_s[0]) if result.time_s.size else 0.0,
        "dwell_time_s": _partial_feedback_dwell_time(candidate, result, primitive_success),
        "energy_initial_m": _specific_energy_height_m(state0),
        "energy_final_m": _specific_energy_height_m(terminal),
        "energy_residual_m": _specific_energy_height_m(terminal) - _specific_energy_height_m(state0),
        "minimum_margin_m": _minimum_true_margin(result.x),
        "minimum_speed_m_s": _minimum_speed_m_s(result.x),
        "exit_speed_m_s": _speed_m_s(terminal),
        "control_saturation": float(metrics.get("saturation_fraction", 0.0)),
        "rollout_integrity_success": rollout_success,
        "entry_check_status": _entry_check_status(candidate, state0, entry_pass),
        "exit_check_status": "primitive_exit_pass" if exit_pass else _exit_check_failure_status(exit_checks, failure_label),
        "mission_feedback_path_status": mission_feedback_status,
        **_terminal_state_fields(terminal),
        **descriptor,
        **latency_fields,
    }


def _blocked_feedback_row(
    candidate: dict[str, object],
    config: FixedGatePrimitiveRolloutConfig,
    seed: int,
    *,
    reason: str = BLOCKED_FEEDBACK_STATUS,
    binding: WindBinding | None = None,
) -> dict[str, object]:
    del seed
    state0 = _initial_state_from_candidate(candidate)
    wind_binding = _wind_binding(candidate) if binding is None else binding
    admission = str(candidate.get("initial_state_admission_status", launch_gate_admission_status(state0)))
    descriptor = _wind_descriptor(candidate, state0, wind_binding)
    latency_fields = _latency_fields(config.latency_case, accepted=False, state_feedback_delay_applied=False, blocked_before_execution=True)
    return {
        **_common_row(candidate, state0),
        **_wind_binding_fields(wind_binding),
        "controller_mode": MISSION_CONTROLLER_MODE,
        "feedback_mode": "unavailable",
        "latency_case": str(config.latency_case),
        "claim_status": "simulation_only",
        "evidence_role": "blocked_partial",
        "governor_decision_status": "blocked_before_governor",
        "primary_rejection_reason": str(reason),
        "all_rejection_reasons": str(reason),
        "outcome_class": "blocked_partial",
        "accepted": False,
        "failure_label": str(reason),
        "duration_s": 0.0,
        "dwell_time_s": 0.0,
        "energy_initial_m": _specific_energy_height_m(state0),
        "energy_final_m": _specific_energy_height_m(state0),
        "energy_residual_m": 0.0,
        "minimum_margin_m": _initial_margin_m(state0),
        "minimum_speed_m_s": _speed_m_s(state0),
        "exit_speed_m_s": _speed_m_s(state0),
        "control_saturation": 0.0,
        "rollout_integrity_success": False,
        "entry_check_status": admission,
        "exit_check_status": "blocked_no_feedback_rollout",
        "mission_feedback_path_status": str(reason),
        **_terminal_state_fields(state0),
        **descriptor,
        **latency_fields,
    }


def _common_row(candidate: dict[str, object], state0: np.ndarray) -> dict[str, object]:
    family = str(candidate.get("primitive_family", candidate.get("family", "glide")))
    primitive_id = str(candidate.get("primitive_id", candidate.get("candidate_id", f"{candidate.get('sample_id', 'sample')}__{family}")))
    speed0 = _speed_m_s(state0)
    return {
        "sample_id": str(candidate.get("sample_id", "")),
        "paired_sample_key": str(candidate.get("paired_sample_key", "")),
        "fan_branch": str(candidate.get("fan_branch", "single_fan_branch")),
        "W_layer": str(candidate.get("W_layer", "W1")),
        "test_environment_mode": str(candidate.get("test_environment_mode", _environment_mode(candidate))),
        "entry_source": str(candidate.get("entry_source", "launch_gate_main")),
        "launch_gate_id": str(candidate.get("launch_gate_id", FIXED_LAUNCH_GATE.launch_gate_id)),
        "initial_state_vector": _state_text(state0),
        "initial_state_admission_status": str(candidate.get("initial_state_admission_status", launch_gate_admission_status(state0))),
        "primitive_id": primitive_id,
        "primitive_family": family,
        "x0_w_m": float(state0[0]),
        "y0_w_m": float(state0[1]),
        "z0_w_m": float(state0[2]),
        "phi0_rad": float(state0[3]),
        "theta0_rad": float(state0[4]),
        "psi0_rad": float(state0[5]),
        "speed0_m_s": speed0,
    }


def _initial_state_from_candidate(candidate: dict[str, object]) -> np.ndarray:
    if "initial_state_vector" in candidate and str(candidate["initial_state_vector"]).strip():
        values = [float(item) for item in str(candidate["initial_state_vector"]).split(";")]
        return as_state_vector(values)
    state = np.zeros(STATE_SIZE, dtype=float)
    source_names = (
        ("x_w_m", "x0_w_m"),
        ("y_w_m", "y0_w_m"),
        ("z_w_m", "z0_w_m"),
        ("phi_rad", "phi0_rad"),
        ("theta_rad", "theta0_rad"),
        ("psi_rad", "psi0_rad"),
        ("u_m_s",),
        ("v_m_s",),
        ("w_m_s",),
        ("p_rad_s",),
        ("q_rad_s",),
        ("r_rad_s",),
        ("delta_a_rad",),
        ("delta_e_rad",),
        ("delta_r_rad",),
    )
    for index, names in enumerate(source_names):
        state[index] = _first_float(candidate, names, 0.0)
    if state[6] == 0.0 and "speed0_m_s" in candidate:
        state[6] = _first_float(candidate, ("speed0_m_s",), 0.0)
    if state[6] == 0.0 and "speed_m_s" in candidate:
        state[6] = _first_float(candidate, ("speed_m_s",), 0.0)
    return as_state_vector(state)


# =============================================================================
# 4) Command-Template and Metric Helpers
# =============================================================================
def _command_schedule(candidate: dict[str, object], config: FixedGatePrimitiveRolloutConfig) -> CommandSchedule:
    time_s = np.arange(int(round(float(config.horizon_s) / float(config.dt_s))) + 1, dtype=float) * float(config.dt_s)
    spec = _candidate_spec(candidate, horizon_s=float(config.horizon_s))
    command, _ = generate_command_profile(spec, time_s)
    return CommandSchedule(times_s=time_s, u_norm_requested=command)


def _candidate_spec(candidate: dict[str, object], *, horizon_s: float) -> PrimitiveCandidateSpec:
    family = str(candidate.get("primitive_family", candidate.get("family", "glide")))
    mapped_family, direction_sign, target_heading_deg = _library_family_mapping(family)
    primitive_id = str(candidate.get("primitive_id", candidate.get("candidate_id", f"{candidate.get('sample_id', 'sample')}__{family}")))
    return PrimitiveCandidateSpec(
        primitive_id=primitive_id,
        parent_primitive_id=family,
        variant_id=primitive_id,
        family=mapped_family,
        target_heading_deg=target_heading_deg,
        updraft_config=str(candidate.get("test_environment_mode", "fixed_gate")),
        wind_fidelity=str(candidate.get("W_layer", "W1")),
        start_condition="fixed_gate",
        direction_sign=int(direction_sign),
        horizon_s=float(horizon_s),
    )


def _run_local_feedback_primitive(
    candidate: dict[str, object],
    state0: np.ndarray,
    config: FixedGatePrimitiveRolloutConfig,
    seed: int,
    latency_cfg: object,
    binding: WindBinding,
) -> FixedGateFeedbackReplayResult:
    family = str(candidate.get("primitive_family", candidate.get("family", "glide")))
    speed = max(_speed_m_s(state0), 1e-6)
    altitude = float(state0[2])
    tau = actuator_tau_for_case(latency_cfg)
    primitive_spec, feedback_context = _feedback_spec_and_context(
        family=family,
        state0=state0,
        dt_s=float(config.dt_s),
        horizon_s=float(config.horizon_s),
        speed_m_s=float(speed),
        altitude_m=float(altitude),
        actuator_tau_s=tau,
        seed=int(seed),
    )
    entry_checks = evaluate_entry_set(state0, primitive_spec.entry_set)
    if not _checks_pass(entry_checks):
        command0 = _feedback_command(
            family,
            state0,
            state0,
            np.zeros(3, dtype=float),
            0.0,
            feedback_context,
        )
        rollout_config = RolloutConfig(
            dt_s=float(config.dt_s),
            t_final_s=float(config.horizon_s),
            wind_mode=str(binding.wind_mode),
            latency_case=str(config.latency_case),
            actuator_tau_s=tau,
        )
        result = RolloutResult(
            time_s=np.array([0.0]),
            x=state0.reshape(1, STATE_SIZE),
            u_norm_requested=command0.reshape(1, 3),
            u_norm_applied=clip_normalised_command(command0).reshape(1, 3),
            delta_cmd_rad=normalised_command_to_surface_rad(command0).reshape(1, 3),
            success=False,
            failure_label="primitive_entry_check_failed",
            metrics=_feedback_metrics(
                np.array([0.0]),
                state0.reshape(1, STATE_SIZE),
                command0.reshape(1, 3),
                clip_normalised_command(command0).reshape(1, 3),
                rollout_config,
                failure_label="primitive_entry_check_failed",
            ),
            notes="entry_check_failed_before_feedback_replay",
        )
        exit_checks = evaluate_exit_checks(primitive_spec, result)
        return FixedGateFeedbackReplayResult(
            primitive_spec=primitive_spec,
            time_s=result.time_s,
            x=result.x,
            u_norm_requested=result.u_norm_requested,
            u_norm_applied=result.u_norm_applied,
            delta_cmd_rad=result.delta_cmd_rad,
            entry_checks=entry_checks,
            exit_checks=exit_checks,
            metrics=result.metrics,
            success=False,
            failure_label=result.failure_label,
            notes=result.notes,
        )
    return _feedback_replay_loop(
        candidate=candidate,
        family=family,
        state0=state0,
        config=config,
        seed=seed,
        latency_cfg=latency_cfg,
        binding=binding,
        primitive_spec=primitive_spec,
        feedback_context=feedback_context,
        entry_checks=entry_checks,
    )


def _feedback_spec_and_context(
    *,
    family: str,
    state0: np.ndarray,
    dt_s: float,
    horizon_s: float,
    speed_m_s: float,
    altitude_m: float,
    actuator_tau_s: tuple[float, float, float],
    seed: int,
) -> tuple[object, dict[str, object]]:
    x_ref, u_trim_norm = _trim_reference(speed_m_s, altitude_m, actuator_tau_s)
    x_ref = x_ref.copy()
    x_ref[0:3] = as_state_vector(state0)[0:3]
    if family in {"glide", "lift_entry"}:
        primitive_config = GlidePrimitiveConfig(
            dt_s=float(dt_s),
            t_final_s=float(horizon_s),
            speed_m_s=float(speed_m_s),
            altitude_m=float(altitude_m),
            wind_mode="none",
            latency_case="none",
            actuator_tau_s=actuator_tau_s,
            seed=int(seed),
            scenario_name=f"fixed_gate_{family}_partial_feedback",
        )
        return build_glide_primitive_spec(primitive_config), {
            "x_ref": x_ref,
            "u_trim_norm": u_trim_norm,
            "gains": GlideFeedbackGains(),
            "primitive_config": primitive_config,
        }
    if family == "recovery":
        primitive_config = RecoveryPrimitiveConfig(
            dt_s=float(dt_s),
            t_final_s=float(horizon_s),
            speed_m_s=float(speed_m_s),
            altitude_m=float(altitude_m),
            wind_mode="none",
            latency_case="none",
            actuator_tau_s=actuator_tau_s,
            seed=int(seed),
            scenario_name="fixed_gate_recovery_partial_feedback",
        )
        return build_recovery_primitive_spec(primitive_config), {
            "x_ref": x_ref,
            "u_trim_norm": u_trim_norm,
            "gains": RecoveryFeedbackGains(),
            "primitive_config": primitive_config,
        }
    if family in {
        "mild_coordinated_turn_left",
        "mild_coordinated_turn_right",
        "energy_retaining_bank",
        "lift_dwell_arc",
    }:
        primitive_config = BankPrimitiveConfig(
            dt_s=float(dt_s),
            t_final_s=float(horizon_s),
            speed_m_s=float(speed_m_s),
            altitude_m=float(altitude_m),
            wind_mode="none",
            latency_case="none",
            actuator_tau_s=actuator_tau_s,
            seed=int(seed),
            scenario_name=f"fixed_gate_{family}_partial_feedback",
            target_bank_deg=10.0,
        )
        bank_case = BankCaseSpec(
            name=f"fixed_gate_{family}_{seed}",
            role="fixed_gate_candidate",
            description=f"fixed-gate {family} partial-feedback candidate",
            direction_sign=int(_bank_direction_sign(family)),
            x0=state0,
            t_final_s=float(horizon_s),
        )
        return build_bank_primitive_spec(primitive_config), {
            "x_ref": x_ref,
            "u_trim_norm": u_trim_norm,
            "gains": BankFeedbackGains(),
            "primitive_config": primitive_config,
            "bank_case": bank_case,
        }
    raise ValueError(f"unsupported feedback primitive family: {family}")


_TRIM_REFERENCE_CACHE: dict[tuple[float, float, tuple[float, float, float]], tuple[np.ndarray, np.ndarray]] = {}


def _trim_reference(
    speed_m_s: float,
    altitude_m: float,
    actuator_tau_s: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    speed_key = round(float(speed_m_s), 2)
    altitude_key = round(float(altitude_m), 2)
    tau_key = tuple(round(float(value), 4) for value in actuator_tau_s)
    key = (speed_key, altitude_key, tau_key)
    if key not in _TRIM_REFERENCE_CACHE:
        aircraft = adapt_glider(build_nausicaa_glider())
        trim = solve_straight_trim(
            aircraft,
            TrimTarget(
                speed_m_s=float(speed_key),
                altitude_m=float(altitude_key),
                wind_model=None,
                wind_mode="none",
                actuator_tau_s=tuple(float(value) for value in tau_key),
            ),
        )
        if not bool(trim.converged):
            raise ValueError("straight trim did not converge for fixed-gate feedback replay.")
        _TRIM_REFERENCE_CACHE[key] = (
            as_state_vector(trim.x_trim).copy(),
            surface_rad_to_normalised_command(trim.u_cmd_trim).copy(),
        )
    x_ref, u_trim_norm = _TRIM_REFERENCE_CACHE[key]
    return x_ref.copy(), u_trim_norm.copy()


def _feedback_command(
    family: str,
    state: np.ndarray,
    x_ref: np.ndarray,
    u_trim_norm: np.ndarray,
    time_s: float,
    context: dict[str, object],
) -> np.ndarray:
    if family in {"glide", "lift_entry"}:
        return glide_feedback_command_norm(state, x_ref, u_trim_norm, context["gains"])
    if family == "recovery":
        return recovery_feedback_command_norm(state, x_ref, u_trim_norm, context["gains"])
    if family in {
        "mild_coordinated_turn_left",
        "mild_coordinated_turn_right",
        "energy_retaining_bank",
        "lift_dwell_arc",
    }:
        bank_ref = bank_reference_bank_rad(
            float(time_s),
            context["bank_case"],
            context["primitive_config"],
        )
        return bank_feedback_command_norm(state, x_ref, bank_ref, u_trim_norm, context["gains"])
    raise ValueError(f"unsupported feedback primitive family: {family}")


def _feedback_replay_loop(
    *,
    candidate: dict[str, object],
    family: str,
    state0: np.ndarray,
    config: FixedGatePrimitiveRolloutConfig,
    seed: int,
    latency_cfg: object,
    binding: WindBinding,
    primitive_spec: object,
    feedback_context: dict[str, object],
    entry_checks: tuple[object, ...],
) -> FixedGateFeedbackReplayResult:
    del candidate, seed
    rollout_config = RolloutConfig(
        dt_s=float(config.dt_s),
        t_final_s=float(config.horizon_s),
        wind_mode=str(binding.wind_mode),
        latency_case=str(config.latency_case),
        actuator_tau_s=actuator_tau_for_case(latency_cfg),
    )
    time_s = _feedback_time_grid(config)
    sample_count = int(time_s.size)
    x_log = np.empty((sample_count, STATE_SIZE), dtype=float)
    requested_log = np.empty((sample_count, 3), dtype=float)
    applied_log = np.empty((sample_count, 3), dtype=float)
    command_rad_log = np.empty((sample_count, 3), dtype=float)
    x_log[0] = state0
    aircraft = adapt_glider(build_nausicaa_glider())
    x_ref = np.asarray(feedback_context["x_ref"], dtype=float).reshape(STATE_SIZE)
    u_trim_norm = np.asarray(feedback_context["u_trim_norm"], dtype=float).reshape(3)
    failure_label = "not_run"
    notes = "instant_state_feedback_replay"
    final_index = sample_count - 1

    for index in range(sample_count):
        requested = _feedback_command(
            family,
            x_log[index],
            x_ref,
            u_trim_norm,
            float(time_s[index]),
            feedback_context,
        )
        requested_log[index] = requested
        delayed_requested = latency_adjusted_command_sample(
            time_s[: index + 1],
            requested_log[: index + 1],
            float(time_s[index]),
            latency_cfg,
        )
        applied = clip_normalised_command(delayed_requested)
        command_rad = normalised_command_to_surface_rad(applied)
        if str(config.latency_case) == "none":
            x_log[index, 12:15] = command_rad
        applied_log[index] = applied
        command_rad_log[index] = command_rad
        if index == sample_count - 1:
            break
        next_state = rk4_step(
            x_log[index],
            command_rad,
            float(config.dt_s),
            aircraft,
            binding.wind_model,
            str(binding.wind_mode),
            actuator_tau_for_case(latency_cfg),
        )
        x_log[index + 1] = next_state
        if not np.all(np.isfinite(next_state)):
            failure_label = "nonfinite_state"
            notes = "nonfinite_state"
            final_index = index + 1
            break
        if _initial_margin_m(next_state) < 0.0:
            failure_label = "true_safety_violation"
            notes = "true_safety_violation"
            final_index = index + 1
            break

    time_s = time_s[: final_index + 1]
    x_log = x_log[: final_index + 1]
    requested_log = requested_log[: final_index + 1]
    applied_log = applied_log[: final_index + 1]
    command_rad_log = command_rad_log[: final_index + 1]
    metrics = _feedback_metrics(
        time_s,
        x_log,
        requested_log,
        applied_log,
        rollout_config,
        failure_label=failure_label,
    )
    rollout_result = RolloutResult(
        time_s=time_s,
        x=x_log,
        u_norm_requested=requested_log,
        u_norm_applied=applied_log,
        delta_cmd_rad=command_rad_log,
        success=bool(metrics["rollout_success"]),
        failure_label=failure_label,
        metrics=metrics,
        notes=notes,
    )
    exit_checks = evaluate_exit_checks(primitive_spec, rollout_result)
    success = bool(_checks_pass(entry_checks) and _checks_pass(exit_checks))
    final_failure = "success" if success else (failure_label if failure_label != "not_run" else "primitive_exit_check_failed")
    return FixedGateFeedbackReplayResult(
        primitive_spec=primitive_spec,
        time_s=time_s,
        x=x_log,
        u_norm_requested=requested_log,
        u_norm_applied=applied_log,
        delta_cmd_rad=command_rad_log,
        entry_checks=entry_checks,
        exit_checks=exit_checks,
        metrics=metrics,
        success=success,
        failure_label=final_failure,
        notes=notes,
    )


def _feedback_time_grid(config: FixedGatePrimitiveRolloutConfig) -> np.ndarray:
    ratio = float(config.horizon_s) / float(config.dt_s)
    rounded = int(round(ratio))
    if not np.isclose(ratio, rounded, rtol=1e-12, atol=1e-9):
        raise ValueError("fixed-gate feedback horizon_s must be an integer multiple of dt_s.")
    return np.arange(rounded + 1, dtype=float) * float(config.dt_s)


def _feedback_metrics(
    time_s: np.ndarray,
    x_log: np.ndarray,
    requested_log: np.ndarray,
    applied_log: np.ndarray,
    config: RolloutConfig,
    *,
    failure_label: str,
) -> dict[str, object]:
    finite = bool(np.all(np.isfinite(x_log)))
    in_bounds = bool(all(_initial_margin_m(state) >= 0.0 for state in np.asarray(x_log, dtype=float)))
    rollout_success = bool(finite and in_bounds and str(failure_label) == "not_run")
    saturation = np.any(np.abs(np.asarray(requested_log) - np.asarray(applied_log)) > 1e-12, axis=1)
    if saturation.size > 1:
        saturation_fraction = float(np.count_nonzero(saturation[:-1]) / max(1, saturation.size - 1))
    else:
        saturation_fraction = 0.0
    return {
        "wind_mode": str(config.wind_mode),
        "latency_case": str(config.latency_case),
        "finite_state_success": finite,
        "rollout_success": rollout_success,
        "closed_loop_replay_success": rollout_success,
        "primitive_success": False,
        "failure_label": str(failure_label),
        "duration_s": float(time_s[-1] - time_s[0]) if time_s.size else 0.0,
        "saturation_fraction": saturation_fraction,
    }


def _bank_direction_sign(family: str) -> int:
    if str(family) == "mild_coordinated_turn_left":
        return -1
    return 1


def _library_family_mapping(family: str) -> tuple[str, int, float | None]:
    if family == "glide":
        return "glide", 1, None
    if family == "recovery":
        return "recovery", 1, None
    if family == "mild_coordinated_turn_left":
        return "mild_bank", 1, None
    if family == "mild_coordinated_turn_right":
        return "mild_bank", -1, None
    if family == "energy_retaining_bank":
        return "mild_bank", 1, None
    if family in {"lift_entry", "lift_dwell_arc"}:
        # No measured lift-entry feedback primitive exists yet. The glide
        # command template keeps the ablation row dynamically executable while
        # the evidence_role prevents mission promotion.
        return "glide", 1, None
    return "glide", 1, None


def _wind_binding(candidate: dict[str, object]) -> WindBinding:
    layer = str(candidate.get("W_layer", "W1"))
    branch = str(candidate.get("fan_branch", "single_fan_branch"))
    if layer == "W0":
        return WindBinding(
            wind_model=None,
            wind_mode="none",
            updraft_model_id="no_updraft_dry_air",
            updraft_model_source="dry_air_no_updraft_w0_baseline",
            wind_binding_status="dry_air_w0_baseline",
            dry_air=True,
        )
    if layer in {"W1", "W2"}:
        model_id = W1_MODEL_BY_BRANCH.get(branch, "")
        if not model_id:
            reason = f"{WIND_BINDING_BLOCKED_STATUS}:unsupported_fan_branch:{branch}"
            return WindBinding(
                wind_model=None,
                wind_mode="none",
                updraft_model_id="unavailable",
                updraft_model_source=reason,
                wind_binding_status=reason,
                blocked_reason=reason,
            )
        try:
            model = load_updraft_model(model_id)
        except Exception as exc:
            reason = f"{WIND_BINDING_BLOCKED_STATUS}:{model_id}:{exc}"
            return WindBinding(
                wind_model=None,
                wind_mode="none",
                updraft_model_id=model_id,
                updraft_model_source=reason,
                wind_binding_status=reason,
                blocked_reason=reason,
            )
        return WindBinding(
            wind_model=model,
            wind_mode="panel",
            updraft_model_id=str(getattr(model, "name", model_id)),
            updraft_model_source=str(getattr(model, "source", "unknown_measured_updraft_source")),
            wind_binding_status=W1_MEASURED_WIND_STATUS if layer == "W1" else "measured_updraft_bound_focused_replay",
            dry_air=False,
        )
    reason = "blocked_domain_randomised_wind_binding_not_implemented"
    return WindBinding(
        wind_model=None,
        wind_mode="none",
        updraft_model_id="unavailable",
        updraft_model_source=reason,
        wind_binding_status=reason,
        blocked_reason=reason,
    )


def _wind_binding_fields(binding: WindBinding) -> dict[str, object]:
    return {
        "wind_mode": str(binding.wind_mode),
        "updraft_model_id": str(binding.updraft_model_id),
        "updraft_model_source": str(binding.updraft_model_source),
        "wind_binding_status": str(binding.wind_binding_status),
    }


def _wind_descriptor(candidate: dict[str, object], state: np.ndarray, binding: WindBinding) -> dict[str, object]:
    environment = str(candidate.get("test_environment_mode", _environment_mode(candidate)))
    if binding.blocked_reason:
        return _blocked_wind_descriptor(candidate, binding, environment)
    row = wing_wind_descriptor_row(
        wind_field=None if binding.dry_air else binding.wind_model,
        x_w_m=float(state[0]),
        y_w_m=float(state[1]),
        z_w_m=float(state[2]),
        phi_rad=float(state[3]),
        theta_rad=float(state[4]),
        psi_rad=float(state[5]),
        fan_layout=_fan_layout(str(candidate.get("fan_branch", "single_fan_branch"))),
        fan_config_id=str(candidate.get("fan_branch", "single_fan_branch")),
        environment_mode=environment,
        model_id=str(binding.updraft_model_id),
        model_source=str(binding.updraft_model_source),
        dry_air=bool(binding.dry_air),
    )
    row["spanwise_gradient_m_s_m"] = float(row["spanwise_w_gradient_m_s_per_m"])
    return row


def _blocked_wind_descriptor(
    candidate: dict[str, object],
    binding: WindBinding,
    environment: str,
) -> dict[str, object]:
    del candidate
    return {
        "wind_descriptor_status": str(binding.wind_binding_status),
        "wind_descriptor_environment_mode": str(environment),
        "wind_descriptor_model_id": str(binding.updraft_model_id),
        "wind_descriptor_model_source": str(binding.updraft_model_source),
        "w_wing_mean_m_s": float("nan"),
        "delta_w_lr_m_s": float("nan"),
        "spanwise_gradient_m_s_m": float("nan"),
        "spanwise_w_gradient_m_s_per_m": float("nan"),
    }


def _latency_fields(
    latency_case: str,
    *,
    accepted: bool,
    state_feedback_delay_applied: bool,
    blocked_before_execution: bool = False,
) -> dict[str, object]:
    latency_cfg = latency_case_config(latency_case)
    fields = latency_audit_fields_from_case(latency_cfg, active_actuator_tau_s=actuator_tau_for_case(latency_cfg))
    if blocked_before_execution:
        flags = {
            "state_feedback_delay_applied": False,
            "command_delay_applied": False,
            "actuator_lag_applied": False,
        }
    else:
        flags = latency_mechanism_flags_from_case(
            latency_case,
            state_feedback_delay_applied=bool(state_feedback_delay_applied),
        )
    fields.update(flags)
    fields["latency_pass_label"] = latency_pass_label_for_single_run(latency_case, accepted)
    return fields


def _partial_feedback_dwell_time(candidate: dict[str, object], result: object, primitive_success: bool) -> float:
    family = str(candidate.get("primitive_family", ""))
    if not primitive_success or family not in {"lift_entry", "lift_dwell_arc"}:
        return 0.0
    return float(result.time_s[-1] - result.time_s[0]) if getattr(result, "time_s", np.array([])).size else 0.0


def _entry_check_status(candidate: dict[str, object], state0: np.ndarray, primitive_entry_pass: bool) -> str:
    admission = str(candidate.get("initial_state_admission_status", launch_gate_admission_status(state0)))
    suffix = "primitive_entry_pass" if primitive_entry_pass else "primitive_entry_failed"
    return f"{admission}|{suffix}"


def _exit_check_failure_status(exit_checks: object, failure_label: str) -> str:
    checks = tuple(exit_checks) if exit_checks else ()
    failed = [
        str(getattr(check, "name", "unknown_exit_check"))
        for check in checks
        if bool(getattr(check, "required", True)) and not bool(getattr(check, "pass_check", False))
    ]
    if failed:
        return "primitive_exit_failed:" + ";".join(failed)
    return f"primitive_exit_not_passed:{failure_label}"


def _checks_pass(checks: object) -> bool:
    check_tuple = tuple(checks) if checks else ()
    return bool(check_tuple and all(bool(getattr(check, "pass_check", False)) for check in check_tuple if bool(getattr(check, "required", True))))


def _terminal_state_fields(state: np.ndarray) -> dict[str, object]:
    terminal = as_state_vector(state)
    return {
        "x_terminal_w_m": float(terminal[0]),
        "y_terminal_w_m": float(terminal[1]),
        "z_terminal_w_m": float(terminal[2]),
        "phi_terminal_rad": float(terminal[3]),
        "theta_terminal_rad": float(terminal[4]),
        "psi_terminal_rad": float(terminal[5]),
        "u_terminal_m_s": float(terminal[6]),
        "v_terminal_m_s": float(terminal[7]),
        "w_terminal_m_s": float(terminal[8]),
        "p_terminal_rad_s": float(terminal[9]),
        "q_terminal_rad_s": float(terminal[10]),
        "r_terminal_rad_s": float(terminal[11]),
    }


def _minimum_true_margin(x_log: np.ndarray) -> float:
    values = [_initial_margin_m(state) for state in np.asarray(x_log, dtype=float)]
    finite = [value for value in values if np.isfinite(value)]
    return float(min(finite)) if finite else float("nan")


def _initial_margin_m(state: np.ndarray) -> float:
    margins = position_margin_m(as_state_vector(state)[0:3], TRUE_SAFE_BOUNDS)
    return float(min(margins["min_wall_margin_m"], margins["floor_margin_m"], margins["ceiling_margin_m"]))


def _minimum_speed_m_s(x_log: np.ndarray) -> float:
    speeds = np.linalg.norm(np.asarray(x_log, dtype=float)[:, 6:9], axis=1)
    finite = speeds[np.isfinite(speeds)]
    return float(np.min(finite)) if finite.size else float("nan")


def _specific_energy_height_m(state: np.ndarray) -> float:
    return float(as_state_vector(state)[2] + _speed_m_s(state) ** 2 / (2.0 * 9.81))


def _speed_m_s(state: np.ndarray) -> float:
    return float(np.linalg.norm(as_state_vector(state)[6:9]))


def _state_text(state: np.ndarray) -> str:
    return ";".join(f"{float(value):.12g}" for value in as_state_vector(state))


# =============================================================================
# 5) Readiness Summaries
# =============================================================================
def _mission_or_partial_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "evidence_role" not in frame.columns:
        return pd.DataFrame(columns=frame.columns)
    return frame[
        frame["evidence_role"].astype(str).isin({"mission_candidate", "partial_feedback"})
        & frame["accepted"].astype(bool)
    ].copy()


def _accepted_role_rows(frame: pd.DataFrame, role: str) -> pd.DataFrame:
    if frame.empty or "evidence_role" not in frame.columns:
        return pd.DataFrame(columns=frame.columns)
    return frame[
        frame["evidence_role"].astype(str).eq(str(role))
        & frame["accepted"].astype(bool)
    ].copy()


def _accepted_role_layer_rows(frame: pd.DataFrame, role: str, layer: str) -> pd.DataFrame:
    if frame.empty or not {"evidence_role", "W_layer", "accepted"}.issubset(frame.columns):
        return pd.DataFrame(columns=frame.columns)
    return frame[
        frame["evidence_role"].astype(str).eq(str(role))
        & frame["W_layer"].astype(str).eq(str(layer))
        & frame["accepted"].astype(bool)
    ].copy()


def _role_count(frame: pd.DataFrame, role: str) -> int:
    if frame.empty or "evidence_role" not in frame.columns:
        return 0
    return int(frame["evidence_role"].astype(str).eq(str(role)).sum())


def _layer_count_by_branch(frame: pd.DataFrame, layer: str) -> dict[str, int]:
    if frame.empty or not {"fan_branch", "W_layer"}.issubset(frame.columns):
        return {}
    layer_rows = frame[frame["W_layer"].astype(str).eq(str(layer))]
    return {str(key): int(value) for key, value in layer_rows.groupby("fan_branch").size().to_dict().items()}


def _w1_measured_updraft_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=frame.columns)
    required = {"W_layer", "wind_mode", "wind_binding_status", "wind_descriptor_status", "wind_descriptor_model_source"}
    if not required.issubset(frame.columns):
        return pd.DataFrame(columns=frame.columns)
    w1 = frame[frame["W_layer"].astype(str).eq("W1")].copy()
    if w1.empty:
        return w1
    measured = (
        w1["wind_mode"].astype(str).ne("none")
        & w1["wind_binding_status"].astype(str).eq(W1_MEASURED_WIND_STATUS)
        & w1["wind_descriptor_model_source"].astype(str).ne("dry_air_no_updraft_w0_baseline")
        & w1["wind_descriptor_status"].astype(str).eq("wind_model_evaluated")
    )
    return w1[measured].copy()


def _archive_prepared_block_reason(paired_branch_layers_present: bool, measured_w1_present: bool) -> str:
    if not paired_branch_layers_present:
        return "blocked_missing_branch_layer_rows"
    if not measured_w1_present:
        return "blocked_missing_measured_w1_wind_binding_for_both_branches"
    return "blocked_archive_prepared_unknown"


def _has_required_evidence_columns(frame: pd.DataFrame) -> bool:
    return bool(set(ROW_COLUMNS).issubset(frame.columns))


def _has_promoted_diagnostic_rows(frame: pd.DataFrame) -> bool:
    if frame.empty or not {"controller_mode", "evidence_role"}.issubset(frame.columns):
        return False
    diagnostic_controller = frame["controller_mode"].astype(str).isin({"open_loop_rollout", "command_template_replay"})
    promoted = diagnostic_controller & frame["evidence_role"].astype(str).isin({"mission_candidate", "partial_feedback"})
    return bool(promoted.any())


def _w1_independent(frame: pd.DataFrame) -> bool:
    if frame.empty:
        return False
    keys = ["paired_sample_key", "fan_branch", "primitive_family"]
    w0 = set(map(tuple, frame[frame["W_layer"].astype(str).eq("W0")][keys].astype(str).to_numpy()))
    w1 = set(map(tuple, frame[frame["W_layer"].astype(str).eq("W1")][keys].astype(str).to_numpy()))
    return bool(w0 and w0 == w1)


def _requested_execution_modes(
    candidate: dict[str, object],
    config: FixedGatePrimitiveRolloutConfig,
) -> tuple[tuple[str, str], ...]:
    candidate_controller = str(candidate.get("controller_mode", "")).strip()
    candidate_feedback = str(candidate.get("feedback_mode", "")).strip()
    if candidate_controller:
        feedback = _normalised_feedback_request(candidate_controller, candidate_feedback or config.feedback_mode)
        return ((candidate_controller, feedback),)
    controller = str(config.controller_mode)
    if controller == "both":
        return (
            (DIAGNOSTIC_CONTROLLER_MODE, DIAGNOSTIC_FEEDBACK_MODE),
            (MISSION_CONTROLLER_MODE, PARTIAL_FEEDBACK_MODE),
        )
    feedback = _normalised_feedback_request(controller, str(config.feedback_mode))
    return ((controller, feedback),)


def _normalised_feedback_request(controller_mode: str, feedback_mode: str) -> str:
    if str(controller_mode) != MISSION_CONTROLLER_MODE:
        return DIAGNOSTIC_FEEDBACK_MODE
    if str(feedback_mode) in {"", DIAGNOSTIC_FEEDBACK_MODE, "not_applicable"}:
        return PARTIAL_FEEDBACK_MODE
    return str(feedback_mode)


def _is_delayed_feedback_request(feedback_mode: str) -> bool:
    return str(feedback_mode) == MISSION_FEEDBACK_MODE


def _environment_mode(candidate: dict[str, object]) -> str:
    branch = str(candidate.get("fan_branch", "single_fan_branch"))
    layer = str(candidate.get("W_layer", "W1"))
    if branch == "single_fan_branch":
        return "W0_single_fan_branch" if layer == "W0" else f"{layer}_single_fan"
    return "W0_four_fan_branch" if layer == "W0" else f"{layer}_four_fan"


def _fan_layout(fan_branch: str) -> str:
    return "four_fan" if str(fan_branch) == "four_fan_branch" else "single_fan"


def _first_float(row: dict[str, object], names: tuple[str, ...], default: float) -> float:
    for name in names:
        if name in row and str(row[name]) != "":
            try:
                value = float(row[name])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                return value
    return float(default)
