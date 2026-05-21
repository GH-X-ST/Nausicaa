from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from arena_contract import TRUE_SAFE_BOUNDS, position_margin_m
from episode_schema import validate_primitive_rollout_evidence_frame
from fixed_gate_contract import FIXED_LAUNCH_GATE, launch_gate_admission_status
from latency import (
    actuator_tau_for_case,
    latency_audit_fields_from_case,
    latency_case_config,
    latency_mechanism_flags_from_case,
    latency_pass_label_for_single_run,
)
from primitive_library_generators import generate_command_profile
from primitive_library_schema import PrimitiveCandidateSpec
from rollout import CommandSchedule, RolloutConfig, rollout_open_loop_normalised
from state_contract import STATE_SIZE, as_state_vector
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
DIAGNOSTIC_CONTROLLER_MODE = "open_loop_rollout"
DIAGNOSTIC_FEEDBACK_MODE = "open_loop"
BLOCKED_FEEDBACK_STATUS = "blocked_true_delayed_state_feedback_unavailable"


@dataclass(frozen=True)
class FixedGatePrimitiveRolloutConfig:
    dt_s: float = 0.02
    horizon_s: float = 0.60
    latency_case: str = "nominal"
    random_seed: int = 20260521
    controller_mode: str = DIAGNOSTIC_CONTROLLER_MODE
    feedback_mode: str = DIAGNOSTIC_FEEDBACK_MODE
    allow_open_loop_diagnostic: bool = True


# =============================================================================
# 2) Public Rollout Workflow
# =============================================================================
def run_fixed_gate_primitive_rollouts(
    candidate_rows: pd.DataFrame,
    config: FixedGatePrimitiveRolloutConfig | None = None,
) -> pd.DataFrame:
    """Execute fixed-gate primitive row evidence with explicit hierarchy labels.

    The current reusable primitive execution backend is command-template/open
    loop. These rows are retained because they validate sampling, dynamics,
    latency labels, descriptors, and storage, but they are intentionally marked
    as diagnostic. Mission-candidate rows are emitted only by a delayed-state
    feedback primitive path, which is not yet available in this adapter.
    """

    cfg = FixedGatePrimitiveRolloutConfig() if config is None else config
    if candidate_rows.empty:
        return pd.DataFrame(columns=ROW_COLUMNS)
    latency_cfg = latency_case_config(cfg.latency_case)
    rows: list[dict[str, object]] = []
    for local_index, candidate in enumerate(candidate_rows.to_dict(orient="records")):
        requested_controller = str(candidate.get("controller_mode", cfg.controller_mode))
        requested_feedback = str(candidate.get("feedback_mode", cfg.feedback_mode))
        seed = int(cfg.random_seed) + int(local_index)
        if _requests_mission_feedback(requested_controller, requested_feedback):
            rows.append(_blocked_feedback_row(candidate, cfg, seed))
            continue
        if not bool(cfg.allow_open_loop_diagnostic):
            rows.append(_blocked_feedback_row(candidate, cfg, seed))
            continue
        rows.append(_open_loop_diagnostic_row(candidate, cfg, seed, latency_cfg))
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
    mission = _mission_rows(rollout_rows)
    branch_layer_counts = (
        mission.groupby(["fan_branch", "W_layer"]).size().to_dict()
        if not mission.empty
        else {}
    )
    branches = {"single_fan_branch", "four_fan_branch"}
    archive_ready = all(
        int(branch_layer_counts.get((branch, layer), 0)) > 0
        for branch in branches
        for layer in ("W0", "W1")
    )
    return {
        "code_readiness": "ready" if _has_required_evidence_columns(rollout_rows) else "blocked_schema",
        "archive_readiness": "ready" if archive_ready else "blocked_no_mission_candidate_rows_for_both_branches",
        "w_ladder_readiness": "ready" if _w1_independent(rollout_rows) else "blocked_w1_pairing_missing",
        "reachable_state_readiness": "ready" if reachable_count > 0 else "blocked_no_reachable_downstream_extraction",
        "mission_candidate_row_count": int(len(mission)),
        "ablation_diagnostic_row_count": int(rollout_rows["evidence_role"].astype(str).eq("ablation_diagnostic").sum()) if not rollout_rows.empty else 0,
        "reachable_downstream_row_count": reachable_count,
        "feedback_path_status": BLOCKED_FEEDBACK_STATUS,
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
    admission = str(candidate.get("initial_state_admission_status", launch_gate_admission_status(state0)))
    schedule = _command_schedule(candidate, config)
    wind_model = None
    result = rollout_open_loop_normalised(
        state0,
        schedule,
        RolloutConfig(
            dt_s=float(config.dt_s),
            t_final_s=float(config.horizon_s),
            wind_mode="none",
            latency_case=str(config.latency_case),
            actuator_tau_s=actuator_tau_for_case(latency_cfg),
        ),
        wind_model=wind_model,
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
    descriptor = _dry_wind_descriptor(candidate, state0)
    return {
        **_common_row(candidate, state0),
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


def _blocked_feedback_row(
    candidate: dict[str, object],
    config: FixedGatePrimitiveRolloutConfig,
    seed: int,
) -> dict[str, object]:
    del seed
    state0 = _initial_state_from_candidate(candidate)
    admission = str(candidate.get("initial_state_admission_status", launch_gate_admission_status(state0)))
    descriptor = _dry_wind_descriptor(candidate, state0)
    latency_fields = _latency_fields(config.latency_case, accepted=False, state_feedback_delay_applied=False, blocked_before_execution=True)
    return {
        **_common_row(candidate, state0),
        "controller_mode": MISSION_CONTROLLER_MODE,
        "feedback_mode": "unavailable",
        "latency_case": str(config.latency_case),
        "claim_status": "simulation_only",
        "evidence_role": "blocked_partial",
        "governor_decision_status": "blocked_before_governor",
        "primary_rejection_reason": BLOCKED_FEEDBACK_STATUS,
        "all_rejection_reasons": BLOCKED_FEEDBACK_STATUS,
        "outcome_class": "blocked_partial",
        "accepted": False,
        "failure_label": BLOCKED_FEEDBACK_STATUS,
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
        "exit_check_status": "blocked_no_delayed_state_feedback_rollout",
        "mission_feedback_path_status": BLOCKED_FEEDBACK_STATUS,
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


def _dry_wind_descriptor(candidate: dict[str, object], state: np.ndarray) -> dict[str, object]:
    environment = str(candidate.get("test_environment_mode", _environment_mode(candidate)))
    row = wing_wind_descriptor_row(
        wind_field=None,
        x_w_m=float(state[0]),
        y_w_m=float(state[1]),
        z_w_m=float(state[2]),
        phi_rad=float(state[3]),
        theta_rad=float(state[4]),
        psi_rad=float(state[5]),
        fan_layout=_fan_layout(str(candidate.get("fan_branch", "single_fan_branch"))),
        fan_config_id=str(candidate.get("fan_branch", "single_fan_branch")),
        environment_mode=environment,
        model_id="dry_air_fixed_gate_diagnostic",
        model_source="dry_air_no_updraft_diagnostic",
        dry_air=True,
    )
    row["spanwise_gradient_m_s_m"] = float(row["spanwise_w_gradient_m_s_per_m"])
    return row


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
def _mission_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "evidence_role" not in frame.columns:
        return pd.DataFrame(columns=frame.columns)
    return frame[
        frame["evidence_role"].astype(str).eq("mission_candidate")
        & frame["accepted"].astype(bool)
    ].copy()


def _has_required_evidence_columns(frame: pd.DataFrame) -> bool:
    return bool(set(ROW_COLUMNS).issubset(frame.columns))


def _w1_independent(frame: pd.DataFrame) -> bool:
    if frame.empty:
        return False
    keys = ["paired_sample_key", "fan_branch", "primitive_family"]
    w0 = set(map(tuple, frame[frame["W_layer"].astype(str).eq("W0")][keys].astype(str).to_numpy()))
    w1 = set(map(tuple, frame[frame["W_layer"].astype(str).eq("W1")][keys].astype(str).to_numpy()))
    return bool(w0 and w0 == w1)


def _requests_mission_feedback(controller_mode: str, feedback_mode: str) -> bool:
    return str(controller_mode) == MISSION_CONTROLLER_MODE or str(feedback_mode) == MISSION_FEEDBACK_MODE


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
