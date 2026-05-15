from __future__ import annotations

from pathlib import Path

import numpy as np

from arena import ArenaConfig, safety_margins
from linearisation import STATE_INDEX


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Stable metrics schema
# 2) Path and failure-class helpers
# 3) Rollout metric extraction
# 4) Governor-rejected metric rows
# =============================================================================

# =============================================================================
# 1) Stable Metrics Schema
# =============================================================================
# CSV field order is part of the reproducibility contract for scenario comparisons.
METRIC_SCHEMA_KEYS = (
    "run_id",
    "scenario_id",
    "primitive_name",
    "selected_primitive",
    "seed",
    "wind_model",
    "wind_mode",
    "wind_param_label",
    "latency_mode",
    "latency_s",
    "latency_range_s",
    "state_feedback_delay_s",
    "actuator_t10_s",
    "actuator_t50_nominal_s",
    "actuator_t90_s",
    "conservative_actuator_bound_s",
    "vicon_filter_cutoff_hz",
    "vicon_filter_model",
    "duration_s",
    "termination_reason",
    "success",
    "failure_class",
    "primitive_family",
    "candidate_id",
    "is_full_turn_claim",
    "target_heading_deg",
    "heading_target_deg",
    "heading_change_deg",
    "actual_heading_change_deg",
    "heading_error_deg",
    "forward_travel_m",
    "turn_volume_proxy_m2",
    "height_change_m",
    "terminal_speed_m_s",
    "max_alpha_deg",
    "max_beta_deg",
    "max_bank_deg",
    "min_wall_distance_m",
    "min_floor_margin_m",
    "min_ceiling_margin_m",
    "saturation_time_s",
    "time_at_full_travel_s",
    "saturation_fraction",
    "tracking_error_rms",
    "exit_recoverable",
    "fixed_start_success",
    "feasibility_label",
    "governor_rejection_reason",
    "candidate_count",
    "rejected_count",
    "log_path",
)


# =============================================================================
# 2) Path and Failure-Class Helpers
# =============================================================================
def relative_path(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repo_root.resolve())).replace("\\", "/")
    except ValueError:
        return str(resolved).replace("\\", "/")


def failure_class(
    termination_reason: str,
    governor_rejection_reason: str = "",
    wind_model: str = "",
) -> str:
    # Failure classes stay conservative when logs cannot isolate the source
    reason = f"{termination_reason} {governor_rejection_reason}".lower()
    wind = wind_model.lower()
    if not reason.strip():
        return "none"
    if "governor" in reason or "entry" in reason:
        return "governor"
    if "speed" in reason or "angle of attack" in reason or "bank" in reason or "pitch" in reason:
        return "model"
    if "wall" in reason or "floor" in reason or "ceiling" in reason or "safe volume" in reason:
        return "governor"
    if "saturation" in reason:
        return "actuator"
    if "latency" in reason or "timing" in reason:
        return "latency"
    if "non-finite" in reason or "nonfinite" in reason or "nan" in reason:
        return "model"
    if ("updraft" in reason or "wind" in reason) and wind not in {"", "none"}:
        return "environment"
    if termination_reason == "governor_rejected":
        return "governor"
    return "unknown"


def _speed(states: np.ndarray) -> np.ndarray:
    return np.linalg.norm(states[:, 6:9], axis=1)


def _heading_delta_deg(states: np.ndarray) -> float:
    delta = float(states[-1, STATE_INDEX["psi"]] - states[0, STATE_INDEX["psi"]])
    return float(np.rad2deg((delta + np.pi) % (2.0 * np.pi) - np.pi))


def _metric_with_required_schema(row: dict[str, object]) -> dict[str, object]:
    stable = {key: row.get(key, None) for key in METRIC_SCHEMA_KEYS}
    for key, value in row.items():
        if key not in stable:
            stable[key] = value
    return stable


# =============================================================================
# 3) Rollout Metric Extraction
# =============================================================================
def rollout_metrics(
    scenario_id: str,
    seed: int,
    wind_model: str,
    wind_mode: str,
    latency_mode: str,
    latency_s: float | None,
    latency_range_s: str | None,
    primitive_selected: str,
    success: bool,
    termination_reason: str,
    states: np.ndarray,
    log_path: Path,
    repo_root: Path,
    arena_config: ArenaConfig,
    saturation_fraction: float,
    duration_s: float = 0.0,
    wind_param_label: str = "",
    selected_primitive: str | None = None,
    saturation_time_s: float | None = None,
    tracking_error_rms: float | None = None,
    state_feedback_delay_s: float | None = None,
    actuator_t10_s: float | None = None,
    actuator_t50_nominal_s: float | None = None,
    actuator_t90_s: float | None = None,
    conservative_actuator_bound_s: float | None = None,
    vicon_filter_cutoff_hz: float | None = None,
    vicon_filter_model: str = "",
    governor_rejection_reason: str = "",
    candidate_count: int = 0,
    rejected_count: int = 0,
    primitive_metadata: dict[str, object] | None = None,
) -> dict[str, float | str | bool | int]:
    state_arr = np.asarray(states, dtype=float)
    final = state_arr[-1]
    margins = [safety_margins(row, arena_config) for row in state_arr]
    min_wall = min(float(row["min_wall_distance_m"]) for row in margins)
    min_floor = min(float(row["floor_margin_m"]) for row in margins)
    min_ceiling = min(float(row["ceiling_margin_m"]) for row in margins)
    inside = all(bool(row["inside_safe_volume"]) for row in margins)
    speed = _speed(state_arr)
    x_span = float(
        np.max(state_arr[:, STATE_INDEX["x_w"]]) - np.min(state_arr[:, STATE_INDEX["x_w"]])
    )
    y_span = float(
        np.max(state_arr[:, STATE_INDEX["y_w"]]) - np.min(state_arr[:, STATE_INDEX["y_w"]])
    )
    # Alpha and beta are reported in degrees for human-facing metrics
    alpha = np.arctan2(
        state_arr[:, STATE_INDEX["w"]],
        np.maximum(state_arr[:, STATE_INDEX["u"]], 1e-12),
    )
    beta = np.arcsin(
        np.clip(
            state_arr[:, STATE_INDEX["v"]] / np.maximum(speed, 1e-12),
            -1.0,
            1.0,
        )
    )
    rel_log_path = relative_path(log_path, repo_root) if str(log_path) else ""
    selected_name = selected_primitive or primitive_selected
    reason = str(termination_reason)
    heading_change_deg = _heading_delta_deg(state_arr)
    metadata = _scenario_target_metadata(scenario_id, primitive_selected)
    if primitive_metadata is not None:
        metadata.update(dict(primitive_metadata))
    terminal_speed = float(speed[-1])
    terminal_recoverable = bool(
        success
        and inside
        and 2.5 <= terminal_speed <= 9.5
        and abs(float(final[STATE_INDEX["phi"]])) <= np.deg2rad(65.0)
        and abs(float(final[STATE_INDEX["theta"]])) <= np.deg2rad(40.0)
    )
    feasibility = _agile_feasibility_fields(
        metadata=metadata,
        success=bool(success),
        termination_reason=reason,
        heading_change_deg=heading_change_deg,
        min_wall_distance_m=float(min_wall),
        min_floor_margin_m=float(min_floor),
        min_ceiling_margin_m=float(min_ceiling),
        exit_recoverable=terminal_recoverable,
        max_alpha_deg=float(np.rad2deg(np.max(np.abs(alpha)))),
        saturation_fraction=float(saturation_fraction),
    )
    row = {
        "run_id": f"{scenario_id}_seed{int(seed)}",
        "scenario_id": scenario_id,
        "primitive_name": primitive_selected,
        "selected_primitive": selected_name,
        "seed": int(seed),
        "wind_model": wind_model,
        "wind_mode": wind_mode,
        "wind_param_label": wind_param_label,
        "latency_mode": latency_mode,
        "latency_s": latency_s,
        "latency_range_s": latency_range_s,
        "state_feedback_delay_s": state_feedback_delay_s,
        "actuator_t10_s": actuator_t10_s,
        "actuator_t50_nominal_s": actuator_t50_nominal_s,
        "actuator_t90_s": actuator_t90_s,
        "conservative_actuator_bound_s": conservative_actuator_bound_s,
        "vicon_filter_cutoff_hz": vicon_filter_cutoff_hz,
        "vicon_filter_model": vicon_filter_model,
        "duration_s": float(duration_s),
        "termination_reason": reason,
        "success": bool(success),
        "failure_class": failure_class(reason, governor_rejection_reason, wind_model),
        **feasibility,
        "heading_change_deg": heading_change_deg,
        "actual_heading_change_deg": heading_change_deg,
        "forward_travel_m": float(
            final[STATE_INDEX["x_w"]] - state_arr[0, STATE_INDEX["x_w"]]
        ),
        "turn_volume_proxy_m2": float(x_span * y_span),
        "height_change_m": float(final[STATE_INDEX["z_w"]] - state_arr[0, STATE_INDEX["z_w"]]),
        "terminal_speed_m_s": terminal_speed,
        "max_alpha_deg": float(np.rad2deg(np.max(np.abs(alpha)))),
        "max_beta_deg": float(np.rad2deg(np.max(np.abs(beta)))),
        "max_bank_deg": float(
            np.rad2deg(np.max(np.abs(state_arr[:, STATE_INDEX["phi"]])))
        ),
        "min_wall_distance_m": float(min_wall),
        "min_floor_margin_m": float(min_floor),
        "min_ceiling_margin_m": float(min_ceiling),
        "saturation_time_s": None if saturation_time_s is None else float(saturation_time_s),
        "time_at_full_travel_s": None if saturation_time_s is None else float(saturation_time_s),
        "saturation_fraction": float(saturation_fraction),
        "tracking_error_rms": None if tracking_error_rms is None else float(tracking_error_rms),
        "exit_recoverable": terminal_recoverable,
        "governor_rejection_reason": governor_rejection_reason,
        "candidate_count": int(candidate_count),
        "rejected_count": int(rejected_count),
        "log_path": rel_log_path,
        # Legacy CSV readers still consume the original field names
        "primitive_selected": primitive_selected,
        "max_abs_phi_deg": float(
            np.rad2deg(np.max(np.abs(state_arr[:, STATE_INDEX["phi"]])))
        ),
        "inside_safe_volume": bool(inside),
        "log_path_relative": rel_log_path,
    }
    return _metric_with_required_schema(row)


# =============================================================================
# 4) Governor-Rejected Metric Rows
# =============================================================================
# Rejected rows preserve failure provenance when no rollout log exists.
def rejected_metrics(
    scenario_id: str,
    seed: int,
    wind_model: str,
    wind_mode: str,
    wind_param_label: str,
    latency_mode: str,
    latency_s: float | None,
    latency_range_s: str | None,
    primitive_name: str,
    x0: np.ndarray,
    log_path: Path,
    repo_root: Path,
    arena_config: ArenaConfig,
    governor_rejection_reason: str,
    state_feedback_delay_s: float | None = None,
    actuator_t10_s: float | None = None,
    actuator_t50_nominal_s: float | None = None,
    actuator_t90_s: float | None = None,
    conservative_actuator_bound_s: float | None = None,
    vicon_filter_cutoff_hz: float | None = None,
    vicon_filter_model: str = "",
    candidate_count: int = 1,
    rejected_count: int = 1,
    primitive_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    return rollout_metrics(
        scenario_id=scenario_id,
        seed=seed,
        wind_model=wind_model,
        wind_mode=wind_mode,
        wind_param_label=wind_param_label,
        latency_mode=latency_mode,
        latency_s=latency_s,
        latency_range_s=latency_range_s,
        state_feedback_delay_s=state_feedback_delay_s,
        actuator_t10_s=actuator_t10_s,
        actuator_t50_nominal_s=actuator_t50_nominal_s,
        actuator_t90_s=actuator_t90_s,
        conservative_actuator_bound_s=conservative_actuator_bound_s,
        vicon_filter_cutoff_hz=vicon_filter_cutoff_hz,
        vicon_filter_model=vicon_filter_model,
        primitive_selected=primitive_name,
        selected_primitive=primitive_name,
        success=False,
        termination_reason="governor_rejected",
        states=np.asarray(x0, dtype=float).reshape(1, 15),
        log_path=log_path,
        repo_root=repo_root,
        arena_config=arena_config,
        saturation_fraction=0.0,
        saturation_time_s=0.0,
        governor_rejection_reason=governor_rejection_reason,
        candidate_count=candidate_count,
        rejected_count=rejected_count,
        primitive_metadata=primitive_metadata,
    )


def _agile_feasibility_fields(
    metadata: dict[str, object],
    success: bool,
    termination_reason: str,
    heading_change_deg: float,
    min_wall_distance_m: float,
    min_floor_margin_m: float,
    min_ceiling_margin_m: float,
    exit_recoverable: bool,
    max_alpha_deg: float,
    saturation_fraction: float,
) -> dict[str, object]:
    target = _optional_metadata_float(metadata.get("target_heading_deg"))
    primitive_family = str(metadata.get("primitive_family", ""))
    candidate_id = str(metadata.get("candidate_id", ""))
    is_full_turn_claim = bool(metadata.get("is_full_turn_claim", False))
    if target is None:
        return {
            "primitive_family": primitive_family,
            "candidate_id": candidate_id,
            "is_full_turn_claim": is_full_turn_claim,
            "target_heading_deg": None,
            "heading_target_deg": None,
            "heading_error_deg": None,
            "fixed_start_success": False,
            "feasibility_label": "not_tested",
        }

    actual_abs = abs(float(heading_change_deg))
    heading_error = abs(float(target) - actual_abs)
    target_reached = actual_abs >= 0.8 * float(target)
    safe_exit = bool(success and exit_recoverable and min_wall_distance_m > 0.0)
    reason = str(termination_reason).lower()
    if "latency" in reason or "timing" in reason:
        label = "latency_limited"
    elif float(saturation_fraction) >= 0.95 and not target_reached:
        label = "actuator_limited"
    elif abs(float(max_alpha_deg)) > 25.0 or "angle of attack" in reason:
        label = "model_limited_high_alpha"
    elif min_wall_distance_m <= 0.0 or "wall" in reason:
        label = "fixed_start_unsafe_wall"
    elif (
        min_floor_margin_m <= 0.0
        or min_ceiling_margin_m <= 0.0
        or "floor" in reason
        or "ceiling" in reason
    ):
        label = "fixed_start_unsafe_floor_or_ceiling"
    elif not success:
        label = "physically_infeasible_candidate"
    elif not exit_recoverable:
        label = "fixed_start_unrecoverable"
    elif not target_reached:
        label = "fixed_start_safe_but_under_turning"
    else:
        label = "fixed_start_feasible"
    return {
        "primitive_family": primitive_family,
        "candidate_id": candidate_id,
        "is_full_turn_claim": is_full_turn_claim,
        "target_heading_deg": float(target),
        "heading_target_deg": float(target),
        "heading_error_deg": float(heading_error),
        "fixed_start_success": safe_exit,
        "feasibility_label": label,
    }


def _optional_metadata_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    value_float = float(value)
    if not np.isfinite(value_float):
        return None
    return value_float


def _scenario_target_metadata(
    scenario_id: str,
    primitive_name: str,
) -> dict[str, object]:
    del scenario_id, primitive_name
    return {}
