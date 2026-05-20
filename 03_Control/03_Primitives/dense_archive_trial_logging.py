from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from arena_contract import TRUE_SAFE_BOUNDS, position_margin_m
from primitive_library_schema import (
    classify_wind_query_region,
    entry_clearance_metrics,
    path_metrics,
)
from wing_wind_descriptors import WING_WIND_DESCRIPTOR_COLUMNS


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Dense trial descriptor schema
# 2) Public key and row builders
# 3) Metric helpers
# 4) Conversion and validation helpers
# =============================================================================


# =============================================================================
# 1) Dense Trial Descriptor Schema
# =============================================================================
DENSE_TRIAL_DESCRIPTOR_STATUS_VALUES = (
    "synthetic_descriptor_only",
    "replay_evaluated",
    "not_replayed",
    "entry_invalid",
    "nonfinite_state",
)

DENSE_TRIAL_DESCRIPTOR_COLUMNS = (
    "trial_descriptor_id",
    "layout_branch_id",
    "fan_layout",
    "fan_config_id",
    "test_environment_mode",
    "paired_environment_mode",
    "environment_role",
    "validity_gate_role",
    "acceptance_interpretation",
    "candidate_id",
    "sample_id",
    "paired_sample_key",
    "seed",
    "replay_seed",
    "sampling_round",
    "updraft_model_id",
    "family",
    "target_heading_deg",
    "direction_sign",
    "start_class",
    "x0_w_m",
    "y0_w_m",
    "z0_w_m",
    "speed0_m_s",
    "phi0_rad",
    "theta0_rad",
    "psi0_rad",
    "u0_m_s",
    "v0_m_s",
    "w0_m_s",
    "p0_rad_s",
    "q0_rad_s",
    "r0_rad_s",
    "updraft_center_x_m",
    "updraft_center_y_m",
    "updraft_relative_radius_m",
    "updraft_relative_azimuth_rad",
    "updraft_relative_height_m",
    "updraft_sector_label",
    *WING_WIND_DESCRIPTOR_COLUMNS,
    "duration_s",
    "heading_initial_deg",
    "heading_terminal_deg",
    "heading_change_deg",
    "heading_error_deg",
    "path_length_xy_m",
    "path_length_3d_m",
    "forward_displacement_m",
    "lateral_displacement_m",
    "xy_bounding_box_area_m2",
    "turn_footprint_proxy_m2",
    "entry_clearance_required_x_plus_m",
    "entry_clearance_required_x_minus_m",
    "entry_clearance_required_y_plus_m",
    "entry_clearance_required_y_minus_m",
    "floor_margin_required_m",
    "ceiling_margin_required_m",
    "min_true_margin_m",
    "floor_margin_min_m",
    "ceiling_margin_min_m",
    "speed_min_m_s",
    "terminal_speed_m_s",
    "specific_energy_initial_m",
    "specific_energy_terminal_m",
    "energy_residual_m",
    "lift_dwell_fraction",
    "lift_dwell_fraction_status",
    "wind_query_region",
    "saturation_fraction",
    "latency_case",
    "latency_acceptance_scope",
    "latency_pass_label",
    "state_feedback_delay_s",
    "command_onset_delay_s",
    "command_transport_delay_s",
    "actuator_tau_s",
    "actuator_t50_s",
    "actuator_t90_s",
    "latency_jitter_s",
    "timing_model_version",
    "state_feedback_delay_applied",
    "success_flag",
    "failure_label",
    "governor_rejection_cause",
    "robustness_label",
    "physics_priority_level",
    "sim_real_match_key",
    "sim_real_match_key_version",
    "sim_real_transfer_result",
    "descriptor_status",
)


@dataclass(frozen=True)
class DenseTrialDescriptorConfig:
    dt_s: float = 0.02
    z_outlet_m: float = 0.330
    lift_dwell_threshold_m_s: float = 0.05
    saturation_tolerance: float = 1e-12
    sim_real_match_key_version: str = "dense_trial_match_key_v1"
    trial_descriptor_id_version: str = "trial_descriptor_id_v1"


# =============================================================================
# 2) Public Key and Row Builders
# =============================================================================
def dense_trial_match_key(
    *,
    layout_branch_id: str,
    fan_layout: str,
    fan_config_id: str,
    test_environment_mode: str,
    paired_environment_mode: str,
    candidate_id: str,
    sample_id: str,
    paired_sample_key: str,
    seed: int | str,
    replay_seed: int | str,
    latency_case: str,
    config: DenseTrialDescriptorConfig | None = None,
) -> str:
    """Return a deterministic branch-local sim-real match key."""

    active_config = DenseTrialDescriptorConfig() if config is None else config
    return _descriptor_key(
        version=active_config.sim_real_match_key_version,
        layout_branch_id=layout_branch_id,
        fan_layout=fan_layout,
        fan_config_id=fan_config_id,
        test_environment_mode=test_environment_mode,
        paired_environment_mode=paired_environment_mode,
        candidate_id=candidate_id,
        sample_id=sample_id,
        paired_sample_key=paired_sample_key,
        seed=seed,
        replay_seed=replay_seed,
        latency_case=latency_case,
    )


def dense_trial_descriptor_row(
    *,
    start_row: Mapping[str, object],
    candidate_row: Mapping[str, object],
    time_s: np.ndarray,
    x_ref: np.ndarray,
    u_norm_requested: np.ndarray,
    u_norm_effective_target: np.ndarray,
    u_norm_applied: np.ndarray,
    delta_cmd_rad: np.ndarray,
    latency_fields: Mapping[str, object],
    failure_label: str,
    governor_rejection_cause: str,
    robustness_label: str,
    sim_real_match_key: str,
    descriptor_status: str,
    replay_seed: int | str,
    lift_exposure_m_s: np.ndarray | None = None,
    sim_real_transfer_result: str = "not_evaluated",
    wind_model_z_axis_m: np.ndarray | None = None,
    config: DenseTrialDescriptorConfig | None = None,
) -> dict[str, object]:
    """Build one dense-trial descriptor row from supplied replay-like arrays."""

    active_config = DenseTrialDescriptorConfig() if config is None else config
    _validate_descriptor_status(descriptor_status)
    time, state, _, effective_target, applied, _ = _validated_arrays(
        time_s,
        x_ref,
        u_norm_requested,
        u_norm_effective_target,
        u_norm_applied,
        delta_cmd_rad,
        descriptor_status,
    )
    exposure = _validated_lift_exposure(lift_exposure_m_s, state.shape[0])

    layout_branch_id = _text(_first(candidate_row, start_row, "layout_branch_id"))
    fan_layout = _text(_first(candidate_row, start_row, "fan_layout"))
    fan_config_id = _text(_first(candidate_row, start_row, "fan_config_id"))
    test_environment_mode = _text(_first(candidate_row, start_row, "test_environment_mode"))
    paired_environment_mode = _text(_first(candidate_row, start_row, "paired_environment_mode"))
    candidate_id = _text(_first(candidate_row, start_row, "candidate_id"))
    sample_id = _text(_first(candidate_row, start_row, "sample_id"))
    paired_sample_key = _text(_first(candidate_row, start_row, "paired_sample_key"))
    seed = _first(candidate_row, start_row, "seed")
    family = _text(_first(candidate_row, start_row, "family"))
    direction_sign = _direction_sign(_first(candidate_row, start_row, "direction_sign", default=1))
    target_heading_deg = _finite_float_or_nan(
        _first(candidate_row, start_row, "target_heading_deg", default=np.nan)
    )
    latency_case = _text(latency_fields["latency_case"])

    trial_descriptor_id = _descriptor_key(
        version=active_config.trial_descriptor_id_version,
        layout_branch_id=layout_branch_id,
        fan_layout=fan_layout,
        fan_config_id=fan_config_id,
        test_environment_mode=test_environment_mode,
        paired_environment_mode=paired_environment_mode,
        candidate_id=candidate_id,
        sample_id=sample_id,
        paired_sample_key=paired_sample_key,
        seed=seed,
        replay_seed=replay_seed,
        latency_case=latency_case,
    )

    positions = state[:, 0:3]
    heading = _heading_metrics(state, direction_sign, target_heading_deg)
    path = path_metrics(positions)
    clearance = entry_clearance_metrics(positions, TRUE_SAFE_BOUNDS)
    margins = _margin_metrics(positions)
    speed = _speed_metrics(state)
    energy = _specific_energy_metrics(state)
    lift_dwell, lift_status = _lift_dwell_fraction(
        time,
        exposure,
        _first(candidate_row, start_row, "w_wing_mean_m_s", default=np.nan),
        active_config,
    )

    row: dict[str, object] = {
        "trial_descriptor_id": trial_descriptor_id,
        "layout_branch_id": layout_branch_id,
        "fan_layout": fan_layout,
        "fan_config_id": fan_config_id,
        "test_environment_mode": test_environment_mode,
        "paired_environment_mode": paired_environment_mode,
        "environment_role": _text(_first(candidate_row, start_row, "environment_role")),
        "validity_gate_role": _text(_first(candidate_row, start_row, "validity_gate_role")),
        "acceptance_interpretation": _text(
            _first(candidate_row, start_row, "acceptance_interpretation")
        ),
        "candidate_id": candidate_id,
        "sample_id": sample_id,
        "paired_sample_key": paired_sample_key,
        "seed": _csv_scalar(seed),
        "replay_seed": _csv_scalar(replay_seed),
        "sampling_round": _csv_scalar(_first(candidate_row, start_row, "sampling_round")),
        "updraft_model_id": _text(_first(candidate_row, start_row, "updraft_model_id")),
        "family": family,
        "target_heading_deg": target_heading_deg,
        "direction_sign": direction_sign,
        "start_class": _text(_first(candidate_row, start_row, "start_class")),
        "x0_w_m": _state_value(start_row, "x_w_m", "x0_w_m"),
        "y0_w_m": _state_value(start_row, "y_w_m", "y0_w_m"),
        "z0_w_m": _state_value(start_row, "z_w_m", "z0_w_m"),
        "speed0_m_s": _state_value(start_row, "speed_m_s", "speed0_m_s"),
        "phi0_rad": _state_value(start_row, "phi_rad", "phi0_rad"),
        "theta0_rad": _state_value(start_row, "theta_rad", "theta0_rad"),
        "psi0_rad": _state_value(start_row, "psi_rad", "psi0_rad"),
        "u0_m_s": _state_value(start_row, "u_m_s", "u0_m_s"),
        "v0_m_s": _state_value(start_row, "v_m_s", "v0_m_s"),
        "w0_m_s": _state_value(start_row, "w_m_s", "w0_m_s"),
        "p0_rad_s": _state_value(start_row, "p_rad_s", "p0_rad_s"),
        "q0_rad_s": _state_value(start_row, "q_rad_s", "q0_rad_s"),
        "r0_rad_s": _state_value(start_row, "r_rad_s", "r0_rad_s"),
        "updraft_center_x_m": _finite_float_or_nan(
            _first(start_row, candidate_row, "updraft_center_x_m", default=np.nan)
        ),
        "updraft_center_y_m": _finite_float_or_nan(
            _first(start_row, candidate_row, "updraft_center_y_m", default=np.nan)
        ),
        "updraft_relative_radius_m": _finite_float_or_nan(
            _first(start_row, candidate_row, "updraft_relative_radius_m", default=np.nan)
        ),
        "updraft_relative_azimuth_rad": _finite_float_or_nan(
            _first(start_row, candidate_row, "updraft_relative_azimuth_rad", default=np.nan)
        ),
        "updraft_relative_height_m": _finite_float_or_nan(
            _first(start_row, candidate_row, "updraft_relative_height_m", default=np.nan)
        ),
        "updraft_sector_label": _text(
            _first(start_row, candidate_row, "updraft_sector_label")
        ),
        "duration_s": float(time[-1] - time[0]),
        **heading,
        **path,
        **clearance,
        **margins,
        **speed,
        **energy,
        "lift_dwell_fraction": lift_dwell,
        "lift_dwell_fraction_status": lift_status,
        "wind_query_region": classify_wind_query_region(
            positions[:, 2],
            wind_model_z_axis_m,
            active_config.z_outlet_m,
        ),
        "saturation_fraction": _saturation_fraction(
            effective_target,
            applied,
            active_config.saturation_tolerance,
        ),
        "latency_case": latency_case,
        "latency_acceptance_scope": _text(latency_fields["latency_acceptance_scope"]),
        "latency_pass_label": _text(latency_fields["latency_pass_label"]),
        "state_feedback_delay_s": float(latency_fields["state_feedback_delay_s"]),
        "command_onset_delay_s": float(latency_fields["command_onset_delay_s"]),
        "command_transport_delay_s": float(latency_fields["command_transport_delay_s"]),
        "actuator_tau_s": _text(latency_fields["actuator_tau_s"]),
        "actuator_t50_s": float(latency_fields["actuator_t50_s"]),
        "actuator_t90_s": float(latency_fields["actuator_t90_s"]),
        "latency_jitter_s": float(latency_fields["latency_jitter_s"]),
        "timing_model_version": _text(latency_fields["timing_model_version"]),
        "state_feedback_delay_applied": bool(latency_fields["state_feedback_delay_applied"]),
        "success_flag": _success_flag(
            failure_label,
            governor_rejection_cause,
            robustness_label,
        ),
        "failure_label": _text(failure_label),
        "governor_rejection_cause": _text(governor_rejection_cause),
        "robustness_label": _text(robustness_label),
        "physics_priority_level": _physics_priority_level(family, target_heading_deg),
        "sim_real_match_key": _text(sim_real_match_key),
        "sim_real_match_key_version": active_config.sim_real_match_key_version,
        "sim_real_transfer_result": _text(sim_real_transfer_result),
        "descriptor_status": _text(descriptor_status),
    }
    for column in WING_WIND_DESCRIPTOR_COLUMNS:
        row[column] = _wing_descriptor_value(column, candidate_row, start_row)
    return _ordered_csv_row(row)


# =============================================================================
# 3) Metric Helpers
# =============================================================================
def _heading_metrics(
    state: np.ndarray,
    direction_sign: int,
    target_heading_deg: float,
) -> dict[str, float]:
    yaw_deg = np.rad2deg(np.unwrap(state[:, 5]))
    initial = float(yaw_deg[0])
    terminal = float(yaw_deg[-1])
    change = float(direction_sign * (terminal - initial))
    if np.isfinite(target_heading_deg):
        error = float(abs(change - target_heading_deg))
    else:
        error = float("nan")
    return {
        "heading_initial_deg": initial,
        "heading_terminal_deg": terminal,
        "heading_change_deg": change,
        "heading_error_deg": error,
    }


def _margin_metrics(positions: np.ndarray) -> dict[str, float]:
    rows = [
        position_margin_m(position, TRUE_SAFE_BOUNDS)
        for position in positions
        if np.all(np.isfinite(position))
    ]
    if not rows:
        return {
            "min_true_margin_m": float("nan"),
            "floor_margin_min_m": float("nan"),
            "ceiling_margin_min_m": float("nan"),
        }
    return {
        "min_true_margin_m": float(min(row["min_margin_m"] for row in rows)),
        "floor_margin_min_m": float(min(row["floor_margin_m"] for row in rows)),
        "ceiling_margin_min_m": float(min(row["ceiling_margin_m"] for row in rows)),
    }


def _speed_metrics(state: np.ndarray) -> dict[str, float]:
    speed = np.linalg.norm(state[:, 6:9], axis=1)
    finite_speed = speed[np.isfinite(speed)]
    return {
        "speed_min_m_s": float(np.min(finite_speed)) if finite_speed.size else float("nan"),
        "terminal_speed_m_s": float(speed[-1]) if np.isfinite(speed[-1]) else float("nan"),
    }


def _specific_energy_metrics(state: np.ndarray) -> dict[str, float]:
    speed = np.linalg.norm(state[:, 6:9], axis=1)
    initial = _specific_energy(state[0, 2], speed[0])
    terminal = _specific_energy(state[-1, 2], speed[-1])
    residual = terminal - initial if np.isfinite(initial) and np.isfinite(terminal) else float("nan")
    return {
        "specific_energy_initial_m": initial,
        "specific_energy_terminal_m": terminal,
        "energy_residual_m": float(residual),
    }


def _specific_energy(z_w_m: float, speed_m_s: float) -> float:
    if not np.isfinite(z_w_m) or not np.isfinite(speed_m_s):
        return float("nan")
    return float(z_w_m + speed_m_s**2 / (2.0 * 9.81))


def _lift_dwell_fraction(
    time_s: np.ndarray,
    lift_exposure_m_s: np.ndarray | None,
    wing_mean_m_s: object,
    config: DenseTrialDescriptorConfig,
) -> tuple[float, str]:
    if lift_exposure_m_s is not None:
        intervals = np.diff(time_s)
        total = float(np.sum(intervals))
        if intervals.size == 0 or total <= 0.0:
            return float("nan"), "not_available"
        active = np.isfinite(lift_exposure_m_s[:-1]) & (
            lift_exposure_m_s[:-1] >= float(config.lift_dwell_threshold_m_s)
        )
        return float(np.sum(intervals[active]) / total), "trajectory_exposure"

    wing_mean = _finite_float_or_nan(wing_mean_m_s)
    if not np.isfinite(wing_mean):
        return float("nan"), "not_available"
    dwell = 1.0 if wing_mean >= float(config.lift_dwell_threshold_m_s) else 0.0
    return float(dwell), "start_state_proxy"


def _saturation_fraction(
    effective_target: np.ndarray,
    applied: np.ndarray,
    tolerance: float,
) -> float:
    clipped = np.any(np.abs(effective_target - applied) > float(tolerance), axis=1)
    return float(np.count_nonzero(clipped) / max(1, clipped.size))


def _success_flag(
    failure_label: str,
    governor_rejection_cause: str,
    robustness_label: str,
) -> bool:
    return bool(
        str(failure_label) == "success"
        and str(governor_rejection_cause) == "none"
        and str(robustness_label) in {"none", "not_evaluated"}
    )


def _physics_priority_level(family: str, target_heading_deg: float) -> str:
    if str(family) in {"glide", "recovery", "baseline"} or not np.isfinite(target_heading_deg):
        return "not_priority_ranked"
    if np.isclose(target_heading_deg, 30.0, atol=1e-9):
        return "target_steering_30deg_priority"
    if np.isclose(target_heading_deg, 15.0, atol=1e-9):
        return "target_steering_15deg_priority"
    return "target_steering_future_target_priority"


# =============================================================================
# 4) Conversion and Validation Helpers
# =============================================================================
def _descriptor_key(
    *,
    version: str,
    layout_branch_id: str,
    fan_layout: str,
    fan_config_id: str,
    test_environment_mode: str,
    paired_environment_mode: str,
    candidate_id: str,
    sample_id: str,
    paired_sample_key: str,
    seed: int | str,
    replay_seed: int | str,
    latency_case: str,
) -> str:
    return "|".join(
        (
            _key_part(version),
            f"branch={_key_part(layout_branch_id)}",
            f"fan_layout={_key_part(fan_layout)}",
            f"fan_config={_key_part(fan_config_id)}",
            f"test_env={_key_part(test_environment_mode)}",
            f"paired_env={_key_part(paired_environment_mode)}",
            f"candidate={_key_part(candidate_id)}",
            f"sample={_key_part(sample_id)}",
            f"paired_sample={_key_part(paired_sample_key)}",
            f"seed={_key_part(seed)}",
            f"replay_seed={_key_part(replay_seed)}",
            f"latency={_key_part(latency_case)}",
        )
    )


def _key_part(value: object) -> str:
    return str(value).replace("|", "_").replace("\n", "_").replace("\r", "_")


def _validate_descriptor_status(descriptor_status: str) -> None:
    if descriptor_status not in DENSE_TRIAL_DESCRIPTOR_STATUS_VALUES:
        raise ValueError(f"unknown dense trial descriptor status: {descriptor_status!r}")


def _validated_arrays(
    time_s: np.ndarray,
    x_ref: np.ndarray,
    u_norm_requested: np.ndarray,
    u_norm_effective_target: np.ndarray,
    u_norm_applied: np.ndarray,
    delta_cmd_rad: np.ndarray,
    descriptor_status: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time = np.asarray(time_s, dtype=float)
    state = np.asarray(x_ref, dtype=float)
    requested = np.asarray(u_norm_requested, dtype=float)
    effective = np.asarray(u_norm_effective_target, dtype=float)
    applied = np.asarray(u_norm_applied, dtype=float)
    delta = np.asarray(delta_cmd_rad, dtype=float)
    if time.ndim != 1:
        raise ValueError("time_s must have shape (N,).")
    if time.size == 0:
        raise ValueError("time_s must contain at least one sample.")
    if not np.all(np.isfinite(time)):
        raise ValueError("time_s must contain only finite values.")
    if np.any(np.diff(time) < 0.0):
        raise ValueError("time_s must be monotonic nondecreasing.")
    if state.ndim != 2 or state.shape[1] != 15:
        raise ValueError("x_ref must have shape (N, 15).")
    for name, command in (
        ("u_norm_requested", requested),
        ("u_norm_effective_target", effective),
        ("u_norm_applied", applied),
        ("delta_cmd_rad", delta),
    ):
        if command.ndim != 2 or command.shape[1] != 3:
            raise ValueError(f"{name} must have shape (N, 3).")
    sample_count = time.size
    arrays = (state, requested, effective, applied, delta)
    if any(array.shape[0] != sample_count for array in arrays):
        raise ValueError("time_s, x_ref, and command arrays must share the same N.")
    for name, command in (
        ("u_norm_requested", requested),
        ("u_norm_effective_target", effective),
        ("u_norm_applied", applied),
        ("delta_cmd_rad", delta),
    ):
        if not np.all(np.isfinite(command)):
            raise ValueError(f"{name} must contain only finite values.")
    if not np.all(np.isfinite(state)) and descriptor_status != "nonfinite_state":
        raise ValueError("nonfinite x_ref requires descriptor_status='nonfinite_state'.")
    if np.all(np.isfinite(state)) and descriptor_status == "nonfinite_state":
        raise ValueError("descriptor_status='nonfinite_state' requires nonfinite x_ref.")
    return time, state, requested, effective, applied, delta


def _validated_lift_exposure(
    lift_exposure_m_s: np.ndarray | None,
    sample_count: int,
) -> np.ndarray | None:
    if lift_exposure_m_s is None:
        return None
    exposure = np.asarray(lift_exposure_m_s, dtype=float)
    if exposure.ndim != 1 or exposure.shape[0] != sample_count:
        raise ValueError("lift_exposure_m_s must have shape (N,) when supplied.")
    return exposure


def _first(
    primary: Mapping[str, object],
    secondary: Mapping[str, object],
    key: str,
    *,
    default: object = "",
) -> object:
    if key in primary:
        return primary[key]
    if key in secondary:
        return secondary[key]
    return default


def _state_value(row: Mapping[str, object], primary_key: str, fallback_key: str) -> float:
    return _finite_float_or_nan(row[primary_key] if primary_key in row else row.get(fallback_key, np.nan))


def _wing_descriptor_value(
    column: str,
    candidate_row: Mapping[str, object],
    start_row: Mapping[str, object],
) -> object:
    value = _first(candidate_row, start_row, column, default=np.nan)
    if column in {
        "wind_descriptor_status",
        "wind_descriptor_environment_mode",
        "wind_descriptor_model_id",
        "wind_descriptor_model_source",
        "local_updraft_uncertainty_status",
    }:
        return _text(value)
    return _finite_float_or_nan(value)


def _finite_float_or_nan(value: object) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, str) and value.strip() == "":
        return float("nan")
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return float(result) if np.isfinite(result) else float("nan")


def _direction_sign(value: object) -> int:
    numeric = _finite_float_or_nan(value)
    if not np.isfinite(numeric) or numeric == 0.0:
        return 1
    return -1 if numeric < 0.0 else 1


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
        return ""
    return str(value)


def _csv_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _ordered_csv_row(row: dict[str, object]) -> dict[str, object]:
    missing = [column for column in DENSE_TRIAL_DESCRIPTOR_COLUMNS if column not in row]
    if missing:
        raise KeyError(f"dense trial descriptor row missing columns: {missing}")
    return {
        column: _csv_scalar(row[column])
        for column in DENSE_TRIAL_DESCRIPTOR_COLUMNS
    }
