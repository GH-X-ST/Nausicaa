"""Prepare real-glider calibration summaries from 04_Flight_Test logs.

This script is intentionally offline. It reads completed flight-test calibration
sessions, extracts neutral-glide and invalid-start metrics, and writes compact
calibration targets for grey-box model fitting. It does not edit the controller,
primitive library, or real-flight runtime, and it reports which dry-air model
calibration constants were active during measured-launch replay.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
FLIGHT_RUNTIME_ROOT = REPO_ROOT / "04_Flight_Test" / "01_Runtime"
if str(FLIGHT_RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(FLIGHT_RUNTIME_ROOT))
PRIMITIVE_ROOT = REPO_ROOT / "03_Control" / "03_Primitives"
SCENARIO_ROOT = REPO_ROOT / "03_Control" / "04_Scenarios"
for path in (PRIMITIVE_ROOT, SCENARIO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from launch_gate import LAUNCH_GATE_X_W_M, LAUNCH_GATE_Y_W_M, LAUNCH_GATE_Z_W_M  # noqa: E402
from arena_contract import TRUE_SAFE_BOUNDS, position_margin_m  # noqa: E402
from command_contract import normalised_command_to_surface_rad  # noqa: E402
from latency import latency_case_config  # noqa: E402
from flight_dynamics import adapt_glider, state_derivative  # noqa: E402
from glider import build_nausicaa_glider  # noqa: E402
from A_model_parameters.neutral_dry_air_calibration import (  # noqa: E402
    CALIBRATION_ACTIVE as NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE,
    CALIBRATION_ID as NEUTRAL_DRY_AIR_CALIBRATION_ID,
    CD0_STRIP_SCALE as NEUTRAL_DRY_AIR_CD0_STRIP_SCALE,
    DRAG_AREA_FUSE_SCALE as NEUTRAL_DRY_AIR_DRAG_AREA_FUSE_SCALE,
    DELTA_A_TRIM_RAD as NEUTRAL_DRY_AIR_DELTA_A_TRIM_RAD,
    DELTA_E_TRIM_RAD as NEUTRAL_DRY_AIR_DELTA_E_TRIM_RAD,
    DELTA_R_TRIM_RAD as NEUTRAL_DRY_AIR_DELTA_R_TRIM_RAD,
    EFFICIENCY_STRIP_SCALE as NEUTRAL_DRY_AIR_EFFICIENCY_STRIP_SCALE,
    HELDOUT_POLICY as NEUTRAL_DRY_AIR_HELDOUT_POLICY,
    HELDOUT_SEED as NEUTRAL_DRY_AIR_HELDOUT_SEED,
    ROLL_MOMENT_BIAS_COEFF as NEUTRAL_DRY_AIR_ROLL_MOMENT_BIAS_COEFF,
    SOURCE_PREP_RUN as NEUTRAL_DRY_AIR_SOURCE_PREP_RUN,
    SOURCE_THROW_COUNT as NEUTRAL_DRY_AIR_SOURCE_THROW_COUNT,
    YAW_MOMENT_BIAS_COEFF as NEUTRAL_DRY_AIR_YAW_MOMENT_BIAS_COEFF,
)

DEFAULT_SESSION_SEARCH_ROOT = REPO_ROOT / "04_Flight_Test" / "05_Results"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "03_Control" / "05_Results" / "glider_model_calibration_prep"
DEFAULT_HELDOUT_COUNT = 5
DEFAULT_HELDOUT_SEED = 603
DEFAULT_REPLAY_DT_S = 0.005
DEFAULT_ALIGNMENT_WINDOW_S = 0.10
DEFAULT_REPLAY_COMMAND_ONSET_DELAY_S = float(latency_case_config("nominal").command_onset_delay_s)
DEFAULT_WORKERS = 8
RIDGE_ALPHA = 1.0e-6
LAUNCH_NOMINAL_X_M = 0.5 * (LAUNCH_GATE_X_W_M[0] + LAUNCH_GATE_X_W_M[1])
LAUNCH_NOMINAL_Y_M = 0.5 * (LAUNCH_GATE_Y_W_M[0] + LAUNCH_GATE_Y_W_M[1])
LAUNCH_NOMINAL_Z_M = 0.5 * (LAUNCH_GATE_Z_W_M[0] + LAUNCH_GATE_Z_W_M[1])
_REPLAY_WORKER_AIRCRAFT: Any | None = None

CURRENT_MODEL_CALIBRATION = {
    "neutral_dry_air_calibration_active": bool(NEUTRAL_DRY_AIR_CALIBRATION_ACTIVE),
    "neutral_dry_air_calibration_id": str(NEUTRAL_DRY_AIR_CALIBRATION_ID),
    "source_prep_run": str(NEUTRAL_DRY_AIR_SOURCE_PREP_RUN),
    "source_throw_count": int(NEUTRAL_DRY_AIR_SOURCE_THROW_COUNT),
    "heldout_policy": str(NEUTRAL_DRY_AIR_HELDOUT_POLICY),
    "heldout_seed": int(NEUTRAL_DRY_AIR_HELDOUT_SEED),
    "cd0_strip_scale": float(NEUTRAL_DRY_AIR_CD0_STRIP_SCALE),
    "drag_area_fuse_scale": float(NEUTRAL_DRY_AIR_DRAG_AREA_FUSE_SCALE),
    "efficiency_strip_scale": float(NEUTRAL_DRY_AIR_EFFICIENCY_STRIP_SCALE),
    "roll_moment_bias_coeff": float(NEUTRAL_DRY_AIR_ROLL_MOMENT_BIAS_COEFF),
    "yaw_moment_bias_coeff": float(NEUTRAL_DRY_AIR_YAW_MOMENT_BIAS_COEFF),
    "delta_a_trim_rad": float(NEUTRAL_DRY_AIR_DELTA_A_TRIM_RAD),
    "delta_e_trim_rad": float(NEUTRAL_DRY_AIR_DELTA_E_TRIM_RAD),
    "delta_r_trim_rad": float(NEUTRAL_DRY_AIR_DELTA_R_TRIM_RAD),
}

VALID_THROW_FIELDS = [
    "session_label",
    "source_session_root",
    "calibration_profile_id",
    "calibration_profile_hash",
    "vicon_calibration_source",
    "case_id",
    "case_name",
    "throw_id",
    "command_axis",
    "command_value",
    "pulse_start_s",
    "pulse_duration_s",
    "termination_reason",
    "termination_group",
    "launch_speed_m_s",
    "duration_s",
    "x0_m",
    "y0_m",
    "z0_m",
    "launch_x_offset_m",
    "launch_y_offset_m",
    "launch_z_offset_m",
    "u0_m_s",
    "v0_m_s",
    "w0_m_s",
    "p0_rad_s",
    "q0_rad_s",
    "r0_rad_s",
    "phi0_deg",
    "theta0_deg",
    "psi0_deg",
    "speed0_m_s",
    "horizontal_speed0_m_s",
    "flight_path_angle_proxy0_deg",
    "sideslip_proxy0_deg",
    "vicon_offset_x_m",
    "vicon_offset_y_m",
    "vicon_offset_z_m",
    "x_end_m",
    "y_end_m",
    "z_end_m",
    "dx_m",
    "dy_m",
    "dz_m",
    "altitude_loss_m",
    "horizontal_distance_m",
    "glide_ratio_x_over_altloss",
    "glide_ratio_horizontal_over_altloss",
    "sink_rate_m_s",
    "mean_speed_m_s",
    "mean_forward_speed_m_s",
    "max_abs_phi_deg",
    "max_abs_theta_deg",
    "max_abs_psi_deg",
    "max_abs_p_rad_s",
    "max_abs_q_rad_s",
    "max_abs_r_rad_s",
    "mean_rate_confidence",
    "min_rate_confidence",
    "spike_downweighted_fraction",
    "body_rate_limited_fraction",
    "sample_count",
]

INVALID_ATTEMPT_FIELDS = [
    "session_label",
    "case_id",
    "case_name",
    "attempt_id",
    "cancellation_reason",
    "launch_gate_reason",
    "trigger_source",
    "speed_m_s",
    "x_w_m",
    "y_w_m",
    "z_w_m",
    "phi_deg",
    "theta_deg",
    "psi_deg",
    "p_rad_s",
    "q_rad_s",
    "r_rad_s",
]

AGGREGATE_FIELDS = [
    "metric",
    "count",
    "mean",
    "median",
    "std",
    "min",
    "max",
    "role_for_model_calibration",
]

FEATURE_TARGET_FIELDS = [
    "session_label",
    "calibration_profile_hash",
    "termination_group",
    "throw_id",
    "x0_m",
    "y0_m",
    "z0_m",
    "launch_x_offset_m",
    "launch_y_offset_m",
    "launch_z_offset_m",
    "u0_m_s",
    "v0_m_s",
    "w0_m_s",
    "p0_rad_s",
    "q0_rad_s",
    "r0_rad_s",
    "phi0_deg",
    "theta0_deg",
    "psi0_deg",
    "speed0_m_s",
    "horizontal_speed0_m_s",
    "flight_path_angle_proxy0_deg",
    "sideslip_proxy0_deg",
    "termination_reason",
    "duration_s",
    "dx_m",
    "dy_m",
    "altitude_loss_m",
    "sink_rate_m_s",
    "glide_ratio_x_over_altloss",
]

EMPIRICAL_FEATURES = [
    "x0_m",
    "y0_m",
    "z0_m",
    "launch_x_offset_m",
    "launch_y_offset_m",
    "launch_z_offset_m",
    "u0_m_s",
    "v0_m_s",
    "w0_m_s",
    "p0_rad_s",
    "q0_rad_s",
    "r0_rad_s",
    "phi0_deg",
    "theta0_deg",
    "psi0_deg",
    "speed0_m_s",
    "horizontal_speed0_m_s",
    "flight_path_angle_proxy0_deg",
    "sideslip_proxy0_deg",
]

EMPIRICAL_TARGETS = ["dx_m", "dy_m", "sink_rate_m_s", "duration_s"]

EMPIRICAL_VALIDATION_FIELDS = [
    "split",
    "session_label",
    "calibration_profile_hash",
    "termination_group",
    "throw_id",
    "dx_m",
    "dx_pred_m",
    "dx_residual_m",
    "dy_m",
    "dy_pred_m",
    "dy_residual_m",
    "sink_rate_m_s",
    "sink_rate_pred_m_s",
    "sink_rate_residual_m_s",
    "duration_s",
    "duration_pred_s",
    "duration_residual_s",
    "altitude_loss_m",
    "altitude_loss_pred_m",
    "altitude_loss_residual_m",
    "glide_ratio_x_over_altloss",
    "derived_glide_ratio_pred",
    "derived_glide_ratio_residual",
]

EMPIRICAL_FIT_SUMMARY_FIELDS = [
    "target",
    "feature",
    "standardized_coefficient",
    "intercept",
    "train_count",
    "heldout_count",
    "train_mae",
    "heldout_mae",
]

SESSION_TERMINATION_SUMMARY_FIELDS = [
    "session_label",
    "calibration_profile_hash",
    "termination_group",
    "throw_count",
    "mean_launch_speed_m_s",
    "mean_sink_rate_m_s",
    "mean_dx_m",
    "mean_dy_m",
    "mean_duration_s",
    "mean_glide_ratio_x_over_altloss",
]

MEASURED_LAUNCH_REPLAY_FIELDS = [
    "split",
    "session_label",
    "calibration_profile_hash",
    "case_id",
    "case_name",
    "throw_id",
    "command_axis",
    "command_value",
    "pulse_start_s",
    "pulse_duration_s",
    "replay_status",
    "replay_policy",
    "replay_dt_s",
    "replay_command_source",
    "replay_command_onset_delay_s",
    "actual_termination_group",
    "sim_first_exit_reason",
    "sim_first_exit_time_s",
    "duration_s",
    "x0_m",
    "y0_m",
    "z0_m",
    "u0_m_s",
    "v0_m_s",
    "w0_m_s",
    "p0_rad_s",
    "q0_rad_s",
    "r0_rad_s",
    "phi0_deg",
    "theta0_deg",
    "psi0_deg",
    "actual_x_end_m",
    "actual_y_end_m",
    "actual_z_end_m",
    "sim_x_end_m",
    "sim_y_end_m",
    "sim_z_end_m",
    "actual_dx_m",
    "sim_dx_m",
    "dx_residual_actual_minus_sim_m",
    "actual_dy_m",
    "sim_dy_m",
    "dy_residual_actual_minus_sim_m",
    "actual_altitude_loss_m",
    "sim_altitude_loss_m",
    "altitude_loss_residual_actual_minus_sim_m",
    "actual_sink_rate_m_s",
    "sim_sink_rate_m_s",
    "sink_rate_residual_actual_minus_sim_m_s",
    "actual_glide_ratio_x_over_altloss",
    "sim_glide_ratio_x_over_altloss",
    "glide_ratio_residual_actual_minus_sim",
    "actual_final_phi_deg",
    "sim_final_phi_deg",
    "actual_final_theta_deg",
    "sim_final_theta_deg",
    "actual_final_psi_deg",
    "sim_final_psi_deg",
    "max_abs_sim_phi_deg",
    "max_abs_sim_theta_deg",
    "max_abs_sim_p_rad_s",
    "max_abs_sim_q_rad_s",
    "max_abs_sim_r_rad_s",
]

ALIGNED_MOTION_REPLAY_FIELDS = [
    "split",
    "session_label",
    "calibration_profile_hash",
    "case_id",
    "case_name",
    "throw_id",
    "command_axis",
    "command_value",
    "pulse_start_s",
    "pulse_duration_s",
    "replay_status",
    "replay_policy",
    "replay_dt_s",
    "alignment_window_s",
    "alignment_elapsed_s",
    "alignment_sample_count",
    "alignment_method",
    "replay_command_source",
    "replay_command_onset_delay_s",
    "actual_termination_group",
    "sim_first_exit_reason",
    "sim_first_exit_time_s",
    "duration_s",
    "x0_m",
    "y0_m",
    "z0_m",
    "u0_m_s",
    "v0_m_s",
    "w0_m_s",
    "p0_rad_s",
    "q0_rad_s",
    "r0_rad_s",
    "phi0_deg",
    "theta0_deg",
    "psi0_deg",
    "actual_x_end_m",
    "actual_y_end_m",
    "actual_z_end_m",
    "sim_x_end_m",
    "sim_y_end_m",
    "sim_z_end_m",
    "actual_dx_m",
    "sim_dx_m",
    "dx_residual_actual_minus_sim_m",
    "actual_dy_m",
    "sim_dy_m",
    "dy_residual_actual_minus_sim_m",
    "actual_altitude_loss_m",
    "sim_altitude_loss_m",
    "altitude_loss_residual_actual_minus_sim_m",
    "actual_sink_rate_m_s",
    "sim_sink_rate_m_s",
    "sink_rate_residual_actual_minus_sim_m_s",
    "actual_glide_ratio_x_over_altloss",
    "sim_glide_ratio_x_over_altloss",
    "glide_ratio_residual_actual_minus_sim",
    "actual_final_phi_deg",
    "sim_final_phi_deg",
    "actual_final_theta_deg",
    "sim_final_theta_deg",
    "actual_final_psi_deg",
    "sim_final_psi_deg",
    "max_abs_sim_phi_deg",
    "max_abs_sim_theta_deg",
    "max_abs_sim_p_rad_s",
    "max_abs_sim_q_rad_s",
    "max_abs_sim_r_rad_s",
]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_value(row.get(key, "")) for key in fieldnames})


def _format_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.10g}"
        return ""
    return value


def _float(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        value = row.get(key, default)
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_mean(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return mean(finite) if finite else float("nan")


def _safe_median(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return median(finite) if finite else float("nan")


def _safe_std(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return pstdev(finite) if len(finite) > 1 else 0.0 if finite else float("nan")


def _safe_min(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return min(finite) if finite else float("nan")


def _safe_max(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return max(finite) if finite else float("nan")


def _ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or abs(denominator) < 1e-9:
        return float("nan")
    return numerator / denominator


def _sequence(value: Any, length: int) -> list[float]:
    if not isinstance(value, (list, tuple)):
        return [float("nan")] * length
    output: list[float] = []
    for item in list(value)[:length]:
        output.append(_to_float(item))
    while len(output) < length:
        output.append(float("nan"))
    return output


def _termination_group(reason: str) -> str:
    lowered = str(reason).lower()
    if "floor" in lowered:
        return "floor"
    if "front" in lowered:
        return "front_wall"
    if "wall" in lowered:
        return "other_wall"
    if not lowered:
        return "unknown"
    return lowered


def _session_summary_path(path: Path) -> Path:
    return path / "manifests" / "glider_calibration_sequence_final_summary.json"


def resolve_session_roots(path: Path) -> list[Path]:
    """Resolve either one session root or all completed calibration sessions below a directory."""
    if _session_summary_path(path).exists():
        return [path]
    candidates: list[Path] = []
    if path.exists():
        for summary_path in path.rglob("manifests/glider_calibration_sequence_final_summary.json"):
            session_root = summary_path.parents[1]
            summary = _load_json(summary_path)
            if int(summary.get("total_valid_throw_count", 0) or 0) > 0:
                candidates.append(session_root)
    if not candidates:
        raise FileNotFoundError(
            f"No completed glider calibration session found under {path}. "
            "Pass --session-root to a folder containing manifests/glider_calibration_sequence_final_summary.json."
        )
    unique = {candidate.resolve(): candidate for candidate in candidates}
    return sorted(unique.values(), key=lambda item: item.name)


def _case_dirs(session_root: Path) -> list[Path]:
    ignored = {"manifests", "metrics", "reports"}
    return sorted(
        child for child in session_root.iterdir() if child.is_dir() and child.name not in ignored
    )


def _throw_summary(throw_dir: Path) -> dict[str, Any]:
    return _load_json(throw_dir / "manifests" / "glider_calibration_throw_summary.json")


def _throw_manifest(throw_dir: Path) -> dict[str, Any]:
    return _load_json(throw_dir / "manifests" / "glider_calibration_throw_manifest.json")


def _valid_throw_dirs(session_root: Path) -> list[Path]:
    throw_dirs: list[Path] = []
    for case_dir in _case_dirs(session_root):
        candidate_dirs = list(case_dir.glob("throw_*")) + list(case_dir.glob("v[0-9]*"))
        for throw_dir in sorted(candidate_dirs):
            summary = _throw_summary(throw_dir)
            if summary.get("valid_throw") is True:
                throw_dirs.append(throw_dir)
    return throw_dirs


def _invalid_attempt_dirs(session_root: Path) -> list[Path]:
    attempt_dirs: list[Path] = []
    for case_dir in _case_dirs(session_root):
        old_invalid_root = case_dir / "invalid_attempts"
        if old_invalid_root.exists():
            attempt_dirs.extend(sorted(old_invalid_root.glob("attempt_*")))
        new_invalid_root = case_dir / "bad"
        if new_invalid_root.exists():
            attempt_dirs.extend(sorted(new_invalid_root.glob("i[0-9]*")))
    return attempt_dirs


def summarize_valid_throw(session_label: str, session_root: Path, throw_dir: Path) -> dict[str, Any]:
    summary = _throw_summary(throw_dir)
    manifest = _throw_manifest(throw_dir)
    case = manifest.get("calibration_case", {})
    profile = manifest.get("calibration_profile", {})
    config = manifest.get("config", {})
    rows = _read_csv(throw_dir / "metrics" / "state_samples.csv")
    if not rows:
        raise ValueError(f"No state_samples.csv rows found for {throw_dir}")

    first = rows[0]
    last = rows[-1]
    t0 = _float(first, "t_s", 0.0)
    t1 = _float(last, "t_s", t0)
    duration_s = max(0.0, t1 - t0)
    u0, v0, w0 = _float(first, "u"), _float(first, "v"), _float(first, "w")
    p0, q0, r0 = _float(first, "p"), _float(first, "q"), _float(first, "r")
    phi0, theta0, psi0 = _float(first, "phi"), _float(first, "theta"), _float(first, "psi")
    x0, y0, z0 = _float(first, "x_w"), _float(first, "y_w"), _float(first, "z_w")
    x1, y1, z1 = _float(last, "x_w"), _float(last, "y_w"), _float(last, "z_w")
    dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
    altitude_loss_m = z0 - z1
    horizontal_distance_m = math.hypot(dx, dy)
    speed0 = math.sqrt(u0**2 + v0**2 + w0**2)
    horizontal_speed0 = math.hypot(u0, v0)
    flight_path_angle0 = math.degrees(math.atan2(-w0, max(horizontal_speed0, 1e-9)))
    sideslip_proxy0 = math.degrees(math.atan2(v0, max(abs(u0), 1e-9)))
    vicon_offset = _sequence(config.get("vicon_position_offset_m"), 3)
    termination_reason = str(summary.get("termination_reason", ""))

    speeds = [
        math.sqrt(_float(row, "u", 0.0) ** 2 + _float(row, "v", 0.0) ** 2 + _float(row, "w", 0.0) ** 2)
        for row in rows
    ]
    forward_speeds = [_float(row, "u") for row in rows]
    confidence = [_float(row, "estimator_rate_confidence") for row in rows]
    spike_flags = [row.get("estimator_spike_rejected", "").strip().lower() == "true" for row in rows]
    limited_flags = [row.get("estimator_body_rate_limited", "").strip().lower() == "true" for row in rows]

    return {
        "_throw_dir": throw_dir.as_posix(),
        "session_label": session_label,
        "source_session_root": session_root.as_posix(),
        "calibration_profile_id": profile.get("profile_id", ""),
        "calibration_profile_hash": summary.get("calibration_profile_hash", profile.get("profile_hash", "")),
        "vicon_calibration_source": profile.get("vicon_calibration_source", ""),
        "case_id": summary.get("case_id", case.get("case_id", "")),
        "case_name": summary.get("case_name", case.get("case_name", "")),
        "throw_id": throw_dir.name,
        "command_axis": case.get("command_axis", ""),
        "command_value": case.get("command_value", ""),
        "pulse_start_s": case.get("pulse_start_s", ""),
        "pulse_duration_s": case.get("pulse_duration_s", ""),
        "termination_reason": termination_reason,
        "termination_group": _termination_group(termination_reason),
        "launch_speed_m_s": summary.get("launch_speed_m_s", float("nan")),
        "duration_s": duration_s,
        "x0_m": x0,
        "y0_m": y0,
        "z0_m": z0,
        "launch_x_offset_m": x0 - LAUNCH_NOMINAL_X_M,
        "launch_y_offset_m": y0 - LAUNCH_NOMINAL_Y_M,
        "launch_z_offset_m": z0 - LAUNCH_NOMINAL_Z_M,
        "u0_m_s": u0,
        "v0_m_s": v0,
        "w0_m_s": w0,
        "p0_rad_s": p0,
        "q0_rad_s": q0,
        "r0_rad_s": r0,
        "phi0_deg": math.degrees(phi0),
        "theta0_deg": math.degrees(theta0),
        "psi0_deg": math.degrees(psi0),
        "speed0_m_s": speed0,
        "horizontal_speed0_m_s": horizontal_speed0,
        "flight_path_angle_proxy0_deg": flight_path_angle0,
        "sideslip_proxy0_deg": sideslip_proxy0,
        "vicon_offset_x_m": vicon_offset[0],
        "vicon_offset_y_m": vicon_offset[1],
        "vicon_offset_z_m": vicon_offset[2],
        "x_end_m": x1,
        "y_end_m": y1,
        "z_end_m": z1,
        "dx_m": dx,
        "dy_m": dy,
        "dz_m": dz,
        "altitude_loss_m": altitude_loss_m,
        "horizontal_distance_m": horizontal_distance_m,
        "glide_ratio_x_over_altloss": _ratio(dx, altitude_loss_m),
        "glide_ratio_horizontal_over_altloss": _ratio(horizontal_distance_m, altitude_loss_m),
        "sink_rate_m_s": _ratio(altitude_loss_m, duration_s),
        "mean_speed_m_s": _safe_mean(speeds),
        "mean_forward_speed_m_s": _safe_mean(forward_speeds),
        "max_abs_phi_deg": math.degrees(_safe_max([abs(_float(row, "phi")) for row in rows])),
        "max_abs_theta_deg": math.degrees(_safe_max([abs(_float(row, "theta")) for row in rows])),
        "max_abs_psi_deg": math.degrees(_safe_max([abs(_float(row, "psi")) for row in rows])),
        "max_abs_p_rad_s": _safe_max([abs(_float(row, "p")) for row in rows]),
        "max_abs_q_rad_s": _safe_max([abs(_float(row, "q")) for row in rows]),
        "max_abs_r_rad_s": _safe_max([abs(_float(row, "r")) for row in rows]),
        "mean_rate_confidence": _safe_mean(confidence),
        "min_rate_confidence": _safe_min(confidence),
        "spike_downweighted_fraction": _ratio(sum(spike_flags), len(spike_flags)),
        "body_rate_limited_fraction": _ratio(sum(limited_flags), len(limited_flags)),
        "sample_count": len(rows),
    }


def summarize_invalid_attempt(session_label: str, attempt_dir: Path) -> dict[str, Any]:
    summary = _throw_summary(attempt_dir)
    manifest = _throw_manifest(attempt_dir)
    case = manifest.get("calibration_case", {})
    event_rows = _read_csv(attempt_dir / "metrics" / "runtime_events.csv")
    details: dict[str, Any] = {}
    for row in event_rows:
        if "rejected_launch_attempt" in row.get("event", ""):
            try:
                details = json.loads(row.get("details_json", "{}"))
            except json.JSONDecodeError:
                details = {}
    return {
        "session_label": session_label,
        "case_id": summary.get("case_id", case.get("case_id", "")),
        "case_name": summary.get("case_name", case.get("case_name", "")),
        "attempt_id": attempt_dir.name,
        "cancellation_reason": summary.get("cancellation_reason", ""),
        "launch_gate_reason": details.get("launch_gate_reason", ""),
        "trigger_source": details.get("trigger_source", ""),
        "speed_m_s": details.get("speed_m_s", details.get("launch_attempt_speed_m_s", float("nan"))),
        "x_w_m": details.get("x_w_m", float("nan")),
        "y_w_m": details.get("y_w_m", float("nan")),
        "z_w_m": details.get("z_w_m", float("nan")),
        "phi_deg": details.get("phi_deg", float("nan")),
        "theta_deg": details.get("theta_deg", float("nan")),
        "psi_deg": details.get("psi_deg", float("nan")),
        "p_rad_s": details.get("p_rad_s", float("nan")),
        "q_rad_s": details.get("q_rad_s", float("nan")),
        "r_rad_s": details.get("r_rad_s", float("nan")),
    }


def aggregate_valid_metrics(valid_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    roles = {
        "launch_speed_m_s": "stratify model error versus launch energy; do not fit directly",
        "sink_rate_m_s": "primary neutral-glide target for drag/lift/trim consistency",
        "glide_ratio_x_over_altloss": "derived consistency check from distance and altitude loss; not direct fit target",
        "dy_m": "lateral trim/asymmetry diagnostic before fitting aileron/rudder effects",
        "launch_y_offset_m": "manual-launch lateral offset feature; include in empirical residual model",
        "launch_z_offset_m": "manual-launch height offset feature; include in empirical residual model",
        "flight_path_angle_proxy0_deg": "launch-condition feature for sink/distance residuals",
        "sideslip_proxy0_deg": "launch-condition feature for lateral/yaw residuals",
        "max_abs_phi_deg": "attitude envelope check; should not be used alone as fit objective",
        "max_abs_theta_deg": "pitch/CG/static-margin diagnostic before controller retuning",
        "max_abs_p_rad_s": "roll-rate envelope diagnostic for future damping/control derivative fit",
        "max_abs_q_rad_s": "pitch-rate envelope diagnostic for future damping/control derivative fit",
        "max_abs_r_rad_s": "yaw-rate envelope diagnostic for future damping/control derivative fit",
        "mean_rate_confidence": "state-estimator evidence quality; exclude low-confidence throws from fitting",
        "spike_downweighted_fraction": "state-estimator health diagnostic",
    }
    rows: list[dict[str, Any]] = []
    for metric, role in roles.items():
        values = [_to_float(row.get(metric)) for row in valid_rows]
        finite = [value for value in values if math.isfinite(value)]
        rows.append(
            {
                "metric": metric,
                "count": len(finite),
                "mean": _safe_mean(finite),
                "median": _safe_median(finite),
                "std": _safe_std(finite),
                "min": _safe_min(finite),
                "max": _safe_max(finite),
                "role_for_model_calibration": role,
            }
        )
    return rows


def _to_float(value: Any) -> float:
    try:
        if value in ("", None):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def build_feature_target_rows(valid_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{field: row.get(field, "") for field in FEATURE_TARGET_FIELDS} for row in valid_rows]


def session_termination_summary(valid_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in valid_rows:
        key = (
            str(row.get("session_label", "")),
            str(row.get("calibration_profile_hash", "")),
            str(row.get("termination_group", "")),
        )
        grouped.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for (session_label, profile_hash, termination_group), rows in sorted(grouped.items()):
        output.append(
            {
                "session_label": session_label,
                "calibration_profile_hash": profile_hash,
                "termination_group": termination_group,
                "throw_count": len(rows),
                "mean_launch_speed_m_s": _safe_mean([_to_float(row.get("launch_speed_m_s")) for row in rows]),
                "mean_sink_rate_m_s": _safe_mean([_to_float(row.get("sink_rate_m_s")) for row in rows]),
                "mean_dx_m": _safe_mean([_to_float(row.get("dx_m")) for row in rows]),
                "mean_dy_m": _safe_mean([_to_float(row.get("dy_m")) for row in rows]),
                "mean_duration_s": _safe_mean([_to_float(row.get("duration_s")) for row in rows]),
                "mean_glide_ratio_x_over_altloss": _safe_mean(
                    [_to_float(row.get("glide_ratio_x_over_altloss")) for row in rows]
                ),
            }
        )
    return output


def stratified_heldout_indices(
    valid_rows: list[dict[str, Any]],
    *,
    heldout_count: int,
    heldout_seed: int,
    group_key: str = "session_label",
) -> set[int]:
    """Return randomized held-out indices spread across session/profile groups."""

    if not valid_rows or int(heldout_count) <= 0:
        return set()
    rng = random.Random(int(heldout_seed))
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(valid_rows):
        grouped.setdefault(str(row.get(group_key, "")), []).append(index)
    for indices in grouped.values():
        rng.shuffle(indices)
    group_names = sorted(grouped)
    rng.shuffle(group_names)
    selected: set[int] = set()
    while len(selected) < min(int(heldout_count), len(valid_rows)):
        progressed = False
        for group_name in group_names:
            if len(selected) >= int(heldout_count):
                break
            indices = grouped[group_name]
            while indices and indices[0] in selected:
                indices.pop(0)
            if indices:
                selected.add(indices.pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def measured_launch_replay_diagnostics(
    valid_rows: list[dict[str, Any]],
    *,
    replay_dt_s: float = DEFAULT_REPLAY_DT_S,
    workers: int = DEFAULT_WORKERS,
) -> list[dict[str, Any]]:
    """Replay each real throw from its measured launch state in the dry-air model.

    This is the fair simulator comparison used for SysID: launch variability stays
    in the initial condition, while residuals expose airframe/control mismatch.
    """

    worker_count = max(1, int(workers))
    if worker_count > 1 and len(valid_rows) > 1:
        payloads = [(row, float(replay_dt_s)) for row in valid_rows]
        with ProcessPoolExecutor(max_workers=worker_count, initializer=_initialise_replay_worker) as executor:
            return list(executor.map(_measured_launch_replay_worker, payloads))

    aircraft = adapt_glider(build_nausicaa_glider())
    rows: list[dict[str, Any]] = []
    for row in valid_rows:
        throw_dir_value = row.get("_throw_dir", "")
        throw_dir = Path(str(throw_dir_value)) if throw_dir_value else None
        if throw_dir is None or not throw_dir.exists():
            rows.append(_blocked_replay_row(row, "missing_throw_dir", replay_dt_s))
            continue
        rows.append(_simulate_measured_launch_replay(row, throw_dir, aircraft, replay_dt_s))
    return rows


def aligned_motion_replay_diagnostics(
    valid_rows: list[dict[str, Any]],
    *,
    replay_dt_s: float = DEFAULT_REPLAY_DT_S,
    alignment_window_s: float = DEFAULT_ALIGNMENT_WINDOW_S,
    workers: int = DEFAULT_WORKERS,
) -> list[dict[str, Any]]:
    """Replay from a state aligned to the first short segment of measured motion.

    This diagnostic separates model mismatch from launch-frame derivative noise.
    It is not a deployment validation metric because it conditions on the first
    measured flight segment before predicting the remaining trajectory.
    """

    worker_count = max(1, int(workers))
    if worker_count > 1 and len(valid_rows) > 1:
        payloads = [(row, float(replay_dt_s), float(alignment_window_s)) for row in valid_rows]
        with ProcessPoolExecutor(max_workers=worker_count, initializer=_initialise_replay_worker) as executor:
            return list(executor.map(_aligned_motion_replay_worker, payloads))

    aircraft = adapt_glider(build_nausicaa_glider())
    rows: list[dict[str, Any]] = []
    for row in valid_rows:
        throw_dir_value = row.get("_throw_dir", "")
        throw_dir = Path(str(throw_dir_value)) if throw_dir_value else None
        if throw_dir is None or not throw_dir.exists():
            rows.append(_blocked_aligned_replay_row(row, "missing_throw_dir", replay_dt_s, alignment_window_s))
            continue
        rows.append(_simulate_aligned_motion_replay(row, throw_dir, aircraft, replay_dt_s, alignment_window_s))
    return rows


def _initialise_replay_worker() -> None:
    global _REPLAY_WORKER_AIRCRAFT
    _REPLAY_WORKER_AIRCRAFT = adapt_glider(build_nausicaa_glider())


def _measured_launch_replay_worker(payload: tuple[dict[str, Any], float]) -> dict[str, Any]:
    row, replay_dt_s = payload
    aircraft = _REPLAY_WORKER_AIRCRAFT
    if aircraft is None:
        aircraft = adapt_glider(build_nausicaa_glider())
    throw_dir_value = row.get("_throw_dir", "")
    throw_dir = Path(str(throw_dir_value)) if throw_dir_value else None
    if throw_dir is None or not throw_dir.exists():
        return _blocked_replay_row(row, "missing_throw_dir", replay_dt_s)
    return _simulate_measured_launch_replay(row, throw_dir, aircraft, replay_dt_s)


def _aligned_motion_replay_worker(payload: tuple[dict[str, Any], float, float]) -> dict[str, Any]:
    row, replay_dt_s, alignment_window_s = payload
    aircraft = _REPLAY_WORKER_AIRCRAFT
    if aircraft is None:
        aircraft = adapt_glider(build_nausicaa_glider())
    throw_dir_value = row.get("_throw_dir", "")
    throw_dir = Path(str(throw_dir_value)) if throw_dir_value else None
    if throw_dir is None or not throw_dir.exists():
        return _blocked_aligned_replay_row(row, "missing_throw_dir", replay_dt_s, alignment_window_s)
    return _simulate_aligned_motion_replay(row, throw_dir, aircraft, replay_dt_s, alignment_window_s)


def _simulate_measured_launch_replay(
    row: dict[str, Any],
    throw_dir: Path,
    aircraft: Any,
    replay_dt_s: float,
) -> dict[str, Any]:
    sample_rows = _read_csv(throw_dir / "metrics" / "state_samples.csv")
    if not sample_rows:
        return _blocked_replay_row(row, "missing_state_samples", replay_dt_s)
    x0 = _state_vector_from_sample_row(sample_rows[0])
    if not np.all(np.isfinite(x0)):
        return _blocked_replay_row(row, "nonfinite_initial_state", replay_dt_s)
    t0 = _float(sample_rows[0], "t_s", 0.0)
    t1 = _float(sample_rows[-1], "t_s", t0)
    duration_s = max(0.0, t1 - t0)
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        return _blocked_replay_row(row, "invalid_duration", replay_dt_s)

    manifest = _throw_manifest(throw_dir)
    actuator_tau_s = _actuator_tau_from_manifest(manifest)
    command_onset_delay_s = DEFAULT_REPLAY_COMMAND_ONSET_DELAY_S
    command_schedule, command_source = _load_replay_command_schedule(
        throw_dir,
        row,
        command_onset_delay_s=command_onset_delay_s,
    )
    x = x0.copy()
    t_s = 0.0
    command_index = 0
    first_exit_reason = ""
    first_exit_time_s = float("nan")
    max_abs_phi = abs(float(x[3]))
    max_abs_theta = abs(float(x[4]))
    max_abs_p = abs(float(x[9]))
    max_abs_q = abs(float(x[10]))
    max_abs_r = abs(float(x[11]))

    while t_s < duration_s - 1e-12:
        dt_s = min(float(replay_dt_s), duration_s - t_s)
        command_index = _advance_command_index(command_schedule, command_index, t_s)
        command = command_schedule[command_index][1]
        try:
            x = _rk4_step_measured_launch(
                x=x,
                command=command,
                aircraft=aircraft,
                actuator_tau_s=actuator_tau_s,
                dt_s=dt_s,
            )
        except Exception:
            return _blocked_replay_row(row, "state_derivative_failed", replay_dt_s)
        t_s += dt_s
        if not np.all(np.isfinite(x)):
            return _blocked_replay_row(row, "nonfinite_replay_state", replay_dt_s)
        max_abs_phi = max(max_abs_phi, abs(float(x[3])))
        max_abs_theta = max(max_abs_theta, abs(float(x[4])))
        max_abs_p = max(max_abs_p, abs(float(x[9])))
        max_abs_q = max(max_abs_q, abs(float(x[10])))
        max_abs_r = max(max_abs_r, abs(float(x[11])))
        if not first_exit_reason:
            reason = _true_safe_exit_reason(x)
            if reason:
                first_exit_reason = reason
                first_exit_time_s = t_s

    actual_dx = _to_float(row.get("dx_m"))
    actual_dy = _to_float(row.get("dy_m"))
    actual_altitude_loss = _to_float(row.get("altitude_loss_m"))
    actual_sink_rate = _to_float(row.get("sink_rate_m_s"))
    actual_glide = _to_float(row.get("glide_ratio_x_over_altloss"))
    sim_dx = float(x[0] - x0[0])
    sim_dy = float(x[1] - x0[1])
    sim_altitude_loss = float(x0[2] - x[2])
    sim_sink_rate = _ratio(sim_altitude_loss, duration_s)
    sim_glide = _ratio(sim_dx, sim_altitude_loss)
    actual_final = _state_vector_from_sample_row(sample_rows[-1])

    return {
        "session_label": row.get("session_label", ""),
        "calibration_profile_hash": row.get("calibration_profile_hash", ""),
        "case_id": row.get("case_id", ""),
        "case_name": row.get("case_name", ""),
        "throw_id": row.get("throw_id", ""),
        "command_axis": row.get("command_axis", ""),
        "command_value": row.get("command_value", ""),
        "pulse_start_s": row.get("pulse_start_s", ""),
        "pulse_duration_s": row.get("pulse_duration_s", ""),
        "replay_status": "ok",
        "replay_policy": "dry_air_current_model_exact_measured_launch_state_same_command_history_same_duration",
        "replay_dt_s": float(replay_dt_s),
        "replay_command_source": command_source,
        "replay_command_onset_delay_s": command_onset_delay_s,
        "actual_termination_group": row.get("termination_group", ""),
        "sim_first_exit_reason": first_exit_reason or "none_before_actual_duration",
        "sim_first_exit_time_s": first_exit_time_s,
        "duration_s": duration_s,
        "x0_m": float(x0[0]),
        "y0_m": float(x0[1]),
        "z0_m": float(x0[2]),
        "u0_m_s": float(x0[6]),
        "v0_m_s": float(x0[7]),
        "w0_m_s": float(x0[8]),
        "p0_rad_s": float(x0[9]),
        "q0_rad_s": float(x0[10]),
        "r0_rad_s": float(x0[11]),
        "phi0_deg": math.degrees(float(x0[3])),
        "theta0_deg": math.degrees(float(x0[4])),
        "psi0_deg": math.degrees(float(x0[5])),
        "actual_x_end_m": row.get("x_end_m", ""),
        "actual_y_end_m": row.get("y_end_m", ""),
        "actual_z_end_m": row.get("z_end_m", ""),
        "sim_x_end_m": float(x[0]),
        "sim_y_end_m": float(x[1]),
        "sim_z_end_m": float(x[2]),
        "actual_dx_m": actual_dx,
        "sim_dx_m": sim_dx,
        "dx_residual_actual_minus_sim_m": actual_dx - sim_dx,
        "actual_dy_m": actual_dy,
        "sim_dy_m": sim_dy,
        "dy_residual_actual_minus_sim_m": actual_dy - sim_dy,
        "actual_altitude_loss_m": actual_altitude_loss,
        "sim_altitude_loss_m": sim_altitude_loss,
        "altitude_loss_residual_actual_minus_sim_m": actual_altitude_loss - sim_altitude_loss,
        "actual_sink_rate_m_s": actual_sink_rate,
        "sim_sink_rate_m_s": sim_sink_rate,
        "sink_rate_residual_actual_minus_sim_m_s": actual_sink_rate - sim_sink_rate,
        "actual_glide_ratio_x_over_altloss": actual_glide,
        "sim_glide_ratio_x_over_altloss": sim_glide,
        "glide_ratio_residual_actual_minus_sim": actual_glide - sim_glide,
        "actual_final_phi_deg": math.degrees(float(actual_final[3])),
        "sim_final_phi_deg": math.degrees(float(x[3])),
        "actual_final_theta_deg": math.degrees(float(actual_final[4])),
        "sim_final_theta_deg": math.degrees(float(x[4])),
        "actual_final_psi_deg": math.degrees(float(actual_final[5])),
        "sim_final_psi_deg": math.degrees(float(x[5])),
        "max_abs_sim_phi_deg": math.degrees(max_abs_phi),
        "max_abs_sim_theta_deg": math.degrees(max_abs_theta),
        "max_abs_sim_p_rad_s": max_abs_p,
        "max_abs_sim_q_rad_s": max_abs_q,
        "max_abs_sim_r_rad_s": max_abs_r,
    }


def _simulate_aligned_motion_replay(
    row: dict[str, Any],
    throw_dir: Path,
    aircraft: Any,
    replay_dt_s: float,
    alignment_window_s: float,
) -> dict[str, Any]:
    sample_rows = _read_csv(throw_dir / "metrics" / "state_samples.csv")
    if not sample_rows:
        return _blocked_aligned_replay_row(row, "missing_state_samples", replay_dt_s, alignment_window_s)
    aligned = _aligned_state_from_sample_rows(sample_rows, alignment_window_s)
    if aligned["status"] != "ok":
        return _blocked_aligned_replay_row(row, str(aligned["status"]), replay_dt_s, alignment_window_s)
    x0 = np.asarray(aligned["state"], dtype=float)
    if not np.all(np.isfinite(x0)):
        return _blocked_aligned_replay_row(row, "nonfinite_aligned_state", replay_dt_s, alignment_window_s)

    t_first = _float(sample_rows[0], "t_s", 0.0)
    t_last = _float(sample_rows[-1], "t_s", t_first)
    alignment_elapsed_s = float(aligned["alignment_elapsed_s"])
    duration_s = max(0.0, t_last - t_first - alignment_elapsed_s)
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        return _blocked_aligned_replay_row(row, "invalid_aligned_duration", replay_dt_s, alignment_window_s)

    manifest = _throw_manifest(throw_dir)
    actuator_tau_s = _actuator_tau_from_manifest(manifest)
    command_onset_delay_s = DEFAULT_REPLAY_COMMAND_ONSET_DELAY_S
    command_schedule, command_source = _load_replay_command_schedule(
        throw_dir,
        row,
        command_onset_delay_s=command_onset_delay_s,
    )
    x = x0.copy()
    t_s = 0.0
    command_index = _advance_command_index(command_schedule, 0, alignment_elapsed_s)
    first_exit_reason = ""
    first_exit_time_s = float("nan")
    max_abs_phi = abs(float(x[3]))
    max_abs_theta = abs(float(x[4]))
    max_abs_p = abs(float(x[9]))
    max_abs_q = abs(float(x[10]))
    max_abs_r = abs(float(x[11]))

    while t_s < duration_s - 1e-12:
        dt_s = min(float(replay_dt_s), duration_s - t_s)
        absolute_replay_time_s = alignment_elapsed_s + t_s
        command_index = _advance_command_index(command_schedule, command_index, absolute_replay_time_s)
        command = command_schedule[command_index][1]
        try:
            x = _rk4_step_measured_launch(
                x=x,
                command=command,
                aircraft=aircraft,
                actuator_tau_s=actuator_tau_s,
                dt_s=dt_s,
            )
        except Exception:
            return _blocked_aligned_replay_row(row, "state_derivative_failed", replay_dt_s, alignment_window_s)
        t_s += dt_s
        if not np.all(np.isfinite(x)):
            return _blocked_aligned_replay_row(row, "nonfinite_replay_state", replay_dt_s, alignment_window_s)
        max_abs_phi = max(max_abs_phi, abs(float(x[3])))
        max_abs_theta = max(max_abs_theta, abs(float(x[4])))
        max_abs_p = max(max_abs_p, abs(float(x[9])))
        max_abs_q = max(max_abs_q, abs(float(x[10])))
        max_abs_r = max(max_abs_r, abs(float(x[11])))
        if not first_exit_reason:
            reason = _true_safe_exit_reason(x)
            if reason:
                first_exit_reason = reason
                first_exit_time_s = t_s

    actual_final = _state_vector_from_sample_row(sample_rows[-1])
    actual_dx = float(actual_final[0] - x0[0])
    actual_dy = float(actual_final[1] - x0[1])
    actual_altitude_loss = float(x0[2] - actual_final[2])
    actual_sink_rate = _ratio(actual_altitude_loss, duration_s)
    actual_glide = _ratio(actual_dx, actual_altitude_loss)
    sim_dx = float(x[0] - x0[0])
    sim_dy = float(x[1] - x0[1])
    sim_altitude_loss = float(x0[2] - x[2])
    sim_sink_rate = _ratio(sim_altitude_loss, duration_s)
    sim_glide = _ratio(sim_dx, sim_altitude_loss)

    return {
        "session_label": row.get("session_label", ""),
        "calibration_profile_hash": row.get("calibration_profile_hash", ""),
        "case_id": row.get("case_id", ""),
        "case_name": row.get("case_name", ""),
        "throw_id": row.get("throw_id", ""),
        "command_axis": row.get("command_axis", ""),
        "command_value": row.get("command_value", ""),
        "pulse_start_s": row.get("pulse_start_s", ""),
        "pulse_duration_s": row.get("pulse_duration_s", ""),
        "replay_status": "ok",
        "replay_policy": "dry_air_current_model_aligned_first_motion_window_same_command_history_remaining_duration",
        "replay_dt_s": float(replay_dt_s),
        "alignment_window_s": float(alignment_window_s),
        "alignment_elapsed_s": alignment_elapsed_s,
        "alignment_sample_count": int(aligned["alignment_sample_count"]),
        "alignment_method": aligned["alignment_method"],
        "replay_command_source": command_source,
        "replay_command_onset_delay_s": command_onset_delay_s,
        "actual_termination_group": row.get("termination_group", ""),
        "sim_first_exit_reason": first_exit_reason or "none_before_actual_remaining_duration",
        "sim_first_exit_time_s": first_exit_time_s,
        "duration_s": duration_s,
        "x0_m": float(x0[0]),
        "y0_m": float(x0[1]),
        "z0_m": float(x0[2]),
        "u0_m_s": float(x0[6]),
        "v0_m_s": float(x0[7]),
        "w0_m_s": float(x0[8]),
        "p0_rad_s": float(x0[9]),
        "q0_rad_s": float(x0[10]),
        "r0_rad_s": float(x0[11]),
        "phi0_deg": math.degrees(float(x0[3])),
        "theta0_deg": math.degrees(float(x0[4])),
        "psi0_deg": math.degrees(float(x0[5])),
        "actual_x_end_m": float(actual_final[0]),
        "actual_y_end_m": float(actual_final[1]),
        "actual_z_end_m": float(actual_final[2]),
        "sim_x_end_m": float(x[0]),
        "sim_y_end_m": float(x[1]),
        "sim_z_end_m": float(x[2]),
        "actual_dx_m": actual_dx,
        "sim_dx_m": sim_dx,
        "dx_residual_actual_minus_sim_m": actual_dx - sim_dx,
        "actual_dy_m": actual_dy,
        "sim_dy_m": sim_dy,
        "dy_residual_actual_minus_sim_m": actual_dy - sim_dy,
        "actual_altitude_loss_m": actual_altitude_loss,
        "sim_altitude_loss_m": sim_altitude_loss,
        "altitude_loss_residual_actual_minus_sim_m": actual_altitude_loss - sim_altitude_loss,
        "actual_sink_rate_m_s": actual_sink_rate,
        "sim_sink_rate_m_s": sim_sink_rate,
        "sink_rate_residual_actual_minus_sim_m_s": actual_sink_rate - sim_sink_rate,
        "actual_glide_ratio_x_over_altloss": actual_glide,
        "sim_glide_ratio_x_over_altloss": sim_glide,
        "glide_ratio_residual_actual_minus_sim": actual_glide - sim_glide,
        "actual_final_phi_deg": math.degrees(float(actual_final[3])),
        "sim_final_phi_deg": math.degrees(float(x[3])),
        "actual_final_theta_deg": math.degrees(float(actual_final[4])),
        "sim_final_theta_deg": math.degrees(float(x[4])),
        "actual_final_psi_deg": math.degrees(float(actual_final[5])),
        "sim_final_psi_deg": math.degrees(float(x[5])),
        "max_abs_sim_phi_deg": math.degrees(max_abs_phi),
        "max_abs_sim_theta_deg": math.degrees(max_abs_theta),
        "max_abs_sim_p_rad_s": max_abs_p,
        "max_abs_sim_q_rad_s": max_abs_q,
        "max_abs_sim_r_rad_s": max_abs_r,
    }


def _blocked_replay_row(row: dict[str, Any], status: str, replay_dt_s: float) -> dict[str, Any]:
    return {
        "session_label": row.get("session_label", ""),
        "calibration_profile_hash": row.get("calibration_profile_hash", ""),
        "case_id": row.get("case_id", ""),
        "case_name": row.get("case_name", ""),
        "throw_id": row.get("throw_id", ""),
        "command_axis": row.get("command_axis", ""),
        "command_value": row.get("command_value", ""),
        "pulse_start_s": row.get("pulse_start_s", ""),
        "pulse_duration_s": row.get("pulse_duration_s", ""),
        "replay_status": status,
        "replay_policy": "dry_air_current_model_exact_measured_launch_state_same_command_history_same_duration",
        "replay_dt_s": float(replay_dt_s),
        "replay_command_source": "",
        "replay_command_onset_delay_s": DEFAULT_REPLAY_COMMAND_ONSET_DELAY_S,
        "actual_termination_group": row.get("termination_group", ""),
    }


def _blocked_aligned_replay_row(
    row: dict[str, Any],
    status: str,
    replay_dt_s: float,
    alignment_window_s: float,
) -> dict[str, Any]:
    blocked = _blocked_replay_row(row, status, replay_dt_s)
    blocked.update(
        {
            "replay_policy": "dry_air_current_model_aligned_first_motion_window_same_command_history_remaining_duration",
            "alignment_window_s": float(alignment_window_s),
            "alignment_elapsed_s": "",
            "alignment_sample_count": "",
            "alignment_method": "",
        }
    )
    return blocked


def _state_vector_from_sample_row(row: dict[str, Any]) -> np.ndarray:
    names = ("x_w", "y_w", "z_w", "phi", "theta", "psi", "u", "v", "w", "p", "q", "r", "delta_a", "delta_e", "delta_r")
    return np.array([_float(row, name) for name in names], dtype=float)


def _aligned_state_from_sample_rows(sample_rows: list[dict[str, Any]], alignment_window_s: float) -> dict[str, Any]:
    if len(sample_rows) < 3:
        return {"status": "too_few_state_samples"}
    t0 = _float(sample_rows[0], "t_s", 0.0)
    rel_times = np.asarray([_float(row, "t_s", t0) - t0 for row in sample_rows], dtype=float)
    if not np.all(np.isfinite(rel_times)):
        return {"status": "nonfinite_sample_time"}
    target_time = max(0.0, float(alignment_window_s))
    target_candidates = np.where(rel_times >= target_time)[0]
    if len(target_candidates):
        target_index = int(target_candidates[0])
    else:
        target_index = len(sample_rows) - 1
    alignment_elapsed_s = float(rel_times[target_index])
    if alignment_elapsed_s < 0.05:
        return {"status": "alignment_window_too_short"}
    window_indices = np.where((rel_times >= -1e-12) & (rel_times <= alignment_elapsed_s + 1e-12))[0]
    if len(window_indices) < 3:
        return {"status": "too_few_alignment_samples"}

    target_state = _state_vector_from_sample_row(sample_rows[target_index])
    if not np.all(np.isfinite(target_state)):
        return {"status": "nonfinite_alignment_target_state"}

    t = rel_times[window_indices]
    world_velocity = np.asarray(
        [
            _least_squares_slope(t, [_float(sample_rows[int(index)], name) for index in window_indices])
            for name in ("x_w", "y_w", "z_w")
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(world_velocity)):
        return {"status": "nonfinite_alignment_velocity"}

    start_state = _state_vector_from_sample_row(sample_rows[0])
    if not np.all(np.isfinite(start_state[:6])):
        return {"status": "nonfinite_alignment_start_attitude"}

    aligned_state = target_state.copy()
    aligned_state[6:9] = _body_velocity_from_world_up(world_velocity, aligned_state[3:6])
    aligned_state[9:12] = _body_rates_from_rotation_delta(
        _c_wb_numpy(float(start_state[3]), float(start_state[4]), float(start_state[5])),
        _c_wb_numpy(float(aligned_state[3]), float(aligned_state[4]), float(aligned_state[5])),
        alignment_elapsed_s,
    )
    return {
        "status": "ok",
        "state": aligned_state,
        "alignment_elapsed_s": alignment_elapsed_s,
        "alignment_sample_count": int(len(window_indices)),
        "alignment_method": "target_pose_plus_regressed_world_velocity_and_so3_rate_over_first_window",
    }


def _least_squares_slope(t_s: np.ndarray, values: list[float]) -> float:
    t = np.asarray(t_s, dtype=float)
    y = np.asarray(values, dtype=float)
    valid = np.isfinite(t) & np.isfinite(y)
    if int(np.sum(valid)) < 2:
        return float("nan")
    t = t[valid]
    y = y[valid]
    t_centered = t - float(np.mean(t))
    denominator = float(np.sum(t_centered * t_centered))
    if denominator < 1e-12:
        return float("nan")
    return float(np.sum(t_centered * (y - float(np.mean(y)))) / denominator)


def _body_velocity_from_world_up(world_velocity_m_s: np.ndarray, euler_rad: np.ndarray) -> np.ndarray:
    velocity_internal = np.asarray(world_velocity_m_s, dtype=float).reshape(3).copy()
    velocity_internal[2] *= -1.0
    c_wb = _c_wb_numpy(*np.asarray(euler_rad, dtype=float).reshape(3))
    return c_wb.T @ velocity_internal


def _c_wb_numpy(phi: float, theta: float, psi: float) -> np.ndarray:
    c_phi = np.cos(phi)
    s_phi = np.sin(phi)
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    c_psi = np.cos(psi)
    s_psi = np.sin(psi)
    return np.asarray(
        [
            [
                c_theta * c_psi,
                s_phi * s_theta * c_psi - c_phi * s_psi,
                c_phi * s_theta * c_psi + s_phi * s_psi,
            ],
            [
                c_theta * s_psi,
                s_phi * s_theta * s_psi + c_phi * c_psi,
                c_phi * s_theta * s_psi - s_phi * c_psi,
            ],
            [-s_theta, s_phi * c_theta, c_phi * c_theta],
        ],
        dtype=float,
    )


def _body_rates_from_rotation_delta(previous_c_wb: np.ndarray, current_c_wb: np.ndarray, dt_s: float) -> np.ndarray:
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        return np.zeros(3)
    previous = np.asarray(previous_c_wb, dtype=float).reshape(3, 3)
    current = np.asarray(current_c_wb, dtype=float).reshape(3, 3)
    relative_body_rotation = previous.T @ current
    trace_term = float(np.clip((np.trace(relative_body_rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(trace_term))
    vee = np.asarray(
        [
            relative_body_rotation[2, 1] - relative_body_rotation[1, 2],
            relative_body_rotation[0, 2] - relative_body_rotation[2, 0],
            relative_body_rotation[1, 0] - relative_body_rotation[0, 1],
        ],
        dtype=float,
    )
    if angle < 1e-6:
        rotation_vector = 0.5 * vee
    else:
        rotation_vector = (angle / (2.0 * np.sin(angle))) * vee
    return rotation_vector / float(dt_s)


def _actuator_tau_from_manifest(manifest: dict[str, Any]) -> tuple[float, float, float]:
    values = manifest.get("config", {}).get("actuator_tau_s", (0.06, 0.06, 0.06))
    sequence = _sequence(values, 3)
    if not all(math.isfinite(value) and value > 0.0 for value in sequence):
        return (0.06, 0.06, 0.06)
    return tuple(float(value) for value in sequence)


def _load_replay_command_schedule(
    throw_dir: Path,
    row: dict[str, Any],
    *,
    command_onset_delay_s: float,
) -> tuple[list[tuple[float, np.ndarray]], str]:
    schedule_rows = _read_csv(throw_dir / "metrics" / "command_schedule.csv")
    schedule: list[tuple[float, np.ndarray]] = []
    for command_row in schedule_rows:
        command_norm = np.array(
            [
                _float(command_row, "delta_a_cmd_norm", 0.0),
                _float(command_row, "delta_e_cmd_norm", 0.0),
                _float(command_row, "delta_r_cmd_norm", 0.0),
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(command_norm)):
            continue
        schedule.append(
            (
                max(0.0, _float(command_row, "t_s", 0.0)),
                normalised_command_to_surface_rad(command_norm),
            )
        )
    if schedule:
        return (
            _apply_replay_command_onset_delay(
                sorted(schedule, key=lambda item: item[0]),
                command_onset_delay_s,
            ),
            "command_schedule_csv_normalised_aggregate_to_radians_with_nominal_onset_delay",
        )

    return (
        _apply_replay_command_onset_delay(
            _case_metadata_command_schedule(row),
            command_onset_delay_s,
        ),
        "case_metadata_pulse_normalised_aggregate_to_radians_with_nominal_onset_delay",
    )


def _apply_replay_command_onset_delay(
    schedule: list[tuple[float, np.ndarray]],
    command_onset_delay_s: float,
) -> list[tuple[float, np.ndarray]]:
    if not schedule:
        return [(0.0, normalised_command_to_surface_rad(np.zeros(3, dtype=float)))]
    onset = max(0.0, float(command_onset_delay_s))
    if onset <= 0.0:
        return [(float(time_s), command.copy()) for time_s, command in schedule]
    neutral = normalised_command_to_surface_rad(np.zeros(3, dtype=float))
    first_time_s, first_command = schedule[0]
    if float(first_time_s) <= 1e-12:
        delayed = [(0.0, first_command.copy())]
        remaining = schedule[1:]
    else:
        delayed = [(0.0, neutral)]
        remaining = schedule
    for time_s, command in remaining:
        delayed.append((max(0.0, float(time_s) + onset), command.copy()))
    return sorted(delayed, key=lambda item: item[0])


def _case_metadata_command_schedule(row: dict[str, Any]) -> list[tuple[float, np.ndarray]]:
    axis = str(row.get("command_axis", "neutral"))
    value = _to_float(row.get("command_value"))
    pulse_start_s = _to_float(row.get("pulse_start_s"))
    pulse_duration_s = _to_float(row.get("pulse_duration_s"))
    neutral = normalised_command_to_surface_rad(np.zeros(3, dtype=float))
    if axis not in {"delta_a", "delta_e", "delta_r"} or not math.isfinite(value):
        return [(0.0, neutral)]
    if not math.isfinite(pulse_start_s) or not math.isfinite(pulse_duration_s) or pulse_duration_s <= 0.0:
        return [(0.0, neutral)]
    command_norm = np.zeros(3, dtype=float)
    command_norm[{"delta_a": 0, "delta_e": 1, "delta_r": 2}[axis]] = value
    command = normalised_command_to_surface_rad(command_norm)
    return [
        (0.0, neutral),
        (max(0.0, pulse_start_s), command),
        (max(0.0, pulse_start_s + pulse_duration_s), neutral),
    ]


def _advance_command_index(
    schedule: list[tuple[float, np.ndarray]],
    current_index: int,
    t_s: float,
) -> int:
    index = int(current_index)
    while index + 1 < len(schedule) and float(schedule[index + 1][0]) <= float(t_s) + 1e-12:
        index += 1
    return index


def _rk4_step_measured_launch(
    *,
    x: np.ndarray,
    command: np.ndarray,
    aircraft: Any,
    actuator_tau_s: tuple[float, float, float],
    dt_s: float,
) -> np.ndarray:
    k1 = state_derivative(x, command, aircraft, wind_model=None, actuator_tau_s=actuator_tau_s, wind_mode="panel")
    k2 = state_derivative(
        x + 0.5 * dt_s * k1,
        command,
        aircraft,
        wind_model=None,
        actuator_tau_s=actuator_tau_s,
        wind_mode="panel",
    )
    k3 = state_derivative(
        x + 0.5 * dt_s * k2,
        command,
        aircraft,
        wind_model=None,
        actuator_tau_s=actuator_tau_s,
        wind_mode="panel",
    )
    k4 = state_derivative(
        x + dt_s * k3,
        command,
        aircraft,
        wind_model=None,
        actuator_tau_s=actuator_tau_s,
        wind_mode="panel",
    )
    return x + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _true_safe_exit_reason(state: np.ndarray) -> str:
    margins = position_margin_m(np.asarray(state[:3], dtype=float), TRUE_SAFE_BOUNDS)
    if float(margins["min_margin_m"]) >= 0.0:
        return ""
    x_w, y_w, z_w = (float(state[0]), float(state[1]), float(state[2]))
    if z_w < TRUE_SAFE_BOUNDS.z_w_m[0]:
        return "floor"
    if z_w > TRUE_SAFE_BOUNDS.z_w_m[1]:
        return "ceiling"
    if x_w > TRUE_SAFE_BOUNDS.x_w_m[1]:
        return "front_wall"
    if x_w < TRUE_SAFE_BOUNDS.x_w_m[0]:
        return "rear_wall"
    if y_w < TRUE_SAFE_BOUNDS.y_w_m[0]:
        return "right_or_low_y_wall"
    if y_w > TRUE_SAFE_BOUNDS.y_w_m[1]:
        return "left_or_high_y_wall"
    return "outside_true_safe"


def empirical_fit_diagnostics(
    valid_rows: list[dict[str, Any]],
    *,
    heldout_count: int,
    heldout_seed: int,
    heldout_indices: set[int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    usable_indices = [
        index
        for index, row in enumerate(valid_rows)
        if all(math.isfinite(_to_float(row.get(feature))) for feature in EMPIRICAL_FEATURES)
        and all(math.isfinite(_to_float(row.get(target))) for target in EMPIRICAL_TARGETS)
    ]
    if len(usable_indices) < 8:
        return [], []

    if heldout_indices is None:
        rng = random.Random(int(heldout_seed))
        shuffled = list(usable_indices)
        rng.shuffle(shuffled)
        n_heldout = min(max(0, int(heldout_count)), max(1, len(shuffled) - 4))
        heldout = set(shuffled[:n_heldout])
    else:
        heldout = set(index for index in heldout_indices if index in set(usable_indices))
        if not heldout:
            return [], []
    train = [index for index in usable_indices if index not in heldout]

    predictions: dict[str, dict[int, float]] = {}
    fit_rows: list[dict[str, Any]] = []
    for target in EMPIRICAL_TARGETS:
        model = _fit_ridge_model(valid_rows, train, target)
        if model is None:
            continue
        target_predictions: dict[int, float] = {}
        for index in usable_indices:
            target_predictions[index] = _predict_ridge_model(model, valid_rows[index])
        predictions[target] = target_predictions

        train_errors = [
            abs(_to_float(valid_rows[index].get(target)) - target_predictions[index])
            for index in train
            if index in target_predictions
        ]
        heldout_errors = [
            abs(_to_float(valid_rows[index].get(target)) - target_predictions[index])
            for index in heldout
            if index in target_predictions
        ]
        for feature, coefficient in zip(EMPIRICAL_FEATURES, model["coef"]):
            fit_rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "standardized_coefficient": coefficient,
                    "intercept": model["intercept"],
                    "train_count": len(train),
                    "heldout_count": len(heldout),
                    "train_mae": _safe_mean(train_errors),
                    "heldout_mae": _safe_mean(heldout_errors),
                }
            )

    validation_rows: list[dict[str, Any]] = []
    required_predictions = {"dx_m", "dy_m", "sink_rate_m_s", "duration_s"}
    if not required_predictions.issubset(predictions):
        return fit_rows, validation_rows
    for index in usable_indices:
        row = valid_rows[index]
        dx_pred = predictions["dx_m"][index]
        dy_pred = predictions["dy_m"][index]
        sink_pred = predictions["sink_rate_m_s"][index]
        duration_pred = predictions["duration_s"][index]
        altitude_loss_pred = sink_pred * duration_pred
        glide_pred = _ratio(dx_pred, altitude_loss_pred)
        actual_glide = _to_float(row.get("glide_ratio_x_over_altloss"))
        actual_altitude_loss = _to_float(row.get("altitude_loss_m"))
        validation_rows.append(
            {
                "split": "heldout" if index in heldout else "train",
                "session_label": row.get("session_label", ""),
                "calibration_profile_hash": row.get("calibration_profile_hash", ""),
                "termination_group": row.get("termination_group", ""),
                "throw_id": row.get("throw_id", ""),
                "dx_m": row.get("dx_m", ""),
                "dx_pred_m": dx_pred,
                "dx_residual_m": _to_float(row.get("dx_m")) - dx_pred,
                "dy_m": row.get("dy_m", ""),
                "dy_pred_m": dy_pred,
                "dy_residual_m": _to_float(row.get("dy_m")) - dy_pred,
                "sink_rate_m_s": row.get("sink_rate_m_s", ""),
                "sink_rate_pred_m_s": sink_pred,
                "sink_rate_residual_m_s": _to_float(row.get("sink_rate_m_s")) - sink_pred,
                "duration_s": row.get("duration_s", ""),
                "duration_pred_s": duration_pred,
                "duration_residual_s": _to_float(row.get("duration_s")) - duration_pred,
                "altitude_loss_m": actual_altitude_loss,
                "altitude_loss_pred_m": altitude_loss_pred,
                "altitude_loss_residual_m": actual_altitude_loss - altitude_loss_pred,
                "glide_ratio_x_over_altloss": actual_glide,
                "derived_glide_ratio_pred": glide_pred,
                "derived_glide_ratio_residual": actual_glide - glide_pred,
            }
        )
    return fit_rows, validation_rows


def _fit_ridge_model(rows: list[dict[str, Any]], indices: list[int], target: str) -> dict[str, Any] | None:
    if len(indices) < 4:
        return None
    x_raw = np.array([[_to_float(rows[index].get(feature)) for feature in EMPIRICAL_FEATURES] for index in indices])
    y = np.array([_to_float(rows[index].get(target)) for index in indices])
    mask = np.isfinite(x_raw).all(axis=1) & np.isfinite(y)
    x_raw = x_raw[mask]
    y = y[mask]
    if x_raw.shape[0] < 4:
        return None
    mean_x = x_raw.mean(axis=0)
    std_x = x_raw.std(axis=0)
    std_x[std_x < 1e-9] = 1.0
    x = (x_raw - mean_x) / std_x
    design = np.column_stack([np.ones(x.shape[0]), x])
    penalty = np.eye(design.shape[1]) * RIDGE_ALPHA
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return {
        "target": target,
        "mean_x": mean_x,
        "std_x": std_x,
        "intercept": float(beta[0]),
        "coef": [float(value) for value in beta[1:]],
    }


def _predict_ridge_model(model: dict[str, Any], row: dict[str, Any]) -> float:
    x_raw = np.array([_to_float(row.get(feature)) for feature in EMPIRICAL_FEATURES])
    x = (x_raw - model["mean_x"]) / model["std_x"]
    return float(model["intercept"] + np.dot(np.array(model["coef"]), x))


def write_report(
    report_path: Path,
    session_roots: list[Path],
    output_root: Path,
    valid_rows: list[dict[str, Any]],
    invalid_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    empirical_validation_rows: list[dict[str, Any]],
    measured_replay_rows: list[dict[str, Any]],
    aligned_replay_rows: list[dict[str, Any]],
    alignment_window_s: float,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    termination_counts: dict[str, int] = {}
    for row in valid_rows:
        key = str(row.get("termination_reason", ""))
        termination_counts[key] = termination_counts.get(key, 0) + 1

    def agg(metric: str, key: str = "mean") -> float:
        for row in aggregate_rows:
            if row["metric"] == metric:
                return _to_float(row.get(key))
        return float("nan")

    lines = [
        "# Real Glider Calibration Prep Report",
        "",
        f"- source session count: `{len(session_roots)}`",
        f"- source sessions: `{[root.as_posix() for root in session_roots]}`",
        f"- output root: `{output_root.as_posix()}`",
        f"- generated: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- valid throws: `{len(valid_rows)}`",
        f"- invalid launch attempts: `{len(invalid_rows)}`",
        f"- termination counts: `{termination_counts}`",
        "",
        "## Calibration Data Targets",
        "",
        "These values are evidence targets for later grey-box fitting. They do not by themselves update the simulator.",
        "The empirical diagnostic fits `dx`, `dy`, `sink_rate`, and `duration` from launch-conditioned features; glide ratio is reported only as a derived residual from predicted distance and altitude loss.",
        "",
        f"- mean sink rate: `{_format_value(agg('sink_rate_m_s'))}` m/s",
        f"- mean derived x/altitude-loss glide ratio: `{_format_value(agg('glide_ratio_x_over_altloss'))}`",
        f"- mean launch speed: `{_format_value(agg('launch_speed_m_s'))}` m/s",
        f"- mean lateral displacement: `{_format_value(agg('dy_m'))}` m",
        f"- mean launch lateral offset: `{_format_value(agg('launch_y_offset_m'))}` m",
        f"- mean launch height offset: `{_format_value(agg('launch_z_offset_m'))}` m",
        f"- mean rate-estimator confidence: `{_format_value(agg('mean_rate_confidence'))}`",
        f"- mean spike-downweighted fraction: `{_format_value(agg('spike_downweighted_fraction'))}`",
        "",
        "## Empirical Held-Out Check",
        "",
        f"- empirical validation rows: `{len(empirical_validation_rows)}`",
        f"- held-out rows: `{sum(1 for row in empirical_validation_rows if row.get('split') == 'heldout')}`",
        f"- held-out derived glide-ratio MAE: `{_format_value(_heldout_mae(empirical_validation_rows, 'derived_glide_ratio_residual'))}`",
        f"- held-out dx MAE: `{_format_value(_heldout_mae(empirical_validation_rows, 'dx_residual_m'))}` m",
        f"- held-out sink-rate MAE: `{_format_value(_heldout_mae(empirical_validation_rows, 'sink_rate_residual_m_s'))}` m/s",
        "",
        "## Measured-Launch Simulation Replay",
        "",
        "Each valid throw is replayed in the current dry-air simulator from the measured launch-plane state and the logged command history. Residuals are actual minus simulation, so launch variability is not fitted away as model error.",
        "",
        f"- replay rows: `{len(measured_replay_rows)}`",
        f"- successful replay rows: `{sum(1 for row in measured_replay_rows if row.get('replay_status') == 'ok')}`",
        f"- mean replay dx residual: `{_format_value(_replay_mean(measured_replay_rows, 'dx_residual_actual_minus_sim_m'))}` m",
        f"- mean replay dy residual: `{_format_value(_replay_mean(measured_replay_rows, 'dy_residual_actual_minus_sim_m'))}` m",
        f"- mean replay altitude-loss residual: `{_format_value(_replay_mean(measured_replay_rows, 'altitude_loss_residual_actual_minus_sim_m'))}` m",
        f"- mean replay sink-rate residual: `{_format_value(_replay_mean(measured_replay_rows, 'sink_rate_residual_actual_minus_sim_m_s'))}` m/s",
        f"- replay dy residual MAE: `{_format_value(_replay_mae(measured_replay_rows, 'dy_residual_actual_minus_sim_m'))}` m",
        "",
        "## First-Motion-Aligned Simulation Replay",
        "",
        "Each valid throw is also replayed from a state aligned to the first short segment of measured motion. This diagnostic uses the target pose after the alignment window, regresses world velocity over the window, estimates body rate from the SO(3) rotation change, then predicts only the remaining trajectory.",
        "",
        f"- alignment window: `{alignment_window_s}` s",
        f"- aligned replay rows: `{len(aligned_replay_rows)}`",
        f"- successful aligned replay rows: `{sum(1 for row in aligned_replay_rows if row.get('replay_status') == 'ok')}`",
        f"- mean aligned dx residual: `{_format_value(_replay_mean(aligned_replay_rows, 'dx_residual_actual_minus_sim_m'))}` m",
        f"- mean aligned dy residual: `{_format_value(_replay_mean(aligned_replay_rows, 'dy_residual_actual_minus_sim_m'))}` m",
        f"- mean aligned altitude-loss residual: `{_format_value(_replay_mean(aligned_replay_rows, 'altitude_loss_residual_actual_minus_sim_m'))}` m",
        f"- mean aligned sink-rate residual: `{_format_value(_replay_mean(aligned_replay_rows, 'sink_rate_residual_actual_minus_sim_m_s'))}` m/s",
        f"- aligned dy residual MAE: `{_format_value(_replay_mae(aligned_replay_rows, 'dy_residual_actual_minus_sim_m'))}` m",
        "",
        "## Current Dry-Air Model Calibration",
        "",
        f"- calibration active: `{CURRENT_MODEL_CALIBRATION['neutral_dry_air_calibration_active']}`",
        f"- calibration id: `{CURRENT_MODEL_CALIBRATION['neutral_dry_air_calibration_id']}`",
        f"- source prep run: `{CURRENT_MODEL_CALIBRATION['source_prep_run']}`",
        f"- source throw count: `{CURRENT_MODEL_CALIBRATION['source_throw_count']}`",
        f"- held-out policy: `{CURRENT_MODEL_CALIBRATION['heldout_policy']}`",
        f"- cd0 strip scale: `{CURRENT_MODEL_CALIBRATION['cd0_strip_scale']}`",
        f"- fuselage drag-area scale: `{CURRENT_MODEL_CALIBRATION['drag_area_fuse_scale']}`",
        f"- strip efficiency scale: `{CURRENT_MODEL_CALIBRATION['efficiency_strip_scale']}`",
        f"- roll moment bias coefficient: `{CURRENT_MODEL_CALIBRATION['roll_moment_bias_coeff']}`",
        f"- yaw moment bias coefficient: `{CURRENT_MODEL_CALIBRATION['yaw_moment_bias_coeff']}`",
        f"- aileron neutral trim: `{CURRENT_MODEL_CALIBRATION['delta_a_trim_rad']}` rad",
        f"- elevator neutral trim: `{CURRENT_MODEL_CALIBRATION['delta_e_trim_rad']}` rad",
        f"- rudder neutral trim: `{CURRENT_MODEL_CALIBRATION['delta_r_trim_rad']}` rad",
        "",
        "## Recommended Calibration Order",
        "",
        "1. Use measured-launch replay residuals as the primary fair SysID target; do not compare real throws to a nominal launch.",
        "2. Fit bare-airframe trim/polar consistency first: distance, sink rate, duration, and pitch tendency from neutral throws.",
        "3. Check whether lateral residual remains after measured launch conditioning before assigning it to rudder/aileron trim, wing asymmetry, or y-CG offset.",
        "4. Use pulse-ladder throws only after the neutral fit is stable, fitting control effectiveness and damping separately.",
        "5. Regenerate R5/R7/R8/R10/R11 only after the model update is fixed and documented.",
        "6. Inspect physical neutral trim before assigning residuals to aerodynamic coefficients: rudder/aileron zero, wing/tail asymmetry, CG yaw/roll bias, and elevator trim.",
        "",
        "## Files Written",
        "",
        "- `metrics/neutral_throw_summary.csv`",
        "- `metrics/neutral_feature_target_table.csv`",
        "- `metrics/session_termination_summary.csv`",
        "- `metrics/neutral_aggregate_summary.csv`",
        "- `metrics/empirical_fit_coefficients.csv`",
        "- `metrics/empirical_heldout_validation.csv`",
        "- `metrics/measured_launch_replay_residuals.csv`",
        "- `metrics/aligned_motion_replay_residuals.csv`",
        "- `metrics/invalid_attempt_summary.csv`",
        "- `manifests/calibration_prep_manifest.json`",
        "",
        "## Claims Not Made",
        "",
        "- No controller/library evidence was regenerated.",
        "- The current aerodynamic correction is neutral dry-air only; pulse/control-effectiveness fitting is still separate.",
        "- No zero-shot transfer claim is made from this prep report alone.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _heldout_mae(rows: list[dict[str, Any]], residual_key: str) -> float:
    values = [
        abs(_to_float(row.get(residual_key)))
        for row in rows
        if row.get("split") == "heldout" and math.isfinite(_to_float(row.get(residual_key)))
    ]
    return _safe_mean(values)


def _replay_mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [
        _to_float(row.get(key))
        for row in rows
        if row.get("replay_status") == "ok" and math.isfinite(_to_float(row.get(key)))
    ]
    return _safe_mean(values)


def _replay_mae(rows: list[dict[str, Any]], key: str) -> float:
    values = [
        abs(_to_float(row.get(key)))
        for row in rows
        if row.get("replay_status") == "ok" and math.isfinite(_to_float(row.get(key)))
    ]
    return _safe_mean(values)


def run_calibration_prep(
    session_root: Path,
    output_root: Path,
    run_label: str | None,
    *,
    heldout_count: int = DEFAULT_HELDOUT_COUNT,
    heldout_seed: int = DEFAULT_HELDOUT_SEED,
    replay_dt_s: float = DEFAULT_REPLAY_DT_S,
    alignment_window_s: float = DEFAULT_ALIGNMENT_WINDOW_S,
    workers: int = DEFAULT_WORKERS,
) -> Path:
    session_roots = resolve_session_roots(session_root)
    label = run_label or f"calibration_{len(session_roots)}sessions_prep"
    output_dir = output_root / label

    valid_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    for root in session_roots:
        session_label = root.name
        valid_rows.extend(summarize_valid_throw(session_label, root, throw_dir) for throw_dir in _valid_throw_dirs(root))
        invalid_rows.extend(summarize_invalid_attempt(session_label, attempt_dir) for attempt_dir in _invalid_attempt_dirs(root))
    aggregate_rows = aggregate_valid_metrics(valid_rows)
    feature_target_rows = build_feature_target_rows(valid_rows)
    session_summary_rows = session_termination_summary(valid_rows)
    heldout_indices = stratified_heldout_indices(
        valid_rows,
        heldout_count=heldout_count,
        heldout_seed=heldout_seed,
        group_key="session_label",
    )
    fit_rows, empirical_validation_rows = empirical_fit_diagnostics(
        valid_rows,
        heldout_count=heldout_count,
        heldout_seed=heldout_seed,
        heldout_indices=heldout_indices,
    )
    measured_replay_rows = measured_launch_replay_diagnostics(
        valid_rows,
        replay_dt_s=replay_dt_s,
        workers=workers,
    )
    aligned_replay_rows = aligned_motion_replay_diagnostics(
        valid_rows,
        replay_dt_s=replay_dt_s,
        alignment_window_s=alignment_window_s,
        workers=workers,
    )
    for index, row in enumerate(measured_replay_rows):
        row["split"] = "heldout" if index in heldout_indices else "train"
    for index, row in enumerate(aligned_replay_rows):
        row["split"] = "heldout" if index in heldout_indices else "train"

    _write_csv(output_dir / "metrics" / "neutral_throw_summary.csv", valid_rows, VALID_THROW_FIELDS)
    _write_csv(output_dir / "metrics" / "neutral_feature_target_table.csv", feature_target_rows, FEATURE_TARGET_FIELDS)
    _write_csv(output_dir / "metrics" / "session_termination_summary.csv", session_summary_rows, SESSION_TERMINATION_SUMMARY_FIELDS)
    _write_csv(output_dir / "metrics" / "invalid_attempt_summary.csv", invalid_rows, INVALID_ATTEMPT_FIELDS)
    _write_csv(output_dir / "metrics" / "neutral_aggregate_summary.csv", aggregate_rows, AGGREGATE_FIELDS)
    _write_csv(output_dir / "metrics" / "empirical_fit_coefficients.csv", fit_rows, EMPIRICAL_FIT_SUMMARY_FIELDS)
    _write_csv(
        output_dir / "metrics" / "empirical_heldout_validation.csv",
        empirical_validation_rows,
        EMPIRICAL_VALIDATION_FIELDS,
    )
    _write_csv(
        output_dir / "metrics" / "measured_launch_replay_residuals.csv",
        measured_replay_rows,
        MEASURED_LAUNCH_REPLAY_FIELDS,
    )
    _write_csv(
        output_dir / "metrics" / "aligned_motion_replay_residuals.csv",
        aligned_replay_rows,
        ALIGNED_MOTION_REPLAY_FIELDS,
    )

    manifest = {
        "source_session_roots": [root.as_posix() for root in session_roots],
        "output_dir": output_dir.as_posix(),
        "session_count": len(session_roots),
        "valid_throw_count": len(valid_rows),
        "invalid_attempt_count": len(invalid_rows),
        "heldout_count": int(heldout_count),
        "heldout_seed": int(heldout_seed),
        "heldout_selection_policy": "randomised_stratified_by_session_label",
        "heldout_throw_keys": [
            {
                "index": int(index),
                "session_label": str(valid_rows[index].get("session_label", "")),
                "throw_id": str(valid_rows[index].get("throw_id", "")),
                "case_id": str(valid_rows[index].get("case_id", "")),
            }
            for index in sorted(heldout_indices)
        ],
        "measured_launch_replay_dt_s": float(replay_dt_s),
        "measured_launch_replay_workers": int(workers),
        "measured_launch_replay_policy": "dry_air_current_model_exact_measured_launch_state_same_command_history_same_duration",
        "aligned_motion_replay_alignment_window_s": float(alignment_window_s),
        "aligned_motion_replay_policy": "dry_air_current_model_aligned_first_motion_window_same_command_history_remaining_duration",
        "current_model_calibration": dict(CURRENT_MODEL_CALIBRATION),
        "empirical_feature_names": list(EMPIRICAL_FEATURES),
        "empirical_target_names": list(EMPIRICAL_TARGETS),
        "derived_glide_ratio_policy": "computed_from_dx_pred_divided_by_sink_rate_pred_times_duration_pred",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "calibration_stage": "offline_current_model_replay_no_controller_library_regeneration",
    }
    manifest_path = output_dir / "manifests" / "calibration_prep_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    write_report(
        output_dir / "reports" / "neutral_glide_calibration_prep_report.md",
        session_roots,
        output_dir,
        valid_rows,
        invalid_rows,
        aggregate_rows,
        empirical_validation_rows,
        measured_replay_rows,
        aligned_replay_rows,
        alignment_window_s,
    )
    return output_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare offline real-glider calibration summaries from completed 04_Flight_Test logs."
    )
    parser.add_argument(
        "--session-root",
        type=Path,
        default=DEFAULT_SESSION_SEARCH_ROOT,
        help=(
            "Completed session root, or a directory containing session roots. Defaults to searching "
            "04_Flight_Test/05_Results for all completed glider-calibration sessions with valid throws."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output directory for compact calibration-prep evidence.",
    )
    parser.add_argument(
        "--run-label",
        default=None,
        help="Optional output run label. Defaults to 'calibration_<session_count>sessions_prep'.",
    )
    parser.add_argument(
        "--heldout-count",
        type=int,
        default=DEFAULT_HELDOUT_COUNT,
        help="Number of calibration throws to reserve for empirical held-out residual diagnostics.",
    )
    parser.add_argument(
        "--heldout-seed",
        type=int,
        default=DEFAULT_HELDOUT_SEED,
        help="Deterministic seed for empirical held-out selection.",
    )
    parser.add_argument(
        "--replay-dt-s",
        type=float,
        default=DEFAULT_REPLAY_DT_S,
        help="Fixed RK4 step for measured-launch dry-air simulation replay.",
    )
    parser.add_argument(
        "--alignment-window-s",
        type=float,
        default=DEFAULT_ALIGNMENT_WINDOW_S,
        help="First-motion alignment window for the additional aligned replay diagnostic.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Parallel workers for replay diagnostics. Defaults to 8.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = run_calibration_prep(
        args.session_root,
        args.output_root,
        args.run_label,
        heldout_count=args.heldout_count,
        heldout_seed=args.heldout_seed,
        replay_dt_s=args.replay_dt_s,
        alignment_window_s=args.alignment_window_s,
        workers=args.workers,
    )
    print(f"[DONE] calibration prep written to {output_dir}")


if __name__ == "__main__":
    main()
