"""Neutral-only aero residual identification from Vicon trajectories.

Pipeline:
    Vicon trajectory -> force/moment residuals -> regime-split coefficient fit
    -> held-out replay validation.

Pulse/control-effectiveness throws are intentionally excluded.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (
    REPO_ROOT / "04_Flight_Test" / "01_Runtime",
    REPO_ROOT / "03_Control" / "03_Primitives",
    REPO_ROOT / "03_Control" / "04_Scenarios",
    Path(__file__).resolve().parent,
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_fit_neutral_dry_air_calibration as replay_fit  # noqa: E402
import run_real_glider_calibration_prep as prep  # noqa: E402
from A_model_parameters import neutral_dry_air_calibration as active_calibration  # noqa: E402
from flight_dynamics import STALL_BLEND_ALPHA_RAD, evaluate_state, post_stall_residual_activation_numpy  # noqa: E402


FIT_VERSION = "N16_full_6dof_alpha_rbf_surface_residual_fit"
DEFAULT_SESSION_ROOT = REPO_ROOT / "04_Flight_Test" / "05_Results" / "cal" / "n30"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "03_Control" / "05_Results" / "glider_model_calibration_prep"
DEFAULT_RUN_LABEL = "n30_neutral_aero_residual_fit"
DEFAULT_WORKERS = 8
DEFAULT_HELDOUT_COUNT = 10
DEFAULT_HELDOUT_SEED = 606
DEFAULT_ALIGNMENT_WINDOW_S = 0.10
DEFAULT_DERIVATIVE_WINDOW_S = 0.040
DEFAULT_REPLAY_DT_S = 0.005
DEFAULT_RIDGE_LAMBDA = 1.0e-3
DEFAULT_MIN_SPEED_M_S = 1.5
RHO_KG_M3 = 1.225
STALL_ALPHA_DEG = float(math.degrees(STALL_BLEND_ALPHA_RAD))
POST_STALL_ALPHA_DEG = 20.0
SURFACE_SCALE_CANDIDATES = (0.0, 0.25, 0.5, 0.75, 1.0)
SURFACE_RBF_ALPHA_CENTERS_DEG = tuple(
    float(value) for value in getattr(active_calibration, "POST_STALL_RBF_ALPHA_CENTERS_DEG", (20.0, 45.0, 70.0))
)
SURFACE_RBF_ALPHA_WIDTH_DEG = float(getattr(active_calibration, "POST_STALL_RBF_ALPHA_WIDTH_DEG", 15.0))
LATERAL_SURFACE_FEATURES = ("bias", "beta", "p_hat", "r_hat")
LATERAL_SURFACE_PREFIXES = (
    "post_stall_side_force",
    "post_stall_roll_moment",
    "post_stall_yaw_moment",
)


AERO_RESIDUAL_FIELDS = [
    "split",
    "session_label",
    "case_id",
    "throw_id",
    "residual_status",
    "sample_index",
    "t_since_alignment_s",
    "regime",
    "stage_fit_group",
    "post_stall_seen_before_sample",
    "speed_m_s",
    "q_bar_pa",
    "alpha_deg",
    "beta_deg",
    "q_rad_s",
    "q_hat",
    "p_rad_s",
    "p_hat",
    "r_rad_s",
    "r_hat",
    "post_stall_activation",
    "theta_deg",
    "z_m",
    "cx_required",
    "cx_model",
    "cx_residual",
    "cz_required",
    "cz_model",
    "cz_residual",
    "cl_required",
    "cl_model",
    "cl_residual",
    "cd_required",
    "cd_model",
    "cd_residual",
    "cy_required",
    "cy_model",
    "cy_residual",
    "cm_required",
    "cm_model",
    "cm_residual",
    "cl_roll_required",
    "cl_roll_model",
    "cl_roll_residual",
    "cn_yaw_required",
    "cn_yaw_model",
    "cn_yaw_residual",
    "pitch_moment_required_n_m",
    "pitch_moment_model_n_m",
    "pitch_moment_residual_n_m",
    "force_residual_norm_n",
    "angular_accel_q_rad_s2",
]
REGIME_SUMMARY_FIELDS = [
    "split",
    "regime",
    "count",
    "alpha_min_deg",
    "alpha_max_deg",
    "cm_residual_mean",
    "cm_residual_mae",
    "cm_fit_residual_mae",
    "cd_residual_mean",
    "cl_residual_mean",
    "cy_residual_mean",
    "cl_roll_residual_mean",
    "cn_yaw_residual_mean",
    "q_hat_mean",
]
STAGE_FIT_SUMMARY_FIELDS = [
    "split",
    "stage_fit_group",
    "count",
    "alpha_min_deg",
    "alpha_max_deg",
    "cm_residual_mean",
    "cm_residual_mae",
    "cm_fit_residual_mae",
    "cd_residual_mean",
    "cl_residual_mean",
    "cy_residual_mean",
    "cl_roll_residual_mean",
    "cn_yaw_residual_mean",
    "q_hat_mean",
]
COEFFICIENT_FIELDS = ["parameter", "value", "applied_to_replay", "description"]
SURFACE_SCALE_SELECTION_FIELDS = [
    "surface_scale",
    "selected",
    "objective",
    "dx_mae_m",
    "dy_mae_m",
    "altitude_loss_mae_m",
    "sink_rate_mae_m_s",
    "final_theta_mae_deg",
    "final_phi_mae_deg",
    "final_psi_mae_deg",
]
REPLAY_VALIDATION_FIELDS = ["model_id", *replay_fit.REPLAY_RESIDUAL_FIELDS]
STAGE_REPLAY_SUMMARY_FIELDS = [
    "model_id",
    "split",
    "regime",
    "sample_count",
    "throw_count",
    "dx_mae_m",
    "dy_mae_m",
    "altitude_loss_mae_m",
    "sink_rate_mae_m_s",
    "roll_mae_deg",
    "pitch_mae_deg",
    "yaw_mae_deg",
]


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = run_fit(
        session_root=args.session_root,
        output_root=args.output_root,
        run_label=args.run_label,
        heldout_count=args.heldout_count,
        heldout_seed=args.heldout_seed,
        alignment_window_s=args.alignment_window_s,
        derivative_window_s=args.derivative_window_s,
        replay_dt_s=args.replay_dt_s,
        ridge_lambda=args.ridge_lambda,
        min_speed_m_s=args.min_speed_m_s,
        workers=args.workers,
        apply_attached_cm_bias=args.apply_attached_cm_bias,
        fit_post_stall_damping=args.fit_post_stall_damping,
        fit_lateral_surfaces=args.fit_lateral_surfaces,
    )
    print(f"[DONE] neutral aero residual fit written to {output_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit neutral-launch aero residual coefficients from Vicon trajectory residuals.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--session-root", type=Path, default=DEFAULT_SESSION_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-label", default=DEFAULT_RUN_LABEL)
    parser.add_argument("--heldout-count", type=int, default=DEFAULT_HELDOUT_COUNT)
    parser.add_argument("--heldout-seed", type=int, default=DEFAULT_HELDOUT_SEED)
    parser.add_argument("--alignment-window-s", type=float, default=DEFAULT_ALIGNMENT_WINDOW_S)
    parser.add_argument("--derivative-window-s", type=float, default=DEFAULT_DERIVATIVE_WINDOW_S)
    parser.add_argument("--replay-dt-s", type=float, default=DEFAULT_REPLAY_DT_S)
    parser.add_argument("--ridge-lambda", type=float, default=DEFAULT_RIDGE_LAMBDA)
    parser.add_argument("--min-speed-m-s", type=float, default=DEFAULT_MIN_SPEED_M_S)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--apply-attached-cm-bias",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply attached-regime Cm residual as Cm0. Disabled by default so high-AoA launch data cannot move normal-flight Cm0.",
    )
    parser.add_argument(
        "--fit-post-stall-damping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fit and apply alpha-dependent post-stall Cmq residuals. Keep disabled only for static-surface ablations.",
    )
    parser.add_argument(
        "--fit-lateral-surfaces",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fit and apply post-stall CY/Cl/Cn residual surfaces using beta, p_hat, and r_hat features.",
    )
    return parser


def run_fit(
    *,
    session_root: Path,
    output_root: Path,
    run_label: str,
    heldout_count: int,
    heldout_seed: int,
    alignment_window_s: float,
    derivative_window_s: float,
    replay_dt_s: float,
    ridge_lambda: float,
    min_speed_m_s: float,
    workers: int,
    apply_attached_cm_bias: bool,
    fit_post_stall_damping: bool,
    fit_lateral_surfaces: bool,
) -> Path:
    output_dir = output_root / run_label
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_rows = load_neutral_rows(session_root)
    heldout_indices = prep.stratified_heldout_indices(
        valid_rows,
        heldout_count=heldout_count,
        heldout_seed=heldout_seed,
        group_key="session_label",
    )
    train_rows = [row for index, row in enumerate(valid_rows) if index not in heldout_indices]
    heldout_rows = [row for index, row in enumerate(valid_rows) if index in heldout_indices]
    base_parameters = active_parameter_dict()

    train_residuals = residual_rows(
        train_rows,
        split="train",
        parameters=base_parameters,
        alignment_window_s=alignment_window_s,
        derivative_window_s=derivative_window_s,
        min_speed_m_s=min_speed_m_s,
        workers=workers,
    )
    heldout_residuals = residual_rows(
        heldout_rows,
        split="heldout",
        parameters=base_parameters,
        alignment_window_s=alignment_window_s,
        derivative_window_s=derivative_window_s,
        min_speed_m_s=min_speed_m_s,
        workers=workers,
    )
    fit_result = fit_pitch_residual_coefficients(
        train_residuals,
        ridge_lambda=ridge_lambda,
        fit_post_stall_damping=fit_post_stall_damping,
        fit_lateral_surfaces=fit_lateral_surfaces,
    )
    surface_scale_rows = select_surface_scale_rows(
        base_parameters=base_parameters,
        fit_result=fit_result,
        train_rows=train_rows,
        replay_dt_s=replay_dt_s,
        alignment_window_s=alignment_window_s,
        workers=workers,
        apply_attached_cm_bias=apply_attached_cm_bias,
        fit_post_stall_damping=fit_post_stall_damping,
        fit_lateral_surfaces=fit_lateral_surfaces,
    )
    selected_surface_scale = selected_surface_scale_from_rows(surface_scale_rows)
    fit_result["surface_scale_selection"] = {
        "candidate_scales": list(SURFACE_SCALE_CANDIDATES),
        "selected_surface_scale": float(selected_surface_scale),
        "selection_metric": "train_replay_combined_objective",
    }
    candidate_parameters = candidate_from_fit(
        base_parameters,
        fit_result,
        apply_attached_cm_bias=apply_attached_cm_bias,
        fit_post_stall_damping=fit_post_stall_damping,
        fit_lateral_surfaces=fit_lateral_surfaces,
        surface_scale=selected_surface_scale,
    )
    validation_rows = replay_validation_rows(
        train_rows=train_rows,
        heldout_rows=heldout_rows,
        base_parameters=base_parameters,
        candidate_parameters=candidate_parameters,
        replay_dt_s=replay_dt_s,
        alignment_window_s=alignment_window_s,
        workers=workers,
    )
    stage_replay_rows = stage_replay_summary_rows(
        train_rows=train_rows,
        heldout_rows=heldout_rows,
        base_parameters=base_parameters,
        candidate_parameters=candidate_parameters,
        replay_dt_s=replay_dt_s,
        alignment_window_s=alignment_window_s,
        workers=workers,
    )
    all_residuals = train_residuals + heldout_residuals
    regime_summary = summarize_regimes(all_residuals, fit_result)
    stage_fit_summary = summarize_stage_fit_groups(all_residuals, fit_result)
    coefficient_rows = coefficient_output_rows(
        fit_result,
        apply_attached_cm_bias,
        fit_post_stall_damping,
        fit_lateral_surfaces,
    )

    write_csv(output_dir / "metrics" / "neutral_aero_residual_samples.csv", all_residuals, AERO_RESIDUAL_FIELDS)
    write_csv(output_dir / "metrics" / "neutral_aero_residual_regime_summary.csv", regime_summary, REGIME_SUMMARY_FIELDS)
    write_csv(output_dir / "metrics" / "neutral_aero_residual_stage_fit_summary.csv", stage_fit_summary, STAGE_FIT_SUMMARY_FIELDS)
    write_csv(output_dir / "metrics" / "neutral_aero_residual_fit_coefficients.csv", coefficient_rows, COEFFICIENT_FIELDS)
    write_csv(output_dir / "metrics" / "neutral_aero_residual_surface_scale_selection.csv", surface_scale_rows, SURFACE_SCALE_SELECTION_FIELDS)
    write_csv(output_dir / "metrics" / "neutral_aero_residual_replay_validation.csv", validation_rows, REPLAY_VALIDATION_FIELDS)
    write_csv(output_dir / "metrics" / "neutral_aero_residual_stage_replay_errors.csv", stage_replay_rows, STAGE_REPLAY_SUMMARY_FIELDS)
    write_manifest(
        output_dir,
        run_label=run_label,
        session_root=session_root,
        valid_rows=valid_rows,
        heldout_indices=heldout_indices,
        heldout_seed=heldout_seed,
        alignment_window_s=alignment_window_s,
        derivative_window_s=derivative_window_s,
        replay_dt_s=replay_dt_s,
        ridge_lambda=ridge_lambda,
        min_speed_m_s=min_speed_m_s,
        workers=workers,
        apply_attached_cm_bias=apply_attached_cm_bias,
        fit_post_stall_damping=fit_post_stall_damping,
        fit_lateral_surfaces=fit_lateral_surfaces,
        base_parameters=base_parameters,
        candidate_parameters=candidate_parameters,
        fit_result=fit_result,
    )
    write_report(
        output_dir,
        run_label=run_label,
        session_root=session_root,
        heldout_count=heldout_count,
        heldout_seed=heldout_seed,
        alignment_window_s=alignment_window_s,
        derivative_window_s=derivative_window_s,
        replay_dt_s=replay_dt_s,
        ridge_lambda=ridge_lambda,
        min_speed_m_s=min_speed_m_s,
        workers=workers,
        apply_attached_cm_bias=apply_attached_cm_bias,
        fit_post_stall_damping=fit_post_stall_damping,
        fit_lateral_surfaces=fit_lateral_surfaces,
        fit_result=fit_result,
        regime_summary=regime_summary,
        stage_fit_summary=stage_fit_summary,
        validation_rows=validation_rows,
        stage_replay_rows=stage_replay_rows,
        base_parameters=base_parameters,
        candidate_parameters=candidate_parameters,
    )
    return output_dir


def active_parameter_dict() -> dict[str, float]:
    parameters = {
        "cd0_strip_scale": float(active_calibration.CD0_STRIP_SCALE),
        "drag_area_fuse_scale": float(active_calibration.DRAG_AREA_FUSE_SCALE),
        "efficiency_strip_scale": float(active_calibration.EFFICIENCY_STRIP_SCALE),
        "roll_moment_bias_coeff": float(getattr(active_calibration, "ROLL_MOMENT_BIAS_COEFF", 0.0)),
        "pitch_moment_bias_coeff": float(getattr(active_calibration, "PITCH_MOMENT_BIAS_COEFF", 0.0)),
        "post_stall_lift_residual_coeff": float(getattr(active_calibration, "POST_STALL_LIFT_RESIDUAL_COEFF", 0.0)),
        "post_stall_drag_residual_coeff": float(getattr(active_calibration, "POST_STALL_DRAG_RESIDUAL_COEFF", 0.0)),
        "post_stall_pitch_moment_coeff": float(getattr(active_calibration, "POST_STALL_PITCH_MOMENT_COEFF", 0.0)),
        "post_stall_pitch_damping_coeff": float(getattr(active_calibration, "POST_STALL_PITCH_DAMPING_COEFF", 0.0)),
        "post_stall_residual_blend_start_alpha_deg": float(
            getattr(active_calibration, "POST_STALL_RESIDUAL_BLEND_START_ALPHA_DEG", STALL_ALPHA_DEG)
        ),
        "post_stall_residual_blend_full_alpha_deg": float(
            getattr(active_calibration, "POST_STALL_RESIDUAL_BLEND_FULL_ALPHA_DEG", POST_STALL_ALPHA_DEG)
        ),
        "post_stall_tail_effectiveness_drop": float(getattr(active_calibration, "POST_STALL_TAIL_EFFECTIVENESS_DROP", 0.0)),
        "yaw_moment_bias_coeff": float(getattr(active_calibration, "YAW_MOMENT_BIAS_COEFF", 0.0)),
        "delta_a_trim_rad": float(getattr(active_calibration, "DELTA_A_TRIM_RAD", 0.0)),
        "delta_e_trim_rad": float(getattr(active_calibration, "DELTA_E_TRIM_RAD", 0.0)),
        "delta_r_trim_rad": float(getattr(active_calibration, "DELTA_R_TRIM_RAD", 0.0)),
    }
    for prefix, values in (
        ("post_stall_lift_rbf", getattr(active_calibration, "POST_STALL_LIFT_RBF_COEFFS", (0.0, 0.0, 0.0))),
        ("post_stall_drag_rbf", getattr(active_calibration, "POST_STALL_DRAG_RBF_COEFFS", (0.0, 0.0, 0.0))),
        (
            "post_stall_pitch_moment_rbf",
            getattr(active_calibration, "POST_STALL_PITCH_MOMENT_RBF_COEFFS", (0.0, 0.0, 0.0)),
        ),
        (
            "post_stall_pitch_damping_rbf",
            getattr(active_calibration, "POST_STALL_PITCH_DAMPING_RBF_COEFFS", (0.0, 0.0, 0.0)),
        ),
    ):
        value_list = list(values)
        for index, centre_deg in enumerate(SURFACE_RBF_ALPHA_CENTERS_DEG):
            parameters[surface_rbf_parameter_name(prefix, centre_deg)] = float(value_list[index]) if index < len(value_list) else 0.0
    for prefix, values in (
        ("post_stall_side_force", getattr(active_calibration, "POST_STALL_SIDE_FORCE_RBF_COEFFS", ())),
        ("post_stall_roll_moment", getattr(active_calibration, "POST_STALL_ROLL_MOMENT_RBF_COEFFS", ())),
        ("post_stall_yaw_moment", getattr(active_calibration, "POST_STALL_YAW_MOMENT_RBF_COEFFS", ())),
    ):
        matrix = np.asarray(values, dtype=float)
        expected_shape = (len(LATERAL_SURFACE_FEATURES), len(SURFACE_RBF_ALPHA_CENTERS_DEG))
        if matrix.shape != expected_shape:
            matrix = np.zeros(expected_shape, dtype=float)
        for feature_index, feature in enumerate(LATERAL_SURFACE_FEATURES):
            for centre_index, centre_deg in enumerate(SURFACE_RBF_ALPHA_CENTERS_DEG):
                parameters[lateral_surface_parameter_name(prefix, feature, centre_deg)] = float(matrix[feature_index, centre_index])
    return parameters


def load_neutral_rows(session_root: Path) -> list[dict[str, Any]]:
    try:
        return replay_fit.load_neutral_open_loop_rows(session_root)
    except FileNotFoundError:
        pass

    roots = [session_root] if (session_root / "c0_neu").exists() else [
        path for path in sorted(session_root.iterdir()) if path.is_dir() and (path / "c0_neu").exists()
    ]
    rows: list[dict[str, Any]] = []
    for root in roots:
        for throw_dir in sorted((root / "c0_neu").glob("v*")):
            if not throw_dir.is_dir():
                continue
            if not (throw_dir / "metrics" / "state_samples.csv").exists():
                continue
            rows.append(
                {
                    "session_label": root.name,
                    "case_id": "C0_neutral",
                    "throw_id": throw_dir.name,
                    "command_axis": "neutral",
                    "_throw_dir": str(throw_dir),
                }
            )
    if len(rows) < 4:
        raise FileNotFoundError(
            f"Need neutral throw folders with metrics/state_samples.csv under {session_root}; found {len(rows)} usable throws."
        )
    return rows


def residual_rows(
    rows: list[dict[str, Any]],
    *,
    split: str,
    parameters: dict[str, float],
    alignment_window_s: float,
    derivative_window_s: float,
    min_speed_m_s: float,
    workers: int,
) -> list[dict[str, Any]]:
    payloads = [(row, split, parameters, alignment_window_s, derivative_window_s, min_speed_m_s) for row in rows]
    if int(workers) > 1 and len(payloads) > 1:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            nested = list(executor.map(residual_payload, payloads))
    else:
        nested = [residual_payload(payload) for payload in payloads]
    return [item for group in nested for item in group]


def residual_payload(payload: tuple[dict[str, Any], str, dict[str, float], float, float, float]) -> list[dict[str, Any]]:
    row, split, parameters, alignment_window_s, derivative_window_s, min_speed_m_s = payload
    throw_dir = Path(str(row.get("_throw_dir", "")))
    if not throw_dir.exists():
        return [blocked_residual_row(row, split, "missing_throw_dir")]
    sample_rows = prep._read_csv(throw_dir / "metrics" / "state_samples.csv")
    if len(sample_rows) < 8:
        return [blocked_residual_row(row, split, "too_few_state_samples")]
    aligned = prep._aligned_state_from_sample_rows(sample_rows, alignment_window_s)
    if aligned.get("status") != "ok":
        return [blocked_residual_row(row, split, str(aligned.get("status", "alignment_failed")))]

    t0 = prep._float(sample_rows[0], "t_s", 0.0)
    alignment_elapsed_s = float(aligned["alignment_elapsed_s"])
    times = np.asarray([prep._float(sample, "t_s", t0) - t0 - alignment_elapsed_s for sample in sample_rows], dtype=float)
    states = np.asarray([prep._state_vector_from_sample_row(sample) for sample in sample_rows], dtype=float)
    indices = [index for index, time_s in enumerate(times) if time_s >= -1e-9]
    if len(indices) < 6:
        return [blocked_residual_row(row, split, "too_few_post_alignment_samples")]

    aircraft = replay_fit.calibrated_aircraft(parameters)
    inertia_b = np.asarray(aircraft.inertia_b, dtype=float)
    neutral_command = np.zeros(3, dtype=float)
    out: list[dict[str, Any]] = []
    post_stall_seen = False
    for sample_index in indices:
        x = states[sample_index]
        if not np.all(np.isfinite(x)):
            continue
        v_dot_b = np.asarray([local_linear_slope(times, states[:, component], sample_index, derivative_window_s) for component in (6, 7, 8)], dtype=float)
        omega_dot_b = np.asarray([local_linear_slope(times, states[:, component], sample_index, derivative_window_s) for component in (9, 10, 11)], dtype=float)
        if not np.all(np.isfinite(v_dot_b)) or not np.all(np.isfinite(omega_dot_b)):
            continue
        loads = evaluate_state(x, neutral_command, aircraft, wind_model=None, rho=RHO_KG_M3, wind_mode="panel")
        speed_m_s = float(loads["speed_m_s"])
        if speed_m_s < float(min_speed_m_s):
            continue
        q_bar = 0.5 * RHO_KG_M3 * speed_m_s**2
        force_denom = q_bar * aircraft.s_ref_m2
        pitch_moment_denom = force_denom * aircraft.c_ref_m
        roll_yaw_moment_denom = force_denom * aircraft.b_ref_m
        if force_denom <= 1e-9 or pitch_moment_denom <= 1e-9 or roll_yaw_moment_denom <= 1e-9:
            continue

        v_b = x[6:9]
        omega_b = x[9:12]
        f_total_required_b = aircraft.mass_kg * (v_dot_b + np.cross(omega_b, v_b))
        f_aero_required_b = f_total_required_b - aircraft.mass_kg * np.asarray(loads["gravity_b"], dtype=float)
        m_aero_required_b = inertia_b @ omega_dot_b + np.cross(omega_b, inertia_b @ omega_b)
        f_model_b = np.asarray(loads["f_aero_b"], dtype=float)
        m_model_b = np.asarray(loads["m_aero_b"], dtype=float)
        f_residual_b = f_aero_required_b - f_model_b
        m_residual_b = m_aero_required_b - m_model_b
        cl_required, cd_required = lift_drag_coefficients(f_aero_required_b, x[6], x[8], force_denom)
        cl_model, cd_model = lift_drag_coefficients(f_model_b, x[6], x[8], force_denom)
        alpha_deg = math.degrees(float(loads["alpha_rad"]))
        beta_rad = float(loads.get("beta_rad", math.asin(np.clip(x[7] / max(speed_m_s, 1.0e-9), -1.0, 1.0))))
        regime = alpha_regime(alpha_deg)
        stage_fit_group = independent_stage_fit_group(regime, post_stall_seen)
        q_hat = float(loads.get("pitch_rate_hat", x[10] * aircraft.c_ref_m / (2.0 * speed_m_s)))
        p_hat = float(loads.get("roll_rate_hat", x[9] * aircraft.b_ref_m / (2.0 * speed_m_s)))
        r_hat = float(loads.get("yaw_rate_hat", x[11] * aircraft.b_ref_m / (2.0 * speed_m_s)))
        post_stall_activation = float(
            loads.get(
                "post_stall_residual_activation",
                loads.get(
                    "post_stall_pitch_activation",
                    post_stall_residual_activation_numpy(
                        float(loads["alpha_rad"]),
                        aircraft.post_stall_residual_blend_start_alpha_rad,
                        aircraft.post_stall_residual_blend_full_alpha_rad,
                    ),
                ),
            )
        )

        out.append(
            {
                "split": split,
                "session_label": row.get("session_label", ""),
                "case_id": row.get("case_id", ""),
                "throw_id": row.get("throw_id", ""),
                "residual_status": "ok",
                "sample_index": int(sample_index),
                "t_since_alignment_s": float(times[sample_index]),
                "regime": regime,
                "stage_fit_group": stage_fit_group,
                "post_stall_seen_before_sample": bool(post_stall_seen),
                "speed_m_s": speed_m_s,
                "q_bar_pa": float(q_bar),
                "alpha_deg": alpha_deg,
                "beta_deg": math.degrees(beta_rad),
                "p_rad_s": float(x[9]),
                "p_hat": p_hat,
                "q_rad_s": float(x[10]),
                "q_hat": q_hat,
                "r_rad_s": float(x[11]),
                "r_hat": r_hat,
                "post_stall_activation": post_stall_activation,
                "theta_deg": math.degrees(float(x[4])),
                "z_m": float(x[2]),
                "cx_required": float(f_aero_required_b[0] / force_denom),
                "cx_model": float(f_model_b[0] / force_denom),
                "cx_residual": float(f_residual_b[0] / force_denom),
                "cz_required": float(f_aero_required_b[2] / force_denom),
                "cz_model": float(f_model_b[2] / force_denom),
                "cz_residual": float(f_residual_b[2] / force_denom),
                "cl_required": cl_required,
                "cl_model": cl_model,
                "cl_residual": cl_required - cl_model,
                "cd_required": cd_required,
                "cd_model": cd_model,
                "cd_residual": cd_required - cd_model,
                "cy_required": float(f_aero_required_b[1] / force_denom),
                "cy_model": float(f_model_b[1] / force_denom),
                "cy_residual": float(f_residual_b[1] / force_denom),
                "cm_required": float(m_aero_required_b[1] / pitch_moment_denom),
                "cm_model": float(m_model_b[1] / pitch_moment_denom),
                "cm_residual": float(m_residual_b[1] / pitch_moment_denom),
                "cl_roll_required": float(m_aero_required_b[0] / roll_yaw_moment_denom),
                "cl_roll_model": float(m_model_b[0] / roll_yaw_moment_denom),
                "cl_roll_residual": float(m_residual_b[0] / roll_yaw_moment_denom),
                "cn_yaw_required": float(m_aero_required_b[2] / roll_yaw_moment_denom),
                "cn_yaw_model": float(m_model_b[2] / roll_yaw_moment_denom),
                "cn_yaw_residual": float(m_residual_b[2] / roll_yaw_moment_denom),
                "pitch_moment_required_n_m": float(m_aero_required_b[1]),
                "pitch_moment_model_n_m": float(m_model_b[1]),
                "pitch_moment_residual_n_m": float(m_residual_b[1]),
                "force_residual_norm_n": float(np.linalg.norm(f_residual_b)),
                "angular_accel_q_rad_s2": float(omega_dot_b[1]),
            }
        )
        if regime == "post_stall":
            post_stall_seen = True
    return out if out else [blocked_residual_row(row, split, "no_valid_residual_samples")]


def blocked_residual_row(row: dict[str, Any], split: str, status: str) -> dict[str, Any]:
    return {
        "split": split,
        "session_label": row.get("session_label", ""),
        "case_id": row.get("case_id", ""),
        "throw_id": row.get("throw_id", ""),
        "residual_status": status,
    }


def local_linear_slope(times: np.ndarray, values: np.ndarray, index: int, window_s: float) -> float:
    t_centre = float(times[index])
    mask = np.isfinite(times) & np.isfinite(values) & (np.abs(times - t_centre) <= float(window_s))
    if int(np.count_nonzero(mask)) < 4:
        finite_indices = np.where(np.isfinite(times) & np.isfinite(values))[0]
        if len(finite_indices) < 4:
            return float("nan")
        nearest = finite_indices[np.argsort(np.abs(times[finite_indices] - t_centre))[:7]]
        mask = np.zeros_like(times, dtype=bool)
        mask[nearest] = True
    t = times[mask] - t_centre
    y = values[mask]
    if len(t) < 2 or float(np.ptp(t)) <= 1e-9:
        return float("nan")
    design = np.column_stack([np.ones_like(t), t])
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coeffs[1])


def lift_drag_coefficients(force_b: np.ndarray, u: float, w: float, force_denom: float) -> tuple[float, float]:
    v_plane = np.array([float(u), 0.0, float(w)], dtype=float)
    norm = float(np.linalg.norm(v_plane))
    if norm <= 1e-9:
        return float("nan"), float("nan")
    drag_dir = -v_plane / norm
    lift_dir = np.array([-drag_dir[2], 0.0, drag_dir[0]], dtype=float)
    return float(np.dot(force_b, lift_dir) / force_denom), float(np.dot(force_b, drag_dir) / force_denom)


def alpha_regime(alpha_deg: float) -> str:
    if alpha_deg < STALL_ALPHA_DEG:
        return "attached"
    if alpha_deg < POST_STALL_ALPHA_DEG:
        return "transition"
    return "post_stall"


def independent_stage_fit_group(regime: str, post_stall_seen_before_sample: bool) -> str:
    if regime == "transition":
        return "transition_after_post_stall" if post_stall_seen_before_sample else "transition_before_post_stall"
    if regime in {"attached", "post_stall"}:
        return regime
    return str(regime)


def surface_rbf_parameter_name(prefix: str, centre_deg: float) -> str:
    centre_label = f"{float(centre_deg):g}".replace(".", "p").replace("-", "m")
    return f"{prefix}_{centre_label}_coeff"


def lateral_surface_parameter_name(prefix: str, feature: str, centre_deg: float) -> str:
    centre_label = f"{float(centre_deg):g}".replace(".", "p").replace("-", "m")
    return f"{prefix}_{feature}_rbf_{centre_label}_coeff"


def surface_rbf_basis_deg(alpha_deg: float, *, start_alpha_deg: float, full_alpha_deg: float) -> np.ndarray:
    activation = residual_blend_activation_deg(float(alpha_deg), float(start_alpha_deg), float(full_alpha_deg))
    centres = np.asarray(SURFACE_RBF_ALPHA_CENTERS_DEG, dtype=float)
    width = max(float(SURFACE_RBF_ALPHA_WIDTH_DEG), 1.0e-9)
    return activation * np.exp(-0.5 * ((float(alpha_deg) - centres) / width) ** 2)


def surface_rbf_prediction(
    coeffs: dict[str, float],
    prefix: str,
    alpha_deg: float,
    *,
    start_alpha_deg: float | None = None,
    full_alpha_deg: float | None = None,
) -> float:
    if not math.isfinite(float(alpha_deg)):
        return float("nan")
    start = (
        float(start_alpha_deg)
        if start_alpha_deg is not None
        else float(coeffs.get("post_stall_residual_blend_start_alpha_deg", STALL_ALPHA_DEG))
    )
    full = (
        float(full_alpha_deg)
        if full_alpha_deg is not None
        else float(coeffs.get("post_stall_residual_blend_full_alpha_deg", POST_STALL_ALPHA_DEG))
    )
    basis = surface_rbf_basis_deg(float(alpha_deg), start_alpha_deg=start, full_alpha_deg=full)
    values = np.asarray(
        [float(coeffs.get(surface_rbf_parameter_name(prefix, centre_deg), 0.0)) for centre_deg in SURFACE_RBF_ALPHA_CENTERS_DEG],
        dtype=float,
    )
    return float(np.dot(values, basis))


def lateral_surface_prediction(
    coeffs: dict[str, float],
    prefix: str,
    alpha_deg: float,
    beta_rad: float,
    p_hat: float,
    r_hat: float,
    *,
    start_alpha_deg: float | None = None,
    full_alpha_deg: float | None = None,
) -> float:
    if not all(math.isfinite(float(value)) for value in (alpha_deg, beta_rad, p_hat, r_hat)):
        return float("nan")
    start = (
        float(start_alpha_deg)
        if start_alpha_deg is not None
        else float(coeffs.get("post_stall_residual_blend_start_alpha_deg", STALL_ALPHA_DEG))
    )
    full = (
        float(full_alpha_deg)
        if full_alpha_deg is not None
        else float(coeffs.get("post_stall_residual_blend_full_alpha_deg", POST_STALL_ALPHA_DEG))
    )
    basis = surface_rbf_basis_deg(float(alpha_deg), start_alpha_deg=start, full_alpha_deg=full)
    features = np.asarray([1.0, float(beta_rad), float(p_hat), float(r_hat)], dtype=float)
    values = np.asarray(
        [
            [
                float(coeffs.get(lateral_surface_parameter_name(prefix, feature, centre_deg), 0.0))
                for centre_deg in SURFACE_RBF_ALPHA_CENTERS_DEG
            ]
            for feature in LATERAL_SURFACE_FEATURES
        ],
        dtype=float,
    )
    return float(features @ values @ basis)


def lateral_surface_design_matrix(samples: list[dict[str, Any]]) -> np.ndarray:
    rows = []
    for sample in samples:
        alpha_deg = float(sample.get("alpha_deg", float("nan")))
        beta_rad = math.radians(float(sample.get("beta_deg", float("nan"))))
        p_hat = float(sample.get("p_hat", 0.0)) if math.isfinite(float(sample.get("p_hat", 0.0))) else 0.0
        r_hat = float(sample.get("r_hat", 0.0)) if math.isfinite(float(sample.get("r_hat", 0.0))) else 0.0
        basis = surface_rbf_basis_deg(
            alpha_deg,
            start_alpha_deg=STALL_ALPHA_DEG,
            full_alpha_deg=POST_STALL_ALPHA_DEG,
        )
        features = np.asarray([1.0, beta_rad, p_hat, r_hat], dtype=float)
        rows.append(
            np.asarray([feature_value * basis_value for feature_value in features for basis_value in basis], dtype=float)
        )
    return np.asarray(rows, dtype=float)


def lateral_surface_coeff_keys(prefix: str) -> list[str]:
    return [
        lateral_surface_parameter_name(prefix, feature, centre_deg)
        for feature in LATERAL_SURFACE_FEATURES
        for centre_deg in SURFACE_RBF_ALPHA_CENTERS_DEG
    ]


def fit_pitch_residual_coefficients(
    rows: list[dict[str, Any]],
    *,
    ridge_lambda: float,
    fit_post_stall_damping: bool,
    fit_lateral_surfaces: bool,
) -> dict[str, Any]:
    samples = []
    for row in rows:
        if row.get("residual_status") != "ok":
            continue
        cm = finite_value(row.get("cm_residual"))
        if not math.isfinite(cm):
            continue
        samples.append(
            {
                "cm": cm,
                "cl": finite_value(row.get("cl_residual")),
                "cd": finite_value(row.get("cd_residual")),
                "cy": finite_value(row.get("cy_residual")),
                "cl_roll": finite_value(row.get("cl_roll_residual")),
                "cn_yaw": finite_value(row.get("cn_yaw_residual")),
                "q_bar": finite_value(row.get("q_bar_pa")),
                "q_hat": finite_value(row.get("q_hat")),
                "beta_deg": finite_value(row.get("beta_deg")),
                "p_hat": finite_value(row.get("p_hat")),
                "r_hat": finite_value(row.get("r_hat")),
                "alpha_deg": finite_value(row.get("alpha_deg")),
                "activation": finite_value(row.get("post_stall_activation")),
                "regime": str(row.get("regime", "")),
                "stage_fit_group": str(row.get("stage_fit_group") or row.get("regime", "")),
                "throw_key": f"{row.get('session_label', '')}/{row.get('throw_id', '')}",
            }
        )
    if len(samples) < 8:
        return {"status": "too_few_samples", "sample_count": len(samples), "coefficients": zero_coefficients()}

    attached_samples = [sample for sample in samples if sample["stage_fit_group"] == "attached"]
    transition_before_samples = [
        sample for sample in samples if sample["stage_fit_group"] == "transition_before_post_stall"
    ]
    transition_after_samples = [
        sample for sample in samples if sample["stage_fit_group"] == "transition_after_post_stall"
    ]
    transition_samples = [sample for sample in samples if sample["regime"] == "transition"]
    post_stall_samples = [sample for sample in samples if sample["regime"] == "post_stall"]

    surface_fit = fit_post_stall_surface_coefficients(
        post_stall_samples,
        ridge_lambda=ridge_lambda,
        fit_post_stall_damping=fit_post_stall_damping,
        fit_lateral_surfaces=fit_lateral_surfaces,
    )
    surface_coeffs = surface_fit["coefficients"]
    coeffs = {
        "attached_cm_bias_coeff": fit_stage_constant_residual(attached_samples, "cm", ridge_lambda=ridge_lambda),
        "transition_cm_bias_coeff": fit_stage_constant_residual(transition_samples, "cm", ridge_lambda=ridge_lambda),
        "transition_before_post_stall_cm_bias_coeff": fit_stage_constant_residual(
            transition_before_samples,
            "cm",
            ridge_lambda=ridge_lambda,
        ),
        "transition_after_post_stall_cm_bias_coeff": fit_stage_constant_residual(
            transition_after_samples,
            "cm",
            ridge_lambda=ridge_lambda,
        ),
        "post_stall_lift_residual_coeff": 0.0,
        "post_stall_drag_residual_coeff": 0.0,
        "post_stall_pitch_moment_coeff": 0.0,
        "post_stall_pitch_damping_coeff": float(surface_coeffs.get("post_stall_pitch_damping_coeff", 0.0)),
    }
    coeffs.update(surface_coeffs)
    blender_samples = transition_before_samples if len(transition_before_samples) >= 12 else transition_samples
    blender_fit_group = "transition_before_post_stall" if len(transition_before_samples) >= 12 else "all_transition"
    blend_fit = fit_transition_blender(
        blender_samples,
        coeffs,
        fit_group=blender_fit_group,
    )
    coeffs["post_stall_residual_blend_start_alpha_deg"] = float(blend_fit["start_alpha_deg"])
    coeffs["post_stall_residual_blend_full_alpha_deg"] = float(blend_fit["full_alpha_deg"])
    residual_after = np.asarray([cm_fit_residual_for_sample(sample, coeffs) for sample in samples], dtype=float)
    mask = np.isfinite(residual_after)
    return {
        "status": "ok",
        "sample_count": int(len(samples)),
        "used_sample_count": int(np.count_nonzero(mask)),
        "post_stall_used_sample_count": int(surface_fit.get("used_sample_count", 0)),
        "transition_blender_fit": blend_fit,
        "surface_fit": surface_fit,
        "stage_sample_counts": {
            "attached": len(attached_samples),
            "transition_before_post_stall": len(transition_before_samples),
            "transition_after_post_stall": len(transition_after_samples),
            "post_stall": len(post_stall_samples),
        },
        "ridge_lambda": float(ridge_lambda),
        "fit_policy": (
            "sequential_stage_6dof_surface_fit; attached diagnostic first, post-stall alpha-RBF "
            "CL/CD/Cm/Cmq surfaces plus optional CY/Cl/Cn surfaces second, transition blender third"
        ),
        "coefficients": coeffs,
        "fit_rmse_cm": float(np.sqrt(np.mean(residual_after[mask] ** 2))),
        "fit_mae_cm": float(np.mean(np.abs(residual_after[mask]))),
        "cm_residual_mean_before": mean([sample["cm"] for sample in samples]),
    }


def fit_stage_constant_residual(samples: list[dict[str, Any]], key: str, *, ridge_lambda: float) -> float:
    valid = [
        sample
        for sample in samples
        if math.isfinite(float(sample.get(key, float("nan"))))
        and math.isfinite(float(sample.get("q_bar", float("nan"))))
    ]
    if len(valid) < 4:
        return 0.0
    y = np.asarray([float(sample[key]) for sample in valid], dtype=float)
    x = np.ones((len(valid), 1), dtype=float)
    q_bar = np.asarray([float(sample["q_bar"]) for sample in valid], dtype=float)
    weights = dynamic_pressure_weights(q_bar) * throw_balance_weights(valid)
    mask = np.ones(len(y), dtype=bool)
    coeff = np.zeros(1, dtype=float)
    for _ in range(2):
        coeff = weighted_ridge_fit(x[mask], y[mask], weights[mask], float(ridge_lambda))
        residual = y - x @ coeff
        sigma = robust_sigma(residual[mask])
        if not math.isfinite(sigma) or sigma <= 1e-9:
            break
        next_mask = np.abs(residual - float(np.nanmedian(residual[mask]))) <= 4.0 * sigma
        if int(np.count_nonzero(next_mask)) < 4:
            break
        mask = next_mask
    return float(coeff[0])


def fit_post_stall_surface_coefficients(
    samples: list[dict[str, Any]],
    *,
    ridge_lambda: float,
    fit_post_stall_damping: bool,
    fit_lateral_surfaces: bool,
) -> dict[str, Any]:
    valid = [
        sample
        for sample in samples
        if math.isfinite(float(sample.get("alpha_deg", float("nan"))))
        and math.isfinite(float(sample.get("q_bar", float("nan"))))
        and math.isfinite(float(sample.get("cl", float("nan"))))
        and math.isfinite(float(sample.get("cd", float("nan"))))
        and math.isfinite(float(sample.get("cm", float("nan"))))
    ]
    coeffs = zero_surface_coefficients()
    if len(valid) < 8:
        return {
            "status": "too_few_post_stall_samples",
            "sample_count": len(valid),
            "used_sample_count": 0,
            "coefficients": coeffs,
        }

    x_surface = np.asarray(
        [
            surface_rbf_basis_deg(
                float(sample["alpha_deg"]),
                start_alpha_deg=STALL_ALPHA_DEG,
                full_alpha_deg=POST_STALL_ALPHA_DEG,
            )
            for sample in valid
        ],
        dtype=float,
    )
    basis_norm = np.linalg.norm(x_surface, axis=1)
    finite_mask = np.isfinite(x_surface).all(axis=1) & (basis_norm > 1.0e-9)
    if int(np.count_nonzero(finite_mask)) < 8:
        return {
            "status": "degenerate_post_stall_basis",
            "sample_count": len(valid),
            "used_sample_count": int(np.count_nonzero(finite_mask)),
            "coefficients": coeffs,
        }

    valid = [sample for sample, keep in zip(valid, finite_mask) if keep]
    x_surface = x_surface[finite_mask]
    q_bar = np.asarray([float(sample["q_bar"]) for sample in valid], dtype=float)
    weights = dynamic_pressure_weights(q_bar) * throw_balance_weights(valid)

    fit_details: dict[str, Any] = {}
    for residual_key, prefix in (
        ("cl", "post_stall_lift_rbf"),
        ("cd", "post_stall_drag_rbf"),
        ("cm", "post_stall_pitch_moment_rbf"),
    ):
        y = np.asarray([float(sample[residual_key]) for sample in valid], dtype=float)
        x = x_surface
        if residual_key == "cm" and fit_post_stall_damping:
            q_hat = np.asarray(
                [
                    float(sample.get("q_hat", 0.0))
                    if math.isfinite(float(sample.get("q_hat", 0.0)))
                    else 0.0
                    for sample in valid
                ],
                dtype=float,
            )
            x = np.column_stack([x_surface, x_surface * q_hat[:, None]])
        coeff, used_count, mae_value, rmse_value = robust_weighted_ridge_fit(
            x,
            y,
            weights,
            ridge_lambda=float(ridge_lambda),
            min_used_count=8,
        )
        for index, centre_deg in enumerate(SURFACE_RBF_ALPHA_CENTERS_DEG):
            key = surface_rbf_parameter_name(prefix, centre_deg)
            coeffs[key] = replay_fit.bounded_parameter_value(key, float(coeff[index]))
        if residual_key == "cm" and fit_post_stall_damping:
            offset = len(SURFACE_RBF_ALPHA_CENTERS_DEG)
            for index, centre_deg in enumerate(SURFACE_RBF_ALPHA_CENTERS_DEG):
                key = surface_rbf_parameter_name("post_stall_pitch_damping_rbf", centre_deg)
                coeffs[key] = replay_fit.bounded_parameter_value(key, float(coeff[offset + index]))
        fit_details[residual_key] = {
            "used_sample_count": int(used_count),
            "mae": float(mae_value),
            "rmse": float(rmse_value),
        }

    if fit_lateral_surfaces:
        x_lateral = lateral_surface_design_matrix(valid)
        for residual_key, prefix in (
            ("cy", "post_stall_side_force"),
            ("cl_roll", "post_stall_roll_moment"),
            ("cn_yaw", "post_stall_yaw_moment"),
        ):
            y = np.asarray([float(sample.get(residual_key, float("nan"))) for sample in valid], dtype=float)
            finite_lateral = np.isfinite(x_lateral).all(axis=1) & np.isfinite(y)
            if int(np.count_nonzero(finite_lateral)) < 8:
                fit_details[residual_key] = {
                    "used_sample_count": int(np.count_nonzero(finite_lateral)),
                    "mae": float("nan"),
                    "rmse": float("nan"),
                }
                continue
            coeff, used_count, mae_value, rmse_value = robust_weighted_ridge_fit(
                x_lateral[finite_lateral],
                y[finite_lateral],
                weights[finite_lateral],
                ridge_lambda=float(ridge_lambda),
                min_used_count=8,
            )
            for key, value in zip(lateral_surface_coeff_keys(prefix), coeff):
                coeffs[key] = replay_fit.bounded_parameter_value(key, float(value))
            fit_details[residual_key] = {
                "used_sample_count": int(used_count),
                "mae": float(mae_value),
                "rmse": float(rmse_value),
            }
    else:
        for residual_key in ("cy", "cl_roll", "cn_yaw"):
            fit_details[residual_key] = {
                "used_sample_count": 0,
                "mae": float("nan"),
                "rmse": float("nan"),
                "status": "disabled",
            }

    return {
        "status": "ok",
        "sample_count": len(valid),
        "used_sample_count": min(int(detail["used_sample_count"]) for detail in fit_details.values()),
        "basis_centres_deg": list(SURFACE_RBF_ALPHA_CENTERS_DEG),
        "basis_width_deg": float(SURFACE_RBF_ALPHA_WIDTH_DEG),
        "fit_details": fit_details,
        "coefficients": coeffs,
    }


def fit_post_stall_pitch_coefficients(
    samples: list[dict[str, Any]],
    *,
    ridge_lambda: float,
    fit_post_stall_damping: bool,
) -> tuple[float, float, int]:
    valid = [
        sample
        for sample in samples
        if math.isfinite(float(sample.get("cm", float("nan"))))
        and math.isfinite(float(sample.get("activation", float("nan"))))
        and sample["activation"] > 1.0e-6
        and math.isfinite(float(sample.get("q_bar", float("nan"))))
    ]
    if len(valid) < 8:
        return 0.0, 0.0, 0
    y = np.asarray([float(sample["cm"]) for sample in valid], dtype=float)
    if fit_post_stall_damping:
        x = np.asarray(
            [[float(sample["activation"]), float(sample["activation"]) * float(sample.get("q_hat", 0.0))] for sample in valid],
            dtype=float,
        )
    else:
        x = np.asarray([[float(sample["activation"])] for sample in valid], dtype=float)
    q_bar = np.asarray([float(sample["q_bar"]) for sample in valid], dtype=float)
    weights = dynamic_pressure_weights(q_bar)
    mask = np.ones(len(y), dtype=bool)
    coeff = np.zeros(x.shape[1], dtype=float)
    for _ in range(2):
        coeff = weighted_ridge_fit(x[mask], y[mask], weights[mask], float(ridge_lambda))
        residual = y - x @ coeff
        sigma = robust_sigma(residual[mask])
        if not math.isfinite(sigma) or sigma <= 1e-9:
            break
        next_mask = np.abs(residual - float(np.nanmedian(residual[mask]))) <= 4.0 * sigma
        if int(np.count_nonzero(next_mask)) < 8:
            break
        mask = next_mask
    cm_coeff = float(coeff[0])
    cmq_coeff = float(coeff[1]) if fit_post_stall_damping and coeff.size > 1 else 0.0
    return cm_coeff, cmq_coeff, int(np.count_nonzero(mask))


def fit_transition_blender(samples: list[dict[str, Any]], coeffs: dict[str, float], *, fit_group: str) -> dict[str, Any]:
    valid = [
        sample
        for sample in samples
        if math.isfinite(float(sample.get("alpha_deg", float("nan"))))
        and math.isfinite(float(sample.get("cm", float("nan"))))
        and math.isfinite(float(sample.get("cl", float("nan"))))
        and math.isfinite(float(sample.get("cd", float("nan"))))
    ]
    if len(valid) < 8:
        return {
            "status": "too_few_transition_samples",
            "fit_group": fit_group,
            "sample_count": len(valid),
            "start_alpha_deg": STALL_ALPHA_DEG,
            "full_alpha_deg": POST_STALL_ALPHA_DEG,
            "objective": float("nan"),
        }
    post_cmq = float(coeffs.get("post_stall_pitch_damping_coeff", 0.0))
    surface_values = [
        abs(float(coeffs.get(surface_rbf_parameter_name(prefix, centre_deg), 0.0)))
        for prefix in (
            "post_stall_lift_rbf",
            "post_stall_drag_rbf",
            "post_stall_pitch_moment_rbf",
            "post_stall_pitch_damping_rbf",
        )
        for centre_deg in SURFACE_RBF_ALPHA_CENTERS_DEG
    ]
    surface_values.extend(
        abs(float(coeffs.get(key, 0.0)))
        for prefix in LATERAL_SURFACE_PREFIXES
        for key in lateral_surface_coeff_keys(prefix)
    )
    if max([abs(post_cmq), *surface_values]) <= 1.0e-12:
        return {
            "status": "zero_post_stall_coefficients",
            "fit_group": fit_group,
            "sample_count": len(valid),
            "start_alpha_deg": STALL_ALPHA_DEG,
            "full_alpha_deg": POST_STALL_ALPHA_DEG,
            "objective": float("nan"),
        }

    best = {
        "status": "ok",
        "fit_group": fit_group,
        "sample_count": len(valid),
        "start_alpha_deg": STALL_ALPHA_DEG,
        "full_alpha_deg": POST_STALL_ALPHA_DEG,
        "objective": float("inf"),
    }
    start_values = np.arange(8.0, min(16.0, POST_STALL_ALPHA_DEG - 3.0) + 1.0e-9, 1.0)
    for start_alpha_deg in start_values:
        full_min = max(start_alpha_deg + 3.0, 14.0)
        full_values = np.arange(full_min, POST_STALL_ALPHA_DEG + 1.0e-9, 1.0)
        for full_alpha_deg in full_values:
            cm_resid = []
            cl_resid = []
            cd_resid = []
            cy_resid = []
            cl_roll_resid = []
            cn_yaw_resid = []
            for sample in valid:
                alpha_deg = float(sample["alpha_deg"])
                activation = residual_blend_activation_deg(alpha_deg, float(start_alpha_deg), float(full_alpha_deg))
                q_hat = float(sample.get("q_hat", 0.0)) if math.isfinite(float(sample.get("q_hat", 0.0))) else 0.0
                beta_rad = math.radians(float(sample.get("beta_deg", 0.0))) if math.isfinite(float(sample.get("beta_deg", 0.0))) else 0.0
                p_hat = float(sample.get("p_hat", 0.0)) if math.isfinite(float(sample.get("p_hat", 0.0))) else 0.0
                r_hat = float(sample.get("r_hat", 0.0)) if math.isfinite(float(sample.get("r_hat", 0.0))) else 0.0
                cm_model = surface_rbf_prediction(
                    coeffs,
                    "post_stall_pitch_moment_rbf",
                    alpha_deg,
                    start_alpha_deg=float(start_alpha_deg),
                    full_alpha_deg=float(full_alpha_deg),
                )
                cmq_model = surface_rbf_prediction(
                    coeffs,
                    "post_stall_pitch_damping_rbf",
                    alpha_deg,
                    start_alpha_deg=float(start_alpha_deg),
                    full_alpha_deg=float(full_alpha_deg),
                )
                cl_model = surface_rbf_prediction(
                    coeffs,
                    "post_stall_lift_rbf",
                    alpha_deg,
                    start_alpha_deg=float(start_alpha_deg),
                    full_alpha_deg=float(full_alpha_deg),
                )
                cd_model = surface_rbf_prediction(
                    coeffs,
                    "post_stall_drag_rbf",
                    alpha_deg,
                    start_alpha_deg=float(start_alpha_deg),
                    full_alpha_deg=float(full_alpha_deg),
                )
                cy_model = lateral_surface_prediction(
                    coeffs,
                    "post_stall_side_force",
                    alpha_deg,
                    beta_rad,
                    p_hat,
                    r_hat,
                    start_alpha_deg=float(start_alpha_deg),
                    full_alpha_deg=float(full_alpha_deg),
                )
                cl_roll_model = lateral_surface_prediction(
                    coeffs,
                    "post_stall_roll_moment",
                    alpha_deg,
                    beta_rad,
                    p_hat,
                    r_hat,
                    start_alpha_deg=float(start_alpha_deg),
                    full_alpha_deg=float(full_alpha_deg),
                )
                cn_yaw_model = lateral_surface_prediction(
                    coeffs,
                    "post_stall_yaw_moment",
                    alpha_deg,
                    beta_rad,
                    p_hat,
                    r_hat,
                    start_alpha_deg=float(start_alpha_deg),
                    full_alpha_deg=float(full_alpha_deg),
                )
                cm_resid.append(float(sample["cm"]) - (cm_model + (activation * post_cmq + cmq_model) * q_hat))
                cl_resid.append(float(sample["cl"]) - cl_model)
                cd_resid.append(float(sample["cd"]) - cd_model)
                if math.isfinite(float(sample.get("cy", float("nan")))) and math.isfinite(cy_model):
                    cy_resid.append(float(sample["cy"]) - cy_model)
                if math.isfinite(float(sample.get("cl_roll", float("nan")))) and math.isfinite(cl_roll_model):
                    cl_roll_resid.append(float(sample["cl_roll"]) - cl_roll_model)
                if math.isfinite(float(sample.get("cn_yaw", float("nan")))) and math.isfinite(cn_yaw_model):
                    cn_yaw_resid.append(float(sample["cn_yaw"]) - cn_yaw_model)
            objective = (
                mae(cm_resid) / 0.08
                + 0.5 * mae(cl_resid) / 0.25
                + 0.5 * mae(cd_resid) / 0.25
                + 0.25 * mae(cy_resid) / 0.25
                + 0.25 * mae(cl_roll_resid) / 0.12
                + 0.25 * mae(cn_yaw_resid) / 0.12
                + 0.05 * ((float(start_alpha_deg) - STALL_ALPHA_DEG) / 4.0) ** 2
                + 0.05 * ((float(full_alpha_deg) - POST_STALL_ALPHA_DEG) / 4.0) ** 2
            )
            if objective < float(best["objective"]):
                best = {
                    "status": "ok",
                    "fit_group": fit_group,
                    "sample_count": len(valid),
                    "start_alpha_deg": float(start_alpha_deg),
                    "full_alpha_deg": float(full_alpha_deg),
                    "objective": float(objective),
                    "cm_mae": mae(cm_resid),
                    "cl_mae": mae(cl_resid),
                    "cd_mae": mae(cd_resid),
                    "cy_mae": mae(cy_resid),
                    "cl_roll_mae": mae(cl_roll_resid),
                    "cn_yaw_mae": mae(cn_yaw_resid),
                }
    return best


def residual_blend_activation_deg(alpha_deg: float, start_alpha_deg: float, full_alpha_deg: float) -> float:
    t = np.clip(
        (float(alpha_deg) - float(start_alpha_deg)) / max(float(full_alpha_deg) - float(start_alpha_deg), 1.0e-9),
        0.0,
        1.0,
    )
    return float(t * t * (3.0 - 2.0 * t))


def fit_activated_scalar_residual(samples: list[dict[str, float]], key: str, *, ridge_lambda: float) -> float:
    valid = [
        sample
        for sample in samples
        if math.isfinite(sample.get(key, float("nan")))
        and math.isfinite(sample.get("activation", float("nan")))
        and sample["activation"] > 1.0e-6
    ]
    if len(valid) < 8:
        return 0.0
    y = np.asarray([sample[key] for sample in valid], dtype=float)
    x = np.asarray([[sample["activation"]] for sample in valid], dtype=float)
    q_bar = np.asarray([sample["q_bar"] for sample in valid], dtype=float)
    weights = dynamic_pressure_weights(q_bar)
    coeff = weighted_ridge_fit(x, y, weights, float(ridge_lambda))
    return float(coeff[0])


def zero_coefficients() -> dict[str, float]:
    coeffs = {
        "attached_cm_bias_coeff": 0.0,
        "transition_cm_bias_coeff": 0.0,
        "transition_before_post_stall_cm_bias_coeff": 0.0,
        "transition_after_post_stall_cm_bias_coeff": 0.0,
        "post_stall_lift_residual_coeff": 0.0,
        "post_stall_drag_residual_coeff": 0.0,
        "post_stall_pitch_moment_coeff": 0.0,
        "post_stall_pitch_damping_coeff": 0.0,
        "post_stall_residual_blend_start_alpha_deg": STALL_ALPHA_DEG,
        "post_stall_residual_blend_full_alpha_deg": POST_STALL_ALPHA_DEG,
    }
    coeffs.update(zero_surface_coefficients())
    return coeffs


def zero_surface_coefficients() -> dict[str, float]:
    coeffs = {"post_stall_pitch_damping_coeff": 0.0}
    for prefix in (
        "post_stall_lift_rbf",
        "post_stall_drag_rbf",
        "post_stall_pitch_moment_rbf",
        "post_stall_pitch_damping_rbf",
    ):
        for centre_deg in SURFACE_RBF_ALPHA_CENTERS_DEG:
            coeffs[surface_rbf_parameter_name(prefix, centre_deg)] = 0.0
    for prefix in LATERAL_SURFACE_PREFIXES:
        for key in lateral_surface_coeff_keys(prefix):
            coeffs[key] = 0.0
    return coeffs


def weighted_ridge_fit(x: np.ndarray, y: np.ndarray, weights: np.ndarray, ridge_lambda: float) -> np.ndarray:
    w = np.asarray(weights, dtype=float).reshape(-1, 1)
    xw = x * w
    yw = y * w[:, 0]
    return np.linalg.solve(xw.T @ xw + ridge_lambda * np.eye(x.shape[1]), xw.T @ yw)


def robust_weighted_ridge_fit(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    ridge_lambda: float,
    min_used_count: int,
) -> tuple[np.ndarray, int, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)
    finite = np.isfinite(x).all(axis=1) & np.isfinite(y) & np.isfinite(weights)
    if int(np.count_nonzero(finite)) < int(min_used_count):
        return np.zeros(x.shape[1], dtype=float), int(np.count_nonzero(finite)), float("nan"), float("nan")
    mask = finite.copy()
    coeff = np.zeros(x.shape[1], dtype=float)
    for _ in range(3):
        coeff = weighted_ridge_fit(x[mask], y[mask], weights[mask], float(ridge_lambda))
        residual = y - x @ coeff
        sigma = robust_sigma(residual[mask])
        if not math.isfinite(sigma) or sigma <= 1e-9:
            break
        centre = float(np.nanmedian(residual[mask]))
        next_mask = finite & (np.abs(residual - centre) <= 4.0 * sigma)
        if int(np.count_nonzero(next_mask)) < int(min_used_count):
            break
        if np.array_equal(next_mask, mask):
            break
        mask = next_mask
    residual = y - x @ coeff
    used_residual = residual[mask]
    return (
        coeff,
        int(np.count_nonzero(mask)),
        mae([float(value) for value in used_residual if math.isfinite(float(value))]),
        float(np.sqrt(np.mean(used_residual[np.isfinite(used_residual)] ** 2))) if np.any(np.isfinite(used_residual)) else float("nan"),
    )


def dynamic_pressure_weights(q_bar: np.ndarray) -> np.ndarray:
    q_bar = np.asarray(q_bar, dtype=float)
    q_bar_median = float(np.nanmedian(q_bar)) if np.any(np.isfinite(q_bar)) else 1.0
    return np.sqrt(np.clip(q_bar / max(q_bar_median, 1e-9), 0.25, 4.0))


def throw_balance_weights(samples: list[dict[str, Any]]) -> np.ndarray:
    counts: dict[str, int] = {}
    for sample in samples:
        key = str(sample.get("throw_key", ""))
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return np.ones(len(samples), dtype=float)
    median_count = float(np.median(list(counts.values())))
    return np.asarray(
        [
            math.sqrt(median_count / max(float(counts.get(str(sample.get("throw_key", "")), 1)), 1.0))
            for sample in samples
        ],
        dtype=float,
    )


def cm_fit_residual_for_sample(sample: dict[str, Any], coeffs: dict[str, float]) -> float:
    cm = float(sample.get("cm", float("nan")))
    if not math.isfinite(cm):
        return float("nan")
    fitted = 0.0
    stage_fit_group = str(sample.get("stage_fit_group", ""))
    if stage_fit_group == "attached":
        fitted += float(coeffs.get("attached_cm_bias_coeff", 0.0))
    elif str(sample.get("regime", "")) in {"transition", "post_stall"}:
        alpha_deg = float(sample.get("alpha_deg", float("nan")))
        activation = residual_blend_activation_from_coeffs(alpha_deg, coeffs)
        q_hat = float(sample.get("q_hat", float("nan")))
        fitted_surface = surface_rbf_prediction(coeffs, "post_stall_pitch_moment_rbf", alpha_deg)
        damping_surface = surface_rbf_prediction(coeffs, "post_stall_pitch_damping_rbf", alpha_deg)
        if math.isfinite(fitted_surface):
            fitted += fitted_surface
        if math.isfinite(activation):
            if math.isfinite(q_hat):
                fitted += (
                    float(coeffs.get("post_stall_pitch_damping_coeff", 0.0)) * activation
                    + (damping_surface if math.isfinite(damping_surface) else 0.0)
                ) * q_hat
    return cm - fitted


def robust_sigma(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    return 1.4826 * mad


def candidate_from_fit(
    base: dict[str, float],
    fit_result: dict[str, Any],
    *,
    apply_attached_cm_bias: bool,
    fit_post_stall_damping: bool,
    fit_lateral_surfaces: bool,
    surface_scale: float = 1.0,
) -> dict[str, float]:
    coeffs = fit_result.get("coefficients", zero_coefficients())
    candidate = dict(base)
    if apply_attached_cm_bias:
        candidate["pitch_moment_bias_coeff"] = replay_fit.bounded_parameter_value(
            "pitch_moment_bias_coeff",
            candidate["pitch_moment_bias_coeff"] + float(coeffs.get("attached_cm_bias_coeff", 0.0)),
        )
    candidate["post_stall_lift_residual_coeff"] = replay_fit.bounded_parameter_value(
        "post_stall_lift_residual_coeff",
        candidate["post_stall_lift_residual_coeff"] + float(coeffs.get("post_stall_lift_residual_coeff", 0.0)),
    )
    candidate["post_stall_drag_residual_coeff"] = replay_fit.bounded_parameter_value(
        "post_stall_drag_residual_coeff",
        candidate["post_stall_drag_residual_coeff"] + float(coeffs.get("post_stall_drag_residual_coeff", 0.0)),
    )
    candidate["post_stall_pitch_moment_coeff"] = replay_fit.bounded_parameter_value(
        "post_stall_pitch_moment_coeff",
        candidate["post_stall_pitch_moment_coeff"] + float(coeffs.get("post_stall_pitch_moment_coeff", 0.0)),
    )
    candidate["post_stall_pitch_damping_coeff"] = replay_fit.bounded_parameter_value(
        "post_stall_pitch_damping_coeff",
        candidate["post_stall_pitch_damping_coeff"]
        + (float(coeffs.get("post_stall_pitch_damping_coeff", 0.0)) if fit_post_stall_damping else 0.0),
    )
    for prefix in (
        "post_stall_lift_rbf",
        "post_stall_drag_rbf",
        "post_stall_pitch_moment_rbf",
        "post_stall_pitch_damping_rbf",
    ):
        for centre_deg in SURFACE_RBF_ALPHA_CENTERS_DEG:
            key = surface_rbf_parameter_name(prefix, centre_deg)
            candidate[key] = replay_fit.bounded_parameter_value(
                key,
                float(candidate.get(key, 0.0)) + float(surface_scale) * float(coeffs.get(key, 0.0)),
            )
    if fit_lateral_surfaces:
        for prefix in LATERAL_SURFACE_PREFIXES:
            for key in lateral_surface_coeff_keys(prefix):
                candidate[key] = replay_fit.bounded_parameter_value(
                    key,
                    float(candidate.get(key, 0.0)) + float(surface_scale) * float(coeffs.get(key, 0.0)),
                )
    candidate["post_stall_residual_blend_start_alpha_deg"] = replay_fit.bounded_parameter_value(
        "post_stall_residual_blend_start_alpha_deg",
        float(coeffs.get("post_stall_residual_blend_start_alpha_deg", candidate["post_stall_residual_blend_start_alpha_deg"])),
    )
    candidate["post_stall_residual_blend_full_alpha_deg"] = replay_fit.bounded_parameter_value(
        "post_stall_residual_blend_full_alpha_deg",
        max(
            candidate["post_stall_residual_blend_start_alpha_deg"] + 1.0,
            float(coeffs.get("post_stall_residual_blend_full_alpha_deg", candidate["post_stall_residual_blend_full_alpha_deg"])),
        ),
    )
    return candidate


def select_surface_scale_rows(
    *,
    base_parameters: dict[str, float],
    fit_result: dict[str, Any],
    train_rows: list[dict[str, Any]],
    replay_dt_s: float,
    alignment_window_s: float,
    workers: int,
    apply_attached_cm_bias: bool,
    fit_post_stall_damping: bool,
    fit_lateral_surfaces: bool,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    best_index = 0
    best_objective = float("inf")
    for index, scale in enumerate(SURFACE_SCALE_CANDIDATES):
        candidate = candidate_from_fit(
            base_parameters,
            fit_result,
            apply_attached_cm_bias=apply_attached_cm_bias,
            fit_post_stall_damping=fit_post_stall_damping,
            fit_lateral_surfaces=fit_lateral_surfaces,
            surface_scale=float(scale),
        )
        replay_rows = replay_fit.simulate_rows(
            train_rows,
            candidate,
            replay_dt_s=replay_dt_s,
            alignment_window_s=alignment_window_s,
            workers=workers,
        )
        summary = replay_fit.objective_summary(replay_rows, objective_mode="combined")
        objective = finite_value(summary.get("objective"))
        if math.isfinite(objective) and objective < best_objective:
            best_objective = objective
            best_index = index
        output.append(
            {
                "surface_scale": float(scale),
                "selected": False,
                "objective": objective,
                "dx_mae_m": finite_value(summary.get("dx_mae_m")),
                "dy_mae_m": finite_value(summary.get("dy_mae_m")),
                "altitude_loss_mae_m": finite_value(summary.get("altitude_loss_mae_m")),
                "sink_rate_mae_m_s": finite_value(summary.get("sink_mae_m_s")),
                "final_theta_mae_deg": finite_value(summary.get("final_theta_mae_deg")),
                "final_phi_mae_deg": finite_value(summary.get("final_phi_mae_deg")),
                "final_psi_mae_deg": finite_value(summary.get("final_psi_mae_deg")),
            }
        )
    if output:
        output[best_index]["selected"] = True
    return output


def selected_surface_scale_from_rows(rows: list[dict[str, Any]]) -> float:
    for row in rows:
        if bool(row.get("selected", False)):
            return float(row.get("surface_scale", 1.0))
    return 1.0


def replay_validation_rows(
    *,
    train_rows: list[dict[str, Any]],
    heldout_rows: list[dict[str, Any]],
    base_parameters: dict[str, float],
    candidate_parameters: dict[str, float],
    replay_dt_s: float,
    alignment_window_s: float,
    workers: int,
) -> list[dict[str, Any]]:
    out = []
    for model_id, parameters in (("baseline_active", base_parameters), ("coefficient_candidate", candidate_parameters)):
        for split, rows in (("train", train_rows), ("heldout", heldout_rows)):
            replay_rows = replay_fit.simulate_rows(
                rows,
                parameters,
                replay_dt_s=replay_dt_s,
                alignment_window_s=alignment_window_s,
                workers=workers,
            )
            for row in replay_rows:
                row["model_id"] = model_id
                row["split"] = split
                out.append(row)
    return out


def stage_replay_summary_rows(
    *,
    train_rows: list[dict[str, Any]],
    heldout_rows: list[dict[str, Any]],
    base_parameters: dict[str, float],
    candidate_parameters: dict[str, float],
    replay_dt_s: float,
    alignment_window_s: float,
    workers: int,
) -> list[dict[str, Any]]:
    payloads = []
    for model_id, parameters in (
        ("baseline_active", base_parameters),
        ("coefficient_candidate", candidate_parameters),
    ):
        for split, rows in (("train", train_rows), ("heldout", heldout_rows)):
            payloads.extend((model_id, split, row, parameters, replay_dt_s, alignment_window_s) for row in rows)
    if int(workers) > 1 and len(payloads) > 1:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            nested = list(executor.map(stage_replay_sample_rows_payload, payloads))
    else:
        nested = [stage_replay_sample_rows_payload(payload) for payload in payloads]
    sample_rows = [row for rows in nested for row in rows]
    return summarize_stage_replay_samples(sample_rows)


def stage_replay_sample_rows_payload(payload: tuple[str, str, dict[str, Any], dict[str, float], float, float]) -> list[dict[str, Any]]:
    model_id, split, row, parameters, replay_dt_s, alignment_window_s = payload
    throw_dir = Path(str(row.get("_throw_dir", "")))
    if not throw_dir.exists():
        return []
    sample_rows = prep._read_csv(throw_dir / "metrics" / "state_samples.csv")
    if not sample_rows:
        return []
    aligned = prep._aligned_state_from_sample_rows(sample_rows, alignment_window_s)
    if aligned.get("status") != "ok":
        return []
    x0 = np.asarray(aligned["state"], dtype=float)
    if not np.all(np.isfinite(x0)):
        return []

    t_first = prep._float(sample_rows[0], "t_s", 0.0)
    rel_times = np.asarray([prep._float(sample, "t_s", t_first) - t_first for sample in sample_rows], dtype=float)
    alignment_elapsed_s = float(aligned["alignment_elapsed_s"])
    aircraft = replay_fit.calibrated_aircraft(parameters)
    actuator_tau_s = prep._actuator_tau_from_manifest(prep._throw_manifest(throw_dir))
    neutral_command = np.zeros(3, dtype=float)

    x = x0.copy()
    sim_t_s = 0.0
    out: list[dict[str, Any]] = []
    for sample, rel_time in zip(sample_rows, rel_times):
        if not math.isfinite(float(rel_time)) or float(rel_time) < alignment_elapsed_s - 1e-12:
            continue
        target_t_s = float(rel_time) - alignment_elapsed_s
        while sim_t_s < target_t_s - 1e-12:
            dt_s = min(float(replay_dt_s), target_t_s - sim_t_s)
            try:
                x = replay_fit.rk4_step(x, neutral_command, aircraft, actuator_tau_s, dt_s)
            except Exception:
                return out
            sim_t_s += dt_s
            if not np.all(np.isfinite(x)):
                return out
        if target_t_s <= 1e-9:
            continue
        actual = prep._state_vector_from_sample_row(sample)
        if not np.all(np.isfinite(actual)):
            continue
        try:
            actual_loads = evaluate_state(actual, neutral_command, aircraft, wind_model=None, rho=RHO_KG_M3, wind_mode="panel")
        except Exception:
            continue
        regime = alpha_regime(math.degrees(float(actual_loads["alpha_rad"])))
        actual_altitude_loss = float(x0[2] - actual[2])
        sim_altitude_loss = float(x0[2] - x[2])
        out.append(
            {
                "model_id": model_id,
                "split": split,
                "session_label": row.get("session_label", ""),
                "throw_id": row.get("throw_id", ""),
                "regime": regime,
                "t_since_alignment_s": target_t_s,
                "dx_residual_actual_minus_sim_m": float((actual[0] - x0[0]) - (x[0] - x0[0])),
                "dy_residual_actual_minus_sim_m": float((actual[1] - x0[1]) - (x[1] - x0[1])),
                "altitude_loss_residual_actual_minus_sim_m": float(actual_altitude_loss - sim_altitude_loss),
                "sink_rate_residual_actual_minus_sim_m_s": prep._ratio(actual_altitude_loss - sim_altitude_loss, target_t_s),
                "roll_residual_actual_minus_sim_deg": replay_fit.angular_residual_deg(math.degrees(float(actual[3])), math.degrees(float(x[3]))),
                "pitch_residual_actual_minus_sim_deg": replay_fit.angular_residual_deg(math.degrees(float(actual[4])), math.degrees(float(x[4]))),
                "yaw_residual_actual_minus_sim_deg": replay_fit.angular_residual_deg(math.degrees(float(actual[5])), math.degrees(float(x[5]))),
            }
        )
    return out


def summarize_stage_replay_samples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model_id in ("baseline_active", "coefficient_candidate"):
        for split in ("train", "heldout"):
            for regime in ("attached", "transition", "post_stall"):
                subset = [
                    row
                    for row in rows
                    if row.get("model_id") == model_id
                    and row.get("split") == split
                    and row.get("regime") == regime
                ]
                if not subset:
                    output.append(
                        {
                            "model_id": model_id,
                            "split": split,
                            "regime": regime,
                            "sample_count": 0,
                            "throw_count": 0,
                        }
                    )
                    continue
                throw_keys = {(row.get("session_label", ""), row.get("throw_id", "")) for row in subset}
                output.append(
                    {
                        "model_id": model_id,
                        "split": split,
                        "regime": regime,
                        "sample_count": len(subset),
                        "throw_count": len(throw_keys),
                        "dx_mae_m": mae(finite_values(subset, "dx_residual_actual_minus_sim_m")),
                        "dy_mae_m": mae(finite_values(subset, "dy_residual_actual_minus_sim_m")),
                        "altitude_loss_mae_m": mae(finite_values(subset, "altitude_loss_residual_actual_minus_sim_m")),
                        "sink_rate_mae_m_s": mae(finite_values(subset, "sink_rate_residual_actual_minus_sim_m_s")),
                        "roll_mae_deg": mae(finite_values(subset, "roll_residual_actual_minus_sim_deg")),
                        "pitch_mae_deg": mae(finite_values(subset, "pitch_residual_actual_minus_sim_deg")),
                        "yaw_mae_deg": mae(finite_values(subset, "yaw_residual_actual_minus_sim_deg")),
                    }
                )
    return output


def coefficient_output_rows(
    fit_result: dict[str, Any],
    apply_attached_cm_bias: bool,
    fit_post_stall_damping: bool,
    fit_lateral_surfaces: bool,
) -> list[dict[str, Any]]:
    coeffs = fit_result.get("coefficients", zero_coefficients())
    rows = [
        {
            "parameter": "post_stall_surface_replay_scale",
            "value": float(
                fit_result.get("surface_scale_selection", {}).get("selected_surface_scale", 1.0)
            ),
            "applied_to_replay": True,
            "description": "Train-replay-selected multiplier applied to all fitted alpha-RBF post-stall surface coefficients.",
        },
        {
            "parameter": "attached_cm_bias_coeff",
            "value": float(coeffs.get("attached_cm_bias_coeff", 0.0)),
            "applied_to_replay": bool(apply_attached_cm_bias),
            "description": "Attached-regime Cm residual. Report-only by default.",
        },
        {
            "parameter": "transition_cm_bias_coeff",
            "value": float(coeffs.get("transition_cm_bias_coeff", 0.0)),
            "applied_to_replay": False,
            "description": "All-transition aggregate Cm diagnostic. Not applied to replay.",
        },
        {
            "parameter": "transition_before_post_stall_cm_bias_coeff",
            "value": float(coeffs.get("transition_before_post_stall_cm_bias_coeff", 0.0)),
            "applied_to_replay": False,
            "description": "Transition Cm diagnostic before first post-stall exposure in each throw.",
        },
        {
            "parameter": "transition_after_post_stall_cm_bias_coeff",
            "value": float(coeffs.get("transition_after_post_stall_cm_bias_coeff", 0.0)),
            "applied_to_replay": False,
            "description": "Transition Cm diagnostic after the throw has already entered post-stall.",
        },
        {
            "parameter": "post_stall_lift_residual_coeff",
            "value": float(coeffs.get("post_stall_lift_residual_coeff", 0.0)),
            "applied_to_replay": False,
            "description": "Legacy scalar CL residual. Disabled when alpha-RBF post-stall surface fit is active.",
        },
        {
            "parameter": "post_stall_drag_residual_coeff",
            "value": float(coeffs.get("post_stall_drag_residual_coeff", 0.0)),
            "applied_to_replay": False,
            "description": "Legacy scalar CD residual. Disabled when alpha-RBF post-stall surface fit is active.",
        },
        {
            "parameter": "post_stall_pitch_moment_coeff",
            "value": float(coeffs.get("post_stall_pitch_moment_coeff", 0.0)),
            "applied_to_replay": False,
            "description": "Legacy scalar Cm residual. Disabled when alpha-RBF post-stall surface fit is active.",
        },
        {
            "parameter": "post_stall_pitch_damping_coeff",
            "value": float(coeffs.get("post_stall_pitch_damping_coeff", 0.0)),
            "applied_to_replay": bool(fit_post_stall_damping),
            "description": "Positive-AoA Cmq-style residual using q_hat = q c / (2V). Opt-in because it is derivative-sensitive.",
        },
        {
            "parameter": "post_stall_residual_blend_start_alpha_deg",
            "value": float(coeffs.get("post_stall_residual_blend_start_alpha_deg", STALL_ALPHA_DEG)),
            "applied_to_replay": True,
            "description": "Residual activation start alpha fitted after attached and post-stall coefficients.",
        },
        {
            "parameter": "post_stall_residual_blend_full_alpha_deg",
            "value": float(coeffs.get("post_stall_residual_blend_full_alpha_deg", POST_STALL_ALPHA_DEG)),
            "applied_to_replay": True,
            "description": "Residual activation full-alpha point fitted from transition behaviour while preserving post-stall full activation.",
        },
    ]
    for prefix, coefficient_name in (
        ("post_stall_lift_rbf", "CL"),
        ("post_stall_drag_rbf", "CD"),
        ("post_stall_pitch_moment_rbf", "Cm"),
        ("post_stall_pitch_damping_rbf", "Cmq"),
    ):
        for centre_deg in SURFACE_RBF_ALPHA_CENTERS_DEG:
            key = surface_rbf_parameter_name(prefix, centre_deg)
            rows.append(
                {
                    "parameter": key,
                    "value": float(coeffs.get(key, 0.0)),
                    "applied_to_replay": True,
                    "description": (
                        f"Neutral post-stall {coefficient_name} residual alpha-RBF coefficient "
                        f"at {centre_deg:g} deg AoA."
                    ),
                }
            )
    for prefix, coefficient_name in (
        ("post_stall_side_force", "CY"),
        ("post_stall_roll_moment", "Cl"),
        ("post_stall_yaw_moment", "Cn"),
    ):
        for feature in LATERAL_SURFACE_FEATURES:
            for centre_deg in SURFACE_RBF_ALPHA_CENTERS_DEG:
                key = lateral_surface_parameter_name(prefix, feature, centre_deg)
                rows.append(
                    {
                        "parameter": key,
                        "value": float(coeffs.get(key, 0.0)),
                        "applied_to_replay": bool(fit_lateral_surfaces),
                        "description": (
                            f"Neutral post-stall {coefficient_name} residual coefficient multiplying "
                            f"{feature} at {centre_deg:g} deg AoA."
                        ),
                    }
                )
    return rows


def summarize_regimes(rows: list[dict[str, Any]], fit_result: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for split in ("train", "heldout"):
        for regime in ("attached", "transition", "post_stall"):
            subset = [row for row in rows if row.get("split") == split and row.get("residual_status") == "ok" and row.get("regime") == regime]
            out.append(regime_summary_row(split, regime, subset, fit_result))
    return out


def summarize_stage_fit_groups(rows: list[dict[str, Any]], fit_result: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for split in ("train", "heldout"):
        for stage_fit_group in ("attached", "transition_before_post_stall", "transition_after_post_stall", "post_stall"):
            subset = [
                row
                for row in rows
                if row.get("split") == split
                and row.get("residual_status") == "ok"
                and row.get("stage_fit_group") == stage_fit_group
            ]
            out.append(stage_fit_summary_row(split, stage_fit_group, subset, fit_result))
    return out


def regime_summary_row(split: str, regime: str, rows: list[dict[str, Any]], fit_result: dict[str, Any]) -> dict[str, Any]:
    if not rows:
        return {"split": split, "regime": regime, "count": 0}
    alpha = finite_values(rows, "alpha_deg")
    cm = finite_values(rows, "cm_residual")
    cd = finite_values(rows, "cd_residual")
    cl = finite_values(rows, "cl_residual")
    cy = finite_values(rows, "cy_residual")
    cl_roll = finite_values(rows, "cl_roll_residual")
    cn_yaw = finite_values(rows, "cn_yaw_residual")
    q_hat = finite_values(rows, "q_hat")
    fit_resid = [cm_fit_residual(row, fit_result) for row in rows]
    fit_resid = [value for value in fit_resid if math.isfinite(value)]
    return {
        "split": split,
        "regime": regime,
        "count": int(len(rows)),
        "alpha_min_deg": min(alpha) if alpha else float("nan"),
        "alpha_max_deg": max(alpha) if alpha else float("nan"),
        "cm_residual_mean": mean(cm),
        "cm_residual_mae": mae(cm),
        "cm_fit_residual_mae": mae(fit_resid),
        "cd_residual_mean": mean(cd),
        "cl_residual_mean": mean(cl),
        "cy_residual_mean": mean(cy),
        "cl_roll_residual_mean": mean(cl_roll),
        "cn_yaw_residual_mean": mean(cn_yaw),
        "q_hat_mean": mean(q_hat),
    }


def stage_fit_summary_row(
    split: str,
    stage_fit_group: str,
    rows: list[dict[str, Any]],
    fit_result: dict[str, Any],
) -> dict[str, Any]:
    if not rows:
        return {"split": split, "stage_fit_group": stage_fit_group, "count": 0}
    alpha = finite_values(rows, "alpha_deg")
    cm = finite_values(rows, "cm_residual")
    cd = finite_values(rows, "cd_residual")
    cl = finite_values(rows, "cl_residual")
    cy = finite_values(rows, "cy_residual")
    cl_roll = finite_values(rows, "cl_roll_residual")
    cn_yaw = finite_values(rows, "cn_yaw_residual")
    q_hat = finite_values(rows, "q_hat")
    fit_resid = [cm_fit_residual(row, fit_result) for row in rows]
    fit_resid = [value for value in fit_resid if math.isfinite(value)]
    return {
        "split": split,
        "stage_fit_group": stage_fit_group,
        "count": int(len(rows)),
        "alpha_min_deg": min(alpha) if alpha else float("nan"),
        "alpha_max_deg": max(alpha) if alpha else float("nan"),
        "cm_residual_mean": mean(cm),
        "cm_residual_mae": mae(cm),
        "cm_fit_residual_mae": mae(fit_resid),
        "cd_residual_mean": mean(cd),
        "cl_residual_mean": mean(cl),
        "cy_residual_mean": mean(cy),
        "cl_roll_residual_mean": mean(cl_roll),
        "cn_yaw_residual_mean": mean(cn_yaw),
        "q_hat_mean": mean(q_hat),
    }


def cm_fit_residual(row: dict[str, Any], fit_result: dict[str, Any]) -> float:
    cm = finite_value(row.get("cm_residual"))
    if not math.isfinite(cm):
        return float("nan")
    coeffs = fit_result.get("coefficients", zero_coefficients())
    fitted = 0.0
    stage_fit_group = str(row.get("stage_fit_group", ""))
    if stage_fit_group == "attached":
        fitted += float(coeffs.get("attached_cm_bias_coeff", 0.0))
    elif row.get("regime") in {"transition", "post_stall"}:
        alpha_deg = finite_value(row.get("alpha_deg"))
        activation = residual_blend_activation_from_coeffs(alpha_deg, coeffs)
        q_hat = finite_value(row.get("q_hat"))
        fitted_surface = surface_rbf_prediction(coeffs, "post_stall_pitch_moment_rbf", alpha_deg)
        damping_surface = surface_rbf_prediction(coeffs, "post_stall_pitch_damping_rbf", alpha_deg)
        if math.isfinite(fitted_surface):
            fitted += fitted_surface
        if math.isfinite(activation):
            if math.isfinite(q_hat):
                fitted += (
                    float(coeffs.get("post_stall_pitch_damping_coeff", 0.0)) * activation
                    + (damping_surface if math.isfinite(damping_surface) else 0.0)
                ) * q_hat
    return cm - fitted


def residual_blend_activation_from_coeffs(alpha_deg: float, coeffs: dict[str, float]) -> float:
    if not math.isfinite(float(alpha_deg)):
        return float("nan")
    return residual_blend_activation_deg(
        float(alpha_deg),
        float(coeffs.get("post_stall_residual_blend_start_alpha_deg", STALL_ALPHA_DEG)),
        float(coeffs.get("post_stall_residual_blend_full_alpha_deg", POST_STALL_ALPHA_DEG)),
    )


def finite_value(value: Any) -> float:
    converted = prep._to_float(value)
    return float(converted) if converted is not None and math.isfinite(converted) else float("nan")


def finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [value for value in (finite_value(row.get(key)) for row in rows) if math.isfinite(value)]


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def mae(values: list[float]) -> float:
    return float(sum(abs(value) for value in values) / len(values)) if values else float("nan")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: replay_fit.format_value(row.get(field, "")) for field in fieldnames})


def write_manifest(
    output_dir: Path,
    *,
    run_label: str,
    session_root: Path,
    valid_rows: list[dict[str, Any]],
    heldout_indices: set[int],
    heldout_seed: int,
    alignment_window_s: float,
    derivative_window_s: float,
    replay_dt_s: float,
    ridge_lambda: float,
    min_speed_m_s: float,
    workers: int,
    apply_attached_cm_bias: bool,
    fit_post_stall_damping: bool,
    fit_lateral_surfaces: bool,
    base_parameters: dict[str, float],
    candidate_parameters: dict[str, float],
    fit_result: dict[str, Any],
) -> None:
    manifest = {
        "fit_id": str(run_label),
        "fit_version": FIT_VERSION,
        "fit_scope": "neutral_30_open_loop_vicon_6dof_force_moment_residual_independent_stage_fit",
        "session_root": str(session_root),
        "valid_throw_count": len(valid_rows),
        "heldout_policy": "randomised_stratified_by_session_label",
        "heldout_seed": int(heldout_seed),
        "heldout_indices": sorted(int(index) for index in heldout_indices),
        "alignment_window_s": float(alignment_window_s),
        "derivative_window_s": float(derivative_window_s),
        "replay_dt_s": float(replay_dt_s),
        "ridge_lambda": float(ridge_lambda),
        "min_speed_m_s": float(min_speed_m_s),
        "workers": int(workers),
        "apply_attached_cm_bias": bool(apply_attached_cm_bias),
        "fit_post_stall_damping": bool(fit_post_stall_damping),
        "fit_lateral_surfaces": bool(fit_lateral_surfaces),
        "stage_fit_policy": {
            "attached": "direct Cm residual diagnostic; not applied unless explicitly requested",
            "transition_before_post_stall": "direct Cm residual diagnostic before first post-stall exposure; not applied",
            "transition_after_post_stall": "direct Cm residual diagnostic after post-stall exposure; not applied",
            "post_stall": "neutral CL/CD/Cm/Cmq and lateral-directional CY/Cl/Cn residual alpha-RBF coefficient surfaces; applied to replay candidate through smooth alpha activation",
            "transition_blender": "fits only the residual activation start/full alpha after post-stall surfaces are fixed",
        },
        "base_parameters": dict(base_parameters),
        "candidate_parameters": dict(candidate_parameters),
        "fit_result": fit_result,
        "rerun_command": fit_rerun_command(
            run_label=run_label,
            session_root=session_root,
            heldout_count=len(heldout_indices),
            heldout_seed=heldout_seed,
            alignment_window_s=alignment_window_s,
            derivative_window_s=derivative_window_s,
            replay_dt_s=replay_dt_s,
            ridge_lambda=ridge_lambda,
            min_speed_m_s=min_speed_m_s,
            workers=workers,
            apply_attached_cm_bias=apply_attached_cm_bias,
            fit_post_stall_damping=fit_post_stall_damping,
            fit_lateral_surfaces=fit_lateral_surfaces,
        ),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    path = output_dir / "manifests" / "neutral_aero_residual_fit_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def fit_rerun_command(
    *,
    run_label: str,
    session_root: Path,
    heldout_count: int,
    heldout_seed: int,
    alignment_window_s: float,
    derivative_window_s: float,
    replay_dt_s: float,
    ridge_lambda: float,
    min_speed_m_s: float,
    workers: int,
    apply_attached_cm_bias: bool,
    fit_post_stall_damping: bool,
    fit_lateral_surfaces: bool,
) -> list[str]:
    command = [
        "python",
        "03_Control/02_Inner_Loop/run_fit_neutral_aero_residual_calibration.py",
        "--session-root",
        str(session_root),
        "--run-label",
        str(run_label),
        "--heldout-count",
        str(int(heldout_count)),
        "--heldout-seed",
        str(int(heldout_seed)),
        "--alignment-window-s",
        f"{float(alignment_window_s):.6g}",
        "--derivative-window-s",
        f"{float(derivative_window_s):.6g}",
        "--replay-dt-s",
        f"{float(replay_dt_s):.6g}",
        "--ridge-lambda",
        f"{float(ridge_lambda):.6g}",
        "--min-speed-m-s",
        f"{float(min_speed_m_s):.6g}",
        "--workers",
        str(int(workers)),
    ]
    command.append("--apply-attached-cm-bias" if apply_attached_cm_bias else "--no-apply-attached-cm-bias")
    command.append("--fit-post-stall-damping" if fit_post_stall_damping else "--no-fit-post-stall-damping")
    command.append("--fit-lateral-surfaces" if fit_lateral_surfaces else "--no-fit-lateral-surfaces")
    return command


def write_report(
    output_dir: Path,
    *,
    run_label: str,
    session_root: Path,
    heldout_count: int,
    heldout_seed: int,
    alignment_window_s: float,
    derivative_window_s: float,
    replay_dt_s: float,
    ridge_lambda: float,
    min_speed_m_s: float,
    workers: int,
    apply_attached_cm_bias: bool,
    fit_post_stall_damping: bool,
    fit_lateral_surfaces: bool,
    fit_result: dict[str, Any],
    regime_summary: list[dict[str, Any]],
    stage_fit_summary: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    stage_replay_rows: list[dict[str, Any]],
    base_parameters: dict[str, float],
    candidate_parameters: dict[str, float],
) -> None:
    baseline_train = replay_summary(validation_rows, "baseline_active", "train")
    baseline_heldout = replay_summary(validation_rows, "baseline_active", "heldout")
    candidate_train = replay_summary(validation_rows, "coefficient_candidate", "train")
    candidate_heldout = replay_summary(validation_rows, "coefficient_candidate", "heldout")
    coeffs = fit_result.get("coefficients", zero_coefficients())
    blend_fit = fit_result.get("transition_blender_fit", {})
    surface_scale_selection = fit_result.get("surface_scale_selection", {})
    rerun_command = replay_fit.powershell_command_line(
        fit_rerun_command(
            run_label=run_label,
            session_root=session_root,
            heldout_count=heldout_count,
            heldout_seed=heldout_seed,
            alignment_window_s=alignment_window_s,
            derivative_window_s=derivative_window_s,
            replay_dt_s=replay_dt_s,
            ridge_lambda=ridge_lambda,
            min_speed_m_s=min_speed_m_s,
            workers=workers,
            apply_attached_cm_bias=apply_attached_cm_bias,
            fit_post_stall_damping=fit_post_stall_damping,
            fit_lateral_surfaces=fit_lateral_surfaces,
        )
    )
    lines = [
        "# Neutral Aero Residual Regime Fit",
        "",
        "This run uses only neutral open-loop real throws. It estimates 6-DoF force/moment residuals from Vicon state trajectories, keeps attached and transition Cm offsets report-only, fits compact post-stall alpha-RBF residual surfaces for CL/CD/Cm/Cmq plus CY/Cl/Cn lateral-directional coupling, then validates the candidate by held-out dry-air replay.",
        "",
        "## Rerun Recipe",
        "",
        f"- source session root: `{session_root}`",
        f"- alignment window: `{alignment_window_s:.3f}` s",
        f"- derivative window: `{derivative_window_s:.3f}` s",
        f"- replay dt: `{replay_dt_s:.4f}` s",
        f"- ridge lambda: `{ridge_lambda:.3g}`",
        f"- min speed: `{min_speed_m_s:.2f}` m/s",
        f"- workers: `{workers}`",
        f"- apply attached Cm bias: `{apply_attached_cm_bias}`",
        f"- fit post-stall damping: `{fit_post_stall_damping}`",
        f"- fit lateral surfaces: `{fit_lateral_surfaces}`",
        "",
        "```powershell",
        rerun_command,
        "```",
        "",
        "## Coefficient Fit",
        "",
        f"- fit status: `{fit_result.get('status', '')}`",
        f"- sample count: `{fit_result.get('sample_count', 0)}`",
        f"- used sample count: `{fit_result.get('used_sample_count', 0)}`",
        f"- post-stall used sample count: `{fit_result.get('post_stall_used_sample_count', 0)}`",
        f"- fit MAE in Cm: `{float(fit_result.get('fit_mae_cm', float('nan'))):.5f}`",
        f"- attached Cm residual: `{float(coeffs.get('attached_cm_bias_coeff', 0.0)):.6g}`",
        f"- transition Cm residual before post-stall: `{float(coeffs.get('transition_before_post_stall_cm_bias_coeff', 0.0)):.6g}`",
        f"- transition Cm residual after post-stall: `{float(coeffs.get('transition_after_post_stall_cm_bias_coeff', 0.0)):.6g}`",
        f"- post-stall surface centres: `{', '.join(f'{centre:g}' for centre in SURFACE_RBF_ALPHA_CENTERS_DEG)}` deg",
        f"- post-stall surface width: `{SURFACE_RBF_ALPHA_WIDTH_DEG:.3g}` deg",
        surface_coeff_report_lines(coeffs),
        f"- post-stall Cmq residual: `{float(coeffs.get('post_stall_pitch_damping_coeff', 0.0)):.6g}`",
        f"- selected post-stall surface replay scale: `{float(surface_scale_selection.get('selected_surface_scale', 1.0)):.3f}`",
        f"- transition blender status: `{blend_fit.get('status', '')}`",
        f"- transition blender fit group: `{blend_fit.get('fit_group', '')}`",
        f"- transition blender start alpha: `{float(coeffs.get('post_stall_residual_blend_start_alpha_deg', STALL_ALPHA_DEG)):.3f}` deg",
        f"- transition blender full alpha: `{float(coeffs.get('post_stall_residual_blend_full_alpha_deg', POST_STALL_ALPHA_DEG)):.3f}` deg",
        "",
        "## Replay Validation",
        "",
        f"- baseline train pitch MAE: `{baseline_train['final_theta_mae_deg']:.3f}` deg",
        f"- candidate train pitch MAE: `{candidate_train['final_theta_mae_deg']:.3f}` deg",
        f"- baseline held-out pitch MAE: `{baseline_heldout['final_theta_mae_deg']:.3f}` deg",
        f"- candidate held-out pitch MAE: `{candidate_heldout['final_theta_mae_deg']:.3f}` deg",
        f"- baseline held-out altitude-loss MAE: `{baseline_heldout['altitude_loss_mae_m']:.4f}` m",
        f"- candidate held-out altitude-loss MAE: `{candidate_heldout['altitude_loss_mae_m']:.4f}` m",
        f"- baseline held-out dx MAE: `{baseline_heldout['dx_mae_m']:.4f}` m",
        f"- candidate held-out dx MAE: `{candidate_heldout['dx_mae_m']:.4f}` m",
        "",
        "## Stage Replay Errors",
        "",
        "These are sample-aligned replay residuals grouped by the measured Vicon alpha regime. Use them to see whether the candidate fixes attached, transition, or post-stall behaviour; keep the full-throw held-out replay gate as the acceptance criterion.",
        "",
        stage_replay_report_lines(stage_replay_rows),
        "",
        "## Candidate Parameters",
        "",
        f"- legacy scalar post-stall CL residual: baseline `{base_parameters['post_stall_lift_residual_coeff']:.6g}`, candidate `{candidate_parameters['post_stall_lift_residual_coeff']:.6g}`",
        f"- legacy scalar post-stall CD residual: baseline `{base_parameters['post_stall_drag_residual_coeff']:.6g}`, candidate `{candidate_parameters['post_stall_drag_residual_coeff']:.6g}`",
        f"- legacy scalar post-stall Cm residual: baseline `{base_parameters['post_stall_pitch_moment_coeff']:.6g}`, candidate `{candidate_parameters['post_stall_pitch_moment_coeff']:.6g}`",
        candidate_surface_report_lines(base_parameters, candidate_parameters),
        f"- baseline post-stall Cmq: `{base_parameters['post_stall_pitch_damping_coeff']:.6g}`",
        f"- candidate post-stall Cmq: `{candidate_parameters['post_stall_pitch_damping_coeff']:.6g}`",
        f"- baseline residual blend start alpha: `{base_parameters['post_stall_residual_blend_start_alpha_deg']:.3f}` deg",
        f"- candidate residual blend start alpha: `{candidate_parameters['post_stall_residual_blend_start_alpha_deg']:.3f}` deg",
        f"- baseline residual blend full alpha: `{base_parameters['post_stall_residual_blend_full_alpha_deg']:.3f}` deg",
        f"- candidate residual blend full alpha: `{candidate_parameters['post_stall_residual_blend_full_alpha_deg']:.3f}` deg",
        "",
        "## Regime Summary",
        "",
        regime_report_lines(regime_summary),
        "",
        "## Independent Stage Fit Summary",
        "",
        "Transition is split into samples before and after the first post-stall exposure in each throw. This separates real transition-model evidence from transition samples that may already be contaminated by earlier post-stall divergence.",
        "",
        stage_fit_report_lines(stage_fit_summary),
        "",
        "## Interpretation",
        "",
        "Accept the candidate only if held-out replay improves pitch/q and lateral trajectory behaviour without damaging x, altitude loss, or sink. Attached and transition Cm residuals are diagnostic-only by default; accepted high-AoA changes should enter through the compact alpha-dependent post-stall static, pitch-damping, and lateral-directional residual surfaces.",
    ]
    path = output_dir / "reports" / "neutral_aero_residual_fit_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def replay_summary(rows: list[dict[str, Any]], model_id: str, split: str) -> dict[str, float]:
    subset = [row for row in rows if row.get("model_id") == model_id and row.get("split") == split]
    return replay_fit.objective_summary(subset)


def surface_coeff_report_lines(coeffs: dict[str, float]) -> str:
    lines = []
    for prefix, label in (
        ("post_stall_lift_rbf", "CL"),
        ("post_stall_drag_rbf", "CD"),
        ("post_stall_pitch_moment_rbf", "Cm"),
        ("post_stall_pitch_damping_rbf", "Cmq"),
    ):
        values = [
            f"{centre:g} deg `{float(coeffs.get(surface_rbf_parameter_name(prefix, centre), 0.0)):.6g}`"
            for centre in SURFACE_RBF_ALPHA_CENTERS_DEG
        ]
        lines.append(f"- post-stall {label} surface: " + ", ".join(values))
    for prefix, label in (
        ("post_stall_side_force", "CY"),
        ("post_stall_roll_moment", "Cl"),
        ("post_stall_yaw_moment", "Cn"),
    ):
        values = []
        for feature in LATERAL_SURFACE_FEATURES:
            entries = [
                f"{centre:g} deg `{float(coeffs.get(lateral_surface_parameter_name(prefix, feature, centre), 0.0)):.6g}`"
                for centre in SURFACE_RBF_ALPHA_CENTERS_DEG
            ]
            values.append(f"{feature}: " + ", ".join(entries))
        lines.append(f"- post-stall {label} surface: " + "; ".join(values))
    return "\n".join(lines)


def candidate_surface_report_lines(base_parameters: dict[str, float], candidate_parameters: dict[str, float]) -> str:
    lines = []
    for prefix, label in (
        ("post_stall_lift_rbf", "CL"),
        ("post_stall_drag_rbf", "CD"),
        ("post_stall_pitch_moment_rbf", "Cm"),
        ("post_stall_pitch_damping_rbf", "Cmq"),
    ):
        values = []
        for centre in SURFACE_RBF_ALPHA_CENTERS_DEG:
            key = surface_rbf_parameter_name(prefix, centre)
            values.append(
                f"{centre:g} deg baseline `{float(base_parameters.get(key, 0.0)):.6g}` -> "
                f"candidate `{float(candidate_parameters.get(key, 0.0)):.6g}`"
            )
        lines.append(f"- post-stall {label} surface: " + ", ".join(values))
    for prefix, label in (
        ("post_stall_side_force", "CY"),
        ("post_stall_roll_moment", "Cl"),
        ("post_stall_yaw_moment", "Cn"),
    ):
        values = []
        for feature in LATERAL_SURFACE_FEATURES:
            entries = []
            for centre in SURFACE_RBF_ALPHA_CENTERS_DEG:
                key = lateral_surface_parameter_name(prefix, feature, centre)
                entries.append(
                    f"{centre:g} deg baseline `{float(base_parameters.get(key, 0.0)):.6g}` -> "
                    f"candidate `{float(candidate_parameters.get(key, 0.0)):.6g}`"
                )
            values.append(f"{feature}: " + ", ".join(entries))
        lines.append(f"- post-stall {label} surface: " + "; ".join(values))
    return "\n".join(lines)


def regime_report_lines(regime_summary: list[dict[str, Any]]) -> str:
    lines = []
    for row in regime_summary:
        lines.append(
            f"- {row.get('split', '')}/{row.get('regime', '')}: count `{row.get('count', 0)}`, "
            f"Cm mean `{finite_value(row.get('cm_residual_mean')):.5f}`, "
            f"Cm MAE `{finite_value(row.get('cm_residual_mae')):.5f}`, "
            f"CY mean `{finite_value(row.get('cy_residual_mean')):.5f}`, "
            f"Cl mean `{finite_value(row.get('cl_roll_residual_mean')):.5f}`, "
            f"Cn mean `{finite_value(row.get('cn_yaw_residual_mean')):.5f}`"
        )
    return "\n".join(lines)


def stage_fit_report_lines(stage_fit_summary: list[dict[str, Any]]) -> str:
    lines = []
    for row in stage_fit_summary:
        lines.append(
            f"- {row.get('split', '')}/{row.get('stage_fit_group', '')}: count `{row.get('count', 0)}`, "
            f"Cm mean `{finite_value(row.get('cm_residual_mean')):.5f}`, "
            f"Cm MAE `{finite_value(row.get('cm_residual_mae')):.5f}`, "
            f"Cm fit residual MAE `{finite_value(row.get('cm_fit_residual_mae')):.5f}`, "
            f"CY mean `{finite_value(row.get('cy_residual_mean')):.5f}`, "
            f"Cl mean `{finite_value(row.get('cl_roll_residual_mean')):.5f}`, "
            f"Cn mean `{finite_value(row.get('cn_yaw_residual_mean')):.5f}`"
        )
    return "\n".join(lines)


def stage_replay_report_lines(stage_replay_rows: list[dict[str, Any]]) -> str:
    lines = []
    for split in ("train", "heldout"):
        for regime in ("attached", "transition", "post_stall"):
            baseline = stage_replay_row(stage_replay_rows, "baseline_active", split, regime)
            candidate = stage_replay_row(stage_replay_rows, "coefficient_candidate", split, regime)
            if int(baseline.get("sample_count", 0) or 0) <= 0 and int(candidate.get("sample_count", 0) or 0) <= 0:
                lines.append(f"- {split}/{regime}: no replay samples")
                continue
            lines.append(
                f"- {split}/{regime}: samples `{candidate.get('sample_count', baseline.get('sample_count', 0))}`, "
                f"throws `{candidate.get('throw_count', baseline.get('throw_count', 0))}`"
            )
            lines.append(
                f"  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: "
                f"`{finite_value(baseline.get('dx_mae_m')):.3f}` m, "
                f"`{finite_value(baseline.get('dy_mae_m')):.3f}` m, "
                f"`{finite_value(baseline.get('altitude_loss_mae_m')):.3f}` m, "
                f"`{finite_value(baseline.get('sink_rate_mae_m_s')):.3f}` m/s, "
                f"`{finite_value(baseline.get('roll_mae_deg')):.2f}` deg, "
                f"`{finite_value(baseline.get('pitch_mae_deg')):.2f}` deg, "
                f"`{finite_value(baseline.get('yaw_mae_deg')):.2f}` deg"
            )
            lines.append(
                f"  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: "
                f"`{finite_value(candidate.get('dx_mae_m')):.3f}` m, "
                f"`{finite_value(candidate.get('dy_mae_m')):.3f}` m, "
                f"`{finite_value(candidate.get('altitude_loss_mae_m')):.3f}` m, "
                f"`{finite_value(candidate.get('sink_rate_mae_m_s')):.3f}` m/s, "
                f"`{finite_value(candidate.get('roll_mae_deg')):.2f}` deg, "
                f"`{finite_value(candidate.get('pitch_mae_deg')):.2f}` deg, "
                f"`{finite_value(candidate.get('yaw_mae_deg')):.2f}` deg"
            )
    return "\n".join(lines)


def stage_replay_row(stage_replay_rows: list[dict[str, Any]], model_id: str, split: str, regime: str) -> dict[str, Any]:
    for row in stage_replay_rows:
        if row.get("model_id") == model_id and row.get("split") == split and row.get("regime") == regime:
            return row
    return {"model_id": model_id, "split": split, "regime": regime, "sample_count": 0, "throw_count": 0}


if __name__ == "__main__":
    main()
