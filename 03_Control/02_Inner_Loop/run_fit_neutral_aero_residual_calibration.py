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


FIT_VERSION = "N13_neutral_post_stall_residual_blend"
DEFAULT_SESSION_ROOT = REPO_ROOT / "04_Flight_Test" / "05_Results" / "cal" / "n30"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "03_Control" / "05_Results" / "glider_model_calibration_prep"
DEFAULT_RUN_LABEL = "n30_neutral_aero_residual_fit"
DEFAULT_WORKERS = 8
DEFAULT_HELDOUT_COUNT = 2
DEFAULT_HELDOUT_SEED = 606
DEFAULT_ALIGNMENT_WINDOW_S = 0.10
DEFAULT_DERIVATIVE_WINDOW_S = 0.040
DEFAULT_REPLAY_DT_S = 0.005
DEFAULT_RIDGE_LAMBDA = 1.0e-3
DEFAULT_MIN_SPEED_M_S = 1.5
RHO_KG_M3 = 1.225
STALL_ALPHA_DEG = float(math.degrees(STALL_BLEND_ALPHA_RAD))
POST_STALL_ALPHA_DEG = 20.0


AERO_RESIDUAL_FIELDS = [
    "split",
    "session_label",
    "case_id",
    "throw_id",
    "residual_status",
    "sample_index",
    "t_since_alignment_s",
    "regime",
    "speed_m_s",
    "q_bar_pa",
    "alpha_deg",
    "q_rad_s",
    "q_hat",
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
    "cm_required",
    "cm_model",
    "cm_residual",
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
    "q_hat_mean",
]
COEFFICIENT_FIELDS = ["parameter", "value", "applied_to_replay", "description"]
REPLAY_VALIDATION_FIELDS = ["model_id", *replay_fit.REPLAY_RESIDUAL_FIELDS]


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
        default=False,
        help="Fit and apply a post-stall Cmq residual. Disabled by default because q-dot residuals are noise-sensitive.",
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
    )
    candidate_parameters = candidate_from_fit(
        base_parameters,
        fit_result,
        apply_attached_cm_bias=apply_attached_cm_bias,
        fit_post_stall_damping=fit_post_stall_damping,
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
    all_residuals = train_residuals + heldout_residuals
    regime_summary = summarize_regimes(all_residuals, fit_result)
    coefficient_rows = coefficient_output_rows(
        fit_result,
        apply_attached_cm_bias,
        fit_post_stall_damping,
    )

    write_csv(output_dir / "metrics" / "neutral_aero_residual_samples.csv", all_residuals, AERO_RESIDUAL_FIELDS)
    write_csv(output_dir / "metrics" / "neutral_aero_residual_regime_summary.csv", regime_summary, REGIME_SUMMARY_FIELDS)
    write_csv(output_dir / "metrics" / "neutral_aero_residual_fit_coefficients.csv", coefficient_rows, COEFFICIENT_FIELDS)
    write_csv(output_dir / "metrics" / "neutral_aero_residual_replay_validation.csv", validation_rows, REPLAY_VALIDATION_FIELDS)
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
        fit_result=fit_result,
        regime_summary=regime_summary,
        validation_rows=validation_rows,
        base_parameters=base_parameters,
        candidate_parameters=candidate_parameters,
    )
    return output_dir


def active_parameter_dict() -> dict[str, float]:
    return {
        "cd0_strip_scale": float(active_calibration.CD0_STRIP_SCALE),
        "drag_area_fuse_scale": float(active_calibration.DRAG_AREA_FUSE_SCALE),
        "efficiency_strip_scale": float(active_calibration.EFFICIENCY_STRIP_SCALE),
        "roll_moment_bias_coeff": float(getattr(active_calibration, "ROLL_MOMENT_BIAS_COEFF", 0.0)),
        "pitch_moment_bias_coeff": float(getattr(active_calibration, "PITCH_MOMENT_BIAS_COEFF", 0.0)),
        "post_stall_lift_residual_coeff": float(getattr(active_calibration, "POST_STALL_LIFT_RESIDUAL_COEFF", 0.0)),
        "post_stall_drag_residual_coeff": float(getattr(active_calibration, "POST_STALL_DRAG_RESIDUAL_COEFF", 0.0)),
        "post_stall_pitch_moment_coeff": float(getattr(active_calibration, "POST_STALL_PITCH_MOMENT_COEFF", 0.0)),
        "post_stall_pitch_damping_coeff": float(getattr(active_calibration, "POST_STALL_PITCH_DAMPING_COEFF", 0.0)),
        "post_stall_tail_effectiveness_drop": float(getattr(active_calibration, "POST_STALL_TAIL_EFFECTIVENESS_DROP", 0.0)),
        "yaw_moment_bias_coeff": float(getattr(active_calibration, "YAW_MOMENT_BIAS_COEFF", 0.0)),
        "delta_a_trim_rad": float(getattr(active_calibration, "DELTA_A_TRIM_RAD", 0.0)),
        "delta_e_trim_rad": float(getattr(active_calibration, "DELTA_E_TRIM_RAD", 0.0)),
        "delta_r_trim_rad": float(getattr(active_calibration, "DELTA_R_TRIM_RAD", 0.0)),
    }


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
        if force_denom <= 1e-9 or pitch_moment_denom <= 1e-9:
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
        q_hat = float(loads.get("pitch_rate_hat", x[10] * aircraft.c_ref_m / (2.0 * speed_m_s)))
        post_stall_activation = float(
            loads.get(
                "post_stall_residual_activation",
                loads.get(
                    "post_stall_pitch_activation",
                    post_stall_residual_activation_numpy(float(loads["alpha_rad"])),
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
                "regime": alpha_regime(alpha_deg),
                "speed_m_s": speed_m_s,
                "q_bar_pa": float(q_bar),
                "alpha_deg": alpha_deg,
                "q_rad_s": float(x[10]),
                "q_hat": q_hat,
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
                "cm_required": float(m_aero_required_b[1] / pitch_moment_denom),
                "cm_model": float(m_model_b[1] / pitch_moment_denom),
                "cm_residual": float(m_residual_b[1] / pitch_moment_denom),
                "pitch_moment_required_n_m": float(m_aero_required_b[1]),
                "pitch_moment_model_n_m": float(m_model_b[1]),
                "pitch_moment_residual_n_m": float(m_residual_b[1]),
                "force_residual_norm_n": float(np.linalg.norm(f_residual_b)),
                "angular_accel_q_rad_s2": float(omega_dot_b[1]),
            }
        )
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


def fit_pitch_residual_coefficients(
    rows: list[dict[str, Any]],
    *,
    ridge_lambda: float,
    fit_post_stall_damping: bool,
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
                "q_bar": finite_value(row.get("q_bar_pa")),
                "q_hat": finite_value(row.get("q_hat")),
                "activation": finite_value(row.get("post_stall_activation")),
                "attached": 1.0 if row.get("regime") == "attached" else 0.0,
                "transition": 1.0 if row.get("regime") == "transition" else 0.0,
            }
        )
    if len(samples) < 8:
        return {"status": "too_few_samples", "sample_count": len(samples), "coefficients": zero_coefficients()}

    y = np.asarray([sample["cm"] for sample in samples], dtype=float)
    x = np.asarray(
        [
            [
                sample["attached"],
                sample["transition"],
                sample["activation"],
                sample["activation"] * sample["q_hat"] if fit_post_stall_damping else 0.0,
            ]
            for sample in samples
        ],
        dtype=float,
    )
    q_bar = np.asarray([sample["q_bar"] for sample in samples], dtype=float)
    q_bar_median = float(np.nanmedian(q_bar)) if np.any(np.isfinite(q_bar)) else 1.0
    weights = np.sqrt(np.clip(q_bar / max(q_bar_median, 1e-9), 0.25, 4.0))
    mask = np.ones(len(y), dtype=bool)
    coeffs = np.zeros(4, dtype=float)
    for _ in range(2):
        coeffs = weighted_ridge_fit(x[mask], y[mask], weights[mask], float(ridge_lambda))
        residual = y - x @ coeffs
        sigma = robust_sigma(residual[mask])
        if not math.isfinite(sigma) or sigma <= 1e-9:
            break
        next_mask = np.abs(residual - float(np.nanmedian(residual[mask]))) <= 4.0 * sigma
        if int(np.count_nonzero(next_mask)) < 8:
            break
        mask = next_mask
    residual_after = y - x @ coeffs
    cl_coeff = fit_activated_scalar_residual(samples, "cl", ridge_lambda=ridge_lambda)
    cd_coeff = fit_activated_scalar_residual(samples, "cd", ridge_lambda=ridge_lambda)
    return {
        "status": "ok",
        "sample_count": int(len(y)),
        "used_sample_count": int(np.count_nonzero(mask)),
        "ridge_lambda": float(ridge_lambda),
        "coefficients": {
            "attached_cm_bias_coeff": float(coeffs[0]),
            "transition_cm_bias_coeff": float(coeffs[1]),
            "post_stall_lift_residual_coeff": float(cl_coeff),
            "post_stall_drag_residual_coeff": float(cd_coeff),
            "post_stall_pitch_moment_coeff": float(coeffs[2]),
            "post_stall_pitch_damping_coeff": float(coeffs[3]),
        },
        "fit_rmse_cm": float(np.sqrt(np.mean(residual_after[mask] ** 2))),
        "fit_mae_cm": float(np.mean(np.abs(residual_after[mask]))),
        "cm_residual_mean_before": float(np.mean(y[mask])),
    }


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
    q_bar_median = float(np.nanmedian(q_bar)) if np.any(np.isfinite(q_bar)) else 1.0
    weights = np.sqrt(np.clip(q_bar / max(q_bar_median, 1e-9), 0.25, 4.0))
    coeff = weighted_ridge_fit(x, y, weights, float(ridge_lambda))
    return float(coeff[0])


def zero_coefficients() -> dict[str, float]:
    return {
        "attached_cm_bias_coeff": 0.0,
        "transition_cm_bias_coeff": 0.0,
        "post_stall_lift_residual_coeff": 0.0,
        "post_stall_drag_residual_coeff": 0.0,
        "post_stall_pitch_moment_coeff": 0.0,
        "post_stall_pitch_damping_coeff": 0.0,
    }


def weighted_ridge_fit(x: np.ndarray, y: np.ndarray, weights: np.ndarray, ridge_lambda: float) -> np.ndarray:
    w = np.asarray(weights, dtype=float).reshape(-1, 1)
    xw = x * w
    yw = y * w[:, 0]
    return np.linalg.solve(xw.T @ xw + ridge_lambda * np.eye(x.shape[1]), xw.T @ yw)


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
    return candidate


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


def coefficient_output_rows(
    fit_result: dict[str, Any],
    apply_attached_cm_bias: bool,
    fit_post_stall_damping: bool,
) -> list[dict[str, Any]]:
    coeffs = fit_result.get("coefficients", zero_coefficients())
    return [
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
            "description": "Transition-regime diagnostic only. Transition replay uses the smooth post-stall activation blend.",
        },
        {
            "parameter": "post_stall_lift_residual_coeff",
            "value": float(coeffs.get("post_stall_lift_residual_coeff", 0.0)),
            "applied_to_replay": True,
            "description": "Positive-AoA whole-aircraft CL residual.",
        },
        {
            "parameter": "post_stall_drag_residual_coeff",
            "value": float(coeffs.get("post_stall_drag_residual_coeff", 0.0)),
            "applied_to_replay": True,
            "description": "Positive-AoA whole-aircraft CD residual.",
        },
        {
            "parameter": "post_stall_pitch_moment_coeff",
            "value": float(coeffs.get("post_stall_pitch_moment_coeff", 0.0)),
            "applied_to_replay": True,
            "description": "Positive-AoA post-stall Cm residual.",
        },
        {
            "parameter": "post_stall_pitch_damping_coeff",
            "value": float(coeffs.get("post_stall_pitch_damping_coeff", 0.0)),
            "applied_to_replay": bool(fit_post_stall_damping),
            "description": "Positive-AoA Cmq-style residual using q_hat = q c / (2V). Opt-in because it is derivative-sensitive.",
        },
    ]


def summarize_regimes(rows: list[dict[str, Any]], fit_result: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for split in ("train", "heldout"):
        for regime in ("attached", "transition", "post_stall"):
            subset = [row for row in rows if row.get("split") == split and row.get("residual_status") == "ok" and row.get("regime") == regime]
            out.append(regime_summary_row(split, regime, subset, fit_result))
    return out


def regime_summary_row(split: str, regime: str, rows: list[dict[str, Any]], fit_result: dict[str, Any]) -> dict[str, Any]:
    if not rows:
        return {"split": split, "regime": regime, "count": 0}
    alpha = finite_values(rows, "alpha_deg")
    cm = finite_values(rows, "cm_residual")
    cd = finite_values(rows, "cd_residual")
    cl = finite_values(rows, "cl_residual")
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
        "q_hat_mean": mean(q_hat),
    }


def cm_fit_residual(row: dict[str, Any], fit_result: dict[str, Any]) -> float:
    cm = finite_value(row.get("cm_residual"))
    if not math.isfinite(cm):
        return float("nan")
    coeffs = fit_result.get("coefficients", zero_coefficients())
    fitted = 0.0
    if row.get("regime") == "attached":
        fitted += float(coeffs.get("attached_cm_bias_coeff", 0.0))
    if row.get("regime") == "transition":
        fitted += float(coeffs.get("transition_cm_bias_coeff", 0.0))
    activation = finite_value(row.get("post_stall_activation"))
    q_hat = finite_value(row.get("q_hat"))
    if math.isfinite(activation):
        fitted += float(coeffs.get("post_stall_pitch_moment_coeff", 0.0)) * activation
        if math.isfinite(q_hat):
            fitted += float(coeffs.get("post_stall_pitch_damping_coeff", 0.0)) * activation * q_hat
    return cm - fitted


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
    base_parameters: dict[str, float],
    candidate_parameters: dict[str, float],
    fit_result: dict[str, Any],
) -> None:
    manifest = {
        "fit_id": str(run_label),
        "fit_version": FIT_VERSION,
        "fit_scope": "neutral_30_open_loop_vicon_force_moment_residual_regime_fit",
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
    fit_result: dict[str, Any],
    regime_summary: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    base_parameters: dict[str, float],
    candidate_parameters: dict[str, float],
) -> None:
    baseline_train = replay_summary(validation_rows, "baseline_active", "train")
    baseline_heldout = replay_summary(validation_rows, "baseline_active", "heldout")
    candidate_train = replay_summary(validation_rows, "coefficient_candidate", "train")
    candidate_heldout = replay_summary(validation_rows, "coefficient_candidate", "heldout")
    coeffs = fit_result.get("coefficients", zero_coefficients())
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
        )
    )
    lines = [
        "# Neutral Aero Residual Regime Fit",
        "",
        "This run uses only neutral open-loop real throws. It estimates force/moment residuals from Vicon state trajectories, keeps attached-flow corrections report-only, uses transition only as a smooth activation band, and validates post-stall residual candidates by held-out dry-air replay.",
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
        f"- fit MAE in Cm: `{float(fit_result.get('fit_mae_cm', float('nan'))):.5f}`",
        f"- attached Cm residual: `{float(coeffs.get('attached_cm_bias_coeff', 0.0)):.6g}`",
        f"- transition Cm residual: `{float(coeffs.get('transition_cm_bias_coeff', 0.0)):.6g}`",
        f"- post-stall CL residual: `{float(coeffs.get('post_stall_lift_residual_coeff', 0.0)):.6g}`",
        f"- post-stall CD residual: `{float(coeffs.get('post_stall_drag_residual_coeff', 0.0)):.6g}`",
        f"- post-stall Cm residual: `{float(coeffs.get('post_stall_pitch_moment_coeff', 0.0)):.6g}`",
        f"- post-stall Cmq residual: `{float(coeffs.get('post_stall_pitch_damping_coeff', 0.0)):.6g}`",
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
        "## Candidate Parameters",
        "",
        f"- baseline post-stall CL residual: `{base_parameters['post_stall_lift_residual_coeff']:.6g}`",
        f"- candidate post-stall CL residual: `{candidate_parameters['post_stall_lift_residual_coeff']:.6g}`",
        f"- baseline post-stall CD residual: `{base_parameters['post_stall_drag_residual_coeff']:.6g}`",
        f"- candidate post-stall CD residual: `{candidate_parameters['post_stall_drag_residual_coeff']:.6g}`",
        f"- baseline post-stall Cm: `{base_parameters['post_stall_pitch_moment_coeff']:.6g}`",
        f"- candidate post-stall Cm: `{candidate_parameters['post_stall_pitch_moment_coeff']:.6g}`",
        f"- baseline post-stall Cmq: `{base_parameters['post_stall_pitch_damping_coeff']:.6g}`",
        f"- candidate post-stall Cmq: `{candidate_parameters['post_stall_pitch_damping_coeff']:.6g}`",
        "",
        "## Regime Summary",
        "",
        regime_report_lines(regime_summary),
        "",
        "## Interpretation",
        "",
        "Accept the candidate only if held-out replay improves pitch/q behaviour without damaging x, altitude loss, or sink. Attached and transition residuals are diagnostic-only by default; accepted model changes should enter through the smoothly activated post-stall residual terms.",
    ]
    path = output_dir / "reports" / "neutral_aero_residual_fit_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def replay_summary(rows: list[dict[str, Any]], model_id: str, split: str) -> dict[str, float]:
    subset = [row for row in rows if row.get("model_id") == model_id and row.get("split") == split]
    return replay_fit.objective_summary(subset)


def regime_report_lines(regime_summary: list[dict[str, Any]]) -> str:
    lines = []
    for row in regime_summary:
        lines.append(
            f"- {row.get('split', '')}/{row.get('regime', '')}: count `{row.get('count', 0)}`, "
            f"Cm mean `{finite_value(row.get('cm_residual_mean')):.5f}`, "
            f"Cm MAE `{finite_value(row.get('cm_residual_mae')):.5f}`"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
