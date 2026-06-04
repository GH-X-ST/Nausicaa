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
import warnings
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


FIT_VERSION = "N19_regime_separated_longitudinal_cm"
DEFAULT_FIT_WORKFLOW = "cm_regime_staged"
DEFAULT_SESSION_ROOT = REPO_ROOT / "04_Flight_Test" / "05_Results" / "cal" / "n30"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "03_Control" / "05_Results" / "glider_model_calibration_prep"
DEFAULT_RUN_LABEL = "n30_neutral_aero_residual_fit"
DEFAULT_WORKERS = 8
DEFAULT_HELDOUT_COUNT = 0
DEFAULT_HELDOUT_FRACTION = 0.15
DEFAULT_HELDOUT_SEED = 606
DEFAULT_ALIGNMENT_WINDOW_S = 0.10
DEFAULT_DERIVATIVE_WINDOW_S = 0.040
DEFAULT_REPLAY_DT_S = 0.005
DEFAULT_RIDGE_LAMBDA = 1.0e-3
DEFAULT_MIN_SPEED_M_S = 1.5
DEFAULT_FILTER_ALIGNED_LAUNCH_STATE = True
# Replay starts 0.10 s after first motion, so u may have decayed from the
# hardware launch-gate value. Keep this as a relaxed post-alignment sanity
# bound, not a second copy of the real launch gate.
DEFAULT_ALIGNED_U_MIN_M_S = 3.0
DEFAULT_ALIGNED_U_MAX_M_S = 8.0
DEFAULT_ALIGNED_V_ABS_MAX_M_S = 1.5
DEFAULT_ALIGNED_W_ABS_MAX_M_S = 0.9
DEFAULT_ALIGNED_ROLL_ABS_MAX_DEG = 20.0
DEFAULT_ALIGNED_YAW_ABS_MAX_DEG = 20.0
DEFAULT_ALIGNED_P_ABS_MAX_RAD_S = 1.2
DEFAULT_ALIGNED_Q_ABS_MAX_RAD_S = 1.2
DEFAULT_ALIGNED_R_ABS_MAX_RAD_S = 1.8
DEFAULT_LAUNCH_CONFIDENCE_MIN_WEIGHT = 0.25
DEFAULT_LAUNCH_CONFIDENCE_EXPONENT = 1.5
DEFAULT_GROUP_ITERATIONS = 3
DEFAULT_GROUP_IMPROVEMENT_TOL = 1.0e-3
DEFAULT_APPLY_ATTACHED_CM_BIAS = False
DEFAULT_FIT_TRANSITION_PITCH_MOMENT = False
DEFAULT_FIT_POST_STALL_LIFT_DRAG = True
DEFAULT_FIT_POST_STALL_PITCH_MOMENT = True
DEFAULT_FIT_POST_STALL_LONGITUDINAL = True
DEFAULT_FIT_TRANSITION_BLENDER = True
DEFAULT_FIT_POST_STALL_SURFACES = False
DEFAULT_FIT_POST_STALL_DAMPING = True
DEFAULT_FIT_ATTACHED_LATERAL_COUPLING = False
DEFAULT_FIT_TRANSITION_LATERAL_COUPLING = False
DEFAULT_FIT_LATERAL_SURFACES = False
DEFAULT_FIT_SECONDARY_LATERAL_DIAGNOSTIC = True
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
LATERAL_COUPLING_GROUPS = ("side_force", "roll_moment", "yaw_moment")
LATERAL_COUPLING_FEATURE_NAMES = ("bias", "beta", "p_hat", "r_hat")
MINIMAL_LATERAL_FEATURES_BY_GROUP = {
    "side_force": ("beta",),
    "roll_moment": ("p_hat",),
    "yaw_moment": ("r_hat",),
}
MINIMAL_ATTACHED_LATERAL_PARAMETER_KEYS = (
    "side_force_beta_coeff",
    "roll_moment_p_hat_coeff",
    "yaw_moment_r_hat_coeff",
)
MINIMAL_TRANSITION_LATERAL_PARAMETER_KEYS = (
    "transition_side_force_beta_coeff",
    "transition_roll_moment_p_hat_coeff",
    "transition_yaw_moment_r_hat_coeff",
)
ATTACHED_LATERAL_PARAMETER_KEYS = (
    "side_force_bias_coeff",
    "side_force_beta_coeff",
    "side_force_p_hat_coeff",
    "side_force_r_hat_coeff",
    "roll_moment_bias_coeff",
    "roll_moment_beta_coeff",
    "roll_moment_p_hat_coeff",
    "roll_moment_r_hat_coeff",
    "yaw_moment_bias_coeff",
    "yaw_moment_beta_coeff",
    "yaw_moment_p_hat_coeff",
    "yaw_moment_r_hat_coeff",
)
TRANSITION_LATERAL_PARAMETER_KEYS = (
    "transition_side_force_bias_coeff",
    "transition_side_force_beta_coeff",
    "transition_side_force_p_hat_coeff",
    "transition_side_force_r_hat_coeff",
    "transition_roll_moment_bias_coeff",
    "transition_roll_moment_beta_coeff",
    "transition_roll_moment_p_hat_coeff",
    "transition_roll_moment_r_hat_coeff",
    "transition_yaw_moment_bias_coeff",
    "transition_yaw_moment_beta_coeff",
    "transition_yaw_moment_p_hat_coeff",
    "transition_yaw_moment_r_hat_coeff",
)
GROUP_SCALE_CANDIDATES = (-0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
GROUPED_FIT_GROUP_ORDER = (
    "attached_longitudinal",
    "transition_pitch_moment",
    "post_stall_pitch_moment",
    "post_stall_pitch_damping",
    "post_stall_lift_drag",
    "attached_lateral",
    "post_stall_longitudinal",
    "transition_lateral",
    "post_stall_lateral",
    "transition_blender",
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
    "launch_lateral_score",
    "launch_confidence_weight",
    "u0_m_s",
    "v0_m_s",
    "w0_m_s",
    "p0_rad_s",
    "q0_rad_s",
    "r0_rad_s",
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
GROUP_ITERATION_FIELDS = [
    "iteration",
    "pass_index",
    "group",
    "candidate_scale",
    "selected",
    "objective",
    "dx_mae_m",
    "dy_mae_m",
    "altitude_loss_mae_m",
    "sink_rate_mae_m_s",
    "final_theta_mae_deg",
    "final_phi_mae_deg",
    "final_psi_mae_deg",
    "group_scales_json",
]
CM_STAGE_HISTORY_FIELDS = [
    "stage_index",
    "stage_id",
    "fit_parameter_group",
    "accepted",
    "acceptance_reason",
    "train_objective_before",
    "train_objective_after",
    "heldout_objective_before",
    "heldout_objective_after",
    "heldout_dx_mae_before_m",
    "heldout_dx_mae_after_m",
    "heldout_altitude_loss_mae_before_m",
    "heldout_altitude_loss_mae_after_m",
    "heldout_sink_mae_before_m_s",
    "heldout_sink_mae_after_m_s",
    "heldout_pitch_mae_before_deg",
    "heldout_pitch_mae_after_deg",
    "attached_sample_fraction",
    "transition_sample_fraction",
    "post_stall_sample_fraction",
    "parameter_updates_json",
]
REPLAY_VALIDATION_FIELDS = ["model_id", *replay_fit.REPLAY_RESIDUAL_FIELDS]
FILTERED_THROW_FIELDS = [
    "row_index",
    "kept",
    "filter_reason",
    "session_label",
    "case_id",
    "throw_id",
    "alignment_window_s",
    "alignment_elapsed_s",
    "phi0_deg",
    "psi0_deg",
    "u0_m_s",
    "v0_m_s",
    "w0_m_s",
    "speed0_m_s",
    "p0_rad_s",
    "q0_rad_s",
    "r0_rad_s",
    "launch_lateral_score",
    "launch_confidence_weight",
]
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
        heldout_fraction=args.heldout_fraction,
        heldout_seed=args.heldout_seed,
        alignment_window_s=args.alignment_window_s,
        derivative_window_s=args.derivative_window_s,
        replay_dt_s=args.replay_dt_s,
        ridge_lambda=args.ridge_lambda,
        min_speed_m_s=args.min_speed_m_s,
        workers=args.workers,
        fit_workflow=args.fit_workflow,
        group_iterations=args.group_iterations,
        group_improvement_tol=args.group_improvement_tol,
        filter_aligned_launch_state=args.filter_aligned_launch_state,
        aligned_u_min_m_s=args.aligned_u_min_m_s,
        aligned_u_max_m_s=args.aligned_u_max_m_s,
        aligned_v_abs_max_m_s=args.aligned_v_abs_max_m_s,
        aligned_w_abs_max_m_s=args.aligned_w_abs_max_m_s,
        apply_attached_cm_bias=args.apply_attached_cm_bias,
        fit_transition_pitch_moment=args.fit_transition_pitch_moment,
        fit_post_stall_lift_drag=args.fit_post_stall_lift_drag,
        fit_post_stall_pitch_moment=args.fit_post_stall_pitch_moment,
        fit_post_stall_longitudinal=args.fit_post_stall_longitudinal,
        fit_transition_blender=args.fit_transition_blender,
        fit_post_stall_surfaces=args.fit_post_stall_surfaces,
        fit_post_stall_damping=args.fit_post_stall_damping,
        fit_attached_lateral_coupling=args.fit_attached_lateral_coupling,
        fit_transition_lateral_coupling=args.fit_transition_lateral_coupling,
        fit_lateral_surfaces=args.fit_lateral_surfaces,
        fit_secondary_lateral_diagnostic=args.fit_secondary_lateral_diagnostic,
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
    parser.add_argument(
        "--heldout-count",
        type=int,
        default=DEFAULT_HELDOUT_COUNT,
        help="Held-out throw count. Use 0 to choose round(heldout_fraction * filtered_valid_count).",
    )
    parser.add_argument("--heldout-fraction", type=float, default=DEFAULT_HELDOUT_FRACTION)
    parser.add_argument("--heldout-seed", type=int, default=DEFAULT_HELDOUT_SEED)
    parser.add_argument("--alignment-window-s", type=float, default=DEFAULT_ALIGNMENT_WINDOW_S)
    parser.add_argument("--derivative-window-s", type=float, default=DEFAULT_DERIVATIVE_WINDOW_S)
    parser.add_argument("--replay-dt-s", type=float, default=DEFAULT_REPLAY_DT_S)
    parser.add_argument("--ridge-lambda", type=float, default=DEFAULT_RIDGE_LAMBDA)
    parser.add_argument("--min-speed-m-s", type=float, default=DEFAULT_MIN_SPEED_M_S)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--fit-workflow",
        choices=("cm_regime_staged", "grouped_iterative", "residual_only"),
        default=DEFAULT_FIT_WORKFLOW,
        help=(
            "cm_regime_staged is the default claim workflow. grouped_iterative keeps the older block-scale "
            "coordinate refinement; residual_only applies the direct residual coefficients."
        ),
    )
    parser.add_argument("--group-iterations", type=int, default=DEFAULT_GROUP_ITERATIONS)
    parser.add_argument("--group-improvement-tol", type=float, default=DEFAULT_GROUP_IMPROVEMENT_TOL)
    parser.add_argument(
        "--filter-aligned-launch-state",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_FILTER_ALIGNED_LAUNCH_STATE,
        help=(
            "Exclude throws whose replay-aligned start is outside the post-alignment sanity envelope. "
            "The lower u bound is intentionally relaxed relative to the real launch gate because replay "
            "starts after the alignment delay."
        ),
    )
    parser.add_argument("--aligned-u-min-m-s", type=float, default=DEFAULT_ALIGNED_U_MIN_M_S)
    parser.add_argument("--aligned-u-max-m-s", type=float, default=DEFAULT_ALIGNED_U_MAX_M_S)
    parser.add_argument("--aligned-v-abs-max-m-s", type=float, default=DEFAULT_ALIGNED_V_ABS_MAX_M_S)
    parser.add_argument("--aligned-w-abs-max-m-s", type=float, default=DEFAULT_ALIGNED_W_ABS_MAX_M_S)
    parser.add_argument(
        "--apply-attached-cm-bias",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_APPLY_ATTACHED_CM_BIAS,
        help="Apply attached-regime Cm residual to the attached-weighted pitch-moment term.",
    )
    parser.add_argument(
        "--fit-transition-pitch-moment",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_FIT_TRANSITION_PITCH_MOMENT,
        help="Apply the transition-weighted Cm residual term. The staged workflow enables this only at its transition stage.",
    )
    parser.add_argument(
        "--fit-post-stall-lift-drag",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_FIT_POST_STALL_LIFT_DRAG,
        help="Apply compact scalar post-stall CL/CD residuals when post-stall longitudinal fitting is enabled.",
    )
    parser.add_argument(
        "--fit-post-stall-pitch-moment",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_FIT_POST_STALL_PITCH_MOMENT,
        help="Apply compact scalar post-stall Cm residuals when post-stall longitudinal fitting is enabled.",
    )
    parser.add_argument(
        "--fit-post-stall-longitudinal",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_FIT_POST_STALL_LONGITUDINAL,
        help="Apply compact scalar post-stall CL/CD/Cm/Cmq residuals. Disable for attached/transition-only ablations.",
    )
    parser.add_argument(
        "--fit-transition-blender",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_FIT_TRANSITION_BLENDER,
        help="Apply fitted residual activation start/full alpha. Disable when freezing transition-blend timing.",
    )
    parser.add_argument(
        "--fit-post-stall-surfaces",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_FIT_POST_STALL_SURFACES,
        help=(
            "Fit multi-centre post-stall CL/CD/Cm/Cmq alpha-RBF surfaces. "
            "Disabled by default; the default promoted path uses compact scalar post-stall residuals."
        ),
    )
    parser.add_argument(
        "--fit-post-stall-damping",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_FIT_POST_STALL_DAMPING,
        help="Fit and apply alpha-dependent post-stall Cmq residuals. Keep disabled only for static-surface ablations.",
    )
    parser.add_argument(
        "--fit-attached-lateral-coupling",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_FIT_ATTACHED_LATERAL_COUPLING,
        help=(
            "Apply the compact attached lateral set CY_beta, Cl_p, and Cn_r inside the primary candidate. "
            "Disabled by default; use the secondary lateral diagnostic for claim-safe lateral checks."
        ),
    )
    parser.add_argument(
        "--fit-transition-lateral-coupling",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_FIT_TRANSITION_LATERAL_COUPLING,
        help="Diagnostic only by default: fit compact transition-window CY_beta, Cl_p, and Cn_r deltas.",
    )
    parser.add_argument(
        "--fit-lateral-surfaces",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_FIT_LATERAL_SURFACES,
        help="Diagnostic only by default: fit and apply post-stall CY/Cl/Cn residual surfaces.",
    )
    parser.add_argument(
        "--fit-secondary-lateral-diagnostic",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_FIT_SECONDARY_LATERAL_DIAGNOSTIC,
        help=(
            "After the primary longitudinal candidate is fixed, fit a separate diagnostic CY_beta/Cl_p/Cn_r "
            "candidate with longitudinal terms frozen and launch-confidence weighting disabled."
        ),
    )
    return parser


def run_fit(
    *,
    session_root: Path,
    output_root: Path,
    run_label: str,
    heldout_count: int,
    heldout_fraction: float,
    heldout_seed: int,
    alignment_window_s: float,
    derivative_window_s: float,
    replay_dt_s: float,
    ridge_lambda: float,
    min_speed_m_s: float,
    workers: int,
    fit_workflow: str,
    group_iterations: int,
    group_improvement_tol: float,
    filter_aligned_launch_state: bool,
    aligned_u_min_m_s: float,
    aligned_u_max_m_s: float,
    aligned_v_abs_max_m_s: float,
    aligned_w_abs_max_m_s: float,
    apply_attached_cm_bias: bool,
    fit_transition_pitch_moment: bool,
    fit_post_stall_lift_drag: bool,
    fit_post_stall_pitch_moment: bool,
    fit_post_stall_longitudinal: bool,
    fit_transition_blender: bool,
    fit_post_stall_surfaces: bool,
    fit_post_stall_damping: bool,
    fit_attached_lateral_coupling: bool,
    fit_transition_lateral_coupling: bool,
    fit_lateral_surfaces: bool,
    fit_secondary_lateral_diagnostic: bool,
) -> Path:
    output_dir = output_root / run_label
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded_rows = load_neutral_rows(session_root)
    valid_rows, filtered_throw_rows = filter_aligned_launch_rows(
        loaded_rows,
        alignment_window_s=alignment_window_s,
        enabled=filter_aligned_launch_state,
        u_min_m_s=aligned_u_min_m_s,
        u_max_m_s=aligned_u_max_m_s,
        v_abs_max_m_s=aligned_v_abs_max_m_s,
        w_abs_max_m_s=aligned_w_abs_max_m_s,
    )
    if len(valid_rows) < 4:
        raise FileNotFoundError(
            "Need at least 4 neutral throws after aligned replay-start filtering; "
            f"loaded {len(loaded_rows)} and kept {len(valid_rows)}."
        )
    requested_heldout_count = int(heldout_count)
    if requested_heldout_count <= 0:
        requested_heldout_count = int(round(float(heldout_fraction) * len(valid_rows)))
    requested_heldout_count = int(np.clip(requested_heldout_count, 1, max(1, len(valid_rows) - 1)))
    heldout_indices = prep.stratified_heldout_indices(
        valid_rows,
        heldout_count=requested_heldout_count,
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
        fit_post_stall_surfaces=fit_post_stall_surfaces,
        fit_post_stall_damping=fit_post_stall_damping,
        fit_attached_lateral_coupling=fit_attached_lateral_coupling,
        fit_transition_lateral_coupling=fit_transition_lateral_coupling,
        fit_lateral_surfaces=fit_lateral_surfaces,
        lateral_use_confidence_weights=True,
    )
    surface_scale_rows = (
        []
        if fit_workflow == "cm_regime_staged"
        else select_surface_scale_rows(
            base_parameters=base_parameters,
            fit_result=fit_result,
            train_rows=train_rows,
            replay_dt_s=replay_dt_s,
            alignment_window_s=alignment_window_s,
            workers=workers,
            apply_attached_cm_bias=apply_attached_cm_bias,
            fit_transition_pitch_moment=fit_transition_pitch_moment,
            fit_post_stall_lift_drag=fit_post_stall_lift_drag,
            fit_post_stall_pitch_moment=fit_post_stall_pitch_moment,
            fit_post_stall_longitudinal=fit_post_stall_longitudinal,
            fit_transition_blender=fit_transition_blender,
            fit_post_stall_surfaces=fit_post_stall_surfaces,
            fit_post_stall_damping=fit_post_stall_damping,
            fit_attached_lateral_coupling=fit_attached_lateral_coupling,
            fit_transition_lateral_coupling=fit_transition_lateral_coupling,
            fit_lateral_surfaces=fit_lateral_surfaces,
        )
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
        fit_transition_pitch_moment=fit_transition_pitch_moment,
        fit_post_stall_lift_drag=fit_post_stall_lift_drag,
        fit_post_stall_pitch_moment=fit_post_stall_pitch_moment,
        fit_post_stall_longitudinal=fit_post_stall_longitudinal,
        fit_transition_blender=fit_transition_blender,
        fit_post_stall_surfaces=fit_post_stall_surfaces,
        fit_post_stall_damping=fit_post_stall_damping,
        fit_attached_lateral_coupling=fit_attached_lateral_coupling,
        fit_transition_lateral_coupling=fit_transition_lateral_coupling,
        fit_lateral_surfaces=fit_lateral_surfaces,
        surface_scale=selected_surface_scale,
    )
    group_iteration_rows: list[dict[str, Any]] = []
    cm_stage_history_rows: list[dict[str, Any]] = []
    if fit_workflow == "cm_regime_staged":
        candidate_parameters, cm_stage_history_rows = cm_regime_staged_refinement(
            base_parameters=base_parameters,
            train_rows=train_rows,
            heldout_rows=heldout_rows,
            replay_dt_s=replay_dt_s,
            alignment_window_s=alignment_window_s,
            derivative_window_s=derivative_window_s,
            min_speed_m_s=min_speed_m_s,
            ridge_lambda=ridge_lambda,
            workers=workers,
            fit_post_stall_damping=fit_post_stall_damping,
            fit_transition_blender=fit_transition_blender,
            fit_post_stall_lift_drag=fit_post_stall_lift_drag,
            fit_post_stall_surfaces=fit_post_stall_surfaces,
        )
        fit_result["cm_regime_staged"] = {
            "enabled": True,
            "history_csv": "metrics/neutral_aero_residual_cm_stage_history.csv",
            "stage_count": len(cm_stage_history_rows),
        }
    elif fit_workflow == "grouped_iterative":
        candidate_parameters, group_iteration_rows = grouped_iterative_refinement(
            base_parameters=base_parameters,
            fit_result=fit_result,
            train_rows=train_rows,
            replay_dt_s=replay_dt_s,
            alignment_window_s=alignment_window_s,
            workers=workers,
            apply_attached_cm_bias=apply_attached_cm_bias,
            fit_transition_pitch_moment=fit_transition_pitch_moment,
            fit_post_stall_lift_drag=fit_post_stall_lift_drag,
            fit_post_stall_pitch_moment=fit_post_stall_pitch_moment,
            fit_post_stall_longitudinal=fit_post_stall_longitudinal,
            fit_transition_blender=fit_transition_blender,
            fit_post_stall_surfaces=fit_post_stall_surfaces,
            fit_post_stall_damping=fit_post_stall_damping,
            fit_attached_lateral_coupling=fit_attached_lateral_coupling,
            fit_transition_lateral_coupling=fit_transition_lateral_coupling,
            fit_lateral_surfaces=fit_lateral_surfaces,
            initial_surface_scale=selected_surface_scale,
            group_iterations=group_iterations,
            improvement_tol=group_improvement_tol,
        )
    elif fit_workflow != "residual_only":
        raise ValueError(f"Unsupported fit workflow: {fit_workflow}")
    lateral_diagnostic_result, lateral_diagnostic_parameters = secondary_lateral_diagnostic_candidate(
        train_rows=train_rows,
        primary_parameters=candidate_parameters,
        alignment_window_s=alignment_window_s,
        derivative_window_s=derivative_window_s,
        min_speed_m_s=min_speed_m_s,
        ridge_lambda=ridge_lambda,
        workers=workers,
        enabled=fit_secondary_lateral_diagnostic,
    )
    extra_validation_models = []
    if lateral_diagnostic_parameters is not None:
        extra_validation_models.append(("lateral_diagnostic_candidate", lateral_diagnostic_parameters))
    validation_rows = replay_validation_rows(
        train_rows=train_rows,
        heldout_rows=heldout_rows,
        base_parameters=base_parameters,
        candidate_parameters=candidate_parameters,
        extra_models=extra_validation_models,
        replay_dt_s=replay_dt_s,
        alignment_window_s=alignment_window_s,
        workers=workers,
    )
    acceptance_summary = candidate_acceptance_summary(validation_rows)
    lateral_diagnostic_acceptance = lateral_diagnostic_acceptance_summary(validation_rows)
    stage_replay_rows = stage_replay_summary_rows(
        train_rows=train_rows,
        heldout_rows=heldout_rows,
        base_parameters=base_parameters,
        candidate_parameters=candidate_parameters,
        extra_models=extra_validation_models,
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
        fit_transition_pitch_moment,
        fit_post_stall_lift_drag,
        fit_post_stall_pitch_moment,
        fit_post_stall_longitudinal,
        fit_transition_blender,
        fit_post_stall_surfaces,
        fit_post_stall_damping,
        fit_attached_lateral_coupling,
        fit_transition_lateral_coupling,
        fit_lateral_surfaces,
        base_parameters=base_parameters,
        candidate_parameters=candidate_parameters,
    )

    write_csv(output_dir / "metrics" / "neutral_aero_residual_samples.csv", all_residuals, AERO_RESIDUAL_FIELDS)
    write_csv(output_dir / "metrics" / "neutral_aero_residual_regime_summary.csv", regime_summary, REGIME_SUMMARY_FIELDS)
    write_csv(output_dir / "metrics" / "neutral_aero_residual_stage_fit_summary.csv", stage_fit_summary, STAGE_FIT_SUMMARY_FIELDS)
    write_csv(output_dir / "metrics" / "neutral_aero_residual_fit_coefficients.csv", coefficient_rows, COEFFICIENT_FIELDS)
    write_csv(
        output_dir / "metrics" / "neutral_aero_residual_lateral_diagnostic_coefficients.csv",
        coefficient_output_rows(
            lateral_diagnostic_result.get("fit_result", {}),
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
        )
        if lateral_diagnostic_result.get("enabled", False)
        else [],
        COEFFICIENT_FIELDS,
    )
    write_csv(output_dir / "metrics" / "neutral_aero_residual_filtered_throws.csv", filtered_throw_rows, FILTERED_THROW_FIELDS)
    write_csv(output_dir / "metrics" / "neutral_aero_residual_surface_scale_selection.csv", surface_scale_rows, SURFACE_SCALE_SELECTION_FIELDS)
    write_csv(output_dir / "metrics" / "neutral_aero_residual_group_iteration_history.csv", group_iteration_rows, GROUP_ITERATION_FIELDS)
    write_csv(output_dir / "metrics" / "neutral_aero_residual_cm_stage_history.csv", cm_stage_history_rows, CM_STAGE_HISTORY_FIELDS)
    write_csv(output_dir / "metrics" / "neutral_aero_residual_replay_validation.csv", validation_rows, REPLAY_VALIDATION_FIELDS)
    write_csv(output_dir / "metrics" / "neutral_aero_residual_stage_replay_errors.csv", stage_replay_rows, STAGE_REPLAY_SUMMARY_FIELDS)
    write_manifest(
        output_dir,
        run_label=run_label,
        session_root=session_root,
        loaded_throw_count=len(loaded_rows),
        valid_rows=valid_rows,
        filtered_throw_rows=filtered_throw_rows,
        heldout_indices=heldout_indices,
        heldout_seed=heldout_seed,
        alignment_window_s=alignment_window_s,
        derivative_window_s=derivative_window_s,
        replay_dt_s=replay_dt_s,
        ridge_lambda=ridge_lambda,
        min_speed_m_s=min_speed_m_s,
        workers=workers,
        fit_workflow=fit_workflow,
        group_iterations=group_iterations,
        group_improvement_tol=group_improvement_tol,
        filter_aligned_launch_state=filter_aligned_launch_state,
        aligned_u_min_m_s=aligned_u_min_m_s,
        aligned_u_max_m_s=aligned_u_max_m_s,
        aligned_v_abs_max_m_s=aligned_v_abs_max_m_s,
        aligned_w_abs_max_m_s=aligned_w_abs_max_m_s,
        apply_attached_cm_bias=apply_attached_cm_bias,
        fit_transition_pitch_moment=fit_transition_pitch_moment,
        fit_post_stall_lift_drag=fit_post_stall_lift_drag,
        fit_post_stall_pitch_moment=fit_post_stall_pitch_moment,
        fit_post_stall_longitudinal=fit_post_stall_longitudinal,
        fit_transition_blender=fit_transition_blender,
        fit_post_stall_damping=fit_post_stall_damping,
        fit_post_stall_surfaces=fit_post_stall_surfaces,
        fit_attached_lateral_coupling=fit_attached_lateral_coupling,
        fit_transition_lateral_coupling=fit_transition_lateral_coupling,
        fit_lateral_surfaces=fit_lateral_surfaces,
        fit_secondary_lateral_diagnostic=fit_secondary_lateral_diagnostic,
        base_parameters=base_parameters,
        candidate_parameters=candidate_parameters,
        lateral_diagnostic_parameters=lateral_diagnostic_parameters,
        fit_result=fit_result,
        lateral_diagnostic_result=lateral_diagnostic_result,
        group_iteration_rows=group_iteration_rows,
        cm_stage_history_rows=cm_stage_history_rows,
        acceptance_summary=acceptance_summary,
        lateral_diagnostic_acceptance=lateral_diagnostic_acceptance,
    )
    write_report(
        output_dir,
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
        fit_workflow=fit_workflow,
        group_iterations=group_iterations,
        group_improvement_tol=group_improvement_tol,
        loaded_throw_count=len(loaded_rows),
        filtered_throw_rows=filtered_throw_rows,
        filter_aligned_launch_state=filter_aligned_launch_state,
        aligned_u_min_m_s=aligned_u_min_m_s,
        aligned_u_max_m_s=aligned_u_max_m_s,
        aligned_v_abs_max_m_s=aligned_v_abs_max_m_s,
        aligned_w_abs_max_m_s=aligned_w_abs_max_m_s,
        apply_attached_cm_bias=apply_attached_cm_bias,
        fit_transition_pitch_moment=fit_transition_pitch_moment,
        fit_post_stall_lift_drag=fit_post_stall_lift_drag,
        fit_post_stall_pitch_moment=fit_post_stall_pitch_moment,
        fit_post_stall_longitudinal=fit_post_stall_longitudinal,
        fit_transition_blender=fit_transition_blender,
        fit_post_stall_damping=fit_post_stall_damping,
        fit_post_stall_surfaces=fit_post_stall_surfaces,
        fit_attached_lateral_coupling=fit_attached_lateral_coupling,
        fit_transition_lateral_coupling=fit_transition_lateral_coupling,
        fit_lateral_surfaces=fit_lateral_surfaces,
        fit_secondary_lateral_diagnostic=fit_secondary_lateral_diagnostic,
        fit_result=fit_result,
        lateral_diagnostic_result=lateral_diagnostic_result,
        group_iteration_rows=group_iteration_rows,
        cm_stage_history_rows=cm_stage_history_rows,
        acceptance_summary=acceptance_summary,
        lateral_diagnostic_acceptance=lateral_diagnostic_acceptance,
        regime_summary=regime_summary,
        stage_fit_summary=stage_fit_summary,
        validation_rows=validation_rows,
        stage_replay_rows=stage_replay_rows,
        base_parameters=base_parameters,
        candidate_parameters=candidate_parameters,
        lateral_diagnostic_parameters=lateral_diagnostic_parameters,
    )
    return output_dir


def active_parameter_dict() -> dict[str, float]:
    parameters = {
        "cd0_strip_scale": float(active_calibration.CD0_STRIP_SCALE),
        "drag_area_fuse_scale": float(active_calibration.DRAG_AREA_FUSE_SCALE),
        "efficiency_strip_scale": float(active_calibration.EFFICIENCY_STRIP_SCALE),
        "side_force_bias_coeff": float(getattr(active_calibration, "SIDE_FORCE_BIAS_COEFF", 0.0)),
        "side_force_beta_coeff": float(getattr(active_calibration, "SIDE_FORCE_BETA_COEFF", 0.0)),
        "side_force_p_hat_coeff": float(getattr(active_calibration, "SIDE_FORCE_P_HAT_COEFF", 0.0)),
        "side_force_r_hat_coeff": float(getattr(active_calibration, "SIDE_FORCE_R_HAT_COEFF", 0.0)),
        "roll_moment_bias_coeff": float(getattr(active_calibration, "ROLL_MOMENT_BIAS_COEFF", 0.0)),
        "roll_moment_beta_coeff": float(getattr(active_calibration, "ROLL_MOMENT_BETA_COEFF", 0.0)),
        "roll_moment_p_hat_coeff": float(getattr(active_calibration, "ROLL_MOMENT_P_HAT_COEFF", 0.0)),
        "roll_moment_r_hat_coeff": float(getattr(active_calibration, "ROLL_MOMENT_R_HAT_COEFF", 0.0)),
        "pitch_moment_bias_coeff": float(getattr(active_calibration, "PITCH_MOMENT_BIAS_COEFF", 0.0)),
        "attached_pitch_moment_bias_coeff": float(
            getattr(active_calibration, "ATTACHED_PITCH_MOMENT_BIAS_COEFF", 0.0)
        ),
        "transition_pitch_moment_bias_coeff": float(
            getattr(active_calibration, "TRANSITION_PITCH_MOMENT_BIAS_COEFF", 0.0)
        ),
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
        "yaw_moment_beta_coeff": float(getattr(active_calibration, "YAW_MOMENT_BETA_COEFF", 0.0)),
        "yaw_moment_p_hat_coeff": float(getattr(active_calibration, "YAW_MOMENT_P_HAT_COEFF", 0.0)),
        "yaw_moment_r_hat_coeff": float(getattr(active_calibration, "YAW_MOMENT_R_HAT_COEFF", 0.0)),
        "transition_side_force_bias_coeff": float(getattr(active_calibration, "TRANSITION_SIDE_FORCE_BIAS_COEFF", 0.0)),
        "transition_side_force_beta_coeff": float(getattr(active_calibration, "TRANSITION_SIDE_FORCE_BETA_COEFF", 0.0)),
        "transition_side_force_p_hat_coeff": float(getattr(active_calibration, "TRANSITION_SIDE_FORCE_P_HAT_COEFF", 0.0)),
        "transition_side_force_r_hat_coeff": float(getattr(active_calibration, "TRANSITION_SIDE_FORCE_R_HAT_COEFF", 0.0)),
        "transition_roll_moment_bias_coeff": float(getattr(active_calibration, "TRANSITION_ROLL_MOMENT_BIAS_COEFF", 0.0)),
        "transition_roll_moment_beta_coeff": float(getattr(active_calibration, "TRANSITION_ROLL_MOMENT_BETA_COEFF", 0.0)),
        "transition_roll_moment_p_hat_coeff": float(getattr(active_calibration, "TRANSITION_ROLL_MOMENT_P_HAT_COEFF", 0.0)),
        "transition_roll_moment_r_hat_coeff": float(getattr(active_calibration, "TRANSITION_ROLL_MOMENT_R_HAT_COEFF", 0.0)),
        "transition_yaw_moment_bias_coeff": float(getattr(active_calibration, "TRANSITION_YAW_MOMENT_BIAS_COEFF", 0.0)),
        "transition_yaw_moment_beta_coeff": float(getattr(active_calibration, "TRANSITION_YAW_MOMENT_BETA_COEFF", 0.0)),
        "transition_yaw_moment_p_hat_coeff": float(getattr(active_calibration, "TRANSITION_YAW_MOMENT_P_HAT_COEFF", 0.0)),
        "transition_yaw_moment_r_hat_coeff": float(getattr(active_calibration, "TRANSITION_YAW_MOMENT_R_HAT_COEFF", 0.0)),
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


def filter_aligned_launch_rows(
    rows: list[dict[str, Any]],
    *,
    alignment_window_s: float,
    enabled: bool,
    u_min_m_s: float,
    u_max_m_s: float,
    v_abs_max_m_s: float,
    w_abs_max_m_s: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        keep = True
        reason = "filter_disabled" if not enabled else "kept"
        aligned: dict[str, Any] = {}
        state = np.full(12, np.nan, dtype=float)
        launch_lateral_score = float("nan")
        launch_confidence_weight = 1.0
        if enabled:
            throw_dir = Path(str(row.get("_throw_dir", "")))
            if not throw_dir.exists():
                keep = False
                reason = "missing_throw_dir"
            else:
                sample_rows = prep._read_csv(throw_dir / "metrics" / "state_samples.csv")
                if len(sample_rows) < 8:
                    keep = False
                    reason = "too_few_state_samples"
                else:
                    aligned = prep._aligned_state_from_sample_rows(sample_rows, alignment_window_s)
                    if aligned.get("status") != "ok":
                        keep = False
                        reason = str(aligned.get("status", "alignment_failed"))
                    else:
                        state = np.asarray(aligned["state"], dtype=float).reshape(-1)
                        if state.size < 12 or not np.all(np.isfinite(state[:12])):
                            keep = False
                            reason = "aligned_state_nonfinite"
                        else:
                            launch_lateral_score = launch_quality_score_from_state(state)
                            launch_confidence_weight = launch_confidence_weight_from_state(state)
                            u0_m_s = float(state[6])
                            v0_m_s = float(state[7])
                            w0_m_s = float(state[8])
                            checks = [
                                (float(u_min_m_s) <= u0_m_s <= float(u_max_m_s), "aligned_u_outside_replay_filter"),
                                (abs(v0_m_s) <= float(v_abs_max_m_s), "aligned_v_outside_launch_gate"),
                                (abs(w0_m_s) <= float(w_abs_max_m_s), "aligned_w_outside_launch_gate"),
                            ]
                            failed = [label for ok, label in checks if not ok]
                            if failed:
                                keep = False
                                reason = ";".join(failed)
        if keep:
            kept_row = dict(row)
            kept_row["_aligned_launch_lateral_score"] = launch_lateral_score
            kept_row["_aligned_launch_confidence_weight"] = launch_confidence_weight
            out.append(kept_row)
        audit_rows.append(
            {
                "row_index": index,
                "kept": keep,
                "filter_reason": reason,
                "session_label": row.get("session_label", ""),
                "case_id": row.get("case_id", ""),
                "throw_id": row.get("throw_id", ""),
                "alignment_window_s": alignment_window_s,
                "alignment_elapsed_s": aligned.get("alignment_elapsed_s", ""),
                "phi0_deg": float(math.degrees(state[3])) if state.size >= 6 and math.isfinite(float(state[3])) else "",
                "psi0_deg": float(math.degrees(state[5])) if state.size >= 6 and math.isfinite(float(state[5])) else "",
                "u0_m_s": float(state[6]) if state.size >= 9 and math.isfinite(float(state[6])) else "",
                "v0_m_s": float(state[7]) if state.size >= 9 and math.isfinite(float(state[7])) else "",
                "w0_m_s": float(state[8]) if state.size >= 9 and math.isfinite(float(state[8])) else "",
                "speed0_m_s": float(np.linalg.norm(state[6:9])) if state.size >= 9 and np.all(np.isfinite(state[6:9])) else "",
                "p0_rad_s": float(state[9]) if state.size >= 12 and math.isfinite(float(state[9])) else "",
                "q0_rad_s": float(state[10]) if state.size >= 12 and math.isfinite(float(state[10])) else "",
                "r0_rad_s": float(state[11]) if state.size >= 12 and math.isfinite(float(state[11])) else "",
                "launch_lateral_score": launch_lateral_score if math.isfinite(launch_lateral_score) else "",
                "launch_confidence_weight": launch_confidence_weight,
            }
        )
    return out, audit_rows


def launch_quality_score_from_state(state: np.ndarray) -> float:
    state = np.asarray(state, dtype=float).reshape(-1)
    if state.size < 12 or not np.all(np.isfinite(state[:12])):
        return float("nan")
    components = np.asarray(
        [
            abs(float(state[3])) / max(math.radians(DEFAULT_ALIGNED_ROLL_ABS_MAX_DEG), 1.0e-9),
            abs(float(state[5])) / max(math.radians(DEFAULT_ALIGNED_YAW_ABS_MAX_DEG), 1.0e-9),
            abs(float(state[7])) / max(DEFAULT_ALIGNED_V_ABS_MAX_M_S, 1.0e-9),
            abs(float(state[9])) / max(DEFAULT_ALIGNED_P_ABS_MAX_RAD_S, 1.0e-9),
            abs(float(state[11])) / max(DEFAULT_ALIGNED_R_ABS_MAX_RAD_S, 1.0e-9),
        ],
        dtype=float,
    )
    return float(np.sqrt(np.mean(components**2)))


def launch_confidence_weight_from_state(state: np.ndarray) -> float:
    score = launch_quality_score_from_state(state)
    if not math.isfinite(score):
        return 1.0
    weight = math.exp(-float(DEFAULT_LAUNCH_CONFIDENCE_EXPONENT) * score**2)
    return float(np.clip(weight, DEFAULT_LAUNCH_CONFIDENCE_MIN_WEIGHT, 1.0))


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
    launch_state = np.asarray(aligned["state"], dtype=float).reshape(-1)
    launch_lateral_score = finite_value(row.get("_aligned_launch_lateral_score", float("nan")))
    launch_confidence_weight = finite_value(row.get("_aligned_launch_confidence_weight", float("nan")))
    if not math.isfinite(launch_lateral_score):
        launch_lateral_score = launch_quality_score_from_state(launch_state)
    if not math.isfinite(launch_confidence_weight):
        launch_confidence_weight = launch_confidence_weight_from_state(launch_state)

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
                "launch_lateral_score": launch_lateral_score,
                "launch_confidence_weight": launch_confidence_weight,
                "u0_m_s": float(launch_state[6]) if launch_state.size >= 9 and math.isfinite(float(launch_state[6])) else float("nan"),
                "v0_m_s": float(launch_state[7]) if launch_state.size >= 9 and math.isfinite(float(launch_state[7])) else float("nan"),
                "w0_m_s": float(launch_state[8]) if launch_state.size >= 9 and math.isfinite(float(launch_state[8])) else float("nan"),
                "p0_rad_s": float(launch_state[9]) if launch_state.size >= 12 and math.isfinite(float(launch_state[9])) else float("nan"),
                "q0_rad_s": float(launch_state[10]) if launch_state.size >= 12 and math.isfinite(float(launch_state[10])) else float("nan"),
                "r0_rad_s": float(launch_state[11]) if launch_state.size >= 12 and math.isfinite(float(launch_state[11])) else float("nan"),
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


def lateral_coupling_coeff_keys(group: str, *, transition: bool) -> list[str]:
    prefix = f"transition_{group}" if transition else group
    return [
        f"{prefix}_bias_coeff",
        f"{prefix}_beta_coeff",
        f"{prefix}_p_hat_coeff",
        f"{prefix}_r_hat_coeff",
    ]


def lateral_coupling_features(sample: dict[str, Any]) -> np.ndarray:
    beta_rad = math.radians(float(sample.get("beta_deg", float("nan"))))
    p_hat = float(sample.get("p_hat", 0.0)) if math.isfinite(float(sample.get("p_hat", 0.0))) else 0.0
    r_hat = float(sample.get("r_hat", 0.0)) if math.isfinite(float(sample.get("r_hat", 0.0))) else 0.0
    return np.asarray([1.0, beta_rad, p_hat, r_hat], dtype=float)


def minimal_lateral_feature_indices(group: str) -> list[int]:
    fitted_features = MINIMAL_LATERAL_FEATURES_BY_GROUP.get(str(group), ())
    return [LATERAL_COUPLING_FEATURE_NAMES.index(name) for name in fitted_features]


def minimal_lateral_coupling_coeff_keys(group: str, *, transition: bool) -> list[str]:
    all_keys = lateral_coupling_coeff_keys(group, transition=transition)
    return [all_keys[index] for index in minimal_lateral_feature_indices(group)]


def active_attached_lateral_parameter_keys() -> tuple[str, ...]:
    return MINIMAL_ATTACHED_LATERAL_PARAMETER_KEYS


def active_transition_lateral_parameter_keys() -> tuple[str, ...]:
    return MINIMAL_TRANSITION_LATERAL_PARAMETER_KEYS


def transition_window_weight_deg(alpha_deg: float, coeffs: dict[str, float]) -> float:
    activation = residual_blend_activation_from_coeffs(float(alpha_deg), coeffs)
    if not math.isfinite(activation):
        return 0.0
    return float(4.0 * activation * (1.0 - activation))


def lateral_coupling_prediction(
    coeffs: dict[str, float],
    group: str,
    sample: dict[str, Any],
    *,
    transition: bool,
) -> float:
    features = lateral_coupling_features(sample)
    keys = lateral_coupling_coeff_keys(group, transition=transition)
    values = np.asarray([float(coeffs.get(key, 0.0)) for key in keys], dtype=float)
    prediction = float(features @ values)
    if transition:
        prediction *= transition_window_weight_deg(float(sample.get("alpha_deg", float("nan"))), coeffs)
    return prediction


def fit_lateral_coupling_coefficients(
    samples: list[dict[str, Any]],
    *,
    ridge_lambda: float,
    transition: bool,
    base_coeffs: dict[str, float],
    use_confidence_weights: bool,
) -> tuple[dict[str, float], dict[str, Any]]:
    residual_map = {
        "side_force": "cy",
        "roll_moment": "cl_roll",
        "yaw_moment": "cn_yaw",
    }
    coeffs = {key: 0.0 for group in LATERAL_COUPLING_GROUPS for key in lateral_coupling_coeff_keys(group, transition=transition)}
    fit_details: dict[str, Any] = {}
    valid = [
        sample
        for sample in samples
        if math.isfinite(float(sample.get("q_bar", float("nan"))))
        and math.isfinite(float(sample.get("alpha_deg", float("nan"))))
    ]
    if not valid:
        for group in LATERAL_COUPLING_GROUPS:
            fit_details[group] = {"used_sample_count": 0, "mae": float("nan"), "rmse": float("nan")}
        return coeffs, fit_details

    q_bar = np.asarray([float(sample["q_bar"]) for sample in valid], dtype=float)
    confidence_weights = sample_confidence_weights(valid) if bool(use_confidence_weights) else np.ones(len(valid), dtype=float)
    weights = dynamic_pressure_weights(q_bar) * throw_balance_weights(valid) * confidence_weights
    x_full = np.asarray([lateral_coupling_features(sample) for sample in valid], dtype=float)
    x_base = x_full
    if transition:
        transition_weights = np.asarray(
            [transition_window_weight_deg(float(sample["alpha_deg"]), base_coeffs) for sample in valid],
            dtype=float,
        )
        x_full = x_full * transition_weights[:, None]
    for group, residual_key in residual_map.items():
        feature_indices = minimal_lateral_feature_indices(group)
        fit_keys = minimal_lateral_coupling_coeff_keys(group, transition=transition)
        if not feature_indices:
            fit_details[group] = {
                "used_sample_count": 0,
                "mae": float("nan"),
                "rmse": float("nan"),
                "status": "disabled_no_minimal_feature",
                "fitted_parameter_keys": [],
            }
            continue
        x_base = x_full[:, feature_indices]
        y_values = []
        x_rows = []
        w_rows = []
        for index, sample in enumerate(valid):
            y = float(sample.get(residual_key, float("nan")))
            if not math.isfinite(y):
                continue
            if transition:
                y -= lateral_coupling_prediction(base_coeffs, group, sample, transition=False)
            if not np.isfinite(x_base[index]).all() or np.linalg.norm(x_base[index]) <= 1.0e-12:
                continue
            y_values.append(y)
            x_rows.append(x_base[index])
            w_rows.append(weights[index])
        min_count = max(8 if not transition else 12, 4 * len(feature_indices))
        if len(y_values) < min_count:
            fit_details[group] = {
                "used_sample_count": len(y_values),
                "mae": float("nan"),
                "rmse": float("nan"),
                "fitted_parameter_keys": fit_keys,
            }
            continue
        x = np.asarray(x_rows, dtype=float)
        y_array = np.asarray(y_values, dtype=float)
        w_array = np.asarray(w_rows, dtype=float)
        coeff, used_count, mae_value, rmse_value = robust_weighted_ridge_fit(
            x,
            y_array,
            w_array,
            ridge_lambda=float(ridge_lambda),
            min_used_count=min_count,
        )
        for key, value in zip(fit_keys, coeff):
            coeffs[key] = replay_fit.bounded_parameter_value(key, float(value))
        fit_details[group] = {
            "used_sample_count": int(used_count),
            "mae": float(mae_value),
            "rmse": float(rmse_value),
            "fitted_parameter_keys": fit_keys,
        }
    return coeffs, fit_details


def fit_pitch_residual_coefficients(
    rows: list[dict[str, Any]],
    *,
    ridge_lambda: float,
    fit_post_stall_surfaces: bool,
    fit_post_stall_damping: bool,
    fit_attached_lateral_coupling: bool,
    fit_transition_lateral_coupling: bool,
    fit_lateral_surfaces: bool,
    lateral_use_confidence_weights: bool,
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
                "launch_lateral_score": finite_value(row.get("launch_lateral_score")),
                "launch_confidence_weight": finite_value(row.get("launch_confidence_weight")),
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
        "post_stall_pitch_damping_coeff": 0.0,
    }
    if fit_attached_lateral_coupling:
        attached_lateral_coeffs, attached_lateral_fit = fit_lateral_coupling_coefficients(
            attached_samples,
            ridge_lambda=ridge_lambda,
            transition=False,
            base_coeffs=coeffs,
            use_confidence_weights=lateral_use_confidence_weights,
        )
        coeffs.update(attached_lateral_coeffs)
    else:
        attached_lateral_fit = {
            group: {"used_sample_count": 0, "mae": float("nan"), "rmse": float("nan"), "status": "disabled"}
            for group in LATERAL_COUPLING_GROUPS
        }
    if fit_transition_lateral_coupling:
        transition_lateral_coeffs, transition_lateral_fit = fit_lateral_coupling_coefficients(
            transition_samples,
            ridge_lambda=ridge_lambda,
            transition=True,
            base_coeffs=coeffs,
            use_confidence_weights=lateral_use_confidence_weights,
        )
        coeffs.update(transition_lateral_coeffs)
    else:
        transition_lateral_fit = {
            group: {"used_sample_count": 0, "mae": float("nan"), "rmse": float("nan"), "status": "disabled"}
            for group in LATERAL_COUPLING_GROUPS
        }

    post_stall_surface_samples = adjusted_lateral_surface_samples(post_stall_samples, coeffs)
    if fit_post_stall_surfaces:
        surface_fit = fit_post_stall_surface_coefficients(
            post_stall_surface_samples,
            ridge_lambda=ridge_lambda,
            fit_post_stall_damping=fit_post_stall_damping,
            fit_lateral_surfaces=fit_lateral_surfaces,
        )
    else:
        surface_fit = fit_compact_post_stall_coefficients(
            post_stall_surface_samples,
            ridge_lambda=ridge_lambda,
            fit_post_stall_damping=fit_post_stall_damping,
        )
    surface_coeffs = surface_fit["coefficients"]
    coeffs["post_stall_pitch_damping_coeff"] = float(surface_coeffs.get("post_stall_pitch_damping_coeff", 0.0))
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
        "attached_lateral_fit": attached_lateral_fit,
        "transition_lateral_fit": transition_lateral_fit,
        "stage_sample_counts": {
            "attached": len(attached_samples),
            "transition_before_post_stall": len(transition_before_samples),
            "transition_after_post_stall": len(transition_after_samples),
            "post_stall": len(post_stall_samples),
        },
        "ridge_lambda": float(ridge_lambda),
        "post_stall_fit_profile": "alpha_rbf_surface" if fit_post_stall_surfaces else "compact_scalar_activation",
        "fit_policy": (
            "compact_neutral_launch_fit; primary claim-bearing path is longitudinal-only by default, "
            "compact scalar post-stall CL/CD/Cm/Cmq by default, transition blender last, "
            "and lateral terms are report-only unless explicit primary lateral flags or the secondary diagnostic are used"
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
    weights = dynamic_pressure_weights(q_bar) * throw_balance_weights(valid) * sample_confidence_weights(valid)
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


def adjusted_lateral_surface_samples(samples: list[dict[str, Any]], coeffs: dict[str, float]) -> list[dict[str, Any]]:
    adjusted: list[dict[str, Any]] = []
    residual_map = {
        "side_force": "cy",
        "roll_moment": "cl_roll",
        "yaw_moment": "cn_yaw",
    }
    for sample in samples:
        row = dict(sample)
        for group, residual_key in residual_map.items():
            value = float(row.get(residual_key, float("nan")))
            if not math.isfinite(value):
                continue
            value -= lateral_coupling_prediction(coeffs, group, row, transition=False)
            value -= lateral_coupling_prediction(coeffs, group, row, transition=True)
            row[residual_key] = float(value)
        adjusted.append(row)
    return adjusted


def fit_compact_post_stall_coefficients(
    samples: list[dict[str, Any]],
    *,
    ridge_lambda: float,
    fit_post_stall_damping: bool,
) -> dict[str, Any]:
    valid = [
        sample
        for sample in samples
        if math.isfinite(float(sample.get("activation", float("nan"))))
        and float(sample.get("activation", 0.0)) > 1.0e-6
        and math.isfinite(float(sample.get("alpha_deg", float("nan"))))
        and math.isfinite(float(sample.get("q_bar", float("nan"))))
        and math.isfinite(float(sample.get("cl", float("nan"))))
        and math.isfinite(float(sample.get("cd", float("nan"))))
        and math.isfinite(float(sample.get("cm", float("nan"))))
    ]
    coeffs = zero_surface_coefficients()
    coeffs.update(
        {
            "post_stall_lift_residual_coeff": 0.0,
            "post_stall_drag_residual_coeff": 0.0,
            "post_stall_pitch_moment_coeff": 0.0,
            "post_stall_pitch_damping_coeff": 0.0,
        }
    )
    if len(valid) < 8:
        return {
            "status": "too_few_post_stall_samples",
            "sample_count": len(valid),
            "used_sample_count": 0,
            "fit_details": {},
            "coefficients": coeffs,
        }

    q_bar = np.asarray([float(sample["q_bar"]) for sample in valid], dtype=float)
    weights = dynamic_pressure_weights(q_bar) * throw_balance_weights(valid) * sample_confidence_weights(valid)
    activation = np.asarray([float(sample["activation"]) for sample in valid], dtype=float)
    x_scalar = activation.reshape(-1, 1)
    fit_details: dict[str, Any] = {}

    for residual_key, coeff_key in (
        ("cl", "post_stall_lift_residual_coeff"),
        ("cd", "post_stall_drag_residual_coeff"),
    ):
        y = np.asarray([float(sample[residual_key]) for sample in valid], dtype=float)
        coeff, used_count, mae_value, rmse_value = robust_weighted_ridge_fit(
            x_scalar,
            y,
            weights,
            ridge_lambda=float(ridge_lambda),
            min_used_count=8,
        )
        coeffs[coeff_key] = replay_fit.bounded_parameter_value(coeff_key, float(coeff[0]))
        fit_details[residual_key] = {
            "used_sample_count": int(used_count),
            "mae": float(mae_value),
            "rmse": float(rmse_value),
            "fit_profile": "compact_scalar_activation",
        }

    q_hat = np.asarray(
        [
            float(sample.get("q_hat", 0.0))
            if math.isfinite(float(sample.get("q_hat", 0.0)))
            else 0.0
            for sample in valid
        ],
        dtype=float,
    )
    x_cm = np.column_stack([activation, activation * q_hat]) if fit_post_stall_damping else x_scalar
    y_cm = np.asarray([float(sample["cm"]) for sample in valid], dtype=float)
    cm_coeff, used_count, mae_value, rmse_value = robust_weighted_ridge_fit(
        x_cm,
        y_cm,
        weights,
        ridge_lambda=float(ridge_lambda),
        min_used_count=8,
    )
    coeffs["post_stall_pitch_moment_coeff"] = replay_fit.bounded_parameter_value(
        "post_stall_pitch_moment_coeff",
        float(cm_coeff[0]),
    )
    if fit_post_stall_damping and cm_coeff.size > 1:
        coeffs["post_stall_pitch_damping_coeff"] = replay_fit.bounded_parameter_value(
            "post_stall_pitch_damping_coeff",
            float(cm_coeff[1]),
        )
    fit_details["cm"] = {
        "used_sample_count": int(used_count),
        "mae": float(mae_value),
        "rmse": float(rmse_value),
        "fit_profile": "compact_scalar_activation_plus_q_hat" if fit_post_stall_damping else "compact_scalar_activation",
    }
    for residual_key in ("cy", "cl_roll", "cn_yaw"):
        fit_details[residual_key] = {
            "used_sample_count": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "status": "disabled_in_compact_default",
        }

    return {
        "status": "ok",
        "sample_count": len(valid),
        "used_sample_count": min(int(detail["used_sample_count"]) for detail in fit_details.values() if "status" not in detail),
        "basis_centres_deg": [],
        "basis_width_deg": float("nan"),
        "fit_profile": "compact_scalar_activation",
        "fit_details": fit_details,
        "coefficients": coeffs,
    }


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
    weights = dynamic_pressure_weights(q_bar) * throw_balance_weights(valid) * sample_confidence_weights(valid)

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
    weights = dynamic_pressure_weights(q_bar) * sample_confidence_weights(valid)
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
    post_cl = float(coeffs.get("post_stall_lift_residual_coeff", 0.0))
    post_cd = float(coeffs.get("post_stall_drag_residual_coeff", 0.0))
    post_cm = float(coeffs.get("post_stall_pitch_moment_coeff", 0.0))
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
    if max([abs(post_cl), abs(post_cd), abs(post_cm), abs(post_cmq), *surface_values]) <= 1.0e-12:
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
                cm_model = activation * post_cm + surface_rbf_prediction(
                    coeffs,
                    "post_stall_pitch_moment_rbf",
                    alpha_deg,
                    start_alpha_deg=float(start_alpha_deg),
                    full_alpha_deg=float(full_alpha_deg),
                )
                cmq_model = activation * post_cmq + surface_rbf_prediction(
                    coeffs,
                    "post_stall_pitch_damping_rbf",
                    alpha_deg,
                    start_alpha_deg=float(start_alpha_deg),
                    full_alpha_deg=float(full_alpha_deg),
                )
                cl_model = activation * post_cl + surface_rbf_prediction(
                    coeffs,
                    "post_stall_lift_rbf",
                    alpha_deg,
                    start_alpha_deg=float(start_alpha_deg),
                    full_alpha_deg=float(full_alpha_deg),
                )
                cd_model = activation * post_cd + surface_rbf_prediction(
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
                transition_weight = float(4.0 * activation * (1.0 - activation))
                lateral_feature_values = np.asarray([1.0, beta_rad, p_hat, r_hat], dtype=float)
                cy_model += float(
                    lateral_feature_values
                    @ np.asarray(
                        [float(coeffs.get(key, 0.0)) for key in lateral_coupling_coeff_keys("side_force", transition=False)],
                        dtype=float,
                    )
                )
                cy_model += transition_weight * float(
                    lateral_feature_values
                    @ np.asarray(
                        [float(coeffs.get(key, 0.0)) for key in lateral_coupling_coeff_keys("side_force", transition=True)],
                        dtype=float,
                    )
                )
                cl_roll_model += float(
                    lateral_feature_values
                    @ np.asarray(
                        [float(coeffs.get(key, 0.0)) for key in lateral_coupling_coeff_keys("roll_moment", transition=False)],
                        dtype=float,
                    )
                )
                cl_roll_model += transition_weight * float(
                    lateral_feature_values
                    @ np.asarray(
                        [float(coeffs.get(key, 0.0)) for key in lateral_coupling_coeff_keys("roll_moment", transition=True)],
                        dtype=float,
                    )
                )
                cn_yaw_model += float(
                    lateral_feature_values
                    @ np.asarray(
                        [float(coeffs.get(key, 0.0)) for key in lateral_coupling_coeff_keys("yaw_moment", transition=False)],
                        dtype=float,
                    )
                )
                cn_yaw_model += transition_weight * float(
                    lateral_feature_values
                    @ np.asarray(
                        [float(coeffs.get(key, 0.0)) for key in lateral_coupling_coeff_keys("yaw_moment", transition=True)],
                        dtype=float,
                    )
                )
                cm_resid.append(float(sample["cm"]) - (cm_model + cmq_model * q_hat))
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


def pitch_moment_regime_weights_from_activation(activation: float) -> tuple[float, float, float]:
    post_raw = float(np.clip(activation, 0.0, 1.0))
    attached_raw = 1.0 - post_raw
    transition_raw = 4.0 * post_raw * (1.0 - post_raw)
    total = max(attached_raw + transition_raw + post_raw, 1.0e-12)
    return attached_raw / total, transition_raw / total, post_raw / total


def pitch_moment_regime_weights_deg(alpha_deg: float, coeffs: dict[str, float]) -> tuple[float, float, float]:
    activation = residual_blend_activation_from_coeffs(float(alpha_deg), coeffs)
    if not math.isfinite(activation):
        return float("nan"), float("nan"), float("nan")
    return pitch_moment_regime_weights_from_activation(activation)


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
    weights = dynamic_pressure_weights(q_bar) * sample_confidence_weights(valid)
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
    for key in ATTACHED_LATERAL_PARAMETER_KEYS:
        coeffs[key] = 0.0
    for key in TRANSITION_LATERAL_PARAMETER_KEYS:
        coeffs[key] = 0.0
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


def sample_confidence_weights(samples: list[dict[str, Any]]) -> np.ndarray:
    weights = []
    for sample in samples:
        value = finite_value(sample.get("launch_confidence_weight", float("nan")))
        if not math.isfinite(value):
            value = 1.0
        weights.append(float(np.clip(value, DEFAULT_LAUNCH_CONFIDENCE_MIN_WEIGHT, 1.0)))
    return np.asarray(weights, dtype=float)


def cm_fit_residual_for_sample(sample: dict[str, Any], coeffs: dict[str, float]) -> float:
    cm = float(sample.get("cm", float("nan")))
    if not math.isfinite(cm):
        return float("nan")
    fitted = 0.0
    alpha_deg = float(sample.get("alpha_deg", float("nan")))
    if not math.isfinite(alpha_deg):
        return float("nan")
    attached_weight, transition_weight, post_weight = pitch_moment_regime_weights_deg(alpha_deg, coeffs)
    if math.isfinite(attached_weight):
        fitted += attached_weight * float(coeffs.get("attached_cm_bias_coeff", 0.0))
    if math.isfinite(transition_weight):
        fitted += transition_weight * float(coeffs.get("transition_cm_bias_coeff", 0.0))
    q_hat = float(sample.get("q_hat", float("nan")))
    fitted_surface = surface_rbf_prediction(coeffs, "post_stall_pitch_moment_rbf", alpha_deg)
    damping_surface = surface_rbf_prediction(coeffs, "post_stall_pitch_damping_rbf", alpha_deg)
    if math.isfinite(post_weight):
        fitted += post_weight * float(coeffs.get("post_stall_pitch_moment_coeff", 0.0))
    if math.isfinite(fitted_surface):
        fitted += fitted_surface
    if math.isfinite(post_weight) and math.isfinite(q_hat):
        fitted += (
            float(coeffs.get("post_stall_pitch_damping_coeff", 0.0)) * post_weight
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
    fit_post_stall_longitudinal: bool,
    fit_transition_blender: bool,
    fit_post_stall_surfaces: bool,
    fit_post_stall_damping: bool,
    fit_attached_lateral_coupling: bool,
    fit_transition_lateral_coupling: bool,
    fit_lateral_surfaces: bool,
    fit_transition_pitch_moment: bool = False,
    fit_post_stall_lift_drag: bool | None = None,
    fit_post_stall_pitch_moment: bool | None = None,
    surface_scale: float = 1.0,
    group_scales: dict[str, float] | None = None,
) -> dict[str, float]:
    coeffs = fit_result.get("coefficients", zero_coefficients())
    candidate = dict(base)
    scales = dict(group_scales or {})
    if fit_post_stall_lift_drag is None:
        fit_post_stall_lift_drag = bool(fit_post_stall_longitudinal)
    if fit_post_stall_pitch_moment is None:
        fit_post_stall_pitch_moment = bool(fit_post_stall_longitudinal)
    attached_longitudinal_scale = float(scales.get("attached_longitudinal", 1.0))
    transition_pitch_moment_scale = float(scales.get("transition_pitch_moment", 1.0))
    attached_lateral_scale = float(scales.get("attached_lateral", 1.0))
    post_stall_longitudinal_scale = (
        float(scales.get("post_stall_longitudinal", surface_scale)) if bool(fit_post_stall_longitudinal) else 0.0
    )
    post_stall_lift_drag_scale = (
        float(scales.get("post_stall_lift_drag", post_stall_longitudinal_scale))
        if bool(fit_post_stall_longitudinal and fit_post_stall_lift_drag)
        else 0.0
    )
    post_stall_pitch_moment_scale = (
        float(scales.get("post_stall_pitch_moment", post_stall_longitudinal_scale))
        if bool(fit_post_stall_longitudinal and fit_post_stall_pitch_moment)
        else 0.0
    )
    post_stall_pitch_damping_scale = (
        float(scales.get("post_stall_pitch_damping", post_stall_longitudinal_scale))
        if bool(fit_post_stall_longitudinal and fit_post_stall_damping)
        else 0.0
    )
    post_stall_lateral_scale = float(scales.get("post_stall_lateral", surface_scale))
    transition_lateral_scale = float(scales.get("transition_lateral", 1.0))
    transition_blender_scale = float(scales.get("transition_blender", 1.0)) if bool(fit_transition_blender) else 0.0
    if apply_attached_cm_bias:
        candidate["attached_pitch_moment_bias_coeff"] = replay_fit.bounded_parameter_value(
            "attached_pitch_moment_bias_coeff",
            float(candidate.get("attached_pitch_moment_bias_coeff", 0.0))
            + attached_longitudinal_scale * float(coeffs.get("attached_cm_bias_coeff", 0.0)),
        )
    if fit_transition_pitch_moment:
        candidate["transition_pitch_moment_bias_coeff"] = replay_fit.bounded_parameter_value(
            "transition_pitch_moment_bias_coeff",
            float(candidate.get("transition_pitch_moment_bias_coeff", 0.0))
            + transition_pitch_moment_scale * float(coeffs.get("transition_cm_bias_coeff", 0.0)),
        )
    if fit_attached_lateral_coupling:
        for key in active_attached_lateral_parameter_keys():
            candidate[key] = replay_fit.bounded_parameter_value(
                key,
                float(candidate.get(key, 0.0)) + attached_lateral_scale * float(coeffs.get(key, 0.0)),
            )
    if fit_transition_lateral_coupling:
        for key in active_transition_lateral_parameter_keys():
            candidate[key] = replay_fit.bounded_parameter_value(
                key,
                float(candidate.get(key, 0.0)) + transition_lateral_scale * float(coeffs.get(key, 0.0)),
            )
    candidate["post_stall_lift_residual_coeff"] = replay_fit.bounded_parameter_value(
        "post_stall_lift_residual_coeff",
        candidate["post_stall_lift_residual_coeff"]
        + post_stall_lift_drag_scale * float(coeffs.get("post_stall_lift_residual_coeff", 0.0)),
    )
    candidate["post_stall_drag_residual_coeff"] = replay_fit.bounded_parameter_value(
        "post_stall_drag_residual_coeff",
        candidate["post_stall_drag_residual_coeff"]
        + post_stall_lift_drag_scale * float(coeffs.get("post_stall_drag_residual_coeff", 0.0)),
    )
    candidate["post_stall_pitch_moment_coeff"] = replay_fit.bounded_parameter_value(
        "post_stall_pitch_moment_coeff",
        candidate["post_stall_pitch_moment_coeff"]
        + post_stall_pitch_moment_scale * float(coeffs.get("post_stall_pitch_moment_coeff", 0.0)),
    )
    candidate["post_stall_pitch_damping_coeff"] = replay_fit.bounded_parameter_value(
        "post_stall_pitch_damping_coeff",
        candidate["post_stall_pitch_damping_coeff"]
        + post_stall_pitch_damping_scale * float(coeffs.get("post_stall_pitch_damping_coeff", 0.0)),
    )
    if fit_post_stall_surfaces:
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
                    float(candidate.get(key, 0.0)) + post_stall_longitudinal_scale * float(coeffs.get(key, 0.0)),
                )
    if fit_lateral_surfaces:
        for prefix in LATERAL_SURFACE_PREFIXES:
            for key in lateral_surface_coeff_keys(prefix):
                candidate[key] = replay_fit.bounded_parameter_value(
                    key,
                    float(candidate.get(key, 0.0)) + post_stall_lateral_scale * float(coeffs.get(key, 0.0)),
                )
    if transition_blender_scale > 0.5:
        candidate["post_stall_residual_blend_start_alpha_deg"] = replay_fit.bounded_parameter_value(
            "post_stall_residual_blend_start_alpha_deg",
            float(coeffs.get("post_stall_residual_blend_start_alpha_deg", candidate["post_stall_residual_blend_start_alpha_deg"])),
        )
        candidate["post_stall_residual_blend_full_alpha_deg"] = replay_fit.bounded_parameter_value(
            "post_stall_residual_blend_full_alpha_deg",
            max(
                candidate["post_stall_residual_blend_start_alpha_deg"] + 1.0,
                float(
                    coeffs.get(
                        "post_stall_residual_blend_full_alpha_deg",
                        candidate["post_stall_residual_blend_full_alpha_deg"],
                    )
                ),
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
    fit_transition_pitch_moment: bool,
    fit_post_stall_lift_drag: bool,
    fit_post_stall_pitch_moment: bool,
    fit_post_stall_longitudinal: bool,
    fit_transition_blender: bool,
    fit_post_stall_surfaces: bool,
    fit_post_stall_damping: bool,
    fit_attached_lateral_coupling: bool,
    fit_transition_lateral_coupling: bool,
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
            fit_transition_pitch_moment=fit_transition_pitch_moment,
            fit_post_stall_lift_drag=fit_post_stall_lift_drag,
            fit_post_stall_pitch_moment=fit_post_stall_pitch_moment,
            fit_post_stall_longitudinal=fit_post_stall_longitudinal,
            fit_transition_blender=fit_transition_blender,
            fit_post_stall_surfaces=fit_post_stall_surfaces,
            fit_post_stall_damping=fit_post_stall_damping,
            fit_attached_lateral_coupling=fit_attached_lateral_coupling,
            fit_transition_lateral_coupling=fit_transition_lateral_coupling,
            fit_lateral_surfaces=fit_lateral_surfaces,
            surface_scale=float(scale),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
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


def cm_regime_staged_refinement(
    *,
    base_parameters: dict[str, float],
    train_rows: list[dict[str, Any]],
    heldout_rows: list[dict[str, Any]],
    replay_dt_s: float,
    alignment_window_s: float,
    derivative_window_s: float,
    min_speed_m_s: float,
    ridge_lambda: float,
    workers: int,
    fit_post_stall_damping: bool,
    fit_transition_blender: bool,
    fit_post_stall_lift_drag: bool,
    fit_post_stall_surfaces: bool,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    current = dict(base_parameters)
    history: list[dict[str, Any]] = []
    train_before = replay_longitudinal_summary(
        train_rows,
        current,
        replay_dt_s=replay_dt_s,
        alignment_window_s=alignment_window_s,
        workers=workers,
    )
    heldout_before = replay_longitudinal_summary(
        heldout_rows,
        current,
        replay_dt_s=replay_dt_s,
        alignment_window_s=alignment_window_s,
        workers=workers,
    )
    baseline_residuals = residual_rows(
        train_rows,
        split="train",
        parameters=current,
        alignment_window_s=alignment_window_s,
        derivative_window_s=derivative_window_s,
        min_speed_m_s=min_speed_m_s,
        workers=workers,
    )
    history.append(
        cm_stage_history_row(
            stage_index=0,
            stage_id="stage0_baseline",
            fit_parameter_group="none",
            accepted=True,
            reason="baseline_active",
            train_before=train_before,
            train_after=train_before,
            heldout_before=heldout_before,
            heldout_after=heldout_before,
            residuals=baseline_residuals,
            before_parameters=current,
            after_parameters=current,
        )
    )
    stages = [
        ("stage1_attached_cm", "attached_pitch_moment", "attached"),
        ("stage2_transition_cm", "transition_pitch_moment", "transition"),
        ("stage3_post_stall_cm", "post_stall_pitch_moment", "post_stall_cm"),
    ]
    if bool(fit_post_stall_damping):
        stages.append(("stage4_post_stall_cmq", "post_stall_pitch_damping", "post_stall_cmq"))
    if bool(fit_transition_blender):
        stages.append(("stage5_transition_blend", "transition_blend_start_full", "transition_blend"))
    if bool(fit_post_stall_lift_drag):
        stages.append(("stage6_post_stall_lift_drag", "post_stall_lift_drag", "post_stall_lift_drag"))

    for stage_index, (stage_id, group_name, action) in enumerate(stages, start=1):
        residuals = residual_rows(
            train_rows,
            split="train",
            parameters=current,
            alignment_window_s=alignment_window_s,
            derivative_window_s=derivative_window_s,
            min_speed_m_s=min_speed_m_s,
            workers=workers,
        )
        fit_stage = fit_pitch_residual_coefficients(
            residuals,
            ridge_lambda=ridge_lambda,
            fit_post_stall_surfaces=bool(fit_post_stall_surfaces and action == "post_stall_lift_drag"),
            fit_post_stall_damping=bool(action == "post_stall_cmq"),
            fit_attached_lateral_coupling=False,
            fit_transition_lateral_coupling=False,
            fit_lateral_surfaces=False,
            lateral_use_confidence_weights=True,
        )
        if action == "transition_blend":
            proposed = transition_blend_grid_candidate(
                current,
                train_rows=train_rows,
                replay_dt_s=replay_dt_s,
                alignment_window_s=alignment_window_s,
                workers=workers,
            )
        else:
            proposed = candidate_from_fit(
                current,
                fit_stage,
                apply_attached_cm_bias=action == "attached",
                fit_transition_pitch_moment=action == "transition",
                fit_post_stall_lift_drag=action == "post_stall_lift_drag",
                fit_post_stall_pitch_moment=action == "post_stall_cm",
                fit_post_stall_longitudinal=action in {"post_stall_cm", "post_stall_cmq", "post_stall_lift_drag"},
                fit_transition_blender=False,
                fit_post_stall_surfaces=bool(fit_post_stall_surfaces and action == "post_stall_lift_drag"),
                fit_post_stall_damping=action == "post_stall_cmq",
                fit_attached_lateral_coupling=False,
                fit_transition_lateral_coupling=False,
                fit_lateral_surfaces=False,
            )
        train_current = replay_longitudinal_summary(
            train_rows,
            current,
            replay_dt_s=replay_dt_s,
            alignment_window_s=alignment_window_s,
            workers=workers,
        )
        heldout_current = replay_longitudinal_summary(
            heldout_rows,
            current,
            replay_dt_s=replay_dt_s,
            alignment_window_s=alignment_window_s,
            workers=workers,
        )
        train_proposed = replay_longitudinal_summary(
            train_rows,
            proposed,
            replay_dt_s=replay_dt_s,
            alignment_window_s=alignment_window_s,
            workers=workers,
        )
        heldout_proposed = replay_longitudinal_summary(
            heldout_rows,
            proposed,
            replay_dt_s=replay_dt_s,
            alignment_window_s=alignment_window_s,
            workers=workers,
        )
        accepted, reason = longitudinal_stage_acceptance(heldout_current, heldout_proposed)
        next_parameters = proposed if accepted else current
        history.append(
            cm_stage_history_row(
                stage_index=stage_index,
                stage_id=stage_id,
                fit_parameter_group=group_name,
                accepted=accepted,
                reason=reason,
                train_before=train_current,
                train_after=train_proposed,
                heldout_before=heldout_current,
                heldout_after=heldout_proposed,
                residuals=residuals,
                before_parameters=current,
                after_parameters=next_parameters,
            )
        )
        current = next_parameters
    return current, history


def replay_longitudinal_summary(
    rows: list[dict[str, Any]],
    parameters: dict[str, float],
    *,
    replay_dt_s: float,
    alignment_window_s: float,
    workers: int,
) -> dict[str, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        replay_rows = replay_fit.simulate_rows(
            rows,
            parameters,
            replay_dt_s=replay_dt_s,
            alignment_window_s=alignment_window_s,
            workers=workers,
        )
    return replay_fit.objective_summary(replay_rows, objective_mode="longitudinal")


def longitudinal_stage_acceptance(before: dict[str, float], after: dict[str, float]) -> tuple[bool, str]:
    keys = ("dx_mae_m", "altitude_loss_mae_m", "sink_mae_m_s", "final_theta_mae_deg")
    passed = []
    for key in keys:
        before_value = finite_value(before.get(key))
        after_value = finite_value(after.get(key))
        passed.append(math.isfinite(before_value) and math.isfinite(after_value) and after_value <= before_value + 1.0e-9)
    if all(passed):
        return True, "heldout_longitudinal_improved_or_preserved"
    return False, "rejected_by_heldout_longitudinal_gate"


def transition_blend_grid_candidate(
    current: dict[str, float],
    *,
    train_rows: list[dict[str, Any]],
    replay_dt_s: float,
    alignment_window_s: float,
    workers: int,
) -> dict[str, float]:
    best = dict(current)
    best_summary = replay_longitudinal_summary(
        train_rows,
        best,
        replay_dt_s=replay_dt_s,
        alignment_window_s=alignment_window_s,
        workers=workers,
    )
    best_objective = finite_value(best_summary.get("objective"))
    for start_deg in (10.0, 12.0, 14.0):
        for full_deg in (18.0, 20.0, 22.0):
            if full_deg <= start_deg + 1.0:
                continue
            candidate = dict(current)
            candidate["post_stall_residual_blend_start_alpha_deg"] = replay_fit.bounded_parameter_value(
                "post_stall_residual_blend_start_alpha_deg",
                start_deg,
            )
            candidate["post_stall_residual_blend_full_alpha_deg"] = replay_fit.bounded_parameter_value(
                "post_stall_residual_blend_full_alpha_deg",
                full_deg,
            )
            summary = replay_longitudinal_summary(
                train_rows,
                candidate,
                replay_dt_s=replay_dt_s,
                alignment_window_s=alignment_window_s,
                workers=workers,
            )
            objective = finite_value(summary.get("objective"))
            if math.isfinite(objective) and objective < best_objective - 1.0e-9:
                best = candidate
                best_objective = objective
    return best


def cm_stage_history_row(
    *,
    stage_index: int,
    stage_id: str,
    fit_parameter_group: str,
    accepted: bool,
    reason: str,
    train_before: dict[str, float],
    train_after: dict[str, float],
    heldout_before: dict[str, float],
    heldout_after: dict[str, float],
    residuals: list[dict[str, Any]],
    before_parameters: dict[str, float],
    after_parameters: dict[str, float],
) -> dict[str, Any]:
    fractions = regime_sample_fractions(residuals)
    return {
        "stage_index": int(stage_index),
        "stage_id": str(stage_id),
        "fit_parameter_group": str(fit_parameter_group),
        "accepted": bool(accepted),
        "acceptance_reason": str(reason),
        "train_objective_before": finite_value(train_before.get("objective")),
        "train_objective_after": finite_value(train_after.get("objective")),
        "heldout_objective_before": finite_value(heldout_before.get("objective")),
        "heldout_objective_after": finite_value(heldout_after.get("objective")),
        "heldout_dx_mae_before_m": finite_value(heldout_before.get("dx_mae_m")),
        "heldout_dx_mae_after_m": finite_value(heldout_after.get("dx_mae_m")),
        "heldout_altitude_loss_mae_before_m": finite_value(heldout_before.get("altitude_loss_mae_m")),
        "heldout_altitude_loss_mae_after_m": finite_value(heldout_after.get("altitude_loss_mae_m")),
        "heldout_sink_mae_before_m_s": finite_value(heldout_before.get("sink_mae_m_s")),
        "heldout_sink_mae_after_m_s": finite_value(heldout_after.get("sink_mae_m_s")),
        "heldout_pitch_mae_before_deg": finite_value(heldout_before.get("final_theta_mae_deg")),
        "heldout_pitch_mae_after_deg": finite_value(heldout_after.get("final_theta_mae_deg")),
        "attached_sample_fraction": fractions.get("attached", 0.0),
        "transition_sample_fraction": fractions.get("transition", 0.0),
        "post_stall_sample_fraction": fractions.get("post_stall", 0.0),
        "parameter_updates_json": json.dumps(parameter_updates(before_parameters, after_parameters), sort_keys=True),
    }


def regime_sample_fractions(rows: list[dict[str, Any]]) -> dict[str, float]:
    valid = [row for row in rows if row.get("residual_status") == "ok"]
    total = max(len(valid), 1)
    return {
        regime: sum(1 for row in valid if row.get("regime") == regime) / total
        for regime in ("attached", "transition", "post_stall")
    }


def parameter_updates(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    keys = sorted(set(before) | set(after))
    return {
        key: float(after.get(key, 0.0))
        for key in keys
        if abs(float(after.get(key, 0.0)) - float(before.get(key, 0.0))) > 1.0e-12
    }


def grouped_iterative_refinement(
    *,
    base_parameters: dict[str, float],
    fit_result: dict[str, Any],
    train_rows: list[dict[str, Any]],
    replay_dt_s: float,
    alignment_window_s: float,
    workers: int,
    apply_attached_cm_bias: bool,
    fit_transition_pitch_moment: bool,
    fit_post_stall_lift_drag: bool,
    fit_post_stall_pitch_moment: bool,
    fit_post_stall_longitudinal: bool,
    fit_transition_blender: bool,
    fit_post_stall_surfaces: bool,
    fit_post_stall_damping: bool,
    fit_attached_lateral_coupling: bool,
    fit_transition_lateral_coupling: bool,
    fit_lateral_surfaces: bool,
    initial_surface_scale: float,
    group_iterations: int,
    improvement_tol: float,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    group_scales = {
        "attached_longitudinal": 1.0 if apply_attached_cm_bias else 0.0,
        "transition_pitch_moment": 1.0 if fit_transition_pitch_moment else 0.0,
        "post_stall_pitch_moment": float(initial_surface_scale)
        if fit_post_stall_longitudinal and fit_post_stall_pitch_moment
        else 0.0,
        "post_stall_pitch_damping": float(initial_surface_scale)
        if fit_post_stall_longitudinal and fit_post_stall_damping
        else 0.0,
        "post_stall_lift_drag": float(initial_surface_scale)
        if fit_post_stall_longitudinal and fit_post_stall_lift_drag
        else 0.0,
        "attached_lateral": 1.0 if fit_attached_lateral_coupling else 0.0,
        "post_stall_longitudinal": float(initial_surface_scale) if fit_post_stall_longitudinal else 0.0,
        "post_stall_lateral": float(initial_surface_scale) if fit_lateral_surfaces else 0.0,
        "transition_lateral": 1.0 if fit_transition_lateral_coupling else 0.0,
        "transition_blender": 1.0 if fit_transition_blender else 0.0,
    }
    enabled_groups = [
        group
        for group in GROUPED_FIT_GROUP_ORDER
        if group_enabled(
            group,
            apply_attached_cm_bias=apply_attached_cm_bias,
            fit_transition_pitch_moment=fit_transition_pitch_moment,
            fit_post_stall_lift_drag=fit_post_stall_lift_drag,
            fit_post_stall_pitch_moment=fit_post_stall_pitch_moment,
            fit_post_stall_longitudinal=fit_post_stall_longitudinal,
            fit_transition_blender=fit_transition_blender,
            fit_attached_lateral_coupling=fit_attached_lateral_coupling,
            fit_transition_lateral_coupling=fit_transition_lateral_coupling,
            fit_lateral_surfaces=fit_lateral_surfaces,
        )
    ]
    current_candidate = candidate_from_fit(
        base_parameters,
        fit_result,
        apply_attached_cm_bias=apply_attached_cm_bias,
        fit_transition_pitch_moment=fit_transition_pitch_moment,
        fit_post_stall_lift_drag=fit_post_stall_lift_drag,
        fit_post_stall_pitch_moment=fit_post_stall_pitch_moment,
        fit_post_stall_longitudinal=fit_post_stall_longitudinal,
        fit_transition_blender=fit_transition_blender,
        fit_post_stall_surfaces=fit_post_stall_surfaces,
        fit_post_stall_damping=fit_post_stall_damping,
        fit_attached_lateral_coupling=fit_attached_lateral_coupling,
        fit_transition_lateral_coupling=fit_transition_lateral_coupling,
        fit_lateral_surfaces=fit_lateral_surfaces,
        group_scales=group_scales,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        current_replay_rows = replay_fit.simulate_rows(
            train_rows,
            current_candidate,
            replay_dt_s=replay_dt_s,
            alignment_window_s=alignment_window_s,
            workers=workers,
        )
    current_summary = replay_fit.objective_summary(
        current_replay_rows,
        objective_mode="combined",
    )
    current_objective = finite_value(current_summary.get("objective"))
    history = [
        group_iteration_row(
            iteration=0,
            pass_index=0,
            group="initial_grouped_candidate",
            candidate_scale=float("nan"),
            selected=True,
            summary=current_summary,
            group_scales=group_scales,
        )
    ]
    iteration = 1

    for pass_index in range(1, max(0, int(group_iterations)) + 1):
        pass_improved = False
        for group in enabled_groups:
            scale_values = group_scale_candidates(group, group_scales[group])
            candidate_rows = evaluate_group_scale_candidates(
                base_parameters=base_parameters,
                fit_result=fit_result,
                train_rows=train_rows,
                replay_dt_s=replay_dt_s,
                alignment_window_s=alignment_window_s,
                workers=workers,
                apply_attached_cm_bias=apply_attached_cm_bias,
                fit_transition_pitch_moment=fit_transition_pitch_moment,
                fit_post_stall_lift_drag=fit_post_stall_lift_drag,
                fit_post_stall_pitch_moment=fit_post_stall_pitch_moment,
                fit_post_stall_longitudinal=fit_post_stall_longitudinal,
                fit_transition_blender=fit_transition_blender,
                fit_post_stall_surfaces=fit_post_stall_surfaces,
                fit_post_stall_damping=fit_post_stall_damping,
                fit_attached_lateral_coupling=fit_attached_lateral_coupling,
                fit_transition_lateral_coupling=fit_transition_lateral_coupling,
                fit_lateral_surfaces=fit_lateral_surfaces,
                group_scales=group_scales,
                group=group,
                scale_values=scale_values,
                iteration=iteration,
                pass_index=pass_index,
            )
            best_row = min(candidate_rows, key=lambda row: finite_value(row.get("objective")))
            best_objective = finite_value(best_row.get("objective"))
            accepted_scale = float(group_scales[group])
            if math.isfinite(best_objective) and best_objective < current_objective - float(improvement_tol):
                accepted_scale = float(best_row["candidate_scale"])
                group_scales[group] = accepted_scale
                current_objective = best_objective
                pass_improved = True
            for row in candidate_rows:
                row["selected"] = abs(float(row["candidate_scale"]) - accepted_scale) <= 1.0e-12
            history.extend(candidate_rows)
            iteration += 1
        if not pass_improved:
            break

    final_candidate = candidate_from_fit(
        base_parameters,
        fit_result,
        apply_attached_cm_bias=apply_attached_cm_bias,
        fit_transition_pitch_moment=fit_transition_pitch_moment,
        fit_post_stall_lift_drag=fit_post_stall_lift_drag,
        fit_post_stall_pitch_moment=fit_post_stall_pitch_moment,
        fit_post_stall_longitudinal=fit_post_stall_longitudinal,
        fit_transition_blender=fit_transition_blender,
        fit_post_stall_surfaces=fit_post_stall_surfaces,
        fit_post_stall_damping=fit_post_stall_damping,
        fit_attached_lateral_coupling=fit_attached_lateral_coupling,
        fit_transition_lateral_coupling=fit_transition_lateral_coupling,
        fit_lateral_surfaces=fit_lateral_surfaces,
        group_scales=group_scales,
    )
    fit_result["grouped_iteration"] = {
        "enabled": True,
        "group_iterations_requested": int(group_iterations),
        "improvement_tol": float(improvement_tol),
        "final_group_scales": dict(group_scales),
        "history_csv": "metrics/neutral_aero_residual_group_iteration_history.csv",
    }
    return final_candidate, history


def secondary_lateral_diagnostic_candidate(
    *,
    train_rows: list[dict[str, Any]],
    primary_parameters: dict[str, float],
    alignment_window_s: float,
    derivative_window_s: float,
    min_speed_m_s: float,
    ridge_lambda: float,
    workers: int,
    enabled: bool,
) -> tuple[dict[str, Any], dict[str, float] | None]:
    if not bool(enabled):
        return {"enabled": False, "status": "disabled"}, None
    residuals = residual_rows(
        train_rows,
        split="train",
        parameters=primary_parameters,
        alignment_window_s=alignment_window_s,
        derivative_window_s=derivative_window_s,
        min_speed_m_s=min_speed_m_s,
        workers=workers,
    )
    fit_result = fit_pitch_residual_coefficients(
        residuals,
        ridge_lambda=ridge_lambda,
        fit_post_stall_surfaces=False,
        fit_post_stall_damping=False,
        fit_attached_lateral_coupling=True,
        fit_transition_lateral_coupling=False,
        fit_lateral_surfaces=False,
        lateral_use_confidence_weights=False,
    )
    lateral_only_scales = {
        "attached_longitudinal": 0.0,
        "transition_pitch_moment": 0.0,
        "post_stall_pitch_moment": 0.0,
        "post_stall_pitch_damping": 0.0,
        "post_stall_lift_drag": 0.0,
        "attached_lateral": 1.0,
        "post_stall_longitudinal": 0.0,
        "post_stall_lateral": 0.0,
        "transition_lateral": 0.0,
        "transition_blender": 0.0,
    }
    parameters = candidate_from_fit(
        primary_parameters,
        fit_result,
        apply_attached_cm_bias=False,
        fit_post_stall_longitudinal=False,
        fit_transition_blender=False,
        fit_post_stall_surfaces=False,
        fit_post_stall_damping=False,
        fit_attached_lateral_coupling=True,
        fit_transition_lateral_coupling=False,
        fit_lateral_surfaces=False,
        group_scales=lateral_only_scales,
    )
    return (
        {
            "enabled": True,
            "status": fit_result.get("status", ""),
            "policy": (
                "Longitudinal terms are frozen at the primary coefficient candidate; only CY_beta, "
                "Cl_p, and Cn_r are fitted, and launch-confidence weighting is disabled."
            ),
            "fit_result": fit_result,
            "group_scales": lateral_only_scales,
        },
        parameters,
    )


def group_enabled(
    group: str,
    *,
    apply_attached_cm_bias: bool,
    fit_post_stall_longitudinal: bool,
    fit_transition_blender: bool,
    fit_attached_lateral_coupling: bool,
    fit_transition_lateral_coupling: bool,
    fit_lateral_surfaces: bool,
) -> bool:
    if group == "attached_longitudinal":
        return bool(apply_attached_cm_bias)
    if group == "transition_pitch_moment":
        return bool(fit_transition_pitch_moment)
    if group == "post_stall_pitch_moment":
        return bool(fit_post_stall_longitudinal and fit_post_stall_pitch_moment)
    if group == "post_stall_pitch_damping":
        return bool(fit_post_stall_longitudinal and fit_post_stall_damping)
    if group == "post_stall_lift_drag":
        return bool(fit_post_stall_longitudinal and fit_post_stall_lift_drag)
    if group == "attached_lateral":
        return bool(fit_attached_lateral_coupling)
    if group == "post_stall_longitudinal":
        return bool(fit_post_stall_longitudinal)
    if group == "post_stall_lateral":
        return bool(fit_lateral_surfaces)
    if group == "transition_lateral":
        return bool(fit_transition_lateral_coupling)
    if group == "transition_blender":
        return bool(fit_transition_blender)
    return False


def group_scale_candidates(group: str, current_scale: float) -> list[float]:
    if group == "transition_blender":
        return [0.0, 1.0]
    values = sorted({float(value) for value in GROUP_SCALE_CANDIDATES} | {float(current_scale)})
    return values


def evaluate_group_scale_candidates(
    *,
    base_parameters: dict[str, float],
    fit_result: dict[str, Any],
    train_rows: list[dict[str, Any]],
    replay_dt_s: float,
    alignment_window_s: float,
    workers: int,
    apply_attached_cm_bias: bool,
    fit_transition_pitch_moment: bool,
    fit_post_stall_lift_drag: bool,
    fit_post_stall_pitch_moment: bool,
    fit_post_stall_longitudinal: bool,
    fit_transition_blender: bool,
    fit_post_stall_surfaces: bool,
    fit_post_stall_damping: bool,
    fit_attached_lateral_coupling: bool,
    fit_transition_lateral_coupling: bool,
    fit_lateral_surfaces: bool,
    group_scales: dict[str, float],
    group: str,
    scale_values: list[float],
    iteration: int,
    pass_index: int,
) -> list[dict[str, Any]]:
    payloads = []
    for scale in scale_values:
        scales = dict(group_scales)
        scales[group] = float(scale)
        candidate = candidate_from_fit(
            base_parameters,
            fit_result,
            apply_attached_cm_bias=apply_attached_cm_bias,
            fit_transition_pitch_moment=fit_transition_pitch_moment,
            fit_post_stall_lift_drag=fit_post_stall_lift_drag,
            fit_post_stall_pitch_moment=fit_post_stall_pitch_moment,
            fit_post_stall_longitudinal=fit_post_stall_longitudinal,
            fit_transition_blender=fit_transition_blender,
            fit_post_stall_surfaces=fit_post_stall_surfaces,
            fit_post_stall_damping=fit_post_stall_damping,
            fit_attached_lateral_coupling=fit_attached_lateral_coupling,
            fit_transition_lateral_coupling=fit_transition_lateral_coupling,
            fit_lateral_surfaces=fit_lateral_surfaces,
            group_scales=scales,
        )
        payloads.append((candidate, train_rows, replay_dt_s, alignment_window_s, scales, iteration, pass_index, group, float(scale)))
    if int(workers) > 1 and len(payloads) > 1:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            return list(executor.map(evaluate_group_scale_payload, payloads))
    return [evaluate_group_scale_payload(payload) for payload in payloads]


def evaluate_group_scale_payload(payload: tuple[dict[str, float], list[dict[str, Any]], float, float, dict[str, float], int, int, str, float]) -> dict[str, Any]:
    candidate, rows, replay_dt_s, alignment_window_s, group_scales, iteration, pass_index, group, scale = payload
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        replay_rows = replay_fit.simulate_rows(
            rows,
            candidate,
            replay_dt_s=replay_dt_s,
            alignment_window_s=alignment_window_s,
            workers=1,
        )
    summary = replay_fit.objective_summary(replay_rows, objective_mode="combined")
    return group_iteration_row(
        iteration=iteration,
        pass_index=pass_index,
        group=group,
        candidate_scale=scale,
        selected=False,
        summary=summary,
        group_scales=group_scales,
    )


def group_iteration_row(
    *,
    iteration: int,
    pass_index: int,
    group: str,
    candidate_scale: float,
    selected: bool,
    summary: dict[str, Any],
    group_scales: dict[str, float],
) -> dict[str, Any]:
    return {
        "iteration": int(iteration),
        "pass_index": int(pass_index),
        "group": str(group),
        "candidate_scale": float(candidate_scale),
        "selected": bool(selected),
        "objective": finite_value(summary.get("objective")),
        "dx_mae_m": finite_value(summary.get("dx_mae_m")),
        "dy_mae_m": finite_value(summary.get("dy_mae_m")),
        "altitude_loss_mae_m": finite_value(summary.get("altitude_loss_mae_m")),
        "sink_rate_mae_m_s": finite_value(summary.get("sink_mae_m_s")),
        "final_theta_mae_deg": finite_value(summary.get("final_theta_mae_deg")),
        "final_phi_mae_deg": finite_value(summary.get("final_phi_mae_deg")),
        "final_psi_mae_deg": finite_value(summary.get("final_psi_mae_deg")),
        "group_scales_json": json.dumps(group_scales, sort_keys=True),
    }


def replay_validation_rows(
    *,
    train_rows: list[dict[str, Any]],
    heldout_rows: list[dict[str, Any]],
    base_parameters: dict[str, float],
    candidate_parameters: dict[str, float],
    extra_models: list[tuple[str, dict[str, float]]] | None = None,
    replay_dt_s: float,
    alignment_window_s: float,
    workers: int,
) -> list[dict[str, Any]]:
    out = []
    models = [
        ("baseline_active", base_parameters),
        ("coefficient_candidate", candidate_parameters),
        *(extra_models or []),
    ]
    for model_id, parameters in models:
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
    extra_models: list[tuple[str, dict[str, float]]] | None = None,
    replay_dt_s: float,
    alignment_window_s: float,
    workers: int,
) -> list[dict[str, Any]]:
    payloads = []
    models = [
        ("baseline_active", base_parameters),
        ("coefficient_candidate", candidate_parameters),
        *(extra_models or []),
    ]
    for model_id, parameters in models:
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
    warnings.filterwarnings("ignore", category=RuntimeWarning)
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
    model_ids = sorted({str(row.get("model_id", "")) for row in rows if str(row.get("model_id", ""))})
    for model_id in model_ids:
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
    fit_transition_pitch_moment: bool,
    fit_post_stall_lift_drag: bool,
    fit_post_stall_pitch_moment: bool,
    fit_post_stall_longitudinal: bool,
    fit_transition_blender: bool,
    fit_post_stall_surfaces: bool,
    fit_post_stall_damping: bool,
    fit_attached_lateral_coupling: bool,
    fit_transition_lateral_coupling: bool,
    fit_lateral_surfaces: bool,
    *,
    base_parameters: dict[str, float] | None = None,
    candidate_parameters: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    coeffs = fit_result.get("coefficients", zero_coefficients())
    base = dict(base_parameters or {})
    candidate = dict(candidate_parameters or {})

    def output_value(candidate_key: str, fit_key: str | None = None) -> float:
        if candidate_key in candidate:
            return float(candidate.get(candidate_key, 0.0))
        return float(coeffs.get(fit_key or candidate_key, 0.0))

    def changed(candidate_key: str, fallback: bool) -> bool:
        if candidate_key not in candidate:
            return bool(fallback)
        return abs(float(candidate.get(candidate_key, 0.0)) - float(base.get(candidate_key, 0.0))) > 1.0e-12

    rows = [
        {
            "parameter": "post_stall_surface_replay_scale",
            "value": float(
                fit_result.get("surface_scale_selection", {}).get("selected_surface_scale", 1.0)
            ),
            "applied_to_replay": bool(fit_post_stall_longitudinal),
            "description": "Train-replay-selected multiplier used by non-staged post-stall ablations.",
        },
        {
            "parameter": "attached_cm_bias_coeff",
            "value": output_value("attached_pitch_moment_bias_coeff", "attached_cm_bias_coeff"),
            "applied_to_replay": changed("attached_pitch_moment_bias_coeff", apply_attached_cm_bias),
            "description": "Attached-flow Cm value applied through attached_pitch_moment_bias_coeff when accepted.",
        },
        {
            "parameter": "transition_cm_bias_coeff",
            "value": output_value("transition_pitch_moment_bias_coeff", "transition_cm_bias_coeff"),
            "applied_to_replay": changed("transition_pitch_moment_bias_coeff", fit_transition_pitch_moment),
            "description": "Transition-window Cm value applied through transition_pitch_moment_bias_coeff when accepted.",
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
            "value": output_value("post_stall_lift_residual_coeff"),
            "applied_to_replay": changed(
                "post_stall_lift_residual_coeff",
                bool(fit_post_stall_longitudinal and fit_post_stall_lift_drag and not fit_post_stall_surfaces),
            ),
            "description": "Compact scalar post-stall CL value applied when the staged replay gate accepts lift cleanup.",
        },
        {
            "parameter": "post_stall_drag_residual_coeff",
            "value": output_value("post_stall_drag_residual_coeff"),
            "applied_to_replay": changed(
                "post_stall_drag_residual_coeff",
                bool(fit_post_stall_longitudinal and fit_post_stall_lift_drag and not fit_post_stall_surfaces),
            ),
            "description": "Compact scalar post-stall CD value applied when the staged replay gate accepts drag cleanup.",
        },
        {
            "parameter": "post_stall_pitch_moment_coeff",
            "value": output_value("post_stall_pitch_moment_coeff"),
            "applied_to_replay": changed(
                "post_stall_pitch_moment_coeff",
                bool(fit_post_stall_longitudinal and fit_post_stall_pitch_moment and not fit_post_stall_surfaces),
            ),
            "description": "Compact scalar post-stall Cm value applied only through the post-stall pitch weight when accepted.",
        },
        {
            "parameter": "post_stall_pitch_damping_coeff",
            "value": output_value("post_stall_pitch_damping_coeff"),
            "applied_to_replay": changed(
                "post_stall_pitch_damping_coeff",
                bool(fit_post_stall_longitudinal and fit_post_stall_damping),
            ),
            "description": "Positive-AoA Cmq-style value using q_hat = q c / (2V), applied only when accepted.",
        },
        {
            "parameter": "post_stall_residual_blend_start_alpha_deg",
            "value": output_value("post_stall_residual_blend_start_alpha_deg"),
            "applied_to_replay": changed("post_stall_residual_blend_start_alpha_deg", fit_transition_blender),
            "description": "Residual activation start alpha applied after the staged blend gate accepts it.",
        },
        {
            "parameter": "post_stall_residual_blend_full_alpha_deg",
            "value": output_value("post_stall_residual_blend_full_alpha_deg"),
            "applied_to_replay": changed("post_stall_residual_blend_full_alpha_deg", fit_transition_blender),
            "description": "Residual activation full-alpha point applied after the staged blend gate accepts it.",
        },
    ]
    for key in ATTACHED_LATERAL_PARAMETER_KEYS:
        rows.append(
            {
                "parameter": key,
                "value": output_value(key),
                "applied_to_replay": changed(
                    key,
                    bool(fit_attached_lateral_coupling and key in active_attached_lateral_parameter_keys()),
                ),
                "description": (
                    "Compact attached lateral residual term. Default primary fit leaves attached lateral terms report-only; "
                    "the secondary diagnostic applies only CY_beta, Cl_p, and Cn_r."
                ),
            }
        )
    for key in TRANSITION_LATERAL_PARAMETER_KEYS:
        rows.append(
            {
                "parameter": key,
                "value": output_value(key),
                "applied_to_replay": changed(
                    key,
                    bool(fit_transition_lateral_coupling and key in active_transition_lateral_parameter_keys()),
                ),
                "description": (
                    "Compact transition-window lateral delta multiplied by 4*smoothstep*(1-smoothstep). "
                    "Disabled by default; only CY_beta, Cl_p, and Cn_r are fit when enabled."
                ),
            }
        )
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
                    "value": output_value(key),
                    "applied_to_replay": changed(key, fit_post_stall_surfaces),
                    "description": (
                        f"Diagnostic rich post-stall {coefficient_name} alpha-RBF coefficient "
                        f"at {centre_deg:g} deg AoA. Disabled in the compact default."
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
                        "value": output_value(key),
                        "applied_to_replay": changed(key, fit_lateral_surfaces),
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
    alpha_deg = finite_value(row.get("alpha_deg"))
    if not math.isfinite(alpha_deg):
        return float("nan")
    attached_weight, transition_weight, post_weight = pitch_moment_regime_weights_deg(alpha_deg, coeffs)
    if math.isfinite(attached_weight):
        fitted += attached_weight * float(coeffs.get("attached_cm_bias_coeff", 0.0))
    if math.isfinite(transition_weight):
        fitted += transition_weight * float(coeffs.get("transition_cm_bias_coeff", 0.0))
    q_hat = finite_value(row.get("q_hat"))
    fitted_surface = surface_rbf_prediction(coeffs, "post_stall_pitch_moment_rbf", alpha_deg)
    damping_surface = surface_rbf_prediction(coeffs, "post_stall_pitch_damping_rbf", alpha_deg)
    if math.isfinite(post_weight):
        fitted += post_weight * float(coeffs.get("post_stall_pitch_moment_coeff", 0.0))
    if math.isfinite(fitted_surface):
        fitted += fitted_surface
    if math.isfinite(post_weight) and math.isfinite(q_hat):
        fitted += (
            float(coeffs.get("post_stall_pitch_damping_coeff", 0.0)) * post_weight
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
    loaded_throw_count: int,
    valid_rows: list[dict[str, Any]],
    filtered_throw_rows: list[dict[str, Any]],
    heldout_indices: set[int],
    heldout_seed: int,
    alignment_window_s: float,
    derivative_window_s: float,
    replay_dt_s: float,
    ridge_lambda: float,
    min_speed_m_s: float,
    workers: int,
    fit_workflow: str,
    group_iterations: int,
    group_improvement_tol: float,
    filter_aligned_launch_state: bool,
    aligned_u_min_m_s: float,
    aligned_u_max_m_s: float,
    aligned_v_abs_max_m_s: float,
    aligned_w_abs_max_m_s: float,
    apply_attached_cm_bias: bool,
    fit_transition_pitch_moment: bool,
    fit_post_stall_lift_drag: bool,
    fit_post_stall_pitch_moment: bool,
    fit_post_stall_longitudinal: bool,
    fit_transition_blender: bool,
    fit_post_stall_surfaces: bool,
    fit_post_stall_damping: bool,
    fit_attached_lateral_coupling: bool,
    fit_transition_lateral_coupling: bool,
    fit_lateral_surfaces: bool,
    fit_secondary_lateral_diagnostic: bool,
    base_parameters: dict[str, float],
    candidate_parameters: dict[str, float],
    lateral_diagnostic_parameters: dict[str, float] | None,
    fit_result: dict[str, Any],
    lateral_diagnostic_result: dict[str, Any],
    group_iteration_rows: list[dict[str, Any]],
    cm_stage_history_rows: list[dict[str, Any]],
    acceptance_summary: dict[str, Any],
    lateral_diagnostic_acceptance: dict[str, Any],
) -> None:
    manifest = {
        "fit_id": str(run_label),
        "fit_version": FIT_VERSION,
        "fit_scope": "neutral_open_loop_vicon_6dof_force_moment_residual_regime_staged_fit",
        "session_root": str(session_root),
        "loaded_throw_count": int(loaded_throw_count),
        "valid_throw_count": len(valid_rows),
        "filtered_throw_count": int(sum(not bool(row.get("kept", False)) for row in filtered_throw_rows)),
        "aligned_launch_filter": {
            "enabled": bool(filter_aligned_launch_state),
            "policy": "recheck replay-aligned start state before held-out split",
            "u_min_m_s": float(aligned_u_min_m_s),
            "u_max_m_s": float(aligned_u_max_m_s),
            "v_abs_max_m_s": float(aligned_v_abs_max_m_s),
            "w_abs_max_m_s": float(aligned_w_abs_max_m_s),
            "metrics_csv": "metrics/neutral_aero_residual_filtered_throws.csv",
        },
        "launch_confidence_weighting": launch_confidence_summary(filtered_throw_rows),
        "heldout_policy": "randomised_stratified_by_session_label",
        "heldout_seed": int(heldout_seed),
        "heldout_indices": sorted(int(index) for index in heldout_indices),
        "alignment_window_s": float(alignment_window_s),
        "derivative_window_s": float(derivative_window_s),
        "replay_dt_s": float(replay_dt_s),
        "ridge_lambda": float(ridge_lambda),
        "min_speed_m_s": float(min_speed_m_s),
        "workers": int(workers),
        "fit_workflow": str(fit_workflow),
        "group_iterations": int(group_iterations),
        "group_improvement_tol": float(group_improvement_tol),
        "apply_attached_cm_bias": bool(apply_attached_cm_bias),
        "fit_transition_pitch_moment": bool(fit_transition_pitch_moment),
        "fit_post_stall_lift_drag": bool(fit_post_stall_lift_drag),
        "fit_post_stall_pitch_moment": bool(fit_post_stall_pitch_moment),
        "fit_post_stall_longitudinal": bool(fit_post_stall_longitudinal),
        "fit_transition_blender": bool(fit_transition_blender),
        "fit_post_stall_surfaces": bool(fit_post_stall_surfaces),
        "fit_post_stall_damping": bool(fit_post_stall_damping),
        "fit_attached_lateral_coupling": bool(fit_attached_lateral_coupling),
        "fit_transition_lateral_coupling": bool(fit_transition_lateral_coupling),
        "fit_lateral_surfaces": bool(fit_lateral_surfaces),
        "fit_secondary_lateral_diagnostic": bool(fit_secondary_lateral_diagnostic),
        "group_iteration_history_count": len(group_iteration_rows),
        "cm_stage_history_count": len(cm_stage_history_rows),
        "stage_fit_policy": {
            "primary": "claim-bearing candidate is longitudinal-only by default; lateral residuals are reported but not claimed",
            "attached": "attached Cm is applied only through attached_pitch_moment_bias_coeff in the staged workflow",
            "transition": "transition Cm is applied only through transition_pitch_moment_bias_coeff; blend start/full is tested after Cm stages",
            "post_stall": "post-stall Cm/Cmq use post-stall pitch weights; CL/CD cleanup is a later compact stage",
            "secondary_lateral_diagnostic": "optional frozen-longitudinal CY_beta/Cl_p/Cn_r diagnostic, accepted only by held-out lateral improvement without longitudinal degradation",
            "staged_replay": "cm_regime_staged fits attached, transition, post-stall, blend, then optional CL/CD cleanup with held-out gates at each stage",
        },
        "base_parameters": dict(base_parameters),
        "candidate_parameters": dict(candidate_parameters),
        "lateral_diagnostic_parameters": dict(lateral_diagnostic_parameters or {}),
        "candidate_acceptance": acceptance_summary,
        "lateral_diagnostic_acceptance": lateral_diagnostic_acceptance,
        "fit_result": fit_result,
        "lateral_diagnostic_result": lateral_diagnostic_result,
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
            fit_workflow=fit_workflow,
            group_iterations=group_iterations,
            group_improvement_tol=group_improvement_tol,
            filter_aligned_launch_state=filter_aligned_launch_state,
            aligned_u_min_m_s=aligned_u_min_m_s,
            aligned_u_max_m_s=aligned_u_max_m_s,
            aligned_v_abs_max_m_s=aligned_v_abs_max_m_s,
            aligned_w_abs_max_m_s=aligned_w_abs_max_m_s,
            apply_attached_cm_bias=apply_attached_cm_bias,
            fit_transition_pitch_moment=fit_transition_pitch_moment,
            fit_post_stall_lift_drag=fit_post_stall_lift_drag,
            fit_post_stall_pitch_moment=fit_post_stall_pitch_moment,
            fit_post_stall_longitudinal=fit_post_stall_longitudinal,
            fit_transition_blender=fit_transition_blender,
            fit_post_stall_surfaces=fit_post_stall_surfaces,
            fit_post_stall_damping=fit_post_stall_damping,
            fit_attached_lateral_coupling=fit_attached_lateral_coupling,
            fit_transition_lateral_coupling=fit_transition_lateral_coupling,
            fit_lateral_surfaces=fit_lateral_surfaces,
            fit_secondary_lateral_diagnostic=fit_secondary_lateral_diagnostic,
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
    fit_workflow: str,
    group_iterations: int,
    group_improvement_tol: float,
    filter_aligned_launch_state: bool,
    aligned_u_min_m_s: float,
    aligned_u_max_m_s: float,
    aligned_v_abs_max_m_s: float,
    aligned_w_abs_max_m_s: float,
    apply_attached_cm_bias: bool,
    fit_transition_pitch_moment: bool,
    fit_post_stall_lift_drag: bool,
    fit_post_stall_pitch_moment: bool,
    fit_post_stall_longitudinal: bool,
    fit_transition_blender: bool,
    fit_post_stall_surfaces: bool,
    fit_post_stall_damping: bool,
    fit_attached_lateral_coupling: bool,
    fit_transition_lateral_coupling: bool,
    fit_lateral_surfaces: bool,
    fit_secondary_lateral_diagnostic: bool,
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
        "--fit-workflow",
        str(fit_workflow),
        "--group-iterations",
        str(int(group_iterations)),
        "--group-improvement-tol",
        f"{float(group_improvement_tol):.6g}",
        "--aligned-u-min-m-s",
        f"{float(aligned_u_min_m_s):.6g}",
        "--aligned-u-max-m-s",
        f"{float(aligned_u_max_m_s):.6g}",
        "--aligned-v-abs-max-m-s",
        f"{float(aligned_v_abs_max_m_s):.6g}",
        "--aligned-w-abs-max-m-s",
        f"{float(aligned_w_abs_max_m_s):.6g}",
    ]
    command.append("--filter-aligned-launch-state" if filter_aligned_launch_state else "--no-filter-aligned-launch-state")
    command.append("--apply-attached-cm-bias" if apply_attached_cm_bias else "--no-apply-attached-cm-bias")
    command.append(
        "--fit-transition-pitch-moment" if fit_transition_pitch_moment else "--no-fit-transition-pitch-moment"
    )
    command.append("--fit-post-stall-lift-drag" if fit_post_stall_lift_drag else "--no-fit-post-stall-lift-drag")
    command.append(
        "--fit-post-stall-pitch-moment" if fit_post_stall_pitch_moment else "--no-fit-post-stall-pitch-moment"
    )
    command.append(
        "--fit-post-stall-longitudinal" if fit_post_stall_longitudinal else "--no-fit-post-stall-longitudinal"
    )
    command.append("--fit-transition-blender" if fit_transition_blender else "--no-fit-transition-blender")
    command.append("--fit-post-stall-surfaces" if fit_post_stall_surfaces else "--no-fit-post-stall-surfaces")
    command.append("--fit-post-stall-damping" if fit_post_stall_damping else "--no-fit-post-stall-damping")
    command.append(
        "--fit-attached-lateral-coupling"
        if fit_attached_lateral_coupling
        else "--no-fit-attached-lateral-coupling"
    )
    command.append(
        "--fit-transition-lateral-coupling"
        if fit_transition_lateral_coupling
        else "--no-fit-transition-lateral-coupling"
    )
    command.append("--fit-lateral-surfaces" if fit_lateral_surfaces else "--no-fit-lateral-surfaces")
    command.append(
        "--fit-secondary-lateral-diagnostic"
        if fit_secondary_lateral_diagnostic
        else "--no-fit-secondary-lateral-diagnostic"
    )
    return command


def filtered_throw_report_lines(filtered_throw_rows: list[dict[str, Any]]) -> str:
    rejected = [row for row in filtered_throw_rows if not bool(row.get("kept", False))]
    if not rejected:
        return "- no logged-valid throws were rejected by the replay-start filter"
    lines = ["- rejected logged-valid throws:"]
    for row in rejected:
        lines.append(
            (
                f"  - `{row.get('session_label', '')}/{row.get('throw_id', '')}`: "
                f"{row.get('filter_reason', '')}; "
                f"u0 `{replay_fit.format_value(row.get('u0_m_s', ''))}`, "
                f"v0 `{replay_fit.format_value(row.get('v0_m_s', ''))}`, "
                f"w0 `{replay_fit.format_value(row.get('w0_m_s', ''))}` m/s"
            )
        )
    return "\n".join(lines)


def launch_confidence_summary(filtered_throw_rows: list[dict[str, Any]]) -> dict[str, Any]:
    kept = [row for row in filtered_throw_rows if bool(row.get("kept", False))]
    weights = finite_values(kept, "launch_confidence_weight")
    scores = finite_values(kept, "launch_lateral_score")
    return {
        "enabled": True,
        "policy": (
            "valid throws remain usable, but residual fitting is downweighted for replay-aligned starts "
            "with larger lateral contamination: |v0|, |p0|, |r0|, |phi0|, or |psi0| relative to the launch gate"
        ),
        "perfect_lateral_launch_reference": {
            "phi0_deg": 0.0,
            "psi0_deg": 0.0,
            "v0_m_s": 0.0,
            "p0_rad_s": 0.0,
            "r0_rad_s": 0.0,
        },
        "minimum_weight": float(DEFAULT_LAUNCH_CONFIDENCE_MIN_WEIGHT),
        "exponent": float(DEFAULT_LAUNCH_CONFIDENCE_EXPONENT),
        "kept_throw_count": len(kept),
        "weight_min": min(weights) if weights else float("nan"),
        "weight_mean": mean(weights),
        "weight_max": max(weights) if weights else float("nan"),
        "lateral_score_min": min(scores) if scores else float("nan"),
        "lateral_score_mean": mean(scores),
        "lateral_score_max": max(scores) if scores else float("nan"),
    }


def launch_confidence_report_lines(filtered_throw_rows: list[dict[str, Any]]) -> str:
    summary = launch_confidence_summary(filtered_throw_rows)
    return "\n".join(
        [
            "- launch-confidence weighting: enabled for residual coefficient fitting",
            (
                "- confidence reference: replay-aligned lateral contamination `phi0=psi0=v0=p0=r0=0`; "
                f"minimum weight `{DEFAULT_LAUNCH_CONFIDENCE_MIN_WEIGHT:.2f}`"
            ),
            (
                "- kept-throw confidence weight min/mean/max: "
                f"`{replay_fit.format_value(summary.get('weight_min', ''))}`, "
                f"`{replay_fit.format_value(summary.get('weight_mean', ''))}`, "
                f"`{replay_fit.format_value(summary.get('weight_max', ''))}`"
            ),
            (
                "- kept-throw lateral score min/mean/max: "
                f"`{replay_fit.format_value(summary.get('lateral_score_min', ''))}`, "
                f"`{replay_fit.format_value(summary.get('lateral_score_mean', ''))}`, "
                f"`{replay_fit.format_value(summary.get('lateral_score_max', ''))}`"
            ),
        ]
    )


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
    fit_workflow: str,
    group_iterations: int,
    group_improvement_tol: float,
    loaded_throw_count: int,
    filtered_throw_rows: list[dict[str, Any]],
    filter_aligned_launch_state: bool,
    aligned_u_min_m_s: float,
    aligned_u_max_m_s: float,
    aligned_v_abs_max_m_s: float,
    aligned_w_abs_max_m_s: float,
    apply_attached_cm_bias: bool,
    fit_transition_pitch_moment: bool,
    fit_post_stall_lift_drag: bool,
    fit_post_stall_pitch_moment: bool,
    fit_post_stall_longitudinal: bool,
    fit_transition_blender: bool,
    fit_post_stall_surfaces: bool,
    fit_post_stall_damping: bool,
    fit_attached_lateral_coupling: bool,
    fit_transition_lateral_coupling: bool,
    fit_lateral_surfaces: bool,
    fit_secondary_lateral_diagnostic: bool,
    fit_result: dict[str, Any],
    lateral_diagnostic_result: dict[str, Any],
    group_iteration_rows: list[dict[str, Any]],
    cm_stage_history_rows: list[dict[str, Any]],
    acceptance_summary: dict[str, Any],
    lateral_diagnostic_acceptance: dict[str, Any],
    regime_summary: list[dict[str, Any]],
    stage_fit_summary: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    stage_replay_rows: list[dict[str, Any]],
    base_parameters: dict[str, float],
    candidate_parameters: dict[str, float],
    lateral_diagnostic_parameters: dict[str, float] | None,
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
            fit_workflow=fit_workflow,
            group_iterations=group_iterations,
            group_improvement_tol=group_improvement_tol,
            filter_aligned_launch_state=filter_aligned_launch_state,
            aligned_u_min_m_s=aligned_u_min_m_s,
            aligned_u_max_m_s=aligned_u_max_m_s,
            aligned_v_abs_max_m_s=aligned_v_abs_max_m_s,
            aligned_w_abs_max_m_s=aligned_w_abs_max_m_s,
            apply_attached_cm_bias=apply_attached_cm_bias,
            fit_transition_pitch_moment=fit_transition_pitch_moment,
            fit_post_stall_lift_drag=fit_post_stall_lift_drag,
            fit_post_stall_pitch_moment=fit_post_stall_pitch_moment,
            fit_post_stall_longitudinal=fit_post_stall_longitudinal,
            fit_transition_blender=fit_transition_blender,
            fit_post_stall_surfaces=fit_post_stall_surfaces,
            fit_post_stall_damping=fit_post_stall_damping,
            fit_attached_lateral_coupling=fit_attached_lateral_coupling,
            fit_transition_lateral_coupling=fit_transition_lateral_coupling,
            fit_lateral_surfaces=fit_lateral_surfaces,
            fit_secondary_lateral_diagnostic=fit_secondary_lateral_diagnostic,
        )
    )
    lateral_diagnostic_heldout = replay_summary(validation_rows, "lateral_diagnostic_candidate", "heldout")
    lateral_diag_coeffs = lateral_diagnostic_result.get("fit_result", {}).get("coefficients", zero_coefficients())
    lines = [
        "# Neutral Aero Residual Regime Fit",
        "",
        "This run uses only neutral open-loop real throws. It estimates 6-DoF force/moment residuals from Vicon state trajectories, fits a claim-bearing longitudinal candidate by default, reports lateral residuals without claiming accurate lateral SysID, and validates the primary candidate by held-out dry-air replay. The default `cm_regime_staged` workflow fits attached Cm, transition Cm, post-stall Cm/Cmq, transition blend, and optional post-stall CL/CD cleanup in separate held-out-gated stages. The optional secondary lateral diagnostic freezes the longitudinal candidate and fits only `CY_beta`, `Cl_p`, and `Cn_r`; rich transition lateral deltas, post-stall lateral surfaces, and post-stall alpha-RBF longitudinal surfaces are diagnostic-only unless explicitly enabled.",
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
        f"- fit workflow: `{fit_workflow}`",
        f"- group iterations: `{group_iterations}`",
        f"- group improvement tolerance: `{group_improvement_tol:.3g}`",
        f"- aligned launch filter: `{filter_aligned_launch_state}`",
        f"- aligned launch filter bounds: `u=[{aligned_u_min_m_s:.2f}, {aligned_u_max_m_s:.2f}]` m/s, `|v|<={aligned_v_abs_max_m_s:.2f}` m/s, `|w|<={aligned_w_abs_max_m_s:.2f}` m/s",
        f"- apply attached Cm bias: `{apply_attached_cm_bias}`",
        f"- fit transition Cm bias: `{fit_transition_pitch_moment}`",
        f"- fit post-stall CL/CD cleanup: `{fit_post_stall_lift_drag}`",
        f"- fit post-stall Cm bias: `{fit_post_stall_pitch_moment}`",
        f"- fit compact post-stall longitudinal residuals: `{fit_post_stall_longitudinal}`",
        f"- fit transition blender: `{fit_transition_blender}`",
        f"- fit post-stall alpha-RBF surfaces: `{fit_post_stall_surfaces}`",
        f"- fit post-stall damping: `{fit_post_stall_damping}`",
        f"- fit attached lateral coupling: `{fit_attached_lateral_coupling}`",
        f"- fit transition lateral coupling: `{fit_transition_lateral_coupling}`",
        f"- fit lateral surfaces: `{fit_lateral_surfaces}`",
        f"- fit secondary lateral diagnostic: `{fit_secondary_lateral_diagnostic}`",
        "",
        "```powershell",
        rerun_command,
        "```",
        "",
        "## Aligned Launch Filter",
        "",
        f"- loaded logged-valid throws: `{loaded_throw_count}`",
        f"- kept throws after replay-start filter: `{sum(bool(row.get('kept', False)) for row in filtered_throw_rows)}`",
        f"- filtered throws: `{sum(not bool(row.get('kept', False)) for row in filtered_throw_rows)}`",
        "- filter audit CSV: `metrics/neutral_aero_residual_filtered_throws.csv`",
        launch_confidence_report_lines(filtered_throw_rows),
        filtered_throw_report_lines(filtered_throw_rows),
        "",
        "## Coefficient Fit",
        "",
        f"- fit status: `{fit_result.get('status', '')}`",
        f"- sample count: `{fit_result.get('sample_count', 0)}`",
        f"- used sample count: `{fit_result.get('used_sample_count', 0)}`",
        f"- post-stall used sample count: `{fit_result.get('post_stall_used_sample_count', 0)}`",
        f"- post-stall fit profile: `{fit_result.get('post_stall_fit_profile', '')}`",
        f"- fit MAE in Cm: `{float(fit_result.get('fit_mae_cm', float('nan'))):.5f}`",
        f"- attached Cm residual: `{float(coeffs.get('attached_cm_bias_coeff', 0.0)):.6g}`",
        f"- transition Cm residual before post-stall: `{float(coeffs.get('transition_before_post_stall_cm_bias_coeff', 0.0)):.6g}`",
        f"- transition Cm residual after post-stall: `{float(coeffs.get('transition_after_post_stall_cm_bias_coeff', 0.0)):.6g}`",
        lateral_coupling_report_lines(coeffs),
        "- primary lateral-coupling interpretation: report-only unless `--fit-attached-lateral-coupling` or other lateral flags are explicitly enabled; default primary SysID does not claim accurate lateral identification",
        f"- post-stall surface centres: `{', '.join(f'{centre:g}' for centre in SURFACE_RBF_ALPHA_CENTERS_DEG)}` deg (`diagnostic only unless fit_post_stall_surfaces=True`)",
        f"- post-stall surface width: `{SURFACE_RBF_ALPHA_WIDTH_DEG:.3g}` deg",
        surface_coeff_report_lines(coeffs),
        f"- post-stall Cmq residual: `{float(coeffs.get('post_stall_pitch_damping_coeff', 0.0)):.6g}`",
        f"- selected compact post-stall replay scale: `{float(surface_scale_selection.get('selected_surface_scale', 1.0)):.3f}`",
        f"- transition blender status: `{blend_fit.get('status', '')}`",
        f"- transition blender fit group: `{blend_fit.get('fit_group', '')}`",
        f"- transition blender start alpha: `{float(coeffs.get('post_stall_residual_blend_start_alpha_deg', STALL_ALPHA_DEG)):.3f}` deg",
        f"- transition blender full alpha: `{float(coeffs.get('post_stall_residual_blend_full_alpha_deg', POST_STALL_ALPHA_DEG)):.3f}` deg",
        "",
        "## Grouped Replay Refinement",
        "",
        f"- grouped history rows: `{len(group_iteration_rows)}`",
        "- grouped history CSV: `metrics/neutral_aero_residual_group_iteration_history.csv`",
        grouped_iteration_report_lines(group_iteration_rows),
        "",
        "## Regime-Staged Cm Workflow",
        "",
        f"- staged history rows: `{len(cm_stage_history_rows)}`",
        "- staged history CSV: `metrics/neutral_aero_residual_cm_stage_history.csv`",
        cm_stage_history_report_lines(cm_stage_history_rows),
        "",
        "## Secondary Lateral Diagnostic",
        "",
        f"- enabled: `{bool(lateral_diagnostic_result.get('enabled', False))}`",
        f"- status: `{lateral_diagnostic_result.get('status', '')}`",
        "- diagnostic coefficients CSV: `metrics/neutral_aero_residual_lateral_diagnostic_coefficients.csv`",
        "- diagnostic policy: longitudinal parameters are frozen at the primary candidate; only `CY_beta`, `Cl_p`, and `Cn_r` may change; launch-confidence weighting is ignored for this lateral-only fit",
        lateral_coupling_report_lines(lateral_diag_coeffs),
        (
            f"- lateral diagnostic held-out dy/roll/yaw MAE: "
            f"`{lateral_diagnostic_heldout.get('dy_mae_m', float('nan')):.4f}` m, "
            f"`{lateral_diagnostic_heldout.get('final_phi_mae_deg', float('nan')):.3f}` deg, "
            f"`{lateral_diagnostic_heldout.get('final_psi_mae_deg', float('nan')):.3f}` deg"
            if lateral_diagnostic_heldout
            else "- lateral diagnostic held-out dy/roll/yaw MAE: not available"
        ),
        f"- lateral diagnostic acceptance: `{'accepted' if lateral_diagnostic_acceptance.get('accepted', False) else 'rejected_diagnostic_only'}`",
        lateral_diagnostic_acceptance_report_lines(lateral_diagnostic_acceptance),
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
        f"- held-out acceptance: `{'accepted' if acceptance_summary.get('accepted', False) else 'rejected_diagnostic_only'}`",
        acceptance_report_lines(acceptance_summary),
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
        f"- legacy global Cm bias: baseline `{base_parameters['pitch_moment_bias_coeff']:.6g}`, candidate `{candidate_parameters['pitch_moment_bias_coeff']:.6g}`",
        f"- attached Cm bias: baseline `{base_parameters['attached_pitch_moment_bias_coeff']:.6g}`, candidate `{candidate_parameters['attached_pitch_moment_bias_coeff']:.6g}`",
        f"- transition Cm bias: baseline `{base_parameters['transition_pitch_moment_bias_coeff']:.6g}`, candidate `{candidate_parameters['transition_pitch_moment_bias_coeff']:.6g}`",
        f"- post-stall Cm bias: baseline `{base_parameters['post_stall_pitch_moment_coeff']:.6g}`, candidate `{candidate_parameters['post_stall_pitch_moment_coeff']:.6g}`",
        candidate_surface_report_lines(base_parameters, candidate_parameters),
        candidate_lateral_coupling_report_lines(base_parameters, candidate_parameters),
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
        "Accept the primary candidate only for the longitudinal claim-bearing model when held-out dx, altitude loss, sink, and pitch improve or preserve the active baseline. Treat dy, roll, and yaw as reported residual evidence unless the secondary lateral diagnostic improves held-out dy/roll/yaw without damaging those longitudinal metrics.",
    ]
    path = output_dir / "reports" / "neutral_aero_residual_fit_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def replay_summary(rows: list[dict[str, Any]], model_id: str, split: str) -> dict[str, float]:
    subset = [row for row in rows if row.get("model_id") == model_id and row.get("split") == split]
    return replay_fit.objective_summary(subset)


def candidate_acceptance_summary(validation_rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = replay_summary(validation_rows, "baseline_active", "heldout")
    candidate = replay_summary(validation_rows, "coefficient_candidate", "heldout")
    metric_keys = (
        "dx_mae_m",
        "altitude_loss_mae_m",
        "sink_mae_m_s",
        "final_theta_mae_deg",
    )
    metrics = []
    accepted = True
    for key in metric_keys:
        base_value = finite_value(baseline.get(key))
        candidate_value = finite_value(candidate.get(key))
        passed = math.isfinite(base_value) and math.isfinite(candidate_value) and candidate_value <= base_value + 1.0e-9
        metrics.append(
            {
                "metric": key,
                "baseline": base_value,
                "candidate": candidate_value,
                "delta_candidate_minus_baseline": candidate_value - base_value,
                "passed": bool(passed),
            }
        )
        accepted = accepted and passed
    return {
        "accepted": bool(accepted),
        "policy": (
            "primary longitudinal candidate must improve or preserve held-out dx, altitude loss, sink, "
            "and pitch MAE versus baseline_active; lateral errors are reported but not claim-bearing"
        ),
        "metrics": metrics,
    }


def lateral_diagnostic_acceptance_summary(validation_rows: list[dict[str, Any]]) -> dict[str, Any]:
    primary = replay_summary(validation_rows, "coefficient_candidate", "heldout")
    diagnostic = replay_summary(validation_rows, "lateral_diagnostic_candidate", "heldout")
    if not diagnostic:
        return {
            "accepted": False,
            "policy": "secondary lateral diagnostic not run",
            "metrics": [],
        }
    lateral_metric_keys = ("dy_mae_m", "final_phi_mae_deg", "final_psi_mae_deg")
    longitudinal_metric_keys = ("dx_mae_m", "altitude_loss_mae_m", "sink_mae_m_s", "final_theta_mae_deg")
    metrics = []
    accepted = True
    for key in lateral_metric_keys:
        primary_value = finite_value(primary.get(key))
        diagnostic_value = finite_value(diagnostic.get(key))
        passed = (
            math.isfinite(primary_value)
            and math.isfinite(diagnostic_value)
            and diagnostic_value < primary_value - 1.0e-9
        )
        metrics.append(
            {
                "metric": key,
                "primary": primary_value,
                "diagnostic": diagnostic_value,
                "delta_diagnostic_minus_primary": diagnostic_value - primary_value,
                "role": "lateral_improvement_required",
                "passed": bool(passed),
            }
        )
        accepted = accepted and passed
    for key in longitudinal_metric_keys:
        primary_value = finite_value(primary.get(key))
        diagnostic_value = finite_value(diagnostic.get(key))
        passed = (
            math.isfinite(primary_value)
            and math.isfinite(diagnostic_value)
            and diagnostic_value <= primary_value + 1.0e-9
        )
        metrics.append(
            {
                "metric": key,
                "primary": primary_value,
                "diagnostic": diagnostic_value,
                "delta_diagnostic_minus_primary": diagnostic_value - primary_value,
                "role": "longitudinal_preservation_required",
                "passed": bool(passed),
            }
        )
        accepted = accepted and passed
    return {
        "accepted": bool(accepted),
        "policy": (
            "secondary lateral diagnostic is reportable only if held-out dy, roll, and yaw improve versus "
            "the primary longitudinal candidate while dx, altitude loss, sink, and pitch do not degrade"
        ),
        "metrics": metrics,
    }


def acceptance_report_lines(summary: dict[str, Any]) -> str:
    lines = [f"- acceptance policy: {summary.get('policy', '')}"]
    for row in summary.get("metrics", []):
        lines.append(
            f"  - {row.get('metric', '')}: baseline `{finite_value(row.get('baseline')):.4f}`, "
            f"candidate `{finite_value(row.get('candidate')):.4f}`, "
            f"delta `{finite_value(row.get('delta_candidate_minus_baseline')):.4f}`, "
            f"pass `{bool(row.get('passed', False))}`"
        )
    return "\n".join(lines)


def lateral_diagnostic_acceptance_report_lines(summary: dict[str, Any]) -> str:
    lines = [f"- lateral diagnostic policy: {summary.get('policy', '')}"]
    for row in summary.get("metrics", []):
        lines.append(
            f"  - {row.get('metric', '')} ({row.get('role', '')}): primary "
            f"`{finite_value(row.get('primary')):.4f}`, diagnostic "
            f"`{finite_value(row.get('diagnostic')):.4f}`, delta "
            f"`{finite_value(row.get('delta_diagnostic_minus_primary')):.4f}`, "
            f"pass `{bool(row.get('passed', False))}`"
        )
    return "\n".join(lines)


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


def lateral_coupling_report_lines(coeffs: dict[str, float]) -> str:
    lines = ["- attached lateral coupling:"]
    for group, label in (("side_force", "CY"), ("roll_moment", "Cl"), ("yaw_moment", "Cn")):
        keys = lateral_coupling_coeff_keys(group, transition=False)
        lines.append(
            f"  - {label}: bias `{float(coeffs.get(keys[0], 0.0)):.6g}`, "
            f"beta `{float(coeffs.get(keys[1], 0.0)):.6g}`, "
            f"p_hat `{float(coeffs.get(keys[2], 0.0)):.6g}`, "
            f"r_hat `{float(coeffs.get(keys[3], 0.0)):.6g}`"
        )
    lines.append("- transition-window lateral coupling:")
    for group, label in (("side_force", "CY"), ("roll_moment", "Cl"), ("yaw_moment", "Cn")):
        keys = lateral_coupling_coeff_keys(group, transition=True)
        lines.append(
            f"  - {label}: bias `{float(coeffs.get(keys[0], 0.0)):.6g}`, "
            f"beta `{float(coeffs.get(keys[1], 0.0)):.6g}`, "
            f"p_hat `{float(coeffs.get(keys[2], 0.0)):.6g}`, "
            f"r_hat `{float(coeffs.get(keys[3], 0.0)):.6g}`"
        )
    return "\n".join(lines)


def grouped_iteration_report_lines(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "- grouped replay refinement disabled"
    selected = [row for row in rows if bool(row.get("selected", False))]
    if not selected:
        return "- no grouped replay scale was selected"
    lines = ["- selected grouped replay scales:"]
    for row in selected:
        lines.append(
            f"  - pass `{row.get('pass_index', '')}`, group `{row.get('group', '')}`, "
            f"scale `{replay_fit.format_value(row.get('candidate_scale', ''))}`, "
            f"objective `{finite_value(row.get('objective')):.4f}`"
        )
    return "\n".join(lines)


def cm_stage_history_report_lines(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "- regime-staged Cm workflow disabled"
    lines = ["- staged decisions:"]
    for row in rows:
        lines.append(
            (
                f"  - `{row.get('stage_id', '')}` `{row.get('fit_parameter_group', '')}`: "
                f"accepted `{bool(row.get('accepted', False))}`, "
                f"held-out pitch `{finite_value(row.get('heldout_pitch_mae_before_deg')):.3f}` -> "
                f"`{finite_value(row.get('heldout_pitch_mae_after_deg')):.3f}` deg, "
                f"altitude-loss `{finite_value(row.get('heldout_altitude_loss_mae_before_m')):.4f}` -> "
                f"`{finite_value(row.get('heldout_altitude_loss_mae_after_m')):.4f}` m"
            )
        )
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


def candidate_lateral_coupling_report_lines(base_parameters: dict[str, float], candidate_parameters: dict[str, float]) -> str:
    lines = []
    for heading, keys in (
        ("attached lateral coupling", ATTACHED_LATERAL_PARAMETER_KEYS),
        ("transition lateral coupling", TRANSITION_LATERAL_PARAMETER_KEYS),
    ):
        changed = [
            (
                key,
                float(base_parameters.get(key, 0.0)),
                float(candidate_parameters.get(key, 0.0)),
            )
            for key in keys
            if abs(float(base_parameters.get(key, 0.0)) - float(candidate_parameters.get(key, 0.0))) > 1.0e-12
        ]
        if not changed:
            lines.append(f"- {heading}: unchanged")
            continue
        entries = [f"{key} `{old:.6g}` -> `{new:.6g}`" for key, old, new in changed]
        lines.append(f"- {heading}: " + "; ".join(entries))
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
