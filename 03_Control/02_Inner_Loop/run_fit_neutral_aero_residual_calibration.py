"""Neutral-only aero residual identification from Vicon trajectories.

Pipeline:
    Vicon trajectory -> force/moment residuals -> regime-split coefficient fit
    -> held-out replay validation.

Pulse/control-effectiveness throws are intentionally excluded.
"""

from __future__ import annotations

import argparse
import csv
import itertools
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


FIT_VERSION = "N20_compact_joint_sweep_from_active"
DEFAULT_FIT_WORKFLOW = "cm_regime_staged"
DEFAULT_SESSION_ROOT = REPO_ROOT / "04_Flight_Test" / "05_Results" / "cal" / "n30"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "03_Control" / "05_Results" / "glider_model_calibration_prep"
DEFAULT_RUN_LABEL = "n30_neutral_aero_residual_fit"
DEFAULT_WORKERS = 8
DEFAULT_HELDOUT_COUNT = 0
DEFAULT_HELDOUT_FRACTION = 0.15
DEFAULT_HELDOUT_SEED = 606
DEFAULT_ALIGNMENT_WINDOW_S = 0.040
DEFAULT_SENSITIVITY_ALIGNMENT_WINDOWS_S: tuple[float, ...] = ()
DEFAULT_JOINT_PARETO_AUDIT = True
DEFAULT_JOINT_PARETO_AUDIT_ALIGNMENT_WINDOW_S = 0.040
DEFAULT_JOINT_PARETO_PROFILE = "small"
DEFAULT_DERIVATIVE_WINDOW_S = 0.040
DEFAULT_REPLAY_DT_S = 0.005
DEFAULT_RIDGE_LAMBDA = 1.0e-3
DEFAULT_MIN_SPEED_M_S = 1.5
DEFAULT_FILTER_ALIGNED_LAUNCH_STATE = True
# Replay starts at the synchronized 40 ms launch-handoff boundary. Keep this as
# a relaxed post-alignment sanity bound, not a second copy of the real launch gate.
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
LONGITUDINAL_STAGE_DX_TOL_M = 0.02
LONGITUDINAL_STAGE_ALTITUDE_LOSS_TOL_M = 0.02
LONGITUDINAL_STAGE_SINK_TOL_M_S = 0.02
LONGITUDINAL_STAGE_PITCH_TOL_DEG = 0.50
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
COMPACT_JOINT_SWEEP_BEAM_WIDTH = 6
COMPACT_JOINT_SWEEP_HELDOUT_EVAL_LIMIT = 24
COMPACT_JOINT_SWEEP_MULTIPLIERS = (0.0, 0.5, 0.75, 1.0, 1.25, 1.5)
JOINT_PARETO_AUDIT_TOP_LONGITUDINAL = 4
JOINT_PARETO_AUDIT_TOP_LATERAL = 4
JOINT_PARETO_AUDIT_SELECTED_LIMIT = 8
JOINT_PARETO_HEAVY_SCALE_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
JOINT_PARETO_PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "small": {
        "top_longitudinal": JOINT_PARETO_AUDIT_TOP_LONGITUDINAL,
        "top_lateral": JOINT_PARETO_AUDIT_TOP_LATERAL,
        "max_lateral_order": 1,
        "top_triples": 0,
        "max_candidates": 64,
        "selected_limit": JOINT_PARETO_AUDIT_SELECTED_LIMIT,
        "scale_grid": (1.0,),
        "scaled_single_term_bundles": False,
        "include_rejected_stage_candidates": False,
        "reference_policy": "global_best",
    },
    "heavy": {
        "top_longitudinal": 6,
        "top_lateral": 10,
        "max_lateral_order": 3,
        "top_triples": 80,
        "max_candidates": 900,
        "selected_limit": 8,
        "scale_grid": JOINT_PARETO_HEAVY_SCALE_GRID,
        "scaled_single_term_bundles": True,
        "include_rejected_stage_candidates": True,
        "reference_policy": "per_base",
    },
}
COMPACT_JOINT_SWEEP_BLEND_GRID_DEG = (
    (12.0, 20.0),
    (12.0, 22.0),
    (12.0, 24.0),
    (14.0, 20.0),
    (14.0, 22.0),
    (14.0, 24.0),
)
JOINT_SWEEP_STRICT_LONGITUDINAL_TOLERANCES = {
    "dx_mae_m": LONGITUDINAL_STAGE_DX_TOL_M,
    "altitude_loss_mae_m": LONGITUDINAL_STAGE_ALTITUDE_LOSS_TOL_M,
    "sink_mae_m_s": LONGITUDINAL_STAGE_SINK_TOL_M_S,
    "final_theta_mae_deg": LONGITUDINAL_STAGE_PITCH_TOL_DEG,
}
JOINT_SWEEP_BALANCED_LONGITUDINAL_TOLERANCES = {
    "dx_mae_m": 0.05,
    "altitude_loss_mae_m": 0.04,
    "sink_mae_m_s": 0.04,
    "final_theta_mae_deg": 1.0,
}
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
LATERAL_ABLATION_TERMS = (
    {
        "term": "CY_beta",
        "group": "side_force",
        "residual_key": "cy",
        "feature_key": "beta_rad",
        "attached_key": "side_force_beta_coeff",
        "transition_key": "transition_side_force_beta_coeff",
        "post_stall_prefix": "post_stall_side_force",
        "post_stall_feature": "beta",
    },
    {
        "term": "CY_p",
        "group": "side_force",
        "residual_key": "cy",
        "feature_key": "p_hat",
        "attached_key": "side_force_p_hat_coeff",
        "transition_key": "transition_side_force_p_hat_coeff",
        "post_stall_prefix": "post_stall_side_force",
        "post_stall_feature": "p_hat",
    },
    {
        "term": "CY_r",
        "group": "side_force",
        "residual_key": "cy",
        "feature_key": "r_hat",
        "attached_key": "side_force_r_hat_coeff",
        "transition_key": "transition_side_force_r_hat_coeff",
        "post_stall_prefix": "post_stall_side_force",
        "post_stall_feature": "r_hat",
    },
    {
        "term": "Cl_beta",
        "group": "roll_moment",
        "residual_key": "cl_roll",
        "feature_key": "beta_rad",
        "attached_key": "roll_moment_beta_coeff",
        "transition_key": "transition_roll_moment_beta_coeff",
        "post_stall_prefix": "post_stall_roll_moment",
        "post_stall_feature": "beta",
    },
    {
        "term": "Cl_p",
        "group": "roll_moment",
        "residual_key": "cl_roll",
        "feature_key": "p_hat",
        "attached_key": "roll_moment_p_hat_coeff",
        "transition_key": "transition_roll_moment_p_hat_coeff",
        "post_stall_prefix": "post_stall_roll_moment",
        "post_stall_feature": "p_hat",
    },
    {
        "term": "Cl_r",
        "group": "roll_moment",
        "residual_key": "cl_roll",
        "feature_key": "r_hat",
        "attached_key": "roll_moment_r_hat_coeff",
        "transition_key": "transition_roll_moment_r_hat_coeff",
        "post_stall_prefix": "post_stall_roll_moment",
        "post_stall_feature": "r_hat",
    },
    {
        "term": "Cn_beta",
        "group": "yaw_moment",
        "residual_key": "cn_yaw",
        "feature_key": "beta_rad",
        "attached_key": "yaw_moment_beta_coeff",
        "transition_key": "transition_yaw_moment_beta_coeff",
        "post_stall_prefix": "post_stall_yaw_moment",
        "post_stall_feature": "beta",
    },
    {
        "term": "Cn_p",
        "group": "yaw_moment",
        "residual_key": "cn_yaw",
        "feature_key": "p_hat",
        "attached_key": "yaw_moment_p_hat_coeff",
        "transition_key": "transition_yaw_moment_p_hat_coeff",
        "post_stall_prefix": "post_stall_yaw_moment",
        "post_stall_feature": "p_hat",
    },
    {
        "term": "Cn_r",
        "group": "yaw_moment",
        "residual_key": "cn_yaw",
        "feature_key": "r_hat",
        "attached_key": "yaw_moment_r_hat_coeff",
        "transition_key": "transition_yaw_moment_r_hat_coeff",
        "post_stall_prefix": "post_stall_yaw_moment",
        "post_stall_feature": "r_hat",
    },
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
    "phi0_deg",
    "psi0_deg",
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
    "proposed_parameter_updates_json",
]
REPLAY_VALIDATION_FIELDS = ["model_id", *replay_fit.REPLAY_RESIDUAL_FIELDS]
REPLAY_SENSITIVITY_FIELDS = [
    "primary_alignment_window_s",
    "sensitivity_alignment_window_s",
    *REPLAY_VALIDATION_FIELDS,
]
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
STAGE_REPLAY_SENSITIVITY_FIELDS = [
    "primary_alignment_window_s",
    "sensitivity_alignment_window_s",
    *STAGE_REPLAY_SUMMARY_FIELDS,
]
LATERAL_ABLATION_FIELDS = [
    "candidate_id",
    "baseline_model_id",
    "term",
    "regime_family",
    "split",
    "fit_sample_count",
    "fit_used_sample_count",
    "fit_coefficient",
    "fit_sign",
    "parameter_keys_json",
    "baseline_dy_mae_m",
    "candidate_dy_mae_m",
    "delta_dy_mae_m",
    "baseline_roll_mae_deg",
    "candidate_roll_mae_deg",
    "delta_roll_mae_deg",
    "baseline_yaw_mae_deg",
    "candidate_yaw_mae_deg",
    "delta_yaw_mae_deg",
    "baseline_dx_mae_m",
    "candidate_dx_mae_m",
    "delta_dx_mae_m",
    "baseline_altitude_loss_mae_m",
    "candidate_altitude_loss_mae_m",
    "delta_altitude_loss_mae_m",
    "baseline_sink_mae_m_s",
    "candidate_sink_mae_m_s",
    "delta_sink_mae_m_s",
    "baseline_pitch_mae_deg",
    "candidate_pitch_mae_deg",
    "delta_pitch_mae_deg",
    "accepted",
    "acceptance_reason",
]
LATERAL_LAUNCH_CORRELATION_FIELDS = [
    "model_id",
    "split",
    "residual_metric",
    "launch_variable",
    "sample_count",
    "pearson_r",
    "abs_pearson_r",
    "slope",
    "intercept",
    "residual_mean",
    "launch_variable_mean",
]
JOINT_SWEEP_CANDIDATE_FIELDS = [
    "candidate_id",
    "selection_class",
    "sweep_stage",
    "split",
    "score",
    "longitudinal_score",
    "lateral_score",
    "parameter_count",
    "lateral_parameter_count",
    "dx_mae_m",
    "dy_mae_m",
    "altitude_loss_mae_m",
    "sink_mae_m_s",
    "roll_mae_deg",
    "pitch_mae_deg",
    "yaw_mae_deg",
    "parameter_updates_json",
]
JOINT_SWEEP_SELECTED_FIELDS = [
    "selection_class",
    "candidate_id",
    "selection_reason",
    "score",
    "longitudinal_score",
    "lateral_score",
    "parameter_count",
    "lateral_parameter_count",
    "dx_mae_m",
    "dy_mae_m",
    "altitude_loss_mae_m",
    "sink_mae_m_s",
    "roll_mae_deg",
    "pitch_mae_deg",
    "yaw_mae_deg",
    "parameter_updates_json",
]
JOINT_PARETO_AUDIT_FIELDS = [
    "candidate_id",
    "selection_class",
    "joint_pareto_profile",
    "accepted",
    "acceptance_reason",
    "pareto_member",
    "alignment_window_s",
    "reference_candidate_id",
    "global_reference_candidate_id",
    "longitudinal_source_id",
    "lateral_source_id",
    "split",
    "score",
    "longitudinal_score",
    "reference_longitudinal_score",
    "delta_longitudinal_score",
    "lateral_score",
    "reference_lateral_score",
    "delta_lateral_score",
    "global_reference_longitudinal_score",
    "global_delta_longitudinal_score",
    "global_reference_lateral_score",
    "global_delta_lateral_score",
    "parameter_count",
    "lateral_parameter_count",
    "dx_mae_m",
    "reference_dx_mae_m",
    "delta_dx_mae_m",
    "global_reference_dx_mae_m",
    "global_delta_dx_mae_m",
    "dy_mae_m",
    "reference_dy_mae_m",
    "delta_dy_mae_m",
    "global_reference_dy_mae_m",
    "global_delta_dy_mae_m",
    "altitude_loss_mae_m",
    "reference_altitude_loss_mae_m",
    "delta_altitude_loss_mae_m",
    "global_reference_altitude_loss_mae_m",
    "global_delta_altitude_loss_mae_m",
    "sink_mae_m_s",
    "reference_sink_mae_m_s",
    "delta_sink_mae_m_s",
    "global_reference_sink_mae_m_s",
    "global_delta_sink_mae_m_s",
    "roll_mae_deg",
    "reference_roll_mae_deg",
    "delta_roll_mae_deg",
    "global_reference_roll_mae_deg",
    "global_delta_roll_mae_deg",
    "pitch_mae_deg",
    "reference_pitch_mae_deg",
    "delta_pitch_mae_deg",
    "global_reference_pitch_mae_deg",
    "global_delta_pitch_mae_deg",
    "yaw_mae_deg",
    "reference_yaw_mae_deg",
    "delta_yaw_mae_deg",
    "global_reference_yaw_mae_deg",
    "global_delta_yaw_mae_deg",
    "parameter_updates_json",
]
JOINT_PARETO_HEAVY_STAGE_REPLAY_FIELDS = [
    "model_id",
    "model_role",
    "matched_candidate_id",
    "split",
    "regime",
    "report_regime",
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
        sensitivity_alignment_windows_s=(
            ()
            if args.no_sensitivity_alignment
            else tuple(args.sensitivity_alignment_window_s or DEFAULT_SENSITIVITY_ALIGNMENT_WINDOWS_S)
        ),
        joint_pareto_audit=bool(args.joint_pareto_audit),
        joint_pareto_audit_alignment_window_s=args.joint_pareto_audit_alignment_window_s,
        joint_pareto_profile=args.joint_pareto_profile,
        joint_pareto_top_longitudinal=args.joint_pareto_top_longitudinal,
        joint_pareto_top_lateral=args.joint_pareto_top_lateral,
        joint_pareto_max_lateral_order=args.joint_pareto_max_lateral_order,
        joint_pareto_top_triples=args.joint_pareto_top_triples,
        joint_pareto_max_candidates=args.joint_pareto_max_candidates,
        joint_pareto_selected_limit=args.joint_pareto_selected_limit,
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
    parser.add_argument(
        "--sensitivity-alignment-window-s",
        type=float,
        action="append",
        default=None,
        help=(
            "Replay-only diagnostic alignment window. Repeat to add windows. "
            "Disabled by default now that 0.040 s is the primary alignment; does not change fitted coefficients or acceptance gates."
        ),
    )
    parser.add_argument(
        "--no-sensitivity-alignment",
        action="store_true",
        help="Disable replay-only alignment-window sensitivity diagnostics.",
    )
    parser.add_argument(
        "--joint-pareto-audit",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_JOINT_PARETO_AUDIT,
        help=(
            "Run a small held-out 0.040 s joint Pareto audit combining top longitudinal and compact "
            "lateral/cross-coupling candidates. Diagnostic only; it does not promote coefficients."
        ),
    )
    parser.add_argument(
        "--joint-pareto-audit-alignment-window-s",
        type=float,
        default=DEFAULT_JOINT_PARETO_AUDIT_ALIGNMENT_WINDOW_S,
        help="Replay alignment window used by the diagnostic joint Pareto audit.",
    )
    parser.add_argument(
        "--joint-pareto-profile",
        choices=tuple(JOINT_PARETO_PROFILE_DEFAULTS),
        default=DEFAULT_JOINT_PARETO_PROFILE,
        help="Diagnostic joint Pareto search scale. heavy keeps the same parameter family but expands candidate combinations.",
    )
    parser.add_argument("--joint-pareto-top-longitudinal", type=int, default=None)
    parser.add_argument("--joint-pareto-top-lateral", type=int, default=None)
    parser.add_argument("--joint-pareto-max-lateral-order", type=int, default=None)
    parser.add_argument("--joint-pareto-top-triples", type=int, default=None)
    parser.add_argument("--joint-pareto-max-candidates", type=int, default=None)
    parser.add_argument("--joint-pareto-selected-limit", type=int, default=None)
    parser.add_argument("--derivative-window-s", type=float, default=DEFAULT_DERIVATIVE_WINDOW_S)
    parser.add_argument("--replay-dt-s", type=float, default=DEFAULT_REPLAY_DT_S)
    parser.add_argument("--ridge-lambda", type=float, default=DEFAULT_RIDGE_LAMBDA)
    parser.add_argument("--min-speed-m-s", type=float, default=DEFAULT_MIN_SPEED_M_S)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--fit-workflow",
        choices=("cm_regime_staged", "compact_joint_sweep", "grouped_iterative", "residual_only"),
        default=DEFAULT_FIT_WORKFLOW,
        help=(
            "cm_regime_staged is the default claim workflow. compact_joint_sweep starts from active constants, "
            "discovers compact signs/ranges, and jointly sweeps supported terms. grouped_iterative keeps the "
            "older block-scale coordinate refinement; residual_only applies the direct residual coefficients."
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
            "candidate with longitudinal terms frozen and excitation-aware lateral weighting."
        ),
    )
    return parser


def normalized_sensitivity_alignment_windows(
    primary_alignment_window_s: float,
    sensitivity_alignment_windows_s: tuple[float, ...],
) -> tuple[float, ...]:
    primary = float(primary_alignment_window_s)
    out: list[float] = []
    for raw_window in sensitivity_alignment_windows_s:
        window = float(raw_window)
        if not math.isfinite(window) or window <= 0.0:
            raise ValueError(f"Sensitivity alignment window must be positive and finite, got {raw_window!r}.")
        if abs(window - primary) <= 1.0e-12:
            continue
        if any(abs(window - existing) <= 1.0e-12 for existing in out):
            continue
        out.append(window)
    return tuple(out)


def resolved_joint_pareto_profile_config(
    *,
    profile: str,
    top_longitudinal: int | None,
    top_lateral: int | None,
    max_lateral_order: int | None,
    top_triples: int | None,
    max_candidates: int | None,
    selected_limit: int | None,
) -> dict[str, Any]:
    if str(profile) not in JOINT_PARETO_PROFILE_DEFAULTS:
        raise ValueError(f"Unsupported joint Pareto profile: {profile!r}.")
    config = dict(JOINT_PARETO_PROFILE_DEFAULTS[str(profile)])
    config["profile"] = str(profile)
    overrides = {
        "top_longitudinal": top_longitudinal,
        "top_lateral": top_lateral,
        "max_lateral_order": max_lateral_order,
        "top_triples": top_triples,
        "max_candidates": max_candidates,
        "selected_limit": selected_limit,
    }
    for key, value in overrides.items():
        if value is None:
            continue
        if int(value) < 0:
            raise ValueError(f"Joint Pareto {key} must be non-negative, got {value!r}.")
        config[key] = int(value)
    config["top_longitudinal"] = max(1, int(config["top_longitudinal"]))
    config["top_lateral"] = max(0, int(config["top_lateral"]))
    config["max_lateral_order"] = max(1, min(3, int(config["max_lateral_order"])))
    config["top_triples"] = max(0, int(config["top_triples"]))
    config["max_candidates"] = max(1, int(config["max_candidates"]))
    config["selected_limit"] = max(1, int(config["selected_limit"]))
    config["scale_grid"] = tuple(float(value) for value in config["scale_grid"])
    if "longitudinal_tolerances" in config:
        config["longitudinal_tolerances"] = {
            str(key): float(value)
            for key, value in dict(config["longitudinal_tolerances"]).items()
        }
    return config


def run_fit(
    *,
    session_root: Path,
    output_root: Path,
    run_label: str,
    heldout_count: int,
    heldout_fraction: float,
    heldout_seed: int,
    alignment_window_s: float,
    sensitivity_alignment_windows_s: tuple[float, ...],
    joint_pareto_audit: bool,
    joint_pareto_audit_alignment_window_s: float,
    joint_pareto_profile: str,
    joint_pareto_top_longitudinal: int | None,
    joint_pareto_top_lateral: int | None,
    joint_pareto_max_lateral_order: int | None,
    joint_pareto_top_triples: int | None,
    joint_pareto_max_candidates: int | None,
    joint_pareto_selected_limit: int | None,
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
    sensitivity_alignment_windows = normalized_sensitivity_alignment_windows(
        alignment_window_s,
        sensitivity_alignment_windows_s,
    )
    joint_pareto_config = resolved_joint_pareto_profile_config(
        profile=joint_pareto_profile,
        top_longitudinal=joint_pareto_top_longitudinal,
        top_lateral=joint_pareto_top_lateral,
        max_lateral_order=joint_pareto_max_lateral_order,
        top_triples=joint_pareto_top_triples,
        max_candidates=joint_pareto_max_candidates,
        selected_limit=joint_pareto_selected_limit,
    )

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
    requested_heldout_count = resolved_heldout_count(
        filtered_valid_count=len(valid_rows),
        heldout_count=heldout_count,
        heldout_fraction=heldout_fraction,
    )
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
    joint_sweep_candidate_rows: list[dict[str, Any]] = []
    joint_sweep_pareto_rows: list[dict[str, Any]] = []
    joint_sweep_selected_rows: list[dict[str, Any]] = []
    joint_sweep_extra_models: list[tuple[str, dict[str, float]]] = []
    joint_pareto_audit_candidate_rows: list[dict[str, Any]] = []
    joint_pareto_audit_selected_rows: list[dict[str, Any]] = []
    joint_pareto_heavy_stage_replay_rows: list[dict[str, Any]] = []
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
    elif fit_workflow == "compact_joint_sweep":
        (
            candidate_parameters,
            joint_sweep_candidate_rows,
            joint_sweep_pareto_rows,
            joint_sweep_selected_rows,
            joint_sweep_extra_models,
        ) = compact_joint_sweep_refinement(
            base_parameters=base_parameters,
            fit_result=fit_result,
            train_residuals=train_residuals,
            train_rows=train_rows,
            heldout_rows=heldout_rows,
            replay_dt_s=replay_dt_s,
            alignment_window_s=alignment_window_s,
            ridge_lambda=ridge_lambda,
            workers=workers,
        )
        fit_result["compact_joint_sweep"] = {
            "enabled": True,
            "candidates_csv": "metrics/neutral_aero_residual_joint_sweep_candidates.csv",
            "pareto_csv": "metrics/neutral_aero_residual_joint_sweep_pareto.csv",
            "selected_csv": "metrics/neutral_aero_residual_joint_sweep_selected.csv",
            "beam_width": COMPACT_JOINT_SWEEP_BEAM_WIDTH,
            "heldout_eval_limit": COMPACT_JOINT_SWEEP_HELDOUT_EVAL_LIMIT,
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
    extra_validation_models.extend(joint_sweep_extra_models)
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
    lateral_ablation_rows = lateral_ablation_diagnostic_rows(
        train_rows=train_rows,
        heldout_rows=heldout_rows,
        primary_parameters=candidate_parameters,
        primary_validation_rows=validation_rows,
        alignment_window_s=alignment_window_s,
        derivative_window_s=derivative_window_s,
        min_speed_m_s=min_speed_m_s,
        ridge_lambda=ridge_lambda,
        replay_dt_s=replay_dt_s,
        workers=workers,
    )
    joint_pareto_audit_candidate_rows, joint_pareto_audit_selected_rows = small_joint_pareto_audit(
        base_parameters=base_parameters,
        candidate_parameters=candidate_parameters,
        fit_result=fit_result,
        lateral_diagnostic_parameters=lateral_diagnostic_parameters,
        cm_stage_history_rows=cm_stage_history_rows,
        joint_sweep_extra_models=joint_sweep_extra_models,
        lateral_ablation_rows=lateral_ablation_rows,
        heldout_rows=heldout_rows,
        replay_dt_s=replay_dt_s,
        alignment_window_s=joint_pareto_audit_alignment_window_s,
        workers=workers,
        enabled=joint_pareto_audit,
        config=joint_pareto_config,
    )
    if str(joint_pareto_config["profile"]) == "heavy":
        joint_pareto_heavy_stage_replay_rows = joint_pareto_heavy_stage_replay_summary_rows(
            candidate_rows=joint_pareto_audit_candidate_rows,
            selected_rows=joint_pareto_audit_selected_rows,
            base_parameters=base_parameters,
            heldout_rows=heldout_rows,
            replay_dt_s=replay_dt_s,
            alignment_window_s=joint_pareto_audit_alignment_window_s,
            workers=workers,
        )
    lateral_launch_correlation_rows = lateral_launch_correlation_report_rows(validation_rows)
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
    sensitivity_validation_rows: list[dict[str, Any]] = []
    sensitivity_stage_replay_rows: list[dict[str, Any]] = []
    for sensitivity_window_s in sensitivity_alignment_windows:
        window_validation_rows = replay_validation_rows(
            train_rows=train_rows,
            heldout_rows=heldout_rows,
            base_parameters=base_parameters,
            candidate_parameters=candidate_parameters,
            extra_models=extra_validation_models,
            replay_dt_s=replay_dt_s,
            alignment_window_s=sensitivity_window_s,
            workers=workers,
        )
        sensitivity_validation_rows.extend(
            tag_alignment_sensitivity_rows(
                window_validation_rows,
                primary_alignment_window_s=alignment_window_s,
                sensitivity_alignment_window_s=sensitivity_window_s,
            )
        )
        window_stage_replay_rows = stage_replay_summary_rows(
            train_rows=train_rows,
            heldout_rows=heldout_rows,
            base_parameters=base_parameters,
            candidate_parameters=candidate_parameters,
            extra_models=extra_validation_models,
            replay_dt_s=replay_dt_s,
            alignment_window_s=sensitivity_window_s,
            workers=workers,
        )
        sensitivity_stage_replay_rows.extend(
            tag_alignment_sensitivity_rows(
                window_stage_replay_rows,
                primary_alignment_window_s=alignment_window_s,
                sensitivity_alignment_window_s=sensitivity_window_s,
            )
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
    write_csv(
        output_dir / "metrics" / "neutral_aero_residual_replay_sensitivity.csv",
        sensitivity_validation_rows,
        REPLAY_SENSITIVITY_FIELDS,
    )
    write_csv(
        output_dir / "metrics" / "neutral_aero_residual_stage_replay_sensitivity.csv",
        sensitivity_stage_replay_rows,
        STAGE_REPLAY_SENSITIVITY_FIELDS,
    )
    write_csv(output_dir / "metrics" / "neutral_aero_residual_lateral_ablation.csv", lateral_ablation_rows, LATERAL_ABLATION_FIELDS)
    write_csv(
        output_dir / "metrics" / "neutral_aero_residual_joint_sweep_candidates.csv",
        joint_sweep_candidate_rows,
        JOINT_SWEEP_CANDIDATE_FIELDS,
    )
    write_csv(
        output_dir / "metrics" / "neutral_aero_residual_joint_sweep_pareto.csv",
        joint_sweep_pareto_rows,
        JOINT_SWEEP_CANDIDATE_FIELDS,
    )
    write_csv(
        output_dir / "metrics" / "neutral_aero_residual_joint_sweep_selected.csv",
        joint_sweep_selected_rows,
        JOINT_SWEEP_SELECTED_FIELDS,
    )
    write_csv(
        output_dir / "metrics" / "neutral_aero_residual_joint_pareto_audit_candidates.csv",
        joint_pareto_audit_candidate_rows,
        JOINT_PARETO_AUDIT_FIELDS,
    )
    write_csv(
        output_dir / "metrics" / "neutral_aero_residual_joint_pareto_audit_selected.csv",
        joint_pareto_audit_selected_rows,
        JOINT_PARETO_AUDIT_FIELDS,
    )
    if str(joint_pareto_config["profile"]) == "heavy":
        write_csv(
            output_dir / "metrics" / "neutral_aero_residual_joint_pareto_heavy_candidates.csv",
            joint_pareto_audit_candidate_rows,
            JOINT_PARETO_AUDIT_FIELDS,
        )
        write_csv(
            output_dir / "metrics" / "neutral_aero_residual_joint_pareto_heavy_selected.csv",
            joint_pareto_audit_selected_rows,
            JOINT_PARETO_AUDIT_FIELDS,
        )
        write_csv(
            output_dir / "metrics" / "neutral_aero_residual_joint_pareto_heavy_stage_replay.csv",
            joint_pareto_heavy_stage_replay_rows,
            JOINT_PARETO_HEAVY_STAGE_REPLAY_FIELDS,
        )
    write_csv(
        output_dir / "metrics" / "neutral_aero_residual_lateral_launch_correlation.csv",
        lateral_launch_correlation_rows,
        LATERAL_LAUNCH_CORRELATION_FIELDS,
    )
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
        sensitivity_alignment_windows_s=sensitivity_alignment_windows,
        joint_pareto_audit=joint_pareto_audit,
        joint_pareto_audit_alignment_window_s=joint_pareto_audit_alignment_window_s,
        joint_pareto_config=joint_pareto_config,
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
        lateral_ablation_rows=lateral_ablation_rows,
        lateral_launch_correlation_rows=lateral_launch_correlation_rows,
        sensitivity_validation_rows=sensitivity_validation_rows,
        sensitivity_stage_replay_rows=sensitivity_stage_replay_rows,
        joint_sweep_candidate_rows=joint_sweep_candidate_rows,
        joint_sweep_pareto_rows=joint_sweep_pareto_rows,
        joint_sweep_selected_rows=joint_sweep_selected_rows,
        joint_pareto_audit_candidate_rows=joint_pareto_audit_candidate_rows,
        joint_pareto_audit_selected_rows=joint_pareto_audit_selected_rows,
        joint_pareto_heavy_stage_replay_rows=joint_pareto_heavy_stage_replay_rows,
    )
    write_report(
        output_dir,
        run_label=run_label,
        session_root=session_root,
        heldout_count=len(heldout_indices),
        heldout_seed=heldout_seed,
        alignment_window_s=alignment_window_s,
        sensitivity_alignment_windows_s=sensitivity_alignment_windows,
        joint_pareto_audit=joint_pareto_audit,
        joint_pareto_audit_alignment_window_s=joint_pareto_audit_alignment_window_s,
        joint_pareto_config=joint_pareto_config,
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
        lateral_ablation_rows=lateral_ablation_rows,
        lateral_launch_correlation_rows=lateral_launch_correlation_rows,
        joint_sweep_candidate_rows=joint_sweep_candidate_rows,
        joint_sweep_pareto_rows=joint_sweep_pareto_rows,
        joint_sweep_selected_rows=joint_sweep_selected_rows,
        joint_pareto_audit_candidate_rows=joint_pareto_audit_candidate_rows,
        joint_pareto_audit_selected_rows=joint_pareto_audit_selected_rows,
        joint_pareto_heavy_stage_replay_rows=joint_pareto_heavy_stage_replay_rows,
        regime_summary=regime_summary,
        stage_fit_summary=stage_fit_summary,
        validation_rows=validation_rows,
        stage_replay_rows=stage_replay_rows,
        sensitivity_validation_rows=sensitivity_validation_rows,
        sensitivity_stage_replay_rows=sensitivity_stage_replay_rows,
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


def resolved_heldout_count(*, filtered_valid_count: int, heldout_count: int, heldout_fraction: float) -> int:
    requested_heldout_count = int(heldout_count)
    if requested_heldout_count <= 0:
        requested_heldout_count = int(round(float(heldout_fraction) * int(filtered_valid_count)))
    return int(np.clip(requested_heldout_count, 1, max(1, int(filtered_valid_count) - 1)))


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
                "phi0_deg": math.degrees(float(launch_state[3])) if launch_state.size >= 6 and math.isfinite(float(launch_state[3])) else float("nan"),
                "psi0_deg": math.degrees(float(launch_state[5])) if launch_state.size >= 6 and math.isfinite(float(launch_state[5])) else float("nan"),
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
    confidence_weights = (
        lateral_excitation_confidence_weights(valid) if bool(use_confidence_weights) else np.ones(len(valid), dtype=float)
    )
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


def fit_samples_from_residual_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples = []
    for row in rows:
        if row.get("residual_status") != "ok":
            continue
        cm = finite_value(row.get("cm_residual"))
        if not math.isfinite(cm):
            continue
        beta_deg = finite_value(row.get("beta_deg"))
        beta_rad = math.radians(beta_deg) if math.isfinite(beta_deg) else float("nan")
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
                "beta_deg": beta_deg,
                "beta_rad": beta_rad,
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
    return samples


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
    samples = fit_samples_from_residual_rows(rows)
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


def lateral_excitation_confidence_weights(samples: list[dict[str, Any]]) -> np.ndarray:
    weights = []
    for sample in samples:
        beta = abs(finite_value(sample.get("beta_rad")))
        p_hat = abs(finite_value(sample.get("p_hat")))
        r_hat = abs(finite_value(sample.get("r_hat")))
        launch_score = finite_value(sample.get("launch_lateral_score"))
        excitation_terms = [
            min(beta / math.radians(8.0), 2.0) if math.isfinite(beta) else 0.0,
            min(p_hat / 0.12, 2.0) if math.isfinite(p_hat) else 0.0,
            min(r_hat / 0.12, 2.0) if math.isfinite(r_hat) else 0.0,
        ]
        excitation = max(excitation_terms)
        excitation_weight = 0.65 + 0.35 * math.exp(-((excitation - 1.0) / 0.9) ** 2)
        contamination = launch_score if math.isfinite(launch_score) else 0.0
        contamination_weight = math.exp(-0.65 * max(contamination - 0.45, 0.0) ** 2)
        weights.append(float(np.clip(excitation_weight * contamination_weight, 0.35, 1.15)))
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
        stages.append(("stage6_post_blend_transition_cm", "transition_pitch_moment", "transition"))
        stages.append(("stage7_post_blend_post_stall_cm", "post_stall_pitch_moment", "post_stall_cm"))
        if bool(fit_post_stall_damping):
            stages.append(("stage8_post_blend_post_stall_cmq", "post_stall_pitch_damping", "post_stall_cmq"))
        if bool(fit_post_stall_lift_drag):
            stages.append(("stage9_post_blend_post_stall_lift_drag", "post_stall_lift_drag", "post_stall_lift_drag"))
    elif bool(fit_post_stall_lift_drag):
        stages.append(("stage5_post_stall_lift_drag", "post_stall_lift_drag", "post_stall_lift_drag"))

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
                proposed_parameters=proposed,
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
    tolerances = {
        "dx_mae_m": LONGITUDINAL_STAGE_DX_TOL_M,
        "altitude_loss_mae_m": LONGITUDINAL_STAGE_ALTITUDE_LOSS_TOL_M,
        "sink_mae_m_s": LONGITUDINAL_STAGE_SINK_TOL_M_S,
        "final_theta_mae_deg": LONGITUDINAL_STAGE_PITCH_TOL_DEG,
    }
    objective_before = finite_value(before.get("objective"))
    objective_after = finite_value(after.get("objective"))
    if not (math.isfinite(objective_before) and math.isfinite(objective_after)):
        return False, "rejected_by_nonfinite_heldout_objective"

    exact_passed = []
    tolerance_passed = []
    for key, tolerance in tolerances.items():
        before_value = finite_value(before.get(key))
        after_value = finite_value(after.get(key))
        finite_pair = math.isfinite(before_value) and math.isfinite(after_value)
        exact_passed.append(finite_pair and after_value <= before_value + 1.0e-9)
        tolerance_passed.append(finite_pair and after_value <= before_value + float(tolerance))
    if all(exact_passed):
        return True, "heldout_longitudinal_improved_or_preserved"
    if objective_after < objective_before - 1.0e-9 and all(tolerance_passed):
        return True, "heldout_objective_improved_with_practical_metric_tolerance"
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
    proposed_parameters: dict[str, float] | None = None,
) -> dict[str, Any]:
    fractions = regime_sample_fractions(residuals)
    proposed = proposed_parameters if proposed_parameters is not None else after_parameters
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
        "proposed_parameter_updates_json": json.dumps(parameter_updates(before_parameters, proposed), sort_keys=True),
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


def compact_joint_sweep_refinement(
    *,
    base_parameters: dict[str, float],
    fit_result: dict[str, Any],
    train_residuals: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    heldout_rows: list[dict[str, Any]],
    replay_dt_s: float,
    alignment_window_s: float,
    ridge_lambda: float,
    workers: int,
) -> tuple[
    dict[str, float],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[tuple[str, dict[str, float]]],
]:
    samples = fit_samples_from_residual_rows(train_residuals)
    discovery = compact_joint_sweep_discovery(samples, fit_result, base_parameters, ridge_lambda=ridge_lambda)
    blocks = compact_joint_sweep_blocks(discovery)
    train_output: list[dict[str, Any]] = []
    candidate_counter = itertools.count()
    beam = [
        {
            "candidate_id": "joint_0000_active_baseline",
            "parameters": dict(base_parameters),
            "updates": {},
            "sweep_stage": "active_baseline",
            "summary": None,
        }
    ]
    longitudinal_reference_beam: list[dict[str, Any]] = []

    for block in blocks:
        expanded = compact_joint_sweep_expand_block(
            beam,
            block,
            base_parameters=base_parameters,
            candidate_counter=candidate_counter,
        )
        evaluated = compact_joint_sweep_evaluate_states(
            expanded,
            rows=train_rows,
            split="train",
            replay_dt_s=replay_dt_s,
            alignment_window_s=alignment_window_s,
            workers=workers,
        )
        train_output.extend(compact_joint_sweep_candidate_rows(evaluated, base_parameters, selection_class=""))
        evaluated = sorted(evaluated, key=lambda state: finite_value(state.get("score")))
        beam = evaluated[:COMPACT_JOINT_SWEEP_BEAM_WIDTH]
        if str(block["stage"]) == "post_lift_drag_cleanup":
            longitudinal_reference_beam = list(beam)

    if not longitudinal_reference_beam:
        longitudinal_reference_beam = [
            state for state in beam if joint_sweep_lateral_parameter_count(state.get("updates", {})) == 0
        ][: COMPACT_JOINT_SWEEP_BEAM_WIDTH]
    final_states = compact_joint_sweep_unique_states(
        [beam[0], *beam, *longitudinal_reference_beam[:10]],
        base_parameters=base_parameters,
        limit=COMPACT_JOINT_SWEEP_HELDOUT_EVAL_LIMIT,
    )
    heldout_evaluated = compact_joint_sweep_evaluate_states(
        final_states,
        rows=heldout_rows,
        split="heldout",
        replay_dt_s=replay_dt_s,
        alignment_window_s=alignment_window_s,
        workers=workers,
    )
    heldout_rows_out = compact_joint_sweep_candidate_rows(heldout_evaluated, base_parameters, selection_class="")
    candidate_rows = train_output + heldout_rows_out
    pareto_rows = compact_joint_sweep_pareto_rows(heldout_rows_out)
    selected_rows, selected_models = compact_joint_sweep_selected_rows(heldout_evaluated, base_parameters)
    selected_by_class = {str(row["selection_class"]): row for row in selected_rows}
    primary_id = (
        str(selected_by_class.get("strict_best", {}).get("candidate_id", ""))
        or str(selected_by_class.get("balanced_best", {}).get("candidate_id", ""))
        or str(selected_by_class.get("diagnostic_best", {}).get("candidate_id", ""))
    )
    selected_lookup = {state["candidate_id"]: state for state in heldout_evaluated}
    primary_parameters = dict(selected_lookup.get(primary_id, heldout_evaluated[0])["parameters"])
    extra_models = [
        (f"joint_sweep_{model_class}", params)
        for model_class, params in selected_models
        if selected_by_class.get(model_class, {}).get("candidate_id", "") != primary_id
    ]
    fit_result["compact_joint_sweep_discovery"] = discovery
    return primary_parameters, candidate_rows, pareto_rows, selected_rows, extra_models


def compact_joint_sweep_discovery(
    samples: list[dict[str, Any]],
    fit_result: dict[str, Any],
    base_parameters: dict[str, float],
    *,
    ridge_lambda: float,
) -> dict[str, Any]:
    coeffs = dict(fit_result.get("coefficients", zero_coefficients()))
    discovery_base = dict(base_parameters)
    for key in ("post_stall_residual_blend_start_alpha_deg", "post_stall_residual_blend_full_alpha_deg"):
        discovery_base[key] = float(coeffs.get(key, discovery_base.get(key, 0.0)))

    def lateral_fit(term: str, regime_family: str) -> dict[str, Any]:
        spec = next(spec for spec in LATERAL_ABLATION_TERMS if spec["term"] == term)
        return fit_single_lateral_ablation(
            samples,
            spec,
            regime_family=regime_family,
            base_coeffs=discovery_base,
            ridge_lambda=ridge_lambda,
        )

    lateral_fits = {
        "attached_CY_beta": lateral_fit("CY_beta", "attached"),
        "transition_CY_r": lateral_fit("CY_r", "transition"),
        "transition_Cn_p": lateral_fit("Cn_p", "transition"),
        "post_stall_Cn_p": lateral_fit("Cn_p", "post_stall"),
    }
    return {
        "longitudinal": {
            "attached_cm": float(coeffs.get("attached_cm_bias_coeff", 0.0)),
            "transition_cm": float(coeffs.get("transition_cm_bias_coeff", 0.0)),
            "post_stall_cm": float(coeffs.get("post_stall_pitch_moment_coeff", 0.0)),
            "post_stall_cmq": float(coeffs.get("post_stall_pitch_damping_coeff", 0.0)),
            "post_stall_cl": float(coeffs.get("post_stall_lift_residual_coeff", 0.0)),
            "post_stall_cd": float(coeffs.get("post_stall_drag_residual_coeff", 0.0)),
            "blend_start": float(coeffs.get("post_stall_residual_blend_start_alpha_deg", STALL_ALPHA_DEG)),
            "blend_full": float(coeffs.get("post_stall_residual_blend_full_alpha_deg", POST_STALL_ALPHA_DEG)),
        },
        "lateral": lateral_fits,
        "weighting_policy": {
            "longitudinal": "dynamic_pressure * throw_balance * lateral_contamination_confidence",
            "lateral": "dynamic_pressure * throw_balance * excitation_aware_lateral_confidence",
        },
    }


def compact_joint_sweep_blocks(discovery: dict[str, Any]) -> list[dict[str, Any]]:
    longitudinal = discovery["longitudinal"]
    lateral = discovery["lateral"]

    def scalar_block(stage: str, parameter: str, coefficient: float) -> dict[str, Any]:
        return {
            "stage": stage,
            "variants": [
                {"label": f"{stage}_{multiplier:g}", "updates": {parameter: multiplier * float(coefficient)}}
                for multiplier in COMPACT_JOINT_SWEEP_MULTIPLIERS
            ],
        }

    post_cnp_key = lateral_surface_parameter_name(
        "post_stall_yaw_moment",
        "p_hat",
        SURFACE_RBF_ALPHA_CENTERS_DEG[0],
    )
    blocks = [
        scalar_block("attached_cm", "attached_pitch_moment_bias_coeff", longitudinal["attached_cm"]),
        scalar_block("transition_cm", "transition_pitch_moment_bias_coeff", longitudinal["transition_cm"]),
        {
            "stage": "transition_blend",
            "variants": [
                {
                    "label": f"blend_{start:g}_{full:g}",
                    "updates": {
                        "post_stall_residual_blend_start_alpha_deg": start,
                        "post_stall_residual_blend_full_alpha_deg": full,
                    },
                }
                for start, full in COMPACT_JOINT_SWEEP_BLEND_GRID_DEG
            ],
        },
        scalar_block("post_stall_cm", "post_stall_pitch_moment_coeff", longitudinal["post_stall_cm"]),
        scalar_block("post_stall_cmq", "post_stall_pitch_damping_coeff", longitudinal["post_stall_cmq"]),
        {
            "stage": "post_lift_drag_cleanup",
            "variants": [
                {
                    "label": f"post_lift_drag_{multiplier:g}",
                    "updates": {
                        "post_stall_lift_residual_coeff": multiplier * float(longitudinal["post_stall_cl"]),
                        "post_stall_drag_residual_coeff": multiplier * float(longitudinal["post_stall_cd"]),
                    },
                }
                for multiplier in COMPACT_JOINT_SWEEP_MULTIPLIERS
            ],
        },
        scalar_block("attached_CY_beta", "side_force_beta_coeff", lateral["attached_CY_beta"]["coefficient"]),
        scalar_block("transition_CY_r", "transition_side_force_r_hat_coeff", lateral["transition_CY_r"]["coefficient"]),
        scalar_block("transition_Cn_p", "transition_yaw_moment_p_hat_coeff", lateral["transition_Cn_p"]["coefficient"]),
        scalar_block("post_stall_Cn_p", post_cnp_key, lateral["post_stall_Cn_p"]["coefficient"]),
    ]
    return blocks


def compact_joint_sweep_expand_block(
    beam: list[dict[str, Any]],
    block: dict[str, Any],
    *,
    base_parameters: dict[str, float],
    candidate_counter: Any,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for state in beam:
        for variant in block["variants"]:
            updates = dict(state.get("updates", {}))
            updates.update({key: float(value) for key, value in variant["updates"].items()})
            params = compact_joint_sweep_parameters(base_parameters, updates)
            signature = compact_joint_sweep_signature(params, base_parameters)
            if signature in seen:
                continue
            seen.add(signature)
            out.append(
                {
                    "candidate_id": f"joint_{next(candidate_counter) + 1:04d}_{variant['label']}",
                    "parameters": params,
                    "updates": updates,
                    "sweep_stage": str(block["stage"]),
                    "summary": None,
                }
            )
    return out


def compact_joint_sweep_parameters(base_parameters: dict[str, float], updates: dict[str, float]) -> dict[str, float]:
    params = dict(base_parameters)
    for key, value in updates.items():
        if key == "post_stall_residual_blend_start_alpha_deg":
            params[key] = replay_fit.bounded_parameter_value(key, float(value))
            continue
        if key == "post_stall_residual_blend_full_alpha_deg":
            start = float(params.get("post_stall_residual_blend_start_alpha_deg", STALL_ALPHA_DEG))
            value = max(start + 1.0, float(value))
            params[key] = replay_fit.bounded_parameter_value(key, float(value))
            continue
        params[key] = replay_fit.bounded_parameter_value(key, float(base_parameters.get(key, 0.0)) + float(value))
    return params


def compact_joint_sweep_signature(params: dict[str, float], base_parameters: dict[str, float]) -> str:
    updates = parameter_updates(base_parameters, params)
    return json.dumps(updates, sort_keys=True)


def compact_joint_sweep_unique_states(
    states: list[dict[str, Any]],
    *,
    base_parameters: dict[str, float],
    limit: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for state in states:
        signature = compact_joint_sweep_signature(state["parameters"], base_parameters)
        if signature in seen:
            continue
        seen.add(signature)
        out.append(state)
        if len(out) >= int(limit):
            break
    return out


def compact_joint_sweep_evaluate_states(
    states: list[dict[str, Any]],
    *,
    rows: list[dict[str, Any]],
    split: str,
    replay_dt_s: float,
    alignment_window_s: float,
    workers: int,
) -> list[dict[str, Any]]:
    payloads = [
        (str(state["candidate_id"]), row, state["parameters"], replay_dt_s, alignment_window_s)
        for state in states
        for row in rows
    ]
    grouped: dict[str, list[dict[str, Any]]] = {str(state["candidate_id"]): [] for state in states}
    if int(workers) > 1 and len(payloads) > 1:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            replay_items = list(executor.map(compact_joint_sweep_replay_payload, payloads))
    else:
        replay_items = [compact_joint_sweep_replay_payload(payload) for payload in payloads]
    for candidate_id, replay_row in replay_items:
        grouped[str(candidate_id)].append(replay_row)

    evaluated = []
    for state in states:
        replay_rows = grouped[str(state["candidate_id"])]
        summary = replay_fit.objective_summary(replay_rows, objective_mode="combined")
        state = dict(state)
        state["summary"] = summary
        state["split"] = split
        state["score"] = compact_joint_sweep_score(summary)
        state["longitudinal_score"] = compact_joint_sweep_longitudinal_score(summary)
        state["lateral_score"] = compact_joint_sweep_lateral_score(summary)
        evaluated.append(state)
    return evaluated


def compact_joint_sweep_replay_payload(
    payload: tuple[str, dict[str, Any], dict[str, float], float, float]
) -> tuple[str, dict[str, Any]]:
    candidate_id, row, parameters, replay_dt_s, alignment_window_s = payload
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        replay_row = replay_fit.simulate_row_payload((row, parameters, replay_dt_s, alignment_window_s))
    return str(candidate_id), replay_row


def compact_joint_sweep_score(summary: dict[str, float]) -> float:
    return compact_joint_sweep_longitudinal_score(summary) + compact_joint_sweep_lateral_score(summary)


def compact_joint_sweep_longitudinal_score(summary: dict[str, float]) -> float:
    return float(
        finite_value(summary.get("dx_mae_m")) / 0.30
        + finite_value(summary.get("altitude_loss_mae_m")) / 0.12
        + finite_value(summary.get("sink_mae_m_s")) / 0.10
        + finite_value(summary.get("final_theta_mae_deg")) / 10.0
    )


def compact_joint_sweep_lateral_score(summary: dict[str, float]) -> float:
    return float(
        finite_value(summary.get("dy_mae_m")) / 0.45
        + finite_value(summary.get("final_phi_mae_deg")) / 12.0
        + finite_value(summary.get("final_psi_mae_deg")) / 18.0
    )


def compact_joint_sweep_candidate_rows(
    states: list[dict[str, Any]],
    base_parameters: dict[str, float],
    *,
    selection_class: str,
) -> list[dict[str, Any]]:
    return [
        compact_joint_sweep_candidate_row(state, base_parameters, selection_class=selection_class)
        for state in states
    ]


def compact_joint_sweep_candidate_row(
    state: dict[str, Any],
    base_parameters: dict[str, float],
    *,
    selection_class: str,
) -> dict[str, Any]:
    summary = state.get("summary", {}) or {}
    updates = parameter_updates(base_parameters, state["parameters"])
    return {
        "candidate_id": state.get("candidate_id", ""),
        "selection_class": selection_class,
        "sweep_stage": state.get("sweep_stage", ""),
        "split": state.get("split", ""),
        "score": finite_value(state.get("score")),
        "longitudinal_score": finite_value(state.get("longitudinal_score")),
        "lateral_score": finite_value(state.get("lateral_score")),
        "parameter_count": len(updates),
        "lateral_parameter_count": joint_sweep_lateral_parameter_count(updates),
        "dx_mae_m": finite_value(summary.get("dx_mae_m")),
        "dy_mae_m": finite_value(summary.get("dy_mae_m")),
        "altitude_loss_mae_m": finite_value(summary.get("altitude_loss_mae_m")),
        "sink_mae_m_s": finite_value(summary.get("sink_mae_m_s")),
        "roll_mae_deg": finite_value(summary.get("final_phi_mae_deg")),
        "pitch_mae_deg": finite_value(summary.get("final_theta_mae_deg")),
        "yaw_mae_deg": finite_value(summary.get("final_psi_mae_deg")),
        "parameter_updates_json": json.dumps(updates, sort_keys=True),
    }


def joint_sweep_lateral_parameter_count(updates: dict[str, float]) -> int:
    return sum(1 for key in updates if joint_sweep_is_lateral_parameter(key))


def joint_sweep_is_lateral_parameter(key: str) -> bool:
    return key in set(ATTACHED_LATERAL_PARAMETER_KEYS + TRANSITION_LATERAL_PARAMETER_KEYS) or key.startswith(
        ("post_stall_side_force_", "post_stall_roll_moment_", "post_stall_yaw_moment_")
    )


def compact_joint_sweep_pareto_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    heldout = [row for row in rows if row.get("split") == "heldout"]
    pareto = []
    for row in heldout:
        dominated = False
        metrics = (
            "dx_mae_m",
            "dy_mae_m",
            "altitude_loss_mae_m",
            "sink_mae_m_s",
            "roll_mae_deg",
            "pitch_mae_deg",
            "yaw_mae_deg",
        )
        for other in heldout:
            if other is row:
                continue
            no_worse = all(finite_value(other.get(metric)) <= finite_value(row.get(metric)) + 1.0e-12 for metric in metrics)
            strictly_better = any(finite_value(other.get(metric)) < finite_value(row.get(metric)) - 1.0e-12 for metric in metrics)
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            pareto.append(dict(row))
    return sorted(pareto, key=lambda row: finite_value(row.get("score")))[:25]


def compact_joint_sweep_selected_rows(
    states: list[dict[str, Any]],
    base_parameters: dict[str, float],
) -> tuple[list[dict[str, Any]], list[tuple[str, dict[str, float]]]]:
    rows = compact_joint_sweep_candidate_rows(states, base_parameters, selection_class="")
    longitudinal_rows = [row for row in rows if int(finite_value(row.get("lateral_parameter_count"))) == 0]
    reference = min(longitudinal_rows or rows, key=lambda row: finite_value(row.get("longitudinal_score")))

    def within_longitudinal_tolerance(row: dict[str, Any], tolerances: dict[str, float]) -> bool:
        mapping = {
            "dx_mae_m": "dx_mae_m",
            "altitude_loss_mae_m": "altitude_loss_mae_m",
            "sink_mae_m_s": "sink_mae_m_s",
            "final_theta_mae_deg": "pitch_mae_deg",
        }
        for reference_key, row_key in mapping.items():
            if finite_value(row.get(row_key)) > finite_value(reference.get(row_key)) + float(tolerances[reference_key]):
                return False
        return True

    strict_candidates = [
        row
        for row in rows
        if within_longitudinal_tolerance(row, JOINT_SWEEP_STRICT_LONGITUDINAL_TOLERANCES)
        and finite_value(row.get("lateral_score")) <= finite_value(reference.get("lateral_score")) + 1.0e-9
    ]
    balanced_candidates = [
        row
        for row in rows
        if within_longitudinal_tolerance(row, JOINT_SWEEP_BALANCED_LONGITUDINAL_TOLERANCES)
    ]
    diagnostic_candidates = rows
    selected_specs = [
        ("strict_best", strict_candidates, "best score with strict longitudinal tolerance and non-worse lateral score"),
        ("balanced_best", balanced_candidates, "best score with balanced longitudinal tolerance"),
        ("diagnostic_best", diagnostic_candidates, "lowest lateral score; not claim-bearing if longitudinal replay degrades"),
    ]
    state_by_id = {str(state["candidate_id"]): state for state in states}
    selected_rows = []
    selected_models: list[tuple[str, dict[str, float]]] = []
    for selection_class, candidates, reason in selected_specs:
        if not candidates:
            continue
        key = (lambda row: finite_value(row.get("lateral_score"))) if selection_class == "diagnostic_best" else (
            lambda row: finite_value(row.get("score"))
        )
        row = dict(min(candidates, key=key))
        row["selection_class"] = selection_class
        selected_rows.append(
            {
                "selection_class": selection_class,
                "candidate_id": row["candidate_id"],
                "selection_reason": reason,
                "score": row["score"],
                "longitudinal_score": row["longitudinal_score"],
                "lateral_score": row["lateral_score"],
                "parameter_count": row["parameter_count"],
                "lateral_parameter_count": row["lateral_parameter_count"],
                "dx_mae_m": row["dx_mae_m"],
                "dy_mae_m": row["dy_mae_m"],
                "altitude_loss_mae_m": row["altitude_loss_mae_m"],
                "sink_mae_m_s": row["sink_mae_m_s"],
                "roll_mae_deg": row["roll_mae_deg"],
                "pitch_mae_deg": row["pitch_mae_deg"],
                "yaw_mae_deg": row["yaw_mae_deg"],
                "parameter_updates_json": row["parameter_updates_json"],
            }
        )
        selected_models.append((selection_class, dict(state_by_id[row["candidate_id"]]["parameters"])))
    return selected_rows, selected_models


def small_joint_pareto_audit(
    *,
    base_parameters: dict[str, float],
    candidate_parameters: dict[str, float],
    fit_result: dict[str, Any],
    lateral_diagnostic_parameters: dict[str, float] | None,
    cm_stage_history_rows: list[dict[str, Any]],
    joint_sweep_extra_models: list[tuple[str, dict[str, float]]],
    lateral_ablation_rows: list[dict[str, Any]],
    heldout_rows: list[dict[str, Any]],
    replay_dt_s: float,
    alignment_window_s: float,
    workers: int,
    enabled: bool,
    config: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not bool(enabled):
        return [], []
    profile_config = dict(config or JOINT_PARETO_PROFILE_DEFAULTS[DEFAULT_JOINT_PARETO_PROFILE])
    profile = str(profile_config.get("profile", DEFAULT_JOINT_PARETO_PROFILE))
    top_longitudinal = max(1, int(profile_config.get("top_longitudinal", JOINT_PARETO_AUDIT_TOP_LONGITUDINAL)))
    top_lateral = max(0, int(profile_config.get("top_lateral", JOINT_PARETO_AUDIT_TOP_LATERAL)))
    selected_limit = max(1, int(profile_config.get("selected_limit", JOINT_PARETO_AUDIT_SELECTED_LIMIT)))
    audit_window_s = float(alignment_window_s)
    if not math.isfinite(audit_window_s) or audit_window_s <= 0.0:
        raise ValueError(f"Joint Pareto audit alignment window must be positive and finite, got {alignment_window_s!r}.")
    if not heldout_rows:
        return [], []

    longitudinal_sources = joint_pareto_longitudinal_source_states(
        base_parameters=base_parameters,
        candidate_parameters=candidate_parameters,
        cm_stage_history_rows=cm_stage_history_rows,
        joint_sweep_extra_models=joint_sweep_extra_models,
        include_rejected_stage_candidates=bool(
            profile_config.get(
                "include_rejected_stage_candidates",
                profile_config.get("scaled_single_term_bundles", False),
            )
        ),
    )
    if not longitudinal_sources:
        return [], []
    evaluated_longitudinal = compact_joint_sweep_evaluate_states(
        longitudinal_sources,
        rows=heldout_rows,
        split="heldout",
        replay_dt_s=replay_dt_s,
        alignment_window_s=audit_window_s,
        workers=workers,
    )
    top_longitudinal_sources = sorted(
        evaluated_longitudinal,
        key=lambda state: finite_value(state.get("longitudinal_score")),
    )[: max(1, int(top_longitudinal))]
    best_longitudinal = top_longitudinal_sources[0]

    lateral_sources = joint_pareto_lateral_source_states(
        base_parameters=base_parameters,
        candidate_parameters=candidate_parameters,
        fit_result=fit_result,
        lateral_diagnostic_parameters=lateral_diagnostic_parameters,
        lateral_ablation_rows=lateral_ablation_rows,
        joint_sweep_extra_models=joint_sweep_extra_models,
    )
    if bool(profile_config.get("scaled_single_term_bundles", False)):
        combination_states = joint_pareto_heavy_combination_states(
            top_longitudinal_sources=top_longitudinal_sources,
            lateral_sources=lateral_sources,
            base_parameters=base_parameters,
            best_longitudinal=best_longitudinal,
            heldout_rows=heldout_rows,
            replay_dt_s=replay_dt_s,
            alignment_window_s=audit_window_s,
            workers=workers,
            config=profile_config,
        )
    else:
        probe_states = []
        for source in lateral_sources:
            params = joint_pareto_combined_parameters(
                best_longitudinal["parameters"],
                base_parameters=base_parameters,
                lateral_parameters=source["parameters"],
            )
            probe_states.append(
                {
                    "candidate_id": f"probe_{source['candidate_id']}",
                    "parameters": params,
                    "updates": parameter_updates(base_parameters, params),
                    "sweep_stage": "joint_pareto_lateral_probe",
                    "summary": None,
                    "lateral_source_id": source["candidate_id"],
                }
            )
        evaluated_probes = compact_joint_sweep_evaluate_states(
            probe_states,
            rows=heldout_rows,
            split="heldout",
            replay_dt_s=replay_dt_s,
            alignment_window_s=audit_window_s,
            workers=workers,
        )
        probe_score_by_source = {
            str(state.get("lateral_source_id", "")): finite_value(state.get("lateral_score"))
            for state in evaluated_probes
        }
        no_lateral = [source for source in lateral_sources if source["candidate_id"] == "no_lateral_update"]
        lateral_candidates = [source for source in lateral_sources if source["candidate_id"] != "no_lateral_update"]
        top_lateral_sources = no_lateral + sorted(
            lateral_candidates,
            key=lambda source: probe_score_by_source.get(str(source["candidate_id"]), float("inf")),
        )[: max(0, int(top_lateral))]
        combination_states = joint_pareto_raw_combination_states(
            top_longitudinal_sources=top_longitudinal_sources,
            top_lateral_sources=top_lateral_sources,
            base_parameters=base_parameters,
            audit_window_s=audit_window_s,
        )

    if not combination_states:
        return [], []
    evaluated_combinations = compact_joint_sweep_evaluate_states(
        combination_states,
        rows=heldout_rows,
        split="heldout",
        replay_dt_s=replay_dt_s,
        alignment_window_s=audit_window_s,
        workers=workers,
    )
    reference_candidates = [
        state
        for state in evaluated_combinations
        if str(state.get("lateral_source_id", "")) == "no_lateral_update"
    ]
    global_reference_state = min(
        reference_candidates or evaluated_combinations,
        key=lambda state: finite_value(state.get("longitudinal_score")),
    )
    reference_by_longitudinal = {
        str(state.get("longitudinal_source_id", "")): state
        for state in reference_candidates
    }
    if str(profile_config.get("reference_policy", "global_best")) != "per_base":
        reference_by_longitudinal = {}
    rows = joint_pareto_audit_candidate_rows(
        evaluated_combinations,
        base_parameters=base_parameters,
        reference_state=global_reference_state,
        reference_by_longitudinal=reference_by_longitudinal,
        global_reference_state=global_reference_state,
        alignment_window_s=audit_window_s,
        profile=profile,
        longitudinal_tolerances=joint_pareto_profile_longitudinal_tolerances(profile_config),
    )
    joint_pareto_audit_mark_pareto(rows)
    selected_rows = joint_pareto_audit_selected_rows(rows, selected_limit=selected_limit)
    return rows, selected_rows


def joint_pareto_profile_longitudinal_tolerances(config: dict[str, Any]) -> dict[str, float]:
    raw = dict(JOINT_SWEEP_BALANCED_LONGITUDINAL_TOLERANCES)
    raw.update(dict(config.get("longitudinal_tolerances", {}) or {}))
    return {
        key: float(raw[key])
        for key in JOINT_SWEEP_BALANCED_LONGITUDINAL_TOLERANCES
        if key in raw
    }


def joint_pareto_raw_combination_states(
    *,
    top_longitudinal_sources: list[dict[str, Any]],
    top_lateral_sources: list[dict[str, Any]],
    base_parameters: dict[str, float],
    audit_window_s: float,
) -> list[dict[str, Any]]:
    combination_states = []
    seen: set[str] = set()
    audit_prefix = f"jp{int(round(audit_window_s * 1000.0)):03d}"
    for long_index, longitudinal in enumerate(top_longitudinal_sources):
        for lateral_index, lateral in enumerate(top_lateral_sources):
            params = joint_pareto_combined_parameters(
                longitudinal["parameters"],
                base_parameters=base_parameters,
                lateral_parameters=lateral["parameters"],
            )
            signature = compact_joint_sweep_signature(params, base_parameters)
            if signature in seen:
                continue
            seen.add(signature)
            long_id = short_source_id(str(longitudinal["candidate_id"]))
            lateral_id = short_source_id(str(lateral["candidate_id"]))
            combination_states.append(
                {
                    "candidate_id": f"{audit_prefix}_L{long_index:02d}_{long_id}_X{lateral_index:02d}_{lateral_id}",
                    "parameters": params,
                    "updates": parameter_updates(base_parameters, params),
                    "sweep_stage": "joint_pareto_audit",
                    "summary": None,
                    "longitudinal_source_id": str(longitudinal["candidate_id"]),
                    "lateral_source_id": str(lateral["candidate_id"]),
                    "bundle_order": 0 if str(lateral["candidate_id"]) == "no_lateral_update" else 1,
                }
            )
    return combination_states


def joint_pareto_heavy_combination_states(
    *,
    top_longitudinal_sources: list[dict[str, Any]],
    lateral_sources: list[dict[str, Any]],
    base_parameters: dict[str, float],
    best_longitudinal: dict[str, Any],
    heldout_rows: list[dict[str, Any]],
    replay_dt_s: float,
    alignment_window_s: float,
    workers: int,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    top_lateral = max(0, int(config.get("top_lateral", JOINT_PARETO_AUDIT_TOP_LATERAL)))
    max_lateral_order = max(1, min(3, int(config.get("max_lateral_order", 1))))
    top_triples = max(0, int(config.get("top_triples", 0)))
    max_candidates = max(1, int(config.get("max_candidates", 64)))
    scale_grid = tuple(float(value) for value in config.get("scale_grid", JOINT_PARETO_HEAVY_SCALE_GRID))
    audit_prefix = f"jp{int(round(float(alignment_window_s) * 1000.0)):03d}h"

    single_sources = joint_pareto_scaled_single_lateral_sources(
        lateral_sources,
        base_parameters=base_parameters,
        scale_grid=scale_grid,
    )
    if top_lateral <= 0 or not single_sources:
        return joint_pareto_raw_combination_states(
            top_longitudinal_sources=top_longitudinal_sources,
            top_lateral_sources=[
                {
                    "candidate_id": "no_lateral_update",
                    "parameters": dict(base_parameters),
                    "updates": {},
                    "source_priority": 0.0,
                }
            ],
            base_parameters=base_parameters,
            audit_window_s=alignment_window_s,
        )[:max_candidates]

    probe_states: list[dict[str, Any]] = []
    for index, source in enumerate(single_sources):
        params = joint_pareto_combined_parameters(
            best_longitudinal["parameters"],
            base_parameters=base_parameters,
            lateral_parameters=source["parameters"],
        )
        probe_states.append(
            {
                "candidate_id": f"{audit_prefix}_probe_{index:03d}_{short_source_id(str(source['candidate_id']), limit=36)}",
                "parameters": params,
                "updates": parameter_updates(base_parameters, params),
                "sweep_stage": "joint_pareto_heavy_lateral_probe",
                "summary": None,
                "lateral_source_id": source["candidate_id"],
                "single_source": source,
            }
        )
    evaluated_probes = compact_joint_sweep_evaluate_states(
        probe_states,
        rows=heldout_rows,
        split="heldout",
        replay_dt_s=replay_dt_s,
        alignment_window_s=alignment_window_s,
        workers=workers,
    )
    probe_by_source = {
        str(state.get("lateral_source_id", "")): state
        for state in evaluated_probes
    }
    ranked_singles = sorted(
        single_sources,
        key=lambda source: (
            finite_value(probe_by_source.get(str(source["candidate_id"]), {}).get("lateral_score")),
            finite_value(probe_by_source.get(str(source["candidate_id"]), {}).get("longitudinal_score")),
            finite_value(source.get("source_priority")),
            str(source["candidate_id"]),
        ),
    )[:top_lateral]

    bundles = joint_pareto_heavy_lateral_bundles(
        ranked_singles,
        max_lateral_order=max_lateral_order,
        top_triples=top_triples,
        probe_by_source=probe_by_source,
    )
    combination_states: list[dict[str, Any]] = []
    seen: set[str] = set()
    for long_index, longitudinal in enumerate(top_longitudinal_sources):
        no_lateral_state = {
            "candidate_id": f"{audit_prefix}_L{long_index:02d}_{short_source_id(str(longitudinal['candidate_id']))}_X00_no_lateral",
            "parameters": dict(longitudinal["parameters"]),
            "updates": parameter_updates(base_parameters, longitudinal["parameters"]),
            "sweep_stage": "joint_pareto_heavy",
            "summary": None,
            "longitudinal_source_id": str(longitudinal["candidate_id"]),
            "lateral_source_id": "no_lateral_update",
            "bundle_order": 0,
        }
        signature = compact_joint_sweep_signature(no_lateral_state["parameters"], base_parameters)
        if signature not in seen:
            seen.add(signature)
            combination_states.append(no_lateral_state)
            if len(combination_states) >= max_candidates:
                return combination_states

        for bundle_index, bundle in enumerate(bundles, start=1):
            lateral_params = joint_pareto_lateral_bundle_parameters(
                base_parameters=base_parameters,
                bundle_sources=bundle,
            )
            params = joint_pareto_combined_parameters(
                longitudinal["parameters"],
                base_parameters=base_parameters,
                lateral_parameters=lateral_params,
            )
            signature = compact_joint_sweep_signature(params, base_parameters)
            if signature in seen:
                continue
            seen.add(signature)
            bundle_id = "_".join(short_source_id(str(source["candidate_id"]), limit=20) for source in bundle)
            combination_states.append(
                {
                    "candidate_id": (
                        f"{audit_prefix}_L{long_index:02d}_{short_source_id(str(longitudinal['candidate_id']), limit=24)}"
                        f"_X{bundle_index:03d}_{short_source_id(bundle_id, limit=48)}"
                    ),
                    "parameters": params,
                    "updates": parameter_updates(base_parameters, params),
                    "sweep_stage": "joint_pareto_heavy",
                    "summary": None,
                    "longitudinal_source_id": str(longitudinal["candidate_id"]),
                    "lateral_source_id": "+".join(str(source["candidate_id"]) for source in bundle),
                    "bundle_order": len(bundle),
                }
            )
            if len(combination_states) >= max_candidates:
                return combination_states
    return combination_states


def joint_pareto_scaled_single_lateral_sources(
    lateral_sources: list[dict[str, Any]],
    *,
    base_parameters: dict[str, float],
    scale_grid: tuple[float, ...],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in lateral_sources:
        if str(source.get("candidate_id", "")) == "no_lateral_update":
            continue
        updates = {
            key: float(value)
            for key, value in (source.get("updates", {}) or {}).items()
            if joint_pareto_heavy_allowed_lateral_key(str(key))
        }
        for key, value in sorted(updates.items()):
            base_value = float(base_parameters.get(key, 0.0))
            delta = float(value) - base_value
            if abs(delta) <= 1.0e-12:
                continue
            for scale in scale_grid:
                if abs(float(scale)) <= 1.0e-12:
                    continue
                params = dict(base_parameters)
                params[key] = replay_fit.bounded_parameter_value(key, base_value + float(scale) * delta)
                single_updates = parameter_updates(base_parameters, params)
                if not single_updates:
                    continue
                signature = json.dumps(single_updates, sort_keys=True)
                if signature in seen:
                    continue
                seen.add(signature)
                source_id = (
                    f"{short_source_id(str(source.get('candidate_id', 'source')), limit=28)}"
                    f"__{short_source_id(key, limit=30)}__s{str(float(scale)).replace('.', 'p').replace('-', 'm')}"
                )
                out.append(
                    {
                        "candidate_id": source_id,
                        "parameters": params,
                        "updates": single_updates,
                        "sweep_stage": "joint_pareto_heavy_single_lateral_source",
                        "summary": None,
                        "source_priority": finite_value(source.get("source_priority")),
                        "source_key": key,
                        "source_scale": float(scale),
                    }
                )
    return out


def joint_pareto_heavy_allowed_lateral_key(key: str) -> bool:
    if key in set(ATTACHED_LATERAL_PARAMETER_KEYS + TRANSITION_LATERAL_PARAMETER_KEYS):
        return True
    return key.startswith(("post_stall_side_force_", "post_stall_roll_moment_", "post_stall_yaw_moment_"))


def joint_pareto_heavy_lateral_bundles(
    ranked_singles: list[dict[str, Any]],
    *,
    max_lateral_order: int,
    top_triples: int,
    probe_by_source: dict[str, dict[str, Any]],
) -> list[tuple[dict[str, Any], ...]]:
    bundles: list[tuple[dict[str, Any], ...]] = []
    if max_lateral_order >= 1:
        bundles.extend((source,) for source in ranked_singles)
    if max_lateral_order >= 2:
        bundles.extend(
            bundle
            for bundle in itertools.combinations(ranked_singles, 2)
            if joint_pareto_lateral_bundle_has_unique_keys(bundle)
        )
    if max_lateral_order >= 3 and top_triples > 0:
        triples = [
            bundle
            for bundle in itertools.combinations(ranked_singles, 3)
            if joint_pareto_lateral_bundle_has_unique_keys(bundle)
        ]
        triples = sorted(
            triples,
            key=lambda bundle: sum(
                finite_value(probe_by_source.get(str(source["candidate_id"]), {}).get("lateral_score"))
                for source in bundle
            ),
        )[:top_triples]
        bundles.extend(triples)
    return bundles


def joint_pareto_lateral_bundle_has_unique_keys(bundle_sources: tuple[dict[str, Any], ...]) -> bool:
    keys = []
    for source in bundle_sources:
        source_keys = [key for key in (source.get("updates", {}) or {}) if joint_pareto_heavy_allowed_lateral_key(str(key))]
        if len(source_keys) != 1:
            return False
        keys.append(str(source_keys[0]))
    return len(keys) == len(set(keys))


def joint_pareto_lateral_bundle_parameters(
    *,
    base_parameters: dict[str, float],
    bundle_sources: tuple[dict[str, Any], ...],
) -> dict[str, float]:
    params = dict(base_parameters)
    for source in bundle_sources:
        for key, value in (source.get("updates", {}) or {}).items():
            if joint_pareto_heavy_allowed_lateral_key(str(key)):
                params[str(key)] = replay_fit.bounded_parameter_value(str(key), float(value))
    return params


def joint_pareto_longitudinal_source_states(
    *,
    base_parameters: dict[str, float],
    candidate_parameters: dict[str, float],
    cm_stage_history_rows: list[dict[str, Any]],
    joint_sweep_extra_models: list[tuple[str, dict[str, float]]],
    include_rejected_stage_candidates: bool = False,
) -> list[dict[str, Any]]:
    sources: list[tuple[str, dict[str, float]]] = []
    sources.append(("active_baseline", joint_pareto_longitudinal_only_parameters(base_parameters, base_parameters)))
    staged_parameters = dict(base_parameters)
    for row in cm_stage_history_rows:
        stage_id = short_source_id(f"stage_{row.get('stage_index', '')}_{row.get('stage_id', '')}")
        stage_index_value = finite_value(row.get("stage_index"))
        if bool(include_rejected_stage_candidates) and math.isfinite(stage_index_value) and int(stage_index_value) > 0:
            proposed_updates = parse_json_mapping(row.get("proposed_parameter_updates_json", "{}"))
            if proposed_updates:
                proposed_parameters = dict(staged_parameters)
                for key, value in proposed_updates.items():
                    proposed_parameters[str(key)] = replay_fit.bounded_parameter_value(str(key), float(value))
                sources.append(
                    (
                        f"proposal_{stage_id}",
                        joint_pareto_longitudinal_only_parameters(base_parameters, proposed_parameters),
                    )
                )
        if not bool(row.get("accepted", False)):
            continue
        updates = parse_json_mapping(row.get("parameter_updates_json", "{}"))
        for key, value in updates.items():
            staged_parameters[str(key)] = replay_fit.bounded_parameter_value(str(key), float(value))
        sources.append((stage_id, joint_pareto_longitudinal_only_parameters(base_parameters, staged_parameters)))
    sources.append(("primary_candidate", joint_pareto_longitudinal_only_parameters(base_parameters, candidate_parameters)))
    for name, params in joint_sweep_extra_models:
        updates = parameter_updates(base_parameters, params)
        if joint_sweep_lateral_parameter_count(updates) == 0:
            sources.append((f"extra_{short_source_id(name)}", joint_pareto_longitudinal_only_parameters(base_parameters, params)))

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate_id, params in sources:
        signature = compact_joint_sweep_signature(params, base_parameters)
        if signature in seen:
            continue
        seen.add(signature)
        out.append(
            {
                "candidate_id": candidate_id,
                "parameters": params,
                "updates": parameter_updates(base_parameters, params),
                "sweep_stage": "joint_pareto_longitudinal_source",
                "summary": None,
            }
        )
    return out


def joint_pareto_lateral_source_states(
    *,
    base_parameters: dict[str, float],
    candidate_parameters: dict[str, float],
    fit_result: dict[str, Any],
    lateral_diagnostic_parameters: dict[str, float] | None,
    lateral_ablation_rows: list[dict[str, Any]],
    joint_sweep_extra_models: list[tuple[str, dict[str, float]]],
) -> list[dict[str, Any]]:
    sources: list[tuple[str, dict[str, float], float]] = [
        ("no_lateral_update", dict(base_parameters), 0.0),
    ]
    if lateral_diagnostic_parameters is not None:
        sources.append(
            (
                "secondary_lateral_diagnostic",
                joint_pareto_lateral_only_parameters(base_parameters, lateral_diagnostic_parameters),
                -1.0,
            )
        )

    for source_id, params in compact_fit_lateral_source_parameters(base_parameters, fit_result):
        sources.append((source_id, params, 0.0))

    heldout_ablation_rows = [
        row
        for row in lateral_ablation_rows
        if str(row.get("split", "")) == "heldout"
        and str(row.get("baseline_model_id", "")) == "primary_longitudinal"
    ]
    for row in sorted(heldout_ablation_rows, key=lateral_ablation_source_priority)[: max(1, JOINT_PARETO_AUDIT_TOP_LATERAL * 3)]:
        params = joint_pareto_lateral_ablation_parameters(base_parameters, row)
        sources.append((f"ablation_{short_source_id(str(row.get('candidate_id', '')))}", params, lateral_ablation_source_priority(row)))

    for name, params in joint_sweep_extra_models:
        lateral_params = joint_pareto_lateral_only_parameters(base_parameters, params)
        if parameter_updates(base_parameters, lateral_params):
            sources.append((f"extra_{short_source_id(name)}_lateral", lateral_params, 0.0))

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate_id, params, priority in sources:
        lateral_updates = {
            key: value
            for key, value in parameter_updates(base_parameters, params).items()
            if joint_sweep_is_lateral_parameter(key)
        }
        signature = json.dumps(lateral_updates, sort_keys=True)
        if signature in seen:
            continue
        seen.add(signature)
        out.append(
            {
                "candidate_id": candidate_id,
                "parameters": params,
                "updates": lateral_updates,
                "sweep_stage": "joint_pareto_lateral_source",
                "summary": None,
                "source_priority": float(priority),
            }
        )
    return out


def compact_fit_lateral_source_parameters(
    base_parameters: dict[str, float],
    fit_result: dict[str, Any],
) -> list[tuple[str, dict[str, float]]]:
    specs = [
        ("fit_attached_lateral", True, False),
        ("fit_transition_lateral", False, True),
        ("fit_attached_transition_lateral", True, True),
    ]
    out: list[tuple[str, dict[str, float]]] = []
    for source_id, fit_attached, fit_transition in specs:
        candidate = candidate_from_fit(
            base_parameters,
            fit_result,
            apply_attached_cm_bias=False,
            fit_post_stall_longitudinal=False,
            fit_transition_blender=False,
            fit_post_stall_surfaces=False,
            fit_post_stall_damping=False,
            fit_attached_lateral_coupling=fit_attached,
            fit_transition_lateral_coupling=fit_transition,
            fit_lateral_surfaces=False,
        )
        lateral_only = joint_pareto_lateral_only_parameters(base_parameters, candidate)
        if parameter_updates(base_parameters, lateral_only):
            out.append((source_id, lateral_only))
    return out


def joint_pareto_longitudinal_only_parameters(
    base_parameters: dict[str, float],
    source_parameters: dict[str, float],
) -> dict[str, float]:
    params = dict(source_parameters)
    for key, value in base_parameters.items():
        if joint_sweep_is_lateral_parameter(key):
            params[key] = float(value)
    return params


def joint_pareto_lateral_only_parameters(
    base_parameters: dict[str, float],
    source_parameters: dict[str, float],
) -> dict[str, float]:
    params = dict(base_parameters)
    for key, value in source_parameters.items():
        if joint_sweep_is_lateral_parameter(str(key)):
            params[str(key)] = replay_fit.bounded_parameter_value(str(key), float(value))
    return params


def joint_pareto_lateral_ablation_parameters(
    base_parameters: dict[str, float],
    row: dict[str, Any],
) -> dict[str, float]:
    params = dict(base_parameters)
    coefficient = finite_value(row.get("fit_coefficient"))
    if not math.isfinite(coefficient):
        return params
    keys = parse_json_list(row.get("parameter_keys_json", "[]"))
    for key in keys:
        if joint_sweep_is_lateral_parameter(str(key)):
            params[str(key)] = replay_fit.bounded_parameter_value(str(key), coefficient)
    return params


def joint_pareto_combined_parameters(
    longitudinal_parameters: dict[str, float],
    *,
    base_parameters: dict[str, float],
    lateral_parameters: dict[str, float],
) -> dict[str, float]:
    params = dict(longitudinal_parameters)
    for key, value in parameter_updates(base_parameters, lateral_parameters).items():
        if joint_sweep_is_lateral_parameter(key):
            params[key] = replay_fit.bounded_parameter_value(key, float(value))
    return params


def joint_pareto_audit_candidate_rows(
    states: list[dict[str, Any]],
    *,
    base_parameters: dict[str, float],
    reference_state: dict[str, Any],
    reference_by_longitudinal: dict[str, dict[str, Any]] | None = None,
    global_reference_state: dict[str, Any] | None = None,
    alignment_window_s: float,
    profile: str = DEFAULT_JOINT_PARETO_PROFILE,
    longitudinal_tolerances: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    reference_lookup = reference_by_longitudinal or {}
    global_reference = global_reference_state or reference_state
    global_reference_summary = global_reference.get("summary", {}) or {}
    global_reference_longitudinal_score = finite_value(global_reference.get("longitudinal_score"))
    global_reference_lateral_score = finite_value(global_reference.get("lateral_score"))
    rows = []
    for state in states:
        summary = state.get("summary", {}) or {}
        lateral_source_id = str(state.get("lateral_source_id", ""))
        has_lateral_update = lateral_source_id != "no_lateral_update"
        longitudinal_source_id = str(state.get("longitudinal_source_id", ""))
        matched_reference = reference_lookup.get(longitudinal_source_id, reference_state)
        reference_summary = matched_reference.get("summary", {}) or {}
        reference_longitudinal_score = finite_value(matched_reference.get("longitudinal_score"))
        reference_lateral_score = finite_value(matched_reference.get("lateral_score"))
        accepted, reason = joint_pareto_audit_acceptance(
            reference_summary,
            summary,
            has_lateral_update=has_lateral_update,
            longitudinal_tolerances=longitudinal_tolerances,
        )
        updates = parameter_updates(base_parameters, state["parameters"])
        row = {
            "candidate_id": state.get("candidate_id", ""),
            "selection_class": "",
            "joint_pareto_profile": str(profile),
            "accepted": bool(accepted),
            "acceptance_reason": reason,
            "pareto_member": False,
            "alignment_window_s": float(alignment_window_s),
            "reference_candidate_id": matched_reference.get("candidate_id", ""),
            "global_reference_candidate_id": global_reference.get("candidate_id", ""),
            "longitudinal_source_id": longitudinal_source_id,
            "lateral_source_id": lateral_source_id,
            "split": state.get("split", ""),
            "score": finite_value(state.get("score")),
            "longitudinal_score": finite_value(state.get("longitudinal_score")),
            "reference_longitudinal_score": reference_longitudinal_score,
            "delta_longitudinal_score": finite_value(state.get("longitudinal_score")) - reference_longitudinal_score,
            "lateral_score": finite_value(state.get("lateral_score")),
            "reference_lateral_score": reference_lateral_score,
            "delta_lateral_score": finite_value(state.get("lateral_score")) - reference_lateral_score,
            "global_reference_longitudinal_score": global_reference_longitudinal_score,
            "global_delta_longitudinal_score": finite_value(state.get("longitudinal_score")) - global_reference_longitudinal_score,
            "global_reference_lateral_score": global_reference_lateral_score,
            "global_delta_lateral_score": finite_value(state.get("lateral_score")) - global_reference_lateral_score,
            "parameter_count": len(updates),
            "lateral_parameter_count": joint_sweep_lateral_parameter_count(updates),
            "parameter_updates_json": json.dumps(updates, sort_keys=True),
        }
        row.update(joint_pareto_metric_deltas(reference_summary, summary))
        row.update(
            joint_pareto_metric_deltas(
                global_reference_summary,
                summary,
                include_candidate_values=False,
                reference_key_prefix="global_",
                delta_key_prefix="global_",
            )
        )
        rows.append(row)
    return sorted(rows, key=lambda row: (not bool(row.get("accepted", False)), finite_value(row.get("score"))))


def joint_pareto_metric_deltas(
    reference_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    *,
    include_candidate_values: bool = True,
    reference_key_prefix: str = "",
    delta_key_prefix: str = "",
) -> dict[str, float]:
    metrics = {
        "dx_mae_m": "dx_mae_m",
        "dy_mae_m": "dy_mae_m",
        "altitude_loss_mae_m": "altitude_loss_mae_m",
        "sink_mae_m_s": "sink_mae_m_s",
        "roll_mae_deg": "final_phi_mae_deg",
        "pitch_mae_deg": "final_theta_mae_deg",
        "yaw_mae_deg": "final_psi_mae_deg",
    }
    out: dict[str, float] = {}
    for row_key, summary_key in metrics.items():
        candidate_value = finite_value(candidate_summary.get(summary_key))
        reference_value = finite_value(reference_summary.get(summary_key))
        if include_candidate_values:
            out[row_key] = candidate_value
        out[f"{reference_key_prefix}reference_{row_key}"] = reference_value
        out[f"{delta_key_prefix}delta_{row_key}"] = candidate_value - reference_value
    return out


def joint_pareto_audit_acceptance(
    reference_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    *,
    has_lateral_update: bool = True,
    longitudinal_tolerances: dict[str, float] | None = None,
) -> tuple[bool, str]:
    if not bool(has_lateral_update):
        return False, "reference_longitudinal_only"
    tolerances = dict(longitudinal_tolerances or JOINT_SWEEP_BALANCED_LONGITUDINAL_TOLERANCES)
    longitudinal_failures = []
    for key, tolerance in tolerances.items():
        reference_value = finite_value(reference_summary.get(key))
        candidate_value = finite_value(candidate_summary.get(key))
        if not (math.isfinite(reference_value) and math.isfinite(candidate_value)):
            longitudinal_failures.append(key)
            continue
        if candidate_value > reference_value + float(tolerance):
            longitudinal_failures.append(key)
    if longitudinal_failures:
        return False, "rejected_longitudinal_metrics_degraded:" + ",".join(longitudinal_failures)

    lateral_keys = ("dy_mae_m", "final_phi_mae_deg", "final_psi_mae_deg")
    lateral_failures = []
    for key in lateral_keys:
        reference_value = finite_value(reference_summary.get(key))
        candidate_value = finite_value(candidate_summary.get(key))
        if not (
            math.isfinite(reference_value)
            and math.isfinite(candidate_value)
            and candidate_value < reference_value - 1.0e-9
        ):
            lateral_failures.append(key)
    if lateral_failures:
        return False, "rejected_lateral_metrics_not_all_improved:" + ",".join(lateral_failures)
    return True, "accepted_lateral_metrics_improved_with_longitudinal_tolerance"


def joint_pareto_audit_mark_pareto(rows: list[dict[str, Any]]) -> None:
    accepted = [row for row in rows if bool(row.get("accepted", False))]
    for row in rows:
        row["pareto_member"] = False
    for row in accepted:
        dominated = False
        for other in accepted:
            if other is row:
                continue
            other_long = finite_value(other.get("longitudinal_score"))
            other_lat = finite_value(other.get("lateral_score"))
            row_long = finite_value(row.get("longitudinal_score"))
            row_lat = finite_value(row.get("lateral_score"))
            no_worse = other_long <= row_long + 1.0e-12 and other_lat <= row_lat + 1.0e-12
            strictly_better = other_long < row_long - 1.0e-12 or other_lat < row_lat - 1.0e-12
            if no_worse and strictly_better:
                dominated = True
                break
        row["pareto_member"] = not dominated


def joint_pareto_audit_selected_rows(rows: list[dict[str, Any]], *, selected_limit: int = JOINT_PARETO_AUDIT_SELECTED_LIMIT) -> list[dict[str, Any]]:
    selected = [
        dict(row)
        for row in rows
        if bool(row.get("accepted", False)) and bool(row.get("pareto_member", False))
    ]
    selected = sorted(
        selected,
        key=lambda row: (finite_value(row.get("lateral_score")), finite_value(row.get("longitudinal_score"))),
    )[: max(1, int(selected_limit))]
    for row in selected:
        row["selection_class"] = "accepted_pareto"
    return selected


def lateral_ablation_source_priority(row: dict[str, Any]) -> float:
    accepted_bonus = -10.0 if bool(row.get("accepted", False)) else 0.0
    deltas = (
        finite_value(row.get("delta_dy_mae_m")),
        finite_value(row.get("delta_roll_mae_deg")),
        finite_value(row.get("delta_yaw_mae_deg")),
    )
    if not all(math.isfinite(value) for value in deltas):
        return float("inf")
    return float(accepted_bonus + deltas[0] / 0.45 + deltas[1] / 12.0 + deltas[2] / 18.0)


def parse_json_mapping(value: Any) -> dict[str, float]:
    try:
        parsed = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    out: dict[str, float] = {}
    for key, raw_value in parsed.items():
        numeric = finite_value(raw_value)
        if math.isfinite(numeric):
            out[str(key)] = numeric
    return out


def parse_json_list(value: Any) -> list[str]:
    try:
        parsed = json.loads(str(value or "[]"))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def short_source_id(value: str, limit: int = 44) -> str:
    text = "".join(char if char.isalnum() else "_" for char in str(value)).strip("_")
    while "__" in text:
        text = text.replace("__", "_")
    return (text or "source")[: int(limit)]


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
        lateral_use_confidence_weights=True,
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
                "Cl_p, and Cn_r are fitted with excitation-aware lateral confidence weighting."
            ),
            "fit_result": fit_result,
            "group_scales": lateral_only_scales,
        },
        parameters,
    )


def lateral_ablation_diagnostic_rows(
    *,
    train_rows: list[dict[str, Any]],
    heldout_rows: list[dict[str, Any]],
    primary_parameters: dict[str, float],
    primary_validation_rows: list[dict[str, Any]],
    alignment_window_s: float,
    derivative_window_s: float,
    min_speed_m_s: float,
    ridge_lambda: float,
    replay_dt_s: float,
    workers: int,
) -> list[dict[str, Any]]:
    residuals = residual_rows(
        train_rows,
        split="train",
        parameters=primary_parameters,
        alignment_window_s=alignment_window_s,
        derivative_window_s=derivative_window_s,
        min_speed_m_s=min_speed_m_s,
        workers=workers,
    )
    samples = fit_samples_from_residual_rows(residuals)
    primary_train = replay_summary(primary_validation_rows, "coefficient_candidate", "train")
    primary_heldout = replay_summary(primary_validation_rows, "coefficient_candidate", "heldout")
    output: list[dict[str, Any]] = []
    cy_beta_candidate: dict[str, float] | None = None
    cy_beta_train_summary: dict[str, float] | None = None
    cy_beta_heldout_summary: dict[str, float] | None = None
    for spec in LATERAL_ABLATION_TERMS:
        for regime_family in ("attached", "transition", "post_stall"):
            fit = fit_single_lateral_ablation(
                samples,
                spec,
                regime_family=regime_family,
                base_coeffs=primary_parameters,
                ridge_lambda=ridge_lambda,
            )
            rows_to_append, candidate, split_summaries = evaluate_lateral_ablation_fit(
                fit,
                base_parameters=primary_parameters,
                baseline_model_id="primary_longitudinal",
                train_rows=train_rows,
                heldout_rows=heldout_rows,
                baseline_train=primary_train,
                baseline_heldout=primary_heldout,
                replay_dt_s=replay_dt_s,
                alignment_window_s=alignment_window_s,
                workers=workers,
            )
            output.extend(rows_to_append)
            if (
                fit.get("candidate_id") == "attached_CY_beta"
                and bool(split_summaries.get("heldout", {}).get("accepted", False))
            ):
                cy_beta_candidate = candidate
                cy_beta_train_summary = split_summaries.get("train", {}).get("summary")
                cy_beta_heldout_summary = split_summaries.get("heldout", {}).get("summary")
    if cy_beta_candidate is None or cy_beta_train_summary is None or cy_beta_heldout_summary is None:
        return output

    cy_beta_residuals = residual_rows(
        train_rows,
        split="train",
        parameters=cy_beta_candidate,
        alignment_window_s=alignment_window_s,
        derivative_window_s=derivative_window_s,
        min_speed_m_s=min_speed_m_s,
        workers=workers,
    )
    cy_beta_samples = fit_samples_from_residual_rows(cy_beta_residuals)
    for spec in LATERAL_ABLATION_TERMS:
        if str(spec.get("term", "")) == "CY_beta":
            continue
        for regime_family in ("attached", "transition", "post_stall"):
            fit = fit_single_lateral_ablation(
                cy_beta_samples,
                spec,
                regime_family=regime_family,
                base_coeffs=cy_beta_candidate,
                ridge_lambda=ridge_lambda,
            )
            fit["candidate_id"] = f"after_CY_beta_{fit.get('candidate_id', '')}"
            rows_to_append, _, _ = evaluate_lateral_ablation_fit(
                fit,
                base_parameters=cy_beta_candidate,
                baseline_model_id="attached_CY_beta",
                train_rows=train_rows,
                heldout_rows=heldout_rows,
                baseline_train=cy_beta_train_summary,
                baseline_heldout=cy_beta_heldout_summary,
                replay_dt_s=replay_dt_s,
                alignment_window_s=alignment_window_s,
                workers=workers,
            )
            output.extend(rows_to_append)
    return output


def evaluate_lateral_ablation_fit(
    fit: dict[str, Any],
    *,
    base_parameters: dict[str, float],
    baseline_model_id: str,
    train_rows: list[dict[str, Any]],
    heldout_rows: list[dict[str, Any]],
    baseline_train: dict[str, float],
    baseline_heldout: dict[str, float],
    replay_dt_s: float,
    alignment_window_s: float,
    workers: int,
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, dict[str, Any]]]:
    candidate = lateral_ablation_candidate(base_parameters, fit)
    output: list[dict[str, Any]] = []
    split_summaries: dict[str, dict[str, Any]] = {}
    for split, rows, baseline_summary in (
        ("train", train_rows, baseline_train),
        ("heldout", heldout_rows, baseline_heldout),
    ):
        replay_rows = replay_fit.simulate_rows(
            rows,
            candidate,
            replay_dt_s=replay_dt_s,
            alignment_window_s=alignment_window_s,
            workers=workers,
        )
        candidate_summary = replay_fit.objective_summary(replay_rows)
        accepted, reason = lateral_ablation_acceptance(baseline_summary, candidate_summary)
        split_summaries[split] = {
            "summary": candidate_summary,
            "accepted": accepted,
            "reason": reason,
        }
        output.append(
            lateral_ablation_output_row(
                fit,
                baseline_model_id=baseline_model_id,
                split=split,
                primary_summary=baseline_summary,
                candidate_summary=candidate_summary,
                accepted=accepted,
                reason=reason,
            )
        )
    return output, candidate, split_summaries


def fit_single_lateral_ablation(
    samples: list[dict[str, Any]],
    spec: dict[str, str],
    *,
    regime_family: str,
    base_coeffs: dict[str, float],
    ridge_lambda: float,
) -> dict[str, Any]:
    if regime_family == "attached":
        fit_samples = [sample for sample in samples if sample.get("stage_fit_group") == "attached"]
        parameter_keys = [str(spec["attached_key"])]
    elif regime_family == "transition":
        fit_samples = [sample for sample in samples if sample.get("regime") == "transition"]
        parameter_keys = [str(spec["transition_key"])]
    elif regime_family == "post_stall":
        fit_samples = [sample for sample in samples if sample.get("regime") == "post_stall"]
        parameter_keys = [
            lateral_surface_parameter_name(
                str(spec["post_stall_prefix"]),
                str(spec["post_stall_feature"]),
                SURFACE_RBF_ALPHA_CENTERS_DEG[0],
            )
        ]
    else:
        raise ValueError(f"Unsupported lateral ablation regime: {regime_family}")

    x_rows = []
    y_values = []
    q_bar_values = []
    kept_samples = []
    for sample in fit_samples:
        feature = lateral_ablation_feature_value(sample, spec, regime_family, base_coeffs)
        y = finite_value(sample.get(str(spec["residual_key"])))
        q_bar = finite_value(sample.get("q_bar"))
        if not (math.isfinite(feature) and math.isfinite(y) and math.isfinite(q_bar)):
            continue
        if abs(feature) <= 1.0e-12:
            continue
        x_rows.append([feature])
        y_values.append(y)
        q_bar_values.append(q_bar)
        kept_samples.append(sample)

    min_count = 8 if regime_family != "post_stall" else 6
    if len(y_values) < min_count:
        coefficient = 0.0
        used_count = len(y_values)
        fit_mae = float("nan")
        fit_rmse = float("nan")
        status = "too_few_samples"
    else:
        x = np.asarray(x_rows, dtype=float)
        y_array = np.asarray(y_values, dtype=float)
        q_bar = np.asarray(q_bar_values, dtype=float)
        weights = dynamic_pressure_weights(q_bar) * throw_balance_weights(kept_samples) * lateral_excitation_confidence_weights(kept_samples)
        coeff, used_count, fit_mae, fit_rmse = robust_weighted_ridge_fit(
            x,
            y_array,
            weights,
            ridge_lambda=float(ridge_lambda),
            min_used_count=min_count,
        )
        coefficient = replay_fit.bounded_parameter_value(parameter_keys[0], float(coeff[0]))
        status = "ok" if math.isfinite(coefficient) else "nonfinite_fit"
    return {
        "candidate_id": f"{regime_family}_{spec['term']}",
        "term": str(spec["term"]),
        "regime_family": str(regime_family),
        "coefficient": float(coefficient),
        "fit_sign": coefficient_sign(float(coefficient)),
        "fit_sample_count": int(len(y_values)),
        "fit_used_sample_count": int(used_count),
        "fit_mae": float(fit_mae),
        "fit_rmse": float(fit_rmse),
        "status": status,
        "parameter_keys": parameter_keys,
    }


def lateral_ablation_feature_value(
    sample: dict[str, Any],
    spec: dict[str, str],
    regime_family: str,
    base_coeffs: dict[str, float],
) -> float:
    feature = finite_value(sample.get(str(spec["feature_key"])))
    if not math.isfinite(feature):
        return float("nan")
    if regime_family == "transition":
        return feature * transition_window_weight_deg(finite_value(sample.get("alpha_deg")), base_coeffs)
    if regime_family == "post_stall":
        basis = surface_rbf_basis_deg(
            finite_value(sample.get("alpha_deg")),
            start_alpha_deg=float(base_coeffs.get("post_stall_residual_blend_start_alpha_deg", STALL_ALPHA_DEG)),
            full_alpha_deg=float(base_coeffs.get("post_stall_residual_blend_full_alpha_deg", POST_STALL_ALPHA_DEG)),
        )
        return feature * float(basis[0])
    return feature


def lateral_ablation_candidate(base: dict[str, float], fit: dict[str, Any]) -> dict[str, float]:
    candidate = dict(base)
    value = float(fit.get("coefficient", 0.0))
    for key in fit.get("parameter_keys", []):
        candidate[str(key)] = replay_fit.bounded_parameter_value(str(key), value)
    return candidate


def coefficient_sign(value: float) -> str:
    if not math.isfinite(float(value)) or abs(float(value)) <= 1.0e-12:
        return "zero"
    return "positive" if float(value) > 0.0 else "negative"


def lateral_ablation_acceptance(
    primary: dict[str, float],
    candidate: dict[str, float],
) -> tuple[bool, str]:
    lateral_keys = ("dy_mae_m", "final_phi_mae_deg", "final_psi_mae_deg")
    longitudinal_tolerances = {
        "dx_mae_m": LONGITUDINAL_STAGE_DX_TOL_M,
        "altitude_loss_mae_m": LONGITUDINAL_STAGE_ALTITUDE_LOSS_TOL_M,
        "sink_mae_m_s": LONGITUDINAL_STAGE_SINK_TOL_M_S,
        "final_theta_mae_deg": LONGITUDINAL_STAGE_PITCH_TOL_DEG,
    }
    lateral_pass = []
    for key in lateral_keys:
        primary_value = finite_value(primary.get(key))
        candidate_value = finite_value(candidate.get(key))
        lateral_pass.append(
            math.isfinite(primary_value)
            and math.isfinite(candidate_value)
            and candidate_value < primary_value - 1.0e-9
        )
    longitudinal_pass = []
    for key, tolerance in longitudinal_tolerances.items():
        primary_value = finite_value(primary.get(key))
        candidate_value = finite_value(candidate.get(key))
        longitudinal_pass.append(
            math.isfinite(primary_value)
            and math.isfinite(candidate_value)
            and candidate_value <= primary_value + float(tolerance)
        )
    if all(lateral_pass) and all(longitudinal_pass):
        return True, "heldout_lateral_improved_with_longitudinal_tolerance"
    if not all(lateral_pass):
        return False, "rejected_lateral_metrics_not_all_improved"
    return False, "rejected_longitudinal_metrics_degraded"


def lateral_ablation_output_row(
    fit: dict[str, Any],
    *,
    baseline_model_id: str,
    split: str,
    primary_summary: dict[str, float],
    candidate_summary: dict[str, float],
    accepted: bool,
    reason: str,
) -> dict[str, Any]:
    return {
        "candidate_id": fit.get("candidate_id", ""),
        "baseline_model_id": baseline_model_id,
        "term": fit.get("term", ""),
        "regime_family": fit.get("regime_family", ""),
        "split": split,
        "fit_sample_count": int(fit.get("fit_sample_count", 0)),
        "fit_used_sample_count": int(fit.get("fit_used_sample_count", 0)),
        "fit_coefficient": finite_value(fit.get("coefficient")),
        "fit_sign": fit.get("fit_sign", ""),
        "parameter_keys_json": json.dumps(fit.get("parameter_keys", [])),
        "baseline_dy_mae_m": finite_value(primary_summary.get("dy_mae_m")),
        "candidate_dy_mae_m": finite_value(candidate_summary.get("dy_mae_m")),
        "delta_dy_mae_m": finite_value(candidate_summary.get("dy_mae_m")) - finite_value(primary_summary.get("dy_mae_m")),
        "baseline_roll_mae_deg": finite_value(primary_summary.get("final_phi_mae_deg")),
        "candidate_roll_mae_deg": finite_value(candidate_summary.get("final_phi_mae_deg")),
        "delta_roll_mae_deg": finite_value(candidate_summary.get("final_phi_mae_deg")) - finite_value(primary_summary.get("final_phi_mae_deg")),
        "baseline_yaw_mae_deg": finite_value(primary_summary.get("final_psi_mae_deg")),
        "candidate_yaw_mae_deg": finite_value(candidate_summary.get("final_psi_mae_deg")),
        "delta_yaw_mae_deg": finite_value(candidate_summary.get("final_psi_mae_deg")) - finite_value(primary_summary.get("final_psi_mae_deg")),
        "baseline_dx_mae_m": finite_value(primary_summary.get("dx_mae_m")),
        "candidate_dx_mae_m": finite_value(candidate_summary.get("dx_mae_m")),
        "delta_dx_mae_m": finite_value(candidate_summary.get("dx_mae_m")) - finite_value(primary_summary.get("dx_mae_m")),
        "baseline_altitude_loss_mae_m": finite_value(primary_summary.get("altitude_loss_mae_m")),
        "candidate_altitude_loss_mae_m": finite_value(candidate_summary.get("altitude_loss_mae_m")),
        "delta_altitude_loss_mae_m": finite_value(candidate_summary.get("altitude_loss_mae_m")) - finite_value(primary_summary.get("altitude_loss_mae_m")),
        "baseline_sink_mae_m_s": finite_value(primary_summary.get("sink_mae_m_s")),
        "candidate_sink_mae_m_s": finite_value(candidate_summary.get("sink_mae_m_s")),
        "delta_sink_mae_m_s": finite_value(candidate_summary.get("sink_mae_m_s")) - finite_value(primary_summary.get("sink_mae_m_s")),
        "baseline_pitch_mae_deg": finite_value(primary_summary.get("final_theta_mae_deg")),
        "candidate_pitch_mae_deg": finite_value(candidate_summary.get("final_theta_mae_deg")),
        "delta_pitch_mae_deg": finite_value(candidate_summary.get("final_theta_mae_deg")) - finite_value(primary_summary.get("final_theta_mae_deg")),
        "accepted": bool(accepted) if split == "heldout" else "",
        "acceptance_reason": reason if split == "heldout" else "",
    }


def lateral_launch_correlation_report_rows(validation_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    residual_metrics = (
        ("dy_residual_actual_minus_sim_m", "dy"),
        ("final_phi_residual_actual_minus_sim_deg", "roll"),
        ("final_psi_residual_actual_minus_sim_deg", "yaw"),
    )
    launch_variables = ("v0_m_s", "p0_rad_s", "r0_rad_s", "phi0_deg", "psi0_deg")
    output: list[dict[str, Any]] = []
    model_ids = sorted({str(row.get("model_id", "")) for row in validation_rows if row.get("replay_status") == "ok"})
    for model_id in model_ids:
        for split in ("train", "heldout"):
            subset = [
                row
                for row in validation_rows
                if row.get("model_id") == model_id
                and row.get("split") == split
                and row.get("replay_status") == "ok"
            ]
            for residual_key, residual_label in residual_metrics:
                for launch_key in launch_variables:
                    output.append(
                        lateral_launch_correlation_row(
                            subset,
                            model_id=model_id,
                            split=split,
                            residual_key=residual_key,
                            residual_label=residual_label,
                            launch_key=launch_key,
                        )
                    )
    return output


def lateral_launch_correlation_row(
    rows: list[dict[str, Any]],
    *,
    model_id: str,
    split: str,
    residual_key: str,
    residual_label: str,
    launch_key: str,
) -> dict[str, Any]:
    pairs = [
        (finite_value(row.get(launch_key)), finite_value(row.get(residual_key)))
        for row in rows
    ]
    pairs = [(x, y) for x, y in pairs if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return {
            "model_id": model_id,
            "split": split,
            "residual_metric": residual_label,
            "launch_variable": launch_key,
            "sample_count": len(pairs),
            "pearson_r": float("nan"),
            "abs_pearson_r": float("nan"),
            "slope": float("nan"),
            "intercept": float("nan"),
            "residual_mean": float("nan"),
            "launch_variable_mean": float("nan"),
        }
    x = np.asarray([pair[0] for pair in pairs], dtype=float)
    y = np.asarray([pair[1] for pair in pairs], dtype=float)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    x_centred = x - x_mean
    y_centred = y - y_mean
    denom = float(np.sqrt(np.sum(x_centred**2) * np.sum(y_centred**2)))
    pearson = float(np.sum(x_centred * y_centred) / denom) if denom > 1.0e-12 else float("nan")
    slope = float(np.sum(x_centred * y_centred) / np.sum(x_centred**2)) if float(np.sum(x_centred**2)) > 1.0e-12 else float("nan")
    intercept = y_mean - slope * x_mean if math.isfinite(slope) else float("nan")
    return {
        "model_id": model_id,
        "split": split,
        "residual_metric": residual_label,
        "launch_variable": launch_key,
        "sample_count": len(pairs),
        "pearson_r": pearson,
        "abs_pearson_r": abs(pearson) if math.isfinite(pearson) else float("nan"),
        "slope": slope,
        "intercept": intercept,
        "residual_mean": y_mean,
        "launch_variable_mean": x_mean,
    }


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


def joint_pareto_heavy_stage_replay_summary_rows(
    *,
    candidate_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    base_parameters: dict[str, float],
    heldout_rows: list[dict[str, Any]],
    replay_dt_s: float,
    alignment_window_s: float,
    workers: int,
) -> list[dict[str, Any]]:
    if not selected_rows or not heldout_rows:
        return []
    row_by_id = {str(row.get("candidate_id", "")): row for row in candidate_rows}
    model_specs: list[tuple[str, str, str, dict[str, float]]] = []
    for selected in selected_rows:
        selected_id = str(selected.get("candidate_id", ""))
        if not selected_id:
            continue
        model_specs.append(
            (
                selected_id,
                "selected_candidate",
                selected_id,
                joint_pareto_parameters_from_row(base_parameters, selected),
            )
        )
        reference_id = str(selected.get("reference_candidate_id", ""))
        reference_row = row_by_id.get(reference_id)
        if reference_row is not None:
            model_specs.append(
                (
                    f"{selected_id}__matched_reference",
                    "matched_longitudinal_reference",
                    selected_id,
                    joint_pareto_parameters_from_row(base_parameters, reference_row),
                )
            )
        global_reference_id = str(selected.get("global_reference_candidate_id", ""))
        global_reference_row = row_by_id.get(global_reference_id)
        if global_reference_row is not None:
            model_specs.append(
                (
                    f"{selected_id}__global_reference",
                    "global_longitudinal_reference",
                    selected_id,
                    joint_pareto_parameters_from_row(base_parameters, global_reference_row),
                )
            )

    unique_specs: list[tuple[str, str, str, dict[str, float]]] = []
    seen_model_ids: set[str] = set()
    for spec in model_specs:
        if spec[0] in seen_model_ids:
            continue
        seen_model_ids.add(spec[0])
        unique_specs.append(spec)
    if not unique_specs:
        return []

    payloads = [
        (model_id, "heldout", row, parameters, replay_dt_s, alignment_window_s)
        for model_id, _, _, parameters in unique_specs
        for row in heldout_rows
    ]
    if int(workers) > 1 and len(payloads) > 1:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            nested = list(executor.map(stage_replay_sample_rows_payload, payloads))
    else:
        nested = [stage_replay_sample_rows_payload(payload) for payload in payloads]
    sample_rows = [row for rows in nested for row in rows]
    summary_rows = summarize_stage_replay_samples(sample_rows)
    metadata_by_model = {
        model_id: {"model_role": role, "matched_candidate_id": matched_id}
        for model_id, role, matched_id, _ in unique_specs
    }
    out = []
    for model_id, role, matched_id, _ in unique_specs:
        for regime in ("attached", "transition", "post_stall"):
            row = stage_replay_row(summary_rows, model_id, "heldout", regime)
            enriched = dict(row)
            enriched["model_id"] = model_id
            enriched["model_role"] = metadata_by_model[model_id]["model_role"]
            enriched["matched_candidate_id"] = metadata_by_model[model_id]["matched_candidate_id"]
            enriched["split"] = "heldout"
            enriched["regime"] = regime
            enriched["report_regime"] = "normal" if regime == "attached" else regime
            out.append(enriched)
    return out


def joint_pareto_parameters_from_row(base_parameters: dict[str, float], row: dict[str, Any]) -> dict[str, float]:
    params = dict(base_parameters)
    updates = parse_json_mapping(row.get("parameter_updates_json", "{}"))
    for key, value in updates.items():
        params[str(key)] = replay_fit.bounded_parameter_value(str(key), float(value))
    return params


def tag_alignment_sensitivity_rows(
    rows: list[dict[str, Any]],
    *,
    primary_alignment_window_s: float,
    sensitivity_alignment_window_s: float,
) -> list[dict[str, Any]]:
    tagged_rows: list[dict[str, Any]] = []
    for row in rows:
        tagged = dict(row)
        tagged["primary_alignment_window_s"] = float(primary_alignment_window_s)
        tagged["sensitivity_alignment_window_s"] = float(sensitivity_alignment_window_s)
        tagged_rows.append(tagged)
    return tagged_rows


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
    sensitivity_alignment_windows_s: tuple[float, ...],
    joint_pareto_audit: bool,
    joint_pareto_audit_alignment_window_s: float,
    joint_pareto_config: dict[str, Any],
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
    lateral_ablation_rows: list[dict[str, Any]],
    lateral_launch_correlation_rows: list[dict[str, Any]],
    sensitivity_validation_rows: list[dict[str, Any]],
    sensitivity_stage_replay_rows: list[dict[str, Any]],
    joint_sweep_candidate_rows: list[dict[str, Any]],
    joint_sweep_pareto_rows: list[dict[str, Any]],
    joint_sweep_selected_rows: list[dict[str, Any]],
    joint_pareto_audit_candidate_rows: list[dict[str, Any]],
    joint_pareto_audit_selected_rows: list[dict[str, Any]],
    joint_pareto_heavy_stage_replay_rows: list[dict[str, Any]],
) -> None:
    manifest = {
        "fit_id": str(run_label),
        "fit_version": FIT_VERSION,
        "fit_scope": "neutral_open_loop_vicon_6dof_force_moment_residual_regime_staged_and_compact_joint_sweep_fit",
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
        "alignment_sensitivity": {
            "enabled": bool(sensitivity_alignment_windows_s),
            "policy": (
                "replay-only diagnostic using the already fitted baseline/candidate models; "
                "does not change coefficients, held-out split, or acceptance gate"
            ),
            "primary_alignment_window_s": float(alignment_window_s),
            "sensitivity_alignment_windows_s": [float(value) for value in sensitivity_alignment_windows_s],
            "replay_validation_csv": "metrics/neutral_aero_residual_replay_sensitivity.csv",
            "stage_replay_csv": "metrics/neutral_aero_residual_stage_replay_sensitivity.csv",
            "replay_row_count": len(sensitivity_validation_rows),
            "stage_replay_row_count": len(sensitivity_stage_replay_rows),
        },
        "joint_pareto_audit": {
            "enabled": bool(joint_pareto_audit),
            "policy": (
                "diagnostic held-out replay at 0.040 s by default; combines top longitudinal candidates with "
                "compact lateral/cross-coupling candidates and accepts only lateral improvements that keep "
                "longitudinal metrics within balanced tolerance"
            ),
            "alignment_window_s": float(joint_pareto_audit_alignment_window_s),
            "candidate_csv": "metrics/neutral_aero_residual_joint_pareto_audit_candidates.csv",
            "selected_csv": "metrics/neutral_aero_residual_joint_pareto_audit_selected.csv",
            "heavy_candidate_csv": (
                "metrics/neutral_aero_residual_joint_pareto_heavy_candidates.csv"
                if str(joint_pareto_config.get("profile", "")) == "heavy"
                else ""
            ),
            "heavy_selected_csv": (
                "metrics/neutral_aero_residual_joint_pareto_heavy_selected.csv"
                if str(joint_pareto_config.get("profile", "")) == "heavy"
                else ""
            ),
            "heavy_stage_replay_csv": (
                "metrics/neutral_aero_residual_joint_pareto_heavy_stage_replay.csv"
                if str(joint_pareto_config.get("profile", "")) == "heavy"
                else ""
            ),
            "candidate_row_count": len(joint_pareto_audit_candidate_rows),
            "accepted_row_count": sum(bool(row.get("accepted", False)) for row in joint_pareto_audit_candidate_rows),
            "selected_row_count": len(joint_pareto_audit_selected_rows),
            "heavy_stage_replay_row_count": len(joint_pareto_heavy_stage_replay_rows),
            "profile": str(joint_pareto_config.get("profile", "")),
            "top_longitudinal": int(joint_pareto_config.get("top_longitudinal", 0)),
            "top_lateral": int(joint_pareto_config.get("top_lateral", 0)),
            "max_lateral_order": int(joint_pareto_config.get("max_lateral_order", 0)),
            "top_triples": int(joint_pareto_config.get("top_triples", 0)),
            "max_candidates": int(joint_pareto_config.get("max_candidates", 0)),
            "selected_limit": int(joint_pareto_config.get("selected_limit", 0)),
            "scale_grid": [float(value) for value in joint_pareto_config.get("scale_grid", ())],
            "local_longitudinal_source_ids": list(joint_pareto_config.get("local_longitudinal_source_ids", ())),
            "local_yaw_beta_scale_grid": [
                float(value) for value in joint_pareto_config.get("local_yaw_beta_scale_grid", ())
            ],
            "local_post_stall_clr_scale_grid": [
                float(value) for value in joint_pareto_config.get("local_post_stall_clr_scale_grid", ())
            ],
            "reference_policy": str(joint_pareto_config.get("reference_policy", "")),
            "scaled_single_term_bundles": bool(joint_pareto_config.get("scaled_single_term_bundles", False)),
            "include_rejected_stage_candidates": bool(
                joint_pareto_config.get("include_rejected_stage_candidates", False)
            ),
            "longitudinal_tolerances": joint_pareto_profile_longitudinal_tolerances(joint_pareto_config),
        },
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
            "secondary_lateral_diagnostic": "optional frozen-longitudinal CY_beta/Cl_p/Cn_r diagnostic with excitation-aware lateral weighting, accepted only by held-out lateral improvement without longitudinal degradation",
            "staged_replay": "cm_regime_staged fits attached, transition, post-stall, blend, then optional CL/CD cleanup with held-out gates at each stage",
        },
        "base_parameters": dict(base_parameters),
        "candidate_parameters": dict(candidate_parameters),
        "lateral_diagnostic_parameters": dict(lateral_diagnostic_parameters or {}),
        "candidate_acceptance": acceptance_summary,
        "lateral_diagnostic_acceptance": lateral_diagnostic_acceptance,
        "lateral_ablation": {
            "metrics_csv": "metrics/neutral_aero_residual_lateral_ablation.csv",
            "row_count": len(lateral_ablation_rows),
            "policy": "diagnostic one-term lateral ablations with longitudinal candidate frozen",
        },
        "lateral_launch_correlation": {
            "metrics_csv": "metrics/neutral_aero_residual_lateral_launch_correlation.csv",
            "row_count": len(lateral_launch_correlation_rows),
            "policy": "correlate replay lateral residuals with replay-aligned v0, p0, r0, phi0, and psi0",
        },
        "compact_joint_sweep": {
            "metrics_csv": "metrics/neutral_aero_residual_joint_sweep_candidates.csv",
            "pareto_csv": "metrics/neutral_aero_residual_joint_sweep_pareto.csv",
            "selected_csv": "metrics/neutral_aero_residual_joint_sweep_selected.csv",
            "candidate_row_count": len(joint_sweep_candidate_rows),
            "pareto_row_count": len(joint_sweep_pareto_rows),
            "selected_row_count": len(joint_sweep_selected_rows),
            "beam_width": COMPACT_JOINT_SWEEP_BEAM_WIDTH,
            "heldout_eval_limit": COMPACT_JOINT_SWEEP_HELDOUT_EVAL_LIMIT,
            "policy": "from-active compact sign-constrained joint sweep with 8-worker replay evaluation",
        },
        "fit_result": fit_result,
        "lateral_diagnostic_result": lateral_diagnostic_result,
        "rerun_command": fit_rerun_command(
            run_label=run_label,
            session_root=session_root,
            heldout_count=len(heldout_indices),
            heldout_seed=heldout_seed,
            alignment_window_s=alignment_window_s,
            sensitivity_alignment_windows_s=sensitivity_alignment_windows_s,
            joint_pareto_audit=joint_pareto_audit,
            joint_pareto_audit_alignment_window_s=joint_pareto_audit_alignment_window_s,
            joint_pareto_config=joint_pareto_config,
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
    if str(joint_pareto_config.get("profile", "")) == "heavy":
        heavy_manifest = {
            "fit_id": str(run_label),
            "artifact": "neutral_aero_residual_joint_pareto_heavy",
            "alignment_window_s": float(joint_pareto_audit_alignment_window_s),
            "workers": int(workers),
            "profile": str(joint_pareto_config.get("profile", "")),
            "top_longitudinal": int(joint_pareto_config.get("top_longitudinal", 0)),
            "top_lateral": int(joint_pareto_config.get("top_lateral", 0)),
            "max_lateral_order": int(joint_pareto_config.get("max_lateral_order", 0)),
            "top_triples": int(joint_pareto_config.get("top_triples", 0)),
            "max_candidates": int(joint_pareto_config.get("max_candidates", 0)),
            "selected_limit": int(joint_pareto_config.get("selected_limit", 0)),
            "scale_grid": [float(value) for value in joint_pareto_config.get("scale_grid", ())],
            "reference_policy": str(joint_pareto_config.get("reference_policy", "")),
            "include_rejected_stage_candidates": bool(
                joint_pareto_config.get("include_rejected_stage_candidates", False)
            ),
            "candidate_count": len(joint_pareto_audit_candidate_rows),
            "accepted_count": sum(bool(row.get("accepted", False)) for row in joint_pareto_audit_candidate_rows),
            "selected_count": len(joint_pareto_audit_selected_rows),
            "stage_replay_row_count": len(joint_pareto_heavy_stage_replay_rows),
            "candidate_csv": "metrics/neutral_aero_residual_joint_pareto_heavy_candidates.csv",
            "selected_csv": "metrics/neutral_aero_residual_joint_pareto_heavy_selected.csv",
            "stage_replay_csv": "metrics/neutral_aero_residual_joint_pareto_heavy_stage_replay.csv",
            "command": manifest["rerun_command"],
            "command_line": replay_fit.powershell_command_line(manifest["rerun_command"]),
            "generated_at": manifest["generated_at"],
        }
        heavy_path = output_dir / "manifests" / "neutral_aero_residual_joint_pareto_heavy_manifest.json"
        heavy_path.write_text(json.dumps(heavy_manifest, indent=2) + "\n", encoding="utf-8")


def fit_rerun_command(
    *,
    run_label: str,
    session_root: Path,
    heldout_count: int,
    heldout_seed: int,
    alignment_window_s: float,
    sensitivity_alignment_windows_s: tuple[float, ...],
    joint_pareto_audit: bool,
    joint_pareto_audit_alignment_window_s: float,
    joint_pareto_config: dict[str, Any],
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
    if sensitivity_alignment_windows_s:
        for sensitivity_window_s in sensitivity_alignment_windows_s:
            command.extend(["--sensitivity-alignment-window-s", f"{float(sensitivity_window_s):.6g}"])
    else:
        command.append("--no-sensitivity-alignment")
    command.append("--joint-pareto-audit" if joint_pareto_audit else "--no-joint-pareto-audit")
    command.extend(
        [
            "--joint-pareto-audit-alignment-window-s",
            f"{float(joint_pareto_audit_alignment_window_s):.6g}",
            "--joint-pareto-profile",
            str(joint_pareto_config.get("profile", DEFAULT_JOINT_PARETO_PROFILE)),
            "--joint-pareto-top-longitudinal",
            str(int(joint_pareto_config.get("top_longitudinal", JOINT_PARETO_AUDIT_TOP_LONGITUDINAL))),
            "--joint-pareto-top-lateral",
            str(int(joint_pareto_config.get("top_lateral", JOINT_PARETO_AUDIT_TOP_LATERAL))),
            "--joint-pareto-max-lateral-order",
            str(int(joint_pareto_config.get("max_lateral_order", 1))),
            "--joint-pareto-top-triples",
            str(int(joint_pareto_config.get("top_triples", 0))),
            "--joint-pareto-max-candidates",
            str(int(joint_pareto_config.get("max_candidates", 64))),
            "--joint-pareto-selected-limit",
            str(int(joint_pareto_config.get("selected_limit", JOINT_PARETO_AUDIT_SELECTED_LIMIT))),
        ]
    )
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
    sensitivity_alignment_windows_s: tuple[float, ...],
    joint_pareto_audit: bool,
    joint_pareto_audit_alignment_window_s: float,
    joint_pareto_config: dict[str, Any],
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
    lateral_ablation_rows: list[dict[str, Any]],
    lateral_launch_correlation_rows: list[dict[str, Any]],
    joint_sweep_candidate_rows: list[dict[str, Any]],
    joint_sweep_pareto_rows: list[dict[str, Any]],
    joint_sweep_selected_rows: list[dict[str, Any]],
    joint_pareto_audit_candidate_rows: list[dict[str, Any]],
    joint_pareto_audit_selected_rows: list[dict[str, Any]],
    joint_pareto_heavy_stage_replay_rows: list[dict[str, Any]],
    regime_summary: list[dict[str, Any]],
    stage_fit_summary: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    stage_replay_rows: list[dict[str, Any]],
    sensitivity_validation_rows: list[dict[str, Any]],
    sensitivity_stage_replay_rows: list[dict[str, Any]],
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
            sensitivity_alignment_windows_s=sensitivity_alignment_windows_s,
            joint_pareto_audit=joint_pareto_audit,
            joint_pareto_audit_alignment_window_s=joint_pareto_audit_alignment_window_s,
            joint_pareto_config=joint_pareto_config,
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
        "This run uses only neutral open-loop real throws. It estimates 6-DoF force/moment residuals from Vicon state trajectories, fits a claim-bearing longitudinal candidate by default, reports lateral residuals without claiming accurate lateral SysID, and validates candidates by held-out dry-air replay. The default `cm_regime_staged` workflow fits attached Cm, transition Cm, post-stall Cm/Cmq, transition blend, and optional post-stall CL/CD cleanup in separate held-out-gated stages. The `compact_joint_sweep` workflow starts from active constants, keeps the same compact model family, and jointly sweeps longitudinal plus small lateral/coupling terms after sign/range discovery. Longitudinal fitting uses lateral-contamination confidence; lateral/coupling fitting uses excitation-aware confidence. Rich transition lateral deltas, post-stall lateral surfaces, and post-stall alpha-RBF longitudinal surfaces are diagnostic-only unless explicitly enabled.",
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
        "- diagnostic policy: longitudinal parameters are frozen at the primary candidate; only `CY_beta`, `Cl_p`, and `Cn_r` may change; lateral-only fitting uses excitation-aware confidence weighting",
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
        "## Lateral One-Term Ablation",
        "",
        "- ablation CSV: `metrics/neutral_aero_residual_lateral_ablation.csv`",
        "- policy: longitudinal candidate is frozen; each diagnostic candidate fits one lateral term and one regime family at a time. If held-out `attached_CY_beta` is accepted, a second pass tests remaining cross-couplings against that side-force baseline.",
        "- acceptance: held-out dy, roll, and yaw must all improve while dx, altitude loss, sink, and pitch stay within practical tolerance.",
        lateral_ablation_report_lines(lateral_ablation_rows),
        "",
        "## Compact Joint Sweep",
        "",
        "- candidate CSV: `metrics/neutral_aero_residual_joint_sweep_candidates.csv`",
        "- pareto CSV: `metrics/neutral_aero_residual_joint_sweep_pareto.csv`",
        "- selected CSV: `metrics/neutral_aero_residual_joint_sweep_selected.csv`",
        "- policy: from active constants only; signs/ranges are discovered from current residuals, then compact longitudinal/lateral terms are swept jointly with 8-worker replay.",
        compact_joint_sweep_report_lines(joint_sweep_candidate_rows, joint_sweep_pareto_rows, joint_sweep_selected_rows),
        "",
        "## 40 ms Joint Pareto Audit",
        "",
        "- candidate CSV: `metrics/neutral_aero_residual_joint_pareto_audit_candidates.csv`",
        "- selected CSV: `metrics/neutral_aero_residual_joint_pareto_audit_selected.csv`",
        "- policy: diagnostic held-out replay at the launch-handoff-aligned window; accepted rows must keep longitudinal metrics within balanced tolerance while improving dy, roll, and yaw.",
        f"- profile: `{joint_pareto_config.get('profile', DEFAULT_JOINT_PARETO_PROFILE)}`",
        joint_pareto_audit_report_lines(
            candidate_rows=joint_pareto_audit_candidate_rows,
            selected_rows=joint_pareto_audit_selected_rows,
            enabled=joint_pareto_audit,
            alignment_window_s=joint_pareto_audit_alignment_window_s,
        ),
        joint_pareto_heavy_report_lines(
            config=joint_pareto_config,
            selected_rows=joint_pareto_audit_selected_rows,
            stage_replay_rows=joint_pareto_heavy_stage_replay_rows,
        ),
        "",
        "## Lateral Launch-Correlation Audit",
        "",
        "- correlation CSV: `metrics/neutral_aero_residual_lateral_launch_correlation.csv`",
        "- interpretation: strong correlation means the remaining lateral replay error is largely launch-condition dependent, so bad lateral launches should be down-weighted before stronger lateral aerodynamics are promoted.",
        lateral_launch_correlation_report_lines(lateral_launch_correlation_rows),
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
        "## Alignment-Window Sensitivity Replay",
        "",
        alignment_sensitivity_report_lines(
            validation_rows=sensitivity_validation_rows,
            stage_replay_rows=sensitivity_stage_replay_rows,
            primary_alignment_window_s=alignment_window_s,
            sensitivity_alignment_windows_s=sensitivity_alignment_windows_s,
        ),
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
    if str(joint_pareto_config.get("profile", "")) == "heavy":
        heavy_lines = [
            "# Heavy 40 ms Joint Pareto SysID Refit",
            "",
            f"- fit id: `{run_label}`",
            f"- alignment window: `{joint_pareto_audit_alignment_window_s:.3f}` s",
            f"- workers: `{workers}`",
            f"- top longitudinal: `{int(joint_pareto_config.get('top_longitudinal', 0))}`",
            f"- top lateral: `{int(joint_pareto_config.get('top_lateral', 0))}`",
            f"- max lateral order: `{int(joint_pareto_config.get('max_lateral_order', 0))}`",
            f"- top triples: `{int(joint_pareto_config.get('top_triples', 0))}`",
            f"- max candidates: `{int(joint_pareto_config.get('max_candidates', 0))}`",
            f"- selected limit: `{int(joint_pareto_config.get('selected_limit', 0))}`",
            "- candidate CSV: `metrics/neutral_aero_residual_joint_pareto_heavy_candidates.csv`",
            "- selected CSV: `metrics/neutral_aero_residual_joint_pareto_heavy_selected.csv`",
            "- stage replay CSV: `metrics/neutral_aero_residual_joint_pareto_heavy_stage_replay.csv`",
            "",
            "## Selected Candidates",
            "",
            joint_pareto_audit_report_lines(
                candidate_rows=joint_pareto_audit_candidate_rows,
                selected_rows=joint_pareto_audit_selected_rows,
                enabled=joint_pareto_audit,
                alignment_window_s=joint_pareto_audit_alignment_window_s,
            ),
            "",
            "## Held-Out Regime Replay",
            "",
            joint_pareto_heavy_stage_replay_report_lines(joint_pareto_heavy_stage_replay_rows),
        ]
        heavy_path = output_dir / "reports" / "neutral_aero_residual_joint_pareto_heavy_report.md"
        heavy_path.write_text("\n".join(heavy_lines) + "\n", encoding="utf-8")

def alignment_sensitivity_report_lines(
    *,
    validation_rows: list[dict[str, Any]],
    stage_replay_rows: list[dict[str, Any]],
    primary_alignment_window_s: float,
    sensitivity_alignment_windows_s: tuple[float, ...],
) -> str:
    if not sensitivity_alignment_windows_s:
        return "- disabled"
    lines = [
        (
            "- policy: replay-only diagnostic for handoff-aligned starts; fitted coefficients, held-out split, "
            "and acceptance gates stay tied to the primary alignment window"
        ),
        f"- primary alignment window: `{primary_alignment_window_s:.3f}` s",
        "- replay CSV: `metrics/neutral_aero_residual_replay_sensitivity.csv`",
        "- regime-split CSV: `metrics/neutral_aero_residual_stage_replay_sensitivity.csv`",
    ]
    for window_s in sensitivity_alignment_windows_s:
        subset = [
            row
            for row in validation_rows
            if abs(finite_value(row.get("sensitivity_alignment_window_s")) - float(window_s)) <= 1.0e-12
        ]
        stage_subset = [
            row
            for row in stage_replay_rows
            if abs(finite_value(row.get("sensitivity_alignment_window_s")) - float(window_s)) <= 1.0e-12
        ]
        baseline_heldout = replay_summary(subset, "baseline_active", "heldout")
        candidate_heldout = replay_summary(subset, "coefficient_candidate", "heldout")
        lines.extend(
            [
                "",
                f"- sensitivity window `{window_s:.3f}` s held-out baseline -> candidate:",
                "  - "
                + ", ".join(
                    [
                        sensitivity_metric_pair("dx", baseline_heldout, candidate_heldout, "dx_mae_m", "m"),
                        sensitivity_metric_pair(
                            "altitude loss",
                            baseline_heldout,
                            candidate_heldout,
                            "altitude_loss_mae_m",
                            "m",
                        ),
                        sensitivity_metric_pair("sink", baseline_heldout, candidate_heldout, "sink_mae_m_s", "m/s"),
                        sensitivity_metric_pair(
                            "pitch",
                            baseline_heldout,
                            candidate_heldout,
                            "final_theta_mae_deg",
                            "deg",
                        ),
                        sensitivity_metric_pair("dy", baseline_heldout, candidate_heldout, "dy_mae_m", "m"),
                        sensitivity_metric_pair(
                            "roll",
                            baseline_heldout,
                            candidate_heldout,
                            "final_phi_mae_deg",
                            "deg",
                        ),
                        sensitivity_metric_pair(
                            "yaw",
                            baseline_heldout,
                            candidate_heldout,
                            "final_psi_mae_deg",
                            "deg",
                        ),
                    ]
                ),
                "  - regime split:",
                indent_report_lines(stage_replay_report_lines(stage_subset), prefix="    "),
            ]
        )
    return "\n".join(lines)


def sensitivity_metric_pair(
    label: str,
    baseline_summary: dict[str, float],
    candidate_summary: dict[str, float],
    key: str,
    unit: str,
) -> str:
    baseline = finite_value(baseline_summary.get(key))
    candidate = finite_value(candidate_summary.get(key))
    delta = candidate - baseline
    return f"{label} `{baseline:.4g}->{candidate:.4g}` {unit} (delta `{delta:+.4g}`)"


def indent_report_lines(text: str, *, prefix: str) -> str:
    return "\n".join(f"{prefix}{line}" if line else line for line in text.splitlines())


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


def lateral_ablation_report_lines(rows: list[dict[str, Any]]) -> str:
    heldout = [row for row in rows if row.get("split") == "heldout"]
    if not heldout:
        return "- no lateral ablation rows available"
    accepted = [row for row in heldout if bool(row.get("accepted", False))]
    lines = [f"- held-out ablations tested: `{len(heldout)}`"]
    if accepted:
        lines.append("- accepted held-out lateral ablations:")
        for row in accepted:
            lines.append(
                f"  - `{row.get('candidate_id', '')}` vs `{row.get('baseline_model_id', '')}` "
                f"coeff `{finite_value(row.get('fit_coefficient')):.6g}` "
                f"({row.get('fit_sign', '')}); dy `{finite_value(row.get('baseline_dy_mae_m')):.3f}` -> "
                f"`{finite_value(row.get('candidate_dy_mae_m')):.3f}` m, roll "
                f"`{finite_value(row.get('baseline_roll_mae_deg')):.2f}` -> "
                f"`{finite_value(row.get('candidate_roll_mae_deg')):.2f}` deg, yaw "
                f"`{finite_value(row.get('baseline_yaw_mae_deg')):.2f}` -> "
                f"`{finite_value(row.get('candidate_yaw_mae_deg')):.2f}` deg"
            )
    else:
        lines.append("- accepted held-out lateral ablations: none")
    best_by_dy = sorted(
        heldout,
        key=lambda row: finite_value(row.get("delta_dy_mae_m")) if math.isfinite(finite_value(row.get("delta_dy_mae_m"))) else float("inf"),
    )[:5]
    lines.append("- best held-out dy reductions, even if rejected:")
    for row in best_by_dy:
        lines.append(
            f"  - `{row.get('candidate_id', '')}` vs `{row.get('baseline_model_id', '')}` "
            f"coeff `{finite_value(row.get('fit_coefficient')):.6g}` "
            f"delta dy `{finite_value(row.get('delta_dy_mae_m')):.3f}` m, "
            f"delta roll `{finite_value(row.get('delta_roll_mae_deg')):.2f}` deg, "
            f"delta yaw `{finite_value(row.get('delta_yaw_mae_deg')):.2f}` deg, "
            f"reason `{row.get('acceptance_reason', '')}`"
        )
    return "\n".join(lines)


def compact_joint_sweep_report_lines(
    candidate_rows: list[dict[str, Any]],
    pareto_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
) -> str:
    if not candidate_rows:
        return "- compact joint sweep not run for this workflow"
    train_count = sum(1 for row in candidate_rows if row.get("split") == "train")
    heldout_count = sum(1 for row in candidate_rows if row.get("split") == "heldout")
    lines = [
        f"- train candidate rows: `{train_count}`",
        f"- held-out candidate rows: `{heldout_count}`",
        f"- pareto rows: `{len(pareto_rows)}`",
    ]
    if selected_rows:
        lines.append("- selected held-out candidates:")
        for row in selected_rows:
            lines.append(
                f"  - `{row.get('selection_class', '')}` `{row.get('candidate_id', '')}`: "
                f"score `{finite_value(row.get('score')):.3f}`, dx `{finite_value(row.get('dx_mae_m')):.3f}` m, "
                f"dy `{finite_value(row.get('dy_mae_m')):.3f}` m, altitude-loss "
                f"`{finite_value(row.get('altitude_loss_mae_m')):.3f}` m, sink "
                f"`{finite_value(row.get('sink_mae_m_s')):.3f}` m/s, roll "
                f"`{finite_value(row.get('roll_mae_deg')):.2f}` deg, pitch "
                f"`{finite_value(row.get('pitch_mae_deg')):.2f}` deg, yaw "
                f"`{finite_value(row.get('yaw_mae_deg')):.2f}` deg"
            )
    else:
        lines.append("- selected held-out candidates: none")
    return "\n".join(lines)


def joint_pareto_audit_report_lines(
    *,
    candidate_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    enabled: bool,
    alignment_window_s: float,
) -> str:
    if not bool(enabled):
        return "- disabled"
    if not candidate_rows:
        return f"- enabled at `{alignment_window_s:.3f}` s, but no held-out audit candidates were generated"
    accepted = [row for row in candidate_rows if bool(row.get("accepted", False))]
    lines = [
        f"- audit alignment window: `{alignment_window_s:.3f}` s",
        f"- candidate rows: `{len(candidate_rows)}`",
        f"- accepted rows: `{len(accepted)}`",
        f"- selected accepted Pareto rows: `{len(selected_rows)}`",
    ]
    if selected_rows:
        lines.append("- accepted Pareto candidates:")
        for row in selected_rows:
            lines.append(
                f"  - `{row.get('candidate_id', '')}` from `{row.get('longitudinal_source_id', '')}` + "
                f"`{row.get('lateral_source_id', '')}`: dx delta `{finite_value(row.get('delta_dx_mae_m')):.4f}` m, "
                f"dy delta `{finite_value(row.get('delta_dy_mae_m')):.4f}` m, pitch delta "
                f"`{finite_value(row.get('delta_pitch_mae_deg')):.3f}` deg, roll delta "
                f"`{finite_value(row.get('delta_roll_mae_deg')):.3f}` deg, yaw delta "
                f"`{finite_value(row.get('delta_yaw_mae_deg')):.3f}` deg"
            )
    else:
        best_rejected = sorted(
            candidate_rows,
            key=lambda row: finite_value(row.get("delta_lateral_score")),
        )[:5]
        lines.append("- accepted Pareto candidates: none")
        lines.append("- best lateral-score reductions, even if rejected:")
        for row in best_rejected:
            lines.append(
                f"  - `{row.get('candidate_id', '')}` reason `{row.get('acceptance_reason', '')}`, "
                f"lateral-score delta `{finite_value(row.get('delta_lateral_score')):.3f}`, "
                f"longitudinal-score delta `{finite_value(row.get('delta_longitudinal_score')):.3f}`"
            )
    return "\n".join(lines)


def joint_pareto_heavy_report_lines(
    *,
    config: dict[str, Any],
    selected_rows: list[dict[str, Any]],
    stage_replay_rows: list[dict[str, Any]],
) -> str:
    if str(config.get("profile", "")) != "heavy":
        return ""
    return "\n".join(
        [
            "- heavy candidate CSV: `metrics/neutral_aero_residual_joint_pareto_heavy_candidates.csv`",
            "- heavy selected CSV: `metrics/neutral_aero_residual_joint_pareto_heavy_selected.csv`",
            "- heavy stage replay CSV: `metrics/neutral_aero_residual_joint_pareto_heavy_stage_replay.csv`",
            f"- heavy selected row count: `{len(selected_rows)}`",
            f"- heavy stage replay row count: `{len(stage_replay_rows)}`",
        ]
    )


def joint_pareto_heavy_stage_replay_report_lines(stage_replay_rows: list[dict[str, Any]]) -> str:
    if not stage_replay_rows:
        return "- no heavy stage replay rows available"
    lines: list[str] = []
    matched_ids = sorted({str(row.get("matched_candidate_id", "")) for row in stage_replay_rows if str(row.get("matched_candidate_id", ""))})
    for matched_id in matched_ids:
        lines.append(f"- selected candidate `{matched_id}`:")
        subset = [row for row in stage_replay_rows if str(row.get("matched_candidate_id", "")) == matched_id]
        for role in ("matched_longitudinal_reference", "selected_candidate", "global_longitudinal_reference"):
            role_rows = [row for row in subset if str(row.get("model_role", "")) == role]
            if not role_rows:
                continue
            lines.append(f"  - `{role}`:")
            for regime in ("normal", "transition", "post_stall"):
                row = next((item for item in role_rows if str(item.get("report_regime", "")) == regime), None)
                if row is None:
                    continue
                lines.append(
                    "    - "
                    f"{regime}: n `{int(finite_value(row.get('sample_count'))) if math.isfinite(finite_value(row.get('sample_count'))) else 0}`, "
                    f"throws `{int(finite_value(row.get('throw_count'))) if math.isfinite(finite_value(row.get('throw_count'))) else 0}`, "
                    f"dx `{finite_value(row.get('dx_mae_m')):.4f}` m, "
                    f"dy `{finite_value(row.get('dy_mae_m')):.4f}` m, "
                    f"alt `{finite_value(row.get('altitude_loss_mae_m')):.4f}` m, "
                    f"sink `{finite_value(row.get('sink_rate_mae_m_s')):.4f}` m/s, "
                    f"roll `{finite_value(row.get('roll_mae_deg')):.3f}` deg, "
                    f"pitch `{finite_value(row.get('pitch_mae_deg')):.3f}` deg, "
                    f"yaw `{finite_value(row.get('yaw_mae_deg')):.3f}` deg"
                )
    return "\n".join(lines)


def lateral_launch_correlation_report_lines(rows: list[dict[str, Any]]) -> str:
    heldout_candidate = [
        row
        for row in rows
        if row.get("model_id") == "coefficient_candidate" and row.get("split") == "heldout"
    ]
    if not heldout_candidate:
        return "- no held-out launch-correlation rows available"
    strongest = sorted(
        heldout_candidate,
        key=lambda row: finite_value(row.get("abs_pearson_r")) if math.isfinite(finite_value(row.get("abs_pearson_r"))) else -1.0,
        reverse=True,
    )[:8]
    lines = ["- strongest held-out correlations for the accepted longitudinal candidate:"]
    for row in strongest:
        lines.append(
            f"  - `{row.get('residual_metric', '')}` residual vs `{row.get('launch_variable', '')}`: "
            f"r `{finite_value(row.get('pearson_r')):.3f}`, slope `{finite_value(row.get('slope')):.3f}`, "
            f"n `{int(finite_value(row.get('sample_count'))) if math.isfinite(finite_value(row.get('sample_count'))) else 0}`"
        )
    max_abs = max((finite_value(row.get("abs_pearson_r")) for row in heldout_candidate), default=float("nan"))
    if math.isfinite(max_abs) and max_abs >= 0.6:
        lines.append("- interpretation: at least one held-out lateral residual has strong launch-condition correlation; down-weighting contaminated launches is likely safer than promoting stronger lateral aerodynamics.")
    elif math.isfinite(max_abs) and max_abs >= 0.35:
        lines.append("- interpretation: launch-condition correlation is moderate; use lateral coefficient promotion only if one-term held-out ablations also improve roll and yaw.")
    else:
        lines.append("- interpretation: launch-condition correlation is weak in the held-out split; remaining lateral mismatch may need better lateral dynamics or more targeted lateral data.")
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
