"""Control-surface effectiveness diagnostics from real-flight pulse ladders.

This analysis freezes the active neutral calibrated dry-air model, inventories
the sustained aileron/elevator/rudder deflection throws, replays each usable
throw from its measured launch state, and reports launch-level effectiveness
metrics. It is evidence for surface response and replay alignment only; it does
not promote broad aerodynamic SysID parameters.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
import warnings
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
INNER_LOOP = ROOT / "03_Control" / "02_Inner_Loop"
PRIMITIVES = ROOT / "03_Control" / "03_Primitives"
for path in (INNER_LOOP, PRIMITIVES):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from command_contract import normalised_command_to_surface_rad  # noqa: E402
from flight_dynamics import adapt_glider, state_derivative  # noqa: E402
from glider import build_nausicaa_glider  # noqa: E402
import run_real_glider_calibration_prep as prep  # noqa: E402
from A_model_parameters import neutral_dry_air_calibration as active_calibration  # noqa: E402


DEFAULT_INPUT_ROOT = ROOT / "04_Flight_Test" / "05_Results" / "cal"
DEFAULT_OUTPUT_ROOT = ROOT / "03_Control" / "05_Results" / "control_surface_effectiveness"
DEFAULT_RUN_LABEL = "control_surface_effectiveness_v3_0"
DEFAULT_DATASET_ROOTS = ("pa30", "pe30", "pr30")
DEFAULT_RESPONSE_WINDOW_S = 0.65
DEFAULT_MIN_RESPONSE_WINDOW_S = 0.25
DEFAULT_REPLAY_DT_S = 0.005
DEFAULT_HELDOUT_SEED = 606

RELAXED_U_MIN_M_S = 3.0
RELAXED_U_MAX_M_S = 8.0
RELAXED_ABS_V_MAX_M_S = 1.5
RELAXED_ABS_W_MAX_M_S = 0.9
MAX_RESPONSE_SPIKE_FRACTION = 0.20
DEEP_POST_STALL_ALPHA_DEG = 30.0
COMMAND_MATCH_TOL = 1e-6
COMMAND_TIMING_TOL_S = 0.10
LAUNCH_CONFIDENCE_HIGH_THRESHOLD = 0.75
LAUNCH_CONFIDENCE_MEDIUM_THRESHOLD = 0.55
LAUNCH_CONFIDENCE_MIN_WEIGHT = 0.25
LAUNCH_CONFIDENCE_EXPONENT = 1.5
LAUNCH_CONFIDENCE_ROLL_ABS_MAX_DEG = 20.0
LAUNCH_CONFIDENCE_YAW_ABS_MAX_DEG = 20.0
LAUNCH_CONFIDENCE_V_ABS_MAX_M_S = 1.5
LAUNCH_CONFIDENCE_P_ABS_MAX_RAD_S = 1.2
LAUNCH_CONFIDENCE_R_ABS_MAX_RAD_S = 1.8
DERIVATIVE_FIT_MIN_QBAR_PA = 2.0
DERIVATIVE_FIT_MIN_SURFACE_RAD = 0.02
DERIVATIVE_FIT_COEFF_ABS_BOUND = 12.0
SURFACE_AERO_NORMAL_ALPHA_MAX_DEG = 12.0
SURFACE_AERO_TRANSITION_ALPHA_MAX_DEG = 22.0
SURFACE_AERO_ALPHA_REGIMES = ("normal", "transition", "post_stall")

COMMAND_AXIS_TO_SURFACE = {
    "delta_a": "aileron",
    "delta_e": "elevator",
    "delta_r": "rudder",
}
SURFACE_TO_COMMAND_AXIS = {value: key for key, value in COMMAND_AXIS_TO_SURFACE.items()}
COMMAND_AXIS_INDEX = {"delta_a": 0, "delta_e": 1, "delta_r": 2}

PRIMARY_METRICS_BY_SURFACE = {
    "aileron": ("peak_p_rad_s", "p_impulse_rad", "phi_change_deg", "yaw_coupling_psi_change_deg", "response_dy_m"),
    "elevator": ("peak_q_rad_s", "q_impulse_rad", "theta_change_deg", "response_altitude_loss_m", "max_abs_alpha_deg"),
    "rudder": ("peak_r_rad_s", "r_impulse_rad", "psi_change_deg", "roll_coupling_phi_change_deg", "response_dy_m"),
}
ALL_RESPONSE_METRICS = (
    "peak_p_rad_s",
    "p_impulse_rad",
    "phi_change_deg",
    "peak_q_rad_s",
    "q_impulse_rad",
    "theta_change_deg",
    "peak_r_rad_s",
    "r_impulse_rad",
    "psi_change_deg",
    "roll_coupling_phi_change_deg",
    "yaw_coupling_psi_change_deg",
    "response_dy_m",
    "response_dx_m",
    "response_altitude_loss_m",
    "max_abs_alpha_deg",
    "alpha_gt_20_s",
    "alpha_gt_30_s",
    "speed_mean_m_s",
    "response_delay_p_s",
    "response_delay_q_s",
    "response_delay_r_s",
)

INVENTORY_FIELDS = [
    "dataset_root",
    "session_label",
    "trial_id",
    "throw_id",
    "throw_dir",
    "case_id",
    "case_storage_id",
    "surface_axis",
    "command_axis",
    "command_value",
    "command_abs",
    "command_lattice_percent",
    "command_start_s",
    "command_duration_s",
    "effective_flight_duration_s",
    "usable_window_start_s",
    "usable_window_end_s",
    "response_window_s",
    "valid_throw",
    "state_sample_count",
    "command_schedule_status",
    "command_schedule_notes",
    "vicon_validity_flag",
    "tracking_quality_flag",
    "launch_quality_flag",
    "termination_reason",
    "termination_group",
    "x0_m",
    "y0_m",
    "z0_m",
    "u0_m_s",
    "v0_m_s",
    "w0_m_s",
    "speed0_m_s",
    "phi0_deg",
    "theta0_deg",
    "psi0_deg",
    "p0_rad_s",
    "q0_rad_s",
    "r0_rad_s",
    "mean_rate_confidence",
    "min_rate_confidence",
    "launch_confidence_score",
    "launch_confidence_label",
    "launch_confidence_weight",
    "launch_lateral_contamination_score",
    "launch_confidence_reasons",
    "response_spike_fraction",
    "response_body_rate_limited_fraction",
    "max_abs_alpha_deg",
    "deep_post_stall_flag",
    "filter_status",
    "filter_reasons",
    "split",
    "notes",
]

FILTER_SUMMARY_FIELDS = [
    "group",
    "surface_axis",
    "command_abs",
    "total_throw_count",
    "kept_throw_count",
    "filtered_throw_count",
    "train_throw_count",
    "heldout_throw_count",
    "invalid_throw_count",
    "deep_post_stall_count",
    "early_contact_count",
    "filter_reason_counts",
]

LAUNCH_CONFIDENCE_SUMMARY_FIELDS = [
    "group",
    "split",
    "surface_axis",
    "command_abs",
    "replay_count",
    "high_confidence_count",
    "medium_confidence_count",
    "low_confidence_count",
    "mean_launch_confidence_score",
    "mean_launch_lateral_contamination_score",
    "primary_metric",
    "all_abs_antisym_residual",
    "high_confidence_abs_antisym_residual",
    "confidence_weighted_abs_antisym_residual",
    "high_minus_all_abs_residual",
    "weighted_minus_all_abs_residual",
    "replay_dx_mae_m",
    "replay_dy_mae_m",
    "replay_altitude_loss_mae_m",
]

REPLAY_FIELDS = [
    "candidate_id",
    "split",
    "dataset_root",
    "session_label",
    "trial_id",
    "throw_id",
    "case_id",
    "surface_axis",
    "command_axis",
    "command_value",
    "command_abs",
    "launch_confidence_score",
    "launch_confidence_label",
    "launch_confidence_weight",
    "launch_lateral_contamination_score",
    "launch_confidence_reasons",
    "replay_status",
    "replay_policy",
    "replay_dt_s",
    "replay_command_source",
    "replay_command_onset_delay_s",
    "duration_s",
    "response_window_start_s",
    "response_window_end_s",
    "actual_dx_m",
    "sim_dx_m",
    "dx_residual_actual_minus_sim_m",
    "actual_dy_m",
    "sim_dy_m",
    "dy_residual_actual_minus_sim_m",
    "actual_altitude_loss_m",
    "sim_altitude_loss_m",
    "altitude_loss_residual_actual_minus_sim_m",
    "actual_final_phi_deg",
    "sim_final_phi_deg",
    "final_phi_residual_actual_minus_sim_deg",
    "actual_final_theta_deg",
    "sim_final_theta_deg",
    "final_theta_residual_actual_minus_sim_deg",
    "actual_final_psi_deg",
    "sim_final_psi_deg",
    "final_psi_residual_actual_minus_sim_deg",
]
for metric_name in ALL_RESPONSE_METRICS:
    REPLAY_FIELDS.extend([f"actual_{metric_name}", f"sim_{metric_name}", f"{metric_name}_residual_actual_minus_sim"])

EFFECTIVENESS_FIELDS = [
    "split",
    "surface_axis",
    "command_abs",
    "metric",
    "positive_count",
    "negative_count",
    "actual_positive_mean",
    "actual_negative_mean",
    "actual_antisymmetric_response",
    "actual_symmetric_response",
    "sim_positive_mean",
    "sim_negative_mean",
    "sim_antisymmetric_response",
    "sim_symmetric_response",
    "antisymmetric_residual_actual_minus_sim",
    "symmetric_residual_actual_minus_sim",
]

OPTIONAL_CANDIDATE_FIELDS = [
    "candidate_id",
    "status",
    "promoted",
    "delta_a_effectiveness_scale",
    "delta_e_effectiveness_scale",
    "delta_r_effectiveness_scale",
    "delta_a_neutral_bias_rad",
    "delta_e_neutral_bias_rad",
    "delta_r_neutral_bias_rad",
    "aileron_left_right_asymmetry",
    "actuator_time_constant_scale",
    "command_delay_offset_s",
    "evidence_summary",
    "claim_boundary",
]

OPTIONAL_HELDOUT_FIELDS = [
    "candidate_id",
    "surface_axis",
    "metric",
    "heldout_count",
    "baseline_abs_error",
    "candidate_abs_error",
    "improved",
    "promotion_gate_status",
]

OPTIONAL_AERO_COUPLING_FIT_FIELDS = [
    "candidate_id",
    "parameter",
    "surface_axis",
    "command_axis",
    "moment_axis",
    "fit_role",
    "coefficient_per_rad",
    "bounded_coefficient_per_rad",
    "train_sample_count",
    "heldout_sample_count",
    "train_baseline_mae_coeff",
    "train_candidate_mae_coeff",
    "heldout_baseline_mae_coeff",
    "heldout_candidate_mae_coeff",
    "heldout_improved",
    "mean_launch_confidence_weight",
    "physical_sign_status",
    "promotion_gate_status",
    "claim_boundary",
]

REPLAY_ERROR_SUMMARY_FIELDS = [
    "candidate_id",
    "split",
    "surface_axis",
    "replay_count",
    "dx_mae_m",
    "dy_mae_m",
    "altitude_loss_mae_m",
    "final_phi_mae_deg",
    "final_theta_mae_deg",
    "final_psi_mae_deg",
    "primary_antisym_residual",
    "claim_boundary",
]

REGIME_LADDER_ERROR_FIELDS = [
    "candidate_id",
    "split",
    "surface_axis",
    "alpha_regime",
    "command_abs",
    "replay_count",
    "positive_count",
    "negative_count",
    "mean_actual_max_abs_alpha_deg",
    "max_actual_max_abs_alpha_deg",
    "mean_actual_alpha_gt_20_s",
    "mean_actual_alpha_gt_30_s",
    "dx_mae_m",
    "dy_mae_m",
    "altitude_loss_mae_m",
    "final_phi_mae_deg",
    "final_theta_mae_deg",
    "final_psi_mae_deg",
    "primary_metric",
    "actual_antisymmetric_response",
    "sim_antisymmetric_response",
    "primary_antisym_residual",
    "claim_boundary",
]

SURFACE_AERO_COUPLING_SPECS = (
    {
        "parameter": "CY_delta_a_residual",
        "surface_axis": "aileron",
        "command_axis": "delta_a",
        "state_index": 12,
        "fit_type": "side_force",
        "moment_axis": "side_force_y",
        "fit_role": "aileron_side_force",
        "reference": "area",
    },
    {
        "parameter": "Cl_delta_a_residual",
        "surface_axis": "aileron",
        "command_axis": "delta_a",
        "state_index": 12,
        "fit_type": "moment",
        "moment_index": 0,
        "moment_axis": "roll",
        "fit_role": "aileron_roll_effectiveness",
        "reference": "span",
    },
    {
        "parameter": "Cn_delta_a_residual",
        "surface_axis": "aileron",
        "command_axis": "delta_a",
        "state_index": 12,
        "fit_type": "moment",
        "moment_index": 2,
        "moment_axis": "yaw",
        "fit_role": "aileron_yaw_coupling",
        "reference": "span",
    },
    {
        "parameter": "Cm_delta_e_residual",
        "surface_axis": "elevator",
        "command_axis": "delta_e",
        "state_index": 13,
        "fit_type": "moment",
        "moment_index": 1,
        "moment_axis": "pitch",
        "fit_role": "elevator_pitch_effectiveness",
        "reference": "chord",
    },
    {
        "parameter": "CY_delta_r_residual",
        "surface_axis": "rudder",
        "command_axis": "delta_r",
        "state_index": 14,
        "fit_type": "side_force",
        "moment_axis": "side_force_y",
        "fit_role": "rudder_side_force",
        "reference": "area",
    },
    {
        "parameter": "Cn_delta_r_residual",
        "surface_axis": "rudder",
        "command_axis": "delta_r",
        "state_index": 14,
        "fit_type": "moment",
        "moment_index": 2,
        "moment_axis": "yaw",
        "fit_role": "rudder_yaw_effectiveness",
        "reference": "span",
    },
    {
        "parameter": "Cl_delta_r_residual",
        "surface_axis": "rudder",
        "command_axis": "delta_r",
        "state_index": 14,
        "fit_type": "moment",
        "moment_index": 0,
        "moment_axis": "roll",
        "fit_role": "rudder_roll_coupling",
        "reference": "span",
    },
)

SURFACE_AERO_COUPLING_CANDIDATE_FAMILIES = {
    "C0_frozen_neutral": (),
    "C1_primary_moment_derivatives": (
        "Cl_delta_a_residual",
        "Cm_delta_e_residual",
        "Cn_delta_r_residual",
    ),
    "C2_c1_plus_aileron_adverse_yaw": (
        "Cl_delta_a_residual",
        "Cm_delta_e_residual",
        "Cn_delta_r_residual",
        "Cn_delta_a_residual",
    ),
    "C3_c1_plus_rudder_roll": (
        "Cl_delta_a_residual",
        "Cm_delta_e_residual",
        "Cn_delta_r_residual",
        "Cl_delta_r_residual",
    ),
    "C4_c1_plus_surface_side_force": (
        "Cl_delta_a_residual",
        "Cm_delta_e_residual",
        "Cn_delta_r_residual",
        "CY_delta_a_residual",
        "CY_delta_r_residual",
    ),
    "C5_c2_plus_surface_side_force": (
        "Cl_delta_a_residual",
        "Cm_delta_e_residual",
        "Cn_delta_r_residual",
        "Cn_delta_a_residual",
        "CY_delta_a_residual",
        "CY_delta_r_residual",
    ),
    "C6_alpha_regime_primary_derivatives": (
        "Cl_delta_a_residual@normal",
        "Cl_delta_a_residual@transition",
        "Cl_delta_a_residual@post_stall",
        "Cm_delta_e_residual@normal",
        "Cm_delta_e_residual@transition",
        "Cm_delta_e_residual@post_stall",
        "Cn_delta_r_residual@normal",
        "Cn_delta_r_residual@transition",
    ),
    "C7_c6_plus_alpha_regime_aileron_yaw": (
        "Cl_delta_a_residual@normal",
        "Cl_delta_a_residual@transition",
        "Cl_delta_a_residual@post_stall",
        "Cm_delta_e_residual@normal",
        "Cm_delta_e_residual@transition",
        "Cm_delta_e_residual@post_stall",
        "Cn_delta_r_residual@normal",
        "Cn_delta_r_residual@transition",
        "Cn_delta_a_residual@normal",
        "Cn_delta_a_residual@transition",
        "Cn_delta_a_residual@post_stall",
    ),
    "C8_c7_plus_alpha_regime_aileron_side_force": (
        "Cl_delta_a_residual@normal",
        "Cl_delta_a_residual@transition",
        "Cl_delta_a_residual@post_stall",
        "Cm_delta_e_residual@normal",
        "Cm_delta_e_residual@transition",
        "Cm_delta_e_residual@post_stall",
        "Cn_delta_r_residual@normal",
        "Cn_delta_r_residual@transition",
        "Cn_delta_a_residual@normal",
        "Cn_delta_a_residual@transition",
        "Cn_delta_a_residual@post_stall",
        "CY_delta_a_residual@normal",
        "CY_delta_a_residual@transition",
        "CY_delta_a_residual@post_stall",
    ),
}

ALPHA_REGIME_SCHEDULED_PARAMETERS = {
    "Cl_delta_a_residual",
    "Cn_delta_a_residual",
    "Cm_delta_e_residual",
    "Cn_delta_r_residual",
    "CY_delta_a_residual",
}


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = run_study(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        run_label=args.run_label,
        dataset_roots=tuple(args.dataset_root),
        replay_dt_s=float(args.replay_dt_s),
        response_window_s=float(args.response_window_s),
        min_response_window_s=float(args.min_response_window_s),
        heldout_seed=int(args.heldout_seed),
        run_surface_fit_diagnostics=bool(args.surface_fit_diagnostics),
    )
    print(output_dir.as_posix())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run conservative real-flight control-surface effectiveness diagnostics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT.as_posix(), help="Root containing pa30/pe30/pr30 data.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT.as_posix(), help="Root for study outputs.")
    parser.add_argument("--run-label", default=DEFAULT_RUN_LABEL, help="Output subdirectory label.")
    parser.add_argument(
        "--dataset-root",
        action="append",
        default=list(DEFAULT_DATASET_ROOTS),
        help="Dataset root name below input root. Repeat to add roots.",
    )
    parser.add_argument("--replay-dt-s", type=float, default=DEFAULT_REPLAY_DT_S, help="Fixed RK4 replay step.")
    parser.add_argument("--response-window-s", type=float, default=DEFAULT_RESPONSE_WINDOW_S, help="Nominal response metric window after command onset.")
    parser.add_argument("--min-response-window-s", type=float, default=DEFAULT_MIN_RESPONSE_WINDOW_S, help="Minimum usable response window after command onset.")
    parser.add_argument("--heldout-seed", type=int, default=DEFAULT_HELDOUT_SEED, help="Launch-level held-out split seed.")
    parser.add_argument(
        "--surface-fit-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optionally write legacy S1/S2 metric-space surface-scale estimates; disabled by default.",
    )
    return parser


def run_study(
    *,
    input_root: Path,
    output_root: Path,
    run_label: str,
    dataset_roots: tuple[str, ...],
    replay_dt_s: float,
    response_window_s: float,
    min_response_window_s: float,
    heldout_seed: int,
    run_surface_fit_diagnostics: bool,
) -> Path:
    output_dir = output_root / run_label
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    inventory_rows = load_inventory_rows(
        input_root,
        dataset_roots=dataset_roots,
        response_window_s=response_window_s,
        min_response_window_s=min_response_window_s,
    )
    assign_launch_level_split(inventory_rows, heldout_seed=heldout_seed)
    kept_rows = [row for row in inventory_rows if row.get("filter_status") == "kept"]
    replay_rows = replay_kept_rows(kept_rows, replay_dt_s=replay_dt_s, candidate_id="C0_frozen_neutral")
    effectiveness_rows = summarize_effectiveness(replay_rows)
    launch_confidence_rows = launch_confidence_summary(replay_rows, effectiveness_rows)
    aero_coupling_rows = optional_surface_aero_coupling_fit(replay_rows)
    aero_coupling_coefficients = surface_aero_coupling_coefficients(aero_coupling_rows)
    candidate_coefficients = surface_aero_coupling_candidate_coefficients(aero_coupling_coefficients)
    replay_by_candidate = {"C0_frozen_neutral": replay_rows}
    effectiveness_by_candidate = {"C0_frozen_neutral": effectiveness_rows}
    for candidate_id, coeffs in candidate_coefficients.items():
        if candidate_id == "C0_frozen_neutral":
            continue
        candidate_replay_rows = replay_kept_rows(
            kept_rows,
            replay_dt_s=replay_dt_s,
            candidate_id=candidate_id,
            replay_policy=f"{candidate_id}_launch_confidence_weighted_surface_aero_coupling_replay",
            derivative_coeffs=coeffs,
        )
        replay_by_candidate[candidate_id] = candidate_replay_rows
        effectiveness_by_candidate[candidate_id] = summarize_effectiveness(candidate_replay_rows)
    candidate_sweep_replay_rows = [
        row
        for candidate_id, candidate_rows in replay_by_candidate.items()
        if candidate_id != "C0_frozen_neutral"
        for row in candidate_rows
    ]
    replay_error_summary_rows = replay_error_summary(replay_by_candidate, effectiveness_by_candidate)
    regime_ladder_error_rows = regime_ladder_error_summary(replay_by_candidate)
    symmetric_rows = [
        {
            **row,
            "actual_antisymmetric_response": "",
            "sim_antisymmetric_response": "",
            "antisymmetric_residual_actual_minus_sim": "",
        }
        for row in effectiveness_rows
    ]
    optional_candidates, optional_heldout = optional_surface_fit_diagnostics(
        effectiveness_rows,
        run_diagnostics=run_surface_fit_diagnostics,
    )
    append_surface_aero_coupling_candidate(optional_candidates, optional_heldout, aero_coupling_rows)
    filtering_summary_rows = filtering_summary(inventory_rows)

    write_csv(output_dir / "control_surface_inventory.csv", inventory_rows, INVENTORY_FIELDS)
    write_csv(output_dir / "control_surface_filtering_summary.csv", filtering_summary_rows, FILTER_SUMMARY_FIELDS)
    write_csv(output_dir / "control_surface_replay_metrics.csv", replay_rows, REPLAY_FIELDS)
    write_csv(output_dir / "control_surface_replay_metrics_candidate_sweep.csv", candidate_sweep_replay_rows, REPLAY_FIELDS)
    write_csv(output_dir / "control_surface_replay_error_summary.csv", replay_error_summary_rows, REPLAY_ERROR_SUMMARY_FIELDS)
    write_csv(output_dir / "control_surface_regime_ladder_error_summary.csv", regime_ladder_error_rows, REGIME_LADDER_ERROR_FIELDS)
    write_csv(output_dir / "control_surface_effectiveness_summary.csv", effectiveness_rows, EFFECTIVENESS_FIELDS)
    write_csv(output_dir / "control_surface_launch_confidence_summary.csv", launch_confidence_rows, LAUNCH_CONFIDENCE_SUMMARY_FIELDS)
    write_csv(
        output_dir / "control_surface_symmetric_contamination_summary.csv",
        symmetric_rows,
        EFFECTIVENESS_FIELDS,
    )
    write_csv(output_dir / "optional_surface_fit_candidates.csv", optional_candidates, OPTIONAL_CANDIDATE_FIELDS)
    write_csv(output_dir / "optional_surface_fit_heldout_summary.csv", optional_heldout, OPTIONAL_HELDOUT_FIELDS)
    write_csv(output_dir / "optional_surface_aero_coupling_fit.csv", aero_coupling_rows, OPTIONAL_AERO_COUPLING_FIT_FIELDS)

    figures = write_figures(output_dir / "figures", inventory_rows, replay_rows, effectiveness_rows, launch_confidence_rows, optional_heldout)
    manifest = build_manifest(
        input_root=input_root,
        dataset_roots=dataset_roots,
        output_dir=output_dir,
        inventory_rows=inventory_rows,
        replay_rows=replay_rows,
        candidate_sweep_replay_rows=candidate_sweep_replay_rows,
        replay_error_summary_rows=replay_error_summary_rows,
        regime_ladder_error_rows=regime_ladder_error_rows,
        effectiveness_rows=effectiveness_rows,
        launch_confidence_rows=launch_confidence_rows,
        aero_coupling_rows=aero_coupling_rows,
        filtering_summary_rows=filtering_summary_rows,
        optional_candidates=optional_candidates,
        optional_heldout=optional_heldout,
        figures=figures,
        replay_dt_s=replay_dt_s,
        response_window_s=response_window_s,
        min_response_window_s=min_response_window_s,
        heldout_seed=heldout_seed,
    )
    (output_dir / "control_surface_effectiveness_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(
        output_dir / "control_surface_effectiveness_report.md",
        inventory_rows=inventory_rows,
        replay_rows=replay_rows,
        candidate_sweep_replay_rows=candidate_sweep_replay_rows,
        replay_error_summary_rows=replay_error_summary_rows,
        regime_ladder_error_rows=regime_ladder_error_rows,
        effectiveness_rows=effectiveness_rows,
        launch_confidence_rows=launch_confidence_rows,
        aero_coupling_rows=aero_coupling_rows,
        optional_candidates=optional_candidates,
        optional_heldout=optional_heldout,
        filtering_summary_rows=filtering_summary_rows,
        manifest=manifest,
    )
    return output_dir


def load_inventory_rows(
    input_root: Path,
    *,
    dataset_roots: tuple[str, ...],
    response_window_s: float,
    min_response_window_s: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_name in dataset_roots:
        dataset_root = input_root / dataset_name
        if not dataset_root.exists():
            continue
        for manifest_path in sorted(dataset_root.rglob("manifests/glider_calibration_throw_manifest.json")):
            throw_dir = manifest_path.parents[1]
            row = inventory_row_from_throw(
                dataset_root,
                throw_dir,
                response_window_s=response_window_s,
                min_response_window_s=min_response_window_s,
            )
            rows.append(row)
    return sorted(
        rows,
        key=lambda item: (
            str(item.get("dataset_root", "")),
            str(item.get("session_label", "")),
            str(item.get("case_storage_id", "")),
            str(item.get("throw_id", "")),
        ),
    )


def inventory_row_from_throw(
    dataset_root: Path,
    throw_dir: Path,
    *,
    response_window_s: float = DEFAULT_RESPONSE_WINDOW_S,
    min_response_window_s: float = DEFAULT_MIN_RESPONSE_WINDOW_S,
) -> dict[str, Any]:
    manifest = load_json(throw_dir / "manifests" / "glider_calibration_throw_manifest.json")
    summary = load_json(throw_dir / "manifests" / "glider_calibration_throw_summary.json")
    case = manifest.get("calibration_case", {})
    config = manifest.get("config", {})
    state_rows = read_csv(throw_dir / "metrics" / "state_samples.csv")
    command_rows = read_csv(throw_dir / "metrics" / "command_schedule.csv")

    rel_parts = relative_parts(throw_dir, dataset_root)
    session_label = rel_parts[0] if rel_parts else ""
    case_storage_id = str(manifest.get("case_storage_id", summary.get("case_storage_id", rel_parts[1] if len(rel_parts) > 1 else "")))
    throw_id = throw_dir.name
    trial_id = "/".join(rel_parts)
    command_axis = str(case.get("command_axis", summary.get("command_axis", "")))
    command_value = to_float(case.get("command_value", summary.get("command_value", float("nan"))))
    command_start_s = to_float(case.get("pulse_start_s", summary.get("pulse_start_s", float("nan"))))
    command_duration_s = to_float(case.get("pulse_duration_s", summary.get("pulse_duration_s", float("nan"))))
    surface_axis = COMMAND_AXIS_TO_SURFACE.get(command_axis, command_axis or "unknown")

    first = state_rows[0] if state_rows else {}
    last = state_rows[-1] if state_rows else {}
    t_first = to_float(first.get("t_s", 0.0), 0.0)
    t_last = to_float(last.get("t_s", t_first), t_first)
    duration_s = max(0.0, t_last - t_first)
    response_start_s = command_start_s if math.isfinite(command_start_s) else float("nan")
    response_end_s = min(duration_s, response_start_s + float(response_window_s)) if math.isfinite(response_start_s) else float("nan")
    response_duration_s = response_end_s - response_start_s if math.isfinite(response_end_s) else float("nan")
    x0, y0, z0 = to_float(first.get("x_w")), to_float(first.get("y_w")), to_float(first.get("z_w"))
    phi0, theta0, psi0 = to_float(first.get("phi")), to_float(first.get("theta")), to_float(first.get("psi"))
    u0, v0, w0 = to_float(first.get("u")), to_float(first.get("v")), to_float(first.get("w"))
    p0, q0, r0 = to_float(first.get("p")), to_float(first.get("q")), to_float(first.get("r"))
    speed0 = math.sqrt(u0 * u0 + v0 * v0 + w0 * w0) if all_finite(u0, v0, w0) else to_float(summary.get("launch_speed_m_s"))

    response_rows = rows_in_window(state_rows, response_start_s, response_end_s)
    confidence = [to_float(row.get("estimator_rate_confidence")) for row in state_rows]
    response_spikes = [bool_value(row.get("estimator_spike_rejected")) for row in response_rows]
    response_limited = [bool_value(row.get("estimator_body_rate_limited")) for row in response_rows]
    alpha_values = [alpha_deg_from_row(row) for row in state_rows]
    max_abs_alpha_deg = safe_max_abs(alpha_values)
    command_status, command_notes = command_schedule_audit(command_rows, command_axis, command_value, command_start_s)
    termination_reason = str(summary.get("termination_reason", ""))
    termination_group = termination_group_from_reason(termination_reason)
    launch_quality_flag = launch_quality_from_state(phi0, theta0, psi0, u0, v0, w0, p0, q0, r0)

    row = {
        "_throw_dir": throw_dir.as_posix(),
        "dataset_root": dataset_root.as_posix(),
        "session_label": session_label,
        "trial_id": trial_id,
        "throw_id": throw_id,
        "throw_dir": throw_dir.as_posix(),
        "case_id": summary.get("case_id", case.get("case_id", "")),
        "case_storage_id": case_storage_id,
        "surface_axis": surface_axis,
        "command_axis": command_axis,
        "command_value": command_value,
        "command_abs": abs(command_value) if math.isfinite(command_value) else float("nan"),
        "command_lattice_percent": command_value * 100.0 if math.isfinite(command_value) else float("nan"),
        "command_start_s": command_start_s,
        "command_duration_s": command_duration_s,
        "effective_flight_duration_s": duration_s,
        "usable_window_start_s": response_start_s,
        "usable_window_end_s": response_end_s,
        "response_window_s": response_duration_s,
        "valid_throw": bool(summary.get("valid_throw", False)),
        "state_sample_count": int(len(state_rows)),
        "command_schedule_status": command_status,
        "command_schedule_notes": command_notes,
        "vicon_validity_flag": "valid_state_samples" if state_rows else "missing_state_samples",
        "tracking_quality_flag": tracking_quality_flag(confidence, response_spikes),
        "launch_quality_flag": launch_quality_flag,
        "termination_reason": termination_reason,
        "termination_group": termination_group,
        "x0_m": x0,
        "y0_m": y0,
        "z0_m": z0,
        "u0_m_s": u0,
        "v0_m_s": v0,
        "w0_m_s": w0,
        "speed0_m_s": speed0,
        "phi0_deg": math.degrees(phi0) if math.isfinite(phi0) else float("nan"),
        "theta0_deg": math.degrees(theta0) if math.isfinite(theta0) else float("nan"),
        "psi0_deg": math.degrees(psi0) if math.isfinite(psi0) else float("nan"),
        "p0_rad_s": p0,
        "q0_rad_s": q0,
        "r0_rad_s": r0,
        "mean_rate_confidence": safe_mean(confidence),
        "min_rate_confidence": safe_min(confidence),
        "launch_confidence_score": float("nan"),
        "launch_confidence_label": "unknown",
        "launch_confidence_weight": 0.0,
        "launch_lateral_contamination_score": float("nan"),
        "launch_confidence_reasons": "",
        "response_spike_fraction": ratio(sum(response_spikes), len(response_spikes)),
        "response_body_rate_limited_fraction": ratio(sum(response_limited), len(response_limited)),
        "max_abs_alpha_deg": max_abs_alpha_deg,
        "deep_post_stall_flag": bool(math.isfinite(max_abs_alpha_deg) and max_abs_alpha_deg >= DEEP_POST_STALL_ALPHA_DEG),
        "split": "excluded",
        "notes": inventory_notes(summary, config),
    }
    row.update(launch_confidence_from_inventory_row(row))
    kept, reasons = filter_inventory_row(row, min_response_window_s=min_response_window_s)
    row["filter_status"] = "kept" if kept else "filtered"
    row["filter_reasons"] = ";".join(reasons)
    if not kept:
        row["launch_confidence_weight"] = 0.0
    return row


def filter_inventory_row(row: dict[str, Any], *, min_response_window_s: float = DEFAULT_MIN_RESPONSE_WINDOW_S) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if row.get("valid_throw") is not True:
        reasons.append("invalid_or_cancelled_throw")
    if int(row.get("state_sample_count", 0) or 0) < 3:
        reasons.append("missing_or_short_state_samples")
    if str(row.get("command_axis", "")) not in COMMAND_AXIS_TO_SURFACE:
        reasons.append("unknown_surface_axis")
    command_value = to_float(row.get("command_value"))
    if not math.isfinite(command_value) or abs(command_value) < COMMAND_MATCH_TOL:
        reasons.append("not_nonzero_deflection")
    elif not is_lattice_20_percent(command_value):
        reasons.append("not_20_percent_lattice")
    if row.get("command_schedule_status") != "ok":
        reasons.append(str(row.get("command_schedule_status", "command_schedule_problem")))
    if not math.isfinite(to_float(row.get("response_window_s"))) or to_float(row.get("response_window_s")) < min_response_window_s:
        reasons.append("response_window_too_short")
    if early_contact_before_response(row, min_response_window_s):
        reasons.append("floor_or_wall_before_response_window")
    u0, v0, w0 = to_float(row.get("u0_m_s")), to_float(row.get("v0_m_s")), to_float(row.get("w0_m_s"))
    if not all_finite(u0, v0, w0):
        reasons.append("nonfinite_launch_velocity")
    elif (
        u0 < RELAXED_U_MIN_M_S
        or u0 > RELAXED_U_MAX_M_S
        or abs(v0) > RELAXED_ABS_V_MAX_M_S
        or abs(w0) > RELAXED_ABS_W_MAX_M_S
    ):
        reasons.append("outside_relaxed_replay_start_velocity_gate")
    spike_fraction = to_float(row.get("response_spike_fraction"))
    if math.isfinite(spike_fraction) and spike_fraction > MAX_RESPONSE_SPIKE_FRACTION:
        reasons.append("response_marker_jump_or_spike_fraction_high")
    return not reasons, reasons


def assign_launch_level_split(rows: list[dict[str, Any]], *, heldout_seed: int) -> None:
    for row in rows:
        row["split"] = "excluded"
    groups: dict[tuple[str, float], dict[int, list[int]]] = {}
    for index, row in enumerate(rows):
        if row.get("filter_status") != "kept":
            continue
        command_value = to_float(row.get("command_value"))
        if not math.isfinite(command_value) or command_value == 0.0:
            continue
        key = (str(row.get("surface_axis", "")), round(abs(command_value), 6))
        sign = 1 if command_value > 0.0 else -1
        groups.setdefault(key, {-1: [], 1: []})[sign].append(index)

    rng = random.Random(int(heldout_seed))
    heldout: set[int] = set()
    for key in sorted(groups):
        for sign in (-1, 1):
            indices = sorted(groups[key][sign])
            if len(indices) >= 2:
                heldout.add(rng.choice(indices))
    for index, row in enumerate(rows):
        if row.get("filter_status") == "kept":
            row["split"] = "heldout" if index in heldout else "train"


def replay_kept_rows(
    rows: list[dict[str, Any]],
    *,
    replay_dt_s: float,
    candidate_id: str = "C0_frozen_neutral",
    replay_policy: str = "frozen_active_neutral_model_exact_measured_launch_state_same_command_history_same_duration",
    derivative_coeffs: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    aircraft = adapt_glider(build_nausicaa_glider())
    replay_rows: list[dict[str, Any]] = []
    for row in rows:
        replay_rows.append(
            replay_throw(
                row,
                aircraft=aircraft,
                replay_dt_s=replay_dt_s,
                candidate_id=candidate_id,
                replay_policy=replay_policy,
                derivative_coeffs=derivative_coeffs,
            )
        )
    return replay_rows


def replay_throw(
    row: dict[str, Any],
    *,
    aircraft: Any,
    replay_dt_s: float,
    candidate_id: str,
    replay_policy: str,
    derivative_coeffs: dict[str, float] | None = None,
) -> dict[str, Any]:
    throw_dir = Path(str(row.get("_throw_dir", "")))
    sample_rows = read_csv(throw_dir / "metrics" / "state_samples.csv")
    if not sample_rows:
        return blocked_replay_row(row, "missing_state_samples", replay_dt_s, candidate_id=candidate_id)
    try:
        actual_state0 = prep._state_vector_from_sample_row(sample_rows[0])
    except Exception:
        return blocked_replay_row(row, "nonfinite_initial_state", replay_dt_s, candidate_id=candidate_id)
    if not np.all(np.isfinite(actual_state0)):
        return blocked_replay_row(row, "nonfinite_initial_state", replay_dt_s, candidate_id=candidate_id)

    t0 = to_float(sample_rows[0].get("t_s"), 0.0)
    t1 = to_float(sample_rows[-1].get("t_s"), t0)
    duration_s = max(0.0, t1 - t0)
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        return blocked_replay_row(row, "invalid_duration", replay_dt_s, candidate_id=candidate_id)

    manifest = load_json(throw_dir / "manifests" / "glider_calibration_throw_manifest.json")
    actuator_tau_s = prep._actuator_tau_from_manifest(manifest)
    command_onset_delay_s = float(prep.DEFAULT_REPLAY_COMMAND_ONSET_DELAY_S)
    command_schedule, command_source = prep._load_replay_command_schedule(
        throw_dir,
        row,
        command_onset_delay_s=command_onset_delay_s,
    )
    sim_trace, sim_status = simulate_trace(
        actual_state0,
        command_schedule,
        aircraft=aircraft,
        actuator_tau_s=actuator_tau_s,
        duration_s=duration_s,
        replay_dt_s=replay_dt_s,
        derivative_coeffs=derivative_coeffs,
    )
    if sim_status != "ok":
        return blocked_replay_row(row, sim_status, replay_dt_s, candidate_id=candidate_id)

    response_start_s = to_float(row.get("usable_window_start_s"))
    response_end_s = to_float(row.get("usable_window_end_s"))
    actual_metrics = response_metrics_from_rows(sample_rows, response_start_s, response_end_s)
    sim_metrics = response_metrics_from_rows(sim_trace, response_start_s, response_end_s)
    actual_final = prep._state_vector_from_sample_row(sample_rows[-1])
    sim_final = state_vector_from_metric_row(sim_trace[-1])
    actual_dx = float(actual_final[0] - actual_state0[0])
    actual_dy = float(actual_final[1] - actual_state0[1])
    actual_altitude_loss = float(actual_state0[2] - actual_final[2])
    sim_dx = float(sim_final[0] - actual_state0[0])
    sim_dy = float(sim_final[1] - actual_state0[1])
    sim_altitude_loss = float(actual_state0[2] - sim_final[2])

    out = base_replay_row(row, replay_dt_s, candidate_id=candidate_id)
    out.update(
        {
            "replay_status": "ok",
            "replay_policy": replay_policy,
            "replay_command_source": command_source,
            "replay_command_onset_delay_s": command_onset_delay_s,
            "duration_s": duration_s,
            "response_window_start_s": response_start_s,
            "response_window_end_s": response_end_s,
            "actual_dx_m": actual_dx,
            "sim_dx_m": sim_dx,
            "dx_residual_actual_minus_sim_m": actual_dx - sim_dx,
            "actual_dy_m": actual_dy,
            "sim_dy_m": sim_dy,
            "dy_residual_actual_minus_sim_m": actual_dy - sim_dy,
            "actual_altitude_loss_m": actual_altitude_loss,
            "sim_altitude_loss_m": sim_altitude_loss,
            "altitude_loss_residual_actual_minus_sim_m": actual_altitude_loss - sim_altitude_loss,
            "actual_final_phi_deg": math.degrees(float(actual_final[3])),
            "sim_final_phi_deg": math.degrees(float(sim_final[3])),
            "final_phi_residual_actual_minus_sim_deg": angular_residual_deg(
                math.degrees(float(actual_final[3])),
                math.degrees(float(sim_final[3])),
            ),
            "actual_final_theta_deg": math.degrees(float(actual_final[4])),
            "sim_final_theta_deg": math.degrees(float(sim_final[4])),
            "final_theta_residual_actual_minus_sim_deg": angular_residual_deg(
                math.degrees(float(actual_final[4])),
                math.degrees(float(sim_final[4])),
            ),
            "actual_final_psi_deg": math.degrees(float(actual_final[5])),
            "sim_final_psi_deg": math.degrees(float(sim_final[5])),
            "final_psi_residual_actual_minus_sim_deg": angular_residual_deg(
                math.degrees(float(actual_final[5])),
                math.degrees(float(sim_final[5])),
            ),
        }
    )
    for metric in ALL_RESPONSE_METRICS:
        actual_value = actual_metrics.get(metric, float("nan"))
        sim_value = sim_metrics.get(metric, float("nan"))
        out[f"actual_{metric}"] = actual_value
        out[f"sim_{metric}"] = sim_value
        out[f"{metric}_residual_actual_minus_sim"] = actual_value - sim_value if all_finite(actual_value, sim_value) else float("nan")
    return out


def simulate_trace(
    x0: np.ndarray,
    command_schedule: list[tuple[float, np.ndarray]],
    *,
    aircraft: Any,
    actuator_tau_s: tuple[float, float, float],
    duration_s: float,
    replay_dt_s: float,
    derivative_coeffs: dict[str, float] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    x = np.asarray(x0, dtype=float).copy()
    trace = [metric_row_from_state(0.0, x)]
    t_s = 0.0
    command_index = 0
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    while t_s < float(duration_s) - 1e-12:
        dt_s = min(float(replay_dt_s), float(duration_s) - t_s)
        command_index = prep._advance_command_index(command_schedule, command_index, t_s)
        command = command_schedule[command_index][1]
        try:
            if derivative_coeffs:
                x = rk4_step_surface_aero_coupling(
                    x=x,
                    command=command,
                    aircraft=aircraft,
                    actuator_tau_s=actuator_tau_s,
                    dt_s=dt_s,
                    derivative_coeffs=derivative_coeffs,
                )
            else:
                x = prep._rk4_step_measured_launch(
                    x=x,
                    command=command,
                    aircraft=aircraft,
                    actuator_tau_s=actuator_tau_s,
                    dt_s=dt_s,
                )
        except Exception:
            return trace, "state_derivative_failed"
        t_s += dt_s
        if not np.all(np.isfinite(x)):
            return trace, "nonfinite_replay_state"
        trace.append(metric_row_from_state(t_s, x))
    return trace, "ok"


def rk4_step_surface_aero_coupling(
    *,
    x: np.ndarray,
    command: np.ndarray,
    aircraft: Any,
    actuator_tau_s: tuple[float, float, float],
    dt_s: float,
    derivative_coeffs: dict[str, float],
) -> np.ndarray:
    k1 = state_derivative_with_surface_aero_coupling(x, command, aircraft, actuator_tau_s, derivative_coeffs)
    k2 = state_derivative_with_surface_aero_coupling(
        x + 0.5 * dt_s * k1,
        command,
        aircraft,
        actuator_tau_s,
        derivative_coeffs,
    )
    k3 = state_derivative_with_surface_aero_coupling(
        x + 0.5 * dt_s * k2,
        command,
        aircraft,
        actuator_tau_s,
        derivative_coeffs,
    )
    k4 = state_derivative_with_surface_aero_coupling(
        x + dt_s * k3,
        command,
        aircraft,
        actuator_tau_s,
        derivative_coeffs,
    )
    return x + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def state_derivative_with_surface_aero_coupling(
    x: np.ndarray,
    command: np.ndarray,
    aircraft: Any,
    actuator_tau_s: tuple[float, float, float],
    derivative_coeffs: dict[str, float],
) -> np.ndarray:
    x_dot = state_derivative(
        x,
        command,
        aircraft,
        wind_model=None,
        actuator_tau_s=actuator_tau_s,
        wind_mode="panel",
    )
    x_dot = np.asarray(x_dot, dtype=float).reshape(15)
    x_dot[6:9] += surface_aero_coupling_v_dot(x, aircraft, derivative_coeffs)
    x_dot[9:12] += surface_aero_coupling_omega_dot(x, aircraft, derivative_coeffs)
    return x_dot


def surface_aero_coupling_v_dot(x: np.ndarray, aircraft: Any, derivative_coeffs: dict[str, float]) -> np.ndarray:
    state = np.asarray(x, dtype=float).reshape(15)
    speed_m_s = float(np.linalg.norm(state[6:9]))
    qbar = 0.5 * 1.225 * speed_m_s * speed_m_s
    if not math.isfinite(qbar) or qbar <= 0.0:
        return np.zeros(3)
    delta_a, _, delta_r = [float(value) for value in state[12:15]]
    side_force_b = (
        qbar
        * aircraft.s_ref_m2
        * (
            surface_aero_coefficient(derivative_coeffs, "CY_delta_a_residual", state) * delta_a
            + surface_aero_coefficient(derivative_coeffs, "CY_delta_r_residual", state) * delta_r
        )
    )
    return np.array([0.0, side_force_b / aircraft.mass_kg, 0.0], dtype=float)


def surface_aero_coupling_omega_dot(x: np.ndarray, aircraft: Any, derivative_coeffs: dict[str, float]) -> np.ndarray:
    state = np.asarray(x, dtype=float).reshape(15)
    speed_m_s = float(np.linalg.norm(state[6:9]))
    qbar = 0.5 * 1.225 * speed_m_s * speed_m_s
    if not math.isfinite(qbar) or qbar <= 0.0:
        return np.zeros(3)
    delta_a, delta_e, delta_r = [float(value) for value in state[12:15]]
    moment_b = np.array(
        [
            qbar
            * aircraft.s_ref_m2
            * aircraft.b_ref_m
            * (
                surface_aero_coefficient(derivative_coeffs, "Cl_delta_a_residual", state) * delta_a
                + surface_aero_coefficient(derivative_coeffs, "Cl_delta_r_residual", state) * delta_r
            ),
            qbar
            * aircraft.s_ref_m2
            * aircraft.c_ref_m
            * surface_aero_coefficient(derivative_coeffs, "Cm_delta_e_residual", state)
            * delta_e,
            qbar
            * aircraft.s_ref_m2
            * aircraft.b_ref_m
            * (
                surface_aero_coefficient(derivative_coeffs, "Cn_delta_a_residual", state) * delta_a
                + surface_aero_coefficient(derivative_coeffs, "Cn_delta_r_residual", state) * delta_r
            ),
        ],
        dtype=float,
    )
    return aircraft.inertia_inv_b @ moment_b


def surface_aero_coefficient(derivative_coeffs: dict[str, float], parameter: str, state: np.ndarray) -> float:
    regime = alpha_regime_from_state(state)
    value = to_float(derivative_coeffs.get(f"{parameter}@{regime}"))
    if not math.isfinite(value) and regime == "post_stall" and "_delta_r_" in parameter:
        value = to_float(derivative_coeffs.get(f"{parameter}@transition"))
    if not math.isfinite(value):
        value = to_float(derivative_coeffs.get(parameter, 0.0), 0.0)
    return float(value) if math.isfinite(value) else 0.0


def alpha_regime_from_state(state: np.ndarray) -> str:
    values = np.asarray(state, dtype=float).reshape(-1)
    if values.size < 9:
        return "transition"
    u = float(values[6])
    w = float(values[8])
    alpha_abs_deg = abs(math.degrees(math.atan2(w, max(abs(u), 1e-9))))
    return alpha_regime_from_abs_deg(alpha_abs_deg)


def alpha_regime_from_abs_deg(alpha_abs_deg: float) -> str:
    value = abs(to_float(alpha_abs_deg))
    if not math.isfinite(value):
        return "transition"
    if value < SURFACE_AERO_NORMAL_ALPHA_MAX_DEG:
        return "normal"
    if value < SURFACE_AERO_TRANSITION_ALPHA_MAX_DEG:
        return "transition"
    return "post_stall"


def response_metrics_from_rows(rows: list[dict[str, Any]], start_s: float, end_s: float) -> dict[str, float]:
    if not rows or not all_finite(start_s, end_s) or end_s <= start_s:
        return {metric: float("nan") for metric in ALL_RESPONSE_METRICS}
    window = rows_in_window(rows, start_s, end_s)
    if len(window) < 2:
        return {metric: float("nan") for metric in ALL_RESPONSE_METRICS}
    base = nearest_row(rows, start_s)
    last = window[-1]
    times = [to_float(row.get("t_s")) for row in window]
    p_delta = [to_float(row.get("p")) - to_float(base.get("p")) for row in window]
    q_delta = [to_float(row.get("q")) - to_float(base.get("q")) for row in window]
    r_delta = [to_float(row.get("r")) - to_float(base.get("r")) for row in window]
    phi_change = math.degrees(wrap_angle(to_float(last.get("phi")) - to_float(base.get("phi"))))
    theta_change = math.degrees(wrap_angle(to_float(last.get("theta")) - to_float(base.get("theta"))))
    psi_change = math.degrees(wrap_angle(to_float(last.get("psi")) - to_float(base.get("psi"))))
    dx = to_float(last.get("x_w")) - to_float(base.get("x_w"))
    dy = to_float(last.get("y_w")) - to_float(base.get("y_w"))
    altitude_loss = to_float(base.get("z_w")) - to_float(last.get("z_w"))
    alpha_values = [alpha_deg_from_row(row) for row in window]
    speeds = [speed_from_row(row) for row in window]
    return {
        "peak_p_rad_s": signed_peak(p_delta),
        "p_impulse_rad": trapezoid(times, p_delta),
        "phi_change_deg": phi_change,
        "peak_q_rad_s": signed_peak(q_delta),
        "q_impulse_rad": trapezoid(times, q_delta),
        "theta_change_deg": theta_change,
        "peak_r_rad_s": signed_peak(r_delta),
        "r_impulse_rad": trapezoid(times, r_delta),
        "psi_change_deg": psi_change,
        "roll_coupling_phi_change_deg": phi_change,
        "yaw_coupling_psi_change_deg": psi_change,
        "response_dy_m": dy,
        "response_dx_m": dx,
        "response_altitude_loss_m": altitude_loss,
        "max_abs_alpha_deg": safe_max_abs(alpha_values),
        "alpha_gt_20_s": exposure_time(times, [abs(value) >= 20.0 for value in alpha_values]),
        "alpha_gt_30_s": exposure_time(times, [abs(value) >= 30.0 for value in alpha_values]),
        "speed_mean_m_s": safe_mean(speeds),
        "response_delay_p_s": response_delay(times, p_delta, start_s),
        "response_delay_q_s": response_delay(times, q_delta, start_s),
        "response_delay_r_s": response_delay(times, r_delta, start_s),
    }


def summarize_effectiveness(replay_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    split_specs = [
        ("all", None, False),
        ("train", "train", False),
        ("heldout", "heldout", False),
        ("high_confidence_all", None, False),
        ("high_confidence_train", "train", False),
        ("high_confidence_heldout", "heldout", False),
        ("confidence_weighted_all", None, True),
        ("confidence_weighted_train", "train", True),
        ("confidence_weighted_heldout", "heldout", True),
    ]
    for split, base_split, confidence_weighted in split_specs:
        split_rows = [
            row
            for row in replay_rows
            if row.get("replay_status") == "ok"
            and (base_split is None or row.get("split") == base_split)
            and (not split.startswith("high_confidence_") or row.get("launch_confidence_label") == "high")
        ]
        for surface in ("aileron", "elevator", "rudder"):
            magnitudes = sorted({round(to_float(row.get("command_abs")), 6) for row in split_rows if row.get("surface_axis") == surface})
            for magnitude in magnitudes:
                pair_rows = [
                    row
                    for row in split_rows
                    if row.get("surface_axis") == surface and abs(to_float(row.get("command_abs")) - magnitude) < 1e-9
                ]
                positive = [row for row in pair_rows if to_float(row.get("command_value")) > 0.0]
                negative = [row for row in pair_rows if to_float(row.get("command_value")) < 0.0]
                if not positive or not negative:
                    continue
                for metric in PRIMARY_METRICS_BY_SURFACE[surface]:
                    rows.append(
                        effectiveness_metric_row(
                            split,
                            surface,
                            magnitude,
                            metric,
                            positive,
                            negative,
                            confidence_weighted=confidence_weighted,
                        )
                    )
    return rows


def effectiveness_metric_row(
    split: str,
    surface: str,
    magnitude: float,
    metric: str,
    positive: list[dict[str, Any]],
    negative: list[dict[str, Any]],
    confidence_weighted: bool = False,
) -> dict[str, Any]:
    actual_pos = response_mean(positive, f"actual_{metric}", confidence_weighted=confidence_weighted)
    actual_neg = response_mean(negative, f"actual_{metric}", confidence_weighted=confidence_weighted)
    sim_pos = response_mean(positive, f"sim_{metric}", confidence_weighted=confidence_weighted)
    sim_neg = response_mean(negative, f"sim_{metric}", confidence_weighted=confidence_weighted)
    actual_antisym = 0.5 * (actual_pos - actual_neg) if all_finite(actual_pos, actual_neg) else float("nan")
    actual_sym = 0.5 * (actual_pos + actual_neg) if all_finite(actual_pos, actual_neg) else float("nan")
    sim_antisym = 0.5 * (sim_pos - sim_neg) if all_finite(sim_pos, sim_neg) else float("nan")
    sim_sym = 0.5 * (sim_pos + sim_neg) if all_finite(sim_pos, sim_neg) else float("nan")
    return {
        "split": split,
        "surface_axis": surface,
        "command_abs": magnitude,
        "metric": metric,
        "positive_count": len(positive),
        "negative_count": len(negative),
        "actual_positive_mean": actual_pos,
        "actual_negative_mean": actual_neg,
        "actual_antisymmetric_response": actual_antisym,
        "actual_symmetric_response": actual_sym,
        "sim_positive_mean": sim_pos,
        "sim_negative_mean": sim_neg,
        "sim_antisymmetric_response": sim_antisym,
        "sim_symmetric_response": sim_sym,
        "antisymmetric_residual_actual_minus_sim": actual_antisym - sim_antisym if all_finite(actual_antisym, sim_antisym) else float("nan"),
        "symmetric_residual_actual_minus_sim": actual_sym - sim_sym if all_finite(actual_sym, sim_sym) else float("nan"),
    }


def optional_surface_fit_diagnostics(
    effectiveness_rows: list[dict[str, Any]],
    *,
    run_diagnostics: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline = {
        "candidate_id": "S0_frozen_neutral",
        "status": "evaluated_frozen_active_neutral_model",
        "promoted": False,
        "delta_a_effectiveness_scale": 1.0,
        "delta_e_effectiveness_scale": 1.0,
        "delta_r_effectiveness_scale": 1.0,
        "delta_a_neutral_bias_rad": 0.0,
        "delta_e_neutral_bias_rad": 0.0,
        "delta_r_neutral_bias_rad": 0.0,
        "aileron_left_right_asymmetry": 1.0,
        "actuator_time_constant_scale": 1.0,
        "command_delay_offset_s": 0.0,
        "evidence_summary": "baseline replay only; no surface update",
        "claim_boundary": "residual-calibrated replay alignment only",
    }
    if not run_diagnostics:
        return [
            baseline,
            candidate_not_run("S1_surface_effectiveness_scales", "disabled_by_cli"),
            candidate_not_run("S2_scales_plus_neutral_biases", "disabled_by_cli"),
        ], []

    scales = estimate_surface_scales(effectiveness_rows)
    s1 = {
        "candidate_id": "S1_surface_effectiveness_scales",
        "status": "diagnostic_metric_space_estimate_not_promoted",
        "promoted": False,
        "delta_a_effectiveness_scale": scales.get("aileron", float("nan")),
        "delta_e_effectiveness_scale": scales.get("elevator", float("nan")),
        "delta_r_effectiveness_scale": scales.get("rudder", float("nan")),
        "delta_a_neutral_bias_rad": 0.0,
        "delta_e_neutral_bias_rad": 0.0,
        "delta_r_neutral_bias_rad": 0.0,
        "aileron_left_right_asymmetry": 1.0,
        "actuator_time_constant_scale": 1.0,
        "command_delay_offset_s": 0.0,
        "evidence_summary": "train antisymmetric response ratio only; not replayed as a plant change",
        "claim_boundary": "diagnostic surface-only candidate; no checked-in model change",
    }
    s2 = {
        **candidate_not_run("S2_scales_plus_neutral_biases", "neutral_bias_not_fit_from_symmetric_contamination"),
        "evidence_summary": "symmetric response is reported as contamination and is not hidden as a bias fit",
    }
    heldout_rows = surface_scale_heldout_rows(effectiveness_rows, scales)
    return [baseline, s1, s2], heldout_rows


def estimate_surface_scales(effectiveness_rows: list[dict[str, Any]]) -> dict[str, float]:
    scales: dict[str, float] = {}
    target_metric = {
        "aileron": "peak_p_rad_s",
        "elevator": "peak_q_rad_s",
        "rudder": "peak_r_rad_s",
    }
    for surface, metric in target_metric.items():
        ratios: list[float] = []
        for row in effectiveness_rows:
            if row.get("split") != "train" or row.get("surface_axis") != surface or row.get("metric") != metric:
                continue
            actual = to_float(row.get("actual_antisymmetric_response"))
            sim = to_float(row.get("sim_antisymmetric_response"))
            if all_finite(actual, sim) and abs(sim) > 1e-9 and actual * sim > 0.0:
                ratios.append(float(np.clip(actual / sim, 0.25, 4.0)))
        scales[surface] = median(ratios) if ratios else float("nan")
    return scales


def surface_scale_heldout_rows(effectiveness_rows: list[dict[str, Any]], scales: dict[str, float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in effectiveness_rows:
        if row.get("split") != "heldout":
            continue
        surface = str(row.get("surface_axis", ""))
        metric = str(row.get("metric", ""))
        if metric not in PRIMARY_METRICS_BY_SURFACE.get(surface, ()):
            continue
        actual = to_float(row.get("actual_antisymmetric_response"))
        sim = to_float(row.get("sim_antisymmetric_response"))
        scale = to_float(scales.get(surface))
        baseline_error = abs(actual - sim) if all_finite(actual, sim) else float("nan")
        candidate_error = abs(actual - scale * sim) if all_finite(actual, sim, scale) else float("nan")
        rows.append(
            {
                "candidate_id": "S1_surface_effectiveness_scales",
                "surface_axis": surface,
                "metric": metric,
                "heldout_count": min(int(row.get("positive_count", 0) or 0), int(row.get("negative_count", 0) or 0)),
                "baseline_abs_error": baseline_error,
                "candidate_abs_error": candidate_error,
                "improved": bool(all_finite(baseline_error, candidate_error) and candidate_error < baseline_error),
                "promotion_gate_status": "not_promoted_metric_only_no_closed_loop_or_neutral_replay_gate",
            }
        )
    if not rows:
        rows.append(
            {
                "candidate_id": "S1_surface_effectiveness_scales",
                "surface_axis": "all",
                "metric": "none",
                "heldout_count": 0,
                "baseline_abs_error": float("nan"),
                "candidate_abs_error": float("nan"),
                "improved": False,
                "promotion_gate_status": "not_promoted_no_heldout_pair_summary",
            }
        )
    return rows


def candidate_not_run(candidate_id: str, reason: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "status": f"not_run_{reason}",
        "promoted": False,
        "delta_a_effectiveness_scale": "",
        "delta_e_effectiveness_scale": "",
        "delta_r_effectiveness_scale": "",
        "delta_a_neutral_bias_rad": "",
        "delta_e_neutral_bias_rad": "",
        "delta_r_neutral_bias_rad": "",
        "aileron_left_right_asymmetry": "",
        "actuator_time_constant_scale": "",
        "command_delay_offset_s": "",
        "evidence_summary": reason,
        "claim_boundary": "diagnostic only; no model update",
    }


def append_surface_aero_coupling_candidate(
    candidates: list[dict[str, Any]],
    heldout_rows: list[dict[str, Any]],
    aero_coupling_rows: list[dict[str, Any]],
) -> None:
    improved = sum(1 for row in aero_coupling_rows if bool(row.get("heldout_improved")))
    evaluated = [row for row in aero_coupling_rows if int(row.get("heldout_sample_count", 0) or 0) > 0]
    candidates.append(
        {
            "candidate_id": "D0_launch_confidence_weighted_derivative_fit_basis",
            "status": "diagnostic_derivative_level_fit_not_promoted",
            "promoted": False,
            "delta_a_effectiveness_scale": "",
            "delta_e_effectiveness_scale": "",
            "delta_r_effectiveness_scale": "",
            "delta_a_neutral_bias_rad": 0.0,
            "delta_e_neutral_bias_rad": 0.0,
            "delta_r_neutral_bias_rad": 0.0,
            "aileron_left_right_asymmetry": 1.0,
            "actuator_time_constant_scale": 1.0,
            "command_delay_offset_s": 0.0,
            "evidence_summary": (
                "confidence-weighted derivative-level residual force/moment fit; "
                f"held-out derivative residual improved {improved}/{len(evaluated)} coefficients"
            ),
            "claim_boundary": "diagnostic surface-aero/coupling derivative estimate; no checked-in model change",
        }
    )
    for row in aero_coupling_rows:
        heldout_rows.append(
            {
                "candidate_id": row.get("candidate_id", "D0_launch_confidence_weighted_derivative_fit_basis"),
                "surface_axis": row.get("surface_axis", ""),
                "metric": row.get("parameter", ""),
                "heldout_count": row.get("heldout_sample_count", 0),
                "baseline_abs_error": row.get("heldout_baseline_mae_coeff", float("nan")),
                "candidate_abs_error": row.get("heldout_candidate_mae_coeff", float("nan")),
                "improved": bool(row.get("heldout_improved")),
                "promotion_gate_status": row.get("promotion_gate_status", "not_promoted_derivative_diagnostic_only"),
            }
        )


def surface_aero_coupling_coefficients(rows: list[dict[str, Any]]) -> dict[str, float]:
    coeffs: dict[str, float] = {}
    for row in rows:
        parameter = str(row.get("parameter", ""))
        value = to_float(row.get("bounded_coefficient_per_rad"))
        if parameter and math.isfinite(value):
            coeffs[parameter] = value
    return coeffs


def surface_aero_coupling_candidate_coefficients(coefficients: dict[str, float]) -> dict[str, dict[str, float]]:
    families: dict[str, dict[str, float]] = {}
    for candidate_id, names in SURFACE_AERO_COUPLING_CANDIDATE_FAMILIES.items():
        families[candidate_id] = {
            name: float(coefficients.get(name, 0.0))
            for name in names
            if math.isfinite(to_float(coefficients.get(name, 0.0)))
        }
    return families


def replay_error_summary(
    replay_by_candidate: dict[str, list[dict[str, Any]]],
    effectiveness_by_candidate: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate_id, replay_rows in replay_by_candidate.items():
        effectiveness_rows = effectiveness_by_candidate.get(candidate_id, [])
        for split in ("all", "train", "heldout"):
            split_rows = [
                row
                for row in replay_rows
                if row.get("replay_status") == "ok" and (split == "all" or row.get("split") == split)
            ]
            for surface in ("all", "aileron", "elevator", "rudder"):
                surface_rows = split_rows if surface == "all" else [row for row in split_rows if row.get("surface_axis") == surface]
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "split": split,
                        "surface_axis": surface,
                        "replay_count": len(surface_rows),
                        "dx_mae_m": mae(surface_rows, "dx_residual_actual_minus_sim_m"),
                        "dy_mae_m": mae(surface_rows, "dy_residual_actual_minus_sim_m"),
                        "altitude_loss_mae_m": mae(surface_rows, "altitude_loss_residual_actual_minus_sim_m"),
                        "final_phi_mae_deg": mae(surface_rows, "final_phi_residual_actual_minus_sim_deg"),
                        "final_theta_mae_deg": mae(surface_rows, "final_theta_residual_actual_minus_sim_deg"),
                        "final_psi_mae_deg": mae(surface_rows, "final_psi_residual_actual_minus_sim_deg"),
                        "primary_antisym_residual": mean_abs_primary_antisym_residual(effectiveness_rows, split, surface, "all"),
                        "claim_boundary": (
                            "frozen neutral replay"
                            if candidate_id == "C0_frozen_neutral"
                            else "diagnostic launch-confidence-weighted surface aero/coupling replay only; not promoted"
                        ),
                    }
                )
    return rows


def regime_ladder_error_summary(replay_by_candidate: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    magnitudes_by_surface: dict[str, list[float]] = {}
    for replay_rows in replay_by_candidate.values():
        for surface in ("aileron", "elevator", "rudder"):
            values = {
                round(to_float(row.get("command_abs")), 6)
                for row in replay_rows
                if row.get("replay_status") == "ok"
                and row.get("surface_axis") == surface
                and math.isfinite(to_float(row.get("command_abs")))
            }
            existing = set(magnitudes_by_surface.get(surface, []))
            magnitudes_by_surface[surface] = sorted(existing | values)

    candidate_order = [
        candidate_id
        for candidate_id in SURFACE_AERO_COUPLING_CANDIDATE_FAMILIES
        if candidate_id in replay_by_candidate
    ]
    candidate_order.extend(candidate_id for candidate_id in replay_by_candidate if candidate_id not in candidate_order)

    for candidate_id in candidate_order:
        replay_rows = [row for row in replay_by_candidate.get(candidate_id, []) if row.get("replay_status") == "ok"]
        for split in ("all", "train", "heldout"):
            split_rows = [row for row in replay_rows if split == "all" or row.get("split") == split]
            for surface in ("aileron", "elevator", "rudder"):
                surface_magnitudes = magnitudes_by_surface.get(surface, [])
                for regime in SURFACE_AERO_ALPHA_REGIMES:
                    for magnitude in surface_magnitudes:
                        group_rows = [
                            row
                            for row in split_rows
                            if row.get("surface_axis") == surface
                            and abs(to_float(row.get("command_abs")) - magnitude) < 1e-9
                            and alpha_regime_from_abs_deg(to_float(row.get("actual_max_abs_alpha_deg"))) == regime
                        ]
                        positive = [row for row in group_rows if to_float(row.get("command_value")) > 0.0]
                        negative = [row for row in group_rows if to_float(row.get("command_value")) < 0.0]
                        primary_metric = PRIMARY_METRICS_BY_SURFACE[surface][0]
                        if positive and negative:
                            primary = effectiveness_metric_row(split, surface, magnitude, primary_metric, positive, negative)
                            actual_antisym = primary["actual_antisymmetric_response"]
                            sim_antisym = primary["sim_antisymmetric_response"]
                            primary_residual = abs(to_float(primary["antisymmetric_residual_actual_minus_sim"]))
                        else:
                            actual_antisym = float("nan")
                            sim_antisym = float("nan")
                            primary_residual = float("nan")
                        alpha_values = [to_float(row.get("actual_max_abs_alpha_deg")) for row in group_rows]
                        rows.append(
                            {
                                "candidate_id": candidate_id,
                                "split": split,
                                "surface_axis": surface,
                                "alpha_regime": regime,
                                "command_abs": magnitude,
                                "replay_count": len(group_rows),
                                "positive_count": len(positive),
                                "negative_count": len(negative),
                                "mean_actual_max_abs_alpha_deg": safe_mean(alpha_values),
                                "max_actual_max_abs_alpha_deg": safe_max(alpha_values),
                                "mean_actual_alpha_gt_20_s": safe_mean([to_float(row.get("actual_alpha_gt_20_s")) for row in group_rows]),
                                "mean_actual_alpha_gt_30_s": safe_mean([to_float(row.get("actual_alpha_gt_30_s")) for row in group_rows]),
                                "dx_mae_m": mae(group_rows, "dx_residual_actual_minus_sim_m"),
                                "dy_mae_m": mae(group_rows, "dy_residual_actual_minus_sim_m"),
                                "altitude_loss_mae_m": mae(group_rows, "altitude_loss_residual_actual_minus_sim_m"),
                                "final_phi_mae_deg": mae(group_rows, "final_phi_residual_actual_minus_sim_deg"),
                                "final_theta_mae_deg": mae(group_rows, "final_theta_residual_actual_minus_sim_deg"),
                                "final_psi_mae_deg": mae(group_rows, "final_psi_residual_actual_minus_sim_deg"),
                                "primary_metric": primary_metric,
                                "actual_antisymmetric_response": actual_antisym,
                                "sim_antisymmetric_response": sim_antisym,
                                "primary_antisym_residual": primary_residual,
                                "claim_boundary": (
                                    "frozen neutral replay"
                                    if candidate_id == "C0_frozen_neutral"
                                    else "diagnostic alpha-regime command-ladder replay summary only; not promoted"
                                ),
                            }
                        )
    return rows


def optional_surface_aero_coupling_fit(replay_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aircraft = adapt_glider(build_nausicaa_glider())
    samples: list[dict[str, Any]] = []
    for replay_row in replay_rows:
        if replay_row.get("replay_status") != "ok":
            continue
        samples.extend(derivative_fit_samples_for_replay_row(replay_row, aircraft))

    rows: list[dict[str, Any]] = []
    for spec in SURFACE_AERO_COUPLING_SPECS:
        parameter = str(spec["parameter"])
        surface = str(spec["surface_axis"])
        state_index = int(spec["state_index"])
        residual_key = residual_key_for_surface_aero_spec(spec)
        fit_samples = [
            sample
            for sample in samples
            if sample.get("surface_axis") == surface
            and abs(to_float(sample.get(f"state_{state_index}"))) >= DERIVATIVE_FIT_MIN_SURFACE_RAD
            and math.isfinite(to_float(sample.get(residual_key)))
        ]
        train = [sample for sample in fit_samples if sample.get("split") == "train"]
        heldout = [sample for sample in fit_samples if sample.get("split") == "heldout"]
        coefficient = weighted_surface_derivative_fit(train, state_index=state_index, residual_key=residual_key)
        bounded = float(np.clip(coefficient, -DERIVATIVE_FIT_COEFF_ABS_BOUND, DERIVATIVE_FIT_COEFF_ABS_BOUND)) if math.isfinite(coefficient) else float("nan")
        train_baseline = weighted_derivative_mae(train, state_index=state_index, residual_key=residual_key, coefficient=0.0)
        train_candidate = weighted_derivative_mae(train, state_index=state_index, residual_key=residual_key, coefficient=bounded)
        heldout_baseline = weighted_derivative_mae(heldout, state_index=state_index, residual_key=residual_key, coefficient=0.0)
        heldout_candidate = weighted_derivative_mae(heldout, state_index=state_index, residual_key=residual_key, coefficient=bounded)
        rows.append(
            {
                "candidate_id": "D0_launch_confidence_weighted_derivative_fit_basis",
                "parameter": parameter,
                "surface_axis": surface,
                "command_axis": spec["command_axis"],
                "moment_axis": spec["moment_axis"],
                "fit_role": spec["fit_role"],
                "coefficient_per_rad": coefficient,
                "bounded_coefficient_per_rad": bounded,
                "train_sample_count": len(train),
                "heldout_sample_count": len(heldout),
                "train_baseline_mae_coeff": train_baseline,
                "train_candidate_mae_coeff": train_candidate,
                "heldout_baseline_mae_coeff": heldout_baseline,
                "heldout_candidate_mae_coeff": heldout_candidate,
                "heldout_improved": bool(all_finite(heldout_baseline, heldout_candidate) and heldout_candidate < heldout_baseline),
                "mean_launch_confidence_weight": safe_mean([to_float(sample.get("launch_confidence_weight")) for sample in fit_samples]),
                "physical_sign_status": "signed_residual_derivative_diagnostic_only",
                "promotion_gate_status": "not_promoted_no_replay_neutral_or_closed_loop_gate",
                "claim_boundary": "surface/coupling derivative residual diagnostic only",
            }
        )
        if parameter not in ALPHA_REGIME_SCHEDULED_PARAMETERS:
            continue
        for regime in SURFACE_AERO_ALPHA_REGIMES:
            if surface == "rudder" and regime == "post_stall":
                continue
            regime_samples = [sample for sample in fit_samples if sample.get("alpha_regime") == regime]
            if not regime_samples:
                continue
            train = [sample for sample in regime_samples if sample.get("split") == "train"]
            heldout = [sample for sample in regime_samples if sample.get("split") == "heldout"]
            coefficient = weighted_surface_derivative_fit(train, state_index=state_index, residual_key=residual_key)
            bounded = float(np.clip(coefficient, -DERIVATIVE_FIT_COEFF_ABS_BOUND, DERIVATIVE_FIT_COEFF_ABS_BOUND)) if math.isfinite(coefficient) else float("nan")
            train_baseline = weighted_derivative_mae(train, state_index=state_index, residual_key=residual_key, coefficient=0.0)
            train_candidate = weighted_derivative_mae(train, state_index=state_index, residual_key=residual_key, coefficient=bounded)
            heldout_baseline = weighted_derivative_mae(heldout, state_index=state_index, residual_key=residual_key, coefficient=0.0)
            heldout_candidate = weighted_derivative_mae(heldout, state_index=state_index, residual_key=residual_key, coefficient=bounded)
            rows.append(
                {
                    "candidate_id": "D1_alpha_regime_derivative_fit_basis",
                    "parameter": f"{parameter}@{regime}",
                    "surface_axis": surface,
                    "command_axis": spec["command_axis"],
                    "moment_axis": spec["moment_axis"],
                    "fit_role": f"{spec['fit_role']}_alpha_regime_{regime}",
                    "coefficient_per_rad": coefficient,
                    "bounded_coefficient_per_rad": bounded,
                    "train_sample_count": len(train),
                    "heldout_sample_count": len(heldout),
                    "train_baseline_mae_coeff": train_baseline,
                    "train_candidate_mae_coeff": train_candidate,
                    "heldout_baseline_mae_coeff": heldout_baseline,
                    "heldout_candidate_mae_coeff": heldout_candidate,
                    "heldout_improved": bool(all_finite(heldout_baseline, heldout_candidate) and heldout_candidate < heldout_baseline),
                    "mean_launch_confidence_weight": safe_mean([to_float(sample.get("launch_confidence_weight")) for sample in regime_samples]),
                    "physical_sign_status": "alpha_regime_signed_residual_derivative_diagnostic_only",
                    "promotion_gate_status": "not_promoted_alpha_regime_diagnostic_only",
                    "claim_boundary": "alpha-regime surface/coupling derivative residual diagnostic only",
                }
            )
    return rows


def derivative_fit_samples_for_replay_row(row: dict[str, Any], aircraft: Any) -> list[dict[str, Any]]:
    throw_dir = Path(str(row.get("throw_dir", row.get("_throw_dir", ""))))
    sample_rows = read_csv(throw_dir / "metrics" / "state_samples.csv")
    if len(sample_rows) < 3:
        return []
    response_start_s = to_float(row.get("response_window_start_s"))
    response_end_s = to_float(row.get("response_window_end_s"))
    if not all_finite(response_start_s, response_end_s) or response_end_s <= response_start_s:
        return []
    times = np.asarray([to_float(sample.get("t_s")) for sample in sample_rows], dtype=float)
    states = np.asarray([prep._state_vector_from_sample_row(sample) for sample in sample_rows], dtype=float)
    if states.ndim != 2 or states.shape[1] < 15:
        return []
    manifest = load_json(throw_dir / "manifests" / "glider_calibration_throw_manifest.json")
    actuator_tau_s = prep._actuator_tau_from_manifest(manifest)
    out: list[dict[str, Any]] = []
    for index in range(1, len(sample_rows) - 1):
        time_s = float(times[index])
        if not math.isfinite(time_s) or time_s < response_start_s or time_s > response_end_s:
            continue
        dt_s = float(times[index + 1] - times[index - 1])
        if not math.isfinite(dt_s) or dt_s <= 1e-6:
            continue
        state = states[index]
        if not np.all(np.isfinite(state[:15])):
            continue
        measured_v_dot = (states[index + 1, 6:9] - states[index - 1, 6:9]) / dt_s
        measured_omega_dot = (states[index + 1, 9:12] - states[index - 1, 9:12]) / dt_s
        if not np.all(np.isfinite(measured_v_dot)) or not np.all(np.isfinite(measured_omega_dot)):
            continue
        try:
            base_dot = state_derivative(
                state,
                state[12:15],
                aircraft,
                wind_model=None,
                actuator_tau_s=actuator_tau_s,
                wind_mode="panel",
            )
        except Exception:
            continue
        if not np.all(np.isfinite(base_dot[6:12])):
            continue
        v_b = state[6:9]
        speed_m_s = float(np.linalg.norm(v_b))
        qbar = 0.5 * 1.225 * speed_m_s * speed_m_s
        if not math.isfinite(qbar) or qbar < DERIVATIVE_FIT_MIN_QBAR_PA:
            continue
        moment_residual = aircraft.inertia_b @ (measured_omega_dot - base_dot[9:12])
        force_residual_y = aircraft.mass_kg * float(measured_v_dot[1] - base_dot[7])
        coeff_den_force = qbar * aircraft.s_ref_m2
        coeff_den_span = qbar * aircraft.s_ref_m2 * aircraft.b_ref_m
        coeff_den_chord = qbar * aircraft.s_ref_m2 * aircraft.c_ref_m
        if coeff_den_force <= 0.0 or coeff_den_span <= 0.0 or coeff_den_chord <= 0.0:
            continue
        alpha_abs_deg = abs(math.degrees(math.atan2(float(state[8]), max(abs(float(state[6])), 1e-9))))
        out.append(
            {
                "split": row.get("split", ""),
                "surface_axis": row.get("surface_axis", ""),
                "launch_confidence_weight": row.get("launch_confidence_weight", 1.0),
                "alpha_abs_deg": alpha_abs_deg,
                "alpha_regime": alpha_regime_from_abs_deg(alpha_abs_deg),
                "state_12": float(state[12]),
                "state_13": float(state[13]),
                "state_14": float(state[14]),
                "force_coeff_residual_y": float(force_residual_y / coeff_den_force),
                "moment_coeff_residual_0": float(moment_residual[0] / coeff_den_span),
                "moment_coeff_residual_1": float(moment_residual[1] / coeff_den_chord),
                "moment_coeff_residual_2": float(moment_residual[2] / coeff_den_span),
            }
        )
    return out


def residual_key_for_surface_aero_spec(spec: dict[str, Any]) -> str:
    if spec.get("fit_type") == "side_force":
        return "force_coeff_residual_y"
    return f"moment_coeff_residual_{int(spec['moment_index'])}"


def residual_key_from_args(*, residual_key: str | None = None, moment_index: int | None = None) -> str:
    if residual_key:
        return residual_key
    if moment_index is None:
        raise ValueError("Either residual_key or moment_index is required.")
    return f"moment_coeff_residual_{int(moment_index)}"


def weighted_surface_derivative_fit(
    samples: list[dict[str, Any]],
    *,
    state_index: int,
    residual_key: str | None = None,
    moment_index: int | None = None,
) -> float:
    y_key = residual_key_from_args(residual_key=residual_key, moment_index=moment_index)
    numerator = 0.0
    denominator = 0.0
    for sample in samples:
        x_value = to_float(sample.get(f"state_{state_index}"))
        y_value = to_float(sample.get(y_key))
        weight = to_float(sample.get("launch_confidence_weight"), 1.0)
        if not all_finite(x_value, y_value, weight) or weight <= 0.0:
            continue
        numerator += weight * x_value * y_value
        denominator += weight * x_value * x_value
    return numerator / denominator if denominator > 1e-12 else float("nan")


def weighted_derivative_mae(
    samples: list[dict[str, Any]],
    *,
    state_index: int,
    residual_key: str | None = None,
    moment_index: int | None = None,
    coefficient: float,
) -> float:
    y_key = residual_key_from_args(residual_key=residual_key, moment_index=moment_index)
    numerator = 0.0
    denominator = 0.0
    for sample in samples:
        x_value = to_float(sample.get(f"state_{state_index}"))
        y_value = to_float(sample.get(y_key))
        weight = to_float(sample.get("launch_confidence_weight"), 1.0)
        if not all_finite(x_value, y_value, weight, coefficient) or weight <= 0.0:
            continue
        numerator += weight * abs(y_value - float(coefficient) * x_value)
        denominator += weight
    return numerator / denominator if denominator > 1e-12 else float("nan")


def filtering_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[tuple[str, list[dict[str, Any]], str, Any]] = [("all", rows, "all", "all")]
    for surface in ("aileron", "elevator", "rudder"):
        surface_rows = [row for row in rows if row.get("surface_axis") == surface]
        groups.append((surface, surface_rows, surface, "all"))
        for magnitude in sorted({round(to_float(row.get("command_abs")), 6) for row in surface_rows if math.isfinite(to_float(row.get("command_abs")))}):
            groups.append((f"{surface}_{magnitude:g}", [row for row in surface_rows if abs(to_float(row.get("command_abs")) - magnitude) < 1e-9], surface, magnitude))

    summary_rows: list[dict[str, Any]] = []
    for name, group_rows, surface, magnitude in groups:
        summary_rows.append(
            {
                "group": name,
                "surface_axis": surface,
                "command_abs": magnitude,
                "total_throw_count": len(group_rows),
                "kept_throw_count": sum(1 for row in group_rows if row.get("filter_status") == "kept"),
                "filtered_throw_count": sum(1 for row in group_rows if row.get("filter_status") != "kept"),
                "train_throw_count": sum(1 for row in group_rows if row.get("split") == "train"),
                "heldout_throw_count": sum(1 for row in group_rows if row.get("split") == "heldout"),
                "invalid_throw_count": sum(1 for row in group_rows if row.get("valid_throw") is not True),
                "deep_post_stall_count": sum(1 for row in group_rows if bool(row.get("deep_post_stall_flag"))),
                "early_contact_count": sum(1 for row in group_rows if early_contact_before_response(row, DEFAULT_MIN_RESPONSE_WINDOW_S)),
                "filter_reason_counts": json.dumps(reason_counts(group_rows), sort_keys=True),
            }
        )
    return summary_rows


def launch_confidence_summary(replay_rows: list[dict[str, Any]], effectiveness_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok_rows = [row for row in replay_rows if row.get("replay_status") == "ok"]
    groups: list[tuple[str, str, Any, list[dict[str, Any]]]] = [("all", "all", "all", ok_rows)]
    for surface in ("aileron", "elevator", "rudder"):
        surface_rows = [row for row in ok_rows if row.get("surface_axis") == surface]
        groups.append((surface, surface, "all", surface_rows))
        magnitudes = sorted({round(to_float(row.get("command_abs")), 6) for row in surface_rows if math.isfinite(to_float(row.get("command_abs")))})
        for magnitude in magnitudes:
            groups.append(
                (
                    f"{surface}_{magnitude:g}",
                    surface,
                    magnitude,
                    [row for row in surface_rows if abs(to_float(row.get("command_abs")) - magnitude) < 1e-9],
                )
            )

    rows: list[dict[str, Any]] = []
    for split in ("all", "train", "heldout"):
        for group_name, surface, magnitude, group_rows in groups:
            split_rows = [row for row in group_rows if split == "all" or row.get("split") == split]
            all_residual = mean_abs_primary_antisym_residual(effectiveness_rows, split, surface, magnitude)
            high_residual = mean_abs_primary_antisym_residual(effectiveness_rows, f"high_confidence_{split}", surface, magnitude)
            weighted_residual = mean_abs_primary_antisym_residual(effectiveness_rows, f"confidence_weighted_{split}", surface, magnitude)
            rows.append(
                {
                    "group": group_name,
                    "split": split,
                    "surface_axis": surface,
                    "command_abs": magnitude,
                    "replay_count": len(split_rows),
                    "high_confidence_count": sum(1 for row in split_rows if row.get("launch_confidence_label") == "high"),
                    "medium_confidence_count": sum(1 for row in split_rows if row.get("launch_confidence_label") == "medium"),
                    "low_confidence_count": sum(1 for row in split_rows if row.get("launch_confidence_label") == "low"),
                    "mean_launch_confidence_score": safe_mean([to_float(row.get("launch_confidence_score")) for row in split_rows]),
                    "mean_launch_lateral_contamination_score": safe_mean(
                        [to_float(row.get("launch_lateral_contamination_score")) for row in split_rows]
                    ),
                    "primary_metric": "mean_primary" if surface == "all" else PRIMARY_METRICS_BY_SURFACE[str(surface)][0],
                    "all_abs_antisym_residual": all_residual,
                    "high_confidence_abs_antisym_residual": high_residual,
                    "confidence_weighted_abs_antisym_residual": weighted_residual,
                    "high_minus_all_abs_residual": high_residual - all_residual if all_finite(high_residual, all_residual) else float("nan"),
                    "weighted_minus_all_abs_residual": weighted_residual - all_residual if all_finite(weighted_residual, all_residual) else float("nan"),
                    "replay_dx_mae_m": mae(split_rows, "dx_residual_actual_minus_sim_m"),
                    "replay_dy_mae_m": mae(split_rows, "dy_residual_actual_minus_sim_m"),
                    "replay_altitude_loss_mae_m": mae(split_rows, "altitude_loss_residual_actual_minus_sim_m"),
                }
            )
    return rows


def mean_abs_primary_antisym_residual(
    effectiveness_rows: list[dict[str, Any]],
    split: str,
    surface: str,
    magnitude: Any,
) -> float:
    values: list[float] = []
    for row in effectiveness_rows:
        row_surface = str(row.get("surface_axis", ""))
        if row.get("split") != split:
            continue
        if surface != "all" and row_surface != surface:
            continue
        if magnitude != "all" and abs(to_float(row.get("command_abs")) - to_float(magnitude)) > 1e-9:
            continue
        if row.get("metric") != PRIMARY_METRICS_BY_SURFACE.get(row_surface, ("",))[0]:
            continue
        residual = abs(to_float(row.get("antisymmetric_residual_actual_minus_sim")))
        if math.isfinite(residual):
            values.append(residual)
    return safe_mean(values)


def write_figures(
    figure_dir: Path,
    inventory_rows: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
    effectiveness_rows: list[dict[str, Any]],
    launch_confidence_rows: list[dict[str, Any]],
    optional_heldout: list[dict[str, Any]],
) -> list[str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        write_inventory_plot(figure_dir, inventory_rows),
        write_surface_effectiveness_plot(figure_dir, effectiveness_rows, "aileron"),
        write_surface_effectiveness_plot(figure_dir, effectiveness_rows, "elevator"),
        write_surface_effectiveness_plot(figure_dir, effectiveness_rows, "rudder"),
        write_symmetric_contamination_plot(figure_dir, effectiveness_rows),
        write_launch_confidence_plot(figure_dir, launch_confidence_rows),
        write_heldout_summary_plot(figure_dir, optional_heldout),
    ]
    paths.extend(write_representative_replay_plots(figure_dir, replay_rows))
    return [path.as_posix() for path in paths if path is not None]


def write_inventory_plot(figure_dir: Path, rows: list[dict[str, Any]]) -> Path:
    labels: list[str] = []
    kept: list[int] = []
    filtered: list[int] = []
    for surface in ("aileron", "elevator", "rudder"):
        magnitudes = sorted({round(to_float(row.get("command_abs")), 6) for row in rows if row.get("surface_axis") == surface and math.isfinite(to_float(row.get("command_abs")))})
        for magnitude in magnitudes:
            subset = [row for row in rows if row.get("surface_axis") == surface and abs(to_float(row.get("command_abs")) - magnitude) < 1e-9]
            labels.append(f"{surface[0]} {magnitude:g}")
            kept.append(sum(1 for row in subset if row.get("filter_status") == "kept"))
            filtered.append(sum(1 for row in subset if row.get("filter_status") != "kept"))
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(labels))
    ax.bar(x, kept, label="kept", color="#4477AA")
    ax.bar(x, filtered, bottom=kept, label="filtered", color="#CC6677")
    ax.set_ylabel("launch count")
    ax.set_title("Command Ladder Inventory")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend()
    fig.tight_layout()
    path = figure_dir / "command_ladder_inventory.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_surface_effectiveness_plot(figure_dir: Path, rows: list[dict[str, Any]], surface: str) -> Path:
    metrics = PRIMARY_METRICS_BY_SURFACE[surface][:3]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 8), sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics, strict=True):
        subset = [row for row in rows if row.get("split") == "all" and row.get("surface_axis") == surface and row.get("metric") == metric]
        subset = sorted(subset, key=lambda item: to_float(item.get("command_abs")))
        x = [to_float(row.get("command_abs")) for row in subset]
        actual = [to_float(row.get("actual_antisymmetric_response")) for row in subset]
        sim = [to_float(row.get("sim_antisymmetric_response")) for row in subset]
        ax.plot(x, actual, marker="o", label="real")
        ax.plot(x, sim, marker="s", label="frozen replay")
        ax.axhline(0.0, color="0.35", linewidth=0.8)
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.25)
    axes[0].set_title(f"{surface.title()} Antisymmetric Effectiveness")
    axes[-1].set_xlabel("normalised command magnitude")
    axes[0].legend()
    fig.tight_layout()
    path = figure_dir / f"{surface}_effectiveness.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_symmetric_contamination_plot(figure_dir: Path, rows: list[dict[str, Any]]) -> Path:
    labels: list[str] = []
    values: list[float] = []
    for surface in ("aileron", "elevator", "rudder"):
        metric = PRIMARY_METRICS_BY_SURFACE[surface][0]
        subset = [row for row in rows if row.get("split") == "all" and row.get("surface_axis") == surface and row.get("metric") == metric]
        for row in sorted(subset, key=lambda item: to_float(item.get("command_abs"))):
            labels.append(f"{surface[0]} {to_float(row.get('command_abs')):g}")
            values.append(to_float(row.get("actual_symmetric_response")))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(np.arange(len(labels)), values, color="#DDCC77")
    ax.axhline(0.0, color="0.35", linewidth=0.8)
    ax.set_title("Symmetric Launch/Trim Contamination")
    ax.set_ylabel("symmetric primary response")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    fig.tight_layout()
    path = figure_dir / "symmetric_contamination.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_launch_confidence_plot(figure_dir: Path, rows: list[dict[str, Any]]) -> Path:
    subset = [
        row
        for row in rows
        if row.get("split") == "all"
        and row.get("surface_axis") in {"aileron", "elevator", "rudder"}
        and row.get("command_abs") == "all"
    ]
    labels = [str(row.get("surface_axis")) for row in subset]
    all_residual = [to_float(row.get("all_abs_antisym_residual")) for row in subset]
    high_residual = [to_float(row.get("high_confidence_abs_antisym_residual")) for row in subset]
    weighted_residual = [to_float(row.get("confidence_weighted_abs_antisym_residual")) for row in subset]

    fig, ax = plt.subplots(figsize=(8, 4))
    if labels:
        x = np.arange(len(labels))
        ax.bar(x - 0.24, all_residual, width=0.24, label="all kept")
        ax.bar(x, high_residual, width=0.24, label="high confidence")
        ax.bar(x + 0.24, weighted_residual, width=0.24, label="confidence weighted")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No launch-confidence summary rows", ha="center", va="center")
        ax.set_axis_off()
    ax.set_title("Launch-Confidence Effectiveness Residual Check")
    ax.set_ylabel("mean abs primary antisymmetric residual")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path = figure_dir / "launch_confidence_effectiveness_residuals.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_heldout_summary_plot(figure_dir: Path, rows: list[dict[str, Any]]) -> Path:
    usable = [row for row in rows if row.get("candidate_id") == "S1_surface_effectiveness_scales" and math.isfinite(to_float(row.get("baseline_abs_error")))]
    labels = [f"{row.get('surface_axis', '')}:{row.get('metric', '')}" for row in usable]
    baseline = [to_float(row.get("baseline_abs_error")) for row in usable]
    candidate = [to_float(row.get("candidate_abs_error")) for row in usable]
    fig, ax = plt.subplots(figsize=(10, 4))
    if labels:
        x = np.arange(len(labels))
        ax.bar(x - 0.18, baseline, width=0.36, label="S0 frozen")
        ax.bar(x + 0.18, candidate, width=0.36, label="S1 metric diagnostic")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No held-out S1 metric diagnostics", ha="center", va="center")
        ax.set_axis_off()
    ax.set_title("Held-Out Optional Surface Candidate Summary")
    ax.set_ylabel("absolute antisymmetric error")
    fig.tight_layout()
    path = figure_dir / "heldout_optional_surface_candidate.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_representative_replay_plots(figure_dir: Path, replay_rows: list[dict[str, Any]]) -> list[Path]:
    aircraft = adapt_glider(build_nausicaa_glider())
    paths: list[Path] = []
    primary_rate = {"aileron": "p", "elevator": "q", "rudder": "r"}
    for surface in ("aileron", "elevator", "rudder"):
        selected: list[dict[str, Any]] = []
        for sign in (1, -1):
            candidates = [
                row
                for row in replay_rows
                if row.get("surface_axis") == surface and row.get("replay_status") == "ok" and sign * to_float(row.get("command_value")) > 0.0
            ]
            if candidates:
                selected.append(sorted(candidates, key=lambda row: abs(to_float(row.get("command_abs")) - 0.6))[0])
        fig, ax = plt.subplots(figsize=(8, 4))
        for replay_row in selected:
            throw_dir = Path(str(replay_row.get("throw_dir", replay_row.get("_throw_dir", ""))))
            if not throw_dir.exists():
                throw_dir = Path(str(next((row.get("_throw_dir", "") for row in replay_rows if row is replay_row), "")))
            sample_rows = read_csv(throw_dir / "metrics" / "state_samples.csv")
            if not sample_rows:
                continue
            manifest = load_json(throw_dir / "manifests" / "glider_calibration_throw_manifest.json")
            x0 = prep._state_vector_from_sample_row(sample_rows[0])
            duration_s = max(0.0, to_float(sample_rows[-1].get("t_s"), 0.0) - to_float(sample_rows[0].get("t_s"), 0.0))
            command_schedule, _ = prep._load_replay_command_schedule(
                throw_dir,
                replay_row,
                command_onset_delay_s=float(prep.DEFAULT_REPLAY_COMMAND_ONSET_DELAY_S),
            )
            sim_trace, status = simulate_trace(
                x0,
                command_schedule,
                aircraft=aircraft,
                actuator_tau_s=prep._actuator_tau_from_manifest(manifest),
                duration_s=duration_s,
                replay_dt_s=DEFAULT_REPLAY_DT_S,
            )
            if status != "ok":
                continue
            key = primary_rate[surface]
            label = f"{to_float(replay_row.get('command_value')):+.1f}"
            ax.plot([to_float(row.get("t_s")) for row in sample_rows], [to_float(row.get(key)) for row in sample_rows], label=f"real {label}")
            ax.plot([to_float(row.get("t_s")) for row in sim_trace], [to_float(row.get(key)) for row in sim_trace], linestyle="--", label=f"sim {label}")
        ax.set_title(f"{surface.title()} Representative Replay")
        ax.set_xlabel("time since launch trigger [s]")
        ax.set_ylabel(f"{primary_rate[surface]} [rad/s]")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = figure_dir / f"{surface}_representative_replay.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def build_manifest(
    *,
    input_root: Path,
    dataset_roots: tuple[str, ...],
    output_dir: Path,
    inventory_rows: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
    candidate_sweep_replay_rows: list[dict[str, Any]],
    replay_error_summary_rows: list[dict[str, Any]],
    regime_ladder_error_rows: list[dict[str, Any]],
    effectiveness_rows: list[dict[str, Any]],
    launch_confidence_rows: list[dict[str, Any]],
    aero_coupling_rows: list[dict[str, Any]],
    filtering_summary_rows: list[dict[str, Any]],
    optional_candidates: list[dict[str, Any]],
    optional_heldout: list[dict[str, Any]],
    figures: list[str],
    replay_dt_s: float,
    response_window_s: float,
    min_response_window_s: float,
    heldout_seed: int,
) -> dict[str, Any]:
    kept_count = sum(1 for row in inventory_rows if row.get("filter_status") == "kept")
    promoted = any(bool(row.get("promoted")) for row in optional_candidates)
    return {
        "task": "Real-Flight Control Surface Effectiveness Study v3.0",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit_hash": git_commit_hash(),
        "input_root": input_root.as_posix(),
        "input_data_roots": [(input_root / name).as_posix() for name in dataset_roots],
        "output_dir": output_dir.as_posix(),
        "current_neutral_model_coefficient_source": "03_Control/02_Inner_Loop/A_model_parameters/neutral_dry_air_calibration.py",
        "current_neutral_model_calibration_id": getattr(active_calibration, "CALIBRATION_ID", "unknown"),
        "current_neutral_model_claim_boundary": getattr(active_calibration, "CLAIM_BOUNDARY", "unknown"),
        "replay_script": "03_Control/02_Inner_Loop/run_control_surface_effectiveness_study.py",
        "replay_policy": "frozen active neutral model, measured launch state, logged command schedule, existing command delay and actuator lag",
        "replay_dt_s": float(replay_dt_s),
        "filtering_thresholds": {
            "min_response_window_s": float(min_response_window_s),
            "response_window_s": float(response_window_s),
            "u0_range_m_s": [RELAXED_U_MIN_M_S, RELAXED_U_MAX_M_S],
            "abs_v0_max_m_s": RELAXED_ABS_V_MAX_M_S,
            "abs_w0_max_m_s": RELAXED_ABS_W_MAX_M_S,
            "max_response_spike_fraction": MAX_RESPONSE_SPIKE_FRACTION,
            "deep_post_stall_alpha_deg": DEEP_POST_STALL_ALPHA_DEG,
            "launch_confidence_high_threshold": LAUNCH_CONFIDENCE_HIGH_THRESHOLD,
            "launch_confidence_medium_threshold": LAUNCH_CONFIDENCE_MEDIUM_THRESHOLD,
            "launch_confidence_min_weight": LAUNCH_CONFIDENCE_MIN_WEIGHT,
            "launch_confidence_exponent": LAUNCH_CONFIDENCE_EXPONENT,
            "launch_confidence_reference": "same lateral-contamination strategy as neutral SysID: phi0=psi0=v0=p0=r0=0",
            "derivative_fit_min_qbar_pa": DERIVATIVE_FIT_MIN_QBAR_PA,
            "derivative_fit_min_surface_rad": DERIVATIVE_FIT_MIN_SURFACE_RAD,
            "surface_aero_normal_alpha_max_deg": SURFACE_AERO_NORMAL_ALPHA_MAX_DEG,
            "surface_aero_transition_alpha_max_deg": SURFACE_AERO_TRANSITION_ALPHA_MAX_DEG,
            "rudder_post_stall_schedule": "uses transition coefficient fallback; no independent rudder post-stall coefficient is fitted",
        },
        "train_heldout_split_method": "launch-level one held-out launch per sign within each surface/magnitude ladder pair when available",
        "heldout_seed": int(heldout_seed),
        "throw_counts": {
            "inventoried": len(inventory_rows),
            "kept": kept_count,
            "filtered": len(inventory_rows) - kept_count,
            "train": sum(1 for row in inventory_rows if row.get("split") == "train"),
            "heldout": sum(1 for row in inventory_rows if row.get("split") == "heldout"),
            "successful_replays": sum(1 for row in replay_rows if row.get("replay_status") == "ok"),
            "successful_candidate_sweep_replays": sum(1 for row in candidate_sweep_replay_rows if row.get("replay_status") == "ok"),
        },
        "output_files": {
            "inventory": "control_surface_inventory.csv",
            "filtering_summary": "control_surface_filtering_summary.csv",
            "replay_metrics": "control_surface_replay_metrics.csv",
            "candidate_sweep_replay_metrics": "control_surface_replay_metrics_candidate_sweep.csv",
            "replay_error_summary": "control_surface_replay_error_summary.csv",
            "regime_ladder_error_summary": "control_surface_regime_ladder_error_summary.csv",
            "effectiveness_summary": "control_surface_effectiveness_summary.csv",
            "launch_confidence_summary": "control_surface_launch_confidence_summary.csv",
            "symmetric_contamination_summary": "control_surface_symmetric_contamination_summary.csv",
            "optional_surface_fit_candidates": "optional_surface_fit_candidates.csv",
            "optional_surface_fit_heldout_summary": "optional_surface_fit_heldout_summary.csv",
            "optional_surface_aero_coupling_fit": "optional_surface_aero_coupling_fit.csv",
            "report": "control_surface_effectiveness_report.md",
            "figures": figures,
        },
        "promotion_decision": "not_promoted" if not promoted else "promoted",
        "claim_boundary": "control-surface effectiveness diagnostics and residual-calibrated replay alignment only; no broad aero SysID",
        "filtering_summary_groups": len(filtering_summary_rows),
        "candidate_families": {candidate_id: list(parameters) for candidate_id, parameters in SURFACE_AERO_COUPLING_CANDIDATE_FAMILIES.items()},
        "replay_error_summary_rows": len(replay_error_summary_rows),
        "regime_ladder_error_summary_rows": len(regime_ladder_error_rows),
        "effectiveness_summary_rows": len(effectiveness_rows),
        "launch_confidence_summary_rows": len(launch_confidence_rows),
        "optional_surface_aero_coupling_fit_rows": len(aero_coupling_rows),
        "optional_heldout_rows": len(optional_heldout),
    }


def write_report(
    path: Path,
    *,
    inventory_rows: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
    candidate_sweep_replay_rows: list[dict[str, Any]],
    replay_error_summary_rows: list[dict[str, Any]],
    regime_ladder_error_rows: list[dict[str, Any]],
    effectiveness_rows: list[dict[str, Any]],
    launch_confidence_rows: list[dict[str, Any]],
    aero_coupling_rows: list[dict[str, Any]],
    optional_candidates: list[dict[str, Any]],
    optional_heldout: list[dict[str, Any]],
    filtering_summary_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    kept = [row for row in inventory_rows if row.get("filter_status") == "kept"]
    filtered = [row for row in inventory_rows if row.get("filter_status") != "kept"]
    successful_replays = [row for row in replay_rows if row.get("replay_status") == "ok"]
    successful_candidate_sweep_replays = [row for row in candidate_sweep_replay_rows if row.get("replay_status") == "ok"]
    lines = [
        "# Real-Flight Control Surface Effectiveness Study v3.0",
        "",
        "## 1. Purpose and Claim Boundary",
        "",
        "The current neutral fitted model is frozen. Deflection ladder throws are used first as surface-effectiveness evidence and measured-command replay diagnostics. This is not broad aerodynamic SysID and does not claim accurate full 6-DoF lateral derivative identification.",
        "",
        f"- active neutral model: `{manifest.get('current_neutral_model_calibration_id')}`",
        f"- claim boundary: `{manifest.get('claim_boundary')}`",
        f"- promotion decision: `{manifest.get('promotion_decision')}`",
        "",
        "## 2. Data Inventory",
        "",
        f"- inventoried throws: `{len(inventory_rows)}`",
        f"- kept for main effectiveness analysis: `{len(kept)}`",
        f"- filtered but retained in inventory: `{len(filtered)}`",
        f"- train launches: `{sum(1 for row in inventory_rows if row.get('split') == 'train')}`",
        f"- held-out launches: `{sum(1 for row in inventory_rows if row.get('split') == 'heldout')}`",
        "",
        "## 3. Filtering Rules",
        "",
        "Main analysis requires valid state samples, a matching nonzero 20 percent command-lattice schedule, sufficient response time after command onset, relaxed replay-start velocity bounds, and no floor/wall/contact before the response window. Deep post-stall, launch asymmetry, rate outliers, and filtered throws remain reported rather than deleted.",
        "",
        filter_summary_lines(filtering_summary_rows),
        "",
        "## 4. Frozen-Model Replay Setup",
        "",
        "Each usable throw is replayed from its measured launch state using the active neutral calibrated model, logged command schedule, nominal command-onset delay, and actuator lag from the throw manifest.",
        "",
        f"- successful replays: `{len(successful_replays)}` / `{len(replay_rows)}`",
        f"- replay dx MAE: `{format_value(mae(successful_replays, 'dx_residual_actual_minus_sim_m'))}` m",
        f"- replay dy MAE: `{format_value(mae(successful_replays, 'dy_residual_actual_minus_sim_m'))}` m",
        f"- replay altitude-loss MAE: `{format_value(mae(successful_replays, 'altitude_loss_residual_actual_minus_sim_m'))}` m",
        "",
        "## 5. Candidate Replay Error Summary",
        "",
        "The candidate comparison fits only launch-confidence-weighted residual surface aero/coupling derivatives. Surface-scale fitting is not part of the default fit because measured surface magnitudes are already used; scaling remains an optional legacy appendix only. C6-C8 add a diagnostic alpha-regime schedule using normal, transition, and post-stall bins; rudder post-stall shares the transition coefficient because the kept data have no held-out rudder post-stall support.",
        "",
        f"- successful candidate-family replays: `{len(successful_candidate_sweep_replays)}` / `{len(candidate_sweep_replay_rows)}`",
        "",
        replay_error_report_lines(replay_error_summary_rows),
        "",
        "## 6. Alpha-Regime Command-Ladder Replay Error",
        "",
        "Replay error is also reported as an explicit candidate/surface/alpha-regime/20 percent command ladder. Regime is assigned from measured response-window `actual_max_abs_alpha_deg`: normal `<12 deg`, transition `12-22 deg`, and post-stall `>=22 deg`. Empty cells are retained in `control_surface_regime_ladder_error_summary.csv` with `replay_count=0` so missing support is visible.",
        "",
        regime_ladder_report_lines(regime_ladder_error_rows),
        "",
        "## 7. Launch-Confidence Diagnostic",
        "",
        "Launch confidence is a diagnostic weight and grouping variable, not a new acceptance gate. It reuses the neutral SysID lateral-contamination strategy with reference `phi0=psi0=v0=p0=r0=0`, so the study can test whether real-vs-replay mismatch is launch-condition driven.",
        "",
        launch_confidence_report_lines(launch_confidence_rows),
        "",
        "## 8. Aileron Effectiveness",
        "",
        effectiveness_report_lines(effectiveness_rows, "aileron"),
        "",
        "## 9. Elevator Effectiveness",
        "",
        effectiveness_report_lines(effectiveness_rows, "elevator"),
        "",
        "## 10. Rudder Effectiveness",
        "",
        effectiveness_report_lines(effectiveness_rows, "rudder"),
        "",
        "## 11. Cross-Coupling Observations",
        "",
        "Aileron yaw response and rudder roll response are reported as diagnostic coupling evidence. They are not promoted as lateral transition aerodynamic derivatives by this study.",
        "",
        "## 12. Symmetric Launch/Trim Contamination",
        "",
        "Symmetric response is separated from antisymmetric response. Large symmetric terms are interpreted as launch, trim, hardware, or model-mismatch contamination rather than hidden inside a surface effectiveness scale.",
        "",
        symmetric_report_lines(effectiveness_rows),
        "",
        "## 13. Optional Surface/Aero Fit Result",
        "",
        optional_candidate_report_lines(optional_candidates, optional_heldout),
        "",
        surface_aero_coupling_report_lines(aero_coupling_rows),
        "",
        "## 14. Promotion Decision",
        "",
        "No model parameter is promoted by this analysis. A surface-only update would require held-out deflection improvement, neutral replay preservation, interpretable signs/magnitudes, and closed-loop smoke evidence.",
        "",
        "## 15. Limitations",
        "",
        "- Launch-condition contamination remains visible in the symmetric response.",
        "- Deflection data are sustained pulse-ladder throws, not a broad aero excitation design.",
        "- Candidate derivative rows are diagnostic summaries, not checked-in plant changes.",
        "- Alpha-regime candidate rows are diagnostic; they do not establish a validated surface-effectiveness schedule.",
        "- Regime-ladder rows are evidence reporting cells, not independent pass gates.",
        "- S1/S2 surface-scale rows are disabled by default because measured surface magnitudes are already used.",
        "- R5/R7/R8/R10/R11 semantics are unchanged.",
        "",
        "## 16. Reproducibility Commands",
        "",
        "```powershell",
        "python 03_Control/02_Inner_Loop/run_control_surface_effectiveness_study.py",
        "pytest 03_Control/tests/test_control_surface_effectiveness_study.py",
        "pytest 03_Control/tests/test_neutral_aero_residual_sysid_contract.py",
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def filter_summary_lines(rows: list[dict[str, Any]]) -> str:
    all_row = next((row for row in rows if row.get("group") == "all"), {})
    return (
        f"- all throws: `{all_row.get('total_throw_count', 0)}` total, "
        f"`{all_row.get('kept_throw_count', 0)}` kept, "
        f"`{all_row.get('filtered_throw_count', 0)}` filtered; "
        f"reason counts `{all_row.get('filter_reason_counts', '{}')}`"
    )


def launch_confidence_report_lines(rows: list[dict[str, Any]]) -> str:
    all_row = next((row for row in rows if row.get("split") == "all" and row.get("group") == "all"), {})
    lines = [
        (
            f"- all successful replays: `{all_row.get('replay_count', 0)}` total, "
            f"`{all_row.get('high_confidence_count', 0)}` high-confidence, "
            f"`{all_row.get('medium_confidence_count', 0)}` medium-confidence, "
            f"`{all_row.get('low_confidence_count', 0)}` low-confidence; "
            f"mean confidence weight `{format_value(all_row.get('mean_launch_confidence_score'))}`, "
            f"mean lateral-contamination score `{format_value(all_row.get('mean_launch_lateral_contamination_score'))}`"
        ),
        "- primary antisymmetric residual check; lower is better, negative delta means the confidence subset reduced mismatch:",
    ]
    for surface in ("aileron", "elevator", "rudder"):
        row = next(
            (
                item
                for item in rows
                if item.get("split") == "all" and item.get("surface_axis") == surface and item.get("command_abs") == "all"
            ),
            {},
        )
        lines.append(
            "- "
            f"{surface}: all `{format_value(row.get('all_abs_antisym_residual'))}`, "
            f"high-confidence `{format_value(row.get('high_confidence_abs_antisym_residual'))}` "
            f"(delta `{format_value(row.get('high_minus_all_abs_residual'))}`), "
            f"weighted `{format_value(row.get('confidence_weighted_abs_antisym_residual'))}` "
            f"(delta `{format_value(row.get('weighted_minus_all_abs_residual'))}`)"
        )
    return "\n".join(lines)


def regime_ladder_report_lines(rows: list[dict[str, Any]]) -> str:
    heldout = [
        row
        for row in rows
        if row.get("split") == "heldout" and int(row.get("replay_count", 0) or 0) > 0
    ]
    if not heldout:
        return "- no held-out regime/command replay cells available"
    lines = [
        "- held-out non-empty cells; lower is better:",
        "`candidate | surface | regime | |cmd| | n | dx | dy | altitude | phi | theta | psi | primary`",
    ]
    candidate_order = {candidate_id: index for index, candidate_id in enumerate(SURFACE_AERO_COUPLING_CANDIDATE_FAMILIES)}
    surface_order = {"aileron": 0, "elevator": 1, "rudder": 2}
    regime_order = {regime: index for index, regime in enumerate(SURFACE_AERO_ALPHA_REGIMES)}
    for row in sorted(
        heldout,
        key=lambda item: (
            candidate_order.get(str(item.get("candidate_id")), 999),
            surface_order.get(str(item.get("surface_axis")), 999),
            regime_order.get(str(item.get("alpha_regime")), 999),
            to_float(item.get("command_abs")),
        ),
    ):
        lines.append(
            "- "
            f"`{row.get('candidate_id')} | "
            f"{row.get('surface_axis')} | "
            f"{row.get('alpha_regime')} | "
            f"{format_value(row.get('command_abs'))} | "
            f"{row.get('replay_count')} | "
            f"{format_value(row.get('dx_mae_m'))} | "
            f"{format_value(row.get('dy_mae_m'))} | "
            f"{format_value(row.get('altitude_loss_mae_m'))} | "
            f"{format_value(row.get('final_phi_mae_deg'))} | "
            f"{format_value(row.get('final_theta_mae_deg'))} | "
            f"{format_value(row.get('final_psi_mae_deg'))} | "
            f"{format_value(row.get('primary_antisym_residual'))}`"
        )
    return "\n".join(lines)


def replay_error_report_lines(rows: list[dict[str, Any]]) -> str:
    lines = [
        "- replay MAE comparison; lower is better:",
        (
            "`candidate | split | surface | dx | dy | altitude | phi | theta | psi | primary antisym`"
        ),
    ]
    candidate_order = [
        candidate_id
        for candidate_id in SURFACE_AERO_COUPLING_CANDIDATE_FAMILIES
        if any(row.get("candidate_id") == candidate_id for row in rows)
    ]
    for candidate_id in candidate_order:
        for split in ("all", "heldout"):
            for surface in ("all", "aileron", "elevator", "rudder"):
                row = next(
                    (
                        item
                        for item in rows
                        if item.get("candidate_id") == candidate_id
                        and item.get("split") == split
                        and item.get("surface_axis") == surface
                    ),
                    {},
                )
                if not row:
                    continue
                lines.append(
                    "- "
                    f"`{candidate_id} | {split} | {surface} | "
                    f"{format_value(row.get('dx_mae_m'))} | "
                    f"{format_value(row.get('dy_mae_m'))} | "
                    f"{format_value(row.get('altitude_loss_mae_m'))} | "
                    f"{format_value(row.get('final_phi_mae_deg'))} | "
                    f"{format_value(row.get('final_theta_mae_deg'))} | "
                    f"{format_value(row.get('final_psi_mae_deg'))} | "
                    f"{format_value(row.get('primary_antisym_residual'))}`"
                )
    return "\n".join(lines)


def effectiveness_report_lines(rows: list[dict[str, Any]], surface: str) -> str:
    subset = [
        row
        for row in rows
        if row.get("split") == "all" and row.get("surface_axis") == surface and row.get("metric") in PRIMARY_METRICS_BY_SURFACE[surface][:3]
    ]
    if not subset:
        return "- no paired kept launches available"
    lines: list[str] = []
    for row in sorted(subset, key=lambda item: (str(item.get("metric")), to_float(item.get("command_abs")))):
        lines.append(
            "- "
            f"`{row.get('metric')}` at |cmd| `{format_value(row.get('command_abs'))}`: "
            f"real antisym `{format_value(row.get('actual_antisymmetric_response'))}`, "
            f"frozen replay antisym `{format_value(row.get('sim_antisymmetric_response'))}`, "
            f"symmetric `{format_value(row.get('actual_symmetric_response'))}`"
        )
    return "\n".join(lines)


def symmetric_report_lines(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for surface in ("aileron", "elevator", "rudder"):
        metric = PRIMARY_METRICS_BY_SURFACE[surface][0]
        values = [
            abs(to_float(row.get("actual_symmetric_response")))
            for row in rows
            if row.get("split") == "all" and row.get("surface_axis") == surface and row.get("metric") == metric
        ]
        lines.append(f"- {surface}: mean absolute primary symmetric response `{format_value(safe_mean(values))}`")
    return "\n".join(lines)


def optional_candidate_report_lines(candidates: list[dict[str, Any]], heldout_rows: list[dict[str, Any]]) -> str:
    lines = [
        f"- `{row.get('candidate_id')}`: `{row.get('status')}`, promoted `{row.get('promoted')}`"
        for row in candidates
    ]
    s1_rows = [row for row in heldout_rows if row.get("candidate_id") == "S1_surface_effectiveness_scales"]
    if s1_rows:
        improved = sum(1 for row in s1_rows if bool(row.get("improved")))
        lines.append(f"- S1 held-out metric diagnostics improved `{improved}` / `{len(s1_rows)}` rows, but remain not promoted.")
    else:
        lines.append("- S1/S2 surface-scale diagnostics are disabled by default because measured surface magnitudes are used.")
    return "\n".join(lines)


def surface_aero_coupling_report_lines(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "- Surface aero/coupling derivative diagnostic was not run."
    lines = [
        "- The derivative diagnostic fits residual control force/moment coefficients from measured acceleration with launch-confidence weighting; it is not replay-promoted.",
        "- `@normal`, `@transition`, and `@post_stall` rows are alpha-regime diagnostics. Rudder post-stall is intentionally not fitted independently and falls back to transition in scheduled replay candidates.",
    ]
    for row in rows:
        lines.append(
            "- "
            f"`{row.get('parameter')}`: coeff `{format_value(row.get('bounded_coefficient_per_rad'))}`, "
            f"held-out baseline `{format_value(row.get('heldout_baseline_mae_coeff'))}`, "
            f"candidate `{format_value(row.get('heldout_candidate_mae_coeff'))}`, "
            f"improved `{row.get('heldout_improved')}`"
        )
    return "\n".join(lines)


def command_schedule_audit(
    schedule_rows: list[dict[str, Any]],
    command_axis: str,
    command_value: float,
    command_start_s: float,
) -> tuple[str, str]:
    if not schedule_rows:
        return "missing_command_schedule", "metrics/command_schedule.csv missing or empty"
    if command_axis not in COMMAND_AXIS_INDEX or not math.isfinite(command_value):
        return "invalid_command_metadata", "manifest command axis/value missing"
    axis_index = COMMAND_AXIS_INDEX[command_axis]
    axis_fields = ("delta_a_cmd_norm", "delta_e_cmd_norm", "delta_r_cmd_norm")
    active_rows: list[dict[str, Any]] = []
    for row in schedule_rows:
        values = [to_float(row.get(field), 0.0) for field in axis_fields]
        if any(abs(value) > COMMAND_MATCH_TOL for value in values) or bool_value(row.get("pulse_active")):
            active_rows.append(row)
    if not active_rows:
        return "missing_active_pulse", "no active command rows"
    active_values = [to_float(row.get(axis_fields[axis_index]), 0.0) for row in active_rows]
    off_axis_values = [
        abs(to_float(row.get(field), 0.0))
        for row in active_rows
        for index, field in enumerate(axis_fields)
        if index != axis_index
    ]
    active_start = min(to_float(row.get("t_s"), 0.0) for row in active_rows)
    median_active = median(active_values)
    if abs(median_active - command_value) > 1e-6:
        return "command_value_mismatch", f"active median {median_active:g} != manifest {command_value:g}"
    if off_axis_values and max(off_axis_values) > 1e-6:
        return "off_axis_command_present", f"max off-axis command {max(off_axis_values):g}"
    if math.isfinite(command_start_s) and abs(active_start - command_start_s) > COMMAND_TIMING_TOL_S:
        return "command_timing_mismatch", f"active start {active_start:g}, manifest start {command_start_s:g}"
    return "ok", f"active rows {len(active_rows)}, active start {active_start:g}"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field, "")) for field in fieldnames})


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def base_replay_row(row: dict[str, Any], replay_dt_s: float, *, candidate_id: str = "C0_frozen_neutral") -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "split": row.get("split", ""),
        "dataset_root": row.get("dataset_root", ""),
        "session_label": row.get("session_label", ""),
        "trial_id": row.get("trial_id", ""),
        "throw_id": row.get("throw_id", ""),
        "throw_dir": row.get("throw_dir", row.get("_throw_dir", "")),
        "case_id": row.get("case_id", ""),
        "surface_axis": row.get("surface_axis", ""),
        "command_axis": row.get("command_axis", ""),
        "command_value": row.get("command_value", ""),
        "command_abs": row.get("command_abs", ""),
        "launch_confidence_score": row.get("launch_confidence_score", ""),
        "launch_confidence_label": row.get("launch_confidence_label", ""),
        "launch_confidence_weight": row.get("launch_confidence_weight", ""),
        "launch_lateral_contamination_score": row.get("launch_lateral_contamination_score", ""),
        "launch_confidence_reasons": row.get("launch_confidence_reasons", ""),
        "replay_status": "",
        "replay_policy": "",
        "replay_dt_s": float(replay_dt_s),
        "replay_command_source": "",
        "replay_command_onset_delay_s": "",
    }


def blocked_replay_row(
    row: dict[str, Any],
    status: str,
    replay_dt_s: float,
    *,
    candidate_id: str = "C0_frozen_neutral",
) -> dict[str, Any]:
    out = base_replay_row(row, replay_dt_s, candidate_id=candidate_id)
    out["replay_status"] = status
    out["replay_policy"] = "blocked_before_replay"
    return out


def metric_row_from_state(t_s: float, x: np.ndarray) -> dict[str, Any]:
    return {
        "t_s": float(t_s),
        "x_w": float(x[0]),
        "y_w": float(x[1]),
        "z_w": float(x[2]),
        "phi": float(x[3]),
        "theta": float(x[4]),
        "psi": float(x[5]),
        "u": float(x[6]),
        "v": float(x[7]),
        "w": float(x[8]),
        "p": float(x[9]),
        "q": float(x[10]),
        "r": float(x[11]),
        "delta_a": float(x[12]),
        "delta_e": float(x[13]),
        "delta_r": float(x[14]),
    }


def state_vector_from_metric_row(row: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            to_float(row.get("x_w")),
            to_float(row.get("y_w")),
            to_float(row.get("z_w")),
            to_float(row.get("phi")),
            to_float(row.get("theta")),
            to_float(row.get("psi")),
            to_float(row.get("u")),
            to_float(row.get("v")),
            to_float(row.get("w")),
            to_float(row.get("p")),
            to_float(row.get("q")),
            to_float(row.get("r")),
            to_float(row.get("delta_a")),
            to_float(row.get("delta_e")),
            to_float(row.get("delta_r")),
        ],
        dtype=float,
    )


def rows_in_window(rows: list[dict[str, Any]], start_s: float, end_s: float) -> list[dict[str, Any]]:
    if not all_finite(start_s, end_s):
        return []
    out = [row for row in rows if start_s - 1e-9 <= to_float(row.get("t_s")) <= end_s + 1e-9]
    if len(out) >= 2:
        return out
    return [row for row in rows if to_float(row.get("t_s")) >= start_s - 1e-9][:2]


def nearest_row(rows: list[dict[str, Any]], t_s: float) -> dict[str, Any]:
    return min(rows, key=lambda row: abs(to_float(row.get("t_s")) - t_s))


def signed_peak(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return float("nan")
    return max(finite, key=lambda value: abs(value))


def trapezoid(times: list[float], values: list[float]) -> float:
    if len(times) < 2 or len(times) != len(values):
        return float("nan")
    total = 0.0
    for t0, t1, y0, y1 in zip(times[:-1], times[1:], values[:-1], values[1:], strict=True):
        if all_finite(t0, t1, y0, y1):
            total += 0.5 * (y0 + y1) * (t1 - t0)
    return total


def exposure_time(times: list[float], flags: list[bool]) -> float:
    if len(times) < 2 or len(times) != len(flags):
        return 0.0
    total = 0.0
    for t0, t1, flag0, flag1 in zip(times[:-1], times[1:], flags[:-1], flags[1:], strict=True):
        if flag0 or flag1:
            total += max(0.0, t1 - t0)
    return total


def response_delay(times: list[float], values: list[float], start_s: float) -> float:
    peak = signed_peak(values)
    if not math.isfinite(peak) or abs(peak) < 1e-9:
        return float("nan")
    sign = 1.0 if peak >= 0.0 else -1.0
    threshold = max(0.2 * abs(peak), 0.05)
    for time_s, value in zip(times, values, strict=True):
        if sign * value >= threshold:
            return max(0.0, time_s - start_s)
    return float("nan")


def alpha_deg_from_row(row: dict[str, Any]) -> float:
    u = to_float(row.get("u"))
    w = to_float(row.get("w"))
    if not all_finite(u, w):
        return float("nan")
    return math.degrees(math.atan2(w, max(abs(u), 1e-9)))


def speed_from_row(row: dict[str, Any]) -> float:
    u, v, w = to_float(row.get("u")), to_float(row.get("v")), to_float(row.get("w"))
    if not all_finite(u, v, w):
        return float("nan")
    return math.sqrt(u * u + v * v + w * w)


def launch_quality_from_state(phi: float, theta: float, psi: float, u: float, v: float, w: float, p: float, q: float, r: float) -> str:
    flags: list[str] = []
    if math.isfinite(v) and abs(v) > 1.0:
        flags.append("lateral_velocity_asymmetry")
    if math.isfinite(phi) and abs(math.degrees(phi)) > 15.0:
        flags.append("roll_angle_asymmetry")
    if math.isfinite(psi) and abs(math.degrees(psi)) > 15.0:
        flags.append("yaw_angle_asymmetry")
    if any(
        [
            math.isfinite(p) and abs(p) > 1.2,
            math.isfinite(q) and abs(q) > 1.2,
            math.isfinite(r) and abs(r) > 1.8,
        ]
    ):
        flags.append("rate_outlier")
    if math.isfinite(u) and u < 4.0:
        flags.append("below_real_launch_gate_u")
    if math.isfinite(w) and abs(w) > 0.9:
        flags.append("vertical_velocity_edge")
    return ";".join(flags) if flags else "clean"


def launch_confidence_from_inventory_row(row: dict[str, Any]) -> dict[str, Any]:
    components = [
        ("roll_angle", math.radians(to_float(row.get("phi0_deg"))) / max(math.radians(LAUNCH_CONFIDENCE_ROLL_ABS_MAX_DEG), 1e-9)),
        ("yaw_angle", math.radians(to_float(row.get("psi0_deg"))) / max(math.radians(LAUNCH_CONFIDENCE_YAW_ABS_MAX_DEG), 1e-9)),
        ("lateral_velocity", to_float(row.get("v0_m_s")) / max(LAUNCH_CONFIDENCE_V_ABS_MAX_M_S, 1e-9)),
        ("roll_rate", to_float(row.get("p0_rad_s")) / max(LAUNCH_CONFIDENCE_P_ABS_MAX_RAD_S, 1e-9)),
        ("yaw_rate", to_float(row.get("r0_rad_s")) / max(LAUNCH_CONFIDENCE_R_ABS_MAX_RAD_S, 1e-9)),
    ]
    finite_components = [(name, abs(value)) for name, value in components if math.isfinite(value)]
    if len(finite_components) != len(components):
        lateral_score = float("nan")
        weight = 1.0
        reasons = [f"missing_{name}" for name, value in components if not math.isfinite(value)]
    else:
        lateral_score = float(np.sqrt(np.mean([value * value for _, value in finite_components])))
        weight = math.exp(-float(LAUNCH_CONFIDENCE_EXPONENT) * lateral_score**2)
        weight = float(np.clip(weight, LAUNCH_CONFIDENCE_MIN_WEIGHT, 1.0))
        reasons = [name for name, value in finite_components if value >= 0.45]

    if weight >= LAUNCH_CONFIDENCE_HIGH_THRESHOLD:
        label = "high"
    elif weight >= LAUNCH_CONFIDENCE_MEDIUM_THRESHOLD:
        label = "medium"
    else:
        label = "low"
    return {
        "launch_confidence_score": weight,
        "launch_confidence_label": label,
        "launch_confidence_weight": weight,
        "launch_lateral_contamination_score": lateral_score,
        "launch_confidence_reasons": ";".join(reasons),
    }


def tracking_quality_flag(confidence: list[float], response_spikes: list[bool]) -> str:
    mean_conf = safe_mean(confidence)
    spike_fraction = ratio(sum(response_spikes), len(response_spikes))
    if math.isfinite(spike_fraction) and spike_fraction > MAX_RESPONSE_SPIKE_FRACTION:
        return "response_spike_fraction_high"
    if math.isfinite(mean_conf) and mean_conf < 0.65:
        return "low_rate_confidence"
    return "nominal"


def early_contact_before_response(row: dict[str, Any], min_response_window_s: float) -> bool:
    group = str(row.get("termination_group", ""))
    if group not in {"floor", "front_wall", "other_wall"}:
        return False
    start_s = to_float(row.get("command_start_s"))
    duration_s = to_float(row.get("effective_flight_duration_s"))
    return all_finite(start_s, duration_s) and duration_s < start_s + float(min_response_window_s)


def reason_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for reason in str(row.get("filter_reasons", "")).split(";"):
            if not reason:
                continue
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def inventory_notes(summary: dict[str, Any], config: dict[str, Any]) -> str:
    notes: list[str] = []
    if summary.get("flight_cancelled"):
        notes.append(f"cancelled:{summary.get('cancellation_reason', '')}")
    if summary.get("exit_gate_triggered") is True:
        notes.append("exit_gate_triggered")
    if config.get("controller_mode"):
        notes.append(f"controller_mode:{config.get('controller_mode')}")
    return ";".join(str(note) for note in notes if note)


def relative_parts(path: Path, root: Path) -> tuple[str, ...]:
    try:
        return path.relative_to(root).parts
    except ValueError:
        return path.parts


def is_lattice_20_percent(value: float) -> bool:
    if not math.isfinite(value):
        return False
    return abs(value * 5.0 - round(value * 5.0)) < 1e-6 and abs(value) <= 1.0 + 1e-9


def command_norm_to_surface_rad(command_norm: tuple[float, float, float] | list[float] | np.ndarray) -> np.ndarray:
    return normalised_command_to_surface_rad(np.asarray(command_norm, dtype=float).reshape(3))


def termination_group_from_reason(reason: str) -> str:
    lowered = str(reason).lower()
    if "floor" in lowered:
        return "floor"
    if "front" in lowered:
        return "front_wall"
    if "wall" in lowered:
        return "other_wall"
    if "abort" in lowered or "cancel" in lowered:
        return "manual_abort"
    if not lowered:
        return "unknown"
    return lowered


def wrap_angle(value_rad: float) -> float:
    if not math.isfinite(value_rad):
        return float("nan")
    return (value_rad + math.pi) % (2.0 * math.pi) - math.pi


def angular_residual_deg(actual_deg: float, simulated_deg: float) -> float:
    if not all_finite(actual_deg, simulated_deg):
        return float("nan")
    return math.degrees(wrap_angle(math.radians(actual_deg - simulated_deg)))


def to_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def all_finite(*values: float) -> bool:
    for value in values:
        try:
            if not math.isfinite(float(value)):
                return False
        except (TypeError, ValueError):
            return False
    return True


def safe_mean(values: list[float]) -> float:
    finite = [to_float(value) for value in values if math.isfinite(to_float(value))]
    return mean(finite) if finite else float("nan")


def response_mean(rows: list[dict[str, Any]], key: str, *, confidence_weighted: bool = False) -> float:
    if not confidence_weighted:
        return safe_mean([to_float(row.get(key)) for row in rows])
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        value = to_float(row.get(key))
        weight = to_float(row.get("launch_confidence_weight"))
        if not all_finite(value, weight) or weight <= 0.0:
            continue
        numerator += value * weight
        denominator += weight
    return numerator / denominator if denominator > 0.0 else float("nan")


def safe_min(values: list[float]) -> float:
    finite = [to_float(value) for value in values if math.isfinite(to_float(value))]
    return min(finite) if finite else float("nan")


def safe_max(values: list[float]) -> float:
    finite = [to_float(value) for value in values if math.isfinite(to_float(value))]
    return max(finite) if finite else float("nan")


def safe_max_abs(values: list[float]) -> float:
    finite = [abs(to_float(value)) for value in values if math.isfinite(to_float(value))]
    return max(finite) if finite else float("nan")


def ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(float(numerator)) or not math.isfinite(float(denominator)) or abs(float(denominator)) < 1e-12:
        return float("nan")
    return float(numerator) / float(denominator)


def mae(rows: list[dict[str, Any]], key: str) -> float:
    values = [abs(to_float(row.get(key))) for row in rows if math.isfinite(to_float(row.get(key)))]
    return safe_mean(values)


def format_value(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if math.isfinite(value):
            return f"{value:.10g}"
        return ""
    return value


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
