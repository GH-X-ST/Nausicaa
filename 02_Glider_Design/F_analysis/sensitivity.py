"""Local first-order sensitivity analysis for the selected Nausicaa design.

Usage:
    python F_analysis/solve_step_size.py
    python F_analysis/sensitivity.py

Inputs:
    - C_results/nausicaa_workflow.xlsx
    - C_results/nausicaa_results.xlsx
    - C_results/step_size_table.csv

Outputs:
    - C_results/sensitivity_analysis.xlsx
    - C_results/sensitivity_table.csv
    - C_results/sensitivity_thesis_table.csv

Notes:
    - Run `solve_step_size.py` first. It performs the expensive finite-
      difference step-size sweep and stores the candidate derivative table.
    - This script reads the saved step-size table and extracts the selected
      derivative for each parameter/quantity pair.
    - Requirement perturbations do not re-optimise geometry.
    - Requirement terms wired through module-level constants in nausicaa.py are
      handled by local wrappers in this script so the analysis remains additive.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import sys
from typing import Any

import aerosandbox as asb
import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import nausicaa

nausicaa.IPOPT_VERBOSE = False

RESULTS_DIR = PROJECT_ROOT / "C_results"
FIGURES_DIR = PROJECT_ROOT / "B_figures"
WORKFLOW_XLSX = RESULTS_DIR / "nausicaa_workflow.xlsx"
RESULTS_XLSX = RESULTS_DIR / "nausicaa_results.xlsx"
OUTPUT_XLSX = RESULTS_DIR / "sensitivity_analysis.xlsx"
OUTPUT_TABLE_CSV = RESULTS_DIR / "sensitivity_table.csv"
OUTPUT_THESIS_CSV = RESULTS_DIR / "sensitivity_thesis_table.csv"
OUTPUT_STEP_SIZE_CSV = RESULTS_DIR / "step_size_table.csv"
OUTPUT_STEP_SIZE_XLSX = RESULTS_DIR / "step_size_analysis.xlsx"
OUTPUT_FIGURE = FIGURES_DIR / "step_size_tradeoff.png"
LEGACY_HEATMAP_FIGURE = FIGURES_DIR / "sensitivity_matrix.png"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

FD_REL_STEP = 2e-3
FD_STABLE_REL_TOL = 0.10
FD_STABLE_ABS_TOL = 1e-9
TURN_TAU_FLOOR_S = 1e-4
FD_ROUNDOFF_SAFETY = 50.0
FD_ROUNDOFF_REL_SCALE = math.sqrt(np.finfo(float).eps)
FD_STEP_LADDER_MULTIPLIERS = (
    8.0,
    4.0,
    2.0,
    1.0,
    0.5,
    0.25,
    0.125,
    0.0625,
)
FD_STEP_LADDER_LEVELS = len(FD_STEP_LADDER_MULTIPLIERS)

GEOMETRY_PARAM_ORDER = [
    "wing_span_m",
    "wing_chord_m",
    "tail_arm_m",
    "htail_span_m",
    "vtail_height_m",
]
REQUIREMENT_PARAM_ORDER = [
    "bank_entry_time_s",
    "wall_clearance_m",
    "turn_bank_deg",
    "static_margin_min",
    "max_cl_nominal",
    "max_roll_tau_s",
]

PRIMARY_QUANTITY_ORDER = [
    "objective",
    "sink_rate_mps",
    "mass_total_kg",
    "roll_tau_s",
    "static_margin",
    "nom_cl_margin_to_cap",
    "nom_util_e",
    "nom_lateral_residual",
]
MARGIN_QUANTITY_ORDER = [
    "static_margin_min_margin",
    "roll_tau_limit_margin",
    "elevator_util_margin",
    "bank_entry_margin_deg",
    "turn_radius_allow_m",
    "turn_radius_ach_m",
    "turn_footprint_margin_m",
    "agility_lateral_margin_mps2",
]
ALL_QUANTITY_ORDER = PRIMARY_QUANTITY_ORDER + MARGIN_QUANTITY_ORDER

GEOMETRY_THESIS_METRICS = [
    "sink_rate_mps",
    "mass_total_kg",
    "roll_tau_s",
    "static_margin",
]
REQUIREMENT_THESIS_METRICS = [
    "objective",
    "sink_rate_mps",
    "roll_tau_s",
]
GEOMETRY_STEP_PLOT_QUANTITIES = [
    "objective",
    "sink_rate_mps",
    "mass_total_kg",
    "roll_tau_s",
    "static_margin",
    "static_margin_min_margin",
    "bank_entry_margin_deg",
    "turn_footprint_margin_m",
]
REQUIREMENT_STEP_PLOT_QUANTITIES = [
    "objective",
    "sink_rate_mps",
    "roll_tau_s",
    "static_margin",
    "static_margin_min_margin",
    "nom_cl_margin_to_cap",
    "roll_tau_limit_margin",
    "bank_entry_margin_deg",
    "turn_footprint_margin_m",
    "agility_lateral_margin_mps2",
]

PARAMETER_LABELS = {
    "wing_span_m": "Wing span",
    "wing_chord_m": "Wing chord",
    "tail_arm_m": "Tail arm",
    "htail_span_m": "H-tail span",
    "vtail_height_m": "V-tail height",
    "bank_entry_time_s": "Bank-entry time",
    "wall_clearance_m": "Wall clearance",
    "turn_bank_deg": "Turn bank angle",
    "static_margin_min": "Static-margin minimum",
    "max_cl_nominal": "Nominal CL cap",
    "max_roll_tau_s": "Roll-tau limit",
}
PARAMETER_SYMBOLS = {
    "wing_span_m": "b",
    "wing_chord_m": "c",
    "tail_arm_m": "l_t",
    "htail_span_m": "b_h",
    "vtail_height_m": "h_v",
    "bank_entry_time_s": "t_be",
    "wall_clearance_m": "c_w",
    "turn_bank_deg": "phi_turn",
    "static_margin_min": "SM_min",
    "max_cl_nominal": "CL_cap",
    "max_roll_tau_s": "tau_roll,max",
}
QUANTITY_LABELS = {
    "objective": "Objective",
    "sink_rate_mps": "Nominal sink rate",
    "mass_total_kg": "Total mass",
    "roll_tau_s": "Roll time constant",
    "static_margin": "Static margin",
    "nom_cl_margin_to_cap": "Nominal CL margin",
    "nom_util_e": "Nominal elevator utilisation",
    "nom_lateral_residual": "Nominal lateral residual",
    "static_margin_min_margin": "Static-margin min margin",
    "roll_tau_limit_margin": "Roll-tau limit margin",
    "elevator_util_margin": "Elevator-utilisation margin",
    "bank_entry_margin_deg": "Bank-entry margin",
    "turn_radius_allow_m": "Allowable turn radius",
    "turn_radius_ach_m": "Achieved turn radius",
    "turn_footprint_margin_m": "Turn-footprint margin",
    "agility_lateral_margin_mps2": "Lateral-agility margin",
}
QUANTITY_SYMBOLS = {
    "objective": "J",
    "sink_rate_mps": "V_sink",
    "mass_total_kg": "m",
    "roll_tau_s": "tau_roll",
    "static_margin": "SM",
    "nom_cl_margin_to_cap": "Delta_CL",
    "nom_util_e": "u_e",
    "nom_lateral_residual": "r_lat",
    "static_margin_min_margin": "Delta_SM_min",
    "roll_tau_limit_margin": "Delta_tau_max",
    "elevator_util_margin": "Delta_u_e",
    "bank_entry_margin_deg": "Delta_phi_be",
    "turn_radius_allow_m": "R_allow",
    "turn_radius_ach_m": "R_ach",
    "turn_footprint_margin_m": "Delta_footprint",
    "agility_lateral_margin_mps2": "Delta_a_lat",
}
QUANTITY_UNITS = {
    "objective": "-",
    "sink_rate_mps": "m/s",
    "mass_total_kg": "kg",
    "roll_tau_s": "s",
    "static_margin": "%MAC",
    "nom_cl_margin_to_cap": "-",
    "nom_util_e": "-",
    "nom_lateral_residual": "-",
    "static_margin_min_margin": "%MAC",
    "roll_tau_limit_margin": "s",
    "elevator_util_margin": "-",
    "bank_entry_margin_deg": "deg",
    "turn_radius_allow_m": "m",
    "turn_radius_ach_m": "m",
    "turn_footprint_margin_m": "m",
    "agility_lateral_margin_mps2": "m/s^2",
}
QUANTITY_FLOORS = {
    "objective": 1e-6,
    "sink_rate_mps": 1e-6,
    "mass_total_kg": 1e-6,
    "roll_tau_s": 1e-6,
    "static_margin": 1e-4,
    "nom_cl_margin_to_cap": 1e-6,
    "nom_util_e": 1e-6,
    "nom_lateral_residual": 1e-6,
    "static_margin_min_margin": 1e-4,
    "roll_tau_limit_margin": 1e-6,
    "elevator_util_margin": 1e-6,
    "bank_entry_margin_deg": 1e-4,
    "turn_radius_allow_m": 1e-6,
    "turn_radius_ach_m": 1e-6,
    "turn_footprint_margin_m": 1e-6,
    "agility_lateral_margin_mps2": 1e-6,
}

OVERLAP_METRIC_KEYS = [
    "objective",
    "sink_rate_mps",
    "mass_total_kg",
    "static_margin",
    "roll_tau_s",
    "bank_entry_margin_deg",
    "nom_cl_margin_to_cap",
]


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    group: str
    unit: str
    baseline_value: float
    rel_step: float
    abs_floor: float
    symbol: str
    evaluation_path: str
    lower_bound: float | None = None
    notes: str = ""


@dataclass(frozen=True)
class QuantitySpec:
    name: str
    unit: str
    symbol: str
    q_floor: float
    notes: str = ""


@dataclass
class SensitivityResult:
    group: str
    parameter_name: str
    parameter_symbol: str
    baseline_parameter_value: float
    step_used: float
    quantity_name: str
    quantity_symbol: str
    baseline_quantity_value: float
    sensitivity_raw: float
    sensitivity_normalized: float
    difference_scheme: str
    fd_stability_abs: float
    fd_stability_rel: float
    fd_stable: bool
    unit_raw: str
    step_selection_reason: str
    notes: str
    reeval_path: str


@dataclass
class StepSizeResult:
    group: str
    parameter_name: str
    parameter_symbol: str
    baseline_parameter_value: float
    step_level: int
    quantity_name: str
    quantity_symbol: str
    baseline_quantity_value: float
    difference_scheme: str
    step_size: float
    derivative_estimate: float
    normalized_sensitivity: float
    absolute_error_estimate: float
    relative_error_estimate: float
    fd_stability_abs: float
    fd_stability_rel: float
    fd_stable: bool
    signal_amplitude: float
    roundoff_floor: float
    roundoff_error_proxy: float
    total_error_proxy: float
    roundoff_limited: bool
    selected_for_final: bool
    selected_step_size: float
    unit_raw: str
    error_reference_method: str
    selection_reason: str
    reeval_path: str
    notes: str


@dataclass(frozen=True)
class BaselineContext:
    workbook_path: Path
    selected_candidate_id: int
    selected_row: dict[str, float]
    summary_map: dict[str, Any]
    design_points_map: dict[str, Any]
    run_info_map: dict[str, Any]
    constraints_df: pd.DataFrame
    active_constraints_df: pd.DataFrame
    geometry: nausicaa.GeometryVars
    objective_weights: nausicaa.ObjectiveWeights
    objective_scales: nausicaa.ObjectiveScales
    cfg: nausicaa.Config
    workflow_cfg: nausicaa.WorkflowConfig
    requirement_values: dict[str, float]
    trim_seed: dict[str, float]
    workbook_baseline_values: dict[str, float]


@dataclass
class EvaluationResult:
    success: bool
    metrics: dict[str, float]
    state: dict[str, float]
    notes: list[str]


def _to_float(value: Any) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return float("nan")
    return float(numeric)


def _series_lookup(df: pd.DataFrame, key_col: str, value_col: str) -> dict[str, Any]:
    if df.empty or key_col not in df.columns or value_col not in df.columns:
        return {}
    return pd.Series(df[value_col].to_numpy(), index=df[key_col].astype(str)).to_dict()


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"1", "true", "yes"})


def _read_sheet(book: pd.ExcelFile, sheet_name: str) -> pd.DataFrame:
    if sheet_name in book.sheet_names:
        return pd.read_excel(book, sheet_name=sheet_name)
    return pd.DataFrame()


def open_canonical_workbook() -> tuple[Path, pd.ExcelFile]:
    for path in (WORKFLOW_XLSX, RESULTS_XLSX):
        if path.exists():
            return path, pd.ExcelFile(path)
    raise FileNotFoundError(
        "No canonical workbook found. Expected "
        f"{WORKFLOW_XLSX} or {RESULTS_XLSX}."
    )


def _resolve_tail_metric_name(df: pd.DataFrame) -> str:
    if "nom_sink_tail_mean_k" in df.columns:
        return "nom_sink_tail_mean_k"
    if "nom_sink_cvar_20" in df.columns:
        return "nom_sink_cvar_20"
    return "nom_sink_tail_mean_k"


def resolve_selected_candidate_id(
    candidates_df: pd.DataFrame,
    robust_summary_df: pd.DataFrame,
) -> int:
    if not robust_summary_df.empty and "is_selected" in robust_summary_df.columns:
        selected_mask = _coerce_bool_series(robust_summary_df["is_selected"])
        if selected_mask.any():
            return int(robust_summary_df.loc[selected_mask, "candidate_id"].iloc[0])

    if not candidates_df.empty and not robust_summary_df.empty:
        tail_metric = _resolve_tail_metric_name(robust_summary_df)
        merged_df = robust_summary_df.merge(
            candidates_df[["candidate_id", "objective"]],
            on="candidate_id",
            how="left",
        )
        ranked_df = merged_df.sort_values(
            by=["nom_success_rate", tail_metric, "objective"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        if not ranked_df.empty:
            return int(ranked_df.iloc[0]["candidate_id"])

    if not candidates_df.empty:
        ranked_df = candidates_df.sort_values("objective", kind="mergesort")
        return int(ranked_df.iloc[0]["candidate_id"])

    return 1


def _constraint_row(df: pd.DataFrame, name: str) -> pd.Series | None:
    if df.empty or "Constraint" not in df.columns:
        return None
    match_df = df.loc[df["Constraint"] == name]
    if match_df.empty:
        return None
    return match_df.iloc[0]


def _tail_arm_from_boom_length(boom_length_m: float, htail_span_m: float) -> float:
    htail_chord_m = htail_span_m / max(float(nausicaa.HT_AR), 1e-9)
    return (
        boom_length_m
        + float(nausicaa.NOSE_X_M)
        - float(nausicaa.BOOM_END_BEFORE_ELEV_FRAC) * htail_chord_m
    )


def _build_workflow_config(run_info_map: dict[str, Any]) -> nausicaa.WorkflowConfig:
    workflow_cfg = nausicaa.WorkflowConfig()
    override_keys = {
        "servo_rate_deg_s",
        "nom_trim_time_s",
        "rate_util_fraction",
        "turn_deflection_util",
        "max_trim_util_fraction",
    }
    override_values: dict[str, Any] = {}
    for key in override_keys:
        if key in run_info_map:
            value = _to_float(run_info_map[key])
            if math.isfinite(value):
                override_values[key] = value
    if override_values:
        workflow_cfg = replace(workflow_cfg, **override_values)
    return workflow_cfg


def _build_fallback_selected_row(
    summary_map: dict[str, Any],
    geometry_df: pd.DataFrame,
) -> dict[str, float]:
    geometry_map = _series_lookup(geometry_df, "Parameter", "Value")
    row = {
        "candidate_id": 1.0,
        "objective": _to_float(summary_map.get("objective", float("nan"))),
        "wing_span_m": _to_float(geometry_map.get("wing_span_m", float("nan"))),
        "wing_chord_m": _to_float(geometry_map.get("wing_chord_m", float("nan"))),
        "boom_length_m": _to_float(geometry_map.get("boom_length_m", float("nan"))),
        "htail_span_m": _to_float(geometry_map.get("htail_span_m", float("nan"))),
        "vtail_height_m": _to_float(geometry_map.get("vtail_height_m", float("nan"))),
    }
    for key in (
        "objective_weight_w_sink",
        "objective_weight_w_mass",
        "objective_weight_w_trim_effort",
        "objective_weight_w_wing_deflection",
        "objective_weight_w_htail_deflection",
        "objective_weight_w_roll_tau",
        "objective_scale_sink_mps",
        "objective_scale_mass_kg",
        "objective_scale_trim_deg",
        "objective_scale_roll_tau_s",
    ):
        row[key] = _to_float(summary_map.get(key, float("nan")))
    return row


def _lookup_numeric(
    mapping: dict[str, Any],
    *keys: str,
    default: float = float("nan"),
) -> float:
    for key in keys:
        if key not in mapping:
            continue
        value = _to_float(mapping[key])
        if math.isfinite(value):
            return value
    return default


def _build_objective_weights(
    selected_row: dict[str, float],
    summary_map: dict[str, Any],
) -> nausicaa.ObjectiveWeights:
    defaults = asdict(nausicaa.ObjectiveWeights())
    column_map = {
        "w_sink": "objective_weight_w_sink",
        "w_mass": "objective_weight_w_mass",
        "w_trim_effort": "objective_weight_w_trim_effort",
        "w_wing_deflection": "objective_weight_w_wing_deflection",
        "w_htail_deflection": "objective_weight_w_htail_deflection",
        "w_roll_tau": "objective_weight_w_roll_tau",
    }
    kwargs: dict[str, float] = {}
    for field_name, column_name in column_map.items():
        value = _lookup_numeric(
            selected_row,
            column_name,
            default=_lookup_numeric(
                summary_map,
                column_name,
                default=defaults[field_name],
            ),
        )
        if not math.isfinite(value):
            value = float(defaults[field_name])
        kwargs[field_name] = float(value)
    return nausicaa.ObjectiveWeights(**kwargs)


def _build_objective_scales(
    selected_row: dict[str, float],
    summary_map: dict[str, Any],
) -> nausicaa.ObjectiveScales:
    defaults = asdict(nausicaa.ObjectiveScales())
    column_map = {
        "sink_mps": "objective_scale_sink_mps",
        "mass_kg": "objective_scale_mass_kg",
        "trim_deg": "objective_scale_trim_deg",
        "roll_tau_s": "objective_scale_roll_tau_s",
    }
    kwargs: dict[str, float] = {}
    for field_name, column_name in column_map.items():
        value = _lookup_numeric(
            selected_row,
            column_name,
            default=_lookup_numeric(
                summary_map,
                column_name,
                default=defaults[field_name],
            ),
        )
        if not math.isfinite(value):
            value = float(defaults[field_name])
        kwargs[field_name] = float(value)
    return nausicaa.ObjectiveScales(**kwargs)


def _build_requirement_values(
    summary_map: dict[str, Any],
    design_points_map: dict[str, Any],
    constraints_df: pd.DataFrame,
) -> dict[str, float]:
    default_cfg = nausicaa.Config()
    cl_constraint_row = _constraint_row(constraints_df, "CL <= CLmax")
    static_margin_row = _constraint_row(constraints_df, "Static margin minimum")

    max_cl_nominal = float(default_cfg.max_cl_nominal)
    if cl_constraint_row is not None:
        constraint_upper = _to_float(cl_constraint_row.get("Upper"))
        if math.isfinite(constraint_upper):
            max_cl_nominal = constraint_upper

    static_margin_min = float(nausicaa.STATIC_MARGIN_MIN)
    if static_margin_row is not None:
        constraint_lower = _to_float(static_margin_row.get("Lower"))
        if math.isfinite(constraint_lower):
            static_margin_min = constraint_lower

    return {
        "bank_entry_time_s": _lookup_numeric(
            summary_map,
            "bank_entry_time_s",
            default=_lookup_numeric(
                design_points_map,
                "BANK_ENTRY_TIME_S",
                default=float(default_cfg.bank_entry_time_s),
            ),
        ),
        "wall_clearance_m": _lookup_numeric(
            summary_map,
            "wall_clearance_m",
            default=_lookup_numeric(
                design_points_map,
                "WALL_CLEARANCE_M",
                default=float(default_cfg.wall_clearance_m),
            ),
        ),
        "turn_bank_deg": _lookup_numeric(
            summary_map,
            "turn_bank_deg",
            default=_lookup_numeric(
                design_points_map,
                "TURN_BANK_DEG",
                default=float(default_cfg.turn_bank_deg),
            ),
        ),
        "static_margin_min": 100.0 * float(static_margin_min),
        "max_cl_nominal": float(max_cl_nominal),
        "max_roll_tau_s": _lookup_numeric(
            summary_map,
            "max_roll_tau_s",
            default=float(default_cfg.max_roll_tau_s),
        ),
    }


def _build_cfg(
    summary_map: dict[str, Any],
    design_points_map: dict[str, Any],
    requirement_values: dict[str, float],
) -> nausicaa.Config:
    base_cfg = nausicaa.Config()
    overrides = {
        "v_nom_mps": _lookup_numeric(
            summary_map,
            "v_nom_mps",
            default=_lookup_numeric(
                design_points_map,
                "V_NOM_MPS",
                default=float(base_cfg.v_nom_mps),
            ),
        ),
        "v_turn_mps": _lookup_numeric(
            summary_map,
            "v_turn_mps",
            default=_lookup_numeric(
                design_points_map,
                "V_TURN_MPS",
                default=float(base_cfg.v_turn_mps),
            ),
        ),
        "arena_width_m": _lookup_numeric(
            summary_map,
            "arena_width_m",
            default=_lookup_numeric(
                design_points_map,
                "ARENA_WIDTH_M",
                default=float(base_cfg.arena_width_m),
            ),
        ),
        "arena_length_m": _lookup_numeric(
            summary_map,
            "arena_length_m",
            default=_lookup_numeric(
                design_points_map,
                "ARENA_LENGTH_M",
                default=float(base_cfg.arena_length_m),
            ),
        ),
        "arena_height_m": _lookup_numeric(
            summary_map,
            "arena_height_m",
            default=_lookup_numeric(
                design_points_map,
                "ARENA_HEIGHT_M",
                default=float(base_cfg.arena_height_m),
            ),
        ),
        "turn_bank_deg": float(requirement_values["turn_bank_deg"]),
        "wall_clearance_m": float(requirement_values["wall_clearance_m"]),
        "bank_entry_time_s": float(requirement_values["bank_entry_time_s"]),
        "max_cl_nominal": float(requirement_values["max_cl_nominal"]),
        "max_roll_tau_s": float(requirement_values["max_roll_tau_s"]),
    }
    return replace(base_cfg, **overrides)


def _build_workbook_baseline_values(
    selected_row: dict[str, float],
    summary_map: dict[str, Any],
    constraints_df: pd.DataFrame,
) -> dict[str, float]:
    cl_constraint_row = _constraint_row(constraints_df, "CL <= CLmax")
    cl_margin = float("nan")
    if cl_constraint_row is not None:
        cl_margin = _to_float(cl_constraint_row.get("Margin"))
        if not math.isfinite(cl_margin):
            upper = _to_float(cl_constraint_row.get("Upper"))
            value = _to_float(cl_constraint_row.get("Value"))
            if math.isfinite(upper) and math.isfinite(value):
                cl_margin = upper - value

    return {
        "objective": _lookup_numeric(
            selected_row,
            "objective",
            default=_lookup_numeric(summary_map, "objective"),
        ),
        "sink_rate_mps": _lookup_numeric(
            selected_row,
            "sink_rate_mps",
            default=_lookup_numeric(summary_map, "sink_rate_mps"),
        ),
        "mass_total_kg": _lookup_numeric(
            selected_row,
            "mass_total_kg",
            default=_lookup_numeric(summary_map, "mass_total_kg"),
        ),
        "static_margin": 100.0
        * _lookup_numeric(
            selected_row,
            "static_margin",
            default=_lookup_numeric(summary_map, "static_margin"),
        ),
        "roll_tau_s": _lookup_numeric(
            selected_row,
            "roll_tau_s",
            default=_lookup_numeric(summary_map, "roll_tau_s"),
        ),
        "bank_entry_margin_deg": _lookup_numeric(summary_map, "bank_entry_margin_deg"),
        "nom_cl_margin_to_cap": float(cl_margin),
    }


def load_selected_baseline() -> BaselineContext:
    workbook_path, book = open_canonical_workbook()
    try:
        candidates_df = _read_sheet(book, "Candidates")
        robust_summary_df = _read_sheet(book, "RobustSummary")
        summary_df = _read_sheet(book, "Summary")
        geometry_df = _read_sheet(book, "Geometry")
        design_points_df = _read_sheet(book, "DesignPoints")
        constraints_df = _read_sheet(book, "Constraints")
        active_constraints_df = _read_sheet(book, "ActiveConstraints")
        run_info_df = _read_sheet(book, "RunInfo")
    finally:
        book.close()

    summary_map = _series_lookup(summary_df, "Metric", "Value")
    design_points_map = _series_lookup(design_points_df, "Metric", "Value")
    run_info_map = _series_lookup(run_info_df, "Key", "Value")

    selected_candidate_id = resolve_selected_candidate_id(
        candidates_df=candidates_df,
        robust_summary_df=robust_summary_df,
    )

    if not candidates_df.empty and "candidate_id" in candidates_df.columns:
        candidate_ids = pd.to_numeric(candidates_df["candidate_id"], errors="coerce")
        match_df = candidates_df.loc[candidate_ids == selected_candidate_id]
        if match_df.empty:
            match_df = candidates_df.sort_values("objective", kind="mergesort").head(1)
        selected_row_raw = match_df.iloc[0].to_dict()
    else:
        selected_row_raw = _build_fallback_selected_row(summary_map, geometry_df)

    selected_row = {
        str(key): _to_float(value)
        if not isinstance(value, str)
        else value
        for key, value in selected_row_raw.items()
    }

    wing_span_m = _lookup_numeric(selected_row, "wing_span_m")
    wing_chord_m = _lookup_numeric(selected_row, "wing_chord_m")
    htail_span_m = _lookup_numeric(selected_row, "htail_span_m")
    vtail_height_m = _lookup_numeric(selected_row, "vtail_height_m")
    tail_arm_m = _lookup_numeric(selected_row, "tail_arm_m")
    if not math.isfinite(tail_arm_m):
        boom_length_m = _lookup_numeric(
            selected_row,
            "boom_length_m",
            default=_lookup_numeric(summary_map, "boom_length_m"),
        )
        tail_arm_m = _tail_arm_from_boom_length(boom_length_m, htail_span_m)

    geometry = nausicaa.GeometryVars(
        wing_span_m=float(wing_span_m),
        wing_chord_m=float(wing_chord_m),
        tail_arm_m=float(tail_arm_m),
        htail_span_m=float(htail_span_m),
        vtail_height_m=float(vtail_height_m),
    )
    objective_weights = _build_objective_weights(selected_row, summary_map)
    objective_scales = _build_objective_scales(selected_row, summary_map)
    requirement_values = _build_requirement_values(
        summary_map=summary_map,
        design_points_map=design_points_map,
        constraints_df=constraints_df,
    )
    cfg = _build_cfg(
        summary_map=summary_map,
        design_points_map=design_points_map,
        requirement_values=requirement_values,
    )
    workflow_cfg = _build_workflow_config(run_info_map)
    trim_seed = {
        "alpha_deg": _lookup_numeric(
            summary_map,
            "alpha_trim_deg",
            default=_lookup_numeric(
                design_points_map,
                "alpha_nom_deg",
                default=_lookup_numeric(selected_row, "alpha_deg", default=5.0),
            ),
        ),
        "delta_a_deg": _lookup_numeric(
            summary_map,
            "delta_a_trim_deg",
            default=_lookup_numeric(
                design_points_map,
                "delta_a_nom_deg",
                default=_lookup_numeric(selected_row, "delta_a_deg", default=0.0),
            ),
        ),
        "delta_e_deg": _lookup_numeric(
            summary_map,
            "delta_e_trim_deg",
            default=_lookup_numeric(
                design_points_map,
                "delta_e_nom_deg",
                default=_lookup_numeric(selected_row, "delta_e_deg", default=0.0),
            ),
        ),
        "delta_r_deg": _lookup_numeric(
            summary_map,
            "delta_r_trim_deg",
            default=_lookup_numeric(
                design_points_map,
                "delta_r_nom_deg",
                default=_lookup_numeric(selected_row, "delta_r_deg", default=0.0),
            ),
        ),
    }
    workbook_baseline_values = _build_workbook_baseline_values(
        selected_row=selected_row,
        summary_map=summary_map,
        constraints_df=constraints_df,
    )

    return BaselineContext(
        workbook_path=workbook_path,
        selected_candidate_id=int(selected_candidate_id),
        selected_row=selected_row,
        summary_map=summary_map,
        design_points_map=design_points_map,
        run_info_map=run_info_map,
        constraints_df=constraints_df,
        active_constraints_df=active_constraints_df,
        geometry=geometry,
        objective_weights=objective_weights,
        objective_scales=objective_scales,
        cfg=cfg,
        workflow_cfg=workflow_cfg,
        requirement_values=requirement_values,
        trim_seed=trim_seed,
        workbook_baseline_values=workbook_baseline_values,
    )


def build_geometry_parameter_specs(context: BaselineContext) -> list[ParameterSpec]:
    cfg = context.cfg
    baseline_map = {
        "wing_span_m": float(context.geometry.wing_span_m),
        "wing_chord_m": float(context.geometry.wing_chord_m),
        "tail_arm_m": float(context.geometry.tail_arm_m),
        "htail_span_m": float(context.geometry.htail_span_m),
        "vtail_height_m": float(context.geometry.vtail_height_m),
    }
    lower_bounds = {
        "wing_span_m": float(cfg.wing_span_min_m),
        "wing_chord_m": float(cfg.wing_chord_min_m),
        "tail_arm_m": float(cfg.tail_arm_min_m),
        "htail_span_m": float(cfg.htail_span_min_m),
        "vtail_height_m": float(cfg.vtail_height_min_m),
    }
    notes_map = {
        "tail_arm_m": "Recovered from workbook boom_length_m using the exported boom geometry relation.",
    }
    return [
        ParameterSpec(
            name=name,
            group="geometry",
            unit="m",
            baseline_value=baseline_map[name],
            rel_step=FD_REL_STEP,
            abs_floor=1e-4,
            symbol=PARAMETER_SYMBOLS[name],
            evaluation_path="full_trim",
            lower_bound=lower_bounds[name],
            notes=notes_map.get(name, ""),
        )
        for name in GEOMETRY_PARAM_ORDER
    ]


def build_requirement_parameter_specs(context: BaselineContext) -> list[ParameterSpec]:
    baseline_map = context.requirement_values
    units = {
        "bank_entry_time_s": "s",
        "wall_clearance_m": "m",
        "turn_bank_deg": "deg",
        "static_margin_min": "%MAC",
        "max_cl_nominal": "-",
        "max_roll_tau_s": "s",
    }
    abs_floors = {
        "bank_entry_time_s": 1e-4,
        "wall_clearance_m": 1e-4,
        "turn_bank_deg": 1e-3,
        "static_margin_min": 1e-2,
        "max_cl_nominal": 1e-4,
        "max_roll_tau_s": 1e-4,
    }
    lower_bounds = {
        "bank_entry_time_s": 0.0,
        "wall_clearance_m": 0.0,
        "turn_bank_deg": 0.0,
        "static_margin_min": 0.0,
        "max_cl_nominal": 0.0,
        "max_roll_tau_s": 0.0,
    }
    evaluation_path = {
        "bank_entry_time_s": "derived_only",
        "wall_clearance_m": "derived_only",
        "turn_bank_deg": "derived_only",
        "static_margin_min": "derived_only",
        "max_cl_nominal": "full_trim",
        "max_roll_tau_s": "derived_only",
    }
    return [
        ParameterSpec(
            name=name,
            group="requirement",
            unit=units[name],
            baseline_value=float(baseline_map[name]),
            rel_step=FD_REL_STEP,
            abs_floor=abs_floors[name],
            symbol=PARAMETER_SYMBOLS[name],
            evaluation_path=evaluation_path[name],
            lower_bound=lower_bounds[name],
        )
        for name in REQUIREMENT_PARAM_ORDER
    ]


def build_quantity_specs() -> list[QuantitySpec]:
    return [
        QuantitySpec(
            name=name,
            unit=QUANTITY_UNITS[name],
            symbol=QUANTITY_SYMBOLS[name],
            q_floor=QUANTITY_FLOORS[name],
        )
        for name in ALL_QUANTITY_ORDER
    ]


def compute_fd_step(parameter_spec: ParameterSpec) -> float:
    return max(
        abs(parameter_spec.baseline_value) * parameter_spec.rel_step,
        parameter_spec.abs_floor,
    )


def build_step_ladder(
    parameter_spec: ParameterSpec,
    base_step: float,
    scheme: str,
) -> list[float]:
    lower_bound = parameter_spec.lower_bound
    step_sizes: list[float] = []
    for multiplier in FD_STEP_LADDER_MULTIPLIERS:
        step_size = float(base_step) * float(multiplier)
        if (
            scheme == "central"
            and lower_bound is not None
            and parameter_spec.baseline_value - step_size <= lower_bound
        ):
            continue
        step_sizes.append(step_size)
    if not step_sizes:
        step_sizes.append(float(base_step))
    return step_sizes


def _difference_order(scheme: str) -> int:
    if scheme == "central":
        return 2
    return 1


def _richardson_reference(
    coarse_step: float,
    coarse_derivative: float,
    fine_step: float,
    fine_derivative: float,
    scheme: str,
) -> float:
    order = _difference_order(scheme)
    step_ratio = coarse_step / max(fine_step, 1e-30)
    denominator = max(step_ratio**order - 1.0, 1e-12)
    return fine_derivative + (fine_derivative - coarse_derivative) / denominator


def _compose_derivative_unit(quantity_unit: str, parameter_unit: str) -> str:
    if quantity_unit == "-" and parameter_unit == "-":
        return "-"
    if quantity_unit == "-":
        return f"1/{parameter_unit}"
    if parameter_unit == "-":
        return quantity_unit
    return f"{quantity_unit}/{parameter_unit}"


def normalize_sensitivity(
    parameter_value: float,
    quantity_value: float,
    dq_dp: float,
    q_floor: float,
) -> float:
    if not math.isfinite(dq_dp):
        return float("nan")
    denominator = max(abs(quantity_value), q_floor)
    return (parameter_value / denominator) * dq_dp


def classify_interpretation(
    normalized_values: list[float],
    stable_flags: list[bool],
) -> str:
    stable_values = [
        abs(value)
        for value, is_stable in zip(normalized_values, stable_flags, strict=False)
        if is_stable and math.isfinite(value)
    ]
    if not stable_values:
        return "numerically unstable estimate"
    max_value = max(stable_values)
    if max_value >= 1.0:
        return "strong local driver"
    if max_value >= 0.2:
        return "moderate local driver"
    return "weak local driver"


@contextmanager
def suppress_solver_output() -> Any:
    try:
        stdout_fd = sys.stdout.fileno()
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        yield
        return

    saved_stdout = os.dup(stdout_fd)
    saved_stderr = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_RDWR)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(devnull_fd, stdout_fd)
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        os.dup2(saved_stdout, stdout_fd)
        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stdout)
        os.close(saved_stderr)
        os.close(devnull_fd)


def _build_seed_candidate(
    context: BaselineContext,
    geometry: nausicaa.GeometryVars,
) -> nausicaa.Candidate:
    return nausicaa.Candidate(
        candidate_id=int(context.selected_candidate_id),
        objective=float(context.workbook_baseline_values.get("objective", float("nan"))),
        wing_span_m=float(geometry.wing_span_m),
        wing_chord_m=float(geometry.wing_chord_m),
        tail_arm_m=float(geometry.tail_arm_m),
        htail_span_m=float(geometry.htail_span_m),
        vtail_height_m=float(geometry.vtail_height_m),
        alpha_deg=float(context.trim_seed["alpha_deg"]),
        delta_a_deg=float(context.trim_seed["delta_a_deg"]),
        delta_e_deg=float(context.trim_seed["delta_e_deg"]),
        delta_r_deg=float(context.trim_seed["delta_r_deg"]),
        sink_rate_mps=float(
            context.workbook_baseline_values.get("sink_rate_mps", float("nan"))
        ),
        l_over_d=_lookup_numeric(context.summary_map, "L_over_D"),
        mass_total_kg=float(
            context.workbook_baseline_values.get("mass_total_kg", float("nan"))
        ),
        static_margin=float(
            context.workbook_baseline_values.get("static_margin", float("nan"))
        ),
        vh=_lookup_numeric(context.summary_map, "tail_volume_horizontal"),
        vv=_lookup_numeric(context.summary_map, "tail_volume_vertical"),
        roll_tau_s=float(context.workbook_baseline_values.get("roll_tau_s", float("nan"))),
        roll_rate_ss_radps=_lookup_numeric(context.summary_map, "roll_rate_ss_radps"),
        roll_accel0_rad_s2=_lookup_numeric(context.summary_map, "roll_accel0_rad_s2"),
        max_servo_util=_lookup_numeric(
            context.summary_map,
            "max_servo_utilization",
            default=_lookup_numeric(context.selected_row, "max_servo_util"),
        ),
        objective_weights=asdict(context.objective_weights),
        objective_scales=asdict(context.objective_scales),
        airfoil_label=str(context.summary_map.get("airfoil_model", "")),
    )


def _compute_turn_metrics(
    cfg: nausicaa.Config,
    workflow_cfg: nausicaa.WorkflowConfig,
    state: dict[str, float],
) -> dict[str, float]:
    wing_span_m = float(state["wing_span_m"])
    wing_area_m2 = float(state["wing_area_m2"])
    mass_total_kg = float(state["mass_total_kg"])
    lift_n = max(float(state["lift_n"]), 0.0)
    cl_delta_a_proxy = abs(float(state["cl_delta_a_proxy"]))
    clp_mag = max(abs(float(state["clp_mag"])), 1e-5)
    ixx_kg_m2 = max(float(state["ixx_kg_m2"]), 1e-8)

    delta_a_turn_cmd_deg = float(workflow_cfg.turn_deflection_util) * float(
        cfg.delta_a_max_deg
    )
    delta_a_turn_rate_limited_deg = float(workflow_cfg.servo_rate_deg_s) * float(
        cfg.bank_entry_time_s
    )
    if bool(nausicaa.INCLUDE_SERVO_RATE_IN_BANK_ENTRY):
        delta_a_turn_eff_deg = min(
            delta_a_turn_cmd_deg,
            delta_a_turn_rate_limited_deg,
        )
    else:
        delta_a_turn_eff_deg = delta_a_turn_cmd_deg

    q_dyn_agility = 0.5 * float(cfg.rho) * float(cfg.v_turn_mps) ** 2
    delta_a_turn_rad = float(np.radians(delta_a_turn_eff_deg))
    roll_tau_proxy_s = (
        2.0
        * ixx_kg_m2
        * float(cfg.v_turn_mps)
        / max(q_dyn_agility * wing_area_m2 * (wing_span_m**2) * clp_mag, 1e-8)
    )
    roll_rate_ss_turn_radps = (
        2.0
        * float(cfg.v_turn_mps)
        / max(wing_span_m, 1e-8)
        * cl_delta_a_proxy
        * delta_a_turn_rad
        / clp_mag
    )
    tau_turn_eff_s = math.sqrt(roll_tau_proxy_s**2 + TURN_TAU_FLOOR_S**2)
    bank_entry_phi_achieved_rad = roll_rate_ss_turn_radps * (
        float(cfg.bank_entry_time_s)
        - tau_turn_eff_s
        * (1.0 - math.exp(-float(cfg.bank_entry_time_s) / tau_turn_eff_s))
    )
    bank_entry_margin_deg = float(
        np.degrees(bank_entry_phi_achieved_rad - np.radians(float(cfg.turn_bank_deg)))
    )

    turn_radius_allow_raw_m = 0.5 * float(cfg.arena_width_m) - (
        0.5 * wing_span_m + float(cfg.wall_clearance_m)
    )
    turn_radius_allow_m = max(turn_radius_allow_raw_m, 1e-6)
    a_lat_req_mps2 = float(cfg.v_turn_mps) ** 2 / turn_radius_allow_m
    a_lat_ach_mps2 = (
        lift_n
        * math.sin(math.radians(float(cfg.turn_bank_deg)))
        / max(mass_total_kg, 1e-8)
    )
    turn_radius_ach_m = float(cfg.v_turn_mps) ** 2 / max(a_lat_ach_mps2, 1e-6)
    turn_footprint_margin_m = 0.5 * float(cfg.arena_width_m) - (
        turn_radius_ach_m + 0.5 * wing_span_m + float(cfg.wall_clearance_m)
    )
    agility_lateral_margin_mps2 = a_lat_ach_mps2 - a_lat_req_mps2

    return {
        "bank_entry_margin_deg": bank_entry_margin_deg,
        "turn_radius_allow_m": turn_radius_allow_m,
        "turn_radius_ach_m": turn_radius_ach_m,
        "turn_footprint_margin_m": turn_footprint_margin_m,
        "agility_lateral_margin_mps2": agility_lateral_margin_mps2,
        "a_lat_req_mps2": a_lat_req_mps2,
        "a_lat_ach_mps2": a_lat_ach_mps2,
        "roll_tau_proxy_s": roll_tau_proxy_s,
        "roll_rate_ss_turn_radps": roll_rate_ss_turn_radps,
    }


def _empty_evaluation_result(note: str) -> EvaluationResult:
    metric_map = {name: float("nan") for name in ALL_QUANTITY_ORDER}
    return EvaluationResult(success=False, metrics=metric_map, state={}, notes=[note])


def _roundoff_signal_floor(
    quantity_floor: float,
    *values: float,
) -> float:
    finite_values = [abs(value) for value in values if math.isfinite(value)]
    scale = max([quantity_floor, *finite_values], default=quantity_floor)
    return FD_ROUNDOFF_SAFETY * FD_ROUNDOFF_REL_SCALE * max(scale, quantity_floor)


def _roundoff_derivative_proxy(
    roundoff_floor: float,
    step_size: float,
    scheme: str,
) -> float:
    denominator = step_size if scheme == "forward" else 2.0 * step_size
    return float(roundoff_floor) / max(float(denominator), 1e-30)


def evaluate_full_trim(
    context: BaselineContext,
    geometry: nausicaa.GeometryVars,
    requirement_values: dict[str, float],
) -> EvaluationResult:
    cfg = _build_cfg(
        summary_map=context.summary_map,
        design_points_map=context.design_points_map,
        requirement_values=requirement_values,
    )
    workflow_cfg = context.workflow_cfg
    seed_candidate = _build_seed_candidate(context, geometry)
    policy = nausicaa.get_constraint_policy(cfg)

    opti = asb.Opti()
    try:
        airfoil, _airfoil_label = nausicaa.get_reference_airfoil_cached()
        airframe_bundle = nausicaa.get_airframe_bundle_cached(
            geometry=geometry,
            airfoil=airfoil,
            cfg=cfg,
        )
        wing = airframe_bundle.wing
        htail = airframe_bundle.htail
        vtail = airframe_bundle.vtail
        airplane_base = airframe_bundle.airplane_base
        htail_chord_m = airframe_bundle.htail_chord_m
        vtail_chord_m = airframe_bundle.vtail_chord_m

        mass_props, total_mass = nausicaa.build_mass_model(
            opti=opti,
            wing=wing,
            htail=htail,
            vtail=vtail,
            wing_chord_m=float(geometry.wing_chord_m),
            tail_arm_m=float(geometry.tail_arm_m),
            boom_end_x_m=float(airframe_bundle.boom_end_x_m),
            vtail_chord_m=float(vtail_chord_m),
            cfg=cfg,
        )
        mass_penalty_expr = nausicaa.objective_mass_penalty_mass_kg(
            mass_props=mass_props,
            total_mass_kg=total_mass.mass,
        )

        u_a_min, u_a_max = nausicaa.servo_command_bounds(
            float(cfg.delta_a_min_deg),
            float(cfg.delta_a_max_deg),
            1.0,
            0.0,
            float(workflow_cfg.max_trim_util_fraction),
        )
        u_e_min, u_e_max = nausicaa.servo_command_bounds(
            float(cfg.delta_e_min_deg),
            float(cfg.delta_e_max_deg),
            1.0,
            0.0,
            float(workflow_cfg.max_trim_util_fraction),
        )
        u_r_min, u_r_max = nausicaa.servo_command_bounds(
            float(cfg.delta_r_min_deg),
            float(cfg.delta_r_max_deg),
            1.0,
            0.0,
            float(workflow_cfg.max_trim_util_fraction),
        )
        rate_limit_deg = (
            float(workflow_cfg.rate_util_fraction)
            * float(workflow_cfg.servo_rate_deg_s)
            * float(workflow_cfg.nom_trim_time_s)
        )

        alpha_trim_deg = opti.variable(
            init_guess=float(
                np.clip(
                    seed_candidate.alpha_deg,
                    float(cfg.alpha_min_deg),
                    float(cfg.alpha_max_deg),
                )
            ),
            lower_bound=float(cfg.alpha_min_deg),
            upper_bound=float(cfg.alpha_max_deg),
        )
        u_a_deg = opti.variable(
            init_guess=float(np.clip(seed_candidate.delta_a_deg, u_a_min, u_a_max))
        )
        u_e_deg = opti.variable(
            init_guess=float(np.clip(seed_candidate.delta_e_deg, u_e_min, u_e_max))
        )
        u_r_deg = opti.variable(
            init_guess=float(np.clip(seed_candidate.delta_r_deg, u_r_min, u_r_max))
        )

        airplane = airplane_base.with_control_deflections(
            {
                "aileron": u_a_deg,
                "elevator": u_e_deg,
                "rudder": u_r_deg,
            }
        )
        trim_metrics = nausicaa.build_trim_constraints_and_metrics(
            opti=opti,
            airplane=airplane,
            xyz_ref=[total_mass.x_cg, total_mass.y_cg, total_mass.z_cg],
            velocity_mps=float(cfg.v_nom_mps),
            alpha_deg=alpha_trim_deg,
            mass_kg=total_mass.mass,
            mode="nominal",
            bank_angle_deg=0.0,
            lift_k=1.0,
            cl_cap=float(requirement_values["max_cl_nominal"]),
            enforce_lateral_trim=bool(cfg.nom_lateral_trim),
            use_coordinated_turn=False,
            atmosphere=asb.Atmosphere(altitude=0.0),
            cfg=cfg,
            policy=policy,
        )
        aero = trim_metrics["aero"]

        sink_rate_expr = aero["D"] * float(cfg.v_nom_mps) / np.maximum(
            total_mass.mass * float(cfg.g),
            1e-8,
        )
        trim_penalty_expr = u_e_deg**2 + 0.3 * u_r_deg**2 + 0.15 * u_a_deg**2
        opti.minimize(
            sink_rate_expr + float(nausicaa.CONTROL_TRIM_WEIGHT) * trim_penalty_expr
        )
        opti.subject_to(
            [
                *trim_metrics["constraints"],
                opti.bounded(u_a_min, u_a_deg, u_a_max),
                opti.bounded(u_e_min, u_e_deg, u_e_max),
                opti.bounded(u_r_min, u_r_deg, u_r_max),
                opti.bounded(
                    -rate_limit_deg,
                    u_a_deg - float(seed_candidate.delta_a_deg),
                    rate_limit_deg,
                ),
                opti.bounded(
                    -rate_limit_deg,
                    u_e_deg - float(seed_candidate.delta_e_deg),
                    rate_limit_deg,
                ),
                opti.bounded(
                    -rate_limit_deg,
                    u_r_deg - float(seed_candidate.delta_r_deg),
                    rate_limit_deg,
                ),
            ]
        )
        opti.solver(
            "ipopt",
            {
                "print_time": False,
                "verbose": False,
                "ipopt.print_level": 0,
                "ipopt.sb": "yes",
            },
            {
                "max_iter": 500,
                "hessian_approximation": "limited-memory",
                "check_derivatives_for_naninf": "no",
                "print_level": 0,
                "sb": "yes",
                **nausicaa.ipopt_verbosity_options(),
            },
        )
        with suppress_solver_output():
            solution = opti.solve()
    except Exception as exc:
        return _empty_evaluation_result(
            f"full-trim solve failed for geometry/requirement state: {exc}"
        )

    aero_num = solution(aero)
    alpha_num = float(nausicaa.to_scalar(solution(alpha_trim_deg)))
    delta_a_num = float(nausicaa.to_scalar(solution(u_a_deg)))
    delta_e_num = float(nausicaa.to_scalar(solution(u_e_deg)))
    delta_r_num = float(nausicaa.to_scalar(solution(u_r_deg)))
    mass_total_kg = float(nausicaa.to_scalar(solution(total_mass.mass)))
    total_cg_x_m = float(nausicaa.to_scalar(solution(total_mass.x_cg)))
    total_cg_y_m = float(nausicaa.to_scalar(solution(total_mass.y_cg)))
    total_cg_z_m = float(nausicaa.to_scalar(solution(total_mass.z_cg)))
    wing_area_m2 = float(nausicaa.to_scalar(solution(wing.area())))
    wing_mac_m = float(nausicaa.to_scalar(solution(wing.mean_aerodynamic_chord())))
    ixx_kg_m2 = abs(
        float(nausicaa.to_scalar(solution(total_mass.inertia_tensor[0, 0])))
    )
    lift_n = float(nausicaa.to_scalar(aero_num["L"]))
    drag_n = float(nausicaa.to_scalar(aero_num["D"]))
    cl_num = float(nausicaa.to_scalar(aero_num["CL"]))
    cl_lat_num = float(nausicaa.to_scalar(aero_num["Cl"]))
    cn_lat_num = float(nausicaa.to_scalar(aero_num["Cn"]))
    x_np_m = float(nausicaa.to_scalar(aero_num["x_np"]))
    clp_mag = max(abs(float(nausicaa.to_scalar(aero_num["Clp"]))), 1e-5)
    sink_rate_mps = drag_n * float(cfg.v_nom_mps) / max(
        mass_total_kg * float(cfg.g),
        1e-8,
    )
    static_margin = 100.0 * (x_np_m - total_cg_x_m) / max(wing_mac_m, 1e-8)
    lateral_residual = float(np.hypot(cl_lat_num, cn_lat_num))

    cl_delta_a_proxy = float(
        nausicaa.to_scalar(
            nausicaa.aileron_effectiveness_proxy(
                aero=aero_num,
                eta_inboard=float(cfg.aileron_eta_inboard),
                eta_outboard=float(cfg.aileron_eta_outboard),
                chord_fraction=float(cfg.aileron_chord_fraction),
            )
        )
    )
    cl_delta_a_fd = float("nan")
    cl_delta_a_mag = abs(cl_delta_a_proxy)
    delta_a_max_rad = float(np.radians(float(cfg.delta_a_max_deg)))
    q_dyn = 0.5 * float(cfg.rho) * float(cfg.v_nom_mps) ** 2
    roll_rate_ss_radps = (
        2.0
        * float(cfg.v_nom_mps)
        / max(float(geometry.wing_span_m), 1e-8)
        * cl_delta_a_mag
        * delta_a_max_rad
        / clp_mag
    )
    roll_tau_s = (
        2.0
        * max(ixx_kg_m2, 1e-8)
        * float(cfg.v_nom_mps)
        / max(q_dyn * wing_area_m2 * (float(geometry.wing_span_m) ** 2) * clp_mag, 1e-8)
    )

    struct_force_n = max(lift_n, 0.0)
    wing_struct = nausicaa.struct_tip_deflection_proxy_composite(
        total_force_n=struct_force_n,
        span_m=float(geometry.wing_span_m),
        chord_m=float(geometry.wing_chord_m),
        foam_thickness_m=float(nausicaa.WING_THICKNESS_M),
        e_foam_pa=float(nausicaa.WING_E_SECANT_PA),
        e_foam_scale=1.0,
        allow_frac=float(nausicaa.WING_DEFLECTION_ALLOW_FRAC),
        thickness_scale=1.0,
        include_spar=bool(nausicaa.WING_SPAR_ENABLE),
        spar_od_m=float(nausicaa.WING_SPAR_OD_M),
        spar_id_m=float(nausicaa.WING_SPAR_ID_M),
        e_spar_pa=float(nausicaa.WING_SPAR_E_FLEX_PA),
        spar_z_from_lower_m=float(nausicaa.WING_SPAR_Z_FROM_LOWER_M),
        include_tape=bool(nausicaa.TAPE_ENABLE_WING),
        tape_width_m=float(
            2.0
            * nausicaa.TAPE_WIDTH_M
            * nausicaa.WING_TAPE_SPAN_FRACTION_OF_SEMISPAN
        ),
        tape_thickness_m=float(nausicaa.TAPE_THICKNESS_M),
        e_tape_pa=float(nausicaa.TAPE_EFFICIENCY * nausicaa.TAPE_E_EFFECTIVE_PA),
    )
    htail_struct = nausicaa.struct_tip_deflection_proxy_composite(
        total_force_n=float(nausicaa.HT_LOAD_FRACTION) * struct_force_n,
        span_m=float(geometry.htail_span_m),
        chord_m=float(nausicaa.to_scalar(solution(htail_chord_m))),
        foam_thickness_m=float(nausicaa.TAIL_THICKNESS_M),
        e_foam_pa=float(nausicaa.HTAIL_E_SECANT_PA),
        e_foam_scale=1.0,
        allow_frac=float(nausicaa.HT_DEFLECTION_ALLOW_FRAC),
        thickness_scale=1.0,
        include_spar=False,
        spar_od_m=float(nausicaa.WING_SPAR_OD_M),
        spar_id_m=float(nausicaa.WING_SPAR_ID_M),
        e_spar_pa=float(nausicaa.WING_SPAR_E_FLEX_PA),
        spar_z_from_lower_m=0.0,
        include_tape=bool(nausicaa.TAPE_ENABLE_TAIL),
        tape_width_m=float(nausicaa.TAPE_WIDTH_M),
        tape_thickness_m=float(nausicaa.TAPE_THICKNESS_M),
        e_tape_pa=float(nausicaa.TAPE_EFFICIENCY * nausicaa.TAPE_E_EFFECTIVE_PA),
    )

    wing_deflection_over_allow = float(
        nausicaa.to_scalar(
            nausicaa.stable_softplus(
                float(wing_struct["delta_tip_m"]) - float(wing_struct["delta_allow_m"]),
                float(cfg.softplus_k),
            )
        )
    ) / max(float(wing_struct["delta_allow_m"]), 1e-6)
    htail_deflection_over_allow = float(
        nausicaa.to_scalar(
            nausicaa.stable_softplus(
                float(htail_struct["delta_tip_m"]) - float(htail_struct["delta_allow_m"]),
                float(cfg.softplus_k),
            )
        )
    ) / max(float(htail_struct["delta_allow_m"]), 1e-6)

    mass_penalty_mass_kg = float(nausicaa.to_scalar(solution(mass_penalty_expr)))
    trim_effort_deg2 = (
        delta_e_num**2 + 0.3 * delta_r_num**2 + 0.15 * delta_a_num**2
    )
    objective_value, objective_terms = nausicaa.build_dimensionless_objective_terms(
        sink_rate_mps=sink_rate_mps,
        mass_penalty_kg=mass_penalty_mass_kg,
        trim_effort_deg2=trim_effort_deg2,
        wing_deflection_over_allow=wing_deflection_over_allow,
        htail_deflection_over_allow=htail_deflection_over_allow,
        roll_tau_s=roll_tau_s,
        scales=context.objective_scales,
        weights=context.objective_weights,
    )
    objective_num = float(nausicaa.to_scalar(objective_value))
    turn_metrics = _compute_turn_metrics(
        cfg=cfg,
        workflow_cfg=workflow_cfg,
        state={
            "wing_span_m": float(geometry.wing_span_m),
            "wing_area_m2": wing_area_m2,
            "mass_total_kg": mass_total_kg,
            "lift_n": lift_n,
            "cl_delta_a_proxy": cl_delta_a_proxy,
            "clp_mag": clp_mag,
            "ixx_kg_m2": ixx_kg_m2,
        },
    )

    elevator_util = abs(delta_e_num) / max(abs(float(cfg.delta_e_max_deg)), 1e-8)
    metrics = {
        "objective": objective_num,
        "sink_rate_mps": sink_rate_mps,
        "mass_total_kg": mass_total_kg,
        "roll_tau_s": roll_tau_s,
        "static_margin": static_margin,
        "nom_cl_margin_to_cap": float(requirement_values["max_cl_nominal"]) - cl_num,
        "nom_util_e": elevator_util,
        "nom_lateral_residual": lateral_residual,
        "static_margin_min_margin": static_margin
        - float(requirement_values["static_margin_min"]),
        "roll_tau_limit_margin": float(requirement_values["max_roll_tau_s"]) - roll_tau_s,
        "elevator_util_margin": 1.0 - elevator_util,
        "bank_entry_margin_deg": float(turn_metrics["bank_entry_margin_deg"]),
        "turn_radius_allow_m": float(turn_metrics["turn_radius_allow_m"]),
        "turn_radius_ach_m": float(turn_metrics["turn_radius_ach_m"]),
        "turn_footprint_margin_m": float(turn_metrics["turn_footprint_margin_m"]),
        "agility_lateral_margin_mps2": float(
            turn_metrics["agility_lateral_margin_mps2"]
        ),
    }
    state = {
        "alpha_deg": alpha_num,
        "delta_a_deg": delta_a_num,
        "delta_e_deg": delta_e_num,
        "delta_r_deg": delta_r_num,
        "lift_n": lift_n,
        "drag_n": drag_n,
        "wing_span_m": float(geometry.wing_span_m),
        "wing_area_m2": wing_area_m2,
        "wing_mac_m": wing_mac_m,
        "mass_total_kg": mass_total_kg,
        "ixx_kg_m2": ixx_kg_m2,
        "cl_delta_a_proxy": cl_delta_a_proxy,
        "cl_delta_a_mag": cl_delta_a_mag,
        "clp_mag": clp_mag,
        "a_lat_req_mps2": float(turn_metrics["a_lat_req_mps2"]),
        "a_lat_ach_mps2": float(turn_metrics["a_lat_ach_mps2"]),
        "roll_rate_ss_radps": roll_rate_ss_radps,
        "objective_J_sink": float(nausicaa.to_scalar(objective_terms["J_sink"])),
        "objective_J_mass": float(nausicaa.to_scalar(objective_terms["J_mass"])),
    }
    return EvaluationResult(success=True, metrics=metrics, state=state, notes=[])


def evaluate_with_geometry_override(
    context: BaselineContext,
    parameter_name: str,
    value: float,
) -> EvaluationResult:
    geometry = replace(context.geometry, **{parameter_name: float(value)})
    return evaluate_full_trim(
        context=context,
        geometry=geometry,
        requirement_values=context.requirement_values,
    )


def evaluate_with_requirement_override(
    context: BaselineContext,
    baseline_result: EvaluationResult,
    parameter_name: str,
    value: float,
) -> EvaluationResult:
    requirement_values = dict(context.requirement_values)
    requirement_values[parameter_name] = float(value)
    if parameter_name == "max_cl_nominal":
        return evaluate_full_trim(
            context=context,
            geometry=context.geometry,
            requirement_values=requirement_values,
        )

    if not baseline_result.success:
        return _empty_evaluation_result(
            "derived-only requirement evaluation requested without a valid baseline"
        )

    cfg = _build_cfg(
        summary_map=context.summary_map,
        design_points_map=context.design_points_map,
        requirement_values=requirement_values,
    )
    metrics = dict(baseline_result.metrics)
    turn_metrics = _compute_turn_metrics(
        cfg=cfg,
        workflow_cfg=context.workflow_cfg,
        state=baseline_result.state,
    )
    metrics["static_margin_min_margin"] = (
        float(metrics["static_margin"]) - float(requirement_values["static_margin_min"])
    )
    metrics["roll_tau_limit_margin"] = (
        float(requirement_values["max_roll_tau_s"]) - float(metrics["roll_tau_s"])
    )
    metrics["bank_entry_margin_deg"] = float(turn_metrics["bank_entry_margin_deg"])
    metrics["turn_radius_allow_m"] = float(turn_metrics["turn_radius_allow_m"])
    metrics["turn_radius_ach_m"] = float(turn_metrics["turn_radius_ach_m"])
    metrics["turn_footprint_margin_m"] = float(turn_metrics["turn_footprint_margin_m"])
    metrics["agility_lateral_margin_mps2"] = float(
        turn_metrics["agility_lateral_margin_mps2"]
    )
    return EvaluationResult(
        success=True,
        metrics=metrics,
        state=dict(baseline_result.state),
        notes=[],
    )


def _adjust_fd_scheme(parameter_spec: ParameterSpec) -> tuple[str, float, list[str]]:
    step = compute_fd_step(parameter_spec)
    notes: list[str] = []
    scheme = "central"
    lower_bound = parameter_spec.lower_bound
    if lower_bound is not None and parameter_spec.baseline_value - step <= lower_bound:
        shrunk_step = 0.5 * step
        if parameter_spec.baseline_value - shrunk_step > lower_bound:
            step = shrunk_step
            notes.append("Centered step halved once to stay above the lower bound.")
        else:
            scheme = "forward"
            notes.append(
                "Forward difference used because the centered step violates the lower bound."
            )
    return scheme, step, notes


def _evaluate_parameter_value(
    context: BaselineContext,
    baseline_result: EvaluationResult,
    parameter_spec: ParameterSpec,
    value: float,
) -> EvaluationResult:
    if parameter_spec.group == "geometry":
        return evaluate_with_geometry_override(
            context=context,
            parameter_name=parameter_spec.name,
            value=value,
        )
    return evaluate_with_requirement_override(
        context=context,
        baseline_result=baseline_result,
        parameter_name=parameter_spec.name,
        value=value,
    )


def _build_step_candidates(
    step_sizes: list[float],
    evaluation_points: dict[str, EvaluationResult],
    quantity_name: str,
    baseline_quantity: float,
    quantity_floor: float,
    scheme: str,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for level, step_size in enumerate(step_sizes):
        derivative_estimate = float("nan")
        signal_amplitude = float("nan")
        roundoff_floor = float("nan")
        roundoff_limited = False

        if scheme == "central":
            minus_eval = evaluation_points[f"minus_{level}"]
            plus_eval = evaluation_points[f"plus_{level}"]
            if (
                minus_eval.success
                and plus_eval.success
                and math.isfinite(minus_eval.metrics[quantity_name])
                and math.isfinite(plus_eval.metrics[quantity_name])
            ):
                derivative_estimate = (
                    plus_eval.metrics[quantity_name] - minus_eval.metrics[quantity_name]
                ) / (2.0 * step_size)
                signal_amplitude = abs(
                    plus_eval.metrics[quantity_name] - minus_eval.metrics[quantity_name]
                )
                roundoff_floor = _roundoff_signal_floor(
                    quantity_floor,
                    baseline_quantity,
                    plus_eval.metrics[quantity_name],
                    minus_eval.metrics[quantity_name],
                )
                roundoff_limited = signal_amplitude <= roundoff_floor
        else:
            plus_eval = evaluation_points[f"plus_{level}"]
            if (
                plus_eval.success
                and math.isfinite(plus_eval.metrics[quantity_name])
                and math.isfinite(baseline_quantity)
            ):
                derivative_estimate = (
                    plus_eval.metrics[quantity_name] - baseline_quantity
                ) / step_size
                signal_amplitude = abs(
                    plus_eval.metrics[quantity_name] - baseline_quantity
                )
                roundoff_floor = _roundoff_signal_floor(
                    quantity_floor,
                    baseline_quantity,
                    plus_eval.metrics[quantity_name],
                )
                roundoff_limited = signal_amplitude <= roundoff_floor

        candidates.append(
            {
                "step_level": level,
                "step_size": float(step_size),
                "derivative_estimate": float(derivative_estimate),
                "signal_amplitude": float(signal_amplitude),
                "roundoff_floor": float(roundoff_floor),
                "roundoff_limited": bool(roundoff_limited),
            }
        )
    return candidates


def _estimate_reference_derivative(
    candidates: list[dict[str, Any]],
    scheme: str,
) -> tuple[float, str]:
    finite_candidates = [
        candidate
        for candidate in candidates
        if math.isfinite(candidate["derivative_estimate"])
    ]
    nonroundoff_candidates = [
        candidate
        for candidate in finite_candidates
        if not candidate["roundoff_limited"]
    ]
    reference_pool = nonroundoff_candidates
    method = "richardson_smallest_nonroundoff_pair"
    if len(reference_pool) < 2:
        reference_pool = finite_candidates
        method = "richardson_smallest_finite_pair_fallback"
    if len(reference_pool) >= 2:
        reference_pool = sorted(reference_pool, key=lambda row: row["step_size"])
        fine_candidate = reference_pool[0]
        coarse_candidate = reference_pool[1]
        reference = _richardson_reference(
            coarse_step=float(coarse_candidate["step_size"]),
            coarse_derivative=float(coarse_candidate["derivative_estimate"]),
            fine_step=float(fine_candidate["step_size"]),
            fine_derivative=float(fine_candidate["derivative_estimate"]),
            scheme=scheme,
        )
        return float(reference), method
    if finite_candidates:
        return (
            float(min(finite_candidates, key=lambda row: row["step_size"])[
                "derivative_estimate"
            ]),
            "single_finite_candidate_fallback",
        )
    return float("nan"), "no_finite_reference"


def _annotate_candidate_errors(
    candidates: list[dict[str, Any]],
    reference_derivative: float,
    scheme: str,
) -> None:
    for candidate in candidates:
        derivative_estimate = float(candidate["derivative_estimate"])
        if math.isfinite(derivative_estimate) and math.isfinite(reference_derivative):
            absolute_error = abs(derivative_estimate - reference_derivative)
            relative_error = absolute_error / max(abs(reference_derivative), 1e-9)
        else:
            absolute_error = float("nan")
            relative_error = float("nan")
        roundoff_error_proxy = _roundoff_derivative_proxy(
            roundoff_floor=float(candidate["roundoff_floor"]),
            step_size=float(candidate["step_size"]),
            scheme=scheme,
        )
        if math.isfinite(absolute_error):
            total_error_proxy = absolute_error + roundoff_error_proxy
        else:
            total_error_proxy = float("nan")
        candidate["absolute_error_estimate"] = float(absolute_error)
        candidate["relative_error_estimate"] = float(relative_error)
        candidate["roundoff_error_proxy"] = float(roundoff_error_proxy)
        candidate["total_error_proxy"] = float(total_error_proxy)


def _select_step_candidate(
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    viable_candidates = [
        candidate
        for candidate in candidates
        if math.isfinite(candidate["derivative_estimate"])
        and math.isfinite(candidate["absolute_error_estimate"])
        and not candidate["roundoff_limited"]
    ]
    if viable_candidates:
        selected_candidate = min(
            viable_candidates,
            key=lambda row: (
                float(row["absolute_error_estimate"]),
                float(row["step_size"]),
            ),
        )
        smaller_roundoff_exists = any(
            candidate["roundoff_limited"]
            and float(candidate["step_size"]) < float(selected_candidate["step_size"])
            for candidate in candidates
            if math.isfinite(candidate["derivative_estimate"])
        )
        if smaller_roundoff_exists:
            return (
                selected_candidate,
                "minimum_absolute_error_before_roundoff_onset",
            )
        return selected_candidate, "minimum_absolute_error_nonroundoff"

    finite_candidates = [
        candidate
        for candidate in candidates
        if math.isfinite(candidate["derivative_estimate"])
    ]
    if finite_candidates:
        return (
            max(finite_candidates, key=lambda row: float(row["step_size"])),
            "largest_step_selected_all_candidates_roundoff_fragile",
        )
    return None, "no_finite_derivative_available"


def compute_sensitivity_for_parameter(
    context: BaselineContext,
    baseline_result: EvaluationResult,
    parameter_spec: ParameterSpec,
    quantity_specs: list[QuantitySpec],
) -> tuple[list[SensitivityResult], list[StepSizeResult]]:
    scheme, step, scheme_notes = _adjust_fd_scheme(parameter_spec)
    step_sizes = build_step_ladder(
        parameter_spec=parameter_spec,
        base_step=step,
        scheme=scheme,
    )
    baseline_parameter = float(parameter_spec.baseline_value)
    baseline_values = baseline_result.metrics
    evaluation_points: dict[str, EvaluationResult] = {}
    notes = [note for note in (parameter_spec.notes, *scheme_notes) if note]

    for level, step_size in enumerate(step_sizes):
        if scheme == "central":
            offset_map = {
                f"minus_{level}": -step_size,
                f"plus_{level}": step_size,
            }
        else:
            offset_map = {f"plus_{level}": step_size}
        for label, offset in offset_map.items():
            value = baseline_parameter + offset
            evaluation = _evaluate_parameter_value(
                context=context,
                baseline_result=baseline_result,
                parameter_spec=parameter_spec,
                value=value,
            )
            evaluation_points[label] = evaluation
            if not evaluation.success:
                notes.extend(evaluation.notes)

    results: list[SensitivityResult] = []
    step_size_rows: list[StepSizeResult] = []
    for quantity_spec in quantity_specs:
        quantity_name = quantity_spec.name
        baseline_quantity = float(baseline_values[quantity_name])
        sensitivity_raw = float("nan")
        selected_step = float("nan")
        step_selection_reason = ""
        fd_stability_abs = float("nan")
        fd_stability_rel = float("nan")
        fd_stable = False
        row_notes = list(notes)
        candidate_data = _build_step_candidates(
            step_sizes=step_sizes,
            evaluation_points=evaluation_points,
            quantity_name=quantity_name,
            baseline_quantity=baseline_quantity,
            quantity_floor=quantity_spec.q_floor,
            scheme=scheme,
        )
        finite_candidates = [
            candidate
            for candidate in candidate_data
            if math.isfinite(candidate["derivative_estimate"])
        ]
        if finite_candidates:
            reference_derivative, error_reference_method = (
                _estimate_reference_derivative(
                    candidates=candidate_data,
                    scheme=scheme,
                )
            )
            _annotate_candidate_errors(
                candidates=candidate_data,
                reference_derivative=reference_derivative,
                scheme=scheme,
            )
            selected_candidate, step_selection_reason = _select_step_candidate(
                candidate_data
            )
            if selected_candidate is not None:
                sensitivity_raw = float(selected_candidate["derivative_estimate"])
                selected_step = float(selected_candidate["step_size"])
                fd_stability_abs = float(selected_candidate["absolute_error_estimate"])
                fd_stability_rel = float(selected_candidate["relative_error_estimate"])
                fd_stable = (
                    math.isfinite(fd_stability_abs)
                    and math.isfinite(fd_stability_rel)
                    and not bool(selected_candidate["roundoff_limited"])
                    and (
                        fd_stability_rel <= FD_STABLE_REL_TOL
                        or fd_stability_abs <= FD_STABLE_ABS_TOL
                    )
                )
                if error_reference_method == "richardson_smallest_nonroundoff_pair":
                    row_notes.append(
                        "Absolute error was estimated against a Richardson-extrapolated derivative from the two smallest non-roundoff steps."
                    )
                elif error_reference_method == "richardson_smallest_finite_pair_fallback":
                    row_notes.append(
                        "Absolute error was estimated against a Richardson-extrapolated derivative from the two smallest finite steps because no non-roundoff pair was available."
                    )
                elif error_reference_method == "single_finite_candidate_fallback":
                    row_notes.append(
                        "Only one finite derivative estimate was available; the step-size trade-off is weakly resolved."
                    )
                if any(
                    candidate["roundoff_limited"]
                    for candidate in candidate_data
                    if math.isfinite(candidate["derivative_estimate"])
                ):
                    row_notes.append(
                        "Round-off onset was detected on at least one smaller step in the evaluated ladder."
                    )
                else:
                    row_notes.append(
                        "No round-off onset was detected across the evaluated step ladder."
                    )
                if not fd_stable:
                    row_notes.append(
                        "The selected step remains numerically fragile under the current absolute/relative finite-difference stability tolerances."
                    )
                if (
                    step_selection_reason
                    == "minimum_absolute_error_before_roundoff_onset"
                ):
                    row_notes.append(
                        "The selected step is the minimum derivative-error point before the smaller steps enter the round-off-limited regime."
                    )
                elif (
                    step_selection_reason
                    == "minimum_absolute_error_nonroundoff"
                ):
                    row_notes.append(
                        "The selected step is the minimum derivative-error non-roundoff point within the evaluated ladder."
                    )
                elif (
                    step_selection_reason
                    == "largest_step_selected_all_candidates_roundoff_fragile"
                ):
                    row_notes.append(
                        "All finite ladder points were flagged as round-off fragile; the largest step was retained for robustness."
                    )
            else:
                row_notes.append(
                    "No finite derivative estimate was available across the evaluated step ladder."
                )
                error_reference_method = "no_finite_reference"
        else:
            error_reference_method = "no_finite_reference"
            row_notes.append(
                "Perturbed evaluations failed or produced non-finite quantities across the full evaluated step ladder."
            )

        normalized = normalize_sensitivity(
            parameter_value=baseline_parameter,
            quantity_value=baseline_quantity,
            dq_dp=sensitivity_raw,
            q_floor=quantity_spec.q_floor,
        )
        results.append(
            SensitivityResult(
                group=parameter_spec.group,
                parameter_name=parameter_spec.name,
                parameter_symbol=parameter_spec.symbol,
                baseline_parameter_value=baseline_parameter,
                step_used=selected_step if math.isfinite(selected_step) else step,
                quantity_name=quantity_name,
                quantity_symbol=quantity_spec.symbol,
                baseline_quantity_value=baseline_quantity,
                sensitivity_raw=sensitivity_raw,
                sensitivity_normalized=normalized,
                difference_scheme=scheme,
                fd_stability_abs=fd_stability_abs,
                fd_stability_rel=fd_stability_rel,
                fd_stable=bool(fd_stable),
                unit_raw=_compose_derivative_unit(
                    quantity_unit=quantity_spec.unit,
                    parameter_unit=parameter_spec.unit,
                ),
                step_selection_reason=step_selection_reason,
                notes="; ".join(dict.fromkeys(note for note in row_notes if note)),
                reeval_path=parameter_spec.evaluation_path,
            )
        )
        for candidate in candidate_data:
            step_size_rows.append(
                StepSizeResult(
                    group=parameter_spec.group,
                    parameter_name=parameter_spec.name,
                    parameter_symbol=parameter_spec.symbol,
                    baseline_parameter_value=baseline_parameter,
                    step_level=int(candidate["step_level"]),
                    quantity_name=quantity_name,
                    quantity_symbol=quantity_spec.symbol,
                    baseline_quantity_value=baseline_quantity,
                    difference_scheme=scheme,
                    step_size=float(candidate["step_size"]),
                    derivative_estimate=float(candidate["derivative_estimate"]),
                    normalized_sensitivity=normalize_sensitivity(
                        parameter_value=baseline_parameter,
                        quantity_value=baseline_quantity,
                        dq_dp=float(candidate["derivative_estimate"]),
                        q_floor=quantity_spec.q_floor,
                    ),
                    absolute_error_estimate=float(
                        candidate.get("absolute_error_estimate", float("nan"))
                    ),
                    relative_error_estimate=float(
                        candidate.get("relative_error_estimate", float("nan"))
                    ),
                    fd_stability_abs=float(
                        candidate.get("absolute_error_estimate", float("nan"))
                    ),
                    fd_stability_rel=float(
                        candidate.get("relative_error_estimate", float("nan"))
                    ),
                    fd_stable=(
                        math.isfinite(
                            float(candidate.get("absolute_error_estimate", float("nan")))
                        )
                        and math.isfinite(
                            float(candidate.get("relative_error_estimate", float("nan")))
                        )
                        and not bool(candidate["roundoff_limited"])
                        and (
                            float(candidate.get("relative_error_estimate", float("nan")))
                            <= FD_STABLE_REL_TOL
                            or float(
                                candidate.get("absolute_error_estimate", float("nan"))
                            )
                            <= FD_STABLE_ABS_TOL
                        )
                    ),
                    signal_amplitude=float(candidate["signal_amplitude"]),
                    roundoff_floor=float(candidate["roundoff_floor"]),
                    roundoff_error_proxy=float(
                        candidate.get("roundoff_error_proxy", float("nan"))
                    ),
                    total_error_proxy=float(
                        candidate.get("total_error_proxy", float("nan"))
                    ),
                    roundoff_limited=bool(candidate["roundoff_limited"]),
                    selected_for_final=math.isfinite(selected_step)
                    and math.isclose(
                        float(candidate["step_size"]),
                        float(selected_step),
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    ),
                    selected_step_size=selected_step,
                    unit_raw=_compose_derivative_unit(
                        quantity_unit=quantity_spec.unit,
                        parameter_unit=parameter_spec.unit,
                    ),
                    error_reference_method=error_reference_method,
                    selection_reason=step_selection_reason,
                    reeval_path=parameter_spec.evaluation_path,
                    notes="; ".join(dict.fromkeys(note for note in row_notes if note)),
                )
            )
    return results, step_size_rows


def _mismatch_rows(
    context: BaselineContext,
    baseline_result: EvaluationResult,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric_name in OVERLAP_METRIC_KEYS:
        workbook_value = _to_float(context.workbook_baseline_values.get(metric_name))
        reevaluated_value = _to_float(baseline_result.metrics.get(metric_name))
        delta_value = reevaluated_value - workbook_value
        rows.append(
            {
                "group": "workbook_delta",
                "name": metric_name,
                "label": QUANTITY_LABELS[metric_name],
                "symbol": QUANTITY_SYMBOLS[metric_name],
                "value": reevaluated_value,
                "unit": QUANTITY_UNITS[metric_name],
                "workbook_value": workbook_value,
                "delta_from_workbook": delta_value,
                "source": "baseline_sanity_check",
            }
        )
    return rows


def build_baseline_table(
    context: BaselineContext,
    baseline_result: EvaluationResult,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    geometry_values = {
        "wing_span_m": float(context.geometry.wing_span_m),
        "wing_chord_m": float(context.geometry.wing_chord_m),
        "tail_arm_m": float(context.geometry.tail_arm_m),
        "htail_span_m": float(context.geometry.htail_span_m),
        "vtail_height_m": float(context.geometry.vtail_height_m),
    }
    for name in GEOMETRY_PARAM_ORDER:
        rows.append(
            {
                "group": "geometry",
                "name": name,
                "label": PARAMETER_LABELS[name],
                "symbol": PARAMETER_SYMBOLS[name],
                "value": geometry_values[name],
                "unit": "m",
                "workbook_value": np.nan,
                "delta_from_workbook": np.nan,
                "source": "selected_design_geometry",
            }
        )

    requirement_units = {
        "bank_entry_time_s": "s",
        "wall_clearance_m": "m",
        "turn_bank_deg": "deg",
        "static_margin_min": "%MAC",
        "max_cl_nominal": "-",
        "max_roll_tau_s": "s",
    }
    for name in REQUIREMENT_PARAM_ORDER:
        rows.append(
            {
                "group": "requirement",
                "name": name,
                "label": PARAMETER_LABELS[name],
                "symbol": PARAMETER_SYMBOLS[name],
                "value": float(context.requirement_values[name]),
                "unit": requirement_units[name],
                "workbook_value": np.nan,
                "delta_from_workbook": np.nan,
                "source": "baseline_requirement_setting",
            }
        )

    for name in PRIMARY_QUANTITY_ORDER:
        rows.append(
            {
                "group": "primary_output",
                "name": name,
                "label": QUANTITY_LABELS[name],
                "symbol": QUANTITY_SYMBOLS[name],
                "value": float(baseline_result.metrics[name]),
                "unit": QUANTITY_UNITS[name],
                "workbook_value": np.nan,
                "delta_from_workbook": np.nan,
                "source": "deterministic_baseline_evaluation",
            }
        )

    for name in MARGIN_QUANTITY_ORDER:
        rows.append(
            {
                "group": "margin_output",
                "name": name,
                "label": QUANTITY_LABELS[name],
                "symbol": QUANTITY_SYMBOLS[name],
                "value": float(baseline_result.metrics[name]),
                "unit": QUANTITY_UNITS[name],
                "workbook_value": np.nan,
                "delta_from_workbook": np.nan,
                "source": "deterministic_baseline_evaluation",
            }
        )

    rows.extend(_mismatch_rows(context, baseline_result))
    return pd.DataFrame(rows)


def build_metadata_table(
    context: BaselineContext,
    baseline_result: EvaluationResult,
    sensitivity_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {"Key": "timestamp_utc", "Value": datetime.now(timezone.utc).isoformat()},
        {"Key": "script_path", "Value": str(Path(__file__).resolve())},
        {"Key": "workbook_path", "Value": str(context.workbook_path.resolve())},
        {"Key": "selected_candidate_id", "Value": context.selected_candidate_id},
        {"Key": "fd_rel_step", "Value": FD_REL_STEP},
        {"Key": "fd_step_ladder_levels", "Value": FD_STEP_LADDER_LEVELS},
        {
            "Key": "fd_step_ladder_multipliers",
            "Value": ",".join(str(value) for value in FD_STEP_LADDER_MULTIPLIERS),
        },
        {"Key": "fd_stable_rel_tol", "Value": FD_STABLE_REL_TOL},
        {"Key": "fd_stable_abs_tol", "Value": FD_STABLE_ABS_TOL},
        {"Key": "fd_roundoff_safety", "Value": FD_ROUNDOFF_SAFETY},
        {"Key": "fd_roundoff_rel_scale", "Value": FD_ROUNDOFF_REL_SCALE},
        {
            "Key": "step_selection_basis",
            "Value": (
                "Selected step minimizes absolute_error_estimate = "
                "|D(h) - D_ref| over non-roundoff ladder points, where D_ref "
                "is the Richardson-based reference derivative."
            ),
        },
        {"Key": "step_size_table_path", "Value": str(OUTPUT_STEP_SIZE_CSV.resolve())},
        {
            "Key": "static_margin_reporting",
            "Value": (
                "Static-margin quantities are reported in %MAC and use the "
                "workbook-consistent neutral-point definition normalized by wing MAC."
            ),
        },
        {"Key": "objective_weights", "Value": str(asdict(context.objective_weights))},
        {"Key": "objective_scales", "Value": str(asdict(context.objective_scales))},
        {"Key": "baseline_success", "Value": baseline_result.success},
    ]
    for row in _mismatch_rows(context, baseline_result):
        rows.append(
            {
                "Key": f"baseline_delta_{row['name']}",
                "Value": row["delta_from_workbook"],
            }
        )

    unstable_df = sensitivity_df.loc[
        sensitivity_df["sensitivity_raw"].isna()
        | ~sensitivity_df["fd_stable"].astype(bool)
        | sensitivity_df["notes"].astype(str).str.len().gt(0)
    ]
    for idx, row in unstable_df.iterrows():
        rows.append(
            {
                "Key": f"issue_{idx:03d}",
                "Value": (
                    f"{row['parameter_name']} -> {row['quantity_name']}: "
                    f"{row['notes'] or 'unstable finite-difference estimate'}"
                ),
            }
        )
    return pd.DataFrame(rows)


def load_saved_step_size_table(
    path: Path | None = None,
) -> pd.DataFrame:
    csv_path = path or OUTPUT_STEP_SIZE_CSV
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Saved step-size table not found at {csv_path}. "
            "Run `python F_analysis/solve_step_size.py` first."
        )
    table_df = pd.read_csv(csv_path)
    required_columns = {
        "group",
        "parameter_name",
        "parameter_symbol",
        "baseline_parameter_value",
        "quantity_name",
        "quantity_symbol",
        "baseline_quantity_value",
        "difference_scheme",
        "step_size",
        "derivative_estimate",
        "normalized_sensitivity",
        "absolute_error_estimate",
        "relative_error_estimate",
        "fd_stability_abs",
        "fd_stability_rel",
        "fd_stable",
        "roundoff_limited",
        "selected_for_final",
        "selected_step_size",
        "roundoff_error_proxy",
        "total_error_proxy",
        "unit_raw",
        "selection_reason",
        "reeval_path",
        "notes",
    }
    missing_columns = required_columns.difference(table_df.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(
            "Saved step-size table uses an older schema and is missing required "
            f"columns: {missing_list}. Run `python F_analysis/solve_step_size.py` "
            "to regenerate it."
        )
    for column_name in ("selected_for_final", "fd_stable", "roundoff_limited"):
        table_df[column_name] = _coerce_bool_series(table_df[column_name])
    return table_df


def build_sensitivity_table_from_step_size_table(
    step_size_df: pd.DataFrame,
) -> pd.DataFrame:
    selected_df = step_size_df.loc[
        step_size_df["selected_for_final"].astype(bool)
    ].copy()
    if selected_df.empty:
        raise ValueError(
            "Saved step-size table contains no selected rows. "
            "Run `python F_analysis/solve_step_size.py` again."
        )

    selected_df.sort_values(
        by=["parameter_name", "quantity_name", "step_size"],
        ascending=[True, True, True],
        kind="mergesort",
        inplace=True,
    )
    selected_df = selected_df.drop_duplicates(
        subset=["parameter_name", "quantity_name"],
        keep="first",
    ).copy()

    sensitivity_df = pd.DataFrame(
        {
            "group": selected_df["group"],
            "parameter_name": selected_df["parameter_name"],
            "parameter_symbol": selected_df["parameter_symbol"],
            "baseline_parameter_value": selected_df["baseline_parameter_value"],
            "step_used": selected_df["step_size"],
            "quantity_name": selected_df["quantity_name"],
            "quantity_symbol": selected_df["quantity_symbol"],
            "baseline_quantity_value": selected_df["baseline_quantity_value"],
            "sensitivity_raw": selected_df["derivative_estimate"],
            "sensitivity_normalized": selected_df["normalized_sensitivity"],
            "difference_scheme": selected_df["difference_scheme"],
            "fd_stability_abs": selected_df["fd_stability_abs"],
            "fd_stability_rel": selected_df["fd_stability_rel"],
            "fd_stable": selected_df["fd_stable"],
            "unit_raw": selected_df["unit_raw"],
            "step_selection_reason": selected_df["selection_reason"],
            "notes": selected_df["notes"],
            "reeval_path": selected_df["reeval_path"],
        }
    )
    sensitivity_df["parameter_name"] = pd.Categorical(
        sensitivity_df["parameter_name"],
        categories=[*GEOMETRY_PARAM_ORDER, *REQUIREMENT_PARAM_ORDER],
        ordered=True,
    )
    sensitivity_df["quantity_name"] = pd.Categorical(
        sensitivity_df["quantity_name"],
        categories=ALL_QUANTITY_ORDER,
        ordered=True,
    )
    sensitivity_df.sort_values(
        by=["parameter_name", "quantity_name"],
        kind="mergesort",
        inplace=True,
    )
    sensitivity_df["parameter_name"] = sensitivity_df["parameter_name"].astype(str)
    sensitivity_df["quantity_name"] = sensitivity_df["quantity_name"].astype(str)
    return sensitivity_df.reset_index(drop=True)


def _primary_margin_lookup(parameter_df: pd.DataFrame) -> str:
    margin_metric_names = ["nom_cl_margin_to_cap", *MARGIN_QUANTITY_ORDER]
    margin_df = parameter_df.loc[
        parameter_df["quantity_name"].isin(margin_metric_names)
        & parameter_df["fd_stable"].astype(bool)
        & parameter_df["sensitivity_normalized"].notna()
    ].copy()
    if margin_df.empty:
        return ""
    margin_df["abs_norm"] = margin_df["sensitivity_normalized"].abs()
    best_row = margin_df.sort_values("abs_norm", ascending=False, kind="mergesort").iloc[0]
    return QUANTITY_LABELS[str(best_row["quantity_name"])]


def build_thesis_table(sensitivity_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    variants = [
        ("GeometryToPerformance", GEOMETRY_PARAM_ORDER, GEOMETRY_THESIS_METRICS),
        ("RequirementToPerformance", REQUIREMENT_PARAM_ORDER, REQUIREMENT_THESIS_METRICS),
    ]
    for variant, parameter_order, metric_names in variants:
        for parameter_name in parameter_order:
            parameter_df = sensitivity_df.loc[
                sensitivity_df["parameter_name"] == parameter_name
            ].copy()
            metric_df = parameter_df.loc[
                parameter_df["quantity_name"].isin(metric_names)
            ].copy()
            metric_values = {
                metric_name: float("nan")
                for metric_name in set(GEOMETRY_THESIS_METRICS + REQUIREMENT_THESIS_METRICS)
            }
            normalized_values: list[float] = []
            stable_flags: list[bool] = []
            for metric_name in metric_names:
                metric_row = metric_df.loc[metric_df["quantity_name"] == metric_name]
                if metric_row.empty:
                    continue
                metric_value = float(metric_row.iloc[0]["sensitivity_normalized"])
                metric_values[metric_name] = metric_value
                normalized_values.append(metric_value)
                stable_flags.append(bool(metric_row.iloc[0]["fd_stable"]))

            rows.append(
                {
                    "variant": variant,
                    "parameter_name": parameter_name,
                    "parameter_label": PARAMETER_LABELS[parameter_name],
                    "parameter_symbol": PARAMETER_SYMBOLS[parameter_name],
                    "baseline_parameter_value": float(
                        parameter_df["baseline_parameter_value"].iloc[0]
                    ),
                    "objective": metric_values["objective"],
                    "sink_rate_mps": metric_values["sink_rate_mps"],
                    "mass_total_kg": metric_values["mass_total_kg"],
                    "roll_tau_s": metric_values["roll_tau_s"],
                    "static_margin": metric_values["static_margin"],
                    "primary_affected_margin": (
                        _primary_margin_lookup(parameter_df)
                        if variant == "RequirementToPerformance"
                        else ""
                    ),
                    "interpretation": classify_interpretation(
                        normalized_values=normalized_values,
                        stable_flags=stable_flags,
                    ),
                }
            )
    return pd.DataFrame(rows)


def _step_plot_quantity_order(parameter_name: str) -> list[str]:
    if parameter_name in GEOMETRY_PARAM_ORDER:
        return GEOMETRY_STEP_PLOT_QUANTITIES
    return REQUIREMENT_STEP_PLOT_QUANTITIES


def _rank_step_plot_quantities(
    parameter_df: pd.DataFrame,
    parameter_name: str,
    max_quantities: int,
) -> list[str]:
    base_order = [
        quantity_name
        for quantity_name in _step_plot_quantity_order(parameter_name)
        if quantity_name in set(parameter_df["quantity_name"])
    ]
    if max_quantities <= 0 or len(base_order) <= max_quantities:
        return base_order

    selected_df = parameter_df.loc[
        parameter_df["selected_for_final"].astype(bool)
    ].copy()
    if selected_df.empty:
        return base_order[:max_quantities]

    selected_df["rel_error"] = pd.to_numeric(
        selected_df["fd_stability_rel"],
        errors="coerce",
    )
    selected_df["abs_error"] = pd.to_numeric(
        selected_df["absolute_error_estimate"],
        errors="coerce",
    ).abs()
    selected_df["order_rank"] = selected_df["quantity_name"].map(
        {name: index for index, name in enumerate(base_order)}
    )
    selected_df.sort_values(
        by=["rel_error", "abs_error", "order_rank"],
        ascending=[False, False, True],
        kind="mergesort",
        inplace=True,
    )
    ranked = selected_df["quantity_name"].drop_duplicates().tolist()
    if len(ranked) < max_quantities:
        for quantity_name in base_order:
            if quantity_name not in ranked:
                ranked.append(quantity_name)
            if len(ranked) >= max_quantities:
                break
    return ranked[:max_quantities]


def make_step_size_figure(
    step_size_df: pd.DataFrame,
    figure_path: Path | None = None,
    max_quantities: int = 3,
) -> None:
    if step_size_df.empty:
        raise ValueError("Step-size table is empty; no trade-off figure can be drawn.")

    figure_path = figure_path or OUTPUT_FIGURE
    parameter_order = [
        *GEOMETRY_PARAM_ORDER,
        *REQUIREMENT_PARAM_ORDER,
    ]
    parameter_names = [
        name for name in parameter_order if name in set(step_size_df["parameter_name"])
    ]
    if not parameter_names:
        parameter_names = sorted(step_size_df["parameter_name"].unique().tolist())

    all_quantities = []
    for parameter_name in parameter_names:
        for quantity_name in _step_plot_quantity_order(parameter_name):
            if quantity_name in set(
                step_size_df.loc[
                    step_size_df["parameter_name"] == parameter_name,
                    "quantity_name",
                ]
            ):
                all_quantities.append(quantity_name)
    ordered_quantities = list(dict.fromkeys(all_quantities))
    colors = {
        quantity_name: plt.get_cmap("tab10")(index % 10)
        for index, quantity_name in enumerate(ordered_quantities)
    }

    n_panels = len(parameter_names)
    ncols = 3 if n_panels > 6 else 2 if n_panels > 1 else 1
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.6 * ncols, 4.8 * nrows),
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes).reshape(-1)
    style_handles = [
        plt.Line2D(
            [0],
            [0],
            color="black",
            linewidth=2.0,
            linestyle="-",
        ),
        plt.Line2D(
            [0],
            [0],
            color="black",
            linestyle="None",
            marker="o",
            markersize=4.8,
            alpha=0.45,
        ),
        plt.Line2D(
            [0],
            [0],
            color="black",
            marker="*",
            linestyle="None",
            markersize=10,
        ),
        plt.Line2D(
            [0],
            [0],
            color="black",
            marker="s",
            markerfacecolor="none",
            linestyle="None",
            markersize=7,
        ),
    ]
    style_labels = [
        "Total-error proxy",
        "Observed absolute error",
        "Selected final step (min derivative error)",
        "Round-off-limited point",
    ]

    for axis, parameter_name in zip(axes_array, parameter_names, strict=False):
        parameter_df = step_size_df.loc[
            step_size_df["parameter_name"] == parameter_name
        ].copy()
        quantity_order = _rank_step_plot_quantities(
            parameter_df=parameter_df,
            parameter_name=parameter_name,
            max_quantities=max_quantities,
        )
        quantity_handles = []
        quantity_labels = []
        for quantity_name in quantity_order:
            quantity_df = parameter_df.loc[
                parameter_df["quantity_name"] == quantity_name
            ].sort_values("step_size", kind="mergesort")
            x_values = quantity_df["step_size"].to_numpy(dtype=float)
            y_observed = np.maximum(
                quantity_df["absolute_error_estimate"].to_numpy(dtype=float),
                1e-18,
            )
            if "total_error_proxy" in quantity_df.columns:
                y_proxy = np.maximum(
                    quantity_df["total_error_proxy"].to_numpy(dtype=float),
                    1e-18,
                )
            else:
                y_proxy = y_observed
            selected_y_column = "absolute_error_estimate"
            color = colors[quantity_name]
            axis.scatter(
                x_values,
                y_observed,
                s=20,
                color=color,
                alpha=0.35,
                zorder=3,
            )
            axis.plot(
                x_values,
                y_proxy,
                linestyle="-",
                linewidth=2.1,
                color=color,
                alpha=0.92,
            )
            quantity_handles.append(
                plt.Line2D([0], [0], color=color, linewidth=2.1)
            )
            quantity_labels.append(QUANTITY_LABELS[quantity_name])

            selected_df = quantity_df.loc[quantity_df["selected_for_final"].astype(bool)]
            if not selected_df.empty:
                axis.scatter(
                    selected_df["step_size"],
                    np.maximum(selected_df[selected_y_column], 1e-18),
                    marker="*",
                    s=110,
                    zorder=6,
                    color=color,
                    edgecolors="black",
                    linewidths=0.8,
                )

            roundoff_df = quantity_df.loc[quantity_df["roundoff_limited"].astype(bool)]
            if not roundoff_df.empty:
                axis.scatter(
                    roundoff_df["step_size"],
                    np.maximum(roundoff_df[selected_y_column], 1e-18),
                    marker="s",
                    s=44,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=1.1,
                    zorder=5,
                )

        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.22)
        axis.set_xlabel("Step size")
        axis.set_ylabel("Error")
        axis.set_title(PARAMETER_LABELS.get(parameter_name, parameter_name))
        axis.text(
            0.02,
            0.02,
            f"Top {len(quantity_order)} quantities by derivative-error difficulty",
            transform=axis.transAxes,
            fontsize=8,
            color="#444444",
            ha="left",
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65},
        )
        if quantity_handles:
            axis.legend(
                quantity_handles,
                quantity_labels,
                loc="upper right",
                fontsize=7.5,
                frameon=False,
            )

    for axis in axes_array[n_panels:]:
        axis.axis("off")

    fig.legend(
        style_handles,
        style_labels,
        loc="upper center",
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        "Finite-difference step-size ladder",
        fontsize=15,
        y=1.01,
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_console_summary(
    sensitivity_df: pd.DataFrame,
    thesis_df: pd.DataFrame,
) -> None:
    print("Sensitivity analysis summary")
    for variant, parameter_order in (
        ("GeometryToPerformance", GEOMETRY_PARAM_ORDER),
        ("RequirementToPerformance", REQUIREMENT_PARAM_ORDER),
    ):
        variant_df = thesis_df.loc[thesis_df["variant"] == variant].copy()
        metric_names = (
            GEOMETRY_THESIS_METRICS
            if variant == "GeometryToPerformance"
            else REQUIREMENT_THESIS_METRICS
        )
        variant_df["max_abs_norm"] = variant_df[metric_names].abs().max(axis=1, skipna=True)
        ranked_df = variant_df.sort_values(
            "max_abs_norm",
            ascending=False,
            kind="mergesort",
        )
        print(f"{variant}:")
        for _, row in ranked_df.iterrows():
            print(
                f"  {row['parameter_label']}: "
                f"max |S_bar| = {row['max_abs_norm']:.3f}, "
                f"{row['interpretation']}"
            )

    unstable_df = sensitivity_df.loc[
        sensitivity_df["sensitivity_raw"].notna()
        & ~sensitivity_df["fd_stable"].astype(bool)
    ]
    if not unstable_df.empty:
        print("Unstable estimates:")
        for _, row in unstable_df.iterrows():
            print(
                f"  {row['parameter_name']} -> {row['quantity_name']}: "
                f"rel={row['fd_stability_rel']:.3g}, abs={row['fd_stability_abs']:.3g}"
            )

    skipped_df = sensitivity_df.loc[sensitivity_df["sensitivity_raw"].isna()]
    if not skipped_df.empty:
        print("Failed or skipped perturbations:")
        for _, row in skipped_df.iterrows():
            print(f"  {row['parameter_name']} -> {row['quantity_name']}: {row['notes']}")


def write_excel_outputs(
    metadata_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    step_size_df: pd.DataFrame,
    thesis_df: pd.DataFrame,
) -> None:
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        metadata_df.to_excel(writer, sheet_name="Metadata", index=False)
        baseline_df.to_excel(writer, sheet_name="Baseline", index=False)
        sensitivity_df.to_excel(writer, sheet_name="SensitivityTable", index=False)
        step_size_df.to_excel(writer, sheet_name="StepSizeTable", index=False)
        thesis_df.to_excel(writer, sheet_name="ThesisTable", index=False)


def write_step_size_excel(
    metadata_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    step_size_df: pd.DataFrame,
    output_path: Path | None = None,
) -> None:
    workbook_path = output_path or OUTPUT_STEP_SIZE_XLSX
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        metadata_df.to_excel(writer, sheet_name="Metadata", index=False)
        baseline_df.to_excel(writer, sheet_name="Baseline", index=False)
        step_size_df.to_excel(writer, sheet_name="StepSizeTable", index=False)


def compute_study_tables(
    context: BaselineContext,
) -> tuple[EvaluationResult, pd.DataFrame, pd.DataFrame]:
    quantity_specs = build_quantity_specs()
    parameter_specs = (
        build_geometry_parameter_specs(context)
        + build_requirement_parameter_specs(context)
    )
    baseline_result = evaluate_full_trim(
        context=context,
        geometry=context.geometry,
        requirement_values=context.requirement_values,
    )
    if not baseline_result.success:
        raise RuntimeError("Baseline deterministic trim evaluation failed.")

    sensitivity_rows: list[SensitivityResult] = []
    step_size_rows: list[StepSizeResult] = []
    for parameter_spec in parameter_specs:
        parameter_results, parameter_step_rows = compute_sensitivity_for_parameter(
            context=context,
            baseline_result=baseline_result,
            parameter_spec=parameter_spec,
            quantity_specs=quantity_specs,
        )
        sensitivity_rows.extend(parameter_results)
        step_size_rows.extend(parameter_step_rows)

    sensitivity_df = pd.DataFrame([asdict(row) for row in sensitivity_rows])
    step_size_df = pd.DataFrame([asdict(row) for row in step_size_rows])
    return baseline_result, sensitivity_df, step_size_df


def main() -> None:
    context = load_selected_baseline()
    baseline_result = evaluate_full_trim(
        context=context,
        geometry=context.geometry,
        requirement_values=context.requirement_values,
    )
    if not baseline_result.success:
        raise RuntimeError("Baseline deterministic trim evaluation failed.")

    step_size_df = load_saved_step_size_table()
    sensitivity_df = build_sensitivity_table_from_step_size_table(step_size_df)
    baseline_df = build_baseline_table(context, baseline_result)
    thesis_df = build_thesis_table(sensitivity_df)
    metadata_df = build_metadata_table(context, baseline_result, sensitivity_df)

    sensitivity_df.to_csv(OUTPUT_TABLE_CSV, index=False)
    thesis_df.to_csv(OUTPUT_THESIS_CSV, index=False)
    write_excel_outputs(
        metadata_df,
        baseline_df,
        sensitivity_df,
        step_size_df,
        thesis_df,
    )
    if LEGACY_HEATMAP_FIGURE.exists():
        LEGACY_HEATMAP_FIGURE.unlink()
    print_console_summary(sensitivity_df, thesis_df)
    print(f"Saved workbook: {OUTPUT_XLSX}")
    print(f"Saved table: {OUTPUT_TABLE_CSV}")
    print(f"Read step-size table: {OUTPUT_STEP_SIZE_CSV}")
    print(f"Saved thesis table: {OUTPUT_THESIS_CSV}")


if __name__ == "__main__":
    main()
