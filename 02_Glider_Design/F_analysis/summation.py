"""Build thesis-ready summary tables from the canonical design workbook."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Allows direct execution as `python F_analysis/summation.py`.
    sys.path.insert(0, str(PROJECT_ROOT))

from F_analysis.analysis_common import (
    open_canonical_workbook,
    read_sheet_optional,
    read_sheet_required,
    resolve_selected_candidate_id,
    resolve_tail_metric_name,
    sort_scenario_tags,
)

# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Workbook paths
# 2) Formatting and aggregation helpers
# 3) Workbook source loading
# 4) Main-text table builders
# 5) Workbook export and CLI
# =============================================================================

# =============================================================================
# 1) Workbook Paths
# =============================================================================

RESULTS_DIR = PROJECT_ROOT / "C_results"
FIGURES_DIR = PROJECT_ROOT / "B_figures"
OUTPUT_XLSX = RESULTS_DIR / "main_text_tables.xlsx"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2) Formatting and Aggregation Helpers
# =============================================================================

def _series_lookup(df: pd.DataFrame, key_col: str, value_col: str) -> dict[str, object]:
    if df.empty or key_col not in df.columns or value_col not in df.columns:
        return {}
    return pd.Series(df[value_col].to_numpy(), index=df[key_col].astype(str)).to_dict()


def _format_value(value: object, digits: int = 4) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric):
        return "n/a"
    return f"{numeric:.{digits}g}"


def _format_range(lower: object, upper: object, digits: int = 4) -> str:
    if pd.isna(lower) or pd.isna(upper):
        return "n/a"
    return f"[{_format_value(lower, digits)}, {_format_value(upper, digits)}]"


def _constraint_row(
    name: str,
    definition: str,
    unit: str,
    source_name: str,
    constraints_df: pd.DataFrame,
) -> dict[str, str]:
    if not constraints_df.empty and "Constraint" in constraints_df.columns:
        match_df = constraints_df.loc[constraints_df["Constraint"] == source_name]
    else:
        match_df = pd.DataFrame()

    implemented_as = "hard constraint"
    source = source_name
    if not match_df.empty:
        # Preserve the optimizer-exported type and margin when the sheet has it.
        row = match_df.iloc[0]
        implemented_as = (
            f"{row.get('Type', 'hard')} constraint; margin={_format_value(row.get('Margin'))}"
        )
        source = f"Constraints: {source_name}"

    return {
        "type": "hard constraint",
        "name": name,
        "definition": definition,
        "unit": unit,
        "implemented_as": implemented_as,
        "source": source,
    }


def _penalty_row(
    name: str,
    definition: str,
    objective_terms_df: pd.DataFrame,
) -> dict[str, str]:
    if objective_terms_df.empty:
        return {
            "type": "soft penalty",
            "name": name,
            "definition": definition,
            "unit": "-",
            "implemented_as": "objective term",
            "source": "ObjectiveTerms",
        }

    match_df = objective_terms_df.loc[objective_terms_df["Term"] == name]
    if match_df.empty:
        return {
            "type": "soft penalty",
            "name": name,
            "definition": definition,
            "unit": "-",
            "implemented_as": "objective term",
            "source": "ObjectiveTerms",
        }

    row = match_df.iloc[0]
    # Objective weights and values are provenance for the soft-penalty table.
    implemented_as = (
        "objective term; "
        f"weight={_format_value(row.get('Weight'))}, "
        f"value={_format_value(row.get('Value'))}"
    )
    return {
        "type": "soft penalty",
        "name": name,
        "definition": definition,
        "unit": "-",
        "implemented_as": implemented_as,
        "source": "ObjectiveTerms",
    }


def _aggregate_tag_summary(selected_scenarios_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if selected_scenarios_df.empty:
        return pd.DataFrame()

    for scenario_tag, group_df in selected_scenarios_df.groupby("scenario_tag", sort=False):
        sink = (
            pd.to_numeric(group_df["nom_sink_rate_mps"], errors="coerce")
            if "nom_sink_rate_mps" in group_df.columns
            else pd.Series(dtype=float)
        )
        sink = sink.dropna()
        sink_sorted = np.sort(sink.to_numpy()) if not sink.empty else np.array([])
        # Tail-risk sink is the mean of the worst 20% of selected-scenario sinks.
        tail_count = max(1, int(np.ceil(0.2 * len(sink_sorted)))) if sink_sorted.size else 0
        tail_mean = (
            float(np.mean(sink_sorted[-tail_count:])) if tail_count > 0 else np.nan
        )

        resid_value = np.nan
        if "nom_lateral_residual" in group_df.columns and "nom_success" in group_df.columns:
            # Residual RMSE excludes failed trims so it reflects successful-case fit.
            success_mask = group_df["nom_success"].astype(bool)
            resid = group_df.loc[success_mask, "nom_lateral_residual"].astype(float)
            if not resid.empty:
                resid_value = float(np.sqrt(np.mean(np.square(resid.to_numpy()))))

        success_rate = (
            float(group_df["nom_success"].astype(float).mean())
            if "nom_success" in group_df.columns
            else np.nan
        )
        rows.append(
            {
                "scenario_tag": scenario_tag,
                "n_total": int(len(group_df)),
                "n_success": int(group_df["nom_success"].astype(float).sum())
                if "nom_success" in group_df.columns
                else np.nan,
                "n_fail": int(len(group_df) - group_df["nom_success"].astype(float).sum())
                if "nom_success" in group_df.columns
                else np.nan,
                "success_rate": success_rate,
                "failure_rate": 1.0 - success_rate if np.isfinite(success_rate) else np.nan,
                "sink_mean": float(sink.mean()) if not sink.empty else np.nan,
                "sink_p95": float(np.quantile(sink.to_numpy(), 0.95)) if sink.size else np.nan,
                "sink_worst": float(sink.max()) if not sink.empty else np.nan,
                "nom_sink_tail_mean_k": tail_mean,
                "nom_resid_rmse_success_only": resid_value,
            }
        )

    summary_df = pd.DataFrame(rows)
    tag_order = sort_scenario_tags(summary_df["scenario_tag"])
    summary_df["tag_sort"] = summary_df["scenario_tag"].map(
        {tag: idx for idx, tag in enumerate(tag_order)}
    ).fillna(len(tag_order))
    summary_df = summary_df.sort_values(
        by=["tag_sort", "scenario_tag"],
        kind="mergesort",
    ).drop(columns="tag_sort")
    return summary_df.reset_index(drop=True)


# =============================================================================
# 3) Workbook Source Loading
# =============================================================================

def load_table_sources() -> dict[str, pd.DataFrame]:
    _, workbook = open_canonical_workbook()
    try:
        return {
            "RunInfo": read_sheet_optional(workbook, "RunInfo"),
            "Candidates": read_sheet_required(workbook, "Candidates"),
            "RobustSummary": read_sheet_required(workbook, "RobustSummary"),
            "Definitions": read_sheet_optional(workbook, "Definitions"),
            "Summary": read_sheet_required(workbook, "Summary"),
            "ObjectiveTerms": read_sheet_optional(workbook, "ObjectiveTerms"),
            "Geometry": read_sheet_required(workbook, "Geometry"),
            "Constraints": read_sheet_optional(workbook, "Constraints"),
            "RobustSummaryByTag": read_sheet_optional(workbook, "RobustSummaryByTag"),
            "RobustScenarios": read_sheet_required(workbook, "RobustScenarios"),
            "DesignVarBounds": read_sheet_optional(workbook, "DesignVarBounds"),
            "DesignPoints": read_sheet_optional(workbook, "DesignPoints"),
        }
    finally:
        workbook.close()


# =============================================================================
# 4) Main-Text Table Builders
# =============================================================================

def build_variables_sheet(
    sources: dict[str, pd.DataFrame],
    selected_candidate_id: int,
) -> pd.DataFrame:
    run_info_map = _series_lookup(sources["RunInfo"], "Key", "Value")
    summary_map = _series_lookup(sources["Summary"], "Metric", "Value")
    candidates_df = sources["Candidates"]
    selected_candidate = candidates_df.loc[
        candidates_df["candidate_id"] == selected_candidate_id
    ].iloc[0]
    design_bounds_df = sources["DesignVarBounds"]
    bounds_map = (
        design_bounds_df.set_index("Variable") if not design_bounds_df.empty else pd.DataFrame()
    )

    rows: list[dict[str, str]] = []

    def add_row(
        group: str,
        name: str,
        symbol_or_key: str,
        value_or_range: object,
        unit: str,
        source: str,
    ) -> None:
        rows.append(
            {
                "group": group,
                "name": name,
                "symbol_or_key": symbol_or_key,
                "value_or_range": str(value_or_range),
                "unit": unit,
                "source": source,
            }
        )

    geometry_rows = [
        ("wing span", "wing_span_m"),
        ("wing chord", "wing_chord_m"),
        ("tail arm", "tail_arm_m"),
        ("h-tail span", "htail_span_m"),
        ("v-tail height", "vtail_height_m"),
    ]
    for name, key in geometry_rows:
        if not bounds_map.empty and key in bounds_map.index:
            # DesignVarBounds documents the optimisation search domain.
            bounds_row = bounds_map.loc[key]
            value_or_range = _format_range(bounds_row["Lower"], bounds_row["Upper"])
            source = "DesignVarBounds"
            unit = bounds_row.get("Unit", "m")
        elif f"{key}_min" in run_info_map and f"{key}_max" in run_info_map:
            value_or_range = _format_range(
                run_info_map.get(f"{key}_min"),
                run_info_map.get(f"{key}_max"),
            )
            source = "RunInfo"
            unit = "m"
        else:
            # Legacy workbooks may only expose the selected candidate value.
            selected_key = "boom_length_m" if key == "tail_arm_m" else key
            value_or_range = _format_value(selected_candidate.get(selected_key))
            source = "Candidates(selected)"
            unit = "m"
        add_row("optimised geometry", name, key, value_or_range, unit, source)

    trim_rows = [
        ("angle of attack", "alpha_nom_deg"),
        ("aileron trim", "delta_a_nom_deg"),
        ("elevator trim", "delta_e_nom_deg"),
        ("rudder trim", "delta_r_nom_deg"),
    ]
    for name, key in trim_rows:
        if not bounds_map.empty and key in bounds_map.index:
            # Trim angles are exported in degrees for human-facing thesis tables.
            bounds_row = bounds_map.loc[key]
            value_or_range = _format_range(bounds_row["Lower"], bounds_row["Upper"])
            source = "DesignVarBounds"
            unit = bounds_row.get("Unit", "deg")
        else:
            summary_key = key.replace("_nom_", "_trim_")
            value_or_range = _format_value(summary_map.get(summary_key))
            source = "Summary"
            unit = "deg"
        add_row("trim variables", name, key, value_or_range, unit, source)

    settings_rows = [
        ("nominal speed", "v_nom_mps", "m/s"),
        ("turn speed", "v_turn_mps", "m/s"),
        ("turn bank angle", "turn_bank_deg", "deg"),
        ("bank-entry time", "bank_entry_time_s", "s"),
        ("arena width", "arena_width_m", "m"),
        ("wall clearance", "wall_clearance_m", "m"),
        ("nominal CL cap", "max_cl_nominal", "-"),
        ("turn CL cap", "max_cl_turn", "-"),
    ]
    design_points_map = _series_lookup(sources["DesignPoints"], "Metric", "Value")
    for name, key, unit in settings_rows:
        value = summary_map.get(
            key,
            design_points_map.get(
                key.upper(),
                run_info_map.get(key),
            ),
        )
        add_row(
            "mission / settings",
            name,
            key,
            _format_value(value),
            unit,
            "Summary/DesignPoints/RunInfo",
        )

    selected_rows = [
        ("span", "wing_span_m", selected_candidate.get("wing_span_m"), "m"),
        ("chord", "wing_chord_m", selected_candidate.get("wing_chord_m"), "m"),
        ("boom length", "boom_length_m", selected_candidate.get("boom_length_m"), "m"),
        ("static margin", "static_margin", selected_candidate.get("static_margin"), "-"),
        ("wing loading", "wing_loading_n_m2", summary_map.get("wing_loading_n_m2"), "N/m^2"),
        ("mass", "mass_total_kg", selected_candidate.get("mass_total_kg"), "kg"),
        ("roll tau", "roll_tau_s", selected_candidate.get("roll_tau_s"), "s"),
    ]
    for name, key, value, unit in selected_rows:
        add_row("selected design", name, key, _format_value(value), unit, "Candidates/Summary")

    return pd.DataFrame(rows)


def build_constraints_penalties_sheet(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    constraints_df = sources["Constraints"]
    objective_terms_df = sources["ObjectiveTerms"]
    rows = [
        _constraint_row("lift balance", "Nominal lift must be at least equal to aircraft weight.", "N", "Lift >= Weight", constraints_df),
        _constraint_row("trim Cm condition", "The nominal pitching-moment trim condition must stay within the trim tolerance band.", "-", "Nominal Trim Cm tolerance", constraints_df),
        _constraint_row("nominal lateral trim Cl", "Nominal roll moment must remain near zero when lateral trim is enforced.", "-", "Nominal Trim Cl", constraints_df),
        _constraint_row("nominal lateral trim Cn", "Nominal yaw moment must remain near zero when lateral trim is enforced.", "-", "Nominal Trim Cn", constraints_df),
        _constraint_row("nominal CL cap", "The nominal lift coefficient must not exceed the imposed CL limit.", "-", "CL <= CLmax", constraints_df),
        _constraint_row("minimum L/D", "Nominal lift-to-drag ratio must stay above the mission minimum.", "-", "L/D minimum", constraints_df),
        _constraint_row("minimum Reynolds number", "Wing Reynolds number must stay above the aerodynamic validity floor.", "-", "Wing Reynolds", constraints_df),
        _constraint_row("wing-loading lower bound", "Wing loading must stay above the minimum feasible loading.", "N/m^2", "Wing loading minimum", constraints_df),
        _constraint_row("wing-loading upper bound", "Wing loading must stay below the structural and handling ceiling.", "N/m^2", "Wing loading maximum", constraints_df),
        _constraint_row("static-margin lower bound", "Static margin must stay above the required longitudinal stability floor.", "MAC fraction", "Static margin minimum", constraints_df),
        _constraint_row("static-margin upper bound", "Static margin must stay below the imposed handling limit.", "MAC fraction", "Static margin maximum", constraints_df),
        _constraint_row("horizontal tail-volume lower bound", "Horizontal tail volume must stay above the required stability/control minimum.", "-", "Vh minimum", constraints_df),
        _constraint_row("horizontal tail-volume upper bound", "Horizontal tail volume must remain below the imposed sizing limit.", "-", "Vh maximum", constraints_df),
        _constraint_row("vertical tail-volume lower bound", "Vertical tail volume must stay above the directional-stability minimum.", "-", "Vv minimum", constraints_df),
        _constraint_row("vertical tail-volume upper bound", "Vertical tail volume must remain below the imposed sizing ceiling.", "-", "Vv maximum", constraints_df),
        _constraint_row("dihedral effect sign", "The roll response to sideslip must be non-positive.", "-", "Clb <= 0", constraints_df),
        _constraint_row("weathercock stability sign", "The yaw response to sideslip must be non-negative.", "-", "Cnb >= 0", constraints_df),
        _constraint_row("roll damping sign", "Roll damping must remain sufficiently negative.", "-", "Clp nominal <= -eps", constraints_df),
        _constraint_row("aileron servo torque", "Aileron hinge moment demand must stay below the servo torque limit.", "N m", "Aileron servo torque", constraints_df),
        _constraint_row("elevator servo torque", "Elevator hinge moment demand must stay below the servo torque limit.", "N m", "Elevator servo torque", constraints_df),
        _constraint_row("rudder servo torque", "Rudder hinge moment demand must stay below the servo torque limit.", "N m", "Rudder servo torque", constraints_df),
        {
            "type": "hard constraint",
            "name": "boom length bounds",
            "definition": "The boom length design variable is boxed between the allowed minimum and maximum.",
            "unit": "m",
            "implemented_as": "design-variable box bounds",
            "source": "DesignVarBounds: boom_length_m",
        },
        {
            "type": "hard constraint",
            "name": "positive principal inertias",
            "definition": "Principal inertias must remain positive to preserve a physical mass model.",
            "unit": "kg m^2",
            "implemented_as": "algebraic positivity constraint",
            "source": "mass-property feasibility logic",
        },
        _penalty_row("J_sink", "Primary objective term penalising nominal sink rate.", objective_terms_df),
        _penalty_row("J_mass", "Mass penalty term applied to the selected mass measure.", objective_terms_df),
        _penalty_row("J_trim", "Trim-effort penalty based on control deflection demand.", objective_terms_df),
        _penalty_row("J_wing_deflection", "Wing deflection proxy penalty that discourages overly soft wing designs.", objective_terms_df),
        _penalty_row("J_htail_deflection", "Horizontal-tail deflection proxy penalty that discourages overly soft tail designs.", objective_terms_df),
        _penalty_row("J_roll_tau", "Roll-time-constant penalty that favours fast roll response.", objective_terms_df),
    ]
    return pd.DataFrame(rows)


def build_uncertainty_model_sheet(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    robust_scenarios_df = sources["RobustScenarios"]
    definitions_df = sources["Definitions"]
    unique_scenarios_df = robust_scenarios_df.drop_duplicates("scenario_id").copy()
    definitions_map = (
        definitions_df.set_index("name").to_dict("index") if not definitions_df.empty else {}
    )

    ordered_tags: list[str] = []
    if "scenario_tag" in unique_scenarios_df.columns:
        ordered_tags = sort_scenario_tags(unique_scenarios_df["scenario_tag"])

    variable_names = [
        "mass_scale",
        "cg_x_shift_mac",
        "incidence_bias_deg",
        "eff_a",
        "eff_e",
        "eff_r",
        "bias_a_deg",
        "bias_e_deg",
        "bias_r_deg",
        "ixx_scale",
        "iyy_scale",
        "izz_scale",
        "wing_E_scale",
        "htail_E_scale",
        "w_gust_nom",
        "w_gust_turn",
        "drag_factor",
        "q",
        "scenario_tag",
    ]

    rows: list[dict[str, str]] = []
    for name in variable_names:
        definition_entry = definitions_map.get(name, {})
        if name == "scenario_tag":
            range_or_values = ", ".join(ordered_tags) if ordered_tags else "not available"
        elif name in unique_scenarios_df.columns:
            # Ranges report sampled scenario inputs, not imposed uncertainty bounds.
            values = pd.to_numeric(unique_scenarios_df[name], errors="coerce")
            finite_values = values[np.isfinite(values)]
            range_or_values = (
                _format_range(finite_values.min(), finite_values.max())
                if not finite_values.empty
                else "not available"
            )
        else:
            range_or_values = "not available"

        rows.append(
            {
                "name": name,
                "unit": str(definition_entry.get("unit", "-")),
                "range_or_values": range_or_values,
                "meaning": str(
                    definition_entry.get(
                        "definition",
                        "Scenario uncertainty input or grouping label.",
                    )
                ),
                "notes": str(definition_entry.get("notes", "")),
            }
        )

    return pd.DataFrame(rows)


def build_final_selection_summary_sheet(
    sources: dict[str, pd.DataFrame],
    selected_candidate_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates_df = sources["Candidates"]
    robust_summary_df = sources["RobustSummary"]
    merged_df = candidates_df.merge(
        robust_summary_df,
        on="candidate_id",
        how="left",
        suffixes=("", "_robust"),
    )
    tail_metric = resolve_tail_metric_name(robust_summary_df)

    best_nominal_candidate_id = int(
        candidates_df.sort_values("objective", kind="mergesort").iloc[0]["candidate_id"]
    )
    best_robust_candidate_id = resolve_selected_candidate_id(
        candidates_df,
        robust_summary_df,
    )

    comparison_rows = [
        # Compare the exported selection with nominal-objective and robust choices.
        ("selected_candidate", selected_candidate_id),
        ("best_nominal_objective_candidate", best_nominal_candidate_id),
        ("best_robust_success_candidate", best_robust_candidate_id),
    ]
    summary_columns = [
        "comparison_row",
        "candidate_id",
        "robust_rank",
        "selected_flag",
        "objective",
        "n_success",
        "n_total",
        "nom_success_rate",
        "nom_failure_rate",
        "nom_sink_mean",
        tail_metric,
        "nom_sink_p95",
        "nom_sink_worst",
        "nom_resid_rmse_success_only",
        "nom_roll_tau_p95",
        "nom_roll_tau_worst",
        "nom_delta_e_util_p95",
        "nom_delta_e_util_worst",
        "nom_alpha_p95",
        "nom_alpha_margin_p05",
        "nom_alpha_margin_worst",
        "nom_cl_margin_p05",
        "nom_cl_margin_worst",
        "nom_wing_deflection_over_allow_worst",
        "nom_htail_deflection_over_allow_worst",
        "wing_span_m",
        "wing_chord_m",
        "boom_length_m",
        "mass_total_kg",
        "static_margin",
        "L_over_D",
        "sink_rate_mps",
    ]

    comparison_row_frames: list[pd.DataFrame] = []
    for label, candidate_id in comparison_rows:
        row_df = merged_df.loc[merged_df["candidate_id"] == candidate_id].copy()
        if row_df.empty:
            continue
        row_df.insert(0, "comparison_row", label)
        row_df["selected_flag"] = candidate_id == selected_candidate_id
        comparison_row_frames.append(row_df.reindex(columns=summary_columns))

    comparison_df = (
        pd.concat(comparison_row_frames, ignore_index=True)
        if comparison_row_frames
        else pd.DataFrame(columns=summary_columns)
    )

    summary_by_tag_df = sources["RobustSummaryByTag"]
    if not summary_by_tag_df.empty and "candidate_id" in summary_by_tag_df.columns:
        summary_by_tag_df = summary_by_tag_df.loc[
            summary_by_tag_df["candidate_id"] == selected_candidate_id
        ].copy()
    elif summary_by_tag_df.empty:
        # Reconstruct by-tag summaries for older workbooks that only store rows.
        robust_scenarios_df = sources["RobustScenarios"]
        selected_scenarios_df = robust_scenarios_df.loc[
            robust_scenarios_df["candidate_id"] == selected_candidate_id
        ].copy()
        summary_by_tag_df = _aggregate_tag_summary(selected_scenarios_df)

    if not summary_by_tag_df.empty:
        if "sink_mean" not in summary_by_tag_df.columns and "nom_sink_mean" in summary_by_tag_df.columns:
            summary_by_tag_df = summary_by_tag_df.rename(columns={"nom_sink_mean": "sink_mean"})
        if "sink_worst" not in summary_by_tag_df.columns and "nom_sink_worst" in summary_by_tag_df.columns:
            summary_by_tag_df = summary_by_tag_df.rename(columns={"nom_sink_worst": "sink_worst"})
        if "sink_tail_mean_k" in summary_by_tag_df.columns:
            # Legacy column name maps to the current robust-ranking metric.
            summary_by_tag_df = summary_by_tag_df.rename(
                columns={"sink_tail_mean_k": "nom_sink_tail_mean_k"}
            )

        tag_order = sort_scenario_tags(summary_by_tag_df["scenario_tag"])
        summary_by_tag_df["tag_sort"] = summary_by_tag_df["scenario_tag"].map(
            {tag: idx for idx, tag in enumerate(tag_order)}
        ).fillna(len(tag_order))
        summary_by_tag_df = summary_by_tag_df.sort_values(
            by=["tag_sort", "scenario_tag"],
            kind="mergesort",
        ).drop(columns="tag_sort")
        summary_tail_metric = resolve_tail_metric_name(summary_by_tag_df)
        summary_by_tag_df = summary_by_tag_df.reindex(
            columns=[
                "scenario_tag",
                "n_fail",
                "failure_rate",
                "success_rate",
                "sink_mean",
                "sink_p95",
                "sink_worst",
                summary_tail_metric,
                "nom_resid_rmse_success_only",
                "nom_alpha_margin_worst",
                "nom_cl_margin_worst",
                "nom_delta_e_util_worst",
            ]
        )

    return comparison_df, summary_by_tag_df


# =============================================================================
# 5) Workbook Export and CLI
# =============================================================================

def write_tables_workbook(
    variables_df: pd.DataFrame,
    constraints_df: pd.DataFrame,
    uncertainty_df: pd.DataFrame,
    final_summary_df: pd.DataFrame,
    summary_by_tag_df: pd.DataFrame,
) -> Path:
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        variables_df.to_excel(writer, sheet_name="variables", index=False)
        constraints_df.to_excel(writer, sheet_name="constraints and penalties", index=False)
        uncertainty_df.to_excel(writer, sheet_name="uncertainty model", index=False)
        final_summary_df.to_excel(writer, sheet_name="final selection summary", index=False)
        if not summary_by_tag_df.empty:
            # Place by-tag detail below the candidate-comparison block in one sheet.
            start_row = len(final_summary_df) + 3
            title_df = pd.DataFrame([{"scenario_tag": "selected candidate by tag"}])
            title_df.to_excel(
                writer,
                sheet_name="final selection summary",
                index=False,
                startrow=start_row,
            )
            summary_by_tag_df.to_excel(
                writer,
                sheet_name="final selection summary",
                index=False,
                startrow=start_row + 2,
            )
    return OUTPUT_XLSX


def main() -> None:
    sources = load_table_sources()
    selected_candidate_id = resolve_selected_candidate_id(
        sources["Candidates"],
        sources["RobustSummary"],
    )
    variables_df = build_variables_sheet(sources, selected_candidate_id)
    constraints_df = build_constraints_penalties_sheet(sources)
    uncertainty_df = build_uncertainty_model_sheet(sources)
    final_summary_df, summary_by_tag_df = build_final_selection_summary_sheet(
        sources,
        selected_candidate_id,
    )
    output_path = write_tables_workbook(
        variables_df=variables_df,
        constraints_df=constraints_df,
        uncertainty_df=uncertainty_df,
        final_summary_df=final_summary_df,
        summary_by_tag_df=summary_by_tag_df,
    )
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
