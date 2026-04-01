from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from mpl_toolkits.mplot3d import proj3d

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "C_results"
FIGURES_DIR = PROJECT_ROOT / "B_figures"
WORKFLOW_XLSX = RESULTS_DIR / "nausicaa_workflow.xlsx"
RESULTS_XLSX = RESULTS_DIR / "nausicaa_results.xlsx"
WEIGHT_SWEEP_XLSX = RESULTS_DIR / "weight_sweep.xlsx"
TOP_RERUN_XLSX = RESULTS_DIR / "weight_sweep_top_rerun.xlsx"
FIGURE_PATH = FIGURES_DIR / "candidate_selection.png"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STAGE_ORDER = [
    "Weight sweep",
    "Selected sweep rows",
    "Top rerun starts",
    "Final rerun starts",
    "Retained robust rank",
]
STAGE_JITTER = {
    "Weight sweep": 0.08,
    "Selected sweep rows": 0.07,
    "Top rerun starts": 0.07,
    "Final rerun starts": 0.06,
    "Retained robust rank": 0.045,
}
EDGE_STYLE_MAP = {
    "background_terminated": {
        "color": "#9e9e9e",
        "linewidth": 0.75,
        "alpha": 0.12,
        "zorder": 1,
    },
    "promoted": {
        "color": "#7f9fc5",
        "linewidth": 0.85,
        "alpha": 0.22,
        "zorder": 2,
    },
    "final_rerun": {
        "color": "#4e79a7",
        "linewidth": 1.05,
        "alpha": 0.36,
        "zorder": 3,
    },
    "kept": {
        "color": "#4e79a7",
        "linewidth": 1.15,
        "alpha": 0.52,
        "zorder": 4,
    },
    "selected": {
        "color": "#f28e2b",
        "linewidth": 2.45,
        "alpha": 0.98,
        "zorder": 7,
    },
}
TRADEOFF_STAGE_ORDER = [
    "Weight sweep",
    "Selected sweep rows",
    "Top rerun summary",
    "Top rerun all starts",
    "All starts",
    "Candidates",
    "Robust summary",
]
TRADEOFF_STAGE_STYLES = {
    "Weight sweep": {"color": "#bdbdbd", "marker": "o", "size": 34.0, "alpha": 0.30},
    "Selected sweep rows": {"color": "#4e79a7", "marker": "s", "size": 54.0, "alpha": 0.58},
    "Top rerun summary": {"color": "#59a14f", "marker": "^", "size": 66.0, "alpha": 0.72},
    "Top rerun all starts": {"color": "#76b7b2", "marker": "X", "size": 44.0, "alpha": 0.42},
    "All starts": {"color": "#edc948", "marker": "D", "size": 40.0, "alpha": 0.42},
    "Candidates": {"color": "#e15759", "marker": "P", "size": 78.0, "alpha": 0.76},
    "Robust summary": {"color": "#af7aa1", "marker": "h", "size": 88.0, "alpha": 0.88},
}
TRACE_STAGE_MARKERS = {
    "Weight sweep": "o",
    "Selected sweep rows": "s",
    "Top rerun starts": "X",
    "Final rerun starts": "D",
    "Retained robust rank": "h",
}
TRACE_STAGE_Z_LABELS = {
    "Weight sweep": "Weight\nsweep",
    "Selected sweep rows": "Selected\nsweep rows",
    "Top rerun starts": "Top rerun\nstarts",
    "Final rerun starts": "Final rerun\nstarts",
    "Retained robust rank": "Retained\nrobust rank",
}
TRADEOFF_TO_TRACE_STAGE_MAP = {
    "Weight sweep": "Weight sweep",
    "Selected sweep rows": "Selected sweep rows",
    "Top rerun summary": "Top rerun starts",
    "Top rerun all starts": "Top rerun starts",
    "All starts": "Final rerun starts",
    "Candidates": "Retained robust rank",
    "Robust summary": "Retained robust rank",
}


@dataclass(frozen=True)
class CandidateSelectionData:
    weight_sweep_df: pd.DataFrame
    selected_sweep_rows_df: pd.DataFrame
    top_rerun_summary_df: pd.DataFrame
    top_rerun_all_starts_df: pd.DataFrame
    all_starts_df: pd.DataFrame
    candidates_df: pd.DataFrame
    robust_summary_df: pd.DataFrame


@dataclass(frozen=True)
class StageDisplayTransform:
    region_a_x_min: float
    region_a_x_span: float
    region_a_y_min: float
    region_a_y_span: float
    region_b_x_min: float
    region_b_x_span: float
    region_b_y_min: float
    region_b_y_span: float
    weight_x_min: float
    weight_x_span: float
    weight_y_min: float
    weight_y_span: float
    later_x_min: float
    later_x_span: float
    later_y_min: float
    later_y_span: float


def _open_workbook(path: Path) -> pd.ExcelFile | None:
    if path.exists():
        return pd.ExcelFile(path)
    return None


def _read_sheet(
    sheet_name: str,
    workflow_book: pd.ExcelFile | None,
    results_book: pd.ExcelFile | None,
) -> pd.DataFrame | None:
    if workflow_book is not None and sheet_name in workflow_book.sheet_names:
        return pd.read_excel(workflow_book, sheet_name=sheet_name)
    if results_book is not None and sheet_name in results_book.sheet_names:
        return pd.read_excel(results_book, sheet_name=sheet_name)
    return None


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"1", "true", "yes"})


def _sort_all_starts(all_starts_df: pd.DataFrame) -> pd.DataFrame:
    return all_starts_df.sort_values(by=["start_index"], kind="mergesort").reset_index(
        drop=True
    )


def _sort_feasible(feasible_df: pd.DataFrame) -> pd.DataFrame:
    return feasible_df.sort_values(
        by=["objective", "start_index"],
        kind="mergesort",
    ).reset_index(drop=True)


def _sort_kept(kept_df: pd.DataFrame) -> pd.DataFrame:
    if "kept_rank" in kept_df.columns and kept_df["kept_rank"].notna().any():
        return kept_df.sort_values(
            by=["kept_rank", "start_index"],
            kind="mergesort",
        ).reset_index(drop=True)
    return kept_df.sort_values(
        by=["objective", "start_index"],
        kind="mergesort",
    ).reset_index(drop=True)


def _resolve_kept_candidate_map(
    kept_df: pd.DataFrame,
    ranked_df: pd.DataFrame,
) -> dict[int, int]:
    candidate_ids = set(ranked_df["candidate_id"].astype(int).tolist())
    start_ids = kept_df["start_index"].astype(int)
    start_overlap = int(start_ids.isin(candidate_ids).sum())

    if "kept_rank" in kept_df.columns and kept_df["kept_rank"].notna().any():
        kept_rank_ids = kept_df["kept_rank"].dropna().astype(int)
        kept_rank_overlap = int(kept_rank_ids.isin(candidate_ids).sum())
        if kept_rank_overlap >= start_overlap and kept_rank_overlap > 0:
            return {
                int(row["start_index"]): int(row["kept_rank"])
                for _, row in kept_df.iterrows()
                if pd.notna(row["kept_rank"])
                and int(row["kept_rank"]) in candidate_ids
            }

    return {
        int(row["start_index"]): int(row["start_index"])
        for _, row in kept_df.iterrows()
        if int(row["start_index"]) in candidate_ids
    }


def _normalise_boolean_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    normalized_df = df.copy()
    for column in columns:
        if column in normalized_df.columns:
            normalized_df[column] = _coerce_bool_series(normalized_df[column])
    return normalized_df


def load_candidate_selection_data() -> CandidateSelectionData:
    workflow_book = _open_workbook(WORKFLOW_XLSX)
    results_book = _open_workbook(RESULTS_XLSX)
    weight_sweep_book = _open_workbook(WEIGHT_SWEEP_XLSX)
    top_rerun_book = _open_workbook(TOP_RERUN_XLSX)

    if weight_sweep_book is None:
        raise FileNotFoundError(f"Required workbook not found: {WEIGHT_SWEEP_XLSX}")
    if top_rerun_book is None:
        raise FileNotFoundError(f"Required workbook not found: {TOP_RERUN_XLSX}")

    weight_sweep_df = _read_sheet("WeightSweep", weight_sweep_book, None)
    if weight_sweep_df is None:
        weight_sweep_df = _read_sheet("TradeStudy", weight_sweep_book, None)
    selected_sweep_rows_df = _read_sheet("SelectedSweepRows", top_rerun_book, None)
    top_rerun_summary_df = _read_sheet("TopRerunSummary", top_rerun_book, None)
    top_rerun_all_starts_df = _read_sheet("TopRerunAllStarts", top_rerun_book, None)
    all_starts_df = _read_sheet("AllStarts", workflow_book, results_book)
    candidates_df = _read_sheet("Candidates", workflow_book, results_book)
    robust_summary_df = _read_sheet("RobustSummary", workflow_book, results_book)

    if weight_sweep_df is None:
        raise FileNotFoundError(
            "Required sheet 'WeightSweep' (or fallback 'TradeStudy') was not found in "
            f"{WEIGHT_SWEEP_XLSX}."
        )
    if (
        selected_sweep_rows_df is None
        or top_rerun_summary_df is None
        or top_rerun_all_starts_df is None
    ):
        raise FileNotFoundError(
            "Required top-rerun sheets were not found in "
            f"{TOP_RERUN_XLSX}."
        )
    if all_starts_df is None or candidates_df is None or robust_summary_df is None:
        raise FileNotFoundError(
            "Required workflow sheets were not found. Expected "
            f"{WORKFLOW_XLSX} with 'AllStarts', 'Candidates', and 'RobustSummary'."
        )

    weight_sweep_df = _normalise_boolean_columns(
        weight_sweep_df,
        ["success", "is_best"],
    )
    selected_sweep_rows_df = _normalise_boolean_columns(
        selected_sweep_rows_df,
        ["success", "is_best"],
    )
    top_rerun_summary_df = _normalise_boolean_columns(
        top_rerun_summary_df,
        ["success", "is_best"],
    )
    top_rerun_all_starts_df = _normalise_boolean_columns(
        top_rerun_all_starts_df,
        ["success", "kept_after_dedup"],
    )
    all_starts_df = _normalise_boolean_columns(
        all_starts_df,
        ["success", "kept_after_dedup"],
    )
    robust_summary_df = _normalise_boolean_columns(
        robust_summary_df,
        ["is_selected"],
    )

    return CandidateSelectionData(
        weight_sweep_df=weight_sweep_df,
        selected_sweep_rows_df=selected_sweep_rows_df,
        top_rerun_summary_df=top_rerun_summary_df,
        top_rerun_all_starts_df=top_rerun_all_starts_df,
        all_starts_df=all_starts_df,
        candidates_df=candidates_df,
        robust_summary_df=robust_summary_df,
    )


def resolve_tail_metric_column(df: pd.DataFrame) -> str:
    if "nom_sink_tail_mean_k" in df.columns:
        return "nom_sink_tail_mean_k"
    if "nom_sink_cvar_20" in df.columns:
        return "nom_sink_cvar_20"
    raise KeyError("Neither 'nom_sink_tail_mean_k' nor 'nom_sink_cvar_20' is available.")


def build_rank_columns(df: pd.DataFrame, tail_metric_col: str) -> pd.DataFrame:
    ranked_df = df.sort_values(
        by=["nom_success_rate", tail_metric_col, "objective"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked_df["robust_rank"] = np.arange(1, len(ranked_df) + 1, dtype=int)

    if "is_selected" in ranked_df.columns and ranked_df["is_selected"].any():
        ranked_df["is_selected"] = _coerce_bool_series(ranked_df["is_selected"])
    else:
        ranked_df["is_selected"] = ranked_df["robust_rank"].eq(1)

    return ranked_df


def _resolve_objective_column(df: pd.DataFrame) -> str:
    if "objective_nominal" in df.columns:
        return "objective_nominal"
    if "objective" in df.columns:
        return "objective"
    raise KeyError("Neither 'objective_nominal' nor 'objective' is available.")


def _resolve_tradeoff_tail_metric_column(
    df: pd.DataFrame,
    preferred_tail_metric_col: str,
) -> str:
    if "trace_sink_tail_mean_k" in df.columns:
        return "trace_sink_tail_mean_k"
    if "trace_sink_cvar_20" in df.columns:
        return "trace_sink_cvar_20"
    if preferred_tail_metric_col in df.columns:
        return preferred_tail_metric_col
    if "nom_sink_tail_mean_k" in df.columns:
        return "nom_sink_tail_mean_k"
    if "sink_tail_mean_k" in df.columns:
        return "sink_tail_mean_k"
    if "nom_sink_cvar_20" in df.columns:
        return "nom_sink_cvar_20"
    raise KeyError(
        "No tail-risk sink metric is available for trade-off plotting."
    )


def _build_tradeoff_stage_points(
    df: pd.DataFrame,
    stage: str,
    preferred_tail_metric_col: str,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "stage",
                "objective_plot",
                "tail_metric_plot",
                "nom_success_rate",
                "candidate_id",
                "is_selected",
            ]
        )

    try:
        objective_col = _resolve_objective_column(df)
        tail_metric_col = _resolve_tradeoff_tail_metric_column(
            df,
            preferred_tail_metric_col,
        )
    except KeyError:
        return pd.DataFrame(
            columns=[
                "stage",
                "objective_plot",
                "tail_metric_plot",
                "nom_success_rate",
                "candidate_id",
                "is_selected",
            ]
        )

    stage_df = df.copy()
    stage_df["objective_plot"] = pd.to_numeric(
        stage_df[objective_col],
        errors="coerce",
    )
    stage_df["tail_metric_plot"] = pd.to_numeric(
        stage_df[tail_metric_col],
        errors="coerce",
    )
    if "nom_success_rate" in stage_df.columns:
        stage_df["nom_success_rate"] = pd.to_numeric(
            stage_df["nom_success_rate"],
            errors="coerce",
        )
    else:
        stage_df["nom_success_rate"] = np.nan

    if "candidate_id" in stage_df.columns:
        stage_df["candidate_id"] = pd.to_numeric(
            stage_df["candidate_id"],
            errors="coerce",
        )
    else:
        stage_df["candidate_id"] = np.nan

    if "is_selected" in stage_df.columns:
        is_selected = stage_df["is_selected"]
        if is_selected.dtype != bool:
            is_selected = _coerce_bool_series(is_selected)
        stage_df["is_selected"] = is_selected
    else:
        stage_df["is_selected"] = False

    stage_df = stage_df.loc[
        stage_df["objective_plot"].notna() & stage_df["tail_metric_plot"].notna()
    ].copy()
    stage_df["stage"] = stage

    return stage_df[
        [
            "stage",
            "objective_plot",
            "tail_metric_plot",
            "nom_success_rate",
            "candidate_id",
            "is_selected",
        ]
    ].reset_index(drop=True)


def _coerce_float(value: object) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else float("nan")


def _safe_positive_span(span: float) -> float:
    return span if span > 1e-12 else 1.0


def _map_region_coordinates(
    x_values: pd.Series,
    y_values: pd.Series,
    source_x_min: float,
    source_x_span: float,
    source_y_min: float,
    source_y_span: float,
    target_x_min: float,
    target_x_span: float,
    target_y_min: float,
    target_y_span: float,
) -> tuple[pd.Series, pd.Series]:
    x_scaled = target_x_min + (x_values - source_x_min) * target_x_span / source_x_span
    y_scaled = target_y_min + (y_values - source_y_min) * target_y_span / source_y_span
    return x_scaled, y_scaled


def _build_stage_display_transform(tradeoff_df: pd.DataFrame) -> StageDisplayTransform | None:
    weight_df = tradeoff_df.loc[tradeoff_df["stage"] == "Weight sweep"].copy()
    later_df = tradeoff_df.loc[tradeoff_df["stage"] != "Weight sweep"].copy()
    if weight_df.empty or later_df.empty:
        return None

    weight_x_min = float(weight_df["objective_plot"].min())
    weight_x_max = float(weight_df["objective_plot"].max())
    weight_y_min = float(weight_df["tail_metric_plot"].min())
    weight_y_max = float(weight_df["tail_metric_plot"].max())
    later_x_min = float(later_df["objective_plot"].min())
    later_x_max = float(later_df["objective_plot"].max())
    later_y_min = float(later_df["tail_metric_plot"].min())
    later_y_max = float(later_df["tail_metric_plot"].max())

    later_x_span = _safe_positive_span(later_x_max - later_x_min)
    later_y_span = _safe_positive_span(later_y_max - later_y_min)
    weight_x_span = _safe_positive_span(weight_x_max - weight_x_min)
    weight_y_span = _safe_positive_span(weight_y_max - weight_y_min)
    panel_x_span = 4.8
    panel_y_span = 3.1

    return StageDisplayTransform(
        region_a_x_min=0.0,
        region_a_x_span=panel_x_span,
        region_a_y_min=0.0,
        region_a_y_span=panel_y_span,
        region_b_x_min=6.9,
        region_b_x_span=panel_x_span,
        region_b_y_min=0.55,
        region_b_y_span=panel_y_span,
        weight_x_min=weight_x_min,
        weight_x_span=weight_x_span,
        weight_y_min=weight_y_min,
        weight_y_span=weight_y_span,
        later_x_min=later_x_min,
        later_x_span=later_x_span,
        later_y_min=later_y_min,
        later_y_span=later_y_span,
    )


def _map_weight_sweep_display_coordinates(
    x_values: pd.Series,
    y_values: pd.Series,
    transform: StageDisplayTransform,
) -> tuple[pd.Series, pd.Series]:
    return _map_region_coordinates(
        x_values=x_values,
        y_values=y_values,
        source_x_min=transform.weight_x_min,
        source_x_span=transform.weight_x_span,
        source_y_min=transform.weight_y_min,
        source_y_span=transform.weight_y_span,
        target_x_min=transform.region_a_x_min,
        target_x_span=transform.region_a_x_span,
        target_y_min=transform.region_a_y_min,
        target_y_span=transform.region_a_y_span,
    )


def _map_later_stage_display_coordinates(
    x_values: pd.Series,
    y_values: pd.Series,
    transform: StageDisplayTransform,
) -> tuple[pd.Series, pd.Series]:
    return _map_region_coordinates(
        x_values=x_values,
        y_values=y_values,
        source_x_min=transform.later_x_min,
        source_x_span=transform.later_x_span,
        source_y_min=transform.later_y_min,
        source_y_span=transform.later_y_span,
        target_x_min=transform.region_b_x_min,
        target_x_span=transform.region_b_x_span,
        target_y_min=transform.region_b_y_min,
        target_y_span=transform.region_b_y_span,
    )


def apply_stage_display_transform(
    tradeoff_df: pd.DataFrame,
    trace_nodes_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StageDisplayTransform | None]:
    transform = _build_stage_display_transform(tradeoff_df)

    tradeoff_plot_df = tradeoff_df.copy()
    tradeoff_plot_df["objective_display"] = tradeoff_plot_df["objective_plot"]
    tradeoff_plot_df["tail_metric_display"] = tradeoff_plot_df["tail_metric_plot"]

    trace_plot_df = trace_nodes_df.copy()
    trace_plot_df["objective_display"] = trace_plot_df["objective_plot"]
    trace_plot_df["tail_metric_display"] = trace_plot_df["tail_metric_plot"]

    if transform is None:
        return tradeoff_plot_df, trace_plot_df, None

    weight_mask = tradeoff_plot_df["stage"].eq("Weight sweep")
    if weight_mask.any():
        x_scaled, y_scaled = _map_weight_sweep_display_coordinates(
            tradeoff_plot_df.loc[weight_mask, "objective_plot"],
            tradeoff_plot_df.loc[weight_mask, "tail_metric_plot"],
            transform,
        )
        tradeoff_plot_df.loc[weight_mask, "objective_display"] = x_scaled
        tradeoff_plot_df.loc[weight_mask, "tail_metric_display"] = y_scaled

    later_mask = ~weight_mask
    if later_mask.any():
        x_scaled, y_scaled = _map_later_stage_display_coordinates(
            tradeoff_plot_df.loc[later_mask, "objective_plot"],
            tradeoff_plot_df.loc[later_mask, "tail_metric_plot"],
            transform,
        )
        tradeoff_plot_df.loc[later_mask, "objective_display"] = x_scaled
        tradeoff_plot_df.loc[later_mask, "tail_metric_display"] = y_scaled

    trace_weight_mask = trace_plot_df["stage"].eq("Weight sweep")
    if trace_weight_mask.any():
        x_scaled, y_scaled = _map_weight_sweep_display_coordinates(
            trace_plot_df.loc[trace_weight_mask, "objective_plot"],
            trace_plot_df.loc[trace_weight_mask, "tail_metric_plot"],
            transform,
        )
        trace_plot_df.loc[trace_weight_mask, "objective_display"] = x_scaled
        trace_plot_df.loc[trace_weight_mask, "tail_metric_display"] = y_scaled

    trace_later_mask = ~trace_weight_mask
    if trace_later_mask.any():
        x_scaled, y_scaled = _map_later_stage_display_coordinates(
            trace_plot_df.loc[trace_later_mask, "objective_plot"],
            trace_plot_df.loc[trace_later_mask, "tail_metric_plot"],
            transform,
        )
        trace_plot_df.loc[trace_later_mask, "objective_display"] = x_scaled
        trace_plot_df.loc[trace_later_mask, "tail_metric_display"] = y_scaled

    return tradeoff_plot_df, trace_plot_df, transform


def _panel_tick_values(
    min_value: float,
    span_value: float,
    count: int = 3,
) -> np.ndarray:
    if span_value <= 1e-12:
        return np.asarray([min_value], dtype=float)
    return np.linspace(min_value, min_value + span_value, count)


def _map_tick_values_to_region(
    tick_values: np.ndarray,
    source_min: float,
    source_span: float,
    target_min: float,
    target_span: float,
) -> np.ndarray:
    return target_min + (tick_values - source_min) * target_span / source_span


def _draw_trade_space_panel(
    ax: plt.Axes,
    x_min: float,
    x_span: float,
    y_min: float,
    y_span: float,
    z_level: float,
    source_x_min: float,
    source_x_span: float,
    source_y_min: float,
    source_y_span: float,
    panel_label: str,
    frame_color: str,
) -> None:
    x_max = x_min + x_span
    y_max = y_min + y_span
    x_tick_values = _panel_tick_values(source_x_min, source_x_span)
    y_tick_values = _panel_tick_values(source_y_min, source_y_span)
    x_tick_positions = _map_tick_values_to_region(
        x_tick_values,
        source_x_min,
        source_x_span,
        x_min,
        x_span,
    )
    y_tick_positions = _map_tick_values_to_region(
        y_tick_values,
        source_y_min,
        source_y_span,
        y_min,
        y_span,
    )
    x_tick_len = 0.06 * y_span
    y_tick_len = 0.06 * x_span

    for x_pair, y_pair in [
        ([x_min, x_max], [y_min, y_min]),
        ([x_max, x_max], [y_min, y_max]),
        ([x_max, x_min], [y_max, y_max]),
        ([x_min, x_min], [y_max, y_min]),
    ]:
        ax.plot(
            x_pair,
            y_pair,
            [z_level, z_level],
            color=frame_color,
            linewidth=0.85,
            linestyle="--",
            alpha=0.72,
            zorder=1,
        )

    for tick_position, tick_value in zip(x_tick_positions, x_tick_values, strict=False):
        ax.plot(
            [tick_position, tick_position],
            [y_min, y_min - x_tick_len],
            [z_level, z_level],
            color=frame_color,
            linewidth=0.7,
            alpha=0.75,
            zorder=1,
        )
        ax.text(
            tick_position,
            y_min - 1.9 * x_tick_len,
            z_level,
            f"{tick_value:.2f}",
            fontsize=7,
            ha="center",
            va="top",
            color=frame_color,
        )

    for tick_position, tick_value in zip(y_tick_positions, y_tick_values, strict=False):
        ax.plot(
            [x_min, x_min - y_tick_len],
            [tick_position, tick_position],
            [z_level, z_level],
            color=frame_color,
            linewidth=0.7,
            alpha=0.75,
            zorder=1,
        )
        ax.text(
            x_min - 1.4 * y_tick_len,
            tick_position,
            z_level,
            f"{tick_value:.2f}",
            fontsize=7,
            ha="right",
            va="center",
            color=frame_color,
        )

    ax.text(
        0.5 * (x_min + x_max),
        y_min - 3.2 * x_tick_len,
        z_level,
        "Nominal objective",
        fontsize=8,
        ha="center",
        va="top",
        color=frame_color,
    )
    ax.text(
        x_min - 2.8 * y_tick_len,
        0.5 * (y_min + y_max),
        z_level,
        "Tail-risk mean\n(worst 20%)",
        fontsize=8,
        ha="center",
        va="center",
        color=frame_color,
    )
    ax.text(
        x_min + 0.05 * x_span,
        y_max + 0.08 * y_span,
        z_level,
        panel_label,
        fontsize=8,
        fontweight="bold",
        color=frame_color,
    )


def _extract_tradeoff_pair(
    row: pd.Series | dict[str, object],
    preferred_tail_metric_col: str,
) -> tuple[float, float]:
    source = row if isinstance(row, dict) else row.to_dict()
    objective = float("nan")
    for key in ["objective_nominal", "objective"]:
        if key in source:
            objective = _coerce_float(source.get(key))
        if np.isfinite(objective):
            break

    tail_metric = float("nan")
    for key in [
        "trace_sink_tail_mean_k",
        "trace_sink_cvar_20",
        preferred_tail_metric_col,
        "nom_sink_tail_mean_k",
        "sink_tail_mean_k",
        "nom_sink_cvar_20",
    ]:
        if key in source:
            tail_metric = _coerce_float(source.get(key))
        if np.isfinite(tail_metric):
            break

    return objective, tail_metric


def build_tradeoff_points(
    data: CandidateSelectionData,
    ranked_df: pd.DataFrame,
    tail_metric_col: str,
) -> pd.DataFrame:
    stage_frames = [
        _build_tradeoff_stage_points(
            data.weight_sweep_df,
            stage="Weight sweep",
            preferred_tail_metric_col=tail_metric_col,
        ),
        _build_tradeoff_stage_points(
            data.selected_sweep_rows_df,
            stage="Selected sweep rows",
            preferred_tail_metric_col=tail_metric_col,
        ),
        _build_tradeoff_stage_points(
            data.top_rerun_summary_df,
            stage="Top rerun summary",
            preferred_tail_metric_col=tail_metric_col,
        ),
        _build_tradeoff_stage_points(
            data.top_rerun_all_starts_df,
            stage="Top rerun all starts",
            preferred_tail_metric_col=tail_metric_col,
        ),
        _build_tradeoff_stage_points(
            data.all_starts_df,
            stage="All starts",
            preferred_tail_metric_col=tail_metric_col,
        ),
        _build_tradeoff_stage_points(
            data.candidates_df,
            stage="Candidates",
            preferred_tail_metric_col=tail_metric_col,
        ),
        _build_tradeoff_stage_points(
            ranked_df,
            stage="Robust summary",
            preferred_tail_metric_col=tail_metric_col,
        ),
    ]

    non_empty_stage_frames = [frame for frame in stage_frames if not frame.empty]
    if not non_empty_stage_frames:
        return pd.DataFrame(
            columns=[
                "stage",
                "objective_plot",
                "tail_metric_plot",
                "nom_success_rate",
                "candidate_id",
                "is_selected",
            ]
        )

    tradeoff_df = pd.concat(non_empty_stage_frames, ignore_index=True)
    tradeoff_df["stage"] = pd.Categorical(
        tradeoff_df["stage"],
        categories=TRADEOFF_STAGE_ORDER,
        ordered=True,
    )
    return tradeoff_df.sort_values(
        by=["stage", "objective_plot", "tail_metric_plot"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_tradeoff_trace_nodes(
    data: CandidateSelectionData,
    nodes_df: pd.DataFrame,
    ranked_df: pd.DataFrame,
    tail_metric_col: str,
) -> pd.DataFrame:
    trace_nodes_df = nodes_df.copy()
    trace_nodes_df["objective_plot"] = np.nan
    trace_nodes_df["tail_metric_plot"] = np.nan

    ranked_lookup = {
        str(int(row["candidate_id"])): _extract_tradeoff_pair(row, tail_metric_col)
        for _, row in ranked_df.iterrows()
    }

    weight_sweep_lookup = {
        str(int(row["run_index"])): _extract_tradeoff_pair(row, tail_metric_col)
        for _, row in data.weight_sweep_df.iterrows()
        if pd.notna(row.get("run_index"))
    }
    selected_sweep_lookup = {
        str(int(row["run_index"])): _extract_tradeoff_pair(row, tail_metric_col)
        for _, row in data.selected_sweep_rows_df.iterrows()
        if pd.notna(row.get("run_index"))
    }

    final_all_starts_df = _sort_all_starts(data.all_starts_df.copy())
    top_rerun_all_starts_df = data.top_rerun_all_starts_df.sort_values(
        by=["rerun_rank", "start_index"],
        kind="mergesort",
    ).reset_index(drop=True)
    final_kept_df = _sort_kept(
        final_all_starts_df.loc[final_all_starts_df["kept_after_dedup"]].copy()
    )
    kept_to_candidate_map = _resolve_final_kept_candidate_map(final_kept_df, ranked_df)
    final_rerun_rank = infer_final_rerun_rank(
        final_all_starts_df,
        top_rerun_all_starts_df,
    )

    top_rerun_start_lookup: dict[str, tuple[float, float]] = {}
    for _, row in top_rerun_all_starts_df.iterrows():
        rerun_rank = int(row["rerun_rank"])
        start_index = int(row["start_index"])
        objective_value, tail_metric_value = _extract_tradeoff_pair(row, tail_metric_col)
        if (
            not np.isfinite(tail_metric_value)
            and rerun_rank == final_rerun_rank
            and pd.notna(row.get("kept_rank"))
        ):
            candidate_id = kept_to_candidate_map.get(int(row["kept_rank"]))
            if candidate_id is not None:
                _, tail_metric_value = ranked_lookup.get(
                    str(int(candidate_id)),
                    (float("nan"), float("nan")),
                )
        top_rerun_start_lookup[f"{rerun_rank}:{start_index}"] = (
            objective_value,
            tail_metric_value,
        )

    final_start_lookup: dict[str, tuple[float, float]] = {}
    for _, row in final_all_starts_df.iterrows():
        start_index = int(row["start_index"])
        objective_value, tail_metric_value = _extract_tradeoff_pair(row, tail_metric_col)
        if not np.isfinite(tail_metric_value) and pd.notna(row.get("kept_rank")):
            candidate_id = kept_to_candidate_map.get(int(row["kept_rank"]))
            if candidate_id is not None:
                _, tail_metric_value = ranked_lookup.get(
                    str(int(candidate_id)),
                    (float("nan"), float("nan")),
                )
        final_start_lookup[str(start_index)] = (
            objective_value,
            tail_metric_value,
        )

    final_kept_lookup: dict[str, tuple[float, float]] = {}
    for _, row in final_kept_df.iterrows():
        kept_rank = int(row["kept_rank"])
        objective_value, tail_metric_value = _extract_tradeoff_pair(row, tail_metric_col)
        candidate_id = kept_to_candidate_map.get(kept_rank)
        if candidate_id is not None:
            _, ranked_tail_metric = ranked_lookup.get(
                str(int(candidate_id)),
                (float("nan"), float("nan")),
            )
            if not np.isfinite(tail_metric_value):
                tail_metric_value = ranked_tail_metric
        final_kept_lookup[str(kept_rank)] = (
            objective_value,
            tail_metric_value,
        )

    lookup_by_stage = {
        "Weight sweep": weight_sweep_lookup,
        "Selected sweep rows": selected_sweep_lookup,
        "Top rerun starts": top_rerun_start_lookup,
        "Final rerun starts": final_start_lookup,
        "Retained robust rank": ranked_lookup,
    }

    for index, row in trace_nodes_df.iterrows():
        stage_lookup = lookup_by_stage.get(str(row["stage"]))
        if stage_lookup is None:
            continue
        objective_value, tail_metric_value = stage_lookup.get(
            str(row["node_key"]),
            (float("nan"), float("nan")),
        )
        trace_nodes_df.at[index, "objective_plot"] = objective_value
        trace_nodes_df.at[index, "tail_metric_plot"] = tail_metric_value

    return trace_nodes_df


def _stable_jitter(identifier: int, scale: float) -> float:
    value = np.sin(float(identifier) * 12.9898) * 43758.5453
    frac = value - np.floor(value)
    return scale * (frac - 0.5)


def _draw_curve(
    ax: plt.Axes,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: str,
    linewidth: float,
    alpha: float,
    zorder: int,
) -> None:
    t = np.linspace(0.0, 1.0, 40)
    c1x = x0 + 0.35 * (x1 - x0)
    c2x = x0 + 0.65 * (x1 - x0)
    c1y = y0
    c2y = y1
    x = (
        (1.0 - t) ** 3 * x0
        + 3.0 * (1.0 - t) ** 2 * t * c1x
        + 3.0 * (1.0 - t) * t**2 * c2x
        + t**3 * x1
    )
    y = (
        (1.0 - t) ** 3 * y0
        + 3.0 * (1.0 - t) ** 2 * t * c1y
        + 3.0 * (1.0 - t) * t**2 * c2y
        + t**3 * y1
    )
    ax.plot(
        x,
        y,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
    )


def _draw_curve_3d(
    ax: plt.Axes,
    x0: float,
    y0: float,
    z0: float,
    x1: float,
    y1: float,
    z1: float,
    color: str,
    linewidth: float,
    alpha: float,
    zorder: int,
) -> None:
    t = np.linspace(0.0, 1.0, 40)
    c1z = z0 + 0.35 * (z1 - z0)
    c2z = z0 + 0.65 * (z1 - z0)
    x = (
        (1.0 - t) ** 3 * x0
        + 3.0 * (1.0 - t) ** 2 * t * x0
        + 3.0 * (1.0 - t) * t**2 * x1
        + t**3 * x1
    )
    y = (
        (1.0 - t) ** 3 * y0
        + 3.0 * (1.0 - t) ** 2 * t * y0
        + 3.0 * (1.0 - t) * t**2 * y1
        + t**3 * y1
    )
    z = (
        (1.0 - t) ** 3 * z0
        + 3.0 * (1.0 - t) ** 2 * t * c1z
        + 3.0 * (1.0 - t) * t**2 * c2z
        + t**3 * z1
    )
    ax.plot(
        x,
        y,
        z,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
    )


def infer_final_rerun_rank(
    final_all_starts_df: pd.DataFrame,
    top_rerun_all_starts_df: pd.DataFrame,
) -> int:
    compare_columns = [
        "start_index",
        "objective",
        "success",
        "kept_after_dedup",
        "kept_rank",
    ]
    required_columns = [
        column
        for column in compare_columns
        if column in final_all_starts_df.columns and column in top_rerun_all_starts_df.columns
    ]
    if len(required_columns) != len(compare_columns):
        raise ValueError(
            "Cannot infer final rerun rank because the required AllStarts signature "
            "columns are incomplete."
        )

    final_df = _sort_all_starts(final_all_starts_df[required_columns].copy())
    final_df["start_index"] = pd.to_numeric(
        final_df["start_index"],
        errors="coerce",
    ).fillna(-1).astype(int)
    final_df["objective"] = pd.to_numeric(final_df["objective"], errors="coerce")
    final_df["kept_rank"] = pd.to_numeric(final_df["kept_rank"], errors="coerce")

    matched_ranks: list[int] = []
    for rerun_rank, group_df in top_rerun_all_starts_df.groupby("rerun_rank", sort=False):
        group_compare_df = _sort_all_starts(group_df[required_columns].copy())
        group_compare_df["start_index"] = pd.to_numeric(
            group_compare_df["start_index"],
            errors="coerce",
        ).fillna(-1).astype(int)
        group_compare_df["objective"] = pd.to_numeric(
            group_compare_df["objective"],
            errors="coerce",
        )
        group_compare_df["kept_rank"] = pd.to_numeric(
            group_compare_df["kept_rank"],
            errors="coerce",
        )

        if len(group_compare_df) != len(final_df):
            continue
        if not np.array_equal(
            group_compare_df["start_index"].to_numpy(),
            final_df["start_index"].to_numpy(),
        ):
            continue
        if not np.array_equal(
            group_compare_df["success"].to_numpy(dtype=bool),
            final_df["success"].to_numpy(dtype=bool),
        ):
            continue
        if not np.array_equal(
            group_compare_df["kept_after_dedup"].to_numpy(dtype=bool),
            final_df["kept_after_dedup"].to_numpy(dtype=bool),
        ):
            continue
        if not np.allclose(
            group_compare_df["objective"].fillna(np.nan).to_numpy(dtype=float),
            final_df["objective"].fillna(np.nan).to_numpy(dtype=float),
            atol=1e-9,
            rtol=1e-9,
            equal_nan=True,
        ):
            continue
        if not np.allclose(
            group_compare_df["kept_rank"].fillna(-1.0).to_numpy(dtype=float),
            final_df["kept_rank"].fillna(-1.0).to_numpy(dtype=float),
            atol=1e-9,
            rtol=1e-9,
            equal_nan=True,
        ):
            continue
        matched_ranks.append(int(rerun_rank))

    if len(matched_ranks) != 1:
        raise ValueError(
            "Could not infer a unique final rerun rank from TopRerunAllStarts "
            f"and final AllStarts. Matches found: {matched_ranks}"
        )

    return matched_ranks[0]


def _resolve_final_kept_candidate_map(
    final_kept_df: pd.DataFrame,
    ranked_df: pd.DataFrame,
) -> dict[int, int]:
    candidate_ids = set(ranked_df["candidate_id"].astype(int).tolist())
    kept_rank_ids = (
        final_kept_df["kept_rank"].dropna().astype(int)
        if "kept_rank" in final_kept_df.columns
        else pd.Series(dtype=int)
    )
    start_ids = final_kept_df["start_index"].astype(int)
    kept_rank_overlap = int(kept_rank_ids.isin(candidate_ids).sum())
    start_overlap = int(start_ids.isin(candidate_ids).sum())

    if kept_rank_overlap >= start_overlap and kept_rank_overlap > 0:
        return {
            int(row["kept_rank"]): int(row["kept_rank"])
            for _, row in final_kept_df.iterrows()
            if pd.notna(row["kept_rank"])
            and int(row["kept_rank"]) in candidate_ids
        }

    return {
        int(row["kept_rank"]): int(row["start_index"])
        for _, row in final_kept_df.iterrows()
        if pd.notna(row["kept_rank"])
        and int(row["start_index"]) in candidate_ids
    }


def build_full_provenance(
    data: CandidateSelectionData,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, int]:
    retained_df = data.candidates_df.merge(
        data.robust_summary_df,
        on="candidate_id",
        how="left",
        suffixes=("", "_robust"),
    )
    tail_metric_col = resolve_tail_metric_column(retained_df)
    ranked_df = build_rank_columns(retained_df, tail_metric_col)
    ranked_df = ranked_df.sort_values(by="robust_rank", kind="mergesort").reset_index(
        drop=True
    )
    selected_candidate_id = int(
        ranked_df.loc[ranked_df["is_selected"], "candidate_id"].iloc[0]
    )

    weight_sweep_df = data.weight_sweep_df.sort_values(
        by=["run_index"],
        kind="mergesort",
    ).reset_index(drop=True)
    selected_sweep_rows_df = data.selected_sweep_rows_df.sort_values(
        by=["run_index"],
        kind="mergesort",
    ).reset_index(drop=True)
    top_rerun_summary_df = data.top_rerun_summary_df.sort_values(
        by=["rerun_rank"],
        kind="mergesort",
    ).reset_index(drop=True)
    top_rerun_all_starts_df = data.top_rerun_all_starts_df.sort_values(
        by=["rerun_rank", "start_index"],
        kind="mergesort",
    ).reset_index(drop=True)
    final_all_starts_df = _sort_all_starts(data.all_starts_df)
    final_rerun_rank = infer_final_rerun_rank(
        final_all_starts_df,
        top_rerun_all_starts_df,
    )
    final_rerun_source_run_index = int(
        top_rerun_summary_df.loc[
            top_rerun_summary_df["rerun_rank"] == final_rerun_rank,
            "source_sweep_run_index",
        ].iloc[0]
    )

    final_kept_df = _sort_kept(
        final_all_starts_df.loc[final_all_starts_df["kept_after_dedup"]].copy()
    )
    start_to_candidate_map = _resolve_kept_candidate_map(final_kept_df, ranked_df)
    kept_to_candidate_map = _resolve_final_kept_candidate_map(final_kept_df, ranked_df)
    selected_start_id = next(
        start_id
        for start_id, candidate_id in start_to_candidate_map.items()
        if candidate_id == selected_candidate_id
    )
    selected_sweep_run_indices = set(
        selected_sweep_rows_df["run_index"].astype(int).tolist()
    )

    node_rows: list[dict[str, object]] = []
    for _, row in weight_sweep_df.iterrows():
        run_index = int(row["run_index"])
        node_rows.append(
            {
                "stage": "Weight sweep",
                "node_key": str(run_index),
                "display_label": str(run_index),
                "order_value": float(run_index),
                "group_key": "weight_sweep",
                "status_class": (
                    "final_rerun"
                    if run_index == final_rerun_source_run_index
                    else "promoted"
                    if run_index in selected_sweep_run_indices
                    else "background_terminated"
                ),
            }
        )

    for _, row in selected_sweep_rows_df.iterrows():
        run_index = int(row["run_index"])
        node_rows.append(
            {
                "stage": "Selected sweep rows",
                "node_key": str(run_index),
                "display_label": str(run_index),
                "order_value": float(run_index),
                "group_key": "selected_sweep_rows",
                "status_class": (
                    "final_rerun"
                    if run_index == final_rerun_source_run_index
                    else "promoted"
                ),
            }
        )

    for _, row in top_rerun_all_starts_df.iterrows():
        rerun_rank = int(row["rerun_rank"])
        start_index = int(row["start_index"])
        node_rows.append(
            {
                "stage": "Top rerun starts",
                "node_key": f"{rerun_rank}:{start_index}",
                "display_label": f"{rerun_rank}:{start_index}",
                "order_value": float(1000 * rerun_rank + start_index),
                "group_key": str(rerun_rank),
                "status_class": (
                    "final_rerun"
                    if rerun_rank == final_rerun_rank
                    else "promoted"
                ),
            }
        )

    for _, row in final_all_starts_df.iterrows():
        start_index = int(row["start_index"])
        node_rows.append(
            {
                "stage": "Final rerun starts",
                "node_key": str(start_index),
                "display_label": str(start_index),
                "order_value": float(start_index),
                "group_key": "final_rerun",
                "status_class": (
                    "kept" if bool(row["kept_after_dedup"]) else "background_terminated"
                ),
            }
        )

    for _, row in ranked_df.iterrows():
        candidate_id = int(row["candidate_id"])
        node_rows.append(
            {
                "stage": "Retained robust rank",
                "node_key": str(candidate_id),
                "display_label": f"#{int(row['robust_rank'])}",
                "order_value": float(row["robust_rank"]),
                "group_key": "retained_robust_rank",
                "status_class": (
                    "selected" if candidate_id == selected_candidate_id else "kept"
                ),
                "nom_success_rate": float(row["nom_success_rate"]),
            }
        )

    edge_rows: list[dict[str, object]] = []
    for _, row in selected_sweep_rows_df.iterrows():
        run_index = int(row["run_index"])
        edge_rows.append(
            {
                "source_stage": "Weight sweep",
                "source_key": str(run_index),
                "target_stage": "Selected sweep rows",
                "target_key": str(run_index),
                "edge_class": "promoted",
                "trace_group": f"sweep:{run_index}",
                "is_selected_path": run_index == final_rerun_source_run_index,
            }
        )

    for _, row in top_rerun_all_starts_df.iterrows():
        rerun_rank = int(row["rerun_rank"])
        source_run_index = int(row["source_sweep_run_index"])
        start_index = int(row["start_index"])
        edge_rows.append(
            {
                "source_stage": "Selected sweep rows",
                "source_key": str(source_run_index),
                "target_stage": "Top rerun starts",
                "target_key": f"{rerun_rank}:{start_index}",
                "edge_class": "promoted",
                "trace_group": f"rerun:{rerun_rank}",
                "is_selected_path": (
                    rerun_rank == final_rerun_rank and start_index == selected_start_id
                ),
            }
        )

    final_top_rerun_starts_df = _sort_all_starts(
        top_rerun_all_starts_df.loc[
            top_rerun_all_starts_df["rerun_rank"] == final_rerun_rank
        ].copy()
    )
    for _, row in final_top_rerun_starts_df.iterrows():
        start_index = int(row["start_index"])
        edge_rows.append(
            {
                "source_stage": "Top rerun starts",
                "source_key": f"{final_rerun_rank}:{start_index}",
                "target_stage": "Final rerun starts",
                "target_key": str(start_index),
                "edge_class": "final_rerun",
                "trace_group": f"final_rerun:{start_index}",
                "is_selected_path": start_index == selected_start_id,
            }
        )

    for _, row in final_kept_df.iterrows():
        start_index = int(row["start_index"])
        kept_rank = int(row["kept_rank"])
        candidate_id = kept_to_candidate_map[kept_rank]
        edge_rows.append(
            {
                "source_stage": "Final rerun starts",
                "source_key": str(start_index),
                "target_stage": "Retained robust rank",
                "target_key": str(candidate_id),
                "edge_class": "kept",
                "trace_group": f"candidate:{candidate_id}",
                "is_selected_path": start_index == selected_start_id,
            }
        )

    nodes_df = pd.DataFrame(node_rows)
    edges_df = pd.DataFrame(edge_rows)
    return nodes_df, edges_df, ranked_df, tail_metric_col, selected_candidate_id


def build_stage_positions(
    nodes_df: pd.DataFrame,
) -> tuple[dict[str, float], dict[tuple[str, str], float]]:
    stage_x_map = {stage: float(index) for index, stage in enumerate(STAGE_ORDER)}
    position_map: dict[tuple[str, str], float] = {}

    for stage in STAGE_ORDER:
        stage_df = nodes_df.loc[nodes_df["stage"] == stage].copy()
        stage_df = stage_df.sort_values(
            by=["order_value", "node_key"],
            kind="mergesort",
        ).reset_index(drop=True)
        stage_count = len(stage_df)
        jitter_scale = STAGE_JITTER.get(stage, 0.05)
        for index, row in stage_df.iterrows():
            base_y = float(stage_count - index - 1)
            identifier_str = f"{stage}|{row['node_key']}"
            identifier = sum(
                (char_index + 1) * ord(character)
                for char_index, character in enumerate(identifier_str)
            )
            position_map[(stage, str(row["node_key"]))] = base_y + _stable_jitter(
                identifier,
                jitter_scale,
            )

    return stage_x_map, position_map


def draw_provenance_panel(
    ax: plt.Axes,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    ranked_df: pd.DataFrame,
    stage_x_map: dict[str, float],
    position_map: dict[tuple[str, str], float],
    selected_candidate_id: int,
) -> None:
    max_stage_size = max(nodes_df.groupby("stage", sort=False).size().tolist() + [1])
    y_top = float(max_stage_size) + 0.7

    for stage in STAGE_ORDER:
        x_center = stage_x_map[stage]
        ax.axvspan(
            x_center - 0.10,
            x_center + 0.10,
            facecolor="#d9d9d9",
            alpha=0.08,
            zorder=0,
        )
        ax.axvline(
            x_center,
            color="#c7c7c7",
            linewidth=0.6,
            alpha=0.45,
            zorder=0,
        )

    edge_class_order = {
        "background_terminated": 0,
        "promoted": 1,
        "final_rerun": 2,
        "kept": 3,
        "selected": 4,
    }
    edges_to_draw = edges_df.copy()
    edges_to_draw["draw_class"] = np.where(
        edges_to_draw["is_selected_path"],
        "selected",
        edges_to_draw["edge_class"],
    )
    edges_to_draw["draw_order"] = edges_to_draw["draw_class"].map(edge_class_order)
    edges_to_draw = edges_to_draw.sort_values(
        by=["draw_order", "source_stage", "target_stage", "trace_group"],
        kind="mergesort",
    )

    for _, row in edges_to_draw.iterrows():
        draw_class = str(row["draw_class"])
        style = EDGE_STYLE_MAP[draw_class]
        source_key = (str(row["source_stage"]), str(row["source_key"]))
        target_key = (str(row["target_stage"]), str(row["target_key"]))
        _draw_curve(
            ax,
            stage_x_map[source_key[0]],
            position_map[source_key],
            stage_x_map[target_key[0]],
            position_map[target_key],
            color=str(style["color"]),
            linewidth=float(style["linewidth"]),
            alpha=float(style["alpha"]),
            zorder=int(style["zorder"]),
        )

    rank_success_map = {
        str(int(row["candidate_id"])): float(row["nom_success_rate"])
        for _, row in ranked_df.iterrows()
    }
    selected_trace_key = str(selected_candidate_id)

    for stage in STAGE_ORDER:
        stage_df = nodes_df.loc[nodes_df["stage"] == stage].copy()
        stage_df = stage_df.sort_values(
            by=["order_value", "node_key"],
            kind="mergesort",
        ).reset_index(drop=True)
        x_values = np.full(len(stage_df), stage_x_map[stage])
        y_values = np.asarray(
            [position_map[(stage, str(node_key))] for node_key in stage_df["node_key"]],
            dtype=float,
        )

        if stage == "Weight sweep":
            promoted_mask = stage_df["status_class"].isin(["promoted", "final_rerun"])
            background_mask = ~promoted_mask
            if background_mask.any():
                ax.scatter(
                    x_values[background_mask.to_numpy()],
                    y_values[background_mask.to_numpy()],
                    s=16,
                    facecolors="#bfbfbf",
                    edgecolors="none",
                    alpha=0.75,
                    zorder=4,
                )
            if promoted_mask.any():
                ax.scatter(
                    x_values[promoted_mask.to_numpy()],
                    y_values[promoted_mask.to_numpy()],
                    s=22,
                    facecolors="#7f9fc5",
                    edgecolors="white",
                    linewidths=0.4,
                    alpha=0.9,
                    zorder=5,
                )
        elif stage == "Selected sweep rows":
            final_mask = stage_df["status_class"].eq("final_rerun")
            other_mask = ~final_mask
            if other_mask.any():
                ax.scatter(
                    x_values[other_mask.to_numpy()],
                    y_values[other_mask.to_numpy()],
                    s=26,
                    facecolors="#7f9fc5",
                    edgecolors="white",
                    linewidths=0.5,
                    alpha=0.92,
                    zorder=5,
                )
            if final_mask.any():
                ax.scatter(
                    x_values[final_mask.to_numpy()],
                    y_values[final_mask.to_numpy()],
                    s=34,
                    facecolors="#4e79a7",
                    edgecolors="black",
                    linewidths=0.6,
                    alpha=0.96,
                    zorder=6,
                )
        elif stage == "Top rerun starts":
            final_mask = stage_df["status_class"].eq("final_rerun")
            other_mask = ~final_mask
            if other_mask.any():
                ax.scatter(
                    x_values[other_mask.to_numpy()],
                    y_values[other_mask.to_numpy()],
                    s=18,
                    facecolors="#c9d7e6",
                    edgecolors="white",
                    linewidths=0.4,
                    alpha=0.85,
                    zorder=4,
                )
            if final_mask.any():
                ax.scatter(
                    x_values[final_mask.to_numpy()],
                    y_values[final_mask.to_numpy()],
                    s=24,
                    facecolors="#4e79a7",
                    edgecolors="white",
                    linewidths=0.4,
                    alpha=0.92,
                    zorder=5,
                )
        elif stage == "Final rerun starts":
            kept_mask = stage_df["status_class"].eq("kept")
            background_mask = ~kept_mask
            if background_mask.any():
                ax.scatter(
                    x_values[background_mask.to_numpy()],
                    y_values[background_mask.to_numpy()],
                    s=22,
                    facecolors="white",
                    edgecolors="#9e9e9e",
                    linewidths=0.8,
                    alpha=0.9,
                    zorder=4,
                )
            if kept_mask.any():
                ax.scatter(
                    x_values[kept_mask.to_numpy()],
                    y_values[kept_mask.to_numpy()],
                    s=26,
                    facecolors="#4e79a7",
                    edgecolors="white",
                    linewidths=0.4,
                    alpha=0.94,
                    zorder=5,
                )
        elif stage == "Retained robust rank":
            selected_mask = stage_df["status_class"].eq("selected")
            kept_mask = ~selected_mask
            if kept_mask.any():
                ax.scatter(
                    x_values[kept_mask.to_numpy()],
                    y_values[kept_mask.to_numpy()],
                    s=30,
                    facecolors="#4e79a7",
                    edgecolors="white",
                    linewidths=0.5,
                    alpha=0.96,
                    zorder=5,
                )
            if selected_mask.any():
                ax.scatter(
                    x_values[selected_mask.to_numpy()],
                    y_values[selected_mask.to_numpy()],
                    s=46,
                    facecolors="#f28e2b",
                    edgecolors="black",
                    linewidths=0.8,
                    alpha=0.98,
                    zorder=7,
                )
        else:
            color_values = np.asarray(
                [rank_success_map[str(node_key)] for node_key in stage_df["node_key"]],
                dtype=float,
            )
            ax.scatter(
                x_values,
                y_values,
                c=color_values,
                cmap="viridis",
                vmin=float(ranked_df["nom_success_rate"].min()),
                vmax=float(ranked_df["nom_success_rate"].max()),
                s=40,
                edgecolors="white",
                linewidths=0.5,
                alpha=0.96,
                zorder=5,
            )
            ax.scatter(
                [stage_x_map[stage]],
                [position_map[(stage, selected_trace_key)]],
                marker="*",
                s=180,
                c="#ffcc00",
                edgecolors="black",
                linewidths=1.0,
                zorder=8,
            )

    ax.annotate(
        "selected",
        xy=(
            stage_x_map["Retained robust rank"],
            position_map[("Retained robust rank", selected_trace_key)],
        ),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold",
    )

    top_rank_df = ranked_df.nsmallest(min(3, len(ranked_df)), "robust_rank")
    for _, row in top_rank_df.iterrows():
        candidate_id = str(int(row["candidate_id"]))
        if candidate_id == selected_trace_key:
            continue
        ax.annotate(
            f"#{int(row['robust_rank'])}",
            xy=(
                stage_x_map["Retained robust rank"],
                position_map[("Retained robust rank", candidate_id)],
            ),
            xytext=(6, -10),
            textcoords="offset points",
            fontsize=8,
        )

    for stage in STAGE_ORDER:
        stage_count = int((nodes_df["stage"] == stage).sum())
        ax.text(
            stage_x_map[stage],
            y_top,
            f"{stage}\n$n$ = {stage_count}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title("Selection trace")
    ax.set_xlim(-0.35, float(len(STAGE_ORDER) - 1) + 0.35)
    ax.set_ylim(-1.0, y_top + 1.2)
    ax.set_xticks([stage_x_map[stage] for stage in STAGE_ORDER], labels=STAGE_ORDER)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(False)


def draw_tradeoff_panel(
    ax: plt.Axes,
    tradeoff_df: pd.DataFrame,
    trace_nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    ranked_df: pd.DataFrame,
    tail_metric_col: str,
) -> None:
    legend_handles: list[Line2D] = []
    for stage in TRADEOFF_STAGE_ORDER:
        stage_df = tradeoff_df.loc[tradeoff_df["stage"] == stage].copy()
        if stage_df.empty:
            continue
        style = TRADEOFF_STAGE_STYLES[stage]
        ax.scatter(
            stage_df["objective_plot"],
            stage_df["tail_metric_plot"],
            s=float(style["size"]),
            c=str(style["color"]),
            marker=str(style["marker"]),
            edgecolors="white",
            linewidths=0.6,
            alpha=float(style["alpha"]),
            zorder=2,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=str(style["marker"]),
                color="w",
                label=f"{stage} (n={len(stage_df)})",
                markerfacecolor=str(style["color"]),
                markeredgecolor="white",
                markeredgewidth=0.6,
                markersize=max(5.5, np.sqrt(float(style["size"]))),
                alpha=float(style["alpha"]),
                linewidth=0,
            )
        )

    trace_position_map: dict[tuple[str, str], tuple[float, float]] = {}
    trace_plot_df = trace_nodes_df.loc[
        trace_nodes_df["objective_plot"].notna() & trace_nodes_df["tail_metric_plot"].notna()
    ].copy()
    for _, row in trace_plot_df.iterrows():
        trace_position_map[(str(row["stage"]), str(row["node_key"]))] = (
            float(row["objective_plot"]),
            float(row["tail_metric_plot"]),
        )

    plotted_trace_edges = 0
    for _, edge in edges_df.iterrows():
        source_key = (str(edge["source_stage"]), str(edge["source_key"]))
        target_key = (str(edge["target_stage"]), str(edge["target_key"]))
        if source_key not in trace_position_map or target_key not in trace_position_map:
            continue
        source_x, source_y = trace_position_map[source_key]
        target_x, target_y = trace_position_map[target_key]
        edge_style = EDGE_STYLE_MAP[str(edge["edge_class"])]
        _draw_curve(
            ax=ax,
            x0=source_x,
            y0=source_y,
            x1=target_x,
            y1=target_y,
            color=str(edge_style["color"]),
            linewidth=float(edge_style["linewidth"]),
            alpha=float(edge_style["alpha"]),
            zorder=int(edge_style["zorder"]) + 1,
        )
        plotted_trace_edges += 1

    for stage in STAGE_ORDER:
        stage_trace_df = trace_plot_df.loc[trace_plot_df["stage"] == stage].copy()
        if stage_trace_df.empty:
            continue
        marker = TRACE_STAGE_MARKERS.get(stage, "o")
        colors = [
            EDGE_STYLE_MAP.get(str(status), EDGE_STYLE_MAP["background_terminated"])["color"]
            for status in stage_trace_df["status_class"]
        ]
        sizes = [
            92.0 if str(status) == "selected" else 54.0
            for status in stage_trace_df["status_class"]
        ]
        ax.scatter(
            stage_trace_df["objective_plot"],
            stage_trace_df["tail_metric_plot"],
            s=sizes,
            c=colors,
            marker=marker,
            edgecolors="black",
            linewidths=0.55,
            alpha=0.9,
            zorder=5,
        )

    selected_row = ranked_df.loc[ranked_df["is_selected"]].iloc[0]
    selected_x = float(selected_row[_resolve_objective_column(ranked_df)])
    selected_y = float(selected_row[tail_metric_col])
    ax.scatter(
        [selected_x],
        [selected_y],
        marker="*",
        s=260,
        c="#ffcc00",
        edgecolors="black",
        linewidths=1.1,
        zorder=4,
    )
    ax.annotate(
        "selected",
        xy=(selected_x, selected_y),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
    )

    top_rank_df = ranked_df.nsmallest(min(3, len(ranked_df)), "robust_rank")
    for _, row in top_rank_df.iterrows():
        if bool(row["is_selected"]):
            continue
        ax.annotate(
            f"#{int(row['robust_rank'])}",
            xy=(row["objective"], row[tail_metric_col]),
            xytext=(6, -12),
            textcoords="offset points",
            fontsize=9,
        )

    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper right",
            fontsize=8,
            framealpha=0.9,
        )

    metric_label = (
        "Tail-risk sink mean (worst 20%) [m/s]"
        if tail_metric_col == "nom_sink_tail_mean_k"
        else "Sink CVaR20 [m/s]"
    )
    ax.set_title("Workflow trade-off space")
    ax.set_xlabel("Nominal objective")
    ax.set_ylabel(metric_label)
    ax.grid(True, alpha=0.2)

    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    ax.plot(
        [x_limits[0], selected_x],
        [selected_y, selected_y],
        color="#777777",
        linewidth=0.8,
        linestyle="--",
        alpha=0.35,
        zorder=1,
    )
    ax.plot(
        [selected_x, selected_x],
        [y_limits[0], selected_y],
        color="#777777",
        linewidth=0.8,
        linestyle="--",
        alpha=0.35,
        zorder=1,
    )
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)

    ax.text(
        0.03,
        0.97,
        "\n".join(
            [
                "ranking: success desc",
                "tail risk asc",
                "objective asc",
                f"retained candidates: {len(ranked_df)}",
                f"plotted points: {len(tradeoff_df)}",
                f"trace nodes: {len(trace_plot_df)}",
                f"trace edges: {plotted_trace_edges}",
            ]
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
    )


def draw_tradeoff_trace_3d_panel(
    ax: plt.Axes,
    tradeoff_df: pd.DataFrame,
    trace_nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    ranked_df: pd.DataFrame,
    tail_metric_col: str,
    display_transform: StageDisplayTransform | None,
) -> None:
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        axis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 1.0)
        axis._axinfo["grid"]["linewidth"] = 0.4

    ax.set_facecolor("white")
    ax.grid(True)

    stage_z_map = {stage: float(index) for index, stage in enumerate(STAGE_ORDER)}
    legend_handles: list[Line2D] = []
    for stage in TRADEOFF_STAGE_ORDER:
        stage_df = tradeoff_df.loc[tradeoff_df["stage"] == stage].copy()
        if stage_df.empty:
            continue
        style = TRADEOFF_STAGE_STYLES[stage]
        z_value = stage_z_map[TRADEOFF_TO_TRACE_STAGE_MAP[stage]]
        ax.scatter(
            stage_df["objective_display"],
            stage_df["tail_metric_display"],
            np.full(len(stage_df), z_value, dtype=float),
            s=float(style["size"]),
            c=str(style["color"]),
            marker=str(style["marker"]),
            edgecolors="white",
            linewidths=0.6,
            alpha=float(style["alpha"]),
            depthshade=False,
            zorder=2,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=str(style["marker"]),
                color="w",
                label=f"{stage} (n={len(stage_df)})",
                markerfacecolor=str(style["color"]),
                markeredgecolor="white",
                markeredgewidth=0.6,
                markersize=max(5.5, np.sqrt(float(style["size"]))),
                alpha=float(style["alpha"]),
                linewidth=0,
            )
        )

    trace_position_map: dict[tuple[str, str], tuple[float, float, float]] = {}
    trace_plot_df = trace_nodes_df.loc[
        trace_nodes_df["objective_plot"].notna() & trace_nodes_df["tail_metric_plot"].notna()
    ].copy()
    for _, row in trace_plot_df.iterrows():
        stage = str(row["stage"])
        trace_position_map[(stage, str(row["node_key"]))] = (
            float(row["objective_display"]),
            float(row["tail_metric_display"]),
            stage_z_map[stage],
        )

    edge_class_order = {
        "background_terminated": 0,
        "promoted": 1,
        "final_rerun": 2,
        "kept": 3,
        "selected": 4,
    }
    edges_to_draw = edges_df.copy()
    edges_to_draw["draw_class"] = np.where(
        edges_to_draw["is_selected_path"],
        "selected",
        edges_to_draw["edge_class"],
    )
    edges_to_draw["draw_order"] = edges_to_draw["draw_class"].map(edge_class_order)
    edges_to_draw = edges_to_draw.sort_values(
        by=["draw_order", "source_stage", "target_stage", "trace_group"],
        kind="mergesort",
    )

    for _, edge in edges_to_draw.iterrows():
        source_key = (str(edge["source_stage"]), str(edge["source_key"]))
        target_key = (str(edge["target_stage"]), str(edge["target_key"]))
        if source_key not in trace_position_map or target_key not in trace_position_map:
            continue
        source_x, source_y, source_z = trace_position_map[source_key]
        target_x, target_y, target_z = trace_position_map[target_key]
        edge_style = EDGE_STYLE_MAP[str(edge["draw_class"])]
        _draw_curve_3d(
            ax=ax,
            x0=source_x,
            y0=source_y,
            z0=source_z,
            x1=target_x,
            y1=target_y,
            z1=target_z,
            color=str(edge_style["color"]),
            linewidth=float(edge_style["linewidth"]),
            alpha=float(edge_style["alpha"]),
            zorder=int(edge_style["zorder"]) + 1,
        )

        if (
            source_key[0] == "Weight sweep"
            and target_key[0] == "Selected sweep rows"
        ):
            bridge_color = (
                EDGE_STYLE_MAP["selected"]["color"]
                if bool(edge["is_selected_path"])
                else "#9bb7d7"
            )
            bridge_linewidth = 2.0 if bool(edge["is_selected_path"]) else 1.15
            bridge_alpha = 0.96 if bool(edge["is_selected_path"]) else 0.68
            _draw_curve_3d(
                ax=ax,
                x0=source_x,
                y0=source_y,
                z0=source_z,
                x1=target_x,
                y1=target_y,
                z1=target_z,
                color=bridge_color,
                linewidth=bridge_linewidth,
                alpha=bridge_alpha,
                zorder=8 if bool(edge["is_selected_path"]) else 6,
            )

    for stage in STAGE_ORDER:
        stage_trace_df = trace_plot_df.loc[trace_plot_df["stage"] == stage].copy()
        if stage_trace_df.empty:
            continue
        marker = TRACE_STAGE_MARKERS.get(stage, "o")
        colors = [
            EDGE_STYLE_MAP.get(str(status), EDGE_STYLE_MAP["background_terminated"])["color"]
            for status in stage_trace_df["status_class"]
        ]
        sizes = [
            92.0 if str(status) == "selected" else 54.0
            for status in stage_trace_df["status_class"]
        ]
        ax.scatter(
            stage_trace_df["objective_display"],
            stage_trace_df["tail_metric_display"],
            np.full(len(stage_trace_df), stage_z_map[stage], dtype=float),
            s=sizes,
            c=colors,
            marker=marker,
            edgecolors="black",
            linewidths=0.55,
            alpha=0.9,
            depthshade=False,
            zorder=5,
        )

    selected_row = ranked_df.loc[ranked_df["is_selected"]].iloc[0]
    selected_trace_key = ("Retained robust rank", str(int(selected_row["candidate_id"])))
    selected_stage = stage_z_map["Retained robust rank"]
    selected_x, selected_y, _ = trace_position_map[selected_trace_key]
    ax.scatter(
        [selected_x],
        [selected_y],
        [selected_stage],
        marker="*",
        s=260,
        c="#ffcc00",
        edgecolors="black",
        linewidths=1.1,
        depthshade=False,
        zorder=6,
    )
    ax.text(
        selected_x,
        selected_y,
        selected_stage + 0.12,
        "selected",
        fontsize=10,
        fontweight="bold",
    )

    if display_transform is not None:
        _draw_trade_space_panel(
            ax=ax,
            x_min=display_transform.region_a_x_min,
            x_span=display_transform.region_a_x_span,
            y_min=display_transform.region_a_y_min,
            y_span=display_transform.region_a_y_span,
            z_level=stage_z_map["Weight sweep"] - 0.03,
            source_x_min=display_transform.weight_x_min,
            source_x_span=display_transform.weight_x_span,
            source_y_min=display_transform.weight_y_min,
            source_y_span=display_transform.weight_y_span,
            panel_label="Region A: weight sweep",
            frame_color="#7a7a7a",
        )
        _draw_trade_space_panel(
            ax=ax,
            x_min=display_transform.region_b_x_min,
            x_span=display_transform.region_b_x_span,
            y_min=display_transform.region_b_y_min,
            y_span=display_transform.region_b_y_span,
            z_level=stage_z_map["Selected sweep rows"] - 0.08,
            source_x_min=display_transform.later_x_min,
            source_x_span=display_transform.later_x_span,
            source_y_min=display_transform.later_y_min,
            source_y_span=display_transform.later_y_span,
            panel_label="Region B: refinement",
            frame_color="#5f5f5f",
        )

    ax.set_title("Workflow trade-off space and selection trace")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("Stage", labelpad=6)
    ax.set_zticks([stage_z_map[stage] for stage in STAGE_ORDER])
    ax.set_zticklabels([TRACE_STAGE_Z_LABELS[stage] for stage in STAGE_ORDER])
    ax.set_zlim(-0.2, float(len(STAGE_ORDER) - 1) + 0.35)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0, pad=2)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.tick_params(axis="z", labelsize=9, pad=12)

    if display_transform is not None:
        x_min = min(display_transform.region_a_x_min, display_transform.region_b_x_min) - 0.45
        x_max = max(
            display_transform.region_a_x_min + display_transform.region_a_x_span,
            display_transform.region_b_x_min + display_transform.region_b_x_span,
        ) + 0.45
        y_min = min(display_transform.region_a_y_min, display_transform.region_b_y_min) - 0.95
        y_max = max(
            display_transform.region_a_y_min + display_transform.region_a_y_span,
            display_transform.region_b_y_min + display_transform.region_b_y_span,
        ) + 0.40
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    if legend_handles:
        legend = ax.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(1.22, 0.98),
            fontsize=8,
            framealpha=0.95,
        )
        legend.get_frame().set_edgecolor("black")

    try:
        ax.set_box_aspect((1.65, 1.35, 1.20))
    except AttributeError:
        pass
    ax.view_init(elev=22, azim=-63)


def _set_3d_axes_style(ax: plt.Axes) -> None:
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        axis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 1.0)
        axis._axinfo["grid"]["linewidth"] = 0.4
    ax.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.grid(True)


def _set_flat_weight_plane_style(ax: plt.Axes) -> None:
    _set_3d_axes_style(ax)
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis._axinfo["grid"]["color"] = (1.0, 1.0, 1.0, 0.0)
    ax.zaxis._axinfo["grid"]["linewidth"] = 0.0
    ax.zaxis.line.set_color((0.0, 0.0, 0.0, 0.0))
    ax.zaxis.line.set_linewidth(0.0)


def _draw_flat_trade_plane(
    ax: plt.Axes,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    z_level: float,
) -> None:
    x_min, x_max = x_limits
    y_min, y_max = y_limits
    plane_color = "#8d8d8d"
    corners = [
        (x_min, y_min),
        (x_max, y_min),
        (x_max, y_max),
        (x_min, y_max),
        (x_min, y_min),
    ]
    for (x0, y0), (x1, y1) in zip(corners[:-1], corners[1:]):
        ax.plot(
            [x0, x1],
            [y0, y1],
            [z_level, z_level],
            color=plane_color,
            linewidth=0.85,
            alpha=0.70,
            zorder=1,
        )


def _compute_padded_limits(values: pd.Series, pad_fraction: float) -> tuple[float, float]:
    finite_values = pd.to_numeric(values, errors="coerce")
    finite_values = finite_values[np.isfinite(finite_values.to_numpy(dtype=float))]
    if finite_values.empty:
        return -1.0, 1.0

    value_min = float(finite_values.min())
    value_max = float(finite_values.max())
    span = value_max - value_min
    if span <= 1e-12:
        pad = max(0.05 * max(abs(value_min), 1.0), 0.05)
        return value_min - pad, value_max + pad
    pad = pad_fraction * span
    return value_min - pad, value_max + pad


def _build_refinement_stage_z_map() -> dict[str, float]:
    return {
        "Selected sweep rows": 0.0,
        "Top rerun starts": 1.0,
        "Final rerun starts": 2.0,
        "Retained robust rank": 3.0,
    }


def _project_3d_point_to_figure(
    fig: plt.Figure,
    ax: plt.Axes,
    x_value: float,
    y_value: float,
    z_value: float,
) -> tuple[float, float]:
    x_proj, y_proj, _ = proj3d.proj_transform(
        x_value,
        y_value,
        z_value,
        ax.get_proj(),
    )
    display_xy = ax.transData.transform((x_proj, y_proj))
    figure_xy = fig.transFigure.inverted().transform(display_xy)
    return float(figure_xy[0]), float(figure_xy[1])


def draw_weight_sweep_subplot(
    ax: plt.Axes,
    tradeoff_df: pd.DataFrame,
    trace_nodes_df: pd.DataFrame,
) -> tuple[dict[str, tuple[float, float, float]], list[Line2D]]:
    _set_flat_weight_plane_style(ax)

    weight_df = tradeoff_df.loc[tradeoff_df["stage"] == "Weight sweep"].copy()
    if weight_df.empty:
        ax.set_title("")
        return {}, []

    weight_trace_df = trace_nodes_df.loc[
        (trace_nodes_df["stage"] == "Weight sweep")
        & trace_nodes_df["objective_plot"].notna()
        & trace_nodes_df["tail_metric_plot"].notna()
    ].copy()
    weight_trace_df["node_key"] = weight_trace_df["node_key"].astype(str)
    promoted_mask = weight_trace_df["status_class"].isin(["promoted", "final_rerun"])
    selected_mask = weight_trace_df["status_class"].eq("final_rerun")

    z_level = 0.0
    z_cloud = np.full(len(weight_df), z_level, dtype=float)
    ax.scatter(
        weight_df["objective_plot"],
        weight_df["tail_metric_plot"],
        z_cloud,
        s=float(TRADEOFF_STAGE_STYLES["Weight sweep"]["size"]),
        c=TRADEOFF_STAGE_STYLES["Weight sweep"]["color"],
        marker=TRADEOFF_STAGE_STYLES["Weight sweep"]["marker"],
        edgecolors="white",
        linewidths=0.45,
        alpha=0.32,
        depthshade=False,
        zorder=2,
    )

    if promoted_mask.any():
        promoted_df = weight_trace_df.loc[promoted_mask].copy()
        ax.scatter(
            promoted_df["objective_plot"],
            promoted_df["tail_metric_plot"],
            np.full(len(promoted_df), z_level, dtype=float),
            s=74.0,
            c="#4e79a7",
            marker="o",
            edgecolors="white",
            linewidths=0.75,
            alpha=0.96,
            depthshade=False,
            zorder=5,
        )

    if selected_mask.any():
        selected_df = weight_trace_df.loc[selected_mask].copy()
        ax.scatter(
            selected_df["objective_plot"],
            selected_df["tail_metric_plot"],
            np.full(len(selected_df), z_level, dtype=float),
            s=132.0,
            c="#f28e2b",
            marker="*",
            edgecolors="black",
            linewidths=0.9,
            alpha=0.98,
            depthshade=False,
            zorder=7,
        )

    carry_over_positions = {
        str(row["node_key"]): (
            float(row["objective_plot"]),
            float(row["tail_metric_plot"]),
            z_level,
        )
        for _, row in weight_trace_df.loc[promoted_mask].iterrows()
    }

    x_limits = _compute_padded_limits(weight_df["objective_plot"], pad_fraction=0.10)
    y_limits = _compute_padded_limits(weight_df["tail_metric_plot"], pad_fraction=0.16)
    _draw_flat_trade_plane(
        ax=ax,
        x_limits=x_limits,
        y_limits=y_limits,
        z_level=z_level,
    )
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_zlim(-0.015, 0.085)
    ax.set_zticks([])
    ax.set_xlabel("Nominal objective", labelpad=10)
    ax.set_ylabel("Tail-risk mean (worst 20%)", labelpad=10)
    ax.set_zlabel("")
    ax.set_title("")
    ax.tick_params(axis="x", labelsize=8, pad=2)
    ax.tick_params(axis="y", labelsize=8, pad=2)
    ax.tick_params(axis="z", length=0, pad=0, labelsize=0)
    ax.text(
        x_limits[1] - 0.04 * (x_limits[1] - x_limits[0]),
        y_limits[0] + 0.06 * (y_limits[1] - y_limits[0]),
        z_level,
        "weight sweep",
        fontsize=8,
        color="#5f5f5f",
        ha="right",
        va="bottom",
    )
    try:
        ax.set_box_aspect((1.55, 1.08, 0.10))
    except AttributeError:
        pass
    ax.view_init(elev=23, azim=-58)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"Weight sweep cloud (n={len(weight_df)})",
            markerfacecolor=str(TRADEOFF_STAGE_STYLES["Weight sweep"]["color"]),
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=6.5,
            alpha=0.55,
            linewidth=0,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"Carry-over sweep rows (n={int(promoted_mask.sum())})",
            markerfacecolor="#4e79a7",
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=7.5,
            linewidth=0,
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            label="Selected handoff path",
            markerfacecolor="#f28e2b",
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=10.5,
            linewidth=0,
        ),
    ]
    return carry_over_positions, legend_handles


def draw_refinement_subplot(
    ax: plt.Axes,
    trace_nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
) -> tuple[dict[tuple[str, str], tuple[float, float, float]], list[Line2D]]:
    _set_3d_axes_style(ax)

    refinement_stage_order = [
        "Selected sweep rows",
        "Top rerun starts",
        "Final rerun starts",
        "Retained robust rank",
    ]
    stage_z_map = _build_refinement_stage_z_map()
    refinement_df = trace_nodes_df.loc[
        trace_nodes_df["stage"].isin(refinement_stage_order)
        & trace_nodes_df["objective_plot"].notna()
        & trace_nodes_df["tail_metric_plot"].notna()
    ].copy()
    refinement_df["node_key"] = refinement_df["node_key"].astype(str)

    position_map = {
        (str(row["stage"]), str(row["node_key"])): (
            float(row["objective_plot"]),
            float(row["tail_metric_plot"]),
            stage_z_map[str(row["stage"])],
        )
        for _, row in refinement_df.iterrows()
    }

    edge_class_order = {
        "promoted": 0,
        "final_rerun": 1,
        "kept": 2,
        "selected": 3,
    }
    refinement_edges_df = edges_df.loc[
        edges_df["source_stage"].isin(refinement_stage_order)
        & edges_df["target_stage"].isin(refinement_stage_order)
    ].copy()
    refinement_edges_df["draw_class"] = np.where(
        refinement_edges_df["is_selected_path"],
        "selected",
        refinement_edges_df["edge_class"],
    )
    refinement_edges_df["draw_order"] = refinement_edges_df["draw_class"].map(
        edge_class_order
    )
    refinement_edges_df = refinement_edges_df.sort_values(
        by=["draw_order", "source_stage", "target_stage", "trace_group"],
        kind="mergesort",
    )

    for _, edge in refinement_edges_df.iterrows():
        source_key = (str(edge["source_stage"]), str(edge["source_key"]))
        target_key = (str(edge["target_stage"]), str(edge["target_key"]))
        if source_key not in position_map or target_key not in position_map:
            continue
        source_x, source_y, source_z = position_map[source_key]
        target_x, target_y, target_z = position_map[target_key]
        edge_style = EDGE_STYLE_MAP[str(edge["draw_class"])]
        _draw_curve_3d(
            ax=ax,
            x0=source_x,
            y0=source_y,
            z0=source_z,
            x1=target_x,
            y1=target_y,
            z1=target_z,
            color=str(edge_style["color"]),
            linewidth=float(edge_style["linewidth"]),
            alpha=float(edge_style["alpha"]),
            zorder=int(edge_style["zorder"]) + 1,
        )

    stage_style_map = {
        "Selected sweep rows": {
            "marker": "s",
            "size": 64.0,
            "facecolor": "#4e79a7",
            "edgecolor": "white",
        },
        "Top rerun starts": {
            "marker": "X",
            "size": 58.0,
            "facecolor": "#76b7b2",
            "edgecolor": "white",
        },
        "Final rerun starts": {
            "marker": "D",
            "size": 56.0,
            "facecolor": "#edc948",
            "edgecolor": "white",
        },
        "Retained robust rank": {
            "marker": "h",
            "size": 72.0,
            "facecolor": "#af7aa1",
            "edgecolor": "white",
        },
    }

    selected_point_xyz: tuple[float, float, float] | None = None
    for stage in refinement_stage_order:
        stage_df = refinement_df.loc[refinement_df["stage"] == stage].copy()
        if stage_df.empty:
            continue
        stage_style = stage_style_map[stage]
        ax.scatter(
            stage_df["objective_plot"],
            stage_df["tail_metric_plot"],
            np.full(len(stage_df), stage_z_map[stage], dtype=float),
            s=float(stage_style["size"]),
            c=str(stage_style["facecolor"]),
            marker=str(stage_style["marker"]),
            edgecolors=str(stage_style["edgecolor"]),
            linewidths=0.7,
            alpha=0.92,
            depthshade=False,
            zorder=5,
        )
        if stage == "Retained robust rank":
            selected_df = stage_df.loc[stage_df["status_class"] == "selected"].copy()
            if not selected_df.empty:
                selected_row = selected_df.iloc[0]
                selected_point_xyz = (
                    float(selected_row["objective_plot"]),
                    float(selected_row["tail_metric_plot"]),
                    stage_z_map[stage],
                )
                ax.scatter(
                    [selected_point_xyz[0]],
                    [selected_point_xyz[1]],
                    [selected_point_xyz[2]],
                    marker="*",
                    s=240.0,
                    c="#ffcc00",
                    edgecolors="black",
                    linewidths=1.0,
                    depthshade=False,
                    zorder=7,
                )
                ax.text(
                    selected_point_xyz[0],
                    selected_point_xyz[1],
                    selected_point_xyz[2] + 0.10,
                    "selected",
                    fontsize=9,
                    fontweight="bold",
                )

    x_limits = _compute_padded_limits(refinement_df["objective_plot"], pad_fraction=0.18)
    y_limits = _compute_padded_limits(refinement_df["tail_metric_plot"], pad_fraction=0.22)
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_zlim(-0.2, float(len(refinement_stage_order) - 1) + 0.45)
    ax.set_zticks(
        [stage_z_map[stage] for stage in refinement_stage_order],
        labels=[
            "Selected\nsweep rows",
            "Top rerun\nstarts",
            "Final rerun\nstarts",
            "Retained\nrobust rank",
        ],
    )
    ax.set_xlabel("Nominal objective", labelpad=10)
    ax.set_ylabel("Tail-risk mean (worst 20%)", labelpad=10)
    ax.set_zlabel("Stage", labelpad=6)
    ax.set_title("Refinement workflow", pad=1)
    ax.tick_params(axis="x", labelsize=8, pad=2)
    ax.tick_params(axis="y", labelsize=8, pad=2)
    ax.tick_params(axis="z", labelsize=8, pad=8)
    try:
        ax.set_box_aspect((1.20, 1.04, 1.28))
    except AttributeError:
        pass
    ax.view_init(elev=23, azim=-58)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label=f"Selected sweep rows (n={int((refinement_df['stage'] == 'Selected sweep rows').sum())})",
            markerfacecolor="#4e79a7",
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=7.2,
            linewidth=0,
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            color="w",
            label=f"Top rerun starts (n={int((refinement_df['stage'] == 'Top rerun starts').sum())})",
            markerfacecolor="#76b7b2",
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=7.0,
            linewidth=0,
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            label=f"Final rerun starts (n={int((refinement_df['stage'] == 'Final rerun starts').sum())})",
            markerfacecolor="#edc948",
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=6.8,
            linewidth=0,
        ),
        Line2D(
            [0],
            [0],
            marker="h",
            color="w",
            label=f"Retained robust rank (n={int((refinement_df['stage'] == 'Retained robust rank').sum())})",
            markerfacecolor="#af7aa1",
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=7.6,
            linewidth=0,
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            label="Selected final design",
            markerfacecolor="#ffcc00",
            markeredgecolor="black",
            markeredgewidth=0.9,
            markersize=10.0,
            linewidth=0,
        ),
    ]
    return position_map, legend_handles


def draw_cross_panel_handoff_connectors(
    fig: plt.Figure,
    left_ax: plt.Axes,
    right_ax: plt.Axes,
    edges_df: pd.DataFrame,
    left_position_map: dict[str, tuple[float, float, float]],
    right_position_map: dict[tuple[str, str], tuple[float, float, float]],
) -> None:
    fig.canvas.draw()
    handoff_edges_df = edges_df.loc[
        (edges_df["source_stage"] == "Weight sweep")
        & (edges_df["target_stage"] == "Selected sweep rows")
    ].copy()
    handoff_edges_df["draw_class"] = np.where(
        handoff_edges_df["is_selected_path"],
        "selected",
        handoff_edges_df["edge_class"],
    )

    for _, edge in handoff_edges_df.iterrows():
        source_key = str(edge["source_key"])
        target_key = ("Selected sweep rows", str(edge["target_key"]))
        if source_key not in left_position_map or target_key not in right_position_map:
            continue

        left_x, left_y, left_z = left_position_map[source_key]
        right_x, right_y, right_z = right_position_map[target_key]
        x0_fig, y0_fig = _project_3d_point_to_figure(
            fig,
            left_ax,
            left_x,
            left_y,
            left_z,
        )
        x1_fig, y1_fig = _project_3d_point_to_figure(
            fig,
            right_ax,
            right_x,
            right_y,
            right_z,
        )

        draw_class = str(edge["draw_class"])
        edge_style = EDGE_STYLE_MAP[draw_class]
        dx = x1_fig - x0_fig
        dy = y1_fig - y0_fig
        control_1 = (
            x0_fig + 0.26 * dx,
            y0_fig + max(0.05, 0.48 * dy),
        )
        control_2 = (
            x0_fig + 0.76 * dx,
            y0_fig + max(0.08, 1.02 * dy),
        )
        path = MplPath(
            [
                (x0_fig, y0_fig),
                control_1,
                control_2,
                (x1_fig, y1_fig),
            ],
            [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
        )
        patch = PathPatch(
            path,
            transform=fig.transFigure,
            facecolor="none",
            edgecolor=str(edge_style["color"]),
            linewidth=float(edge_style["linewidth"])
            + (0.85 if draw_class == "selected" else 0.20),
            alpha=0.96 if draw_class == "selected" else 0.52,
            zorder=20 if draw_class == "selected" else 12,
            capstyle="round",
            joinstyle="round",
        )
        fig.add_artist(patch)


def make_candidate_selection_plot(data: CandidateSelectionData) -> Path:
    nodes_df, edges_df, ranked_df, tail_metric_col, _selected_candidate_id = (
        build_full_provenance(data)
    )
    tradeoff_df = build_tradeoff_points(
        data=data,
        ranked_df=ranked_df,
        tail_metric_col=tail_metric_col,
    )
    trace_nodes_df = build_tradeoff_trace_nodes(
        data=data,
        nodes_df=nodes_df,
        ranked_df=ranked_df,
        tail_metric_col=tail_metric_col,
    )

    fig = plt.figure(figsize=(12.8, 7.4))
    fig.patch.set_facecolor("white")
    left_ax = fig.add_axes([0.09, 0.07, 0.52, 0.40], projection="3d")
    right_ax = fig.add_axes([0.50, 0.22, 0.41, 0.56], projection="3d")

    left_position_map, left_legend_handles = draw_weight_sweep_subplot(
        ax=left_ax,
        tradeoff_df=tradeoff_df,
        trace_nodes_df=trace_nodes_df,
    )
    right_position_map, right_legend_handles = draw_refinement_subplot(
        ax=right_ax,
        trace_nodes_df=trace_nodes_df,
        edges_df=edges_df,
    )
    draw_cross_panel_handoff_connectors(
        fig=fig,
        left_ax=left_ax,
        right_ax=right_ax,
        edges_df=edges_df,
        left_position_map=left_position_map,
        right_position_map=right_position_map,
    )

    legend = fig.legend(
        handles=left_legend_handles + right_legend_handles,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.50, 0.95),
        fontsize=8,
        framealpha=0.95,
    )
    legend.get_frame().set_edgecolor("black")

    fig.savefig(FIGURE_PATH, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return FIGURE_PATH


def main() -> None:
    data = load_candidate_selection_data()
    figure_path = make_candidate_selection_plot(data)
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
