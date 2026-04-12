from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cmocean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from F_analysis.analysis_common import (
    open_canonical_workbook,
    read_sheet_optional,
    read_sheet_required,
    resolve_selected_candidate_id,
    resolve_tail_metric_name,
    sort_scenario_tags,
)

RESULTS_DIR = PROJECT_ROOT / "C_results"
FIGURES_DIR = PROJECT_ROOT / "B_figures"
FIGURE_PATH = FIGURES_DIR / "robustness_map.png"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

METRIC_ORDER = [
    "nom_sink_rate_mps",
    "nom_alpha_margin_deg",
    "nom_cl_margin_to_cap",
    "nom_util_e",
    "nom_roll_tau_s",
    "nom_lateral_residual",
]
LOWER_IS_BETTER = {
    "nom_sink_rate_mps",
    "nom_util_e",
    "nom_roll_tau_s",
    "nom_lateral_residual",
}
METRIC_LABEL_MAP = {
    "nom_sink_rate_mps": "sink rate",
    "nom_alpha_margin_deg": "alpha margin",
    "nom_cl_margin_to_cap": "CL margin",
    "nom_util_e": "elevator util.",
    "nom_roll_tau_s": "roll tau",
    "nom_lateral_residual": "lat. residual",
}
TAG_BASE_COLORS = [
    "#7f3c8d",
    "#11a579",
    "#3969ac",
    "#f2b701",
    "#e73f74",
    "#80ba5a",
]
AXIS_EDGE_LW = 0.80
CBAR_EDGE_LW = AXIS_EDGE_LW
CELL_EDGE_LW = 0.30
LEGEND_FONT_SIZE = 8.4


def _aggregate_tag_summary(selected_scenarios_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for scenario_tag, group_df in selected_scenarios_df.groupby("scenario_tag", sort=False):
        success = (
            group_df["nom_success"].astype(float)
            if "nom_success" in group_df.columns
            else pd.Series(dtype=float)
        )
        sink = (
            group_df["nom_sink_rate_mps"].astype(float)
            if "nom_sink_rate_mps" in group_df.columns
            else pd.Series(dtype=float)
        )
        sink_sorted = np.sort(sink.to_numpy()) if not sink.empty else np.array([])
        tail_count = max(1, int(np.ceil(0.2 * len(sink_sorted)))) if sink_sorted.size else 0
        tail_value = (
            float(np.mean(sink_sorted[-tail_count:])) if tail_count > 0 else np.nan
        )
        rows.append(
            {
                "scenario_tag": str(scenario_tag),
                "success_rate": float(success.mean()) if not success.empty else np.nan,
                "nom_sink_tail_mean_k": tail_value,
            }
        )

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        return summary_df

    tag_order = sort_scenario_tags(summary_df["scenario_tag"])
    summary_df["tag_sort"] = summary_df["scenario_tag"].map(
        {tag: idx for idx, tag in enumerate(tag_order)}
    ).fillna(len(tag_order))
    summary_df = summary_df.sort_values(
        by=["tag_sort", "scenario_tag"],
        kind="mergesort",
    ).drop(columns="tag_sort")
    return summary_df.reset_index(drop=True)


def load_robustness_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _, book = open_canonical_workbook()
    try:
        candidates_df = read_sheet_required(book, "Candidates")
        robust_summary_df = read_sheet_required(book, "RobustSummary")
        scenarios_df = read_sheet_optional(book, "RobustScenarios")
        if scenarios_df.empty:
            scenarios_df = read_sheet_required(book, "PlotDataRobust")
        summary_by_tag_df = read_sheet_optional(book, "RobustSummaryByTag")
    finally:
        book.close()

    return candidates_df, robust_summary_df, scenarios_df, summary_by_tag_df


def get_selected_candidate_id(
    candidates_df: pd.DataFrame,
    robust_summary_df: pd.DataFrame,
) -> int:
    return resolve_selected_candidate_id(candidates_df, robust_summary_df)


def sort_selected_scenarios(selected_scenarios_df: pd.DataFrame) -> pd.DataFrame:
    sorted_df = selected_scenarios_df.copy()
    tag_order = sort_scenario_tags(sorted_df["scenario_tag"])
    sorted_df["tag_sort"] = sorted_df["scenario_tag"].map(
        {tag: idx for idx, tag in enumerate(tag_order)}
    ).fillna(len(tag_order))
    sort_columns = ["tag_sort"]
    ascending = [True]
    if "scenario_id" in sorted_df.columns:
        sort_columns.append("scenario_id")
        ascending.append(True)
    elif "selection_id" in sorted_df.columns:
        sort_columns.append("selection_id")
        ascending.append(True)
    sorted_df = sorted_df.sort_values(
        by=sort_columns,
        ascending=ascending,
        kind="mergesort",
    )
    return sorted_df.reset_index(drop=True)


def build_metric_matrix(selected_scenarios_df: pd.DataFrame) -> pd.DataFrame:
    available_metrics = [
        metric for metric in METRIC_ORDER if metric in selected_scenarios_df.columns
    ]
    metric_df = selected_scenarios_df[available_metrics].transpose()
    metric_df.columns = selected_scenarios_df["scenario_id"].astype(int).astype(str)
    metric_df.index = available_metrics
    return metric_df


def compute_rowwise_risk_map(metric_matrix: pd.DataFrame) -> np.ndarray:
    risk_rows: list[np.ndarray] = []
    for metric_name, row in metric_matrix.iterrows():
        values = row.to_numpy(dtype=float)
        finite_mask = np.isfinite(values)
        if not finite_mask.any():
            risk_rows.append(np.zeros_like(values))
            continue

        transformed = values.copy()
        if metric_name not in LOWER_IS_BETTER:
            finite_values = transformed[finite_mask]
            transformed[finite_mask] = np.nanmax(finite_values) - finite_values

        finite_values = transformed[finite_mask]
        min_value = np.nanmin(finite_values)
        max_value = np.nanmax(finite_values)
        if np.isclose(max_value, min_value):
            scaled = np.zeros_like(values)
        else:
            scaled = (transformed - min_value) / (max_value - min_value)
        scaled[~finite_mask] = np.nan
        risk_rows.append(scaled)

    return np.vstack(risk_rows) if risk_rows else np.zeros((0, 0))


def _centers_to_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    edges = np.empty(values.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * (values[1] - values[0]) if values.size > 1 else values[0] - 0.5
    edges[-1] = (
        values[-1] + 0.5 * (values[-1] - values[-2]) if values.size > 1 else values[-1] + 0.5
    )
    return edges


def _prepare_summary_by_tag(
    summary_by_tag_df: pd.DataFrame,
    selected_candidate_id: int,
    selected_scenarios_df: pd.DataFrame,
) -> pd.DataFrame:
    if summary_by_tag_df.empty:
        return _aggregate_tag_summary(selected_scenarios_df)

    summary_df = summary_by_tag_df.copy()
    if "candidate_id" in summary_df.columns:
        summary_df = summary_df.loc[
            summary_df["candidate_id"] == selected_candidate_id
        ].copy()

    if summary_df.empty or "scenario_tag" not in summary_df.columns:
        return _aggregate_tag_summary(selected_scenarios_df)

    if "sink_tail_mean_k" in summary_df.columns:
        summary_df = summary_df.rename(
            columns={"sink_tail_mean_k": "nom_sink_tail_mean_k"}
        )

    tag_order = sort_scenario_tags(summary_df["scenario_tag"])
    summary_df["tag_sort"] = summary_df["scenario_tag"].map(
        {tag: idx for idx, tag in enumerate(tag_order)}
    ).fillna(len(tag_order))
    summary_df = summary_df.sort_values(
        by=["tag_sort", "scenario_tag"],
        kind="mergesort",
    ).drop(columns="tag_sort")
    return summary_df.reset_index(drop=True)


def _tag_color_map(tags: list[str]) -> dict[str, str]:
    return {
        tag: TAG_BASE_COLORS[index % len(TAG_BASE_COLORS)]
        for index, tag in enumerate(tags)
    }


def _style_axes(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_EDGE_LW)
        spine.set_edgecolor("black")
    ax.tick_params(axis="both", which="major", length=2.0, width=0.6)


def _format_tag_label(tag: str) -> str:
    return tag.replace("_", "\n")


def _scenario_family_positions(
    selected_scenarios_df: pd.DataFrame,
) -> tuple[np.ndarray, list[str], list[float]]:
    tag_series = selected_scenarios_df["scenario_tag"].astype(str).reset_index(drop=True)
    centers: list[float] = []
    labels: list[str] = []
    boundaries: list[float] = []

    start_index = 0
    while start_index < len(tag_series):
        current_tag = tag_series.iloc[start_index]
        end_index = start_index
        while end_index < len(tag_series) and tag_series.iloc[end_index] == current_tag:
            end_index += 1
        centers.append(0.5 * (start_index + end_index - 1))
        labels.append(_format_tag_label(current_tag))
        if end_index < len(tag_series):
            boundaries.append(float(end_index) - 0.5)
        start_index = end_index

    return np.asarray(centers, dtype=float), labels, boundaries


def make_robustness_map(
    selected_candidate_id: int,
    selected_scenarios_df: pd.DataFrame,
    summary_by_tag_df: pd.DataFrame,
) -> Path:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.edgecolor": "black",
            "axes.linewidth": AXIS_EDGE_LW,
            "patch.edgecolor": "black",
        }
    )

    metric_matrix = build_metric_matrix(selected_scenarios_df)
    risk_map = compute_rowwise_risk_map(metric_matrix)
    summary_df = _prepare_summary_by_tag(
        summary_by_tag_df=summary_by_tag_df,
        selected_candidate_id=selected_candidate_id,
        selected_scenarios_df=selected_scenarios_df,
    )

    fig = plt.figure(figsize=(9.4, 4.6), dpi=600)
    grid = fig.add_gridspec(1, 2, width_ratios=[2.9, 1.2], wspace=0.42)
    heatmap_ax = fig.add_subplot(grid[0, 0])
    summary_ax = fig.add_subplot(grid[0, 1])

    x_centers = np.arange(len(metric_matrix.columns), dtype=float)
    y_centers = np.arange(len(metric_matrix.index), dtype=float)
    x_edges = _centers_to_edges(x_centers)
    y_edges = _centers_to_edges(y_centers)
    family_tick_positions, family_tick_labels, family_boundaries = _scenario_family_positions(
        selected_scenarios_df
    )

    image = heatmap_ax.pcolormesh(
        x_edges,
        y_edges,
        risk_map,
        shading="auto",
        cmap=cmocean.cm.matter,
        vmin=0.0,
        vmax=1.0,
        edgecolors=(0.0, 0.0, 0.0, 0.22),
        linewidth=CELL_EDGE_LW,
    )

    heatmap_ax.set_title(f"Robustness heat map: candidate {selected_candidate_id}")
    heatmap_ax.set_xlabel("Scenario family ordering")
    heatmap_ax.set_ylabel("Metric")
    heatmap_ax.set_yticks(
        y_centers,
        labels=[METRIC_LABEL_MAP.get(name, name) for name in metric_matrix.index],
    )
    heatmap_ax.set_xticks(family_tick_positions, labels=family_tick_labels)
    heatmap_ax.set_xlim(x_edges[0], x_edges[-1])
    heatmap_ax.set_ylim(y_edges[0], y_edges[-1])
    heatmap_ax.invert_yaxis()
    for boundary in family_boundaries:
        heatmap_ax.axvline(
            boundary,
            color=(0.0, 0.0, 0.0, 0.55),
            linewidth=0.65,
            linestyle="--",
            zorder=4,
        )
    _style_axes(heatmap_ax)

    divider = make_axes_locatable(heatmap_ax)
    cax = divider.append_axes("right", size="2.95%", pad=0.18)
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label("Relative risk", fontsize=9)
    cbar.set_ticks(np.linspace(0.0, 1.0, 6))
    cbar.formatter = FormatStrFormatter("%.2f")
    cbar.update_ticks()
    cbar.ax.tick_params(width=0.6, length=2.0, labelsize=9)
    cbar.outline.set_linewidth(CBAR_EDGE_LW)
    cbar.outline.set_edgecolor("black")
    for spine in cbar.ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(CBAR_EDGE_LW)

    selection_id_ax = heatmap_ax.secondary_xaxis("top")
    selection_id_ax.set_xticks(x_centers, labels=metric_matrix.columns.tolist())
    selection_id_ax.set_xlabel("Selection ID")
    selection_id_ax.tick_params(
        axis="x",
        which="major",
        length=2.0,
        width=0.6,
        labelsize=6,
        pad=1.5,
    )
    selection_id_ax.spines["top"].set_linewidth(AXIS_EDGE_LW)
    selection_id_ax.spines["top"].set_edgecolor("black")
    selection_id_ax.spines["bottom"].set_visible(False)
    selection_id_ax.spines["left"].set_visible(False)
    selection_id_ax.spines["right"].set_visible(False)

    if summary_df.empty:
        summary_ax.text(
            0.5,
            0.5,
            "No scenario-family summary available",
            ha="center",
            va="center",
        )
        summary_ax.set_axis_off()
    else:
        tag_metric_col = resolve_tail_metric_name(summary_df)
        tail_label = (
            "Tail-risk sink"
            if tag_metric_col == "nom_sink_tail_mean_k"
            else "Tail-risk sink (CVaR 20%)"
        )
        tag_labels = summary_df["scenario_tag"].astype(str).tolist()
        x = np.arange(len(tag_labels), dtype=float)
        color_map = _tag_color_map(tag_labels)
        bar_colors = [color_map[tag] for tag in tag_labels]

        summary_ax.bar(
            x,
            pd.to_numeric(summary_df[tag_metric_col], errors="coerce"),
            color=bar_colors,
            edgecolor="black",
            linewidth=0.45,
            width=0.72,
        )
        summary_ax.set_title("Scenario-family summary")
        summary_ax.set_ylabel(f"{tail_label} [m/s]")
        summary_ax.set_xticks(x, labels=[_format_tag_label(tag) for tag in tag_labels])
        summary_ax.tick_params(axis="x", rotation=25)
        summary_ax.grid(True, axis="y", alpha=0.20, linewidth=0.45)
        _style_axes(summary_ax)

        success_ax = summary_ax.twinx()
        success_ax.plot(
            x,
            pd.to_numeric(summary_df["success_rate"], errors="coerce"),
            color="#1f1f1f",
            linewidth=1.2,
            marker="o",
            markersize=3.8,
            zorder=4,
        )
        success_ax.set_ylabel("Success rate")
        success_ax.set_ylim(0.0, 1.05)
        success_ax.tick_params(axis="y", which="major", length=2.0, width=0.6, labelsize=9)
        for spine in success_ax.spines.values():
            spine.set_linewidth(AXIS_EDGE_LW)
            spine.set_edgecolor("black")

        legend_handles = [
            Line2D(
                [0],
                [0],
                color="#808080",
                linewidth=0.0,
                marker="s",
                markerfacecolor="#808080",
                markeredgecolor="black",
                markersize=7,
                label="Tail-risk sink",
            ),
            Line2D(
                [0],
                [0],
                color="#1f1f1f",
                linewidth=1.2,
                marker="o",
                markersize=4,
                label="Success rate",
            ),
        ]
        summary_ax.legend(
            handles=legend_handles,
            loc="upper left",
            frameon=True,
            framealpha=1.0,
            edgecolor="black",
            fontsize=LEGEND_FONT_SIZE,
            handlelength=1.5,
            borderpad=0.5,
            labelspacing=0.2,
        )

    fig.savefig(
        FIGURE_PATH,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
    )
    plt.close(fig)
    return FIGURE_PATH


def main() -> None:
    candidates_df, robust_summary_df, scenarios_df, summary_by_tag_df = load_robustness_data()
    selected_candidate_id = get_selected_candidate_id(candidates_df, robust_summary_df)
    selected_scenarios_df = scenarios_df.loc[
        scenarios_df["candidate_id"] == selected_candidate_id
    ].copy()
    selected_scenarios_df = sort_selected_scenarios(selected_scenarios_df)

    figure_path = make_robustness_map(
        selected_candidate_id=selected_candidate_id,
        selected_scenarios_df=selected_scenarios_df,
        summary_by_tag_df=summary_by_tag_df,
    )
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
