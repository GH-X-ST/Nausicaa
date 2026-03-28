from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "C_results"
FIGURES_DIR = PROJECT_ROOT / "B_figures"
WORKFLOW_XLSX = RESULTS_DIR / "nausicaa_workflow.xlsx"
RESULTS_XLSX = RESULTS_DIR / "nausicaa_results.xlsx"
FIGURE_PATH = FIGURES_DIR / "robustness_map.png"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TAG_ORDER = [
    "harsh_compound",
    "harsh_build",
    "gusty_only",
    "mild_compound",
    "mild_build",
]
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


def _resolve_tail_metric_name(df: pd.DataFrame) -> str:
    if "nom_sink_tail_mean_k" in df.columns:
        return "nom_sink_tail_mean_k"
    if "nom_sink_cvar_20" in df.columns:
        return "nom_sink_cvar_20"
    raise KeyError("Neither robust tail-risk metric is available.")


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
        resid_success_only = np.nan
        if "nom_lateral_residual" in group_df.columns and "nom_success" in group_df.columns:
            success_mask = group_df["nom_success"].astype(bool)
            resid = group_df.loc[success_mask, "nom_lateral_residual"].astype(float)
            if not resid.empty:
                resid_success_only = float(
                    np.sqrt(np.mean(np.square(resid.to_numpy())))
                )

        rows.append(
            {
                "scenario_tag": scenario_tag,
                "success_rate": float(success.mean()) if not success.empty else np.nan,
                "sink_tail_mean_k": tail_value,
                "nom_resid_rmse_success_only": resid_success_only,
            }
        )

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        return summary_df

    summary_df["tag_sort"] = summary_df["scenario_tag"].map(
        {tag: idx for idx, tag in enumerate(TAG_ORDER)}
    ).fillna(len(TAG_ORDER))
    summary_df = summary_df.sort_values(
        by=["tag_sort", "scenario_tag"],
        kind="mergesort",
    ).drop(columns="tag_sort")
    return summary_df.reset_index(drop=True)


def load_robustness_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    workflow_book = _open_workbook(WORKFLOW_XLSX)
    results_book = _open_workbook(RESULTS_XLSX)
    candidates_df = _read_sheet("Candidates", workflow_book, results_book)
    robust_summary_df = _read_sheet("RobustSummary", workflow_book, results_book)
    scenarios_df = _read_sheet("RobustScenarios", workflow_book, results_book)
    if scenarios_df is None:
        scenarios_df = _read_sheet("PlotDataRobust", workflow_book, results_book)
    summary_by_tag_df = _read_sheet("RobustSummaryByTag", workflow_book, results_book)

    if candidates_df is None or robust_summary_df is None or scenarios_df is None:
        raise FileNotFoundError(
            "Required workbook sheets were not found. Expected canonical "
            "workflow outputs with 'Candidates', 'RobustSummary', and "
            "'RobustScenarios' or 'PlotDataRobust'."
        )

    if summary_by_tag_df is None:
        summary_by_tag_df = pd.DataFrame()

    return candidates_df, robust_summary_df, scenarios_df, summary_by_tag_df


def get_selected_candidate_id(
    candidates_df: pd.DataFrame,
    robust_summary_df: pd.DataFrame,
) -> int:
    if "is_selected" in robust_summary_df.columns:
        selected_mask = _coerce_bool_series(robust_summary_df["is_selected"])
        if selected_mask.any():
            return int(robust_summary_df.loc[selected_mask, "candidate_id"].iloc[0])

    tail_metric_col = _resolve_tail_metric_name(robust_summary_df)
    merged_df = robust_summary_df.merge(
        candidates_df[["candidate_id", "objective"]],
        on="candidate_id",
        how="left",
    )
    ranked_df = merged_df.sort_values(
        by=["nom_success_rate", tail_metric_col, "objective"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    if not ranked_df.empty:
        return int(ranked_df.iloc[0]["candidate_id"])

    return int(candidates_df.sort_values("objective", kind="mergesort").iloc[0]["candidate_id"])


def sort_selected_scenarios(selected_scenarios_df: pd.DataFrame) -> pd.DataFrame:
    sorted_df = selected_scenarios_df.copy()
    sorted_df["tag_sort"] = sorted_df["scenario_tag"].map(
        {tag: idx for idx, tag in enumerate(TAG_ORDER)}
    ).fillna(len(TAG_ORDER))
    sort_columns = ["tag_sort"]
    ascending = [True]
    if "q" in sorted_df.columns:
        sort_columns.append("q")
        ascending.append(False)
    sort_columns.append("scenario_id")
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


def make_robustness_map(
    selected_candidate_id: int,
    selected_scenarios_df: pd.DataFrame,
    summary_by_tag_df: pd.DataFrame,
) -> Path:
    metric_matrix = build_metric_matrix(selected_scenarios_df)
    risk_map = compute_rowwise_risk_map(metric_matrix)

    fig = plt.figure(figsize=(11, 6.5), constrained_layout=True)
    outer = fig.add_gridspec(1, 2, width_ratios=[2.6, 1.1])
    left = outer[0].subgridspec(2, 1, height_ratios=[0.22, 1.0], hspace=0.05)
    right = outer[1].subgridspec(2, 1, hspace=0.25)

    strip_ax = fig.add_subplot(left[0])
    heatmap_ax = fig.add_subplot(left[1])
    success_ax = fig.add_subplot(right[0])
    tail_ax = fig.add_subplot(right[1])

    unique_tags = list(dict.fromkeys(selected_scenarios_df["scenario_tag"].astype(str)))
    base_colors = ["#7f3c8d", "#11a579", "#3969ac", "#f2b701", "#e73f74", "#80ba5a"]
    tag_color_map = {
        tag: base_colors[idx % len(base_colors)] for idx, tag in enumerate(unique_tags)
    }
    strip_rgb = np.zeros((2, len(selected_scenarios_df), 3))
    success_colors = np.where(
        selected_scenarios_df["nom_success"].astype(bool).to_numpy(),
        "#1a9641",
        "#d7191c",
    )
    strip_rgb[0] = np.array([to_rgb(color) for color in success_colors])
    strip_rgb[1] = np.array(
        [to_rgb(tag_color_map[tag]) for tag in selected_scenarios_df["scenario_tag"].astype(str)]
    )
    strip_ax.imshow(strip_rgb, aspect="auto")
    strip_ax.set_yticks([0, 1], labels=["ok", "tag"])
    strip_ax.set_xticks([])
    strip_ax.set_title(
        f"Selected candidate {selected_candidate_id}: row-wise normalized risk intensity",
        fontsize=11,
    )

    image = heatmap_ax.imshow(risk_map, aspect="auto", cmap="Reds", vmin=0.0, vmax=1.0)
    label_map = {
        "nom_sink_rate_mps": "sink rate",
        "nom_alpha_margin_deg": "alpha margin",
        "nom_cl_margin_to_cap": "CL margin",
        "nom_util_e": "elevator util.",
        "nom_roll_tau_s": "roll tau",
        "nom_lateral_residual": "lat. residual",
    }
    heatmap_ax.set_yticks(
        np.arange(len(metric_matrix.index)),
        labels=[label_map.get(name, name) for name in metric_matrix.index],
    )
    step = max(1, len(metric_matrix.columns) // 12)
    tick_positions = np.arange(0, len(metric_matrix.columns), step)
    heatmap_ax.set_xticks(tick_positions, labels=metric_matrix.columns[tick_positions])
    heatmap_ax.set_xlabel("Scenario ID")
    heatmap_ax.set_ylabel("Metric")
    fig.colorbar(image, ax=heatmap_ax, fraction=0.046, pad=0.02, label="Relative risk")

    if summary_by_tag_df.empty:
        success_ax.text(
            0.5,
            0.5,
            "No scenario-family summary available",
            ha="center",
            va="center",
        )
        success_ax.axis("off")
        tail_ax.axis("off")
    else:
        summary_df = summary_by_tag_df.copy()
        if "candidate_id" in summary_df.columns:
            summary_df = summary_df.loc[
                summary_df["candidate_id"] == selected_candidate_id
            ].copy()
        if "scenario_tag" not in summary_df.columns:
            summary_df = pd.DataFrame()

        if summary_df.empty:
            success_ax.text(
                0.5,
                0.5,
                "No selected-candidate tag summary available",
                ha="center",
                va="center",
            )
            success_ax.axis("off")
            tail_ax.axis("off")
        else:
            summary_df["tag_sort"] = summary_df["scenario_tag"].map(
                {tag: idx for idx, tag in enumerate(TAG_ORDER)}
            ).fillna(len(TAG_ORDER))
            summary_df = summary_df.sort_values(
                by=["tag_sort", "scenario_tag"],
                kind="mergesort",
            ).drop(columns="tag_sort")
            tail_metric_name = (
                "sink_tail_mean_k"
                if "sink_tail_mean_k" in summary_df.columns
                else "nom_sink_tail_mean_k"
                if "nom_sink_tail_mean_k" in summary_df.columns
                else "nom_sink_cvar_20"
            )
            x = np.arange(len(summary_df))
            colors = [tag_color_map[tag] for tag in summary_df["scenario_tag"].astype(str)]

            success_ax.bar(
                x,
                summary_df["success_rate"],
                color=colors,
                edgecolor="black",
                linewidth=0.4,
            )
            success_ax.set_ylabel("Success rate")
            success_ax.set_ylim(0.0, 1.05)
            success_ax.set_xticks(
                x,
                labels=summary_df["scenario_tag"],
                rotation=25,
                ha="right",
            )
            success_ax.set_title("Scenario-family summary")
            success_ax.grid(True, axis="y", alpha=0.25)

            tail_ax.bar(
                x,
                summary_df[tail_metric_name],
                color=colors,
                edgecolor="black",
                linewidth=0.4,
            )
            tail_ax.set_ylabel("Tail-risk sink [m/s]")
            tail_ax.set_xticks(
                x,
                labels=summary_df["scenario_tag"],
                rotation=25,
                ha="right",
            )
            tail_ax.grid(True, axis="y", alpha=0.25)

    legend_handles = [
        Patch(facecolor=tag_color_map[tag], edgecolor="black", label=tag)
        for tag in unique_tags
    ]
    if legend_handles:
        fig.legend(handles=legend_handles, loc="upper center", ncol=min(4, len(legend_handles)))

    fig.savefig(FIGURE_PATH, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return FIGURE_PATH


def main() -> None:
    candidates_df, robust_summary_df, scenarios_df, summary_by_tag_df = load_robustness_data()
    selected_candidate_id = get_selected_candidate_id(candidates_df, robust_summary_df)
    selected_scenarios_df = scenarios_df.loc[
        scenarios_df["candidate_id"] == selected_candidate_id
    ].copy()
    selected_scenarios_df = sort_selected_scenarios(selected_scenarios_df)

    if summary_by_tag_df.empty:
        summary_by_tag_df = _aggregate_tag_summary(selected_scenarios_df)

    figure_path = make_robustness_map(
        selected_candidate_id=selected_candidate_id,
        selected_scenarios_df=selected_scenarios_df,
        summary_by_tag_df=summary_by_tag_df,
    )
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
