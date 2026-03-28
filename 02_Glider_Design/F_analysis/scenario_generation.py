from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "C_results"
FIGURES_DIR = PROJECT_ROOT / "B_figures"
WORKFLOW_XLSX = RESULTS_DIR / "nausicaa_workflow.xlsx"
RESULTS_XLSX = RESULTS_DIR / "nausicaa_results.xlsx"
FIGURE_PATH = FIGURES_DIR / "scenario_generation.png"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TAG_ORDER = [
    "harsh_compound",
    "harsh_build",
    "gusty_only",
    "mild_compound",
    "mild_build",
]
KEY_INPUTS = [
    "mass_scale",
    "cg_x_shift_mac",
    "incidence_bias_deg",
    "control_eff",
    "drag_factor",
    "w_gust_nom",
    "q",
]


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


def load_scenario_generation_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    workflow_book = _open_workbook(WORKFLOW_XLSX)
    results_book = _open_workbook(RESULTS_XLSX)
    scenario_inputs_long_df = _read_sheet("ScenarioInputsLong", workflow_book, results_book)
    robust_scenarios_df = _read_sheet("RobustScenarios", workflow_book, results_book)
    definitions_df = _read_sheet("Definitions", workflow_book, results_book)

    if robust_scenarios_df is None:
        raise FileNotFoundError(
            "Expected canonical workflow workbook with a 'RobustScenarios' sheet."
        )

    if scenario_inputs_long_df is None:
        input_columns = [column for column in KEY_INPUTS if column in robust_scenarios_df.columns]
        scenario_inputs_long_df = robust_scenarios_df[
            ["scenario_id", "scenario_tag", *input_columns]
        ].drop_duplicates("scenario_id").melt(
            id_vars=["scenario_id", "scenario_tag"],
            var_name="input_name",
            value_name="input_value",
        )

    if definitions_df is None:
        definitions_df = pd.DataFrame(
            columns=["name", "unit", "definition", "computed_in", "notes"]
        )

    return scenario_inputs_long_df, robust_scenarios_df, definitions_df


def get_unique_scenarios(robust_scenarios_df: pd.DataFrame) -> pd.DataFrame:
    unique_df = robust_scenarios_df.drop_duplicates(subset=["scenario_id"]).copy()
    if "control_eff" not in unique_df.columns:
        eff_columns = [column for column in ["eff_a", "eff_e", "eff_r"] if column in unique_df.columns]
        if eff_columns:
            unique_df["control_eff"] = unique_df[eff_columns].mean(axis=1)
    return unique_df


def sort_tags(tags: list[str] | pd.Index | pd.Series) -> list[str]:
    unique_tags = list(dict.fromkeys(pd.Series(tags).dropna().astype(str)))
    tag_lookup = {tag: idx for idx, tag in enumerate(TAG_ORDER)}
    return sorted(unique_tags, key=lambda tag: (tag_lookup.get(tag, len(TAG_ORDER)), tag))


def _point_positions(center: float, count: int, span: float = 0.18) -> np.ndarray:
    if count <= 1:
        return np.array([center], dtype=float)
    return np.linspace(center - span / 2.0, center + span / 2.0, count)


def _boxplot_by_tag(
    ax: plt.Axes,
    unique_scenarios_df: pd.DataFrame,
    tags: list[str],
    column: str,
) -> None:
    data = [
        unique_scenarios_df.loc[unique_scenarios_df["scenario_tag"] == tag, column]
        .dropna()
        .to_numpy(dtype=float)
        for tag in tags
    ]
    boxplot = ax.boxplot(data, tick_labels=tags, patch_artist=True, showfliers=False)
    for patch in boxplot["boxes"]:
        patch.set_facecolor("#9ecae1")
        patch.set_edgecolor("black")
        patch.set_alpha(0.85)

    for index, values in enumerate(data, start=1):
        if values.size == 0:
            continue
        ax.scatter(
            _point_positions(float(index), int(values.size)),
            values,
            s=18,
            facecolors="white",
            edgecolors="#3b3b3b",
            linewidths=0.5,
            alpha=0.85,
            zorder=3,
        )
        ax.scatter(
            [float(index)],
            [float(np.mean(values))],
            marker="D",
            s=26,
            facecolors="#1f1f1f",
            edgecolors="white",
            linewidths=0.5,
            zorder=4,
        )
        ax.text(
            float(index),
            0.03,
            f"n={values.size}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=7,
            color="#4a4a4a",
        )

    ax.set_title(column)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.2)


def _boxplot_columns(
    ax: plt.Axes,
    data_df: pd.DataFrame,
    columns: list[str],
    colors: list[str],
) -> None:
    data = [data_df[column].dropna().to_numpy(dtype=float) for column in columns]
    boxplot = ax.boxplot(
        data,
        tick_labels=columns,
        patch_artist=True,
        showfliers=False,
    )
    for patch, color in zip(boxplot["boxes"], colors, strict=False):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_alpha(0.85)

    column_positions = np.arange(1, len(columns) + 1, dtype=float)
    for _, row in data_df[columns].iterrows():
        values = row.to_numpy(dtype=float)
        finite_mask = np.isfinite(values)
        if finite_mask.sum() < 2:
            continue
        ax.plot(
            column_positions[finite_mask],
            values[finite_mask],
            color="#7a7a7a",
            linewidth=0.6,
            alpha=0.18,
            zorder=2,
        )

    for index, values in enumerate(data, start=1):
        if values.size == 0:
            continue
        ax.scatter(
            _point_positions(float(index), int(values.size)),
            values,
            s=18,
            facecolors="white",
            edgecolors="#3b3b3b",
            linewidths=0.5,
            alpha=0.85,
            zorder=3,
        )
        ax.scatter(
            [float(index)],
            [float(np.mean(values))],
            marker="D",
            s=26,
            facecolors="#1f1f1f",
            edgecolors="white",
            linewidths=0.5,
            zorder=4,
        )
        ax.text(
            float(index),
            0.03,
            f"n={values.size}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=7,
            color="#4a4a4a",
        )

    ax.grid(True, axis="y", alpha=0.2)


def make_scenario_generation_figure(
    scenario_inputs_long_df: pd.DataFrame,
    robust_scenarios_df: pd.DataFrame,
    definitions_df: pd.DataFrame,
) -> Path:
    del scenario_inputs_long_df
    del definitions_df

    unique_scenarios_df = get_unique_scenarios(robust_scenarios_df)
    tags = sort_tags(unique_scenarios_df["scenario_tag"])

    fig = plt.figure(figsize=(12, 24), constrained_layout=True)
    outer = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.6], width_ratios=[1.0, 1.0])
    counts_ax = fig.add_subplot(outer[0, 0])
    control_grid = outer[0, 1].subgridspec(2, 1, hspace=0.35)
    eff_ax = fig.add_subplot(control_grid[0, 0])
    bias_ax = fig.add_subplot(control_grid[1, 0])

    available_inputs = [column for column in KEY_INPUTS if column in unique_scenarios_df.columns]
    n_cols = 4
    n_rows = int(np.ceil(max(len(available_inputs), 1) / n_cols))
    input_grid = outer[1, :].subgridspec(n_rows, n_cols, hspace=0.45, wspace=0.25)
    input_axes = [
        fig.add_subplot(input_grid[row, col]) for row in range(n_rows) for col in range(n_cols)
    ]

    counts = (
        unique_scenarios_df.groupby("scenario_tag")["scenario_id"]
        .nunique()
        .reindex(tags, fill_value=0)
    )
    counts_ax.bar(counts.index, counts.values, color="#4c78a8", edgecolor="black", linewidth=0.5)
    counts_ax.set_title("Scenario family counts")
    counts_ax.set_ylabel("Unique scenarios")
    counts_ax.tick_params(axis="x", rotation=25)
    counts_ax.grid(True, axis="y", alpha=0.25)
    for index, value in enumerate(counts.values):
        counts_ax.text(
            index,
            float(value) + 0.05,
            str(int(value)),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    for ax, column in zip(input_axes, available_inputs, strict=False):
        _boxplot_by_tag(ax, unique_scenarios_df, tags, column)
    for ax in input_axes[len(available_inputs):]:
        ax.axis("off")

    eff_columns = [column for column in ["eff_a", "eff_e", "eff_r"] if column in unique_scenarios_df.columns]
    if eff_columns:
        _boxplot_columns(
            eff_ax,
            unique_scenarios_df,
            eff_columns,
            ["#59a14f", "#76b7b2", "#edc948"],
        )
        eff_ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        eff_ax.set_title("Per-axis control effectiveness")
    else:
        eff_ax.text(0.5, 0.5, "No per-axis effectiveness data", ha="center", va="center")
        eff_ax.axis("off")

    bias_columns = [
        column for column in ["bias_a_deg", "bias_e_deg", "bias_r_deg"] if column in unique_scenarios_df.columns
    ]
    if bias_columns:
        _boxplot_columns(
            bias_ax,
            unique_scenarios_df,
            bias_columns,
            ["#e15759", "#f28e2b", "#b07aa1"],
        )
        bias_ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        bias_ax.set_title("Per-axis control bias [deg]")
    else:
        bias_ax.text(0.5, 0.5, "No per-axis bias data", ha="center", va="center")
        bias_ax.axis("off")

    fig.suptitle("Scenario Generation and Uncertainty Families", fontsize=13)
    fig.savefig(FIGURE_PATH, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return FIGURE_PATH


def main() -> None:
    scenario_inputs_long_df, robust_scenarios_df, definitions_df = load_scenario_generation_data()
    figure_path = make_scenario_generation_figure(
        scenario_inputs_long_df=scenario_inputs_long_df,
        robust_scenarios_df=robust_scenarios_df,
        definitions_df=definitions_df,
    )
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
