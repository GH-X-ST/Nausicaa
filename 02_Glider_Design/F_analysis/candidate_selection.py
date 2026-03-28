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
FIGURE_PATH = FIGURES_DIR / "candidate_selection.png"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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


def load_candidate_selection_data() -> pd.DataFrame:
    workflow_book = _open_workbook(WORKFLOW_XLSX)
    results_book = _open_workbook(RESULTS_XLSX)
    candidates_df = _read_sheet("Candidates", workflow_book, results_book)
    robust_summary_df = _read_sheet("RobustSummary", workflow_book, results_book)

    if candidates_df is None or robust_summary_df is None:
        raise FileNotFoundError(
            "Required workbook sheets were not found. Expected "
            f"{WORKFLOW_XLSX} with 'Candidates' and 'RobustSummary'."
        )

    return candidates_df.merge(
        robust_summary_df,
        on="candidate_id",
        how="left",
        suffixes=("", "_robust"),
    )


def resolve_tail_metric_column(df: pd.DataFrame) -> str:
    if "nom_sink_tail_mean_k" in df.columns:
        return "nom_sink_tail_mean_k"
    if "nom_sink_cvar_20" in df.columns:
        return "nom_sink_cvar_20"
    raise KeyError("Neither 'nom_sink_tail_mean_k' nor 'nom_sink_cvar_20' is available.")


def build_rank_columns(df: pd.DataFrame, tail_metric_col: str) -> pd.DataFrame:
    ranked_df = df.copy()
    if "nom_success_rate" not in ranked_df.columns:
        ranked_df["nom_success_rate"] = np.nan

    ranked_df = ranked_df.sort_values(
        by=["nom_success_rate", tail_metric_col, "objective"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked_df["robust_rank"] = np.arange(1, len(ranked_df) + 1, dtype=int)

    if "is_selected" in ranked_df.columns and _coerce_bool_series(
        ranked_df["is_selected"]
    ).any():
        ranked_df["is_selected"] = _coerce_bool_series(ranked_df["is_selected"])
    else:
        ranked_df["is_selected"] = False
        if not ranked_df.empty:
            ranked_df.loc[0, "is_selected"] = True

    return ranked_df


def make_candidate_selection_plot(df: pd.DataFrame, tail_metric_col: str) -> Path:
    plot_df = build_rank_columns(df, tail_metric_col)
    selected_df = plot_df.loc[plot_df["is_selected"]]
    top_rank_df = plot_df.nsmallest(min(3, len(plot_df)), "robust_rank")

    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    if "mass_total_kg" in plot_df.columns and plot_df["mass_total_kg"].notna().any():
        mass = plot_df["mass_total_kg"].to_numpy(dtype=float)
        span = np.nanmax(mass) - np.nanmin(mass)
        sizes = 70.0 + 110.0 * (mass - np.nanmin(mass)) / max(span, 1e-9)
    else:
        sizes = np.full(len(plot_df), 90.0)

    scatter = ax.scatter(
        plot_df["objective"],
        plot_df[tail_metric_col],
        c=plot_df["nom_success_rate"],
        s=sizes,
        cmap="viridis",
        edgecolors="white",
        linewidths=0.8,
        alpha=0.9,
    )

    if not selected_df.empty:
        ax.scatter(
            selected_df["objective"],
            selected_df[tail_metric_col],
            marker="*",
            s=340,
            c="#ffcc00",
            edgecolors="black",
            linewidths=1.2,
            zorder=5,
        )
        selected_row = selected_df.iloc[0]
        ax.annotate(
            "selected",
            xy=(selected_row["objective"], selected_row[tail_metric_col]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )

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

    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Nominal robust success rate")

    metric_label = (
        "Tail-risk sink mean (worst 20%)"
        if tail_metric_col == "nom_sink_tail_mean_k"
        else "Sink CVaR20"
    )
    ax.set_xlabel("Nominal objective")
    ax.set_ylabel(metric_label + " [m/s]")
    ax.set_title("Candidate Selection")
    ax.grid(True, alpha=0.25)

    note_lines = [
        f"candidates: {len(plot_df)}",
        "ranking: success desc, tail risk asc, objective asc",
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(note_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    fig.savefig(FIGURE_PATH, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return FIGURE_PATH


def main() -> None:
    merged_df = load_candidate_selection_data()
    tail_metric_col = resolve_tail_metric_column(merged_df)
    figure_path = make_candidate_selection_plot(merged_df, tail_metric_col)
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
