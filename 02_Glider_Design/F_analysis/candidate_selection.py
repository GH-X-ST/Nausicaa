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


def load_candidate_selection_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    workflow_book = _open_workbook(WORKFLOW_XLSX)
    results_book = _open_workbook(RESULTS_XLSX)
    all_starts_df = _read_sheet("AllStarts", workflow_book, results_book)
    candidates_df = _read_sheet("Candidates", workflow_book, results_book)
    robust_summary_df = _read_sheet("RobustSummary", workflow_book, results_book)

    if all_starts_df is None:
        raise FileNotFoundError(
            "Required sheet 'AllStarts' was not found in "
            f"{WORKFLOW_XLSX}. The trace figure requires workflow data."
        )
    if candidates_df is None or robust_summary_df is None:
        raise FileNotFoundError(
            "Required workbook sheets were not found. Expected "
            f"{WORKFLOW_XLSX} with 'Candidates' and 'RobustSummary'."
        )

    if "success" in all_starts_df.columns:
        all_starts_df["success"] = _coerce_bool_series(all_starts_df["success"])
    if "kept_after_dedup" in all_starts_df.columns:
        all_starts_df["kept_after_dedup"] = _coerce_bool_series(
            all_starts_df["kept_after_dedup"]
        )
    elif "status" in all_starts_df.columns:
        all_starts_df["kept_after_dedup"] = (
            all_starts_df["status"].astype(str).str.strip().str.lower().eq("kept")
        )

    if "is_selected" in robust_summary_df.columns:
        robust_summary_df["is_selected"] = _coerce_bool_series(
            robust_summary_df["is_selected"]
        )

    return all_starts_df, candidates_df, robust_summary_df


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


def build_stage_position_map(ids: list[int]) -> dict[int, float]:
    n_ids = len(ids)
    return {
        int(identifier): float(n_ids - index - 1)
        for index, identifier in enumerate(ids)
    }


def make_candidate_selection_plot(
    all_starts_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    robust_summary_df: pd.DataFrame,
) -> Path:
    retained_df = candidates_df.merge(
        robust_summary_df,
        on="candidate_id",
        how="left",
        suffixes=("", "_robust"),
    )
    tail_metric_col = resolve_tail_metric_column(retained_df)
    ranked_df = build_rank_columns(retained_df, tail_metric_col)

    all_sorted_df = _sort_all_starts(all_starts_df)
    feasible_df = _sort_feasible(all_sorted_df.loc[all_sorted_df["success"]].copy())
    kept_df = _sort_kept(
        feasible_df.loc[feasible_df["kept_after_dedup"]].copy()
    )
    ranked_sorted_df = ranked_df.sort_values(
        by=["robust_rank"],
        kind="mergesort",
    ).reset_index(drop=True)

    all_map = build_stage_position_map(all_sorted_df["start_index"].astype(int).tolist())
    feasible_map = build_stage_position_map(feasible_df["start_index"].astype(int).tolist())
    kept_map = build_stage_position_map(kept_df["start_index"].astype(int).tolist())
    robust_map = build_stage_position_map(
        ranked_sorted_df["candidate_id"].astype(int).tolist()
    )
    kept_to_candidate_map = _resolve_kept_candidate_map(kept_df, ranked_sorted_df)

    all_x = 0.0
    feasible_x = 1.0
    kept_x = 2.0
    robust_x = 3.0

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 5.8),
        gridspec_kw={"width_ratios": [1.0, 1.25]},
    )
    ax_trace, ax_trade = axes

    failed_df = all_sorted_df.loc[~all_sorted_df["success"]].copy()
    discarded_df = feasible_df.loc[~feasible_df["kept_after_dedup"]].copy()

    discarded_color = "#9e9e9e"
    kept_color = "#4e79a7"
    selected_color = "#f28e2b"

    for _, row in discarded_df.iterrows():
        start_id = int(row["start_index"])
        ax_trace.plot(
            [all_x, feasible_x],
            [all_map[start_id], feasible_map[start_id]],
            color=discarded_color,
            linewidth=0.8,
            alpha=0.18,
            zorder=1,
        )

    for _, row in kept_df.iterrows():
        start_id = int(row["start_index"])
        candidate_id = kept_to_candidate_map.get(start_id)
        if candidate_id is None or candidate_id not in robust_map:
            continue
        ax_trace.plot(
            [all_x, feasible_x],
            [all_map[start_id], feasible_map[start_id]],
            color=kept_color,
            linewidth=0.9,
            alpha=0.3,
            zorder=1,
        )
        ax_trace.plot(
            [feasible_x, kept_x],
            [feasible_map[start_id], kept_map[start_id]],
            color=kept_color,
            linewidth=0.9,
            alpha=0.3,
            zorder=1,
        )
        ax_trace.plot(
            [kept_x, robust_x],
            [kept_map[start_id], robust_map[candidate_id]],
            color=kept_color,
            linewidth=0.9,
            alpha=0.3,
            zorder=1,
        )

    selected_row = ranked_sorted_df.loc[ranked_sorted_df["is_selected"]].iloc[0]
    selected_candidate_id = int(selected_row["candidate_id"])
    selected_start_ids = [
        start_id
        for start_id, candidate_id in kept_to_candidate_map.items()
        if candidate_id == selected_candidate_id
    ]
    if selected_start_ids:
        selected_start_id = int(selected_start_ids[0])
        ax_trace.plot(
            [all_x, feasible_x, kept_x, robust_x],
            [
                all_map[selected_start_id],
                feasible_map[selected_start_id],
                kept_map[selected_start_id],
                robust_map[selected_candidate_id],
            ],
            color=selected_color,
            linewidth=2.2,
            alpha=0.95,
            zorder=3,
        )

    if not failed_df.empty:
        ax_trace.scatter(
            np.full(len(failed_df), all_x),
            failed_df["start_index"].map(all_map),
            marker="x",
            s=18,
            color=discarded_color,
            linewidths=0.8,
            alpha=0.7,
            zorder=4,
        )

    ax_trace.scatter(
        np.full(len(feasible_df), all_x),
        feasible_df["start_index"].map(all_map),
        s=16,
        facecolors="#bfbfbf",
        edgecolors="none",
        alpha=0.8,
        zorder=4,
    )
    ax_trace.scatter(
        np.full(len(discarded_df), feasible_x),
        discarded_df["start_index"].map(feasible_map),
        s=24,
        facecolors="white",
        edgecolors=discarded_color,
        linewidths=0.8,
        alpha=0.9,
        zorder=4,
    )
    ax_trace.scatter(
        np.full(len(kept_df), feasible_x),
        kept_df["start_index"].map(feasible_map),
        s=24,
        facecolors=kept_color,
        edgecolors="white",
        linewidths=0.4,
        alpha=0.9,
        zorder=5,
    )
    ax_trace.scatter(
        np.full(len(kept_df), kept_x),
        kept_df["start_index"].map(kept_map),
        s=28,
        facecolors=kept_color,
        edgecolors="white",
        linewidths=0.4,
        alpha=0.95,
        zorder=5,
    )

    if selected_start_ids:
        ax_trace.scatter(
            [kept_x],
            [kept_map[selected_start_id]],
            s=46,
            facecolors=selected_color,
            edgecolors="black",
            linewidths=0.8,
            zorder=6,
        )

    trace_scatter = ax_trade.scatter(
        ranked_sorted_df["objective"],
        ranked_sorted_df[tail_metric_col],
        c=ranked_sorted_df["nom_success_rate"],
        cmap="viridis",
        s=np.full(len(ranked_sorted_df), 100.0),
        edgecolors="white",
        linewidths=0.8,
        alpha=0.95,
        zorder=2,
    )

    if "mass_total_kg" in ranked_sorted_df.columns and ranked_sorted_df["mass_total_kg"].notna().any():
        mass = ranked_sorted_df["mass_total_kg"].to_numpy(dtype=float)
        span = np.nanmax(mass) - np.nanmin(mass)
        sizes = 70.0 + 110.0 * (mass - np.nanmin(mass)) / max(span, 1e-9)
        trace_scatter.set_sizes(sizes)

    ax_trace.scatter(
        np.full(len(ranked_sorted_df), robust_x),
        ranked_sorted_df["candidate_id"].map(robust_map),
        c=ranked_sorted_df["nom_success_rate"],
        cmap="viridis",
        vmin=float(ranked_sorted_df["nom_success_rate"].min()),
        vmax=float(ranked_sorted_df["nom_success_rate"].max()),
        s=34,
        edgecolors="white",
        linewidths=0.5,
        alpha=0.95,
        zorder=5,
    )
    ax_trace.scatter(
        [robust_x],
        [robust_map[selected_candidate_id]],
        marker="*",
        s=180,
        c="#ffcc00",
        edgecolors="black",
        linewidths=1.0,
        zorder=7,
    )

    ax_trade.scatter(
        [selected_row["objective"]],
        [selected_row[tail_metric_col]],
        marker="*",
        s=260,
        c="#ffcc00",
        edgecolors="black",
        linewidths=1.1,
        zorder=4,
    )
    ax_trade.annotate(
        "selected",
        xy=(selected_row["objective"], selected_row[tail_metric_col]),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
    )
    ax_trace.annotate(
        "selected",
        xy=(robust_x, robust_map[selected_candidate_id]),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold",
    )

    top_rank_df = ranked_sorted_df.nsmallest(min(3, len(ranked_sorted_df)), "robust_rank")
    for _, row in top_rank_df.iterrows():
        if int(row["candidate_id"]) == selected_candidate_id:
            continue
        label = f"#{int(row['robust_rank'])}"
        ax_trade.annotate(
            label,
            xy=(row["objective"], row[tail_metric_col]),
            xytext=(6, -12),
            textcoords="offset points",
            fontsize=9,
        )
        ax_trace.annotate(
            label,
            xy=(robust_x, robust_map[int(row["candidate_id"])]),
            xytext=(6, -10),
            textcoords="offset points",
            fontsize=8,
        )

    stage_counts = [
        ("All starts", len(all_sorted_df), all_x),
        ("Feasible", len(feasible_df), feasible_x),
        ("Kept", len(kept_df), kept_x),
        ("Robust rank", len(ranked_sorted_df), robust_x),
    ]
    max_stage_size = max(
        len(all_sorted_df),
        len(feasible_df),
        len(kept_df),
        len(ranked_sorted_df),
        1,
    )
    y_top = float(max_stage_size) + 0.3
    for label, count, xpos in stage_counts:
        ax_trace.text(
            xpos,
            y_top,
            f"{label}\n$n$ = {count}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax_trace.set_title("Selection trace")
    ax_trace.set_xlim(-0.35, 3.35)
    ax_trace.set_ylim(-1.0, y_top + 1.2)
    ax_trace.set_xticks(
        [all_x, feasible_x, kept_x, robust_x],
        labels=["All starts", "Feasible", "Kept", "Robust rank"],
    )
    ax_trace.set_yticks([])
    ax_trace.spines["left"].set_visible(False)
    ax_trace.spines["right"].set_visible(False)
    ax_trace.spines["top"].set_visible(False)
    ax_trace.grid(False)

    colorbar = fig.colorbar(trace_scatter, ax=ax_trade)
    colorbar.set_label("Nominal robust success rate")

    metric_label = (
        "Tail-risk sink mean (worst 20%) [m/s]"
        if tail_metric_col == "nom_sink_tail_mean_k"
        else "Sink CVaR20 [m/s]"
    )
    ax_trade.set_title("Robust trade-off")
    ax_trade.set_xlabel("Nominal objective")
    ax_trade.set_ylabel(metric_label)
    ax_trade.grid(True, alpha=0.2)
    ax_trade.text(
        0.03,
        0.97,
        "\n".join(
            [
                "ranking: success desc",
                "tail risk asc",
                "objective asc",
                f"retained candidates: {len(ranked_sorted_df)}",
            ]
        ),
        transform=ax_trade.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
    )

    fig.savefig(FIGURE_PATH, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return FIGURE_PATH


def main() -> None:
    all_starts_df, candidates_df, robust_summary_df = load_candidate_selection_data()
    figure_path = make_candidate_selection_plot(
        all_starts_df,
        candidates_df,
        robust_summary_df,
    )
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
