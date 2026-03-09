"""
Plot multistart objective distributions for top rerun weight sets.

Input workbook:
    C_results/weight_sweep_top_rerun_iter5.xlsx
Sheet:
    TopRerunAllStarts

Output figure:
    B_figures/06_weight_sweep_top_rerun_multistart_distribution.png
"""

###### Initialization

### Imports
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


### User settings
INPUT_XLSX = Path("C_results/weight_sweep_top_rerun_iter5.xlsx")
SHEET_NAME = "TopRerunAllStarts"
OUT_DIR = Path("B_figures")
OUT_PATH = OUT_DIR / "06_weight_sweep_top_rerun_multistart_distribution.png"

REQUIRED_COLUMNS = [
    "rerun_rank",
    "start_index",
    "success",
    "objective",
    "kept_after_dedup",
    "kept_rank",
]

# Plot style aligned with F_analysis references
AXIS_EDGE_LW = 0.80
GRID_COLOR = (0.85, 0.85, 0.85, 1.0)
GRID_LINEWIDTH = 0.4

COLOR_SUCCESS = "#4c78a8"
COLOR_KEPT = "#ffa600"
COLOR_FAILURE = "#e45756"
COLOR_BOX = "#72b7b2"

POINT_SIZE_SUCCESS = 18
POINT_SIZE_KEPT = 34
POINT_ALPHA = 0.65


### Helpers
def coerce_bool_like(series: pd.Series) -> pd.Series:
    """Convert bool-like values to strict boolean."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float).astype(int).astype(bool)

    lowered = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
        "t": True,
        "f": False,
    }
    mapped = lowered.map(mapping)
    if mapped.isna().any():
        bad_values = series[mapped.isna()].dropna().unique().tolist()
        raise ValueError(f"Unable to coerce bool-like column. Bad values: {bad_values}")
    return mapped.astype(bool)


def load_top_rerun_allstarts(xlsx_path: Path) -> pd.DataFrame:
    """Load TopRerunAllStarts and enforce required dtypes."""
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Workbook not found: {xlsx_path}")

    xls = pd.ExcelFile(xlsx_path)
    if SHEET_NAME not in xls.sheet_names:
        raise KeyError(f"Sheet '{SHEET_NAME}' not found in {xlsx_path}.")

    df = pd.read_excel(xlsx_path, sheet_name=SHEET_NAME)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in '{SHEET_NAME}': {missing}")

    df = df.copy()
    df["rerun_rank"] = pd.to_numeric(df["rerun_rank"], errors="raise").astype(int)
    df["start_index"] = pd.to_numeric(df["start_index"], errors="coerce")
    df["objective"] = pd.to_numeric(df["objective"], errors="coerce")
    df["kept_rank"] = pd.to_numeric(df["kept_rank"], errors="coerce")

    df["success"] = coerce_bool_like(df["success"])
    df["kept_after_dedup"] = coerce_bool_like(df["kept_after_dedup"])

    df = df.sort_values(by=["rerun_rank", "start_index"], ascending=[True, True])
    return df.reset_index(drop=True)


def build_rank_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-rerun counts for annotation."""
    grouped = df.groupby("rerun_rank", sort=True, observed=True)
    summary = grouped.agg(
        n_starts=("start_index", "size"),
        n_success=("success", "sum"),
        n_kept_after_dedup=("kept_after_dedup", "sum"),
    )
    summary["n_starts"] = summary["n_starts"].astype(int)
    summary["n_success"] = summary["n_success"].astype(int)
    summary["n_kept_after_dedup"] = summary["n_kept_after_dedup"].astype(int)
    return summary


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID_COLOR, linewidth=GRID_LINEWIDTH)
    ax.tick_params(axis="both", which="major", length=2, width=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_EDGE_LW)


def plot_distribution(df: pd.DataFrame, out_path: Path) -> None:
    """
    Box plot + jittered points per rerun_rank for objective distribution.
    """
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.edgecolor": "k",
            "axes.linewidth": AXIS_EDGE_LW,
        }
    )

    df_success = df[df["success"]].copy()
    df_fail = df[~df["success"]].copy()

    rerun_ranks = np.sort(df["rerun_rank"].unique())
    rank_summary = build_rank_summary(df)

    fig, ax = plt.subplots(figsize=(6.0, 4.3), dpi=600)
    fig.patch.set_facecolor("white")
    style_axes(ax)

    # Boxplot data from successful starts with finite objective.
    box_positions: List[int] = []
    box_values: List[np.ndarray] = []
    for rr in rerun_ranks:
        vals = (
            df_success.loc[df_success["rerun_rank"] == rr, "objective"]
            .dropna()
            .to_numpy(dtype=float)
        )
        if vals.size > 0:
            box_positions.append(int(rr))
            box_values.append(vals)

    if box_values:
        ax.boxplot(
            box_values,
            positions=box_positions,
            widths=0.52,
            patch_artist=True,
            boxprops={
                "facecolor": COLOR_BOX,
                "edgecolor": "black",
                "alpha": 0.22,
                "linewidth": 0.8,
            },
            whiskerprops={"color": "black", "linewidth": 0.8},
            capprops={"color": "black", "linewidth": 0.8},
            medianprops={"color": "#cc4a74", "linewidth": 1.2},
            flierprops={
                "marker": "o",
                "markersize": 2,
                "markerfacecolor": "black",
                "markeredgecolor": "black",
                "alpha": 0.2,
            },
            zorder=2,
        )

    rng = np.random.default_rng(42)

    # Success jittered points split by kept_after_dedup.
    for rr in rerun_ranks:
        ds = df_success[df_success["rerun_rank"] == rr].copy()
        if ds.empty:
            continue

        x_center = float(rr)
        x_jitter = rng.uniform(-0.16, 0.16, size=len(ds))
        x_vals = x_center + x_jitter
        y_vals = ds["objective"].to_numpy(dtype=float)
        kept_mask = ds["kept_after_dedup"].to_numpy(dtype=bool)

        if np.any(~kept_mask):
            ax.scatter(
                x_vals[~kept_mask],
                y_vals[~kept_mask],
                s=POINT_SIZE_SUCCESS,
                color=COLOR_SUCCESS,
                alpha=POINT_ALPHA,
                edgecolors="none",
                zorder=4,
            )

        if np.any(kept_mask):
            ax.scatter(
                x_vals[kept_mask],
                y_vals[kept_mask],
                s=POINT_SIZE_KEPT,
                color=COLOR_KEPT,
                alpha=0.95,
                edgecolors="black",
                linewidths=0.45,
                zorder=6,
            )

    # Failure points at a fixed y-level near top, plus count shown in annotation.
    finite_objective = pd.to_numeric(df_success["objective"], errors="coerce")
    if finite_objective.notna().any():
        y_data_min = float(finite_objective.min())
        y_data_max = float(finite_objective.max())
    else:
        y_data_min, y_data_max = 0.0, 1.0

    y_span = max(1e-9, y_data_max - y_data_min)
    y_fail = y_data_max + 0.08 * y_span

    if not df_fail.empty:
        for rr in rerun_ranks:
            n_fail = int((df_fail["rerun_rank"] == rr).sum())
            if n_fail <= 0:
                continue
            x_vals = float(rr) + rng.uniform(-0.14, 0.14, size=n_fail)
            y_vals = np.full(n_fail, y_fail, dtype=float)
            ax.scatter(
                x_vals,
                y_vals,
                s=16,
                marker="x",
                color=COLOR_FAILURE,
                alpha=0.9,
                linewidths=0.8,
                zorder=7,
            )

    # Per-rank counts.
    y_anno = y_data_max + 0.17 * y_span
    for rr in rerun_ranks:
        stats = rank_summary.loc[rr]
        txt = (
            f"n={int(stats['n_starts'])}\n"
            f"succ={int(stats['n_success'])}\n"
            f"kept={int(stats['n_kept_after_dedup'])}"
        )
        ax.text(
            float(rr),
            y_anno,
            txt,
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=(0.05, 0.05, 0.05, 0.95),
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": (1.0, 1.0, 1.0, 0.75),
                "edgecolor": "none",
            },
            zorder=10,
        )

    ax.set_xlabel("Rerun rank")
    ax.set_ylabel("Objective [-]")

    ax.set_xticks(rerun_ranks)
    ax.set_xticklabels([str(int(v)) for v in rerun_ranks])
    ax.set_xlim(float(np.min(rerun_ranks)) - 0.5, float(np.max(rerun_ranks)) + 0.5)
    ax.set_ylim(y_data_min - 0.08 * y_span, y_data_max + 0.32 * y_span)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLOR_SUCCESS,
            markeredgecolor="none",
            markersize=5,
            alpha=POINT_ALPHA,
            label="Successful start",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLOR_KEPT,
            markeredgecolor="black",
            markeredgewidth=0.45,
            markersize=6,
            label="Kept after dedup",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color=COLOR_FAILURE,
            markersize=5,
            linewidth=0,
            label="Failed start",
        ),
    ]
    leg = ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        framealpha=1.0,
        edgecolor="black",
        fontsize=8.5,
        handlelength=1.5,
        borderpad=0.5,
        labelspacing=0.2,
    )
    if leg is not None:
        leg.get_frame().set_linewidth(AXIS_EDGE_LW)

    ax.text(
        0.01,
        0.99,
        "Counts per rank: n=starts, succ=successes, kept=kept_after_dedup",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": (1.0, 1.0, 1.0, 0.75),
            "edgecolor": "none",
        },
        zorder=12,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


### Entrypoint
def main() -> None:
    df = load_top_rerun_allstarts(INPUT_XLSX)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_distribution(df=df, out_path=OUT_PATH)

    print(f"Source sheet used: {SHEET_NAME}")
    print(f"Rows plotted: {len(df)}")
    print(f"Rerun ranks: {sorted(df['rerun_rank'].unique().tolist())}")
    print(f"Success count: {int(df['success'].sum())}")
    print(f"Kept-after-dedup count: {int(df['kept_after_dedup'].sum())}")
    print(f"Saved: {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()

