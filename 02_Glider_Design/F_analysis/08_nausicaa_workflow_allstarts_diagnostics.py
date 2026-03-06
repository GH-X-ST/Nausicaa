"""
Plot multistart diagnostics for one workflow run.

Input workbook:
    C_results/nausicaa_workflow_iter3.xlsx
Sheet:
    AllStarts

Output figure:
    B_figures/08_nausicaa_workflow_allstarts_diagnostics.png
"""

###### Initialization

### Imports
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


### User settings
INPUT_XLSX = Path("C_results/nausicaa_workflow_iter3.xlsx")
SHEET_NAME = "AllStarts"
OUT_DIR = Path("B_figures")
OUT_PATH = OUT_DIR / "08_nausicaa_workflow_allstarts_diagnostics.png"

REQUIRED_COLUMNS = [
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
COLOR_FAILURE = "#e45756"
COLOR_KEPT = "#ffa600"
COLOR_BEST = "#ffd200"

SIZE_BASE = 28
SIZE_KEPT = 42
SIZE_BEST = 130


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


def load_allstarts_df(xlsx_path: Path) -> pd.DataFrame:
    """Load AllStarts sheet and enforce required fields."""
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
    df["start_index"] = pd.to_numeric(df["start_index"], errors="raise").astype(int)
    df["success"] = coerce_bool_like(df["success"])
    df["objective"] = pd.to_numeric(df["objective"], errors="coerce")
    df["kept_after_dedup"] = coerce_bool_like(df["kept_after_dedup"])
    df["kept_rank"] = pd.to_numeric(df["kept_rank"], errors="coerce")

    df = df.sort_values(by="start_index", ascending=True).reset_index(drop=True)
    return df


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID_COLOR, linewidth=GRID_LINEWIDTH)
    ax.tick_params(axis="both", which="major", length=2, width=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_EDGE_LW)


def compute_failure_nan_level(df: pd.DataFrame) -> float:
    """
    Y-level to place failures with NaN objective.
    """
    success_obj = df.loc[df["success"] & df["objective"].notna(), "objective"]

    if not success_obj.empty:
        y_min = float(success_obj.min())
        y_max = float(success_obj.max())
    else:
        finite_obj = df.loc[df["objective"].notna(), "objective"]
        if not finite_obj.empty:
            y_min = float(finite_obj.min())
            y_max = float(finite_obj.max())
        else:
            y_min, y_max = 0.0, 1.0

    y_span = max(1e-9, y_max - y_min)
    return y_max + 0.08 * y_span


def plot_allstarts_diagnostics(df: pd.DataFrame, out_path: Path) -> None:
    """Scatter diagnostics: objective over start index with pipeline highlighting."""
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

    fig, ax = plt.subplots(figsize=(5.9, 4.0), dpi=600)
    fig.patch.set_facecolor("white")
    style_axes(ax)

    x = df["start_index"].to_numpy(dtype=float)
    success_mask = df["success"].to_numpy(dtype=bool)
    kept_mask = df["kept_after_dedup"].to_numpy(dtype=bool)
    obj = df["objective"].to_numpy(dtype=float)

    fail_nan_level = compute_failure_nan_level(df)
    y = obj.copy()
    fail_nan_mask = (~success_mask) & (~np.isfinite(y))
    y[fail_nan_mask] = fail_nan_level

    # Success non-kept
    mask_success_not_kept = success_mask & (~kept_mask) & np.isfinite(y)
    if np.any(mask_success_not_kept):
        ax.scatter(
            x[mask_success_not_kept],
            y[mask_success_not_kept],
            s=SIZE_BASE,
            color=COLOR_SUCCESS,
            alpha=0.78,
            edgecolors="none",
            marker="o",
            zorder=4,
        )

    # Success kept-after-dedup
    mask_success_kept = success_mask & kept_mask & np.isfinite(y)
    if np.any(mask_success_kept):
        ax.scatter(
            x[mask_success_kept],
            y[mask_success_kept],
            s=SIZE_KEPT,
            color=COLOR_KEPT,
            alpha=0.95,
            edgecolors="black",
            linewidths=0.55,
            marker="D",
            zorder=6,
        )

    # Failures with finite objective
    mask_fail_finite = (~success_mask) & np.isfinite(y) & (~fail_nan_mask)
    if np.any(mask_fail_finite):
        ax.scatter(
            x[mask_fail_finite],
            y[mask_fail_finite],
            s=SIZE_BASE,
            color=COLOR_FAILURE,
            alpha=0.90,
            edgecolors="none",
            marker="x",
            linewidths=0.9,
            zorder=5,
        )

    # Failures with NaN objective shown at fixed y-level.
    if np.any(fail_nan_mask):
        ax.scatter(
            x[fail_nan_mask],
            y[fail_nan_mask],
            s=SIZE_BASE,
            color=COLOR_FAILURE,
            alpha=0.95,
            edgecolors="none",
            marker="x",
            linewidths=0.9,
            zorder=7,
        )
        for xi in x[fail_nan_mask]:
            ax.text(
                float(xi),
                fail_nan_level,
                " FAIL",
                ha="left",
                va="center",
                fontsize=7.5,
                color=COLOR_FAILURE,
                zorder=8,
            )

    # Best kept (rank 1) highlight.
    best_rank1_mask = kept_mask & df["kept_rank"].notna().to_numpy(dtype=bool) & (df["kept_rank"].to_numpy(dtype=float) == 1.0)
    if np.any(best_rank1_mask):
        i_best = int(np.flatnonzero(best_rank1_mask)[0])
        ax.scatter(
            [x[i_best]],
            [y[i_best]],
            s=SIZE_BEST,
            marker="*",
            color=COLOR_BEST,
            edgecolors="black",
            linewidths=0.9,
            zorder=12,
        )
        ax.annotate(
            "Best kept (rank 1)",
            xy=(x[i_best], y[i_best]),
            xytext=(10, 12),
            textcoords="offset points",
            fontsize=9,
            ha="left",
            va="bottom",
            arrowprops={
                "arrowstyle": "->",
                "lw": 0.7,
                "color": "black",
            },
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": (1.0, 1.0, 1.0, 0.82),
                "edgecolor": "none",
            },
            zorder=20,
        )

    # Axis limits with margins.
    finite_y = y[np.isfinite(y)]
    if finite_y.size > 0:
        y_min = float(np.min(finite_y))
        y_max = float(np.max(finite_y))
    else:
        y_min, y_max = 0.0, 1.0

    y_span = max(1e-9, y_max - y_min)
    ax.set_xlim(float(np.min(x)) - 0.4, float(np.max(x)) + 0.4)
    ax.set_ylim(y_min - 0.08 * y_span, y_max + 0.18 * y_span)

    ax.set_xlabel("Multistart index")
    ax.set_ylabel("Objective [-]")

    summary_txt = (
        f"starts={len(df)}, success={int(success_mask.sum())}, "
        f"kept_after_dedup={int(kept_mask.sum())}"
    )
    ax.text(
        0.01,
        0.99,
        summary_txt,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": (1.0, 1.0, 1.0, 0.75),
            "edgecolor": "none",
        },
        zorder=20,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLOR_SUCCESS,
            markeredgecolor="none",
            markersize=5,
            alpha=0.8,
            label="Success",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color=COLOR_FAILURE,
            markersize=6,
            linewidth=0,
            label="Failure",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor=COLOR_KEPT,
            markeredgecolor="black",
            markeredgewidth=0.55,
            markersize=6,
            label="Kept after dedup",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor=COLOR_BEST,
            markeredgecolor="black",
            markeredgewidth=0.9,
            markersize=8,
            label="Best kept (rank 1)",
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

    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


### Entrypoint
def main() -> None:
    df = load_allstarts_df(INPUT_XLSX)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_allstarts_diagnostics(df=df, out_path=OUT_PATH)

    best_rank1_count = int(((df["kept_rank"] == 1) & df["kept_after_dedup"]).sum())
    print(f"Source sheet used: {SHEET_NAME}")
    print(f"Rows plotted: {len(df)}")
    print(f"Success count: {int(df['success'].sum())}")
    print(f"Kept-after-dedup count: {int(df['kept_after_dedup'].sum())}")
    print(f"Best kept rank-1 count: {best_rank1_count}")
    print(f"Saved: {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()

