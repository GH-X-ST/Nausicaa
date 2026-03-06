"""
Plot key trade-study ranking metrics in final lexicographic selection order.

Input workbook:
    C_results/weight_sweep_iter3.xlsx
Preferred sheet:
    TradeStudy
Fallback sheet:
    WeightSweep (manually sorted by selection rule)

Output figure:
    B_figures/01_weight_sweep_selection_rule.png
"""

###### Initialization

### Imports
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter


### User settings
INPUT_XLSX = Path("C_results/weight_sweep_iter3.xlsx")
OUT_DIR = Path("B_figures")
OUT_PATH = OUT_DIR / "01_weight_sweep_selection_rule.png"

PREFERRED_SHEET = "TradeStudy"
FALLBACK_SHEET = "WeightSweep"

REQUIRED_COLUMNS = [
    "run_index",
    "feasible_rate",
    "sink_cvar_20",
    "wing_span_m",
    "is_best",
]

# Plot style aligned with existing F_analysis scripts
AXIS_EDGE_LW = 0.80
GRID_COLOR = (0.85, 0.85, 0.85, 1.0)
GRID_LINEWIDTH = 0.4

COLOR_FEASIBLE = "#4c78a8"
COLOR_CVAR = "#cc4a74"
COLOR_SPAN = "#ffa600"
COLOR_BEST = "#ffd200"


### Helpers
def coerce_bool_like(series: pd.Series) -> pd.Series:
    """
    Convert bool-like values to strict boolean.
    """
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
        raise ValueError(f"Unable to coerce 'is_best' to bool. Bad values: {bad_values}")
    return mapped.astype(bool)


def validate_required_columns(df: pd.DataFrame, sheet_name: str) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in '{sheet_name}': {missing}")


def load_trade_study_frame(xlsx_path: Path) -> Tuple[pd.DataFrame, str]:
    """
    Load TradeStudy if available; otherwise load WeightSweep and sort by selection logic.
    Returns the dataframe in plotting order and selected sheet name.
    """
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Workbook not found: {xlsx_path}")

    xls = pd.ExcelFile(xlsx_path)
    sheet_names = xls.sheet_names

    if PREFERRED_SHEET in sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=PREFERRED_SHEET)
        source_sheet = PREFERRED_SHEET
        fallback_sorted = False
    elif FALLBACK_SHEET in sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=FALLBACK_SHEET)
        source_sheet = FALLBACK_SHEET
        fallback_sorted = True
    else:
        raise KeyError(
            f"Neither '{PREFERRED_SHEET}' nor '{FALLBACK_SHEET}' found in {xlsx_path}."
        )

    validate_required_columns(df, source_sheet)

    df = df.copy()
    df["run_index"] = pd.to_numeric(df["run_index"], errors="raise").astype(int)
    df["feasible_rate"] = pd.to_numeric(df["feasible_rate"], errors="raise")
    df["sink_cvar_20"] = pd.to_numeric(df["sink_cvar_20"], errors="raise")
    df["wing_span_m"] = pd.to_numeric(df["wing_span_m"], errors="raise")
    df["is_best"] = coerce_bool_like(df["is_best"])

    if fallback_sorted:
        df = df.sort_values(
            by=["feasible_rate", "sink_cvar_20", "wing_span_m", "run_index"],
            ascending=[False, True, True, True],
            kind="mergesort",
        )

    df = df.reset_index(drop=True)
    return df, source_sheet


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID_COLOR, linewidth=GRID_LINEWIDTH)
    ax.tick_params(axis="both", which="major", length=2, width=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_EDGE_LW)


def annotate_best_row(ax: plt.Axes, x_best: int, y_best: float, n_rows: int) -> None:
    """
    Add one best-row annotation near the selected row.
    """
    x_offset = -5 if x_best >= int(0.75 * n_rows) else 1
    y_span = max(1e-9, ax.get_ylim()[1] - ax.get_ylim()[0])
    y_text = y_best + 0.06 * y_span
    ax.annotate(
        "Best (is_best=True)",
        xy=(x_best, y_best),
        xytext=(x_best + x_offset, y_text),
        textcoords="data",
        fontsize=9,
        arrowprops={
            "arrowstyle": "->",
            "lw": 0.7,
            "color": "black",
        },
        ha="left" if x_offset > 0 else "right",
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": (1.0, 1.0, 1.0, 0.8),
            "edgecolor": "none",
        },
        zorder=20,
    )


def plot_selection_rule(df: pd.DataFrame, out_path: Path) -> None:
    """
    Plot 3-row shared-x figure in the same row order as lexicographic selection.
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

    x = np.arange(len(df), dtype=int)
    feasible = df["feasible_rate"].to_numpy(dtype=float)
    sink_cvar = df["sink_cvar_20"].to_numpy(dtype=float)
    span = df["wing_span_m"].to_numpy(dtype=float)
    best_idx = np.flatnonzero(df["is_best"].to_numpy(dtype=bool))

    fig, axes = plt.subplots(3, 1, figsize=(5.7, 6.8), dpi=600, sharex=True)
    fig.patch.set_facecolor("white")

    for ax in axes:
        style_axes(ax)
        for i_best in best_idx:
            ax.axvline(
                x=i_best,
                color=(0.0, 0.0, 0.0, 0.60),
                linewidth=0.7,
                linestyle=(0, (3, 2)),
                zorder=1,
            )

    # Panel A: feasible rate
    bars = axes[0].bar(
        x,
        feasible,
        width=0.82,
        color=COLOR_FEASIBLE,
        alpha=0.85,
        edgecolor="none",
        zorder=3,
    )
    for i_best in best_idx:
        bars[i_best].set_edgecolor("black")
        bars[i_best].set_linewidth(0.95)
    axes[0].scatter(
        best_idx,
        feasible[best_idx],
        marker="*",
        s=70,
        color=COLOR_BEST,
        edgecolors="black",
        linewidths=0.7,
        zorder=6,
    )
    axes[0].set_ylabel("Feasible rate [-]")
    axes[0].set_ylim(0.0, max(1.0, float(np.nanmax(feasible)) * 1.08))
    if best_idx.size > 0:
        first_best = int(best_idx[0])
        annotate_best_row(
            ax=axes[0],
            x_best=first_best,
            y_best=float(feasible[first_best]),
            n_rows=len(df),
        )

    # Panel B: sink CVaR20
    axes[1].plot(
        x,
        sink_cvar,
        color=COLOR_CVAR,
        linewidth=1.0,
        marker="o",
        markersize=3.0,
        markerfacecolor=COLOR_CVAR,
        markeredgewidth=0.0,
        zorder=4,
    )
    axes[1].scatter(
        best_idx,
        sink_cvar[best_idx],
        marker="*",
        s=70,
        color=COLOR_BEST,
        edgecolors="black",
        linewidths=0.7,
        zorder=7,
    )
    axes[1].set_ylabel(r"Sink CVaR$_{20}$ [m/s]")
    axes[1].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    # Panel C: wing span
    axes[2].plot(
        x,
        span,
        color=COLOR_SPAN,
        linewidth=1.0,
        marker="o",
        markersize=3.0,
        markerfacecolor=COLOR_SPAN,
        markeredgewidth=0.0,
        zorder=4,
    )
    axes[2].scatter(
        best_idx,
        span[best_idx],
        marker="*",
        s=70,
        color=COLOR_BEST,
        edgecolors="black",
        linewidths=0.7,
        zorder=7,
    )
    axes[2].set_ylabel("Wing span [m]")
    axes[2].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    axes[2].set_xlabel("TradeStudy order (lexicographic selection)")

    n_rows = len(df)
    tick_step = max(1, int(np.ceil(n_rows / 16)))
    xticks = np.arange(0, n_rows, tick_step, dtype=int)
    axes[2].set_xticks(xticks)
    axes[2].set_xlim(-0.5, n_rows - 0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


### Entrypoint
def main() -> None:
    df, source_sheet = load_trade_study_frame(INPUT_XLSX)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_selection_rule(df, OUT_PATH)

    print(f"Source sheet used: {source_sheet}")
    print(f"Rows plotted: {len(df)}")
    print(f"Saved: {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()

