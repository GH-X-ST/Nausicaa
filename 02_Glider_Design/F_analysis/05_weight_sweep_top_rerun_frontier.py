"""
Plot top-K rerun frontier from weight-sweep follow-up optimization.

Input workbook:
    C_results/weight_sweep_top_rerun_iter3.xlsx
Sheet:
    TopRerunSummary

Output figure:
    B_figures/05_weight_sweep_top_rerun_frontier.png
"""

###### Initialization

### Imports
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

try:
    import cmocean
except ImportError:
    cmocean = None


### User settings
INPUT_XLSX = Path("C_results/weight_sweep_top_rerun_iter3.xlsx")
SHEET_NAME = "TopRerunSummary"
OUT_DIR = Path("B_figures")
OUT_PATH = OUT_DIR / "05_weight_sweep_top_rerun_frontier.png"

REQUIRED_COLUMNS = [
    "rerun_rank",
    "source_sweep_run_index",
    "feasible_rate",
    "sink_cvar_20",
    "wing_span_m",
    "is_best",
]

OPTIONAL_COLUMNS = [
    "objective",
]

# Plot style aligned with F_analysis references
AXIS_EDGE_LW = 0.80
GRID_COLOR = (0.85, 0.85, 0.85, 1.0)
GRID_LINEWIDTH = 0.4

BASE_POINT_ALPHA = 0.90
BEST_MARKER_SIZE = 140
BEST_COLOR = "#ffd200"
ANNOTATION_FONT_SIZE = 8


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
        raise ValueError(f"Unable to coerce 'is_best' to bool. Bad values: {bad_values}")
    return mapped.astype(bool)


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in '{SHEET_NAME}': {missing}")


def load_top_rerun_df(xlsx_path: Path) -> pd.DataFrame:
    """Load and standardize TopRerunSummary sheet."""
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Workbook not found: {xlsx_path}")

    xls = pd.ExcelFile(xlsx_path)
    if SHEET_NAME not in xls.sheet_names:
        raise KeyError(f"Sheet '{SHEET_NAME}' not found in {xlsx_path}.")

    df = pd.read_excel(xlsx_path, sheet_name=SHEET_NAME)
    validate_columns(df)

    df = df.copy()
    df["rerun_rank"] = pd.to_numeric(df["rerun_rank"], errors="raise").astype(int)
    df["source_sweep_run_index"] = pd.to_numeric(
        df["source_sweep_run_index"], errors="raise"
    ).astype(int)
    df["feasible_rate"] = pd.to_numeric(df["feasible_rate"], errors="raise")
    df["sink_cvar_20"] = pd.to_numeric(df["sink_cvar_20"], errors="raise")
    df["wing_span_m"] = pd.to_numeric(df["wing_span_m"], errors="raise")
    df["is_best"] = coerce_bool_like(df["is_best"])

    for col in OPTIONAL_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(by=["rerun_rank", "source_sweep_run_index"], ascending=[True, True])
    return df.reset_index(drop=True)


def marker_sizes_from_span(span: np.ndarray, size_min: float = 40.0, size_max: float = 180.0) -> np.ndarray:
    """Scale wing span to marker areas, with robust fallback for near-constant span."""
    s_min = float(np.nanmin(span))
    s_max = float(np.nanmax(span))

    if np.isclose(s_min, s_max):
        return np.full_like(span, fill_value=0.5 * (size_min + size_max), dtype=float)

    t = (span - s_min) / (s_max - s_min)
    return size_min + t * (size_max - size_min)


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID_COLOR, linewidth=GRID_LINEWIDTH)
    ax.tick_params(axis="both", which="major", length=2, width=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_EDGE_LW)


def annotate_points(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Annotate each point with rerun rank and source sweep run index."""
    x = df["sink_cvar_20"].to_numpy(dtype=float)
    y = df["feasible_rate"].to_numpy(dtype=float)
    ranks = df["rerun_rank"].to_numpy(dtype=int)
    src = df["source_sweep_run_index"].to_numpy(dtype=int)

    for xi, yi, ri, si in zip(x, y, ranks, src):
        label = f"R{ri} (S{si})"
        ax.annotate(
            label,
            xy=(xi, yi),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=ANNOTATION_FONT_SIZE,
            ha="left",
            va="bottom",
            color=(0.05, 0.05, 0.05, 0.95),
            zorder=10,
            path_effects=[
                pe.withStroke(
                    linewidth=2.2,
                    foreground=(1.0, 1.0, 1.0, 0.65),
                )
            ],
        )


def plot_top_rerun_frontier(df: pd.DataFrame, out_path: Path) -> None:
    """Scatter plot: sink CVaR20 vs feasible rate, size by wing span."""
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

    x = df["sink_cvar_20"].to_numpy(dtype=float)
    y = df["feasible_rate"].to_numpy(dtype=float)
    span = df["wing_span_m"].to_numpy(dtype=float)
    rank = df["rerun_rank"].to_numpy(dtype=float)
    best_mask = df["is_best"].to_numpy(dtype=bool)

    sizes = marker_sizes_from_span(span)

    cmap = cmocean.cm.phase if cmocean is not None else plt.get_cmap("viridis")
    rank_min = float(np.nanmin(rank))
    rank_max = float(np.nanmax(rank))
    if np.isclose(rank_min, rank_max):
        rank_norm = np.full_like(rank, 0.5)
    else:
        rank_norm = (rank - rank_min) / (rank_max - rank_min)
    colors = [cmap(v) for v in rank_norm]

    fig, ax = plt.subplots(figsize=(5.7, 3.9), dpi=600)
    fig.patch.set_facecolor("white")
    style_axes(ax)

    # Base points
    ax.scatter(
        x,
        y,
        s=sizes,
        c=colors,
        alpha=BASE_POINT_ALPHA,
        edgecolors="black",
        linewidths=0.45,
        zorder=4,
    )

    annotate_points(ax, df)

    # Best highlight
    if np.any(best_mask):
        xb = x[best_mask]
        yb = y[best_mask]
        ax.scatter(
            xb,
            yb,
            marker="*",
            s=BEST_MARKER_SIZE,
            color=BEST_COLOR,
            edgecolors="black",
            linewidths=0.9,
            zorder=12,
        )

        # Label only the first best if multiple rows happen to be marked.
        i_best = int(np.flatnonzero(best_mask)[0])
        ax.annotate(
            "Best after rerun",
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
                "facecolor": (1.0, 1.0, 1.0, 0.8),
                "edgecolor": "none",
            },
            zorder=20,
        )

    ax.set_xlabel(r"Sink CVaR$_{20}$ [m/s]")
    ax.set_ylabel("Feasible rate [-]")

    # Tight, readable bounds with small margins.
    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    x_pad = 0.03 * (x_max - x_min) if x_max > x_min else 0.01
    y_pad = 0.05 * (y_max - y_min) if y_max > y_min else 0.02
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(max(0.0, y_min - y_pad), min(1.05, y_max + y_pad))

    # Legend: best marker + span-size note + rank-color note.
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor=BEST_COLOR,
            markeredgecolor="black",
            markeredgewidth=0.9,
            markersize=8,
            label="Best after rerun",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            markerfacecolor=(0.8, 0.8, 0.8, 1.0),
            markeredgecolor="black",
            markeredgewidth=0.45,
            markersize=6,
            linewidth=0,
            label="Marker size proportional to wing_span_m",
        ),
    ]
    leg = ax.legend(
        handles=legend_handles,
        loc="lower right",
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
    df = load_top_rerun_df(INPUT_XLSX)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_top_rerun_frontier(df=df, out_path=OUT_PATH)

    best_count = int(df["is_best"].sum())
    print(f"Source sheet used: {SHEET_NAME}")
    print(f"Rows plotted: {len(df)}")
    print(f"Best rows: {best_count}")
    if best_count > 0:
        best_df = df[df["is_best"]]
        print(
            "Best source_sweep_run_index: "
            + ", ".join(str(int(v)) for v in best_df["source_sweep_run_index"].tolist())
        )
    print(f"Saved: {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()

