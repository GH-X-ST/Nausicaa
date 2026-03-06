"""
Plot objective-term composition across the weight sweep.

Input workbook:
    C_results/weight_sweep_iter3.xlsx
Preferred sheet:
    TradeStudy
Fallback sheet:
    WeightSweep (sorted by selection logic)

Output figure:
    B_figures/03_weight_sweep_objective_term_shares.png
"""

###### Initialization

### Imports
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter


### User settings
INPUT_XLSX = Path("C_results/weight_sweep_iter3.xlsx")
OUT_DIR = Path("B_figures")
OUT_PATH = OUT_DIR / "03_weight_sweep_objective_term_shares.png"

PREFERRED_SHEET = "TradeStudy"
FALLBACK_SHEET = "WeightSweep"

# Use normalized shares by default (recommended for explanation)
USE_NORMALIZED_SHARES = True

TERM_COLUMNS = [
    "J_sink",
    "J_mass",
    "J_trim",
    "J_wing_deflection",
    "J_htail_deflection",
    "J_roll_tau",
]

REQUIRED_COLUMNS = [
    "run_index",
    "is_best",
    "J_total",
] + TERM_COLUMNS

OPTIONAL_OVERLAY_COLUMN = "objective_term_spread_ratio"

TERM_LABELS = {
    "J_sink": "J_sink",
    "J_mass": "J_mass",
    "J_trim": "J_trim",
    "J_wing_deflection": "J_wing_deflection",
    "J_htail_deflection": "J_htail_deflection",
    "J_roll_tau": "J_roll_tau",
}

TERM_COLORS = {
    "J_sink": "#4c78a8",
    "J_mass": "#f58518",
    "J_trim": "#54a24b",
    "J_wing_deflection": "#e45756",
    "J_htail_deflection": "#72b7b2",
    "J_roll_tau": "#9d755d",
}

# Style aligned with F_analysis references
AXIS_EDGE_LW = 0.80
GRID_COLOR = (0.85, 0.85, 0.85, 1.0)
GRID_LINEWIDTH = 0.4


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


def validate_required_columns(df: pd.DataFrame, sheet_name: str) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in '{sheet_name}': {missing}")


def load_trade_study_frame(xlsx_path: Path) -> Tuple[pd.DataFrame, str]:
    """
    Load TradeStudy if available; otherwise WeightSweep sorted by selection logic.
    """
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Workbook not found: {xlsx_path}")

    xls = pd.ExcelFile(xlsx_path)
    sheet_names = xls.sheet_names

    if PREFERRED_SHEET in sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=PREFERRED_SHEET)
        source_sheet = PREFERRED_SHEET
        needs_sort = False
    elif FALLBACK_SHEET in sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=FALLBACK_SHEET)
        source_sheet = FALLBACK_SHEET
        needs_sort = True
    else:
        raise KeyError(
            f"Neither '{PREFERRED_SHEET}' nor '{FALLBACK_SHEET}' found in {xlsx_path}."
        )

    validate_required_columns(df, source_sheet)

    df = df.copy()
    df["run_index"] = pd.to_numeric(df["run_index"], errors="raise").astype(int)
    df["is_best"] = coerce_bool_like(df["is_best"])
    df["J_total"] = pd.to_numeric(df["J_total"], errors="coerce")

    for col in TERM_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if OPTIONAL_OVERLAY_COLUMN in df.columns:
        df[OPTIONAL_OVERLAY_COLUMN] = pd.to_numeric(df[OPTIONAL_OVERLAY_COLUMN], errors="coerce")

    if needs_sort:
        sort_cols = ["feasible_rate", "sink_cvar_20", "wing_span_m", "run_index"]
        missing_sort = [c for c in sort_cols if c not in df.columns]
        if missing_sort:
            raise KeyError(
                "Fallback sorting requires columns missing from WeightSweep: "
                f"{missing_sort}"
            )
        df["feasible_rate"] = pd.to_numeric(df["feasible_rate"], errors="coerce")
        df["sink_cvar_20"] = pd.to_numeric(df["sink_cvar_20"], errors="coerce")
        df["wing_span_m"] = pd.to_numeric(df["wing_span_m"], errors="coerce")

        df = df.sort_values(
            by=sort_cols,
            ascending=[False, True, True, True],
            kind="mergesort",
        )

    return df.reset_index(drop=True), source_sheet


def prepare_term_values(df: pd.DataFrame, use_shares: bool) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Build plotted term table. If normalized, guard against small-total divide issues.
    """
    values = df[TERM_COLUMNS].copy()
    values = values.fillna(0.0)
    sum_terms = values.sum(axis=1).to_numpy(dtype=float)

    if not use_shares:
        return values, sum_terms

    j_total = df["J_total"].to_numpy(dtype=float)
    eps = 1e-12

    # Prefer J_total but fall back to sum of terms when J_total is too small/noisy.
    denom = np.where(np.isfinite(j_total) & (np.abs(j_total) > eps), j_total, sum_terms)
    denom = np.where(np.abs(denom) > eps, denom, np.nan)

    shares = values.to_numpy(dtype=float) / denom[:, None]
    shares = np.where(np.isfinite(shares), shares, 0.0)

    share_df = pd.DataFrame(shares, columns=TERM_COLUMNS, index=df.index)
    total_share = share_df.sum(axis=1).to_numpy(dtype=float)
    return share_df, total_share


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=GRID_LINEWIDTH)
    ax.tick_params(axis="both", which="major", length=2, width=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_EDGE_LW)


def add_best_annotation(ax: plt.Axes, x_best: int, y_best: float, n_rows: int) -> None:
    """Annotate the selected best bar."""
    x_offset = -5 if x_best >= int(0.75 * n_rows) else 1
    y_top = ax.get_ylim()[1]
    y_text = min(y_top * 0.98, y_best + 0.06 * y_top)
    ax.annotate(
        "Best (is_best=True)",
        xy=(x_best, y_best),
        xytext=(x_best + x_offset, y_text),
        fontsize=9,
        textcoords="data",
        ha="left" if x_offset > 0 else "right",
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


def plot_objective_term_stack(df: pd.DataFrame, out_path: Path, use_shares: bool) -> None:
    """
    Plot stacked objective terms per run and highlight best row.
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

    values_df, total_vals = prepare_term_values(df=df, use_shares=use_shares)
    x = np.arange(len(df), dtype=int)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=600)
    fig.patch.set_facecolor("white")
    style_axes(ax)

    bottoms = np.zeros(len(df), dtype=float)
    for col in TERM_COLUMNS:
        vals = values_df[col].to_numpy(dtype=float)
        ax.bar(
            x,
            vals,
            width=0.84,
            bottom=bottoms,
            color=TERM_COLORS[col],
            edgecolor="none",
            alpha=0.90,
            label=TERM_LABELS[col],
            zorder=3,
        )
        bottoms += vals

    best_idx = np.flatnonzero(df["is_best"].to_numpy(dtype=bool))
    for i_best in best_idx:
        ax.bar(
            int(i_best),
            float(total_vals[i_best]),
            width=0.84,
            bottom=0.0,
            fill=False,
            edgecolor="black",
            linewidth=1.2,
            hatch="///",
            zorder=9,
        )

    if use_shares:
        y_max = max(1.0, float(np.nanmax(total_vals)) * 1.06)
        ax.set_ylim(0.0, y_max)
        ax.set_ylabel("Objective share [-]")
    else:
        y_max = float(np.nanmax(total_vals)) * 1.10 if np.nanmax(total_vals) > 0 else 1.0
        ax.set_ylim(0.0, y_max)
        ax.set_ylabel("Objective term value [-]")

    ax.set_xlabel("TradeStudy order (lexicographic selection)")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    n_rows = len(df)
    tick_step = max(1, int(np.ceil(n_rows / 16)))
    xticks = np.arange(0, n_rows, tick_step, dtype=int)
    ax.set_xticks(xticks)
    ax.set_xlim(-0.5, n_rows - 0.5)

    if best_idx.size > 0:
        i_best = int(best_idx[0])
        add_best_annotation(ax=ax, x_best=i_best, y_best=float(total_vals[i_best]), n_rows=n_rows)

    spread_added = False
    line_handle = None
    if use_shares and OPTIONAL_OVERLAY_COLUMN in df.columns:
        spread = df[OPTIONAL_OVERLAY_COLUMN].to_numpy(dtype=float)
        if np.isfinite(spread).any():
            ax2 = ax.twinx()
            ax2.plot(
                x,
                spread,
                color=(0.15, 0.15, 0.15, 0.70),
                linewidth=1.0,
                linestyle="--",
                marker="o",
                markersize=2.2,
                zorder=6,
            )
            ax2.set_ylabel("Objective term spread ratio [-]")
            ax2.tick_params(axis="y", which="major", length=2, width=0.6)
            for spine in ax2.spines.values():
                spine.set_linewidth(AXIS_EDGE_LW)
            spread_added = True
            line_handle = Line2D(
                [0],
                [0],
                color=(0.15, 0.15, 0.15, 0.70),
                linewidth=1.0,
                linestyle="--",
                marker="o",
                markersize=3.0,
                label="objective_term_spread_ratio",
            )

    legend_handles = [
        Patch(facecolor=TERM_COLORS[col], edgecolor="none", label=TERM_LABELS[col])
        for col in TERM_COLUMNS
    ]
    legend_handles.append(
        Patch(
            facecolor=(1.0, 1.0, 1.0, 0.0),
            edgecolor="black",
            linewidth=1.2,
            hatch="///",
            label="Best (is_best=True)",
        )
    )
    if spread_added and line_handle is not None:
        legend_handles.append(line_handle)

    leg = ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=4,
        frameon=True,
        framealpha=1.0,
        edgecolor="black",
        fontsize=8.5,
        handlelength=1.5,
        borderpad=0.5,
        labelspacing=0.2,
        columnspacing=0.9,
    )
    if leg is not None:
        leg.get_frame().set_linewidth(AXIS_EDGE_LW)

    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


### Entrypoint
def main() -> None:
    df, source_sheet = load_trade_study_frame(INPUT_XLSX)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_objective_term_stack(df=df, out_path=OUT_PATH, use_shares=USE_NORMALIZED_SHARES)

    print(f"Source sheet used: {source_sheet}")
    print(f"Rows plotted: {len(df)}")
    print(f"Best rows: {int(df['is_best'].sum())}")
    print(f"Mode: {'normalized shares' if USE_NORMALIZED_SHARES else 'absolute terms'}")
    print(f"Saved: {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()

