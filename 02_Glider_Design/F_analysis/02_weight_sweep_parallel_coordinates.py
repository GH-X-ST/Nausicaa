"""
Plot the 6D objective-weight sweep as parallel coordinates.

Input workbook:
    C_results/weight_sweep_iter5.xlsx
Preferred sheet:
    WeightSweep
Fallback sheet:
    TradeStudy

Output figure:
    B_figures/02_weight_sweep_parallel_coordinates.png
"""

###### Initialization

### Imports
from pathlib import Path
from typing import List, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

try:
    import cmocean
except ImportError:
    cmocean = None


### User settings
INPUT_XLSX = Path("C_results/weight_sweep_iter5.xlsx")
OUT_DIR = Path("B_figures")
OUT_PATH = OUT_DIR / "02_weight_sweep_parallel_coordinates.png"

PREFERRED_SHEET = "WeightSweep"
FALLBACK_SHEET = "TradeStudy"

WEIGHT_COLUMNS = [
    "log10_objective_weight_w_sink",
    "log10_objective_weight_w_mass",
    "log10_objective_weight_w_trim_effort",
    "log10_objective_weight_w_wing_deflection",
    "log10_objective_weight_w_htail_deflection",
    "log10_objective_weight_w_roll_tau",
]

REQUIRED_COLUMNS = WEIGHT_COLUMNS + [
    "run_index",
    "feasible_rate",
    "sink_cvar_20",
    "is_best",
]

# Labels follow the objective-weight order
XTICK_LABELS = [
    "log10(w_sink)",
    "log10(w_mass)",
    "log10(w_trim)",
    "log10(w_wing_def)",
    "log10(w_htail_def)",
    "log10(w_roll_tau)",
]

# Plot style aligned with existing F_analysis scripts
AXIS_EDGE_LW = 0.80
GRID_COLOR = (0.85, 0.85, 0.85, 1.0)
GRID_LINEWIDTH = 0.4

NON_BEST_ALPHA = 0.20
NON_BEST_LW = 0.8
BEST_LW_OUTLINE = 3.0
BEST_LW_MAIN = 2.1


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


def choose_sheet(xlsx_path: Path) -> Tuple[pd.DataFrame, str]:
    """
    Load WeightSweep if available; otherwise TradeStudy.
    """
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Workbook not found: {xlsx_path}")

    xls = pd.ExcelFile(xlsx_path)
    sheet_names = xls.sheet_names

    if PREFERRED_SHEET in sheet_names:
        sheet_name = PREFERRED_SHEET
    elif FALLBACK_SHEET in sheet_names:
        sheet_name = FALLBACK_SHEET
    else:
        raise KeyError(
            f"Neither '{PREFERRED_SHEET}' nor '{FALLBACK_SHEET}' found in {xlsx_path}."
        )

    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    validate_required_columns(df, sheet_name)

    df = df.copy()
    df["run_index"] = pd.to_numeric(df["run_index"], errors="raise").astype(int)
    for col in WEIGHT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="raise")
    df["feasible_rate"] = pd.to_numeric(df["feasible_rate"], errors="raise")
    df["sink_cvar_20"] = pd.to_numeric(df["sink_cvar_20"], errors="raise")
    df["is_best"] = coerce_bool_like(df["is_best"])

    return df, sheet_name


def sorted_for_drawing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Draw low-feasibility / high-CVaR runs first so stronger runs are less occluded.
    """
    return df.sort_values(
        by=["feasible_rate", "sink_cvar_20", "run_index"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def axis_limits(values: np.ndarray) -> Tuple[float, float]:
    """
    Build robust axis limits with small padding.
    """
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if np.isclose(vmin, vmax):
        delta = max(0.10, 0.05 * max(1.0, abs(vmin)))
        return vmin - delta, vmax + delta
    pad = 0.04 * (vmax - vmin)
    return vmin - pad, vmax + pad


def setup_parallel_axes(
    fig: plt.Figure,
    mins: np.ndarray,
    maxs: np.ndarray,
    x: np.ndarray,
) -> Tuple[plt.Axes, List[plt.Axes]]:
    """
    Create one host axis + extra y-axes, one vertical scale per dimension.
    """
    host = fig.add_subplot(111)
    host.set_facecolor("white")
    host.set_xlim(float(x[0]), float(x[-1]))
    host.set_ylim(0.0, 1.0)
    host.set_axisbelow(True)
    host.grid(True, axis="y", color=GRID_COLOR, linewidth=GRID_LINEWIDTH)

    for xi in x:
        host.axvline(x=xi, color=GRID_COLOR, linewidth=GRID_LINEWIDTH, zorder=0)

    host.tick_params(axis="x", which="major", length=2, width=0.6, pad=3)
    host.tick_params(axis="y", which="major", length=2, width=0.6, pad=2)
    for spine in host.spines.values():
        spine.set_linewidth(AXIS_EDGE_LW)

    dim0_ticks = np.linspace(mins[0], maxs[0], 5)
    dim0_norm_ticks = (dim0_ticks - mins[0]) / (maxs[0] - mins[0])
    host.set_yticks(dim0_norm_ticks)
    host.set_yticklabels([f"{v:.2f}" for v in dim0_ticks])
    host.set_ylabel("log10(weight)")

    axes = [host]
    for i in range(1, x.size):
        ax = host.twinx()
        ax.set_ylim(float(mins[i]), float(maxs[i]))
        ax.spines["right"].set_position(("data", float(x[i])))
        ax.spines["right"].set_linewidth(AXIS_EDGE_LW)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.patch.set_alpha(0.0)

        yticks = np.linspace(mins[i], maxs[i], 5)
        ax.set_yticks(yticks)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.tick_params(axis="y", which="major", length=2, width=0.6, pad=2, labelsize=8)
        axes.append(ax)

    return host, axes


def normalize_values(values: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    """
    Normalize each dimension to [0, 1] using axis limits.
    """
    spans = maxs - mins
    return (values - mins) / spans


def plot_parallel_coordinates(df: pd.DataFrame, out_path: Path) -> None:
    """
    Plot 6D weight settings as parallel coordinates with feasible-rate coloring.
    """
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 8,
            "axes.edgecolor": "k",
            "axes.linewidth": AXIS_EDGE_LW,
        }
    )

    x = np.arange(len(WEIGHT_COLUMNS), dtype=float)

    mins = []
    maxs = []
    for col in WEIGHT_COLUMNS:
        lo, hi = axis_limits(df[col].to_numpy(dtype=float))
        mins.append(lo)
        maxs.append(hi)
    mins_arr = np.asarray(mins, dtype=float)
    maxs_arr = np.asarray(maxs, dtype=float)

    values = df[WEIGHT_COLUMNS].to_numpy(dtype=float)
    values_norm = normalize_values(values, mins_arr, maxs_arr)

    feasible = df["feasible_rate"].to_numpy(dtype=float)
    best_mask = df["is_best"].to_numpy(dtype=bool)

    cmap = cmocean.cm.thermal if cmocean is not None else plt.get_cmap("viridis")
    color_norm = mcolors.Normalize(
        vmin=float(np.nanmin(feasible)),
        vmax=float(np.nanmax(feasible)),
    )

    fig = plt.figure(figsize=(7.4, 4.3), dpi=600)
    fig.patch.set_facecolor("white")
    host, _ = setup_parallel_axes(fig=fig, mins=mins_arr, maxs=maxs_arr, x=x)

    host.set_xticks(x)
    host.set_xticklabels(XTICK_LABELS, rotation=-18, ha="left")

    # Non-best runs
    for i in range(len(df)):
        if best_mask[i]:
            continue
        line_color = cmap(color_norm(feasible[i]))
        host.plot(
            x,
            values_norm[i, :],
            color=line_color,
            alpha=NON_BEST_ALPHA,
            linewidth=NON_BEST_LW,
            zorder=2,
        )

    # Best run(s), drawn last with strong contrast
    best_indices = np.flatnonzero(best_mask)
    for i in best_indices:
        line_color = cmap(color_norm(feasible[i]))
        host.plot(
            x,
            values_norm[i, :],
            color=(0.0, 0.0, 0.0, 0.95),
            linewidth=BEST_LW_OUTLINE,
            zorder=8,
        )
        host.plot(
            x,
            values_norm[i, :],
            color=line_color,
            linewidth=BEST_LW_MAIN,
            alpha=1.0,
            zorder=9,
        )
        host.scatter(
            x,
            values_norm[i, :],
            s=16,
            color=line_color,
            edgecolors="black",
            linewidths=0.5,
            zorder=10,
        )

    if best_indices.size > 0:
        i_best = int(best_indices[0])
        y_best = float(values_norm[i_best, -1])
        y_text = min(0.98, y_best + 0.09)
        run_index = int(df.loc[i_best, "run_index"])
        host.annotate(
            "Best (is_best=True)",
            xy=(x[-1], y_best),
            xytext=(x[-1] - 0.35, y_text),
            textcoords="data",
            fontsize=9,
            ha="right",
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
        host.text(
            0.01,
            0.98,
            (
                f"best run_index={run_index}, "
                f"feasible_rate={df.loc[i_best, 'feasible_rate']:.3f}, "
                f"sink_cvar_20={df.loc[i_best, 'sink_cvar_20']:.3f} m/s"
            ),
            transform=host.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": (1.0, 1.0, 1.0, 0.8),
                "edgecolor": "none",
            },
            zorder=20,
        )

    sm = ScalarMappable(norm=color_norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=host, pad=0.02, fraction=0.05)
    cbar.set_label("Feasible rate [-]")
    cbar.ax.tick_params(width=0.6, length=2)
    cbar.outline.set_linewidth(AXIS_EDGE_LW)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=(0.2, 0.2, 0.2, 0.35),
            linewidth=1.2,
            label="Weight-sweep runs",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=2.2,
            label="Best (is_best=True)",
        ),
    ]
    leg = host.legend(
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
    df, source_sheet = choose_sheet(INPUT_XLSX)
    df = sorted_for_drawing(df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_parallel_coordinates(df=df, out_path=OUT_PATH)

    print(f"Source sheet used: {source_sheet}")
    print(f"Rows plotted: {len(df)}")
    print(f"Best rows: {int(df['is_best'].sum())}")
    print(f"Saved: {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()

