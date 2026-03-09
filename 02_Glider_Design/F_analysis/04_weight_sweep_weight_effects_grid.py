"""
Plot weight-effect sensitivity panels for the 6D objective-weight sweep.

Input workbook:
    C_results/weight_sweep_iter5.xlsx
Sheet:
    WeightSweep

Output figure:
    B_figures/04_weight_sweep_weight_effects_grid.png
"""

###### Initialization

### Imports
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:
    import cmocean
except ImportError:
    cmocean = None


### User settings
INPUT_XLSX = Path("C_results/weight_sweep_iter5.xlsx")
SHEET_NAME = "WeightSweep"
OUT_DIR = Path("B_figures")
OUT_PATH = OUT_DIR / "04_weight_sweep_weight_effects_grid.png"

WEIGHT_COLUMNS = [
    "log10_objective_weight_w_sink",
    "log10_objective_weight_w_mass",
    "log10_objective_weight_w_trim_effort",
    "log10_objective_weight_w_wing_deflection",
    "log10_objective_weight_w_htail_deflection",
    "log10_objective_weight_w_roll_tau",
]

WEIGHT_LABELS = {
    "log10_objective_weight_w_sink": "log10(w_sink)",
    "log10_objective_weight_w_mass": "log10(w_mass)",
    "log10_objective_weight_w_trim_effort": "log10(w_trim)",
    "log10_objective_weight_w_wing_deflection": "log10(w_wing_def)",
    "log10_objective_weight_w_htail_deflection": "log10(w_htail_def)",
    "log10_objective_weight_w_roll_tau": "log10(w_roll_tau)",
}

METRICS = [
    ("feasible_rate", "Feasible rate [-]"),
    ("sink_cvar_20", r"Sink CVaR$_{20}$ [m/s]"),
]

REQUIRED_COLUMNS = WEIGHT_COLUMNS + [
    "feasible_rate",
    "sink_cvar_20",
    "is_best",
]

# Plot style aligned with F_analysis references
AXIS_EDGE_LW = 0.80
GRID_COLOR = (0.85, 0.85, 0.85, 1.0)
GRID_LINEWIDTH = 0.4

POINT_SIZE = 18
POINT_ALPHA = 0.75
TREND_LW = 1.1
TREND_COLOR = (0.10, 0.10, 0.10, 0.95)
IQR_ALPHA = 0.10

BEST_MARKER_SIZE = 80
BEST_FACE_COLOR = "#ffd200"

N_BINS = 10


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


def load_data(xlsx_path: Path) -> pd.DataFrame:
    """Load WeightSweep data and enforce numeric/bool dtypes used in plots."""
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Workbook not found: {xlsx_path}")

    xls = pd.ExcelFile(xlsx_path)
    if SHEET_NAME not in xls.sheet_names:
        raise KeyError(f"Sheet '{SHEET_NAME}' not found in {xlsx_path}.")

    df = pd.read_excel(xlsx_path, sheet_name=SHEET_NAME)
    validate_columns(df)

    df = df.copy()
    for col in WEIGHT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["feasible_rate"] = pd.to_numeric(df["feasible_rate"], errors="coerce")
    df["sink_cvar_20"] = pd.to_numeric(df["sink_cvar_20"], errors="coerce")
    df["is_best"] = coerce_bool_like(df["is_best"])

    if "success" in df.columns:
        # Preserve bool if possible for discrete fallback coloring.
        try:
            df["success"] = coerce_bool_like(df["success"])
        except Exception:
            pass

    return df


def choose_coloring(df: pd.DataFrame) -> Dict[str, object]:
    """
    Choose coloring priority:
      1) discrete by trade_label (if present and non-empty)
      2) discrete by success (if present)
      3) continuous by feasible_rate
    """
    if "trade_label" in df.columns:
        labels = df["trade_label"].astype(str).str.strip()
        valid = labels.replace("", np.nan).dropna()
        if not valid.empty:
            cats = sorted(valid.unique().tolist())
            palette = [
                "#4c78a8",
                "#cc4a74",
                "#ffa600",
                "#54a24b",
                "#72b7b2",
                "#9d755d",
                "#b279a2",
                "#f58518",
            ]
            color_map = {c: palette[i % len(palette)] for i, c in enumerate(cats)}
            return {
                "mode": "discrete_trade_label",
                "series": labels,
                "categories": cats,
                "color_map": color_map,
                "label": "trade_label",
            }

    if "success" in df.columns:
        s = df["success"]
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
            s_bool = coerce_bool_like(s)
            color_map = {True: "#4c78a8", False: "#e45756"}
            return {
                "mode": "discrete_success",
                "series": s_bool,
                "categories": [True, False],
                "color_map": color_map,
                "label": "success",
            }

    values = df["feasible_rate"].to_numpy(dtype=float)
    cmap = cmocean.cm.thermal if cmocean is not None else plt.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=float(np.nanmin(values)), vmax=float(np.nanmax(values)))
    return {
        "mode": "continuous_feasible_rate",
        "values": values,
        "cmap": cmap,
        "norm": norm,
        "label": "Feasible rate [-]",
    }


def binned_stats(x: np.ndarray, y: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin x and compute median(y), q25(y), q75(y), median(x) per bin.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x_f = x[mask]
    y_f = y[mask]

    if x_f.size < 4:
        return np.array([]), np.array([]), np.array([]), np.array([])
    if np.isclose(float(np.nanmin(x_f)), float(np.nanmax(x_f))):
        return np.array([]), np.array([]), np.array([]), np.array([])

    x_series = pd.Series(x_f)
    y_series = pd.Series(y_f)

    bins = pd.cut(x_series, bins=n_bins, include_lowest=True, duplicates="drop")
    grouped = pd.DataFrame({"x": x_series, "y": y_series, "bin": bins}).groupby("bin", observed=False)

    x_med = grouped["x"].median()
    y_med = grouped["y"].median()
    y_q25 = grouped["y"].quantile(0.25)
    y_q75 = grouped["y"].quantile(0.75)

    data = pd.concat(
        [x_med.rename("x"), y_med.rename("med"), y_q25.rename("q25"), y_q75.rename("q75")],
        axis=1,
    ).dropna()

    if data.empty:
        return np.array([]), np.array([]), np.array([]), np.array([])

    data = data.sort_values("x")
    return (
        data["x"].to_numpy(dtype=float),
        data["med"].to_numpy(dtype=float),
        data["q25"].to_numpy(dtype=float),
        data["q75"].to_numpy(dtype=float),
    )


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID_COLOR, linewidth=GRID_LINEWIDTH)
    ax.tick_params(axis="both", which="major", length=2, width=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_EDGE_LW)


def draw_scatter(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    color_info: Dict[str, object],
) -> None:
    """Draw panel scatter using selected color mode."""
    mode = color_info["mode"]

    if mode in {"discrete_trade_label", "discrete_success"}:
        series = color_info["series"]
        categories = color_info["categories"]
        color_map = color_info["color_map"]
        for cat in categories:
            mask = (series == cat).to_numpy(dtype=bool)
            if not np.any(mask):
                continue
            ax.scatter(
                x[mask],
                y[mask],
                s=POINT_SIZE,
                color=color_map[cat],
                alpha=POINT_ALPHA,
                edgecolors="none",
                zorder=3,
            )
    else:
        values = color_info["values"]
        cmap = color_info["cmap"]
        norm = color_info["norm"]
        ax.scatter(
            x,
            y,
            c=values,
            cmap=cmap,
            norm=norm,
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            edgecolors="none",
            zorder=3,
        )

    best_mask = df["is_best"].to_numpy(dtype=bool)
    if np.any(best_mask):
        ax.scatter(
            x[best_mask],
            y[best_mask],
            marker="*",
            s=BEST_MARKER_SIZE,
            color=BEST_FACE_COLOR,
            edgecolors="black",
            linewidths=0.7,
            zorder=8,
        )


def add_trend(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    """Add binned median trend with IQR band."""
    x_med, y_med, y_q25, y_q75 = binned_stats(x=x, y=y, n_bins=N_BINS)
    if x_med.size == 0:
        return

    ax.fill_between(
        x_med,
        y_q25,
        y_q75,
        color=TREND_COLOR,
        alpha=IQR_ALPHA,
        edgecolor="none",
        zorder=4,
    )
    ax.plot(
        x_med,
        y_med,
        color=TREND_COLOR,
        linewidth=TREND_LW,
        linestyle="-",
        zorder=5,
    )


def build_legend_handles(color_info: Dict[str, object]) -> List[object]:
    """Build legend handles for color encoding + best marker + trend."""
    handles: List[object] = []

    mode = color_info["mode"]
    if mode in {"discrete_trade_label", "discrete_success"}:
        categories = color_info["categories"]
        color_map = color_info["color_map"]
        for cat in categories:
            handles.append(
                Patch(facecolor=color_map[cat], edgecolor="none", label=f"{color_info['label']}={cat}")
            )

    handles.append(
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor=BEST_FACE_COLOR,
            markeredgecolor="black",
            markeredgewidth=0.7,
            markersize=8,
            label="Best (is_best=True)",
        )
    )
    handles.append(
        Line2D(
            [0],
            [0],
            color=TREND_COLOR,
            linewidth=TREND_LW,
            label="Binned median trend",
        )
    )
    return handles


def plot_weight_effect_grid(df: pd.DataFrame, out_path: Path) -> None:
    """Create 2x6 grid: each weight vs feasible_rate and sink_cvar_20."""
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.edgecolor": "k",
            "axes.linewidth": AXIS_EDGE_LW,
        }
    )

    color_info = choose_coloring(df)

    fig, axes = plt.subplots(2, 6, figsize=(14.4, 5.6), dpi=600, sharey="row")
    fig.patch.set_facecolor("white")

    for col_idx, weight_col in enumerate(WEIGHT_COLUMNS):
        x = df[weight_col].to_numpy(dtype=float)

        for row_idx, (metric_col, y_label) in enumerate(METRICS):
            ax = axes[row_idx, col_idx]
            y = df[metric_col].to_numpy(dtype=float)

            style_axes(ax)
            draw_scatter(ax=ax, x=x, y=y, df=df, color_info=color_info)
            add_trend(ax=ax, x=x, y=y)

            if row_idx == 0:
                ax.set_title(WEIGHT_LABELS[weight_col], pad=4)

            if col_idx == 0:
                ax.set_ylabel(y_label)

            if row_idx == 1:
                ax.set_xlabel(WEIGHT_LABELS[weight_col])

    # Add legend and colorbar once for entire figure.
    handles = build_legend_handles(color_info)
    leg = fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.07),
        ncol=min(5, len(handles)),
        frameon=True,
        framealpha=1.0,
        edgecolor="black",
        fontsize=8.5,
        handlelength=1.5,
        borderpad=0.5,
        labelspacing=0.2,
        columnspacing=0.8,
    )
    if leg is not None:
        leg.get_frame().set_linewidth(AXIS_EDGE_LW)

    if color_info["mode"] == "continuous_feasible_rate":
        sm = ScalarMappable(norm=color_info["norm"], cmap=color_info["cmap"])
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), pad=0.01, fraction=0.02)
        cbar.set_label("Feasible rate [-]")
        cbar.ax.tick_params(width=0.6, length=2)
        cbar.outline.set_linewidth(AXIS_EDGE_LW)

    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


### Entrypoint
def main() -> None:
    df = load_data(INPUT_XLSX)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_weight_effect_grid(df=df, out_path=OUT_PATH)

    color_mode = choose_coloring(df)["mode"]
    print(f"Source sheet used: {SHEET_NAME}")
    print(f"Rows plotted: {len(df)}")
    print(f"Best rows: {int(df['is_best'].sum())}")
    print(f"Color mode: {color_mode}")
    print(f"Saved: {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()

