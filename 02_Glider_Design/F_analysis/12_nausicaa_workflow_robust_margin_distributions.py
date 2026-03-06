"""
Plot robust-margin distributions per candidate under uncertainty.

Input workbook:
    C_results/nausicaa_workflow_iter3.xlsx
Sheet:
    PlotDataRobust

Output figure:
    B_figures/12_nausicaa_workflow_robust_margin_distributions.png
"""

###### Initialization

### Imports
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


### User settings
INPUT_XLSX = Path("C_results/nausicaa_workflow_iter3.xlsx")
SHEET_NAME = "PlotDataRobust"
OUT_DIR = Path("B_figures")
OUT_PATH = OUT_DIR / "12_nausicaa_workflow_robust_margin_distributions.png"

# Optional metrics (set True to include nom_util_a and nom_util_r panels if available)
INCLUDE_OPTIONAL_UTILS = False

REQUIRED_COLUMNS = [
    "candidate_id",
    "nom_success",
    "nom_alpha_margin_deg",
    "nom_cl_margin_to_cap",
    "nom_util_e",
    "nom_roll_tau_s",
]

OPTIONAL_UTIL_COLUMNS = ["nom_util_a", "nom_util_r"]

METRIC_LABELS = {
    "nom_alpha_margin_deg": "Nom alpha margin [deg]",
    "nom_cl_margin_to_cap": "Nom CL margin to cap [-]",
    "nom_util_e": "Nom elevator utilization [-]",
    "nom_roll_tau_s": "Nom roll tau [s]",
    "nom_util_a": "Nom aileron utilization [-]",
    "nom_util_r": "Nom rudder utilization [-]",
}

# Plot style aligned with F_analysis references
AXIS_EDGE_LW = 0.80
GRID_COLOR = (0.85, 0.85, 0.85, 1.0)
GRID_LINEWIDTH = 0.4

POINT_ALPHA = 0.65
POINT_SIZE = 18
BOX_ALPHA = 0.22

COLOR_BOUNDARY_MARGIN = "#e45756"
COLOR_BOUNDARY_UTIL = "#cc4a74"

PALETTE = ["#4c78a8", "#72b7b2", "#54a24b", "#f58518", "#9d755d", "#b279a2"]


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


def load_plot_data(xlsx_path: Path) -> pd.DataFrame:
    """Load PlotDataRobust and enforce required schema."""
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
    df["candidate_id"] = pd.to_numeric(df["candidate_id"], errors="raise").astype(int)
    df["nom_success"] = coerce_bool_like(df["nom_success"])

    numeric_cols = REQUIRED_COLUMNS[2:] + [c for c in OPTIONAL_UTIL_COLUMNS if c in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(by=["candidate_id"]).reset_index(drop=True)
    return df


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=GRID_LINEWIDTH)
    ax.tick_params(axis="both", which="major", length=2, width=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_EDGE_LW)


def candidate_color_map(candidate_ids: List[int]) -> Dict[int, str]:
    """Assign stable colors per candidate ID."""
    out: Dict[int, str] = {}
    for i, cid in enumerate(candidate_ids):
        out[cid] = PALETTE[i % len(PALETTE)]
    return out


def metric_boundaries(metric: str) -> List[tuple[float, str, str]]:
    """
    Return boundary lines as (y_value, label, color).
    """
    if metric in {"nom_alpha_margin_deg", "nom_cl_margin_to_cap"}:
        return [(0.0, "Boundary = 0", COLOR_BOUNDARY_MARGIN)]
    if metric in {"nom_util_e", "nom_util_a", "nom_util_r"}:
        return [(1.0, "Boundary = 1.0", COLOR_BOUNDARY_UTIL)]
    return []


def plot_metric_panel(
    ax: plt.Axes,
    d_success: pd.DataFrame,
    metric: str,
    candidate_ids: List[int],
    color_map: Dict[int, str],
) -> None:
    """Draw one box+jitter panel for a metric."""
    style_axes(ax)

    rng = np.random.default_rng(42)

    box_data = []
    positions = []
    box_colors = []

    for pos, cid in enumerate(candidate_ids, start=1):
        vals = d_success.loc[d_success["candidate_id"] == cid, metric].dropna().to_numpy(dtype=float)
        if vals.size > 0:
            box_data.append(vals)
            positions.append(pos)
            box_colors.append(color_map[cid])

    if box_data:
        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            boxprops={"edgecolor": "black", "linewidth": 0.8},
            whiskerprops={"color": "black", "linewidth": 0.8},
            capprops={"color": "black", "linewidth": 0.8},
            medianprops={"color": "#cc4a74", "linewidth": 1.2},
            flierprops={
                "marker": "o",
                "markersize": 2,
                "markerfacecolor": "black",
                "markeredgecolor": "black",
                "alpha": 0.18,
            },
            zorder=2,
        )
        for patch, c in zip(bp["boxes"], box_colors):
            patch.set_facecolor(c)
            patch.set_alpha(BOX_ALPHA)

    # Jittered points per candidate.
    for pos, cid in enumerate(candidate_ids, start=1):
        vals = d_success.loc[d_success["candidate_id"] == cid, metric].dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        x = np.full(vals.shape, float(pos)) + rng.uniform(-0.15, 0.15, size=vals.size)
        ax.scatter(
            x,
            vals,
            s=POINT_SIZE,
            color=color_map[cid],
            alpha=POINT_ALPHA,
            edgecolors="none",
            zorder=4,
        )

    # Boundary/reference lines.
    for y0, label, c in metric_boundaries(metric):
        ax.axhline(
            y=y0,
            color=c,
            linewidth=1.0,
            linestyle=(0, (3, 2)),
            zorder=3,
            label=label,
        )

    ax.set_title(METRIC_LABELS[metric], pad=5)
    ax.set_xlabel("Candidate ID")
    ax.set_xticks(np.arange(1, len(candidate_ids) + 1, dtype=int))
    ax.set_xticklabels([str(cid) for cid in candidate_ids], rotation=0)


def plot_margin_distributions(df: pd.DataFrame, out_path: Path) -> None:
    """
    Multi-panel robust metric distributions (success scenarios only).
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

    candidate_ids = sorted(df["candidate_id"].dropna().astype(int).unique().tolist())
    if not candidate_ids:
        raise ValueError("No candidate_id values found.")

    color_map = candidate_color_map(candidate_ids)

    metrics = [
        "nom_alpha_margin_deg",
        "nom_cl_margin_to_cap",
        "nom_util_e",
        "nom_roll_tau_s",
    ]
    if INCLUDE_OPTIONAL_UTILS:
        for c in OPTIONAL_UTIL_COLUMNS:
            if c in df.columns:
                metrics.append(c)

    d_success = df[df["nom_success"]].copy()

    n_panels = len(metrics)
    ncols = 2 if n_panels <= 4 else 3
    nrows = int(np.ceil(n_panels / ncols))

    fig_w = 5.6 * ncols
    fig_h = 3.7 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), dpi=600)
    fig.patch.set_facecolor("white")
    axes_arr = np.atleast_1d(axes).ravel()

    for i, metric in enumerate(metrics):
        ax = axes_arr[i]
        plot_metric_panel(
            ax=ax,
            d_success=d_success,
            metric=metric,
            candidate_ids=candidate_ids,
            color_map=color_map,
        )

    for j in range(n_panels, len(axes_arr)):
        axes_arr[j].axis("off")

    # Add a global note so filtering choice is explicit.
    n_total = len(df)
    n_success = int(df["nom_success"].sum())
    n_fail = n_total - n_success
    fig.text(
        0.01,
        0.995,
        (
            "Distributions use successful scenarios only "
            f"(success={n_success}, fail={n_fail}, total={n_total}). "
            "Failure visibility is handled in the uncertainty map figure."
        ),
        ha="left",
        va="top",
        fontsize=8.2,
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": (1.0, 1.0, 1.0, 0.78),
            "edgecolor": "none",
        },
    )

    # Combined legend for candidate colors and boundaries.
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color_map[cid],
            markeredgecolor="none",
            markersize=6,
            label=f"Candidate {cid}",
        )
        for cid in candidate_ids
    ]
    legend_handles += [
        Line2D(
            [0],
            [0],
            color=COLOR_BOUNDARY_MARGIN,
            linewidth=1.0,
            linestyle=(0, (3, 2)),
            label="Margin boundary (0)",
        ),
        Line2D(
            [0],
            [0],
            color=COLOR_BOUNDARY_UTIL,
            linewidth=1.0,
            linestyle=(0, (3, 2)),
            label="Utilization boundary (1.0)",
        ),
    ]

    leg = fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=min(6, len(legend_handles)),
        frameon=True,
        framealpha=1.0,
        edgecolor="black",
        fontsize=8.3,
        handlelength=1.5,
        borderpad=0.5,
        labelspacing=0.2,
        columnspacing=0.8,
    )
    if leg is not None:
        leg.get_frame().set_linewidth(AXIS_EDGE_LW)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


### Entrypoint
def main() -> None:
    df = load_plot_data(INPUT_XLSX)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_margin_distributions(df=df, out_path=OUT_PATH)

    n_total = len(df)
    n_success = int(df["nom_success"].sum())
    n_fail = n_total - n_success
    print(f"Source sheet used: {SHEET_NAME}")
    print(f"Rows plotted (input): {n_total}")
    print(f"Success rows used in distributions: {n_success}")
    print(f"Failure rows excluded from distributions: {n_fail}")
    print(f"Candidates: {sorted(df['candidate_id'].unique().tolist())}")
    print(f"Saved: {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()

