"""
Plot robustness success/failure map in uncertainty space per candidate.

Input workbook:
    C_results/nausicaa_workflow_iter3.xlsx
Sheet:
    PlotDataRobust

Output figure:
    B_figures/10_nausicaa_workflow_robust_uncertainty_map.png
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
OUT_PATH = OUT_DIR / "10_nausicaa_workflow_robust_uncertainty_map.png"

REQUIRED_COLUMNS = [
    "candidate_id",
    "scenario_id",
    "scenario_tag",
    "nom_success",
    "mass_scale",
    "cg_x_shift_mac",
]

# Plot style aligned with F_analysis references
AXIS_EDGE_LW = 0.80
GRID_COLOR = (0.85, 0.85, 0.85, 1.0)
GRID_LINEWIDTH = 0.4

COLOR_SUCCESS = "#4c78a8"
COLOR_FAILURE = "#e45756"

MARKER_CYCLE = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h"]


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
    df["scenario_id"] = pd.to_numeric(df["scenario_id"], errors="coerce")
    df["scenario_tag"] = df["scenario_tag"].astype(str)
    df["nom_success"] = coerce_bool_like(df["nom_success"])
    df["mass_scale"] = pd.to_numeric(df["mass_scale"], errors="raise")
    df["cg_x_shift_mac"] = pd.to_numeric(df["cg_x_shift_mac"], errors="raise")

    return df.sort_values(by=["candidate_id", "scenario_id"], ascending=[True, True]).reset_index(drop=True)


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID_COLOR, linewidth=GRID_LINEWIDTH)
    ax.tick_params(axis="both", which="major", length=2, width=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_EDGE_LW)


def scenario_marker_map(tags: List[str]) -> Dict[str, str]:
    """Map each scenario_tag to a marker symbol."""
    out: Dict[str, str] = {}
    for i, t in enumerate(tags):
        out[t] = MARKER_CYCLE[i % len(MARKER_CYCLE)]
    return out


def plot_robust_uncertainty_map(df: pd.DataFrame, out_path: Path) -> None:
    """
    Faceted scatter by candidate:
      x = cg_x_shift_mac,
      y = mass_scale,
      color = nom_success,
      marker shape = scenario_tag.
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

    candidate_ids = sorted(df["candidate_id"].unique().tolist())
    tags = sorted(df["scenario_tag"].unique().tolist())
    marker_map = scenario_marker_map(tags)

    n_cands = len(candidate_ids)
    fig_w = max(5.7, 3.15 * n_cands)
    fig, axes = plt.subplots(1, n_cands, figsize=(fig_w, 3.9), dpi=600, sharex=True, sharey=True)
    fig.patch.set_facecolor("white")

    if n_cands == 1:
        axes = [axes]

    x_all = df["cg_x_shift_mac"].to_numpy(dtype=float)
    y_all = df["mass_scale"].to_numpy(dtype=float)

    x_min, x_max = float(np.nanmin(x_all)), float(np.nanmax(x_all))
    y_min, y_max = float(np.nanmin(y_all)), float(np.nanmax(y_all))

    x_span = max(1e-12, x_max - x_min)
    y_span = max(1e-12, y_max - y_min)

    x_pad = 0.08 * x_span
    y_pad = 0.08 * y_span

    for ax, cid in zip(axes, candidate_ids):
        style_axes(ax)

        d = df[df["candidate_id"] == cid].copy()

        for tag in tags:
            dt = d[d["scenario_tag"] == tag]
            if dt.empty:
                continue

            success_mask = dt["nom_success"].to_numpy(dtype=bool)
            x = dt["cg_x_shift_mac"].to_numpy(dtype=float)
            y = dt["mass_scale"].to_numpy(dtype=float)
            mk = marker_map[tag]

            if np.any(success_mask):
                ax.scatter(
                    x[success_mask],
                    y[success_mask],
                    marker=mk,
                    s=32,
                    color=COLOR_SUCCESS,
                    alpha=0.85,
                    edgecolors="black",
                    linewidths=0.35,
                    zorder=4,
                )

            if np.any(~success_mask):
                ax.scatter(
                    x[~success_mask],
                    y[~success_mask],
                    marker=mk,
                    s=38,
                    color=COLOR_FAILURE,
                    alpha=0.95,
                    edgecolors="black",
                    linewidths=0.45,
                    zorder=6,
                )

        n_total = len(d)
        n_success = int(d["nom_success"].sum())
        n_fail = n_total - n_success
        ax.set_title(f"Candidate {cid}\n(n={n_total}, succ={n_success}, fail={n_fail})", pad=5)

        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_xlabel("CG shift [MAC]")

    axes[0].set_ylabel("Mass scale [-]")

    # Combined legend: success/failure colors + scenario_tag shapes.
    success_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLOR_SUCCESS,
            markeredgecolor="black",
            markeredgewidth=0.35,
            markersize=6,
            label="nom_success=True",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLOR_FAILURE,
            markeredgecolor="black",
            markeredgewidth=0.45,
            markersize=6,
            label="nom_success=False",
        ),
    ]

    tag_handles = []
    for tag in tags:
        tag_handles.append(
            Line2D(
                [0],
                [0],
                marker=marker_map[tag],
                color="black",
                markerfacecolor=(0.8, 0.8, 0.8, 1.0),
                markeredgecolor="black",
                markeredgewidth=0.4,
                linewidth=0,
                markersize=6,
                label=f"scenario_tag={tag}",
            )
        )

    legend_handles = success_handles + tag_handles
    leg = fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
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

    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


### Entrypoint
def main() -> None:
    df = load_plot_data(INPUT_XLSX)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_robust_uncertainty_map(df=df, out_path=OUT_PATH)

    print(f"Source sheet used: {SHEET_NAME}")
    print(f"Rows plotted: {len(df)}")
    print(f"Candidates: {sorted(df['candidate_id'].unique().tolist())}")
    print(
        "Success count: "
        f"{int(df['nom_success'].sum())}, failures: {int((~df['nom_success']).sum())}"
    )
    print(f"Saved: {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()

