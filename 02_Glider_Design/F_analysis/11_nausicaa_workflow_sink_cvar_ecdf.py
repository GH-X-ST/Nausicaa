"""
Plot empirical CDF of nominal sink rate and mark mean/CVaR20/worst per candidate.

Input workbook:
    C_results/nausicaa_workflow_iter3.xlsx
Sheets:
    PlotDataRobust
    RobustSummary

Output figure:
    B_figures/11_nausicaa_workflow_sink_cvar_ecdf.png
"""

###### Initialization

### Imports
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


### User settings
INPUT_XLSX = Path("C_results/nausicaa_workflow_iter3.xlsx")
SHEET_PLOT = "PlotDataRobust"
SHEET_SUMMARY = "RobustSummary"
OUT_DIR = Path("B_figures")
OUT_PATH = OUT_DIR / "11_nausicaa_workflow_sink_cvar_ecdf.png"

REQUIRED_PLOT_COLUMNS = [
    "candidate_id",
    "nom_success",
    "nom_sink_rate_mps",
]

REQUIRED_SUMMARY_COLUMNS = [
    "candidate_id",
    "nom_sink_mean",
    "nom_sink_worst",
    "nom_sink_cvar_20",
]

# Plot style aligned with F_analysis references
AXIS_EDGE_LW = 0.80
GRID_COLOR = (0.85, 0.85, 0.85, 1.0)
GRID_LINEWIDTH = 0.4

CDF_COLORS = ["#4c78a8", "#72b7b2", "#54a24b", "#f58518", "#9d755d"]
MEAN_COLOR = "#222222"
CVAR_COLOR = "#cc4a74"
WORST_COLOR = "#e45756"


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


def validate_columns(df: pd.DataFrame, required: List[str], sheet_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in '{sheet_name}': {missing}")


def load_data(xlsx_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and type-check PlotDataRobust and RobustSummary."""
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Workbook not found: {xlsx_path}")

    xls = pd.ExcelFile(xlsx_path)
    if SHEET_PLOT not in xls.sheet_names:
        raise KeyError(f"Sheet '{SHEET_PLOT}' not found in {xlsx_path}.")
    if SHEET_SUMMARY not in xls.sheet_names:
        raise KeyError(f"Sheet '{SHEET_SUMMARY}' not found in {xlsx_path}.")

    df_plot = pd.read_excel(xlsx_path, sheet_name=SHEET_PLOT)
    df_summary = pd.read_excel(xlsx_path, sheet_name=SHEET_SUMMARY)

    validate_columns(df_plot, REQUIRED_PLOT_COLUMNS, SHEET_PLOT)
    validate_columns(df_summary, REQUIRED_SUMMARY_COLUMNS, SHEET_SUMMARY)

    df_plot = df_plot.copy()
    df_summary = df_summary.copy()

    df_plot["candidate_id"] = pd.to_numeric(df_plot["candidate_id"], errors="raise").astype(int)
    df_plot["nom_success"] = coerce_bool_like(df_plot["nom_success"])
    df_plot["nom_sink_rate_mps"] = pd.to_numeric(df_plot["nom_sink_rate_mps"], errors="coerce")

    df_summary["candidate_id"] = pd.to_numeric(df_summary["candidate_id"], errors="raise").astype(int)
    df_summary["nom_sink_mean"] = pd.to_numeric(df_summary["nom_sink_mean"], errors="coerce")
    df_summary["nom_sink_worst"] = pd.to_numeric(df_summary["nom_sink_worst"], errors="coerce")
    df_summary["nom_sink_cvar_20"] = pd.to_numeric(df_summary["nom_sink_cvar_20"], errors="coerce")

    return df_plot, df_summary


def empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute empirical CDF coordinates from 1D array."""
    x = np.sort(values)
    y = np.arange(1, x.size + 1, dtype=float) / float(x.size)
    return x, y


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID_COLOR, linewidth=GRID_LINEWIDTH)
    ax.tick_params(axis="both", which="major", length=2, width=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_EDGE_LW)


def plot_sink_ecdf(df_plot: pd.DataFrame, df_summary: pd.DataFrame, out_path: Path) -> None:
    """
    Plot per-candidate ECDF of successful nominal sink rates with summary vertical lines.
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

    candidate_ids = sorted(df_summary["candidate_id"].dropna().astype(int).unique().tolist())
    if not candidate_ids:
        raise ValueError("No candidate_id found in RobustSummary.")

    n = len(candidate_ids)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    fig_w = 5.4 * ncols
    fig_h = 3.9 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), dpi=600, sharey=True)
    fig.patch.set_facecolor("white")

    axes_arr = np.atleast_1d(axes).ravel()

    for idx, cid in enumerate(candidate_ids):
        ax = axes_arr[idx]
        style_axes(ax)

        d_success = df_plot[
            (df_plot["candidate_id"] == cid)
            & (df_plot["nom_success"] == True)
        ]
        vals = d_success["nom_sink_rate_mps"].dropna().to_numpy(dtype=float)

        d_all = df_plot[df_plot["candidate_id"] == cid]
        n_all = len(d_all)
        n_success = len(vals)

        row = df_summary[df_summary["candidate_id"] == cid]
        if row.empty:
            mean_v = np.nan
            cvar_v = np.nan
            worst_v = np.nan
        else:
            r0 = row.iloc[0]
            mean_v = float(r0["nom_sink_mean"])
            cvar_v = float(r0["nom_sink_cvar_20"])
            worst_v = float(r0["nom_sink_worst"])

        if vals.size > 0:
            x, y = empirical_cdf(vals)
            cdf_color = CDF_COLORS[idx % len(CDF_COLORS)]
            ax.plot(
                x,
                y,
                color=cdf_color,
                linewidth=1.3,
                label=f"Empirical CDF (n={vals.size})",
                zorder=4,
            )

            if np.isfinite(mean_v):
                ax.axvline(
                    mean_v,
                    color=MEAN_COLOR,
                    linewidth=1.0,
                    linestyle=(0, (3, 2)),
                    label=f"Mean = {mean_v:.3f}",
                    zorder=5,
                )
            if np.isfinite(cvar_v):
                ax.axvline(
                    cvar_v,
                    color=CVAR_COLOR,
                    linewidth=1.4,
                    linestyle="-",
                    label=f"CVaR20 = {cvar_v:.3f}",
                    zorder=6,
                )
            if np.isfinite(worst_v):
                ax.axvline(
                    worst_v,
                    color=WORST_COLOR,
                    linewidth=1.1,
                    linestyle=(0, (4, 1, 1, 1)),
                    label=f"Worst = {worst_v:.3f}",
                    zorder=5,
                )

            x_ref = x.copy()
            for v in [mean_v, cvar_v, worst_v]:
                if np.isfinite(v):
                    x_ref = np.append(x_ref, v)
            x_min = float(np.min(x_ref))
            x_max = float(np.max(x_ref))
            x_span = max(1e-9, x_max - x_min)
            x_pad = 0.07 * x_span
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
        else:
            ax.text(
                0.5,
                0.5,
                "No successful scenarios",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color=(0.2, 0.2, 0.2, 1.0),
            )

        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("Nominal sink rate [m/s]")
        ax.set_title(f"Candidate {cid} (success {n_success}/{n_all})", pad=5)

        leg = ax.legend(
            loc="lower right",
            frameon=True,
            framealpha=1.0,
            edgecolor="black",
            fontsize=8.3,
            handlelength=1.5,
            borderpad=0.5,
            labelspacing=0.2,
        )
        if leg is not None:
            leg.get_frame().set_linewidth(AXIS_EDGE_LW)

    for j in range(len(candidate_ids), len(axes_arr)):
        axes_arr[j].axis("off")

    # Left-most visible axis gets y-label.
    for ax in axes_arr:
        if ax.get_visible():
            ax.set_ylabel("Empirical CDF [-]")
            break

    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


### Entrypoint
def main() -> None:
    df_plot, df_summary = load_data(INPUT_XLSX)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_sink_ecdf(df_plot=df_plot, df_summary=df_summary, out_path=OUT_PATH)

    n_success = int(df_plot["nom_success"].sum())
    n_total = len(df_plot)
    print(f"Source sheets used: {SHEET_PLOT}, {SHEET_SUMMARY}")
    print(f"Rows in PlotDataRobust: {n_total}")
    print(f"Successful scenarios used for ECDF: {n_success}")
    print(f"Candidates plotted: {sorted(df_summary['candidate_id'].astype(int).unique().tolist())}")
    print(f"Saved: {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()

