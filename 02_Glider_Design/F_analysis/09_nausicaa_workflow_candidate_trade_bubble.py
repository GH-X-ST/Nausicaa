"""
Plot candidate trade bubble chart and highlight robust-selected candidate.

Input workbook:
    C_results/nausicaa_workflow_iter5.xlsx
Sheets:
    Candidates
    RobustSummary

Output figure:
    B_figures/09_nausicaa_workflow_candidate_trade_bubble.png
"""

###### Initialization

### Imports
from pathlib import Path
from typing import Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable

try:
    import cmocean
except ImportError:
    cmocean = None


### User settings
INPUT_XLSX = Path("C_results/nausicaa_workflow_iter5.xlsx")
SHEET_CANDIDATES = "Candidates"
SHEET_ROBUST = "RobustSummary"
OUT_DIR = Path("B_figures")
OUT_PATH = OUT_DIR / "09_nausicaa_workflow_candidate_trade_bubble.png"

REQUIRED_CANDIDATE_COLUMNS = [
    "candidate_id",
    "wing_span_m",
    "sink_rate_mps",
    "roll_tau_s",
    "mass_total_kg",
]

REQUIRED_ROBUST_COLUMNS = [
    "candidate_id",
    "is_selected",
]

# Plot style aligned with F_analysis references
AXIS_EDGE_LW = 0.80
GRID_COLOR = (0.85, 0.85, 0.85, 1.0)
GRID_LINEWIDTH = 0.4

SELECTED_STAR_SIZE = 150
SELECTED_STAR_COLOR = "#ffd200"


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


def validate_columns(df: pd.DataFrame, required: list, sheet_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in '{sheet_name}': {missing}")


def load_joined_df(xlsx_path: Path) -> pd.DataFrame:
    """
    Load Candidates and RobustSummary, then left-join is_selected on candidate_id.
    """
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Workbook not found: {xlsx_path}")

    xls = pd.ExcelFile(xlsx_path)
    if SHEET_CANDIDATES not in xls.sheet_names:
        raise KeyError(f"Sheet '{SHEET_CANDIDATES}' not found in {xlsx_path}.")
    if SHEET_ROBUST not in xls.sheet_names:
        raise KeyError(f"Sheet '{SHEET_ROBUST}' not found in {xlsx_path}.")

    df_cand = pd.read_excel(xlsx_path, sheet_name=SHEET_CANDIDATES)
    df_rob = pd.read_excel(xlsx_path, sheet_name=SHEET_ROBUST)

    validate_columns(df_cand, REQUIRED_CANDIDATE_COLUMNS, SHEET_CANDIDATES)
    validate_columns(df_rob, REQUIRED_ROBUST_COLUMNS, SHEET_ROBUST)

    df_cand = df_cand.copy()
    df_rob = df_rob.copy()

    df_cand["candidate_id"] = pd.to_numeric(df_cand["candidate_id"], errors="raise").astype(int)
    df_cand["wing_span_m"] = pd.to_numeric(df_cand["wing_span_m"], errors="raise")
    df_cand["sink_rate_mps"] = pd.to_numeric(df_cand["sink_rate_mps"], errors="raise")
    df_cand["roll_tau_s"] = pd.to_numeric(df_cand["roll_tau_s"], errors="raise")
    df_cand["mass_total_kg"] = pd.to_numeric(df_cand["mass_total_kg"], errors="raise")

    df_rob["candidate_id"] = pd.to_numeric(df_rob["candidate_id"], errors="raise").astype(int)
    df_rob["is_selected"] = coerce_bool_like(df_rob["is_selected"])

    df = df_cand.merge(
        df_rob[["candidate_id", "is_selected"]],
        on="candidate_id",
        how="left",
    )
    df["is_selected"] = df["is_selected"].fillna(False).astype(bool)

    df = df.sort_values(by="candidate_id", ascending=True).reset_index(drop=True)
    return df


def marker_sizes_from_mass(mass: np.ndarray, size_min: float = 80.0, size_max: float = 300.0) -> np.ndarray:
    """Scale mass to marker area with robust handling for near-constant mass."""
    m_min = float(np.nanmin(mass))
    m_max = float(np.nanmax(mass))

    if np.isclose(m_min, m_max):
        return np.full_like(mass, 0.5 * (size_min + size_max), dtype=float)

    t = (mass - m_min) / (m_max - m_min)
    return size_min + t * (size_max - size_min)


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID_COLOR, linewidth=GRID_LINEWIDTH)
    ax.tick_params(axis="both", which="major", length=2, width=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_EDGE_LW)


def plot_candidate_trade(df: pd.DataFrame, out_path: Path) -> None:
    """
    Bubble scatter:
      x = wing_span_m,
      y = sink_rate_mps,
      color = roll_tau_s,
      size = mass_total_kg,
      highlight selected candidate.
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

    x = df["wing_span_m"].to_numpy(dtype=float)
    y = df["sink_rate_mps"].to_numpy(dtype=float)
    tau = df["roll_tau_s"].to_numpy(dtype=float)
    mass = df["mass_total_kg"].to_numpy(dtype=float)
    cand_id = df["candidate_id"].to_numpy(dtype=int)
    selected_mask = df["is_selected"].to_numpy(dtype=bool)

    sizes = marker_sizes_from_mass(mass)

    # Use reversed map so lower tau (better agility) maps to visually stronger colors.
    cmap = cmocean.cm.thermal_r if cmocean is not None else plt.get_cmap("viridis_r")
    norm = mcolors.Normalize(vmin=float(np.nanmin(tau)), vmax=float(np.nanmax(tau)))

    fig, ax = plt.subplots(figsize=(5.8, 4.0), dpi=600)
    fig.patch.set_facecolor("white")
    style_axes(ax)

    sc = ax.scatter(
        x,
        y,
        c=tau,
        cmap=cmap,
        norm=norm,
        s=sizes,
        alpha=0.90,
        edgecolors="black",
        linewidths=0.5,
        zorder=4,
    )

    # Label all candidates by ID for traceability.
    for xi, yi, ci in zip(x, y, cand_id):
        ax.annotate(
            f"C{int(ci)}",
            xy=(xi, yi),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
            ha="left",
            va="bottom",
            color=(0.05, 0.05, 0.05, 0.95),
            zorder=8,
        )

    if np.any(selected_mask):
        x_sel = x[selected_mask]
        y_sel = y[selected_mask]
        ax.scatter(
            x_sel,
            y_sel,
            marker="*",
            s=SELECTED_STAR_SIZE,
            color=SELECTED_STAR_COLOR,
            edgecolors="black",
            linewidths=0.9,
            zorder=10,
        )

        i_sel = int(np.flatnonzero(selected_mask)[0])
        ax.annotate(
            "Selected after robust post-check",
            xy=(x[i_sel], y[i_sel]),
            xytext=(10, 13),
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
                "facecolor": (1.0, 1.0, 1.0, 0.80),
                "edgecolor": "none",
            },
            zorder=20,
        )

    ax.set_xlabel("Wing span [m]")
    ax.set_ylabel("Nominal sink rate [m/s]")

    # Robust axis margins for tiny x-range cases.
    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))

    x_span = x_max - x_min
    y_span = y_max - y_min

    x_pad = 0.08 * x_span if x_span > 1e-12 else max(0.002, 0.02 * max(1.0, abs(x_min)))
    y_pad = 0.08 * y_span if y_span > 1e-12 else max(0.02, 0.04 * max(1.0, abs(y_min)))

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02, fraction=0.05)
    cbar.set_label(r"Roll time constant $\tau$ [s]")
    cbar.ax.tick_params(width=0.6, length=2)
    cbar.outline.set_linewidth(AXIS_EDGE_LW)

    # Size legend anchors mass encoding.
    m_min = float(np.nanmin(mass))
    m_max = float(np.nanmax(mass))
    m_mid = 0.5 * (m_min + m_max)
    size_samples = np.array([m_min, m_mid, m_max], dtype=float)
    area_samples = marker_sizes_from_mass(size_samples)

    size_handles = []
    for m_val, a_val in zip(size_samples, area_samples):
        size_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=(0.82, 0.82, 0.82, 1.0),
                markeredgecolor="black",
                markeredgewidth=0.5,
                markersize=float(np.sqrt(a_val) / 2.0),
                label=f"mass={m_val:.3f} kg",
            )
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor=SELECTED_STAR_COLOR,
            markeredgecolor="black",
            markeredgewidth=0.9,
            markersize=8,
            label="Selected after robust post-check",
        )
    ] + size_handles

    leg = ax.legend(
        handles=legend_handles,
        loc="upper left",
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

    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


### Entrypoint
def main() -> None:
    df = load_joined_df(INPUT_XLSX)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_candidate_trade(df=df, out_path=OUT_PATH)

    print(f"Source sheets used: {SHEET_CANDIDATES}, {SHEET_ROBUST}")
    print(f"Candidates plotted: {len(df)}")
    print(f"Selected candidates: {int(df['is_selected'].sum())}")
    print(f"Saved: {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()

