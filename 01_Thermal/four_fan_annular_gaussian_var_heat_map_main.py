"""
Plot annular-Gaussian model heat maps using the same style as annuli heat maps.

For each height sheet, fitted ring parameters are loaded from
B_results/four_annular_var_params.xlsx and used to generate a continuous
model field w(x, y). The field is evaluated on a dense uniform grid (not the
measurement grid) and then plotted.
"""


from pathlib import Path
from typing import Tuple
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable

# cmocean provides perceptual thermal colormaps used consistently across figures.
import cmocean


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Plot Configuration and Data Sources
# 2) Workbook Loading and Plot Construction
# 3) Batch Figure Export
# =============================================================================

# =============================================================================
# 1) Plot Configuration and Data Sources
# =============================================================================
# Workbook, parameter, and output paths below define the data-provenance boundary for this run.

XLSX_PATH = "S02.xlsx"
SHEETS = ["z020", "z035", "z050", "z075", "z110", "z160", "z220"]

OUT_DIR = Path("A_figures/Four_Fan_Annular_Gaussian_Var")
OUT_DIR.mkdir(exist_ok=True)

PARAMS_XLSX = Path("B_results/four_annular_var_params.xlsx")

FOUR_FAN_CENTERS_XY = (
    (3.0, 3.6),
    (5.4, 3.6),
    (3.0, 1.2),
    (5.4, 1.2),
)

# Axis and colorbar labels use metres and metres per second in exported figures.
# Colourbar reports vertical velocity in metres per second.
CBAR_LABEL = r"$w$ (m $\!$s$^{-1}$)"
XLABEL = r"$x$ (m)"
YLABEL = r"$y$ (m)"

# Ticks follow measured arena grid labels while keeping compact A4 panels readable.
# Line widths are fixed for figure-to-figure comparability.
CELL_EDGE_LW = 0.30
AXIS_EDGE_LW = 0.80

CBAR_EDGE_LW = AXIS_EDGE_LW
GRID_COLOR = (0.85, 0.85, 0.85, 1.0)
GRID_LINEWIDTH = 0.4


# Exponential opacity mapping versus normalized w (= 0..1).
# Alpha maps normalized velocity monotonically so weak updraft remains visually subordinate.
ALPHA_EXP_RATE = 0.005
LEGEND_FONTSIZE = 8.5

# Fan outlet markers in arena coordinates (m).
FAN_OUTLET_POINTS = list(FOUR_FAN_CENTERS_XY)
FAN_OUTLET_DIAMETER = 0.8
FAN_OUTLET_EDGE_LW = 0.7
FAN_OUTLET_ALPHA = 0.6
FAN_OUTLET_DASH = (0, (2, 2))

# Fixed color scale keeps heights and model families visually comparable.
PLOT_VMIN = 0.0
PLOT_VMAX = 8.0

# Dense display grid only affects plot interpolation, not fitted data.
GRID_NX = 240
GRID_NY = 180

# z020 -> 0.20 m, z110 -> 1.10 m, etc.
SHEET_HEIGHT_DIVISOR = 100.0


# =============================================================================
# 2) Workbook Loading and Plot Construction
# =============================================================================
# Parsing and plotting helpers keep measured workbook coordinates in arena metres.

# Alpha mapping keeps low-speed regions visible while preserving a common thermal colour scale.
def build_alpha_cmap() -> mcolors.ListedColormap:
    """
    Build a thermal colormap with exponential alpha versus normalized w.
    """
    base_cmap = cmocean.cm.thermal
    colors = base_cmap(np.linspace(0.0, 1.0, 256))

    t_norm = np.linspace(0.0, 1.0, colors.shape[0])
    exp_scale = np.exp(ALPHA_EXP_RATE * t_norm)
    exp_full = np.exp(ALPHA_EXP_RATE)
    alpha = (exp_scale - 1.0) / (exp_full - 1.0)
    alpha[0] = 0.0
    alpha[-1] = 1.0
    colors[:, 3] = alpha

    return mcolors.ListedColormap(colors)

# Pcolormesh uses cell edges, so measured centre coordinates are expanded without changing sample values.
def centers_to_edges(c: np.ndarray) -> np.ndarray:
    """
    Convert 1D array of cell centers -> cell edges for pcolormesh.
    """
    c = np.asarray(c, dtype=float)
    if c.size < 2:
        raise ValueError("Need at least 2 center points to compute edges.")
    edges = np.empty(c.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (c[:-1] + c[1:])
    edges[0] = c[0] - 0.5 * (c[1] - c[0])
    edges[-1] = c[-1] + 0.5 * (c[-1] - c[-2])
    return edges


# Sheet names encode height in centimetres; parsing converts that label to metres.
def parse_sheet_height_m(sheet_name: str) -> float:
    """
    Parse height in meters from sheet names like 'z020', 'z110', 'z220'.
    """
    if not sheet_name.startswith("z"):
        raise ValueError(f"Invalid sheet name (expected 'z###'): {sheet_name}")
    suffix = sheet_name[1:]
    if not suffix.isdigit():
        raise ValueError(f"Invalid height code in sheet name: {sheet_name}")
    return int(suffix) / SHEET_HEIGHT_DIVISOR


# Workbook sheets store x in the first row, y in the first column, and vertical velocity in m/s inside the grid.
def read_slice_from_sheet(xlsx_path: str, sheet_name: str):
    """
    Read the workbook grid layout:
      - row 0, col 1.. = x coordinates
      - col 0, row 1.. = y coordinates
      - interior = scalar field values
    Returns x_centers, y_centers, W (Ny x Nx)
    """
    raw = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)

    # Workbook grid stores x coordinates in the first row after the corner cell.
    x = pd.to_numeric(raw.iloc[0, 1:], errors="coerce").to_numpy(dtype=float)

    # Workbook grid stores y coordinates in the first column after the corner cell.
    y = pd.to_numeric(raw.iloc[1:, 0], errors="coerce").to_numpy(dtype=float)

    # Measured vertical-velocity block (m/s).
    W = raw.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    # Workbook grid shape must match y-by-x coordinates.
    if W.shape != (y.size, x.size):
        raise ValueError(
            f"Shape mismatch in {sheet_name}: W{W.shape}, y({y.size}), x({x.size})."
        )

    # Plot convention uses increasing arena y from bottom to top.
    if y.size >= 2 and y[0] > y[-1]:
        y = y[::-1]
        W = W[::-1, :]

    return x, y, W


# Display-grid resolution is a plotting choice and must not be interpreted as measurement density.
def build_continuous_grid(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a dense uniform grid covering the measurement extents.
    """
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    y_min = float(np.min(y))
    y_max = float(np.max(y))

    x_lin = np.linspace(x_min, x_max, GRID_NX, dtype=float)
    y_lin = np.linspace(y_min, y_max, GRID_NY, dtype=float)
    xg, yg = np.meshgrid(x_lin, y_lin)
    return xg, yg


# Ring-parameter tables are fitted-model inputs and remain separate from measured workbook grids.
def load_ring_params(xlsx_path: Path) -> pd.DataFrame:
    """
    Load fitted ring parameters from Excel.
    """
    df = pd.read_excel(xlsx_path)
    if "z_m" not in df.columns:
        raise ValueError(f"Missing column 'z_m' in {xlsx_path}.")
    return df.copy()


# Fan-ID discovery treats suffixed columns as the interface for per-fan fitted parameters.
def discover_fan_ids(df: pd.DataFrame) -> Tuple[str, ...]:
    """
    Discover fan IDs from columns like A_ring_F01.
    """
    pattern = re.compile(r"^A_ring_(F\d{2})$")
    fan_ids = []
    for col in df.columns:
        match = pattern.match(str(col))
        if match is not None:
            fan_ids.append(match.group(1))
    fan_ids = sorted(set(fan_ids))

    valid = []
    for fan_id in fan_ids:
        required = (
            f"A_ring_{fan_id}",
            f"r_ring_{fan_id}",
            f"delta_r_{fan_id}",
            f"w0_{fan_id}",
        )
        if all(col in df.columns for col in required):
            valid.append(fan_id)
    return tuple(valid)


# Height selection keeps plotted fields tied to the nearest fitted or interpolated z sample.
def params_for_height(df: pd.DataFrame, z_m: float, fan_ids: Tuple[str, ...]) -> np.ndarray:
    """
    Extract per-fan [A_ring, r_ring, delta_r, w0] for a given height.
    """
    mask = np.isclose(df["z_m"].to_numpy(dtype=float), float(z_m), atol=1e-6)
    if not np.any(mask):
        available = ", ".join([f"{v:.2f}" for v in df["z_m"].to_numpy(dtype=float)])
        raise ValueError(f"No parameters for z={z_m:.2f}. Available: {available}")
    row = df.loc[mask].iloc[0]

    if len(fan_ids) == 0:
        required = ("A_ring", "r_ring", "delta_r", "w0")
        missing = [c for c in required if c not in row.index]
        if missing:
            raise ValueError(f"Missing shared parameter columns: {missing}")
        shared = np.array(
            [
                float(row["A_ring"]),
                float(row["r_ring"]),
                float(row["delta_r"]),
                float(row["w0"]),
            ],
            dtype=float,
        )
        return np.repeat(shared[None, :], len(FOUR_FAN_CENTERS_XY), axis=0)

    params = np.empty((len(fan_ids), 4), dtype=float)
    for fan_idx, fan_id in enumerate(fan_ids):
        params[fan_idx, 0] = float(row[f"A_ring_{fan_id}"])
        params[fan_idx, 1] = float(row[f"r_ring_{fan_id}"])
        params[fan_idx, 2] = float(row[f"delta_r_{fan_id}"])
        params[fan_idx, 3] = float(row[f"w0_{fan_id}"])
    return params


# Nearest-fan radius maps each display point to the controlling plume centre.
def nearest_fan_radius_map(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    fan_centers_xy: Tuple[Tuple[float, float], ...],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute nearest-fan index and radius for each grid point.
    """
    fan_xy = np.asarray(fan_centers_xy, dtype=float)
    d_sample_fan = np.sqrt(
        (x_grid[..., None] - fan_xy[:, 0]) ** 2
        + (y_grid[..., None] - fan_xy[:, 1]) ** 2
    )
    nearest_idx = np.argmin(d_sample_fan, axis=-1)
    nearest_r = np.min(d_sample_fan, axis=-1)
    return nearest_idx, nearest_r


# Continuous plots show model or interpolated fields on a display grid, not new measurements.
def plot_continuous_heatmap(x, y, W, outpath: Path):
    """
    Plot continuous model heat map using the same axis settings as single_fan_heat_map.py.
    """
    # Pcolormesh requires cell edges; this preserves the measured cell-centre layout.
    x_edges = centers_to_edges(x)
    y_edges = centers_to_edges(y)

    # White background and hidden top/right spines match the thesis figure style.
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": LEGEND_FONTSIZE,
        "axes.edgecolor": "k",
        "axes.linewidth": AXIS_EDGE_LW,
        "patch.edgecolor": "k",
    })

    # A4 thesis sizing supports two panels per landscape row at fixed data scale.
    fig, ax = plt.subplots(figsize=(5.2, 3.0), dpi=600)

    # Interpolated grid is for display only; fitted or measured values remain unchanged.
    cmap_alpha = build_alpha_cmap()
    im = ax.pcolormesh(
        x_edges,
        y_edges,
        W,
        shading="auto",
        cmap=cmap_alpha,
        vmin=PLOT_VMIN,
        vmax=PLOT_VMAX,
    )
    ax.hlines(
        y=float(y_edges[0]),
        xmin=float(x_edges[0]),
        xmax=float(x_edges[-1]),
        colors=(0.0, 0.0, 0.0, 0.70),
        linewidth=0.30,
        zorder=4,
    )

    for idx, (fx, fy) in enumerate(FAN_OUTLET_POINTS):
        outlet = Circle(
            (fx, fy),
            radius=FAN_OUTLET_DIAMETER / 2.0,
            fill=False,
            edgecolor=(0, 0, 0, FAN_OUTLET_ALPHA),
            linewidth=FAN_OUTLET_EDGE_LW,
            linestyle=FAN_OUTLET_DASH,
            label="Fan outlet" if idx == 0 else None,
            zorder=5,
            clip_on=True,
        )
        ax.add_patch(outlet)

    # Colorbar ticks use the fixed velocity scale for cross-figure comparison.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.990%", pad=0.22)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(CBAR_LABEL, fontsize=9)
    cbar.set_ticks(np.arange(0.0, 8.0 + 1e-9, 1.0))
    cbar.formatter = FormatStrFormatter("%.2f")
    cbar.update_ticks()
    cbar.ax.tick_params(width=0.6, length=2, labelsize=9)
    cbar.outline.set_linewidth(CBAR_EDGE_LW)
    cbar.outline.set_edgecolor("k")
    cbar.outline.set_visible(True)
    cbar.ax.patch.set_edgecolor("k")
    cbar.ax.patch.set_linewidth(CBAR_EDGE_LW)
    cbar.ax.set_frame_on(True)
    for spine in cbar.ax.spines.values():
        spine.set_edgecolor("k")
        spine.set_linewidth(CBAR_EDGE_LW)

    # Equal aspect preserves arena geometry in plan view.
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    # Equal aspect keeps arena distances physically meaningful in the rendered plot.
    ax.set_aspect("equal", adjustable="box")
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID_COLOR, linewidth=GRID_LINEWIDTH)
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_EDGE_LW)
    # Fixed two-decimal ticks avoid implicit re-centering between figures.
    xticks = np.arange(0.0, 8.4 + 1e-9, 1.4)
    yticks = np.arange(0.0, 4.8 + 1e-9, 0.8)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f"{v:.2f}" for v in xticks])
    ax.set_yticklabels([f"{v:.2f}" for v in yticks])
    ax.tick_params(axis="both", which="major", length=2, width=0.6)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(1.15, -0.25),
        frameon=True,
        framealpha=1.0,
        edgecolor="black",
        fontsize=(LEGEND_FONTSIZE - 0.2),
        handlelength=1.5,
        borderpad=0.5,
        labelspacing=0.2,
    )
    leg = ax.get_legend()
    if leg is not None:
        leg.get_frame().set_linewidth(AXIS_EDGE_LW)

    # Axis limits match the annuli heat-map domain instead of padded defaults.
    ax.set_xlim(0.0, 8.4)
    ax.set_ylim(0.0, 4.8)

    fig.tight_layout()
    # Colorbar is manually aligned to the x-axis baseline for thesis layout.
    ax_pos = ax.get_position()
    cax_pos = cax.get_position()
    new_h = ax_pos.height * 0.82
    new_y0 = ax_pos.y0
    cax.set_position([cax_pos.x0, new_y0, cax_pos.width, new_h])
    fig.savefig(
        outpath,
        bbox_inches="tight",
        facecolor="white",
        dpi=600,
    )
    plt.close(fig)


# =============================================================================
# 3) Batch Figure Export
# =============================================================================
# Entry points write deterministic artifacts so regenerated figures and tables can be compared by path and sheet name.

# Main execution keeps data loading, evaluation, and export order deterministic.
def main():
    params_df = load_ring_params(PARAMS_XLSX)
    fan_ids = discover_fan_ids(params_df)
    if len(fan_ids) > 0 and len(fan_ids) != len(FOUR_FAN_CENTERS_XY):
        raise ValueError(
            f"Parameter table has {len(fan_ids)} fan IDs but expected {len(FOUR_FAN_CENTERS_XY)}."
        )

    for sh in SHEETS:
        x, y, _w = read_slice_from_sheet(XLSX_PATH, sh)
        z_m = parse_sheet_height_m(sh)

        params_fan = params_for_height(params_df, z_m, fan_ids=fan_ids)

        x_grid, y_grid = build_continuous_grid(x, y)
        fan_xy = np.asarray(FOUR_FAN_CENTERS_XY, dtype=float)
        r_all = np.sqrt(
            (x_grid[:, :, None] - fan_xy[None, None, :, 0]) ** 2
            + (y_grid[:, :, None] - fan_xy[None, None, :, 1]) ** 2
        )

        W_model = np.zeros((x_grid.shape[0], x_grid.shape[1]), dtype=float)
        n_fans = len(FOUR_FAN_CENTERS_XY)
        for fan_idx in range(n_fans):
            p = params_fan[fan_idx]
            W_model += p[3] + p[0] * np.exp(
                -((r_all[:, :, fan_idx] - p[1]) / p[2]) ** 2
            )

        out_png = OUT_DIR / f"{sh}_four_annular_gaussian_var_heatmap_main.png"
        plot_continuous_heatmap(
            x_grid[0, :],
            y_grid[:, 0],
            W_model,
            outpath=out_png,
        )

    print(f"Saved figures to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()


