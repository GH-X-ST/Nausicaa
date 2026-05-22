"""
Plot GP mean-field heat maps using the annular-Gaussian heat-map style.

For each height sheet, GP mean predictions are loaded from:
    B_results/Single_Fan_GP/single_fan_gp_grid_predictions.xlsx
Then the field is interpolated onto a dense uniform grid and plotted.
"""


from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import RegularGridInterpolator

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

SHEETS = ["z020", "z035", "z050", "z075", "z110", "z160", "z220"]

GP_GRID_XLSX = Path("B_results/Single_Fan_GP/single_gp_grid_predictions.xlsx")
OUT_DIR = Path("A_figures/Single_Fan_GP")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Axis and colorbar labels use metres and metres per second in exported figures.
CBAR_LABEL = r"$w$ (m $\!$s$^{-1}$)"
XLABEL = r"$x$ (m)"
YLABEL = r"$y$ (m)"

# Line widths are fixed for figure-to-figure comparability.

AXIS_EDGE_LW = 0.80
CBAR_EDGE_LW = AXIS_EDGE_LW
GRID_COLOR = (0.85, 0.85, 0.85, 1.0)
GRID_LINEWIDTH = 0.4


# Exponential opacity mapping versus normalized w (= 0..1).
# Alpha maps normalized velocity monotonically so weak updraft remains visually subordinate.
ALPHA_EXP_RATE = 0.005

# Fan outlet marker in arena coordinates (m).
FAN_OUTLET_X = 4.2
FAN_OUTLET_Y = 2.4
FAN_OUTLET_DIAMETER = 0.8
FAN_OUTLET_EDGE_LW = 1.1
FAN_OUTLET_ALPHA = 0.6
FAN_OUTLET_DASH = (0, (2, 2))

# Fixed color scale keeps heights and model families visually comparable.
PLOT_VMIN = 0.0
PLOT_VMAX = 8.0

# Dense display grid only affects plot interpolation, not fitted data.
GRID_NX = 240
GRID_NY = 180


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


# GP workbooks are generated model outputs; loading keeps coordinates and velocity predictions paired by sheet.
def load_gp_mean_sheet(
    xlsx_path: Path,
    sheet_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load one GP mean sheet (format: y/x index, x columns).
    Returns x, y, W with shape (Ny, Nx), sorted ascending.
    """
    raw = pd.read_excel(xlsx_path, sheet_name=sheet_name, index_col=0)
    if raw.empty:
        raise ValueError(f"Sheet '{sheet_name}' is empty in {xlsx_path}.")

    x = pd.to_numeric(raw.columns, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(raw.index, errors="coerce").to_numpy(dtype=float)
    w = raw.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    if w.shape != (y.size, x.size):
        raise ValueError(
            f"Shape mismatch in {sheet_name}: W{w.shape}, y({y.size}), x({x.size})."
        )
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError(f"Non-numeric x/y axis values found in sheet '{sheet_name}'.")

    x_order = np.argsort(x)
    y_order = np.argsort(y)
    x = x[x_order]
    y = y[y_order]
    w = w[np.ix_(y_order, x_order)]

    if np.any(np.diff(x) <= 0.0) or np.any(np.diff(y) <= 0.0):
        raise ValueError(f"x/y coordinates must be strictly increasing in '{sheet_name}'.")

    return x, y, w


# Display-grid resolution is a plotting choice and must not be interpreted as measurement density.
def build_continuous_grid(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a dense uniform grid covering the measured extents.
    """
    x_lin = np.linspace(float(np.min(x)), float(np.max(x)), GRID_NX, dtype=float)
    y_lin = np.linspace(float(np.min(y)), float(np.max(y)), GRID_NY, dtype=float)
    xg, yg = np.meshgrid(x_lin, y_lin)
    return xg, yg


# Display interpolation is only for figure smoothness; model fitting remains on source samples.
def interpolate_to_continuous_grid(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
) -> np.ndarray:
    """
    Bilinear interpolation of GP mean map onto dense grid.
    """
    interp = RegularGridInterpolator(
        (y, x),
        w,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    points = np.column_stack([y_grid.ravel(), x_grid.ravel()])
    w_dense = interp(points).reshape(x_grid.shape)

    if np.any(~np.isfinite(w_dense)):
        interp_nn = RegularGridInterpolator(
            (y, x),
            w,
            method="nearest",
            bounds_error=False,
            fill_value=0.0,
        )
        missing = ~np.isfinite(w_dense)
        w_dense[missing] = interp_nn(points[missing.ravel()])

    return w_dense


# Continuous plots show model or interpolated fields on a display grid, not new measurements.
def plot_continuous_heatmap(x: np.ndarray, y: np.ndarray, w: np.ndarray, outpath: Path) -> None:
    """
    Plot continuous GP heat map using the annular-Gaussian avg style.
    """
    x_edges = centers_to_edges(x)
    y_edges = centers_to_edges(y)

    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.edgecolor": "k",
        "axes.linewidth": AXIS_EDGE_LW,
        "patch.edgecolor": "k",
    })

    fig, ax = plt.subplots(figsize=(6.8, 5.6), dpi=600)
    cmap_alpha = build_alpha_cmap()
    im = ax.pcolormesh(
        x_edges,
        y_edges,
        w,
        shading="auto",
        cmap=cmap_alpha,
        vmin=PLOT_VMIN,
        vmax=PLOT_VMAX,
    )

    outlet = Circle(
        (FAN_OUTLET_X, FAN_OUTLET_Y),
        radius=FAN_OUTLET_DIAMETER / 2.0,
        fill=False,
        edgecolor=(0, 0, 0, FAN_OUTLET_ALPHA),
        linewidth=FAN_OUTLET_EDGE_LW,
        linestyle=FAN_OUTLET_DASH,
        label="Fan outlet",
        zorder=5,
        clip_on=True,
    )
    ax.add_patch(outlet)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.6%", pad=0.15)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(CBAR_LABEL)
    cbar.formatter = FormatStrFormatter("%.2f")
    cbar.update_ticks()
    cbar.ax.tick_params(width=0.6, length=2)
    cbar.outline.set_linewidth(CBAR_EDGE_LW)
    cbar.outline.set_edgecolor("k")
    cbar.outline.set_visible(True)
    cbar.ax.patch.set_edgecolor("k")
    cbar.ax.patch.set_linewidth(CBAR_EDGE_LW)
    cbar.ax.set_frame_on(True)
    for spine in cbar.ax.spines.values():
        spine.set_edgecolor("k")
        spine.set_linewidth(CBAR_EDGE_LW)

    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID_COLOR, linewidth=GRID_LINEWIDTH)
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_EDGE_LW)
    xticks = np.arange(0.0, 8.4 + 1e-9, 0.6)
    yticks = np.arange(0.0, 4.8 + 1e-9, 0.4)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f"{v:.2f}" for v in xticks])
    ax.set_yticklabels([f"{v:.2f}" for v in yticks])
    ax.tick_params(axis="x", labelrotation=30)
    ax.tick_params(axis="both", which="major", length=2, width=0.6)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.97, -0.18),
        frameon=True,
        framealpha=1.0,
        edgecolor="black",
        fontsize=7,
        handlelength=1.5,
        borderpad=0.7,
        labelspacing=0.2,
    )
    leg = ax.get_legend()
    if leg is not None:
        leg.get_frame().set_linewidth(AXIS_EDGE_LW)

    ax.set_xlim(0.0, 8.4)
    ax.set_ylim(0.0, 4.8)

    fig.tight_layout()
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
def main() -> None:
    if not GP_GRID_XLSX.exists():
        raise FileNotFoundError(f"Missing GP grid workbook: {GP_GRID_XLSX}")

    for sh in SHEETS:
        x, y, w_mean = load_gp_mean_sheet(GP_GRID_XLSX, f"{sh}_gp_mean")
        x_grid, y_grid = build_continuous_grid(x, y)
        w_dense = interpolate_to_continuous_grid(x, y, w_mean, x_grid, y_grid)

        out_png = OUT_DIR / f"{sh}_single_gp_heatmap.png"
        plot_continuous_heatmap(
            x=x_grid[0, :],
            y=y_grid[:, 0],
            w=w_dense,
            outpath=out_png,
        )

    print(f"Saved figures to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()


