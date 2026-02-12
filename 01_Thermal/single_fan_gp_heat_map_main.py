"""
Plot GP mean-field heat maps using the annular-Gaussian main-figure style.

For each height sheet, GP mean predictions are loaded from:
    B_results/Single_Fan_GP/single_fan_gp_grid_predictions.xlsx
Then the field is interpolated onto a dense uniform grid and plotted.
"""

###### Initialization

### Imports
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import RegularGridInterpolator

import cmocean  # https://matplotlib.org/cmocean

### User settings
SHEETS = ["z020", "z035", "z050", "z075", "z110", "z160", "z220"]

GP_GRID_XLSX = Path("B_results/Single_Fan_GP/single_fan_gp_grid_predictions.xlsx")
OUT_DIR = Path("A_figures/Single_Fan_GP")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Units / labels
CBAR_LABEL = r"$w$ (m$\cdot$s$^{-1}$)"
XLABEL = r"$x$ (m)"
YLABEL = r"$y$ (m)"

# Line widths
AXIS_EDGE_LW = 0.30
CBAR_EDGE_LW = 0.30
LEGEND_FONTSIZE = 8.5

# Fan outlet marker (single fan)
FAN_OUTLET_X = 4.2
FAN_OUTLET_Y = 2.4
FAN_OUTLET_DIAMETER = 0.8
FAN_OUTLET_EDGE_LW = 1.1
FAN_OUTLET_ALPHA = 0.6
FAN_OUTLET_DASH = (0, (2, 2))

# Color scale
PLOT_VMIN = 0.0
PLOT_VMAX = 8.0

# Continuous grid resolution
GRID_NX = 240
GRID_NY = 180


### Helpers
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


def build_continuous_grid(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a dense uniform grid covering the measured extents.
    """
    x_lin = np.linspace(float(np.min(x)), float(np.max(x)), GRID_NX, dtype=float)
    y_lin = np.linspace(float(np.min(y)), float(np.max(y)), GRID_NY, dtype=float)
    xg, yg = np.meshgrid(x_lin, y_lin)
    return xg, yg


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


def plot_continuous_heatmap(x: np.ndarray, y: np.ndarray, w: np.ndarray, outpath: Path) -> None:
    """
    Plot continuous GP heat map using the annular-Gaussian avg main style.
    """
    x_edges = centers_to_edges(x)
    y_edges = centers_to_edges(y)

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": LEGEND_FONTSIZE,
        "axes.edgecolor": "k",
        "axes.linewidth": AXIS_EDGE_LW,
        "patch.edgecolor": "k",
    })

    fig, ax = plt.subplots(figsize=(5.7, 3.9), dpi=600)
    im = ax.pcolormesh(
        x_edges,
        y_edges,
        w,
        shading="auto",
        cmap=cmocean.cm.thermal,
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
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_EDGE_LW)
    xticks = np.arange(0.0, 8.4 + 1e-9, 1.4)
    yticks = np.arange(0.0, 4.8 + 1e-9, 0.8)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f"{v:.2f}" for v in xticks])
    ax.set_yticklabels([f"{v:.2f}" for v in yticks])
    ax.tick_params(axis="both", which="major", length=2, width=0.6)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.97, -0.22),
        frameon=True,
        framealpha=1.0,
        edgecolor="black",
        fontsize=LEGEND_FONTSIZE,
        handlelength=1.5,
        borderpad=0.5,
        labelspacing=0.2,
    )
    leg = ax.get_legend()
    if leg is not None:
        leg.get_frame().set_linewidth(0.3)

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


### Export each sheet as PNG
def main() -> None:
    if not GP_GRID_XLSX.exists():
        raise FileNotFoundError(f"Missing GP grid workbook: {GP_GRID_XLSX}")

    for sh in SHEETS:
        x, y, w_mean = load_gp_mean_sheet(GP_GRID_XLSX, f"{sh}_gp_mean")
        x_grid, y_grid = build_continuous_grid(x, y)
        w_dense = interpolate_to_continuous_grid(x, y, w_mean, x_grid, y_grid)

        out_png = OUT_DIR / f"{sh}_single_fan_gp_heatmap_main.png"
        plot_continuous_heatmap(
            x=x_grid[0, :],
            y=y_grid[:, 0],
            w=w_dense,
            outpath=out_png,
        )

    print(f"Saved figures to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
