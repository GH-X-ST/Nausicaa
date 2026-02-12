"""
Plot annular-Gaussian model heat maps using the same style as annuli heat maps.

For each height sheet, fitted ring parameters are loaded from
B_results/annular_gaussian_avg_params.xlsx and used to generate a continuous
model field w(x, y). The field is evaluated on a dense uniform grid (not the
measurement grid) and then plotted.
"""

###### Initialization

### Imports
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cmocean  # https://matplotlib.org/cmocean

### User settings
XLSX_PATH = "S01.xlsx"
SHEETS = ["z020", "z035", "z050", "z075", "z110", "z160", "z220"]

OUT_DIR = Path("A_figures/Single_Fan_Annular_Gaussian_Avg_Heat_Map")
OUT_DIR.mkdir(exist_ok=True)

PARAMS_XLSX = Path("B_results/annular_gaussian_avg_params.xlsx")

# Fan centre (x_c, y_c)
FAN_CENTER_XY = (4.2, 2.4)

# Units / labels
CBAR_LABEL = r"$w$ (m$\cdot$s$^{-1}$)"  # vertical velocity
XLABEL = r"$x$ (m)"
YLABEL = r"$y$ (m)"

# Line widths
CELL_EDGE_LW = 0.30
AXIS_EDGE_LW = 0.30
CBAR_EDGE_LW = 0.30

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

# z020 -> 0.20 m, z110 -> 1.10 m, etc.
SHEET_HEIGHT_DIVISOR = 100.0


# Helpers
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


def read_slice_from_sheet(xlsx_path: str, sheet_name: str):
    """
    Reads your grid sheet:
      - row 0, col 1.. = x coordinates
      - col 0, row 1.. = y coordinates
      - interior = scalar field values
    Returns x_centers, y_centers, W (Ny x Nx)
    """
    raw = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)

    # x along first row (skip [0,0])
    x = pd.to_numeric(raw.iloc[0, 1:], errors="coerce").to_numpy(dtype=float)

    # y along first column (skip [0,0])
    y = pd.to_numeric(raw.iloc[1:, 0], errors="coerce").to_numpy(dtype=float)

    # field values
    W = raw.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    # sanity checks
    if W.shape != (y.size, x.size):
        raise ValueError(
            f"Shape mismatch in {sheet_name}: W{W.shape}, y({y.size}), x({x.size})."
        )

    # Ensure y increases bottom-to-top on the plot (0 -> max)
    if y.size >= 2 and y[0] > y[-1]:
        y = y[::-1]
        W = W[::-1, :]

    return x, y, W


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


def load_ring_params(xlsx_path: Path) -> pd.DataFrame:
    """
    Load fitted ring parameters from Excel.
    """
    df = pd.read_excel(xlsx_path)
    required = {"z_m", "A_ring", "r_ring", "delta_r", "w0"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing columns in {xlsx_path}: {sorted(missing)}")
    return df.copy()


def params_for_height(df: pd.DataFrame, z_m: float) -> Tuple[float, float, float, float]:
    """
    Extract [A_ring, r_ring, delta_r, w0] for a given height.
    """
    mask = np.isclose(df["z_m"].to_numpy(dtype=float), float(z_m), atol=1e-6)
    if not np.any(mask):
        available = ", ".join([f"{v:.2f}" for v in df["z_m"].to_numpy(dtype=float)])
        raise ValueError(f"No parameters for z={z_m:.2f}. Available: {available}")
    row = df.loc[mask].iloc[0]
    return float(row["A_ring"]), float(row["r_ring"]), float(row["delta_r"]), float(row["w0"])


def plot_continuous_heatmap(x, y, W, outpath: Path):
    """
    Plot continuous model heat map using the same axis settings as single_fan_heat_map.py.
    """
    # Convert center grids -> edges for pcolormesh
    x_edges = centers_to_edges(x)
    y_edges = centers_to_edges(y)

    # Figure styling
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

    fig, ax = plt.subplots(figsize=(6.8, 5.6), dpi=600)  # larger for readability

    # Continuous heatmap
    im = ax.pcolormesh(
        x_edges,
        y_edges,
        W,
        shading="auto",
        cmap=cmocean.cm.thermal,
        vmin=PLOT_VMIN,
        vmax=PLOT_VMAX,
    )

    # Fan outlet marker (thin dashed ring)
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

    # Colorbar
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

    # Axes
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_aspect("equal", adjustable="box")  # 1:1 grid without stretching
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_EDGE_LW)
    # Axis ticks: fixed spacing and range to match annuli heat map
    xticks = np.arange(0.0, 8.4 + 1e-9, 0.6)
    yticks = np.arange(0.0, 4.8 + 1e-9, 0.4)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f"{v:.2f}" for v in xticks])
    ax.set_yticklabels([f"{v:.2f}" for v in yticks])
    ax.tick_params(axis="x", labelrotation=-30)
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
        leg.get_frame().set_linewidth(0.3)

    # Tighten limits to match annuli heat map extents
    ax.set_xlim(0.0, 8.4)
    ax.set_ylim(0.0, 4.8)

    fig.tight_layout()
    # Shorten colorbar and align its bottom with the x-axis baseline
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
def main():
    params_df = load_ring_params(PARAMS_XLSX)

    for sh in SHEETS:
        x, y, _w = read_slice_from_sheet(XLSX_PATH, sh)
        z_m = parse_sheet_height_m(sh)

        a_ring, r_ring, delta_r_model, w0 = params_for_height(params_df, z_m)

        x_grid, y_grid = build_continuous_grid(x, y)
        xc, yc = FAN_CENTER_XY
        r = np.sqrt((x_grid - xc) ** 2 + (y_grid - yc) ** 2)
        W_model = a_ring * np.exp(-((r - r_ring) / delta_r_model) ** 2)

        out_png = OUT_DIR / f"{sh}_single_annular_gaussian_avg_heatmap.png"
        plot_continuous_heatmap(
            x_grid[0, :],
            y_grid[:, 0],
            W_model,
            outpath=out_png,
        )

    print(f"Saved figures to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
