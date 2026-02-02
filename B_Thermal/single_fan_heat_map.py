###### Initialization

### Imports
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cmocean # https://matplotlib.org/cmocean

### User settings
XLSX_PATH = "S01.xlsx"
SHEETS = ["z020", "z035", "z050", "z075", "z110", "z160", "z220"]

OUT_DIR = Path("A_figures/Single_Fan_Heat_Map")
OUT_DIR.mkdir(exist_ok=True)

MASK_ZEROS_AS_NODATA = False

# Units / labels
CBAR_LABEL = r"$w$ (m$\cdot$s$^{-1}$)"   # vertical velocity
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


def plot_heatmap(x, y, W, outpath: Path, mask_zeros: bool = True):
    """
    Uses pcolormesh with edges for nonuniform grids.
    """
    # Masking policy
    W_plot = W.copy()
    if mask_zeros:
        W_plot[W_plot == 0.0] = np.nan  # treat zeros as missing/outside

    # Convert center grids -> edges for pcolormesh
    x_edges = centers_to_edges(x)
    y_edges = centers_to_edges(y)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

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

    # Heatmap with cell edges
    im = ax.pcolormesh(
        x_edges,
        y_edges,
        W_plot,
        shading="auto",
        cmap=cmocean.cm.thermal,
        vmin=0.0,
        vmax=7.0,
        edgecolors=(0, 0, 0, 0.3),
        linewidth=CELL_EDGE_LW,
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
    )
    ax.add_patch(outlet)

    # Annotate each cell with its value
    for iy, y0 in enumerate(y_centers):
        for ix, x0 in enumerate(x_centers):
            val = W_plot[iy, ix]
            if np.isfinite(val):
                r, g, b, _a = im.cmap(im.norm(val))
                luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                text_color = "white" if luminance < 0.5 else "black"
                ax.text(
                    x0,
                    y0,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color=text_color,
                )

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
    ax.set_xticks(x_centers)
    ax.set_yticks(y_centers)
    ax.set_xticklabels([f"{v:g}" for v in x])
    ax.set_yticklabels([f"{v:g}" for v in y])
    ax.tick_params(axis="both", which="major", length=2, width=0.6)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.97, -0.13),
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

    # Tighten limits to data extents
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])

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
    for sh in SHEETS:
        x, y, W = read_slice_from_sheet(XLSX_PATH, sh)

        out_png = OUT_DIR / f"{sh}_single_heatmap.png"

        plot_heatmap(x, y, W, out_png, mask_zeros=MASK_ZEROS_AS_NODATA)

    print(f"Saved figures to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
