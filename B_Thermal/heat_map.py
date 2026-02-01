###### Initialization

### Imports
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### User settings
XLSX_PATH = "S01.xlsx"
SHEETS = ["z020", "z035", "z050", "z075", "z110", "z160", "z220"]

OUT_DIR = Path("A_figures/Single_Fan_Heat_Map")
OUT_DIR.mkdir(exist_ok=True)

MASK_ZEROS_AS_NODATA = False

# Units / labels (edit to match your notation)
CBAR_LABEL = r"$w$ (m$\cdot$s$^{-1}$)"   # vertical velocity
XLABEL = r"$x$ (m)"
YLABEL = r"$y$ (m)"


# ----------------------------
# Helpers
# ----------------------------
def centers_to_edges(c: np.ndarray) -> np.ndarray:
    """
    Convert 1D array of cell centers -> cell edges for pcolormesh.
    Works for nonuniform spacing.
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

    return x, y, W


def plot_heatmap(x, y, W, title: str, outpath: Path, mask_zeros: bool = True):
    """
    IEEE-ish single-panel heatmap with colorbar, equal aspect, clean layout.
    Uses pcolormesh with edges for nonuniform grids.
    """
    # Masking policy
    W_plot = W.copy()
    if mask_zeros:
        W_plot[W_plot == 0.0] = np.nan  # treat zeros as missing/outside

    # Convert center grids -> edges for pcolormesh
    x_edges = centers_to_edges(x)
    y_edges = centers_to_edges(y)

    # Figure styling (reasonable for IEEE 2-col; adjust if needed)
    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    })

    fig, ax = plt.subplots(figsize=(3.4, 2.8), dpi=300)  # ~single-column width

    # Heatmap
    im = ax.pcolormesh(x_edges, y_edges, W_plot, shading="auto")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(CBAR_LABEL)

    # Axes
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_title(title)
    ax.set_aspect("equal")  # preserve geometry

    # Optional: tighten limits to data extents
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Main: export each sheet as a PDF/PNG
# ----------------------------
def main():
    for sh in SHEETS:
        x, y, W = read_slice_from_sheet(XLSX_PATH, sh)

        # Example title: z020 -> "z = 0.20 m" (edit if your naming means something else)
        # If z020 means 0.20 m, this is appropriate:
        z_m = float(sh[1:]) / 100.0
        title = rf"Slice at $z = {z_m:.2f}$ m"

        out_pdf = OUT_DIR / f"{sh}_heatmap.pdf"
        out_png = OUT_DIR / f"{sh}_heatmap.png"

        plot_heatmap(x, y, W, title, out_pdf, mask_zeros=MASK_ZEROS_AS_NODATA)
        plot_heatmap(x, y, W, title, out_png, mask_zeros=MASK_ZEROS_AS_NODATA)

    print(f"Saved figures to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()