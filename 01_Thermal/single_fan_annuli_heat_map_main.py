"""
Plot annuli heat maps using the same plotting style as single_fan_heat_map.py.

For each height sheet, annuli are plotted as rings using the radii stored in
the annuli profile CSVs (r_m). This ensures the plot uses the exact r values
from annuli_cut.py, while keeping the same x/y axis settings as the original
heat map.
"""

###### Initialization

### Imports
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Circle, Rectangle, Wedge
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cmocean  # https://matplotlib.org/cmocean

### User settings
XLSX_PATH = "S01.xlsx"
SHEETS = ["z020", "z035", "z050", "z075", "z110", "z160", "z220"]

OUT_DIR = Path("A_figures/Single_Fan_Annuli_Heat_Map")
OUT_DIR.mkdir(exist_ok=True)

ANNULI_PROFILE_DIR = Path("B_results/Single_Fan_Annuli_Profile")

MASK_ZEROS_AS_NODATA = False

# Annulus thickness (m)
# Used only as a fallback if spacing cannot be inferred from r_m.
DELTA_R_M = 0.30

# Use median rather than mean within each annulus
USE_MEDIAN_PROFILE = False

# Fan centre (x_c, y_c)
FAN_CENTER_XY = (4.2, 2.4)

# Units / labels
CBAR_LABEL = r"$w$ (m$\cdot$s$^{-1}$)"  # vertical velocity
XLABEL = r"$x$ (m)"
YLABEL = r"$y$ (m)"

# Tick positions for compact A4 layout
# Line widths
CELL_EDGE_LW = 0.30
AXIS_EDGE_LW = 0.80
CBAR_EDGE_LW = AXIS_EDGE_LW
LEGEND_FONTSIZE = 8.5

# Exponential opacity mapping versus normalized w (= 0..1).
# alpha(0) = 0 (fully transparent), alpha(1) = 1 (fully opaque).
ALPHA_EXP_RATE = 0.005

# Fan outlet marker (single fan)
FAN_OUTLET_X = 4.2
FAN_OUTLET_Y = 2.4
FAN_OUTLET_DIAMETER = 0.8
FAN_OUTLET_EDGE_LW = 1.1
FAN_OUTLET_ALPHA = 0.6
FAN_OUTLET_DASH = (0, (2, 2))


# Helpers
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


def build_annuli_bins(
    x: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
    fan_center_xy: Tuple[float, float],
    delta_r: float,
    use_median: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute annulus centres and their aggregated values (fallback).
    """
    if delta_r <= 0.0:
        raise ValueError("delta_r must be positive.")

    W_work = W.copy()
    if MASK_ZEROS_AS_NODATA:
        W_work[W_work == 0.0] = np.nan

    x_grid, y_grid = np.meshgrid(x, y)
    xc, yc = fan_center_xy
    r = np.sqrt((x_grid - xc) ** 2 + (y_grid - yc) ** 2)

    valid = np.isfinite(W_work) & np.isfinite(r)
    if not np.any(valid):
        raise ValueError("No finite samples available to construct annuli map.")

    r_valid = r[valid]
    w_valid = W_work[valid]

    # Nearest-centre binning (matches annuli_cut.py)
    k = np.floor(r_valid / delta_r + 0.5).astype(int)
    r_bins = k.astype(float) * float(delta_r)

    uniq_k = np.unique(k)
    r_list = []
    w_list = []
    for kk in uniq_k:
        in_bin = k == kk
        w_slice = w_valid[in_bin]
        if w_slice.size == 0:
            continue
        r_list.append(float(kk) * float(delta_r))
        w_list.append(float(np.median(w_slice) if use_median else np.mean(w_slice)))

    r_bins = np.array(r_list, dtype=float)
    w_bins = np.array(w_list, dtype=float)
    order = np.argsort(r_bins)
    return r_bins[order], w_bins[order]


def load_annuli_profile_csv(profile_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load annuli profile CSV (r_m, w_mps).
    """
    df = pd.read_csv(profile_path)
    r_bins = pd.to_numeric(df["r_m"], errors="coerce").to_numpy(dtype=float)
    w_bins = pd.to_numeric(df["w_mps"], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(r_bins) & np.isfinite(w_bins)
    r_bins = r_bins[mask]
    w_bins = w_bins[mask]

    order = np.argsort(r_bins)
    return r_bins[order], w_bins[order]


def infer_delta_r(r_bins: np.ndarray, fallback: float) -> float:
    """
    Infer annulus spacing from r_bins. Uses the smallest positive gap as base delta_r.
    """
    if r_bins.size < 2:
        return float(fallback)

    r_unique = np.unique(r_bins)
    if r_unique.size < 2:
        return float(fallback)

    diffs = np.diff(r_unique)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if diffs.size == 0:
        return float(fallback)

    return float(np.min(diffs))


def plot_annuli(x, y, r_bins, w_bins, delta_r: float, outpath: Path):
    """
    Plot annuli as rings using the same axis settings as single_fan_heat_map.py.
    """
    if delta_r <= 0.0:
        raise ValueError("delta_r must be positive.")

    # Convert center grids -> edges for axis settings
    x_edges = centers_to_edges(x)
    y_edges = centers_to_edges(y)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    # Figure styling
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

    fig, ax = plt.subplots(figsize=(5.7, 3.9), dpi=600)  # 2-per-row on A4 landscape

    # Annulus rings
    vmin = 0.0
    vmax = 8.0
    cmap_alpha = build_alpha_cmap()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # Background fill for empty areas (w = 0)
    bg = Rectangle(
        (x_edges[0], y_edges[0]),
        x_edges[-1] - x_edges[0],
        y_edges[-1] - y_edges[0],
        facecolor=cmap_alpha(norm(0.0)),
        edgecolor="none",
        zorder=0,
    )
    ax.add_patch(bg)

    for r_c, w_val in zip(r_bins, w_bins):
        if not np.isfinite(w_val):
            continue

        r_in = max(float(r_c) - 0.5 * float(delta_r), 0.0)
        r_out = float(r_c) + 0.5 * float(delta_r)
        if r_out <= r_in:
            continue

        ring = Wedge(
            (FAN_OUTLET_X, FAN_OUTLET_Y),
            r_out,
            0.0,
            360.0,
            width=r_out - r_in,
            facecolor=cmap_alpha(norm(w_val)),
            edgecolor=(0, 0, 0, 0.3),
            linewidth=CELL_EDGE_LW,
            clip_on=True,
        )
        ax.add_patch(ring)

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
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap_alpha)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, cax=cax)
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
    # Fixed axis ticks (no centering) with 2-decimal labels
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
        leg.get_frame().set_linewidth(AXIS_EDGE_LW)

    # Axis limits start at (0, 0)
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
    for sh in SHEETS:
        x, y, W = read_slice_from_sheet(XLSX_PATH, sh)
        profile_path = ANNULI_PROFILE_DIR / f"{sh}_annuli_profile.csv"
        if profile_path.exists():
            r_bins, w_bins = load_annuli_profile_csv(profile_path)
        else:
            r_bins, w_bins = build_annuli_bins(
                x=x,
                y=y,
                W=W,
                fan_center_xy=FAN_CENTER_XY,
                delta_r=DELTA_R_M,
                use_median=USE_MEDIAN_PROFILE,
            )

        delta_r = infer_delta_r(r_bins, DELTA_R_M)
        out_png = OUT_DIR / f"{sh}_single_annuli_heatmap_main.png"
        plot_annuli(x, y, r_bins, w_bins, delta_r, out_png)

    print(f"Saved figures to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()



