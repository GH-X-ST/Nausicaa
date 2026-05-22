"""
Plot annuli heat maps using a structure aligned with single_fan_annuli_heat_map.py.

For four fans, each grid point uses nearest-fan radius:
    r(x, y) = min_f ||[x, y] - c_f||_2
and w(x, y) is assigned from annuli profile (r_m, w_mps) by nearest r_m.
"""


from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Circle, Rectangle, Wedge
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

OUT_DIR = Path("A_figures/Four_Fan_Annuli_Heat_Map")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANNULI_PROFILE_DIR = Path("B_results/Four_Fan_Annuli_Profile")

MASK_ZEROS_AS_NODATA = False

# Annulus thickness (m)
# Used only as a fallback if spacing cannot be inferred from r_m.
DELTA_R_M = 0.30

# Median aggregation is an optional robustness choice for outlier-prone annuli.
USE_MEDIAN_PROFILE = False

FOUR_FAN_CENTERS_XY = (
    (3.0, 3.6),
    (5.4, 3.6),
    (3.0, 1.2),
    (5.4, 1.2),
)

# Axis and colorbar labels use metres and metres per second in exported figures.
CBAR_LABEL = r"$w$ (m $\!$s$^{-1}$)"
XLABEL = r"$x$ (m)"
YLABEL = r"$y$ (m)"

# Line widths are fixed for figure-to-figure comparability.
CELL_EDGE_LW = 0.30
AXIS_EDGE_LW = 0.80
CBAR_EDGE_LW = AXIS_EDGE_LW

# Exponential opacity mapping versus normalized w (= 0..1).
# Alpha maps normalized velocity monotonically so weak updraft remains visually subordinate.
ALPHA_EXP_RATE = 0.005

# Fan outlet markers in arena coordinates (m).
FAN_OUTLET_POINTS = [
    (3.0, 3.6),
    (5.4, 3.6),
    (5.4, 1.2),
    (3.0, 1.2),
]
FAN_OUTLET_DIAMETER = 0.8
FAN_OUTLET_EDGE_LW = 1.1
FAN_OUTLET_ALPHA = 0.6
FAN_OUTLET_DASH = (0, (2, 2))

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
    w = raw.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    # Workbook grid shape must match y-by-x coordinates.
    if w.shape != (y.size, x.size):
        raise ValueError(
            f"Shape mismatch in {sheet_name}: W{w.shape}, y({y.size}), x({x.size})."
        )

    # Plot convention uses increasing arena y from bottom to top.
    if y.size >= 2 and y[0] > y[-1]:
        y = y[::-1]
        w = w[::-1, :]

    return x, y, w


# Annuli CSV files are derived measured profiles and must stay traceable to source sheets.
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


# Nearest-fan radius maps each display point to the controlling plume centre.
def _nearest_fan_radius_map(
    x: np.ndarray,
    y: np.ndarray,
    fan_centers_xy: Tuple[Tuple[float, float], ...],
) -> np.ndarray:
    """
    Return nearest-fan radius map r(x, y) for each grid point.
    """
    x_grid, y_grid = np.meshgrid(x, y)
    fan_xy = np.asarray(fan_centers_xy, dtype=float)

    d_sample_fan = np.sqrt(
        (x_grid[:, :, None] - fan_xy[None, None, :, 0]) ** 2
        + (y_grid[:, :, None] - fan_xy[None, None, :, 1]) ** 2
    )
    return np.min(d_sample_fan, axis=2)


# Radius and fan-index maps preserve which plume owns each annular bin.
def _nearest_fan_radius_and_index(
    x: np.ndarray,
    y: np.ndarray,
    fan_centers_xy: Tuple[Tuple[float, float], ...],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return nearest-fan radius map and nearest-fan index map.
    """
    x_grid, y_grid = np.meshgrid(x, y)
    fan_xy = np.asarray(fan_centers_xy, dtype=float)

    d_sample_fan = np.sqrt(
        (x_grid[:, :, None] - fan_xy[None, None, :, 0]) ** 2
        + (y_grid[:, :, None] - fan_xy[None, None, :, 1]) ** 2
    )
    nearest_idx = np.argmin(d_sample_fan, axis=2)
    nearest_r = np.min(d_sample_fan, axis=2)
    return nearest_r, nearest_idx


# Boundary levels expose annular bin edges for plotting the fitted support.
def _annulus_boundary_levels(r_bins: np.ndarray) -> np.ndarray:
    """
    Return mid-point radii between consecutive annulus centers.
    """
    r_unique = np.unique(np.asarray(r_bins, dtype=float))
    if r_unique.size < 2:
        return np.asarray([], dtype=float)
    return 0.5 * (r_unique[:-1] + r_unique[1:])


# Annulus bins define the radial support used when reconstructing profile-based fields.
def build_annuli_bins(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    fan_centers_xy: Tuple[Tuple[float, float], ...],
    delta_r: float,
    use_median: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute annulus centres and their aggregated values (fallback).
    """
    if delta_r <= 0.0:
        raise ValueError("delta_r must be positive.")

    w_work = w.copy()
    if MASK_ZEROS_AS_NODATA:
        w_work[w_work == 0.0] = np.nan

    r_map = _nearest_fan_radius_map(x=x, y=y, fan_centers_xy=fan_centers_xy)

    valid = np.isfinite(w_work) & np.isfinite(r_map)
    if not np.any(valid):
        raise ValueError("No finite samples available to construct annuli map.")

    r_valid = r_map[valid]
    w_valid = w_work[valid]

    # Nearest-centre binning (matches annuli_cut.py)
    k = np.floor(r_valid / delta_r + 0.5).astype(int)

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


# Annulus width is inferred from stored bin centres to preserve the profile convention.
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


# Four-fan reconstruction assigns each point to the nearest fan annulus before summing fields.
def reconstruct_four_fan_annuli_field(
    x: np.ndarray,
    y: np.ndarray,
    r_bins: np.ndarray,
    w_bins: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct w(x, y) from annuli profile using nearest-fan radius.
    """
    r_map = _nearest_fan_radius_map(x=x, y=y, fan_centers_xy=FOUR_FAN_CENTERS_XY)
    idx = np.argmin(np.abs(r_map[:, :, None] - r_bins[None, None, :]), axis=2)
    return w_bins[idx].astype(float)


# Annulus plots reconstruct radial summaries using the same fan-centred distance convention as fitting.
def plot_annuli(x, y, r_bins, w_bins, delta_r: float, outpath: Path):
    """
    Plot annuli map in the same axis/tick style as single_fan_annuli_heat_map.py.
    """
    if delta_r <= 0.0:
        raise ValueError("delta_r must be positive.")

    # Cell edges keep axis limits tied to measured sample centres.
    x_edges = centers_to_edges(x)
    y_edges = centers_to_edges(y)

    # White background and hidden top/right spines match the thesis figure style.
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.edgecolor": "k",
            "axes.linewidth": AXIS_EDGE_LW,
            "patch.edgecolor": "k",
        }
    )

    # Figure size leaves room for raw-cell annotations without changing data scale.
    fig, ax = plt.subplots(figsize=(6.8, 5.6), dpi=600)

    # Annular rings show the radial binning used for profile construction.
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

    for fx, fy in FAN_OUTLET_POINTS:
        for r_c, w_val in zip(r_bins, w_bins):
            if not np.isfinite(w_val):
                continue

            r_in = max(float(r_c) - 0.5 * float(delta_r), 0.0)
            r_out = float(r_c) + 0.5 * float(delta_r)
            if r_out <= r_in:
                continue

            ring = Wedge(
                (fx, fy),
                r_out,
                0.0,
                360.0,
                width=r_out - r_in,
                facecolor=cmap_alpha(norm(w_val)),
                edgecolor=(0, 0, 0, 0.3),
                linewidth=CELL_EDGE_LW,
                clip_on=True,
                zorder=1,
            )
            ax.add_patch(ring)

    # Dashed fan outlines anchor the arena-frame fan geometry.
    for i, (fx, fy) in enumerate(FAN_OUTLET_POINTS):
        outlet = Circle(
            (fx, fy),
            radius=FAN_OUTLET_DIAMETER / 2.0,
            fill=False,
            edgecolor=(0, 0, 0, FAN_OUTLET_ALPHA),
            linewidth=FAN_OUTLET_EDGE_LW,
            linestyle=FAN_OUTLET_DASH,
            label="Fan outlet" if i == 0 else None,
            zorder=5,
        )
        ax.add_patch(outlet)

    # Colorbar ticks use the fixed velocity scale for cross-figure comparison.
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

    # Equal aspect preserves arena geometry in plan view.
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    # Equal aspect keeps arena distances physically meaningful in the rendered plot.
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_EDGE_LW)

    # Fixed ticks keep spatial comparisons aligned across exported figures.
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

    # Axis limits match the measured domain instead of padded defaults.
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
def main() -> None:
    for sh in SHEETS:
        x, y, w = read_slice_from_sheet(XLSX_PATH, sh)
        profile_path = ANNULI_PROFILE_DIR / f"{sh}_four_annuli_profile.csv"
        if profile_path.exists():
            r_bins, w_bins = load_annuli_profile_csv(profile_path)
        else:
            r_bins, w_bins = build_annuli_bins(
                x=x,
                y=y,
                w=w,
                fan_centers_xy=FOUR_FAN_CENTERS_XY,
                delta_r=DELTA_R_M,
                use_median=USE_MEDIAN_PROFILE,
            )

        delta_r = infer_delta_r(r_bins, DELTA_R_M)
        out_png = OUT_DIR / f"{sh}_four_annuli_heatmap.png"
        plot_annuli(x, y, r_bins, w_bins, delta_r, out_png)

    print(f"Saved figures to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()


