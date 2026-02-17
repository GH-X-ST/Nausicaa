"""
Plot four-fan annuli heat maps in compact main-figure style.

Field reconstruction uses nearest-fan distance r(x, y) and annuli profile
values w(r) from B_results/Four_Fan_Annuli_Profile/*_four_annuli_profile.csv.
"""

###### Initialization

### Imports
from pathlib import Path
from typing import Sequence, Tuple

import cmocean
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Rectangle, Wedge
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

### User settings
XLSX_PATH = "S02.xlsx"
SHEETS = ["z020", "z035", "z050", "z075", "z110", "z160", "z220"]

OUT_DIR = Path("A_figures/Four_Fan_Annuli_Heat_Map")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANNULI_PROFILE_DIR = Path("B_results/Four_Fan_Annuli_Profile")

MASK_ZEROS_AS_NODATA = False

# Annulus thickness (m)
# Used as fallback when spacing cannot be inferred from r_m.
DELTA_R_M = 0.30

FOUR_FAN_CENTERS_XY = (
    (3.0, 3.6),
    (5.4, 3.6),
    (3.0, 1.2),
    (5.4, 1.2),
)

# Units / labels
CBAR_LABEL = r"$w$ (m$\cdot$s$^{-1}$)"
XLABEL = r"$x$ (m)"
YLABEL = r"$y$ (m)"

# Line widths
CELL_EDGE_LW = 0.30
AXIS_EDGE_LW = 0.80
CBAR_EDGE_LW = AXIS_EDGE_LW
LEGEND_FONTSIZE = 8.5

# Exponential opacity mapping versus normalized w (= 0..1).
# alpha(0) = 0 (fully transparent), alpha(1) = 1 (fully opaque).
ALPHA_EXP_RATE = 0.005

# Fan outlet markers (four fan)
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


def build_alpha_cmap() -> mcolors.ListedColormap:
    """Build a thermal colormap with exponential alpha versus normalized w."""
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
    """Convert 1D array of cell centers to cell edges."""
    c = np.asarray(c, dtype=float)
    if c.size < 2:
        raise ValueError("Need at least 2 center points to compute edges.")
    edges = np.empty(c.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (c[:-1] + c[1:])
    edges[0] = c[0] - 0.5 * (c[1] - c[0])
    edges[-1] = c[-1] + 0.5 * (c[-1] - c[-2])
    return edges


def read_slice_from_sheet(xlsx_path: str, sheet_name: str):
    """Read one grid sheet and return x_centers, y_centers, W (Ny x Nx)."""
    raw = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)

    x = pd.to_numeric(raw.iloc[0, 1:], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(raw.iloc[1:, 0], errors="coerce").to_numpy(dtype=float)
    w = raw.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    if w.shape != (y.size, x.size):
        raise ValueError(
            f"Shape mismatch in {sheet_name}: W{w.shape}, y({y.size}), x({x.size})."
        )

    if y.size >= 2 and y[0] > y[-1]:
        y = y[::-1]
        w = w[::-1, :]

    return x, y, w


def load_annuli_profile_csv(profile_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load annuli profile CSV (r_m, w_mps)."""
    df = pd.read_csv(profile_path)
    r_bins = pd.to_numeric(df["r_m"], errors="coerce").to_numpy(dtype=float)
    w_bins = pd.to_numeric(df["w_mps"], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(r_bins) & np.isfinite(w_bins)
    r_bins = r_bins[mask]
    w_bins = w_bins[mask]

    if r_bins.size == 0:
        raise ValueError(f"No valid annuli rows in {profile_path}.")

    order = np.argsort(r_bins)
    return r_bins[order], w_bins[order]


def infer_delta_r(r_bins: np.ndarray, fallback: float) -> float:
    """Infer annulus spacing from r_bins."""
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


def reconstruct_four_fan_annuli_field(
    x: np.ndarray,
    y: np.ndarray,
    r_bins: np.ndarray,
    w_bins: np.ndarray,
    fan_centers_xy: Sequence[Tuple[float, float]],
) -> np.ndarray:
    """Reconstruct w(x, y) from radial profile using nearest-fan radius."""
    x_grid, y_grid = np.meshgrid(x, y)
    fan_xy = np.asarray(fan_centers_xy, dtype=float)

    d_sample_fan = np.sqrt(
        (x_grid[:, :, None] - fan_xy[None, None, :, 0]) ** 2
        + (y_grid[:, :, None] - fan_xy[None, None, :, 1]) ** 2
    )
    r_nearest = np.min(d_sample_fan, axis=2)

    idx = np.argmin(np.abs(r_nearest[:, :, None] - r_bins[None, None, :]), axis=2)
    w_annuli = w_bins[idx]
    return w_annuli.astype(float)


def nearest_fan_radius_and_index(
    x: np.ndarray,
    y: np.ndarray,
    fan_centers_xy: Sequence[Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return nearest-fan radius map and nearest-fan index map."""
    x_grid, y_grid = np.meshgrid(x, y)
    fan_xy = np.asarray(fan_centers_xy, dtype=float)

    d_sample_fan = np.sqrt(
        (x_grid[:, :, None] - fan_xy[None, None, :, 0]) ** 2
        + (y_grid[:, :, None] - fan_xy[None, None, :, 1]) ** 2
    )
    nearest_idx = np.argmin(d_sample_fan, axis=2)
    nearest_r = np.min(d_sample_fan, axis=2)
    return nearest_r, nearest_idx


def annulus_boundary_levels(r_bins: np.ndarray) -> np.ndarray:
    """Return mid-point radii between consecutive annulus centers."""
    r_unique = np.unique(np.asarray(r_bins, dtype=float))
    if r_unique.size < 2:
        return np.asarray([], dtype=float)
    return 0.5 * (r_unique[:-1] + r_unique[1:])


def plot_annuli(
    x: np.ndarray,
    y: np.ndarray,
    r_bins: np.ndarray,
    w_bins: np.ndarray,
    delta_r: float,
    outpath: Path,
) -> None:
    """Plot full annuli rings for all four fans (overlap allowed)."""
    if delta_r <= 0.0:
        raise ValueError("delta_r must be positive.")

    x_edges = centers_to_edges(x)
    y_edges = centers_to_edges(y)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": LEGEND_FONTSIZE,
            "axes.edgecolor": "k",
            "axes.linewidth": AXIS_EDGE_LW,
            "patch.edgecolor": "k",
        }
    )

    fig, ax = plt.subplots(figsize=(5.7, 3.9), dpi=600)

    vmin = 0.0
    vmax = 8.0
    cmap_alpha = build_alpha_cmap()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

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
        leg.get_frame().set_linewidth(AXIS_EDGE_LW)

    ax.set_xlim(0.0, 8.4)
    ax.set_ylim(0.0, 4.8)

    fig.tight_layout()
    ax_pos = ax.get_position()
    cax_pos = cax.get_position()
    new_h = ax_pos.height * 0.82
    new_y0 = ax_pos.y0
    cax.set_position([cax_pos.x0, new_y0, cax_pos.width, new_h])

    fig.savefig(outpath, bbox_inches="tight", facecolor="white", dpi=600)
    plt.close(fig)


### Export each sheet as PNG
def main() -> None:
    for sh in SHEETS:
        x, y, _w_raw = read_slice_from_sheet(XLSX_PATH, sh)
        profile_path = ANNULI_PROFILE_DIR / f"{sh}_four_annuli_profile.csv"
        if not profile_path.exists():
            raise FileNotFoundError(f"Missing annuli profile CSV: {profile_path}")

        r_bins, w_bins = load_annuli_profile_csv(profile_path)
        delta_r = infer_delta_r(r_bins, DELTA_R_M)
        out_png = OUT_DIR / f"{sh}_four_annuli_heatmap_main.png"
        plot_annuli(x, y, r_bins, w_bins, delta_r, out_png)

    print(f"Saved figures to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()



