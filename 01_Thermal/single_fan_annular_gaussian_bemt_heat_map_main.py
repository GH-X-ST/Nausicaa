"""
Plot non-axisymmetric annular-Gaussian BEMT heat maps.

For each height sheet, fitted non-axisymmetric parameters are loaded from
B_results/single_annular_bemt_params.xlsx and used to generate a continuous
model field w(x, y) on a dense uniform grid.

Model:
    theta = atan2(y - y_c, x - x_c)
    r     = sqrt((x - x_c)^2 + (y - y_c)^2)
    g(r)  = exp(-((r - r_ring) / delta_ring)^2)

    A(theta) = a0 + sum_{n=1..N}(a_n cos(n theta) + b_n sin(n theta))
    w_model(x, y) = w0 + g(r) * A(theta)
"""

###### Initialization

### Imports
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cmocean  # https://matplotlib.org/cmocean

### User settings
XLSX_PATH = "S01.xlsx"
SHEETS = ["z020", "z035", "z050", "z075", "z110", "z160", "z220"]

OUT_DIR = Path("A_figures/Single_Fan_Annular_Gaussian_BEMT")
OUT_DIR.mkdir(exist_ok=True)

PARAMS_XLSX = Path("B_results/single_annular_bemt_params.xlsx")
PARAMS_SHEET = "single_bemt_az_fit"

# Fan centre (x_c, y_c)
FAN_CENTER_XY = (4.2, 2.4)

# Units / labels
CBAR_LABEL = r"$w$ (m $\!$s$^{-1}$)"  # vertical velocity
XLABEL = r"$x$ (m)"
YLABEL = r"$y$ (m)"

# Tick positions for compact A4 layout
# Line widths

AXIS_EDGE_LW = 0.80
CBAR_EDGE_LW = AXIS_EDGE_LW
GRID_COLOR = (0.85, 0.85, 0.85, 1.0)
GRID_LINEWIDTH = 0.4


# Exponential opacity mapping versus normalized w (= 0..1).
# alpha(0) = 0 (fully transparent), alpha(1) = 1 (fully opaque).
ALPHA_EXP_RATE = 0.005
LEGEND_FONTSIZE = 8.5

# Fan outlet marker (single fan)
FAN_OUTLET_X = 4.2
FAN_OUTLET_Y = 2.4
FAN_OUTLET_DIAMETER = 0.8
FAN_OUTLET_EDGE_LW = 0.7
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
    w_map = raw.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    # sanity checks
    if w_map.shape != (y.size, x.size):
        raise ValueError(
            f"Shape mismatch in {sheet_name}: W{w_map.shape}, y({y.size}), x({x.size})."
        )

    # Ensure y increases bottom-to-top on the plot (0 -> max)
    if y.size >= 2 and y[0] > y[-1]:
        y = y[::-1]
        w_map = w_map[::-1, :]

    return x, y, w_map


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


def discover_param_columns(df: pd.DataFrame) -> List[str]:
    """
    Discover parameter columns:
        [w0, r_ring, delta_ring, a0, a1, b1, ..., aN, bN]
    """
    base = ["w0", "r_ring", "delta_ring", "a0"]
    missing = [col for col in base if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required parameter columns: {missing}")

    a_orders = []
    b_orders = []
    for col in df.columns:
        if col.startswith("a") and col[1:].isdigit():
            order = int(col[1:])
            if order >= 1:
                a_orders.append(order)
        if col.startswith("b") and col[1:].isdigit():
            order = int(col[1:])
            if order >= 1:
                b_orders.append(order)

    harmonic_orders = sorted(set(a_orders).intersection(set(b_orders)))

    param_cols = list(base)
    for n_idx in harmonic_orders:
        param_cols.append(f"a{n_idx}")
        param_cols.append(f"b{n_idx}")

    return param_cols


def load_bemt_params(
    xlsx_path: Path,
    sheet_name: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load fitted non-axisymmetric BEMT parameters from Excel.
    """
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Missing fitted-parameter file: {xlsx_path}")

    xls = pd.ExcelFile(xlsx_path)
    sheet_to_use = sheet_name if sheet_name in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(xlsx_path, sheet_name=sheet_to_use)

    param_cols = discover_param_columns(df)

    for col in ["z_m"] + param_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=param_cols).copy()
    if df.empty:
        raise ValueError("No valid fitted rows in fitted-parameter table.")

    return df, param_cols


def params_for_height(
    df: pd.DataFrame,
    param_cols: List[str],
    sheet_name: str,
) -> np.ndarray:
    """
    Select parameter vector for requested sheet.

    Preference:
    1) exact 'sheet' match (if available)
    2) nearest z_m to parsed sheet height
    """
    if "sheet" in df.columns:
        series = df["sheet"].astype(str).str.strip().str.lower()
        mask = series == sheet_name.strip().lower()
        if np.any(mask):
            row = df.loc[mask].iloc[0]
            return np.array([float(row[c]) for c in param_cols], dtype=float)

    if "z_m" not in df.columns:
        raise ValueError("Fitted table has neither matching 'sheet' nor 'z_m'.")

    z_m = parse_sheet_height_m(sheet_name)
    z_vals = pd.to_numeric(df["z_m"], errors="coerce").to_numpy(dtype=float)
    if not np.any(np.isfinite(z_vals)):
        raise ValueError("No finite z_m values in fitted table.")

    idx = int(np.nanargmin(np.abs(z_vals - z_m)))
    row = df.iloc[idx]
    return np.array([float(row[c]) for c in param_cols], dtype=float)


def evaluate_model(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    params: np.ndarray,
    fan_center_xy: Tuple[float, float],
) -> np.ndarray:
    """
    Evaluate non-axisymmetric annular-Gaussian model on a grid.
    """
    w0 = float(params[0])
    r_ring = float(params[1])
    delta_ring = float(max(params[2], 1e-12))
    coeffs = params[3:]

    xc, yc = fan_center_xy
    r = np.sqrt((x_grid - xc) ** 2 + (y_grid - yc) ** 2)
    theta = np.arctan2(y_grid - yc, x_grid - xc)

    amp = np.full_like(theta, float(coeffs[0]), dtype=float)  # a0
    fourier_order = (coeffs.size - 1) // 2
    for n_idx in range(1, fourier_order + 1):
        a_n = float(coeffs[2 * n_idx - 1])
        b_n = float(coeffs[2 * n_idx])
        amp += a_n * np.cos(n_idx * theta) + b_n * np.sin(n_idx * theta)

    g_r = np.exp(-((r - r_ring) / delta_ring) ** 2)
    return w0 + g_r * amp


def plot_continuous_heatmap(x, y, w_field, outpath: Path):
    """
    Plot continuous model heat map using the same axis settings as annuli heat maps.
    """
    # Convert center grids -> edges for pcolormesh
    x_edges = centers_to_edges(x)
    y_edges = centers_to_edges(y)

    # Figure styling
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

    fig, ax = plt.subplots(figsize=(5.2, 3.0), dpi=600)  # 2-per-row on A4 landscape

    # Continuous heatmap
    cmap_alpha = build_alpha_cmap()
    im = ax.pcolormesh(
        x_edges,
        y_edges,
        w_field,
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

    # Axes
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_aspect("equal", adjustable="box")  # 1:1 grid without stretching
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID_COLOR, linewidth=GRID_LINEWIDTH)
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
def main() -> None:
    params_df, param_cols = load_bemt_params(PARAMS_XLSX, PARAMS_SHEET)

    for sh in SHEETS:
        x, y, _w = read_slice_from_sheet(XLSX_PATH, sh)

        params = params_for_height(params_df, param_cols, sh)

        x_grid, y_grid = build_continuous_grid(x, y)
        w_model = evaluate_model(x_grid, y_grid, params, fan_center_xy=FAN_CENTER_XY)

        out_png = OUT_DIR / f"{sh}_single_annular_gaussian_bemt_heatmap_main.png"
        plot_continuous_heatmap(
            x_grid[0, :],
            y_grid[:, 0],
            w_model,
            outpath=out_png,
        )

    print(f"Saved figures to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()












