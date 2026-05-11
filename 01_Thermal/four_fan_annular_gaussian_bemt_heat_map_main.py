"""
Plot non-axisymmetric annular-Gaussian BEMT heat maps.

For each height sheet, fitted non-axisymmetric parameters are loaded from
B_results/four_annular_bemt_params.xlsx and used to generate a continuous
model field w(x, y) on a dense uniform grid.

Model per fan:
    theta = atan2(y - y_f, x - x_f)
    r     = sqrt((x - x_f)^2 + (y - y_f)^2)
    g(r)  = exp(-((r - r_ring) / delta_ring)^2)

    A(theta) = a0 + sum_{n=1..N}(a_n cos(n theta) + b_n sin(n theta))
    w_f(x, y) = w0 + g(r) * A(theta)

Total model:
    w_model(x, y) = sum_f w_f(x, y)
"""


from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cmocean  # https://matplotlib.org/cmocean


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

XLSX_PATH = "S02.xlsx"
SHEETS = ["z020", "z035", "z050", "z075", "z110", "z160", "z220"]

OUT_DIR = Path("A_figures/Four_Fan_Annular_Gaussian_BEMT")
OUT_DIR.mkdir(exist_ok=True)

PARAMS_XLSX = Path("B_results/four_annular_bemt_params.xlsx")
PARAMS_SHEET = "four_bemt_az_fit"

FOUR_FAN_CENTERS_XY = (
    (3.0, 3.6),
    (5.4, 3.6),
    (3.0, 1.2),
    (5.4, 1.2),
)

# Axis and colorbar units used in exported figures.
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
FAN_COL_PATTERN = re.compile(r"^a0_(F\d{2})$")


# =============================================================================
# 2) Workbook Loading and Plot Construction
# =============================================================================

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
    Reads the grid sheet:
      - row 0, col 1.. = x coordinates
      - col 0, row 1.. = y coordinates
      - interior = scalar field values
    Returns x_centers, y_centers, W (Ny x Nx).
    """
    raw = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)

    x = pd.to_numeric(raw.iloc[0, 1:], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(raw.iloc[1:, 0], errors="coerce").to_numpy(dtype=float)
    w_map = raw.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    if w_map.shape != (y.size, x.size):
        raise ValueError(
            f"Shape mismatch in {sheet_name}: W{w_map.shape}, y({y.size}), x({x.size})."
        )

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
    x_grid, y_grid = np.meshgrid(x_lin, y_lin)
    return x_grid, y_grid


def load_bemt_params(xlsx_path: Path, sheet_name: str) -> pd.DataFrame:
    """
    Load fitted non-axisymmetric BEMT parameters from Excel.
    """
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Missing fitted-parameter file: {xlsx_path}")

    xls = pd.ExcelFile(xlsx_path)
    chosen_sheet = sheet_name if sheet_name in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(xlsx_path, sheet_name=chosen_sheet)

    if "z_m" not in df.columns:
        raise ValueError("Missing required column: z_m")

    for col in df.columns:
        if col != "sheet":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["z_m"]).copy()
    if df.empty:
        raise ValueError("No valid parameter rows in fitted-parameter table.")

    return df


def discover_fan_ids(df: pd.DataFrame) -> Tuple[str, ...]:
    """
    Discover fan IDs from columns like a0_F01.
    """
    fan_ids = []
    for col in df.columns:
        match = FAN_COL_PATTERN.match(str(col))
        if match is not None:
            fan_ids.append(match.group(1))

    fan_ids = sorted(set(fan_ids))
    valid = []
    for fan_id in fan_ids:
        required = (
            f"w0_{fan_id}",
            f"r_ring_{fan_id}",
            f"delta_ring_{fan_id}",
            f"a0_{fan_id}",
        )
        if all(col in df.columns for col in required):
            valid.append(fan_id)

    return tuple(valid)


def discover_shared_param_columns(df: pd.DataFrame) -> Tuple[List[str], List[int]]:
    """
    Discover shared parameter columns and harmonic orders.
    """
    base = ["w0", "r_ring", "delta_ring", "a0"]
    missing = [col for col in base if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required shared parameter columns: {missing}")

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

    return param_cols, harmonic_orders


def discover_fan_param_columns(df: pd.DataFrame, fan_id: str) -> Tuple[List[str], List[int]]:
    """
    Discover per-fan parameter columns and harmonic orders.
    """
    base = [f"w0_{fan_id}", f"r_ring_{fan_id}", f"delta_ring_{fan_id}", f"a0_{fan_id}"]
    missing = [col for col in base if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {fan_id}: {missing}")

    a_orders = []
    b_orders = []
    suffix = f"_{fan_id}"
    for col in df.columns:
        if col.startswith("a") and col.endswith(suffix):
            core = col[1 : -len(suffix)]
            if core.isdigit() and int(core) >= 1:
                a_orders.append(int(core))
        if col.startswith("b") and col.endswith(suffix):
            core = col[1 : -len(suffix)]
            if core.isdigit() and int(core) >= 1:
                b_orders.append(int(core))

    harmonic_orders = sorted(set(a_orders).intersection(set(b_orders)))
    param_cols = list(base)
    for n_idx in harmonic_orders:
        param_cols.append(f"a{n_idx}_{fan_id}")
        param_cols.append(f"b{n_idx}_{fan_id}")

    return param_cols, harmonic_orders


def select_row_for_sheet(df: pd.DataFrame, sheet_name: str) -> pd.Series:
    """
    Select parameter row for a requested sheet.

    Preference:
    1) exact 'sheet' match (if available)
    2) nearest z_m to parsed sheet height
    """
    if "sheet" in df.columns:
        series = df["sheet"].astype(str).str.strip().str.lower()
        mask = series == sheet_name.strip().lower()
        if np.any(mask):
            return df.loc[mask].iloc[0]

    z_m = parse_sheet_height_m(sheet_name)
    z_vals = pd.to_numeric(df["z_m"], errors="coerce").to_numpy(dtype=float)
    if not np.any(np.isfinite(z_vals)):
        raise ValueError("No finite z_m values in fitted table.")

    idx = int(np.nanargmin(np.abs(z_vals - z_m)))
    return df.iloc[idx]


def params_for_height(
    df: pd.DataFrame,
    sheet_name: str,
    fan_ids: Tuple[str, ...],
) -> Tuple[List[Dict[str, float]], List[List[int]]]:
    """
    Extract per-fan model parameters for the requested sheet.
    """
    row = select_row_for_sheet(df, sheet_name)

    if len(fan_ids) == 0:
        param_cols, harmonic_orders = discover_shared_param_columns(df)
        shared = {name: float(row[name]) for name in param_cols}

        params_list = []
        orders_list = []
        for _ in FOUR_FAN_CENTERS_XY:
            params_list.append(shared.copy())
            orders_list.append(list(harmonic_orders))

        return params_list, orders_list

    params_list = []
    orders_list = []
    for fan_id in fan_ids:
        _, harmonic_orders = discover_fan_param_columns(df, fan_id)
        params = {
            "w0": float(row[f"w0_{fan_id}"]),
            "r_ring": float(row[f"r_ring_{fan_id}"]),
            "delta_ring": float(row[f"delta_ring_{fan_id}"]),
            "a0": float(row[f"a0_{fan_id}"]),
        }
        for n_idx in harmonic_orders:
            params[f"a{n_idx}"] = float(row[f"a{n_idx}_{fan_id}"])
            params[f"b{n_idx}"] = float(row[f"b{n_idx}_{fan_id}"])

        params_list.append(params)
        orders_list.append(list(harmonic_orders))

    return params_list, orders_list


def azimuthal_ring_model(
    r: np.ndarray,
    theta: np.ndarray,
    params: Dict[str, float],
    harmonic_orders: Sequence[int],
) -> np.ndarray:
    """
    Evaluate one fan contribution:
        w = w0 + exp(-((r - r_ring)/delta_ring)^2) * A(theta)
    """
    delta_ring = max(float(params["delta_ring"]), 1e-12)
    g_r = np.exp(-((r - float(params["r_ring"])) / delta_ring) ** 2)

    amp = np.full_like(theta, float(params["a0"]), dtype=float)
    for order in harmonic_orders:
        amp += float(params[f"a{order}"]) * np.cos(order * theta)
        amp += float(params[f"b{order}"]) * np.sin(order * theta)

    return float(params["w0"]) + g_r * amp


def plot_continuous_heatmap(
    x: np.ndarray,
    y: np.ndarray,
    w_field: np.ndarray,
    outpath: Path,
) -> None:
    """
    Plot continuous model heat map using the same axis settings as annuli heat maps.
    """
    x_edges = centers_to_edges(x)
    y_edges = centers_to_edges(y)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": LEGEND_FONTSIZE,
            "axes.edgecolor": "k",
            "axes.linewidth": AXIS_EDGE_LW,
            "patch.edgecolor": "k",
        }
    )

    fig, ax = plt.subplots(figsize=(5.2, 3.0), dpi=600)

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

    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID_COLOR, linewidth=GRID_LINEWIDTH)
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

def main() -> None:
    params_df = load_bemt_params(PARAMS_XLSX, PARAMS_SHEET)
    fan_ids = discover_fan_ids(params_df)
    if len(fan_ids) > 0 and len(fan_ids) != len(FOUR_FAN_CENTERS_XY):
        raise ValueError(
            f"Parameter table has {len(fan_ids)} fan IDs but expected {len(FOUR_FAN_CENTERS_XY)}."
        )

    for sh in SHEETS:
        x, y, _w = read_slice_from_sheet(XLSX_PATH, sh)
        params_fan, harmonic_orders_fan = params_for_height(
            params_df,
            sheet_name=sh,
            fan_ids=fan_ids,
        )

        x_grid, y_grid = build_continuous_grid(x, y)
        fan_xy = np.asarray(FOUR_FAN_CENTERS_XY, dtype=float)
        dx_all = x_grid[:, :, None] - fan_xy[None, None, :, 0]
        dy_all = y_grid[:, :, None] - fan_xy[None, None, :, 1]
        r_all = np.sqrt(dx_all**2 + dy_all**2)
        theta_all = np.arctan2(dy_all, dx_all)

        w_model = np.zeros((x_grid.shape[0], x_grid.shape[1]), dtype=float)
        for fan_idx in range(len(FOUR_FAN_CENTERS_XY)):
            w_model += azimuthal_ring_model(
                r=r_all[:, :, fan_idx],
                theta=theta_all[:, :, fan_idx],
                params=params_fan[fan_idx],
                harmonic_orders=harmonic_orders_fan[fan_idx],
            )

        out_png = OUT_DIR / f"{sh}_four_annular_gaussian_bemt_heatmap_main.png"
        plot_continuous_heatmap(
            x_grid[0, :],
            y_grid[:, 0],
            w_model,
            outpath=out_png,
        )

    print(f"Saved figures to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()


