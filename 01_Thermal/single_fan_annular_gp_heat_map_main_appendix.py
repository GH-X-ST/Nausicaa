"""
Plot single-fan annular-GP heat maps using the appendix figure style.

For each height sheet, GP mean predictions are loaded from:
    B_results/Single_Fan_Annular_GP/single_annular_gp_grid_predictions.xlsx
using the z###_annular_gp_mean sheets written by single_fan_annular_gp.py.
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
from scipy.interpolate import RegularGridInterpolator

# cmocean provides perceptual thermal colormaps used consistently across figures.
import cmocean


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Figure Routing Constants
# 2) Figure Routing and Rendering
# 3) Batch Figure Export
# =============================================================================

# =============================================================================
# 1) Figure Routing Constants
# =============================================================================
# Routing constants keep thesis and appendix figures traceable to the same metric source tables.

SHEETS = ["z020", "z035", "z050", "z075", "z110", "z160", "z220"]

GP_GRID_XLSX = Path(
    "B_results/Single_Fan_Annular_GP/single_annular_gp_grid_predictions.xlsx"
)
OUT_DIR = Path("A_figures/Single_Fan_Annular_GP")

# Axis and colorbar labels use metres and metres per second in exported figures.
CBAR_LABEL = r"$w$ (m$\cdot$s$^{-1}$)"
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

# Fan outlet marker in arena coordinates (m).
FAN_OUTLET_POINTS = [(4.2, 2.4)]
FAN_OUTLET_DIAMETER = 0.8
FAN_OUTLET_EDGE_LW = 1.1
FAN_OUTLET_ALPHA = 0.6
FAN_OUTLET_DASH = (0, (2, 2))

# Fixed color scale keeps heights and model families visually comparable.
PLOT_VMIN = 0.0
PLOT_VMAX = 8.0
CBAR_TICK_STEP = None

# Dense display grid only affects plot interpolation, not fitted data.
GRID_NX = 240
GRID_NY = 180

# z020 -> 0.20 m, z110 -> 1.10 m, etc.
SHEET_HEIGHT_DIVISOR = 100.0
FAN_COL_PATTERN = re.compile(r"^a0_(F\d{2})$")


# =============================================================================
# 2) Figure Routing and Rendering
# =============================================================================
# Rendering functions keep thesis and appendix figure sizes separate from data loading.

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


# Sheet names encode height in centimetres; parsing converts that label to metres.
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


# Workbook sheets store x in the first row, y in the first column, and vertical velocity in m/s inside the grid.
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
    w_map = (
        raw.iloc[1:, 1:]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )

    if w_map.shape != (y.size, x.size):
        raise ValueError(
            (
                f"Shape mismatch in {sheet_name}: W{w_map.shape}, "
                f"y({y.size}), x({x.size})."
            )
        )

    if y.size >= 2 and y[0] > y[-1]:
        y = y[::-1]
        w_map = w_map[::-1, :]

    return x, y, w_map


# Display-grid resolution is a plotting choice and must not be interpreted as measurement density.
def build_continuous_grid(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
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


# Display interpolation is only for figure smoothness; model fitting remains on source samples.
def interpolate_to_continuous_grid(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
) -> np.ndarray:
    """Bilinearly interpolate a stored field onto the display grid."""
    interp = RegularGridInterpolator(
        (y, x),
        w,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    query = np.column_stack([y_grid.ravel(), x_grid.ravel()])
    w_dense = interp(query).reshape(x_grid.shape)

    if np.any(~np.isfinite(w_dense)):
        interp_nn = RegularGridInterpolator(
            (y, x),
            w,
            method="nearest",
            bounds_error=False,
            fill_value=0.0,
        )
        mask = ~np.isfinite(w_dense)
        w_dense[mask] = interp_nn(query[mask.ravel()])

    return np.asarray(w_dense, dtype=float)


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
            (
                f"Shape mismatch in {sheet_name}: W{w.shape}, "
                f"y({y.size}), x({x.size})."
            )
        )
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError(
            f"Non-numeric x/y axis values found in sheet '{sheet_name}'."
        )

    x_order = np.argsort(x)
    y_order = np.argsort(y)
    x = x[x_order]
    y = y[y_order]
    w = w[np.ix_(y_order, x_order)]

    if np.any(np.diff(x) <= 0.0) or np.any(np.diff(y) <= 0.0):
        raise ValueError(
            f"x/y coordinates must be strictly increasing in '{sheet_name}'."
        )

    return x, y, w


# BEMT parameter tables are read as fitted-model inputs before grid evaluation.
def load_bemt_params(xlsx_path: Path, sheet_name: str) -> pd.DataFrame:
    """
    Load fitted non-axisymmetric BEMT parameters from Excel.
    """
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Missing fitted-parameter file: {xlsx_path}")

    xls = pd.ExcelFile(xlsx_path)
    if sheet_name in xls.sheet_names:
        chosen_sheet = sheet_name
    else:
        chosen_sheet = xls.sheet_names[0]
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


# Fan-ID discovery treats suffixed columns as the interface for per-fan fitted parameters.
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


# Shared-column discovery supports legacy averaged parameter tables without hiding per-fan data.
def discover_shared_param_columns(
    df: pd.DataFrame,
) -> Tuple[List[str], List[int]]:
    """
    Discover shared parameter columns and harmonic orders.
    """
    base = ["w0", "r_ring", "delta_ring", "a0"]
    missing = [col for col in base if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required shared parameter columns: {missing}"
        )

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


# Per-fan column discovery preserves fan identity through model evaluation.
def discover_fan_param_columns(
    df: pd.DataFrame,
    fan_id: str,
) -> Tuple[List[str], List[int]]:
    """
    Discover per-fan parameter columns and harmonic orders.
    """
    base = [
        f"w0_{fan_id}",
        f"r_ring_{fan_id}",
        f"delta_ring_{fan_id}",
        f"a0_{fan_id}",
    ]
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


# Sheet-to-parameter matching keeps measured z planes aligned with fitted model rows.
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


# Height selection keeps plotted fields tied to the nearest fitted or interpolated z sample.
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


# Azimuthal ring evaluation adds Fourier variation around the fan-centred plume.
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


# Continuous plots show model or interpolated fields on a display grid, not new measurements.
def plot_continuous_heatmap(
    x: np.ndarray,
    y: np.ndarray,
    w_field: np.ndarray,
    outpath: Path,
) -> None:
    """
    Plot continuous model heat map using the same axis settings as
    annuli heat maps.
    """
    x_grid, y_grid = build_continuous_grid(x, y)
    w_dense = interpolate_to_continuous_grid(x, y, w_field, x_grid, y_grid)
    x_plot = x_grid[0, :]
    y_plot = y_grid[:, 0]
    x_edges = centers_to_edges(x_plot)
    y_edges = centers_to_edges(y_plot)

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

    cmap_alpha = build_alpha_cmap()
    im = ax.pcolormesh(
        x_edges,
        y_edges,
        w_dense,
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
    cax = divider.append_axes("right", size="2.6%", pad=0.15)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(CBAR_LABEL)
    if CBAR_TICK_STEP is not None:
        cbar.set_ticks(
            np.arange(
                float(PLOT_VMIN),
                float(PLOT_VMAX) + 0.5 * float(CBAR_TICK_STEP),
                float(CBAR_TICK_STEP),
                dtype=float,
            )
        )
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
        raise FileNotFoundError(
            f"Missing annular-GP grid workbook: {GP_GRID_XLSX}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sh in SHEETS:
        x, y, w_model = load_gp_mean_sheet(
            GP_GRID_XLSX,
            f"{sh}_annular_gp_mean",
        )
        out_png = OUT_DIR / f"{sh}_single_annular_gp_heatmap_main_appendix.png"
        plot_continuous_heatmap(
            x,
            y,
            w_model,
            outpath=out_png,
        )

    print(f"Saved figures to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
