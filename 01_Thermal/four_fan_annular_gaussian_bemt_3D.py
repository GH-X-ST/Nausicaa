"""
Plot a 3D non-axisymmetric four-fan annular-Gaussian updraft field using
fitted BEMT parameter output.

Model per fan:
    theta = atan2(y - y_f, x - x_f)
    r     = sqrt((x - x_f)^2 + (y - y_f)^2)
    g(r, z) = exp(-((r - r_ring(z)) / delta_ring(z))**2)

    A(theta, z) = a0(z) + sum_{n=1..N}(a_n(z) cos(n theta) + b_n(z) sin(n theta))
    w_f(x, y, z) = w0(z) + g(r, z) * A(theta, z)

Total field:
    w(x, y, z) = sum_f w_f(x, y, z)
"""

###### Initialization

### Imports
from pathlib import Path
from typing import List, Tuple
import re

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cmocean
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes


### User settings
MAKE_PLOTS = True

PARAMS_XLSX = Path("B_results/four_annular_bemt_params_pchip.xlsx")
PARAMS_SHEET = "four_bemt_az_pchip"

OUT_DIR = Path("A_figures/Four_Fan_Annular_Gaussian_BEMT")
OUT_3D_NAME = "four_annular_gaussian_bemt_3d.png"
FIGSIZE_3D = (7.45 / 1.3, 3.0)

# Layout controls (figure-fraction coordinates).
AXES_X = 0.13
AXES_Y = -0.05
AXES_W = 1.25
AXES_H = 1.25

# Independent colorbar box controls (figure-fraction coordinates).
CBAR_X = 0.87
CBAR_Y = 0.15
CBAR_W = 0.02
CBAR_H = 0.77

CBAR_LABEL = r"$w$ (m$\cdot$s$^{-1}$)"
AXIS_EDGE_LW = 0.80
CBAR_EDGE_LW = AXIS_EDGE_LW
CBAR_VMIN = 0.0
CBAR_VMAX = 8.0
CBAR_TICK_STEP = 1.0
CBAR_LABEL_FONTSIZE = 10
CBAR_TICK_FONTSIZE = 9

# Typography (match heat-map-main script)
AXIS_LABEL_FONTSIZE = 10
TICK_LABEL_FONTSIZE = 9
LEGEND_FONTSIZE = 8.5

# Four-fan centres in arena coordinates (m)
FOUR_FAN_CENTERS_XY = (
    (3.0, 3.6),
    (5.4, 3.6),
    (3.0, 1.2),
    (5.4, 1.2),
)
FAN_OUTLET_DIAMETER = 0.8
FAN_OUTLET_EDGE_LW = 1.1
FAN_OUTLET_ALPHA = 0.6
FAN_OUTLET_DASH = (0, (2, 2))
FAN_VERTICAL_OFFSET_M = 0.330

# Slice planes through the geometric center of the four fans.
SLICE_CENTER_X = 4.2
SLICE_CENTER_Y = 2.4

# Arena limits (absolute height z measured from floor)
X_MIN, X_MAX = 0.0, 8.4
Y_MIN, Y_MAX = 0.0, 4.8
Z_MIN, Z_MAX = 0.0, 3.5

# 3D sampling resolution
NX_3D, NY_3D, NZ_3D = 240, 160, 100

# Isosurface levels as fractions of max(w)
ISO_MIN_FRAC = 0.01
ISO_MAX_FRAC = 0.97
N_ISO_LEVELS = 10

# Include fitted baseline term w0(z) in each fan contribution.
INCLUDE_W0 = True

# Exponential opacity mapping versus normalized w (= 0..1).
# alpha(0) = 0 (fully transparent), alpha(1) = 1 (fully opaque).
ALPHA_EXP_RATE = 3.0

# Interior visibility aids.
SHOW_CENTER_SLICES = True
SLICE_EDGE_COLOR = (0.0, 0.0, 0.0, 0.20)
SLICE_EDGE_LW = 0.2


### Helpers
REQUIRED_SHARED_COLUMNS = ("z_m", "w0", "r_ring", "delta_ring", "a0")
FAN_COL_PATTERN = re.compile(r"^a0_(F\d{2})$")


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


def load_params_table(xlsx_path: Path, sheet_name: str) -> pd.DataFrame:
    """
    Load precomputed PCHIP parameter table from fit-script output.

    If the provided extension does not exist, the alternate xls/xlsx
    extension is tried automatically.
    """
    table_path = xlsx_path
    if not table_path.exists():
        alt_suffix = ".xlsx" if table_path.suffix.lower() == ".xls" else ".xls"
        alt_path = table_path.with_suffix(alt_suffix)
        if alt_path.exists():
            table_path = alt_path
        else:
            raise FileNotFoundError(
                f"Missing PCHIP table: '{xlsx_path}' (also checked '{alt_path}')."
            )

    xls = pd.ExcelFile(table_path)
    chosen_sheet = sheet_name if sheet_name in xls.sheet_names else xls.sheet_names[0]
    return pd.read_excel(table_path, sheet_name=chosen_sheet)


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
    missing = [col for col in REQUIRED_SHARED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required shared columns: {missing}")

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
    param_cols = ["w0", "r_ring", "delta_ring", "a0"]
    for n_idx in harmonic_orders:
        param_cols.append(f"a{n_idx}")
        param_cols.append(f"b{n_idx}")

    return param_cols, harmonic_orders


def discover_fan_param_columns(
    df: pd.DataFrame,
    fan_id: str,
) -> Tuple[List[str], List[str], List[int]]:
    """
    Discover per-fan source columns, local parameter names, and harmonic orders.
    """
    source_cols = [
        f"w0_{fan_id}",
        f"r_ring_{fan_id}",
        f"delta_ring_{fan_id}",
        f"a0_{fan_id}",
    ]
    missing = [col for col in source_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required per-fan columns for {fan_id}: {missing}")

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

    local_cols = ["w0", "r_ring", "delta_ring", "a0"]
    for n_idx in harmonic_orders:
        source_cols.append(f"a{n_idx}_{fan_id}")
        source_cols.append(f"b{n_idx}_{fan_id}")
        local_cols.append(f"a{n_idx}")
        local_cols.append(f"b{n_idx}")

    return source_cols, local_cols, harmonic_orders


def extract_clean_arrays(
    df: pd.DataFrame,
    value_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract sorted numeric arrays for z and selected columns.
    """
    local = df[["z_m"] + list(value_cols)].copy()
    for col in local.columns:
        local[col] = pd.to_numeric(local[col], errors="coerce")

    local = local.dropna(subset=["z_m"] + list(value_cols))
    if local.empty:
        raise ValueError("No valid parameter rows after cleaning.")

    z_vals = local["z_m"].to_numpy(dtype=float)
    params = local[value_cols].to_numpy(dtype=float)

    order = np.argsort(z_vals)
    z_vals = z_vals[order]
    params = params[order, :]

    if np.any(np.diff(z_vals) <= 0.0):
        raise ValueError("z_m must be strictly increasing for interpolation.")

    return z_vals, params


def extract_param_arrays(
    df: pd.DataFrame,
    fan_ids: Tuple[str, ...],
) -> List[Tuple[np.ndarray, np.ndarray, List[str], List[int]]]:
    """
    Extract per-fan interpolation tables.

    Each list entry is:
        (z_vals, params_matrix, param_cols, harmonic_orders)
    where params_matrix columns align with param_cols.
    """
    n_fans_expected = len(FOUR_FAN_CENTERS_XY)
    fan_data = []

    if len(fan_ids) == 0:
        param_cols, harmonic_orders = discover_shared_param_columns(df)
        z_vals, params = extract_clean_arrays(df, param_cols)

        delta_idx = param_cols.index("delta_ring")
        if np.any(params[:, delta_idx] <= 0.0):
            raise ValueError("delta_ring must be positive for interpolation.")

        for _ in FOUR_FAN_CENTERS_XY:
            fan_data.append((z_vals, params, list(param_cols), list(harmonic_orders)))
        return fan_data

    if len(fan_ids) != n_fans_expected:
        raise ValueError(
            f"Parameter table has {len(fan_ids)} fan IDs but expected {n_fans_expected}."
        )

    for fan_id in fan_ids:
        source_cols, local_cols, harmonic_orders = discover_fan_param_columns(df, fan_id)
        z_vals, params = extract_clean_arrays(df, source_cols)

        delta_idx = local_cols.index("delta_ring")
        if np.any(params[:, delta_idx] <= 0.0):
            raise ValueError(f"delta_ring must be positive for {fan_id}.")

        fan_data.append((z_vals, params, local_cols, harmonic_orders))

    return fan_data


def make_3d_grid() -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """
    Build 3D sampling grid and voxel spacing.
    """
    x_3d = np.linspace(X_MIN, X_MAX, NX_3D)
    y_3d = np.linspace(Y_MIN, Y_MAX, NY_3D)
    z_3d = np.linspace(Z_MIN, Z_MAX, NZ_3D)

    dx = x_3d[1] - x_3d[0]
    dy = y_3d[1] - y_3d[0]
    dz = z_3d[1] - z_3d[0]

    x_grid, y_grid, z_grid = np.meshgrid(x_3d, y_3d, z_3d, indexing="ij")
    return x_grid, y_grid, z_grid, dx, dy, dz


def evaluate_field(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    fan_data: List[Tuple[np.ndarray, np.ndarray, List[str], List[int]]],
    include_w0: bool,
) -> np.ndarray:
    """
    Evaluate w(x, y, z_abs) from the precomputed BEMT PCHIP table.

    The field is formed by superposing all four fan contributions.
    """
    fan_xy = np.asarray(FOUR_FAN_CENTERS_XY, dtype=float)
    r_all = np.sqrt(
        (x_grid[..., None] - fan_xy[:, 0]) ** 2
        + (y_grid[..., None] - fan_xy[:, 1]) ** 2
    )
    theta_all = np.arctan2(y_grid[..., None] - fan_xy[:, 1], x_grid[..., None] - fan_xy[:, 0])

    z_rel = z_grid - FAN_VERTICAL_OFFSET_M
    w_grid = np.zeros_like(z_grid, dtype=float)

    for fan_idx, (z_vals, params, param_cols, harmonic_orders) in enumerate(fan_data):
        z_min = float(np.min(z_vals))
        z_max = float(np.max(z_vals))
        mask = (z_rel >= z_min) & (z_rel <= z_max)
        if not np.any(mask):
            continue

        z_eval = z_rel[mask]
        r_eval = r_all[:, :, :, fan_idx][mask]
        theta_eval = theta_all[:, :, :, fan_idx][mask]

        p_eval = {}
        for col_idx, col_name in enumerate(param_cols):
            p_eval[col_name] = np.interp(z_eval, z_vals, params[:, col_idx])

        delta_eval = np.maximum(p_eval["delta_ring"], 1e-12)
        amp_eval = p_eval["a0"].copy()
        for n_idx in harmonic_orders:
            amp_eval += p_eval[f"a{n_idx}"] * np.cos(n_idx * theta_eval)
            amp_eval += p_eval[f"b{n_idx}"] * np.sin(n_idx * theta_eval)

        g_eval = np.exp(-((r_eval - p_eval["r_ring"]) / delta_eval) ** 2)
        w_eval = g_eval * amp_eval
        if include_w0:
            w_eval += p_eval["w0"]

        w_grid[mask] += w_eval

    return w_grid


def plot_isosurfaces(
    w_grid: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    cmap_alpha: mcolors.ListedColormap,
    output_path: Path,
) -> None:
    """
    Plot 3D isosurfaces of w and save figure.
    """
    w_max = float(np.max(w_grid))
    if w_max <= 0.0:
        raise ValueError("Computed field has non-positive max; cannot build isosurfaces.")

    iso_fracs = np.linspace(ISO_MIN_FRAC, ISO_MAX_FRAC, N_ISO_LEVELS)
    iso_levels = [float(frac * w_max) for frac in iso_fracs]

    fig_3d = plt.figure(figsize=FIGSIZE_3D)
    ax3d = fig_3d.add_subplot(111, projection="3d")
    fig_3d.patch.set_facecolor("white")
    ax3d.set_facecolor("white")
    ax3d.grid(True)

    norm = mpl.colors.Normalize(vmin=CBAR_VMIN, vmax=CBAR_VMAX)

    if SHOW_CENTER_SLICES:
        x_vec = np.linspace(X_MIN, X_MAX, w_grid.shape[0])
        y_vec = np.linspace(Y_MIN, Y_MAX, w_grid.shape[1])
        z_vec = np.linspace(Z_MIN, Z_MAX, w_grid.shape[2])

        ix_center = int(np.argmin(np.abs(x_vec - SLICE_CENTER_X)))
        iy_center = int(np.argmin(np.abs(y_vec - SLICE_CENTER_Y)))

        y_mesh_x, z_mesh_x = np.meshgrid(y_vec, z_vec, indexing="ij")
        x_mesh_x = np.full_like(y_mesh_x, x_vec[ix_center])
        w_slice_x = w_grid[ix_center, :, :]
        fc_x = cmap_alpha(norm(w_slice_x))
        ax3d.plot_surface(
            x_mesh_x,
            y_mesh_x,
            z_mesh_x,
            facecolors=fc_x,
            rstride=1,
            cstride=1,
            linewidth=0.0,
            antialiased=False,
            shade=False,
            zorder=5,
        )

        x_mesh_y, z_mesh_y = np.meshgrid(x_vec, z_vec, indexing="ij")
        y_mesh_y = np.full_like(x_mesh_y, y_vec[iy_center])
        w_slice_y = w_grid[:, iy_center, :]
        fc_y = cmap_alpha(norm(w_slice_y))
        ax3d.plot_surface(
            x_mesh_y,
            y_mesh_y,
            z_mesh_y,
            facecolors=fc_y,
            rstride=1,
            cstride=1,
            linewidth=0.0,
            antialiased=False,
            shade=False,
            zorder=5,
        )

        ax3d.plot(
            np.full_like(y_vec, x_vec[ix_center]),
            y_vec,
            np.full_like(y_vec, Z_MIN),
            color=SLICE_EDGE_COLOR,
            linewidth=SLICE_EDGE_LW,
            zorder=6,
        )
        ax3d.plot(
            x_vec,
            np.full_like(x_vec, y_vec[iy_center]),
            np.full_like(x_vec, Z_MIN),
            color=SLICE_EDGE_COLOR,
            linewidth=SLICE_EDGE_LW,
            zorder=6,
        )

    for level in iso_levels:
        verts, faces, _, _ = marching_cubes(
            w_grid,
            level=level,
            spacing=(dx, dy, dz),
        )

        verts[:, 0] += X_MIN
        verts[:, 1] += Y_MIN
        verts[:, 2] += Z_MIN

        mesh = Poly3DCollection(
            verts[faces],
            linewidths=0.05,
            zorder=10,
        )
        mesh.set_facecolor(cmap_alpha(norm(level)))
        mesh.set_edgecolor("none")
        ax3d.add_collection3d(mesh)

    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    for idx, (fcx, fcy) in enumerate(FOUR_FAN_CENTERS_XY):
        circle_x = fcx + 0.5 * FAN_OUTLET_DIAMETER * np.cos(theta)
        circle_y = fcy + 0.5 * FAN_OUTLET_DIAMETER * np.sin(theta)
        circle_z = FAN_VERTICAL_OFFSET_M * np.ones_like(theta)
        ax3d.plot(
            circle_x,
            circle_y,
            circle_z,
            color=(0.0, 0.0, 0.0, FAN_OUTLET_ALPHA),
            linewidth=FAN_OUTLET_EDGE_LW,
            linestyle=FAN_OUTLET_DASH,
            label="Fan outlet" if idx == 0 else None,
            zorder=0,
        )

    ax3d.set_xlim(X_MIN, X_MAX)
    ax3d.set_ylim(Y_MIN, Y_MAX)
    ax3d.set_zlim(Z_MIN, Z_MAX)
    ax3d.set_xticks(np.arange(X_MIN, X_MAX + 1e-9, 1.4))
    ax3d.set_yticks(np.arange(Y_MIN, Y_MAX + 1e-9, 1.2))
    ax3d.set_zticks(np.arange(Z_MIN, Z_MAX + 1e-9, 0.7))

    ax3d.set_xlabel("$x$ (m)", labelpad=17)
    ax3d.set_ylabel("$y$ (m)", labelpad=10)
    ax3d.set_zlabel("$z$ (m)", labelpad=5, rotation=90)
    ax3d.zaxis.set_rotate_label(False)
    ax3d.xaxis.label.set_size(AXIS_LABEL_FONTSIZE)
    ax3d.yaxis.label.set_size(AXIS_LABEL_FONTSIZE)
    ax3d.zaxis.label.set_size(AXIS_LABEL_FONTSIZE)
    ax3d.tick_params(axis="x", which="major", labelsize=TICK_LABEL_FONTSIZE)
    ax3d.tick_params(axis="y", which="major", labelsize=TICK_LABEL_FONTSIZE)
    ax3d.tick_params(axis="z", which="major", labelsize=TICK_LABEL_FONTSIZE)
    for label in ax3d.get_xticklabels():
        label.set_rotation(-20)
    for label in ax3d.get_yticklabels():
        label.set_rotation(20)

    ax3d.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax3d.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax3d.zaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    leg = ax3d.legend(
        loc="upper right",
        bbox_to_anchor=(1.1, 0.8),
        frameon=True,
        framealpha=1.0,
        edgecolor="black",
        fontsize=LEGEND_FONTSIZE,
        handlelength=1.5,
        borderpad=0.5,
        labelspacing=0.2,
    )
    if leg is not None:
        leg.get_frame().set_linewidth(AXIS_EDGE_LW)

    try:
        ax3d.set_box_aspect((X_MAX - X_MIN, Y_MAX - Y_MIN, Z_MAX - Z_MIN))
    except AttributeError:
        pass

    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_alpha)
    mappable.set_array([])
    cax = fig_3d.add_axes([CBAR_X, CBAR_Y, CBAR_W, CBAR_H])
    cbar = fig_3d.colorbar(mappable, cax=cax)
    cbar.set_label(CBAR_LABEL, fontsize=CBAR_LABEL_FONTSIZE)
    cbar.set_ticks(np.arange(CBAR_VMIN, CBAR_VMAX + 1e-9, CBAR_TICK_STEP))
    cbar.formatter = FormatStrFormatter("%.2f")
    cbar.update_ticks()
    cbar.ax.tick_params(width=0.6, length=2, labelsize=CBAR_TICK_FONTSIZE)
    cbar.outline.set_linewidth(CBAR_EDGE_LW)
    cbar.outline.set_edgecolor("k")
    cbar.outline.set_visible(True)
    cbar.ax.patch.set_edgecolor("k")
    cbar.ax.patch.set_linewidth(CBAR_EDGE_LW)
    cbar.ax.set_frame_on(True)
    for spine in cbar.ax.spines.values():
        spine.set_edgecolor("k")
        spine.set_linewidth(CBAR_EDGE_LW)

    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        axis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 1.0)
        axis._axinfo["grid"]["linewidth"] = 0.4

    ax3d.view_init(elev=20, azim=-130)
    ax3d.set_position([AXES_X, AXES_Y, AXES_W, AXES_H])
    ax3d.set_anchor("W")

    fig_3d.savefig(
        output_path,
        dpi=600,
        facecolor="white",
    )
    plt.close(fig_3d)


### Main
def main() -> None:
    if not MAKE_PLOTS:
        print("MAKE_PLOTS is False; nothing to do.")
        return

    params_df = load_params_table(xlsx_path=PARAMS_XLSX, sheet_name=PARAMS_SHEET)
    fan_ids = discover_fan_ids(params_df)
    fan_data = extract_param_arrays(params_df, fan_ids=fan_ids)

    x_grid, y_grid, z_grid, dx, dy, dz = make_3d_grid()
    w_grid = evaluate_field(
        x_grid=x_grid,
        y_grid=y_grid,
        z_grid=z_grid,
        fan_data=fan_data,
        include_w0=INCLUDE_W0,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_3d = OUT_DIR / OUT_3D_NAME
    cmap_alpha = build_alpha_cmap()
    plot_isosurfaces(
        w_grid=w_grid,
        dx=dx,
        dy=dy,
        dz=dz,
        cmap_alpha=cmap_alpha,
        output_path=out_3d,
    )

    print(f"Saved 3D annular-Gaussian plot to: {out_3d.resolve()}")


if __name__ == "__main__":
    main()
