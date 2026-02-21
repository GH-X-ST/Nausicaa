"""
Plot a 3D GP updraft field using saved grid-prediction output.

Input is read from:
    B_results/Single_Fan_GP/single_fan_gp_grid_predictions.xlsx
using the z###_gp_mean sheets written by single_fan_gp.py.
"""

###### Initialization

### Imports
from pathlib import Path
from typing import Sequence, Tuple
import warnings

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cmocean
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import RegularGridInterpolator
from skimage.measure import marching_cubes
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler


### User settings
MAKE_PLOTS = True

SHEETS = ["z020", "z035", "z050", "z075", "z110", "z160", "z220"]
SHEET_HEIGHT_DIVISOR = 100.0

GRID_PRED_XLSX = Path("B_results/Single_Fan_GP/single_gp_grid_predictions.xlsx")

OUT_DIR = Path("A_figures/Single_Fan_GP")
OUT_3D_NAME = "single_gp_3d.png"
# Match heat-map-main output aspect ratio
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

CBAR_LABEL = r"$w$ (m $\!$s$^{-1}$)"
AXIS_EDGE_LW = 0.80
CBAR_EDGE_LW = AXIS_EDGE_LW
CBAR_VMIN = 0.0
CBAR_VMAX = 8.0
CBAR_TICK_STEP = 1.0
CBAR_LABEL_FONTSIZE = 10
CBAR_TICK_FONTSIZE = 9

# Typography (match single_fan_gp_heat_map_main.py intent)
AXIS_LABEL_FONTSIZE = 10
TICK_LABEL_FONTSIZE = 9
LEGEND_FONTSIZE = 8.5

# Single-fan centre in arena coordinates (m)
FAN_CENTER_X = 4.2
FAN_CENTER_Y = 2.4
FAN_OUTLET_DIAMETER = 0.8
FAN_OUTLET_EDGE_LW = 1.1
FAN_OUTLET_ALPHA = 0.6
FAN_OUTLET_DASH = (0, (2, 2))
FAN_VERTICAL_OFFSET_M = 0.330

# Arena limits (absolute height z measured from floor)
X_MIN, X_MAX = 0.0, 8.4
Y_MIN, Y_MAX = 0.0, 4.8
# Plotting range after adding fan vertical offset (+0.33 m).
Z_MIN, Z_MAX = 0.00, 3.5

# 3D sampling resolution
NX_3D, NY_3D, NZ_3D = 240, 160, 100

# Isosurface levels as fractions of max(w)
ISO_MIN_FRAC = 0.01
ISO_MAX_FRAC = 0.97
N_ISO_LEVELS = 10

# Relative-z GP extrapolation range (before adding fan offset).
REL_Z_EXTRAP_MIN = 0.00
REL_Z_EXTRAP_MAX = 3.17

# 1D GP settings for per-(x,y) extrapolation in z.
GP_Z_N_RESTARTS_OPTIMIZER = 1
GP_Z_RANDOM_STATE = 42
GP_Z_ALPHA_JITTER = 1e-8
CONST_SERIES_TOL = 1e-10

# Exponential opacity mapping versus normalized w (= 0..1).
# alpha(0) = 0 (fully transparent), alpha(1) = 1 (fully opaque).
ALPHA_EXP_RATE = 3.0

# Interior visibility aids.
SHOW_CENTER_SLICES = True
SLICE_EDGE_COLOR = (0.0, 0.0, 0.0, 0.20)
SLICE_EDGE_LW = 0.2


### Helpers
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


def parse_sheet_height_m(sheet_name: str) -> float:
    """
    Parse height in meters from names like 'z020', 'z110', 'z220'.
    """
    if not sheet_name.startswith("z"):
        raise ValueError(f"Invalid sheet name (expected 'z###'): {sheet_name}")
    suffix = sheet_name[1:]
    if not suffix.isdigit():
        raise ValueError(f"Invalid height code in sheet name: {sheet_name}")
    return int(suffix) / SHEET_HEIGHT_DIVISOR


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
            f"Shape mismatch in {sheet_name}: W{w.shape}, y({y.size}), x({x.size})."
        )
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError(f"Non-numeric x/y axis values found in sheet '{sheet_name}'.")

    x_order = np.argsort(x)
    y_order = np.argsort(y)
    x = x[x_order]
    y = y[y_order]
    w = w[np.ix_(y_order, x_order)]

    if np.any(np.diff(x) <= 0.0) or np.any(np.diff(y) <= 0.0):
        raise ValueError(f"x/y coordinates must be strictly increasing in '{sheet_name}'.")

    return x, y, w


def load_gp_stack(
    xlsx_path: Path,
    sheet_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load stacked GP mean maps across z from z###_gp_mean sheets.
    Returns x_axis, y_axis, z_axis, w_stack (Nz, Ny, Nx).
    """
    x_ref = None
    y_ref = None
    z_list = []
    w_layers = []

    for sh in sheet_names:
        x, y, w = load_gp_mean_sheet(xlsx_path, f"{sh}_gp_mean")
        if x_ref is None:
            x_ref = x
            y_ref = y
        else:
            if not np.allclose(x, x_ref, atol=1e-9, rtol=0.0):
                raise ValueError(f"Inconsistent x-grid found in sheet '{sh}_gp_mean'.")
            if not np.allclose(y, y_ref, atol=1e-9, rtol=0.0):
                raise ValueError(f"Inconsistent y-grid found in sheet '{sh}_gp_mean'.")

        z_list.append(parse_sheet_height_m(sh))
        w_layers.append(w)

    z_axis = np.asarray(z_list, dtype=float)
    order = np.argsort(z_axis)
    z_axis = z_axis[order]
    w_stack = np.stack([w_layers[i] for i in order], axis=0)

    if np.any(np.diff(z_axis) <= 0.0):
        raise ValueError("z levels must be strictly increasing.")

    return x_ref, y_ref, z_axis, w_stack


def gp_predict_series_vs_z(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_query: np.ndarray,
) -> np.ndarray:
    """
    Fit a 1D GP y(z) and predict at z_query.
    """
    z_train = np.asarray(z_train, dtype=float).reshape(-1, 1)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    z_query = np.asarray(z_query, dtype=float).reshape(-1, 1)

    if float(np.std(y_train)) < CONST_SERIES_TOL:
        return np.full(z_query.shape[0], max(float(np.mean(y_train)), 0.0), dtype=float)

    scaler = StandardScaler()
    z_train_scaled = scaler.fit_transform(z_train)
    z_query_scaled = scaler.transform(z_query)

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=np.ones(1, dtype=float), length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e0))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=float(GP_Z_ALPHA_JITTER),
        normalize_y=True,
        n_restarts_optimizer=int(GP_Z_N_RESTARTS_OPTIMIZER),
        random_state=int(GP_Z_RANDOM_STATE),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        gp.fit(z_train_scaled, y_train)

    y_pred = gp.predict(z_query_scaled, return_std=False).astype(float)
    return np.maximum(y_pred, 0.0)


def extrapolate_stack_in_z_with_gp(
    z_axis: np.ndarray,
    w_stack: np.ndarray,
    z_query: np.ndarray,
) -> np.ndarray:
    """
    Extrapolate/interpolate GP mean maps along z for each coarse (x,y) node.

    Input shape:  w_stack (Nz, Ny, Nx)
    Output shape: (Nq, Ny, Nx)
    """
    z_axis = np.asarray(z_axis, dtype=float).reshape(-1)
    z_query = np.asarray(z_query, dtype=float).reshape(-1)

    if w_stack.shape[0] != z_axis.size:
        raise ValueError(
            f"w_stack first axis ({w_stack.shape[0]}) must equal len(z_axis) ({z_axis.size})."
        )
    if np.any(np.diff(z_axis) <= 0.0):
        raise ValueError("z_axis must be strictly increasing.")

    _nz, ny, nx = w_stack.shape
    out = np.empty((z_query.size, ny, nx), dtype=float)
    for iy in range(ny):
        for ix in range(nx):
            out[:, iy, ix] = gp_predict_series_vs_z(
                z_train=z_axis,
                y_train=w_stack[:, iy, ix],
                z_query=z_query,
            )
    return out


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
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
    w_stack: np.ndarray,
) -> np.ndarray:
    """
    Evaluate interpolated GP mean field on the requested 3D grid.
    """
    # Convert (Nz, Ny, Nx) -> (Nx, Ny, Nz) to match axes order (x, y, z).
    w_xyz = np.transpose(w_stack, (2, 1, 0))

    interp = RegularGridInterpolator(
        (x_axis, y_axis, z_axis),
        w_xyz,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    points = np.column_stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()])
    w_grid = interp(points).reshape(x_grid.shape)

    # Keep non-physical small negative interpolation artifacts from appearing.
    w_grid = np.maximum(w_grid, 0.0)
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

        ix_center = int(np.argmin(np.abs(x_vec - FAN_CENTER_X)))
        iy_center = int(np.argmin(np.abs(y_vec - FAN_CENTER_Y)))

        # Slice at x = fan centre
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

        # Slice at y = fan centre
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
    circle_x = FAN_CENTER_X + 0.5 * FAN_OUTLET_DIAMETER * np.cos(theta)
    circle_y = FAN_CENTER_Y + 0.5 * FAN_OUTLET_DIAMETER * np.sin(theta)
    circle_z = FAN_VERTICAL_OFFSET_M * np.ones_like(theta)
    ax3d.plot(
        circle_x,
        circle_y,
        circle_z,
        color=(0.0, 0.0, 0.0, FAN_OUTLET_ALPHA),
        linewidth=FAN_OUTLET_EDGE_LW,
        linestyle=FAN_OUTLET_DASH,
        label="Fan outlet",
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
    if not GRID_PRED_XLSX.exists():
        raise FileNotFoundError(f"Missing GP grid workbook: {GRID_PRED_XLSX}")

    x_axis, y_axis, z_axis, w_stack = load_gp_stack(
        xlsx_path=GRID_PRED_XLSX,
        sheet_names=SHEETS,
    )

    x_grid, y_grid, z_grid, dx, dy, dz = make_3d_grid()
    # GP extrapolation is performed in relative z (fan-referenced).
    z_query = np.linspace(REL_Z_EXTRAP_MIN, REL_Z_EXTRAP_MAX, z_grid.shape[2], dtype=float)
    w_stack_extrap = extrapolate_stack_in_z_with_gp(
        z_axis=z_axis,
        w_stack=w_stack,
        z_query=z_query,
    )

    # Evaluate at plotted absolute z levels shifted by the fan vertical offset.
    z_grid_rel = z_grid - FAN_VERTICAL_OFFSET_M
    w_grid = evaluate_field(
        x_grid=x_grid,
        y_grid=y_grid,
        z_grid=z_grid_rel,
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=z_query,
        w_stack=w_stack_extrap,
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

    print(f"Saved 3D GP plot to: {out_3d.resolve()}")


if __name__ == "__main__":
    main()




