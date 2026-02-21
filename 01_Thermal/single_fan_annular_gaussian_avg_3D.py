"""
Plot a 3D annular-Gaussian updraft field using fitted parameter output.

The model is
    w_model(r, z) = A_ring(z) * exp(-((r - r_ring(z)) / delta_r(z))**2)
where [A_ring, r_ring, delta_r, w0] are read directly from the
precomputed PCHIP table written by single_fan_annular_gaussian_avg_fit.py.

Input is read from single_fan_annular_gaussian_avg_fit.py output
(by default: B_results/single_annular_avg_params_pchip.xls/.xlsx).
"""

###### Initialization

### Imports
from pathlib import Path
from typing import Tuple

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

PARAMS_XLSX = Path("B_results/single_annular_avg_params_pchip.xls")
PARAMS_SHEET = "single_annular_avg_pchip"

OUT_DIR = Path("A_figures/Single_Fan_Annular_Gaussian_Avg")
OUT_3D_NAME = "single_annular_gaussian_avg_3d.png"
# Match heat-map-main output aspect ratio
FIGSIZE_3D = (7.45/1.3, 3.0)

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

# Typography (match single_fan_annular_gaussian_avg_heat_map_main.py)
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
Z_MIN, Z_MAX = 0.0, 3.5

# 3D sampling resolution
NX_3D, NY_3D, NZ_3D = 240, 160, 100

# Isosurface levels as fractions of max(w)
ISO_MIN_FRAC = 0.01
ISO_MAX_FRAC = 0.97
N_ISO_LEVELS = 10

# Include fitted baseline term if present in table.
# Requested model is ring-only, so keep False by default.
INCLUDE_W0 = True

# Exponential opacity mapping versus normalized w (= 0..1).
# alpha(0) = 0 (fully transparent), alpha(1) = 1 (fully opaque).
ALPHA_EXP_RATE = 3.0

# Interior visibility aids.
SHOW_CENTER_SLICES = True
SLICE_EDGE_COLOR = (0.0, 0.0, 0.0, 0.20)
SLICE_EDGE_LW = 0.2


### Helpers
REQUIRED_COLUMNS = ("z_m", "A_ring", "r_ring", "delta_r")


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


def load_params_table(
    xlsx_path: Path,
    sheet_name: str,
) -> pd.DataFrame:
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


def extract_param_arrays(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract validated arrays from the parameter table.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    local = df.copy()
    w0_default = np.zeros(len(local), dtype=float)
    if "w0" not in local.columns:
        local["w0"] = w0_default

    for col in ("z_m", "A_ring", "r_ring", "delta_r", "w0"):
        local[col] = pd.to_numeric(local[col], errors="coerce")

    local = local.dropna(subset=["z_m", "A_ring", "r_ring", "delta_r", "w0"])
    if local.empty:
        raise ValueError("No valid parameter rows after cleaning.")

    z_vals = local["z_m"].to_numpy(dtype=float)
    a_ring = local["A_ring"].to_numpy(dtype=float)
    r_ring = local["r_ring"].to_numpy(dtype=float)
    delta_r = local["delta_r"].to_numpy(dtype=float)
    w0_vals = local["w0"].to_numpy(dtype=float)

    order = np.argsort(z_vals)
    z_vals = z_vals[order]
    a_ring = a_ring[order]
    r_ring = r_ring[order]
    delta_r = delta_r[order]
    w0_vals = w0_vals[order]

    if np.any(np.diff(z_vals) <= 0.0):
        raise ValueError("z_m must be strictly increasing for interpolation.")

    if np.any(a_ring <= 0.0):
        raise ValueError("A_ring must be positive for log-space interpolation.")

    if np.any(delta_r <= 0.0):
        raise ValueError("delta_r must be positive for log-space interpolation.")

    return z_vals, a_ring, r_ring, delta_r, w0_vals


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
    z_vals: np.ndarray,
    a_ring: np.ndarray,
    r_ring: np.ndarray,
    delta_r: np.ndarray,
    w0_vals: np.ndarray,
    include_w0: bool,
) -> np.ndarray:
    """
    Evaluate w(x, y, z_abs) from the precomputed PCHIP table.
    """
    r_grid = np.sqrt((x_grid - FAN_CENTER_X) ** 2 + (y_grid - FAN_CENTER_Y) ** 2)

    # Convert absolute z to model-relative z referenced at fan height.
    z_rel = z_grid - FAN_VERTICAL_OFFSET_M

    # Only evaluate inside the tabulated z range; keep zero outside.
    w_grid = np.zeros_like(z_grid, dtype=float)
    mask = (z_rel >= float(np.min(z_vals))) & (z_rel <= float(np.max(z_vals)))
    if np.any(mask):
        z_eval = z_rel[mask]
        r_eval = r_grid[mask]

        a_eval = np.interp(z_eval, z_vals, a_ring)
        r_ring_eval = np.interp(z_eval, z_vals, r_ring)
        delta_eval = np.interp(z_eval, z_vals, delta_r)

        w_eval = a_eval * np.exp(-((r_eval - r_ring_eval) / delta_eval) ** 2)
        if include_w0:
            w_eval += np.interp(z_eval, z_vals, w0_vals)

        w_grid[mask] = w_eval

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

        # Outline both slice planes so the cut location is clear.
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

        # Shift from grid coordinates to physical coordinates.
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

    # Fan outlet ring at the specified vertical offset.
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

    # Keep all axis tick labels at 2 decimal places.
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
        # Use physical domain extents so x:y:z follows the real ratio.
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

    # Light grid and pane styling to match ground-effect 3D settings.
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
    z_vals, a_ring, r_ring, delta_r, w0_vals = extract_param_arrays(params_df)

    x_grid, y_grid, z_grid, dx, dy, dz = make_3d_grid()
    w_grid = evaluate_field(
        x_grid=x_grid,
        y_grid=y_grid,
        z_grid=z_grid,
        z_vals=z_vals,
        a_ring=a_ring,
        r_ring=r_ring,
        delta_r=delta_r,
        w0_vals=w0_vals,
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



