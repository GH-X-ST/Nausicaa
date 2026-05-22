import os
import subprocess

import numpy as np
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
# Axes3D import registers Matplotlib's 3D projection used by these figures.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Version Provenance
# 2) Single-Plume Model Evaluation
# 3) Multi-Plume Model Evaluation
# 4) Figure Generation Entry Point
# =============================================================================

# =============================================================================
# 1) Version Provenance
# =============================================================================
# The version stamp records the code state used to generate exported analytic grids.

# Git provenance tags generated analytic grids with the source revision when available.
def get_git_version() -> str:
    """
    Return a short git description string (tag/commit), or 'unknown' if not in a repo.
    """
    try:
        desc = subprocess.check_output(
            ["git", "describe", "--always", "--dirty", "--tags"],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
        return desc
    except Exception:
        return "unknown"


CODE_VERSION = get_git_version()

make_plots = True

# Low-speed fade prevents near-zero updraft from dominating the figure background.
base_cmap = plt.cm.YlOrRd
colors = base_cmap(np.linspace(0.0, 1.0, 256))

N_fade = 15
first_color = colors[N_fade].copy()

for i in range(N_fade):
    t_fade = i / (N_fade - 1.0)
    colors[i] = (1.0 - t_fade) * np.array([1.0, 1.0, 1.0, 1.0]) + t_fade * first_color

cmap_white0 = mcolors.ListedColormap(colors)


# =============================================================================
# 2) Single-Plume Model Evaluation
# =============================================================================
# Plume formulas use arena-frame metres and return vertical velocity in m/s.

# Analytic field returns vertical velocity in m/s in the arena frame.
def vertical_velocity_field(
    Q_v: float,
    R_th0: float,
    k_th: float,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    z0: float,
    x_th: float,
    y_th: float,
) -> np.ndarray:
    """
    Compute w(x, y, z) for a single axisymmetric Gaussian thermal plume:

        w(R, z) = w_th(z) * exp{ - (R / R_th(z))^2 }
        Q_v     = π R_th(z)^2 w_th(z)

    Parameters
    ----------
    Q_v : float
        Vertical volume flux (m^3 s^-1).
    R_th0 : float
        Core radius R_th at reference height z0 (m).
    k_th : float
        Empirical spreading rate in R_th(z) = R_th0 + k_th (z - z0).
    x, y, z : np.ndarray
        Sample locations in the flight arena (m).
    z0 : float
        Reference height for R_th0 (m).
    x_th, y_th : float
        Thermal centre coordinates (m).

    Returns
    -------
    w : np.ndarray
        Vertical velocity field w(x, y, z) in m s^-1.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    # Radius uses horizontal arena distance; height affects plume width separately.
    R = np.sqrt((x - x_th) ** 2 + (y - y_th) ** 2)

    # Core radius increases linearly with z according to the analytic plume assumption.
    R_th = R_th0 + k_th * (z - z0)
    # Radius is floored to keep the volume-flux equation physically valid.
    R_th = np.maximum(R_th, 1e-6)

    # peak vertical velocity w_th(z) from Q_v = π R_th^2 w_th
    w_th = Q_v / (np.pi * R_th**2)

    # Gaussian radial decay converts peak vertical velocity into the spatial field.
    w = w_th * np.exp(-(R / R_th) ** 2)

    # Below z0 the analytic baseline intentionally reports no updraft.
    w = np.where(z < z0, 0.0, w)

    return w


# =============================================================================
# 3) Multi-Plume Model Evaluation
# =============================================================================
# Plume formulas use arena-frame metres and return vertical velocity in m/s.

# Multi-fan analytic fields superpose fan-centred plumes in arena coordinates.
def vertical_velocity_field_multi(
    Q_v: float,
    R_th0: float,
    k_th: float,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    z0: float,
    thermal_centres: list[tuple[float, float]],
) -> np.ndarray:
    """
    Superpose w(x, y, z) from multiple identical thermals.
    """
    w_total = np.zeros_like(x, dtype=float)

    for x_th, y_th in thermal_centres:
        w_total += vertical_velocity_field(
            Q_v=Q_v,
            R_th0=R_th0,
            k_th=k_th,
            x=x,
            y=y,
            z=z,
            z0=z0,
            x_th=x_th,
            y_th=y_th,
        )

    return w_total


# =============================================================================
# 4) Figure Generation Entry Point
# =============================================================================
# Entry points write deterministic artifacts so regenerated figures and tables can be compared by path and sheet name.

if __name__ == "__main__" and make_plots:

    # Output path separates analytic baseline figures from measured-map products.
    output_dir = "A_figures/Four_Fan_Analytic"
    os.makedirs(output_dir, exist_ok=True)

    # CAMAX30 parameters define the analytic baseline rather than measured workbook data.
    # Vertical volume flux uses SI units for the CAMAX30 analytic baseline.
    Q_v = 1.69
    # Thermal-array centre coordinates are arena-frame metres.
    x_th_centre = 4.2
    y_th_centre = 2.4

    # Plume parameters define the linear core-radius law used by the analytic baseline.
    # Core radius, spreading rate, and reference height define R_th(z) in metres.
    R_th0 = 0.381
    k_th = 0.10
    z0 = 0.50

    # Thermal centres encode the four-fan layout in arena coordinates.
    # Fan spacing uses the analytic thermal-array layout in arena metres.
    fan_spacing = 2.0 * R_th0 + 1.7
    thermal_centres = [
        (x_th_centre - fan_spacing / 2.0, y_th_centre - fan_spacing / 2.0),
        (x_th_centre + fan_spacing / 2.0, y_th_centre - fan_spacing / 2.0),
        (x_th_centre - fan_spacing / 2.0, y_th_centre + fan_spacing / 2.0),
        (x_th_centre + fan_spacing / 2.0, y_th_centre + fan_spacing / 2.0),
    ]

    # Arena bounds define the volume over which analytic figures are sampled.
    x_min, x_max = 0.0, 8.4
    y_min, y_max = 0.0, 4.8
    z_min, z_max = 0.0, 3.5

    # 2D diagnostic slice is sampled at the fixed height below.
    n_x_2d, n_y_2d = 120, 80
    x_2d = np.linspace(x_min, x_max, n_x_2d)
    y_2d = np.linspace(y_min, y_max, n_y_2d)
    X_2d, Y_2d = np.meshgrid(x_2d, y_2d, indexing="xy")

    # Slice height is fixed for comparable 2D analytic diagnostics.
    # Slice height is specified in arena metres for the 2D diagnostic plot.
    z_slice = 1.0
    Z_slice = z_slice * np.ones_like(X_2d)

    # Single-height multi-plume velocity slice in m/s.
    W_slice = vertical_velocity_field_multi(
        Q_v=Q_v,
        R_th0=R_th0,
        k_th=k_th,
        x=X_2d,
        y=Y_2d,
        z=Z_slice,
        z0=z0,
        thermal_centres=thermal_centres,
    )

    fig_2d, ax2d = plt.subplots(figsize=(6, 5))

    levels = np.linspace(W_slice.min(), W_slice.max(), 256)
    contour_2d = ax2d.contourf(
        X_2d,
        Y_2d,
        W_slice,
        levels=levels,
        cmap=cmap_white0,
    )

    cbar_2d = fig_2d.colorbar(contour_2d, ax=ax2d)
    cbar_2d.set_label(f"vertical velocity, w (m/s), at z = {z_slice:.2f} m")

    ax2d.set_xlabel("x (m)")
    ax2d.set_ylabel("y (m)")
    ax2d.set_aspect("equal", adjustable="box")
    ax2d.grid(True, linestyle=":", linewidth=0.5)

    fig_2d.tight_layout()
    fig_2d.savefig(
        os.path.join(output_dir, "multi_thermal_slice.png"),
        dpi=300,
        bbox_inches="tight",
    )

    # 3D plume volume is sampled on a regular arena grid for isosurface rendering.
    n_x_3d, n_y_3d, n_z_3d = 150, 100, 50
    x_3d = np.linspace(x_min, x_max, n_x_3d)
    y_3d = np.linspace(y_min, y_max, n_y_3d)
    z_3d = np.linspace(z_min, z_max, n_z_3d)

    dx = x_3d[1] - x_3d[0]
    dy = y_3d[1] - y_3d[0]
    dz = z_3d[1] - z_3d[0]

    X_3d, Y_3d, Z_3d = np.meshgrid(x_3d, y_3d, z_3d, indexing="ij")

    W_3d = vertical_velocity_field_multi(
        Q_v=Q_v,
        R_th0=R_th0,
        k_th=k_th,
        x=X_3d,
        y=Y_3d,
        z=Z_3d,
        z0=z0,
        thermal_centres=thermal_centres,
    )

    w_max = float(W_3d.max())

    # Isosurface levels are fractions of the local maximum for figure comparability.
    # Ten fractional levels give comparable plume-volume slices across analytic figures.
    iso_fracs = np.linspace(0.15, 0.95, 10)
    iso_levels = [frac * w_max for frac in iso_fracs]

    fig_3d = plt.figure(figsize=(8, 6))
    ax3d = fig_3d.add_subplot(111, projection="3d")

    norm = mpl.colors.Normalize(vmin=0.0, vmax=w_max)

    for level in iso_levels:
        verts, faces, _, _ = marching_cubes(
            W_3d,
            level=level,
            spacing=(dx, dy, dz),
        )

        # Marching-cubes vertices are shifted from array indices into arena coordinates.
        verts[:, 0] += x_min
        verts[:, 1] += y_min
        verts[:, 2] += z_min

        alpha = 0.10 + (level / w_max) ** 3.3
        mesh = Poly3DCollection(
            verts[faces],
            linewidths=0.05,
            alpha=alpha,
            zorder=10,
        )
        mesh.set_facecolor(cmap_white0(norm(level)))
        mesh.set_edgecolor("none")

        ax3d.add_collection3d(mesh)

    # Core-radius rings mark the analytic reference radius at z0.
    theta = np.linspace(0.0, 2.0 * np.pi, 200)

    for x_th, y_th in thermal_centres:
        circle_x = x_th + R_th0 * np.cos(theta)
        circle_y = y_th + R_th0 * np.sin(theta)
        circle_z = z0 * np.ones_like(theta)

        ax3d.plot(
            circle_x,
            circle_y,
            circle_z,
            color="k",
            linewidth=1.3,
            zorder=0,
        )

    # Axis limits preserve the physical arena bounds.
    ax3d.set_xlim(x_min, x_max)
    ax3d.set_ylim(y_min, y_max)
    ax3d.set_zlim(z_min, z_max)

    # Axis labels use arena coordinates in metres.
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")

    # Aspect ratio follows arena dimensions so plume shape is not distorted.
    ax3d.set_box_aspect((x_max - x_min, y_max - y_min, z_max - z_min))

    # Colorbar reports vertical velocity in m/s.
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_white0)
    mappable.set_array([])
    fig_3d.colorbar(
        mappable,
        ax=ax3d,
        shrink=0.7,
        aspect=15,
        label="vertical velocity, w (m/s)",
    )

    # Grid styling (matplotlib private API, acceptable for visual only)
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis._axinfo["grid"]["color"] = "k"
        axis._axinfo["grid"]["linestyle"] = ":"
        axis._axinfo["grid"]["linewidth"] = 0.5
        axis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Camera angle exposes plume height and plan-view spread.
    ax3d.view_init(elev=7.0, azim=-111.0)

    fig_3d.tight_layout()
    fig_3d.savefig(
        os.path.join(output_dir, "multi_thermal_plume.png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
