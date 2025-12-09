###### Initialization

### Import
import aerosandbox as asb
import aerosandbox.numpy as np
import numpy as onp
import aerosandbox.tools.units as u
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes
import matplotlib as mpl
import matplotlib.colors as mcolors
import copy
import subprocess
import os


### Code version
def get_git_version():

    try:
        desc = subprocess.check_output(
            ["git", "describe", "--always", "--dirty", "--tags"],
            stderr = subprocess.DEVNULL,
        ).decode("utf-8").strip()
        return desc
    except Exception:
        return "unknown"


CODE_VERSION = get_git_version()

### Aerosandbox setup
opti = asb.Opti()
# variable_categories_to_freeze = "all",
# freeze_style = "float"

make_plots = True

base = plt.cm.YlOrRd
colors = base(np.linspace(0, 1, 256))
colors[0] = [1, 1, 1, 1]  # white
cmap_white0 = mcolors.ListedColormap(colors)

##### Thermal Vertical Velocity Field Model

### Compute w(x, y, z) using the Gaussian jet/plume model

def vertical_velocity_field(Q_v, r_th0, k, x, y, z, z0, x_center, y_center):
    # Q_v      - Vertical volume flux (m^3/s)
    # r_th0    - Core radius at z0 (m)
    # k        - Empirical spreading rate
    # z0       - referemce height for r_th0 (m)
    # x_center - x-coordinate of thermal centre (m)
    # y_center - y-coordinate of thermal centre (m)

    x = onp.asarray(x)
    y = onp.asarray(y)
    z = onp.asarray(z)

    # radial distance from thermal center
    r = onp.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)

    # core radius as function of height
    r_th = r_th0 + k * (z - z0)
    r_th = onp.maximum(r_th, 1e-6) # avoid negative radius 

    # peak vertical velocity
    w_th = Q_v / (onp.pi * r_th ** 2)

    # vertical velocity
    w = w_th * onp.exp(-(r / r_th) ** 2)

    mask = z < z0
    w = onp.where(mask, 0.0, w)

    return w

### Setup

if __name__ == "__main__":

    # output
    out_dir = "figures"
    os.makedirs(out_dir, exist_ok=True) 

    # CAMAX30 fan parameters
    Q_v      = 1.69 # (m^3/s)
    x_center = 4.0  # (m)
    y_center = 2.5  # (m)

    # plume parameters
    r_th0 = 0.381 # assume core radius equal to fan radius (m)
    k     = 0.10  # typical turbulent plume spreading rate
    z0    = 0.50  # reference height at fan centre (m)

    # flight arena volume
    x_min, x_max = 0.0, 8.0
    y_min, y_max = 0.0, 5.0
    z_min, z_max = 0.0, 3.5

    ### Plot 2D slice contour of w
    # grid resolution
    Nx, Ny = 120, 80
    x = onp.linspace(x_min, x_max, Nx)
    y = onp.linspace(y_min, y_max, Ny)
    X, Y = onp.meshgrid(x, y, indexing="xy")

    # choose a single height to visualise
    z_plot = 1.5
    Z_slice = z_plot * onp.ones_like(X)

    # compute vertical velocity field at that height
    W_slice = vertical_velocity_field(
        Q_v = Q_v, r_th0 = r_th0, k = k, x = X, y = Y, z = Z_slice,
        z0 = z0, x_center = x_center, y_center = y_center
    )

    fig_2D, ax2 = plt.subplots(figsize=(6, 5))

    levels = np.linspace(W_slice.min(), W_slice.max(), 256)

    contour = ax2.contourf(X, Y, W_slice, levels = levels, cmap = cmap_white0)

    cbar = fig_2D.colorbar(contour, ax = ax2)
    cbar.set_label(f"vertical velocity, w (m/s), at z = {z_plot:.2f} m")

    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    # ax2.set_title(f"Vertical velocity at z = {z_plot:.2f} m")
    ax2.set_aspect("equal", adjustable = "box")
    ax2.grid(True, linestyle = ":", linewidth = 0.5)

    # output
    fig_2D.tight_layout()
    fig_2D.savefig(os.path.join(out_dir, "thermal_slice.png"),
                   dpi = 300, bbox_inches = "tight")

    ### Plot 3D plume shape coloured by w
    # coarser 3D grid
    Nx3D, Ny3D, Nz3D = 150, 100, 50
    x3 = onp.linspace(x_min, x_max, Nx3D)
    y3 = onp.linspace(y_min, y_max, Ny3D)
    z3 = onp.linspace(z_min, z_max, Nz3D)
    
    dx = x3[1] - x3[0]
    dy = y3[1] - y3[0]
    dz = z3[1] - z3[0]

    X3, Y3, Z3 = onp.meshgrid(x3, y3, z3, indexing = "ij")

    W3 = vertical_velocity_field(
        Q_v = Q_v, r_th0 = r_th0, k = k,
        x = X3, y = Y3, z = Z3,
        z0 = z0, x_center = x_center, y_center = y_center
    )
    W3 = onp.where(Z3 >= z0, W3, 0.0)

    w_max = W3.max()

    # colour layers
    iso_fracs  = onp.linspace(0.15, 0.95, 10) # 10 isosurfaces from 15% to 95% of max w
    iso_levels = [f * w_max for f in iso_fracs]

    fig_3D = plt.figure(figsize=(8, 6))
    ax3 = fig_3D.add_subplot(111, projection = "3d")

    # thermal isosurfaces
    norm = mpl.colors.Normalize(vmin = 0.0, vmax = w_max)

    for level in iso_levels:
        verts, faces, _, _ = marching_cubes(W3, level = level, spacing = (dx, dy, dz))
        verts[:, 0] += x_min
        verts[:, 1] += y_min
        verts[:, 2] += z_min

        alpha = 0.10 + 1.0 * (level / w_max) ** 33.0
        mesh = Poly3DCollection(verts[faces], linewidths = 0.05, alpha = alpha, zorder = 10)
        mesh.set_facecolor(cmap_white0(norm(level)))
        mesh.set_edgecolor("none")

        ax3.add_collection3d(mesh)

    ax3.set_xlim(x_min, x_max)
    ax3.set_ylim(y_min, y_max)
    ax3.set_zlim(z_min, z_max)

    ax3.set_xlabel("x (m)")
    ax3.set_ylabel("y (m)")
    ax3.set_zlabel("z (m)")

    # realistic aspect ratio
    ax3.set_box_aspect((x_max - x_min, y_max - y_min, z_max - z_min))

    mappable = mpl.cm.ScalarMappable(norm = norm, cmap = cmap_white0)
    mappable.set_array([])
    fig_3D.colorbar(mappable, ax = ax3, shrink = 0.7, aspect = 15, label = "vertical velocity, w (m/s)")

    ax3.xaxis._axinfo["grid"]["color"] = "k"
    ax3.xaxis._axinfo["grid"]["linestyle"] = ":"
    ax3.xaxis._axinfo["grid"]["linewidth"] = 0.5
    ax3.yaxis._axinfo["grid"]["color"] = "k"
    ax3.yaxis._axinfo["grid"]["linestyle"] = ":"
    ax3.yaxis._axinfo["grid"]["linewidth"] = 0.5
    ax3.zaxis._axinfo["grid"]["color"] = "k"
    ax3.zaxis._axinfo["grid"]["linestyle"] = ":"
    ax3.zaxis._axinfo["grid"]["linewidth"] = 0.5

    ax3.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax3.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax3.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    ax3.view_init(elev = 11, azim = -111)
    
    # output
    fig_3D.tight_layout()
    fig_3D.savefig(os.path.join(out_dir, "thermal_plume.png"),
                   dpi = 300, bbox_inches = "tight") 
    
    plt.show()