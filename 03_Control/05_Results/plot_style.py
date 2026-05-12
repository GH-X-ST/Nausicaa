from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Plot style dataclasses and constants
# 2) Colormap and export helpers
# 3) Axes and legend styling helpers
# =============================================================================

# =============================================================================
# 1) Plot Style Dataclasses and Constants
# =============================================================================
# Style constants centralise thesis-figure choices so layout and view are easy to tune.
AXIS_EDGE_LW = 0.80
GRID_COLOR = (0.85, 0.85, 0.85, 1.0)
GRID_LW = 0.40
PLOT_VMIN = 0.0
PLOT_VMAX = 8.0
CBAR_LABEL = r"$w$ (m $\!$s$^{-1}$)"
ACTUAL_START_COLOR = "#20b6c7"
ACTUAL_END_COLOR = "#00244c"
CONTROL_REFERENCE_COLOR = "#E66A2C"
ENVIRONMENT_REFERENCE_COLOR = "#7B2CBF"
SIMULATION_FLOOR_COLOR = (0.35, 0.35, 0.35, 1.0)
SIMULATION_FLOOR_LINESTYLE = ":"
SIMULATION_FLOOR_LINEWIDTH = 0.75
TRUE_SAFETY_VOLUME_COLOR = SIMULATION_FLOOR_COLOR
TRUE_SAFETY_VOLUME_LINESTYLE = SIMULATION_FLOOR_LINESTYLE
TRUE_SAFETY_VOLUME_LINEWIDTH = SIMULATION_FLOOR_LINEWIDTH
CONTROL_REFERENCE_ALPHA = 0.55
ENVIRONMENT_REFERENCE_ALPHA = 0.55
DESIRED_COMMAND_ALPHA = CONTROL_REFERENCE_ALPHA
DESIRED_COMMAND_LINEWIDTH = 1.05

TRAJECTORY_SPECS = {
    "actual": {
        "label": "Actual",
        "color": ACTUAL_END_COLOR,
        "linestyle": "-",
        "linewidth": 1.8,
        "alpha": 1.0,
    },
    "control_reference": {
        "label": "Control reference",
        "color": CONTROL_REFERENCE_COLOR,
        "linestyle": "--",
        "linewidth": 1.25,
        "alpha": CONTROL_REFERENCE_ALPHA,
    },
    "environment_reference": {
        "label": "Environment reference",
        "color": ENVIRONMENT_REFERENCE_COLOR,
        "linestyle": ":",
        "linewidth": 1.25,
        "alpha": ENVIRONMENT_REFERENCE_ALPHA,
    },
}

FAN_OUTLET_DIAMETER_M = 0.8
FAN_OUTLET_Z_M = 0.330
FAN_OUTLET_EDGE_LW = 0.7
FAN_OUTLET_ALPHA = 0.55
FAN_OUTLET_DASH = (0, (2, 2))


@dataclass(frozen=True)
class PlotStyle:
    dpi: int = 600
    figsize: tuple[float, float] = (10.0, 4.7)
    width_ratios: tuple[float, float, float, float] = (2.75, 0.06, 0.30, 0.78)
    height_ratios: tuple[float, float, float] = (1.0, 1.0, 1.0)
    wspace: float = 0.28
    hspace: float = 0.16
    colorbar_vertical_shrink: float = 0.74
    elev_deg: float = 20.0
    azim_deg: float = -120.0
    axis_label_size: float = 10.5
    tick_label_size: float = 8.5
    legend_size: float = 7.4
    command_label_size: float = 8.4
    command_tick_size: float = 7.6
    colorbar_label_size: float = 9.0
    colorbar_tick_size: float = 8.0
    x_labelpad: float = 10.0
    y_labelpad: float = 8.0
    z_labelpad: float = 5.0
    updraft_grid_x: int = 32
    updraft_grid_y: int = 22
    updraft_grid_z: int = 16
    updraft_iso_fracs: tuple[float, float, float] = (0.32, 0.52, 0.72)
    updraft_alpha_exp_rate: float = 3.0
    updraft_alpha_scale: float = 0.42
    updraft_alpha_floor: float = 0.16


# =============================================================================
# 2) Colormap and Export Helpers
# =============================================================================
def thermal_cmap() -> mpl.colors.Colormap:
    try:
        import cmocean

        return cmocean.cm.thermal
    except ImportError:
        return plt.get_cmap("inferno")


def build_alpha_cmap(style: PlotStyle) -> mcolors.ListedColormap:
    base_cmap = thermal_cmap()
    colors = base_cmap(np.linspace(0.0, 1.0, 256))

    # Exponential alpha mapping keeps weak updraft visually subordinate to trajectories.
    t_norm = np.linspace(0.0, 1.0, colors.shape[0])
    exp_scale = np.exp(style.updraft_alpha_exp_rate * t_norm)
    exp_full = np.exp(style.updraft_alpha_exp_rate)
    alpha = (exp_scale - 1.0) / (exp_full - 1.0)
    alpha = style.updraft_alpha_floor + (1.0 - style.updraft_alpha_floor) * alpha
    alpha[0] = 0.0
    colors[:, 3] = style.updraft_alpha_scale * alpha
    return mcolors.ListedColormap(colors)


def save_figure(fig: mpl.figure.Figure, path: Path, style: PlotStyle) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=style.dpi, facecolor="white", bbox_inches="tight")


# =============================================================================
# 3) Axes and Legend Styling Helpers
# =============================================================================
def style_3d_axis(
    ax: mpl.axes.Axes,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    zlim: tuple[float, float],
    style: PlotStyle,
) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_xlabel(r"$x$ (m)", labelpad=style.x_labelpad)
    ax.set_ylabel(r"$y$ (m)", labelpad=style.y_labelpad)
    ax.set_zlabel(r"$z$ (m)", labelpad=style.z_labelpad, rotation=90)
    ax.zaxis.set_rotate_label(False)
    ax.tick_params(axis="x", labelsize=style.tick_label_size, width=0.6, length=2)
    ax.tick_params(axis="y", labelsize=style.tick_label_size, width=0.6, length=2)
    ax.tick_params(axis="z", labelsize=style.tick_label_size, width=0.6, length=2)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))
    ax.view_init(elev=style.elev_deg, azim=style.azim_deg)
    ax.grid(True)

    # Pane and grid styling follows the thermal 3D scripts without copying their layout.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        axis._axinfo["grid"]["color"] = GRID_COLOR
        axis._axinfo["grid"]["linewidth"] = GRID_LW


def style_command_axis(ax: mpl.axes.Axes, ylabel: str, style: PlotStyle) -> None:
    ax.set_ylabel(ylabel, fontsize=style.command_label_size)
    ax.tick_params(labelsize=style.command_tick_size, width=0.6, length=2)
    ax.grid(True, color=GRID_COLOR, linewidth=GRID_LW)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_EDGE_LW)


def style_time_axis(
    ax: mpl.axes.Axes,
    ylabel: str,
    style: PlotStyle,
    show_xlabel: bool = False,
) -> None:
    ax.set_facecolor("white")
    ax.set_ylabel(ylabel, fontsize=style.command_label_size)
    if show_xlabel:
        ax.set_xlabel(r"$t$ (s)", fontsize=style.command_label_size)
    ax.tick_params(labelsize=style.command_tick_size, width=0.6, length=2)
    ax.grid(True, color=GRID_COLOR, linewidth=GRID_LW)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_EDGE_LW)


def style_geometry_axis(
    ax: mpl.axes.Axes,
    xlabel: str,
    ylabel: str,
    style: PlotStyle,
    equal_aspect: bool = False,
) -> None:
    ax.set_facecolor("white")
    ax.set_xlabel(xlabel, fontsize=style.command_label_size)
    ax.set_ylabel(ylabel, fontsize=style.command_label_size)
    ax.tick_params(labelsize=style.command_tick_size, width=0.6, length=2)
    ax.grid(True, color=GRID_COLOR, linewidth=GRID_LW)
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_EDGE_LW)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")


def framed_legend(
    ax: mpl.axes.Axes,
    style: PlotStyle,
    loc: str = "best",
    **kwargs: object,
) -> mpl.legend.Legend | None:
    legend = ax.legend(
        loc=loc,
        frameon=True,
        framealpha=1.0,
        edgecolor="black",
        fontsize=style.legend_size,
        handlelength=1.7,
        handletextpad=0.45,
        columnspacing=0.9,
        borderpad=0.45,
        labelspacing=0.25,
        **kwargs,
    )
    if legend is not None:
        legend.get_frame().set_linewidth(AXIS_EDGE_LW)
    return legend
