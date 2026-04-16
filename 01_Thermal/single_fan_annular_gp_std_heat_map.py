from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

from single_fan_gp_heat_map import load_gp_mean_sheet


SHEETS = ["z020", "z035", "z050", "z075", "z110", "z160", "z220"]
GP_GRID_XLSX = Path(
    "B_results/Single_Fan_Annular_GP/single_annular_gp_grid_predictions.xlsx"
)
OUT_DIR = Path("A_figures/Single_Fan_Annular_GP")
PLOT_VMIN = 0.0
PLOT_VMAX = 0.4
CBAR_TICK_STEP = 0.05

CBAR_LABEL = r"$\sigma_{\mathrm{res}}$ (m $\!$s$^{-1}$)"
XLABEL = r"$x$ (m)"
YLABEL = r"$y$ (m)"

AXIS_EDGE_LW = 0.80
CBAR_EDGE_LW = AXIS_EDGE_LW
GRID_COLOR = (0.85, 0.85, 0.85, 1.0)
GRID_LINEWIDTH = 0.4

FAN_OUTLET_X = 4.2
FAN_OUTLET_Y = 2.4
FAN_OUTLET_DIAMETER = 0.8
FAN_OUTLET_EDGE_LW = 1.1
FAN_OUTLET_ALPHA = 0.6
FAN_OUTLET_DASH = (0, (2, 2))


def build_alpha_cmap():
    """Return an opaque coolwarm colormap for std maps."""
    return plt.get_cmap("coolwarm")


def centers_to_edges(c: np.ndarray) -> np.ndarray:
    """Convert cell centers to cell edges for pcolormesh."""
    c = np.asarray(c, dtype=float)
    if c.size < 2:
        raise ValueError("Need at least 2 center points to compute edges.")
    edges = np.empty(c.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (c[:-1] + c[1:])
    edges[0] = c[0] - 0.5 * (c[1] - c[0])
    edges[-1] = c[-1] + 0.5 * (c[-1] - c[-2])
    return edges


def plot_continuous_heatmap(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    outpath: Path,
    plot_vmin: float,
    plot_vmax: float,
) -> None:
    """Plot predictive-std heat map in the annular-GP heat-map style."""
    x_edges = centers_to_edges(x)
    y_edges = centers_to_edges(y)

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.edgecolor": "k",
            "axes.linewidth": AXIS_EDGE_LW,
            "patch.edgecolor": "k",
        }
    )

    fig, ax = plt.subplots(figsize=(6.8, 5.6), dpi=600)
    cmap_alpha = build_alpha_cmap()
    im = ax.pcolormesh(
        x_edges,
        y_edges,
        w,
        shading="auto",
        cmap=cmap_alpha,
        vmin=plot_vmin,
        vmax=plot_vmax,
    )

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

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.6%", pad=0.15)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(CBAR_LABEL)
    cbar.set_ticks(
        np.arange(
            PLOT_VMIN,
            PLOT_VMAX + 0.5 * CBAR_TICK_STEP,
            CBAR_TICK_STEP,
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
    xticks = np.arange(0.0, 8.4 + 1e-9, 0.6)
    yticks = np.arange(0.0, 4.8 + 1e-9, 0.4)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f"{v:.2f}" for v in xticks])
    ax.set_yticklabels([f"{v:.2f}" for v in yticks])
    ax.tick_params(axis="x", labelrotation=30)
    ax.tick_params(axis="both", which="major", length=2, width=0.6)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.97, -0.18),
        frameon=True,
        framealpha=1.0,
        edgecolor="black",
        fontsize=7,
        handlelength=1.5,
        borderpad=0.7,
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


def main() -> None:
    if not GP_GRID_XLSX.exists():
        raise FileNotFoundError(
            f"Missing annular-GP grid workbook: {GP_GRID_XLSX}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sheet_name in SHEETS:
        x, y, w_std = load_gp_mean_sheet(
            GP_GRID_XLSX,
            f"{sheet_name}_annular_gp_std",
        )
        out_png = OUT_DIR / f"{sheet_name}_single_annular_gp_std_heatmap.png"
        plot_continuous_heatmap(
            x=x,
            y=y,
            w=w_std,
            outpath=out_png,
            plot_vmin=PLOT_VMIN,
            plot_vmax=PLOT_VMAX,
        )

    print(f"Saved figures to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
