from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

import single_fan_gp_heat_map_main as base_plot
from single_fan_annular_gp_total_fluc_heat_map import (
    ANNULUS_EDGE_COLOR,
    ANNULUS_EDGE_DASH,
    CBAR_LABEL,
    CBAR_TICK_STEP,
    FAN_CENTER_XY,
    PLOT_VMIN,
    PLOT_VMAX,
    SHEETS,
    build_alpha_cmap,
    build_display_grid,
    interpolate_to_display_grid,
    load_annulus_boundary_levels,
    resolve_grid_xlsx,
    resolve_out_dir,
)


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Plot Configuration and Data Sources
# 2) Workbook Loading and Plot Construction
# 3) Figure Export Entry Point
# =============================================================================

# =============================================================================
# 1) Plot Configuration and Data Sources
# =============================================================================
# Workbook, parameter, and output paths below define the data-provenance boundary for this run.


ANNULUS_EDGE_LW_MAIN = 0.60

# =============================================================================
# 2) Workbook Loading and Plot Construction
# =============================================================================
# Parsing and plotting helpers keep measured workbook coordinates in arena metres.


# Annulus overlays show the binning interface between measured profiles and GP outputs.
def plot_continuous_heatmap_with_annuli(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    outpath: Path,
    sheet_name: str,
) -> None:
    x_edges = base_plot.centers_to_edges(x)
    y_edges = base_plot.centers_to_edges(y)

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": base_plot.LEGEND_FONTSIZE,
        "axes.edgecolor": "k",
        "axes.linewidth": base_plot.AXIS_EDGE_LW,
        "patch.edgecolor": "k",
    })

    fig, ax = plt.subplots(figsize=(5.2, 3.0), dpi=600)
    im = ax.pcolormesh(
        x_edges,
        y_edges,
        w,
        shading="auto",
        cmap=build_alpha_cmap(),
        vmin=PLOT_VMIN,
        vmax=PLOT_VMAX,
    )

    for radius in load_annulus_boundary_levels(sheet_name):
        if not np.isfinite(radius) or radius <= 0.0:
            continue
        ax.add_patch(
            Circle(
                FAN_CENTER_XY,
                radius=float(radius),
                fill=False,
                edgecolor=ANNULUS_EDGE_COLOR,
                linewidth=ANNULUS_EDGE_LW_MAIN,
                linestyle=ANNULUS_EDGE_DASH,
                zorder=4,
                clip_on=True,
            )
        )

    outlet = Circle(
        (base_plot.FAN_OUTLET_X, base_plot.FAN_OUTLET_Y),
        radius=base_plot.FAN_OUTLET_DIAMETER / 2.0,
        fill=False,
        edgecolor=(0, 0, 0, base_plot.FAN_OUTLET_ALPHA),
        linewidth=base_plot.FAN_OUTLET_EDGE_LW,
        linestyle=base_plot.FAN_OUTLET_DASH,
        label="Fan outlet",
        zorder=5,
        clip_on=True,
    )
    ax.add_patch(outlet)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.990%", pad=0.22)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(CBAR_LABEL, fontsize=9)
    cbar.set_ticks(base_plot.build_cbar_ticks(PLOT_VMIN, PLOT_VMAX))
    cbar.formatter = FormatStrFormatter("%.2f")
    cbar.update_ticks()
    cbar.ax.tick_params(width=0.6, length=2, labelsize=9)
    cbar.outline.set_linewidth(base_plot.CBAR_EDGE_LW)
    cbar.outline.set_edgecolor("k")
    cbar.outline.set_visible(True)
    cbar.ax.patch.set_edgecolor("k")
    cbar.ax.patch.set_linewidth(base_plot.CBAR_EDGE_LW)
    cbar.ax.set_frame_on(True)
    for spine in cbar.ax.spines.values():
        spine.set_edgecolor("k")
        spine.set_linewidth(base_plot.CBAR_EDGE_LW)

    ax.set_xlabel(base_plot.XLABEL)
    ax.set_ylabel(base_plot.YLABEL)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axisbelow(True)
    ax.grid(True, color=base_plot.GRID_COLOR, linewidth=base_plot.GRID_LINEWIDTH)
    for spine in ax.spines.values():
        spine.set_linewidth(base_plot.AXIS_EDGE_LW)
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
        fontsize=(base_plot.LEGEND_FONTSIZE - 0.2),
        handlelength=1.5,
        borderpad=0.5,
        labelspacing=0.2,
    )
    leg = ax.get_legend()
    if leg is not None:
        leg.get_frame().set_linewidth(base_plot.AXIS_EDGE_LW)

    ax.set_xlim(0.0, 8.4)
    ax.set_ylim(0.0, 4.8)

    fig.tight_layout()
    ax_pos = ax.get_position()
    cax_pos = cax.get_position()
    cax.set_position([cax_pos.x0, ax_pos.y0, cax_pos.width, ax_pos.height * 0.82])
    fig.savefig(outpath, bbox_inches="tight", facecolor="white", dpi=600)
    plt.close(fig)

# =============================================================================
# 3) Figure Export Entry Point
# =============================================================================
# Entry points write deterministic artifacts so regenerated figures and tables can be compared by path and sheet name.


# Main execution keeps data loading, evaluation, and export order deterministic.
def main() -> None:
    grid_xlsx = resolve_grid_xlsx()
    if not grid_xlsx.exists():
        raise FileNotFoundError(
            f"Missing annular-GP total-fluctuation workbook: {grid_xlsx}"
        )

    out_dir = resolve_out_dir()
    base_plot.CBAR_LABEL = CBAR_LABEL
    base_plot.PLOT_VMIN = float(PLOT_VMIN)
    base_plot.PLOT_VMAX = float(PLOT_VMAX)
    base_plot.CBAR_TICK_STEP = float(CBAR_TICK_STEP)
    base_plot.build_alpha_cmap = build_alpha_cmap

    for sheet_name in SHEETS:
        x, y, sigma_total = base_plot.load_gp_mean_sheet(
            grid_xlsx,
            f"{sheet_name}_annular_gp_total_fluc",
        )
        out_png = (
            out_dir
            / f"{sheet_name}_single_annular_gp_total_fluc_heatmap_main.png"
        )
        x_dense, y_dense = build_display_grid(x, y)
        sigma_dense = interpolate_to_display_grid(
            x=x,
            y=y,
            w=sigma_total,
            x_grid=x_dense,
            y_grid=y_dense,
        )
        plot_continuous_heatmap_with_annuli(
            x=x_dense[0, :],
            y=y_dense[:, 0],
            w=sigma_dense,
            outpath=out_png,
            sheet_name=sheet_name,
        )

    print(f"Saved figures to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
