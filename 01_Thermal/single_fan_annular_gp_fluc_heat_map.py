from __future__ import annotations

from pathlib import Path

import single_fan_annular_gp_total_fluc_heat_map as base_plot


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Plot Configuration and Data Sources
# 2) Figure Export Entry Point
# =============================================================================

# =============================================================================
# 1) Plot Configuration and Data Sources
# =============================================================================
# Workbook, parameter, and output paths below define the data-provenance boundary for this run.


SHEETS = base_plot.SHEETS
PLOT_VMIN = base_plot.PLOT_VMIN
PLOT_VMAX = base_plot.PLOT_VMAX
CBAR_TICK_STEP = base_plot.CBAR_TICK_STEP
CBAR_LABEL = r"$\sigma_i$ (m $\!$s$^{-1}$)"


build_alpha_cmap = base_plot.build_alpha_cmap
resolve_grid_xlsx = base_plot.resolve_grid_xlsx
resolve_out_dir = base_plot.resolve_out_dir
load_annulus_boundary_levels = base_plot.load_annulus_boundary_levels

# =============================================================================
# 2) Figure Export Entry Point
# =============================================================================
# Entry points write deterministic artifacts so regenerated figures and tables can be compared by path and sheet name.


# Main execution keeps data loading, evaluation, and export order deterministic.
def main() -> None:
    grid_xlsx = resolve_grid_xlsx()
    if not grid_xlsx.exists():
        raise FileNotFoundError(
            f"Missing annular-GP fluctuation workbook: {grid_xlsx}"
        )

    out_dir = resolve_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    base_plot.CBAR_LABEL = CBAR_LABEL
    base_plot.CBAR_TICK_STEP = float(CBAR_TICK_STEP)

    for sheet_name in SHEETS:
        x, y, sigma_fluc = base_plot.load_gp_mean_sheet(
            grid_xlsx,
            f"{sheet_name}_annular_gp_fluc",
        )
        out_png = (
            out_dir
            / f"{sheet_name}_single_annular_gp_fluc_heatmap.png"
        )
        base_plot.plot_continuous_heatmap(
            x=x,
            y=y,
            w=sigma_fluc,
            outpath=out_png,
            plot_vmin=PLOT_VMIN,
            plot_vmax=PLOT_VMAX,
            sheet_name=sheet_name,
        )

    print(f"Saved figures to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
