from __future__ import annotations

from pathlib import Path

import numpy as np

import single_fan_annular_gp_heat_map_main_appendix as base_plot
import single_fan_annular_gaussian_var_heat_map_main as model_plot


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Figure Routing Constants
# 2) Figure Export Entry Point
# =============================================================================

# =============================================================================
# 1) Figure Routing Constants
# =============================================================================
# Routing constants keep thesis and appendix figures traceable to the same metric source tables.


OUT_DIR = Path("A_figures/Single_Fan_Annular_Gaussian_Var")

# =============================================================================
# 2) Figure Export Entry Point
# =============================================================================
# Entry points write deterministic artifacts so regenerated figures and tables can be compared by path and sheet name.


# Main execution keeps data loading, evaluation, and export order deterministic.
def main() -> None:
    params_df = model_plot.load_ring_params(model_plot.PARAMS_XLSX)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for sheet_name in model_plot.SHEETS:
        x, y, _ = model_plot.read_slice_from_sheet(
            model_plot.XLSX_PATH,
            sheet_name,
        )
        z_m = model_plot.parse_sheet_height_m(sheet_name)
        amp, r_ring, delta_r, w0 = model_plot.params_for_height(params_df, z_m)

        x_grid, y_grid = model_plot.build_continuous_grid(x, y)
        xc, yc = model_plot.FAN_CENTER_XY
        r = np.sqrt((x_grid - xc) ** 2 + (y_grid - yc) ** 2)
        w_model = w0 + amp * np.exp(-((r - r_ring) / delta_r) ** 2)

        out_png = (
            OUT_DIR
            / (
                f"{sheet_name}_"
                "single_annular_gaussian_var_heatmap_main_appendix.png"
            )
        )
        base_plot.plot_continuous_heatmap(
            x=x_grid[0, :],
            y=y_grid[:, 0],
            w_field=w_model,
            outpath=out_png,
        )

    print(f"Saved figures to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
