from __future__ import annotations

import single_fan_annular_gp_fluc_heat_map as fluc_plot
import single_fan_annular_gp_total_fluc_heat_map_main_appendix as total_appendix


APPENDIX_CBAR_LABEL = r"$\sigma_i$ (m$\cdot$s$^{-1}$)"


def main() -> None:
    grid_xlsx = fluc_plot.resolve_grid_xlsx()
    if not grid_xlsx.exists():
        raise FileNotFoundError(
            f"Missing annular-GP fluctuation workbook: {grid_xlsx}"
        )

    out_dir = fluc_plot.resolve_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    total_appendix.PLOT_VMIN = float(fluc_plot.PLOT_VMIN)
    total_appendix.PLOT_VMAX = float(fluc_plot.PLOT_VMAX)
    total_appendix.CBAR_TICK_STEP = float(fluc_plot.CBAR_TICK_STEP)
    total_appendix.build_alpha_cmap = fluc_plot.build_alpha_cmap
    total_appendix.base_plot.CBAR_LABEL = APPENDIX_CBAR_LABEL
    total_appendix.base_plot.PLOT_VMIN = float(fluc_plot.PLOT_VMIN)
    total_appendix.base_plot.PLOT_VMAX = float(fluc_plot.PLOT_VMAX)
    total_appendix.base_plot.CBAR_TICK_STEP = float(fluc_plot.CBAR_TICK_STEP)
    total_appendix.base_plot.build_alpha_cmap = fluc_plot.build_alpha_cmap

    for sheet_name in fluc_plot.SHEETS:
        x, y, sigma_fluc = total_appendix.base_plot.load_gp_mean_sheet(
            grid_xlsx,
            f"{sheet_name}_annular_gp_fluc",
        )
        x_dense, y_dense = fluc_plot.base_plot.build_display_grid(x, y)
        sigma_dense = fluc_plot.base_plot.interpolate_to_display_grid(
            x=x,
            y=y,
            w=sigma_fluc,
            x_grid=x_dense,
            y_grid=y_dense,
        )
        out_png = (
            out_dir
            / (
                f"{sheet_name}"
                "_single_annular_gp_fluc_heatmap_main_appendix.png"
            )
        )
        total_appendix.plot_continuous_heatmap_with_annuli(
            x=x_dense[0, :],
            y=y_dense[:, 0],
            w_field=sigma_dense,
            outpath=out_png,
            sheet_name=sheet_name,
        )

    print(f"Saved figures to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
