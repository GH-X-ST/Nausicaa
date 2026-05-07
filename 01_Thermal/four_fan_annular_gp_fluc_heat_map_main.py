from __future__ import annotations

import four_fan_annular_gp_fluc_heat_map as fluc_plot
import four_fan_annular_gp_total_fluc_heat_map_main as total_main


def main() -> None:
    grid_xlsx = fluc_plot.resolve_grid_xlsx()
    if not grid_xlsx.exists():
        raise FileNotFoundError(
            f"Missing annular-GP fluctuation workbook: {grid_xlsx}"
        )

    out_dir = fluc_plot.resolve_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    total_main.CBAR_LABEL = fluc_plot.CBAR_LABEL
    total_main.PLOT_VMIN = float(fluc_plot.PLOT_VMIN)
    total_main.PLOT_VMAX = float(fluc_plot.PLOT_VMAX)
    total_main.CBAR_TICK_STEP = float(fluc_plot.CBAR_TICK_STEP)
    total_main.build_alpha_cmap = fluc_plot.build_alpha_cmap
    total_main.base_plot.CBAR_LABEL = fluc_plot.CBAR_LABEL
    total_main.base_plot.PLOT_VMIN = float(fluc_plot.PLOT_VMIN)
    total_main.base_plot.PLOT_VMAX = float(fluc_plot.PLOT_VMAX)
    total_main.base_plot.CBAR_TICK_STEP = float(fluc_plot.CBAR_TICK_STEP)
    total_main.base_plot.build_alpha_cmap = fluc_plot.build_alpha_cmap

    for sheet_name in fluc_plot.SHEETS:
        x, y, sigma_fluc = total_main.base_plot.load_gp_mean_sheet(
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
            / f"{sheet_name}_four_annular_gp_fluc_heatmap_main.png"
        )
        total_main.plot_continuous_heatmap_with_annuli(
            x=x_dense[0, :],
            y=y_dense[:, 0],
            w=sigma_dense,
            outpath=out_png,
            sheet_name=sheet_name,
        )

    print(f"Saved figures to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
