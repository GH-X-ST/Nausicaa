from __future__ import annotations

from pathlib import Path

import single_fan_annular_gp_heat_map_main_appendix as base_plot
import single_fan_annular_gaussian_bemt_heat_map_main as model_plot


OUT_DIR = Path("A_figures/Single_Fan_Annular_Gaussian_BEMT")


def main() -> None:
    params_df, param_cols = model_plot.load_bemt_params(
        model_plot.PARAMS_XLSX,
        model_plot.PARAMS_SHEET,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for sheet_name in model_plot.SHEETS:
        x, y, _ = model_plot.read_slice_from_sheet(
            model_plot.XLSX_PATH,
            sheet_name,
        )
        params = model_plot.params_for_height(
            params_df,
            param_cols,
            sheet_name,
        )

        x_grid, y_grid = model_plot.build_continuous_grid(x, y)
        w_model = model_plot.evaluate_model(
            x_grid,
            y_grid,
            params,
            fan_center_xy=model_plot.FAN_CENTER_XY,
        )

        out_png = (
            OUT_DIR
            / (
                f"{sheet_name}_"
                "single_annular_gaussian_bemt_heatmap_main_appendix.png"
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
