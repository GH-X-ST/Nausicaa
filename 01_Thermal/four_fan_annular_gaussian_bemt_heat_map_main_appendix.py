from __future__ import annotations

from pathlib import Path

import numpy as np

import four_fan_annular_gp_heat_map_main_appendix as base_plot
import four_fan_annular_gaussian_bemt_heat_map_main as model_plot


OUT_DIR = Path("A_figures/Four_Fan_Annular_Gaussian_BEMT")


def main() -> None:
    params_df = model_plot.load_bemt_params(
        model_plot.PARAMS_XLSX,
        model_plot.PARAMS_SHEET,
    )
    fan_ids = model_plot.discover_fan_ids(params_df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if (
        len(fan_ids) > 0
        and len(fan_ids) != len(model_plot.FOUR_FAN_CENTERS_XY)
    ):
        raise ValueError(
            "Parameter table fan count does not match four-fan geometry."
        )

    for sheet_name in model_plot.SHEETS:
        x, y, _ = model_plot.read_slice_from_sheet(
            model_plot.XLSX_PATH,
            sheet_name,
        )
        params_list, orders_list = model_plot.params_for_height(
            params_df,
            sheet_name,
            fan_ids=fan_ids,
        )

        x_grid, y_grid = model_plot.build_continuous_grid(x, y)
        w_model = np.zeros((x_grid.shape[0], x_grid.shape[1]), dtype=float)
        for fan_idx, (fx, fy) in enumerate(model_plot.FOUR_FAN_CENTERS_XY):
            r = np.sqrt((x_grid - fx) ** 2 + (y_grid - fy) ** 2)
            theta = np.arctan2(y_grid - fy, x_grid - fx)
            w_model += model_plot.azimuthal_ring_model(
                r=r,
                theta=theta,
                params=params_list[fan_idx],
                harmonic_orders=orders_list[fan_idx],
            )

        out_png = (
            OUT_DIR
            / (
                f"{sheet_name}_"
                "four_annular_gaussian_bemt_heatmap_main_appendix.png"
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
