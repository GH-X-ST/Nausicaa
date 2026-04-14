from __future__ import annotations

from pathlib import Path

from four_fan_gp_heat_map import (
    build_continuous_grid,
    interpolate_to_continuous_grid,
    load_gp_mean_sheet,
    plot_continuous_heatmap,
)


SHEETS = ["z020", "z035", "z050", "z075", "z110", "z160", "z220"]
GP_GRID_XLSX = Path(
    "B_results/Four_Fan_Annular_GP/four_annular_gp_grid_predictions.xlsx"
)
OUT_DIR = Path("A_figures/Four_Fan_Annular_GP")


def main() -> None:
    if not GP_GRID_XLSX.exists():
        raise FileNotFoundError(
            f"Missing annular-GP grid workbook: {GP_GRID_XLSX}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sheet_name in SHEETS:
        x, y, w_mean = load_gp_mean_sheet(
            GP_GRID_XLSX,
            f"{sheet_name}_annular_gp_mean",
        )
        x_grid, y_grid = build_continuous_grid(x, y)
        w_dense = interpolate_to_continuous_grid(x, y, w_mean, x_grid, y_grid)
        out_png = OUT_DIR / f"{sheet_name}_four_annular_gp_heatmap.png"
        plot_continuous_heatmap(
            x=x_grid[0, :],
            y=y_grid[:, 0],
            w=w_dense,
            outpath=out_png,
        )

    print(f"Saved figures to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
