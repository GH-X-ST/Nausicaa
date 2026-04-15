from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

import single_fan_annular_gp_heat_map_main_appendix as base_plot


SHEETS = ["z020", "z035", "z050", "z075", "z110", "z160", "z220"]
GP_GRID_XLSX = Path(
    "B_results/Single_Fan_Annular_GP/single_annular_gp_grid_predictions.xlsx"
)
OUT_DIR = Path("A_figures/Single_Fan_Annular_GP")
PLOT_VMIN = 0.0
PLOT_VMAX = 0.4


def build_alpha_cmap():
    """Return an opaque summer colormap for std maps."""
    return plt.get_cmap("summer")


def main() -> None:
    if not GP_GRID_XLSX.exists():
        raise FileNotFoundError(
            f"Missing annular-GP grid workbook: {GP_GRID_XLSX}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_plot.CBAR_LABEL = r"$\sigma_{\mathrm{res}}$ (m$\cdot$s$^{-1}$)"
    base_plot.PLOT_VMIN = float(PLOT_VMIN)
    base_plot.PLOT_VMAX = float(PLOT_VMAX)
    base_plot.build_alpha_cmap = build_alpha_cmap

    for sheet_name in SHEETS:
        x, y, w_std = base_plot.load_gp_mean_sheet(
            GP_GRID_XLSX,
            f"{sheet_name}_annular_gp_std",
        )
        out_png = (
            OUT_DIR
            / f"{sheet_name}_single_annular_gp_std_heatmap_main_appendix.png"
        )
        base_plot.plot_continuous_heatmap(
            x=x,
            y=y,
            w_field=w_std,
            outpath=out_png,
        )

    print(f"Saved figures to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
