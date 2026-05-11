from __future__ import annotations

from pathlib import Path

from four_fan_gp_heat_map_main import (
    load_gp_mean_sheet,
    plot_continuous_heatmap,
)


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


SHEETS = ["z020", "z035", "z050", "z075", "z110", "z160", "z220"]
GP_GRID_XLSX = Path(
    "B_results/Four_Fan_Annular_GP/four_annular_gp_grid_predictions.xlsx"
)
OUT_DIR = Path("A_figures/Four_Fan_Annular_GP")

# =============================================================================
# 2) Figure Export Entry Point
# =============================================================================
# Entry points write deterministic artifacts so regenerated figures and tables can be compared by path and sheet name.


# Main execution keeps data loading, evaluation, and export order deterministic.
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
        out_png = OUT_DIR / f"{sheet_name}_four_annular_gp_heatmap_main.png"
        plot_continuous_heatmap(
            x=x,
            y=y,
            w=w_mean,
            outpath=out_png,
        )

    print(f"Saved figures to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
