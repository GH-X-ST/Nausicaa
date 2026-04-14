from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

import four_fan_gp_3D as base_3d


SHEETS = ["z020", "z035", "z050", "z075", "z110", "z160", "z220"]
GRID_PRED_XLSX = Path(
    "B_results/Four_Fan_Annular_GP/four_annular_gp_grid_predictions.xlsx"
)
OUT_DIR = Path("A_figures/Four_Fan_Annular_GP")
OUT_3D_NAME = "four_annular_gp_3d.png"


def load_annular_gp_stack(
    xlsx_path: Path,
    sheet_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_ref = None
    y_ref = None
    z_list = []
    w_layers = []

    for sheet_name in sheet_names:
        x, y, w = base_3d.load_gp_mean_sheet(
            xlsx_path,
            f"{sheet_name}_annular_gp_mean",
        )
        if x_ref is None:
            x_ref = x
            y_ref = y
        else:
            if not np.allclose(x, x_ref, atol=1e-9, rtol=0.0):
                raise ValueError(
                    (
                        "Inconsistent x-grid found in sheet "
                        f"'{sheet_name}_annular_gp_mean'."
                    )
                )
            if not np.allclose(y, y_ref, atol=1e-9, rtol=0.0):
                raise ValueError(
                    (
                        "Inconsistent y-grid found in sheet "
                        f"'{sheet_name}_annular_gp_mean'."
                    )
                )

        z_list.append(base_3d.parse_sheet_height_m(sheet_name))
        w_layers.append(w)

    z_axis = np.asarray(z_list, dtype=float)
    order = np.argsort(z_axis)
    z_axis = z_axis[order]
    w_stack = np.stack([w_layers[idx] for idx in order], axis=0)
    return x_ref, y_ref, z_axis, w_stack


def main() -> None:
    if not base_3d.MAKE_PLOTS:
        print("MAKE_PLOTS is False; nothing to do.")
        return
    if not GRID_PRED_XLSX.exists():
        raise FileNotFoundError(
            f"Missing annular-GP grid workbook: {GRID_PRED_XLSX}"
        )

    x_axis, y_axis, z_axis, w_stack = load_annular_gp_stack(
        xlsx_path=GRID_PRED_XLSX,
        sheet_names=SHEETS,
    )
    x_grid, y_grid, z_grid, dx, dy, dz = base_3d.make_3d_grid()
    z_query = np.linspace(
        base_3d.REL_Z_EXTRAP_MIN,
        base_3d.REL_Z_EXTRAP_MAX,
        z_grid.shape[2],
        dtype=float,
    )
    w_stack_extrap = base_3d.extrapolate_stack_in_z_with_gp(
        z_axis=z_axis,
        w_stack=w_stack,
        z_query=z_query,
    )
    z_grid_rel = z_grid - base_3d.FAN_VERTICAL_OFFSET_M
    w_grid = base_3d.evaluate_field(
        x_grid=x_grid,
        y_grid=y_grid,
        z_grid=z_grid_rel,
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=z_query,
        w_stack=w_stack_extrap,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_3d = OUT_DIR / OUT_3D_NAME
    base_3d.plot_isosurfaces(
        w_grid=w_grid,
        dx=dx,
        dy=dy,
        dz=dz,
        cmap_alpha=base_3d.build_alpha_cmap(),
        output_path=out_3d,
    )
    print(f"Saved 3D annular-GP plot to: {out_3d.resolve()}")


if __name__ == "__main__":
    main()
