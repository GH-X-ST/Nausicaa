from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) 3D Plot Configuration and Data Sources
# 2) Model Loading and 3D Field Evaluation
# 3) 3D Figure Export Entry Point
# =============================================================================

# =============================================================================
# 1) 3D Plot Configuration and Data Sources
# =============================================================================


SHEETS = ["z020", "z035", "z050", "z075", "z110", "z160", "z220"]
GRID_PRED_XLSX = Path(
    "B_results/Single_Fan_Annular_GP/single_annular_gp_grid_predictions.xlsx"
)
OUT_DIR = Path("A_figures/Single_Fan_Annular_GP")
OUT_3D_NAME = "single_annular_gp_3d.png"

# =============================================================================
# 2) Model Loading and 3D Field Evaluation
# =============================================================================


def get_base_3d_module():
    """
    Lazy import of the shared GP-3D helpers.

    The annular-GP 3D wrapper only needs the shared plotting utilities, and
    this keeps the entry point focused on producing one standalone 3D figure.
    """
    import single_fan_gp_3D as base_3d

    return base_3d


def interpolate_stack_in_z_for_plot(
    z_axis: np.ndarray,
    w_stack: np.ndarray,
    z_query: np.ndarray,
) -> np.ndarray:
    """
    Fast, conservative plot-only interpolation in z.

    Annular-GP grid sheets already contain dense x-y fields, so a vectorized
    PCHIP interpolation in z is sufficient for visualization and avoids the
    very expensive per-node 1D GP refits used by the generic GP 3D script.
    Outside the measured z range, hold the nearest measured plane instead of
    extrapolating the final field stack, which can create non-physical
    high-altitude concentration artifacts.
    """
    z_axis = np.asarray(z_axis, dtype=float).reshape(-1)
    z_query = np.asarray(z_query, dtype=float).reshape(-1)
    if w_stack.shape[0] != z_axis.size:
        raise ValueError(
            f"w_stack first axis ({w_stack.shape[0]}) must equal len(z_axis) ({z_axis.size})."
        )

    z_min = float(np.min(z_axis))
    z_max = float(np.max(z_axis))
    z_eval = np.clip(z_query, z_min, z_max)

    interp = PchipInterpolator(z_axis, w_stack, axis=0, extrapolate=False)
    out = np.asarray(interp(z_eval), dtype=float)

    low_mask = z_query < z_min
    if np.any(low_mask):
        out[low_mask, :, :] = w_stack[0]

    high_mask = z_query > z_max
    if np.any(high_mask):
        out[high_mask, :, :] = w_stack[-1]

    return np.maximum(out, 0.0)


def load_annular_gp_stack(
    xlsx_path: Path,
    sheet_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base_3d = get_base_3d_module()
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

# =============================================================================
# 3) 3D Figure Export Entry Point
# =============================================================================


def main() -> None:
    base_3d = get_base_3d_module()
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
    w_stack_extrap = interpolate_stack_in_z_for_plot(
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
