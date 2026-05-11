"""
Compute error metrics for the four-fan non-axisymmetric annular Gaussian model.

Metrics reported:
1) Per-height accumulated SAE across raw grid samples
2) Total accumulated SAE across all heights
3) Per-height weighted RMSE using sigma-weighted raw samples
4) Total weighted RMSE across all heights
"""


from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import re

import numpy as np
import pandas as pd
import four_fan_gp as sigma_gp

from four_fan_annuli_cut import (
    assign_sigma_samples_with_overlap,
    compute_nearest_fan_distances,
    read_slice_from_sheet,
)


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Metric Configuration and Data Sources
# 2) Metric Loading and Diagnostics
# 3) Analysis Export Entry Point
# =============================================================================

# =============================================================================
# 1) Metric Configuration and Data Sources
# =============================================================================

XLSX_PATH = "S02.xlsx"
SHEETS = ["z020", "z035", "z050", "z075", "z110", "z160", "z220"]

FIT_XLSX_PATH = Path("B_results/four_annular_bemt_params_pchip.xlsx")
FIT_SHEET_NAME = "four_bemt_az_pchip"

OUT_XLSX_PATH = Path("B_results/four_annular_bemt_analysis.xlsx")
OUT_SHEET_NAME = "four_annular_bemt_analysis"

FOUR_FAN_CENTERS_XY = (
    (3.0, 3.6),
    (5.4, 3.6),
    (3.0, 1.2),
    (5.4, 1.2),
)
SIGMA_FALLBACK = 0.2
SIGMA_MIN = 1e-3
OVERLAP_RATIO_THRESHOLD = 1.25
OVERLAP_WEIGHT_POWER = 2.0
OVERLAP_SIGMA_BOOST = 1.12

SHEET_HEIGHT_DIVISOR = 100.0
REQUIRED_BASE_COLUMNS = ("z_m", "w0", "r_ring", "delta_ring", "a0")
FAN_COL_PATTERN = re.compile(r"^a0_(F\d{2})$")


# =============================================================================
# 2) Metric Loading and Diagnostics
# =============================================================================

def parse_sheet_height_m(sheet_name: str) -> float:
    """
    Parse heights from names like z020, z110, z220.
    """
    if not sheet_name.startswith("z"):
        raise ValueError(f"Invalid sheet name (expected 'z###'): {sheet_name}")
    suffix = sheet_name[1:]
    if not suffix.isdigit():
        raise ValueError(f"Invalid height code in sheet name: {sheet_name}")
    return int(suffix) / SHEET_HEIGHT_DIVISOR


def load_fit_table(xlsx_path: Path, sheet_name: str) -> pd.DataFrame:
    """
    Load interpolated BEMT-parameter table.
    """
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Missing fit file: {xlsx_path}")

    xls = pd.ExcelFile(xlsx_path)
    sheet_to_use = sheet_name if sheet_name in xls.sheet_names else xls.sheet_names[0]
    return pd.read_excel(xlsx_path, sheet_name=sheet_to_use)


def discover_fan_ids(df: pd.DataFrame) -> Tuple[str, ...]:
    """
    Discover fan IDs from columns like a0_F01.
    """
    fan_ids = []
    for col in df.columns:
        match = FAN_COL_PATTERN.match(str(col))
        if match is not None:
            fan_ids.append(match.group(1))

    fan_ids = sorted(set(fan_ids))
    valid = []
    for fan_id in fan_ids:
        required = (
            f"w0_{fan_id}",
            f"r_ring_{fan_id}",
            f"delta_ring_{fan_id}",
            f"a0_{fan_id}",
        )
        if all(col in df.columns for col in required):
            valid.append(fan_id)
    return tuple(valid)


def discover_param_columns(df: pd.DataFrame) -> Tuple[List[str], List[int]]:
    """
    Discover parameter columns and harmonic orders from shared fit table.
    """
    missing = [col for col in REQUIRED_BASE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in fit table: {missing}")

    a_orders = []
    b_orders = []
    for col in df.columns:
        if col.startswith("a") and col[1:].isdigit():
            order = int(col[1:])
            if order >= 1:
                a_orders.append(order)
        if col.startswith("b") and col[1:].isdigit():
            order = int(col[1:])
            if order >= 1:
                b_orders.append(order)

    harmonic_orders = sorted(set(a_orders).intersection(set(b_orders)))
    param_cols = ["w0", "r_ring", "delta_ring", "a0"]
    for order in harmonic_orders:
        param_cols.append(f"a{order}")
        param_cols.append(f"b{order}")
    return param_cols, harmonic_orders


def discover_param_columns_for_fan(df: pd.DataFrame, fan_id: str) -> Tuple[List[str], List[int]]:
    """
    Discover per-fan parameter columns and harmonic orders.
    """
    base = [f"w0_{fan_id}", f"r_ring_{fan_id}", f"delta_ring_{fan_id}", f"a0_{fan_id}"]
    missing = [col for col in base if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {fan_id}: {missing}")

    a_orders = []
    b_orders = []
    suffix = f"_{fan_id}"
    for col in df.columns:
        if col.startswith("a") and col.endswith(suffix):
            core = col[1 : -len(suffix)]
            if core.isdigit() and int(core) >= 1:
                a_orders.append(int(core))
        if col.startswith("b") and col.endswith(suffix):
            core = col[1 : -len(suffix)]
            if core.isdigit() and int(core) >= 1:
                b_orders.append(int(core))

    harmonic_orders = sorted(set(a_orders).intersection(set(b_orders)))
    cols = list(base)
    for n_idx in harmonic_orders:
        cols.append(f"a{n_idx}_{fan_id}")
        cols.append(f"b{n_idx}_{fan_id}")
    return cols, harmonic_orders


def interpolate_row_at_z(df: pd.DataFrame, z_m: float, cols: Sequence[str]) -> Dict[str, float]:
    """
    Interpolate selected columns to one height.
    """
    z_grid = pd.to_numeric(df["z_m"], errors="coerce").to_numpy(dtype=float)
    order = np.argsort(z_grid)
    z_grid = z_grid[order]

    out = {}
    for col in cols:
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)[order]
        out[col] = float(np.interp(z_m, z_grid, vals))
    return out


def params_by_fan_at_z(
    fit_df: pd.DataFrame,
    fan_ids: Tuple[str, ...],
    z_m: float,
) -> Tuple[List[Dict[str, float]], List[List[int]]]:
    """
    Interpolate per-fan params at one z.
    """
    if len(fan_ids) == 0:
        param_cols, harmonic_orders = discover_param_columns(fit_df)
        shared = interpolate_row_at_z(fit_df, z_m=z_m, cols=param_cols)
        params_list = []
        orders_list = []
        for _ in FOUR_FAN_CENTERS_XY:
            params_list.append(shared.copy())
            orders_list.append(list(harmonic_orders))
        return params_list, orders_list

    params_list = []
    orders_list = []
    for fan_id in fan_ids:
        cols_fan, harmonic_orders = discover_param_columns_for_fan(fit_df, fan_id)
        raw_vals = interpolate_row_at_z(fit_df, z_m=z_m, cols=cols_fan)
        remapped = {
            "w0": raw_vals[f"w0_{fan_id}"],
            "r_ring": raw_vals[f"r_ring_{fan_id}"],
            "delta_ring": raw_vals[f"delta_ring_{fan_id}"],
            "a0": raw_vals[f"a0_{fan_id}"],
        }
        for n_idx in harmonic_orders:
            remapped[f"a{n_idx}"] = raw_vals[f"a{n_idx}_{fan_id}"]
            remapped[f"b{n_idx}"] = raw_vals[f"b{n_idx}_{fan_id}"]
        params_list.append(remapped)
        orders_list.append(list(harmonic_orders))
    return params_list, orders_list


def azimuthal_ring_model(
    r: np.ndarray,
    theta: np.ndarray,
    params: Dict[str, float],
    harmonic_orders: Sequence[int],
) -> np.ndarray:
    """
    Evaluate w_model(x, y) = w0 + g(r) * A(theta).
    """
    delta_ring = max(float(params["delta_ring"]), 1e-12)
    g_r = np.exp(-((r - float(params["r_ring"])) / delta_ring) ** 2)

    amp = np.full_like(theta, float(params["a0"]), dtype=float)
    for order in harmonic_orders:
        amp += float(params[f"a{order}"]) * np.cos(order * theta)
        amp += float(params[f"b{order}"]) * np.sin(order * theta)

    return float(params["w0"]) + g_r * amp


def write_results(df: pd.DataFrame, out_xlsx: Path, sheet_name: str) -> None:
    """
    Write metrics table to an Excel sheet (replace sheet if it exists).
    """
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    if out_xlsx.exists():
        with pd.ExcelWriter(
            out_xlsx,
            engine="openpyxl",
            mode="a",
            if_sheet_exists="replace",
        ) as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)
    else:
        with pd.ExcelWriter(out_xlsx, engine="openpyxl", mode="w") as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)


# =============================================================================
# 3) Analysis Export Entry Point
# =============================================================================

def main() -> None:
    fit_df = load_fit_table(FIT_XLSX_PATH, FIT_SHEET_NAME)
    fan_ids = discover_fan_ids(fit_df)
    if len(fan_ids) > 0 and len(fan_ids) != len(FOUR_FAN_CENTERS_XY):
        raise ValueError(
            f"Per-fan fit table has {len(fan_ids)} fans but expected {len(FOUR_FAN_CENTERS_XY)}."
        )

    rows: List[Dict[str, float]] = []
    total_sae = 0.0
    total_weighted_sse = 0.0
    total_weight_sum = 0.0

    fan_xy = np.asarray(FOUR_FAN_CENTERS_XY, dtype=float)
    for sheet in SHEETS:
        z_m = parse_sheet_height_m(sheet)
        params_fan, harmonic_orders_fan = params_by_fan_at_z(fit_df, fan_ids=fan_ids, z_m=z_m)

        x_centers, y_centers, W = read_slice_from_sheet(XLSX_PATH, sheet)
        x_grid, y_grid = np.meshgrid(x_centers, y_centers)
        x_pts = x_grid.ravel()
        y_pts = y_grid.ravel()
        w_obs = W.ravel()
        valid = np.isfinite(x_pts) & np.isfinite(y_pts) & np.isfinite(w_obs)
        x_pts = x_pts[valid]
        y_pts = y_pts[valid]
        w_obs = w_obs[valid]
        if x_pts.size == 0:
            raise ValueError(f"No valid raw samples found in sheet '{sheet}'.")

        nearest_idx, _nearest_r, _second_idx, _second_r = compute_nearest_fan_distances(
            x_pts=x_pts,
            y_pts=y_pts,
            fan_centers_xy=FOUR_FAN_CENTERS_XY,
        )
        dx_all = x_pts[:, None] - fan_xy[None, :, 0]
        dy_all = y_pts[:, None] - fan_xy[None, :, 1]
        r_all = np.sqrt(dx_all**2 + dy_all**2)
        theta_all = np.arctan2(dy_all, dx_all)

        sigma_pts = sigma_gp.evaluate_sigma_points_annular_blend_pchip_z(
            xlsx_path=XLSX_PATH,
            sheet_names=tuple(SHEETS),
            fan_centers_xy=FOUR_FAN_CENTERS_XY,
            x_pts=x_pts,
            y_pts=y_pts,
            z_pts=np.full_like(x_pts, parse_sheet_height_m(sheet), dtype=float),
            sigma_fallback=SIGMA_FALLBACK,
            sigma_min=SIGMA_MIN,
        )
        sigma_safe = np.maximum(sigma_pts.astype(float), float(SIGMA_MIN))

        w_pred = np.zeros_like(w_obs, dtype=float)
        for fan_idx in range(len(FOUR_FAN_CENTERS_XY)):
            w_pred += azimuthal_ring_model(
                r=r_all[:, fan_idx],
                theta=theta_all[:, fan_idx],
                params=params_fan[fan_idx],
                harmonic_orders=harmonic_orders_fan[fan_idx],
            )

        err = w_pred - w_obs
        sae_k = float(np.sum(np.abs(err)))
        weights = 1.0 / (sigma_safe**2)
        weighted_sse_k = float(np.sum(weights * err**2))
        weight_sum_k = float(np.sum(weights))
        wrmse_k = float(np.sqrt(weighted_sse_k / weight_sum_k))

        total_sae += sae_k
        total_weighted_sse += weighted_sse_k
        total_weight_sum += weight_sum_k

        mean_w0 = float(np.mean([p["w0"] for p in params_fan]))
        mean_r = float(np.mean([p["r_ring"] for p in params_fan]))
        mean_d = float(np.mean([p["delta_ring"] for p in params_fan]))
        mean_a0 = float(np.mean([p["a0"] for p in params_fan]))
        max_harmonic = int(max((len(v) for v in harmonic_orders_fan), default=0))

        rows.append(
            {
                "sheet": sheet,
                "z_m": z_m,
                "n_harmonics_max": max_harmonic,
                "n_samples": int(w_obs.size),
                "accumulate_SAE_mps": sae_k,
                "weighted_RMSE_mps": wrmse_k,
                "weighted_SSE_term": weighted_sse_k,
                "weight_sum_term": weight_sum_k,
                "w0": mean_w0,
                "r_ring": mean_r,
                "delta_ring": mean_d,
                "a0": mean_a0,
            }
        )

    if total_weight_sum <= 0.0:
        raise ValueError("Total WRMSE denominator is non-positive.")

    total_wrmse = float(np.sqrt(total_weighted_sse / total_weight_sum))
    rows.append(
        {
            "sheet": "TOTAL",
            "z_m": np.nan,
            "n_harmonics_max": np.nan,
            "n_samples": int(sum(int(r["n_samples"]) for r in rows)),
            "accumulate_SAE_mps": float(total_sae),
            "weighted_RMSE_mps": total_wrmse,
            "weighted_SSE_term": float(total_weighted_sse),
            "weight_sum_term": float(total_weight_sum),
            "w0": np.nan,
            "r_ring": np.nan,
            "delta_ring": np.nan,
            "a0": np.nan,
        }
    )

    df_out = pd.DataFrame(rows)
    write_results(df_out, OUT_XLSX_PATH, OUT_SHEET_NAME)

    print(f"Saved analysis metrics to: {OUT_XLSX_PATH.resolve()}")
    print(f"Sheet: {OUT_SHEET_NAME}")
    print(df_out.to_string(index=False))


if __name__ == "__main__":
    main()
