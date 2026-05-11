"""
Compute error metrics for the single-fan non-axisymmetric annular Gaussian model.

This script evaluates the BEMT model reconstructed from
single_fan_annular_gaussian_bemt_fit.py output and reports:
1) Per-height accumulated SAE across raw grid samples
2) Total accumulated SAE across all heights
3) Per-height weighted RMSE using sigma-weighted raw samples
4) Total weighted RMSE across all heights
"""


from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import single_fan_gp as sigma_gp

from single_fan_annuli_cut import (
    assign_sigma_points_linear_nearest,
    parse_ts_xy_points_and_sigmas,
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

XLSX_PATH = "S01.xlsx"
SHEETS = ["z020", "z035", "z050", "z075", "z110", "z160", "z220"]

FIT_XLSX_PATH = Path("B_results/single_annular_bemt_params_pchip.xlsx")
FIT_SHEET_NAME = "single_bemt_az_pchip"

OUT_XLSX_PATH = Path("B_results/single_annular_bemt_analysis.xlsx")
OUT_SHEET_NAME = "single_annular_bemt_analysis"

FAN_CENTER_XY = (4.2, 2.4)
SIGMA_FALLBACK = 0.2
SIGMA_MIN = 1e-3

SHEET_HEIGHT_DIVISOR = 100.0


# =============================================================================
# 2) Metric Loading and Diagnostics
# =============================================================================

REQUIRED_BASE_COLUMNS = ("z_m", "w0", "r_ring", "delta_ring", "a0")


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


def discover_param_columns(df: pd.DataFrame) -> Tuple[List[str], List[int]]:
    """
    Discover parameter columns and harmonic orders from fit table.

    Returns:
        param_cols: [w0, r_ring, delta_ring, a0, a1, b1, ...]
        harmonic_orders: [1, 2, ...]
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


def extract_clean_fit_table(df: pd.DataFrame, param_cols: Sequence[str]) -> pd.DataFrame:
    """
    Validate and clean fit table for interpolation.
    """
    cols = ["z_m"] + list(param_cols)
    data = df[cols].copy()
    for col in cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna()

    if data.empty:
        raise ValueError("No valid fit rows found after cleaning.")

    data = data.sort_values("z_m").drop_duplicates(subset=["z_m"], keep="first")

    z_vals = data["z_m"].to_numpy(dtype=float)
    if np.any(np.diff(z_vals) <= 0.0):
        raise ValueError("Fit z_m must be strictly increasing.")

    delta_vals = data["delta_ring"].to_numpy(dtype=float)
    if np.any(delta_vals <= 0.0):
        raise ValueError("delta_ring must be positive.")

    return data


def params_at_z(
    fit_df: pd.DataFrame,
    param_cols: Sequence[str],
    z_m: float,
) -> Dict[str, float]:
    """
    Interpolate fit parameters at one target height z.
    """
    z_grid = fit_df["z_m"].to_numpy(dtype=float)

    z_min = float(np.min(z_grid))
    z_max = float(np.max(z_grid))
    if z_m < z_min - 1e-12 or z_m > z_max + 1e-12:
        raise ValueError(
            f"Requested z={z_m:.3f} m outside fit range [{z_min:.3f}, {z_max:.3f}] m."
        )

    params: Dict[str, float] = {}
    for col in param_cols:
        params[col] = float(np.interp(z_m, z_grid, fit_df[col].to_numpy(dtype=float)))
    return params


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


def compute_height_metrics(
    r_pts: np.ndarray,
    theta_pts: np.ndarray,
    w_obs: np.ndarray,
    alpha: np.ndarray,
    sigma: np.ndarray,
    params: Dict[str, float],
    harmonic_orders: Sequence[int],
) -> Tuple[float, float, float, float, int]:
    """
    Return:
        sae_k, weighted_sse_k, weight_sum_k, wrmse_k, n_samples
    """
    sigma_safe = np.maximum(sigma.astype(float), float(SIGMA_MIN))
    w_pred = azimuthal_ring_model(
        r=r_pts,
        theta=theta_pts,
        params=params,
        harmonic_orders=harmonic_orders,
    )
    err = w_pred - w_obs

    sae_k = float(np.sum(np.abs(err)))

    weights = (alpha / sigma_safe) ** 2
    weighted_sse_k = float(np.sum(weights * err**2))
    weight_sum_k = float(np.sum(weights))
    if weight_sum_k <= 0.0:
        raise ValueError("Non-positive total weight encountered in WRMSE calculation.")
    wrmse_k = float(np.sqrt(weighted_sse_k / weight_sum_k))

    return sae_k, weighted_sse_k, weight_sum_k, wrmse_k, int(r_pts.size)


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
    fit_raw = load_fit_table(FIT_XLSX_PATH, FIT_SHEET_NAME)
    param_cols, harmonic_orders = discover_param_columns(fit_raw)
    fit_df = extract_clean_fit_table(fit_raw, param_cols)

    rows: List[Dict[str, float]] = []
    total_sae = 0.0
    total_weighted_sse = 0.0
    total_weight_sum = 0.0

    for sheet in SHEETS:
        z_m = parse_sheet_height_m(sheet)
        params = params_at_z(fit_df, param_cols, z_m)

        x_centers, y_centers, W = read_slice_from_sheet(XLSX_PATH, sheet)
        x_grid, y_grid = np.meshgrid(x_centers, y_centers)

        xc, yc = FAN_CENTER_XY
        x_pts = x_grid.ravel()
        y_pts = y_grid.ravel()
        dx = x_pts - xc
        dy = y_pts - yc
        r_pts = np.sqrt(dx**2 + dy**2)
        theta_pts = np.arctan2(dy, dx)
        w_obs = W.ravel()

        mask = (
            np.isfinite(x_pts)
            & np.isfinite(y_pts)
            & np.isfinite(r_pts)
            & np.isfinite(theta_pts)
            & np.isfinite(w_obs)
        )
        x_pts = x_pts[mask]
        y_pts = y_pts[mask]
        r_pts = r_pts[mask]
        theta_pts = theta_pts[mask]
        w_obs = w_obs[mask]
        if r_pts.size == 0:
            raise ValueError(f"No valid raw samples found in sheet '{sheet}'.")

        ts_sheet = f"{sheet}_TS"
        sigma_pts = sigma_gp.evaluate_sigma_points_annular_pchip_z(
            xlsx_path=XLSX_PATH,
            sheet_names=tuple(SHEETS),
            fan_center_xy=FAN_CENTER_XY,
            x_pts=x_pts,
            y_pts=y_pts,
            z_pts=np.full_like(x_pts, parse_sheet_height_m(sheet), dtype=float),
            sigma_fallback=SIGMA_FALLBACK,
            sigma_min=SIGMA_MIN,
        )

        alpha_pts = np.ones_like(r_pts, dtype=float)

        sae_k, weighted_sse_k, weight_sum_k, wrmse_k, n_samples = compute_height_metrics(
            r_pts=r_pts,
            theta_pts=theta_pts,
            w_obs=w_obs,
            alpha=alpha_pts,
            sigma=sigma_pts,
            params=params,
            harmonic_orders=harmonic_orders,
        )

        total_sae += sae_k
        total_weighted_sse += weighted_sse_k
        total_weight_sum += weight_sum_k

        row: Dict[str, float] = {
            "sheet": sheet,
            "z_m": z_m,
            "n_harmonics": int(len(harmonic_orders)),
            "n_samples": n_samples,
            "accumulate_SAE_mps": sae_k,
            "weighted_RMSE_mps": wrmse_k,
            "weighted_SSE_term": weighted_sse_k,
            "weight_sum_term": weight_sum_k,
        }
        for col in param_cols:
            row[col] = params[col]
        rows.append(row)

    if total_weight_sum <= 0.0:
        raise ValueError("Total WRMSE denominator is non-positive.")

    total_wrmse = float(np.sqrt(total_weighted_sse / total_weight_sum))
    total_row: Dict[str, float] = {
        "sheet": "TOTAL",
        "z_m": np.nan,
        "n_harmonics": int(len(harmonic_orders)),
        "n_samples": int(sum(int(r["n_samples"]) for r in rows)),
        "accumulate_SAE_mps": float(total_sae),
        "weighted_RMSE_mps": total_wrmse,
        "weighted_SSE_term": float(total_weighted_sse),
        "weight_sum_term": float(total_weight_sum),
    }
    for col in param_cols:
        total_row[col] = np.nan
    rows.append(total_row)

    df_out = pd.DataFrame(rows)
    write_results(df_out, OUT_XLSX_PATH, OUT_SHEET_NAME)

    print(f"Saved analysis metrics to: {OUT_XLSX_PATH.resolve()}")
    print(f"Sheet: {OUT_SHEET_NAME}")
    print(df_out.to_string(index=False))


if __name__ == "__main__":
    main()
