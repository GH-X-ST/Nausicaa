"""
Compute error metrics for the single-fan annular Gaussian model.

Metrics reported:
1) Per-height accumulated SAE across annuli:
       SAE_k = sum_j |w_model(r_{j,k}, z_k) - w_obs_{j,k}|
2) Total accumulated SAE across all heights:
       SAE_total = sum_k SAE_k
3) Per-height weighted RMSE using alpha_j and sigma_j:
       WRMSE_k = sqrt( sum_j ((alpha_{j,k}/sigma_{j,k})^2 * e_{j,k}^2)
                        / sum_j ((alpha_{j,k}/sigma_{j,k})^2) )
4) Total weighted RMSE across all heights:
       WRMSE_total = sqrt( sum_k sum_j ((alpha_{j,k}/sigma_{j,k})^2 * e_{j,k}^2)
                           / sum_k sum_j ((alpha_{j,k}/sigma_{j,k})^2) )

where e_{j,k} = w_model(r_{j,k}, z_k) - w_obs_{j,k}.
"""


from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

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

FIT_XLSX_PATH = Path("B_results/single_annular_var_params_pchip.xlsx")
FIT_SHEET_NAME = "single_annular_var_pchip"

# Output is written to a new/replaced sheet in the fit workbook.
OUT_XLSX_PATH = Path("B_results/single_annular_var_analysis.xlsx")
OUT_SHEET_NAME = "single_annular_var_analysis"

# Keep settings aligned with single_fan_annuli_cut.py.
FAN_CENTER_XY = (4.2, 2.4)
SIGMA_FALLBACK = 0.2
SIGMA_MIN = 1e-3

SHEET_HEIGHT_DIVISOR = 100.0


# =============================================================================
# 2) Metric Loading and Diagnostics
# =============================================================================

REQUIRED_FIT_COLUMNS = ("z_m", "A_ring", "r_ring", "delta_r", "w0")


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


def ring_gaussian(
    r: np.ndarray,
    a_ring: float,
    r_ring: float,
    delta_r: float,
    w0: float,
) -> np.ndarray:
    """
    Evaluate w(r) = w0 + A_ring * exp(-((r - r_ring)/delta_r)^2).
    """
    return w0 + a_ring * np.exp(-((r - r_ring) / delta_r) ** 2)


def load_fit_table(xlsx_path: Path, sheet_name: str) -> pd.DataFrame:
    """
    Load and validate interpolated fit-parameter table.
    """
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Missing fit file: {xlsx_path}")

    xls = pd.ExcelFile(xlsx_path)
    sheet_to_use = sheet_name if sheet_name in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(xlsx_path, sheet_name=sheet_to_use)

    missing = [c for c in REQUIRED_FIT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in fit table: {missing}")

    data = df[list(REQUIRED_FIT_COLUMNS)].copy()
    for col in REQUIRED_FIT_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna()
    if data.empty:
        raise ValueError("No valid fit rows found after cleaning.")

    data = data.sort_values("z_m").drop_duplicates(subset=["z_m"], keep="first")
    z_vals = data["z_m"].to_numpy(dtype=float)
    if np.any(np.diff(z_vals) <= 0.0):
        raise ValueError("Fit z_m must be strictly increasing.")

    return data


def params_at_z(fit_df: pd.DataFrame, z_m: float) -> Dict[str, float]:
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

    return {
        "A_ring": float(np.interp(z_m, z_grid, fit_df["A_ring"].to_numpy(dtype=float))),
        "r_ring": float(np.interp(z_m, z_grid, fit_df["r_ring"].to_numpy(dtype=float))),
        "delta_r": float(np.interp(z_m, z_grid, fit_df["delta_r"].to_numpy(dtype=float))),
        "w0": float(np.interp(z_m, z_grid, fit_df["w0"].to_numpy(dtype=float))),
    }


def compute_height_metrics(
    r_pts: np.ndarray,
    w_obs: np.ndarray,
    alpha: np.ndarray,
    sigma: np.ndarray,
    params: Dict[str, float],
) -> Tuple[float, float, float, float, int]:
    """
    Return:
        sae_k, weighted_sse_k, weight_sum_k, wrmse_k, n_annuli
    """
    sigma_safe = np.maximum(sigma.astype(float), float(SIGMA_MIN))
    w_pred = ring_gaussian(
        r_pts,
        a_ring=params["A_ring"],
        r_ring=params["r_ring"],
        delta_r=params["delta_r"],
        w0=params["w0"],
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
    fit_df = load_fit_table(FIT_XLSX_PATH, FIT_SHEET_NAME)

    rows: List[Dict[str, float]] = []
    total_sae = 0.0
    total_weighted_sse = 0.0
    total_weight_sum = 0.0

    for sheet in SHEETS:
        z_m = parse_sheet_height_m(sheet)
        params = params_at_z(fit_df, z_m)

        # Mean-sheet samples are the measured reference for model residual diagnostics.
        x_centers, y_centers, W = read_slice_from_sheet(XLSX_PATH, sheet)
        x_grid, y_grid = np.meshgrid(x_centers, y_centers)
        xc, yc = FAN_CENTER_XY
        x_pts = x_grid.ravel()
        y_pts = y_grid.ravel()
        r_pts = np.sqrt((x_pts - xc) ** 2 + (y_pts - yc) ** 2)
        w_obs = W.ravel()

        mask = (
            np.isfinite(x_pts)
            & np.isfinite(y_pts)
            & np.isfinite(r_pts)
            & np.isfinite(w_obs)
        )
        x_pts = x_pts[mask]
        y_pts = y_pts[mask]
        r_pts = r_pts[mask]
        w_obs = w_obs[mask]
        if r_pts.size == 0:
            raise ValueError(f"No valid raw samples found in sheet '{sheet}'.")

        # Assign sigma per point using nearest-radius mapping from *_TS,
        # falling back to SIGMA_FALLBACK if needed.
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

        # Raw-point comparison: use unit weights per sample.
        alpha_pts = np.ones_like(r_pts, dtype=float)

        sae_k, weighted_sse_k, weight_sum_k, wrmse_k, n_samples = compute_height_metrics(
            r_pts=r_pts,
            w_obs=w_obs,
            alpha=alpha_pts,
            sigma=sigma_pts,
            params=params,
        )

        total_sae += sae_k
        total_weighted_sse += weighted_sse_k
        total_weight_sum += weight_sum_k

        rows.append(
            {
                "sheet": sheet,
                "z_m": z_m,
                "n_samples": n_samples,
                "accumulate_SAE_mps": sae_k,
                "weighted_RMSE_mps": wrmse_k,
                "weighted_SSE_term": weighted_sse_k,
                "weight_sum_term": weight_sum_k,
                "A_ring": params["A_ring"],
                "r_ring": params["r_ring"],
                "delta_r": params["delta_r"],
                "w0": params["w0"],
            }
        )

    if total_weight_sum <= 0.0:
        raise ValueError("Total WRMSE denominator is non-positive.")

    total_wrmse = float(np.sqrt(total_weighted_sse / total_weight_sum))
    rows.append(
        {
            "sheet": "TOTAL",
            "z_m": np.nan,
            "n_samples": int(sum(int(r["n_samples"]) for r in rows)),
            "accumulate_SAE_mps": float(total_sae),
            "weighted_RMSE_mps": total_wrmse,
            "weighted_SSE_term": float(total_weighted_sse),
            "weight_sum_term": float(total_weight_sum),
            "A_ring": np.nan,
            "r_ring": np.nan,
            "delta_r": np.nan,
            "w0": np.nan,
        }
    )

    df_out = pd.DataFrame(rows)
    write_results(df_out, OUT_XLSX_PATH, OUT_SHEET_NAME)

    print(f"Saved analysis metrics to: {OUT_XLSX_PATH.resolve()}")
    print(f"Sheet: {OUT_SHEET_NAME}")
    print(df_out.to_string(index=False))


if __name__ == "__main__":
    main()
