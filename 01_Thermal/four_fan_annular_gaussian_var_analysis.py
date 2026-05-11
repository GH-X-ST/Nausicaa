"""
Compute error metrics for the four-fan annular Gaussian model.

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
import re

import numpy as np
import pandas as pd
import four_fan_gp as sigma_gp

from four_fan_annuli_cut import (
    assign_sigma_samples_with_overlap,
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

FIT_XLSX_PATH = Path("B_results/four_annular_var_params_pchip.xlsx")
FIT_SHEET_NAME = "four_annular_var_pchip"

# Output is written to a new/replaced sheet in the fit workbook.
OUT_XLSX_PATH = Path("B_results/four_annular_var_analysis.xlsx")
OUT_SHEET_NAME = "four_annular_var_analysis"

# Keep settings aligned with four_fan_annuli_cut.py.
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
    if "z_m" not in df.columns:
        raise ValueError("Fit table must contain 'z_m'.")

    return df.copy()


def discover_fan_ids(fit_df: pd.DataFrame) -> Tuple[str, ...]:
    """
    Discover fan IDs from columns like A_ring_F01.
    """
    pattern = re.compile(r"^A_ring_(F\d{2})$")
    fan_ids = []
    for col in fit_df.columns:
        match = pattern.match(str(col))
        if match is not None:
            fan_ids.append(match.group(1))
    fan_ids = sorted(set(fan_ids))

    valid = []
    for fan_id in fan_ids:
        required = (
            f"A_ring_{fan_id}",
            f"r_ring_{fan_id}",
            f"delta_r_{fan_id}",
            f"w0_{fan_id}",
        )
        if all(col in fit_df.columns for col in required):
            valid.append(fan_id)
    return tuple(valid)


def params_by_fan_at_z(
    fit_df: pd.DataFrame,
    fan_ids: Tuple[str, ...],
    z_m: float,
) -> np.ndarray:
    """
    Interpolate [A_ring, r_ring, delta_r, w0] at one target height for each fan.
    """
    z_grid = pd.to_numeric(fit_df["z_m"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(z_grid)):
        raise ValueError("Non-numeric z_m values found in fit table.")

    order = np.argsort(z_grid)
    z_grid = z_grid[order]
    if np.any(np.diff(z_grid) <= 0.0):
        raise ValueError("Fit z_m must be strictly increasing.")

    z_min = float(np.min(z_grid))
    z_max = float(np.max(z_grid))
    if z_m < z_min - 1e-12 or z_m > z_max + 1e-12:
        raise ValueError(
            f"Requested z={z_m:.3f} m outside fit range [{z_min:.3f}, {z_max:.3f}] m."
        )

    if len(fan_ids) == 0:
        shared_missing = [c for c in REQUIRED_FIT_COLUMNS if c not in fit_df.columns]
        if shared_missing:
            raise ValueError(f"Missing shared columns in fit table: {shared_missing}")
        a = pd.to_numeric(fit_df["A_ring"], errors="coerce").to_numpy(dtype=float)[order]
        r = pd.to_numeric(fit_df["r_ring"], errors="coerce").to_numpy(dtype=float)[order]
        d = pd.to_numeric(fit_df["delta_r"], errors="coerce").to_numpy(dtype=float)[order]
        w0 = pd.to_numeric(fit_df["w0"], errors="coerce").to_numpy(dtype=float)[order]
        shared = np.array(
            [
                float(np.interp(z_m, z_grid, a)),
                float(np.interp(z_m, z_grid, r)),
                float(np.interp(z_m, z_grid, d)),
                float(np.interp(z_m, z_grid, w0)),
            ],
            dtype=float,
        )
        return np.repeat(shared[None, :], len(FOUR_FAN_CENTERS_XY), axis=0)

    params = np.empty((len(fan_ids), 4), dtype=float)
    for fan_idx, fan_id in enumerate(fan_ids):
        a = pd.to_numeric(fit_df[f"A_ring_{fan_id}"], errors="coerce").to_numpy(dtype=float)[order]
        r = pd.to_numeric(fit_df[f"r_ring_{fan_id}"], errors="coerce").to_numpy(dtype=float)[order]
        d = pd.to_numeric(fit_df[f"delta_r_{fan_id}"], errors="coerce").to_numpy(dtype=float)[order]
        w0 = pd.to_numeric(fit_df[f"w0_{fan_id}"], errors="coerce").to_numpy(dtype=float)[order]
        params[fan_idx, 0] = float(np.interp(z_m, z_grid, a))
        params[fan_idx, 1] = float(np.interp(z_m, z_grid, r))
        params[fan_idx, 2] = float(np.interp(z_m, z_grid, d))
        params[fan_idx, 3] = float(np.interp(z_m, z_grid, w0))
    return params


def compute_height_metrics(
    w_pred: np.ndarray,
    w_obs: np.ndarray,
    alpha: np.ndarray,
    sigma: np.ndarray,
) -> Tuple[float, float, float, float, int]:
    """
    Return:
        sae_k, weighted_sse_k, weight_sum_k, wrmse_k, n_annuli
    """
    sigma_safe = np.maximum(sigma.astype(float), float(SIGMA_MIN))
    err = w_pred - w_obs

    sae_k = float(np.sum(np.abs(err)))

    weights = (alpha / sigma_safe) ** 2
    weighted_sse_k = float(np.sum(weights * err**2))
    weight_sum_k = float(np.sum(weights))
    if weight_sum_k <= 0.0:
        raise ValueError("Non-positive total weight encountered in WRMSE calculation.")
    wrmse_k = float(np.sqrt(weighted_sse_k / weight_sum_k))

    return sae_k, weighted_sse_k, weight_sum_k, wrmse_k, int(w_obs.size)


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

    for sheet in SHEETS:
        z_m = parse_sheet_height_m(sheet)
        params_fan = params_by_fan_at_z(fit_df, fan_ids=fan_ids, z_m=z_m)
        params_mean = np.mean(params_fan, axis=0)

        # Mean-sheet samples are the measured reference for model residual diagnostics.
        x_centers, y_centers, W = read_slice_from_sheet(XLSX_PATH, sheet)
        x_grid, y_grid = np.meshgrid(x_centers, y_centers)
        x_pts = x_grid.ravel()
        y_pts = y_grid.ravel()
        fan_xy = np.asarray(FOUR_FAN_CENTERS_XY, dtype=float)
        r_all = np.sqrt(
            (x_pts[:, None] - fan_xy[None, :, 0]) ** 2
            + (y_pts[:, None] - fan_xy[None, :, 1]) ** 2
        )
        w_obs = W.ravel()

        mask = np.isfinite(x_pts) & np.isfinite(y_pts) & np.isfinite(w_obs)
        x_pts = x_pts[mask]
        y_pts = y_pts[mask]
        r_all = r_all[mask, :]
        w_obs = w_obs[mask]
        if x_pts.size == 0:
            raise ValueError(f"No valid raw samples found in sheet '{sheet}'.")

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

        w_pred = np.zeros_like(w_obs, dtype=float)
        n_fans = len(FOUR_FAN_CENTERS_XY)
        for fan_idx in range(n_fans):
            p = params_fan[fan_idx]
            w_pred += ring_gaussian(
                r_all[:, fan_idx],
                a_ring=float(p[0]),
                r_ring=float(p[1]),
                delta_r=float(p[2]),
                w0=float(p[3]),
            )

        # Raw-point comparison: use unit weights per sample.
        alpha_pts = np.ones_like(x_pts, dtype=float)

        sae_k, weighted_sse_k, weight_sum_k, wrmse_k, n_samples = compute_height_metrics(
            w_pred=w_pred,
            w_obs=w_obs,
            alpha=alpha_pts,
            sigma=sigma_pts,
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
                "A_ring": float(params_mean[0]),
                "r_ring": float(params_mean[1]),
                "delta_r": float(params_mean[2]),
                "w0": float(params_mean[3]),
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
