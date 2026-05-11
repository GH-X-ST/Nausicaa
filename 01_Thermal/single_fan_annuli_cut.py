"""
This script reduces a 2D heat map to a radial profile by grouping samples
with similar radii into annuli of width Δr. Each sample (r_i, w_i) is assigned
to an annulus, and for each annulus j:
  - r_j: representative radius
  - w_j: annulus-averaged mean vertical velocity
  - n_j: number of samples in the annulus
  - alpha_j: weight = sqrt(n_j / n_max)
  - sigma_j: uncertainty assigned from *_TS sheets by nearest-radius mapping
"""


from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import QhullError


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Annulus Profile Configuration and Data Sources
# 2) Annulus Binning and Uncertainty Assignment
# 3) Batch CSV Export
# =============================================================================

# =============================================================================
# 1) Annulus Profile Configuration and Data Sources
# =============================================================================

XLSX_PATH = "S01.xlsx"
SHEETS = ["z020", "z035", "z050", "z075", "z110", "z160", "z220"]

OUT_DIR = Path("B_results/Single_Fan_Annuli_Profile")
OUT_DIR.mkdir(exist_ok=True)

# Fan centre (x_c, y_c) in arena metres.
FAN_CENTER_XY = (4.2, 2.4)

# Annulus thickness Δr (m)
DELTA_R_M = 0.30

# Median aggregation is an optional robustness choice for outlier-prone annuli.
USE_MEDIAN_PROFILE = False

# Uncertainty assignment
SIGMA_FALLBACK = 0.2
SIGMA_MIN = 1e-3

# Zero masking is disabled unless zero-valued cells represent missing data.
MASK_ZEROS_AS_NODATA = False

# =============================================================================
# 2) Annulus Binning and Uncertainty Assignment
# =============================================================================


def read_slice_from_sheet(xlsx_path: str, sheet_name: str):
    """
    Reads the grid sheet:
      - row 0, col 1.. = x coordinates
      - col 0, row 1.. = y coordinates
      - interior = scalar field values

    Returns x_centers, y_centers, W (Ny x Nx).
    """
    raw = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)

    # Workbook grid stores x coordinates in the first row after the corner cell.
    x = pd.to_numeric(raw.iloc[0, 1:], errors="coerce").to_numpy(dtype=float)

    # Workbook grid stores y coordinates in the first column after the corner cell.
    y = pd.to_numeric(raw.iloc[1:, 0], errors="coerce").to_numpy(dtype=float)

    # Measured vertical-velocity block (m/s).
    W = raw.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    # Workbook grid shape must match y-by-x coordinates.
    if W.shape != (y.size, x.size):
        raise ValueError(
            f"Shape mismatch in {sheet_name}: W{W.shape}, y({y.size}), x({x.size})."
        )

    # Ensure y increases bottom-to-top on the plot
    if y.size >= 2 and y[0] > y[-1]:
        y = y[::-1]
        W = W[::-1, :]

    return x, y, W


def _cell_is_str(df: pd.DataFrame, r_idx: int, c_idx: int, text: str) -> bool:
    """Case-insensitive equality test for a sheet cell against a string."""
    val = df.iat[r_idx, c_idx]
    return isinstance(val, str) and val.strip().lower() == text.strip().lower()


def _first_numeric_below(df: pd.DataFrame, r_idx: int, c_idx: int) -> Optional[float]:
    """Return the first finite numeric value below (r_idx, c_idx) in the same column."""
    col = pd.to_numeric(df.iloc[r_idx + 1 :, c_idx], errors="coerce").to_numpy(dtype=float)
    col = col[np.isfinite(col)]
    if col.size == 0:
        return None
    return float(col[0])


def parse_ts_points_and_sigmas(
    xlsx_path: str,
    ts_sheet_name: str,
    fan_center_xy: Tuple[float, float],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Parse representative-point radii and sigmas from a *_TS sheet.

    For each block with headers (..., x, y, ..., variance), extract:
      - x_p, y_p from the first data row beneath the header row
      - variance_p from the 'variance' column (first numeric below)
      - sigma_p = sqrt(variance_p)
      - r_p = sqrt((x_p - x_c)^2 + (y_p - y_c)^2)

    Returns:
        r_points: shape (P,), representative-point radii (m)
        sigma_points: shape (P,), representative-point std dev (m/s)
    or None if parsing fails.
    """
    try:
        df = pd.read_excel(xlsx_path, sheet_name=ts_sheet_name, header=None)
    except Exception:
        return None

    xc, yc = fan_center_xy
    r_points = []
    sigma_points = []

    for r_idx in range(df.shape[0]):
        for c_idx in range(df.shape[1]):
            if not _cell_is_str(df, r_idx, c_idx, "variance"):
                continue

            # Prefer the expected block layout: [x, y, time, w, mean, variance]
            x_col = None
            if (
                c_idx >= 5
                and _cell_is_str(df, r_idx, c_idx - 5, "x")
                and _cell_is_str(df, r_idx, c_idx - 4, "y")
            ):
                x_col = c_idx - 5
            else:
                # Fallback: search left for an "x" header followed by "y".
                left_start = max(0, c_idx - 12)
                for cc in range(c_idx - 1, left_start - 1, -1):
                    if (
                        _cell_is_str(df, r_idx, cc, "x")
                        and cc + 1 < df.shape[1]
                        and _cell_is_str(df, r_idx, cc + 1, "y")
                    ):
                        x_col = cc
                        break

            if x_col is None:
                continue

            # Extract x, y from the first data row beneath the header row.
            if r_idx + 1 >= df.shape[0]:
                continue

            x_val = pd.to_numeric(df.iat[r_idx + 1, x_col], errors="coerce")
            y_val = pd.to_numeric(df.iat[r_idx + 1, x_col + 1], errors="coerce")
            if not np.isfinite(x_val) or not np.isfinite(y_val):
                continue

            var_val = _first_numeric_below(df, r_idx, c_idx)
            if var_val is None or not np.isfinite(var_val) or var_val <= 0.0:
                continue

            sigma_val = float(np.sqrt(var_val))
            r_val = float(np.sqrt((float(x_val) - xc) ** 2 + (float(y_val) - yc) ** 2))

            r_points.append(r_val)
            sigma_points.append(sigma_val)

    if not r_points:
        return None

    r_arr = np.array(r_points, dtype=float)
    sigma_arr = np.array(sigma_points, dtype=float)

    order = np.argsort(r_arr)
    return r_arr[order], sigma_arr[order]


def parse_ts_xy_points_and_sigmas(
    xlsx_path: str,
    ts_sheet_name: str,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Parse representative-point coordinates and sigmas from a *_TS sheet.
    """
    try:
        df = pd.read_excel(xlsx_path, sheet_name=ts_sheet_name, header=None)
    except Exception:
        return None

    x_points = []
    y_points = []
    sigma_points = []

    for r_idx in range(df.shape[0]):
        for c_idx in range(df.shape[1]):
            if not _cell_is_str(df, r_idx, c_idx, "variance"):
                continue

            x_col = None
            if (
                c_idx >= 5
                and _cell_is_str(df, r_idx, c_idx - 5, "x")
                and _cell_is_str(df, r_idx, c_idx - 4, "y")
            ):
                x_col = c_idx - 5
            else:
                left_start = max(0, c_idx - 12)
                for cc in range(c_idx - 1, left_start - 1, -1):
                    if (
                        _cell_is_str(df, r_idx, cc, "x")
                        and cc + 1 < df.shape[1]
                        and _cell_is_str(df, r_idx, cc + 1, "y")
                    ):
                        x_col = cc
                        break

            if x_col is None or r_idx + 1 >= df.shape[0]:
                continue

            x_val = pd.to_numeric(df.iat[r_idx + 1, x_col], errors="coerce")
            y_val = pd.to_numeric(df.iat[r_idx + 1, x_col + 1], errors="coerce")
            if not np.isfinite(x_val) or not np.isfinite(y_val):
                continue

            var_val = _first_numeric_below(df, r_idx, c_idx)
            if var_val is None or not np.isfinite(var_val) or var_val <= 0.0:
                continue

            x_points.append(float(x_val))
            y_points.append(float(y_val))
            sigma_points.append(float(np.sqrt(var_val)))

    if not x_points:
        return None

    x_arr = np.asarray(x_points, dtype=float)
    y_arr = np.asarray(y_points, dtype=float)
    sigma_arr = np.asarray(sigma_points, dtype=float)
    order = np.lexsort((y_arr, x_arr))
    return x_arr[order], y_arr[order], sigma_arr[order]


def assign_sigma_bins_nearest(
    r_bins: np.ndarray,
    r_points: np.ndarray,
    sigma_points: np.ndarray,
    sigma_fallback: float,
    sigma_min: float,
) -> np.ndarray:
    """
    Assign sigma_j to each radial bin by nearest-radius mapping.

    sigma_j = sigma_{p*}, where p* = argmin_p |r_j - r_p|.
    """
    if r_points.size == 0 or sigma_points.size == 0:
        sigma_bins = np.full_like(r_bins, float(sigma_fallback), dtype=float)
        return np.maximum(sigma_bins, float(sigma_min))

    diffs = np.abs(r_bins[:, None] - r_points[None, :])
    idx = np.argmin(diffs, axis=1)
    sigma_bins = sigma_points[idx].astype(float)

    return np.maximum(sigma_bins, float(sigma_min))


def assign_sigma_points_linear_nearest(
    x_pts: np.ndarray,
    y_pts: np.ndarray,
    rep_x: np.ndarray,
    rep_y: np.ndarray,
    rep_sigma: np.ndarray,
    sigma_fallback: float,
    sigma_min: float,
) -> np.ndarray:
    """
    Interpolate sigma in x-y with linear interpolation inside the TS-point
    convex hull and nearest-neighbour extrapolation outside it.
    """
    query_x = np.asarray(x_pts, dtype=float).ravel()
    query_y = np.asarray(y_pts, dtype=float).ravel()

    if rep_sigma.size == 0:
        sigma = np.full(query_x.size, float(sigma_fallback), dtype=float)
        return np.maximum(sigma, float(sigma_min))

    if rep_sigma.size == 1:
        sigma = np.full(query_x.size, float(rep_sigma[0]), dtype=float)
        return np.maximum(sigma, float(sigma_min))

    points = np.column_stack([rep_x, rep_y])
    queries = np.column_stack([query_x, query_y])

    nearest_interp = NearestNDInterpolator(points, rep_sigma)
    sigma = np.asarray(nearest_interp(queries), dtype=float)

    if rep_sigma.size >= 3:
        try:
            linear_interp = LinearNDInterpolator(points, rep_sigma, fill_value=np.nan)
            sigma_linear = np.asarray(linear_interp(queries), dtype=float)
            linear_mask = np.isfinite(sigma_linear)
            sigma[linear_mask] = sigma_linear[linear_mask]
        except (QhullError, ValueError):
            pass

    sigma_low = max(float(sigma_min), float(np.nanmin(rep_sigma)))
    sigma_high = float(np.nanmax(rep_sigma))
    sigma = np.clip(sigma, sigma_low, sigma_high)
    return np.maximum(sigma, float(sigma_min))


def aggregate_sigma_to_annuli(
    r_samples: np.ndarray,
    sigma_samples: np.ndarray,
    delta_r: float,
    r_bins: np.ndarray,
    sigma_fallback: float,
    sigma_min: float,
) -> np.ndarray:
    """Aggregate sample-level sigma to annulus centers by annulus mean."""
    mask = np.isfinite(r_samples) & np.isfinite(sigma_samples)
    if not np.any(mask):
        sigma_bins = np.full_like(r_bins, float(sigma_fallback), dtype=float)
        return np.maximum(sigma_bins, float(sigma_min))

    r_valid = r_samples[mask]
    sigma_valid = sigma_samples[mask]
    k = np.floor(r_valid / delta_r + 0.5).astype(int)
    uniq_k = np.unique(k)

    r_points = []
    sigma_points = []
    for kk in uniq_k:
        in_bin = k == kk
        sigma_b = sigma_valid[in_bin]
        if sigma_b.size == 0:
            continue
        r_points.append(float(kk) * float(delta_r))
        sigma_points.append(float(np.mean(sigma_b)))

    if not r_points:
        sigma_bins = np.full_like(r_bins, float(sigma_fallback), dtype=float)
        return np.maximum(sigma_bins, float(sigma_min))

    return assign_sigma_bins_nearest(
        r_bins=r_bins,
        r_points=np.asarray(r_points, dtype=float),
        sigma_points=np.asarray(sigma_points, dtype=float),
        sigma_fallback=sigma_fallback,
        sigma_min=sigma_min,
    )


def make_radial_profile(
    r: np.ndarray,
    w: np.ndarray,
    delta_r: float,
    use_median: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct a 1D radial profile by binning samples (r_i, w_i) into annuli
    using nearest-centre assignment:

        r_i* = round(r_i / Δr) Δr

    Returns:
        r_bins: bin-centre radius per annulus, shape (M,)
        w_bins: aggregated velocity per annulus, shape (M,)
        n_bins: counts per annulus, shape (M,)
        alpha_bins: weights per annulus, shape (M,)
    """
    if delta_r <= 0.0:
        raise ValueError("delta_r must be positive.")

    mask = np.isfinite(r) & np.isfinite(w)
    r = r[mask]
    w = w[mask]

    if r.size == 0:
        raise ValueError("No finite samples available to construct radial profile.")

    # Nearest-centre binning (avoids NumPy's bankers rounding at exact half steps)
    k = np.floor(r / delta_r + 0.5).astype(int)  # integer bin index
    r_star = k.astype(float) * float(delta_r)    # r_i*

    # Group by bin index
    uniq_k = np.unique(k)

    r_list = []
    w_list = []
    n_list = []

    for kk in uniq_k:
        in_bin = k == kk
        w_b = w[in_bin]
        if w_b.size == 0:
            continue

        r_rep = float(kk) * float(delta_r)  # r_j = k Δr (bin centre)
        w_rep = float(np.median(w_b) if use_median else np.mean(w_b))
        n_rep = int(w_b.size)

        r_list.append(r_rep)
        w_list.append(w_rep)
        n_list.append(n_rep)

    r_bins = np.array(r_list, dtype=float)
    w_bins = np.array(w_list, dtype=float)
    n_bins = np.array(n_list, dtype=int)

    order = np.argsort(r_bins)
    r_bins = r_bins[order]
    w_bins = w_bins[order]
    n_bins = n_bins[order]

    n_max = int(np.max(n_bins))
    alpha_bins = np.sqrt(n_bins / n_max)

    return r_bins, w_bins, n_bins, alpha_bins


def build_annuli_profile(
    xlsx_path: str,
    mean_sheet: str,
    fan_center_xy: Tuple[float, float],
    delta_r: float,
    use_median: bool,
    sigma_fallback: float,
    sigma_min: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build annuli profile (r_j, w_j, n_j, alpha_j, sigma_j) for a height sheet.
    """
    x, y, W = read_slice_from_sheet(xlsx_path, mean_sheet)
    if MASK_ZEROS_AS_NODATA:
        W = W.copy()
        W[W == 0.0] = np.nan

    x_grid, y_grid = np.meshgrid(x, y)
    xc, yc = fan_center_xy
    x_pts = x_grid.ravel()
    y_pts = y_grid.ravel()
    r = np.sqrt((x_pts - xc) ** 2 + (y_pts - yc) ** 2)
    w = W.ravel()

    valid = np.isfinite(x_pts) & np.isfinite(y_pts) & np.isfinite(r) & np.isfinite(w)
    x_pts = x_pts[valid]
    y_pts = y_pts[valid]
    r = r[valid]
    w = w[valid]

    r_bins, w_bins, n_bins, alpha_bins = make_radial_profile(
        r=r,
        w=w,
        delta_r=delta_r,
        use_median=use_median,
    )

    ts_sheet = f"{mean_sheet}_TS"
    ts_parsed = parse_ts_xy_points_and_sigmas(
        xlsx_path=xlsx_path,
        ts_sheet_name=ts_sheet,
    )

    if ts_parsed is None:
        sigma_bins = np.full_like(r_bins, float(sigma_fallback), dtype=float)
        sigma_bins = np.maximum(sigma_bins, float(sigma_min))
    else:
        rep_x, rep_y, rep_sigma = ts_parsed
        sigma_samples = assign_sigma_points_linear_nearest(
            x_pts=x_pts,
            y_pts=y_pts,
            rep_x=rep_x,
            rep_y=rep_y,
            rep_sigma=rep_sigma,
            sigma_fallback=sigma_fallback,
            sigma_min=sigma_min,
        )
        sigma_bins = aggregate_sigma_to_annuli(
            r_samples=r,
            sigma_samples=sigma_samples,
            delta_r=delta_r,
            r_bins=r_bins,
            sigma_fallback=sigma_fallback,
            sigma_min=sigma_min,
        )

    return r_bins, w_bins, n_bins, alpha_bins, sigma_bins


def save_profile_csv(
    out_path: Path,
    r_bins: np.ndarray,
    w_bins: np.ndarray,
    n_bins: np.ndarray,
    alpha_bins: np.ndarray,
    sigma_bins: np.ndarray,
) -> None:
    """
    Save annuli profile to CSV.
    """
    df = pd.DataFrame(
        {
            "r_m": r_bins,
            "w_mps": w_bins,
            "n": n_bins,
            "alpha": alpha_bins,
            "sigma_mps": sigma_bins,
        }
    )
    df.to_csv(out_path, index=False)


# =============================================================================
# 3) Batch CSV Export
# =============================================================================

def main():
    for sh in SHEETS:
        r_bins, w_bins, n_bins, alpha_bins, sigma_bins = build_annuli_profile(
            xlsx_path=XLSX_PATH,
            mean_sheet=sh,
            fan_center_xy=FAN_CENTER_XY,
            delta_r=DELTA_R_M,
            use_median=USE_MEDIAN_PROFILE,
            sigma_fallback=SIGMA_FALLBACK,
            sigma_min=SIGMA_MIN,
        )

        out_csv = OUT_DIR / f"{sh}_single_annuli_profile.csv"
        save_profile_csv(out_csv, r_bins, w_bins, n_bins, alpha_bins, sigma_bins)

    print(f"Saved annuli profiles to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
