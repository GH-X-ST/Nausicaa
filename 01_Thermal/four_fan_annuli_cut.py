"""
Build four-fan annuli profiles from measured heat-map sheets.

Each sample point is assigned to the nearest fan center:
    f_i = argmin_f ||p_i - c_f||_2
    r_i = ||p_i - c_{f_i}||_2

Annulus bins are then formed from r_i with width DELTA_R_M.
Uncertainty is assigned at sample level using sampled-fan inheritance and
distance blending between all fan-centred fluctuation profiles.
"""

###### Initialization

### Imports
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import QhullError

import four_fan_gp as sigma_gp


### User settings
XLSX_PATH = "S02.xlsx"
SHEETS = ["z020", "z035", "z050", "z075", "z110", "z160", "z220"]

OUT_DIR = Path("B_results/Four_Fan_Annuli_Profile")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Four-fan centers (x_f, y_f)
FOUR_FAN_CENTERS_XY = (
    (3.0, 3.6),
    (5.4, 3.6),
    (3.0, 1.2),
    (5.4, 1.2),
)

# Annulus thickness (m)
DELTA_R_M = 0.30

# Use median rather than mean within each annulus
USE_MEDIAN_PROFILE = False

# Uncertainty assignment. The overlap constants are retained for compatibility
# with older imports; sigma_mps now uses four_fan_gp's inherited fan blending.
SIGMA_FALLBACK = 0.14
SIGMA_MIN = 0.03
OVERLAP_RATIO_THRESHOLD = 1.25
OVERLAP_WEIGHT_POWER = 2.0
OVERLAP_SIGMA_BOOST = 1.12

# Optional masking
MASK_ZEROS_AS_NODATA = False


def read_slice_from_sheet(xlsx_path: str, sheet_name: str):
    """
    Read one grid sheet:
      - row 0, col 1.. = x coordinates
      - col 0, row 1.. = y coordinates
      - interior = scalar field values

    Returns x_centers, y_centers, W (Ny x Nx).
    """
    raw = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)

    x = pd.to_numeric(raw.iloc[0, 1:], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(raw.iloc[1:, 0], errors="coerce").to_numpy(dtype=float)
    w = raw.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    if w.shape != (y.size, x.size):
        raise ValueError(
            f"Shape mismatch in {sheet_name}: W{w.shape}, y({y.size}), x({x.size})."
        )

    if y.size >= 2 and y[0] > y[-1]:
        y = y[::-1]
        w = w[::-1, :]

    return x, y, w


def _cell_is_str(df: pd.DataFrame, r_idx: int, c_idx: int, text: str) -> bool:
    """Case-insensitive equality test for a sheet cell against a string."""
    val = df.iat[r_idx, c_idx]
    return isinstance(val, str) and val.strip().lower() == text.strip().lower()


def _first_numeric_below(df: pd.DataFrame, r_idx: int, c_idx: int) -> Optional[float]:
    """Return first finite numeric value below (r_idx, c_idx) in same column."""
    col = pd.to_numeric(df.iloc[r_idx + 1 :, c_idx], errors="coerce").to_numpy(
        dtype=float
    )
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

    Returns:
        r_points: representative radii (m)
        sigma_points: representative std dev (m/s)
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
    """Parse representative-point coordinates and sigmas from a *_TS sheet."""
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
    Assign sigma to each radius value by nearest-radius mapping.
    """
    if r_points.size == 0 or sigma_points.size == 0:
        sigma_bins = np.full_like(r_bins, float(sigma_fallback), dtype=float)
        return np.maximum(sigma_bins, float(sigma_min))

    diffs = np.abs(r_bins[:, None] - r_points[None, :])
    idx = np.argmin(diffs, axis=1)
    sigma_bins = sigma_points[idx].astype(float)
    return np.maximum(sigma_bins, float(sigma_min))


def compute_nearest_fan_distances(
    x_pts: np.ndarray,
    y_pts: np.ndarray,
    fan_centers_xy: Sequence[Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute nearest and second-nearest fan distances for each sample.
    """
    fan_xy = np.asarray(fan_centers_xy, dtype=float)
    d_sample_fan = np.sqrt(
        (x_pts[:, None] - fan_xy[None, :, 0]) ** 2
        + (y_pts[:, None] - fan_xy[None, :, 1]) ** 2
    )

    nearest_idx = np.argmin(d_sample_fan, axis=1)
    nearest_r = d_sample_fan[np.arange(x_pts.size), nearest_idx]

    if fan_xy.shape[0] > 1:
        order = np.argsort(d_sample_fan, axis=1)
        second_idx = order[:, 1]
        second_r = d_sample_fan[np.arange(x_pts.size), second_idx]
    else:
        second_idx = nearest_idx.copy()
        second_r = nearest_r.copy()

    return nearest_idx, nearest_r, second_idx, second_r


def assign_sigma_samples_with_overlap(
    xlsx_path: str,
    ts_sheet_name: str,
    x_pts: np.ndarray,
    y_pts: np.ndarray,
    fan_centers_xy: Sequence[Tuple[float, float]],
    sigma_fallback: float,
    sigma_min: float,
    overlap_ratio_threshold: float,
    overlap_weight_power: float,
    overlap_sigma_boost: float,
) -> np.ndarray:
    """
    Assign per-sample sigma from local *_TS representative points in x-y.

    The legacy overlap arguments are retained for call-site compatibility, but
    the actual assignment uses linear interpolation inside the representative
    point convex hull, with nearest-neighbour extrapolation outside it.
    """
    del fan_centers_xy
    del overlap_ratio_threshold
    del overlap_weight_power
    del overlap_sigma_boost

    parsed = parse_ts_xy_points_and_sigmas(
        xlsx_path=xlsx_path,
        ts_sheet_name=ts_sheet_name,
    )
    if parsed is None:
        sigma_arr = np.full(x_pts.size, float(sigma_fallback), dtype=float)
        return np.maximum(sigma_arr, float(sigma_min))

    rep_x, rep_y, rep_sigma = parsed
    query_x = np.asarray(x_pts, dtype=float).ravel()
    query_y = np.asarray(y_pts, dtype=float).ravel()

    if rep_sigma.size == 1:
        sigma_out = np.full(query_x.size, float(rep_sigma[0]), dtype=float)
        return np.maximum(sigma_out, float(sigma_min))

    points = np.column_stack([rep_x, rep_y])
    queries = np.column_stack([query_x, query_y])

    nearest_interp = NearestNDInterpolator(points, rep_sigma)
    sigma_out = np.asarray(nearest_interp(queries), dtype=float)

    if rep_sigma.size >= 3:
        try:
            linear_interp = LinearNDInterpolator(points, rep_sigma, fill_value=np.nan)
            sigma_linear = np.asarray(linear_interp(queries), dtype=float)
            linear_mask = np.isfinite(sigma_linear)
            sigma_out[linear_mask] = sigma_linear[linear_mask]
        except (QhullError, ValueError):
            pass

    sigma_low = max(float(sigma_min), float(np.nanmin(rep_sigma)))
    sigma_high = float(np.nanmax(rep_sigma))
    sigma_out = np.clip(sigma_out, sigma_low, sigma_high)
    return np.maximum(sigma_out, float(sigma_min))


def make_radial_profile(
    r: np.ndarray,
    w: np.ndarray,
    delta_r: float,
    use_median: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct 1D radial profile by nearest-centre annulus binning.
    """
    if delta_r <= 0.0:
        raise ValueError("delta_r must be positive.")

    mask = np.isfinite(r) & np.isfinite(w)
    r = r[mask]
    w = w[mask]

    if r.size == 0:
        raise ValueError("No finite samples available to construct radial profile.")

    k = np.floor(r / delta_r + 0.5).astype(int)
    uniq_k = np.unique(k)

    r_list = []
    w_list = []
    n_list = []

    for kk in uniq_k:
        in_bin = k == kk
        w_b = w[in_bin]
        if w_b.size == 0:
            continue

        r_rep = float(kk) * float(delta_r)
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


def aggregate_sigma_to_annuli(
    r_samples: np.ndarray,
    sigma_samples: np.ndarray,
    delta_r: float,
    r_bins: np.ndarray,
    sigma_fallback: float,
    sigma_min: float,
) -> np.ndarray:
    """
    Aggregate sample-level sigma onto annulus centers.
    """
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


def fill_missing_radial_bins(
    r_bins: np.ndarray,
    w_bins: np.ndarray,
    n_bins: np.ndarray,
    alpha_bins: np.ndarray,
    sigma_bins: np.ndarray,
    delta_r: float,
    sigma_fallback: float,
    sigma_min: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fill missing annulus centers between min and max radius to ensure continuity.
    Missing bins are filled with w=0, n=0, alpha=0 and sigma by nearest-radius map.
    """
    if r_bins.size == 0:
        return r_bins, w_bins, n_bins, alpha_bins, sigma_bins

    k_exist = np.floor(r_bins / delta_r + 0.5).astype(int)
    k_min = int(np.min(k_exist))
    k_max = int(np.max(k_exist))
    k_full = np.arange(k_min, k_max + 1, dtype=int)
    r_full = k_full.astype(float) * float(delta_r)

    w_full = np.zeros(r_full.size, dtype=float)
    n_full = np.zeros(r_full.size, dtype=int)
    alpha_full = np.zeros(r_full.size, dtype=float)

    sigma_full = assign_sigma_bins_nearest(
        r_bins=r_full,
        r_points=r_bins,
        sigma_points=sigma_bins,
        sigma_fallback=sigma_fallback,
        sigma_min=sigma_min,
    )

    for idx, kk in enumerate(k_exist):
        pos = int(kk - k_min)
        w_full[pos] = float(w_bins[idx])
        n_full[pos] = int(n_bins[idx])
        alpha_full[pos] = float(alpha_bins[idx])
        sigma_full[pos] = float(sigma_bins[idx])

    return r_full, w_full, n_full, alpha_full, sigma_full


def build_annuli_profile(
    xlsx_path: str,
    mean_sheet: str,
    fan_centers_xy: Sequence[Tuple[float, float]],
    delta_r: float,
    use_median: bool,
    sigma_fallback: float,
    sigma_min: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build annuli profile (r_j, w_j, n_j, alpha_j, sigma_j) for one sheet.
    """
    x, y, w_map = read_slice_from_sheet(xlsx_path, mean_sheet)
    if MASK_ZEROS_AS_NODATA:
        w_map = w_map.copy()
        w_map[w_map == 0.0] = np.nan

    x_grid, y_grid = np.meshgrid(x, y)
    x_pts = x_grid.ravel()
    y_pts = y_grid.ravel()
    w_pts = w_map.ravel()

    valid = np.isfinite(x_pts) & np.isfinite(y_pts) & np.isfinite(w_pts)
    x_pts = x_pts[valid]
    y_pts = y_pts[valid]
    w_pts = w_pts[valid]

    _nearest_idx, r_pts, _second_idx, _second_r = compute_nearest_fan_distances(
        x_pts=x_pts,
        y_pts=y_pts,
        fan_centers_xy=fan_centers_xy,
    )

    r_bins, w_bins, n_bins, alpha_bins = make_radial_profile(
        r=r_pts,
        w=w_pts,
        delta_r=delta_r,
        use_median=use_median,
    )

    sigma_samples = sigma_gp.evaluate_sigma_points_annular_blend_pchip_z(
        xlsx_path=xlsx_path,
        sheet_names=tuple(SHEETS),
        fan_centers_xy=fan_centers_xy,
        x_pts=x_pts,
        y_pts=y_pts,
        z_pts=np.full_like(
            x_pts,
            sigma_gp.parse_sheet_height_m(mean_sheet),
            dtype=float,
        ),
        sigma_fallback=sigma_fallback,
        sigma_min=sigma_min,
    )

    sigma_bins = aggregate_sigma_to_annuli(
        r_samples=r_pts,
        sigma_samples=sigma_samples,
        delta_r=delta_r,
        r_bins=r_bins,
        sigma_fallback=sigma_fallback,
        sigma_min=sigma_min,
    )

    r_bins, w_bins, n_bins, alpha_bins, sigma_bins = fill_missing_radial_bins(
        r_bins=r_bins,
        w_bins=w_bins,
        n_bins=n_bins,
        alpha_bins=alpha_bins,
        sigma_bins=sigma_bins,
        delta_r=delta_r,
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


### Export each sheet as CSV
def main() -> None:
    for sh in SHEETS:
        r_bins, w_bins, n_bins, alpha_bins, sigma_bins = build_annuli_profile(
            xlsx_path=XLSX_PATH,
            mean_sheet=sh,
            fan_centers_xy=FOUR_FAN_CENTERS_XY,
            delta_r=DELTA_R_M,
            use_median=USE_MEDIAN_PROFILE,
            sigma_fallback=SIGMA_FALLBACK,
            sigma_min=SIGMA_MIN,
        )

        out_csv = OUT_DIR / f"{sh}_four_annuli_profile.csv"
        save_profile_csv(out_csv, r_bins, w_bins, n_bins, alpha_bins, sigma_bins)

    print(f"Saved annuli profiles to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
