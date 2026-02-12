"""
Fit a non-axisymmetric annular-Gaussian model for single-fan BEMT maps.

This script fits, at each measurement height z, the model:

    theta = atan2(y - y_c, x - x_c)
    r     = sqrt((x - x_c)^2 + (y - y_c)^2)
    g(r)  = exp(-((r - r_ring) / delta_ring)^2)

    A(theta) = a0 + sum_{n=1..N}(a_n cos(n theta) + b_n sin(n theta))
    w_model(x, y) = w0 + g(r) * A(theta)

using bounded robust nonlinear least squares with loss="soft_l1", and saves
per-height fitted coefficients to Excel.

PCHIP interpolation in z is handled by single_fan_annular_gaussian_bemt_fit.py.

Input annuli profiles are taken from single_fan_annuli_cut.py:
    B_results/Single_Fan_Annuli_Profile/
    <sheet>_single_annuli_profile.csv
"""

###### Initialization

### Imports
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


### User settings
XLSX_PATH = "S01.xlsx"
SHEETS = ["z020", "z035", "z050", "z075", "z110", "z160", "z220"]
ANNULI_PROFILE_DIR = Path("B_results/Single_Fan_Annuli_Profile")

OUT_AZ_PARAMS_XLSX = Path("B_results/single_annular_bemt_params.xlsx")
OUT_AZ_PARAMS_SHEET = "single_bemt_az_fit"

# Fan centre (x_c, y_c)
FAN_CENTER_XY = (4.2, 2.4)

# Uncertainty assignment
SIGMA_FALLBACK = 0.2
SIGMA_MIN = 1e-3

# Optional masking
MASK_ZEROS_AS_NODATA = False

# Non-axisymmetric model settings
# Tuned default: mild non-axisymmetry with low overfit risk.
FOURIER_ORDER_N = 1
ROBUST_LOSS = "soft_l1"
ROBUST_F_SCALE = 1.0

# Parameter bounds for non-axisymmetric fitting
DELTA_RING_MIN_M = 0.12
DELTA_RING_MAX_M = 1.00
COEFF_ABS_MAX_MPS = 10.0
COEFF_ABS_MIN_MPS = 2.0
COEFF_DYNAMIC_SPAN_SCALE = 1.25
HARMONIC_BOUND_FRACTION = 0.70
HARMONIC_BOUND_MIN_MPS = 0.5
W0_MARGIN_MPS = 1.0

# Harmonic regularization (applied in residual space)
ENABLE_HARMONIC_REGULARIZATION = True
HARMONIC_RIDGE_LAMBDA = 0.02
HARMONIC_REL_CAP_LAMBDA = 0.10
HARMONIC_REL_MAX_TO_A0 = 0.80
HARMONIC_PRIOR_SCALE_MPS = 1.0
HARMONIC_A0_FLOOR_MPS = 0.2
HARMONIC_ORDER_WEIGHT_EXP = 1.0

# z020 -> 0.20 m, z110 -> 1.10 m, etc.
SHEET_HEIGHT_DIVISOR = 100.0


### Data classes
@dataclass(frozen=True)
class AzimuthalBounds:
    """Bounds for non-axisymmetric annular-Gaussian fitting."""

    r_ring_min: float
    r_ring_max: float
    delta_ring_min: float = DELTA_RING_MIN_M
    delta_ring_max: float = DELTA_RING_MAX_M
    coeff_abs_max: float = COEFF_ABS_MAX_MPS


@dataclass
class AzimuthalFitResult:
    """Container for per-height fit outputs and diagnostics."""

    sheet_name: str
    z_m: float
    params: np.ndarray
    n_samples: int
    rmse_mps: float
    mae_mps: float
    success: bool
    cost: float


### Helpers
def parse_sheet_height_m(sheet_name: str) -> float:
    """
    Parse sheet names like z020, z110, z220 into meters.
    """
    if not sheet_name.startswith("z"):
        raise ValueError(f"Invalid sheet name (expected 'z###'): {sheet_name}")

    suffix = sheet_name[1:]
    if not suffix.isdigit():
        raise ValueError(f"Invalid height code in sheet name: {sheet_name}")

    return int(suffix) / SHEET_HEIGHT_DIVISOR


def read_slice_from_sheet(xlsx_path: str, sheet_name: str):
    """
    Reads the grid sheet:
      - row 0, col 1.. = x coordinates
      - col 0, row 1.. = y coordinates
      - interior = scalar field values

    Returns x_centers, y_centers, W (Ny x Nx).
    """
    raw = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)

    # x along first row (skip [0,0])
    x = pd.to_numeric(raw.iloc[0, 1:], errors="coerce").to_numpy(dtype=float)

    # y along first column (skip [0,0])
    y = pd.to_numeric(raw.iloc[1:, 0], errors="coerce").to_numpy(dtype=float)

    # field values
    w_map = raw.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    # sanity checks
    if w_map.shape != (y.size, x.size):
        raise ValueError(
            f"Shape mismatch in {sheet_name}: W{w_map.shape}, y({y.size}), x({x.size})."
        )

    # Ensure y increases bottom-to-top on the plot
    if y.size >= 2 and y[0] > y[-1]:
        y = y[::-1]
        w_map = w_map[::-1, :]

    return x, y, w_map


def load_annuli_profile_csv(
    profile_dir: Path,
    sheet_name: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load annuli profile CSV produced by single_fan_annuli_cut.py.

    Returns:
        r_bins: annulus radii
        w_bins: annulus mean velocities
        sigma_bins: optional per-annulus sigma_j from sigma_mps column
    """
    csv_path = Path(profile_dir) / f"{sheet_name}_single_annuli_profile.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing annuli profile CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    if "r_m" not in df.columns or "w_mps" not in df.columns:
        raise ValueError(f"CSV {csv_path} must contain columns: r_m, w_mps.")

    r_bins = pd.to_numeric(df["r_m"], errors="coerce").to_numpy(dtype=float)
    w_bins = pd.to_numeric(df["w_mps"], errors="coerce").to_numpy(dtype=float)
    sigma_bins = None
    if "sigma_mps" in df.columns:
        sigma_bins = pd.to_numeric(df["sigma_mps"], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(r_bins) & np.isfinite(w_bins)
    r_bins = r_bins[mask]
    w_bins = w_bins[mask]
    if sigma_bins is not None:
        sigma_bins = sigma_bins[mask]

    if r_bins.size == 0:
        raise ValueError(f"No valid annuli data in {csv_path}.")

    order = np.argsort(r_bins)
    r_bins = r_bins[order]
    w_bins = w_bins[order]
    if sigma_bins is not None:
        sigma_bins = sigma_bins[order]

    return r_bins, w_bins, sigma_bins


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


def assign_sigma_samples_nearest(
    r_samples: np.ndarray,
    r_points: np.ndarray,
    sigma_points: np.ndarray,
    sigma_fallback: float,
    sigma_min: float,
) -> np.ndarray:
    """
    Assign sigma_i to raw Cartesian samples by nearest-radius mapping.
    """
    if r_points.size == 0 or sigma_points.size == 0:
        sigma = np.full_like(r_samples, float(sigma_fallback), dtype=float)
        return np.maximum(sigma, float(sigma_min))

    diffs = np.abs(r_samples[:, None] - r_points[None, :])
    idx = np.argmin(diffs, axis=1)
    sigma = sigma_points[idx].astype(float)
    return np.maximum(sigma, float(sigma_min))


def default_r_ring_bounds_by_z(z_m: float) -> Tuple[float, float]:
    """
    Height-dependent bounds for ring radius r_ring(z).
    """
    if z_m <= 0.35:
        return 0.20, 0.65
    if z_m <= 1.10:
        return 0.15, 1.00
    return 0.10, 1.35


def build_param_column_names(fourier_order: int) -> List[str]:
    """
    Build parameter column names:
        [w0, r_ring, delta_ring, a0, a1, b1, ..., aN, bN]
    """
    names = ["w0", "r_ring", "delta_ring", "a0"]
    for n_idx in range(1, fourier_order + 1):
        names.append(f"a{n_idx}")
        names.append(f"b{n_idx}")
    return names


def radial_ring_envelope(
    r: np.ndarray,
    r_ring: float,
    delta_ring: float,
) -> np.ndarray:
    """
    Evaluate g(r) = exp(-((r - r_ring) / delta_ring)^2).
    """
    return np.exp(-((r - r_ring) / delta_ring) ** 2)


def azimuthal_amplitude(
    theta: np.ndarray,
    coeffs: np.ndarray,
    fourier_order: int,
) -> np.ndarray:
    """
    Evaluate A(theta) = a0 + sum_{n=1..N}[a_n cos(n theta) + b_n sin(n theta)].
    """
    expected_size = 1 + 2 * fourier_order
    if coeffs.size != expected_size:
        raise ValueError(
            f"Expected {expected_size} Fourier coefficients, got {coeffs.size}."
        )

    amp = np.full_like(theta, float(coeffs[0]), dtype=float)  # a0
    for n_idx in range(1, fourier_order + 1):
        a_n = float(coeffs[2 * n_idx - 1])
        b_n = float(coeffs[2 * n_idx])
        amp += a_n * np.cos(n_idx * theta) + b_n * np.sin(n_idx * theta)
    return amp


def azimuthal_ring_model(
    r: np.ndarray,
    theta: np.ndarray,
    params: np.ndarray,
    fourier_order: int,
) -> np.ndarray:
    """
    Evaluate w_model(x, y) = w0 + g(r) * A(theta).
    """
    expected_size = 4 + 2 * fourier_order
    if params.size != expected_size:
        raise ValueError(f"Expected {expected_size} model parameters, got {params.size}.")

    w0 = float(params[0])
    r_ring = float(params[1])
    delta_ring = float(params[2])
    coeffs = params[3:]

    g_r = radial_ring_envelope(r, r_ring=r_ring, delta_ring=delta_ring)
    a_theta = azimuthal_amplitude(theta, coeffs=coeffs, fourier_order=fourier_order)
    return w0 + g_r * a_theta


def build_param_bounds(
    w_samples: np.ndarray,
    bounds: AzimuthalBounds,
    fourier_order: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build lower/upper bounds for [w0, r_ring, delta_ring, a0, a1, b1, ...].
    """
    w_min = float(np.min(w_samples))
    w_max = float(np.max(w_samples))
    w_floor = float(np.percentile(w_samples, 10.0))
    w_peak = float(np.percentile(w_samples, 99.5))
    ring_scale = max(w_peak - w_floor, 1e-6)

    coeff_bound = float(
        np.clip(
            COEFF_DYNAMIC_SPAN_SCALE * ring_scale,
            COEFF_ABS_MIN_MPS,
            bounds.coeff_abs_max,
        )
    )
    harmonic_bound = float(
        np.clip(
            HARMONIC_BOUND_FRACTION * coeff_bound,
            HARMONIC_BOUND_MIN_MPS,
            bounds.coeff_abs_max,
        )
    )

    lower = [
        w_min - W0_MARGIN_MPS,
        bounds.r_ring_min,
        bounds.delta_ring_min,
        0.0,
    ]
    upper = [
        w_max + W0_MARGIN_MPS,
        bounds.r_ring_max,
        bounds.delta_ring_max,
        coeff_bound,
    ]

    for _ in range(fourier_order):
        lower.extend([-harmonic_bound, -harmonic_bound])
        upper.extend([harmonic_bound, harmonic_bound])

    return np.array(lower, dtype=float), np.array(upper, dtype=float)


def initial_guess_azimuthal(
    r_samples: np.ndarray,
    w_samples: np.ndarray,
    bounds: AzimuthalBounds,
    fourier_order: int,
    r_profile: Optional[np.ndarray] = None,
    w_profile: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Initial guess for [w0, r_ring, delta_ring, a0, a1, b1, ...].
    """
    if r_samples.size == 0 or w_samples.size == 0:
        raise ValueError("Cannot initialise azimuthal fit with empty samples.")

    p0 = np.zeros(4 + 2 * fourier_order, dtype=float)

    use_profile_guess = (
        r_profile is not None
        and w_profile is not None
        and r_profile.size > 0
        and w_profile.size == r_profile.size
    )

    if use_profile_guess:
        tail_len = max(3, int(0.2 * w_profile.size))
        w0_0 = float(np.median(w_profile[-tail_len:]))
        peak_idx = int(np.argmax(w_profile))
        r_ring_seed = float(r_profile[peak_idx])
        amp_seed = float(w_profile[peak_idx] - w0_0)
    else:
        w0_0 = float(np.percentile(w_samples, 10.0))
        peak_idx = int(np.argmax(w_samples))
        r_ring_seed = float(r_samples[peak_idx])
        amp_seed = float(w_samples[peak_idx] - w0_0)

    r_ring_0 = float(np.clip(r_ring_seed, bounds.r_ring_min, bounds.r_ring_max))
    delta_ring_0 = float(np.clip(0.25, bounds.delta_ring_min, bounds.delta_ring_max))
    a0_0 = float(max(amp_seed, 0.05))
    a0_0 = float(min(a0_0, bounds.coeff_abs_max))

    p0[0] = w0_0
    p0[1] = r_ring_0
    p0[2] = delta_ring_0
    p0[3] = a0_0
    return p0


def fit_azimuthal_model(
    x_samples: np.ndarray,
    y_samples: np.ndarray,
    w_samples: np.ndarray,
    sigma_samples: np.ndarray,
    fan_center_xy: Tuple[float, float],
    z_m: float,
    fourier_order: int,
    r_profile: Optional[np.ndarray] = None,
    w_profile: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, bool, float]:
    """
    Fit non-axisymmetric model parameters at a single height.

    Residuals:
        u_i = (w_model(x_i, y_i) - w_obs(x_i, y_i)) / sigma_i
    """
    if (
        x_samples.shape != y_samples.shape
        or x_samples.shape != w_samples.shape
        or x_samples.shape != sigma_samples.shape
    ):
        raise ValueError("x_samples, y_samples, w_samples, sigma_samples must match.")

    xc, yc = fan_center_xy
    r_samples = np.sqrt((x_samples - xc) ** 2 + (y_samples - yc) ** 2)
    theta_samples = np.arctan2(y_samples - yc, x_samples - xc)

    r_ring_min, r_ring_max = default_r_ring_bounds_by_z(z_m)
    bounds = AzimuthalBounds(r_ring_min=r_ring_min, r_ring_max=r_ring_max)

    lower, upper = build_param_bounds(
        w_samples=w_samples,
        bounds=bounds,
        fourier_order=fourier_order,
    )
    p0 = initial_guess_azimuthal(
        r_samples=r_samples,
        w_samples=w_samples,
        bounds=bounds,
        fourier_order=fourier_order,
        r_profile=r_profile,
        w_profile=w_profile,
    )
    p0 = np.clip(p0, lower + 1e-12, upper - 1e-12)

    sigma_safe = np.maximum(sigma_samples.astype(float), float(SIGMA_MIN))

    def residuals(p: np.ndarray) -> np.ndarray:
        w_pred = azimuthal_ring_model(
            r=r_samples,
            theta=theta_samples,
            params=p,
            fourier_order=fourier_order,
        )
        data_residual = (w_pred - w_samples) / sigma_safe

        if not ENABLE_HARMONIC_REGULARIZATION or fourier_order <= 0:
            return data_residual

        coeffs = p[3:]
        a0 = float(max(coeffs[0], HARMONIC_A0_FLOOR_MPS))
        reg_terms = []
        ridge_weight = float(np.sqrt(max(HARMONIC_RIDGE_LAMBDA, 0.0)))
        relcap_weight = float(np.sqrt(max(HARMONIC_REL_CAP_LAMBDA, 0.0)))
        scale = float(max(HARMONIC_PRIOR_SCALE_MPS, 1e-12))

        for n_idx in range(1, fourier_order + 1):
            a_n = float(coeffs[2 * n_idx - 1])
            b_n = float(coeffs[2 * n_idx])
            order_weight = float(n_idx ** HARMONIC_ORDER_WEIGHT_EXP)

            if ridge_weight > 0.0:
                reg_terms.append(ridge_weight * order_weight * a_n / scale)
                reg_terms.append(ridge_weight * order_weight * b_n / scale)

            if relcap_weight > 0.0:
                harm_mag = float(np.sqrt(a_n**2 + b_n**2))
                cap = float(HARMONIC_REL_MAX_TO_A0 * a0)
                excess = max(harm_mag - cap, 0.0)
                reg_terms.append(relcap_weight * order_weight * excess / scale)

        if not reg_terms:
            return data_residual

        return np.concatenate([data_residual, np.asarray(reg_terms, dtype=float)])

    result = least_squares(
        residuals,
        p0,
        bounds=(lower, upper),
        loss=ROBUST_LOSS,
        f_scale=ROBUST_F_SCALE,
    )
    return result.x.astype(float), bool(result.success), float(result.cost)


def fit_azimuthal_for_sheet(
    xlsx_path: str,
    mean_sheet: str,
    fan_center_xy: Tuple[float, float],
    fourier_order: int,
    sigma_fallback: float,
    sigma_min: float,
) -> AzimuthalFitResult:
    """
    Fit non-axisymmetric annular-Gaussian model for one mean sheet.
    """
    x, y, w_map = read_slice_from_sheet(xlsx_path, mean_sheet)
    r_profile, w_profile, sigma_profile = load_annuli_profile_csv(
        profile_dir=ANNULI_PROFILE_DIR,
        sheet_name=mean_sheet,
    )
    if MASK_ZEROS_AS_NODATA:
        w_map = w_map.copy()
        w_map[w_map == 0.0] = np.nan

    x_grid, y_grid = np.meshgrid(x, y)

    x_samples = x_grid.ravel()
    y_samples = y_grid.ravel()
    w_samples = w_map.ravel()

    valid = np.isfinite(x_samples) & np.isfinite(y_samples) & np.isfinite(w_samples)
    x_samples = x_samples[valid]
    y_samples = y_samples[valid]
    w_samples = w_samples[valid]

    if w_samples.size == 0:
        raise ValueError(f"No finite Cartesian samples in sheet '{mean_sheet}'.")

    xc, yc = fan_center_xy
    r_samples = np.sqrt((x_samples - xc) ** 2 + (y_samples - yc) ** 2)

    # Prefer sigma from annuli profile output (single_fan_annuli_cut.py),
    # then fall back to TS-derived nearest-radius mapping.
    if sigma_profile is not None:
        valid_sigma = (
            np.isfinite(r_profile)
            & np.isfinite(sigma_profile)
            & (sigma_profile > 0.0)
        )
        if np.any(valid_sigma):
            sigma_samples = assign_sigma_samples_nearest(
                r_samples=r_samples,
                r_points=r_profile[valid_sigma],
                sigma_points=sigma_profile[valid_sigma],
                sigma_fallback=sigma_fallback,
                sigma_min=sigma_min,
            )
        else:
            sigma_samples = np.full_like(w_samples, float(sigma_fallback), dtype=float)
    else:
        ts_sheet = f"{mean_sheet}_TS"
        ts_parsed = parse_ts_points_and_sigmas(
            xlsx_path=xlsx_path,
            ts_sheet_name=ts_sheet,
            fan_center_xy=fan_center_xy,
        )
        if ts_parsed is None:
            sigma_samples = np.full_like(w_samples, float(sigma_fallback), dtype=float)
        else:
            r_points, sigma_points = ts_parsed
            sigma_samples = assign_sigma_samples_nearest(
                r_samples=r_samples,
                r_points=r_points,
                sigma_points=sigma_points,
                sigma_fallback=sigma_fallback,
                sigma_min=sigma_min,
            )

    sigma_samples = np.maximum(sigma_samples, float(sigma_min))
    z_m = parse_sheet_height_m(mean_sheet)

    params, success, cost = fit_azimuthal_model(
        x_samples=x_samples,
        y_samples=y_samples,
        w_samples=w_samples,
        sigma_samples=sigma_samples,
        fan_center_xy=fan_center_xy,
        z_m=z_m,
        fourier_order=fourier_order,
        r_profile=r_profile,
        w_profile=w_profile,
    )

    theta_samples = np.arctan2(y_samples - yc, x_samples - xc)
    w_pred = azimuthal_ring_model(
        r=r_samples,
        theta=theta_samples,
        params=params,
        fourier_order=fourier_order,
    )
    err = w_pred - w_samples
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))

    return AzimuthalFitResult(
        sheet_name=mean_sheet,
        z_m=z_m,
        params=params,
        n_samples=int(w_samples.size),
        rmse_mps=rmse,
        mae_mps=mae,
        success=success,
        cost=cost,
    )


def save_azimuthal_fit_table(
    fit_results: List[AzimuthalFitResult],
    out_path: Path,
    sheet_name: str,
    fourier_order: int,
) -> None:
    """
    Save per-height fitted non-axisymmetric parameters to Excel.
    """
    if not fit_results:
        raise ValueError("No azimuthal fit results to save.")

    param_names = build_param_column_names(fourier_order)

    rows = []
    for fit_result in fit_results:
        row = {
            "sheet": fit_result.sheet_name,
            "z_m": fit_result.z_m,
            "n_samples": fit_result.n_samples,
            "rmse_mps": fit_result.rmse_mps,
            "mae_mps": fit_result.mae_mps,
            "success": fit_result.success,
            "cost": fit_result.cost,
        }
        for name, val in zip(param_names, fit_result.params):
            row[name] = float(val)
        rows.append(row)

    df_out = pd.DataFrame(rows).sort_values("z_m").reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_excel(out_path, index=False, sheet_name=sheet_name)


### Main
def main() -> None:
    fit_results: List[AzimuthalFitResult] = []

    for sh in SHEETS:
        fit_result = fit_azimuthal_for_sheet(
            xlsx_path=XLSX_PATH,
            mean_sheet=sh,
            fan_center_xy=FAN_CENTER_XY,
            fourier_order=FOURIER_ORDER_N,
            sigma_fallback=SIGMA_FALLBACK,
            sigma_min=SIGMA_MIN,
        )
        fit_results.append(fit_result)

    save_azimuthal_fit_table(
        fit_results=fit_results,
        out_path=OUT_AZ_PARAMS_XLSX,
        sheet_name=OUT_AZ_PARAMS_SHEET,
        fourier_order=FOURIER_ORDER_N,
    )
    print(f"Saved azimuthal fit parameters to: {OUT_AZ_PARAMS_XLSX.resolve()}")

    param_names = build_param_column_names(FOURIER_ORDER_N)
    print("\nFitted non-axisymmetric annular-Gaussian parameters")
    print(" sheet   z [m]   RMSE      MAE     success")
    for fit_result in sorted(fit_results, key=lambda item: item.z_m):
        print(
            f" {fit_result.sheet_name:>5}  {fit_result.z_m:5.2f}  "
            f"{fit_result.rmse_mps:7.4f}  {fit_result.mae_mps:7.4f}  "
            f"{str(fit_result.success):>7}"
        )

    print("\nParameter order:")
    print(" " + ", ".join(param_names))


if __name__ == "__main__":
    main()
