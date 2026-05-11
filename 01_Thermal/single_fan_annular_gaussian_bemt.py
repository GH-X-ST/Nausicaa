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


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import single_fan_gp as sigma_gp
from single_fan_annuli_cut import (
    assign_sigma_points_linear_nearest,
    parse_ts_xy_points_and_sigmas,
)


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Fitting Configuration and Data Sources
# 2) Data Containers
# 3) Data Loading and Fit Utilities
# 4) Fitting Export Entry Point
# =============================================================================

# =============================================================================
# 1) Fitting Configuration and Data Sources
# =============================================================================

XLSX_PATH = "S01.xlsx"
SHEETS = ["z020", "z035", "z050", "z075", "z110", "z160", "z220"]
ANNULI_PROFILE_DIR = Path("B_results/Single_Fan_Annuli_Profile")

OUT_AZ_PARAMS_XLSX = Path("B_results/single_annular_bemt_params.xlsx")
OUT_AZ_PARAMS_SHEET = "single_bemt_az_fit"

# Fan centre (x_c, y_c) in arena metres.
FAN_CENTER_XY = (4.2, 2.4)

# Uncertainty assignment
SIGMA_FALLBACK = 0.2
SIGMA_MIN = 1e-3

# Zero masking is disabled unless zero-valued cells represent missing data.
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

# Multi-start optimization
ENABLE_MULTI_START = True
MAX_NFEV = 8000
MULTI_START_R_RING_FRACTIONS = (0.35, 0.65)
MULTI_START_DELTA_SCALE = (0.75, 1.00, 1.35)
MULTI_START_A0_SCALE = (0.85, 1.00, 1.15)
MULTI_START_W0_SHIFT_FRACTION = 0.08

# Auto-tune settings
ENABLE_HYPERPARAM_AUTOTUNE = True
AUTO_TUNE_MAE_GUARDRAIL_FRAC = 0.03
AUTO_TUNE_MAX_RMSE_GUARDRAIL_FRAC = 0.10
AUTO_TUNE_RMSE_TIE_TOL = 1.0e-4

# Candidate format:
# (name, fourier_order, robust_loss, robust_f_scale,
#  harmonic_ridge_lambda, harmonic_rel_cap_lambda,
#  harmonic_rel_max_to_a0, harmonic_order_weight_exp)
AUTO_TUNE_CANDIDATES = (
    ("baseline", 1, "soft_l1", 1.0, 0.02, 0.10, 0.80, 1.0),
    ("order2_soft_rmse", 2, "soft_l1", 1.5, 0.05, 0.20, 0.70, 1.5),
    ("order2_soft_bal", 2, "soft_l1", 1.0, 0.05, 0.20, 0.70, 1.5),
    ("order2_huber_mae", 2, "huber", 0.5, 0.05, 0.05, 0.70, 1.5),
    ("order2_soft_cons", 2, "soft_l1", 1.5, 0.03, 0.10, 0.80, 1.5),
)

# z020 -> 0.20 m, z110 -> 1.10 m, etc.
SHEET_HEIGHT_DIVISOR = 100.0


# =============================================================================
# 2) Data Containers
# =============================================================================

@dataclass(frozen=True)
class AzimuthalBounds:
    """Bounds for non-axisymmetric annular-Gaussian fitting."""

    r_ring_min: float
    r_ring_max: float
    delta_ring_min: float = DELTA_RING_MIN_M
    delta_ring_max: float = DELTA_RING_MAX_M
    coeff_abs_max: float = COEFF_ABS_MAX_MPS


@dataclass(frozen=True)
class FitHyperParams:
    """Hyper-parameters controlling one full BEMT fitting run."""

    name: str
    fourier_order: int
    robust_loss: str
    robust_f_scale: float
    harmonic_ridge_lambda: float
    harmonic_rel_cap_lambda: float
    harmonic_rel_max_to_a0: float
    harmonic_order_weight_exp: float


@dataclass(frozen=True)
class FitSummary:
    """Global metrics for one fitted model over all heights."""

    mean_rmse: float
    mean_mae: float
    max_rmse: float
    total_samples: int


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


@dataclass
class HyperTuneTrial:
    """Container for one auto-tune trial."""

    hyper: FitHyperParams
    fit_results: List[AzimuthalFitResult]
    summary: FitSummary


# =============================================================================
# 3) Data Loading and Fit Utilities
# =============================================================================

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

    # Workbook grid stores x coordinates in the first row after the corner cell.
    x = pd.to_numeric(raw.iloc[0, 1:], errors="coerce").to_numpy(dtype=float)

    # Workbook grid stores y coordinates in the first column after the corner cell.
    y = pd.to_numeric(raw.iloc[1:, 0], errors="coerce").to_numpy(dtype=float)

    # Measured vertical-velocity block (m/s).
    w_map = raw.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    # Workbook grid shape must match y-by-x coordinates.
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


def build_multi_start_candidates(
    p0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    bounds: AzimuthalBounds,
    w_samples: np.ndarray,
    fourier_order: int,
) -> List[np.ndarray]:
    """Build seed vectors for multi-start fitting."""
    p0 = np.asarray(p0, dtype=float)
    seeds = [np.clip(p0, lower + 1e-12, upper - 1e-12)]

    if not ENABLE_MULTI_START:
        return seeds

    r_span = float(bounds.r_ring_max - bounds.r_ring_min)
    r_seeds = [float(p0[1])]
    for frac in MULTI_START_R_RING_FRACTIONS:
        r_seeds.append(float(bounds.r_ring_min + frac * r_span))

    delta_seeds = [
        float(np.clip(p0[2] * scale, bounds.delta_ring_min, bounds.delta_ring_max))
        for scale in MULTI_START_DELTA_SCALE
    ]

    w_span = float(np.percentile(w_samples, 95.0) - np.percentile(w_samples, 5.0))
    w_shift = float(max(0.02, MULTI_START_W0_SHIFT_FRACTION * max(w_span, 1e-6)))
    w0_seeds = [float(p0[0] - w_shift), float(p0[0]), float(p0[0] + w_shift)]

    a0_seeds = [float(max(0.0, p0[3] * scale)) for scale in MULTI_START_A0_SCALE]

    for r_seed in r_seeds:
        for d_seed in delta_seeds:
            for w0_seed in w0_seeds:
                for a0_seed in a0_seeds:
                    cand = np.array(p0, dtype=float)
                    cand[0] = w0_seed
                    cand[1] = r_seed
                    cand[2] = d_seed
                    cand[3] = a0_seed
                    if fourier_order > 0:
                        cand[4:] = 0.0
                    cand = np.clip(cand, lower + 1e-12, upper - 1e-12)
                    seeds.append(cand)

    unique = {}
    for seed in seeds:
        key = tuple(np.round(seed, 10).tolist())
        unique[key] = seed
    return list(unique.values())


def fit_azimuthal_model(
    x_samples: np.ndarray,
    y_samples: np.ndarray,
    w_samples: np.ndarray,
    sigma_samples: np.ndarray,
    fan_center_xy: Tuple[float, float],
    z_m: float,
    fit_hyper: FitHyperParams,
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
        fourier_order=fit_hyper.fourier_order,
    )
    p0 = initial_guess_azimuthal(
        r_samples=r_samples,
        w_samples=w_samples,
        bounds=bounds,
        fourier_order=fit_hyper.fourier_order,
        r_profile=r_profile,
        w_profile=w_profile,
    )
    p0 = np.clip(p0, lower + 1e-12, upper - 1e-12)
    p0_candidates = build_multi_start_candidates(
        p0=p0,
        lower=lower,
        upper=upper,
        bounds=bounds,
        w_samples=w_samples,
        fourier_order=fit_hyper.fourier_order,
    )

    sigma_safe = np.maximum(sigma_samples.astype(float), float(SIGMA_MIN))

    def residuals(p: np.ndarray) -> np.ndarray:
        w_pred = azimuthal_ring_model(
            r=r_samples,
            theta=theta_samples,
            params=p,
            fourier_order=fit_hyper.fourier_order,
        )
        data_residual = (w_pred - w_samples) / sigma_safe

        if not ENABLE_HARMONIC_REGULARIZATION or fit_hyper.fourier_order <= 0:
            return data_residual

        coeffs = p[3:]
        a0 = float(max(coeffs[0], HARMONIC_A0_FLOOR_MPS))
        reg_terms = []
        ridge_weight = float(np.sqrt(max(fit_hyper.harmonic_ridge_lambda, 0.0)))
        relcap_weight = float(np.sqrt(max(fit_hyper.harmonic_rel_cap_lambda, 0.0)))
        scale = float(max(HARMONIC_PRIOR_SCALE_MPS, 1e-12))

        for n_idx in range(1, fit_hyper.fourier_order + 1):
            a_n = float(coeffs[2 * n_idx - 1])
            b_n = float(coeffs[2 * n_idx])
            order_weight = float(n_idx ** fit_hyper.harmonic_order_weight_exp)

            if ridge_weight > 0.0:
                reg_terms.append(ridge_weight * order_weight * a_n / scale)
                reg_terms.append(ridge_weight * order_weight * b_n / scale)

            if relcap_weight > 0.0:
                harm_mag = float(np.sqrt(a_n**2 + b_n**2))
                cap = float(fit_hyper.harmonic_rel_max_to_a0 * a0)
                excess = max(harm_mag - cap, 0.0)
                reg_terms.append(relcap_weight * order_weight * excess / scale)

        if not reg_terms:
            return data_residual

        return np.concatenate([data_residual, np.asarray(reg_terms, dtype=float)])

    best_result = None
    for p0_i in p0_candidates:
        result = least_squares(
            residuals,
            p0_i,
            bounds=(lower, upper),
            loss=fit_hyper.robust_loss,
            f_scale=fit_hyper.robust_f_scale,
            max_nfev=MAX_NFEV,
        )
        if best_result is None or result.cost < best_result.cost:
            best_result = result

    if best_result is None:
        raise RuntimeError("least_squares failed for all multi-start seeds.")

    return best_result.x.astype(float), bool(best_result.success), float(best_result.cost)


def fit_azimuthal_for_sheet(
    xlsx_path: str,
    mean_sheet: str,
    fan_center_xy: Tuple[float, float],
    fit_hyper: FitHyperParams,
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
        sigma_samples = sigma_gp.evaluate_sigma_points_annular_pchip_z(
            xlsx_path=xlsx_path,
            sheet_names=tuple(SHEETS),
            fan_center_xy=fan_center_xy,
            x_pts=x_samples,
            y_pts=y_samples,
            z_pts=np.full_like(x_samples, parse_sheet_height_m(mean_sheet), dtype=float),
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
        fit_hyper=fit_hyper,
        r_profile=r_profile,
        w_profile=w_profile,
    )

    theta_samples = np.arctan2(y_samples - yc, x_samples - xc)
    w_pred = azimuthal_ring_model(
        r=r_samples,
        theta=theta_samples,
        params=params,
        fourier_order=fit_hyper.fourier_order,
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


def summarize_fit_results(fit_results: Sequence[AzimuthalFitResult]) -> FitSummary:
    """Compute global summary metrics across all fitted heights."""
    if len(fit_results) == 0:
        raise ValueError("No fit results to summarize.")

    rmses = np.array([res.rmse_mps for res in fit_results], dtype=float)
    maes = np.array([res.mae_mps for res in fit_results], dtype=float)
    samples = np.array([res.n_samples for res in fit_results], dtype=float)

    total_samples = int(np.sum(samples))
    if total_samples <= 0:
        raise ValueError("Total sample count must be positive.")

    mean_rmse = float(np.sum(rmses * samples) / np.sum(samples))
    mean_mae = float(np.sum(maes * samples) / np.sum(samples))
    max_rmse = float(np.max(rmses))

    return FitSummary(
        mean_rmse=mean_rmse,
        mean_mae=mean_mae,
        max_rmse=max_rmse,
        total_samples=total_samples,
    )


def build_fit_hyper_from_globals() -> FitHyperParams:
    """Capture baseline globals into a FitHyperParams object."""
    return FitHyperParams(
        name="baseline",
        fourier_order=int(FOURIER_ORDER_N),
        robust_loss=str(ROBUST_LOSS),
        robust_f_scale=float(ROBUST_F_SCALE),
        harmonic_ridge_lambda=float(HARMONIC_RIDGE_LAMBDA),
        harmonic_rel_cap_lambda=float(HARMONIC_REL_CAP_LAMBDA),
        harmonic_rel_max_to_a0=float(HARMONIC_REL_MAX_TO_A0),
        harmonic_order_weight_exp=float(HARMONIC_ORDER_WEIGHT_EXP),
    )


def build_fit_hyper_candidates() -> List[FitHyperParams]:
    """Build auto-tune hyper-parameter candidates."""
    candidates = []
    for row in AUTO_TUNE_CANDIDATES:
        (
            name,
            fourier_order,
            robust_loss,
            robust_f_scale,
            harmonic_ridge_lambda,
            harmonic_rel_cap_lambda,
            harmonic_rel_max_to_a0,
            harmonic_order_weight_exp,
        ) = row
        candidates.append(
            FitHyperParams(
                name=str(name),
                fourier_order=int(fourier_order),
                robust_loss=str(robust_loss),
                robust_f_scale=float(robust_f_scale),
                harmonic_ridge_lambda=float(harmonic_ridge_lambda),
                harmonic_rel_cap_lambda=float(harmonic_rel_cap_lambda),
                harmonic_rel_max_to_a0=float(harmonic_rel_max_to_a0),
                harmonic_order_weight_exp=float(harmonic_order_weight_exp),
            )
        )
    if len(candidates) == 0:
        raise ValueError("AUTO_TUNE_CANDIDATES must contain at least one candidate.")
    return candidates


def fit_all_sheets(
    fit_hyper: FitHyperParams,
    sheet_names: Iterable[str],
) -> List[AzimuthalFitResult]:
    """Fit all configured sheets with one hyper-parameter set."""
    fit_results: List[AzimuthalFitResult] = []
    for sheet in sheet_names:
        fit_result = fit_azimuthal_for_sheet(
            xlsx_path=XLSX_PATH,
            mean_sheet=sheet,
            fan_center_xy=FAN_CENTER_XY,
            fit_hyper=fit_hyper,
            sigma_fallback=SIGMA_FALLBACK,
            sigma_min=SIGMA_MIN,
        )
        fit_results.append(fit_result)
    return fit_results


def autotune_fit_hyper(
    sheet_names: Iterable[str],
) -> Tuple[FitHyperParams, List[AzimuthalFitResult], List[HyperTuneTrial]]:
    """
    Auto-select fit hyper-parameters with RMSE objective and guardrails.
    """
    candidates = build_fit_hyper_candidates()
    trials: List[HyperTuneTrial] = []

    for hyper in candidates:
        fit_results = fit_all_sheets(hyper, sheet_names=sheet_names)
        summary = summarize_fit_results(fit_results)
        trials.append(HyperTuneTrial(hyper=hyper, fit_results=fit_results, summary=summary))

    baseline = trials[0]
    mae_limit = baseline.summary.mean_mae * (1.0 + float(AUTO_TUNE_MAE_GUARDRAIL_FRAC))
    max_rmse_limit = baseline.summary.max_rmse * (1.0 + float(AUTO_TUNE_MAX_RMSE_GUARDRAIL_FRAC))

    eligible = [
        trial
        for trial in trials
        if trial.summary.mean_mae <= mae_limit and trial.summary.max_rmse <= max_rmse_limit
    ]
    if len(eligible) == 0:
        eligible = trials

    min_rmse = min(trial.summary.mean_rmse for trial in eligible)
    finalists = [
        trial
        for trial in eligible
        if trial.summary.mean_rmse <= min_rmse + float(AUTO_TUNE_RMSE_TIE_TOL)
    ]
    selected = min(finalists, key=lambda trial: trial.summary.mean_mae)

    print("Auto-tuning BEMT candidates:")
    print(" name               N  loss     f_scale  mean_RMSE  mean_MAE   max_RMSE")
    for trial in trials:
        marker = "*" if trial.hyper.name == selected.hyper.name else " "
        h = trial.hyper
        s = trial.summary
        print(
            f"{marker}{h.name:17s}  {h.fourier_order:1d}  {h.robust_loss:7s}  "
            f"{h.robust_f_scale:7.3f}  {s.mean_rmse:9.5f}  {s.mean_mae:9.5f}  {s.max_rmse:9.5f}"
        )
    print(
        f"Selected: {selected.hyper.name} "
        f"(MAE guardrail <= {mae_limit:.5f}, max-RMSE guardrail <= {max_rmse_limit:.5f})"
    )

    return selected.hyper, selected.fit_results, trials


# =============================================================================
# 4) Fitting Export Entry Point
# =============================================================================

def main() -> None:
    if ENABLE_HYPERPARAM_AUTOTUNE:
        selected_hyper, fit_results, _trials = autotune_fit_hyper(SHEETS)
    else:
        selected_hyper = build_fit_hyper_from_globals()
        fit_results = fit_all_sheets(selected_hyper, sheet_names=SHEETS)

    save_azimuthal_fit_table(
        fit_results=fit_results,
        out_path=OUT_AZ_PARAMS_XLSX,
        sheet_name=OUT_AZ_PARAMS_SHEET,
        fourier_order=selected_hyper.fourier_order,
    )
    print(f"Saved azimuthal fit parameters to: {OUT_AZ_PARAMS_XLSX.resolve()}")

    summary = summarize_fit_results(fit_results)
    param_names = build_param_column_names(selected_hyper.fourier_order)
    print("\nFitted non-axisymmetric annular-Gaussian parameters")
    print(
        f"Selected hyper: N={selected_hyper.fourier_order}, loss={selected_hyper.robust_loss}, "
        f"f_scale={selected_hyper.robust_f_scale}, ridge={selected_hyper.harmonic_ridge_lambda}, "
        f"relcap={selected_hyper.harmonic_rel_cap_lambda}, relmax={selected_hyper.harmonic_rel_max_to_a0}, "
        f"order_exp={selected_hyper.harmonic_order_weight_exp}"
    )
    print(" sheet   z [m]   RMSE      MAE     success")
    for fit_result in sorted(fit_results, key=lambda item: item.z_m):
        print(
            f" {fit_result.sheet_name:>5}  {fit_result.z_m:5.2f}  "
            f"{fit_result.rmse_mps:7.4f}  {fit_result.mae_mps:7.4f}  "
            f"{str(fit_result.success):>7}"
        )

    print(
        f"\nGlobal summary: mean_RMSE={summary.mean_rmse:.6f}, "
        f"mean_MAE={summary.mean_mae:.6f}, max_RMSE={summary.max_rmse:.6f}"
    )
    print("\nParameter order:")
    print(" " + ", ".join(param_names))


if __name__ == "__main__":
    main()
