"""
Fit a non-axisymmetric annular-Gaussian model for four-fan BEMT maps.

This script fits, at each measurement height z, the model:

    theta_f = atan2(y - y_f, x - x_f)
    r_f     = sqrt((x - x_f)^2 + (y - y_f)^2)
    g_f(r)  = exp(-((r_f - r_ring_f) / delta_ring_f)^2)

    A_f(theta_f) = a0_f + sum_{n=1..N_f}(a_n_f cos(n theta_f) + b_n_f sin(n theta_f))
    w_model(x, y) = sum_f [w0_f + g_f(r_f) * A_f(theta_f)]

All fan parameters are solved jointly in one optimization per height.
Each fan can use independent N_f, robust loss, robust f_scale, and
harmonic regularization hyper-parameters.
"""


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import four_fan_gp as sigma_gp
from four_fan_annuli_cut import (
    assign_sigma_samples_with_overlap as assign_sigma_samples_with_overlap_four,
    compute_nearest_fan_distances as compute_nearest_fan_distances_four,
    make_radial_profile as make_radial_profile_four,
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

XLSX_PATH = "S02.xlsx"
SHEETS = ["z020", "z035", "z050", "z075", "z110", "z160", "z220"]
ANNULI_PROFILE_DIR = Path("B_results/Four_Fan_Annuli_Profile")

OUT_AZ_PARAMS_XLSX = Path("B_results/four_annular_bemt_params.xlsx")
OUT_AZ_PARAMS_SHEET = "four_bemt_az_fit"
OUT_AZ_HYPER_SHEET = "four_bemt_hyper"

# Fan centres (x_f, y_f)
FOUR_FAN_CENTERS_XY = (
    (3.0, 3.6),
    (5.4, 3.6),
    (3.0, 1.2),
    (5.4, 1.2),
)

# Uncertainty assignment
SIGMA_FALLBACK = 0.2
SIGMA_MIN = 1e-3

# Zero masking is disabled unless zero-valued cells represent missing data.
MASK_ZEROS_AS_NODATA = False

# Non-axisymmetric model settings
FOURIER_ORDER_N = 1
ROBUST_LOSS = "soft_l1"
ROBUST_F_SCALE = 1.0

# Per-fan settings for one joint solve.
FAN_FOURIER_ORDER = (2, 2, 2, 2)
FAN_ROBUST_LOSS = ("soft_l1", "soft_l1", "soft_l1", "soft_l1")
FAN_ROBUST_F_SCALE = (1.0, 1.5, 1.0, 1.5)

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

# Tuned joint-fit regularization (selected by RMSE objective with guardrails).
FAN_HARMONIC_RIDGE_LAMBDA = (0.03, 0.03, 0.03, 0.03)
FAN_HARMONIC_REL_CAP_LAMBDA = (0.10, 0.10, 0.10, 0.10)
FAN_HARMONIC_REL_MAX_TO_A0 = (0.80, 0.80, 0.80, 0.80)
FAN_HARMONIC_ORDER_WEIGHT_EXP = (1.5, 1.5, 1.5, 1.5)

# Multi-start optimization
ENABLE_MULTI_START = False
MAX_NFEV = 8000
MULTI_START_R_RING_FRACTIONS = (0.35, 0.65)
MULTI_START_DELTA_SCALE = (0.75, 1.00, 1.35)
MULTI_START_A0_SCALE = (0.85, 1.00, 1.15)
MULTI_START_W0_SHIFT_FRACTION = 0.08

# Keep overlap/sigma mapping aligned with four_fan_annuli_cut.py.
PROFILE_DELTA_R_M = 0.30
PROFILE_USE_MEDIAN = False
OVERLAP_RATIO_THRESHOLD = 1.25
OVERLAP_WEIGHT_POWER = 2.0
OVERLAP_SIGMA_BOOST = 1.12

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


@dataclass(frozen=True)
class RawMetrics:
    """Aggregate raw-map metrics."""

    total_sae: float
    total_wrmse: float
    n_samples: int


@dataclass
class JointFitResult:
    """Container for one height joint-fit output."""

    sheet_name: str
    z_m: float
    params_by_fan: Tuple[np.ndarray, ...]
    n_samples: int
    rmse_mps: float
    mae_mps: float
    success: bool
    cost: float


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
    r_profile, w_profile, _sigma_profile = load_annuli_profile_csv(
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

    sigma_samples = sigma_gp.evaluate_sigma_points_annular_blend_pchip_z(
        xlsx_path=xlsx_path,
        sheet_names=tuple(SHEETS),
        fan_centers_xy=fan_centers_xy,
        x_pts=x_samples,
        y_pts=y_samples,
        z_pts=np.full_like(
            x_samples,
            parse_sheet_height_m(mean_sheet),
            dtype=float,
        ),
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


def validate_joint_fan_settings() -> None:
    """Validate per-fan joint settings."""
    n_fans = len(FOUR_FAN_CENTERS_XY)
    fields = (
        ("FAN_FOURIER_ORDER", FAN_FOURIER_ORDER),
        ("FAN_ROBUST_LOSS", FAN_ROBUST_LOSS),
        ("FAN_ROBUST_F_SCALE", FAN_ROBUST_F_SCALE),
        ("FAN_HARMONIC_RIDGE_LAMBDA", FAN_HARMONIC_RIDGE_LAMBDA),
        ("FAN_HARMONIC_REL_CAP_LAMBDA", FAN_HARMONIC_REL_CAP_LAMBDA),
        ("FAN_HARMONIC_REL_MAX_TO_A0", FAN_HARMONIC_REL_MAX_TO_A0),
        ("FAN_HARMONIC_ORDER_WEIGHT_EXP", FAN_HARMONIC_ORDER_WEIGHT_EXP),
    )
    for name, values in fields:
        if len(values) != n_fans:
            raise ValueError(f"{name} length={len(values)} must match n_fans={n_fans}.")

    for fan_idx in range(n_fans):
        if int(FAN_FOURIER_ORDER[fan_idx]) < 0:
            raise ValueError(f"FAN_FOURIER_ORDER must be >=0 for F{fan_idx + 1:02d}.")
        if float(FAN_ROBUST_F_SCALE[fan_idx]) <= 0.0:
            raise ValueError(f"FAN_ROBUST_F_SCALE must be positive for F{fan_idx + 1:02d}.")
        if str(FAN_ROBUST_LOSS[fan_idx]) not in {"linear", "soft_l1", "huber", "cauchy", "arctan"}:
            raise ValueError(f"Unsupported FAN_ROBUST_LOSS for F{fan_idx + 1:02d}.")


def robust_residual_transform(
    residual: np.ndarray,
    loss_name: str,
    f_scale: float,
) -> np.ndarray:
    """
    Transform raw residuals so least_squares(loss='linear') reproduces robust cost.
    """
    c = float(f_scale)
    r_scaled = residual / c
    z = r_scaled**2

    if loss_name == "linear":
        rho = z
    elif loss_name == "soft_l1":
        rho = 2.0 * (np.sqrt(1.0 + z) - 1.0)
    elif loss_name == "huber":
        rho = np.where(z <= 1.0, z, 2.0 * np.sqrt(z) - 1.0)
    elif loss_name == "cauchy":
        rho = np.log1p(z)
    elif loss_name == "arctan":
        rho = np.arctan(z)
    else:
        raise ValueError(f"Unsupported robust loss: {loss_name}")

    return c * np.sqrt(np.maximum(rho, 0.0))


def fan_hyper(fan_idx: int) -> FitHyperParams:
    """Build per-fan hyper-parameter bundle."""
    return FitHyperParams(
        name=f"F{fan_idx + 1:02d}",
        fourier_order=int(FAN_FOURIER_ORDER[fan_idx]),
        robust_loss=str(FAN_ROBUST_LOSS[fan_idx]),
        robust_f_scale=float(FAN_ROBUST_F_SCALE[fan_idx]),
        harmonic_ridge_lambda=float(FAN_HARMONIC_RIDGE_LAMBDA[fan_idx]),
        harmonic_rel_cap_lambda=float(FAN_HARMONIC_REL_CAP_LAMBDA[fan_idx]),
        harmonic_rel_max_to_a0=float(FAN_HARMONIC_REL_MAX_TO_A0[fan_idx]),
        harmonic_order_weight_exp=float(FAN_HARMONIC_ORDER_WEIGHT_EXP[fan_idx]),
    )


def prepare_joint_fit_data_for_height(sheet_name: str, z_m: float) -> dict:
    """Build per-fan fit data for one height, solved jointly."""
    x, y, w_map = read_slice_from_sheet(XLSX_PATH, sheet_name)
    if MASK_ZEROS_AS_NODATA:
        w_map = w_map.copy()
        w_map[w_map == 0.0] = np.nan

    x_grid, y_grid = np.meshgrid(x, y)
    x_pts = x_grid.ravel()
    y_pts = y_grid.ravel()
    w_obs = w_map.ravel()

    valid = np.isfinite(x_pts) & np.isfinite(y_pts) & np.isfinite(w_obs)
    x_pts = x_pts[valid]
    y_pts = y_pts[valid]
    w_obs = w_obs[valid]
    if x_pts.size == 0:
        raise ValueError(f"No valid raw samples found in sheet '{sheet_name}'.")

    nearest_idx, nearest_r, _second_idx, _second_r = compute_nearest_fan_distances_four(
        x_pts=x_pts,
        y_pts=y_pts,
        fan_centers_xy=FOUR_FAN_CENTERS_XY,
    )

    sigma_pts = sigma_gp.evaluate_sigma_points_annular_blend_pchip_z(
        xlsx_path=XLSX_PATH,
        sheet_names=tuple(SHEETS),
        fan_centers_xy=FOUR_FAN_CENTERS_XY,
        x_pts=x_pts,
        y_pts=y_pts,
        z_pts=np.full_like(x_pts, parse_sheet_height_m(sheet_name), dtype=float),
        sigma_fallback=float(SIGMA_FALLBACK),
        sigma_min=float(SIGMA_MIN),
    )
    sigma_safe = np.maximum(sigma_pts.astype(float), float(SIGMA_MIN))

    fan_xy = np.asarray(FOUR_FAN_CENTERS_XY, dtype=float)
    dx_all = x_pts[:, None] - fan_xy[None, :, 0]
    dy_all = y_pts[:, None] - fan_xy[None, :, 1]
    r_all = np.sqrt(dx_all**2 + dy_all**2)
    theta_all = np.arctan2(dy_all, dx_all)

    fan_data: List[dict] = []
    for fan_idx in range(len(FOUR_FAN_CENTERS_XY)):
        hyper = fan_hyper(fan_idx)
        fan_mask = nearest_idx == fan_idx
        if not np.any(fan_mask):
            raise ValueError(
                f"No samples assigned to fan F{fan_idx + 1:02d} in sheet '{sheet_name}'."
            )

        r_local = nearest_r[fan_mask]
        w_local = w_obs[fan_mask]
        r_profile, w_profile, _n_profile, _alpha_profile = make_radial_profile_four(
            r=r_local,
            w=w_local,
            delta_r=PROFILE_DELTA_R_M,
            use_median=PROFILE_USE_MEDIAN,
        )

        r_ring_min, r_ring_max = default_r_ring_bounds_by_z(z_m)
        bounds = AzimuthalBounds(
            r_ring_min=r_ring_min,
            r_ring_max=r_ring_max,
            delta_ring_min=DELTA_RING_MIN_M,
            delta_ring_max=DELTA_RING_MAX_M,
            coeff_abs_max=COEFF_ABS_MAX_MPS,
        )
        lower, upper = build_param_bounds(
            w_samples=w_local,
            bounds=bounds,
            fourier_order=hyper.fourier_order,
        )
        p0 = initial_guess_azimuthal(
            r_samples=r_local,
            w_samples=w_local,
            bounds=bounds,
            fourier_order=hyper.fourier_order,
            r_profile=r_profile,
            w_profile=w_profile,
        )
        p0 = np.clip(p0, lower + 1e-12, upper - 1e-12)
        seed_set = build_multi_start_candidates(
            p0=p0,
            lower=lower,
            upper=upper,
            bounds=bounds,
            w_samples=w_local,
            fourier_order=hyper.fourier_order,
        )

        print(
            f"[{sheet_name}][F{fan_idx + 1:02d}] samples={int(np.count_nonzero(fan_mask))}, "
            f"N={hyper.fourier_order}, loss={hyper.robust_loss}, f_scale={hyper.robust_f_scale}"
        )

        fan_data.append(
            {
                "hyper": hyper,
                "lower": lower,
                "upper": upper,
                "seed_set": seed_set,
            }
        )

    return {
        "x_pts": x_pts,
        "y_pts": y_pts,
        "w_obs": w_obs,
        "sigma_safe": sigma_safe,
        "nearest_idx": nearest_idx,
        "r_all": r_all,
        "theta_all": theta_all,
        "fan_data": fan_data,
    }


def fit_joint_azimuthal_at_height(fit_data: dict) -> Tuple[Tuple[np.ndarray, ...], bool, float]:
    """Solve one joint optimization at one height."""
    w_obs = fit_data["w_obs"]
    sigma_safe = fit_data["sigma_safe"]
    nearest_idx = fit_data["nearest_idx"]
    r_all = fit_data["r_all"]
    theta_all = fit_data["theta_all"]
    fan_data = fit_data["fan_data"]
    n_fans = len(fan_data)

    lower_parts = []
    upper_parts = []
    dims = []
    seed_lists = []
    for fan_idx in range(n_fans):
        lower_parts.append(fan_data[fan_idx]["lower"])
        upper_parts.append(fan_data[fan_idx]["upper"])
        dims.append(int(fan_data[fan_idx]["lower"].size))
        seed_lists.append(fan_data[fan_idx]["seed_set"])

    lower = np.concatenate(lower_parts)
    upper = np.concatenate(upper_parts)

    n_seed = max(len(v) for v in seed_lists)
    if not ENABLE_MULTI_START:
        n_seed = 1

    joint_seeds = []
    for seed_idx in range(n_seed):
        parts = []
        for fan_idx in range(n_fans):
            seeds = seed_lists[fan_idx]
            parts.append(seeds[seed_idx % len(seeds)])
        p0 = np.concatenate(parts)
        p0 = np.clip(p0, lower + 1e-12, upper - 1e-12)
        joint_seeds.append(p0)

    def residuals_joint(p: np.ndarray) -> np.ndarray:
        w_pred = np.zeros_like(w_obs, dtype=float)
        reg_terms: List[float] = []

        offset = 0
        for fan_idx in range(n_fans):
            dim = dims[fan_idx]
            p_f = p[offset : offset + dim]
            hyper = fan_data[fan_idx]["hyper"]
            w_pred += azimuthal_ring_model(
                r=r_all[:, fan_idx],
                theta=theta_all[:, fan_idx],
                params=p_f,
                fourier_order=hyper.fourier_order,
            )

            if ENABLE_HARMONIC_REGULARIZATION and hyper.fourier_order > 0:
                coeffs = p_f[3:]
                a0 = float(max(coeffs[0], HARMONIC_A0_FLOOR_MPS))
                ridge_weight = float(np.sqrt(max(hyper.harmonic_ridge_lambda, 0.0)))
                relcap_weight = float(np.sqrt(max(hyper.harmonic_rel_cap_lambda, 0.0)))
                scale = float(max(HARMONIC_PRIOR_SCALE_MPS, 1e-12))
                for n_idx in range(1, hyper.fourier_order + 1):
                    a_n = float(coeffs[2 * n_idx - 1])
                    b_n = float(coeffs[2 * n_idx])
                    order_weight = float(n_idx ** hyper.harmonic_order_weight_exp)
                    if ridge_weight > 0.0:
                        reg_terms.append(ridge_weight * order_weight * a_n / scale)
                        reg_terms.append(ridge_weight * order_weight * b_n / scale)
                    if relcap_weight > 0.0:
                        harm_mag = float(np.sqrt(a_n**2 + b_n**2))
                        cap = float(hyper.harmonic_rel_max_to_a0 * a0)
                        excess = max(harm_mag - cap, 0.0)
                        reg_terms.append(relcap_weight * order_weight * excess / scale)

            offset += dim

        raw_res_all = (w_pred - w_obs) / sigma_safe
        parts = []
        for fan_idx in range(n_fans):
            fan_mask = nearest_idx == fan_idx
            if not np.any(fan_mask):
                continue
            hyper = fan_data[fan_idx]["hyper"]
            parts.append(
                robust_residual_transform(
                    residual=raw_res_all[fan_mask],
                    loss_name=hyper.robust_loss,
                    f_scale=hyper.robust_f_scale,
                )
            )
        if reg_terms:
            parts.append(np.asarray(reg_terms, dtype=float))
        if not parts:
            return raw_res_all
        return np.concatenate(parts)

    best_result = None
    for p0 in joint_seeds:
        result = least_squares(
            residuals_joint,
            p0,
            bounds=(lower, upper),
            loss="linear",
            f_scale=1.0,
            max_nfev=MAX_NFEV,
        )
        if best_result is None or result.cost < best_result.cost:
            best_result = result

    if best_result is None:
        raise RuntimeError("Joint least_squares failed for all multi-start seeds.")

    params_by_fan = []
    offset = 0
    for dim in dims:
        params_by_fan.append(best_result.x[offset : offset + dim].astype(float))
        offset += dim

    return tuple(params_by_fan), bool(best_result.success), float(best_result.cost)


def evaluate_params_on_samples(
    params_by_fan: Sequence[np.ndarray],
    r_all: np.ndarray,
    theta_all: np.ndarray,
) -> np.ndarray:
    """Evaluate summed model on samples for all fans."""
    w_pred = np.zeros(r_all.shape[0], dtype=float)
    for fan_idx, params in enumerate(params_by_fan):
        hyper = fan_hyper(fan_idx)
        w_pred += azimuthal_ring_model(
            r=r_all[:, fan_idx],
            theta=theta_all[:, fan_idx],
            params=np.asarray(params, dtype=float),
            fourier_order=hyper.fourier_order,
        )
    return w_pred


def fit_all_heights_joint(sheet_names: Iterable[str]) -> List[JointFitResult]:
    """Fit all heights with one joint optimization per height."""
    fit_results: List[JointFitResult] = []
    for sheet_name in sheet_names:
        z_m = parse_sheet_height_m(sheet_name)
        fit_data = prepare_joint_fit_data_for_height(sheet_name=sheet_name, z_m=z_m)
        params_by_fan, success, cost = fit_joint_azimuthal_at_height(fit_data=fit_data)
        w_pred = evaluate_params_on_samples(
            params_by_fan=params_by_fan,
            r_all=fit_data["r_all"],
            theta_all=fit_data["theta_all"],
        )
        err = w_pred - fit_data["w_obs"]
        rmse = float(np.sqrt(np.mean(err**2)))
        mae = float(np.mean(np.abs(err)))
        fit_results.append(
            JointFitResult(
                sheet_name=sheet_name,
                z_m=z_m,
                params_by_fan=params_by_fan,
                n_samples=int(fit_data["w_obs"].size),
                rmse_mps=rmse,
                mae_mps=mae,
                success=success,
                cost=cost,
            )
        )

    fit_results.sort(key=lambda item: item.z_m)
    return fit_results


def evaluate_joint_fit_on_raw_maps(
    fit_results: Sequence[JointFitResult],
) -> Tuple[RawMetrics, List[RawMetrics]]:
    """
    Evaluate fitted joint model on raw maps with nearest-region diagnostics.
    """
    n_fans = len(FOUR_FAN_CENTERS_XY)
    by_sheet = {res.sheet_name: res for res in fit_results}

    sae_f = np.zeros(n_fans, dtype=float)
    wsse_f = np.zeros(n_fans, dtype=float)
    wsum_f = np.zeros(n_fans, dtype=float)
    nsamp_f = np.zeros(n_fans, dtype=int)

    fan_xy = np.asarray(FOUR_FAN_CENTERS_XY, dtype=float)
    for sheet_name in SHEETS:
        if sheet_name not in by_sheet:
            continue
        result = by_sheet[sheet_name]

        x, y, w_map = read_slice_from_sheet(XLSX_PATH, sheet_name)
        if MASK_ZEROS_AS_NODATA:
            w_map = w_map.copy()
            w_map[w_map == 0.0] = np.nan

        x_grid, y_grid = np.meshgrid(x, y)
        x_pts = x_grid.ravel()
        y_pts = y_grid.ravel()
        w_obs = w_map.ravel()
        valid = np.isfinite(x_pts) & np.isfinite(y_pts) & np.isfinite(w_obs)
        x_pts = x_pts[valid]
        y_pts = y_pts[valid]
        w_obs = w_obs[valid]
        if x_pts.size == 0:
            continue

        nearest_idx, _nearest_r, _second_idx, _second_r = compute_nearest_fan_distances_four(
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
            z_pts=np.full_like(x_pts, parse_sheet_height_m(sheet_name), dtype=float),
            sigma_fallback=float(SIGMA_FALLBACK),
            sigma_min=float(SIGMA_MIN),
        )
        sigma_safe = np.maximum(sigma_pts.astype(float), float(SIGMA_MIN))

        w_pred = evaluate_params_on_samples(
            params_by_fan=result.params_by_fan,
            r_all=r_all,
            theta_all=theta_all,
        )
        err = w_pred - w_obs
        weights = 1.0 / (sigma_safe**2)

        for fan_idx in range(n_fans):
            fan_mask = nearest_idx == fan_idx
            if not np.any(fan_mask):
                continue
            err_f = err[fan_mask]
            w_f = weights[fan_mask]
            sae_f[fan_idx] += float(np.sum(np.abs(err_f)))
            wsse_f[fan_idx] += float(np.sum(w_f * err_f**2))
            wsum_f[fan_idx] += float(np.sum(w_f))
            nsamp_f[fan_idx] += int(np.count_nonzero(fan_mask))

    total_sae = float(np.sum(sae_f))
    total_wsse = float(np.sum(wsse_f))
    total_wsum = float(np.sum(wsum_f))
    total_nsamp = int(np.sum(nsamp_f))
    overall = RawMetrics(
        total_sae=total_sae,
        total_wrmse=float(np.sqrt(total_wsse / total_wsum)) if total_wsum > 0.0 else float("nan"),
        n_samples=total_nsamp,
    )

    per_fan: List[RawMetrics] = []
    for fan_idx in range(n_fans):
        if wsum_f[fan_idx] <= 0.0:
            per_fan.append(
                RawMetrics(
                    total_sae=float(sae_f[fan_idx]),
                    total_wrmse=float("nan"),
                    n_samples=int(nsamp_f[fan_idx]),
                )
            )
        else:
            per_fan.append(
                RawMetrics(
                    total_sae=float(sae_f[fan_idx]),
                    total_wrmse=float(np.sqrt(wsse_f[fan_idx] / wsum_f[fan_idx])),
                    n_samples=int(nsamp_f[fan_idx]),
                )
            )

    return overall, per_fan


def write_joint_fit_tables(
    fit_results: Sequence[JointFitResult],
    per_fan_metrics: Sequence[RawMetrics],
) -> None:
    """Write fitted parameter table and per-fan hyper table."""
    n_fans = len(FOUR_FAN_CENTERS_XY)
    max_order = max(int(v) for v in FAN_FOURIER_ORDER)

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

        shared = {"w0": [], "r_ring": [], "delta_ring": [], "a0": []}
        harmonic_shared = {f"a{n_idx}": [] for n_idx in range(1, max_order + 1)}
        harmonic_shared.update({f"b{n_idx}": [] for n_idx in range(1, max_order + 1)})

        for fan_idx in range(n_fans):
            fan_id = f"F{fan_idx + 1:02d}"
            order = int(FAN_FOURIER_ORDER[fan_idx])
            params = np.asarray(fit_result.params_by_fan[fan_idx], dtype=float)

            row[f"w0_{fan_id}"] = float(params[0])
            row[f"r_ring_{fan_id}"] = float(params[1])
            row[f"delta_ring_{fan_id}"] = float(params[2])
            row[f"a0_{fan_id}"] = float(params[3])

            shared["w0"].append(float(params[0]))
            shared["r_ring"].append(float(params[1]))
            shared["delta_ring"].append(float(params[2]))
            shared["a0"].append(float(params[3]))

            for n_idx in range(1, order + 1):
                a_val = float(params[3 + 2 * n_idx - 1])
                b_val = float(params[3 + 2 * n_idx])
                row[f"a{n_idx}_{fan_id}"] = a_val
                row[f"b{n_idx}_{fan_id}"] = b_val
                harmonic_shared[f"a{n_idx}"].append(a_val)
                harmonic_shared[f"b{n_idx}"].append(b_val)

        row["w0"] = float(np.mean(shared["w0"]))
        row["r_ring"] = float(np.mean(shared["r_ring"]))
        row["delta_ring"] = float(np.mean(shared["delta_ring"]))
        row["a0"] = float(np.mean(shared["a0"]))
        for n_idx in range(1, max_order + 1):
            a_key = f"a{n_idx}"
            b_key = f"b{n_idx}"
            row[a_key] = float(np.mean(harmonic_shared[a_key])) if harmonic_shared[a_key] else 0.0
            row[b_key] = float(np.mean(harmonic_shared[b_key])) if harmonic_shared[b_key] else 0.0

        rows.append(row)

    df_out = pd.DataFrame(rows).sort_values("z_m").reset_index(drop=True)

    hyper_rows = []
    for fan_idx in range(n_fans):
        fan_id = f"F{fan_idx + 1:02d}"
        cx, cy = FOUR_FAN_CENTERS_XY[fan_idx]
        fan_metrics = per_fan_metrics[fan_idx]
        hyper_rows.append(
            {
                "fan_id": fan_id,
                "center_x_m": float(cx),
                "center_y_m": float(cy),
                "fourier_order": int(FAN_FOURIER_ORDER[fan_idx]),
                "solver_loss": str(FAN_ROBUST_LOSS[fan_idx]),
                "solver_f_scale": float(FAN_ROBUST_F_SCALE[fan_idx]),
                "harmonic_ridge_lambda": float(FAN_HARMONIC_RIDGE_LAMBDA[fan_idx]),
                "harmonic_rel_cap_lambda": float(FAN_HARMONIC_REL_CAP_LAMBDA[fan_idx]),
                "harmonic_rel_max_to_a0": float(FAN_HARMONIC_REL_MAX_TO_A0[fan_idx]),
                "harmonic_order_weight_exp": float(FAN_HARMONIC_ORDER_WEIGHT_EXP[fan_idx]),
                "raw_wrmse_mps": float(fan_metrics.total_wrmse),
                "raw_sae_mps": float(fan_metrics.total_sae),
                "raw_n_samples": int(fan_metrics.n_samples),
                "profile_delta_r_m": float(PROFILE_DELTA_R_M),
                "overlap_ratio_threshold": float(OVERLAP_RATIO_THRESHOLD),
                "overlap_weight_power": float(OVERLAP_WEIGHT_POWER),
                "overlap_sigma_boost": float(OVERLAP_SIGMA_BOOST),
            }
        )
    df_hyper = pd.DataFrame(hyper_rows)

    OUT_AZ_PARAMS_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_AZ_PARAMS_XLSX, engine="openpyxl", mode="w") as writer:
        df_out.to_excel(writer, index=False, sheet_name=OUT_AZ_PARAMS_SHEET)
        df_hyper.to_excel(writer, index=False, sheet_name=OUT_AZ_HYPER_SHEET)


# =============================================================================
# 4) Fitting Export Entry Point
# =============================================================================

def main() -> None:
    validate_joint_fan_settings()

    fit_results = fit_all_heights_joint(SHEETS)
    overall_metrics, per_fan_metrics = evaluate_joint_fit_on_raw_maps(fit_results)

    n_fans = len(FOUR_FAN_CENTERS_XY)
    print("\nFitted per-fan non-axisymmetric annular-Gaussian parameters")
    print(
        f"Joint solve: n_fans={n_fans}, "
        f"total_params_per_height={sum(4 + 2 * int(n) for n in FAN_FOURIER_ORDER)}"
    )
    for fan_idx in range(n_fans):
        print(
            f"F{fan_idx + 1:02d}: "
            f"N={FAN_FOURIER_ORDER[fan_idx]}, "
            f"loss={FAN_ROBUST_LOSS[fan_idx]}, "
            f"f_scale={FAN_ROBUST_F_SCALE[fan_idx]}"
        )

    print(
        f"\nOverall overlap-model raw-map metrics: WRMSE={overall_metrics.total_wrmse:.5f}, "
        f"SAE={overall_metrics.total_sae:.4f}, n_samples={overall_metrics.n_samples}"
    )
    for fan_idx in range(n_fans):
        fan_metrics = per_fan_metrics[fan_idx]
        print(
            f"Region near F{fan_idx + 1:02d}: "
            f"WRMSE={fan_metrics.total_wrmse:.5f}, "
            f"SAE={fan_metrics.total_sae:.4f}, n_samples={fan_metrics.n_samples}"
        )

    print("\nPer-height training diagnostics")
    print(" sheet   z [m]   RMSE      MAE     success")
    for fit_result in fit_results:
        print(
            f" {fit_result.sheet_name:>5}  {fit_result.z_m:5.2f}  "
            f"{fit_result.rmse_mps:7.4f}  {fit_result.mae_mps:7.4f}  "
            f"{str(fit_result.success):>7}"
        )

    write_joint_fit_tables(
        fit_results=fit_results,
        per_fan_metrics=per_fan_metrics,
    )
    print(f"\nSaved azimuthal fit parameters to: {OUT_AZ_PARAMS_XLSX.resolve()}")


if __name__ == "__main__":
    main()
