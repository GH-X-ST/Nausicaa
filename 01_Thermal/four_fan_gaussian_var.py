"""
Plain-Gaussian identification for four-fan updraft maps in S02.xlsx.

This script fits an axisymmetric Gaussian model to each measured height
sheet and writes per-height parameters to Excel.

Model (per height z_k):
    w(r; z_k) = w0(z_k) + A(z_k) * exp(-(r / delta(z_k))**2)
"""


from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.optimize import least_squares
from scipy.spatial import QhullError
import four_fan_gp as sigma_gp


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Fitting Configuration and Data Sources
# 2) Data Containers
# 3) Data Loading and Fit Utilities
# 4) Core Model Evaluation
# 5) Fitting and Diagnostics
# 6) Fitting Export Entry Point
# =============================================================================

# =============================================================================
# 1) Fitting Configuration and Data Sources
# =============================================================================

XLSX_PATH = "S02.xlsx"
ANNULI_PROFILE_DIR = Path("B_results/Four_Fan_Annuli_Profile")
OUT_XLSX_PATH = Path("B_results/four_var_params.xlsx")

FOUR_FAN_CENTERS_XY = (
    (3.0, 3.6),
    (5.4, 3.6),
    (3.0, 1.2),
    (5.4, 1.2),
)

# z020 -> 0.20 m, z110 -> 1.10 m, etc.
SHEET_HEIGHT_DIVISOR = 100.0

# Fitting sheets in ascending z.
MEAN_SHEETS = ("z020", "z035", "z050", "z075", "z110", "z160", "z220")

# Candidate robust solvers for automatic fine-tuning.
SOLVER_CANDIDATES = (
    ("baseline", "soft_l1", 1.0),
    ("huber", "huber", 1.0),
    ("cauchy", "cauchy", 0.5),
)

# If WRMSE values are within this tolerance, prefer lower SAE.
WRMSE_TIE_TOLERANCE = 1.0e-4

# Plane-dependent w0 bounds follow the BEMT-style rule:
#   w0_min = min(w_plane) - W0_MARGIN_MPS
#   w0_max = max(w_plane) + W0_MARGIN_MPS
W0_MARGIN_MPS = 1.0

# Per-fan radial-profile settings (aligned with four_fan_annuli_cut.py style).
PROFILE_DELTA_R_M = 0.30
PROFILE_USE_MEDIAN = False

# Overlap-aware sigma blending (aligned with four_fan_annuli_cut.py).
OVERLAP_RATIO_THRESHOLD = 1.10
OVERLAP_WEIGHT_POWER = 3.0
OVERLAP_SIGMA_BOOST = 1.00
# Per-fan robust settings for one-shot joint optimization (12 parameters).
# Tuned by exhaustive candidate search with SAE guardrail on 2026-02-18.
FAN_ROBUST_LOSS = ("soft_l1", "soft_l1", "soft_l1", "soft_l1")
FAN_ROBUST_F_SCALE = (1.0, 1.0, 1.0, 1.0)


# =============================================================================
# 2) Data Containers
# =============================================================================

@dataclass(frozen=True)
class FitConfig:
    """Configuration for loading, fitting, and auto-tuning."""

    xlsx_path: str = XLSX_PATH
    annuli_profile_dir: Path = ANNULI_PROFILE_DIR
    fan_centers_xy: Tuple[Tuple[float, float], ...] = FOUR_FAN_CENTERS_XY

    # Robust least-squares options for scipy.optimize.least_squares.
    robust_loss: str = "soft_l1"
    robust_f_scale: float = 1.0

    # Per-fan robust settings used in the joint (12-parameter) optimization.
    fan_robust_loss: Tuple[str, ...] = FAN_ROBUST_LOSS
    fan_robust_f_scale: Tuple[float, ...] = FAN_ROBUST_F_SCALE

    # If TS parsing fails, fallback sigma_z (m/s).
    sigma_z_fallback: float = 0.2
    sigma_min: float = 1e-3

    # Per-fan radial profile construction.
    profile_delta_r_m: float = PROFILE_DELTA_R_M
    profile_use_median: bool = PROFILE_USE_MEDIAN

    # Residual weighting by annulus counts n_j.
    use_count_weighting: bool = True

    # Solver behaviour.
    max_nfev: int = 5000
    multi_start_enabled: bool = True
    delta_seed_values: Tuple[float, ...] = (0.10, 0.20, 0.35, 0.50)

    # Ignore potentially unrepresentative core samples in annular-profile fitting.
    # inner_bins_to_ignore removes the smallest-radius bins (typically includes r=0).
    # core_ignore_radius_m additionally removes bins with r <= this value when set.
    inner_bins_to_ignore: int = 1
    core_ignore_radius_m: Optional[float] = None

    # Auto-tuning behaviour (not used in joint one-shot fitting).
    enable_solver_autotune: bool = False
    sae_guardrail_fraction: float = 0.02

    # Overlap-aware uncertainty mapping for raw-map validation.
    overlap_ratio_threshold: float = OVERLAP_RATIO_THRESHOLD
    overlap_weight_power: float = OVERLAP_WEIGHT_POWER
    overlap_sigma_boost: float = OVERLAP_SIGMA_BOOST


@dataclass(frozen=True)
class GaussianBounds:
    """Parameter bounds to keep fits identifiable and physically plausible."""

    delta_min: float = 0.08
    delta_max: float = 1.20
    a_max: float = 50.0
    w0_min: float = -1.0e-4
    w0_max: float = 1.0e-4


@dataclass(frozen=True)
class SolverCandidate:
    """One candidate solver configuration for auto-tuning."""

    name: str
    robust_loss: str
    robust_f_scale: float


@dataclass(frozen=True)
class RawMetrics:
    """Aggregate raw-map error metrics used for solver selection."""

    total_sae: float
    total_wrmse: float
    n_samples: int


@dataclass
class SolverTrial:
    """Container for one auto-tuning trial."""

    candidate: SolverCandidate
    z_vals: np.ndarray
    params: np.ndarray
    metrics: RawMetrics


@dataclass
class SmoothGaussianModel:
    """
    Smooth model w(x, y, z) using PCHIP interpolators over z.

    A(z) and delta(z) are interpolated in log-space to ensure positivity.
    """

    a_log: PchipInterpolator
    delta_log: PchipInterpolator
    w0: PchipInterpolator
    fan_centers_xy: Tuple[Tuple[float, float], ...]

    def __call__(self, x: np.ndarray, y: np.ndarray, z: float) -> np.ndarray:
        a = float(np.exp(self.a_log(z)))
        delta = float(np.exp(self.delta_log(z)))
        w0 = float(self.w0(z))
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        w = np.zeros_like(x_arr, dtype=float)
        for cx, cy in self.fan_centers_xy:
            r = np.sqrt((x_arr - cx) ** 2 + (y_arr - cy) ** 2)
            w += plain_gaussian(r, a=a, delta=delta, w0=w0)
        return w


# =============================================================================
# 3) Data Loading and Fit Utilities
# =============================================================================

def parse_sheet_height_m(sheet_name: str) -> float:
    """
    Parse height in meters from sheet names like 'z020', 'z110', 'z220'.
    """
    if not sheet_name.startswith("z"):
        raise ValueError(f"Invalid sheet name (expected 'z###'): {sheet_name}")

    suffix = sheet_name[1:]
    if not suffix.isdigit():
        raise ValueError(f"Invalid height code in sheet name: {sheet_name}")

    return int(suffix) / SHEET_HEIGHT_DIVISOR


def load_mean_map(
    xlsx_path: str,
    sheet_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a mean velocity map sheet (e.g., 'z020') as X, Y, W arrays.

    Returns:
        X: meshgrid of x-coordinates (m), shape (ny, nx)
        Y: meshgrid of y-coordinates (m), shape (ny, nx)
        W: mean vertical velocity (m/s), shape (ny, nx)
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)

    if df.shape[0] < 2 or df.shape[1] < 2:
        raise ValueError(f"Sheet '{sheet_name}' is too small to contain a grid.")

    xs = pd.to_numeric(df.iloc[0, 1:], errors="coerce").to_numpy(dtype=float)
    ys = pd.to_numeric(df.iloc[1:, 0], errors="coerce").to_numpy(dtype=float)

    w_block = df.iloc[1:, 1:]
    w_vals = pd.to_numeric(w_block.stack(), errors="coerce").unstack().to_numpy(dtype=float)

    if np.any(~np.isfinite(xs)) or np.any(~np.isfinite(ys)) or np.any(~np.isfinite(w_vals)):
        raise ValueError(f"Non-numeric values detected in mean-map sheet '{sheet_name}'.")

    x_grid, y_grid = np.meshgrid(xs, ys)

    if w_vals.shape != x_grid.shape:
        raise ValueError(
            f"Velocity block shape {w_vals.shape} does not match coordinate grid "
            f"shape {x_grid.shape} in sheet '{sheet_name}'."
        )

    return x_grid, y_grid, w_vals


def parse_ts_noise_scale(xlsx_path: str, ts_sheet_name: str) -> Optional[float]:
    """
    Estimate sigma_z (m/s) from a *_TS sheet.

    Strategy:
    - Search for header cells exactly equal to 'variance' (case-insensitive).
    - For each, take the first numeric value below the header in that column.
    - Compute sigma_z = sqrt(mean(variances)).

    Returns None if no variances are found.
    """
    try:
        df = pd.read_excel(xlsx_path, sheet_name=ts_sheet_name, header=None)
    except Exception:
        return None

    positions = []
    for r_idx in range(df.shape[0]):
        for c_idx in range(df.shape[1]):
            val = df.iat[r_idx, c_idx]
            if isinstance(val, str) and val.strip().lower() == "variance":
                positions.append((r_idx, c_idx))

    if not positions:
        return None

    variances = []
    for r_idx, c_idx in positions:
        col = pd.to_numeric(df.iloc[r_idx + 1 :, c_idx], errors="coerce").to_numpy(dtype=float)
        col = col[np.isfinite(col)]
        if col.size > 0:
            variances.append(float(col[0]))

    if not variances:
        return None

    mean_var = float(np.mean(variances))
    if mean_var <= 0.0:
        return None

    return float(np.sqrt(mean_var))


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

    x_points: List[float] = []
    y_points: List[float] = []
    sigma_points: List[float] = []

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
    """Assign sigma_j by nearest-radius mapping."""
    if r_points.size == 0 or sigma_points.size == 0:
        sigma = np.full_like(r_bins, float(sigma_fallback), dtype=float)
        return np.maximum(sigma, float(sigma_min))

    diffs = np.abs(r_bins[:, None] - r_points[None, :])
    idx = np.argmin(diffs, axis=1)
    sigma = sigma_points[idx].astype(float)
    return np.maximum(sigma, float(sigma_min))


def make_radial_profile_local(
    r: np.ndarray,
    w: np.ndarray,
    delta_r: float,
    use_median: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct a 1D radial profile from one fan's local samples."""
    if delta_r <= 0.0:
        raise ValueError("profile_delta_r_m must be positive.")

    mask = np.isfinite(r) & np.isfinite(w)
    r = r[mask]
    w = w[mask]
    if r.size == 0:
        raise ValueError("No finite samples available to construct fan-local profile.")

    k = np.floor(r / delta_r + 0.5).astype(int)
    uniq_k = np.unique(k)

    r_list: List[float] = []
    w_list: List[float] = []
    n_list: List[int] = []
    for kk in uniq_k:
        in_bin = k == kk
        w_b = w[in_bin]
        if w_b.size == 0:
            continue
        r_list.append(float(kk) * float(delta_r))
        w_list.append(float(np.median(w_b) if use_median else np.mean(w_b)))
        n_list.append(int(w_b.size))

    r_bins = np.asarray(r_list, dtype=float)
    w_bins = np.asarray(w_list, dtype=float)
    n_bins = np.asarray(n_list, dtype=int)

    order = np.argsort(r_bins)
    return r_bins[order], w_bins[order], n_bins[order]


def aggregate_sigma_to_annuli_local(
    r_samples: np.ndarray,
    sigma_samples: np.ndarray,
    delta_r: float,
    r_bins: np.ndarray,
    sigma_fallback: float,
    sigma_min: float,
) -> np.ndarray:
    """Aggregate sample-level sigma to fan-local annulus centers."""
    mask = np.isfinite(r_samples) & np.isfinite(sigma_samples)
    if not np.any(mask):
        sigma_bins = np.full_like(r_bins, float(sigma_fallback), dtype=float)
        return np.maximum(sigma_bins, float(sigma_min))

    r_valid = r_samples[mask]
    sigma_valid = sigma_samples[mask]
    k = np.floor(r_valid / delta_r + 0.5).astype(int)
    uniq_k = np.unique(k)

    r_points: List[float] = []
    sigma_points: List[float] = []
    for kk in uniq_k:
        in_bin = k == kk
        sigma_b = sigma_valid[in_bin]
        if sigma_b.size == 0:
            continue
        r_points.append(float(kk) * float(delta_r))
        sigma_points.append(float(np.mean(sigma_b)))

    if len(r_points) == 0:
        sigma_bins = np.full_like(r_bins, float(sigma_fallback), dtype=float)
        return np.maximum(sigma_bins, float(sigma_min))

    return assign_sigma_bins_nearest(
        r_bins=r_bins,
        r_points=np.asarray(r_points, dtype=float),
        sigma_points=np.asarray(sigma_points, dtype=float),
        sigma_fallback=sigma_fallback,
        sigma_min=sigma_min,
    )


def nearest_fan_radius(
    x: np.ndarray,
    y: np.ndarray,
    fan_centers_xy: Sequence[Tuple[float, float]],
) -> np.ndarray:
    """Compute nearest-fan radius for each sample location."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    fan_xy = np.asarray(fan_centers_xy, dtype=float)

    d_sample_fan = np.sqrt(
        (x_arr[..., None] - fan_xy[:, 0]) ** 2
        + (y_arr[..., None] - fan_xy[:, 1]) ** 2
    )
    return np.min(d_sample_fan, axis=-1)


def nearest_and_second_fan_distances(
    x_pts: np.ndarray,
    y_pts: np.ndarray,
    fan_centers_xy: Sequence[Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute nearest/second-nearest fan indices and radii for each sample."""
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


def assign_sigma_points_with_overlap(
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
    """Assign per-point sigma from local *_TS representative points in x-y."""
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


def load_annuli_profile_csv(
    profile_dir: Path,
    sheet_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load annuli profile CSV produced by annuli_cut.py.

    Returns:
        r_bins: annulus radii
        w_bins: annulus mean velocities
        n_bins: annulus counts
        sigma_bins: optional per-annulus sigma_j from sigma_mps column
    """
    csv_path = Path(profile_dir) / f"{sheet_name}_four_annuli_profile.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing annuli profile CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    if "r_m" not in df.columns or "w_mps" not in df.columns or "n" not in df.columns:
        raise ValueError(
            f"CSV {csv_path} must contain columns: r_m, w_mps, n."
        )

    r_bins = pd.to_numeric(df["r_m"], errors="coerce").to_numpy(dtype=float)
    w_bins = pd.to_numeric(df["w_mps"], errors="coerce").to_numpy(dtype=float)
    n_bins = pd.to_numeric(df["n"], errors="coerce").to_numpy(dtype=float)
    sigma_bins = None
    if "sigma_mps" in df.columns:
        sigma_bins = pd.to_numeric(df["sigma_mps"], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(r_bins) & np.isfinite(w_bins) & np.isfinite(n_bins)
    r_bins = r_bins[mask]
    w_bins = w_bins[mask]
    n_bins = n_bins[mask].astype(int)
    if sigma_bins is not None:
        sigma_bins = sigma_bins[mask]

    if r_bins.size == 0:
        raise ValueError(f"No valid annuli data in {csv_path}.")

    order = np.argsort(r_bins)
    r_bins = r_bins[order]
    w_bins = w_bins[order]
    n_bins = n_bins[order]
    if sigma_bins is not None:
        sigma_bins = sigma_bins[order]

    return r_bins, w_bins, n_bins, sigma_bins


# =============================================================================
# 4) Core Model Evaluation
# =============================================================================

def plain_gaussian(r: np.ndarray, a: float, delta: float, w0: float) -> np.ndarray:
    """Evaluate w(r) = w0 + A * exp(-(r / delta)^2)."""
    return w0 + a * np.exp(-((r / delta) ** 2))


def build_smooth_gaussian_model(
    z_vals: np.ndarray,
    params: np.ndarray,
    fan_centers_xy: Tuple[Tuple[float, float], ...],
) -> SmoothGaussianModel:
    """Build PCHIP-smoothed parameter model over z."""
    eps = 1e-12
    a_log = PchipInterpolator(z_vals, np.log(np.maximum(params[:, 0], eps)))
    delta_log = PchipInterpolator(z_vals, np.log(np.maximum(params[:, 1], eps)))
    w0 = PchipInterpolator(z_vals, params[:, 2])

    return SmoothGaussianModel(
        a_log=a_log,
        delta_log=delta_log,
        w0=w0,
        fan_centers_xy=fan_centers_xy,
    )


# =============================================================================
# 5) Fitting and Diagnostics
# =============================================================================


def w0_bounds_from_plane(
    w_bins: np.ndarray,
    margin_mps: float = W0_MARGIN_MPS,
) -> Tuple[float, float]:
    """
    Build plane-dependent w0 bounds from observed annular means.
    """
    w_min = float(np.min(w_bins))
    w_max = float(np.max(w_bins))
    margin = float(margin_mps)
    return w_min - margin, w_max + margin


def initial_guess_gaussian(
    r_bins: np.ndarray,
    w_bins: np.ndarray,
    delta_bounds: Tuple[float, float],
) -> np.ndarray:
    """
    Baseline initial guess for [A, delta, w0].
    """
    tail_len = max(5, int(0.2 * w_bins.size))
    w0 = float(np.median(w_bins[-tail_len:]))

    peak_idx = int(np.argmax(w_bins))
    a0 = float(max(w_bins[peak_idx] - w0, 0.1))

    target = w0 + a0 / np.e
    below = np.where(w_bins <= target)[0]
    if below.size > 0:
        delta0 = float(r_bins[int(below[0])])
    else:
        delta0 = 0.30 * float(np.max(r_bins))
    delta0 = float(np.clip(delta0, delta_bounds[0], delta_bounds[1]))

    return np.array([a0, delta0, w0], dtype=float)


def build_initial_guess_set(
    r_bins: np.ndarray,
    w_bins: np.ndarray,
    bounds: GaussianBounds,
    config: FitConfig,
) -> List[np.ndarray]:
    """Build one or many starting points for multi-start fitting."""
    base = initial_guess_gaussian(r_bins, w_bins, (bounds.delta_min, bounds.delta_max))
    lower = np.array([0.0, bounds.delta_min, bounds.w0_min], dtype=float)
    upper = np.array([bounds.a_max, bounds.delta_max, bounds.w0_max], dtype=float)

    if not config.multi_start_enabled:
        return [np.clip(base, lower + 1e-12, upper - 1e-12)]

    peak_idx = int(np.argmax(w_bins))
    delta_mid = 0.5 * (bounds.delta_min + bounds.delta_max)
    delta_lo = bounds.delta_min + 0.30 * (bounds.delta_max - bounds.delta_min)
    delta_hi = bounds.delta_min + 0.70 * (bounds.delta_max - bounds.delta_min)
    delta_seeds = np.array(
        [base[1], *config.delta_seed_values, delta_mid, delta_lo, delta_hi],
        dtype=float,
    )
    delta_seeds = np.clip(delta_seeds, bounds.delta_min, bounds.delta_max)

    w0_seeds = np.array(
        [
            base[2],
            0.5 * (bounds.w0_min + bounds.w0_max),
            bounds.w0_min + 0.25 * (bounds.w0_max - bounds.w0_min),
            bounds.w0_max - 0.25 * (bounds.w0_max - bounds.w0_min),
        ],
        dtype=float,
    )
    w0_seeds = np.clip(w0_seeds, bounds.w0_min, bounds.w0_max)

    seeds = []
    for delta0 in delta_seeds:
        for w00 in w0_seeds:
            a0 = float(max(w_bins[peak_idx] - w00, 0.05))
            seed = np.array([a0, float(delta0), float(w00)], dtype=float)
            seed = np.clip(seed, lower + 1e-12, upper - 1e-12)
            seeds.append(seed)

    unique = {}
    for seed in seeds:
        key = tuple(np.round(seed, 10).tolist())
        unique[key] = seed

    return list(unique.values())


def build_fit_mask(
    r_bins: np.ndarray,
    config: FitConfig,
) -> np.ndarray:
    """
    Build boolean mask for bins used in fitting (core bins can be ignored).
    """
    if r_bins.ndim != 1:
        raise ValueError("r_bins must be 1D for mask construction.")

    mask = np.ones(r_bins.size, dtype=bool)

    n_ignore = int(max(config.inner_bins_to_ignore, 0))
    if n_ignore > 0:
        if n_ignore >= r_bins.size:
            raise ValueError(
                f"inner_bins_to_ignore={n_ignore} removes all bins (n={r_bins.size})."
            )
        mask[:n_ignore] = False

    if config.core_ignore_radius_m is not None:
        r_cut = float(config.core_ignore_radius_m)
        if r_cut < 0.0:
            raise ValueError("core_ignore_radius_m must be non-negative when provided.")
        mask &= (r_bins > r_cut)

    if int(np.count_nonzero(mask)) < 3:
        raise ValueError(
            "Core-ignored mask leaves fewer than 3 points for 3-parameter fitting."
        )

    return mask


def fit_gaussian_at_height(
    r_bins: np.ndarray,
    w_bins: np.ndarray,
    n_bins: np.ndarray,
    sigma_bins: np.ndarray,
    bounds: GaussianBounds,
    config: FitConfig,
) -> np.ndarray:
    """
    Fit parameters [A, delta, w0] at a single height.
    """
    if r_bins.shape != w_bins.shape:
        raise ValueError("r_bins must have the same shape as w_bins.")
    if n_bins.shape != w_bins.shape:
        raise ValueError("n_bins must have the same shape as w_bins.")
    if sigma_bins.shape != w_bins.shape:
        raise ValueError("sigma_bins must have the same shape as w_bins.")
    if np.any(~np.isfinite(sigma_bins)):
        raise ValueError("sigma_bins contains non-finite values.")
    if np.any(sigma_bins <= 0.0):
        raise ValueError("sigma_bins must be strictly positive.")

    fit_mask = build_fit_mask(r_bins=r_bins, config=config)
    r_fit = r_bins[fit_mask]
    w_fit = w_bins[fit_mask]
    n_fit = n_bins[fit_mask]
    sigma_fit = sigma_bins[fit_mask]

    lower = np.array([0.0, bounds.delta_min, bounds.w0_min], dtype=float)
    upper = np.array([bounds.a_max, bounds.delta_max, bounds.w0_max], dtype=float)

    alpha = (
        np.sqrt(n_fit / np.max(n_fit))
        if config.use_count_weighting
        else np.ones_like(n_fit, dtype=float)
    )
    sigma_safe = np.maximum(sigma_fit.astype(float), 1e-12)

    def residuals(p: np.ndarray) -> np.ndarray:
        w_pred = plain_gaussian(r_fit, a=p[0], delta=p[1], w0=p[2])
        return alpha * (w_pred - w_fit) / sigma_safe

    seeds = build_initial_guess_set(
        r_bins=r_fit,
        w_bins=w_fit,
        bounds=bounds,
        config=config,
    )

    best_result = None
    for p0 in seeds:
        result = least_squares(
            residuals,
            p0,
            bounds=(lower, upper),
            loss=config.robust_loss,
            f_scale=config.robust_f_scale,
            max_nfev=config.max_nfev,
        )
        if best_result is None or result.cost < best_result.cost:
            best_result = result

    if best_result is None:
        raise RuntimeError("least_squares failed for all multi-start seeds.")

    return best_result.x


def validate_joint_fan_settings(config: FitConfig) -> None:
    """Validate per-fan robust settings for joint optimization."""
    n_fans = len(config.fan_centers_xy)
    if len(config.fan_robust_loss) != n_fans:
        raise ValueError(
            f"fan_robust_loss length={len(config.fan_robust_loss)} must match n_fans={n_fans}."
        )
    if len(config.fan_robust_f_scale) != n_fans:
        raise ValueError(
            "fan_robust_f_scale length="
            f"{len(config.fan_robust_f_scale)} must match n_fans={n_fans}."
        )
    for idx, (loss_name, f_scale) in enumerate(
        zip(config.fan_robust_loss, config.fan_robust_f_scale)
    ):
        if loss_name not in {"linear", "soft_l1", "huber", "cauchy", "arctan"}:
            raise ValueError(
                f"Unsupported loss '{loss_name}' for fan F{idx + 1:02d}."
            )
        if float(f_scale) <= 0.0:
            raise ValueError(f"fan_robust_f_scale must be positive for fan F{idx + 1:02d}.")


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


def prepare_fan_fit_data_for_height(
    sheet_name: str,
    config: FitConfig,
) -> dict:
    """
    Build per-fan annular data for one height, to be solved jointly.
    """
    x_grid, y_grid, w_grid = load_mean_map(config.xlsx_path, sheet_name)
    x_pts = x_grid.ravel()
    y_pts = y_grid.ravel()
    w_pts = w_grid.ravel()
    valid = np.isfinite(x_pts) & np.isfinite(y_pts) & np.isfinite(w_pts)
    x_pts = x_pts[valid]
    y_pts = y_pts[valid]
    w_pts = w_pts[valid]

    if x_pts.size == 0:
        raise ValueError(f"No valid raw samples found in sheet '{sheet_name}'.")

    nearest_idx, nearest_r, _second_idx, _second_r = nearest_and_second_fan_distances(
        x_pts=x_pts,
        y_pts=y_pts,
        fan_centers_xy=config.fan_centers_xy,
    )

    sigma_pts = sigma_gp.evaluate_sigma_points_annular_blend_pchip_z(
        xlsx_path=config.xlsx_path,
        sheet_names=MEAN_SHEETS,
        fan_centers_xy=config.fan_centers_xy,
        x_pts=x_pts,
        y_pts=y_pts,
        z_pts=np.full_like(x_pts, parse_sheet_height_m(sheet_name), dtype=float),
        sigma_fallback=float(config.sigma_z_fallback),
        sigma_min=float(config.sigma_min),
    )

    fan_data: List[dict] = []
    for fan_idx in range(len(config.fan_centers_xy)):
        fan_mask = nearest_idx == fan_idx
        if not np.any(fan_mask):
            raise ValueError(
                f"No samples assigned to fan F{fan_idx + 1:02d} in sheet '{sheet_name}'."
            )

        r_local = nearest_r[fan_mask]
        w_local = w_pts[fan_mask]
        sigma_local = sigma_pts[fan_mask]

        r_bins, w_bins, n_bins = make_radial_profile_local(
            r=r_local,
            w=w_local,
            delta_r=config.profile_delta_r_m,
            use_median=config.profile_use_median,
        )
        sigma_bins = aggregate_sigma_to_annuli_local(
            r_samples=r_local,
            sigma_samples=sigma_local,
            delta_r=config.profile_delta_r_m,
            r_bins=r_bins,
            sigma_fallback=float(config.sigma_z_fallback),
            sigma_min=float(config.sigma_min),
        )

        fit_mask = build_fit_mask(r_bins=r_bins, config=config)
        n_fit = int(np.count_nonzero(fit_mask))
        print(
            f"[{sheet_name}][F{fan_idx + 1:02d}] fitting bins used: {n_fit}/{r_bins.size} "
            f"(inner_bins_to_ignore={config.inner_bins_to_ignore}, "
            f"core_ignore_radius_m={config.core_ignore_radius_m})"
        )

        r_fit = r_bins[fit_mask]
        w_fit = w_bins[fit_mask]
        n_fit_bins = n_bins[fit_mask]
        sigma_fit = np.maximum(sigma_bins[fit_mask], float(config.sigma_min))
        alpha_fit = (
            np.sqrt(n_fit_bins / np.max(n_fit_bins))
            if config.use_count_weighting
            else np.ones_like(n_fit_bins, dtype=float)
        )

        w0_min, w0_max = w0_bounds_from_plane(w_bins, margin_mps=W0_MARGIN_MPS)
        bounds = GaussianBounds(
            delta_min=0.08,
            delta_max=1.20,
            a_max=max(50.0, 3.0 * float(np.max(w_bins)) + 1.0),
            w0_min=float(w0_min),
            w0_max=float(w0_max),
        )
        seed_set = build_initial_guess_set(
            r_bins=r_fit,
            w_bins=w_fit,
            bounds=bounds,
            config=config,
        )

        fan_data.append(
            {
                "r_fit": r_fit,
                "w_fit": w_fit,
                "alpha_fit": alpha_fit,
                "sigma_fit": sigma_fit,
                "bounds": bounds,
                "seed_set": seed_set,
            }
        )

    return {
        "fan_data": fan_data,
        "x_pts": x_pts,
        "y_pts": y_pts,
        "w_obs": w_pts,
        "sigma_safe": np.maximum(sigma_pts, float(config.sigma_min)),
        "nearest_idx": nearest_idx,
    }


def fit_joint_gaussian_at_height(
    fit_data: dict,
    config: FitConfig,
) -> np.ndarray:
    """
    Jointly fit all fan parameters for one height (n_fans * 3 variables).
    """
    fan_data = fit_data["fan_data"]
    x_pts = fit_data["x_pts"]
    y_pts = fit_data["y_pts"]
    w_obs = fit_data["w_obs"]
    sigma_safe = fit_data["sigma_safe"]
    nearest_idx = fit_data["nearest_idx"]

    n_fans = len(fan_data)
    fan_xy = np.asarray(config.fan_centers_xy, dtype=float)
    r_all = np.sqrt(
        (x_pts[:, None] - fan_xy[None, :, 0]) ** 2
        + (y_pts[:, None] - fan_xy[None, :, 1]) ** 2
    )

    lower_parts = []
    upper_parts = []
    seed_lists: List[List[np.ndarray]] = []
    for fan_idx in range(n_fans):
        bounds = fan_data[fan_idx]["bounds"]
        lower_parts.append(np.array([0.0, bounds.delta_min, bounds.w0_min], dtype=float))
        upper_parts.append(np.array([bounds.a_max, bounds.delta_max, bounds.w0_max], dtype=float))
        seed_lists.append(fan_data[fan_idx]["seed_set"])

    lower = np.concatenate(lower_parts)
    upper = np.concatenate(upper_parts)

    n_seed = max(len(seeds) for seeds in seed_lists)
    if not config.multi_start_enabled:
        n_seed = 1
    joint_seeds: List[np.ndarray] = []
    for s_idx in range(n_seed):
        parts = []
        for fan_idx in range(n_fans):
            seeds = seed_lists[fan_idx]
            seed = seeds[s_idx % len(seeds)]
            parts.append(seed)
        p0 = np.concatenate(parts)
        p0 = np.clip(p0, lower + 1e-12, upper - 1e-12)
        joint_seeds.append(p0)

    def residuals_joint(p: np.ndarray) -> np.ndarray:
        w_pred = np.zeros_like(w_obs, dtype=float)
        for fan_idx in range(n_fans):
            p_f = p[3 * fan_idx : 3 * fan_idx + 3]
            w_pred += plain_gaussian(
                r_all[:, fan_idx],
                a=float(p_f[0]),
                delta=float(p_f[1]),
                w0=float(p_f[2]),
            )

        raw_res_all = (w_pred - w_obs) / sigma_safe
        parts = []
        for fan_idx in range(n_fans):
            fan_mask = nearest_idx == fan_idx
            if not np.any(fan_mask):
                continue
            transformed = robust_residual_transform(
                residual=raw_res_all[fan_mask],
                loss_name=config.fan_robust_loss[fan_idx],
                f_scale=config.fan_robust_f_scale[fan_idx],
            )
            parts.append(transformed)

        if len(parts) == 0:
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
            max_nfev=config.max_nfev,
        )
        if best_result is None or result.cost < best_result.cost:
            best_result = result

    if best_result is None:
        raise RuntimeError("Joint least_squares failed for all multi-start seeds.")

    return best_result.x.reshape(n_fans, 3)


def fit_all_heights_joint(
    mean_sheet_names: Iterable[str],
    config: FitConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit all heights with one joint optimization per height (n_fans * 3 parameters).

    Returns:
        z_vals: shape (K,), heights in meters
        params_stack: shape (K, n_fans, 3), [A, delta, w0] per fan
    """
    z_list: List[float] = []
    params_list: List[np.ndarray] = []

    for sheet in mean_sheet_names:
        z_m = parse_sheet_height_m(sheet)
        fit_data = prepare_fan_fit_data_for_height(sheet_name=sheet, config=config)
        params_joint = fit_joint_gaussian_at_height(fit_data=fit_data, config=config)
        z_list.append(z_m)
        params_list.append(params_joint)

    z_vals = np.asarray(z_list, dtype=float)
    params_stack = np.stack(params_list, axis=0)
    order = np.argsort(z_vals)
    return z_vals[order], params_stack[order]


def params_by_fan_at_z(
    z_vals: np.ndarray,
    params_stack: np.ndarray,
    z_query: float,
) -> np.ndarray:
    """Interpolate per-fan parameters [A, delta, w0] at one z."""
    if np.any(np.diff(z_vals) <= 0.0):
        raise ValueError("z_vals must be strictly increasing.")

    n_fans = params_stack.shape[1]
    out = np.empty((n_fans, 3), dtype=float)
    for fan_idx in range(n_fans):
        for col_idx in range(3):
            out[fan_idx, col_idx] = float(
                np.interp(z_query, z_vals, params_stack[:, fan_idx, col_idx])
            )
    return out


def evaluate_joint_fit_on_raw_maps(
    mean_sheet_names: Iterable[str],
    z_vals: np.ndarray,
    params_stack: np.ndarray,
    config: FitConfig,
) -> Tuple[RawMetrics, List[RawMetrics]]:
    """
    Evaluate the overlap-model on raw maps.

    Returns:
        overall: global metric over all points using the overlap prediction.
        per_fan: nearest-region diagnostics from the same overlap prediction.
    """
    n_fans = len(config.fan_centers_xy)
    sae_f = np.zeros(n_fans, dtype=float)
    wsse_f = np.zeros(n_fans, dtype=float)
    wsum_f = np.zeros(n_fans, dtype=float)
    nsamp_f = np.zeros(n_fans, dtype=int)

    for sheet in mean_sheet_names:
        z_m = parse_sheet_height_m(sheet)
        p_fan = params_by_fan_at_z(z_vals=z_vals, params_stack=params_stack, z_query=z_m)

        x_grid, y_grid, w_grid = load_mean_map(config.xlsx_path, sheet)
        x_pts = x_grid.ravel()
        y_pts = y_grid.ravel()
        w_obs = w_grid.ravel()
        valid = np.isfinite(x_pts) & np.isfinite(y_pts) & np.isfinite(w_obs)
        x_pts = x_pts[valid]
        y_pts = y_pts[valid]
        w_obs = w_obs[valid]

        nearest_idx, _nearest_r, _second_idx, _second_r = nearest_and_second_fan_distances(
            x_pts=x_pts,
            y_pts=y_pts,
            fan_centers_xy=config.fan_centers_xy,
        )
        fan_xy = np.asarray(config.fan_centers_xy, dtype=float)
        r_all = np.sqrt(
            (x_pts[:, None] - fan_xy[None, :, 0]) ** 2
            + (y_pts[:, None] - fan_xy[None, :, 1]) ** 2
        )

        sigma_pts = sigma_gp.evaluate_sigma_points_annular_blend_pchip_z(
            xlsx_path=config.xlsx_path,
            sheet_names=MEAN_SHEETS,
            fan_centers_xy=config.fan_centers_xy,
            x_pts=x_pts,
            y_pts=y_pts,
            z_pts=np.full_like(x_pts, parse_sheet_height_m(sheet), dtype=float),
            sigma_fallback=float(config.sigma_z_fallback),
            sigma_min=float(config.sigma_min),
        )
        sigma_safe = np.maximum(sigma_pts, float(config.sigma_min))

        w_pred = np.zeros_like(w_obs, dtype=float)
        for fan_idx in range(n_fans):
            pf = p_fan[fan_idx]
            w_pred += plain_gaussian(
                r_all[:, fan_idx],
                a=float(pf[0]),
                delta=float(pf[1]),
                w0=float(pf[2]),
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
    if total_wsum <= 0.0:
        raise ValueError("Total WRMSE denominator is non-positive.")

    overall = RawMetrics(
        total_sae=total_sae,
        total_wrmse=float(np.sqrt(total_wsse / total_wsum)),
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


def fit_all_heights_for_fan(
    mean_sheet_names: Iterable[str],
    config: FitConfig,
    fan_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit plain Gaussian parameters for one specific fan across all heights.

    Returns:
        z_vals: shape (K,), heights in meters
        params: shape (K, 3), fitted parameters [A, delta, w0]
    """
    z_list: List[float] = []
    params_list: List[np.ndarray] = []

    for sheet in mean_sheet_names:
        z_m = parse_sheet_height_m(sheet)
        ts_sheet = f"{sheet}_TS"

        x_grid, y_grid, w_grid = load_mean_map(config.xlsx_path, sheet)
        x_pts = x_grid.ravel()
        y_pts = y_grid.ravel()
        w_pts = w_grid.ravel()
        valid = np.isfinite(x_pts) & np.isfinite(y_pts) & np.isfinite(w_pts)
        x_pts = x_pts[valid]
        y_pts = y_pts[valid]
        w_pts = w_pts[valid]

        if x_pts.size == 0:
            raise ValueError(f"No valid raw samples found in sheet '{sheet}'.")

        nearest_idx, nearest_r, _second_idx, _second_r = nearest_and_second_fan_distances(
            x_pts=x_pts,
            y_pts=y_pts,
            fan_centers_xy=config.fan_centers_xy,
        )
        fan_mask = nearest_idx == int(fan_idx)
        if not np.any(fan_mask):
            raise ValueError(
                f"No samples are assigned to fan F{fan_idx + 1:02d} in sheet '{sheet}'."
            )

        sigma_pts = sigma_gp.evaluate_sigma_points_annular_blend_pchip_z(
            xlsx_path=config.xlsx_path,
            sheet_names=MEAN_SHEETS,
            fan_centers_xy=config.fan_centers_xy,
            x_pts=x_pts,
            y_pts=y_pts,
            z_pts=np.full_like(x_pts, parse_sheet_height_m(sheet), dtype=float),
            sigma_fallback=float(config.sigma_z_fallback),
            sigma_min=float(config.sigma_min),
        )

        r_local = nearest_r[fan_mask]
        w_local = w_pts[fan_mask]
        sigma_local = sigma_pts[fan_mask]

        r_bins, w_bins, n_bins = make_radial_profile_local(
            r=r_local,
            w=w_local,
            delta_r=config.profile_delta_r_m,
            use_median=config.profile_use_median,
        )
        sigma_bins = aggregate_sigma_to_annuli_local(
            r_samples=r_local,
            sigma_samples=sigma_local,
            delta_r=config.profile_delta_r_m,
            r_bins=r_bins,
            sigma_fallback=float(config.sigma_z_fallback),
            sigma_min=float(config.sigma_min),
        )

        fit_mask = build_fit_mask(r_bins=r_bins, config=config)
        n_fit = int(np.count_nonzero(fit_mask))
        print(
            f"[{sheet}][F{fan_idx + 1:02d}] fitting bins used: {n_fit}/{r_bins.size} "
            f"(inner_bins_to_ignore={config.inner_bins_to_ignore}, "
            f"core_ignore_radius_m={config.core_ignore_radius_m})"
        )

        w0_min, w0_max = w0_bounds_from_plane(w_bins, margin_mps=W0_MARGIN_MPS)
        bounds = GaussianBounds(
            delta_min=0.08,
            delta_max=1.20,
            a_max=max(50.0, 3.0 * float(np.max(w_bins)) + 1.0),
            w0_min=float(w0_min),
            w0_max=float(w0_max),
        )

        p = fit_gaussian_at_height(
            r_bins=r_bins,
            w_bins=w_bins,
            n_bins=n_bins,
            sigma_bins=sigma_bins,
            bounds=bounds,
            config=config,
        )

        z_list.append(z_m)
        params_list.append(p)

    z_vals = np.asarray(z_list, dtype=float)
    params = np.vstack(params_list)
    order = np.argsort(z_vals)
    return z_vals[order], params[order]


def evaluate_fit_on_raw_maps_for_fan(
    mean_sheet_names: Iterable[str],
    z_vals: np.ndarray,
    params: np.ndarray,
    config: FitConfig,
    fan_idx: int,
) -> RawMetrics:
    """
    Evaluate fitted parameters for one fan on raw points assigned to that fan.
    """
    total_sae = 0.0
    total_weighted_sse = 0.0
    total_weight_sum = 0.0
    total_samples = 0

    for sheet in mean_sheet_names:
        z_m = parse_sheet_height_m(sheet)
        p = params_at_z(z_vals=z_vals, params=params, z_query=z_m)

        x_grid, y_grid, w_grid = load_mean_map(config.xlsx_path, sheet)
        x_pts = x_grid.ravel()
        y_pts = y_grid.ravel()
        w_obs = w_grid.ravel()
        valid = np.isfinite(x_pts) & np.isfinite(y_pts) & np.isfinite(w_obs)
        x_pts = x_pts[valid]
        y_pts = y_pts[valid]
        w_obs = w_obs[valid]

        if x_pts.size == 0:
            raise ValueError(f"No valid raw samples found in sheet '{sheet}'.")

        nearest_idx, nearest_r, _second_idx, _second_r = nearest_and_second_fan_distances(
            x_pts=x_pts,
            y_pts=y_pts,
            fan_centers_xy=config.fan_centers_xy,
        )
        fan_mask = nearest_idx == int(fan_idx)
        if not np.any(fan_mask):
            continue

        r_local = nearest_r[fan_mask]
        w_local = w_obs[fan_mask]

        sigma_all = sigma_gp.evaluate_sigma_points_annular_blend_pchip_z(
            xlsx_path=config.xlsx_path,
            sheet_names=MEAN_SHEETS,
            fan_centers_xy=config.fan_centers_xy,
            x_pts=x_pts,
            y_pts=y_pts,
            z_pts=np.full_like(x_pts, parse_sheet_height_m(sheet), dtype=float),
            sigma_fallback=float(config.sigma_z_fallback),
            sigma_min=float(config.sigma_min),
        )
        sigma_local = sigma_all[fan_mask]

        sigma_safe = np.maximum(sigma_local, float(config.sigma_min))
        w_pred = plain_gaussian(
            r_local,
            a=float(p[0]),
            delta=float(p[1]),
            w0=float(p[2]),
        )
        err = w_pred - w_local

        total_sae += float(np.sum(np.abs(err)))
        weights = 1.0 / (sigma_safe**2)
        total_weighted_sse += float(np.sum(weights * err**2))
        total_weight_sum += float(np.sum(weights))
        total_samples += int(r_local.size)

    if total_weight_sum <= 0.0:
        raise ValueError(
            f"Total WRMSE denominator is non-positive for fan F{fan_idx + 1:02d}."
        )

    total_wrmse = float(np.sqrt(total_weighted_sse / total_weight_sum))
    return RawMetrics(
        total_sae=float(total_sae),
        total_wrmse=total_wrmse,
        n_samples=total_samples,
    )


def run_solver_autotune_for_fan(
    mean_sheet_names: Iterable[str],
    base_config: FitConfig,
    fan_idx: int,
) -> Tuple[FitConfig, np.ndarray, np.ndarray, List[SolverTrial]]:
    """
    Auto-tune robust solver settings for one fan independently.
    """
    candidates = build_solver_candidates()
    if len(candidates) == 0:
        raise ValueError("No solver candidates configured.")

    trials: List[SolverTrial] = []
    for candidate in candidates:
        trial_config = replace(
            base_config,
            robust_loss=candidate.robust_loss,
            robust_f_scale=candidate.robust_f_scale,
        )
        z_vals, params = fit_all_heights_for_fan(
            mean_sheet_names=mean_sheet_names,
            config=trial_config,
            fan_idx=fan_idx,
        )
        metrics = evaluate_fit_on_raw_maps_for_fan(
            mean_sheet_names=mean_sheet_names,
            z_vals=z_vals,
            params=params,
            config=trial_config,
            fan_idx=fan_idx,
        )
        trials.append(
            SolverTrial(
                candidate=candidate,
                z_vals=z_vals,
                params=params,
                metrics=metrics,
            )
        )

    baseline = trials[0]
    max_sae = baseline.metrics.total_sae * (1.0 + float(base_config.sae_guardrail_fraction))
    eligible = [trial for trial in trials if trial.metrics.total_sae <= max_sae]
    if len(eligible) == 0:
        eligible = trials

    min_wrmse = min(trial.metrics.total_wrmse for trial in eligible)
    finalists = [
        trial
        for trial in eligible
        if trial.metrics.total_wrmse <= min_wrmse + WRMSE_TIE_TOLERANCE
    ]
    selected = min(finalists, key=lambda trial: trial.metrics.total_sae)

    selected_config = replace(
        base_config,
        robust_loss=selected.candidate.robust_loss,
        robust_f_scale=selected.candidate.robust_f_scale,
    )

    print(f"Auto-tuning solver candidates for fan F{fan_idx + 1:02d}:")
    print(" name               loss     f_scale   WRMSE      SAE")
    for trial in trials:
        marker = "*" if trial.candidate.name == selected.candidate.name else " "
        candidate = trial.candidate
        metrics = trial.metrics
        print(
            f"{marker}{candidate.name:17s}  {candidate.robust_loss:7s}  "
            f"{candidate.robust_f_scale:7.3f}  "
            f"{metrics.total_wrmse:8.5f}  {metrics.total_sae:8.4f}"
        )
    print(
        f"Selected F{fan_idx + 1:02d}: {selected.candidate.name} "
        f"(SAE guardrail <= {max_sae:.4f}, baseline SAE={baseline.metrics.total_sae:.4f})"
    )

    return selected_config, selected.z_vals, selected.params, trials


def fit_all_heights(
    mean_sheet_names: Iterable[str],
    config: FitConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit plain Gaussian model at all heights.

    Returns:
        z_vals: shape (K,), heights in meters
        params: shape (K, 3), fitted parameters [A, delta, w0]
    """
    z_list = []
    params_list = []

    for sheet in mean_sheet_names:
        z_m = parse_sheet_height_m(sheet)
        ts_sheet = f"{sheet}_TS"

        r_bins, w_bins, n_bins, sigma_bins_csv = load_annuli_profile_csv(
            profile_dir=config.annuli_profile_dir,
            sheet_name=sheet,
        )

        sigma_z = parse_ts_noise_scale(config.xlsx_path, ts_sheet)
        if sigma_z is None:
            sigma_z = config.sigma_z_fallback

        if sigma_bins_csv is None:
            sigma_bins = np.full_like(w_bins, float(sigma_z), dtype=float)
        else:
            sigma_bins = sigma_bins_csv.astype(float, copy=True)
            bad = ~np.isfinite(sigma_bins) | (sigma_bins <= 0.0)
            if np.any(bad):
                sigma_bins[bad] = float(sigma_z)

        fit_mask = build_fit_mask(r_bins=r_bins, config=config)
        n_fit = int(np.count_nonzero(fit_mask))
        print(
            f"[{sheet}] fitting bins used: {n_fit}/{r_bins.size} "
            f"(inner_bins_to_ignore={config.inner_bins_to_ignore}, "
            f"core_ignore_radius_m={config.core_ignore_radius_m})"
        )

        w0_min, w0_max = w0_bounds_from_plane(w_bins, margin_mps=W0_MARGIN_MPS)
        bounds = GaussianBounds(
            delta_min=0.08,
            delta_max=1.20,
            a_max=max(50.0, 3.0 * float(np.max(w_bins)) + 1.0),
            w0_min=float(w0_min),
            w0_max=float(w0_max),
        )

        p = fit_gaussian_at_height(
            r_bins=r_bins,
            w_bins=w_bins,
            n_bins=n_bins,
            sigma_bins=sigma_bins,
            bounds=bounds,
            config=config,
        )

        z_list.append(z_m)
        params_list.append(p)

    z_vals = np.array(z_list, dtype=float)
    params = np.vstack(params_list)

    order = np.argsort(z_vals)
    return z_vals[order], params[order]


def params_at_z(
    z_vals: np.ndarray,
    params: np.ndarray,
    z_query: float,
) -> np.ndarray:
    """Interpolate [A, delta, w0] at one z using linear interpolation."""
    if np.any(np.diff(z_vals) <= 0.0):
        raise ValueError("z_vals must be strictly increasing.")

    p = np.empty(3, dtype=float)
    for col_idx in range(3):
        p[col_idx] = float(np.interp(z_query, z_vals, params[:, col_idx]))
    return p


def evaluate_fit_on_raw_maps(
    mean_sheet_names: Iterable[str],
    z_vals: np.ndarray,
    params: np.ndarray,
    config: FitConfig,
) -> RawMetrics:
    """
    Evaluate fitted parameters on raw Cartesian maps with TS-derived sigma weights.

    Metrics:
    - total SAE over all raw grid points,
    - total weighted RMSE with weights (1/sigma_i)^2.
    """
    total_sae = 0.0
    total_weighted_sse = 0.0
    total_weight_sum = 0.0
    total_samples = 0

    for sheet in mean_sheet_names:
        z_m = parse_sheet_height_m(sheet)
        p = params_at_z(z_vals=z_vals, params=params, z_query=z_m)

        x_grid, y_grid, w_grid = load_mean_map(config.xlsx_path, sheet)
        x_pts = x_grid.ravel()
        y_pts = y_grid.ravel()
        fan_xy = np.asarray(config.fan_centers_xy, dtype=float)
        r_all = np.sqrt(
            (x_pts[:, None] - fan_xy[None, :, 0]) ** 2
            + (y_pts[:, None] - fan_xy[None, :, 1]) ** 2
        )
        w_obs = w_grid.ravel()

        mask = (
            np.isfinite(x_pts)
            & np.isfinite(y_pts)
            & np.isfinite(w_obs)
        )
        x_pts = x_pts[mask]
        y_pts = y_pts[mask]
        r_all = r_all[mask, :]
        w_obs = w_obs[mask]

        if x_pts.size == 0:
            raise ValueError(f"No valid raw samples found in sheet '{sheet}'.")

        sigma_pts = sigma_gp.evaluate_sigma_points_annular_blend_pchip_z(
            xlsx_path=config.xlsx_path,
            sheet_names=MEAN_SHEETS,
            fan_centers_xy=config.fan_centers_xy,
            x_pts=x_pts,
            y_pts=y_pts,
            z_pts=np.full_like(x_pts, parse_sheet_height_m(sheet), dtype=float),
            sigma_fallback=float(config.sigma_z_fallback),
            sigma_min=1e-3,
        )

        sigma_safe = np.maximum(sigma_pts, 1e-3)
        w_pred = np.zeros_like(w_obs, dtype=float)
        for fan_idx in range(len(config.fan_centers_xy)):
            w_pred += plain_gaussian(
                r_all[:, fan_idx],
                a=float(p[0]),
                delta=float(p[1]),
                w0=float(p[2]),
            )
        err = w_pred - w_obs

        total_sae += float(np.sum(np.abs(err)))

        weights = 1.0 / (sigma_safe**2)
        total_weighted_sse += float(np.sum(weights * err**2))
        total_weight_sum += float(np.sum(weights))
        total_samples += int(x_pts.size)

    if total_weight_sum <= 0.0:
        raise ValueError("Total WRMSE denominator is non-positive.")

    total_wrmse = float(np.sqrt(total_weighted_sse / total_weight_sum))
    return RawMetrics(
        total_sae=float(total_sae),
        total_wrmse=total_wrmse,
        n_samples=total_samples,
    )


def build_solver_candidates() -> Sequence[SolverCandidate]:
    """Build solver-candidate objects from user settings."""
    return tuple(
        SolverCandidate(
            name=name,
            robust_loss=robust_loss,
            robust_f_scale=float(robust_f_scale),
        )
        for (name, robust_loss, robust_f_scale) in SOLVER_CANDIDATES
    )


def run_solver_autotune(
    mean_sheet_names: Iterable[str],
    base_config: FitConfig,
) -> Tuple[FitConfig, np.ndarray, np.ndarray, List[SolverTrial]]:
    """
    Try multiple solver settings and select one by WRMSE under an SAE guardrail.

    Selection rule:
    1) Keep candidates with SAE <= baseline_SAE * (1 + sae_guardrail_fraction).
    2) Among them, choose the smallest WRMSE (tie-break by SAE).
    """
    candidates = build_solver_candidates()
    if len(candidates) == 0:
        raise ValueError("No solver candidates configured.")

    trials: List[SolverTrial] = []

    for candidate in candidates:
        trial_config = replace(
            base_config,
            robust_loss=candidate.robust_loss,
            robust_f_scale=candidate.robust_f_scale,
        )
        z_vals, params = fit_all_heights(mean_sheet_names, config=trial_config)
        metrics = evaluate_fit_on_raw_maps(
            mean_sheet_names=mean_sheet_names,
            z_vals=z_vals,
            params=params,
            config=trial_config,
        )
        trials.append(
            SolverTrial(
                candidate=candidate,
                z_vals=z_vals,
                params=params,
                metrics=metrics,
            )
        )

    baseline = trials[0]
    max_sae = baseline.metrics.total_sae * (1.0 + float(base_config.sae_guardrail_fraction))

    eligible = [trial for trial in trials if trial.metrics.total_sae <= max_sae]
    if not eligible:
        eligible = trials

    min_wrmse = min(trial.metrics.total_wrmse for trial in eligible)
    finalists = [
        trial
        for trial in eligible
        if trial.metrics.total_wrmse <= min_wrmse + WRMSE_TIE_TOLERANCE
    ]
    selected = min(finalists, key=lambda trial: trial.metrics.total_sae)

    selected_config = replace(
        base_config,
        robust_loss=selected.candidate.robust_loss,
        robust_f_scale=selected.candidate.robust_f_scale,
    )

    print("Auto-tuning solver candidates (raw-map validation):")
    print(" name               loss     f_scale   WRMSE      SAE")
    for trial in trials:
        marker = "*" if trial.candidate.name == selected.candidate.name else " "
        candidate = trial.candidate
        metrics = trial.metrics
        print(
            f"{marker}{candidate.name:17s}  {candidate.robust_loss:7s}  "
            f"{candidate.robust_f_scale:7.3f}  "
            f"{metrics.total_wrmse:8.5f}  {metrics.total_sae:8.4f}"
        )
    print(
        f"Selected: {selected.candidate.name} "
        f"(SAE guardrail <= {max_sae:.4f}, baseline SAE={baseline.metrics.total_sae:.4f})"
    )

    return selected_config, selected.z_vals, selected.params, trials


# =============================================================================
# 6) Fitting Export Entry Point
# =============================================================================

def main() -> None:
    config = FitConfig(
        xlsx_path=XLSX_PATH,
        annuli_profile_dir=ANNULI_PROFILE_DIR,
        fan_centers_xy=FOUR_FAN_CENTERS_XY,
        robust_loss="linear",
        robust_f_scale=1.0,
        fan_robust_loss=FAN_ROBUST_LOSS,
        fan_robust_f_scale=FAN_ROBUST_F_SCALE,
        sigma_z_fallback=0.2,
        sigma_min=1e-3,
        profile_delta_r_m=PROFILE_DELTA_R_M,
        profile_use_median=PROFILE_USE_MEDIAN,
        use_count_weighting=True,
        max_nfev=5000,
        multi_start_enabled=False,
        delta_seed_values=(0.10, 0.20, 0.35, 0.50),
        inner_bins_to_ignore=1,
        core_ignore_radius_m=None,
        enable_solver_autotune=False,
        sae_guardrail_fraction=0.02,
        overlap_ratio_threshold=OVERLAP_RATIO_THRESHOLD,
        overlap_weight_power=OVERLAP_WEIGHT_POWER,
        overlap_sigma_boost=OVERLAP_SIGMA_BOOST,
    )

    validate_joint_fan_settings(config)
    n_fans = len(config.fan_centers_xy)
    z_vals, params_stack = fit_all_heights_joint(MEAN_SHEETS, config=config)
    mean_params = np.mean(params_stack, axis=1)
    overall_metrics, per_fan_metrics = evaluate_joint_fit_on_raw_maps(
        mean_sheet_names=MEAN_SHEETS,
        z_vals=z_vals,
        params_stack=params_stack,
        config=config,
    )

    print("\nFitted per-fan plain-Gaussian parameters per height (joint 12-parameter solve)")
    print(
        f"Joint solve: n_fans={len(config.fan_centers_xy)}, "
        f"n_params_per_height={len(config.fan_centers_xy) * 3}",
    )
    print(
        "Per-fan robust settings: "
        + ", ".join(
            [
                f"F{idx + 1:02d}({loss}, f_scale={f_scale})"
                for idx, (loss, f_scale) in enumerate(
                    zip(config.fan_robust_loss, config.fan_robust_f_scale)
                )
            ]
        )
    )
    print(
        f"w0_bounds=plane-dependent (w_min-{W0_MARGIN_MPS:.1f}, w_max+{W0_MARGIN_MPS:.1f}), "
        f"core_ignore: inner_bins={config.inner_bins_to_ignore}, "
        f"radius={config.core_ignore_radius_m}"
    )
    print(
        f"\nOverall overlap-model raw-map metrics: WRMSE={overall_metrics.total_wrmse:.5f}, "
        f"SAE={overall_metrics.total_sae:.4f}, n_samples={overall_metrics.n_samples}"
    )
    print(
        "Nearest-region diagnostics below are computed from the same overlap-model "
        "prediction, segmented by nearest fan center."
    )
    for fan_idx in range(n_fans):
        fan_metrics = per_fan_metrics[fan_idx]
        print(
            f"\nRegion near F{fan_idx + 1:02d} "
            f"(x={config.fan_centers_xy[fan_idx][0]:.2f}, "
            f"y={config.fan_centers_xy[fan_idx][1]:.2f}) "
            f"loss={config.fan_robust_loss[fan_idx]}, "
            f"f_scale={config.fan_robust_f_scale[fan_idx]}, "
            f"WRMSE={fan_metrics.total_wrmse:.5f}, SAE={fan_metrics.total_sae:.4f}"
        )
        print(" z [m]       A        delta       w0")
        for z_m, (a, delta, w0) in zip(z_vals, params_stack[:, fan_idx, :]):
            print(f"{z_m:5.2f}  {a:9.4f}  {delta:10.4f}  {w0:9.4f}")

    out_dir = OUT_XLSX_PATH.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep legacy averaged columns for compatibility, plus explicit per-fan columns.
    out_data = {
        "z_m": z_vals,
        "A": mean_params[:, 0],
        "delta": mean_params[:, 1],
        "w0": mean_params[:, 2],
    }
    for fan_idx in range(n_fans):
        suffix = f"F{fan_idx + 1:02d}"
        out_data[f"A_{suffix}"] = params_stack[:, fan_idx, 0]
        out_data[f"delta_{suffix}"] = params_stack[:, fan_idx, 1]
        out_data[f"w0_{suffix}"] = params_stack[:, fan_idx, 2]
    df_out = pd.DataFrame(out_data)

    hyper_rows = []
    for fan_idx in range(n_fans):
        fan_metrics = per_fan_metrics[fan_idx]
        cx, cy = config.fan_centers_xy[fan_idx]
        hyper_rows.append(
            {
                "fan_id": f"F{fan_idx + 1:02d}",
                "center_x_m": float(cx),
                "center_y_m": float(cy),
                "solver_loss": config.fan_robust_loss[fan_idx],
                "solver_f_scale": float(config.fan_robust_f_scale[fan_idx]),
                "raw_wrmse_mps": float(fan_metrics.total_wrmse),
                "raw_sae_mps": float(fan_metrics.total_sae),
                "raw_n_samples": int(fan_metrics.n_samples),
                "metric_scope": "nearest_region_overlap_model",
                "inner_bins_to_ignore": int(config.inner_bins_to_ignore),
                "profile_delta_r_m": float(config.profile_delta_r_m),
                "overlap_ratio_threshold": float(config.overlap_ratio_threshold),
                "overlap_weight_power": float(config.overlap_weight_power),
                "overlap_sigma_boost": float(config.overlap_sigma_boost),
            }
        )
    df_hyper = pd.DataFrame(hyper_rows)

    with pd.ExcelWriter(OUT_XLSX_PATH, engine="openpyxl", mode="w") as writer:
        df_out.to_excel(writer, index=False, sheet_name="four_var")
        df_hyper.to_excel(writer, index=False, sheet_name="four_var_hyper")

    print(f"Saved Excel results to: {OUT_XLSX_PATH.resolve()}")


if __name__ == "__main__":
    main()
