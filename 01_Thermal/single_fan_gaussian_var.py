"""
Plain-Gaussian identification for single-fan updraft maps in S01.xlsx.

This script fits an axisymmetric Gaussian model to each measured height
sheet and writes per-height parameters to Excel.

Model (per height z_k):
    w(r; z_k) = w0(z_k) + A(z_k) * exp(-(r / delta(z_k))**2)
"""

###### Initialization

### Imports

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
import single_fan_gp as sigma_gp
from single_fan_annuli_cut import (
    assign_sigma_points_linear_nearest,
    parse_ts_xy_points_and_sigmas,
)


### User settings

XLSX_PATH = "S01.xlsx"
ANNULI_PROFILE_DIR = Path("B_results/Single_Fan_Annuli_Profile")
OUT_XLSX_PATH = Path("B_results/single_var_params.xlsx")

# Update if your fan center is different.
FAN_CENTER_XY = (4.2, 2.4)

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


### Data classes

@dataclass(frozen=True)
class FitConfig:
    """Configuration for loading, fitting, and auto-tuning."""

    xlsx_path: str = XLSX_PATH
    annuli_profile_dir: Path = ANNULI_PROFILE_DIR
    fan_center_xy: Tuple[float, float] = FAN_CENTER_XY

    # Robust least-squares options for scipy.optimize.least_squares.
    robust_loss: str = "soft_l1"
    robust_f_scale: float = 1.0

    # If TS parsing fails, fallback sigma_z (m/s).
    sigma_z_fallback: float = 0.2

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

    # Auto-tuning behaviour.
    enable_solver_autotune: bool = True
    sae_guardrail_fraction: float = 0.02


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
    fan_center_xy: Tuple[float, float]

    def __call__(self, x: np.ndarray, y: np.ndarray, z: float) -> np.ndarray:
        xc, yc = self.fan_center_xy
        r = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

        a = float(np.exp(self.a_log(z)))
        delta = float(np.exp(self.delta_log(z)))
        w0 = float(self.w0(z))
        return plain_gaussian(r, a=a, delta=delta, w0=w0)


# Helpers

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
    csv_path = Path(profile_dir) / f"{sheet_name}_single_annuli_profile.csv"
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


# Model

def plain_gaussian(r: np.ndarray, a: float, delta: float, w0: float) -> np.ndarray:
    """Evaluate w(r) = w0 + A * exp(-(r / delta)^2)."""
    return w0 + a * np.exp(-((r / delta) ** 2))


def build_smooth_gaussian_model(
    z_vals: np.ndarray,
    params: np.ndarray,
    fan_center_xy: Tuple[float, float],
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
        fan_center_xy=fan_center_xy,
    )


# Fitting


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

    xc, yc = config.fan_center_xy

    for sheet in mean_sheet_names:
        z_m = parse_sheet_height_m(sheet)
        p = params_at_z(z_vals=z_vals, params=params, z_query=z_m)

        x_grid, y_grid, w_grid = load_mean_map(config.xlsx_path, sheet)
        x_pts = x_grid.ravel()
        y_pts = y_grid.ravel()
        r_pts = np.sqrt((x_pts - xc) ** 2 + (y_pts - yc) ** 2).ravel()
        w_obs = w_grid.ravel()

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

        sigma_pts = sigma_gp.evaluate_sigma_points_annular_pchip_z(
            xlsx_path=config.xlsx_path,
            sheet_names=MEAN_SHEETS,
            fan_center_xy=config.fan_center_xy,
            x_pts=x_pts,
            y_pts=y_pts,
            z_pts=np.full_like(x_pts, parse_sheet_height_m(sheet), dtype=float),
            sigma_fallback=config.sigma_z_fallback,
            sigma_min=1e-3,
        )

        sigma_safe = np.maximum(sigma_pts, 1e-3)
        w_pred = plain_gaussian(
            r_pts,
            a=float(p[0]),
            delta=float(p[1]),
            w0=float(p[2]),
        )
        err = w_pred - w_obs

        total_sae += float(np.sum(np.abs(err)))

        weights = 1.0 / (sigma_safe**2)
        total_weighted_sse += float(np.sum(weights * err**2))
        total_weight_sum += float(np.sum(weights))
        total_samples += int(r_pts.size)

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


### Main

def main() -> None:
    config = FitConfig(
        xlsx_path=XLSX_PATH,
        annuli_profile_dir=ANNULI_PROFILE_DIR,
        fan_center_xy=FAN_CENTER_XY,
        robust_loss="soft_l1",
        robust_f_scale=1.0,
        sigma_z_fallback=0.2,
        use_count_weighting=True,
        max_nfev=5000,
        multi_start_enabled=True,
        delta_seed_values=(0.10, 0.20, 0.35, 0.50),
        inner_bins_to_ignore=1,
        core_ignore_radius_m=None,
        enable_solver_autotune=True,
        sae_guardrail_fraction=0.02,
    )

    if config.enable_solver_autotune:
        config, z_vals, params, _trials = run_solver_autotune(MEAN_SHEETS, base_config=config)
    else:
        z_vals, params = fit_all_heights(MEAN_SHEETS, config=config)

    _model = build_smooth_gaussian_model(z_vals=z_vals, params=params, fan_center_xy=config.fan_center_xy)

    print("\nFitted plain-Gaussian parameters per height")
    print(
        f"Solver: loss={config.robust_loss}, f_scale={config.robust_f_scale}, "
        f"w0_bounds=plane-dependent (w_min-{W0_MARGIN_MPS:.1f}, w_max+{W0_MARGIN_MPS:.1f}), "
        f"core_ignore: inner_bins={config.inner_bins_to_ignore}, "
        f"radius={config.core_ignore_radius_m}"
    )
    print(" z [m]       A        delta       w0")
    for z_m, (a, delta, w0) in zip(z_vals, params):
        print(f"{z_m:5.2f}  {a:9.4f}  {delta:10.4f}  {w0:9.4f}")

    out_dir = OUT_XLSX_PATH.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(
        {
            "z_m": z_vals,
            "A": params[:, 0],
            "delta": params[:, 1],
            "w0": params[:, 2],
        }
    )
    df_out.to_excel(OUT_XLSX_PATH, index=False, sheet_name="single_var")
    print(f"Saved Excel results to: {OUT_XLSX_PATH.resolve()}")


if __name__ == "__main__":
    main()
