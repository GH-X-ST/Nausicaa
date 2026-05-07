###### Initialization

### Imports
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import (
    LinearNDInterpolator,
    NearestNDInterpolator,
    PchipInterpolator,
)
from scipy.spatial import QhullError
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    RBF,
    RationalQuadratic,
    WhiteKernel,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from single_fan_annuli_cut import read_slice_from_sheet


### User settings
XLSX_PATH = "S02.xlsx"
SHEETS = ["z020", "z035", "z050", "z075", "z110", "z160", "z220"]

# Fan centre (x_c, y_c) in meters.
FAN_CENTER_XY = (4.2, 2.4)
SHEET_HEIGHT_DIVISOR = 100.0
FOUR_FAN_CENTERS_XY = (
    (3.0, 3.6),
    (5.4, 3.6),
    (3.0, 1.2),
    (5.4, 1.2),
)
FOUR_FAN_CORE_RADIUS_M = 0.4

# Feature mode:
#   "polar"     -> use (r, cos(theta), sin(theta), z): annular + non-axisymmetric
#   "radial"    -> use (r, z): axisymmetric annular
#   "cartesian" -> use (x, y, z): fully unconstrained spatial structure
FEATURE_MODE = "cartesian"
KERNEL_FAMILY = "matern32_ard"
ALPHA_SCALE = 1.0

# Noise assignment from *_TS.
SIGMA_FALLBACK = 0.14
SIGMA_MIN = 0.03
ALPHA_JITTER = 1e-8
# GP optimizer settings.
N_RESTARTS_OPTIMIZER = 4
RANDOM_STATE = 42

# Automatic GP configuration tuning (grouped CV by sheet height).
ENABLE_AUTO_TUNE = True
AUTO_TUNE_CV_N_SPLITS = 3
AUTO_TUNE_CV_RESTARTS_OPTIMIZER = 0
AUTO_TUNE_RMSE_TIE_TOL = 1e-4

# Candidate format: (name, feature_mode, kernel_family, alpha_scale)
# If empty, a default candidate grid is built from FEATURE_MODE plus the
# kernel/alpha settings below.
AUTO_TUNE_CANDIDATES = (
    ("baseline", "cartesian", "matern32_ard", 1.0),
    ("cartesian_matern32_ard_a0p8", "cartesian", "matern32_ard", 0.8),
    ("cartesian_matern32_ard_a1p2", "cartesian", "matern32_ard", 1.2),
    ("cartesian_matern52_ard_a1", "cartesian", "matern52_ard", 1.0),
    ("cartesian_rbf_ard_a1", "cartesian", "rbf_ard", 1.0),
)
AUTO_TUNE_DEFAULT_KERNEL_FAMILIES = ("matern32_ard", "matern52_ard", "rbf_ard")
AUTO_TUNE_DEFAULT_ALPHA_SCALES = (0.8, 1.0, 1.2)

# Output locations.
OUT_DIR = Path("B_results/Four_Fan_GP")
TRAIN_PRED_CSV_PATH = OUT_DIR / "four_gp_training_predictions.csv"
SUMMARY_XLSX_PATH = OUT_DIR / "four_gp_summary.xlsx"
GRID_PRED_XLSX_PATH = OUT_DIR / "four_gp_grid_predictions.xlsx"
ANALYSIS_XLSX_PATH = "B_results/four_gp_analysis.xlsx"
ANALYSIS_SHEET_NAME = "four_gp_analysis"
CORE_STRENGTH_SHEET_NAME = "four_gp_core_strength"

### Data model
@dataclass
class GPModelBundle:
    """
    Container for a fitted GP model and its feature transform.
    """

    gp: GaussianProcessRegressor
    scaler: StandardScaler
    feature_mode: str
    fan_center_xy: Tuple[float, float]

    def predict(
        self,
        x_m: np.ndarray,
        y_m: np.ndarray,
        z_m: np.ndarray,
        return_std: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean velocity and optional predictive standard deviation.
        """
        feats = build_feature_matrix(
            x_m=x_m,
            y_m=y_m,
            z_m=z_m,
            feature_mode=self.feature_mode,
            fan_center_xy=self.fan_center_xy,
        )
        feats_scaled = self.scaler.transform(feats)

        if return_std:
            w_mean, w_std = self.gp.predict(feats_scaled, return_std=True)
            return w_mean.astype(float), w_std.astype(float)

        w_mean = self.gp.predict(feats_scaled, return_std=False)
        return w_mean.astype(float), np.zeros_like(w_mean, dtype=float)


@dataclass(frozen=True)
class GPTuneCandidate:
    """
    One GP hyper-parameter candidate for grouped CV tuning.
    """

    name: str
    feature_mode: str
    kernel_family: str
    alpha_scale: float


@dataclass
class GPTuneTrial:
    """
    Container for one grouped-CV tuning trial.
    """

    candidate: GPTuneCandidate
    cv_metrics: Dict[str, float]
    n_folds: int


### Helpers
def ensure_path(path_like: str | Path) -> Path:
    """
    Convert a string/Path into a Path object.
    """
    return path_like if isinstance(path_like, Path) else Path(path_like)


def parse_sheet_height_m(sheet_name: str) -> float:
    if not sheet_name.startswith("z"):
        raise ValueError(f"Invalid sheet name (expected 'z###'): {sheet_name}")
    suffix = sheet_name[1:]
    if not suffix.isdigit():
        raise ValueError(f"Invalid height code in sheet name: {sheet_name}")
    return int(suffix) / SHEET_HEIGHT_DIVISOR


@lru_cache(maxsize=None)
def load_ts_sigma_planes(
    xlsx_path: str,
    sheet_names: Tuple[str, ...],
) -> Tuple[np.ndarray, Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray] | None, ...]]:
    """
    Cache representative fluctuation points for all measured z planes.
    """
    z_list: List[float] = []
    parsed_list: List[Tuple[np.ndarray, np.ndarray, np.ndarray] | None] = []

    for sheet_name in sheet_names:
        z_list.append(parse_sheet_height_m(sheet_name))
        parsed_list.append(
            parse_ts_xy_points_and_sigmas(
                xlsx_path=xlsx_path,
                ts_sheet_name=f"{sheet_name}_TS",
            )
        )

    order = np.argsort(np.asarray(z_list, dtype=float))
    z_axis = np.asarray([z_list[idx] for idx in order], dtype=float)
    parsed_sorted = tuple(parsed_list[idx] for idx in order)
    return z_axis, parsed_sorted


def _cell_is_str(df: pd.DataFrame, r_idx: int, c_idx: int, text: str) -> bool:
    """Case-insensitive equality test for a sheet cell against a string."""
    val = df.iat[r_idx, c_idx]
    return isinstance(val, str) and val.strip().lower() == text.strip().lower()


def _first_numeric_below(
    df: pd.DataFrame,
    r_idx: int,
    c_idx: int,
) -> Optional[float]:
    """Return the first finite numeric value below a sheet cell."""
    col = pd.to_numeric(
        df.iloc[r_idx + 1 :, c_idx],
        errors="coerce",
    ).to_numpy(dtype=float)
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
    """
    try:
        df = pd.read_excel(xlsx_path, sheet_name=ts_sheet_name, header=None)
    except Exception:
        return None

    xc, yc = fan_center_xy
    r_points: List[float] = []
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

            sigma_points.append(float(np.sqrt(var_val)))
            r_points.append(
                float(np.sqrt((float(x_val) - float(xc)) ** 2 + (float(y_val) - float(yc)) ** 2))
            )

    if not r_points:
        return None

    r_arr = np.asarray(r_points, dtype=float)
    sigma_arr = np.asarray(sigma_points, dtype=float)
    order = np.argsort(r_arr)
    return r_arr[order], sigma_arr[order]


def parse_ts_xy_points_and_sigmas(
    xlsx_path: str,
    ts_sheet_name: str,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Parse representative-point coordinates and sigmas from one *_TS sheet.

    Each recognised block contributes one representative point:
      - x_p, y_p from the first data row below the header row
      - sigma_p = sqrt(variance_p)
    """
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
            y_val = pd.to_numeric(
                df.iat[r_idx + 1, x_col + 1],
                errors="coerce",
            )
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


def build_feature_matrix(
    x_m: np.ndarray,
    y_m: np.ndarray,
    z_m: np.ndarray,
    feature_mode: str,
    fan_center_xy: Tuple[float, float],
) -> np.ndarray:
    """
    Build GP input matrix from coordinate arrays.
    """
    x_arr = np.asarray(x_m, dtype=float).ravel()
    y_arr = np.asarray(y_m, dtype=float).ravel()
    z_arr = np.asarray(z_m, dtype=float).ravel()

    if not (x_arr.size == y_arr.size == z_arr.size):
        raise ValueError("x_m, y_m, z_m must have the same number of samples.")

    mode = str(feature_mode).strip().lower()
    if mode not in {"polar", "radial", "cartesian"}:
        raise ValueError(
            f"Invalid FEATURE_MODE '{feature_mode}'. Use: polar, radial, cartesian."
        )

    if mode in {"polar", "radial"}:
        xc, yc = fan_center_xy
        r_arr = np.sqrt((x_arr - xc) ** 2 + (y_arr - yc) ** 2)
        if mode == "radial":
            return np.column_stack([r_arr, z_arr])

        theta_arr = np.arctan2(y_arr - yc, x_arr - xc)
        return np.column_stack([r_arr, np.cos(theta_arr), np.sin(theta_arr), z_arr])

    return np.column_stack([x_arr, y_arr, z_arr])


def build_gp_kernel(n_features: int, kernel_family: str):
    """
    Build a GP kernel by family name.
    """
    family = str(kernel_family).strip().lower()

    if family == "rbf_ard":
        return (
            ConstantKernel(1.0, (1e-3, 1e3))
            * RBF(
                length_scale=np.ones(n_features, dtype=float),
                length_scale_bounds=(1e-1, 1e2),
            )
            + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e0))
        )

    if family == "matern32_ard":
        return (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(
                length_scale=np.ones(n_features, dtype=float),
                length_scale_bounds=(1e-1, 1e2),
                nu=1.5,
            )
            + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e0))
        )

    if family == "matern52_ard":
        return (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(
                length_scale=np.ones(n_features, dtype=float),
                length_scale_bounds=(1e-1, 1e2),
                nu=2.5,
            )
            + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e0))
        )

    if family == "rq":
        return (
            ConstantKernel(1.0, (1e-3, 1e3))
            * RationalQuadratic(
                length_scale=1.0,
                alpha=1.0,
                length_scale_bounds=(1e-1, 1e2),
                alpha_bounds=(1e-2, 1e3),
            )
            + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e0))
        )

    raise ValueError(
        f"Invalid KERNEL_FAMILY '{kernel_family}'. "
        "Use: rbf_ard, matern32_ard, matern52_ard, rq."
    )


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
        sigma_out = np.full(query_x.size, float(sigma_fallback), dtype=float)
        return np.maximum(sigma_out, float(sigma_min))

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


def assign_sigma_with_overlap_logic(
    xlsx_path: str,
    ts_sheet_name: str,
    x_pts: np.ndarray,
    y_pts: np.ndarray,
    sigma_fallback: float,
    sigma_min: float,
) -> np.ndarray:
    """
    Four-fan uncertainty mapping from local *_TS representative points.

    Use linear interpolation inside the TS-point convex hull and nearest
    extrapolation outside it, so the fluctuation field stays local without
    inventing long-range radial smearing.
    """
    parsed = parse_ts_xy_points_and_sigmas(
        xlsx_path=xlsx_path,
        ts_sheet_name=ts_sheet_name,
    )
    if parsed is None:
        sigma_arr = np.full(x_pts.size, float(sigma_fallback), dtype=float)
        return np.maximum(sigma_arr, float(sigma_min))

    rep_x, rep_y, rep_sigma = parsed
    return assign_sigma_points_linear_nearest(
        x_pts=x_pts,
        y_pts=y_pts,
        rep_x=rep_x,
        rep_y=rep_y,
        rep_sigma=rep_sigma,
        sigma_fallback=sigma_fallback,
        sigma_min=sigma_min,
    )


def evaluate_sigma_points_pchip_z(
    xlsx_path: str,
    sheet_names: Sequence[str],
    x_pts: np.ndarray,
    y_pts: np.ndarray,
    z_pts: np.ndarray,
    sigma_fallback: float,
    sigma_min: float,
) -> np.ndarray:
    """
    Evaluate fluctuation sigma continuously in z.

    For each measured plane, sigma is assigned in x-y using the local
    representative TS points. The per-point values across z are then blended
    with PCHIP, while heights outside the measured range hold the nearest
    measured plane.
    """
    x_arr = np.asarray(x_pts, dtype=float).ravel()
    y_arr = np.asarray(y_pts, dtype=float).ravel()
    z_arr = np.asarray(z_pts, dtype=float).ravel()
    if not (x_arr.size == y_arr.size == z_arr.size):
        raise ValueError("x_pts, y_pts, z_pts must have the same number of samples.")

    z_axis, parsed_planes = load_ts_sigma_planes(
        str(xlsx_path),
        tuple(str(sheet) for sheet in sheet_names),
    )
    sigma_planes = np.empty((z_axis.size, x_arr.size), dtype=float)

    for idx, parsed in enumerate(parsed_planes):
        if parsed is None:
            sigma_planes[idx] = np.full(
                x_arr.size,
                float(sigma_fallback),
                dtype=float,
            )
            continue

        rep_x, rep_y, rep_sigma = parsed
        sigma_planes[idx] = assign_sigma_points_linear_nearest(
            x_pts=x_arr,
            y_pts=y_arr,
            rep_x=rep_x,
            rep_y=rep_y,
            rep_sigma=rep_sigma,
            sigma_fallback=sigma_fallback,
            sigma_min=sigma_min,
        )

    if z_axis.size == 1:
        return np.maximum(sigma_planes[0], float(sigma_min))

    z_eval = np.clip(z_arr, float(z_axis[0]), float(z_axis[-1]))
    interp = PchipInterpolator(z_axis, sigma_planes, axis=0, extrapolate=False)
    sigma_out = np.empty_like(z_eval, dtype=float)
    for z_val in np.unique(z_eval):
        sigma_at_z = np.asarray(interp(float(z_val)), dtype=float).reshape(-1)
        mask = np.isclose(z_eval, float(z_val), rtol=0.0, atol=1e-12)
        sigma_out[mask] = sigma_at_z[mask]

    sigma_low = np.maximum(float(sigma_min), np.min(sigma_planes, axis=0))
    sigma_high = np.max(sigma_planes, axis=0)
    sigma_out = np.clip(sigma_out, sigma_low, sigma_high)
    return np.maximum(sigma_out, float(sigma_min))


@lru_cache(maxsize=None)
def load_ts_radial_sigma_planes(
    xlsx_path: str,
    sheet_names: Tuple[str, ...],
    fan_centers_xy: Tuple[Tuple[float, float], ...],
) -> Tuple[np.ndarray, Tuple[Tuple[Tuple[np.ndarray, np.ndarray] | None, ...], ...]]:
    """
    Cache per-fan radial representative fluctuation data for all z planes.
    """
    z_list: List[float] = []
    plane_profiles: List[Tuple[Tuple[np.ndarray, np.ndarray] | None, ...]] = []

    for sheet_name in sheet_names:
        z_list.append(parse_sheet_height_m(sheet_name))
        per_fan_profiles: List[Tuple[np.ndarray, np.ndarray] | None] = []
        for fan_center_xy in fan_centers_xy:
            per_fan_profiles.append(
                parse_ts_points_and_sigmas(
                    xlsx_path=xlsx_path,
                    ts_sheet_name=f"{sheet_name}_TS",
                    fan_center_xy=fan_center_xy,
                )
            )
        plane_profiles.append(tuple(per_fan_profiles))

    order = np.argsort(np.asarray(z_list, dtype=float))
    z_axis = np.asarray([z_list[idx] for idx in order], dtype=float)
    profiles_sorted = tuple(plane_profiles[idx] for idx in order)
    return z_axis, profiles_sorted


def evaluate_sigma_points_annular_blend_pchip_z(
    xlsx_path: str,
    sheet_names: Sequence[str],
    fan_centers_xy: Sequence[Tuple[float, float]],
    x_pts: np.ndarray,
    y_pts: np.ndarray,
    z_pts: np.ndarray,
    sigma_fallback: float,
    sigma_min: float,
    blend_power: float = 2.0,
    blend_epsilon_m: float = 0.05,
    near_core_radius_m: float = FOUR_FAN_CORE_RADIUS_M + 0.05,
) -> np.ndarray:
    """
    Evaluate four-fan annular empirical fluctuation sigma continuously in z.

    Fans with near-core TS support retain their own radial fluctuation
    profiles. Fans without near-core support inherit the average sampled-fan
    profile. The four fan-centred profiles are then blended with smooth
    inverse-distance weights in x-y and combined across z using PCHIP.
    """
    x_arr = np.asarray(x_pts, dtype=float).ravel()
    y_arr = np.asarray(y_pts, dtype=float).ravel()
    z_arr = np.asarray(z_pts, dtype=float).ravel()
    if not (x_arr.size == y_arr.size == z_arr.size):
        raise ValueError("x_pts, y_pts, z_pts must have the same number of samples.")

    fan_xy = np.asarray(fan_centers_xy, dtype=float)
    d_fan = np.sqrt(
        (x_arr[:, None] - fan_xy[None, :, 0]) ** 2
        + (y_arr[:, None] - fan_xy[None, :, 1]) ** 2
    )
    weights = (d_fan + float(blend_epsilon_m)) ** (-float(blend_power))
    weights /= np.sum(weights, axis=1, keepdims=True)

    z_axis, plane_profiles = load_ts_radial_sigma_planes(
        str(xlsx_path),
        tuple(str(sheet) for sheet in sheet_names),
        tuple((float(xc), float(yc)) for xc, yc in fan_centers_xy),
    )
    sigma_planes = np.empty((z_axis.size, x_arr.size), dtype=float)

    for z_idx, per_fan_profiles in enumerate(plane_profiles):
        sigma_per_fan = np.empty((x_arr.size, fan_xy.shape[0]), dtype=float)
        sampled_profiles: List[Tuple[np.ndarray, np.ndarray]] = []
        for fan_idx, parsed in enumerate(per_fan_profiles):
            if parsed is None:
                continue

            r_points, sigma_points = parsed
            if np.min(r_points) <= float(near_core_radius_m):
                sampled_profiles.append((r_points, sigma_points))

        for fan_idx, parsed in enumerate(per_fan_profiles):
            if parsed is None:
                sigma_per_fan[:, fan_idx] = float(sigma_fallback)
                continue

            r_points, sigma_points = parsed
            has_near_core = np.min(r_points) <= float(near_core_radius_m)
            if has_near_core or not sampled_profiles:
                sigma_per_fan[:, fan_idx] = assign_sigma_bins_nearest(
                    r_bins=d_fan[:, fan_idx],
                    r_points=r_points,
                    sigma_points=sigma_points,
                    sigma_fallback=sigma_fallback,
                    sigma_min=sigma_min,
                )
                continue

            inherited_sigma = np.empty(
                (x_arr.size, len(sampled_profiles)),
                dtype=float,
            )
            for profile_idx, sampled_profile in enumerate(sampled_profiles):
                sampled_r, sampled_sigma = sampled_profile
                inherited_sigma[:, profile_idx] = assign_sigma_bins_nearest(
                    r_bins=d_fan[:, fan_idx],
                    r_points=sampled_r,
                    sigma_points=sampled_sigma,
                    sigma_fallback=sigma_fallback,
                    sigma_min=sigma_min,
                )
            sigma_per_fan[:, fan_idx] = np.maximum(
                np.mean(inherited_sigma, axis=1),
                float(sigma_min),
            )

        sigma_planes[z_idx] = np.sum(weights * sigma_per_fan, axis=1)

    if z_axis.size == 1:
        return np.maximum(sigma_planes[0], float(sigma_min))

    z_eval = np.clip(z_arr, float(z_axis[0]), float(z_axis[-1]))
    interp = PchipInterpolator(z_axis, sigma_planes, axis=0, extrapolate=False)
    sigma_out = np.empty_like(z_eval, dtype=float)
    for z_val in np.unique(z_eval):
        sigma_at_z = np.asarray(interp(float(z_val)), dtype=float).reshape(-1)
        mask = np.isclose(z_eval, float(z_val), rtol=0.0, atol=1e-12)
        sigma_out[mask] = sigma_at_z[mask]

    sigma_low = np.maximum(float(sigma_min), np.min(sigma_planes, axis=0))
    sigma_high = np.max(sigma_planes, axis=0)
    sigma_out = np.clip(sigma_out, sigma_low, sigma_high)
    return np.maximum(sigma_out, float(sigma_min))


def load_sheet_samples(
    xlsx_path: str,
    sheet_name: str,
    fan_center_xy: Tuple[float, float],
    sigma_fallback: float,
    sigma_min: float,
) -> pd.DataFrame:
    """
    Load one mean sheet and map per-sample sigma from the paired *_TS sheet.
    """
    z_m = parse_sheet_height_m(sheet_name)
    x_centers, y_centers, w_map = read_slice_from_sheet(xlsx_path, sheet_name)

    x_grid, y_grid = np.meshgrid(x_centers, y_centers)
    x_pts = x_grid.ravel()
    y_pts = y_grid.ravel()
    w_obs = w_map.ravel()

    fan_xy = np.asarray(FOUR_FAN_CENTERS_XY, dtype=float)
    d_sample_fan = np.sqrt(
        (x_pts[:, None] - fan_xy[None, :, 0]) ** 2
        + (y_pts[:, None] - fan_xy[None, :, 1]) ** 2
    )
    r_pts = np.min(d_sample_fan, axis=1)

    valid = (
        np.isfinite(x_pts)
        & np.isfinite(y_pts)
        & np.isfinite(r_pts)
        & np.isfinite(w_obs)
    )
    x_pts = x_pts[valid]
    y_pts = y_pts[valid]
    r_pts = r_pts[valid]
    w_obs = w_obs[valid]
    if x_pts.size == 0:
        raise ValueError(f"No valid samples found in sheet '{sheet_name}'.")

    ts_sheet = f"{sheet_name}_TS"
    sigma_pts = assign_sigma_with_overlap_logic(
        xlsx_path=xlsx_path,
        ts_sheet_name=ts_sheet,
        x_pts=x_pts,
        y_pts=y_pts,
        sigma_fallback=sigma_fallback,
        sigma_min=sigma_min,
    )

    return pd.DataFrame(
        {
            "sheet": sheet_name,
            "z_m": z_m,
            "x_m": x_pts,
            "y_m": y_pts,
            "r_m": r_pts,
            "w_obs_mps": w_obs,
            "sigma_mps": sigma_pts,
        }
    )


def build_training_table(
    xlsx_path: str,
    sheet_names: Sequence[str],
    fan_center_xy: Tuple[float, float],
    sigma_fallback: float,
    sigma_min: float,
) -> pd.DataFrame:
    """
    Assemble training samples from all requested sheets.
    """
    parts: List[pd.DataFrame] = []
    for sheet in sheet_names:
        parts.append(
            load_sheet_samples(
                xlsx_path=xlsx_path,
                sheet_name=sheet,
                fan_center_xy=fan_center_xy,
                sigma_fallback=sigma_fallback,
                sigma_min=sigma_min,
            )
        )

    table = pd.concat(parts, axis=0, ignore_index=True)
    if table.empty:
        raise ValueError("Training table is empty after loading all sheets.")
    return table


def fit_gp_model(
    train_df: pd.DataFrame,
    feature_mode: str,
    kernel_family: str,
    alpha_scale: float,
    fan_center_xy: Tuple[float, float],
    n_restarts_optimizer: int,
    random_state: int,
) -> GPModelBundle:
    """
    Fit a Gaussian Process model to training samples.
    """
    x_m = train_df["x_m"].to_numpy(dtype=float)
    y_m = train_df["y_m"].to_numpy(dtype=float)
    z_m = train_df["z_m"].to_numpy(dtype=float)
    w_obs = train_df["w_obs_mps"].to_numpy(dtype=float)
    sigma = np.maximum(train_df["sigma_mps"].to_numpy(dtype=float), float(SIGMA_MIN))

    features = build_feature_matrix(
        x_m=x_m,
        y_m=y_m,
        z_m=z_m,
        feature_mode=feature_mode,
        fan_center_xy=fan_center_xy,
    )
    alpha = np.maximum(float(alpha_scale) * (sigma**2), float(ALPHA_JITTER))

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    n_features = int(features.shape[1])
    kernel = build_gp_kernel(n_features=n_features, kernel_family=kernel_family)

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=True,
        n_restarts_optimizer=int(n_restarts_optimizer),
        random_state=int(random_state),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        gp.fit(features_scaled, w_obs)

    return GPModelBundle(
        gp=gp,
        scaler=scaler,
        feature_mode=str(feature_mode).strip().lower(),
        fan_center_xy=fan_center_xy,
    )


def normalize_feature_mode_name(feature_mode: str) -> str:
    """
    Normalize and validate feature mode token.
    """
    mode = str(feature_mode).strip().lower()
    if mode not in {"polar", "radial", "cartesian"}:
        raise ValueError(
            f"Invalid FEATURE_MODE '{feature_mode}'. Use: polar, radial, cartesian."
        )
    return mode


def normalize_kernel_family_name(kernel_family: str) -> str:
    """
    Normalize and validate kernel-family token.
    """
    family = str(kernel_family).strip().lower()
    if family not in {"rbf_ard", "matern32_ard", "matern52_ard", "rq"}:
        raise ValueError(
            f"Invalid KERNEL_FAMILY '{kernel_family}'. "
            "Use: rbf_ard, matern32_ard, matern52_ard, rq."
        )
    return family


def build_default_tune_candidates() -> List[GPTuneCandidate]:
    """
    Build a default auto-tune grid anchored to FEATURE_MODE.
    """
    mode = normalize_feature_mode_name(FEATURE_MODE)
    base_kernel = normalize_kernel_family_name(KERNEL_FAMILY)
    base_alpha = float(ALPHA_SCALE)
    if not np.isfinite(base_alpha) or base_alpha <= 0.0:
        raise ValueError("ALPHA_SCALE must be a finite positive number.")

    kernels: List[str] = [base_kernel]
    for family in AUTO_TUNE_DEFAULT_KERNEL_FAMILIES:
        fam = normalize_kernel_family_name(family)
        if fam not in kernels:
            kernels.append(fam)

    alpha_scales: List[float] = [base_alpha]
    for alpha_scale in AUTO_TUNE_DEFAULT_ALPHA_SCALES:
        alpha = float(alpha_scale)
        if not np.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("AUTO_TUNE_DEFAULT_ALPHA_SCALES must be finite and > 0.")
        if all(abs(alpha - existing) > 1e-12 for existing in alpha_scales):
            alpha_scales.append(alpha)

    candidates: List[GPTuneCandidate] = []
    for kernel_family in kernels:
        for alpha_scale in alpha_scales:
            if (
                kernel_family == base_kernel
                and abs(alpha_scale - base_alpha) <= 1e-12
            ):
                name = "baseline"
            else:
                alpha_tag = f"{alpha_scale:.3g}".replace(".", "p")
                name = f"{mode}_{kernel_family}_a{alpha_tag}"
            candidates.append(
                GPTuneCandidate(
                    name=name,
                    feature_mode=mode,
                    kernel_family=kernel_family,
                    alpha_scale=alpha_scale,
                )
            )

    return candidates


def build_tune_candidates() -> List[GPTuneCandidate]:
    """
    Build GP tuning candidates.
    """
    candidates: List[GPTuneCandidate] = []
    for row in AUTO_TUNE_CANDIDATES:
        if len(row) != 4:
            raise ValueError(
                "Each AUTO_TUNE_CANDIDATES row must be "
                "(name, feature_mode, kernel_family, alpha_scale)."
            )
        name, feature_mode, kernel_family, alpha_scale = row
        alpha = float(alpha_scale)
        if not np.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("AUTO_TUNE_CANDIDATES alpha_scale must be finite and > 0.")
        candidates.append(
            GPTuneCandidate(
                name=str(name),
                feature_mode=normalize_feature_mode_name(feature_mode),
                kernel_family=normalize_kernel_family_name(kernel_family),
                alpha_scale=alpha,
            )
        )

    if candidates:
        return candidates
    return build_default_tune_candidates()


def evaluate_candidate_group_cv(
    train_df: pd.DataFrame,
    candidate: GPTuneCandidate,
    fan_center_xy: Tuple[float, float],
    n_splits: int,
    n_restarts_optimizer: int,
    random_state: int,
) -> Tuple[Dict[str, float], int]:
    """
    Evaluate one candidate with grouped CV over sheet names.
    """
    groups = train_df["sheet"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        raise ValueError("Need at least two unique sheets for grouped CV.")

    n_splits_eff = int(np.clip(int(n_splits), 2, int(unique_groups.size)))
    splitter = GroupKFold(n_splits=n_splits_eff)

    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []
    sigma_all: List[np.ndarray] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(splitter.split(train_df, groups=groups)):
        train_fold = train_df.iloc[tr_idx].reset_index(drop=True)
        val_fold = train_df.iloc[va_idx].reset_index(drop=True)

        model_fold = fit_gp_model(
            train_df=train_fold,
            feature_mode=candidate.feature_mode,
            kernel_family=candidate.kernel_family,
            alpha_scale=candidate.alpha_scale,
            fan_center_xy=fan_center_xy,
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=int(random_state) + int(fold_idx),
        )

        y_pred_fold, _ = model_fold.predict(
            x_m=val_fold["x_m"].to_numpy(dtype=float),
            y_m=val_fold["y_m"].to_numpy(dtype=float),
            z_m=val_fold["z_m"].to_numpy(dtype=float),
            return_std=False,
        )
        y_true_fold = val_fold["w_obs_mps"].to_numpy(dtype=float)
        sigma_fold = val_fold["sigma_mps"].to_numpy(dtype=float)

        y_true_all.append(y_true_fold)
        y_pred_all.append(y_pred_fold)
        sigma_all.append(sigma_fold)

    metrics = compute_regression_metrics(
        y_true=np.concatenate(y_true_all),
        y_pred=np.concatenate(y_pred_all),
        sigma_mps=np.concatenate(sigma_all),
    )
    return metrics, n_splits_eff


def select_gp_candidate(
    train_df: pd.DataFrame,
    fan_center_xy: Tuple[float, float],
) -> Tuple[GPTuneCandidate, List[GPTuneTrial]]:
    """
    Run grouped-CV tuning and select the best GP candidate.
    """
    candidates = build_tune_candidates()
    trials: List[GPTuneTrial] = []

    for idx, candidate in enumerate(candidates):
        try:
            cv_metrics, n_folds = evaluate_candidate_group_cv(
                train_df=train_df,
                candidate=candidate,
                fan_center_xy=fan_center_xy,
                n_splits=AUTO_TUNE_CV_N_SPLITS,
                n_restarts_optimizer=AUTO_TUNE_CV_RESTARTS_OPTIMIZER,
                random_state=int(RANDOM_STATE) + 17 * idx,
            )
            trials.append(
                GPTuneTrial(
                    candidate=candidate,
                    cv_metrics=cv_metrics,
                    n_folds=n_folds,
                )
            )
        except Exception as exc:
            print(f"Auto-tune candidate '{candidate.name}' failed: {exc}")

    if not trials:
        selected = GPTuneCandidate(
            name="fallback",
            feature_mode=normalize_feature_mode_name(FEATURE_MODE),
            kernel_family=normalize_kernel_family_name(KERNEL_FAMILY),
            alpha_scale=float(ALPHA_SCALE),
        )
        return selected, []

    min_wrmse = min(float(t.cv_metrics["wrmse_mps"]) for t in trials)
    finalists = [
        t for t in trials if float(t.cv_metrics["wrmse_mps"]) <= min_wrmse + float(AUTO_TUNE_RMSE_TIE_TOL)
    ]
    best_trial = min(
        finalists,
        key=lambda t: (
            float(t.cv_metrics["mae_mps"]),
            float(t.cv_metrics["rmse_mps"]),
        ),
    )

    print("Auto-tuning GP candidates (grouped CV by sheet):")
    print(" name               mode       kernel        alpha  CV_WRMSE  CV_RMSE   CV_MAE")
    for trial in trials:
        marker = "*" if trial.candidate.name == best_trial.candidate.name else " "
        c = trial.candidate
        m = trial.cv_metrics
        print(
            f"{marker}{c.name:17s}  {c.feature_mode:9s}  {c.kernel_family:12s}  "
            f"{c.alpha_scale:5.2f}  {m['wrmse_mps']:8.5f}  {m['rmse_mps']:8.5f}  {m['mae_mps']:8.5f}"
        )
    print(f"Selected GP candidate: {best_trial.candidate.name}")

    return best_trial.candidate, trials


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sigma_mps: np.ndarray | None = None,
) -> Dict[str, float]:
    """
    Compute regression metrics (MAE, RMSE, R2, SAE, SSE, WRMSE).

    WRMSE is computed with inverse-variance weights:
        w_i = 1 / sigma_i^2
    When sigma_mps is not provided, uniform weights are used.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true

    # Aggregate-error metrics requested for export.
    sae = float(np.sum(np.abs(err)))
    sse = float(np.sum(err**2))

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = sse
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")

    if sigma_mps is None:
        weights = np.ones_like(err, dtype=float)
    else:
        sigma_arr = np.asarray(sigma_mps, dtype=float)
        if sigma_arr.size != err.size:
            raise ValueError("sigma_mps must have the same length as y_true/y_pred.")
        sigma_safe = np.maximum(sigma_arr, float(SIGMA_MIN))
        weights = 1.0 / (sigma_safe**2)

    w_sum = float(np.sum(weights))
    wrmse = float(np.sqrt(np.sum(weights * (err**2)) / w_sum)) if w_sum > 0.0 else float("nan")

    return {
        "mae_mps": mae,
        "rmse_mps": rmse,
        "sae_mps": sae,
        "sse_mps2": sse,
        "wrmse_mps": wrmse,
        "r2": r2,
        "n_samples": float(y_true.size),
    }


def make_training_prediction_table(
    model: GPModelBundle,
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Predict at training points and return an augmented table.
    """
    w_pred, w_std = model.predict(
        x_m=train_df["x_m"].to_numpy(dtype=float),
        y_m=train_df["y_m"].to_numpy(dtype=float),
        z_m=train_df["z_m"].to_numpy(dtype=float),
        return_std=True,
    )
    out = train_df.copy()
    out["w_pred_mps"] = w_pred
    out["w_pred_std_mps"] = w_std
    out["err_mps"] = out["w_pred_mps"] - out["w_obs_mps"]
    out["abs_err_mps"] = np.abs(out["err_mps"])
    return out


def summarize_by_sheet(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute metrics separately for each height sheet.
    """
    rows: List[Dict[str, float]] = []
    for sheet in SHEETS:
        sub = pred_df[pred_df["sheet"] == sheet]
        if sub.empty:
            continue

        metrics = compute_regression_metrics(
            y_true=sub["w_obs_mps"].to_numpy(dtype=float),
            y_pred=sub["w_pred_mps"].to_numpy(dtype=float),
            sigma_mps=sub["sigma_mps"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "sheet": sheet,
                "z_m": float(sub["z_m"].iloc[0]),
                "n_samples": int(sub.shape[0]),
                "mae_mps": metrics["mae_mps"],
                "rmse_mps": metrics["rmse_mps"],
                "sae_mps": metrics["sae_mps"],
                "sse_mps2": metrics["sse_mps2"],
                "wrmse_mps": metrics["wrmse_mps"],
                "r2": metrics["r2"],
                "mean_pred_std_mps": float(
                    np.mean(sub["w_pred_std_mps"].to_numpy(dtype=float))
                ),
            }
        )

    return pd.DataFrame(rows)


def build_analysis_style_metrics_table(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build an analysis-style table (7 heights + TOTAL) for SAE/SSE/WRMSE.

    Format mirrors single_fan_annular_gaussian_avg_analysis.py.
    """
    rows: List[Dict[str, float]] = []
    total_sae = 0.0
    total_sse = 0.0
    total_weighted_sse = 0.0
    total_weight_sum = 0.0
    total_n = 0

    for sheet in SHEETS:
        sub = pred_df[pred_df["sheet"] == sheet]
        if sub.empty:
            continue

        err = sub["err_mps"].to_numpy(dtype=float)
        sigma = np.maximum(sub["sigma_mps"].to_numpy(dtype=float), float(SIGMA_MIN))
        weights = 1.0 / (sigma**2)

        sae_k = float(np.sum(np.abs(err)))
        sse_k = float(np.sum(err**2))
        weighted_sse_k = float(np.sum(weights * (err**2)))
        weight_sum_k = float(np.sum(weights))
        wrmse_k = (
            float(np.sqrt(weighted_sse_k / weight_sum_k))
            if weight_sum_k > 0.0
            else float("nan")
        )

        total_sae += sae_k
        total_sse += sse_k
        total_weighted_sse += weighted_sse_k
        total_weight_sum += weight_sum_k
        total_n += int(sub.shape[0])

        rows.append(
            {
                "sheet": sheet,
                "z_m": float(sub["z_m"].iloc[0]),
                "n_samples": int(sub.shape[0]),
                "accumulate_SAE_mps": sae_k,
                "SSE_mps2": sse_k,
                "weighted_RMSE_mps": wrmse_k,
                "weighted_SSE_term": weighted_sse_k,
                "weight_sum_term": weight_sum_k,
            }
        )

    if total_weight_sum <= 0.0:
        raise ValueError("Total WRMSE denominator is non-positive.")
    total_wrmse = float(np.sqrt(total_weighted_sse / total_weight_sum))

    rows.append(
        {
            "sheet": "TOTAL",
            "z_m": np.nan,
            "n_samples": int(total_n),
            "accumulate_SAE_mps": float(total_sae),
            "SSE_mps2": float(total_sse),
            "weighted_RMSE_mps": total_wrmse,
            "weighted_SSE_term": float(total_weighted_sse),
            "weight_sum_term": float(total_weight_sum),
        }
    )
    return pd.DataFrame(rows)


def build_four_core_strength_table(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-height four-core strength table:
      - one row per fixed outlet centre (F01..F04)
      - one TOTAL row (sum of outlet core strengths)
    """
    rows: List[Dict[str, float | str]] = []
    for sheet in SHEETS:
        sub = pred_df[pred_df["sheet"] == sheet]
        if sub.empty:
            continue

        z_m = float(sub["z_m"].iloc[0])
        x = sub["x_m"].to_numpy(dtype=float)
        y = sub["y_m"].to_numpy(dtype=float)
        w_obs = sub["w_obs_mps"].to_numpy(dtype=float)
        w_pred = sub["w_pred_mps"].to_numpy(dtype=float)

        outlet_strength_obs: List[float] = []
        outlet_strength_pred: List[float] = []
        n_core_total = 0

        for idx, (cx, cy) in enumerate(FOUR_FAN_CENTERS_XY, start=1):
            in_core = (x - cx) ** 2 + (y - cy) ** 2 <= FOUR_FAN_CORE_RADIUS_M**2
            n_core = int(np.sum(in_core))
            n_core_total += n_core

            if n_core > 0:
                core_strength_obs = float(np.max(w_obs[in_core]))
                core_strength_pred = float(np.max(w_pred[in_core]))
                core_mean_obs = float(np.mean(w_obs[in_core]))
                core_mean_pred = float(np.mean(w_pred[in_core]))
            else:
                core_strength_obs = float("nan")
                core_strength_pred = float("nan")
                core_mean_obs = float("nan")
                core_mean_pred = float("nan")

            if np.isfinite(core_strength_obs):
                outlet_strength_obs.append(core_strength_obs)
            if np.isfinite(core_strength_pred):
                outlet_strength_pred.append(core_strength_pred)

            rows.append(
                {
                    "sheet": sheet,
                    "z_m": z_m,
                    "outlet_id": f"F{idx:02d}",
                    "outlet_x_m": float(cx),
                    "outlet_y_m": float(cy),
                    "core_radius_m": float(FOUR_FAN_CORE_RADIUS_M),
                    "n_core_samples": n_core,
                    "core_strength_obs_mps": core_strength_obs,
                    "core_strength_pred_mps": core_strength_pred,
                    "core_mean_obs_mps": core_mean_obs,
                    "core_mean_pred_mps": core_mean_pred,
                }
            )

        rows.append(
            {
                "sheet": sheet,
                "z_m": z_m,
                "outlet_id": "TOTAL",
                "outlet_x_m": np.nan,
                "outlet_y_m": np.nan,
                "core_radius_m": float(FOUR_FAN_CORE_RADIUS_M),
                "n_core_samples": int(n_core_total),
                "core_strength_obs_mps": float(np.sum(outlet_strength_obs))
                if outlet_strength_obs
                else np.nan,
                "core_strength_pred_mps": float(np.sum(outlet_strength_pred))
                if outlet_strength_pred
                else np.nan,
                "core_mean_obs_mps": float(np.mean(outlet_strength_obs))
                if outlet_strength_obs
                else np.nan,
                "core_mean_pred_mps": float(np.mean(outlet_strength_pred))
                if outlet_strength_pred
                else np.nan,
            }
        )

    return pd.DataFrame(rows)


def make_grid_prediction_tables(
    model: GPModelBundle,
    xlsx_path: str,
    sheet_names: Sequence[str],
) -> Dict[str, pd.DataFrame]:
    """
    Predict GP mean/std on each measured x-y grid at its corresponding z.
    """
    tables: Dict[str, pd.DataFrame] = {}
    for sheet in sheet_names:
        z_m = parse_sheet_height_m(sheet)
        x_centers, y_centers, _w_obs = read_slice_from_sheet(xlsx_path, sheet)
        x_grid, y_grid = np.meshgrid(x_centers, y_centers)
        z_grid = np.full_like(x_grid, z_m, dtype=float)

        w_mean, w_std = model.predict(
            x_m=x_grid.ravel(),
            y_m=y_grid.ravel(),
            z_m=z_grid.ravel(),
            return_std=True,
        )
        mean_map = w_mean.reshape(x_grid.shape)
        std_map = w_std.reshape(x_grid.shape)

        mean_df = pd.DataFrame(mean_map, index=y_centers, columns=x_centers)
        std_df = pd.DataFrame(std_map, index=y_centers, columns=x_centers)
        mean_df.index.name = "y/x"
        std_df.index.name = "y/x"

        tables[f"{sheet}_gp_mean"] = mean_df
        tables[f"{sheet}_gp_std"] = std_df

    return tables


def write_tables_to_excel(
    out_xlsx: str | Path,
    tables: Dict[str, pd.DataFrame],
) -> None:
    """
    Write multiple tables to Excel
    """
    out_path = ensure_path(out_xlsx)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        with pd.ExcelWriter(
            out_path,
            engine="openpyxl",
            mode="a",
            if_sheet_exists="replace",
        ) as writer:
            for sheet_name, table in tables.items():
                table.to_excel(writer, index=True, sheet_name=sheet_name)
    else:
        with pd.ExcelWriter(out_path, engine="openpyxl", mode="w") as writer:
            for sheet_name, table in tables.items():
                table.to_excel(writer, index=True, sheet_name=sheet_name)


def write_table_to_excel_no_index(
    out_xlsx: str | Path,
    table: pd.DataFrame,
    sheet_name: str,
) -> None:
    """
    Write one table to Excel using analysis-style formatting (index=False).
    """
    out_path = ensure_path(out_xlsx)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        with pd.ExcelWriter(
            out_path,
            engine="openpyxl",
            mode="a",
            if_sheet_exists="replace",
        ) as writer:
            table.to_excel(writer, index=False, sheet_name=sheet_name)
    else:
        with pd.ExcelWriter(out_path, engine="openpyxl", mode="w") as writer:
            table.to_excel(writer, index=False, sheet_name=sheet_name)


### Main
def main() -> None:
    out_dir = ensure_path(OUT_DIR)
    train_pred_csv_path = ensure_path(TRAIN_PRED_CSV_PATH)
    summary_xlsx_path = ensure_path(SUMMARY_XLSX_PATH)
    analysis_xlsx_path = ensure_path(ANALYSIS_XLSX_PATH)
    grid_pred_xlsx_path = ensure_path(GRID_PRED_XLSX_PATH)

    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = build_training_table(
        xlsx_path=XLSX_PATH,
        sheet_names=SHEETS,
        fan_center_xy=FAN_CENTER_XY,
        sigma_fallback=SIGMA_FALLBACK,
        sigma_min=SIGMA_MIN,
    )

    if ENABLE_AUTO_TUNE:
        selected_candidate, tune_trials = select_gp_candidate(
            train_df=train_df,
            fan_center_xy=FAN_CENTER_XY,
        )
    else:
        selected_candidate = GPTuneCandidate(
            name="manual",
            feature_mode=normalize_feature_mode_name(FEATURE_MODE),
            kernel_family=normalize_kernel_family_name(KERNEL_FAMILY),
            alpha_scale=float(ALPHA_SCALE),
        )
        tune_trials = []

    model = fit_gp_model(
        train_df=train_df,
        feature_mode=selected_candidate.feature_mode,
        kernel_family=selected_candidate.kernel_family,
        alpha_scale=selected_candidate.alpha_scale,
        fan_center_xy=FAN_CENTER_XY,
        n_restarts_optimizer=N_RESTARTS_OPTIMIZER,
        random_state=RANDOM_STATE,
    )

    pred_df = make_training_prediction_table(model, train_df)
    pred_df.to_csv(train_pred_csv_path, index=False)

    overall_metrics = compute_regression_metrics(
        y_true=pred_df["w_obs_mps"].to_numpy(dtype=float),
        y_pred=pred_df["w_pred_mps"].to_numpy(dtype=float),
        sigma_mps=pred_df["sigma_mps"].to_numpy(dtype=float),
    )
    summary_metrics_df = pd.DataFrame([overall_metrics])
    summary_metrics_df["feature_mode"] = selected_candidate.feature_mode
    summary_metrics_df["kernel_family"] = selected_candidate.kernel_family
    summary_metrics_df["alpha_scale"] = float(selected_candidate.alpha_scale)
    summary_metrics_df["autotune_enabled"] = bool(ENABLE_AUTO_TUNE)
    summary_metrics_df["autotune_selected"] = str(selected_candidate.name)
    summary_metrics_df["use_radial_features"] = (
        selected_candidate.feature_mode == "radial"
    )
    summary_metrics_df["log_marginal_likelihood"] = float(
        model.gp.log_marginal_likelihood_value_
    )
    summary_metrics_df["kernel"] = str(model.gp.kernel_)

    per_sheet_df = summarize_by_sheet(pred_df)
    analysis_df = build_analysis_style_metrics_table(pred_df)
    core_strength_df = build_four_core_strength_table(pred_df)
    fan_specs_df = pd.DataFrame(
        [
            {
                "outlet_id": f"F{idx:02d}",
                "x_m": float(cx),
                "y_m": float(cy),
                "core_radius_m": float(FOUR_FAN_CORE_RADIUS_M),
            }
            for idx, (cx, cy) in enumerate(FOUR_FAN_CENTERS_XY, start=1)
        ]
    )
    hyper_df = pd.DataFrame(
        [
            {"parameter": "xlsx_path", "value": XLSX_PATH},
            {"parameter": "sheet_count", "value": len(SHEETS)},
            {"parameter": "fan_center_x_m", "value": float(FAN_CENTER_XY[0])},
            {"parameter": "fan_center_y_m", "value": float(FAN_CENTER_XY[1])},
            {"parameter": "four_fan_centers_xy", "value": str(FOUR_FAN_CENTERS_XY)},
            {"parameter": "four_fan_core_radius_m", "value": float(FOUR_FAN_CORE_RADIUS_M)},
            {
                "parameter": "sigma_mapping_method",
                "value": "xy_linear_inside_hull_nearest_outside",
            },
            {"parameter": "feature_mode", "value": selected_candidate.feature_mode},
            {"parameter": "kernel_family", "value": selected_candidate.kernel_family},
            {"parameter": "alpha_scale", "value": float(selected_candidate.alpha_scale)},
            {"parameter": "autotune_enabled", "value": bool(ENABLE_AUTO_TUNE)},
            {"parameter": "autotune_selected", "value": str(selected_candidate.name)},
            {
                "parameter": "use_radial_features",
                "value": selected_candidate.feature_mode == "radial",
            },
            {"parameter": "sigma_fallback_mps", "value": float(SIGMA_FALLBACK)},
            {"parameter": "sigma_min_mps", "value": float(SIGMA_MIN)},
            {"parameter": "n_restarts_optimizer", "value": int(N_RESTARTS_OPTIMIZER)},
            {"parameter": "random_state", "value": int(RANDOM_STATE)},
            {"parameter": "autotune_cv_n_splits", "value": int(AUTO_TUNE_CV_N_SPLITS)},
            {
                "parameter": "autotune_cv_restarts_optimizer",
                "value": int(AUTO_TUNE_CV_RESTARTS_OPTIMIZER),
            },
            {"parameter": "kernel_fitted", "value": str(model.gp.kernel_)},
        ]
    )
    if tune_trials:
        cv_rows: List[Dict[str, float]] = []
        for trial in tune_trials:
            row = {
                "name": trial.candidate.name,
                "feature_mode": trial.candidate.feature_mode,
                "kernel_family": trial.candidate.kernel_family,
                "alpha_scale": float(trial.candidate.alpha_scale),
                "n_folds": int(trial.n_folds),
                "cv_mae_mps": float(trial.cv_metrics["mae_mps"]),
                "cv_rmse_mps": float(trial.cv_metrics["rmse_mps"]),
                "cv_wrmse_mps": float(trial.cv_metrics["wrmse_mps"]),
                "cv_sae_mps": float(trial.cv_metrics["sae_mps"]),
                "cv_sse_mps2": float(trial.cv_metrics["sse_mps2"]),
                "cv_r2": float(trial.cv_metrics["r2"]),
                "is_selected": trial.candidate.name == selected_candidate.name,
            }
            cv_rows.append(row)
        autotune_cv_df = pd.DataFrame(cv_rows).sort_values(
            by=["cv_wrmse_mps", "cv_mae_mps"], ascending=[True, True]
        )
    else:
        autotune_cv_df = pd.DataFrame(
            [
                {
                    "name": selected_candidate.name,
                    "feature_mode": selected_candidate.feature_mode,
                    "kernel_family": selected_candidate.kernel_family,
                    "alpha_scale": float(selected_candidate.alpha_scale),
                    "n_folds": np.nan,
                    "cv_mae_mps": np.nan,
                    "cv_rmse_mps": np.nan,
                    "cv_wrmse_mps": np.nan,
                    "cv_sae_mps": np.nan,
                    "cv_sse_mps2": np.nan,
                    "cv_r2": np.nan,
                    "is_selected": True,
                }
            ]
        )

    write_tables_to_excel(
        summary_xlsx_path,
        {
            "overall_metrics": summary_metrics_df,
            "per_sheet_metrics": per_sheet_df,
            "fan_specs": fan_specs_df,
            "hyperparameters": hyper_df,
            "autotune_cv": autotune_cv_df,
        },
    )
    # Add analysis-style SAE/SSE/WRMSE table into summary workbook.
    write_table_to_excel_no_index(
        summary_xlsx_path,
        analysis_df,
        ANALYSIS_SHEET_NAME,
    )
    write_table_to_excel_no_index(
        summary_xlsx_path,
        core_strength_df,
        CORE_STRENGTH_SHEET_NAME,
    )
    # Also export the same analysis table independently.
    write_table_to_excel_no_index(
        analysis_xlsx_path,
        analysis_df,
        ANALYSIS_SHEET_NAME,
    )
    write_table_to_excel_no_index(
        analysis_xlsx_path,
        core_strength_df,
        CORE_STRENGTH_SHEET_NAME,
    )

    grid_tables = make_grid_prediction_tables(
        model=model,
        xlsx_path=XLSX_PATH,
        sheet_names=SHEETS,
    )
    write_tables_to_excel(grid_pred_xlsx_path, grid_tables)

    print("Gaussian Process model fitted successfully.")
    print(f"Samples used: {int(train_df.shape[0])}")
    print(
        "Selected GP config: "
        f"name={selected_candidate.name}, "
        f"feature_mode={selected_candidate.feature_mode}, "
        f"kernel_family={selected_candidate.kernel_family}, "
        f"alpha_scale={selected_candidate.alpha_scale:.2f}"
    )
    print(f"Fitted kernel: {model.gp.kernel_}")
    print(
        "Overall metrics: "
        f"MAE={overall_metrics['mae_mps']:.4f} m/s, "
        f"RMSE={overall_metrics['rmse_mps']:.4f} m/s, "
        f"R2={overall_metrics['r2']:.4f}"
    )
    print(f"Training predictions CSV: {train_pred_csv_path.resolve()}")
    print(f"Summary workbook: {summary_xlsx_path.resolve()}")
    print(f"Analysis workbook: {analysis_xlsx_path.resolve()}")
    print(f"Grid predictions workbook: {grid_pred_xlsx_path.resolve()}")


if __name__ == "__main__":
    main()
