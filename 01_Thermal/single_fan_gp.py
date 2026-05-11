from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
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

from single_fan_annuli_cut import (
    assign_sigma_bins_nearest,
    assign_sigma_points_linear_nearest,
    parse_ts_points_and_sigmas,
    parse_ts_xy_points_and_sigmas,
    read_slice_from_sheet,
)


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) GP Training Configuration and Data Sources
# 2) Data Containers
# 3) GP Feature Engineering and Diagnostics
# 4) Training and Diagnostic Export
# =============================================================================

# =============================================================================
# 1) GP Training Configuration and Data Sources
# =============================================================================

XLSX_PATH = "S01.xlsx"
SHEETS = ["z020", "z035", "z050", "z075", "z110", "z160", "z220"]

# Fan centre (x_c, y_c) in arena metres.
FAN_CENTER_XY = (4.2, 2.4)
SHEET_HEIGHT_DIVISOR = 100.0

# Feature mode:
#   "polar"     -> use (r, cos(theta), sin(theta), z): annular + non-axisymmetric
#   "radial"    -> use (r, z): axisymmetric annular
#   "cartesian" -> use (x, y, z): fully unconstrained spatial structure
FEATURE_MODE = "cartesian"
KERNEL_FAMILY = "rbf_ard"
ALPHA_SCALE = 1.0

# Noise assignment from *_TS.
SIGMA_FALLBACK = 0.14
SIGMA_MIN = 0.03
ALPHA_JITTER = 1e-8

# GP optimizer settings.
N_RESTARTS_OPTIMIZER = 10
RANDOM_STATE = 42

# Automatic GP configuration tuning (grouped CV by sheet height).
ENABLE_AUTO_TUNE = True
AUTO_TUNE_CV_N_SPLITS = 3
AUTO_TUNE_CV_RESTARTS_OPTIMIZER = 1
AUTO_TUNE_RMSE_TIE_TOL = 1e-4

# Candidate format: (name, feature_mode, kernel_family, alpha_scale)
# If empty, a default candidate grid is built from FEATURE_MODE plus the
# kernel/alpha settings below.
AUTO_TUNE_CANDIDATES = ()
AUTO_TUNE_DEFAULT_KERNEL_FAMILIES = ("rbf_ard", "matern32_ard", "matern52_ard")
AUTO_TUNE_DEFAULT_ALPHA_SCALES = (1.0, 1.3)

# Output locations.
OUT_DIR = Path("B_results/Single_Fan_GP")
TRAIN_PRED_CSV_PATH = OUT_DIR / "single_gp_training_predictions.csv"
SUMMARY_XLSX_PATH = OUT_DIR / "single_gp_summary.xlsx"
GRID_PRED_XLSX_PATH = OUT_DIR / "single_gp_grid_predictions.xlsx"
ANALYSIS_XLSX_PATH = "B_results/single_gp_analysis.xlsx"
ANALYSIS_SHEET_NAME = "single_gp_analysis"

# =============================================================================
# 2) Data Containers
# =============================================================================

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


# =============================================================================
# 3) GP Feature Engineering and Diagnostics
# =============================================================================

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
    fan_center_xy: Tuple[float, float],
) -> Tuple[np.ndarray, Tuple[Tuple[np.ndarray, np.ndarray] | None, ...]]:
    """
    Cache radial representative fluctuation data for all measured z planes.
    """
    z_list: List[float] = []
    parsed_list: List[Tuple[np.ndarray, np.ndarray] | None] = []

    for sheet_name in sheet_names:
        z_list.append(parse_sheet_height_m(sheet_name))
        parsed_list.append(
            parse_ts_points_and_sigmas(
                xlsx_path=xlsx_path,
                ts_sheet_name=f"{sheet_name}_TS",
                fan_center_xy=fan_center_xy,
            )
        )

    order = np.argsort(np.asarray(z_list, dtype=float))
    z_axis = np.asarray([z_list[idx] for idx in order], dtype=float)
    parsed_sorted = tuple(parsed_list[idx] for idx in order)
    return z_axis, parsed_sorted


def evaluate_sigma_points_annular_pchip_z(
    xlsx_path: str,
    sheet_names: Sequence[str],
    fan_center_xy: Tuple[float, float],
    x_pts: np.ndarray,
    y_pts: np.ndarray,
    z_pts: np.ndarray,
    sigma_fallback: float,
    sigma_min: float,
) -> np.ndarray:
    """
    Evaluate annular empirical fluctuation sigma continuously in z.

    Each measured plane uses nearest-radius mapping from representative TS
    points relative to the fan centre. The per-point values are then blended
    across z with PCHIP while holding the nearest measured plane outside the
    measured range.
    """
    x_arr = np.asarray(x_pts, dtype=float).ravel()
    y_arr = np.asarray(y_pts, dtype=float).ravel()
    z_arr = np.asarray(z_pts, dtype=float).ravel()
    if not (x_arr.size == y_arr.size == z_arr.size):
        raise ValueError("x_pts, y_pts, z_pts must have the same number of samples.")

    xc, yc = fan_center_xy
    r_query = np.sqrt((x_arr - float(xc)) ** 2 + (y_arr - float(yc)) ** 2)
    z_axis, parsed_planes = load_ts_radial_sigma_planes(
        str(xlsx_path),
        tuple(str(sheet) for sheet in sheet_names),
        (float(xc), float(yc)),
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

        r_points, sigma_points = parsed
        sigma_planes[idx] = assign_sigma_bins_nearest(
            r_bins=r_query,
            r_points=r_points,
            sigma_points=sigma_points,
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

    xc, yc = fan_center_xy
    r_pts = np.sqrt((x_pts - xc) ** 2 + (y_pts - yc) ** 2)

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
    ts_parsed = parse_ts_xy_points_and_sigmas(
        xlsx_path=xlsx_path,
        ts_sheet_name=ts_sheet,
    )

    if ts_parsed is None:
        sigma_pts = np.full_like(r_pts, float(sigma_fallback), dtype=float)
        sigma_pts = np.maximum(sigma_pts, float(sigma_min))
    else:
        rep_x, rep_y, rep_sigma = ts_parsed
        sigma_pts = assign_sigma_points_linear_nearest(
            x_pts=x_pts,
            y_pts=y_pts,
            rep_x=rep_x,
            rep_y=rep_y,
            rep_sigma=rep_sigma,
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


# =============================================================================
# 4) Training and Diagnostic Export
# =============================================================================

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
    hyper_df = pd.DataFrame(
        [
            {"parameter": "xlsx_path", "value": XLSX_PATH},
            {"parameter": "sheet_count", "value": len(SHEETS)},
            {"parameter": "fan_center_x_m", "value": float(FAN_CENTER_XY[0])},
            {"parameter": "fan_center_y_m", "value": float(FAN_CENTER_XY[1])},
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
            "hyperparameters": hyper_df,
            "autotune_cv": autotune_cv_df,
        },
    )
    # Summary workbook includes SAE/SSE/WRMSE for cross-model diagnostics.
    write_table_to_excel_no_index(
        summary_xlsx_path,
        analysis_df,
        ANALYSIS_SHEET_NAME,
    )
    # Also export the same analysis table independently.
    write_table_to_excel_no_index(
        analysis_xlsx_path,
        analysis_df,
        ANALYSIS_SHEET_NAME,
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
