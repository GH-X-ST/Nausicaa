from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Callable, Dict, List, Protocol, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd
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


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Shared GP Constants
# 2) Data Containers
# 3) Feature Engineering and GP Training
# =============================================================================

# =============================================================================
# 1) Shared GP Constants
# =============================================================================


FOUR_FAN_ID_PATTERN = re.compile(r"^a0_(F\d{2})$")
DEFAULT_LENGTH_SCALE_UPPER = 1e2
DEFAULT_NOISE_LEVEL_BOUNDS = (1e-6, 1e0)
DEFAULT_SIGNAL_BOUNDS = (1e-3, 1e3)
DEFAULT_LENGTH_SCALE_FLOORS = {
    "cartesian": np.asarray([0.55, 0.55, 0.25], dtype=float),
    "polar": np.asarray([0.55, 0.45, 0.45, 0.25], dtype=float),
    "radial": np.asarray([0.55, 0.25], dtype=float),
}

# =============================================================================
# 2) Data Containers
# =============================================================================


class MeanModelProtocol(Protocol):
    source_path: Path
    sheet_name: str

    def evaluate(
        self,
        x_m: np.ndarray,
        y_m: np.ndarray,
        z_m: np.ndarray,
    ) -> np.ndarray:
        ...


@dataclass
class ResidualGPModelBundle:
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
        features = build_feature_matrix(
            x_m=x_m,
            y_m=y_m,
            z_m=z_m,
            feature_mode=self.feature_mode,
            fan_center_xy=self.fan_center_xy,
        )
        features_scaled = self.scaler.transform(features)

        if return_std:
            w_mean, w_std = self.gp.predict(features_scaled, return_std=True)
            return w_mean.astype(float), w_std.astype(float)

        w_mean = self.gp.predict(features_scaled, return_std=False)
        return w_mean.astype(float), np.zeros_like(w_mean, dtype=float)

# =============================================================================
# 3) Feature Engineering and GP Training
# =============================================================================


def normalize_feature_mode_name(feature_mode: str) -> str:
    mode = str(feature_mode).strip().lower()
    if mode not in {"polar", "radial", "cartesian"}:
        raise ValueError(
            f"Invalid FEATURE_MODE '{feature_mode}'. "
            "Use: polar, radial, cartesian."
        )
    return mode


def build_feature_matrix(
    x_m: np.ndarray,
    y_m: np.ndarray,
    z_m: np.ndarray,
    feature_mode: str,
    fan_center_xy: Tuple[float, float],
) -> np.ndarray:
    x_arr = np.asarray(x_m, dtype=float).ravel()
    y_arr = np.asarray(y_m, dtype=float).ravel()
    z_arr = np.asarray(z_m, dtype=float).ravel()

    if not (x_arr.size == y_arr.size == z_arr.size):
        raise ValueError("x_m, y_m, z_m must have the same number of samples.")

    mode = normalize_feature_mode_name(feature_mode)
    if mode in {"polar", "radial"}:
        xc, yc = fan_center_xy
        r_arr = np.hypot(x_arr - float(xc), y_arr - float(yc))
        if mode == "radial":
            return np.column_stack([r_arr, z_arr])

        theta_arr = np.arctan2(y_arr - float(yc), x_arr - float(xc))
        return np.column_stack(
            [r_arr, np.cos(theta_arr), np.sin(theta_arr), z_arr]
        )

    return np.column_stack([x_arr, y_arr, z_arr])


def _build_length_scale_bounds(
    feature_mode: str,
    scaler: StandardScaler,
    length_scale_floors: Dict[str, np.ndarray] | None,
) -> Tuple[np.ndarray, np.ndarray]:
    mode = normalize_feature_mode_name(feature_mode)
    floor_map = (
        DEFAULT_LENGTH_SCALE_FLOORS
        if length_scale_floors is None
        else length_scale_floors
    )
    if mode not in floor_map:
        raise ValueError(f"Missing residual length-scale floors for '{mode}'.")

    floors_raw = np.asarray(floor_map[mode], dtype=float).reshape(-1)
    scale = np.asarray(scaler.scale_, dtype=float).reshape(-1)
    if floors_raw.size != scale.size:
        raise ValueError(
            "Residual length-scale floor count must match feature count."
        )

    scale_safe = np.maximum(scale, 1e-9)
    lower = floors_raw / scale_safe
    upper = np.full_like(lower, float(DEFAULT_LENGTH_SCALE_UPPER))
    lower = np.clip(lower, 0.25, DEFAULT_LENGTH_SCALE_UPPER / 10.0)
    return lower, upper


def build_residual_gp_kernel(
    feature_mode: str,
    kernel_family: str,
    scaler: StandardScaler,
    length_scale_floors: Dict[str, np.ndarray] | None = None,
):
    family = str(kernel_family).strip().lower()
    lower, upper = _build_length_scale_bounds(
        feature_mode=feature_mode,
        scaler=scaler,
        length_scale_floors=length_scale_floors,
    )
    bounds = np.column_stack([lower, upper])
    init = np.maximum(1.0, 1.8 * lower)

    if family == "rbf_ard":
        return (
            ConstantKernel(1.0, DEFAULT_SIGNAL_BOUNDS)
            * RBF(length_scale=init, length_scale_bounds=bounds)
            + WhiteKernel(
                noise_level=1e-3,
                noise_level_bounds=DEFAULT_NOISE_LEVEL_BOUNDS,
            )
        )

    if family == "matern32_ard":
        return (
            ConstantKernel(1.0, DEFAULT_SIGNAL_BOUNDS)
            * Matern(
                length_scale=init,
                length_scale_bounds=bounds,
                nu=1.5,
            )
            + WhiteKernel(
                noise_level=1e-3,
                noise_level_bounds=DEFAULT_NOISE_LEVEL_BOUNDS,
            )
        )

    if family == "matern52_ard":
        return (
            ConstantKernel(1.0, DEFAULT_SIGNAL_BOUNDS)
            * Matern(
                length_scale=init,
                length_scale_bounds=bounds,
                nu=2.5,
            )
            + WhiteKernel(
                noise_level=1e-3,
                noise_level_bounds=DEFAULT_NOISE_LEVEL_BOUNDS,
            )
        )

    if family == "rq":
        length_scale_floor = float(np.min(lower))
        return (
            ConstantKernel(1.0, DEFAULT_SIGNAL_BOUNDS)
            * RationalQuadratic(
                length_scale=max(1.0, 1.8 * length_scale_floor),
                alpha=1.0,
                length_scale_bounds=(
                    length_scale_floor,
                    float(DEFAULT_LENGTH_SCALE_UPPER),
                ),
                alpha_bounds=(1e-2, 1e3),
            )
            + WhiteKernel(
                noise_level=1e-3,
                noise_level_bounds=DEFAULT_NOISE_LEVEL_BOUNDS,
            )
        )

    raise ValueError(
        f"Invalid KERNEL_FAMILY '{kernel_family}'. "
        "Use: rbf_ard, matern32_ard, matern52_ard, rq."
    )


@dataclass(frozen=True)
class _FanModelData:
    fan_id: str
    center_xy: Tuple[float, float]
    harmonic_orders: Tuple[int, ...]
    param_names: Tuple[str, ...]
    param_table: np.ndarray


def _load_numeric_sheet(xlsx_path: Path, sheet_name: str) -> pd.DataFrame:
    if not xlsx_path.exists():
        raise FileNotFoundError(
            f"Missing annular-BEMT parameter workbook: {xlsx_path}"
        )

    xls = pd.ExcelFile(xlsx_path)
    if sheet_name not in xls.sheet_names:
        available = ", ".join(xls.sheet_names)
        raise ValueError(
            f"Missing sheet '{sheet_name}' in '{xlsx_path}'. "
            f"Available sheets: {available}"
        )

    return pd.read_excel(xlsx_path, sheet_name=sheet_name)


def _flatten_query_arrays(
    x_m: np.ndarray,
    y_m: np.ndarray,
    z_m: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, ...]]:
    x_arr = np.asarray(x_m, dtype=float)
    y_arr = np.asarray(y_m, dtype=float)
    z_arr = np.asarray(z_m, dtype=float)

    if not (x_arr.size == y_arr.size == z_arr.size):
        raise ValueError(
            "x_m, y_m, and z_m must contain the same number of samples."
        )

    return x_arr.ravel(), y_arr.ravel(), z_arr.ravel(), x_arr.shape


def _sort_and_validate_params(
    df: pd.DataFrame,
    required_columns: Sequence[str],
) -> pd.DataFrame:
    missing = [name for name in required_columns if name not in df.columns]
    if missing:
        raise ValueError(f"Missing required annular-BEMT columns: {missing}")

    numeric_df = df.loc[:, required_columns].copy()
    for name in required_columns:
        numeric_df[name] = pd.to_numeric(numeric_df[name], errors="coerce")
    numeric_df = numeric_df.dropna()

    if numeric_df.empty:
        raise ValueError(
            "No valid annular-BEMT parameter rows remain after cleaning."
        )

    numeric_df = numeric_df.sort_values("z_m").reset_index(drop=True)
    z_axis = numeric_df["z_m"].to_numpy(dtype=float)
    if np.any(np.diff(z_axis) <= 0.0):
        raise ValueError(
            "Annular-BEMT z_m values must be strictly increasing."
        )

    return numeric_df


def _discover_harmonic_orders(
    column_names: Sequence[str],
    suffix: str = "",
) -> Tuple[int, ...]:
    a_orders = set()
    b_orders = set()

    for name in column_names:
        if suffix and not name.endswith(suffix):
            continue
        base_name = name[: -len(suffix)] if suffix else name
        if base_name.startswith("a") and base_name[1:].isdigit():
            order = int(base_name[1:])
            if order >= 1:
                a_orders.add(order)
        if base_name.startswith("b") and base_name[1:].isdigit():
            order = int(base_name[1:])
            if order >= 1:
                b_orders.add(order)

    return tuple(sorted(a_orders.intersection(b_orders)))


def _build_harmonic_column_names(
    harmonic_orders: Sequence[int],
    suffix: str = "",
) -> Tuple[str, ...]:
    names: List[str] = ["a0" + suffix]
    for order in harmonic_orders:
        names.append(f"a{order}{suffix}")
        names.append(f"b{order}{suffix}")
    return tuple(names)


def _interp_columns(
    z_axis: np.ndarray,
    param_table: np.ndarray,
    z_query: np.ndarray,
) -> np.ndarray:
    z_query = np.asarray(z_query, dtype=float).reshape(-1)
    out = np.empty((z_query.size, param_table.shape[1]), dtype=float)

    for idx in range(param_table.shape[1]):
        out[:, idx] = np.interp(
            z_query,
            z_axis,
            param_table[:, idx],
            left=param_table[0, idx],
            right=param_table[-1, idx],
        )

    return out


@dataclass(frozen=True)
class SingleFanBEMTMeanModel:
    source_path: Path
    sheet_name: str
    fan_center_xy: Tuple[float, float]
    z_axis: np.ndarray
    harmonic_orders: Tuple[int, ...]
    param_names: Tuple[str, ...]
    param_table: np.ndarray

    @classmethod
    def from_workbook(
        cls,
        xlsx_path: Path,
        sheet_name: str,
        fan_center_xy: Tuple[float, float],
    ) -> "SingleFanBEMTMeanModel":
        df = _load_numeric_sheet(xlsx_path, sheet_name)
        harmonic_orders = _discover_harmonic_orders(df.columns)
        param_names = (
            "w0",
            "r_ring",
            "delta_ring",
        ) + _build_harmonic_column_names(harmonic_orders=harmonic_orders)
        clean_df = _sort_and_validate_params(
            df=df,
            required_columns=("z_m",) + param_names,
        )

        param_table = clean_df.loc[:, param_names].to_numpy(dtype=float)
        delta_idx = param_names.index("delta_ring")
        if np.any(param_table[:, delta_idx] <= 0.0):
            raise ValueError(
                "delta_ring must stay positive in the annular-BEMT table."
            )

        return cls(
            source_path=xlsx_path,
            sheet_name=sheet_name,
            fan_center_xy=fan_center_xy,
            z_axis=clean_df["z_m"].to_numpy(dtype=float),
            harmonic_orders=harmonic_orders,
            param_names=param_names,
            param_table=param_table,
        )

    def evaluate(
        self,
        x_m: np.ndarray,
        y_m: np.ndarray,
        z_m: np.ndarray,
    ) -> np.ndarray:
        x_arr, y_arr, z_arr, shape = _flatten_query_arrays(
            x_m=x_m,
            y_m=y_m,
            z_m=z_m,
        )
        interp = _interp_columns(
            z_axis=self.z_axis,
            param_table=self.param_table,
            z_query=z_arr,
        )

        xc, yc = self.fan_center_xy
        dx = x_arr - float(xc)
        dy = y_arr - float(yc)
        r_arr = np.hypot(dx, dy)
        theta_arr = np.arctan2(dy, dx)

        w0 = interp[:, 0]
        r_ring = interp[:, 1]
        delta_ring = np.maximum(interp[:, 2], 1e-12)
        amp = interp[:, 3].copy()

        col_idx = 4
        for order in self.harmonic_orders:
            amp += interp[:, col_idx] * np.cos(order * theta_arr)
            amp += interp[:, col_idx + 1] * np.sin(order * theta_arr)
            col_idx += 2

        envelope = np.exp(-((r_arr - r_ring) / delta_ring) ** 2)
        return (w0 + envelope * amp).reshape(shape)


@dataclass(frozen=True)
class FourFanBEMTMeanModel:
    source_path: Path
    sheet_name: str
    z_axis: np.ndarray
    fan_models: Tuple[_FanModelData, ...]

    @classmethod
    def from_workbook(
        cls,
        xlsx_path: Path,
        sheet_name: str,
        fan_centers_by_id: Dict[str, Tuple[float, float]],
    ) -> "FourFanBEMTMeanModel":
        df = _load_numeric_sheet(xlsx_path, sheet_name)

        fan_ids = sorted(
            {
                match.group(1)
                for col in df.columns
                for match in [FOUR_FAN_ID_PATTERN.match(str(col))]
                if match is not None
            }
        )
        if not fan_ids:
            raise ValueError(
                "No per-fan annular-BEMT columns were found in the workbook."
            )

        missing_centers = [
            fan_id for fan_id in fan_ids if fan_id not in fan_centers_by_id
        ]
        if missing_centers:
            raise ValueError(
                f"Missing fan-center definitions for: {missing_centers}"
            )

        required_columns: List[str] = ["z_m"]
        fan_specs: List[Tuple[str, Tuple[int, ...], Tuple[str, ...]]] = []
        for fan_id in fan_ids:
            suffix = f"_{fan_id}"
            harmonic_orders = _discover_harmonic_orders(
                df.columns,
                suffix=suffix,
            )
            param_names = (
                f"w0{suffix}",
                f"r_ring{suffix}",
                f"delta_ring{suffix}",
            ) + _build_harmonic_column_names(harmonic_orders, suffix=suffix)
            required_columns.extend(param_names)
            fan_specs.append((fan_id, harmonic_orders, param_names))

        clean_df = _sort_and_validate_params(
            df=df,
            required_columns=required_columns,
        )
        z_axis = clean_df["z_m"].to_numpy(dtype=float)

        fan_models: List[_FanModelData] = []
        for fan_id, harmonic_orders, param_names in fan_specs:
            param_table = clean_df.loc[:, param_names].to_numpy(dtype=float)
            delta_idx = param_names.index(f"delta_ring_{fan_id}")
            if np.any(param_table[:, delta_idx] <= 0.0):
                raise ValueError(
                    (
                        f"delta_ring_{fan_id} must stay positive in the "
                        "annular-BEMT table."
                    )
                )

            fan_models.append(
                _FanModelData(
                    fan_id=fan_id,
                    center_xy=fan_centers_by_id[fan_id],
                    harmonic_orders=harmonic_orders,
                    param_names=param_names,
                    param_table=param_table,
                )
            )

        return cls(
            source_path=xlsx_path,
            sheet_name=sheet_name,
            z_axis=z_axis,
            fan_models=tuple(fan_models),
        )

    def evaluate(
        self,
        x_m: np.ndarray,
        y_m: np.ndarray,
        z_m: np.ndarray,
    ) -> np.ndarray:
        x_arr, y_arr, z_arr, shape = _flatten_query_arrays(
            x_m=x_m,
            y_m=y_m,
            z_m=z_m,
        )
        total = np.zeros_like(x_arr, dtype=float)

        for fan_data in self.fan_models:
            interp = _interp_columns(
                z_axis=self.z_axis,
                param_table=fan_data.param_table,
                z_query=z_arr,
            )

            xc, yc = fan_data.center_xy
            dx = x_arr - float(xc)
            dy = y_arr - float(yc)
            r_arr = np.hypot(dx, dy)
            theta_arr = np.arctan2(dy, dx)

            w0 = interp[:, 0]
            r_ring = interp[:, 1]
            delta_ring = np.maximum(interp[:, 2], 1e-12)
            amp = interp[:, 3].copy()

            col_idx = 4
            for order in fan_data.harmonic_orders:
                amp += interp[:, col_idx] * np.cos(order * theta_arr)
                amp += interp[:, col_idx + 1] * np.sin(order * theta_arr)
                col_idx += 2

            envelope = np.exp(-((r_arr - r_ring) / delta_ring) ** 2)
            total += w0 + envelope * amp

        return total.reshape(shape)


@dataclass
class AnnularGPModelBundle:
    residual_gp_model: object
    mean_model: MeanModelProtocol

    def predict(
        self,
        x_m: np.ndarray,
        y_m: np.ndarray,
        z_m: np.ndarray,
        return_std: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        w_pred, w_std, _w_prior, _w_residual = self.predict_with_components(
            x_m=x_m,
            y_m=y_m,
            z_m=z_m,
            return_std=return_std,
        )
        return w_pred, w_std

    def predict_with_components(
        self,
        x_m: np.ndarray,
        y_m: np.ndarray,
        z_m: np.ndarray,
        return_std: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        w_prior = np.asarray(
            self.mean_model.evaluate(x_m=x_m, y_m=y_m, z_m=z_m),
            dtype=float,
        )
        w_residual_pred, w_std = self.residual_gp_model.predict(
            x_m=x_m,
            y_m=y_m,
            z_m=z_m,
            return_std=return_std,
        )
        w_pred = w_prior + w_residual_pred
        return (
            np.asarray(w_pred, dtype=float),
            np.asarray(w_std, dtype=float),
            w_prior,
            np.asarray(w_residual_pred, dtype=float),
        )


def fit_residual_gp_model(
    train_df: pd.DataFrame,
    feature_mode: str,
    kernel_family: str,
    alpha_scale: float,
    fan_center_xy: Tuple[float, float],
    sigma_min: float,
    alpha_jitter: float,
    n_restarts_optimizer: int,
    random_state: int,
    length_scale_floors: Dict[str, np.ndarray] | None = None,
) -> ResidualGPModelBundle:
    x_m = train_df["x_m"].to_numpy(dtype=float)
    y_m = train_df["y_m"].to_numpy(dtype=float)
    z_m = train_df["z_m"].to_numpy(dtype=float)
    w_obs = train_df["w_obs_mps"].to_numpy(dtype=float)
    sigma = np.maximum(
        train_df["sigma_mps"].to_numpy(dtype=float),
        float(sigma_min),
    )

    features = build_feature_matrix(
        x_m=x_m,
        y_m=y_m,
        z_m=z_m,
        feature_mode=feature_mode,
        fan_center_xy=fan_center_xy,
    )
    alpha = np.maximum(float(alpha_scale) * (sigma**2), float(alpha_jitter))

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    kernel = build_residual_gp_kernel(
        feature_mode=feature_mode,
        kernel_family=kernel_family,
        scaler=scaler,
        length_scale_floors=length_scale_floors,
    )

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

    return ResidualGPModelBundle(
        gp=gp,
        scaler=scaler,
        feature_mode=normalize_feature_mode_name(feature_mode),
        fan_center_xy=fan_center_xy,
    )


def evaluate_candidate_group_cv(
    train_df: pd.DataFrame,
    candidate,
    fan_center_xy: Tuple[float, float],
    n_splits: int,
    n_restarts_optimizer: int,
    random_state: int,
    sigma_min: float,
    alpha_jitter: float,
    compute_regression_metrics,
    length_scale_floors: Dict[str, np.ndarray] | None = None,
) -> Tuple[Dict[str, float], int]:
    groups = train_df["sheet"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        raise ValueError("Need at least two unique sheets for grouped CV.")

    n_splits_eff = int(np.clip(int(n_splits), 2, int(unique_groups.size)))
    splitter = GroupKFold(n_splits=n_splits_eff)

    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []
    sigma_all: List[np.ndarray] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(
        splitter.split(train_df, groups=groups)
    ):
        train_fold = train_df.iloc[tr_idx].reset_index(drop=True)
        val_fold = train_df.iloc[va_idx].reset_index(drop=True)

        model_fold = fit_residual_gp_model(
            train_df=train_fold,
            feature_mode=candidate.feature_mode,
            kernel_family=candidate.kernel_family,
            alpha_scale=float(candidate.alpha_scale),
            fan_center_xy=fan_center_xy,
            sigma_min=sigma_min,
            alpha_jitter=alpha_jitter,
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=int(random_state) + int(fold_idx),
            length_scale_floors=length_scale_floors,
        )

        y_pred_fold, _ = model_fold.predict(
            x_m=val_fold["x_m"].to_numpy(dtype=float),
            y_m=val_fold["y_m"].to_numpy(dtype=float),
            z_m=val_fold["z_m"].to_numpy(dtype=float),
            return_std=False,
        )
        y_true_all.append(val_fold["w_obs_mps"].to_numpy(dtype=float))
        sigma_all.append(val_fold["sigma_mps"].to_numpy(dtype=float))
        y_pred_all.append(y_pred_fold)

    metrics = compute_regression_metrics(
        y_true=np.concatenate(y_true_all),
        y_pred=np.concatenate(y_pred_all),
        sigma_mps=np.concatenate(sigma_all),
    )
    return metrics, n_splits_eff


def make_training_prediction_table(
    model: AnnularGPModelBundle,
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    w_pred, w_std, w_prior, w_residual_pred = model.predict_with_components(
        x_m=train_df["x_m"].to_numpy(dtype=float),
        y_m=train_df["y_m"].to_numpy(dtype=float),
        z_m=train_df["z_m"].to_numpy(dtype=float),
        return_std=True,
    )
    out = train_df.copy()
    out["w_prior_mps"] = w_prior
    out["w_residual_obs_mps"] = out["w_obs_mps"] - out["w_prior_mps"]
    out["w_residual_pred_mps"] = w_residual_pred
    out["w_pred_mps"] = w_pred
    out["w_pred_std_mps"] = w_std
    out["err_mps"] = out["w_pred_mps"] - out["w_obs_mps"]
    out["abs_err_mps"] = np.abs(out["err_mps"])
    return out


def make_grid_prediction_tables(
    model: AnnularGPModelBundle,
    xlsx_path: str,
    sheet_names: Sequence[str],
    parse_sheet_height_m,
    read_slice_from_sheet,
    sheet_tag: str,
    grid_nx: int | None = None,
    grid_ny: int | None = None,
    evaluate_fluctuation_sigma: Callable[
        [str, np.ndarray, np.ndarray, np.ndarray],
        np.ndarray,
    ]
    | None = None,
) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    for sheet_name in sheet_names:
        z_m = parse_sheet_height_m(sheet_name)
        x_centers, y_centers, _ = read_slice_from_sheet(xlsx_path, sheet_name)
        if grid_nx is None or grid_ny is None:
            x_axis = np.asarray(x_centers, dtype=float)
            y_axis = np.asarray(y_centers, dtype=float)
        else:
            x_axis = np.linspace(
                float(np.min(x_centers)),
                float(np.max(x_centers)),
                int(grid_nx),
                dtype=float,
            )
            y_axis = np.linspace(
                float(np.min(y_centers)),
                float(np.max(y_centers)),
                int(grid_ny),
                dtype=float,
            )

        x_grid, y_grid = np.meshgrid(x_axis, y_axis)
        z_grid = np.full_like(x_grid, z_m, dtype=float)

        w_mean, w_std = model.predict(
            x_m=x_grid.ravel(),
            y_m=y_grid.ravel(),
            z_m=z_grid.ravel(),
            return_std=True,
        )
        mean_map = w_mean.reshape(x_grid.shape)
        std_map = w_std.reshape(x_grid.shape)

        mean_df = pd.DataFrame(mean_map, index=y_axis, columns=x_axis)
        std_df = pd.DataFrame(std_map, index=y_axis, columns=x_axis)
        mean_df.index.name = "y/x"
        std_df.index.name = "y/x"

        tables[f"{sheet_name}_{sheet_tag}_mean"] = mean_df
        tables[f"{sheet_name}_{sheet_tag}_std"] = std_df

        if evaluate_fluctuation_sigma is not None:
            sigma_fluc = np.asarray(
                evaluate_fluctuation_sigma(
                    sheet_name,
                    x_grid.ravel(),
                    y_grid.ravel(),
                    z_grid.ravel(),
                ),
                dtype=float,
            ).reshape(x_grid.shape)
            sigma_total = np.sqrt(
                np.maximum(std_map, 0.0) ** 2 + sigma_fluc**2
            )

            fluc_df = pd.DataFrame(sigma_fluc, index=y_axis, columns=x_axis)
            total_df = pd.DataFrame(sigma_total, index=y_axis, columns=x_axis)
            fluc_df.index.name = "y/x"
            total_df.index.name = "y/x"

            tables[f"{sheet_name}_{sheet_tag}_fluc"] = fluc_df
            tables[f"{sheet_name}_{sheet_tag}_total_fluc"] = total_df

    return tables


def build_autotune_cv_table(
    tune_trials: Sequence[object],
    selected_candidate: object,
) -> pd.DataFrame:
    if tune_trials:
        rows: List[Dict[str, float | str | bool]] = []
        for trial in tune_trials:
            rows.append(
                {
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
                    (
                        "is_selected"
                    ): trial.candidate.name == selected_candidate.name,
                }
            )
        return pd.DataFrame(rows).sort_values(
            by=["cv_wrmse_mps", "cv_mae_mps"],
            ascending=[True, True],
        )

    return pd.DataFrame(
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
