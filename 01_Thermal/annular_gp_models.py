from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Protocol, Sequence, Tuple

import numpy as np
import pandas as pd


FOUR_FAN_ID_PATTERN = re.compile(r"^a0_(F\d{2})$")


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
) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    for sheet_name in sheet_names:
        z_m = parse_sheet_height_m(sheet_name)
        x_centers, y_centers, _ = read_slice_from_sheet(xlsx_path, sheet_name)
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

        tables[f"{sheet_name}_{sheet_tag}_mean"] = mean_df
        tables[f"{sheet_name}_{sheet_tag}_std"] = std_df

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
