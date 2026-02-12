###### Initialization

### Imports
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

from single_fan_annuli_cut import (
    assign_sigma_bins_nearest,
    parse_ts_points_and_sigmas,
    read_slice_from_sheet,
)


### User settings
XLSX_PATH = "S01.xlsx"
SHEETS = ["z020", "z035", "z050", "z075", "z110", "z160", "z220"]

# Fan centre (x_c, y_c) in meters.
FAN_CENTER_XY = (4.2, 2.4)
SHEET_HEIGHT_DIVISOR = 100.0

# Feature mode:
#   True  -> use (r, z), enforcing axisymmetry
#   False -> use (x, y, z), allowing asymmetric structure
USE_RADIAL_FEATURES = False

# Noise assignment from *_TS.
SIGMA_FALLBACK = 0.2
SIGMA_MIN = 1e-3
ALPHA_JITTER = 1e-8

# GP optimizer settings.
N_RESTARTS_OPTIMIZER = 1
RANDOM_STATE = 42

# Output locations.
OUT_DIR = Path("B_results/Single_Fan_GP")
TRAIN_PRED_CSV_PATH = OUT_DIR / "single_fan_gp_training_predictions.csv"
SUMMARY_XLSX_PATH = OUT_DIR / "single_fan_gp_summary.xlsx"
GRID_PRED_XLSX_PATH = OUT_DIR / "single_fan_gp_grid_predictions.xlsx"

### Data model
@dataclass
class GPModelBundle:
    """
    Container for a fitted GP model and its feature transform.
    """

    gp: GaussianProcessRegressor
    scaler: StandardScaler
    use_radial_features: bool
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
            use_radial_features=self.use_radial_features,
            fan_center_xy=self.fan_center_xy,
        )
        feats_scaled = self.scaler.transform(feats)

        if return_std:
            w_mean, w_std = self.gp.predict(feats_scaled, return_std=True)
            return w_mean.astype(float), w_std.astype(float)

        w_mean = self.gp.predict(feats_scaled, return_std=False)
        return w_mean.astype(float), np.zeros_like(w_mean, dtype=float)


### Helpers
def parse_sheet_height_m(sheet_name: str) -> float:
    if not sheet_name.startswith("z"):
        raise ValueError(f"Invalid sheet name (expected 'z###'): {sheet_name}")
    suffix = sheet_name[1:]
    if not suffix.isdigit():
        raise ValueError(f"Invalid height code in sheet name: {sheet_name}")
    return int(suffix) / SHEET_HEIGHT_DIVISOR


def build_feature_matrix(
    x_m: np.ndarray,
    y_m: np.ndarray,
    z_m: np.ndarray,
    use_radial_features: bool,
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

    if use_radial_features:
        xc, yc = fan_center_xy
        r_arr = np.sqrt((x_arr - xc) ** 2 + (y_arr - yc) ** 2)
        return np.column_stack([r_arr, z_arr])

    return np.column_stack([x_arr, y_arr, z_arr])


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
    ts_parsed = parse_ts_points_and_sigmas(
        xlsx_path=xlsx_path,
        ts_sheet_name=ts_sheet,
        fan_center_xy=fan_center_xy,
    )

    if ts_parsed is None:
        sigma_pts = np.full_like(r_pts, float(sigma_fallback), dtype=float)
        sigma_pts = np.maximum(sigma_pts, float(sigma_min))
    else:
        r_points, sigma_points = ts_parsed
        sigma_pts = assign_sigma_bins_nearest(
            r_bins=r_pts,
            r_points=r_points,
            sigma_points=sigma_points,
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
    use_radial_features: bool,
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
        use_radial_features=use_radial_features,
        fan_center_xy=fan_center_xy,
    )
    alpha = np.maximum(sigma**2, float(ALPHA_JITTER))

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    n_features = int(features.shape[1])
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(
            length_scale=np.ones(n_features, dtype=float),
            length_scale_bounds=(1e-2, 1e2),
        )
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e0))
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

    return GPModelBundle(
        gp=gp,
        scaler=scaler,
        use_radial_features=use_radial_features,
        fan_center_xy=fan_center_xy,
    )


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute standard regression metrics.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")

    return {
        "mae_mps": mae,
        "rmse_mps": rmse,
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
        )
        rows.append(
            {
                "sheet": sheet,
                "z_m": float(sub["z_m"].iloc[0]),
                "n_samples": int(sub.shape[0]),
                "mae_mps": metrics["mae_mps"],
                "rmse_mps": metrics["rmse_mps"],
                "r2": metrics["r2"],
                "mean_pred_std_mps": float(
                    np.mean(sub["w_pred_std_mps"].to_numpy(dtype=float))
                ),
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
    out_xlsx: Path,
    tables: Dict[str, pd.DataFrame],
) -> None:
    """
    Write multiple tables to Excel
    """
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    if out_xlsx.exists():
        with pd.ExcelWriter(
            out_xlsx,
            engine="openpyxl",
            mode="a",
            if_sheet_exists="replace",
        ) as writer:
            for sheet_name, table in tables.items():
                table.to_excel(writer, index=True, sheet_name=sheet_name)
    else:
        with pd.ExcelWriter(out_xlsx, engine="openpyxl", mode="w") as writer:
            for sheet_name, table in tables.items():
                table.to_excel(writer, index=True, sheet_name=sheet_name)


### Main
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = build_training_table(
        xlsx_path=XLSX_PATH,
        sheet_names=SHEETS,
        fan_center_xy=FAN_CENTER_XY,
        sigma_fallback=SIGMA_FALLBACK,
        sigma_min=SIGMA_MIN,
    )

    model = fit_gp_model(
        train_df=train_df,
        use_radial_features=USE_RADIAL_FEATURES,
        fan_center_xy=FAN_CENTER_XY,
        n_restarts_optimizer=N_RESTARTS_OPTIMIZER,
        random_state=RANDOM_STATE,
    )

    pred_df = make_training_prediction_table(model, train_df)
    pred_df.to_csv(TRAIN_PRED_CSV_PATH, index=False)

    overall_metrics = compute_regression_metrics(
        y_true=pred_df["w_obs_mps"].to_numpy(dtype=float),
        y_pred=pred_df["w_pred_mps"].to_numpy(dtype=float),
    )
    summary_metrics_df = pd.DataFrame([overall_metrics])
    summary_metrics_df["use_radial_features"] = bool(USE_RADIAL_FEATURES)
    summary_metrics_df["log_marginal_likelihood"] = float(
        model.gp.log_marginal_likelihood_value_
    )
    summary_metrics_df["kernel"] = str(model.gp.kernel_)

    per_sheet_df = summarize_by_sheet(pred_df)
    hyper_df = pd.DataFrame(
        [
            {"parameter": "xlsx_path", "value": XLSX_PATH},
            {"parameter": "sheet_count", "value": len(SHEETS)},
            {"parameter": "fan_center_x_m", "value": float(FAN_CENTER_XY[0])},
            {"parameter": "fan_center_y_m", "value": float(FAN_CENTER_XY[1])},
            {"parameter": "use_radial_features", "value": bool(USE_RADIAL_FEATURES)},
            {"parameter": "sigma_fallback_mps", "value": float(SIGMA_FALLBACK)},
            {"parameter": "sigma_min_mps", "value": float(SIGMA_MIN)},
            {"parameter": "n_restarts_optimizer", "value": int(N_RESTARTS_OPTIMIZER)},
            {"parameter": "random_state", "value": int(RANDOM_STATE)},
            {"parameter": "kernel_fitted", "value": str(model.gp.kernel_)},
        ]
    )
    write_tables_to_excel(
        SUMMARY_XLSX_PATH,
        {
            "overall_metrics": summary_metrics_df,
            "per_sheet_metrics": per_sheet_df,
            "hyperparameters": hyper_df,
        },
    )

    grid_tables = make_grid_prediction_tables(
        model=model,
        xlsx_path=XLSX_PATH,
        sheet_names=SHEETS,
    )
    write_tables_to_excel(GRID_PRED_XLSX_PATH, grid_tables)

    print("Gaussian Process model fitted successfully.")
    print(f"Samples used: {int(train_df.shape[0])}")
    print(f"Fitted kernel: {model.gp.kernel_}")
    print(
        "Overall metrics: "
        f"MAE={overall_metrics['mae_mps']:.4f} m/s, "
        f"RMSE={overall_metrics['rmse_mps']:.4f} m/s, "
        f"R2={overall_metrics['r2']:.4f}"
    )
    print(f"Training predictions CSV: {TRAIN_PRED_CSV_PATH.resolve()}")
    print(f"Summary workbook: {SUMMARY_XLSX_PATH.resolve()}")
    print(f"Grid predictions workbook: {GRID_PRED_XLSX_PATH.resolve()}")


if __name__ == "__main__":
    main()
