"""
Interpolate fitted non-axisymmetric annular-Gaussian parameters vs height z.

This script reads the fitted parameter table produced by
single_fan_annular_gaussian_bemt.py, interpolates coefficients using PCHIP,
and exports a regular-z parameter table for simulation use.

Parameter vector format:
    [w0, r_ring, delta_ring, a0, a1, b1, ..., aN, bN]

The script auto-detects N from available a_n/b_n columns.

Reference upstream profile naming from single_fan_annuli_bemt_cut.py:
    B_results/Single_Fan_Annuli_BEMT_Profile/
    <sheet>_single_annuli_bemt_profile.csv
"""

###### Initialization

### Imports
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


### User settings
PARAMS_XLSX = Path("B_results/single_annular_bemt_params.xlsx")
PARAMS_SHEET = "single_bemt_az_fit"

OUT_XLSX_PATH = Path("B_results/single_annular_bemt_params_pchip.xlsx")
OUT_SHEET_NAME = "single_bemt_az_pchip"

# Output z grid (meters). Use None to infer from fitted data.
# NOTE: Setting Z_MIN_M below fitted min or Z_MAX_M above fitted max extrapolates.
Z_MIN_M = None
Z_MAX_M = None
Z_STEP_M = 0.01


### Helpers
def load_params_table(xlsx_path: Path, sheet_name: str) -> pd.DataFrame:
    """
    Load fitted parameters from Excel.
    """
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Missing parameter file: {xlsx_path}")

    xls = pd.ExcelFile(xlsx_path)
    sheet_to_use = sheet_name if sheet_name in xls.sheet_names else xls.sheet_names[0]
    return pd.read_excel(xlsx_path, sheet_name=sheet_to_use)


def discover_param_columns(df: pd.DataFrame) -> List[str]:
    """
    Discover parameter columns from the fitted table.

    Required base columns:
        w0, r_ring, delta_ring, a0

    Optional harmonics:
        a1,b1,a2,b2,...
    """
    base = ["w0", "r_ring", "delta_ring", "a0"]
    missing = [col for col in base if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required parameter columns: {missing}")

    a_orders = []
    b_orders = []
    for col in df.columns:
        if col.startswith("a") and col[1:].isdigit():
            order = int(col[1:])
            if order >= 1:
                a_orders.append(order)
        if col.startswith("b") and col[1:].isdigit():
            order = int(col[1:])
            if order >= 1:
                b_orders.append(order)

    harmonic_orders = sorted(set(a_orders).intersection(set(b_orders)))

    param_cols = list(base)
    for n_idx in harmonic_orders:
        param_cols.append(f"a{n_idx}")
        param_cols.append(f"b{n_idx}")

    return param_cols


def extract_params(
    df: pd.DataFrame,
    param_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract and validate z values and parameter matrix.
    """
    if "z_m" not in df.columns:
        raise ValueError("Missing required column: z_m")

    data_cols = ["z_m"] + list(param_cols)
    data = df[data_cols].copy()

    for col in data_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna()

    if data.empty:
        raise ValueError("No valid fitted rows after cleaning.")

    z_vals = data["z_m"].to_numpy(dtype=float)
    params = data[param_cols].to_numpy(dtype=float)

    order = np.argsort(z_vals)
    z_vals = z_vals[order]
    params = params[order, :]

    if np.any(np.diff(z_vals) <= 0.0):
        raise ValueError("z_m values must be strictly increasing for PCHIP.")

    delta_idx = param_cols.index("delta_ring")
    if np.any(params[:, delta_idx] <= 0.0):
        raise ValueError("delta_ring must be positive for log-space interpolation.")

    return z_vals, params


def make_z_grid(z_vals: np.ndarray) -> np.ndarray:
    """
    Build a regular z grid for interpolation outputs.
    """
    z_min = float(np.min(z_vals)) if Z_MIN_M is None else float(Z_MIN_M)
    z_max = float(np.max(z_vals)) if Z_MAX_M is None else float(Z_MAX_M)

    if Z_STEP_M <= 0.0:
        raise ValueError("Z_STEP_M must be positive.")

    steps = int(round((z_max - z_min) / Z_STEP_M))
    steps = max(steps, 1)
    return np.linspace(z_min, z_max, steps + 1)


def interpolate_params_pchip(
    z_vals: np.ndarray,
    params: np.ndarray,
    param_cols: List[str],
    z_query: np.ndarray,
) -> np.ndarray:
    """
    Interpolate fitted parameters with PCHIP over z.

    delta_ring is interpolated in log-space to keep positivity.
    """
    if z_vals.size < 2:
        raise ValueError("Need at least two fitted heights for PCHIP interpolation.")

    params_interp = np.empty((z_query.size, params.shape[1]), dtype=float)

    for col_idx, col_name in enumerate(param_cols):
        if col_name == "delta_ring":
            vals = np.maximum(params[:, col_idx], 1e-12)
            interp = PchipInterpolator(z_vals, np.log(vals))
            params_interp[:, col_idx] = np.exp(interp(z_query))
        else:
            interp = PchipInterpolator(z_vals, params[:, col_idx])
            params_interp[:, col_idx] = interp(z_query)

    return params_interp


def write_interpolated_table(
    z_grid: np.ndarray,
    params_interp: np.ndarray,
    param_cols: List[str],
    out_path: Path,
    sheet_name: str,
) -> None:
    """
    Write interpolated parameter table to Excel.
    """
    data = {"z_m": z_grid}
    for idx, name in enumerate(param_cols):
        data[name] = params_interp[:, idx]

    df_out = pd.DataFrame(data)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_excel(out_path, index=False, sheet_name=sheet_name)


### Main
def main() -> None:
    params_df = load_params_table(PARAMS_XLSX, PARAMS_SHEET)
    param_cols = discover_param_columns(params_df)
    z_vals, params = extract_params(params_df, param_cols)

    z_grid = make_z_grid(z_vals)
    params_interp = interpolate_params_pchip(
        z_vals=z_vals,
        params=params,
        param_cols=param_cols,
        z_query=z_grid,
    )

    write_interpolated_table(
        z_grid=z_grid,
        params_interp=params_interp,
        param_cols=param_cols,
        out_path=OUT_XLSX_PATH,
        sheet_name=OUT_SHEET_NAME,
    )

    print("Built continuous non-axisymmetric annular-Gaussian model using PCHIP.")
    print(f"Input:  {PARAMS_XLSX.resolve()}")
    print(f"Output: {OUT_XLSX_PATH.resolve()}")


if __name__ == "__main__":
    main()
