"""
Interpolate fitted non-axisymmetric annular-Gaussian parameters vs height z.

This script reads the fitted parameter table produced by
single_fan_annular_gaussian_bemt.py, interpolates coefficients using PCHIP,
and exports a regular-z parameter table for simulation use.

Parameter vector format:
    [w0, r_ring, delta_ring, a0, a1, b1, ..., aN, bN]

The script auto-detects N from available a_n/b_n columns.

Reference upstream annuli profile naming from single_fan_annuli_cut.py:
    B_results/Single_Fan_Annuli_Profile/
    <sheet>_single_annuli_profile.csv
"""


from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Interpolation Configuration and Data Sources
# 2) Interpolation and Evaluation Helpers
# 3) Parameter Export Entry Point
# =============================================================================

# =============================================================================
# 1) Interpolation Configuration and Data Sources
# =============================================================================
# Workbook, parameter, and output paths below define the data-provenance boundary for this run.

PARAMS_XLSX = Path("B_results/single_annular_bemt_params.xlsx")
PARAMS_SHEET = "single_bemt_az_fit"

OUT_XLSX_PATH = Path("B_results/single_annular_bemt_params_pchip.xlsx")
OUT_SHEET_NAME = "single_bemt_az_pchip"

# Output z grid (meters). Use None to infer from fitted data.
# PCHIP extrapolates when Z_MIN_M or Z_MAX_M lies outside the fitted height range.
# Keep this aligned with single_fan_annular_gaussian_var_fit.py.
Z_MIN_M = 0.0
Z_MAX_M = 3.5
Z_STEP_M = 0.01

# For z values above this threshold, force extrapolation to use only these
# anchor heights from the fitted data table.
HIGH_Z_THRESHOLD_M = 2.20
HIGH_Z_ANCHOR_POINTS_M = (1.10, 1.60, 2.20)
HIGH_Z_ANCHOR_TOL_M = 1e-6
# Smooth half-width (m) around HIGH_Z_THRESHOLD_M for blending between
# main and anchor extrapolation. Set <= 0.0 to recover hard switching.
HIGH_Z_BLEND_HALF_WIDTH_M = 0.10

# Harmonic stabilization for extrapolated heights (z > HIGH_Z_THRESHOLD_M):
# Stabilisation first damps Fourier harmonics with height,
# then caps harmonic magnitude relative to a0.
# This mitigates high-z spikes caused by unconstrained extrapolation.
ENABLE_HARMONIC_STABILIZATION = True
HARMONIC_DECAY_EFOLD_M = 0.60
HARMONIC_MAX_REL_TO_A0 = 0.60
HARMONIC_A0_FLOOR = 1e-3

# Keep a0 positive (if fitted samples are positive) by using log-space
# interpolation/extrapolation. This prevents sign-flip artifacts at high z.
A0_LOG_INTERP_IF_POSITIVE = True

# Below the first fitted z level, hold coefficients at the first fitted row
# instead of extrapolating. This prevents artificial near-outlet blow-up.
ENABLE_LOW_Z_EDGE_HOLD = True

# If anchors are missing in fitted z values, fall back to the main full-range
# interpolation branch instead of raising an exception.
ALLOW_ANCHOR_FALLBACK = True

# Physical floor for interpolated ring radius.
R_RING_MIN_CLIP_M = 0.0


# =============================================================================
# 2) Interpolation and Evaluation Helpers
# =============================================================================
# Interpolation helpers define extrapolation and positivity guardrails before export.

# High-z anchors restrict extrapolation to measured heights that define the upper plume trend.
def select_anchor_indices(z_vals: np.ndarray) -> Optional[np.ndarray]:
    """
    Locate indices for HIGH_Z_ANCHOR_POINTS_M in z_vals.
    """
    anchor_indices = []
    for anchor_z in HIGH_Z_ANCHOR_POINTS_M:
        matches = np.where(
            np.isclose(z_vals, anchor_z, atol=HIGH_Z_ANCHOR_TOL_M, rtol=0.0)
        )[0]
        if matches.size == 0:
            if ALLOW_ANCHOR_FALLBACK:
                return None
            raise ValueError(
                f"Anchor z={anchor_z:.2f} m was not found in fitted data. "
                f"Available z range: [{np.min(z_vals):.2f}, {np.max(z_vals):.2f}] m."
            )
        anchor_indices.append(int(matches[0]))

    if len(set(anchor_indices)) != len(anchor_indices):
        raise ValueError(
            "Duplicate anchor matches detected for HIGH_Z_ANCHOR_POINTS_M. "
            "Check z spacing or HIGH_Z_ANCHOR_TOL_M."
        )

    return np.array(anchor_indices, dtype=int)


# Smoothstep blending avoids a discontinuity at the high-z extrapolation threshold.
def high_branch_weight(z_query: np.ndarray) -> np.ndarray:
    """
    Compute smooth high-branch blending weights in [0, 1].

    The blend region is centered at HIGH_Z_THRESHOLD_M with half-width
    HIGH_Z_BLEND_HALF_WIDTH_M. A smoothstep profile is used to keep first
    derivatives continuous at both blend edges.
    """
    z_query = np.asarray(z_query, dtype=float)
    half_width = float(HIGH_Z_BLEND_HALF_WIDTH_M)
    threshold = float(HIGH_Z_THRESHOLD_M)

    if half_width <= 0.0:
        return (z_query > threshold).astype(float)

    z0 = threshold - half_width
    z1 = threshold + half_width
    t = (z_query - z0) / (z1 - z0)
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# Fourier terms are paired by harmonic order so amplitude caps preserve phase orientation.
def harmonic_pairs(param_cols: List[str]) -> List[Tuple[int, int]]:
    """
    Return aligned (a_n_idx, b_n_idx) index pairs found in param_cols.
    """
    pairs = []
    for col_name in param_cols:
        if not (col_name.startswith("a") and col_name[1:].isdigit()):
            continue
        n_idx = int(col_name[1:])
        if n_idx < 1:
            continue
        b_name = f"b{n_idx}"
        if b_name in param_cols:
            pairs.append((param_cols.index(col_name), param_cols.index(b_name)))
    return pairs


# Harmonic damping limits high-z Fourier growth outside the measured support.
def stabilize_extrapolated_harmonics(
    z_query: np.ndarray,
    params_interp: np.ndarray,
    param_cols: List[str],
) -> np.ndarray:
    """
    Stabilize extrapolated Fourier terms to reduce high-z spikes.
    """
    if not ENABLE_HARMONIC_STABILIZATION:
        return params_interp

    z_query = np.asarray(z_query, dtype=float)
    high_mask = z_query > float(HIGH_Z_THRESHOLD_M)
    if not np.any(high_mask):
        return params_interp

    out = params_interp.copy()

    a0_idx = param_cols.index("a0")
    a0_vals = out[:, a0_idx]
    if A0_LOG_INTERP_IF_POSITIVE:
        a0_vals[high_mask] = np.maximum(a0_vals[high_mask], HARMONIC_A0_FLOOR)
        out[:, a0_idx] = a0_vals

    decay = np.ones_like(z_query)
    if HARMONIC_DECAY_EFOLD_M > 0.0:
        z_high = z_query[high_mask]
        decay[high_mask] = np.exp(
            -(z_high - float(HIGH_Z_THRESHOLD_M)) / float(HARMONIC_DECAY_EFOLD_M)
        )

    cap_ref = np.maximum(out[:, a0_idx], HARMONIC_A0_FLOOR)
    pair_indices = harmonic_pairs(param_cols)
    for a_idx, b_idx in pair_indices:
        a_vals = out[:, a_idx]
        b_vals = out[:, b_idx]
        amp = np.sqrt(a_vals**2 + b_vals**2)
        cap = float(HARMONIC_MAX_REL_TO_A0) * cap_ref

        cap_scale = np.ones_like(amp)
        cap_mask = high_mask & (amp > cap)
        cap_scale[cap_mask] = cap[cap_mask] / np.maximum(amp[cap_mask], 1e-12)

        total_scale = np.ones_like(amp)
        total_scale[high_mask] = decay[high_mask] * cap_scale[high_mask]
        out[:, a_idx] = a_vals * total_scale
        out[:, b_idx] = b_vals * total_scale

    return out


# Parameter workbooks are the fitted-model interface consumed by plotting and simulation scripts.
def load_params_table(xlsx_path: Path, sheet_name: str) -> pd.DataFrame:
    """
    Load fitted parameters from Excel.
    """
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Missing parameter file: {xlsx_path}")

    xls = pd.ExcelFile(xlsx_path)
    if sheet_name not in xls.sheet_names:
        available = ", ".join(xls.sheet_names)
        raise ValueError(
            f"Sheet '{sheet_name}' not found in '{xlsx_path}'. "
            f"Available sheets: {available}"
        )

    return pd.read_excel(xlsx_path, sheet_name=sheet_name)


# Parameter-column discovery keeps harmonic order explicit before array conversion.
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


# Parameter extraction keeps fitted z samples paired with their model coefficients.
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


# The exported z grid is a simulation interface and may be denser than measured heights.
def make_z_grid(z_vals: np.ndarray) -> np.ndarray:
    """
    Build a regular z grid for interpolation outputs.
    """
    z_min = float(np.min(z_vals)) if Z_MIN_M is None else float(Z_MIN_M)
    z_max = float(np.max(z_vals)) if Z_MAX_M is None else float(Z_MAX_M)

    if Z_STEP_M <= 0.0:
        raise ValueError("Z_STEP_M must be positive.")
    if z_max <= z_min:
        raise ValueError("Z_MAX_M must be greater than Z_MIN_M.")

    steps = int(round((z_max - z_min) / Z_STEP_M))
    steps = max(steps, 1)
    return np.linspace(z_min, z_max, steps + 1)


# PCHIP preserves monotone height trends without spline overshoot in fitted parameters.
def interpolate_params_pchip(
    z_vals: np.ndarray,
    params: np.ndarray,
    param_cols: List[str],
    z_query: np.ndarray,
) -> np.ndarray:
    """
    Interpolate fitted parameters with PCHIP over z.

    delta_ring is interpolated in log-space to keep positivity.
    a0 can also use log-space interpolation when all fitted values are positive.
    For high z, extrapolation blends from the full-data PCHIP to the
    anchor-only PCHIP around HIGH_Z_THRESHOLD_M.
    """
    if z_vals.size < 2:
        raise ValueError("Need at least two fitted heights for PCHIP interpolation.")

    params_interp = np.empty((z_query.size, params.shape[1]), dtype=float)
    high_weight = high_branch_weight(z_query)
    high_mask = high_weight > 0.0

    anchor_idx = select_anchor_indices(z_vals)
    high_branch_enabled = anchor_idx is not None
    if high_branch_enabled:
        z_anchor = z_vals[anchor_idx]

    for col_idx, col_name in enumerate(param_cols):
        if col_name == "delta_ring":
            vals = np.maximum(params[:, col_idx], 1e-12)
            interp_main = PchipInterpolator(z_vals, np.log(vals))
            col_interp = np.exp(interp_main(z_query))

            if high_branch_enabled and np.any(high_mask):
                vals_anchor = np.maximum(params[anchor_idx, col_idx], 1e-12)
                interp_high = PchipInterpolator(z_anchor, np.log(vals_anchor))
                high_vals = np.exp(interp_high(z_query[high_mask]))
                col_interp[high_mask] = (
                    (1.0 - high_weight[high_mask]) * col_interp[high_mask]
                    + high_weight[high_mask] * high_vals
                )
        elif (
            col_name == "a0"
            and A0_LOG_INTERP_IF_POSITIVE
            and np.all(params[:, col_idx] > 0.0)
        ):
            vals = np.maximum(params[:, col_idx], HARMONIC_A0_FLOOR)
            interp_main = PchipInterpolator(z_vals, np.log(vals))
            col_interp = np.exp(interp_main(z_query))

            if high_branch_enabled and np.any(high_mask):
                vals_anchor = np.maximum(params[anchor_idx, col_idx], HARMONIC_A0_FLOOR)
                interp_high = PchipInterpolator(z_anchor, np.log(vals_anchor))
                high_vals = np.exp(interp_high(z_query[high_mask]))
                col_interp[high_mask] = (
                    (1.0 - high_weight[high_mask]) * col_interp[high_mask]
                    + high_weight[high_mask] * high_vals
                )
        else:
            interp_main = PchipInterpolator(z_vals, params[:, col_idx])
            col_interp = interp_main(z_query)

            if high_branch_enabled and np.any(high_mask):
                interp_high = PchipInterpolator(z_anchor, params[anchor_idx, col_idx])
                high_vals = interp_high(z_query[high_mask])
                col_interp[high_mask] = (
                    (1.0 - high_weight[high_mask]) * col_interp[high_mask]
                    + high_weight[high_mask] * high_vals
                )

        params_interp[:, col_idx] = col_interp

    params_interp = stabilize_extrapolated_harmonics(
        z_query=z_query,
        params_interp=params_interp,
        param_cols=param_cols,
    )

    if ENABLE_LOW_Z_EDGE_HOLD:
        low_mask = z_query < float(np.min(z_vals))
        if np.any(low_mask):
            params_interp[low_mask, :] = params[0, :]

    # Keep physically constrained columns safe after interpolation.
    r_idx = param_cols.index("r_ring")
    d_idx = param_cols.index("delta_ring")
    params_interp[:, r_idx] = np.maximum(params_interp[:, r_idx], float(R_RING_MIN_CLIP_M))
    params_interp[:, d_idx] = np.maximum(params_interp[:, d_idx], 1e-12)

    return params_interp


# The interpolated table is a simulation handoff, not a replacement for raw fit diagnostics.
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


# =============================================================================
# 3) Parameter Export Entry Point
# =============================================================================
# Entry points write deterministic artifacts so regenerated figures and tables can be compared by path and sheet name.

# Main execution keeps data loading, evaluation, and export order deterministic.
def main() -> None:
    params_df = load_params_table(PARAMS_XLSX, PARAMS_SHEET)
    param_cols = discover_param_columns(params_df)
    z_vals, params = extract_params(params_df, param_cols)
    anchor_idx = select_anchor_indices(z_vals)
    high_branch_enabled = anchor_idx is not None

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
    print(f"High-z anchor branch enabled: {high_branch_enabled}")


if __name__ == "__main__":
    main()
