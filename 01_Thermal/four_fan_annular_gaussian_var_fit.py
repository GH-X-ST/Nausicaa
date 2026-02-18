"""
Build a continuous annular-Gaussian model w_model(r, z) from fitted parameters.

This script reads output from four_fan_annular_gaussian_var.py
(B_results/four_annular_var_params.xlsx), interpolates fitted parameters
vs height z using PCHIP, and writes an interpolated parameter table to Excel.

Model:
    w_model(r, z) = w0(z) + A_ring(z) * exp(-((r - r_ring(z)) / delta_r(z))**2)

Fine-tuning additions in this version:
- smooth blending into the high-z anchor branch,
- low-z edge hold (no extrapolation below first fitted height),
- anchor fallback if configured anchor heights are missing.
"""

###### Initialization

### Imports
from pathlib import Path
from typing import Optional, Tuple
import re

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


### User settings
PARAMS_XLSX = Path("B_results/four_annular_var_params.xlsx")
PARAMS_SHEET = "four_annular_var"

OUT_XLSX_PATH = Path("B_results/four_annular_var_params_pchip.xlsx")
OUT_SHEET_NAME = "four_annular_var_pchip"

# Output z grid (meters). Use None to infer from fitted data.
# NOTE: Setting Z_MIN_M below the fitted min or Z_MAX_M above the fitted max
# extrapolates. Low-z extrapolation can be held with ENABLE_LOW_Z_EDGE_HOLD.
Z_MIN_M = 0.0
Z_MAX_M = 3.5
Z_STEP_M = 0.01

# For z values above this threshold, blend toward a branch built only from
# these anchor heights. This stabilizes extrapolation behaviour.
HIGH_Z_THRESHOLD_M = 2.20
HIGH_Z_ANCHOR_POINTS_M = (1.10, 1.60, 2.20)
HIGH_Z_ANCHOR_TOL_M = 1e-6
HIGH_Z_BLEND_HALF_WIDTH_M = 0.10

# If True, keep params fixed at the first fitted row for z below min(z_fit).
ENABLE_LOW_Z_EDGE_HOLD = True

# If True, and requested high-z anchors are not all present, continue using
# only the main full-range branch (instead of raising an exception).
ALLOW_ANCHOR_FALLBACK = True


### Helpers
REQUIRED_COLUMNS = ("z_m", "A_ring", "r_ring", "delta_r", "w0")
FAN_COL_PATTERN = re.compile(r"^A_ring_(F\d{2})$")


def select_anchor_indices(z_vals: np.ndarray) -> Optional[np.ndarray]:
    """
    Locate indices for HIGH_Z_ANCHOR_POINTS_M in z_vals.

    Returns None when anchors are missing and ALLOW_ANCHOR_FALLBACK is enabled.
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


def high_branch_weight(z_query: np.ndarray) -> np.ndarray:
    """
    Compute smooth high-branch blending weights in [0, 1].

    Blend region is centered at HIGH_Z_THRESHOLD_M with half-width
    HIGH_Z_BLEND_HALF_WIDTH_M and uses a smoothstep profile.
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


def load_params_table(xlsx_path: Path, sheet_name: str) -> pd.DataFrame:
    """
    Load fitted parameters from Excel.
    """
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Missing parameter file: {xlsx_path}")

    xls = pd.ExcelFile(xlsx_path)
    sheet_to_use = sheet_name if sheet_name in xls.sheet_names else xls.sheet_names[0]
    return pd.read_excel(xlsx_path, sheet_name=sheet_to_use)


def extract_params(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract and validate fitted parameters from a DataFrame.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in parameter table: {missing}")

    data = df[list(REQUIRED_COLUMNS)].copy()
    for col in REQUIRED_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna()

    if data.empty:
        raise ValueError("No valid parameter rows found after cleaning.")

    z_vals = data["z_m"].to_numpy(dtype=float)
    a_ring = data["A_ring"].to_numpy(dtype=float)
    r_ring = data["r_ring"].to_numpy(dtype=float)
    delta_r = data["delta_r"].to_numpy(dtype=float)
    w0 = data["w0"].to_numpy(dtype=float)

    order = np.argsort(z_vals)
    z_vals = z_vals[order]
    a_ring = a_ring[order]
    r_ring = r_ring[order]
    delta_r = delta_r[order]
    w0 = w0[order]

    if np.any(np.diff(z_vals) <= 0.0):
        raise ValueError("z_m values must be strictly increasing for PCHIP.")

    if np.any(a_ring <= 0.0):
        raise ValueError("A_ring must be positive for log-space interpolation.")

    if np.any(delta_r <= 0.0):
        raise ValueError("delta_r must be positive for log-space interpolation.")

    return z_vals, a_ring, r_ring, delta_r, w0


def discover_fan_ids(df: pd.DataFrame) -> Tuple[str, ...]:
    """
    Discover fan IDs from columns like A_ring_F01.
    """
    fan_ids = []
    for col in df.columns:
        match = FAN_COL_PATTERN.match(str(col))
        if match is not None:
            fan_ids.append(match.group(1))

    fan_ids = sorted(set(fan_ids))
    valid = []
    for fan_id in fan_ids:
        required = (
            f"A_ring_{fan_id}",
            f"r_ring_{fan_id}",
            f"delta_r_{fan_id}",
            f"w0_{fan_id}",
        )
        if all(col in df.columns for col in required):
            valid.append(fan_id)
    return tuple(valid)


def extract_params_for_fan(
    df: pd.DataFrame,
    fan_id: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract and validate fitted per-fan parameters from a DataFrame.
    """
    cols = (
        "z_m",
        f"A_ring_{fan_id}",
        f"r_ring_{fan_id}",
        f"delta_r_{fan_id}",
        f"w0_{fan_id}",
    )
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in parameter table for {fan_id}: {missing}")

    data = df[list(cols)].copy()
    for col in cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna()

    if data.empty:
        raise ValueError(f"No valid parameter rows found after cleaning for {fan_id}.")

    z_vals = data["z_m"].to_numpy(dtype=float)
    a_ring = data[f"A_ring_{fan_id}"].to_numpy(dtype=float)
    r_ring = data[f"r_ring_{fan_id}"].to_numpy(dtype=float)
    delta_r = data[f"delta_r_{fan_id}"].to_numpy(dtype=float)
    w0 = data[f"w0_{fan_id}"].to_numpy(dtype=float)

    order = np.argsort(z_vals)
    z_vals = z_vals[order]
    a_ring = a_ring[order]
    r_ring = r_ring[order]
    delta_r = delta_r[order]
    w0 = w0[order]

    if np.any(np.diff(z_vals) <= 0.0):
        raise ValueError(f"z_m values must be strictly increasing for {fan_id}.")
    if np.any(a_ring <= 0.0):
        raise ValueError(f"A_ring must be positive for log-space interpolation ({fan_id}).")
    if np.any(delta_r <= 0.0):
        raise ValueError(f"delta_r must be positive for log-space interpolation ({fan_id}).")

    return z_vals, a_ring, r_ring, delta_r, w0


def ring_gaussian(
    r: np.ndarray,
    a_ring: np.ndarray,
    r_ring: np.ndarray,
    delta_r: np.ndarray,
    w0: np.ndarray,
) -> np.ndarray:
    """
    Evaluate the ring-Gaussian model w(r) for given parameters.
    """
    return w0 + a_ring * np.exp(-((r - r_ring) / delta_r) ** 2)


class RingModel:
    """
    Continuous ring-Gaussian model w_model(r, z) using PCHIP interpolation.
    """

    def __init__(
        self,
        z_vals: np.ndarray,
        a_ring: np.ndarray,
        r_ring: np.ndarray,
        delta_r: np.ndarray,
        w0: np.ndarray,
    ) -> None:
        # Main interpolators across the full fitted z range.
        self._a_ring_log = PchipInterpolator(z_vals, np.log(a_ring))
        self._r_ring = PchipInterpolator(z_vals, r_ring)
        self._delta_r_log = PchipInterpolator(z_vals, np.log(delta_r))
        self._w0 = PchipInterpolator(z_vals, w0)

        self._z_min_fit = float(np.min(z_vals))
        self._low_params = (
            float(a_ring[0]),
            float(r_ring[0]),
            float(delta_r[0]),
            float(w0[0]),
        )

        self._high_anchor_enabled = False
        self._high_z_threshold = float(HIGH_Z_THRESHOLD_M)

        anchor_idx = select_anchor_indices(z_vals)
        if anchor_idx is not None:
            z_anchor = z_vals[anchor_idx]
            self._high_a_ring_log = PchipInterpolator(z_anchor, np.log(a_ring[anchor_idx]))
            self._high_r_ring = PchipInterpolator(z_anchor, r_ring[anchor_idx])
            self._high_delta_r_log = PchipInterpolator(z_anchor, np.log(delta_r[anchor_idx]))
            self._high_w0 = PchipInterpolator(z_anchor, w0[anchor_idx])
            self._high_anchor_enabled = True

    @property
    def high_anchor_enabled(self) -> bool:
        """Whether high-z anchor branch is available."""
        return self._high_anchor_enabled

    def params_at(
        self, z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate interpolated parameters at height(s) z.
        """
        z = np.asarray(z, dtype=float)
        z_flat = z.reshape(-1)

        a_ring = np.exp(self._a_ring_log(z_flat))
        r_ring = self._r_ring(z_flat)
        delta_r = np.exp(self._delta_r_log(z_flat))
        w0 = self._w0(z_flat)

        if self._high_anchor_enabled:
            blend = high_branch_weight(z_flat)
            high_mask = blend > 0.0
            if np.any(high_mask):
                z_high = z_flat[high_mask]
                a_high = np.exp(self._high_a_ring_log(z_high))
                r_high = self._high_r_ring(z_high)
                delta_high = np.exp(self._high_delta_r_log(z_high))
                w0_high = self._high_w0(z_high)

                b = blend[high_mask]
                a_ring[high_mask] = (1.0 - b) * a_ring[high_mask] + b * a_high
                r_ring[high_mask] = (1.0 - b) * r_ring[high_mask] + b * r_high
                delta_r[high_mask] = (1.0 - b) * delta_r[high_mask] + b * delta_high
                w0[high_mask] = (1.0 - b) * w0[high_mask] + b * w0_high

        if ENABLE_LOW_Z_EDGE_HOLD:
            low_mask = z_flat < self._z_min_fit
            if np.any(low_mask):
                a_ring[low_mask] = self._low_params[0]
                r_ring[low_mask] = self._low_params[1]
                delta_r[low_mask] = self._low_params[2]
                w0[low_mask] = self._low_params[3]

        # Numerical safety for positive parameters.
        a_ring = np.maximum(a_ring, 1e-12)
        delta_r = np.maximum(delta_r, 1e-12)

        a_ring = a_ring.reshape(z.shape)
        r_ring = r_ring.reshape(z.shape)
        delta_r = delta_r.reshape(z.shape)
        w0 = w0.reshape(z.shape)
        return a_ring, r_ring, delta_r, w0

    def __call__(self, r: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Evaluate w_model(r, z).
        """
        r = np.asarray(r, dtype=float)
        z = np.asarray(z, dtype=float)
        a_ring, r_ring, delta_r, w0 = self.params_at(z)
        return ring_gaussian(r, a_ring=a_ring, r_ring=r_ring, delta_r=delta_r, w0=w0)


def make_z_grid(z_vals: np.ndarray) -> np.ndarray:
    """
    Build a regular z grid for output.
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


def write_interpolated_table(
    z_grid: np.ndarray,
    model: RingModel,
    out_path: Path,
    sheet_name: str,
) -> None:
    """
    Write interpolated parameters to an Excel sheet.
    """
    a_ring, r_ring, delta_r, w0 = model.params_at(z_grid)

    df_out = pd.DataFrame(
        {
            "z_m": z_grid,
            "A_ring": a_ring,
            "r_ring": r_ring,
            "delta_r": delta_r,
            "w0": w0,
        }
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_excel(out_path, index=False, sheet_name=sheet_name)


def write_interpolated_table_multi(
    z_grid: np.ndarray,
    fan_ids: Tuple[str, ...],
    fan_models: Tuple[RingModel, ...],
    out_path: Path,
    sheet_name: str,
) -> None:
    """
    Write interpolated per-fan parameters plus legacy averaged columns.
    """
    out_data = {"z_m": z_grid}

    a_stack = []
    r_stack = []
    d_stack = []
    w0_stack = []
    for fan_id, model in zip(fan_ids, fan_models):
        a_ring, r_ring, delta_r, w0 = model.params_at(z_grid)
        out_data[f"A_ring_{fan_id}"] = a_ring
        out_data[f"r_ring_{fan_id}"] = r_ring
        out_data[f"delta_r_{fan_id}"] = delta_r
        out_data[f"w0_{fan_id}"] = w0
        a_stack.append(a_ring)
        r_stack.append(r_ring)
        d_stack.append(delta_r)
        w0_stack.append(w0)

    out_data["A_ring"] = np.mean(np.vstack(a_stack), axis=0)
    out_data["r_ring"] = np.mean(np.vstack(r_stack), axis=0)
    out_data["delta_r"] = np.mean(np.vstack(d_stack), axis=0)
    out_data["w0"] = np.mean(np.vstack(w0_stack), axis=0)

    df_out = pd.DataFrame(out_data)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_excel(out_path, index=False, sheet_name=sheet_name)


### Main
def main() -> None:
    params_df = load_params_table(PARAMS_XLSX, PARAMS_SHEET)
    fan_ids = discover_fan_ids(params_df)

    if len(fan_ids) == 0:
        z_vals, a_ring, r_ring, delta_r, w0 = extract_params(params_df)
        model = RingModel(
            z_vals=z_vals,
            a_ring=a_ring,
            r_ring=r_ring,
            delta_r=delta_r,
            w0=w0,
        )
        z_grid = make_z_grid(z_vals)
        write_interpolated_table(z_grid, model, OUT_XLSX_PATH, OUT_SHEET_NAME)
        enabled_str = str(model.high_anchor_enabled)
    else:
        models = []
        z_ref = None
        for fan_id in fan_ids:
            z_vals, a_ring, r_ring, delta_r, w0 = extract_params_for_fan(params_df, fan_id)
            if z_ref is None:
                z_ref = z_vals
            elif not np.allclose(z_ref, z_vals, atol=1e-12, rtol=0.0):
                raise ValueError("Per-fan z grids are inconsistent.")

            model = RingModel(
                z_vals=z_vals,
                a_ring=a_ring,
                r_ring=r_ring,
                delta_r=delta_r,
                w0=w0,
            )
            models.append(model)

        if z_ref is None:
            raise ValueError("No fan parameter sets were discovered.")

        z_grid = make_z_grid(z_ref)
        write_interpolated_table_multi(
            z_grid=z_grid,
            fan_ids=fan_ids,
            fan_models=tuple(models),
            out_path=OUT_XLSX_PATH,
            sheet_name=OUT_SHEET_NAME,
        )
        enabled_str = ", ".join(
            [f"{fan_id}:{model.high_anchor_enabled}" for fan_id, model in zip(fan_ids, models)]
        )

    print("Built continuous ring-Gaussian model using PCHIP.")
    print(f"Input:  {PARAMS_XLSX.resolve()}")
    print(f"Output: {OUT_XLSX_PATH.resolve()}")
    print(f"High-z anchor branch enabled: {enabled_str}")


if __name__ == "__main__":
    main()
