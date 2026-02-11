"""
Ring-Gaussian identification for single-fan updraft maps in S01.xlsx (PEP 8).

This script fits an annular (ring-shaped) Gaussian model to each height sheet
(z020, z035, z050, z075, z110, z160, z220) and then interpolates parameters
vs height z using PCHIP.

Data expectations (mean-map sheets):
- Sheet names: z020, z035, z050, z075, z110, z160, z220
- Each sheet stores a rectangular grid:
    Row 0, Col 1..end : x-coordinates (m)
    Col 0, Row 1..end : y-coordinates (m)
    Rows 1..end, Cols 1..end : mean vertical velocity w (m/s)

Time-series sheets:
- Sheet names: z020_TS, z035_TS, ..., z220_TS
- Each contains 15 samples at 1 Hz for 6 representative points, plus mean/variance.
- The parser is intentionally generic: it searches for header cells equal to
  "variance" (case-insensitive) and takes the first numeric cell below each.
  It then sets:
      sigma_z = sqrt(mean(variances))

Model (per height z_k), axisymmetric about (x_c, y_c):
    r = sqrt((x - x_c)^2 + (y - y_c)^2)

    w_m(r; z_k) = w0(z_k) + A_r(z_k) * exp(-((r - r_p(z_k)) / sigma_r(z_k))^2)

This ring model is minimal at r = 0 and increases for r in [0, r_p] (assuming
A_r > 0 and r_p > 0), consistent with an annular peak.

Fitting procedure per height:
1) Convert the 2D grid to (r_i, w_i).
2) Bin in radius with bin width delta_r to obtain a 1D profile (r_j, w_j, n_j).
3) Fit parameters [A_r, r_p, sigma_r, w0] by bounded robust least squares
   using scipy.optimize.least_squares(loss="soft_l1").
   Residuals are normalised by sigma_z (estimated from *_TS), so f_scale can
   remain 1.0. The robust transition scale in physical units is ~sigma_z.

Outputs:
- Printed fitted parameters per height.
- A callable SmoothRingModel for w(x, y, z) via PCHIP interpolation.

Notes:
- Sheet naming convention: z110 is interpreted as 1.10 m (110/100), not 0.110 m.
- Update FAN_CENTER_XY to match your coordinate system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares


# -----------------------------
# User-configurable parameters
# -----------------------------

XLSX_PATH = "/mnt/data/S01.xlsx"

# Update if your fan center is different.
FAN_CENTER_XY = (4.2, 2.4)

# Fan geometry (used only for commentary / reasonable bounds guidance).
FAN_DIAMETER_M = 0.8
FAN_RADIUS_M = FAN_DIAMETER_M / 2.0

# z020 -> 0.20 m, z110 -> 1.10 m, etc.
SHEET_HEIGHT_DIVISOR = 100.0


# -----------------------------
# Data classes
# -----------------------------

@dataclass(frozen=True)
class FitConfig:
    """Configuration for loading, profiling, and fitting."""

    xlsx_path: str = XLSX_PATH
    fan_center_xy: Tuple[float, float] = FAN_CENTER_XY

    # Radial bin width for constructing 1D radial profile from 2D grid (m).
    # Choose comparable to your x/y spacing; 0.03–0.08 m is typical.
    delta_r_m: float = 0.05

    # Robust least-squares options for scipy.optimize.least_squares.
    # Since residuals are normalised by sigma_z, keep f_scale ~ 1.
    robust_loss: str = "soft_l1"
    robust_f_scale: float = 1.0

    # If TS parsing fails, fallback sigma_z (m/s).
    sigma_z_fallback: float = 0.2

    # Use median rather than mean within each radial bin (more robust).
    use_median_profile: bool = False


@dataclass(frozen=True)
class RingBounds:
    """Parameter bounds to keep fits identifiable and physically plausible."""

    rp_min: float
    rp_max: float
    sigma_r_min: float = 0.15
    sigma_r_max: float = 1.00
    a_r_max: float = 50.0
    w0_min: float = -0.50
    w0_max: float = 0.50


@dataclass
class SmoothRingModel:
    """
    Smooth model w(x, y, z) using PCHIP interpolators over z.

    A_r(z) and sigma_r(z) are interpolated in log-space to ensure positivity.
    """

    a_r_log: PchipInterpolator
    r_p: PchipInterpolator
    sigma_r_log: PchipInterpolator
    w0: PchipInterpolator
    fan_center_xy: Tuple[float, float]

    def __call__(self, x: np.ndarray, y: np.ndarray, z: float) -> np.ndarray:
        xc, yc = self.fan_center_xy
        r = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

        a_r = float(np.exp(self.a_r_log(z)))
        r_p = float(self.r_p(z))
        sigma_r = float(np.exp(self.sigma_r_log(z)))
        w0 = float(self.w0(z))

        return ring_gaussian(r, a_r=a_r, r_p=r_p, sigma_r=sigma_r, w0=w0)


# -----------------------------
# Utilities: parsing and loading
# -----------------------------

def parse_sheet_height_m(sheet_name: str) -> float:
    """
    Parse height in meters from sheet names like 'z020', 'z110', 'z220'.

    Interprets zXYZ as XYZ / 100 meters:
        z020 -> 0.20 m
        z110 -> 1.10 m
        z220 -> 2.20 m
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


# -----------------------------
# Model and profiling
# -----------------------------

def ring_gaussian(r: np.ndarray, a_r: float, r_p: float, sigma_r: float, w0: float) -> np.ndarray:
    """Evaluate the ring Gaussian model w(r) = w0 + A_r * exp(-((r - r_p) / sigma_r)^2)."""
    return w0 + a_r * np.exp(-((r - r_p) / sigma_r) ** 2)


def make_radial_profile(
    r: np.ndarray,
    w: np.ndarray,
    delta_r: float,
    use_median: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct a 1D radial profile by binning samples (r_i, w_i) into radius bins.

    Args:
        r: radii, shape (N,)
        w: velocities, shape (N,)
        delta_r: radial bin width (m)
        use_median: if True, use median within bins; else mean

    Returns:
        r_bins: representative radius per bin, shape (M,)
        w_bins: aggregated velocity per bin, shape (M,)
        n_bins: counts per bin, shape (M,)
    """
    if delta_r <= 0.0:
        raise ValueError("delta_r must be positive.")

    mask = np.isfinite(r) & np.isfinite(w)
    r = r[mask]
    w = w[mask]

    if r.size == 0:
        raise ValueError("No finite samples available to construct radial profile.")

    r_max = float(np.max(r))
    edges = np.arange(0.0, r_max + delta_r, delta_r)
    if edges.size < 2:
        edges = np.array([0.0, r_max + delta_r], dtype=float)

    bin_idx = np.digitize(r, edges) - 1
    bin_idx = np.clip(bin_idx, 0, edges.size - 2)

    r_list = []
    w_list = []
    n_list = []

    for b in range(edges.size - 1):
        in_bin = bin_idx == b
        if not np.any(in_bin):
            continue

        r_b = r[in_bin]
        w_b = w[in_bin]

        r_rep = float(np.mean(r_b))
        w_rep = float(np.median(w_b) if use_median else np.mean(w_b))
        n_rep = int(np.sum(in_bin))

        r_list.append(r_rep)
        w_list.append(w_rep)
        n_list.append(n_rep)

    r_bins = np.array(r_list, dtype=float)
    w_bins = np.array(w_list, dtype=float)
    n_bins = np.array(n_list, dtype=int)

    order = np.argsort(r_bins)
    return r_bins[order], w_bins[order], n_bins[order]


# -----------------------------
# Fitting
# -----------------------------

def default_rp_bounds_by_z(z_m: float) -> Tuple[float, float]:
    """
    Height-dependent bounds for ring peak radius r_p(z).

    Near the outlet, constrain around the fan radius band; higher up allow expansion.
    Adjust to taste based on your observed maps.
    """
    if z_m <= 0.35:
        return 0.20, 0.65
    return 0.15, 0.95


def initial_guess_ring(
    r_bins: np.ndarray,
    w_bins: np.ndarray,
    rp_bounds: Tuple[float, float],
) -> np.ndarray:
    """
    Initial guess for parameters [A_r, r_p, sigma_r, w0].

    - w0: median of outer bins
    - r_p: radius at maximum profile value (clipped)
    - A_r: peak - w0
    - sigma_r: moderate default
    """
    tail_len = max(5, int(0.2 * w_bins.size))
    w0 = float(np.median(w_bins[-tail_len:]))

    peak_idx = int(np.argmax(w_bins))
    r_p0 = float(np.clip(r_bins[peak_idx], rp_bounds[0], rp_bounds[1]))

    a_r0 = float(max(w_bins[peak_idx] - w0, 0.1))
    sigma_r0 = 0.25

    return np.array([a_r0, r_p0, sigma_r0, w0], dtype=float)


def fit_ring_at_height(
    r_bins: np.ndarray,
    w_bins: np.ndarray,
    n_bins: np.ndarray,
    sigma_z: float,
    bounds: RingBounds,
    config: FitConfig,
) -> np.ndarray:
    """
    Fit parameters [A_r, r_p, sigma_r, w0] at a single height.

    Residual definition (dimensionless):
        res_j = alpha_j * (w_pred(r_j) - w_j) / sigma_z
    where alpha_j = sqrt(n_j / max(n)).

    With loss="soft_l1" and f_scale=1.0, the transition from quadratic to
    approximately linear occurs at |res_j| ~ 1, i.e. |w_pred - w_j| ~ sigma_z/alpha_j.
    """
    if sigma_z <= 0.0:
        raise ValueError("sigma_z must be positive.")

    p0 = initial_guess_ring(r_bins, w_bins, (bounds.rp_min, bounds.rp_max))

    lower = np.array([0.0, bounds.rp_min, bounds.sigma_r_min, bounds.w0_min], dtype=float)
    upper = np.array([bounds.a_r_max, bounds.rp_max, bounds.sigma_r_max, bounds.w0_max], dtype=float)

    alpha = np.sqrt(n_bins / np.max(n_bins))

    def residuals(p: np.ndarray) -> np.ndarray:
        w_pred = ring_gaussian(r_bins, a_r=p[0], r_p=p[1], sigma_r=p[2], w0=p[3])
        return alpha * (w_pred - w_bins) / sigma_z

    result = least_squares(
        residuals,
        p0,
        bounds=(lower, upper),
        loss=config.robust_loss,
        f_scale=config.robust_f_scale,
    )
    return result.x


def fit_all_heights(
    mean_sheet_names: Iterable[str],
    config: FitConfig,
    rp_bounds_by_z: Callable[[float], Tuple[float, float]] = default_rp_bounds_by_z,
) -> Tuple[np.ndarray, np.ndarray, SmoothRingModel]:
    """
    Fit ring model at all heights and build a smoothed model w(x, y, z).

    Returns:
        z_vals: shape (K,), heights in meters
        params: shape (K, 4), fitted parameters [A_r, r_p, sigma_r, w0]
        model: SmoothRingModel callable
    """
    xc, yc = config.fan_center_xy
    z_list = []
    params_list = []

    for sheet in mean_sheet_names:
        z_m = parse_sheet_height_m(sheet)
        ts_sheet = f"{sheet}_TS"

        x_grid, y_grid, w_grid = load_mean_map(config.xlsx_path, sheet)

        r = np.sqrt((x_grid - xc) ** 2 + (y_grid - yc) ** 2).ravel()
        w = w_grid.ravel()

        r_bins, w_bins, n_bins = make_radial_profile(
            r=r,
            w=w,
            delta_r=config.delta_r_m,
            use_median=config.use_median_profile,
        )

        sigma_z = parse_ts_noise_scale(config.xlsx_path, ts_sheet)
        if sigma_z is None:
            sigma_z = config.sigma_z_fallback

        rp_min, rp_max = rp_bounds_by_z(z_m)
        bounds = RingBounds(rp_min=rp_min, rp_max=rp_max)

        p = fit_ring_at_height(
            r_bins=r_bins,
            w_bins=w_bins,
            n_bins=n_bins,
            sigma_z=sigma_z,
            bounds=bounds,
            config=config,
        )

        z_list.append(z_m)
        params_list.append(p)

    z_vals = np.array(z_list, dtype=float)
    params = np.vstack(params_list)

    # Sort by height for interpolation.
    order = np.argsort(z_vals)
    z_vals = z_vals[order]
    params = params[order]

    # Smooth across z using PCHIP; enforce positivity using log-space.
    eps = 1e-12
    a_r_log = PchipInterpolator(z_vals, np.log(np.maximum(params[:, 0], eps)))
    r_p = PchipInterpolator(z_vals, params[:, 1])
    sigma_r_log = PchipInterpolator(z_vals, np.log(np.maximum(params[:, 2], eps)))
    w0 = PchipInterpolator(z_vals, params[:, 3])

    model = SmoothRingModel(
        a_r_log=a_r_log,
        r_p=r_p,
        sigma_r_log=sigma_r_log,
        w0=w0,
        fan_center_xy=config.fan_center_xy,
    )

    return z_vals, params, model


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    mean_sheets = ["z020", "z035", "z050", "z075", "z110", "z160", "z220"]

    config = FitConfig(
        xlsx_path=XLSX_PATH,
        fan_center_xy=FAN_CENTER_XY,
        delta_r_m=0.05,
        robust_loss="soft_l1",
        robust_f_scale=1.0,
        sigma_z_fallback=0.2,
        use_median_profile=False,
    )

    z_vals, params, _model = fit_all_heights(mean_sheets, config=config)

    print("Fitted ring-Gaussian parameters per height")
    print("z [m]     A_r        r_p      sigma_r       w0")
    for z_m, (a_r, r_p, sigma_r, w0) in zip(z_vals, params):
        print(f"{z_m:5.2f}  {a_r:9.4f}  {r_p:8.4f}  {sigma_r:10.4f}  {w0:9.4f}")


if __name__ == "__main__":
    main()