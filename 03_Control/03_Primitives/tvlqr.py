from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flight_dynamics import AircraftModel, state_derivative


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) TVLQR config
# 2) Trajectory finite-difference linearisation
# 3) Discrete finite-horizon TVLQR
# =============================================================================

# =============================================================================
# 1) TVLQR Config
# =============================================================================
@dataclass(frozen=True)
class TVLQRConfig:
    q_diag: tuple[float, ...]
    r_diag: tuple[float, float, float]
    qf_diag: tuple[float, ...] | None = None
    regularisation: float = 1e-9


# =============================================================================
# 2) Trajectory Finite-Difference Linearisation
# =============================================================================
def linearise_trajectory_finite_difference(
    x_ref: np.ndarray,
    u_ff: np.ndarray,
    aircraft: AircraftModel,
    wind_model: object,
    wind_mode: str,
    rho_kg_m3: float,
    actuator_tau_s: tuple[float, float, float],
    eps_x: float = 1e-5,
    eps_u: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return continuous-time A[k], B[k] for the canonical 15-state model."""
    x_arr = np.asarray(x_ref, dtype=float)
    u_arr = np.asarray(u_ff, dtype=float)
    if x_arr.ndim != 2 or x_arr.shape[1] != 15:
        raise ValueError("x_ref must have shape (N, 15).")
    if u_arr.shape != (x_arr.shape[0], 3):
        raise ValueError("u_ff must have shape (N, 3).")
    n_samples = x_arr.shape[0]
    a_mats = np.empty((n_samples, 15, 15), dtype=float)
    b_mats = np.empty((n_samples, 15, 3), dtype=float)
    for k in range(n_samples):
        x0 = x_arr[k]
        u0 = u_arr[k]
        for idx in range(15):
            dx = np.zeros(15, dtype=float)
            dx[idx] = float(eps_x)
            f_plus = _f(
                x0 + dx,
                u0,
                aircraft,
                wind_model,
                wind_mode,
                rho_kg_m3,
                actuator_tau_s,
            )
            f_minus = _f(
                x0 - dx,
                u0,
                aircraft,
                wind_model,
                wind_mode,
                rho_kg_m3,
                actuator_tau_s,
            )
            a_mats[k, :, idx] = (f_plus - f_minus) / (2.0 * float(eps_x))
        for idx in range(3):
            du = np.zeros(3, dtype=float)
            du[idx] = float(eps_u)
            f_plus = _f(
                x0,
                u0 + du,
                aircraft,
                wind_model,
                wind_mode,
                rho_kg_m3,
                actuator_tau_s,
            )
            f_minus = _f(
                x0,
                u0 - du,
                aircraft,
                wind_model,
                wind_mode,
                rho_kg_m3,
                actuator_tau_s,
            )
            b_mats[k, :, idx] = (f_plus - f_minus) / (2.0 * float(eps_u))
    return a_mats, b_mats


def _f(
    x: np.ndarray,
    u_cmd: np.ndarray,
    aircraft: AircraftModel,
    wind_model: object,
    wind_mode: str,
    rho_kg_m3: float,
    actuator_tau_s: tuple[float, float, float],
) -> np.ndarray:
    return state_derivative(
        x=x,
        u_cmd=u_cmd,
        aircraft=aircraft,
        wind_model=wind_model,
        rho=float(rho_kg_m3),
        actuator_tau_s=actuator_tau_s,
        wind_mode=wind_mode,
    )


# =============================================================================
# 3) Discrete Finite-Horizon TVLQR
# =============================================================================
def solve_discrete_tvlqr(
    a_mats: np.ndarray,
    b_mats: np.ndarray,
    dt_s: float,
    config: TVLQRConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return K[k] and S[k] for a finite-horizon trajectory."""
    a_arr = np.asarray(a_mats, dtype=float)
    b_arr = np.asarray(b_mats, dtype=float)
    if a_arr.ndim != 3 or a_arr.shape[1:] != (15, 15):
        raise ValueError("a_mats must have shape (N, 15, 15).")
    if b_arr.shape != (a_arr.shape[0], 15, 3):
        raise ValueError("b_mats must have shape (N, 15, 3).")
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive.")
    n_samples = a_arr.shape[0]
    q = _diag(config.q_diag, 15, "q_diag")
    qf = q if config.qf_diag is None else _diag(config.qf_diag, 15, "qf_diag")
    r = _diag(config.r_diag, 3, "r_diag")
    regularised_r = r + float(config.regularisation) * np.eye(3)
    k_lqr = np.zeros((n_samples, 3, 15), dtype=float)
    s_mats = np.empty((n_samples, 15, 15), dtype=float)
    s_next = qf
    for k in range(n_samples - 1, -1, -1):
        ad = np.eye(15) + float(dt_s) * a_arr[k]
        bd = float(dt_s) * b_arr[k]
        lhs = regularised_r + bd.T @ s_next @ bd
        rhs = bd.T @ s_next @ ad
        k_gain = np.linalg.solve(lhs, rhs)
        s_current = q + ad.T @ s_next @ (ad - bd @ k_gain)
        s_current = 0.5 * (s_current + s_current.T)
        k_lqr[k] = k_gain
        s_mats[k] = s_current
        s_next = s_current
    if not np.all(np.isfinite(k_lqr)) or not np.all(np.isfinite(s_mats)):
        raise FloatingPointError("TVLQR solution contains non-finite values.")
    return k_lqr, s_mats


def _diag(values: tuple[float, ...], length: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != (length,):
        raise ValueError(f"{name} must contain {length} values.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite.")
    return np.diag(arr)
