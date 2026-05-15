from __future__ import annotations

import numpy as np

from rollout import rk4_step


def discrete_linearise_rollout_map(
    *,
    x_ref: np.ndarray,
    u_ff: np.ndarray,
    times_s: np.ndarray,
    aircraft: object,
    wind_model: object | None,
    wind_mode: str,
    rho_kg_m3: float,
    actuator_tau_s: tuple[float, float, float],
    eps_x: float = 1e-5,
    eps_u: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return A_d[k], B_d[k] for the discrete RK4 rollout map."""
    states = np.asarray(x_ref, dtype=float)
    commands = np.asarray(u_ff, dtype=float)
    times = np.asarray(times_s, dtype=float)
    if states.ndim != 2 or states.shape[1] != 15:
        raise ValueError("x_ref must have shape (N, 15).")
    if commands.shape != (states.shape[0], 3):
        raise ValueError("u_ff must have shape (N, 3).")
    if times.shape != (states.shape[0],):
        raise ValueError("times_s must have shape (N,).")
    n_steps = times.size - 1
    a_d = np.empty((n_steps, 15, 15), dtype=float)
    b_d = np.empty((n_steps, 15, 3), dtype=float)
    for k in range(n_steps):
        dt_s = float(times[k + 1] - times[k])
        xk = states[k]
        uk = commands[k]
        for idx in range(15):
            dx = np.zeros(15, dtype=float)
            dx[idx] = float(eps_x)
            f_plus = _rk4_map(
                xk + dx,
                uk,
                dt_s,
                aircraft,
                wind_model,
                wind_mode,
                rho_kg_m3,
                actuator_tau_s,
            )
            f_minus = _rk4_map(
                xk - dx,
                uk,
                dt_s,
                aircraft,
                wind_model,
                wind_mode,
                rho_kg_m3,
                actuator_tau_s,
            )
            a_d[k, :, idx] = (f_plus - f_minus) / (2.0 * float(eps_x))
        for idx in range(3):
            du = np.zeros(3, dtype=float)
            du[idx] = float(eps_u)
            f_plus = _rk4_map(
                xk,
                uk + du,
                dt_s,
                aircraft,
                wind_model,
                wind_mode,
                rho_kg_m3,
                actuator_tau_s,
            )
            f_minus = _rk4_map(
                xk,
                uk - du,
                dt_s,
                aircraft,
                wind_model,
                wind_mode,
                rho_kg_m3,
                actuator_tau_s,
            )
            b_d[k, :, idx] = (f_plus - f_minus) / (2.0 * float(eps_u))
    if not np.all(np.isfinite(a_d)) or not np.all(np.isfinite(b_d)):
        raise FloatingPointError("discrete aggressive rollout linearisation produced non-finite values.")
    return a_d, b_d


def solve_aggressive_discrete_tvlqr(
    *,
    a_d: np.ndarray,
    b_d: np.ndarray,
    q_diag: tuple[float, ...],
    r_diag: tuple[float, float, float],
    qf_diag: tuple[float, ...] | None = None,
    phase_labels: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return K[k], S[k] for the aggressive reversal trajectory."""
    a = np.asarray(a_d, dtype=float)
    b = np.asarray(b_d, dtype=float)
    if a.ndim != 3 or a.shape[1:] != (15, 15):
        raise ValueError("a_d must have shape (N-1, 15, 15).")
    if b.shape != (a.shape[0], 15, 3):
        raise ValueError("b_d must have shape (N-1, 15, 3).")
    q = np.diag(_diag15(q_diag, "q_diag"))
    r = np.diag(_diag3(r_diag, "r_diag"))
    qf = np.diag(_diag15(qf_diag if qf_diag is not None else q_diag, "qf_diag"))
    n_steps = a.shape[0]
    gains = np.zeros((n_steps + 1, 3, 15), dtype=float)
    s_mats = np.zeros((n_steps + 1, 15, 15), dtype=float)
    s_mats[-1] = qf
    current = qf
    labels = phase_labels or tuple("" for _ in range(n_steps + 1))
    for k in range(n_steps - 1, -1, -1):
        ak = a[k]
        bk = b[k]
        h = r + bk.T @ current @ bk
        g = bk.T @ current @ ak
        try:
            k_gain = np.linalg.solve(h + 1e-9 * np.eye(3), g)
        except np.linalg.LinAlgError:
            k_gain = np.linalg.pinv(h + 1e-7 * np.eye(3)) @ g
        # Feedback is deliberately softened while the high-incidence pitch-brake/redirect
        # phase is trying to create the manoeuvre, then restored for capture/recovery.
        if labels and k < len(labels) and labels[k] in {"pitch_brake", "yaw_roll_redirect"}:
            k_gain *= 0.35
        gains[k] = k_gain
        closed = ak - bk @ k_gain
        current = q + closed.T @ current @ closed + k_gain.T @ r @ k_gain
        current = 0.5 * (current + current.T)
        s_mats[k] = current
    if not np.all(np.isfinite(gains)) or not np.all(np.isfinite(s_mats)):
        raise FloatingPointError("aggressive TVLQR produced non-finite gains.")
    return gains, s_mats


def _rk4_map(
    x: np.ndarray,
    u: np.ndarray,
    dt_s: float,
    aircraft: object,
    wind_model: object | None,
    wind_mode: str,
    rho_kg_m3: float,
    actuator_tau_s: tuple[float, float, float],
) -> np.ndarray:
    return rk4_step(
        x=np.asarray(x, dtype=float),
        u_cmd=np.asarray(u, dtype=float),
        dt_s=float(dt_s),
        aircraft=aircraft,
        wind_model=wind_model,
        rho_kg_m3=float(rho_kg_m3),
        actuator_tau_s=actuator_tau_s,
        wind_mode=wind_mode,
    )


def _diag15(values: tuple[float, ...], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != (15,):
        raise ValueError(f"{name} must contain 15 entries.")
    return arr


def _diag3(values: tuple[float, float, float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must contain 3 entries.")
    return arr

