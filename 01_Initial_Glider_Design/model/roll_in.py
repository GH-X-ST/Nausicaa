from __future__ import annotations

import numpy as onp
import aerosandbox as asb
import aerosandbox.numpy as np

from config import Config


def add_roll_in_constraints(
    opti: asb.Opti,
    cfg: Config,
    mission: dict,
    geom: dict,
    mass: dict,
    aero_pack: dict,
    thermal: dict,
) -> dict:
    """
    Roll-in maneuver constraints (full roll dynamics) + orbit entry constraints.
    Vectorized implementation to reduce Opti graph build time.

    Returns a dict of decision variables / trajectories for objective + postprocessing.
    """
    bnd = cfg.bounds
    cst = cfg.constants

    wing = geom["wing"]
    b_w = geom["b_w"]

    op_point = mission["op_point"]
    phi_rad = mission["phi_rad"]
    r_target = mission["r_target"]

    a = cfg.arena
    x_min, x_max = a.x_min, a.x_max
    y_min, y_max = a.y_min, a.y_max

    # Initial position (as in your original code)
    x0 = 0.0
    y0 = thermal["y_center"]

    # Decision variables
    psi0_deg = opti.variable(
        init_guess=0.0,
        lower_bound=bnd.psi0_min_deg,
        upper_bound=bnd.psi0_max_deg,
    )
    psi0 = np.radians(psi0_deg)

    t_roll = opti.variable(
        init_guess=0.7,
        lower_bound=bnd.t_roll_min,
        upper_bound=bnd.t_roll_max,
    )

    n_roll = cfg.roll_in.n_roll
    if n_roll < 2:
        raise ValueError("cfg.roll_in.n_roll must be >= 2 for roll-in dynamics.")

    dt = t_roll / (n_roll - 1)

    # Trajectory variables
    p_roll = opti.variable(init_guess=np.zeros(n_roll))
    phi_roll = opti.variable(init_guess=np.zeros(n_roll))
    psi = opti.variable(init_guess=np.zeros(n_roll))
    x = opti.variable(init_guess=x0 * np.ones(n_roll))
    y = opti.variable(init_guess=y0 * np.ones(n_roll))

    # Control (aileron) along the roll-in
    delta_a_roll_deg = opti.variable(
        init_guess=5.0 * onp.ones(n_roll),
        lower_bound=bnd.delta_a_min_deg,
        upper_bound=bnd.delta_a_max_deg,
    )
    delta_a_roll_rad = np.radians(delta_a_roll_deg)

    # Initial conditions
    opti.subject_to(
        [
            p_roll[0] == 0.0,
            phi_roll[0] == 0.0,
            psi[0] == psi0,
            x[0] == x0,
            y[0] == y0,
            delta_a_roll_deg[0] == 0.0,
        ]
    )

    # Precompute constants
    V = op_point.velocity
    q_dyn = 0.5 * cst.rho * V**2
    s_w = wing.area()
    i_xx = mass["mass_props_togw"].inertia_tensor[0, 0]

    aero = aero_pack["aero"]
    cl_delta_a = aero_pack["cl_delta_a"]
    Clp = aero["Clp"]

    # Vectorized slices for k = 0..n-2
    p_k = p_roll[:-1]
    phi_k = phi_roll[:-1]
    psi_k = psi[:-1]
    da_k = delta_a_roll_rad[:-1]  # radians for aero model

    # Roll dynamics
    cl_eff = cl_delta_a * da_k + Clp * p_k * b_w / (2 * V)
    l_roll = q_dyn * s_w * b_w * cl_eff
    p_dot = l_roll / i_xx

    # Coordinated turn yaw rate (psi_dot)
    r_k = cst.g * np.tan(phi_k) / V

    # Dynamics constraints (explicit Euler), vectorized
    opti.subject_to(
        [
            p_roll[1:] == p_k + p_dot * dt,
            phi_roll[1:] == phi_k + p_k * dt,
            psi[1:] == psi_k + r_k * dt,
            x[1:] == x[:-1] + V * np.cos(psi_k) * dt,
            y[1:] == y[:-1] + V * np.sin(psi_k) * dt,
        ]
    )

    # Path constraints (vectorized)
    opti.subject_to(
        [
            opti.bounded(x_min, x[1:], x_max),
            opti.bounded(y_min, y[1:], y_max),
            opti.bounded(0.0, p_roll[1:], bnd.p_roll_max),
        ]
    )

    # Aileron rate constraints (vectorized)
    d_delta_a_deg = delta_a_roll_deg[1:] - delta_a_roll_deg[:-1]
    opti.subject_to(
        [
            d_delta_a_deg <= bnd.delta_a_rate_max_deg_s * dt,
            d_delta_a_deg >= -bnd.delta_a_rate_max_deg_s * dt,
            delta_a_roll_deg[1:] <= bnd.delta_a_max_deg,
        ]
    )

    # End-of-roll constraints (enter steady turn)
    dx_end = x[-1] - thermal["x_center"]
    dy_end = y[-1] - thermal["y_center"]

    opti.subject_to(
        [
            p_roll[-1] == 0.0,
            delta_a_roll_deg[-1] == mission["controls"]["delta_a_deg"],
            phi_roll[-1] == phi_rad,
            (x[-1] - thermal["x_center"]) ** 2 + (y[-1] - thermal["y_center"]) ** 2
            == r_target**2,
            # tangency condition
            np.cos(psi[-1]) * dx_end + np.sin(psi[-1]) * dy_end == 0.0,
            # correct direction of travel around circle
            -np.cos(psi[-1]) * dy_end + np.sin(psi[-1]) * dx_end >= 0.0,
        ]
    )

    return {
        "psi0_deg": psi0_deg,
        "psi0": psi0,
        "t_roll": t_roll,
        "dt": dt,
        "p_roll": p_roll,
        "phi_roll": phi_roll,
        "psi": psi,
        "x": x,
        "y": y,
        "delta_a_roll_deg": delta_a_roll_deg,
    }


def orbit_r_max(cfg: Config, x_center: float, y_center: float) -> float:
    """Find maximum orbit radius."""
    a = cfg.arena
    return min(
        x_center - a.x_min,
        a.x_max - x_center,
        y_center - a.y_min,
        a.y_max - y_center,
    )
