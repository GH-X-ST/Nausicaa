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
    Returns the decision variables / trajectories needed for objective + postprocessing.
    """
    bnd = cfg.bounds
    cst = cfg.constants

    wing = geom["wing"]
    b_w = geom["b_w"]

    op_point = mission["op_point"]
    phi_rad = mission["phi_rad"]
    r_target = mission["r_target"]

    x_min, x_max = 0.0, 8.0
    y_min, y_max = 0.0, 5.0

    x0 = 0.0
    y0 = thermal["y_center"]

    psi0_deg = opti.variable(
        init_guess=0.0, lower_bound=bnd.psi0_min_deg, upper_bound=bnd.psi0_max_deg
    )
    psi0 = np.radians(psi0_deg)

    t_roll = opti.variable(init_guess=0.7, lower_bound=bnd.t_roll_min, upper_bound=bnd.t_roll_max)

    n_roll = cfg.roll_in.n_roll
    dt = t_roll / (n_roll - 1)

    p_roll = opti.variable(init_guess=np.zeros(n_roll))
    phi_roll = opti.variable(init_guess=np.zeros(n_roll))
    psi = opti.variable(init_guess=0.0 * np.ones(n_roll))
    x = opti.variable(init_guess=x0 * np.ones(n_roll))
    y = opti.variable(init_guess=y0 * np.ones(n_roll))

    delta_a_roll_deg = opti.variable(
        init_guess=5.0 * onp.ones(n_roll),
        lower_bound=bnd.delta_a_min_deg,
        upper_bound=bnd.delta_a_max_deg,
    )
    delta_a_roll_rad = np.radians(delta_a_roll_deg)

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

    q_dyn = 0.5 * cst.rho * op_point.velocity**2
    s_w = wing.area()
    i_xx = mass["mass_props_togw"].inertia_tensor[0, 0]

    aero = aero_pack["aero"]
    cl_delta_a = aero_pack["cl_delta_a"]

    for k in range(n_roll - 1):
        phi_k = phi_roll[k]
        p_k = p_roll[k]

        cl_eff = cl_delta_a * delta_a_roll_rad[k] + aero["Clp"] * p_k * b_w / (2 * op_point.velocity)
        l_roll = q_dyn * s_w * b_w * cl_eff
        p_dot = l_roll / i_xx

        r_k = cst.g * np.tan(phi_k) / op_point.velocity

        opti.subject_to(
            [
                p_roll[k + 1] == p_k + p_dot * dt,
                phi_roll[k + 1] == phi_k + p_k * dt,
                psi[k + 1] == psi[k] + r_k * dt,
                x[k + 1] == x[k] + op_point.velocity * np.cos(psi[k]) * dt,
                y[k + 1] == y[k] + op_point.velocity * np.sin(psi[k]) * dt,
                opti.bounded(x_min, x[k + 1], x_max),
                opti.bounded(y_min, y[k + 1], y_max),
                p_roll[k + 1] >= 0.0,
                p_roll[k + 1] <= bnd.p_roll_max,
                (delta_a_roll_deg[k + 1] - delta_a_roll_deg[k]) <= bnd.delta_a_rate_max_deg_s * dt,
                (delta_a_roll_deg[k + 1] - delta_a_roll_deg[k]) >= -bnd.delta_a_rate_max_deg_s * dt,
                delta_a_roll_deg[k + 1] <= bnd.delta_a_max_deg,
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
            (x[-1] - thermal["x_center"]) ** 2 + (y[-1] - thermal["y_center"]) ** 2 == r_target**2,
            np.cos(psi[-1]) * dx_end + np.sin(psi[-1]) * dy_end == 0.0,
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
