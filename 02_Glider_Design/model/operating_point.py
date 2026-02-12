from __future__ import annotations

import aerosandbox as asb
import aerosandbox.numpy as np

from config import Config


def build_operating_point(opti: asb.Opti, cfg: Config) -> dict:
    """Create operating point variables and derived turn quantities."""
    bnd = cfg.bounds
    cst = cfg.constants
    msn = cfg.mission

    phi_deg = msn.bank_angle_deg
    phi_rad = np.radians(phi_deg)

    op_point = asb.OperatingPoint(
        velocity=opti.variable(
            init_guess=5.0,
            lower_bound=bnd.v_min,
            upper_bound=bnd.v_max,
            log_transform=True,
        ),
        alpha=opti.variable(
            init_guess=5.0,
            lower_bound=bnd.alpha_min_deg,
            upper_bound=bnd.alpha_max_deg,
        ),
        beta=0.0,
        p=0.0,
    )

    # Coordinated-turn target radius implied by fixed bank angle
    r_target = op_point.velocity**2 / (cst.g * np.tan(phi_rad))
    n_load = 1.0 / np.cos(phi_rad)

    # Control variables
    delta_a_deg = opti.variable(
        init_guess=0.0, lower_bound=bnd.delta_a_min_deg, upper_bound=bnd.delta_a_max_deg
    )
    delta_r_deg = opti.variable(
        init_guess=0.0, lower_bound=bnd.delta_r_min_deg, upper_bound=bnd.delta_r_max_deg
    )
    delta_e_deg = opti.variable(
        init_guess=0.0, lower_bound=bnd.delta_e_min_deg, upper_bound=bnd.delta_e_max_deg
    )

    l_over_d_turn = opti.variable(init_guess=15.0, lower_bound=0.1, log_transform=True)

    # Design mass
    design_mass_togw = opti.variable(init_guess=0.1, lower_bound=bnd.togw_min)
    design_mass_togw = np.maximum(design_mass_togw, bnd.togw_min)  # numeric clamp

    return {
        "phi_deg": phi_deg,
        "phi_rad": phi_rad,
        "z_th": msn.z_th,
        "op_point": op_point,
        "r_target": r_target,
        "n_load": n_load,
        "controls": {
            "delta_a_deg": delta_a_deg,
            "delta_r_deg": delta_r_deg,
            "delta_e_deg": delta_e_deg,
        },
        "l_over_d_turn": l_over_d_turn,
        "design_mass_togw": design_mass_togw,
    }