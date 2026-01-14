from __future__ import annotations

import aerosandbox as asb
import aerosandbox.numpy as np

from config import Config


def aileron_effectiveness_fd(
    airplane: asb.Airplane,
    op_point: asb.OperatingPoint,
    xyz_ref,
    delta_a_deg,
    delta_r_deg,
    delta_e_deg,
    fd_step_deg: float = 1.0,
) -> object:
    """Finite-difference aileron effectiveness Cl_deltaA."""
    airplane_plus = airplane.with_control_deflections(
        {"aileron": delta_a_deg - fd_step_deg, "rudder": delta_r_deg, "elevator": delta_e_deg}
    )
    aero_plus = asb.AeroBuildup(airplane_plus, op_point, xyz_ref=xyz_ref).run()
    cl_plus = aero_plus["Cl"]

    airplane_minus = airplane.with_control_deflections(
        {"aileron": delta_a_deg + fd_step_deg, "rudder": delta_r_deg, "elevator": delta_e_deg}
    )
    aero_minus = asb.AeroBuildup(airplane_minus, op_point, xyz_ref=xyz_ref).run()
    cl_minus = aero_minus["Cl"]

    cl_delta_per_deg = (cl_plus - cl_minus) / (2 * fd_step_deg)
    # Keep your original conversion
    return cl_delta_per_deg * (180 / np.pi)


def build_aero(
    cfg: Config,
    airplane: asb.Airplane,
    op_point: asb.OperatingPoint,
    xyz_ref,
    geom: dict,
    controls: dict,
) -> dict:
    """Compute aerodynamic model, derivatives, and derived performance quantities."""
    ab = asb.AeroBuildup(airplane=airplane, op_point=op_point, xyz_ref=xyz_ref)

    aero = ab.run_with_stability_derivatives(alpha=True, beta=True, p=True, q=True, r=True)

    cl_delta_a = aileron_effectiveness_fd(
        airplane=airplane,
        op_point=op_point,
        xyz_ref=xyz_ref,
        delta_a_deg=controls["delta_a_deg"],
        delta_r_deg=controls["delta_r_deg"],
        delta_e_deg=controls["delta_e_deg"],
        fd_step_deg=1.0,
    )

    l_over_d = aero["L"] / np.maximum(aero["D"], 1e-6)
    power_loss = aero["D"] * op_point.velocity

    return {
        "ab": ab,
        "aero": aero,
        "cl_delta_a": cl_delta_a,
        "l_over_d": l_over_d,
        "power_loss": power_loss,
    }