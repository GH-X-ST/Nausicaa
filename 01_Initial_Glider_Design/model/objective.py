from __future__ import annotations

import aerosandbox as asb
import aerosandbox.numpy as np

from config import Config


def set_objective_and_constraints(
    opti: asb.Opti,
    cfg: Config,
    mission: dict,
    geom: dict,
    mass: dict,
    thermal: dict,
    aero_pack: dict,
    roll: dict,
) -> dict:
    """Add constraints and objective; return objective components for reporting."""
    cst = cfg.constants

    op_point = mission["op_point"]
    phi_rad = mission["phi_rad"]
    n_load = mission["n_load"]

    aero = aero_pack["aero"]
    l_over_d = aero_pack["l_over_d"]

    sink_rate = (aero_pack["power_loss"]) / cst.g / mass["mass_props_togw"].mass
    climb_rate = thermal["w"] - sink_rate

    k_n = (aero["x_np"] - mass["mass_props_togw"].x_cg) / geom["wing"].mean_aerodynamic_chord()

    # “softmax” penalty on climb-rate shortfall
    k_soft = cfg.k_soft
    shortfall = (1 / k_soft) * np.log(1 + np.exp(-k_soft * climb_rate))
    obj_climb = shortfall**2

    obj_span = geom["b_w"] + geom["b_ht"] + geom["b_vt"]
    obj_rolltime = roll["t_roll"]

    controls = mission["controls"]
    obj_control = cfg.w_control_penalty * (
        controls["delta_e_deg"] ** 2 + controls["delta_a_deg"] ** 2 + controls["delta_r_deg"] ** 2
    )

    objective = obj_climb + obj_span + obj_rolltime + obj_control

    # tiny penalty on ballast CG position
    penalty = (mass["mass_props"]["ballast"].x_cg / 1e3) ** 2
    opti.minimize(objective + penalty)

    # Constraints
    opti.subject_to(
        [
            aero["L"] >= n_load * mass["mass_props_togw"].mass * cst.g,
            aero["Cl"] == 0.0,
            aero["Cm"] == 0.0,
            aero["Cn"] == 0.0,
            aero["Clb"] <= -0.08,
            aero["Cnb"] >= 0.03,
            opti.bounded(0.04, k_n, 0.10),
            opti.bounded(0.40, geom["v_ht"], 0.70),
            opti.bounded(0.02, geom["v_vt"], 0.04),
        ]
    )

    # Additional coupling constraints
    opti.subject_to(
        [
            mission["l_over_d_turn"] == l_over_d,
            mission["design_mass_togw"] == mass["mass_props_togw"].mass,
        ]
    )

    return {
        "objective": objective,
        "penalty": penalty,
        "obj_climb": obj_climb,
        "obj_span": obj_span,
        "obj_rolltime": obj_rolltime,
        "obj_control": obj_control,
        "sink_rate": sink_rate,
        "climb_rate": climb_rate,
        "k_n": k_n,
    }
