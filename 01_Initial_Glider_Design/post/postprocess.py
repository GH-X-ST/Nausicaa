from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as onp
import aerosandbox as asb

from config import Config


@dataclass
class SolvedModel:
    airplane: asb.Airplane
    wing: asb.Wing
    htail: asb.Wing
    vtail: asb.Wing
    fuselage: asb.Fuselage

    mass_props: dict
    mass_props_togw: asb.MassProperties

    aero: dict
    performance: dict
    mission: dict
    geom: dict
    thermal: dict
    roll: dict
    objective: dict


def extract_solution(
    sol: asb.OptiSol,
    mission: dict,
    geom: dict,
    mass: dict,
    thermal: dict,
    aero_pack: dict,
    roll: dict,
    objective_pack: dict,
) -> SolvedModel:
    """Convert symbolic objects into solved numeric objects for post/plots/export."""
    # Airplane numeric
    airplane_num = sol(geom["airplane"])
    wing = copy.deepcopy(airplane_num.wings[0])
    htail = copy.deepcopy(airplane_num.wings[1])
    vtail = copy.deepcopy(airplane_num.wings[2])
    fuselage = copy.deepcopy(airplane_num.fuselages[0])

    return SolvedModel(
        airplane=airplane_num,
        wing=wing,
        htail=htail,
        vtail=vtail,
        fuselage=fuselage,
        mass_props=sol(mass["mass_props"]),
        mass_props_togw=sol(mass["mass_props_togw"]),
        aero=sol(aero_pack["aero"]),
        performance={
            "cl_delta_a": sol(aero_pack["cl_delta_a"]),
            "sink_rate": sol(objective_pack["sink_rate"]),
            "w": sol(thermal["w"]),
            "climb_rate": sol(objective_pack["climb_rate"]),
            "l_over_d_turn": sol(mission["l_over_d_turn"]),
        },
        mission={
            "op_point": sol(mission["op_point"]),
            "phi_deg": sol(mission["phi_deg"]),
            "phi_rad": sol(mission["phi_rad"]),
            "r_target": sol(mission["r_target"]),
            "n_load": sol(mission["n_load"]),
            "controls": {k: sol(v) for k, v in mission["controls"].items()},
            "design_mass_togw": sol(mission["design_mass_togw"]),
        },
        geom={k: sol(v) for k, v in geom.items() if k not in {"airplane", "wing", "htail", "vtail", "fuselage"}},
        thermal={k: sol(v) for k, v in thermal.items()},
        roll={
            "t_roll": sol(roll["t_roll"]),
            "psi0_deg": sol(roll["psi0_deg"]),
            "p_roll": sol(roll["p_roll"]),
            "phi_roll": sol(roll["phi_roll"]),
            "psi": sol(roll["psi"]),
            "x": sol(roll["x"]),
            "y": sol(roll["y"]),
            "delta_a_roll_deg": sol(roll["delta_a_roll_deg"]),
        },
        objective={
            "objective": sol(objective_pack["objective"]),
            "penalty": sol(objective_pack["penalty"]),
            "obj_climb": sol(objective_pack["obj_climb"]),
            "obj_span": sol(objective_pack["obj_span"]),
            "obj_rolltime": sol(objective_pack["obj_rolltime"]),
            "obj_control": sol(objective_pack["obj_control"]),
            "k_n": sol(objective_pack["k_n"]),
            "v_ht": sol(geom["v_ht"]),
            "v_vt": sol(geom["v_vt"]),
        },
    )