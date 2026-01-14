from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as onp
import pandas as pd
import aerosandbox.tools.units as u

from config import Config
from post.postprocess import SolvedModel


def _to_scalar(x: Any) -> Any:
    """Safely convert scalars/0-d arrays to float for export."""
    try:
        arr = onp.array(x)
        if arr.shape == ():
            return float(arr)
        return float(arr.flatten()[0])
    except Exception:
        try:
            return float(x)
        except Exception:
            return x


def save_results(cfg: Config, solved: SolvedModel) -> dict[str, Path]:
    """Export key results and roll-in time series to CSV/XLSX."""
    paths = cfg.paths
    paths.results_dir.mkdir(parents=True, exist_ok=True)

    op_point = solved.mission["op_point"]
    wing = solved.wing
    htail = solved.htail
    vtail = solved.vtail

    design_results: dict[str, Any] = {
        # operating point
        "V (m/s)": _to_scalar(op_point.velocity),
        "α (deg)": _to_scalar(op_point.alpha),
        "ϕ (deg)": _to_scalar(solved.mission["phi_deg"]),
        "R_target (m)": _to_scalar(solved.mission["r_target"]),
        "n": _to_scalar(solved.mission["n_load"]),
        # mass/performance
        "m_TOGW (kg)": _to_scalar(solved.mass_props_togw.mass),
        "L/D_turn": _to_scalar(solved.performance["l_over_d_turn"]),
        "sink_rate (m/s)": _to_scalar(solved.performance["sink_rate"]),
        "w (m/s)": _to_scalar(solved.performance["w"]),
        "climb_rate (m/s)": _to_scalar(solved.performance["climb_rate"]),
        "K_n": _to_scalar(solved.objective["k_n"]),
        "V_HT": _to_scalar(solved.objective["v_ht"]),
        "V_VT": _to_scalar(solved.objective["v_vt"]),
        "Re_W": _to_scalar(op_point.reynolds(wing.mean_aerodynamic_chord())),
        # geometry
        "b_W (m)": _to_scalar(wing.span()),
        "S_W (m^2)": _to_scalar(wing.area()),
        "S_HT (m^2)": _to_scalar(htail.area()),
        "S_VT (m^2)": _to_scalar(vtail.area()),
        # controls
        "δ_A (deg)": _to_scalar(solved.mission["controls"]["delta_a_deg"]),
        "δ_R (deg)": _to_scalar(solved.mission["controls"]["delta_r_deg"]),
        "δ_E (deg)": _to_scalar(solved.mission["controls"]["delta_e_deg"]),
        # roll-in
        "t_roll (s)": _to_scalar(solved.roll["t_roll"]),
        "ψ0 (deg)": _to_scalar(solved.roll["psi0_deg"]),
        "Cl,δA (rad^-1)": _to_scalar(solved.performance["cl_delta_a"]),
        # objective
        "objective_total": _to_scalar(solved.objective["objective"]),
        "objective_climb": _to_scalar(solved.objective["obj_climb"]),
        "objective_span": _to_scalar(solved.objective["obj_span"]),
        "objective_roll_in_time": _to_scalar(solved.objective["obj_rolltime"]),
        "objective_control": _to_scalar(solved.objective["obj_control"]),
        "penalty": _to_scalar(solved.objective["penalty"]),
        # CG and inertia
        "x_CG (m)": _to_scalar(solved.mass_props_togw.xyz_cg[0]),
        "y_CG (m)": _to_scalar(solved.mass_props_togw.xyz_cg[1]),
        "z_CG (m)": _to_scalar(solved.mass_props_togw.xyz_cg[2]),
        "I_xx (kg m^2)": _to_scalar(solved.mass_props_togw.inertia_tensor[0, 0]),
        "I_yy (kg m^2)": _to_scalar(solved.mass_props_togw.inertia_tensor[1, 1]),
        "I_zz (kg m^2)": _to_scalar(solved.mass_props_togw.inertia_tensor[2, 2]),
        "I_xy (kg m^2)": _to_scalar(solved.mass_props_togw.inertia_tensor[0, 1]),
        "I_xz (kg m^2)": _to_scalar(solved.mass_props_togw.inertia_tensor[0, 2]),
        "I_yz (kg m^2)": _to_scalar(solved.mass_props_togw.inertia_tensor[1, 2]),
        # convenience
        "m_TOGW (g)": _to_scalar(solved.mass_props_togw.mass * 1e3),
        "m_TOGW (lbm)": _to_scalar(solved.mass_props_togw.mass / u.lbm),
    }

    # component masses
    for name, mp in solved.mass_props.items():
        design_results[f"mass_{name}_kg"] = _to_scalar(mp.mass)

    # aero outputs (flatten dict)
    aero_results = {f"aero_{k}": _to_scalar(v) for k, v in solved.aero.items() if not isinstance(v, list)}

    all_results = {**design_results, **aero_results}
    df = pd.DataFrame(list(all_results.items()), columns=["Variable", "Value"])

    csv_path = paths.results_dir / "nausicaa_results.csv"
    xlsx_path = paths.results_dir / "nausicaa_results.xlsx"
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)

    # Roll-in time series
    t_roll = float(solved.roll["t_roll"])
    x_roll = onp.asarray(solved.roll["x"], dtype=float)
    n = len(x_roll)
    t_vec = onp.linspace(0.0, t_roll, n)

    roll_df = pd.DataFrame(
        {
            "t (s)": t_vec,
            "x (m)": onp.asarray(solved.roll["x"], dtype=float),
            "y (m)": onp.asarray(solved.roll["y"], dtype=float),
            "ψ (rad)": onp.asarray(solved.roll["psi"], dtype=float),
            "ψ (deg)": onp.degrees(onp.asarray(solved.roll["psi"], dtype=float)),
            "p (rad/s)": onp.asarray(solved.roll["p_roll"], dtype=float),
            "ϕ (deg)": onp.degrees(onp.asarray(solved.roll["phi_roll"], dtype=float)),
            "δ_A_roll (deg)": onp.asarray(solved.roll["delta_a_roll_deg"], dtype=float),
        }
    )

    roll_csv = paths.results_dir / "nausicaa_rollin_timeseries.csv"
    roll_xlsx = paths.results_dir / "nausicaa_rollin_timeseries.xlsx"
    roll_df.to_csv(roll_csv, index=False)
    roll_df.to_excel(roll_xlsx, index=False)

    return {
        "results_csv": csv_path,
        "results_xlsx": xlsx_path,
        "roll_csv": roll_csv,
        "roll_xlsx": roll_xlsx,
    }