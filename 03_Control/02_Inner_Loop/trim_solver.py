from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import casadi as ca
import numpy as np

from flight_dynamics import (
    AircraftModel,
    build_symbolic_dynamics,
    evaluate_state,
)


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Trim dataclasses
# 2) Baseline trim seed
# 3) Wind packing
# 4) Straight-flight trim solve
# =============================================================================

# =============================================================================
# 1) Trim Dataclasses
# =============================================================================
@dataclass(frozen=True)
class TrimTarget:
    speed_m_s: float
    altitude_m: float = 0.0
    rho_kg_m3: float = 1.225
    wind_model: np.ndarray | list[float] | tuple[float, ...] | None = None
    wind_mode: str = "panel"
    actuator_tau_s: tuple[float, float, float] = (0.06, 0.06, 0.06)


@dataclass(frozen=True)
class TrimResult:
    x_trim: np.ndarray
    u_cmd_trim: np.ndarray
    converged: bool
    alpha_rad: float
    theta_rad: float
    gamma_rad: float
    sink_rate_m_s: float
    x_force_b: float
    y_force_b: float
    z_force_b: float
    l_moment_b: float
    m_moment_b: float
    n_moment_b: float
    solver_stats: dict[str, object]


# =============================================================================
# 2) Baseline Trim Seed
# =============================================================================
def _baseline_results_path() -> Path:
    # Baseline seed values come from the current design-result CSV
    return (
        Path(__file__).resolve().parents[2]
        / "02_Glider_Design"
        / "C_results"
        / "nausicaa_results.csv"
    )


def _load_baseline_seed() -> tuple[float, float, float]:
    metric_map: dict[str, float] = {}
    with _baseline_results_path().open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                metric_map[row["Metric"]] = float(row["Value"])
            except ValueError:
                continue
    alpha0 = np.deg2rad(metric_map["alpha_trim_deg"])
    delta_e0 = np.deg2rad(metric_map["delta_e_trim_deg"])
    speed0 = metric_map["v_nom_mps"]
    sink0 = metric_map["sink_rate_mps"]
    # Flight-path angle is positive upward in the public z-up frame
    gamma0 = -np.arcsin(np.clip(sink0 / speed0, -1.0, 1.0))
    theta0 = alpha0 + gamma0
    return alpha0, theta0, delta_e0


# =============================================================================
# 3) Wind Packing
# =============================================================================
def _pack_trim_wind(
    target: TrimTarget,
    aircraft: AircraftModel,
) -> tuple[str, np.ndarray | None]:
    if target.wind_model is None:
        return "none", None
    if callable(target.wind_model):
        raise TypeError("Trim solver supports only None or constant wind vectors.")
    wind_w = np.asarray(target.wind_model, dtype=float).reshape(3)
    if target.wind_mode == "cg":
        return "cg", wind_w
    if target.wind_mode != "panel":
        raise ValueError("wind_mode must be 'panel' or 'cg'.")
    # Panel trim packs one CG vector plus one vector per aerodynamic strip
    return "panel", np.tile(wind_w, aircraft.strip_count + 1)


# =============================================================================
# 4) Straight-Flight Trim Solve
# =============================================================================
def solve_straight_trim(
    aircraft: AircraftModel,
    target: TrimTarget,
) -> TrimResult:
    symbolic_wind_mode, wind_param = _pack_trim_wind(target, aircraft)
    dynamics = build_symbolic_dynamics(
        aircraft=aircraft,
        rho=float(target.rho_kg_m3),
        actuator_tau_s=target.actuator_tau_s,
        wind_mode=symbolic_wind_mode,
    )
    alpha = ca.SX.sym("alpha")
    theta = ca.SX.sym("theta")
    delta_e = ca.SX.sym("delta_e")
    z = ca.vertcat(alpha, theta, delta_e)
    speed = float(target.speed_m_s)
    # Straight trim fixes lateral states and solves longitudinal force/moment balance
    x_trim = ca.vertcat(
        0.0,
        0.0,
        float(target.altitude_m),
        0.0,
        theta,
        0.0,
        speed * ca.cos(alpha),
        0.0,
        speed * ca.sin(alpha),
        0.0,
        0.0,
        0.0,
        0.0,
        delta_e,
        0.0,
    )
    u_cmd_trim = ca.vertcat(0.0, delta_e, 0.0)
    if dynamics.wind_param is None:
        x_dot = dynamics.function(x_trim, u_cmd_trim)
    else:
        x_dot = dynamics.function(x_trim, u_cmd_trim, ca.DM(wind_param))
    # Residual enforces body x/z acceleration and pitch moment equilibrium
    residual = ca.vertcat(x_dot[6], x_dot[8], x_dot[10])
    objective = delta_e**2 + 0.1 * alpha**2
    solver = ca.nlpsol(
        "trim_solver",
        "ipopt",
        {"x": z, "f": objective, "g": residual},
        {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.max_iter": 300,
            "ipopt.sb": "yes",
        },
    )
    alpha0, theta0, delta_e0 = _load_baseline_seed()
    solution = solver(
        x0=ca.DM([alpha0, theta0, delta_e0]),
        lbx=ca.DM(
            [
                np.deg2rad(-6.0),
                np.deg2rad(-15.0),
                np.deg2rad(-30.0),
            ]
        ),
        ubx=ca.DM(
            [
                np.deg2rad(12.0),
                np.deg2rad(15.0),
                np.deg2rad(30.0),
            ]
        ),
        lbg=ca.DM.zeros(3, 1),
        ubg=ca.DM.zeros(3, 1),
    )
    alpha_sol, theta_sol, delta_e_sol = np.asarray(solution["x"]).reshape(3)
    x_trim_num = np.array(
        [
            0.0,
            0.0,
            float(target.altitude_m),
            0.0,
            theta_sol,
            0.0,
            speed * np.cos(alpha_sol),
            0.0,
            speed * np.sin(alpha_sol),
            0.0,
            0.0,
            0.0,
            0.0,
            delta_e_sol,
            0.0,
        ],
        dtype=float,
    )
    u_cmd_num = np.array([0.0, delta_e_sol, 0.0], dtype=float)
    loads = evaluate_state(
        x=x_trim_num,
        u_cmd=u_cmd_num,
        aircraft=aircraft,
        wind_model=target.wind_model,
        rho=float(target.rho_kg_m3),
        actuator_tau_s=target.actuator_tau_s,
        wind_mode=target.wind_mode,
    )
    stats = solver.stats()
    solver_stats = {
        "return_status": stats.get("return_status"),
        "success": bool(stats.get("success", False)),
        "iter_count": int(stats.get("iter_count", 0)),
    }
    return TrimResult(
        x_trim=x_trim_num,
        u_cmd_trim=u_cmd_num,
        converged=bool(stats.get("success", False)),
        alpha_rad=float(loads["alpha_rad"]),
        theta_rad=float(theta_sol),
        gamma_rad=float(loads["gamma_rad"]),
        sink_rate_m_s=float(loads["sink_rate_m_s"]),
        x_force_b=float(loads["x_force_b"]),
        y_force_b=float(loads["y_force_b"]),
        z_force_b=float(loads["z_force_b"]),
        l_moment_b=float(loads["l_moment_b"]),
        m_moment_b=float(loads["m_moment_b"]),
        n_moment_b=float(loads["n_moment_b"]),
        solver_stats=solver_stats,
    )
