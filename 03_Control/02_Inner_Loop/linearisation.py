from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np

from flight_dynamics import AircraftModel, adapt_glider, build_symbolic_dynamics
from glider import build_nausicaa_glider
from trim_solver import TrimResult, TrimTarget, solve_straight_trim


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) State and command indexing
# 2) Linear model dataclass
# 3) Linearisation helpers
# 4) Reduced model views
# =============================================================================

# =============================================================================
# 1) State and Command Indexing
# =============================================================================
STATE_NAMES = (
    "x_w",
    "y_w",
    "z_w",
    "phi",
    "theta",
    "psi",
    "u",
    "v",
    "w",
    "p",
    "q",
    "r",
    "delta_a",
    "delta_e",
    "delta_r",
)
INPUT_NAMES = ("delta_a_cmd", "delta_e_cmd", "delta_r_cmd")
STATE_INDEX = {name: idx for idx, name in enumerate(STATE_NAMES)}
INPUT_INDEX = {name: idx for idx, name in enumerate(INPUT_NAMES)}

LONGITUDINAL_STATES = ("u", "w", "q", "theta", "delta_e")
LONGITUDINAL_INPUTS = ("delta_e_cmd",)
LATERAL_STATES = ("v", "p", "r", "phi", "delta_a", "delta_r")
LATERAL_INPUTS = ("delta_a_cmd", "delta_r_cmd")


# =============================================================================
# 2) Linear Model Dataclass
# =============================================================================
@dataclass(frozen=True)
class LinearModel:
    a: np.ndarray
    b: np.ndarray
    x_trim: np.ndarray
    u_trim: np.ndarray
    f_trim: np.ndarray
    state_names: tuple[str, ...]
    input_names: tuple[str, ...]


# =============================================================================
# 3) Linearisation Helpers
# =============================================================================
def _dense(value: object) -> np.ndarray:
    return np.asarray(value, dtype=float)


def linearise_trim(
    aircraft: AircraftModel | None = None,
    trim_result: TrimResult | None = None,
    target: TrimTarget | None = None,
) -> LinearModel:
    if aircraft is None:
        aircraft = adapt_glider(build_nausicaa_glider())
    if target is None:
        target = TrimTarget(speed_m_s=6.5)
    if target.wind_model is not None:
        raise ValueError("linearise_trim currently audits the zero-wind trim only.")
    if trim_result is None:
        trim_result = solve_straight_trim(aircraft=aircraft, target=target)

    x_trim = np.asarray(trim_result.x_trim, dtype=float).reshape(15)
    u_trim = np.array(
        [
            x_trim[STATE_INDEX["delta_a"]],
            x_trim[STATE_INDEX["delta_e"]],
            x_trim[STATE_INDEX["delta_r"]],
        ],
        dtype=float,
    )

    dynamics = build_symbolic_dynamics(
        aircraft=aircraft,
        rho=float(target.rho_kg_m3),
        actuator_tau_s=target.actuator_tau_s,
        wind_mode="none",
    )
    a_sym = ca.jacobian(dynamics.x_dot, dynamics.x)
    b_sym = ca.jacobian(dynamics.x_dot, dynamics.u_cmd)
    lin_fun = ca.Function(
        "trim_linearisation",
        [dynamics.x, dynamics.u_cmd],
        [a_sym, b_sym, dynamics.x_dot],
    )
    a_num, b_num, f_num = lin_fun(ca.DM(x_trim), ca.DM(u_trim))
    return LinearModel(
        a=_dense(a_num),
        b=_dense(b_num),
        x_trim=x_trim,
        u_trim=u_trim,
        f_trim=_dense(f_num).reshape(15),
        state_names=STATE_NAMES,
        input_names=INPUT_NAMES,
    )


# =============================================================================
# 4) Reduced Model Views
# =============================================================================
def reduced_model(
    model: LinearModel,
    state_names: tuple[str, ...],
    input_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    state_idx = [STATE_INDEX[name] for name in state_names]
    input_idx = [INPUT_INDEX[name] for name in input_names]
    return model.a[np.ix_(state_idx, state_idx)], model.b[np.ix_(state_idx, input_idx)]


def key_derivatives(model: LinearModel) -> dict[str, float]:
    a = model.a
    b = model.b
    s = STATE_INDEX
    i = INPUT_INDEX
    return {
        "x_u": float(a[s["u"], s["u"]]),
        "x_w": float(a[s["u"], s["w"]]),
        "z_u": float(a[s["w"], s["u"]]),
        "z_w": float(a[s["w"], s["w"]]),
        "m_u": float(a[s["q"], s["u"]]),
        "m_w": float(a[s["q"], s["w"]]),
        "m_q": float(a[s["q"], s["q"]]),
        "l_p": float(a[s["p"], s["p"]]),
        "l_r": float(a[s["p"], s["r"]]),
        "n_p": float(a[s["r"], s["p"]]),
        "n_r": float(a[s["r"], s["r"]]),
        "x_delta_e": float(a[s["u"], s["delta_e"]]),
        "z_delta_e": float(a[s["w"], s["delta_e"]]),
        "m_delta_e": float(a[s["q"], s["delta_e"]]),
        "y_delta_a": float(a[s["v"], s["delta_a"]]),
        "l_delta_a": float(a[s["p"], s["delta_a"]]),
        "n_delta_a": float(a[s["r"], s["delta_a"]]),
        "y_delta_r": float(a[s["v"], s["delta_r"]]),
        "l_delta_r": float(a[s["p"], s["delta_r"]]),
        "n_delta_r": float(a[s["r"], s["delta_r"]]),
        "delta_a_cmd": float(b[s["delta_a"], i["delta_a_cmd"]]),
        "delta_e_cmd": float(b[s["delta_e"], i["delta_e_cmd"]]),
        "delta_r_cmd": float(b[s["delta_r"], i["delta_r_cmd"]]),
    }
