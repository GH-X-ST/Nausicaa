from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from state_contract import STATE_INDEX, as_state_vector


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Launch-gate constants and data containers
# 2) Gate admission and margins
# 3) CSV/report records
# =============================================================================


# =============================================================================
# 1) Launch-Gate Constants and Data Containers
# =============================================================================
@dataclass(frozen=True)
class LaunchGateBounds:
    """Axis-aligned launch gate in the public world frame.

    Angles are stored in radians internally. The `_deg` fields only appear in
    CSV/report records so unit conversion remains explicit at the boundary.
    """

    launch_gate_id: str
    x_w_m: tuple[float, float]
    y_w_m: tuple[float, float]
    z_w_m: tuple[float, float]
    psi_rad: tuple[float, float]
    phi_rad: tuple[float, float]
    theta_rad: tuple[float, float]
    speed_m_s: tuple[float, float]


FIXED_LAUNCH_GATE = LaunchGateBounds(
    launch_gate_id="fixed_gate_main_v1",
    x_w_m=(1.2, 1.4),
    y_w_m=(1.8, 2.2),
    z_w_m=(1.5, 1.9),
    psi_rad=(np.deg2rad(-30.0), np.deg2rad(30.0)),
    phi_rad=(np.deg2rad(-45.0), np.deg2rad(45.0)),
    theta_rad=(np.deg2rad(-45.0), np.deg2rad(45.0)),
    speed_m_s=(3.0, 8.0),
)

LAUNCH_TOLERANCE_SHELLS: dict[str, LaunchGateBounds] = {
    # This shell matches the project-plan "local robustness shell" while
    # retaining the hard roll/pitch admission bounds used for real launch safety.
    "launch_gate_tolerance_shell": LaunchGateBounds(
        launch_gate_id="launch_gate_tolerance_shell_v1",
        x_w_m=(1.1, 1.6),
        y_w_m=(1.4, 2.6),
        z_w_m=(1.2, 2.2),
        psi_rad=(np.deg2rad(-45.0), np.deg2rad(45.0)),
        phi_rad=FIXED_LAUNCH_GATE.phi_rad,
        theta_rad=FIXED_LAUNCH_GATE.theta_rad,
        speed_m_s=FIXED_LAUNCH_GATE.speed_m_s,
    ),
    "local_robustness_shell": LaunchGateBounds(
        launch_gate_id="local_robustness_shell_v1",
        x_w_m=(1.1, 1.6),
        y_w_m=(1.4, 2.6),
        z_w_m=(1.2, 2.2),
        psi_rad=(np.deg2rad(-45.0), np.deg2rad(45.0)),
        phi_rad=FIXED_LAUNCH_GATE.phi_rad,
        theta_rad=FIXED_LAUNCH_GATE.theta_rad,
        speed_m_s=FIXED_LAUNCH_GATE.speed_m_s,
    ),
}

ADMISSION_STATUS_VALUES = (
    "admitted_main_gate",
    "admitted_tolerance_shell",
    "rejected_position",
    "rejected_yaw",
    "rejected_roll_pitch",
    "rejected_speed",
    "invalid_state",
)


# =============================================================================
# 2) Gate Admission and Margins
# =============================================================================
def inside_fixed_launch_gate(state: np.ndarray) -> bool:
    """Return True when `state` is inside the main fixed launch gate."""

    return _inside_bounds(_state_components(state), FIXED_LAUNCH_GATE)


def inside_launch_tolerance_shell(state: np.ndarray, shell_id: str) -> bool:
    """Return True when `state` is inside the named launch tolerance shell."""

    if shell_id not in LAUNCH_TOLERANCE_SHELLS:
        raise ValueError(f"unknown launch tolerance shell_id: {shell_id!r}.")
    return _inside_bounds(_state_components(state), LAUNCH_TOLERANCE_SHELLS[shell_id])


def launch_gate_margins(state: np.ndarray) -> dict[str, float]:
    """Return signed margins to the main gate.

    Positive margins are inside the gate. Yaw margins are reported in degrees
    only at this public record boundary, with `_deg` suffixes.
    """

    components = _state_components(state)
    return _margins(components, FIXED_LAUNCH_GATE)


def launch_gate_admission_status(state: np.ndarray) -> str:
    """Return a claim-safe fixed-gate admission status."""

    try:
        components = _state_components(state)
    except ValueError:
        return "invalid_state"

    if _inside_bounds(components, FIXED_LAUNCH_GATE):
        return "admitted_main_gate"

    # Roll and pitch are hard admission bounds even for tolerance shells. They
    # are checked before the shell status so a risky launch attitude cannot be
    # hidden by a wider positional shell.
    if not _inside_axis(components["phi_rad"], FIXED_LAUNCH_GATE.phi_rad) or not _inside_axis(
        components["theta_rad"],
        FIXED_LAUNCH_GATE.theta_rad,
    ):
        return "rejected_roll_pitch"

    in_any_shell = any(
        _inside_bounds(components, bounds)
        for bounds in LAUNCH_TOLERANCE_SHELLS.values()
    )
    if in_any_shell:
        return "admitted_tolerance_shell"

    if not (
        _inside_axis(components["x_w_m"], LAUNCH_TOLERANCE_SHELLS["launch_gate_tolerance_shell"].x_w_m)
        and _inside_axis(components["y_w_m"], LAUNCH_TOLERANCE_SHELLS["launch_gate_tolerance_shell"].y_w_m)
        and _inside_axis(components["z_w_m"], LAUNCH_TOLERANCE_SHELLS["launch_gate_tolerance_shell"].z_w_m)
    ):
        return "rejected_position"
    if not _inside_axis(components["psi_rad"], LAUNCH_TOLERANCE_SHELLS["launch_gate_tolerance_shell"].psi_rad):
        return "rejected_yaw"
    if not _inside_axis(components["speed_m_s"], FIXED_LAUNCH_GATE.speed_m_s):
        return "rejected_speed"
    return "rejected_position"


# =============================================================================
# 3) CSV/Report Records
# =============================================================================
def state_to_launch_gate_record(state: np.ndarray) -> dict[str, object]:
    """Return a CSV-ready launch-gate record for one canonical state vector."""

    try:
        components = _state_components(state)
    except ValueError:
        return {
            "launch_gate_id": FIXED_LAUNCH_GATE.launch_gate_id,
            "initial_state_admission_status": "invalid_state",
            "inside_fixed_launch_gate": False,
            "inside_launch_tolerance_shell": False,
        }

    margins = launch_gate_margins(state)
    return {
        "launch_gate_id": FIXED_LAUNCH_GATE.launch_gate_id,
        "initial_state_admission_status": launch_gate_admission_status(state),
        "inside_fixed_launch_gate": inside_fixed_launch_gate(state),
        "inside_launch_tolerance_shell": inside_launch_tolerance_shell(
            state,
            "launch_gate_tolerance_shell",
        ),
        "x_w_m": components["x_w_m"],
        "y_w_m": components["y_w_m"],
        "z_w_m": components["z_w_m"],
        "phi_rad": components["phi_rad"],
        "theta_rad": components["theta_rad"],
        "psi_rad": components["psi_rad"],
        "phi_deg": float(np.rad2deg(components["phi_rad"])),
        "theta_deg": float(np.rad2deg(components["theta_rad"])),
        "psi_deg": float(np.rad2deg(components["psi_rad"])),
        "speed_m_s": components["speed_m_s"],
        **margins,
    }


def _state_components(state: np.ndarray) -> dict[str, float]:
    x = as_state_vector(state)
    speed_m_s = float(np.linalg.norm(x[[STATE_INDEX["u"], STATE_INDEX["v"], STATE_INDEX["w"]]]))
    return {
        "x_w_m": float(x[STATE_INDEX["x_w"]]),
        "y_w_m": float(x[STATE_INDEX["y_w"]]),
        "z_w_m": float(x[STATE_INDEX["z_w"]]),
        "phi_rad": float(x[STATE_INDEX["phi"]]),
        "theta_rad": float(x[STATE_INDEX["theta"]]),
        "psi_rad": float(x[STATE_INDEX["psi"]]),
        "speed_m_s": speed_m_s,
    }


def _inside_axis(value: float, bounds: tuple[float, float]) -> bool:
    return bool(float(bounds[0]) <= float(value) <= float(bounds[1]))


def _axis_margins(value: float, bounds: tuple[float, float]) -> tuple[float, float]:
    return float(value - bounds[0]), float(bounds[1] - value)


def _inside_bounds(components: dict[str, float], bounds: LaunchGateBounds) -> bool:
    margins = _margins(components, bounds)
    metric_names = (
        "x_w_min_margin_m",
        "x_w_max_margin_m",
        "y_w_min_margin_m",
        "y_w_max_margin_m",
        "z_w_min_margin_m",
        "z_w_max_margin_m",
        "phi_min_margin_deg",
        "phi_max_margin_deg",
        "theta_min_margin_deg",
        "theta_max_margin_deg",
        "psi_min_margin_deg",
        "psi_max_margin_deg",
        "speed_min_margin_m_s",
        "speed_max_margin_m_s",
    )
    return bool(min(float(margins[name]) for name in metric_names) >= 0.0)


def _margins(components: dict[str, float], bounds: LaunchGateBounds) -> dict[str, float]:
    x_min, x_max = _axis_margins(components["x_w_m"], bounds.x_w_m)
    y_min, y_max = _axis_margins(components["y_w_m"], bounds.y_w_m)
    z_min, z_max = _axis_margins(components["z_w_m"], bounds.z_w_m)
    phi_min, phi_max = _axis_margins(components["phi_rad"], bounds.phi_rad)
    theta_min, theta_max = _axis_margins(components["theta_rad"], bounds.theta_rad)
    psi_min, psi_max = _axis_margins(components["psi_rad"], bounds.psi_rad)
    speed_min, speed_max = _axis_margins(components["speed_m_s"], bounds.speed_m_s)
    position_margins = (x_min, x_max, y_min, y_max, z_min, z_max)
    attitude_margins_deg = tuple(np.rad2deg(value) for value in (phi_min, phi_max, theta_min, theta_max, psi_min, psi_max))
    return {
        "x_w_min_margin_m": x_min,
        "x_w_max_margin_m": x_max,
        "y_w_min_margin_m": y_min,
        "y_w_max_margin_m": y_max,
        "z_w_min_margin_m": z_min,
        "z_w_max_margin_m": z_max,
        "minimum_position_margin_m": float(min(position_margins)),
        "phi_min_margin_deg": float(attitude_margins_deg[0]),
        "phi_max_margin_deg": float(attitude_margins_deg[1]),
        "theta_min_margin_deg": float(attitude_margins_deg[2]),
        "theta_max_margin_deg": float(attitude_margins_deg[3]),
        "psi_min_margin_deg": float(attitude_margins_deg[4]),
        "psi_max_margin_deg": float(attitude_margins_deg[5]),
        "minimum_attitude_margin_deg": float(min(attitude_margins_deg)),
        "speed_min_margin_m_s": speed_min,
        "speed_max_margin_m_s": speed_max,
        "minimum_speed_margin_m_s": float(min(speed_min, speed_max)),
    }
