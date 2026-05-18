from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CONTROL_DIR = Path(__file__).resolve().parents[1]
INNER_LOOP_DIR = CONTROL_DIR / "02_Inner_Loop"
SCENARIOS_DIR = CONTROL_DIR / "04_Scenarios"
for path in (INNER_LOOP_DIR, SCENARIOS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from arena_contract import TRUE_SAFE_BOUNDS, inside_bounds, position_margin_m
from command_contract import (
    clip_normalised_command,
    normalised_command_to_surface_rad,
)
from flight_dynamics import adapt_glider
from glide_primitive import build_glide_primitive_spec
from glider import build_nausicaa_glider
from primitive_interface import evaluate_entry_set
from recovery_primitive import build_recovery_primitive_spec
from rollout import rk4_step
from state_contract import STATE_INDEX, STATE_SIZE, as_state_vector
from trim_solver import TrimTarget, solve_straight_trim


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Data contracts and thresholds
# 2) Family profiles
# 3) Replay and metrics
# 4) Ranking and comparison
# =============================================================================


# =============================================================================
# 1) Data Contracts and Thresholds
# =============================================================================
AGILE_TURN_CAMPAIGN = "08_agile_turn_family_comparison"
FAMILY_NAMES = (
    "canyon_steep_bank",
    "wingover_lite",
    "bank_yaw_energy_retaining",
)
RETIRED_FAMILY_LABELS = (
    "dive_perch_redirect_30",
    "reduced_perch_redirect_30",
    "early_unload_recovery_30",
    "speed_collapse_pitch_redirect",
)
G_M_S2 = 9.81

STRICT_ALPHA_MAX_DEG = 45.0
STRICT_BETA_MAX_DEG = 25.0
STRICT_RATE_MAX_RAD_S = 4.5
STRICT_SATURATION_MAX = 0.25
STRICT_TERMINAL_SPEED_M_S = 5.0
STRICT_MIN_SPEED_M_S = 4.0

USEFUL_ALPHA_MAX_DEG = 60.0
USEFUL_BETA_MAX_DEG = 30.0
USEFUL_RATE_MAX_RAD_S = 5.0
USEFUL_SATURATION_MAX = 0.40
USEFUL_TERMINAL_SPEED_M_S = 4.5
USEFUL_MIN_SPEED_M_S = 3.8

# Values are nearest fixed-step-compatible horizons to the requested
# 0.50/0.65/0.80, 0.75/0.95/1.15, and 1.15/1.65 grids.
DEFAULT_HORIZON_GRID_S = {
    15.0: (0.50, 0.66, 0.80),
    30.0: (0.76, 0.96, 1.16),
}
PLANNED_ESCALATION_HORIZON_GRID_S = {
    45.0: (1.00, 1.20, 1.40),
    60.0: (1.16, 1.40, 1.66),
}


@dataclass(frozen=True)
class AgileTurnFamilyConfig:
    dt_s: float = 0.02
    t_final_s: float = 0.80
    target_heading_deg: float = 30.0
    direction_sign: int = 1
    speed_m_s: float = 6.5
    altitude_m: float = 1.8
    wind_mode: str = "none"
    latency_case: str = "none"
    actuator_tau_s: tuple[float, float, float] = (0.06, 0.06, 0.06)
    seed: int = 1
    phase_search_scales: tuple[float, ...] = (0.70, 0.85, 1.00, 1.15)


@dataclass(frozen=True)
class AgileTurnCandidateResult:
    family_name: str
    target_heading_deg: float
    direction_sign: int
    success: bool
    failure_label: str
    time_s: np.ndarray
    x_ref: np.ndarray
    u_norm_requested: np.ndarray
    u_norm_applied: np.ndarray
    delta_cmd_rad: np.ndarray
    phase: tuple[str, ...]
    metrics: dict[str, object]
    notes: str


@dataclass(frozen=True)
class AgileTurnFamilyComparisonResult:
    target_heading_deg: float
    family_results: tuple[AgileTurnCandidateResult, ...]
    selected_family: str
    selected_candidate: AgileTurnCandidateResult
    ranking_rows: tuple[dict[str, object], ...]
    notes: str


def family_inventory() -> tuple[str, ...]:
    """Return the active reusable agile-turn families."""

    return FAMILY_NAMES


def target_horizon_s(target_heading_deg: float) -> float:
    """Return the nominal target-specific horizon for family comparison."""

    target = float(target_heading_deg)
    if np.isclose(target, 15.0):
        return 0.66
    if np.isclose(target, 30.0):
        return 0.96
    if np.isclose(target, 45.0):
        return 1.20
    if np.isclose(target, 60.0):
        return 1.40
    raise ValueError("agile turn family comparison supports 15/30 deg by default and planned 45/60 deg escalation only.")


def horizon_grid_s(target_heading_deg: float) -> tuple[float, ...]:
    """Return the horizon grid for a target without changing acceptance gates."""

    target = float(target_heading_deg)
    for key, values in DEFAULT_HORIZON_GRID_S.items():
        if np.isclose(target, key):
            return values
    for key, values in PLANNED_ESCALATION_HORIZON_GRID_S.items():
        if np.isclose(target, key):
            return values
    raise ValueError("unsupported agile turn target horizon grid.")


def acceptance_thresholds() -> dict[str, float]:
    """Return scalar acceptance thresholds for manifest and report evidence."""

    return {
        "strict_terminal_speed_m_s": STRICT_TERMINAL_SPEED_M_S,
        "strict_min_speed_m_s": STRICT_MIN_SPEED_M_S,
        "strict_alpha_max_deg": STRICT_ALPHA_MAX_DEG,
        "strict_beta_max_deg": STRICT_BETA_MAX_DEG,
        "strict_rate_max_rad_s": STRICT_RATE_MAX_RAD_S,
        "strict_saturation_max": STRICT_SATURATION_MAX,
        "useful_terminal_speed_m_s": USEFUL_TERMINAL_SPEED_M_S,
        "useful_min_speed_m_s": USEFUL_MIN_SPEED_M_S,
        "useful_alpha_max_deg": USEFUL_ALPHA_MAX_DEG,
        "useful_beta_max_deg": USEFUL_BETA_MAX_DEG,
        "useful_rate_max_rad_s": USEFUL_RATE_MAX_RAD_S,
        "useful_saturation_max": USEFUL_SATURATION_MAX,
    }


# =============================================================================
# 2) Family Profiles
# =============================================================================
def _validate_config(config: AgileTurnFamilyConfig) -> None:
    if not np.isfinite(float(config.dt_s)) or float(config.dt_s) <= 0.0:
        raise ValueError("dt_s must be finite and positive.")
    if not np.isfinite(float(config.t_final_s)) or float(config.t_final_s) <= 0.0:
        raise ValueError("t_final_s must be finite and positive.")
    step_count = round(float(config.t_final_s) / float(config.dt_s))
    if not np.isclose(float(config.t_final_s), step_count * float(config.dt_s), atol=1e-9):
        raise ValueError("t_final_s must be an integer multiple of dt_s.")
    if int(config.direction_sign) not in (-1, 1):
        raise ValueError("direction_sign must be -1 or +1.")
    if config.wind_mode != "none":
        raise ValueError("this comparison pass is W0/no-wind only.")
    if config.latency_case != "none":
        raise ValueError("this comparison pass uses the nominal no-latency case only.")
    if tuple(config.phase_search_scales) == ():
        raise ValueError("phase_search_scales must not be empty.")


def _time_grid(config: AgileTurnFamilyConfig) -> np.ndarray:
    step_count = int(round(float(config.t_final_s) / float(config.dt_s)))
    return np.arange(step_count + 1, dtype=float) * float(config.dt_s)


def build_family_initial_state(
    config: AgileTurnFamilyConfig,
    aircraft: object | None = None,
) -> np.ndarray:
    """Return safe trim-like W0 initial state in the canonical 15-state order."""

    _validate_config(config)
    aircraft_model = adapt_glider(build_nausicaa_glider()) if aircraft is None else aircraft
    trim = solve_straight_trim(
        aircraft_model,
        TrimTarget(
            speed_m_s=float(config.speed_m_s),
            altitude_m=float(config.altitude_m),
            wind_model=None,
            wind_mode="none",
            actuator_tau_s=config.actuator_tau_s,
        ),
    )
    if not trim.converged:
        raise ValueError("cannot build agile-turn initial state from nonconverged trim.")
    state = as_state_vector(trim.x_trim)
    state[STATE_INDEX["x_w"]] = 1.30
    state[STATE_INDEX["y_w"]] = 2.20
    state[STATE_INDEX["z_w"]] = float(config.altitude_m)
    return state


def _phase_edges(family_name: str, t_final_s: float, timing_scale: float = 1.0) -> tuple[float, ...]:
    scale = float(np.clip(timing_scale, 0.85, 1.15))
    base: dict[str, tuple[float, ...]] = {
        "canyon_steep_bank": (0.14, 0.58, 0.76, 0.90),
        "wingover_lite": (0.12, 0.28, 0.52, 0.74, 0.88),
        "bank_yaw_energy_retaining": (0.12, 0.54, 0.70, 0.86),
    }
    if family_name not in base:
        raise ValueError(f"unknown agile turn family: {family_name}.")
    fractions = np.clip(np.asarray(base[family_name], dtype=float) * scale, 0.05, 0.95)
    fractions = np.maximum.accumulate(fractions)
    for index in range(1, fractions.size):
        if fractions[index] <= fractions[index - 1]:
            fractions[index] = min(0.97, fractions[index - 1] + 0.03)
    return tuple(float(value * t_final_s) for value in fractions)


def _phase_names(family_name: str) -> tuple[str, ...]:
    names = {
        "canyon_steep_bank": (
            "entry",
            "roll_in",
            "turn_hold",
            "heading_capture",
            "unload_exit",
        ),
        "wingover_lite": (
            "entry",
            "shallow_pre_dive",
            "climb_roll",
            "crest_redirect",
            "unload_descend",
            "exit_glide",
        ),
        "bank_yaw_energy_retaining": (
            "entry",
            "bank_yaw_redirect",
            "heading_capture",
            "early_unload",
            "exit_glide",
        ),
    }
    if family_name not in names:
        raise ValueError(f"unknown agile turn family: {family_name}.")
    return names[family_name]


def phase_labels_for_family(
    family_name: str,
    time_s: np.ndarray,
    t_final_s: float,
) -> tuple[str, ...]:
    """Return phase labels for the selected family."""

    time = np.asarray(time_s, dtype=float).reshape(-1)
    names = _phase_names(family_name)
    edges = _phase_edges(family_name, float(t_final_s))
    labels: list[str] = []
    for value in time:
        index = int(np.searchsorted(edges, float(value), side="right"))
        index = min(index, len(names) - 1)
        labels.append(names[index])
    return tuple(labels)


def _phase_labels_for_family_scaled(
    family_name: str,
    time_s: np.ndarray,
    t_final_s: float,
    timing_scale: float,
) -> tuple[str, ...]:
    time = np.asarray(time_s, dtype=float).reshape(-1)
    names = _phase_names(family_name)
    edges = _phase_edges(family_name, float(t_final_s), timing_scale)
    labels: list[str] = []
    for value in time:
        index = int(np.searchsorted(edges, float(value), side="right"))
        index = min(index, len(names) - 1)
        labels.append(names[index])
    return tuple(labels)


def _phase_amplitudes(family_name: str) -> dict[str, tuple[float, float, float]]:
    profiles = {
        "canyon_steep_bank": {
            "entry": (0.0, 0.0, 0.0),
            "roll_in": (0.58, 0.06, 0.08),
            "turn_hold": (0.22, 0.10, 0.10),
            "heading_capture": (-0.34, -0.04, -0.12),
            "unload_exit": (-0.08, -0.10, -0.04),
        },
        "wingover_lite": {
            "entry": (0.0, 0.0, 0.0),
            "shallow_pre_dive": (0.0, -0.18, 0.0),
            "climb_roll": (0.38, 0.34, 0.10),
            "crest_redirect": (0.24, 0.10, 0.34),
            "unload_descend": (-0.16, -0.34, -0.10),
            "exit_glide": (-0.04, -0.08, -0.04),
        },
        "bank_yaw_energy_retaining": {
            "entry": (0.0, 0.0, 0.0),
            "bank_yaw_redirect": (0.44, 0.02, 0.30),
            "heading_capture": (-0.24, -0.04, -0.18),
            "early_unload": (-0.10, -0.22, -0.06),
            "exit_glide": (-0.02, -0.06, -0.02),
        },
    }
    if family_name not in profiles:
        raise ValueError(f"unknown agile turn family: {family_name}.")
    return profiles[family_name]


def family_command_profile(
    config: AgileTurnFamilyConfig,
    time_s: np.ndarray,
    family_name: str,
    amplitude_scale: float = 1.0,
    timing_scale: float = 1.0,
) -> np.ndarray:
    """Return bounded normalised command profile with shape (N, 3)."""

    _validate_config(config)
    time = np.asarray(time_s, dtype=float).reshape(-1)
    names = _phase_names(family_name)
    edges = _phase_edges(family_name, float(config.t_final_s), timing_scale)
    amplitudes = _phase_amplitudes(family_name)
    command = np.zeros((time.size, 3), dtype=float)
    for index, value in enumerate(time):
        phase_index = int(np.searchsorted(edges, float(value), side="right"))
        phase_index = min(phase_index, len(names) - 1)
        aileron, elevator, rudder = amplitudes[names[phase_index]]
        command[index, 0] = int(config.direction_sign) * float(amplitude_scale) * aileron
        command[index, 1] = float(amplitude_scale) * elevator
        command[index, 2] = int(config.direction_sign) * float(amplitude_scale) * rudder
    return np.clip(command, -1.0, 1.0)


# =============================================================================
# 3) Replay and Metrics
# =============================================================================
def replay_family_candidate(
    x0: np.ndarray,
    u_norm_requested: np.ndarray,
    time_s: np.ndarray,
    config: AgileTurnFamilyConfig,
    aircraft: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replay through rk4_step using delta_cmd_rad only."""

    _validate_config(config)
    time = np.asarray(time_s, dtype=float).reshape(-1)
    requested = np.asarray(u_norm_requested, dtype=float)
    if requested.shape != (time.size, 3):
        raise ValueError("u_norm_requested must have shape (N, 3).")
    x_log = np.empty((time.size, STATE_SIZE), dtype=float)
    u_applied = np.empty((time.size, 3), dtype=float)
    command_rad = np.empty((time.size, 3), dtype=float)
    x_log[0] = as_state_vector(x0)
    for index in range(time.size):
        u_applied[index] = clip_normalised_command(requested[index])
        command_rad[index] = normalised_command_to_surface_rad(u_applied[index])
        if index == time.size - 1:
            break
        x_log[index + 1] = rk4_step(
            x_log[index],
            command_rad[index],
            float(time[index + 1] - time[index]),
            aircraft,
            None,
            config.wind_mode,
            config.actuator_tau_s,
        )
    return x_log, u_applied, command_rad


def _speed_alpha_beta(x_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    velocity = np.asarray(x_ref, dtype=float)[:, 6:9]
    speed = np.linalg.norm(velocity, axis=1)
    alpha = np.arctan2(velocity[:, 2], velocity[:, 0])
    beta = np.zeros_like(speed)
    valid = speed > 1e-9
    beta[valid] = np.arcsin(np.clip(velocity[valid, 1] / speed[valid], -1.0, 1.0))
    return speed, alpha, beta


def _specific_energy_height_m(x_ref: np.ndarray) -> np.ndarray:
    speed = np.linalg.norm(np.asarray(x_ref, dtype=float)[:, 6:9], axis=1)
    return np.asarray(x_ref, dtype=float)[:, STATE_INDEX["z_w"]] + speed**2 / (2.0 * G_M_S2)


def _heading_change_deg(x_ref: np.ndarray, direction_sign: int) -> np.ndarray:
    yaw = np.unwrap(np.asarray(x_ref, dtype=float)[:, STATE_INDEX["psi"]])
    return np.rad2deg(int(direction_sign) * (yaw - yaw[0]))


def _margin_metrics(x_ref: np.ndarray) -> dict[str, float]:
    finite_positions = [
        position for position in np.asarray(x_ref, dtype=float)[:, 0:3]
        if np.all(np.isfinite(position))
    ]
    if not finite_positions:
        return {
            "min_true_wall_margin_m": float("nan"),
            "min_floor_margin_m": float("nan"),
            "min_ceiling_margin_m": float("nan"),
            "min_true_margin_m": float("nan"),
        }
    rows = [position_margin_m(position, TRUE_SAFE_BOUNDS) for position in finite_positions]
    return {
        "min_true_wall_margin_m": float(min(row["min_wall_margin_m"] for row in rows)),
        "min_floor_margin_m": float(min(row["floor_margin_m"] for row in rows)),
        "min_ceiling_margin_m": float(min(row["ceiling_margin_m"] for row in rows)),
        "min_true_margin_m": float(min(row["min_margin_m"] for row in rows)),
    }


def _saturation_metrics(
    u_norm_requested: np.ndarray,
    u_norm_applied: np.ndarray,
    dt_s: float,
) -> tuple[float, float]:
    clipped = np.any(np.abs(u_norm_requested - u_norm_applied) > 1e-12, axis=1)
    if clipped.size <= 1:
        return 0.0, 0.0
    interval_clipped = clipped[:-1]
    count = int(np.count_nonzero(interval_clipped))
    return float(count / interval_clipped.size), float(count * float(dt_s))


def _entry_proxy(state: np.ndarray, spec: object) -> bool:
    try:
        checks = evaluate_entry_set(as_state_vector(state), spec.entry_set)
    except ValueError:
        return False
    return bool(all(check.pass_check for check in checks))


def _recovery_extension_compatible(
    terminal_state: np.ndarray,
    terminal_speed: float,
    max_alpha_deg: float,
    max_beta_deg: float,
) -> bool:
    try:
        state = as_state_vector(terminal_state)
    except ValueError:
        return False
    return bool(
        inside_bounds(state[0:3], TRUE_SAFE_BOUNDS)
        and terminal_speed >= USEFUL_TERMINAL_SPEED_M_S
        and abs(float(max_alpha_deg)) <= USEFUL_ALPHA_MAX_DEG
        and abs(float(max_beta_deg)) <= USEFUL_BETA_MAX_DEG
        and np.linalg.norm(state[9:12]) <= USEFUL_RATE_MAX_RAD_S
    )


def _phase_window(
    phase: tuple[str, ...],
    names: tuple[str, ...],
) -> tuple[int, int] | None:
    indices = np.where(np.isin(np.asarray(phase), list(names)))[0]
    if indices.size == 0:
        return None
    return int(indices[0]), int(indices[-1])


def metrics_for_family_candidate(
    config: AgileTurnFamilyConfig,
    family_name: str,
    time_s: np.ndarray,
    x_ref: np.ndarray,
    u_norm_requested: np.ndarray,
    u_norm_applied: np.ndarray,
    phase: tuple[str, ...],
) -> dict[str, object]:
    """Return heading, energy, speed, exposure, safety, and recoverability metrics."""

    _validate_config(config)
    time = np.asarray(time_s, dtype=float).reshape(-1)
    state = np.asarray(x_ref, dtype=float)
    requested = np.asarray(u_norm_requested, dtype=float)
    applied = np.asarray(u_norm_applied, dtype=float)
    finite = bool(np.all(np.isfinite(state)))
    speed, alpha, beta = _speed_alpha_beta(state)
    heading = _heading_change_deg(state, config.direction_sign)
    energy = _specific_energy_height_m(state)
    margins = _margin_metrics(state)
    true_safe = bool(
        finite
        and all(inside_bounds(position, TRUE_SAFE_BOUNDS) for position in state[:, 0:3])
    )
    saturation_fraction, saturation_time_s = _saturation_metrics(
        requested,
        applied,
        config.dt_s,
    )
    rate_norm = np.linalg.norm(state[:, 9:12], axis=1)
    terminal_state = state[-1]
    terminal_glide_proxy = _entry_proxy(terminal_state, build_glide_primitive_spec())
    terminal_recovery_proxy = _entry_proxy(terminal_state, build_recovery_primitive_spec())
    terminal_speed = float(speed[-1])
    max_alpha_deg = float(np.nanmax(np.abs(np.rad2deg(alpha))))
    max_beta_deg = float(np.nanmax(np.abs(np.rad2deg(beta))))
    max_rate_rad_s = float(np.nanmax(rate_norm))
    recovery_extension = _recovery_extension_compatible(
        terminal_state,
        terminal_speed,
        max_alpha_deg,
        max_beta_deg,
    )
    recoverable = bool(terminal_glide_proxy or terminal_recovery_proxy or recovery_extension)
    actual_heading = float(np.nanmax(heading))
    min_speed = float(np.nanmin(speed))
    energy_lost = float(energy[0] - energy[-1])
    energy_lost_per_deg = float(energy_lost / max(actual_heading, 1e-6))
    unload_window = _phase_window(
        phase,
        ("unload_exit", "unload_descend", "early_unload", "exit_glide"),
    )
    if unload_window is None:
        unload_speed_gain = 0.0
        unload_duration = 0.0
    else:
        start, stop = unload_window
        unload_speed_gain = float(speed[stop] - speed[start])
        unload_duration = float(time[stop] - time[start])

    strict_heading_gate = actual_heading >= 0.8 * float(config.target_heading_deg)
    useful_heading_gate = actual_heading >= max(
        0.6 * float(config.target_heading_deg),
        15.0 if np.isclose(config.target_heading_deg, 30.0) else 0.0,
    )
    strict_exposure = bool(
        max_alpha_deg <= STRICT_ALPHA_MAX_DEG
        and max_beta_deg <= STRICT_BETA_MAX_DEG
        and max_rate_rad_s <= STRICT_RATE_MAX_RAD_S
        and saturation_fraction < STRICT_SATURATION_MAX
    )
    useful_exposure = bool(
        max_alpha_deg <= USEFUL_ALPHA_MAX_DEG
        and max_beta_deg <= USEFUL_BETA_MAX_DEG
        and max_rate_rad_s <= USEFUL_RATE_MAX_RAD_S
        and saturation_fraction < USEFUL_SATURATION_MAX
    )
    strict_family_success = bool(
        finite
        and true_safe
        and strict_heading_gate
        and terminal_speed >= STRICT_TERMINAL_SPEED_M_S
        and min_speed >= STRICT_MIN_SPEED_M_S
        and strict_exposure
        and (terminal_glide_proxy or terminal_recovery_proxy)
    )
    useful_recoverable_candidate = bool(
        finite
        and true_safe
        and useful_heading_gate
        and terminal_speed >= USEFUL_TERMINAL_SPEED_M_S
        and min_speed >= USEFUL_MIN_SPEED_M_S
        and useful_exposure
        and recoverable
    )
    speed_preserving = bool(
        terminal_speed >= USEFUL_TERMINAL_SPEED_M_S and min_speed >= USEFUL_MIN_SPEED_M_S
    )
    horizon_limited = bool(
        (finite and true_safe and speed_preserving and recoverable and not useful_heading_gate)
        or (strict_heading_gate and unload_duration < 0.20 and unload_speed_gain < 0.25)
    )
    safety_limited = bool(finite and not true_safe)
    exposure_limited = bool(
        finite
        and true_safe
        and (
            max_alpha_deg > USEFUL_ALPHA_MAX_DEG
            or max_beta_deg > USEFUL_BETA_MAX_DEG
            or max_rate_rad_s > USEFUL_RATE_MAX_RAD_S
            or saturation_fraction >= USEFUL_SATURATION_MAX
        )
    )
    energy_limited = bool(
        finite
        and true_safe
        and not exposure_limited
        and (terminal_speed < USEFUL_TERMINAL_SPEED_M_S or min_speed < USEFUL_MIN_SPEED_M_S)
    )
    turn_authority_limited = bool(
        finite
        and true_safe
        and not horizon_limited
        and not energy_limited
        and not exposure_limited
        and not useful_heading_gate
    )
    if strict_family_success:
        candidate_class = "strict_family_success"
        failure_label = "success"
        notes = "strict_family_success"
    elif useful_recoverable_candidate:
        candidate_class = "useful_recoverable_candidate"
        failure_label = "under_turning" if not strict_heading_gate else "success"
        notes = "useful_recoverable_candidate"
    else:
        candidate_class = "boundary_evidence"
        if not finite:
            failure_label = "nonfinite_state"
            notes = "nonfinite_replay"
        elif not true_safe:
            failure_label = "true_safety_violation"
            notes = "safety_limited"
        elif exposure_limited:
            if max_alpha_deg > USEFUL_ALPHA_MAX_DEG:
                failure_label = "alpha_boundary"
                notes = "exposure_limited"
            elif max_beta_deg > USEFUL_BETA_MAX_DEG:
                failure_label = "beta_boundary"
                notes = "exposure_limited"
            elif max_rate_rad_s > USEFUL_RATE_MAX_RAD_S:
                failure_label = "rate_boundary"
                notes = "exposure_limited"
            else:
                failure_label = "actuator_saturation_limited"
                notes = "exposure_limited"
        elif energy_limited:
            failure_label = "speed_low"
            notes = "energy_limited"
        elif not recoverable:
            failure_label = "terminal_recovery_limited"
            notes = "terminal_recovery_limited"
        elif horizon_limited:
            failure_label = "under_turning"
            notes = "horizon_limited"
        elif not useful_heading_gate:
            failure_label = "under_turning"
            notes = "turn_authority_limited"
        else:
            failure_label = "model_boundary_only"
            notes = "boundary_evidence"

    if safety_limited:
        family_status = "retained_as_thesis_discussion_evidence"
    elif strict_family_success or useful_recoverable_candidate:
        family_status = "selected_for_next_stage"
    elif finite:
        family_status = "retained_as_thesis_discussion_evidence"
    else:
        family_status = "rejected_for_active_primitive"

    return {
        "family_name": family_name,
        "target_heading_deg": float(config.target_heading_deg),
        "horizon_s": float(config.t_final_s),
        "success": strict_family_success,
        "failure_label": failure_label,
        "notes": notes,
        "candidate_class": candidate_class,
        "strict_family_success": strict_family_success,
        "useful_recoverable_candidate": useful_recoverable_candidate,
        "boundary_evidence": bool(not strict_family_success and not useful_recoverable_candidate),
        "actual_heading_change_deg": actual_heading,
        "heading_ratio": float(actual_heading / max(float(config.target_heading_deg), 1e-6)),
        "terminal_speed_m_s": terminal_speed,
        "min_speed_m_s": min_speed,
        "initial_speed_m_s": float(speed[0]),
        "height_change_m": float(state[-1, STATE_INDEX["z_w"]] - state[0, STATE_INDEX["z_w"]]),
        "specific_energy_initial_m": float(energy[0]),
        "specific_energy_terminal_m": float(energy[-1]),
        "specific_energy_lost_m": energy_lost,
        "energy_lost_per_deg_m_per_deg": energy_lost_per_deg,
        "max_alpha_deg": max_alpha_deg,
        "max_beta_deg": max_beta_deg,
        "max_bank_deg": float(np.nanmax(np.abs(np.rad2deg(state[:, STATE_INDEX["phi"]])))),
        "max_pitch_deg": float(np.nanmax(np.abs(np.rad2deg(state[:, STATE_INDEX["theta"]])))),
        "max_rate_rad_s": max_rate_rad_s,
        **margins,
        "saturation_fraction": saturation_fraction,
        "saturation_time_s": saturation_time_s,
        "recoverable": recoverable,
        "terminal_glide_entry_proxy": terminal_glide_proxy,
        "terminal_recovery_entry_proxy": terminal_recovery_proxy,
        "recovery_extension_compatible": recovery_extension,
        "unload_speed_gain_m_s": unload_speed_gain,
        "unload_exit_duration_s": unload_duration,
        "horizon_limited": horizon_limited,
        "turn_authority_limited": turn_authority_limited,
        "energy_limited": energy_limited,
        "safety_limited": safety_limited,
        "exposure_limited": exposure_limited,
        "selected_method": "family_replay_grid",
        "family_status": family_status,
        "finite_state_success": finite,
        "true_safe_trajectory": true_safe,
    }


# =============================================================================
# 4) Ranking and Comparison
# =============================================================================
def candidate_ranking_key(metrics: dict[str, object]) -> tuple[object, ...]:
    """Return ranking key that prefers recoverability over raw heading."""

    recoverable_heading = float(metrics["actual_heading_change_deg"]) if bool(metrics["recoverable"]) else 0.0
    return (
        int(bool(metrics["strict_family_success"])),
        int(bool(metrics["useful_recoverable_candidate"])),
        recoverable_heading,
        float(metrics["terminal_speed_m_s"]),
        float(metrics["min_speed_m_s"]),
        -float(metrics["energy_lost_per_deg_m_per_deg"]),
        -float(metrics["max_alpha_deg"]),
        -float(metrics["max_beta_deg"]),
        -float(metrics["max_rate_rad_s"]),
        float(metrics["min_true_margin_m"]),
        -float(metrics["saturation_fraction"]),
    )


def _candidate_id(
    family_name: str,
    target_heading_deg: float,
    horizon_s: float,
    amplitude_scale: float,
    timing_scale: float,
) -> str:
    target_token = f"{int(round(float(target_heading_deg))):03d}"
    horizon_token = f"{int(round(float(horizon_s) * 100)):03d}"
    amp_token = f"{int(round(float(amplitude_scale) * 100)):03d}"
    timing_token = f"{int(round(float(timing_scale) * 100)):03d}"
    return f"{family_name}_t{target_token}_h{horizon_token}_a{amp_token}_q{timing_token}"


def _result_from_replay(
    config: AgileTurnFamilyConfig,
    family_name: str,
    candidate_id: str,
    amplitude_scale: float,
    timing_scale: float,
    x0: np.ndarray,
    aircraft: object,
) -> AgileTurnCandidateResult:
    time_s = _time_grid(config)
    requested = family_command_profile(
        config,
        time_s,
        family_name,
        amplitude_scale=amplitude_scale,
        timing_scale=timing_scale,
    )
    x_ref, applied, command_rad = replay_family_candidate(
        x0,
        requested,
        time_s,
        config,
        aircraft,
    )
    phase = _phase_labels_for_family_scaled(
        family_name,
        time_s,
        config.t_final_s,
        timing_scale,
    )
    metrics = metrics_for_family_candidate(
        config,
        family_name,
        time_s,
        x_ref,
        requested,
        applied,
        phase,
    )
    metrics.update(
        {
            "candidate_id": candidate_id,
            "phase_search_scale": float(amplitude_scale),
            "timing_scale": float(timing_scale),
        }
    )
    return AgileTurnCandidateResult(
        family_name=family_name,
        target_heading_deg=float(config.target_heading_deg),
        direction_sign=int(config.direction_sign),
        success=bool(metrics["success"]),
        failure_label=str(metrics["failure_label"]),
        time_s=time_s,
        x_ref=x_ref,
        u_norm_requested=requested,
        u_norm_applied=applied,
        delta_cmd_rad=command_rad,
        phase=phase,
        metrics=metrics,
        notes=str(metrics["notes"]),
    )


def _best_by_metric(
    candidates: tuple[AgileTurnCandidateResult, ...],
    metric_name: str,
    require_recoverable: bool = False,
) -> AgileTurnCandidateResult:
    filtered = [
        candidate for candidate in candidates
        if not require_recoverable or bool(candidate.metrics["recoverable"])
    ]
    if not filtered:
        filtered = list(candidates)
    return max(filtered, key=lambda candidate: float(candidate.metrics[metric_name]))


def _ranking_rows(candidates: tuple[AgileTurnCandidateResult, ...]) -> tuple[dict[str, object], ...]:
    best_by_heading = _best_by_metric(candidates, "actual_heading_change_deg")
    best_by_terminal_speed = _best_by_metric(candidates, "terminal_speed_m_s")
    best_by_recoverable_heading = _best_by_metric(
        candidates,
        "actual_heading_change_deg",
        require_recoverable=True,
    )
    best_by_energy = min(
        candidates,
        key=lambda candidate: float(candidate.metrics["energy_lost_per_deg_m_per_deg"]),
    )
    selected = max(candidates, key=lambda candidate: candidate_ranking_key(candidate.metrics))
    rows = []
    for candidate in candidates:
        metrics = candidate.metrics
        rows.append(
            {
                **metrics,
                "best_by_heading": metrics["candidate_id"] == best_by_heading.metrics["candidate_id"],
                "best_by_terminal_speed": metrics["candidate_id"] == best_by_terminal_speed.metrics["candidate_id"],
                "best_by_recoverable_heading": metrics["candidate_id"] == best_by_recoverable_heading.metrics["candidate_id"],
                "best_by_energy_lost_per_deg": metrics["candidate_id"] == best_by_energy.metrics["candidate_id"],
                "selected_candidate": metrics["candidate_id"] == selected.metrics["candidate_id"],
            }
        )
    return tuple(rows)


def compare_agile_turn_families(
    config: AgileTurnFamilyConfig,
    families: tuple[str, ...] | None = None,
    x0: np.ndarray | None = None,
    aircraft: object | None = None,
) -> AgileTurnFamilyComparisonResult:
    """Run bounded family search and select the most promising recoverable family."""

    _validate_config(config)
    selected_families = family_inventory() if families is None else tuple(families)
    unknown = set(selected_families) - set(family_inventory())
    if unknown:
        raise ValueError(f"unknown agile turn families: {sorted(unknown)}")
    aircraft_model = adapt_glider(build_nausicaa_glider()) if aircraft is None else aircraft
    initial_state = build_family_initial_state(config, aircraft_model) if x0 is None else as_state_vector(x0)
    candidates: list[AgileTurnCandidateResult] = []
    for horizon_s in horizon_grid_s(config.target_heading_deg):
        horizon_config = AgileTurnFamilyConfig(
            dt_s=config.dt_s,
            t_final_s=float(horizon_s),
            target_heading_deg=config.target_heading_deg,
            direction_sign=config.direction_sign,
            speed_m_s=config.speed_m_s,
            altitude_m=config.altitude_m,
            wind_mode=config.wind_mode,
            latency_case=config.latency_case,
            actuator_tau_s=config.actuator_tau_s,
            seed=config.seed,
            phase_search_scales=config.phase_search_scales,
        )
        for family_name in selected_families:
            for amplitude_scale in config.phase_search_scales:
                for timing_scale in (0.95, 1.00, 1.05):
                    candidate_id = _candidate_id(
                        family_name,
                        config.target_heading_deg,
                        horizon_s,
                        amplitude_scale,
                        timing_scale,
                    )
                    candidates.append(
                        _result_from_replay(
                            horizon_config,
                            family_name,
                            candidate_id,
                            amplitude_scale,
                            timing_scale,
                            initial_state,
                            aircraft_model,
                        )
                    )
    candidate_tuple = tuple(candidates)
    selected = max(candidate_tuple, key=lambda candidate: candidate_ranking_key(candidate.metrics))
    rows = _ranking_rows(candidate_tuple)
    if bool(selected.metrics["strict_family_success"]):
        reason = "strict_family_success"
    elif bool(selected.metrics["useful_recoverable_candidate"]):
        reason = "useful_recoverable_candidate"
    elif bool(selected.metrics["recoverable"]):
        reason = "largest_recoverable_heading_under_current_gates"
    else:
        reason = "boundary_evidence_best_available"
    for row in rows:
        if row["candidate_id"] == selected.metrics["candidate_id"]:
            row["selection_reason"] = reason
        else:
            row["selection_reason"] = ""
    return AgileTurnFamilyComparisonResult(
        target_heading_deg=float(config.target_heading_deg),
        family_results=candidate_tuple,
        selected_family=selected.family_name,
        selected_candidate=selected,
        ranking_rows=rows,
        notes=reason,
    )
