from __future__ import annotations

import math
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
from command_contract import clip_normalised_command, normalised_command_to_surface_rad
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
# 1) Data contracts and target gates
# 2) Family profiles and replay
# 3) Candidate metrics and classification
# 4) Ranking and comparison
# =============================================================================


# =============================================================================
# 1) Data Contracts and Target Gates
# =============================================================================
AGILE_TURN_CAMPAIGN = "08_agile_turn_family_comparison"
TARGET_LADDER_DEG = (15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 150.0, 180.0)
DEFAULT_TARGETS_DEG = (15.0, 30.0)
FAMILY_NAMES = (
    "canyon_steep_bank",
    "wingover_lite",
    "bank_yaw_energy_retaining",
)
ARCHIVED_BOUNDARY_REFERENCE = (
    "03_Control/05_Results/07_"
    + "aggressive"
    + "_reversal"
    + "_ocp/002"
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

TARGET_HORIZON_GRID_S = {
    15.0: (0.36, 0.46, 0.60, 0.76, 0.90),
    30.0: (0.46, 0.60, 0.80, 1.00, 1.20),
    45.0: (0.60, 0.80, 1.06, 1.30, 1.56),
    60.0: (0.76, 1.00, 1.26, 1.56, 1.86),
    90.0: (1.00, 1.30, 1.70, 2.10, 2.50),
    120.0: (1.20, 1.60, 2.00, 2.50, 3.00),
    150.0: (1.40, 1.90, 2.40, 3.00, 3.60),
    180.0: (1.60, 2.20, 2.80, 3.40, 4.00),
}

CANDIDATE_CLASSES = (
    "commandable_target_candidate",
    "accurate_boundary_evidence",
    "safe_partial_turn_evidence",
    "unsafe_or_nonrecoverable_boundary",
)


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
    amplitude_scales: tuple[float, ...] = (0.70, 0.85, 1.00, 1.15)
    timing_scales: tuple[float, ...] = (0.90, 1.00, 1.10)


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
    selected_candidate: AgileTurnCandidateResult | None
    ranking_rows: tuple[dict[str, object], ...]
    notes: str


def family_inventory() -> tuple[str, ...]:
    """Return active speed-retaining agile-turn family labels."""

    return FAMILY_NAMES


def target_ladder_deg() -> tuple[float, ...]:
    """Return the fixed precision target ladder in degrees."""

    return TARGET_LADDER_DEG


def horizon_grid_s(target_heading_deg: float) -> tuple[float, ...]:
    """Return fixed-step-compatible horizons for a target."""

    target = float(target_heading_deg)
    for key, values in TARGET_HORIZON_GRID_S.items():
        if np.isclose(target, key):
            return values
    raise ValueError("target must be one of the fixed precision ladder values.")


def heading_band_deg(target_heading_deg: float) -> tuple[float, float]:
    """Return terminal heading acceptance band in degrees."""

    target = float(target_heading_deg)
    tolerance = max(2.0, 0.10 * target)
    return target - tolerance, target + tolerance


def heading_accuracy_metrics(
    x_ref: np.ndarray,
    direction_sign: int,
    target_heading_deg: float,
) -> dict[str, float | bool]:
    """Return terminal-band metrics using unwrapped signed yaw."""

    state = np.asarray(x_ref, dtype=float)
    yaw = np.unwrap(state[:, STATE_INDEX["psi"]])
    heading_series = np.rad2deg(int(direction_sign) * (yaw - yaw[0]))
    terminal_heading = float(heading_series[-1])
    peak_heading = float(np.nanmax(heading_series))
    target = float(target_heading_deg)
    lower, upper = heading_band_deg(target)
    error = terminal_heading - target
    return {
        "terminal_heading_change_deg": terminal_heading,
        "peak_heading_change_deg": peak_heading,
        "heading_error_deg": float(error),
        "absolute_heading_error_deg": float(abs(error)),
        "heading_band_lower_deg": float(lower),
        "heading_band_upper_deg": float(upper),
        "heading_band_pass": bool(lower <= terminal_heading <= upper),
        "overshoot_deg": float(max(0.0, terminal_heading - upper)),
        "undershoot_deg": float(max(0.0, lower - terminal_heading)),
    }


def acceptance_thresholds() -> dict[str, float]:
    """Return scalar gates used in candidate classification."""

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
    if not any(np.isclose(float(config.target_heading_deg), value) for value in TARGET_LADDER_DEG):
        raise ValueError("target_heading_deg must be in the fixed precision ladder.")
    if config.wind_mode != "none":
        raise ValueError("this evidence pass is W0/no-wind only.")
    if config.latency_case != "none":
        raise ValueError("this evidence pass uses the nominal no-latency case only.")
    if tuple(config.amplitude_scales) == () or tuple(config.timing_scales) == ():
        raise ValueError("amplitude_scales and timing_scales must not be empty.")


def _time_grid(config: AgileTurnFamilyConfig) -> np.ndarray:
    step_count = int(round(float(config.t_final_s) / float(config.dt_s)))
    return np.arange(step_count + 1, dtype=float) * float(config.dt_s)


# =============================================================================
# 2) Family Profiles and Replay
# =============================================================================
def build_family_initial_state(
    config: AgileTurnFamilyConfig,
    aircraft: object | None = None,
) -> np.ndarray:
    """Return a trim-based W0 initial state in canonical 15-state order."""

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


def _phase_edges(
    family_name: str,
    t_final_s: float,
    timing_scale: float = 1.0,
) -> tuple[float, ...]:
    scale = float(np.clip(timing_scale, 0.85, 1.15))
    base = {
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


def phase_labels_for_family(
    family_name: str,
    time_s: np.ndarray,
    t_final_s: float,
) -> tuple[str, ...]:
    """Return nominal phase labels for human-facing logs."""

    return _phase_labels_for_family_scaled(family_name, time_s, t_final_s, 1.0)


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
        labels.append(names[min(index, len(names) - 1)])
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
    """Return requested normalised command profile with shape ``(N, 3)``."""

    _validate_config(config)
    time = np.asarray(time_s, dtype=float).reshape(-1)
    names = _phase_names(family_name)
    edges = _phase_edges(family_name, float(config.t_final_s), timing_scale)
    amplitudes = _phase_amplitudes(family_name)
    command = np.zeros((time.size, 3), dtype=float)
    for index, value in enumerate(time):
        phase_index = int(np.searchsorted(edges, float(value), side="right"))
        aileron, elevator, rudder = amplitudes[names[min(phase_index, len(names) - 1)]]
        command[index, 0] = int(config.direction_sign) * float(amplitude_scale) * aileron
        command[index, 1] = float(amplitude_scale) * elevator
        command[index, 2] = int(config.direction_sign) * float(amplitude_scale) * rudder
    return np.clip(command, -1.0, 1.0)


def replay_family_candidate(
    x0: np.ndarray,
    u_norm_requested: np.ndarray,
    time_s: np.ndarray,
    config: AgileTurnFamilyConfig,
    aircraft: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replay through ``rk4_step`` with physical radian commands only."""

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


# =============================================================================
# 3) Candidate Metrics and Classification
# =============================================================================
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


def _margin_metrics(x_ref: np.ndarray) -> dict[str, float]:
    positions = np.asarray(x_ref, dtype=float)[:, 0:3]
    finite_positions = [position for position in positions if np.all(np.isfinite(position))]
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
        and max_alpha_deg <= USEFUL_ALPHA_MAX_DEG
        and max_beta_deg <= USEFUL_BETA_MAX_DEG
        and np.linalg.norm(state[9:12]) <= USEFUL_RATE_MAX_RAD_S
    )


def _classification(
    finite: bool,
    true_safe: bool,
    heading_band_pass: bool,
    useful_speed: bool,
    useful_exposure: bool,
    recoverable: bool,
    heading_error_deg: float,
) -> tuple[str, str, str, str]:
    if heading_band_pass and true_safe and finite and useful_speed and useful_exposure and recoverable:
        return "commandable_target_candidate", "success", "target_band_commandable", "success"
    if heading_band_pass:
        if not finite:
            return "accurate_boundary_evidence", "nonfinite_state", "solver_or_horizon_limited", "nonfinite_replay"
        if not true_safe:
            return "accurate_boundary_evidence", "true_safety_violation", "safety_boundary_target_miss", "safety_limited"
        if not useful_exposure:
            return "accurate_boundary_evidence", "alpha_boundary", "exposure_target_miss", "exposure_limited"
        return "accurate_boundary_evidence", "terminal_recovery_limited", "energy_or_recovery_target_miss", "boundary"
    if finite and true_safe and useful_speed and useful_exposure and recoverable:
        mechanism = "under_turning_target_miss" if heading_error_deg < 0.0 else "over_turning_target_miss"
        return "safe_partial_turn_evidence", "under_turning", mechanism, "safe_target_miss"
    if not finite:
        return "unsafe_or_nonrecoverable_boundary", "nonfinite_state", "solver_or_horizon_limited", "nonfinite_replay"
    if not true_safe:
        return "unsafe_or_nonrecoverable_boundary", "true_safety_violation", "safety_boundary_target_miss", "safety_limited"
    if not useful_exposure:
        return "unsafe_or_nonrecoverable_boundary", "alpha_boundary", "exposure_target_miss", "exposure_limited"
    return "unsafe_or_nonrecoverable_boundary", "terminal_recovery_limited", "energy_or_recovery_target_miss", "boundary"


def metrics_for_family_candidate(
    config: AgileTurnFamilyConfig,
    family_name: str,
    time_s: np.ndarray,
    x_ref: np.ndarray,
    u_norm_requested: np.ndarray,
    u_norm_applied: np.ndarray,
    phase: tuple[str, ...],
) -> dict[str, object]:
    """Return target-accuracy, safety, exposure, and recoverability metrics."""

    _validate_config(config)
    time = np.asarray(time_s, dtype=float).reshape(-1)
    state = np.asarray(x_ref, dtype=float)
    requested = np.asarray(u_norm_requested, dtype=float)
    applied = np.asarray(u_norm_applied, dtype=float)
    finite = bool(np.all(np.isfinite(state)))
    speed, alpha, beta = _speed_alpha_beta(state)
    energy = _specific_energy_height_m(state)
    heading = heading_accuracy_metrics(state, config.direction_sign, config.target_heading_deg)
    margins = _margin_metrics(state)
    true_safe = bool(
        finite
        and all(inside_bounds(position, TRUE_SAFE_BOUNDS) for position in state[:, 0:3])
    )
    saturation_fraction, saturation_time_s = _saturation_metrics(requested, applied, config.dt_s)
    rate_norm = np.linalg.norm(state[:, 9:12], axis=1)
    terminal_state = state[-1]
    terminal_glide_proxy = _entry_proxy(terminal_state, build_glide_primitive_spec())
    terminal_recovery_proxy = _entry_proxy(terminal_state, build_recovery_primitive_spec())
    terminal_speed = float(speed[-1])
    min_speed = float(np.nanmin(speed))
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
    strict_speed = terminal_speed >= STRICT_TERMINAL_SPEED_M_S and min_speed >= STRICT_MIN_SPEED_M_S
    useful_speed = terminal_speed >= USEFUL_TERMINAL_SPEED_M_S and min_speed >= USEFUL_MIN_SPEED_M_S
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
    candidate_class, failure_label, mechanism, notes = _classification(
        finite,
        true_safe,
        bool(heading["heading_band_pass"]),
        bool(useful_speed),
        useful_exposure,
        recoverable,
        float(heading["heading_error_deg"]),
    )
    commandable = candidate_class == "commandable_target_candidate"
    strict_target_success = bool(commandable and strict_speed and strict_exposure)
    useful_target_success = bool(commandable and useful_speed and useful_exposure)
    forward_travel = float(state[-1, STATE_INDEX["x_w"]] - state[0, STATE_INDEX["x_w"]])
    lateral_span = float(np.nanmax(state[:, STATE_INDEX["y_w"]]) - np.nanmin(state[:, STATE_INDEX["y_w"]]))
    longitudinal_span = float(np.nanmax(state[:, STATE_INDEX["x_w"]]) - np.nanmin(state[:, STATE_INDEX["x_w"]]))
    energy_lost = float(energy[0] - energy[-1])
    terminal_heading = float(heading["terminal_heading_change_deg"])
    return {
        "family_name": family_name,
        "target_heading_deg": float(config.target_heading_deg),
        "horizon_s": float(config.t_final_s),
        "candidate_class": candidate_class,
        "success": commandable,
        "failure_label": failure_label,
        "notes": notes,
        "active_limiting_mechanism": mechanism,
        "strict_target_success": strict_target_success,
        "useful_target_success": useful_target_success,
        "commandable_target_candidate": commandable,
        "accurate_boundary_evidence": candidate_class == "accurate_boundary_evidence",
        "safe_partial_turn_evidence": candidate_class == "safe_partial_turn_evidence",
        "unsafe_or_nonrecoverable_boundary": candidate_class == "unsafe_or_nonrecoverable_boundary",
        **heading,
        "terminal_speed_m_s": terminal_speed,
        "min_speed_m_s": min_speed,
        "initial_speed_m_s": float(speed[0]),
        "height_change_m": float(state[-1, STATE_INDEX["z_w"]] - state[0, STATE_INDEX["z_w"]]),
        "forward_travel_m": forward_travel,
        "turn_footprint_proxy_m2": abs(longitudinal_span * max(lateral_span, 1e-9)),
        "specific_energy_initial_m": float(energy[0]),
        "specific_energy_terminal_m": float(energy[-1]),
        "specific_energy_lost_m": energy_lost,
        "energy_lost_per_deg_m_per_deg": float(energy_lost / max(abs(terminal_heading), 1e-6)),
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
        "true_safe_trajectory": true_safe,
        "finite_state_success": finite,
        "selected_method": "family_replay_grid",
    }


# =============================================================================
# 4) Ranking and Comparison
# =============================================================================
def candidate_ranking_key(metrics: dict[str, object]) -> tuple[object, ...]:
    """Rank commandable candidates before duration and footprint objectives."""

    commandable = int(bool(metrics["commandable_target_candidate"]))
    heading_pass = int(bool(metrics["heading_band_pass"]))
    true_safe = int(bool(metrics["true_safe_trajectory"]))
    speed_gate = int(
        float(metrics["terminal_speed_m_s"]) >= USEFUL_TERMINAL_SPEED_M_S
        and float(metrics["min_speed_m_s"]) >= USEFUL_MIN_SPEED_M_S
    )
    recoverable = int(bool(metrics["recoverable"]))
    return (
        commandable,
        heading_pass,
        true_safe,
        speed_gate,
        recoverable,
        -float(metrics["horizon_s"]),
        -abs(float(metrics["forward_travel_m"])),
        -float(metrics["turn_footprint_proxy_m2"]),
        -float(metrics["energy_lost_per_deg_m_per_deg"]),
        -float(metrics["max_alpha_deg"]),
        -float(metrics["max_beta_deg"]),
        -float(metrics["max_rate_rad_s"]),
        -float(metrics["saturation_fraction"]),
        float(metrics["min_true_margin_m"]),
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
    phase = _phase_labels_for_family_scaled(family_name, time_s, config.t_final_s, timing_scale)
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
            "amplitude_scale": float(amplitude_scale),
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


def _best_candidate(
    candidates: tuple[AgileTurnCandidateResult, ...],
    predicate: str,
) -> AgileTurnCandidateResult | None:
    filtered = [candidate for candidate in candidates if bool(candidate.metrics[predicate])]
    if not filtered:
        return None
    return max(filtered, key=lambda candidate: candidate_ranking_key(candidate.metrics))


def _best_any_heading(candidates: tuple[AgileTurnCandidateResult, ...]) -> AgileTurnCandidateResult:
    return max(
        candidates,
        key=lambda candidate: float(candidate.metrics["peak_heading_change_deg"]),
    )


def _ranking_rows(candidates: tuple[AgileTurnCandidateResult, ...]) -> tuple[dict[str, object], ...]:
    commandable = _best_candidate(candidates, "commandable_target_candidate")
    safe_partial = _best_candidate(candidates, "safe_partial_turn_evidence")
    accurate_boundary = _best_candidate(candidates, "accurate_boundary_evidence")
    best_any = _best_any_heading(candidates)
    selected_id = commandable.metrics["candidate_id"] if commandable is not None else ""
    rows = []
    for candidate in candidates:
        metrics = candidate.metrics
        rows.append(
            {
                **metrics,
                "selected_candidate": metrics["candidate_id"] == selected_id,
                "best_commandable_candidate": (
                    commandable is not None
                    and metrics["candidate_id"] == commandable.metrics["candidate_id"]
                ),
                "best_safe_partial_candidate": (
                    safe_partial is not None
                    and metrics["candidate_id"] == safe_partial.metrics["candidate_id"]
                ),
                "best_accurate_boundary_candidate": (
                    accurate_boundary is not None
                    and metrics["candidate_id"] == accurate_boundary.metrics["candidate_id"]
                ),
                "best_any_heading_candidate": metrics["candidate_id"] == best_any.metrics["candidate_id"],
            }
        )
    return tuple(rows)


def compare_agile_turn_families(
    config: AgileTurnFamilyConfig,
    families: tuple[str, ...] | None = None,
    x0: np.ndarray | None = None,
    aircraft: object | None = None,
) -> AgileTurnFamilyComparisonResult:
    """Run bounded family search and select only target-accurate commandable candidates."""

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
            amplitude_scales=config.amplitude_scales,
            timing_scales=config.timing_scales,
        )
        for family_name in selected_families:
            for amplitude_scale in config.amplitude_scales:
                for timing_scale in config.timing_scales:
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
    commandable = _best_candidate(candidate_tuple, "commandable_target_candidate")
    rows = _ranking_rows(candidate_tuple)
    if commandable is None:
        reason = "no_commandable_target_candidate"
        selected_family = ""
    else:
        reason = "shortest_commandable_target_candidate"
        selected_family = commandable.family_name
    for row in rows:
        row["selection_reason"] = reason if bool(row["selected_candidate"]) else ""
    return AgileTurnFamilyComparisonResult(
        target_heading_deg=float(config.target_heading_deg),
        family_results=candidate_tuple,
        selected_family=selected_family,
        selected_candidate=commandable,
        ranking_rows=rows,
        notes=reason,
    )
