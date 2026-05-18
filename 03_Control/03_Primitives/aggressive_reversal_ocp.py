from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import casadi as ca
import numpy as np

PRIMITIVES_DIR = Path(__file__).resolve().parent
CONTROL_DIR = PRIMITIVES_DIR.parents[0]
INNER_LOOP_DIR = CONTROL_DIR / "02_Inner_Loop"
SCENARIOS_DIR = CONTROL_DIR / "04_Scenarios"
for path in (INNER_LOOP_DIR, PRIMITIVES_DIR, SCENARIOS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aggressive_reversal_primitive import terminal_aggressive_recoverable_proxy
from arena_contract import TRUE_SAFE_BOUNDS, inside_bounds, position_margin_m
from command_contract import (
    clip_normalised_command,
    normalised_command_to_surface_rad,
)
from flight_dynamics import adapt_glider, build_symbolic_dynamics
from glider import build_nausicaa_glider
from metric_contract import FAILURE_LABELS
from rollout import rk4_step
from state_contract import STATE_INDEX, STATE_SIZE, as_state_vector
from trim_solver import TrimTarget, solve_straight_trim


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Constants and dataclasses
# 2) Seed families and phase profiles
# 3) Replay and metrics
# 4) Guided search and direct OCP attempts
# 5) Public solve workflow
# =============================================================================


# =============================================================================
# 1) Constants and Dataclasses
# =============================================================================
AGGRESSIVE_CAMPAIGN = "07_aggressive_reversal_ocp"
PHASES = (
    "entry",
    "pre_dive_accelerate",
    "pitch_brake",
    "slow_redirect",
    "heading_capture",
    "unload_descend",
    "exit_glide",
)
BASE_SEED_FAMILIES = (
    "short_perch_yaw_redirect",
    "long_perch_slow_redirect",
    "roll_dominant_banked_redirect",
    "split_pulse_redirect",
    "early_unload_descend_capture",
)
THIRTY_DEG_SEED_FAMILIES = (
    "dive_perch_redirect_30",
    "reduced_perch_redirect_30",
    "bank_yaw_redirect_30",
    "early_unload_recovery_30",
)
SEED_FAMILIES = BASE_SEED_FAMILIES + THIRTY_DEG_SEED_FAMILIES
TARGET_HORIZONS_S = {
    15.0: 0.70,
    30.0: 0.86,
    60.0: 1.06,
    90.0: 1.20,
    120.0: 1.40,
    180.0: 1.60,
}
REPLAY_DEFECT_TOL = 1e-5
TARGET_TOKEN_WIDTH = 3
G_M_S2 = 9.81
STRICT_30_HEADING_DEG = 24.0
STRICT_TERMINAL_SPEED_M_S = 5.0
RELAXED_TERMINAL_SPEED_M_S = 4.0
UNLOAD_EXIT_DESCENT_REQUIRED_M = 0.15
UNLOAD_EXIT_SPEED_GAIN_REQUIRED_M_S = 0.50


@dataclass(frozen=True)
class AggressiveReversalOcpConfig:
    dt_s: float = 0.02
    t_final_s: float = 1.20
    target_heading_deg: float = 15.0
    direction_sign: int = 1
    speed_m_s: float = 6.5
    altitude_m: float = 1.8
    wind_mode: str = "none"
    latency_case: str = "none"
    actuator_tau_s: tuple[float, float, float] = (0.06, 0.06, 0.06)
    seed: int = 1
    max_ipopt_iter: int = 80
    use_nonlinear_ocp: bool = True
    use_phase_seed_fallback: bool = True
    phase_search_scales: tuple[float, ...] = (0.85, 1.0, 1.15)
    ocp_node_count: int = 2
    ocp_max_cpu_time_s: float = 0.75
    checkpoint_dir: str | None = None
    candidate_log_dir: str | None = None
    run_id: str = "s001"
    previous_solution_path: str | None = None


@dataclass(frozen=True)
class AggressiveReversalOcpResult:
    target_heading_deg: float
    direction_sign: int
    success: bool
    failure_label: str
    time_s: np.ndarray
    x_ref: np.ndarray
    u_ff_norm: np.ndarray
    u_norm_applied: np.ndarray
    delta_cmd_rad: np.ndarray
    phase: tuple[str, ...]
    metrics: dict[str, object]
    notes: str


@dataclass(frozen=True)
class CandidateResult:
    method: str
    family_name: str
    attempt_index: int
    result: AggressiveReversalOcpResult
    nlp_constructed: bool = False
    ipopt_called: bool = False
    solver_status: str = "not_run"
    solver_iter_count: int = 0
    solver_objective: float = np.nan
    constraint_residual_max: float = np.nan
    direct_ocp_attempted: bool = False
    direct_ocp_converged: bool = False
    replay_defect_max: float = np.nan
    limiting_mechanism: str = ""


# =============================================================================
# 2) Seed Families and Phase Profiles
# =============================================================================
def target_horizon_s(target_heading_deg: float) -> float:
    """Return the configured horizon for a target heading."""

    target = float(target_heading_deg)
    if target in TARGET_HORIZONS_S:
        return TARGET_HORIZONS_S[target]
    if target <= 30.0:
        return 0.85
    if target <= 90.0:
        return 1.20
    return 1.60


def _validate_config(config: AggressiveReversalOcpConfig) -> None:
    if not np.isfinite(config.dt_s) or config.dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive.")
    if not np.isfinite(config.t_final_s) or config.t_final_s <= 0.0:
        raise ValueError("t_final_s must be finite and positive.")
    ratio = config.t_final_s / config.dt_s
    if not np.isclose(ratio, round(ratio), rtol=1e-12, atol=1e-9):
        raise ValueError("t_final_s must be an integer multiple of dt_s.")
    if config.direction_sign not in (-1, 1):
        raise ValueError("direction_sign must be -1 or +1.")
    if config.wind_mode != "none":
        raise ValueError("aggressive reversal search currently supports only W0/no wind.")
    if config.latency_case != "none":
        raise ValueError("aggressive reversal search currently supports latency_case='none'.")
    if config.ocp_node_count < 1:
        raise ValueError("ocp_node_count must be positive.")


def target_config(
    target_heading_deg: float,
    base: AggressiveReversalOcpConfig | None = None,
) -> AggressiveReversalOcpConfig:
    """Return a config with target-specific horizon and target heading."""

    cfg = AggressiveReversalOcpConfig() if base is None else base
    return AggressiveReversalOcpConfig(
        dt_s=cfg.dt_s,
        t_final_s=target_horizon_s(float(target_heading_deg)),
        target_heading_deg=float(target_heading_deg),
        direction_sign=cfg.direction_sign,
        speed_m_s=cfg.speed_m_s,
        altitude_m=cfg.altitude_m,
        wind_mode=cfg.wind_mode,
        latency_case=cfg.latency_case,
        actuator_tau_s=cfg.actuator_tau_s,
        seed=cfg.seed,
        max_ipopt_iter=cfg.max_ipopt_iter,
        use_nonlinear_ocp=cfg.use_nonlinear_ocp,
        use_phase_seed_fallback=cfg.use_phase_seed_fallback,
        phase_search_scales=cfg.phase_search_scales,
        ocp_node_count=cfg.ocp_node_count,
        ocp_max_cpu_time_s=cfg.ocp_max_cpu_time_s,
        checkpoint_dir=cfg.checkpoint_dir,
        candidate_log_dir=cfg.candidate_log_dir,
        run_id=cfg.run_id,
        previous_solution_path=cfg.previous_solution_path,
    )


def _family_phase_labels(family_name: str) -> tuple[str, ...]:
    if family_name == "dive_perch_redirect_30":
        return (
            "entry",
            "pre_dive_accelerate",
            "pitch_brake",
            "slow_redirect",
            "heading_capture",
            "unload_descend",
            "exit_glide",
        )
    return (
        "entry",
        "pitch_brake",
        "slow_redirect",
        "heading_capture",
        "unload_descend",
        "exit_glide",
    )


def _family_phase_edges(family_name: str, t_final_s: float) -> np.ndarray:
    fractions = {
        "short_perch_yaw_redirect": (0.08, 0.26, 0.56, 0.70, 0.86),
        "long_perch_slow_redirect": (0.08, 0.34, 0.64, 0.76, 0.90),
        "roll_dominant_banked_redirect": (0.08, 0.28, 0.58, 0.72, 0.88),
        "split_pulse_redirect": (0.08, 0.26, 0.50, 0.68, 0.86),
        "early_unload_descend_capture": (0.08, 0.24, 0.46, 0.62, 0.82),
        "dive_perch_redirect_30": (0.08, 0.22, 0.38, 0.60, 0.72, 0.88),
        "reduced_perch_redirect_30": (0.10, 0.28, 0.50, 0.66, 0.84),
        "bank_yaw_redirect_30": (0.10, 0.24, 0.52, 0.68, 0.86),
        "early_unload_recovery_30": (0.08, 0.24, 0.42, 0.56, 0.78),
    }
    if family_name not in fractions:
        raise ValueError(f"unknown seed family: {family_name}.")
    return np.array((0.0, *fractions[family_name], 1.0), dtype=float) * float(t_final_s)


def phase_labels_for_family(
    family_name: str,
    time_s: np.ndarray,
    t_final_s: float,
) -> tuple[str, ...]:
    """Return a phase label for each time sample."""

    edges = _family_phase_edges(family_name, t_final_s)
    phase_names = _family_phase_labels(family_name)
    labels: list[str] = []
    for time_value in np.asarray(time_s, dtype=float):
        index = int(np.searchsorted(edges, time_value, side="right") - 1)
        index = int(np.clip(index, 0, len(phase_names) - 1))
        labels.append(phase_names[index])
    return tuple(labels)


def _family_amplitudes(family_name: str) -> dict[str, tuple[float, float, float]]:
    return {
        "short_perch_yaw_redirect": {
            "entry": (0.0, 0.0, 0.0),
            "pitch_brake": (0.10, 0.78, 0.10),
            "slow_redirect": (0.38, 0.55, 0.78),
            "heading_capture": (-0.22, -0.15, -0.25),
            "unload_descend": (0.0, -0.55, 0.0),
            "exit_glide": (0.0, 0.0, 0.0),
        },
        "long_perch_slow_redirect": {
            "entry": (0.0, 0.0, 0.0),
            "pitch_brake": (0.12, 0.92, 0.12),
            "slow_redirect": (0.48, 0.72, 0.95),
            "heading_capture": (-0.26, -0.12, -0.32),
            "unload_descend": (0.0, -0.48, 0.0),
            "exit_glide": (0.0, 0.0, 0.0),
        },
        "roll_dominant_banked_redirect": {
            "entry": (0.0, 0.0, 0.0),
            "pitch_brake": (0.25, 0.70, 0.12),
            "slow_redirect": (0.82, 0.45, 0.42),
            "heading_capture": (-0.36, -0.12, -0.20),
            "unload_descend": (-0.05, -0.45, 0.0),
            "exit_glide": (0.0, 0.0, 0.0),
        },
        "split_pulse_redirect": {
            "entry": (0.0, 0.0, 0.0),
            "pitch_brake": (0.14, 0.88, 0.20),
            "slow_redirect": (0.58, 0.62, 0.92),
            "heading_capture": (-0.48, -0.20, -0.62),
            "unload_descend": (0.35, -0.50, 0.60),
            "exit_glide": (0.0, 0.0, 0.0),
        },
        "early_unload_descend_capture": {
            "entry": (0.0, 0.0, 0.0),
            "pitch_brake": (0.10, 0.72, 0.10),
            "slow_redirect": (0.35, 0.38, 0.70),
            "heading_capture": (-0.22, -0.25, -0.30),
            "unload_descend": (0.0, -0.72, 0.0),
            "exit_glide": (0.0, -0.05, 0.0),
        },
        "dive_perch_redirect_30": {
            "entry": (0.0, 0.0, 0.0),
            "pre_dive_accelerate": (0.0, -0.34, 0.0),
            "pitch_brake": (0.08, 0.58, 0.08),
            "slow_redirect": (0.30, 0.24, 0.55),
            "heading_capture": (-0.18, -0.22, -0.24),
            "unload_descend": (0.0, -0.62, 0.0),
            "exit_glide": (0.0, -0.08, 0.0),
        },
        "reduced_perch_redirect_30": {
            "entry": (0.0, 0.0, 0.0),
            "pitch_brake": (0.05, 0.46, 0.06),
            "slow_redirect": (0.26, 0.22, 0.48),
            "heading_capture": (-0.16, -0.18, -0.24),
            "unload_descend": (0.0, -0.54, 0.0),
            "exit_glide": (0.0, -0.05, 0.0),
        },
        "bank_yaw_redirect_30": {
            "entry": (0.0, 0.0, 0.0),
            "pitch_brake": (0.18, 0.32, 0.06),
            "slow_redirect": (0.58, 0.12, 0.38),
            "heading_capture": (-0.34, -0.10, -0.22),
            "unload_descend": (-0.08, -0.32, 0.0),
            "exit_glide": (0.0, -0.03, 0.0),
        },
        "early_unload_recovery_30": {
            "entry": (0.0, 0.0, 0.0),
            "pitch_brake": (0.08, 0.62, 0.08),
            "slow_redirect": (0.28, 0.24, 0.56),
            "heading_capture": (-0.22, -0.28, -0.32),
            "unload_descend": (0.0, -0.76, 0.0),
            "exit_glide": (0.0, -0.12, 0.0),
        },
    }[family_name]


def seed_family_inventory() -> tuple[str, ...]:
    """Return the fixed guided manoeuvre-family inventory."""

    return SEED_FAMILIES


def seed_family_inventory_for_target(target_heading_deg: float) -> tuple[str, ...]:
    """Return the seed families used for a target-specific search."""

    if np.isclose(float(target_heading_deg), 30.0, atol=1e-9):
        return SEED_FAMILIES
    return BASE_SEED_FAMILIES


def phase_seed_command_profile(
    config: AggressiveReversalOcpConfig,
    time_s: np.ndarray,
    family_name: str = "short_perch_yaw_redirect",
    amplitude_scale: float = 1.0,
) -> np.ndarray:
    """Return a deterministic phase-structured normalised command seed."""

    _validate_config(config)
    time = np.asarray(time_s, dtype=float).reshape(-1)
    labels = phase_labels_for_family(family_name, time, config.t_final_s)
    amplitudes = _family_amplitudes(family_name)
    commands = np.zeros((time.size, 3), dtype=float)
    for index, label in enumerate(labels):
        aileron, elevator, rudder = amplitudes[label]
        commands[index, 0] = config.direction_sign * amplitude_scale * aileron
        commands[index, 1] = amplitude_scale * elevator
        commands[index, 2] = config.direction_sign * amplitude_scale * rudder
    return np.clip(commands, -1.0, 1.0)


def next_family_for_failure(failure_label: str, current_family: str) -> tuple[str, str]:
    """Return deterministic next family and reason for a failure label."""

    is_thirty_family = current_family in THIRTY_DEG_SEED_FAMILIES
    if failure_label == "under_turning":
        if is_thirty_family and current_family != "bank_yaw_redirect_30":
            return "bank_yaw_redirect_30", "under_turning_bank_yaw_redirect"
        if is_thirty_family:
            return "dive_perch_redirect_30", "under_turning_dive_perch_redirect"
        if current_family != "long_perch_slow_redirect":
            return "long_perch_slow_redirect", "under_turning_longer_perch"
        return "split_pulse_redirect", "under_turning_split_redirect"
    if failure_label == "speed_low":
        if is_thirty_family:
            return "early_unload_recovery_30", "speed_low_earlier_unload"
        return "early_unload_descend_capture", "speed_low_earlier_unload"
    if failure_label in {"true_safety_violation", "wall_violation", "floor_violation", "ceiling_violation"}:
        return current_family, "safety_boundary_shorten_redirect_or_raise_penalty"
    if failure_label == "alpha_boundary":
        return current_family, "alpha_boundary_soften_pitch_brake"
    if failure_label == "actuator_saturation_limited":
        return current_family, "saturation_smooth_and_lengthen_phase"
    if failure_label == "terminal_recovery_limited":
        if is_thirty_family:
            return "early_unload_recovery_30", "terminal_recovery_extend_exit_glide"
        return "early_unload_descend_capture", "terminal_recovery_extend_exit_glide"
    if failure_label == "solver_failure":
        return current_family, "solver_failure_retry_best_finite_phase_search"
    if failure_label == "model_boundary_only":
        return current_family, "model_boundary_stop_family_escalation"
    return current_family, "no_family_change"


# =============================================================================
# 3) Replay and Metrics
# =============================================================================
def build_aggressive_initial_state(
    config: AggressiveReversalOcpConfig,
    aircraft: object | None = None,
) -> np.ndarray:
    """Return a safe trim-like initial state inside the true safety volume."""

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
        raise ValueError("cannot build aggressive reversal initial state from nonconverged trim.")
    state = as_state_vector(trim.x_trim).copy()
    state[0:3] = [1.25, 2.2, float(config.altitude_m)]
    return state


def _time_grid(config: AggressiveReversalOcpConfig) -> np.ndarray:
    step_count = int(round(config.t_final_s / config.dt_s))
    return np.arange(step_count + 1, dtype=float) * float(config.dt_s)


def _speed_alpha_beta(x_log: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    velocity_b = np.asarray(x_log, dtype=float)[:, 6:9]
    speed = np.linalg.norm(velocity_b, axis=1)
    alpha = np.arctan2(velocity_b[:, 2], velocity_b[:, 0])
    beta = np.zeros_like(speed)
    valid = speed > 1e-9
    beta[valid] = np.arcsin(np.clip(velocity_b[valid, 1] / speed[valid], -1.0, 1.0))
    return speed, alpha, beta


def speed_m_s(x_log: np.ndarray) -> np.ndarray:
    """Return body-speed magnitude for each state sample."""

    return np.linalg.norm(np.asarray(x_log, dtype=float)[:, 6:9], axis=1)


def specific_energy_height_m(x_log: np.ndarray, g_m_s2: float = G_M_S2) -> np.ndarray:
    """Return specific energy height in public z-up world coordinates."""

    state = np.asarray(x_log, dtype=float)
    speed = speed_m_s(state)
    return state[:, STATE_INDEX["z_w"]] + speed**2 / (2.0 * float(g_m_s2))


def descent_required_to_speed_m(
    current_speed_m_s: float,
    target_speed_m_s: float,
    g_m_s2: float = G_M_S2,
) -> float:
    """Return ideal z-up descent required to accelerate to target speed."""

    required = (float(target_speed_m_s) ** 2 - float(current_speed_m_s) ** 2) / (
        2.0 * float(g_m_s2)
    )
    return float(max(0.0, required))


def recoverable_speed_from_energy_m_s(
    energy_height_m: float,
    floor_z_m: float,
    g_m_s2: float = G_M_S2,
) -> float:
    """Return ideal speed reachable by descending to a floor height."""

    usable_height_m = max(0.0, float(energy_height_m) - float(floor_z_m))
    return float(np.sqrt(2.0 * float(g_m_s2) * usable_height_m))


def _phase_indices(phase: tuple[str, ...], names: tuple[str, ...]) -> np.ndarray:
    labels = np.asarray(phase)
    return np.where(np.isin(labels, list(names)))[0]


def _phase_start_stop(phase: tuple[str, ...], names: tuple[str, ...]) -> tuple[int, int] | None:
    indices = _phase_indices(phase, names)
    if indices.size == 0:
        return None
    return int(indices[0]), int(indices[-1])


def phase_audit_timeseries(
    config: AggressiveReversalOcpConfig,
    time_s: np.ndarray,
    x_log: np.ndarray,
    phase: tuple[str, ...],
) -> list[dict[str, object]]:
    """Return per-sample energy and shape evidence for CSV audit output."""

    speed = speed_m_s(x_log)
    alpha = np.arctan2(x_log[:, STATE_INDEX["w"]], x_log[:, STATE_INDEX["u"]])
    energy = specific_energy_height_m(x_log)
    psi = np.unwrap(x_log[:, STATE_INDEX["psi"]])
    heading = np.rad2deg(int(config.direction_sign) * (psi - psi[0]))
    return [
        {
            "time_s": float(time_s[index]),
            "speed_m_s": float(speed[index]),
            "z_w": float(x_log[index, STATE_INDEX["z_w"]]),
            "specific_energy_height_m": float(energy[index]),
            "alpha_deg": float(np.rad2deg(alpha[index])),
            "theta_deg": float(np.rad2deg(x_log[index, STATE_INDEX["theta"]])),
            "heading_change_deg": float(heading[index]),
            "phase": phase[index],
        }
        for index in range(len(time_s))
    ]


def energy_audit_for_trajectory(
    x_log: np.ndarray,
    phase: tuple[str, ...] | None = None,
) -> dict[str, float]:
    """Return compact energy-budget metrics for one replay."""

    state = np.asarray(x_log, dtype=float)
    speed = speed_m_s(state)
    energy = specific_energy_height_m(state)
    terminal_speed = float(speed[-1])
    terminal_energy = float(energy[-1])
    true_floor = float(TRUE_SAFE_BOUNDS.z_w_m[0])
    audit = {
        "specific_energy_initial_m": float(energy[0]),
        "specific_energy_terminal_m": terminal_energy,
        "specific_energy_min_m": float(np.nanmin(energy)),
        "specific_energy_max_m": float(np.nanmax(energy)),
        "specific_energy_lost_m": float(energy[0] - terminal_energy),
        "speed_loss_m_s": float(speed[0] - terminal_speed),
        "max_altitude_gain_m": float(np.nanmax(state[:, STATE_INDEX["z_w"]]) - state[0, STATE_INDEX["z_w"]]),
        "max_altitude_loss_m": float(state[0, STATE_INDEX["z_w"]] - np.nanmin(state[:, STATE_INDEX["z_w"]])),
        "ideal_descent_to_4ms_m": descent_required_to_speed_m(terminal_speed, 4.0),
        "ideal_descent_to_5ms_m": descent_required_to_speed_m(terminal_speed, 5.0),
        "available_descent_to_true_floor_m": float(max(0.0, state[-1, STATE_INDEX["z_w"]] - true_floor)),
        "available_descent_to_z08_m": float(max(0.0, state[-1, STATE_INDEX["z_w"]] - 0.8)),
        "available_descent_to_z12_m": float(max(0.0, state[-1, STATE_INDEX["z_w"]] - 1.2)),
        "available_descent_to_z15_m": float(max(0.0, state[-1, STATE_INDEX["z_w"]] - 1.5)),
        "recoverable_speed_by_floor_m_s": recoverable_speed_from_energy_m_s(terminal_energy, true_floor),
        "recoverable_speed_by_z08_m_s": recoverable_speed_from_energy_m_s(terminal_energy, 0.8),
        "recoverable_speed_by_z12_m_s": recoverable_speed_from_energy_m_s(terminal_energy, 1.2),
        "recoverable_speed_by_z15_m_s": recoverable_speed_from_energy_m_s(terminal_energy, 1.5),
    }
    if phase is not None:
        pitch = _phase_start_stop(phase, ("pitch_brake",))
        unload = _phase_start_stop(phase, ("unload_descend", "exit_glide"))
        if pitch is not None:
            start, stop = pitch
            audit["pitch_brake_energy_drop_m"] = float(energy[start] - energy[stop])
            audit["pitch_brake_altitude_gain_m"] = float(
                state[stop, STATE_INDEX["z_w"]] - state[start, STATE_INDEX["z_w"]]
            )
        else:
            audit["pitch_brake_energy_drop_m"] = float("nan")
            audit["pitch_brake_altitude_gain_m"] = float("nan")
        if unload is not None:
            start, stop = unload
            audit["unload_descent_m"] = float(
                state[start, STATE_INDEX["z_w"]] - state[stop, STATE_INDEX["z_w"]]
            )
            audit["unload_speed_gain_m_s"] = float(speed[stop] - speed[start])
        else:
            audit["unload_descent_m"] = float("nan")
            audit["unload_speed_gain_m_s"] = float("nan")
    return audit


def phase_shape_audit(
    config: AggressiveReversalOcpConfig,
    x_log: np.ndarray,
    phase: tuple[str, ...],
    terminal_recoverable: bool,
) -> dict[str, object]:
    """Classify whether a replay is a true dive/perch/turn/recover shape."""

    state = np.asarray(x_log, dtype=float)
    speed = speed_m_s(state)
    energy = energy_audit_for_trajectory(state, phase)
    actual_heading = unwrapped_signed_heading_change_deg(
        state[:, STATE_INDEX["psi"]],
        config.direction_sign,
    )
    pitch = _phase_start_stop(phase, ("pitch_brake",))
    redirect = _phase_start_stop(phase, ("slow_redirect",))
    pre_pitch_stop = pitch[0] if pitch is not None else max(1, state.shape[0] // 5)
    pre_speed_gain = float(speed[pre_pitch_stop] - speed[0])
    pre_descent = float(state[0, STATE_INDEX["z_w"]] - np.nanmin(state[: pre_pitch_stop + 1, STATE_INDEX["z_w"]]))
    initial_dive_or_speed_build = bool(
        pre_speed_gain >= 0.10 or (pre_descent >= 0.05 and pre_speed_gain >= -0.05)
    )
    pitch_energy_drop = float(energy["pitch_brake_energy_drop_m"])
    pitch_altitude_gain = float(energy["pitch_brake_altitude_gain_m"])
    perch_up_energy_trade = bool(
        pitch_altitude_gain >= 0.05 and 0.30 <= pitch_energy_drop <= 2.50
    )
    if redirect is None:
        redirect_heading = 0.0
        redirect_min_speed = float("nan")
    else:
        start, stop = redirect
        psi = np.unwrap(state[:, STATE_INDEX["psi"]])
        redirect_heading = float(
            np.rad2deg(config.direction_sign * (psi[stop] - psi[start]))
        )
        redirect_min_speed = float(np.nanmin(speed[start : stop + 1]))
    slow_redirect_heading_gain = bool(
        redirect_heading >= 0.40 * float(config.target_heading_deg)
        and redirect_min_speed >= 3.0
    )
    unload_descent = bool(
        float(energy["unload_descent_m"]) >= UNLOAD_EXIT_DESCENT_REQUIRED_M
    )
    unload_speed_gain = bool(
        float(energy["unload_speed_gain_m_s"]) >= UNLOAD_EXIT_SPEED_GAIN_REQUIRED_M_S
    )
    exit_glide_or_recovery_compatible = bool(terminal_recoverable)
    heading_success = bool(actual_heading >= STRICT_30_HEADING_DEG)
    if (
        heading_success
        and initial_dive_or_speed_build
        and perch_up_energy_trade
        and slow_redirect_heading_gain
        and unload_descent
        and unload_speed_gain
        and exit_glide_or_recovery_compatible
    ):
        shape_class = "dive_perch_turn_recover"
    elif heading_success and (not unload_descent or not unload_speed_gain):
        shape_class = "speed_collapse_pitch_redirect"
    else:
        shape_class = "incomplete_aggressive_reversal"
    return {
        "initial_dive_or_speed_build": initial_dive_or_speed_build,
        "pre_pitch_speed_gain_m_s": pre_speed_gain,
        "pre_pitch_descent_m": pre_descent,
        "perch_up_energy_trade": perch_up_energy_trade,
        "slow_redirect_heading_gain": slow_redirect_heading_gain,
        "redirect_heading_gain_deg": redirect_heading,
        "redirect_min_speed_m_s": redirect_min_speed,
        "unload_descent": unload_descent,
        "unload_speed_gain": unload_speed_gain,
        "unload_exit_descent_m": float(energy["unload_descent_m"]),
        "unload_exit_speed_gain_m_s": float(energy["unload_speed_gain_m_s"]),
        "exit_glide_or_recovery_compatible": exit_glide_or_recovery_compatible,
        "manoeuvre_shape_class": shape_class,
    }


def unwrapped_signed_heading_change_deg(psi_rad: np.ndarray, direction_sign: int) -> float:
    """Return signed target-direction heading progress in degrees."""

    psi = np.unwrap(np.asarray(psi_rad, dtype=float).reshape(-1))
    if psi.size == 0:
        return float("nan")
    return float(np.rad2deg(int(direction_sign) * (psi[-1] - psi[0])))


def _margin_rows(x_log: np.ndarray) -> list[dict[str, float]]:
    return [
        position_margin_m(position, TRUE_SAFE_BOUNDS)
        for position in np.asarray(x_log, dtype=float)[:, 0:3]
        if np.all(np.isfinite(position))
    ]


def _min_metric(rows: list[dict[str, float]], key: str) -> float:
    if not rows:
        return float("nan")
    return float(np.min([row[key] for row in rows]))


def _saturation_metrics(u_requested: np.ndarray, u_applied: np.ndarray, dt_s: float) -> tuple[float, float]:
    clipped = np.any(np.abs(np.asarray(u_requested) - np.asarray(u_applied)) > 1e-12, axis=1)
    if clipped.size <= 1:
        return 0.0, 0.0
    count = int(np.count_nonzero(clipped[:-1]))
    return float(count / clipped[:-1].size), float(count * dt_s)


def replay_aggressive_candidate(
    x0: np.ndarray,
    u_norm: np.ndarray,
    time_s: np.ndarray,
    config: AggressiveReversalOcpConfig,
    aircraft: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replay candidate commands using current plant and command bridge."""

    _validate_config(config)
    time = np.asarray(time_s, dtype=float).reshape(-1)
    requested = np.asarray(u_norm, dtype=float)
    if requested.shape != (time.size, 3):
        raise ValueError("u_norm must have shape (N, 3) matching time_s.")
    x_log = np.empty((time.size, STATE_SIZE), dtype=float)
    applied = np.empty((time.size, 3), dtype=float)
    command_rad = np.empty((time.size, 3), dtype=float)
    x_log[0] = as_state_vector(x0)
    for index in range(time.size):
        applied[index] = clip_normalised_command(requested[index])
        command_rad[index] = normalised_command_to_surface_rad(applied[index])
        if index == time.size - 1:
            break
        dt = float(time[index + 1] - time[index])
        x_log[index + 1] = rk4_step(
            x_log[index],
            command_rad[index],
            dt,
            aircraft,
            None,
            config.wind_mode,
            config.actuator_tau_s,
        )
        if not np.all(np.isfinite(x_log[index + 1])):
            x_log[index + 2 :] = np.nan
            break
    return x_log, applied, command_rad


def _energy_exploitation(x_log: np.ndarray, phase: tuple[str, ...]) -> bool:
    labels = np.asarray(phase)
    mask = labels == "pitch_brake"
    if np.count_nonzero(mask) < 2:
        return False
    indices = np.where(mask)[0]
    start = indices[0]
    stop = indices[-1]
    speed = np.linalg.norm(x_log[:, 6:9], axis=1)
    altitude_gain = float(x_log[stop, STATE_INDEX["z_w"]] - x_log[start, STATE_INDEX["z_w"]])
    speed_gain = float(speed[stop] - speed[start])
    return bool(altitude_gain > 0.02 and speed_gain > 0.05)


def _diagnostic_recovery_extension_success(
    x_terminal: np.ndarray,
    config: AggressiveReversalOcpConfig,
    aircraft: object | None,
) -> bool:
    """Try a documented unload command after a 30 deg segment."""

    if aircraft is None:
        return False
    try:
        state0 = as_state_vector(x_terminal)
    except ValueError:
        return False
    extension_config = AggressiveReversalOcpConfig(
        dt_s=config.dt_s,
        t_final_s=0.60,
        target_heading_deg=config.target_heading_deg,
        direction_sign=config.direction_sign,
        speed_m_s=config.speed_m_s,
        altitude_m=config.altitude_m,
        wind_mode=config.wind_mode,
        latency_case=config.latency_case,
        actuator_tau_s=config.actuator_tau_s,
        seed=config.seed,
        max_ipopt_iter=config.max_ipopt_iter,
        use_nonlinear_ocp=False,
        use_phase_seed_fallback=True,
        phase_search_scales=config.phase_search_scales,
        ocp_node_count=config.ocp_node_count,
        ocp_max_cpu_time_s=config.ocp_max_cpu_time_s,
        checkpoint_dir=None,
        candidate_log_dir=None,
        run_id=config.run_id,
        previous_solution_path=None,
    )
    time_s = _time_grid(extension_config)
    requested = np.zeros((time_s.size, 3), dtype=float)
    requested[:, 1] = -0.55
    x_log, _, _ = replay_aggressive_candidate(
        state0,
        requested,
        time_s,
        extension_config,
        aircraft,
    )
    finite = bool(np.all(np.isfinite(x_log)))
    if not finite:
        return False
    true_safe = all(inside_bounds(position, TRUE_SAFE_BOUNDS) for position in x_log[:, 0:3])
    return bool(true_safe and speed_m_s(x_log)[-1] >= STRICT_TERMINAL_SPEED_M_S)


def metrics_for_candidate(
    config: AggressiveReversalOcpConfig,
    time_s: np.ndarray,
    x_log: np.ndarray,
    u_requested: np.ndarray,
    u_applied: np.ndarray,
    phase: tuple[str, ...],
    aircraft: object | None = None,
) -> dict[str, object]:
    """Return scalar metrics and classification for one candidate."""

    finite = bool(np.all(np.isfinite(x_log)))
    speed, alpha, beta = _speed_alpha_beta(x_log)
    margins = _margin_rows(x_log)
    saturation_fraction, saturation_time_s = _saturation_metrics(
        u_requested,
        u_applied,
        config.dt_s,
    )
    actual_heading = unwrapped_signed_heading_change_deg(
        x_log[:, STATE_INDEX["psi"]],
        config.direction_sign,
    )
    terminal_recoverable = bool(finite and terminal_aggressive_recoverable_proxy(x_log[-1]))
    true_safe = bool(
        finite
        and all(inside_bounds(position, TRUE_SAFE_BOUNDS) for position in x_log[:, 0:3])
    )
    replay_finite = bool(finite)
    energy_audit = energy_audit_for_trajectory(x_log, phase)
    shape_audit = phase_shape_audit(config, x_log, phase, terminal_recoverable) if finite else {
        "initial_dive_or_speed_build": False,
        "pre_pitch_speed_gain_m_s": np.nan,
        "pre_pitch_descent_m": np.nan,
        "perch_up_energy_trade": False,
        "slow_redirect_heading_gain": False,
        "redirect_heading_gain_deg": np.nan,
        "redirect_min_speed_m_s": np.nan,
        "unload_descent": False,
        "unload_speed_gain": False,
        "unload_exit_descent_m": np.nan,
        "unload_exit_speed_gain_m_s": np.nan,
        "exit_glide_or_recovery_compatible": False,
        "manoeuvre_shape_class": "nonfinite",
    }
    is_30deg = bool(np.isclose(float(config.target_heading_deg), 30.0, atol=1e-9))
    heading_threshold = STRICT_30_HEADING_DEG if is_30deg else 0.80 * float(config.target_heading_deg)
    heading_success = bool(actual_heading >= heading_threshold)
    energy_success_strict = bool(float(speed[-1]) >= STRICT_TERMINAL_SPEED_M_S)
    energy_success_relaxed = bool(
        float(speed[-1]) >= RELAXED_TERMINAL_SPEED_M_S
        and float(energy_audit["recoverable_speed_by_z08_m_s"]) >= STRICT_TERMINAL_SPEED_M_S
    )
    recovery_extension_success = bool(
        is_30deg
        and heading_success
        and true_safe
        and _diagnostic_recovery_extension_success(x_log[-1], config, aircraft)
    )
    combined_30deg_recoverable = bool(
        terminal_recoverable or (energy_success_relaxed and recovery_extension_success)
    )
    true_shape = bool(shape_audit["manoeuvre_shape_class"] == "dive_perch_turn_recover")
    primitive_success = bool(
        replay_finite
        and true_safe
        and heading_success
        and energy_success_strict
        and terminal_recoverable
        and (true_shape if is_30deg else True)
    )
    relaxed_recovery_required = bool(
        is_30deg
        and heading_success
        and replay_finite
        and true_safe
        and not primitive_success
        and float(speed[-1]) < STRICT_TERMINAL_SPEED_M_S
        and energy_success_relaxed
        and recovery_extension_success
    )
    energy_exploitation = bool(finite and _energy_exploitation(x_log, phase))
    if primitive_success:
        failure_label = "success"
        notes = "target_achieved_and_recoverable"
    elif not replay_finite:
        failure_label = "nonfinite_state"
        notes = "replay_contains_nonfinite_state"
    elif not true_safe:
        failure_label = "true_safety_violation"
        notes = "trajectory_exits_true_safety_volume"
    elif energy_exploitation:
        failure_label = "model_boundary_only"
        notes = "w0_energy_exploitation"
    elif actual_heading < heading_threshold:
        failure_label = "under_turning"
        notes = "heading_change_below_threshold"
    elif is_30deg and shape_audit["manoeuvre_shape_class"] == "speed_collapse_pitch_redirect":
        failure_label = "speed_low"
        notes = "speed_collapse_pitch_redirect"
    elif float(speed[-1]) < STRICT_TERMINAL_SPEED_M_S:
        failure_label = "speed_low"
        notes = "terminal_speed_below_recoverability_threshold"
    elif np.nanmax(np.abs(np.rad2deg(beta))) > 35.0:
        failure_label = "beta_boundary"
        notes = "beta_exposure_exceeds_boundary"
    elif np.nanmax(np.abs(np.rad2deg(alpha))) > 55.0:
        failure_label = "alpha_boundary"
        notes = "alpha_exposure_exceeds_surrogate_boundary"
    elif np.nanmax(np.linalg.norm(x_log[:, 9:12], axis=1)) > 4.5:
        failure_label = "rate_boundary"
        notes = "body_rate_exceeds_boundary"
    elif saturation_fraction > 0.75:
        failure_label = "actuator_saturation_limited"
        notes = "commands_excessively_clipped"
    elif not terminal_recoverable:
        failure_label = "terminal_recovery_limited"
        notes = "terminal_state_not_glide_or_recovery_compatible"
    else:
        failure_label = "model_boundary_only"
        notes = "simulation_boundary_candidate"
    if failure_label not in FAILURE_LABELS:
        raise ValueError(f"unknown failure label: {failure_label}")
    if not true_safe:
        active_tradeoff = "safety_volume_limited"
    elif is_30deg and heading_success and float(speed[-1]) < RELAXED_TERMINAL_SPEED_M_S:
        active_tradeoff = (
            "high_alpha_drag_limited"
            if np.nanmax(np.abs(np.rad2deg(alpha))) > 70.0
            else "energy_budget_limited"
        )
    elif is_30deg and heading_success and not combined_30deg_recoverable:
        active_tradeoff = "recovery_handoff_limited"
    elif heading_success and not energy_success_strict:
        active_tradeoff = "energy_budget_limited"
    elif not heading_success and energy_success_strict:
        active_tradeoff = "turn_authority_limited"
    elif failure_label == "solver_failure":
        active_tradeoff = "solver_formulation_limited"
    else:
        active_tradeoff = limiting_mechanism(failure_label)
    return {
        "target_heading_deg": float(config.target_heading_deg),
        "actual_heading_change_deg": float(actual_heading),
        "heading_error_deg": float(config.target_heading_deg - actual_heading),
        "forward_travel_m": float(abs(x_log[-1, 0] - x_log[0, 0])),
        "turn_volume_proxy_m2": float(abs(x_log[-1, 0] - x_log[0, 0]) * abs(x_log[-1, 1] - x_log[0, 1])),
        "height_change_m": float(x_log[-1, 2] - x_log[0, 2]),
        "max_altitude_gain_m": float(np.nanmax(x_log[:, 2]) - x_log[0, 2]),
        "speed_min_m_s": float(np.nanmin(speed)),
        "terminal_speed_m_s": float(speed[-1]),
        "max_alpha_deg": float(np.nanmax(np.abs(np.rad2deg(alpha)))),
        "max_beta_deg": float(np.nanmax(np.abs(np.rad2deg(beta)))),
        "max_bank_deg": float(np.nanmax(np.abs(np.rad2deg(x_log[:, STATE_INDEX["phi"]])))),
        "max_pitch_deg": float(np.nanmax(np.abs(np.rad2deg(x_log[:, STATE_INDEX["theta"]])))),
        "max_rate_rad_s": float(np.nanmax(np.linalg.norm(x_log[:, 9:12], axis=1))),
        "min_true_wall_margin_m": _min_metric(margins, "min_wall_margin_m"),
        "min_floor_margin_m": _min_metric(margins, "floor_margin_m"),
        "min_ceiling_margin_m": _min_metric(margins, "ceiling_margin_m"),
        "saturation_fraction": saturation_fraction,
        "saturation_time_s": saturation_time_s,
        "finite_state_success": replay_finite,
        "rollout_success": bool(replay_finite and true_safe),
        "source_trajectory_success": replay_finite,
        "terminal_recoverable_proxy": terminal_recoverable,
        "recoverable": terminal_recoverable,
        "success": primitive_success,
        "primitive_success": primitive_success,
        "failure_label": failure_label,
        "notes": notes,
        "energy_exploitation": energy_exploitation,
        "heading_success": heading_success,
        "energy_success_strict": energy_success_strict,
        "energy_success_relaxed": energy_success_relaxed,
        "strict_30deg_primitive_success": primitive_success if is_30deg else False,
        "relaxed_recovery_required": relaxed_recovery_required,
        "recovery_extension_success": recovery_extension_success,
        "combined_30deg_recoverable": combined_30deg_recoverable,
        "updraft_assisted_boundary_evidence": False,
        "active_tradeoff": active_tradeoff,
        **energy_audit,
        **shape_audit,
    }


def _result_from_replay(
    config: AggressiveReversalOcpConfig,
    family_name: str,
    method: str,
    attempt_index: int,
    x0: np.ndarray,
    time_s: np.ndarray,
    u_requested: np.ndarray,
    aircraft: object,
) -> CandidateResult:
    x_log, u_applied, command_rad = replay_aggressive_candidate(
        x0,
        u_requested,
        time_s,
        config,
        aircraft,
    )
    phase = phase_labels_for_family(family_name, time_s, config.t_final_s)
    metrics = metrics_for_candidate(
        config,
        time_s,
        x_log,
        u_requested,
        u_applied,
        phase,
        aircraft,
    )
    result = AggressiveReversalOcpResult(
        target_heading_deg=float(config.target_heading_deg),
        direction_sign=int(config.direction_sign),
        success=bool(metrics["success"]),
        failure_label=str(metrics["failure_label"]),
        time_s=time_s,
        x_ref=x_log,
        u_ff_norm=u_requested,
        u_norm_applied=u_applied,
        delta_cmd_rad=command_rad,
        phase=phase,
        metrics=metrics,
        notes=str(metrics["notes"]),
    )
    return CandidateResult(
        method=method,
        family_name=family_name,
        attempt_index=attempt_index,
        result=result,
        direct_ocp_attempted=False,
        limiting_mechanism=limiting_mechanism(str(metrics["failure_label"])),
    )


# =============================================================================
# 4) Guided Search and Direct OCP Attempts
# =============================================================================
def _score_candidate(candidate: CandidateResult) -> tuple[int, int, int, int, int, float, float, float, float]:
    metrics = candidate.result.metrics
    shape_class = str(metrics.get("manoeuvre_shape_class", ""))
    return (
        int(bool(metrics["success"])),
        int(bool(metrics.get("relaxed_recovery_required", False))),
        int(bool(metrics.get("combined_30deg_recoverable", False))),
        int(bool(metrics["recoverable"])),
        int(shape_class == "dive_perch_turn_recover"),
        float(metrics["actual_heading_change_deg"]),
        float(metrics.get("specific_energy_terminal_m", -np.inf)),
        float(metrics["terminal_speed_m_s"]),
        -float(metrics["saturation_fraction"]),
    )


def limiting_mechanism(failure_label: str) -> str:
    if failure_label == "success":
        return "none"
    if failure_label == "solver_failure":
        return "optimiser_convergence"
    if failure_label in {"true_safety_violation", "wall_violation", "floor_violation", "ceiling_violation"}:
        return "physical_safety_boundary"
    if failure_label == "model_boundary_only":
        return "model_boundary_behaviour"
    if failure_label in {"under_turning", "terminal_recovery_limited"}:
        return "insufficient_manoeuvre_seed"
    return "physical_boundary"


def _symbolic_command_to_rad(command_norm: ca.MX) -> ca.MX:
    from latency import AGGREGATE_LIMITS

    values = []
    for index, name in enumerate(("delta_a", "delta_e", "delta_r")):
        limit = AGGREGATE_LIMITS[name]
        positive = np.deg2rad(float(limit.positive_deg))
        negative = np.deg2rad(float(limit.negative_deg))
        value = command_norm[index]
        values.append(ca.if_else(value >= 0.0, value * positive, -value * negative))
    return ca.vertcat(*values)


def _downsample_indices(length: int, count: int) -> np.ndarray:
    return np.round(np.linspace(0, length - 1, count)).astype(int)


def _replay_defect(x_ref: np.ndarray, command_rad: np.ndarray, time_s: np.ndarray, config: AggressiveReversalOcpConfig, aircraft: object) -> float:
    if time_s.size < 2:
        return 0.0
    replay = np.empty_like(x_ref)
    replay[0] = x_ref[0]
    for index in range(time_s.size - 1):
        replay[index + 1] = rk4_step(
            replay[index],
            command_rad[index],
            float(time_s[index + 1] - time_s[index]),
            aircraft,
            None,
            config.wind_mode,
            config.actuator_tau_s,
        )
    return float(np.nanmax(np.abs(replay - x_ref)))


def direct_multiple_shooting_attempt(
    seed: CandidateResult,
    config: AggressiveReversalOcpConfig,
    aircraft: object,
    attempt_index: int,
) -> CandidateResult:
    """Attempt a real CasADi/IPOPT multiple-shooting refinement."""

    start = time.monotonic()
    nlp_constructed = False
    ipopt_called = False
    solver_status = "not_run"
    solver_iter_count = 0
    solver_objective = float("nan")
    constraint_residual_max = float("nan")
    replay_defect_max = float("nan")
    try:
        node_count = int(config.ocp_node_count)
        indices = _downsample_indices(seed.result.time_s.size, node_count + 1)
        seed_time = seed.result.time_s[indices]
        seed_state = seed.result.x_ref[indices]
        seed_command = seed.result.u_ff_norm[indices]
        dynamics = build_symbolic_dynamics(
            aircraft,
            actuator_tau_s=config.actuator_tau_s,
            wind_mode="none",
        )
        opti = ca.Opti()
        x_var = opti.variable(STATE_SIZE, node_count + 1)
        u_var = opti.variable(3, node_count + 1)
        opti.subject_to(x_var[:, 0] == seed_state[0])
        objective = 0.0
        defects = []
        for node in range(node_count):
            dt = float(seed_time[node + 1] - seed_time[node])
            u_rad = _symbolic_command_to_rad(u_var[:, node])
            f1 = dynamics.function(x_var[:, node], u_rad)
            f2 = dynamics.function(x_var[:, node] + 0.5 * dt * f1, u_rad)
            f3 = dynamics.function(x_var[:, node] + 0.5 * dt * f2, u_rad)
            f4 = dynamics.function(x_var[:, node] + dt * f3, u_rad)
            x_next = x_var[:, node] + (dt / 6.0) * (f1 + 2.0 * f2 + 2.0 * f3 + f4)
            defect = x_var[:, node + 1] - x_next
            defects.append(defect)
            opti.subject_to(defect == 0)
            opti.subject_to(opti.bounded(-1.0, u_var[:, node], 1.0))
            opti.subject_to(opti.bounded(TRUE_SAFE_BOUNDS.x_w_m[0], x_var[0, node], TRUE_SAFE_BOUNDS.x_w_m[1]))
            opti.subject_to(opti.bounded(TRUE_SAFE_BOUNDS.y_w_m[0], x_var[1, node], TRUE_SAFE_BOUNDS.y_w_m[1]))
            opti.subject_to(opti.bounded(0.05, x_var[2, node], TRUE_SAFE_BOUNDS.z_w_m[1]))
            objective += 0.01 * ca.sumsqr(u_var[:, node] - seed_command[node])
            objective += 0.001 * ca.sumsqr(x_var[9:12, node])
        opti.subject_to(opti.bounded(-1.0, u_var[:, -1], 1.0))
        target_rad = np.deg2rad(float(config.target_heading_deg))
        heading_progress = int(config.direction_sign) * (x_var[STATE_INDEX["psi"], -1] - x_var[STATE_INDEX["psi"], 0])
        terminal_speed = ca.sqrt(ca.sumsqr(x_var[6:9, -1]) + 1e-12)
        objective += 25.0 * (heading_progress - target_rad) ** 2
        objective += 5.0 * ca.fmax(5.0 - terminal_speed, 0.0) ** 2
        objective += 0.1 * ca.sumsqr(x_var[9:12, -1])
        if np.isclose(float(config.target_heading_deg), 30.0, atol=1e-9):
            labels = phase_labels_for_family(
                seed.family_name,
                seed_time,
                float(seed_time[-1]),
            )
            speed_nodes = [
                ca.sqrt(ca.sumsqr(x_var[6:9, node]) + 1e-12)
                for node in range(node_count + 1)
            ]
            energy_nodes = [
                x_var[STATE_INDEX["z_w"], node] + ca.sumsqr(x_var[6:9, node]) / (2.0 * G_M_S2)
                for node in range(node_count + 1)
            ]
            pitch_indices = [
                index for index, label in enumerate(labels) if label == "pitch_brake"
            ]
            redirect_indices = [
                index for index, label in enumerate(labels) if label == "slow_redirect"
            ]
            unload_indices = [
                index
                for index, label in enumerate(labels)
                if label in {"unload_descend", "exit_glide"}
            ]
            if pitch_indices:
                pre_index = max(0, pitch_indices[0] - 1)
                pitch_stop = pitch_indices[-1]
                pitch_energy_drop = energy_nodes[pre_index] - energy_nodes[pitch_stop]
                objective += 2.0 * ca.fmax(float(config.speed_m_s) - speed_nodes[pre_index], 0.0) ** 2
                objective += 4.0 * ca.fmax(pitch_energy_drop - 1.0, 0.0) ** 2
                objective += 2.0 * ca.fmax(4.0 - speed_nodes[pitch_stop], 0.0) ** 2
            for index in redirect_indices:
                objective += 1.5 * ca.fmax(4.0 - speed_nodes[index], 0.0) ** 2
            if redirect_indices:
                redirect_stop = redirect_indices[-1]
                redirect_heading = int(config.direction_sign) * (
                    x_var[STATE_INDEX["psi"], redirect_stop] - x_var[STATE_INDEX["psi"], 0]
                )
                objective += 8.0 * ca.fmax(0.7 * target_rad - redirect_heading, 0.0) ** 2
                objective += 4.0 * ca.fmax(redirect_heading - 1.15 * target_rad, 0.0) ** 2
            if unload_indices:
                unload_start = unload_indices[0]
                unload_stop = unload_indices[-1]
                unload_descent = (
                    x_var[STATE_INDEX["z_w"], unload_start]
                    - x_var[STATE_INDEX["z_w"], unload_stop]
                )
                unload_speed_gain = speed_nodes[unload_stop] - speed_nodes[unload_start]
                objective += 14.0 * ca.fmax(
                    UNLOAD_EXIT_DESCENT_REQUIRED_M - unload_descent,
                    0.0,
                ) ** 2
                objective += 14.0 * ca.fmax(
                    UNLOAD_EXIT_SPEED_GAIN_REQUIRED_M_S - unload_speed_gain,
                    0.0,
                ) ** 2
                objective += 3.0 * ca.fmax(5.0 - speed_nodes[unload_stop], 0.0) ** 2
        opti.minimize(objective)
        opti.set_initial(x_var, seed_state.T)
        opti.set_initial(u_var, seed_command.T)
        opti.solver(
            "ipopt",
            {
                "expand": False,
            },
            {
                "ipopt.print_level": 0,
                "ipopt.sb": "yes",
                "ipopt.max_iter": int(config.max_ipopt_iter),
                "ipopt.max_cpu_time": float(config.ocp_max_cpu_time_s),
                "print_time": 0,
            },
        )
        nlp_constructed = True
        ipopt_called = True
        solution = opti.solve()
        solver_status = str(opti.stats().get("return_status", "unknown"))
        solver_iter_count = int(opti.stats().get("iter_count", 0))
        x_nodes = np.asarray(solution.value(x_var)).T
        u_nodes = np.asarray(solution.value(u_var)).T
        solver_objective = float(solution.value(objective))
        if defects:
            defect_values = [np.asarray(solution.value(defect)).reshape(-1) for defect in defects]
            constraint_residual_max = float(np.max(np.abs(np.concatenate(defect_values))))
        command_rad = np.vstack([normalised_command_to_surface_rad(row) for row in u_nodes])
        replay_defect_max = _replay_defect(x_nodes, command_rad, seed_time, config, aircraft)
        phase = phase_labels_for_family(seed.family_name, seed_time, float(seed_time[-1]))
        metrics = metrics_for_candidate(
            config,
            seed_time,
            x_nodes,
            u_nodes,
            np.clip(u_nodes, -1.0, 1.0),
            phase,
            aircraft,
        )
        if replay_defect_max > REPLAY_DEFECT_TOL:
            metrics["success"] = False
            metrics["primitive_success"] = False
            metrics["failure_label"] = "model_boundary_only"
            metrics["notes"] = "direct_ocp_replay_defect_exceeded"
        result = AggressiveReversalOcpResult(
            target_heading_deg=float(config.target_heading_deg),
            direction_sign=int(config.direction_sign),
            success=bool(metrics["success"]),
            failure_label=str(metrics["failure_label"]),
            time_s=seed_time,
            x_ref=x_nodes,
            u_ff_norm=u_nodes,
            u_norm_applied=np.clip(u_nodes, -1.0, 1.0),
            delta_cmd_rad=command_rad,
            phase=phase,
            metrics=metrics,
            notes=str(metrics["notes"]),
        )
        converged = bool(
            solver_status in {"Solve_Succeeded", "Solved_To_Acceptable_Level"}
            and np.isfinite(replay_defect_max)
            and replay_defect_max <= REPLAY_DEFECT_TOL
        )
        return CandidateResult(
            method="direct_ocp",
            family_name=seed.family_name,
            attempt_index=attempt_index,
            result=result,
            nlp_constructed=nlp_constructed,
            ipopt_called=ipopt_called,
            solver_status=solver_status,
            solver_iter_count=solver_iter_count,
            solver_objective=solver_objective,
            constraint_residual_max=constraint_residual_max,
            direct_ocp_attempted=True,
            direct_ocp_converged=converged,
            replay_defect_max=replay_defect_max,
            limiting_mechanism=limiting_mechanism(str(metrics["failure_label"])),
        )
    except Exception as exc:
        solver_status = f"solver_exception:{type(exc).__name__}"
        metrics = dict(seed.result.metrics)
        metrics["success"] = False
        metrics["primitive_success"] = False
        metrics["failure_label"] = "solver_failure"
        metrics["notes"] = f"direct_ocp_failed:{type(exc).__name__}"
        result = AggressiveReversalOcpResult(
            target_heading_deg=seed.result.target_heading_deg,
            direction_sign=seed.result.direction_sign,
            success=False,
            failure_label="solver_failure",
            time_s=seed.result.time_s,
            x_ref=seed.result.x_ref,
            u_ff_norm=seed.result.u_ff_norm,
            u_norm_applied=seed.result.u_norm_applied,
            delta_cmd_rad=seed.result.delta_cmd_rad,
            phase=seed.result.phase,
            metrics=metrics,
            notes=str(metrics["notes"]),
        )
        return CandidateResult(
            method="direct_ocp",
            family_name=seed.family_name,
            attempt_index=attempt_index,
            result=result,
            nlp_constructed=nlp_constructed,
            ipopt_called=ipopt_called,
            solver_status=solver_status,
            solver_iter_count=solver_iter_count,
            solver_objective=solver_objective,
            constraint_residual_max=constraint_residual_max,
            direct_ocp_attempted=bool(nlp_constructed and ipopt_called),
            direct_ocp_converged=False,
            replay_defect_max=replay_defect_max,
            limiting_mechanism="optimiser_convergence",
        )
    finally:
        _ = time.monotonic() - start


def _write_csv(path: Path, header: tuple[str, ...], rows: list[tuple[object, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="") as handle:
        handle.write(",".join(header) + "\n")
        for row in rows:
            handle.write(",".join(str(item) for item in row) + "\n")


def _json_float(value: object) -> float | None:
    numeric = float(value)
    if not np.isfinite(numeric):
        return None
    return numeric


def checkpoint_candidate(
    candidate: CandidateResult,
    config: AggressiveReversalOcpConfig,
    label: str,
) -> dict[str, str]:
    """Write compact checkpoint JSON plus CSV paths for one candidate."""

    if config.checkpoint_dir is None:
        return {}
    checkpoint_dir = Path(config.checkpoint_dir)
    candidate_dir = Path(config.candidate_log_dir or checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    run_root = checkpoint_dir.parent.parent if checkpoint_dir.parent.name == "logs" else checkpoint_dir.parent
    def _checkpoint_relative(path: Path) -> str:
        try:
            return path.resolve().relative_to(run_root.resolve()).as_posix()
        except ValueError:
            return path.name

    target_token = f"{int(round(config.target_heading_deg)):0{TARGET_TOKEN_WIDTH}d}"
    stem = f"target_{target_token}_{label}_{candidate.attempt_index:03d}_{config.run_id}"
    trajectory_path = candidate_dir / f"{stem}_trajectory.csv"
    command_path = candidate_dir / f"{stem}_commands.csv"
    _write_csv(
        trajectory_path,
        ("time_s", *(f"x{i}" for i in range(STATE_SIZE))),
        [
            (candidate.result.time_s[index], *candidate.result.x_ref[index])
            for index in range(candidate.result.time_s.size)
        ],
    )
    _write_csv(
        command_path,
        (
            "time_s",
            "u_norm_requested_delta_a_norm",
            "u_norm_requested_delta_e_norm",
            "u_norm_requested_delta_r_norm",
            "u_norm_applied_delta_a_norm",
            "u_norm_applied_delta_e_norm",
            "u_norm_applied_delta_r_norm",
            "delta_cmd_rad_delta_a_cmd",
            "delta_cmd_rad_delta_e_cmd",
            "delta_cmd_rad_delta_r_cmd",
        ),
        [
            (
                candidate.result.time_s[index],
                *candidate.result.u_ff_norm[index],
                *candidate.result.u_norm_applied[index],
                *candidate.result.delta_cmd_rad[index],
            )
            for index in range(candidate.result.time_s.size)
        ],
    )
    checkpoint = {
        "target_heading_deg": float(config.target_heading_deg),
        "direction_sign": int(config.direction_sign),
        "method": candidate.method,
        "family_name": candidate.family_name,
        "attempt_index": int(candidate.attempt_index),
        "success": bool(candidate.result.success),
        "failure_label": candidate.result.failure_label,
        "notes": candidate.result.notes,
        "replay_finite": bool(candidate.result.metrics["finite_state_success"]),
        "recoverable": bool(candidate.result.metrics["recoverable"]),
        "primitive_success": bool(candidate.result.metrics["primitive_success"]),
        "actual_heading_change_deg": float(candidate.result.metrics["actual_heading_change_deg"]),
        "terminal_speed_m_s": float(candidate.result.metrics["terminal_speed_m_s"]),
        "specific_energy_initial_m": _json_float(
            candidate.result.metrics.get("specific_energy_initial_m", np.nan)
        ),
        "specific_energy_terminal_m": _json_float(
            candidate.result.metrics.get("specific_energy_terminal_m", np.nan)
        ),
        "specific_energy_lost_m": _json_float(
            candidate.result.metrics.get("specific_energy_lost_m", np.nan)
        ),
        "ideal_descent_to_5ms_m": _json_float(
            candidate.result.metrics.get("ideal_descent_to_5ms_m", np.nan)
        ),
        "initial_dive_or_speed_build": bool(
            candidate.result.metrics.get("initial_dive_or_speed_build", False)
        ),
        "perch_up_energy_trade": bool(
            candidate.result.metrics.get("perch_up_energy_trade", False)
        ),
        "slow_redirect_heading_gain": bool(
            candidate.result.metrics.get("slow_redirect_heading_gain", False)
        ),
        "unload_descent": bool(candidate.result.metrics.get("unload_descent", False)),
        "unload_speed_gain": bool(candidate.result.metrics.get("unload_speed_gain", False)),
        "unload_exit_descent_m": _json_float(
            candidate.result.metrics.get("unload_exit_descent_m", np.nan)
        ),
        "unload_exit_speed_gain_m_s": _json_float(
            candidate.result.metrics.get("unload_exit_speed_gain_m_s", np.nan)
        ),
        "manoeuvre_shape_class": str(
            candidate.result.metrics.get("manoeuvre_shape_class", "")
        ),
        "active_tradeoff": str(candidate.result.metrics.get("active_tradeoff", "")),
        "nlp_constructed": bool(candidate.nlp_constructed),
        "ipopt_called": bool(candidate.ipopt_called),
        "solver_status": candidate.solver_status,
        "solver_iter_count": int(candidate.solver_iter_count),
        "solver_objective": _json_float(candidate.solver_objective),
        "constraint_residual_max": _json_float(candidate.constraint_residual_max),
        "replay_defect_max": _json_float(candidate.replay_defect_max),
        "direct_ocp_attempted": bool(candidate.direct_ocp_attempted),
        "direct_ocp_converged": bool(candidate.direct_ocp_converged),
        "limiting_mechanism": candidate.limiting_mechanism,
        "trajectory_csv": _checkpoint_relative(trajectory_path),
        "commands_csv": _checkpoint_relative(command_path),
    }
    checkpoint_path = checkpoint_dir / f"{stem}.json"
    checkpoint_path.write_text(
        json.dumps(checkpoint, indent=2, allow_nan=False),
        encoding="ascii",
    )
    return {
        "checkpoint_json": _checkpoint_relative(checkpoint_path),
        "trajectory_csv": _checkpoint_relative(trajectory_path),
        "commands_csv": _checkpoint_relative(command_path),
    }


def guided_phase_search(
    config: AggressiveReversalOcpConfig,
    x0: np.ndarray,
    aircraft: object,
) -> list[CandidateResult]:
    """Run fixed manoeuvre-family seeds plus bounded amplitude variations."""

    time_s = _time_grid(config)
    candidates: list[CandidateResult] = []
    attempt_index = 0
    for family_name in seed_family_inventory_for_target(config.target_heading_deg):
        for scale in config.phase_search_scales:
            command = phase_seed_command_profile(config, time_s, family_name, scale)
            candidate = _result_from_replay(
                config,
                family_name,
                "phase_search",
                attempt_index,
                x0,
                time_s,
                command,
                aircraft,
            )
            checkpoint_candidate(candidate, config, "phase_search")
            candidates.append(candidate)
            attempt_index += 1
    return candidates


def _select_candidate(candidates: list[CandidateResult]) -> CandidateResult:
    if not candidates:
        raise ValueError("no aggressive reversal candidates were generated.")
    return max(candidates, key=_score_candidate)


def _best_by(candidates: list[CandidateResult], key: str) -> CandidateResult | None:
    filtered = [candidate for candidate in candidates if bool(candidate.result.metrics[key])]
    if not filtered:
        return None
    return max(filtered, key=_score_candidate)


def _unique_candidates(candidates: list[CandidateResult]) -> list[CandidateResult]:
    unique: list[CandidateResult] = []
    seen: set[tuple[str, str, int]] = set()
    for candidate in candidates:
        marker = (candidate.method, candidate.family_name, candidate.attempt_index)
        if marker in seen:
            continue
        unique.append(candidate)
        seen.add(marker)
    return unique


def _ocp_seed_candidates(
    candidates: list[CandidateResult],
    config: AggressiveReversalOcpConfig,
) -> list[CandidateResult]:
    if not candidates:
        return []
    if not np.isclose(float(config.target_heading_deg), 30.0, atol=1e-9):
        return [_select_candidate(candidates)]
    best_heading = max(
        candidates,
        key=lambda candidate: float(
            candidate.result.metrics.get("actual_heading_change_deg", -np.inf)
        ),
    )
    best_energy = max(
        candidates,
        key=lambda candidate: float(
            candidate.result.metrics.get("specific_energy_terminal_m", -np.inf)
        ),
    )
    best_recoverability = max(
        candidates,
        key=lambda candidate: (
            int(bool(candidate.result.metrics.get("combined_30deg_recoverable", False))),
            int(bool(candidate.result.metrics.get("recoverable", False))),
            float(candidate.result.metrics.get("recoverable_speed_by_z08_m_s", -np.inf)),
            float(candidate.result.metrics.get("terminal_speed_m_s", -np.inf)),
        ),
    )
    selected = _unique_candidates([best_heading, best_energy, best_recoverability])
    for candidate in sorted(candidates, key=_score_candidate, reverse=True):
        if len(selected) >= min(3, len(candidates)):
            break
        selected = _unique_candidates([*selected, candidate])
    return selected


# =============================================================================
# 5) Public Solve Workflow
# =============================================================================
def solve_aggressive_reversal_ocp(
    config: AggressiveReversalOcpConfig,
    x0: np.ndarray | None = None,
    aircraft: object | None = None,
) -> AggressiveReversalOcpResult:
    """Return a finite aggressive-reversal candidate or an honest failure result."""

    _validate_config(config)
    aircraft_model = adapt_glider(build_nausicaa_glider()) if aircraft is None else aircraft
    initial_state = build_aggressive_initial_state(config, aircraft_model) if x0 is None else as_state_vector(x0)
    candidates = guided_phase_search(config, initial_state, aircraft_model)
    phase_best = _select_candidate(candidates)
    next_family, reason = next_family_for_failure(
        phase_best.result.failure_label,
        phase_best.family_name,
    )
    if next_family != phase_best.family_name:
        time_s = _time_grid(config)
        command = phase_seed_command_profile(config, time_s, next_family, 1.0)
        followup = _result_from_replay(
            config,
            next_family,
            "phase_search_followup",
            len(candidates),
            initial_state,
            time_s,
            command,
            aircraft_model,
        )
        followup.result.metrics["next_family_reason"] = reason
        checkpoint_candidate(followup, config, "phase_followup")
        candidates.append(followup)
    if config.use_nonlinear_ocp:
        for ocp_seed in _ocp_seed_candidates(candidates, config):
            ocp_candidate = direct_multiple_shooting_attempt(
                ocp_seed,
                config,
                aircraft_model,
                len(candidates),
            )
            checkpoint_candidate(ocp_candidate, config, "direct_ocp")
            candidates.append(ocp_candidate)
    selected = _select_candidate(candidates)
    ocp_candidates = [
        candidate for candidate in candidates if candidate.method == "direct_ocp"
    ]
    ocp_evidence = ocp_candidates[-1] if ocp_candidates else None
    finite_best = _best_by(candidates, "finite_state_success")
    recoverable_best = _best_by(candidates, "recoverable")
    successful_best = _best_by(candidates, "success")
    selected.result.metrics.update(
        {
            "families_attempted": ";".join(
                seed_family_inventory_for_target(config.target_heading_deg)
            ),
            "selected_family": selected.family_name,
            "selected_method": selected.method,
            "next_family_reason": reason,
            "limiting_mechanism": str(
                selected.result.metrics.get("active_tradeoff", selected.limiting_mechanism)
            ),
            "best_finite_candidate": finite_best.family_name if finite_best else "",
            "best_recoverable_candidate": recoverable_best.family_name if recoverable_best else "",
            "best_successful_candidate": successful_best.family_name if successful_best else "",
            "nlp_constructed": bool(ocp_evidence and ocp_evidence.nlp_constructed),
            "ipopt_called": bool(ocp_evidence and ocp_evidence.ipopt_called),
            "solver_status": ocp_evidence.solver_status if ocp_evidence else "not_run",
            "solver_iter_count": ocp_evidence.solver_iter_count if ocp_evidence else 0,
            "solver_objective": ocp_evidence.solver_objective if ocp_evidence else np.nan,
            "constraint_residual_max": (
                ocp_evidence.constraint_residual_max if ocp_evidence else np.nan
            ),
            "replay_defect_max": ocp_evidence.replay_defect_max if ocp_evidence else np.nan,
            "direct_ocp_attempted": bool(
                ocp_evidence and ocp_evidence.direct_ocp_attempted
            ),
            "direct_ocp_converged": bool(
                ocp_evidence and ocp_evidence.direct_ocp_converged
            ),
            "phase_search_attempted": True,
            "replay_finite": selected.result.metrics["finite_state_success"],
        }
    )
    checkpoint_candidate(selected, config, "selected")
    return selected.result
