from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Command lattice, labels, and latency dataclasses
# 2) Surface limits
# 3) Latency timing helpers
# 4) Command conversion helpers
# =============================================================================

# =============================================================================
# 1) Command lattice, labels, and latency dataclasses
# =============================================================================
# Normalised transmitter levels from the measured command interface
COMMAND_LEVELS = np.array(
    [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    dtype=float,
)
LATENCY_CASES = ("none", "actuator_lag_only", "nominal", "conservative")
LATENCY_PASS_LABELS = (
    "ideal_only",
    "nominal_pass",
    "nominal_fail",
    "conservative_pass",
    "conservative_fail",
    "ideal_only_latency_failed",
)
LATENCY_EXECUTION_STATUSES = (
    "ideal_timing",
    "actuator_lag_only",
    "command_delay_plus_actuator_lag",
    "full_state_command_actuator_latency",
)
TIMING_MODEL_VERSION = "measured_vicon_one_pole_command_response_v2_8hz_state_filter"


@dataclass(frozen=True)
class SurfaceLimit:
    name: str
    positive_deg: float
    negative_deg: float


@dataclass(frozen=True)
class LatencyEnvelope:
    # Command-response values use the latest accepted one-pole filtered runs.
    onset_latency_s: float = 0.073
    half_response_low_s: float = 0.098
    half_response_nominal_s: float = 0.108
    half_response_high_s: float = 0.118
    half_response_upper_s: float = 0.151
    actuator_t90_s: float = 0.130
    conservative_actuator_bound_s: float = 0.151
    # Vicon latency and filter delay are state-feedback terms, not actuator lag.
    vicon_latency_nominal_s: float = 0.0149
    vicon_latency_p95_s: float = 0.0169
    vicon_filter_delay_s: float = 0.0200
    vicon_filter_cutoff_hz: float = 8.0
    vicon_filter_model: str = "one_pole"
    command_dt_s: float = 0.02


DEFAULT_LATENCY_ENVELOPE = LatencyEnvelope()


@dataclass(frozen=True)
class CommandToSurfaceConfig:
    mode: str = "nominal"
    use_state_feedback_delay: bool = True


@dataclass(frozen=True)
class LatencyCaseConfig:
    latency_case: str
    state_feedback_delay_s: float
    command_onset_delay_s: float
    command_transport_delay_s: float
    actuator_tau_s: tuple[float, float, float]
    actuator_t50_s: float
    actuator_t90_s: float
    latency_jitter_s: float
    timing_model_version: str
    latency_pass_label: str


# =============================================================================
# 2) Surface Limits
# =============================================================================
# Direction-specific surface limits are stored in degrees for audit tables
SURFACE_LIMITS = {
    "Aileron_L": SurfaceLimit("Aileron_L", 26.8, -21.5),
    "Aileron_R": SurfaceLimit("Aileron_R", 29.5, -19.3),
    "Rudder": SurfaceLimit("Rudder", 33.0, -33.0),
    "Elevator": SurfaceLimit("Elevator", 23.7, -32.0),
}
AGGREGATE_LIMITS = {
    "delta_a": SurfaceLimit("delta_a_eff", 19.3, -21.5),
    "delta_e": SurfaceLimit("delta_e_eff", 23.7, -32.0),
    "delta_r": SurfaceLimit("delta_r_eff", 33.0, -33.0),
}


# =============================================================================
# 3) Latency Timing Helpers
# =============================================================================
def _first_order_tau_from_t50(t50_s: float, onset_s: float) -> float:
    response_s = max(float(t50_s) - float(onset_s), 0.0)
    return float(response_s / np.log(2.0))


def actuator_tau_s(
    config: CommandToSurfaceConfig | None = None,
    envelope: LatencyEnvelope | None = None,
) -> float:
    config = config or CommandToSurfaceConfig()
    envelope = envelope or LatencyEnvelope()
    half = half_response_s(config, envelope)
    # First-order lag reaches half response after tau * log(2)
    return float((half - envelope.onset_latency_s) / np.log(2.0))


def feedback_delay_s(
    config: CommandToSurfaceConfig | None = None,
    envelope: LatencyEnvelope | None = None,
) -> float:
    config = config or CommandToSurfaceConfig()
    envelope = envelope or LatencyEnvelope()
    if not config.use_state_feedback_delay:
        return 0.0
    # The simulated controller receives Vicon-filtered state, separate from actuator response.
    return float(envelope.vicon_latency_nominal_s + envelope.vicon_filter_delay_s)


def half_response_s(
    config: CommandToSurfaceConfig | None = None,
    envelope: LatencyEnvelope | None = None,
) -> float:
    config = config or CommandToSurfaceConfig()
    envelope = envelope or LatencyEnvelope()
    if config.mode == "low":
        return float(envelope.half_response_low_s)
    if config.mode == "nominal":
        return float(envelope.half_response_nominal_s)
    if config.mode == "high":
        return float(envelope.half_response_high_s)
    if config.mode == "robust_upper":
        return float(envelope.half_response_upper_s)
    raise ValueError("latency mode must be 'low', 'nominal', 'high', or 'robust_upper'.")


def latency_range_s(
    config: CommandToSurfaceConfig | None = None,
    envelope: LatencyEnvelope | None = None,
) -> tuple[float, float]:
    config = config or CommandToSurfaceConfig()
    envelope = envelope or LatencyEnvelope()
    if config.mode == "robust_upper":
        # Robust-upper mode spans nominal to measured upper timing
        return (
            float(envelope.half_response_nominal_s),
            float(envelope.half_response_upper_s),
        )
    return (
        float(envelope.half_response_low_s),
        float(envelope.half_response_high_s),
    )


def latency_range_label(
    config: CommandToSurfaceConfig | None = None,
    envelope: LatencyEnvelope | None = None,
) -> str:
    low, high = latency_range_s(config, envelope)
    return f"{low:.6f}:{high:.6f}"


def latency_audit_fields(
    config: CommandToSurfaceConfig | None = None,
    envelope: LatencyEnvelope | None = None,
) -> dict[str, float | str]:
    config = config or CommandToSurfaceConfig()
    envelope = envelope or LatencyEnvelope()
    # These fields make the measured command path explicit in every scenario log.
    return {
        "state_feedback_delay_s": feedback_delay_s(config, envelope),
        "actuator_t10_s": float(envelope.onset_latency_s),
        "actuator_t50_nominal_s": float(envelope.half_response_nominal_s),
        "actuator_t90_s": float(envelope.actuator_t90_s),
        "conservative_actuator_bound_s": float(envelope.conservative_actuator_bound_s),
        "vicon_filter_cutoff_hz": float(envelope.vicon_filter_cutoff_hz),
        "vicon_filter_model": str(envelope.vicon_filter_model),
    }


def latency_case_config(
    latency_case: str,
    envelope: LatencyEnvelope = DEFAULT_LATENCY_ENVELOPE,
) -> LatencyCaseConfig:
    """Return the active timing contract for one canonical latency case."""

    case = str(latency_case)
    if case not in LATENCY_CASES:
        raise ValueError(
            "latency_case must be one of "
            + ", ".join(f"'{label}'" for label in LATENCY_CASES)
            + "."
        )

    zero_tau = (0.0, 0.0, 0.0)
    nominal_tau = _first_order_tau_from_t50(
        envelope.half_response_nominal_s,
        envelope.onset_latency_s,
    )
    conservative_tau = _first_order_tau_from_t50(
        envelope.conservative_actuator_bound_s,
        envelope.onset_latency_s,
    )
    state_feedback_nominal = float(
        envelope.vicon_latency_nominal_s + envelope.vicon_filter_delay_s
    )

    if case == "none":
        return LatencyCaseConfig(
            latency_case=case,
            state_feedback_delay_s=0.0,
            command_onset_delay_s=0.0,
            command_transport_delay_s=0.0,
            actuator_tau_s=zero_tau,
            actuator_t50_s=0.0,
            actuator_t90_s=0.0,
            latency_jitter_s=0.0,
            timing_model_version=TIMING_MODEL_VERSION,
            latency_pass_label="ideal_only",
        )
    if case == "actuator_lag_only":
        return LatencyCaseConfig(
            latency_case=case,
            state_feedback_delay_s=0.0,
            command_onset_delay_s=0.0,
            command_transport_delay_s=0.0,
            actuator_tau_s=(nominal_tau, nominal_tau, nominal_tau),
            actuator_t50_s=float(envelope.half_response_nominal_s),
            actuator_t90_s=float(envelope.actuator_t90_s),
            latency_jitter_s=0.0,
            timing_model_version=TIMING_MODEL_VERSION,
            latency_pass_label="ideal_only",
        )
    if case == "nominal":
        return LatencyCaseConfig(
            latency_case=case,
            state_feedback_delay_s=state_feedback_nominal,
            command_onset_delay_s=float(envelope.onset_latency_s),
            command_transport_delay_s=0.0,
            actuator_tau_s=(nominal_tau, nominal_tau, nominal_tau),
            actuator_t50_s=float(envelope.half_response_nominal_s),
            actuator_t90_s=float(envelope.actuator_t90_s),
            latency_jitter_s=0.0,
            timing_model_version=TIMING_MODEL_VERSION,
            latency_pass_label="nominal_pass",
        )

    return LatencyCaseConfig(
        latency_case=case,
        state_feedback_delay_s=float(
            envelope.vicon_latency_p95_s + envelope.vicon_filter_delay_s
        ),
        command_onset_delay_s=float(envelope.onset_latency_s),
        command_transport_delay_s=0.0,
        actuator_tau_s=(conservative_tau, conservative_tau, conservative_tau),
        actuator_t50_s=float(envelope.conservative_actuator_bound_s),
        actuator_t90_s=float(
            max(envelope.actuator_t90_s, envelope.conservative_actuator_bound_s)
        ),
        latency_jitter_s=0.0,
        timing_model_version=TIMING_MODEL_VERSION,
        latency_pass_label="conservative_pass",
    )


def latency_audit_fields_from_case(
    config: LatencyCaseConfig,
    active_actuator_tau_s: tuple[float, float, float] | None = None,
) -> dict[str, object]:
    """Return CSV/JSON-ready latency fields for scenario and primitive logs."""

    if active_actuator_tau_s is None or config.latency_case == "none":
        tau = tuple(float(value) for value in config.actuator_tau_s)
        actuator_t50_s = float(config.actuator_t50_s)
        actuator_t90_s = float(config.actuator_t90_s)
    else:
        tau = tuple(
            float(value)
            for value in np.asarray(active_actuator_tau_s, dtype=float).reshape(3)
        )
        if config.latency_case == "actuator_lag_only":
            max_tau = max(tau)
            actuator_t50_s = float(max_tau * np.log(2.0))
            actuator_t90_s = float(max_tau * np.log(10.0))
        else:
            actuator_t50_s = float(config.actuator_t50_s)
            actuator_t90_s = float(config.actuator_t90_s)

    return {
        "latency_case": str(config.latency_case),
        "state_feedback_delay_s": float(config.state_feedback_delay_s),
        "command_onset_delay_s": float(config.command_onset_delay_s),
        "command_transport_delay_s": float(config.command_transport_delay_s),
        "actuator_tau_s": format_actuator_tau_s(tau),
        "actuator_t50_s": actuator_t50_s,
        "actuator_t90_s": actuator_t90_s,
        "latency_jitter_s": float(config.latency_jitter_s),
        "timing_model_version": str(config.timing_model_version),
        "latency_pass_label": str(config.latency_pass_label),
    }


def actuator_tau_for_case(
    config: LatencyCaseConfig,
    fallback_tau_s: tuple[float, float, float] = (0.06, 0.06, 0.06),
) -> tuple[float, float, float]:
    """Return the actuator lag constants active for a latency case."""

    if config.latency_case in ("none", "actuator_lag_only"):
        tau = np.asarray(fallback_tau_s, dtype=float).reshape(3)
    else:
        tau = np.asarray(config.actuator_tau_s, dtype=float).reshape(3)
    if not np.all(np.isfinite(tau)) or np.any(tau <= 0.0):
        raise ValueError("actuator tau must contain three finite positive values.")
    return tuple(float(value) for value in tau)


def format_actuator_tau_s(actuator_tau_s: tuple[float, float, float]) -> str:
    """Return the CSV-ready actuator tau triplet with fixed precision."""

    tau = np.asarray(actuator_tau_s, dtype=float).reshape(3)
    if not np.all(np.isfinite(tau)):
        raise ValueError("actuator_tau_s must contain finite values.")
    return ";".join(f"{float(value):.9f}" for value in tau)


def latency_acceptance_scope(latency_case: str) -> str:
    """Return the evidence scope implied by a single latency replay case."""

    case = str(latency_case)
    if case == "none":
        return "ideal_ablation_only"
    if case == "actuator_lag_only":
        return "actuator_lag_only_ablation"
    if case == "nominal":
        return "command_path_nominal_no_feedback_controller"
    if case == "conservative":
        return "command_path_conservative_no_feedback_controller"
    raise ValueError(
        "latency_case must be one of "
        + ", ".join(f"'{label}'" for label in LATENCY_CASES)
        + "."
    )


def latency_mechanism_flags_from_case(
    latency_case: str,
    *,
    state_feedback_delay_applied: bool = False,
) -> dict[str, bool]:
    """Return the timing mechanisms active for the current replay architecture."""

    case = str(latency_case)
    if case not in LATENCY_CASES:
        raise ValueError(
            "latency_case must be one of "
            + ", ".join(f"'{label}'" for label in LATENCY_CASES)
            + "."
        )
    state_delay = bool(state_feedback_delay_applied)
    if case == "none":
        if state_delay:
            raise ValueError("ideal timing cannot apply state-feedback delay.")
        return {
            "state_feedback_delay_applied": False,
            "command_delay_applied": False,
            "actuator_lag_applied": False,
        }
    if case == "actuator_lag_only":
        if state_delay:
            raise ValueError("actuator lag only cannot apply state-feedback delay.")
        return {
            "state_feedback_delay_applied": False,
            "command_delay_applied": False,
            "actuator_lag_applied": True,
        }
    return {
        "state_feedback_delay_applied": state_delay,
        "command_delay_applied": True,
        "actuator_lag_applied": True,
    }


def latency_execution_status(
    *,
    latency_case: str,
    state_feedback_delay_applied: bool,
    command_delay_applied: bool,
    actuator_lag_applied: bool,
) -> str:
    """Return an audit label for which latency mechanisms were active."""

    case = str(latency_case)
    if case not in LATENCY_CASES:
        raise ValueError(
            "latency_case must be one of "
            + ", ".join(f"'{label}'" for label in LATENCY_CASES)
            + "."
        )
    state_delay = bool(state_feedback_delay_applied)
    command_delay = bool(command_delay_applied)
    actuator_lag = bool(actuator_lag_applied)
    if state_delay and not (command_delay and actuator_lag):
        raise ValueError("state-feedback latency requires command delay and actuator lag.")
    if command_delay and not actuator_lag:
        raise ValueError("command delay cannot be claimed without actuator lag.")
    if case == "none":
        if state_delay or command_delay or actuator_lag:
            raise ValueError("latency_case='none' must use ideal timing mechanisms.")
        return "ideal_timing"
    if case == "actuator_lag_only":
        if state_delay or command_delay or not actuator_lag:
            raise ValueError(
                "latency_case='actuator_lag_only' must apply actuator lag only."
            )
        return "actuator_lag_only"
    if not command_delay or not actuator_lag:
        raise ValueError(
            "nominal/conservative latency requires command delay plus actuator lag."
        )
    if state_delay:
        return "full_state_command_actuator_latency"
    return "command_delay_plus_actuator_lag"


def latency_pass_label_for_single_run(latency_case: str, accepted: bool) -> str:
    """Return the outcome label for one replay without paired-case inference."""

    case = str(latency_case)
    if case not in LATENCY_CASES:
        raise ValueError(
            "latency_case must be one of "
            + ", ".join(f"'{label}'" for label in LATENCY_CASES)
            + "."
        )
    if case in ("none", "actuator_lag_only"):
        return "ideal_only"
    if case == "nominal":
        return "nominal_pass" if bool(accepted) else "nominal_fail"
    return "conservative_pass" if bool(accepted) else "conservative_fail"


def latency_pass_label_for_paired_comparison(
    ideal_or_ablation_accepted: bool,
    nominal_accepted: bool,
) -> str:
    """Return a paired ideal/nominal comparison label when both outcomes exist."""

    if bool(ideal_or_ablation_accepted) and not bool(nominal_accepted):
        return "ideal_only_latency_failed"
    return "nominal_pass" if bool(nominal_accepted) else "nominal_fail"


def delayed_state_sample(
    times_s: np.ndarray,
    states: np.ndarray,
    query_time_s: float,
) -> np.ndarray:
    """Return a state-history sample at query_time_s using column interpolation."""

    times, values = _validate_history(times_s, states, "states")
    query = _finite_query_time(query_time_s)
    if query <= times[0]:
        return values[0].copy()
    if query >= times[-1]:
        return values[-1].copy()
    return np.array(
        [
            np.interp(query, times, values[:, column])
            for column in range(values.shape[1])
        ],
        dtype=float,
    )


def delayed_command_sample(
    times_s: np.ndarray,
    commands: np.ndarray,
    query_time_s: float,
) -> np.ndarray:
    """Return a zero-order-hold command sample at query_time_s."""

    times, values = _validate_history(times_s, commands, "commands")
    if values.shape[1] != 3:
        raise ValueError("commands must have shape (N, 3).")
    query = _finite_query_time(query_time_s)
    index = int(np.searchsorted(times, query, side="right") - 1)
    index = int(np.clip(index, 0, times.size - 1))
    return values[index].copy()


def latency_adjusted_command_sample(
    times_s: np.ndarray,
    commands_norm: np.ndarray,
    query_time_s: float,
    config: LatencyCaseConfig,
) -> np.ndarray:
    """Return delayed executable-lattice commands before surface conversion."""

    delay_s = float(config.command_onset_delay_s + config.command_transport_delay_s)
    commands = np.asarray(commands_norm, dtype=float).reshape(-1, 3)
    quantised = np.asarray(
        [[quantise_command_norm(value) for value in row] for row in commands],
        dtype=float,
    )
    return delayed_command_sample(times_s, quantised, float(query_time_s) - delay_s)


def _validate_history(
    times_s: np.ndarray,
    values: np.ndarray,
    value_label: str,
) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times_s, dtype=float).reshape(-1)
    rows = np.asarray(values, dtype=float)
    if rows.ndim == 1:
        rows = rows.reshape(-1, 1)
    if times.size == 0:
        raise ValueError("history times_s must contain at least one sample.")
    if rows.shape[0] != times.size:
        raise ValueError(f"{value_label} row count must match times_s length.")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(rows)):
        raise ValueError("history times and values must be finite.")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("history times_s must be strictly increasing.")
    return times, rows


def _finite_query_time(query_time_s: float) -> float:
    query = float(query_time_s)
    if not np.isfinite(query):
        raise ValueError("query_time_s must be finite.")
    return query


# =============================================================================
# 4) Command Conversion Helpers
# =============================================================================
def angle_to_command_norm(angle_rad: float, limit: SurfaceLimit) -> float:
    # Endpoint signs are physical surface signs. Aggregate-to-physical mixing
    # handles aileron opposition before this conversion is used for audits.
    angle_deg = float(np.rad2deg(angle_rad))
    if abs(angle_deg) <= 1e-12:
        return 0.0

    candidates: list[float] = []
    if abs(limit.positive_deg) > 1e-12:
        positive_ratio = angle_deg / float(limit.positive_deg)
        if positive_ratio >= 0.0:
            candidates.append(float(positive_ratio))
    if abs(limit.negative_deg) > 1e-12:
        negative_ratio = angle_deg / float(limit.negative_deg)
        if negative_ratio >= 0.0:
            candidates.append(float(-negative_ratio))
    if not candidates:
        endpoint = (
            limit.positive_deg
            if abs(angle_deg - limit.positive_deg) <= abs(angle_deg - limit.negative_deg)
            else limit.negative_deg
        )
        norm = 1.0 if endpoint == limit.positive_deg else -1.0
        return float(norm)
    norm = min(candidates, key=lambda value: abs(abs(value) - 1.0) if abs(value) > 1.0 else abs(value))
    return float(np.clip(norm, -1.0, 1.0))


def command_norm_to_angle(norm: float, limit: SurfaceLimit) -> float:
    value = float(np.clip(norm, -1.0, 1.0))
    if value >= 0.0:
        return float(np.deg2rad(value * limit.positive_deg))
    return float(np.deg2rad((-value) * limit.negative_deg))


def quantise_command_norm(norm: float) -> float:
    idx = int(np.argmin(np.abs(COMMAND_LEVELS - float(norm))))
    return float(COMMAND_LEVELS[idx])


def aggregate_targets_to_surface_degrees(target_rad: np.ndarray) -> dict[str, float]:
    # Aggregate commands are expanded to physical surface directions
    delta_a, delta_e, delta_r = np.asarray(target_rad, dtype=float).reshape(3)
    a_norm = angle_to_command_norm(delta_a, AGGREGATE_LIMITS["delta_a"])
    e_norm = angle_to_command_norm(delta_e, AGGREGATE_LIMITS["delta_e"])
    r_norm = angle_to_command_norm(delta_r, AGGREGATE_LIMITS["delta_r"])
    return {
        "aileron_l_deg": float(np.rad2deg(command_norm_to_angle(a_norm, SURFACE_LIMITS["Aileron_L"]))),
        "aileron_r_deg": float(np.rad2deg(command_norm_to_angle(-a_norm, SURFACE_LIMITS["Aileron_R"]))),
        "elevator_deg": float(np.rad2deg(command_norm_to_angle(e_norm, SURFACE_LIMITS["Elevator"]))),
        "rudder_deg": float(np.rad2deg(command_norm_to_angle(r_norm, SURFACE_LIMITS["Rudder"]))),
    }
