from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from feedback import limit_aggregate_command
from flight_dynamics import AircraftModel
from latency import AGGREGATE_LIMITS, CommandToSurfaceLayer, command_norm_to_angle
from primitive import PrimitiveContext
from rollout import rk4_step
from trajectory_primitive import TrajectoryEntryLimits, TrajectoryPrimitive
from tvlqr import TVLQRConfig, linearise_trajectory_finite_difference, solve_discrete_tvlqr


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Template dataclass
# 2) Candidate builder
# 3) Template helpers
# =============================================================================

# =============================================================================
# 1) Template Dataclass
# =============================================================================
@dataclass(frozen=True)
class AgileTurnTemplate:
    elevator_brake_norm: float
    aileron_roll_norm: float
    rudder_yaw_norm: float
    t_e_start_s: float
    t_a_start_s: float
    t_r_start_s: float
    t_e_duration_s: float
    t_a_duration_s: float
    t_r_duration_s: float
    hold_duration_s: float
    recover_duration_s: float
    target_heading_deg: float | None = None

    @property
    def duration_s(self) -> float:
        return float(
            max(
                self.t_e_start_s + self.t_e_duration_s,
                self.t_a_start_s + self.t_a_duration_s,
                self.t_r_start_s + self.t_r_duration_s,
            )
            + self.hold_duration_s
            + self.recover_duration_s
        )


# =============================================================================
# 2) Candidate Builder
# =============================================================================
def build_agile_reversal_candidate(
    template: AgileTurnTemplate,
    x0: np.ndarray,
    context: PrimitiveContext,
    aircraft: AircraftModel,
    wind_model: object,
    wind_mode: str,
    command_layer: CommandToSurfaceLayer,
    dt_s: float = 0.02,
) -> TrajectoryPrimitive:
    """Create one trajectory-indexed agile reversal primitive candidate."""
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive.")
    x_start = np.asarray(x0, dtype=float).reshape(15).copy()
    times = np.arange(0.0, template.duration_s + 0.5 * float(dt_s), float(dt_s))
    if times[-1] < template.duration_s:
        times = np.append(times, template.duration_s)
    x_ref = np.empty((times.size, 15), dtype=float)
    u_ff = np.empty((times.size, 3), dtype=float)
    x = x_start.copy()
    for idx, t_s in enumerate(times):
        command = _template_command(template, float(t_s), context)
        x_ref[idx] = x
        u_ff[idx] = command
        if idx == times.size - 1:
            break
        step_dt = float(times[idx + 1] - times[idx])
        x = rk4_step(
            x=x,
            u_cmd=command,
            dt_s=step_dt,
            aircraft=aircraft,
            wind_model=wind_model,
            rho_kg_m3=1.225,
            actuator_tau_s=command_layer.actuator_tau_vector_s,
            wind_mode=wind_mode,
        )
    a_mats, b_mats = linearise_trajectory_finite_difference(
        x_ref=x_ref,
        u_ff=u_ff,
        aircraft=aircraft,
        wind_model=wind_model,
        wind_mode=wind_mode,
        rho_kg_m3=1.225,
        actuator_tau_s=command_layer.actuator_tau_vector_s,
    )
    k_lqr, s_mats = solve_discrete_tvlqr(
        a_mats=a_mats,
        b_mats=b_mats,
        dt_s=dt_s,
        config=_default_tvlqr_config(),
    )
    target_heading_deg = _template_target_heading_deg(template)
    target_tag = f"{int(round(target_heading_deg)):03d}"
    return TrajectoryPrimitive(
        name=f"agile_reversal_left_tvlqr_target_{target_tag}",
        times_s=times,
        x_ref=x_ref,
        u_ff=u_ff,
        k_lqr=k_lqr,
        a_mats=a_mats,
        b_mats=b_mats,
        s_mats=s_mats,
        entry_limits=TrajectoryEntryLimits(
            max_position_error_m=0.24,
            max_attitude_error_rad=np.deg2rad(10.0),
            max_surface_error_rad=np.deg2rad(10.0),
        ),
        metadata={
            "target_label": f"agile_reversal_ref_target_{target_tag}",
            "primitive_family": "agile_tvlqr_scaffold",
            "is_full_turn_claim": False,
            "target_heading_deg": target_heading_deg,
            "command_domain": "aggregate surface radians from full [-1,+1] normalised template",
            "template": template,
            "entry_nominal_x": x_start.copy(),
            "dt_s": float(dt_s),
        },
    )


# =============================================================================
# 3) Template Helpers
# =============================================================================
def default_left_agile_reversal_template(
    target_heading_deg: float = 30.0,
) -> AgileTurnTemplate:
    target = _nearest_supported_target(target_heading_deg)
    fields = _target_template_fields(target)
    return AgileTurnTemplate(
        target_heading_deg=target,
        **fields,
    )


def supported_agile_heading_targets_deg() -> tuple[float, ...]:
    return tuple(float(target) for target in sorted(_TARGET_TEMPLATE_TABLE))


def _template_target_heading_deg(template: AgileTurnTemplate) -> float:
    if template.target_heading_deg is None:
        return 30.0
    return float(template.target_heading_deg)


def _nearest_supported_target(target_heading_deg: float) -> float:
    target = float(target_heading_deg)
    supported = supported_agile_heading_targets_deg()
    if target in supported:
        return target
    return min(supported, key=lambda value: abs(value - target))


def _target_template_fields(target_heading_deg: float) -> dict[str, float]:
    try:
        return dict(_TARGET_TEMPLATE_TABLE[float(target_heading_deg)])
    except KeyError as exc:
        raise ValueError(f"unsupported agile heading target {target_heading_deg}") from exc


# The table is deliberately small: each row is a deterministic scaffold target,
# not a broad optimiser or a claim that the target is feasible in the safety box.
_TARGET_TEMPLATE_TABLE = {
    30.0: {
        "elevator_brake_norm": -0.90,
        "aileron_roll_norm": 1.00,
        "rudder_yaw_norm": 0.85,
        "t_e_start_s": 0.08,
        "t_a_start_s": 0.08,
        "t_r_start_s": 0.08,
        "t_e_duration_s": 0.06,
        "t_a_duration_s": 0.34,
        "t_r_duration_s": 0.36,
        "hold_duration_s": 0.06,
        "recover_duration_s": 0.12,
    },
    60.0: {
        "elevator_brake_norm": -1.00,
        "aileron_roll_norm": 1.00,
        "rudder_yaw_norm": 1.00,
        "t_e_start_s": 0.08,
        "t_a_start_s": 0.08,
        "t_r_start_s": 0.08,
        "t_e_duration_s": 0.08,
        "t_a_duration_s": 0.48,
        "t_r_duration_s": 0.52,
        "hold_duration_s": 0.08,
        "recover_duration_s": 0.16,
    },
    90.0: {
        "elevator_brake_norm": -1.00,
        "aileron_roll_norm": 1.00,
        "rudder_yaw_norm": 1.00,
        "t_e_start_s": 0.08,
        "t_a_start_s": 0.08,
        "t_r_start_s": 0.08,
        "t_e_duration_s": 0.10,
        "t_a_duration_s": 0.64,
        "t_r_duration_s": 0.68,
        "hold_duration_s": 0.10,
        "recover_duration_s": 0.18,
    },
    120.0: {
        "elevator_brake_norm": -1.00,
        "aileron_roll_norm": 1.00,
        "rudder_yaw_norm": 1.00,
        "t_e_start_s": 0.08,
        "t_a_start_s": 0.08,
        "t_r_start_s": 0.08,
        "t_e_duration_s": 0.12,
        "t_a_duration_s": 0.76,
        "t_r_duration_s": 0.82,
        "hold_duration_s": 0.12,
        "recover_duration_s": 0.22,
    },
    180.0: {
        "elevator_brake_norm": -1.00,
        "aileron_roll_norm": 1.00,
        "rudder_yaw_norm": 1.00,
        "t_e_start_s": 0.08,
        "t_a_start_s": 0.08,
        "t_r_start_s": 0.08,
        "t_e_duration_s": 0.14,
        "t_a_duration_s": 0.96,
        "t_r_duration_s": 1.02,
        "hold_duration_s": 0.16,
        "recover_duration_s": 0.28,
    },
}


def _template_command(
    template: AgileTurnTemplate,
    t_s: float,
    context: PrimitiveContext,
) -> np.ndarray:
    a_norm = _pulse_then_reverse(
        t_s,
        template.t_a_start_s,
        template.t_a_duration_s,
        template.aileron_roll_norm,
    )
    e_norm = _single_pulse(
        t_s,
        template.t_e_start_s,
        template.t_e_duration_s,
        template.elevator_brake_norm,
    )
    r_norm = _pulse_then_reverse(
        t_s,
        template.t_r_start_s,
        template.t_r_duration_s,
        template.rudder_yaw_norm,
    )
    command = np.asarray(context.u_trim, dtype=float).reshape(3).copy()
    if a_norm != 0.0:
        command[0] = command_norm_to_angle(a_norm, AGGREGATE_LIMITS["delta_a"])
    if e_norm != 0.0:
        command[1] = command_norm_to_angle(e_norm, AGGREGATE_LIMITS["delta_e"])
    if r_norm != 0.0:
        command[2] = command_norm_to_angle(r_norm, AGGREGATE_LIMITS["delta_r"])
    return limit_aggregate_command(command)


def _single_pulse(t_s: float, start_s: float, duration_s: float, value: float) -> float:
    if start_s <= t_s < start_s + duration_s:
        return float(np.clip(value, -1.0, 1.0))
    return 0.0


def _pulse_then_reverse(t_s: float, start_s: float, duration_s: float, value: float) -> float:
    if duration_s <= 0.0:
        return 0.0
    if not (start_s <= t_s < start_s + duration_s):
        return 0.0
    half = 0.5 * duration_s
    sign = 1.0 if t_s < start_s + half else -1.0
    return float(np.clip(sign * value, -1.0, 1.0))


def _default_tvlqr_config() -> TVLQRConfig:
    return TVLQRConfig(
        q_diag=(
            0.05,
            0.20,
            0.05,
            1.60,
            1.20,
            0.80,
            0.20,
            0.25,
            0.25,
            0.50,
            0.50,
            0.50,
            0.08,
            0.08,
            0.08,
        ),
        r_diag=(45.0, 45.0, 45.0),
        qf_diag=(
            0.10,
            0.40,
            0.10,
            2.40,
            1.80,
            1.20,
            0.30,
            0.30,
            0.30,
            0.70,
            0.70,
            0.70,
            0.10,
            0.10,
            0.10,
        ),
    )
