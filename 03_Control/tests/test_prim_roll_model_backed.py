from __future__ import annotations

import json

import numpy as np
import pytest

from env_ctx import EnvironmentMetadata, build_environment_context
from env_surrogate import resolve_surrogate_binding, wind_field_for_binding
from implementation_instance import implementation_instance_for_layer
from lqr_controller import lqr_controller_for_primitive_id
from plant_instance import plant_instance_for_layer
from prim_cat import primitive_by_id
from prim_roll import RolloutConfig, rollout_evidence_row, simulate_primitive_rollout
from state_contract import STATE_INDEX, STATE_SIZE


class ConstantWind:
    name = "constant_unit_test_wind"
    source = "unit_test"

    def __init__(self, value_m_s: float) -> None:
        self.value_m_s = float(value_m_s)

    def __call__(self, points_w_up_m: np.ndarray) -> np.ndarray:
        points = np.asarray(points_w_up_m, dtype=float).reshape(-1, 3)
        return np.column_stack(
            [
                np.zeros(points.shape[0]),
                np.zeros(points.shape[0]),
                np.full(points.shape[0], self.value_m_s),
            ]
        )


def _state(*, x_w_m: float = 2.0, z_w_m: float = 1.6, u_m_s: float = 5.8) -> np.ndarray:
    state = np.zeros(STATE_SIZE)
    state[STATE_INDEX["x_w"]] = x_w_m
    state[STATE_INDEX["y_w"]] = 2.0
    state[STATE_INDEX["z_w"]] = z_w_m
    state[STATE_INDEX["u"]] = u_m_s
    return state


def _context_and_wind(state: np.ndarray):
    metadata = EnvironmentMetadata(
        environment_id="W1_model_backed",
        fan_count=1,
        fan_positions_m=((4.2, 2.4),),
        fan_power_scales=(1.0,),
        updraft_model_id="single_gaussian_var",
    )
    binding = resolve_surrogate_binding("W1", metadata)
    wind = wind_field_for_binding(binding)
    context = build_environment_context(
        state,
        wind_field=wind,
        metadata=metadata,
        latency_case="nominal",
        surrogate_binding=binding,
    )
    return context, wind


def _controller(primitive_id: str):
    return lqr_controller_for_primitive_id(primitive_id)


def test_model_backed_rollout_is_distinct_from_smoke_and_finite() -> None:
    state = _state()
    context, wind = _context_and_wind(state)

    evidence = simulate_primitive_rollout(
        rollout_id="model_rollout_000",
        episode_id="episode_000",
        initial_state=state,
        context=context,
        primitive=primitive_by_id("glide"),
        config=RolloutConfig(W_layer="W1", rollout_backend="model_backed_lqr"),
        wind_field=wind,
        controller=_controller("glide"),
    )

    assert evidence.rollout_backend == "model_backed_lqr"
    assert evidence.evidence_role in {"lqr_rollout_candidate", "blocked_lqr_synthesis"}
    assert evidence.controller_family == "lqr"
    assert evidence.controller_id.startswith("lqrta_glide_")
    assert evidence.controller_design_role == "active_timing_aware_w01"
    assert evidence.timing_augmentation_type == "actuator_surface_state_command_fifo_predictor_compensated"
    assert evidence.timing_state_source == "history_backed_fifo"
    assert evidence.augmented_gain_checksum
    assert evidence.controller_selection_status == "explicit_lqr_unverified"
    assert evidence.controller_evidence_status == "executable_lqr"
    assert evidence.surrogate_binding_status == "ready"
    assert evidence.trajectory_integrity_status == "finite_model_backed"
    assert np.isfinite(evidence.energy_residual_m)
    assert np.isfinite(evidence.floor_margin_m)
    assert np.isfinite(evidence.ceiling_margin_m)


def test_model_backed_rollout_integrates_positive_wing_updraft_along_trajectory() -> None:
    state = _state()
    metadata = EnvironmentMetadata(
        environment_id="constant_updraft",
        fan_count=1,
        fan_positions_m=((4.2, 2.4),),
        fan_power_scales=(1.0,),
        updraft_model_id="constant_unit_test_wind",
    )
    wind = ConstantWind(1.0)
    context = build_environment_context(
        state,
        wind_field=wind,
        metadata=metadata,
        latency_case="nominal",
    )

    evidence = simulate_primitive_rollout(
        rollout_id="model_rollout_integrated_updraft",
        episode_id="episode_integrated_updraft",
        initial_state=state,
        context=context,
        primitive=primitive_by_id("glide"),
        config=RolloutConfig(W_layer="W1", rollout_backend="model_backed_lqr"),
        wind_field=wind,
        controller=_controller("glide"),
    )
    row = rollout_evidence_row(evidence)

    assert evidence.updraft_integration_status == "trajectory_integrated_wing_panel_positive_updraft"
    assert evidence.trajectory_integrated_updraft_gain_m == pytest.approx(evidence.rollout_duration_s)
    assert evidence.trajectory_mean_positive_w_wing_m_s == pytest.approx(1.0)
    assert evidence.trajectory_lift_dwell_time_s == pytest.approx(evidence.rollout_duration_s)
    assert row["trajectory_integrated_updraft_gain_m"] == pytest.approx(evidence.rollout_duration_s)


def test_model_backed_low_speed_initial_state_is_simulated_and_audited() -> None:
    state = _state(u_m_s=2.0)
    context, wind = _context_and_wind(state)
    config = RolloutConfig(W_layer="W1", rollout_backend="model_backed_lqr")

    evidence = simulate_primitive_rollout(
        rollout_id="model_rollout_low_speed",
        initial_state=state,
        context=context,
        primitive=primitive_by_id("glide"),
        config=config,
        wind_field=wind,
        controller=_controller("glide"),
    )

    assert evidence.outcome_class != "blocked"
    assert evidence.trajectory_integrity_status == "finite_model_backed"


def test_model_backed_missing_explicit_controller_is_blocked_before_integration() -> None:
    state = _state()
    context, wind = _context_and_wind(state)

    evidence = simulate_primitive_rollout(
        rollout_id="model_rollout_missing_controller",
        initial_state=state,
        context=context,
        primitive=primitive_by_id("glide"),
        config=RolloutConfig(W_layer="W1", rollout_backend="model_backed_lqr"),
        wind_field=wind,
    )

    assert evidence.outcome_class == "blocked"
    assert evidence.controller_selection_status == "missing_explicit_lqr_controller"
    assert evidence.failure_label == "missing_explicit_lqr_controller"
    assert evidence.entry_rejection_class == "controller_blocked"
    assert evidence.termination_cause == "controller_blocked"
    assert evidence.trajectory_integrity_status == "blocked_before_simulation"
    assert evidence.max_abs_command_norm == 0.0


def test_model_backed_nonfinite_initial_state_is_blocked() -> None:
    context, wind = _context_and_wind(_state())
    state = _state()
    state[STATE_INDEX["u"]] = np.nan

    evidence = simulate_primitive_rollout(
        rollout_id="model_rollout_nonfinite",
        initial_state=state,
        context=context,
        primitive=primitive_by_id("glide"),
        config=RolloutConfig(W_layer="W1", rollout_backend="model_backed_lqr"),
        wind_field=wind,
        controller=_controller("glide"),
    )

    assert evidence.outcome_class == "blocked"
    assert evidence.entry_check_status == "nonfinite_initial_state"
    assert evidence.failure_label == "nonfinite_initial_state"
    assert evidence.boundary_use_class == "hard_failure"
    assert evidence.entry_rejection_class == "physical_hard_failure"
    assert evidence.termination_cause == "nonfinite_initial_state"


def test_model_backed_floor_initial_state_is_blocked() -> None:
    state = _state(z_w_m=0.2)
    context, wind = _context_and_wind(_state())

    evidence = simulate_primitive_rollout(
        rollout_id="model_rollout_floor",
        initial_state=state,
        context=context,
        primitive=primitive_by_id("glide"),
        config=RolloutConfig(W_layer="W1", rollout_backend="model_backed_lqr"),
        wind_field=wind,
        controller=_controller("glide"),
    )

    assert evidence.outcome_class == "blocked"
    assert evidence.failure_label == "initial_floor_violation"
    assert evidence.boundary_use_class == "hard_failure"
    assert evidence.entry_rejection_class == "physical_hard_failure"


def test_model_backed_wall_exit_is_retained_as_terminal_useful_not_continuation_row() -> None:
    state = _state(x_w_m=6.55)
    context, wind = _context_and_wind(state)

    evidence = simulate_primitive_rollout(
        rollout_id="model_rollout_wall",
        initial_state=state,
        context=context,
        primitive=primitive_by_id("mild_turn_left"),
        config=RolloutConfig(W_layer="W1", rollout_backend="model_backed_lqr"),
        wind_field=wind,
        controller=_controller("mild_turn_left"),
    )

    assert evidence.outcome_class in {"weak", "failed"}
    assert evidence.outcome_class != "boundary_terminal"
    assert evidence.failure_label in {"controlled_xy_boundary_terminal", "xy_boundary_terminal"}
    assert evidence.episode_terminal_status == "episode_terminal_useful"
    assert evidence.continuation_status == "not_continuation_valid"
    assert evidence.continuation_valid is False
    assert evidence.episode_terminal_useful is True
    assert evidence.boundary_use_class == "episode_terminal_useful"
    assert evidence.terminal_use_trainable is True
    assert evidence.minimum_wall_margin_m < 0.0


def test_model_backed_controlled_xy_exit_is_safe_terminal_even_with_energy_loss() -> None:
    state = _state(x_w_m=6.55)
    context, wind = _context_and_wind(state)

    evidence = simulate_primitive_rollout(
        rollout_id="model_rollout_controlled_xy_terminal",
        initial_state=state,
        context=context,
        primitive=primitive_by_id("glide"),
        config=RolloutConfig(W_layer="W0", rollout_backend="model_backed_lqr"),
        wind_field=None,
        controller=_controller("glide"),
    )
    row = rollout_evidence_row(evidence)

    assert evidence.termination_cause == "wall_boundary_exit_retained"
    assert evidence.failure_label == "controlled_xy_boundary_terminal"
    assert evidence.boundary_use_class == "episode_terminal_useful"
    assert evidence.episode_terminal_useful is True
    assert evidence.continuation_valid is False
    assert row["failure_label"] == "controlled_xy_boundary_terminal"


def test_retired_feedback_rollout_backends_are_rejected() -> None:
    state = _state()
    context, wind = _context_and_wind(state)
    retired_backend = "model_backed_" + "feedback"

    try:
        simulate_primitive_rollout(
            rollout_id="retired_feedback_rollout_000",
            initial_state=state,
            context=context,
            primitive=primitive_by_id("glide"),
            config=RolloutConfig(
                W_layer="W1",
                rollout_backend=retired_backend,
            ),
            wind_field=wind,
        )
    except ValueError as exc:
        assert "rollout_backend must be one of the retained rollout backends" in str(exc)
    else:
        raise AssertionError("retired feedback rollout backend was accepted")


def test_latency_mechanisms_are_applied_and_logged() -> None:
    state = _state()
    context, wind = _context_and_wind(state)
    ideal_context = context.__class__(**{**context.__dict__, "latency_case": "none"})

    ideal = simulate_primitive_rollout(
        rollout_id="ideal_latency",
        initial_state=state,
        context=ideal_context,
        primitive=primitive_by_id("lift_dwell_arc"),
        config=RolloutConfig(W_layer="W1", rollout_backend="model_backed_lqr"),
        wind_field=wind,
        controller=_controller("lift_dwell_arc"),
    )
    nominal = simulate_primitive_rollout(
        rollout_id="nominal_latency",
        initial_state=state,
        context=context,
        primitive=primitive_by_id("lift_dwell_arc"),
        config=RolloutConfig(W_layer="W1", rollout_backend="model_backed_lqr"),
        wind_field=wind,
        controller=_controller("lift_dwell_arc"),
    )

    assert ideal.state_feedback_delay_applied is False
    assert ideal.command_delay_applied is False
    assert ideal.actuator_lag_applied is False
    assert nominal.state_feedback_delay_applied is True
    assert nominal.command_delay_applied is True
    assert nominal.actuator_lag_applied is True
    assert nominal.latency_execution_status == "full_state_command_actuator_latency"
    assert np.isfinite(nominal.max_abs_command_norm)


def test_command_timing_history_can_continue_across_primitive_rollouts() -> None:
    state = _state()
    context, wind = _context_and_wind(state)
    primitive = primitive_by_id("glide")
    first = simulate_primitive_rollout(
        rollout_id="timing_continuity_first",
        initial_state=state,
        context=context,
        primitive=primitive,
        config=RolloutConfig(W_layer="W1", rollout_backend="model_backed_lqr"),
        wind_field=wind,
        controller=_controller("glide"),
    )
    first_row = rollout_evidence_row(first)
    next_state = np.asarray(json.loads(str(first_row["exit_state_vector"])), dtype=float)

    second = simulate_primitive_rollout(
        rollout_id="timing_continuity_second",
        initial_state=next_state,
        context=context,
        primitive=primitive,
        config=RolloutConfig(
            W_layer="W1",
            rollout_backend="model_backed_lqr",
            absolute_start_time_s=float(first_row["rollout_absolute_end_time_s"]),
            preserve_command_timing_state=True,
            initial_command_history_times_s_json=str(first_row["command_history_times_s_json"]),
            initial_command_norm_history_json=str(first_row["command_norm_history_json"]),
        ),
        wind_field=wind,
        controller=_controller("glide"),
    )
    second_row = rollout_evidence_row(second)
    second_times = json.loads(str(second_row["command_history_times_s_json"]))

    assert first_row["primitive_timing_state_continuity_status"] == "command_timing_state_reset_at_primitive_start"
    assert second_row["primitive_timing_state_continuity_status"] == "continued_from_previous_primitive_command_history"
    assert second_row["rollout_absolute_start_time_s"] == pytest.approx(first_row["rollout_absolute_end_time_s"])
    assert min(second_times) < float(second_row["rollout_absolute_start_time_s"])
    assert max(second_times) >= float(second_row["rollout_absolute_start_time_s"])


def test_rollout_row_logs_full_canonical_entry_state() -> None:
    state = _state()
    context, wind = _context_and_wind(state)

    evidence = simulate_primitive_rollout(
        rollout_id="full_state_logging",
        initial_state=state,
        context=context,
        primitive=primitive_by_id("glide"),
        config=RolloutConfig(W_layer="W1", rollout_backend="model_backed_lqr"),
        wind_field=wind,
        controller=_controller("glide"),
    )
    row = rollout_evidence_row(evidence)

    for name in (
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
    ):
        assert f"initial_{name}" in row


def test_implementation_and_plant_instances_change_rollout_smoke() -> None:
    state = _state()
    context, wind = _context_and_wind(state)
    primitive = primitive_by_id("lift_dwell_arc")
    config = RolloutConfig(W_layer="W3", rollout_backend="model_backed_lqr")

    nominal = simulate_primitive_rollout(
        rollout_id="nominal_instance",
        initial_state=state,
        context=context,
        primitive=primitive,
        config=config,
        wind_field=wind,
        controller=_controller(primitive.primitive_id),
        implementation_instance=implementation_instance_for_layer("W1", 1),
        plant_instance=plant_instance_for_layer("W1", 1),
    )
    randomised = simulate_primitive_rollout(
        rollout_id="randomised_instance",
        initial_state=state,
        context=context,
        primitive=primitive,
        config=config,
        wind_field=wind,
        controller=_controller(primitive.primitive_id),
        implementation_instance=implementation_instance_for_layer("W3", 1),
        plant_instance=plant_instance_for_layer("W3", 1),
    )

    assert (
        nominal.energy_residual_m != randomised.energy_residual_m
        or nominal.max_abs_surface_rad != randomised.max_abs_surface_rad
    )
