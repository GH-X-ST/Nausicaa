from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np
import pytest

from linearisation import linearise_trim
from lqr_controller import (
    ACTIVE_TIMING_AWARE_ROLE,
    LQR_SYNTHESIS_SOLVED,
    TIMING_STATE_HISTORY_BACKED,
    TIMING_STATE_INITIALISED,
    TIMING_AUGMENTATION_TYPE,
    TimingAwareControllerState,
    compare_timing_aware_vs_baseline_nominal,
    controller_is_active_timing_aware_w01,
    initialised_timing_state_for_controller,
    lqr_command_for_state,
    lqr_controller_for_primitive_id,
    lqr_controller_metadata_row,
    synthesize_baseline_trim_lqr_controller,
    synthesis_audit_row,
    timing_augmented_lqr_design_row,
)
from lqr_linearisation import LQR_LOCAL_OPERATING_SPEED_GRID_M_S, LQR_STATE_MASK, ZERO_POSITION_GAIN_STATES
from prim_cat import ACTIVE_PRIMITIVE_IDS, primitive_by_id
from state_contract import STATE_INDEX
from trim_solver import TrimTarget


def test_lqr_controller_audit_contract_for_all_primitives() -> None:
    assert ACTIVE_PRIMITIVE_IDS == (
        "glide",
        "recovery",
        "lift_entry",
        "lift_dwell_arc",
        "mild_turn_left",
        "mild_turn_right",
        "energy_retaining_bank",
        "safe_exit_or_recovery_handoff",
    )

    for primitive_id in ACTIVE_PRIMITIVE_IDS:
        controller = lqr_controller_for_primitive_id(primitive_id)
        metadata = lqr_controller_metadata_row(controller)
        audit = synthesis_audit_row(primitive_by_id(primitive_id))

        assert controller.controller_family == "lqr"
        assert controller.controller_id.startswith(f"lqrta_{primitive_id}_")
        assert controller.controller_version == "predictor_compensated_augmented_discrete_lqr_v1"
        assert controller_is_active_timing_aware_w01(controller)
        assert metadata["controller_design_role"] == ACTIVE_TIMING_AWARE_ROLE
        assert metadata["timing_augmentation_type"] == TIMING_AUGMENTATION_TYPE
        assert controller.lqr_synthesis_status == LQR_SYNTHESIS_SOLVED
        assert controller.reduced_order_lqr is True
        assert controller.actuator_state_count == 3
        assert controller.command_delay_steps >= 1
        assert controller.command_delay_state_count == 3 * controller.command_delay_steps
        assert controller.augmented_state_size > controller.reduced_state_size
        assert controller.augmented_input_size == 3
        assert controller.augmented_A_checksum
        assert controller.augmented_B_checksum
        assert controller.augmented_gain_checksum
        assert tuple(json.loads(metadata["lqr_state_mask_json"])) == LQR_STATE_MASK
        assert metadata["zero_position_gain_expansion_status"] == "zero_position_gains_verified"
        assert metadata["sampled_data_check_status"] == "sampled_stable"
        assert metadata["latency_actuator_survival_status"] == "timing_augmented_discrete_lqr_solved"
        assert audit["primitive_id"] == primitive_id
        assert audit["full_state_care_status"] in {"solved", "unsuitable_use_reduced_order"}
        assert math.isfinite(float(metadata["care_residual_norm"]))
        assert float(metadata["care_residual_norm"]) < 1.0e-6
        assert float(metadata["sampled_data_spectral_radius"]) < 1.0
        assert float(metadata["augmented_closed_loop_spectral_radius"]) < 1.0
        design_row = timing_augmented_lqr_design_row(controller)
        assert design_row["controller_design_role"] == ACTIVE_TIMING_AWARE_ROLE
        assert design_row["delayed_state_lqr_augmentation_status"] == (
            "predictor_compensation_only_no_full_delayed_state_augmentation"
        )

        gain = np.asarray(controller.k_gain_matrix, dtype=float)
        assert gain.shape == (3, 15)
        for state_name in ZERO_POSITION_GAIN_STATES:
            assert np.allclose(gain[:, STATE_INDEX[state_name]], 0.0)


def test_blocked_lqr_controller_cannot_emit_executable_zero_command() -> None:
    controller = replace(
        lqr_controller_for_primitive_id("glide"),
        lqr_synthesis_status="blocked_lqr_synthesis",
        lqr_blocked_reason="unit_test_blocked",
    )

    with pytest.raises(RuntimeError, match="blocked LQR controller"):
        lqr_command_for_state(controller=controller, state_vector=np.zeros(15))


def test_active_trim_and_lqr_require_local_speed_not_global_default() -> None:
    with pytest.raises(ValueError, match="explicit TrimTarget"):
        linearise_trim()

    assert LQR_LOCAL_OPERATING_SPEED_GRID_M_S == (
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        4.5,
        5.0,
        5.5,
        6.0,
        6.5,
        7.0,
        7.5,
        8.0,
        8.5,
        9.0,
    )
    launch_speed_controller = lqr_controller_for_primitive_id("glide", local_reference_speed_m_s=3.0)
    inflight_speed_controller = lqr_controller_for_primitive_id("glide", local_reference_speed_m_s=6.5)
    launch_ref_speed = float(np.linalg.norm(np.asarray(launch_speed_controller.reference_state_vector)[6:9]))
    inflight_ref_speed = float(np.linalg.norm(np.asarray(inflight_speed_controller.reference_state_vector)[6:9]))

    assert np.isclose(launch_ref_speed, 3.0)
    assert np.isclose(inflight_ref_speed, 6.5)
    assert not np.isclose(launch_ref_speed, 6.5)
    assert launch_speed_controller.linearisation_source == "gain_scheduled_passive_speed_operating_point_v2"
    assert launch_speed_controller.linearisation_id != inflight_speed_controller.linearisation_id
    assert linearise_trim(target=TrimTarget(speed_m_s=4.8)).f_trim.shape == (15,)


def test_lqr_command_reports_saturation_from_unclipped_raw_surface_request() -> None:
    base = lqr_controller_for_primitive_id("glide")
    controller = replace(
        base,
        controller_design_role="unit_test_non_timing_controller",
        reference_command_vector=(0.0, float(np.deg2rad(-90.0)), 0.0),
        k_gain_matrix=tuple(tuple(0.0 for _ in range(15)) for _ in range(3)),
    )
    state = np.asarray(controller.reference_state_vector, dtype=float)

    command = lqr_command_for_state(controller=controller, state_vector=state)

    assert command.saturation_applied is True
    assert np.isclose(command.raw_command_rad[1], np.deg2rad(-90.0))
    assert np.isclose(command.command_rad[1], np.deg2rad(-32.0))
    assert command.command_norm[1] == -1.0


def test_timing_aware_controller_ids_are_distinct_from_superseded_baseline() -> None:
    primitive = primitive_by_id("glide")
    timing_aware = lqr_controller_for_primitive_id("glide")
    baseline = synthesize_baseline_trim_lqr_controller(primitive)

    assert timing_aware.controller_id != baseline.controller_id
    assert timing_aware.controller_design_role == ACTIVE_TIMING_AWARE_ROLE
    assert baseline.controller_design_role == "superseded_baseline_not_active_w01"
    assert timing_aware.augmented_gain_checksum != ""


def test_timing_aware_command_path_is_distinct_under_nominal_delay() -> None:
    for primitive_id in ("glide", "lift_entry"):
        comparison = compare_timing_aware_vs_baseline_nominal(primitive_by_id(primitive_id))
        assert comparison["timing_aware_controller_id"].startswith(f"lqrta_{primitive_id}_")
        assert comparison["baseline_controller_id"].startswith(f"lqr_{primitive_id}_")
        assert comparison["timing_aware_role"] == ACTIVE_TIMING_AWARE_ROLE
        assert comparison["baseline_role"] == "superseded_baseline_not_active_w01"
        assert comparison["command_delta_norm"] > 0.0


def test_timing_aware_command_uses_history_backed_fifo_when_supplied() -> None:
    controller = lqr_controller_for_primitive_id("glide")
    state = np.asarray(controller.reference_state_vector, dtype=float).copy()
    state[STATE_INDEX["theta"]] += np.deg2rad(1.0)
    reference = tuple(float(value) for value in controller.reference_command_vector)
    fifo_command = tuple(float(value + 0.02) for value in reference)
    timing_state = TimingAwareControllerState(
        command_fifo_rad=tuple(fifo_command for _ in range(controller.command_delay_steps)),
        last_requested_command_rad=fifo_command,
        last_applied_command_rad=fifo_command,
        predictor_reference_command_rad=reference,
        current_surface_state_rad=tuple(float(state[STATE_INDEX[name]]) for name in ("delta_a", "delta_e", "delta_r")),
        timing_state_source=TIMING_STATE_HISTORY_BACKED,
    )

    compatibility = lqr_command_for_state(controller=controller, state_vector=state)
    initialised = lqr_command_for_state(
        controller=controller,
        state_vector=state,
        timing_state=initialised_timing_state_for_controller(controller, state),
    )
    history_backed = lqr_command_for_state(
        controller=controller,
        state_vector=state,
        timing_state=timing_state,
    )

    assert compatibility.timing_state_source == TIMING_STATE_INITIALISED
    assert initialised.timing_state_source == TIMING_STATE_INITIALISED
    assert history_backed.timing_state_source == TIMING_STATE_HISTORY_BACKED
    assert not np.allclose(history_backed.command_rad, initialised.command_rad)
