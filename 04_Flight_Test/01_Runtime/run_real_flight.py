from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from calibration_profile import ACTIVE_CALIBRATION_PROFILE, calibration_profile_for_runtime_values
from flight_config import (
    CONTROLLER_ROOT,
    DEFAULT_REAL_FLIGHT_LIBRARY_TIER,
    DEFAULT_VICON_ATTITUDE_OFFSET_RAD,
    DEFAULT_VICON_ATTITUDE_SIGNS,
    DEFAULT_VICON_POSITION_OFFSET_M,
    REAL_FLIGHT_LIBRARY_TIER_SELECTION_REASON,
    FlightRuntimeConfig,
    LAUNCH_HANDOFF_DURATION_S,
    default_run_label,
)
from flight_logger import FlightLogger
from frozen_flight_controller import FrozenFlightController
from exit_gate import evaluate_exit_gate, exit_gate_bounds_manifest
from launch_gate import (
    LAUNCH_GATE_X_W_M,
    LAUNCH_TRIGGER_X_W_M,
    evaluate_launch_gate,
    evaluate_launch_plane_gate,
    interpolate_launch_plane_state,
    launch_gate_bounds_manifest,
)
from nano_serial import FakeNanoSerialTx, NanoSerialTx
from safety_monitor import evaluate_safety
from vicon_rigid_body import FanViconSample, LiveNausicaaViconRigidBody, ReplayNausicaaViconRigidBody

if str(CONTROLLER_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTROLLER_ROOT))

from real_flight_io import NausicaaViconStateAdapter, ViconArenaFrameTransform  # noqa: E402
from state_contract import STATE_INDEX, STATE_SIZE, state_dataframe_row  # noqa: E402


ACTIVE_RUNTIME_WAKE_AHEAD_S = 0.002
ACTIVE_METRIC_LOGGING_POLICY = "buffer_active_rows_flush_after_active_record"
ACTIVE_FAN_LOGGING_POLICY = "single_prelaunch_snapshot_only"


def run_real_flight(
    config: FlightRuntimeConfig,
    *,
    mode: str,
    controller: FrozenFlightController | None = None,
    run_root: Path | None = None,
    expected_visible_fan_range: tuple[int, int] | None = None,
) -> dict[str, object]:
    _validate_closed_loop_deployment_evidence(config=config, mode=mode)
    logger = FlightLogger(Path(run_root) if run_root is not None else config.run_root)
    if config.controller_mode not in {"closed_loop", "open_loop_neutral"}:
        raise ValueError("controller_mode must be 'closed_loop' or 'open_loop_neutral'.")
    if not np.isclose(float(config.launch_handoff_duration_s), LAUNCH_HANDOFF_DURATION_S, rtol=0.0, atol=1e-12):
        raise ValueError("launch_handoff_duration_s_not_0p040s")
    handoff_slots = float(config.launch_handoff_duration_s) / float(config.serial_period_s)
    if not np.isclose(handoff_slots, round(handoff_slots), rtol=0.0, atol=1e-9):
        raise ValueError("launch_handoff_duration_s_must_be_integer_serial_slots")
    controller = controller or FrozenFlightController(config)
    deferred_active_metric_rows: list[tuple[str, dict[str, object]]] = []
    adapter = NausicaaViconStateAdapter(
        derivative_cutoff_hz=config.derivative_cutoff_hz,
        body_rate_limit_rad_s=config.body_rate_limit_rad_s,
        body_rate_observer_window_frames=config.body_rate_observer_window_frames,
        actuator_tau_s=config.actuator_tau_s,
        command_delay_s=config.surface_command_delay_s,
        arena_transform=ViconArenaFrameTransform(
            position_offset_m=config.vicon_position_offset_m,
            yaw_alignment_rad=float(np.deg2rad(config.vicon_yaw_alignment_deg)),
            attitude_signs=config.vicon_attitude_signs,
            attitude_offset_rad=config.vicon_attitude_offset_rad,
        ),
    )
    tx = NanoSerialTx(config.serial_port, config.serial_baud) if mode in {"armed", "packet-smoke"} else FakeNanoSerialTx()
    vicon = (
        LiveNausicaaViconRigidBody(host=config.vicon_host, subject_name=config.vicon_subject_name)
        if mode in {"armed", "vicon-smoke"}
        else ReplayNausicaaViconRigidBody(dt_s=config.vicon_poll_period_s)
    )
    logger.write_manifest(
        "real_flight_runtime_manifest.json",
        {
            "mode": mode,
            "config": asdict(config),
            "deployment_library_tier_policy": {
                "default_real_flight_library_tier": DEFAULT_REAL_FLIGHT_LIBRARY_TIER,
                "selection_reason": REAL_FLIGHT_LIBRARY_TIER_SELECTION_REASON,
                "heavy_cluster_role": "compact_fallback_if_runtime_or_library_size_needs_priority",
            },
            "control_boundary": "vicon_rigid_body_to_canonical_state_to_frozen_governor_to_quantised_packet",
            "controller_mode": config.controller_mode,
            "calibration_profile": {
                "profile_id": config.calibration_profile_id,
                "profile_hash": config.calibration_profile_hash,
                "vicon_calibration_source": config.vicon_calibration_source,
                "active_profile_manifest": ACTIVE_CALIBRATION_PROFILE.to_manifest(),
            },
            "deployment_evidence_guard": {
                "required_for_armed_closed_loop": bool(config.deployment_evidence_required_for_armed_closed_loop),
                "manifest_path": Path(config.deployment_evidence_manifest_path).as_posix(),
                "status": _deployment_guard_status(config=config, mode=mode),
            },
            "open_loop_neutral_definition": (
                "launch_gate_and_state_logging_active_but_no_governor_decisions_no_memory_updates_zero_command_only"
                if config.controller_mode == "open_loop_neutral"
                else "not_active"
            ),
            "closed_loop_command_execution_policy": (
                "hybrid_scheduler_prepares_next_0p10s_primitive_before_boundary_from_predicted_boundary_state;"
                "time_critical_governor_commit_and_50hz_packet_send_precede_active_metric_flush;"
                "first_packet_and_50hz_lqr_slot_commands_are_recomputed_from_latest_vicon_state"
                if config.controller_mode == "closed_loop"
                else "not_active"
            ),
            "surface_marker_tracking_enabled": False,
            "latency_quantification_enabled": False,
            "servo_command_limit_norm": [-1.0, 1.0],
            "launch_trigger_policy": "wait_for_r5_launch_gate_before_active_record",
            "launch_handoff_policy": {
                "policy_version": str(config.launch_handoff_policy_version),
                "duration_s": float(config.launch_handoff_duration_s),
                "neutral_slots": int(round(float(config.launch_handoff_duration_s) / float(config.serial_period_s))),
                "active_primitive_finite_horizon_s": float(config.governor_period_s),
                "description": (
                    "after launch gate approval, hold neutral for the fixed handoff duration, "
                    "prepare the first primitive from the approved state, then emit the first "
                    "active command from the latest post-handoff Vicon state"
                ),
            },
            "launch_gate_bounds": launch_gate_bounds_manifest(
                body_rate_limits_rad_s=config.launch_gate_body_rate_limits_rad_s
            ),
            "launch_wait_timeout_s": float(config.launch_wait_timeout_s),
            "launch_gate_required_consecutive_frames": int(config.launch_gate_required_consecutive_frames),
            "rejected_launch_attempt_min_speed_m_s": float(config.rejected_launch_attempt_min_speed_m_s),
            "exit_gate_bounds": exit_gate_bounds_manifest(),
            "post_exit_neutral_tail_s": float(config.post_exit_neutral_tail_s),
            "runtime_rates": {
                "vicon_poll_hz": float(1.0 / config.vicon_poll_period_s),
                "serial_command_repeat_hz": float(1.0 / config.serial_period_s),
                "governor_decision_hz": float(1.0 / config.governor_period_s),
                "closed_loop_slot_command_hz": float(1.0 / config.serial_period_s),
                "active_runtime_wake_ahead_s": ACTIVE_RUNTIME_WAKE_AHEAD_S,
                "active_metric_logging_policy": ACTIVE_METRIC_LOGGING_POLICY,
                "active_fan_logging_policy": ACTIVE_FAN_LOGGING_POLICY,
                "launch_handoff_duration_s": float(config.launch_handoff_duration_s),
                "derivative_cutoff_hz": float(config.derivative_cutoff_hz),
                "body_rate_limit_rad_s": float(config.body_rate_limit_rad_s),
                "body_rate_observer_window_frames": int(config.body_rate_observer_window_frames),
                "launch_gate_rate_confidence_min": float(config.launch_gate_rate_confidence_min),
                "rate_policy": (
                    "Vicon is polled at the tracking rate; serial packets are still repeated "
                    "at the firmware-safe command period; governor selection remains 10 Hz; "
                    "prelaunch SO3 angular-rate observer remains warm through launch approval."
                ),
            },
            "vicon_arena_frame_transform": {
                "description": config.vicon_frame_description,
                "position_offset_m": tuple(float(value) for value in config.vicon_position_offset_m),
                "yaw_alignment_deg": float(config.vicon_yaw_alignment_deg),
                "attitude_signs_phi_theta_psi": tuple(float(value) for value in config.vicon_attitude_signs),
                "attitude_offset_rad_phi_theta_psi": tuple(float(value) for value in config.vicon_attitude_offset_rad),
                "attitude_sign_reason": "recovered_vicon_orientation_check_20260601_205149_pitch_and_yaw_reversed",
            },
            "experiment_case": {
                "case_id": config.experiment_case_id,
                "case_name": config.experiment_case_name,
                "layout_id": config.experiment_layout_id,
                "controller_mode": config.controller_mode,
                "memory_enabled": bool(config.experiment_memory_enabled),
                "throw_index": int(config.throw_index),
                "attempt_index": int(config.attempt_index),
            },
            "expected_visible_fan_range": expected_visible_fan_range,
        },
    )
    summary = {
        "mode": mode,
        "run_root": (Path(run_root) if run_root is not None else config.run_root).as_posix(),
        "experiment_case_id": config.experiment_case_id,
        "experiment_case_name": config.experiment_case_name,
        "experiment_layout_id": config.experiment_layout_id,
        "controller_mode": config.controller_mode,
        "throw_index": int(config.throw_index),
        "attempt_index": int(config.attempt_index),
        "valid_throw": False,
        "state_sample_count": 0,
        "controller_decision_count": 0,
        "packet_count": 0,
        "neutral_failsafe_count": 0,
        "open_loop_neutral_packet_count": 0,
        "slot_command_update_count": 0,
        "serial_write_error_count": 0,
        "serial_write_timeout_count": 0,
        "max_decision_time_s": 0.0,
        "launch_speed_m_s": 0.0,
        "launch_gate_approved": False,
        "launch_handoff_enabled": bool(float(config.launch_handoff_duration_s) > 0.0),
        "launch_handoff_policy_version": str(config.launch_handoff_policy_version),
        "launch_handoff_duration_s": float(config.launch_handoff_duration_s),
        "launch_handoff_completed": False,
        "launch_handoff_neutral_packet_count": 0,
        "first_active_command_elapsed_s": 0.0,
        "first_launch_decision_ready_before_handoff": False,
        "continuation_scheduler_policy": (
            "predict_boundary_prepare_next_primitive_commit_with_latest_vicon_state"
            if config.controller_mode == "closed_loop"
            else "not_active"
        ),
        "continuation_prepare_started_count": 0,
        "continuation_prepared_decision_count": 0,
        "continuation_commit_count": 0,
        "continuation_late_decision_count": 0,
        "active_metric_logging_policy": ACTIVE_METRIC_LOGGING_POLICY,
        "active_metric_buffered_row_count": 0,
        "active_metric_buffer_flush_count": 0,
        "active_fan_logging_policy": ACTIVE_FAN_LOGGING_POLICY,
        "active_runtime_wake_ahead_s": ACTIVE_RUNTIME_WAKE_AHEAD_S,
        "flight_cancelled": False,
        "cancellation_reason": "",
        "exit_gate_triggered": False,
        "termination_reason": "",
        "post_exit_neutral_packets": 0,
        "fan_visible_count_latest": 0,
        "fan_expected_count_ok_latest": False,
        "memory_update_observation_count": 0,
        "memory_cell_count": 0,
        "completed": False,
    }
    primitive_step_index = 0
    latest_decision = None
    pending_slot_packet: bytes | None = None
    latest_state = None
    started = time.perf_counter()
    next_governor_s = 0.0
    next_serial_s = 0.0
    decision_records: list[dict[str, object]] = []
    controller_decision_rows: list[dict[str, object]] = []
    terminal_record_appended = False
    continuation_executor = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="real_flight_governor_prepare")
        if config.controller_mode == "closed_loop"
        else None
    )
    continuation_future: Future[dict[str, object]] | None = None
    continuation_target_step_index = -1
    continuation_target_boundary_s = 0.0
    continuation_prepare_started_elapsed_s = 0.0
    continuation_prediction_dt_s = 0.0
    continuation_late_logged_step = -1

    def _start_continuation_prepare(
        *,
        state: np.ndarray,
        loop_elapsed_s: float,
        target_boundary_s: float,
        target_step_index: int,
    ) -> None:
        nonlocal continuation_future
        nonlocal continuation_target_step_index
        nonlocal continuation_target_boundary_s
        nonlocal continuation_prepare_started_elapsed_s
        nonlocal continuation_prediction_dt_s
        nonlocal continuation_late_logged_step
        if continuation_executor is None:
            return
        if continuation_future is not None and not continuation_future.done():
            return
        prediction_dt_s = max(0.0, float(target_boundary_s) - float(loop_elapsed_s))
        predicted_state = _predict_boundary_state(state, prediction_dt_s)
        continuation_target_step_index = int(target_step_index)
        continuation_target_boundary_s = float(target_boundary_s)
        continuation_prepare_started_elapsed_s = float(loop_elapsed_s)
        continuation_prediction_dt_s = float(prediction_dt_s)
        continuation_late_logged_step = -1
        summary["continuation_prepare_started_count"] = int(summary["continuation_prepare_started_count"]) + 1
        _append_runtime_event(
            logger,
            "continuation_decision_prepare_started",
            primitive_step_index=int(target_step_index),
            prepare_started_elapsed_s=float(loop_elapsed_s),
            target_boundary_s=float(target_boundary_s),
            prediction_dt_s=float(prediction_dt_s),
        )
        continuation_future = continuation_executor.submit(
            controller.prepare_continuation_decision,
            predicted_state,
            primitive_step_index=int(target_step_index),
            target_boundary_s=float(target_boundary_s),
            prepare_started_elapsed_s=float(loop_elapsed_s),
            prediction_dt_s=float(prediction_dt_s),
        )

    def _record_controller_decision(
        decision: object,
        *,
        decision_t_s: float,
        decision_state: np.ndarray,
        executed_primitive_step_index: int,
        scheduler_decision_source: str,
        scheduler_prepared_before_boundary: bool,
        scheduler_target_boundary_s: float,
        scheduler_prepare_started_elapsed_s: float,
        scheduler_prediction_dt_s: float,
        defer_metric_logging: bool = False,
    ) -> None:
        decision_records.append(
            {
                "t_s": float(decision_t_s),
                "state": decision_state.copy(),
                "expected_energy_residual_m": float(decision.expected_energy_residual_m),
                "primitive_variant_id": decision.primitive_variant_id,
            }
        )
        summary["controller_decision_count"] = int(summary["controller_decision_count"]) + 1
        summary["max_decision_time_s"] = max(
            float(summary["max_decision_time_s"]),
            float(decision.decision_time_s),
        )
        decision_row = {
            "t_host_s": time.perf_counter(),
            "decision_elapsed_s": float(decision_t_s),
            "executed_primitive_step_index": int(executed_primitive_step_index),
            "scheduler_policy": "hybrid_predict_boundary_prepare_then_commit_latest_vicon",
            "scheduler_decision_source": str(scheduler_decision_source),
            "scheduler_prepared_before_primitive_boundary": bool(scheduler_prepared_before_boundary),
            "scheduler_target_boundary_s": float(scheduler_target_boundary_s),
            "scheduler_prepare_started_elapsed_s": float(scheduler_prepare_started_elapsed_s),
            "scheduler_prediction_dt_s": float(scheduler_prediction_dt_s),
            "scheduler_commit_lag_s": float(decision_t_s) - float(scheduler_target_boundary_s),
            **asdict(decision),
        }
        controller_decision_rows.append(decision_row)
        _append_metric_row(
            logger,
            "controller_decisions.csv",
            decision_row,
            metric_buffer=deferred_active_metric_rows if defer_metric_logging else None,
        )

    try:
        tx.open()
        vicon.open()
        if mode == "packet-smoke":
            _run_packet_smoke(config=config, tx=tx, controller=controller, logger=logger)
            summary["completed"] = True
            return summary

        launched_state = _await_launch_gate(
            config=config,
            tx=tx,
            vicon=vicon,
            adapter=adapter,
            controller=controller,
            logger=logger,
            mode=mode,
            summary=summary,
            expected_visible_fan_range=expected_visible_fan_range,
        )
        if launched_state is None:
            if _write_packet_safe(
                tx,
                controller.neutral_packet(),
                logger=logger,
                summary=summary,
                event="launch_gate_cancel_neutral",
            ):
                summary["neutral_failsafe_count"] += 1
            summary["flight_cancelled"] = True
            summary["completed"] = False
            return summary
        latest_state = launched_state
        pending_launch_decision_state = launched_state.copy()
        summary["valid_throw"] = True
        summary["launch_speed_m_s"] = float(np.linalg.norm(latest_state[6:9]))
        started = time.perf_counter()
        next_governor_s = 0.0
        next_serial_s = 0.0
        launch_handoff_duration_s = float(config.launch_handoff_duration_s)
        _append_runtime_event(
            logger,
            "launch_handoff_start",
            launch_handoff_policy_version=str(config.launch_handoff_policy_version),
            launch_handoff_duration_s=launch_handoff_duration_s,
        )
        if config.controller_mode == "closed_loop":
            _append_runtime_event(
                logger,
                "first_decision_uses_approved_launch_state",
                source="launch_gate_interpolated_or_window_state",
            )
            prepared = controller.prepare_launch_handoff_decision(
                pending_launch_decision_state,
                primitive_step_index=0,
            )
            summary["first_launch_decision_ready_before_handoff"] = bool(
                prepared.get("ready", False)
                and float(prepared.get("decision_time_s", float("inf"))) <= launch_handoff_duration_s + 1e-12
            )
            _append_runtime_event(
                logger,
                "first_launch_decision_prepared",
                **prepared,
                launch_handoff_duration_s=launch_handoff_duration_s,
            )
            if not bool(summary["first_launch_decision_ready_before_handoff"]):
                reason = "first_launch_decision_missed_handoff_budget"
                summary["valid_throw"] = False
                summary["flight_cancelled"] = True
                summary["cancellation_reason"] = reason
                _append_runtime_event(
                    logger,
                    "launch_handoff_abort",
                    reason=reason,
                    prepared_decision=prepared,
                )
                _write_packet_safe(
                    tx,
                    controller.neutral_packet(),
                    logger=logger,
                    summary=summary,
                    event="launch_handoff_abort_neutral_packet",
                )
                summary["completed"] = False
                return summary

        handoff_sample_count = max(
            1,
            int(np.ceil(launch_handoff_duration_s / float(config.vicon_poll_period_s))),
        )
        for handoff_sample_index in range(handoff_sample_count):
            handoff_elapsed_s = min(
                launch_handoff_duration_s,
                float(handoff_sample_index) * float(config.vicon_poll_period_s),
            )
            while next_serial_s < launch_handoff_duration_s - 1e-12 and handoff_elapsed_s + 1e-12 >= next_serial_s:
                if _write_packet_safe(
                    tx,
                    controller.neutral_packet(),
                    logger=logger,
                    summary=summary,
                    event="launch_handoff_neutral_packet",
                ):
                    summary["launch_handoff_neutral_packet_count"] = (
                        int(summary["launch_handoff_neutral_packet_count"]) + 1
                    )
                next_serial_s += float(config.serial_period_s)

            sample, status = vicon.read_latest()
            if sample is None or not status.valid:
                reason = f"launch_handoff_abort:vicon_invalid:{status.reason}"
                summary["flight_cancelled"] = True
                summary["cancellation_reason"] = reason
                _append_runtime_event(
                    logger,
                    "launch_handoff_abort",
                    reason=reason,
                    handoff_elapsed_s=handoff_elapsed_s,
                )
                _write_packet_safe(
                    tx,
                    controller.neutral_packet(),
                    logger=logger,
                    summary=summary,
                    event="launch_handoff_abort_neutral_packet",
                )
                summary["completed"] = False
                return summary

            latest_state = adapter.update(sample, command_norm=controller.last_command_norm())
            estimator = adapter.estimator_status()
            safety = evaluate_safety(latest_state)
            exit_gate = evaluate_exit_gate(latest_state)
            summary["state_sample_count"] += 1
            logger.append_metric_row(
                "state_samples.csv",
                {
                    "t_host_s": time.perf_counter(),
                    "frame_number": status.frame_number,
                    "vicon_frame_rate_hz": status.frame_rate_hz,
                    "vicon_latency_s": status.vicon_latency_s,
                    **{f"estimator_{key}": value for key, value in estimator.items()},
                    **state_dataframe_row(latest_state),
                    **asdict(safety),
                    **{f"exit_gate_{key}": value for key, value in asdict(exit_gate).items()},
                },
            )
            if not safety.safe or not exit_gate.inside:
                reason = (
                    f"launch_handoff_abort:{safety.reason}"
                    if not safety.safe
                    else f"launch_handoff_abort:{exit_gate.reason}"
                )
                summary["flight_cancelled"] = True
                summary["cancellation_reason"] = reason
                _append_runtime_event(
                    logger,
                    "launch_handoff_abort",
                    reason=reason,
                    handoff_elapsed_s=handoff_elapsed_s,
                    **asdict(safety),
                    **{f"exit_gate_{key}": value for key, value in asdict(exit_gate).items()},
                )
                _write_packet_safe(
                    tx,
                    controller.neutral_packet(),
                    logger=logger,
                    summary=summary,
                    event="launch_handoff_abort_neutral_packet",
                )
                summary["completed"] = False
                return summary
            if mode in {"armed", "vicon-smoke"}:
                time.sleep(float(config.vicon_poll_period_s))

        summary["launch_handoff_completed"] = True
        _append_runtime_event(
            logger,
            "launch_handoff_complete",
            launch_handoff_duration_s=launch_handoff_duration_s,
            neutral_packet_count=int(summary["launch_handoff_neutral_packet_count"]),
        )
        started = time.perf_counter() - launch_handoff_duration_s
        next_serial_s = launch_handoff_duration_s
        next_governor_s = launch_handoff_duration_s
        if config.controller_mode == "closed_loop":
            latest_decision = controller.commit_prepared_launch_handoff_decision(latest_state)
            if not latest_decision.selected:
                reason = "first_launch_decision_missed_handoff_budget"
                summary["valid_throw"] = False
                summary["flight_cancelled"] = True
                summary["cancellation_reason"] = reason
                _append_runtime_event(
                    logger,
                    "launch_handoff_abort",
                    reason=reason,
                    latest_decision=asdict(latest_decision),
                )
                _write_packet_safe(
                    tx,
                    controller.neutral_packet(),
                    logger=logger,
                    summary=summary,
                    event="launch_handoff_abort_neutral_packet",
                )
                summary["completed"] = False
                return summary
            pending_slot_packet = latest_decision.packet_bytes
            _record_controller_decision(
                latest_decision,
                decision_t_s=0.0,
                decision_state=pending_launch_decision_state,
                executed_primitive_step_index=0,
                scheduler_decision_source="initial_launch_precomputed_before_release",
                scheduler_prepared_before_boundary=True,
                scheduler_target_boundary_s=launch_handoff_duration_s,
                scheduler_prepare_started_elapsed_s=0.0,
                scheduler_prediction_dt_s=0.0,
                defer_metric_logging=True,
            )
            pending_launch_decision_state = None
            primitive_step_index = 1
            next_governor_s = launch_handoff_duration_s + float(config.governor_period_s)
            _start_continuation_prepare(
                state=latest_state,
                loop_elapsed_s=launch_handoff_duration_s,
                target_boundary_s=next_governor_s,
                target_step_index=primitive_step_index,
            )

        while (time.perf_counter() - started) <= float(config.max_duration_s):
            loop_elapsed_s = time.perf_counter() - started
            sample, status = vicon.read_latest()
            if sample is None or not status.valid:
                packet = controller.neutral_packet()
                if loop_elapsed_s + 1e-12 >= next_serial_s:
                    if _write_packet_safe(
                        tx,
                        packet,
                        logger=logger,
                        summary=summary,
                        event="vicon_invalid_neutral_command",
                    ):
                        summary["neutral_failsafe_count"] += 1
                    next_serial_s += float(config.serial_period_s)
                _append_runtime_event(
                    logger,
                    "vicon_invalid_neutral_command",
                    metric_buffer=deferred_active_metric_rows,
                    reason=status.reason,
                )
                if mode in {"armed", "vicon-smoke"}:
                    _sleep_until_next_runtime_poll(
                        config=config,
                        started=started,
                        next_serial_s=next_serial_s,
                        next_governor_s=next_governor_s,
                    )
                continue

            latest_state = adapter.update(sample, command_norm=controller.last_command_norm())
            estimator = adapter.estimator_status()
            safety = evaluate_safety(latest_state)
            exit_gate = evaluate_exit_gate(latest_state)
            summary["state_sample_count"] += 1
            state_row = {
                "t_host_s": time.perf_counter(),
                "frame_number": status.frame_number,
                "vicon_frame_rate_hz": status.frame_rate_hz,
                "vicon_latency_s": status.vicon_latency_s,
                **{f"estimator_{key}": value for key, value in estimator.items()},
                **state_dataframe_row(latest_state),
                **asdict(safety),
                **{f"exit_gate_{key}": value for key, value in asdict(exit_gate).items()},
            }
            action_elapsed_s = time.perf_counter() - started

            if not exit_gate.inside:
                _append_metric_row(
                    logger,
                    "state_samples.csv",
                    state_row,
                    metric_buffer=deferred_active_metric_rows,
                )
                if latest_decision is not None:
                    decision_records.append(
                        {
                            "t_s": float(action_elapsed_s),
                            "state": latest_state.copy(),
                            "expected_energy_residual_m": float(latest_decision.expected_energy_residual_m),
                            "primitive_variant_id": latest_decision.primitive_variant_id,
                        }
                    )
                    terminal_record_appended = True
                _send_neutral_tail(
                    config=config,
                    tx=tx,
                    controller=controller,
                    logger=logger,
                    mode=mode,
                    summary=summary,
                    reason=exit_gate.reason,
                    metric_buffer=deferred_active_metric_rows,
                )
                summary["exit_gate_triggered"] = True
                summary["termination_reason"] = str(exit_gate.reason)
                _append_runtime_event(
                    logger,
                    "exit_gate_terminate_active_record",
                    metric_buffer=deferred_active_metric_rows,
                    **asdict(exit_gate),
                )
                break

            if config.controller_mode == "closed_loop" and action_elapsed_s + 1e-12 >= next_governor_s:
                if (
                    continuation_future is not None
                    and continuation_target_step_index == primitive_step_index
                    and continuation_future.done()
                ):
                    prepared_status = continuation_future.result()
                    summary["continuation_prepared_decision_count"] = (
                        int(summary["continuation_prepared_decision_count"]) + 1
                    )
                    latest_decision = controller.commit_prepared_continuation_decision(
                        latest_state,
                        primitive_step_index=primitive_step_index,
                    )
                    pending_slot_packet = latest_decision.packet_bytes
                    commit_lag_s = float(action_elapsed_s) - float(continuation_target_boundary_s)
                    scheduler_source = (
                        "prepared_during_previous_primitive_window"
                        if commit_lag_s <= 1e-12
                        else "late_prepared_after_boundary"
                    )
                    _record_controller_decision(
                        latest_decision,
                        decision_t_s=float(action_elapsed_s),
                        decision_state=latest_state,
                        executed_primitive_step_index=primitive_step_index,
                        scheduler_decision_source=scheduler_source,
                        scheduler_prepared_before_boundary=commit_lag_s <= 1e-12,
                        scheduler_target_boundary_s=continuation_target_boundary_s,
                        scheduler_prepare_started_elapsed_s=continuation_prepare_started_elapsed_s,
                        scheduler_prediction_dt_s=continuation_prediction_dt_s,
                        defer_metric_logging=True,
                    )
                    summary["continuation_commit_count"] = int(summary["continuation_commit_count"]) + 1
                    _append_runtime_event(
                        logger,
                        "continuation_decision_committed",
                        metric_buffer=deferred_active_metric_rows,
                        primitive_step_index=int(primitive_step_index),
                        scheduler_decision_source=scheduler_source,
                        prepared_ready=bool(prepared_status.get("ready", False)),
                        selected=bool(latest_decision.selected),
                        reason=str(latest_decision.reason),
                        target_boundary_s=float(continuation_target_boundary_s),
                        commit_elapsed_s=float(action_elapsed_s),
                        commit_lag_s=float(commit_lag_s),
                        decision_time_s=float(latest_decision.decision_time_s),
                    )
                    primitive_step_index += 1
                    next_governor_s += float(config.governor_period_s)
                    continuation_future = None
                    _start_continuation_prepare(
                        state=latest_state,
                        loop_elapsed_s=float(action_elapsed_s),
                        target_boundary_s=next_governor_s,
                        target_step_index=primitive_step_index,
                    )
                else:
                    if continuation_future is None:
                        _start_continuation_prepare(
                            state=latest_state,
                            loop_elapsed_s=float(action_elapsed_s),
                            target_boundary_s=next_governor_s,
                            target_step_index=primitive_step_index,
                        )
                    if continuation_late_logged_step != primitive_step_index:
                        continuation_late_logged_step = primitive_step_index
                        summary["continuation_late_decision_count"] = (
                            int(summary["continuation_late_decision_count"]) + 1
                        )
                        _append_runtime_event(
                            logger,
                            "continuation_decision_late",
                            metric_buffer=deferred_active_metric_rows,
                            primitive_step_index=int(primitive_step_index),
                            target_boundary_s=float(next_governor_s),
                            elapsed_s=float(action_elapsed_s),
                            active_primitive_variant_id=(
                                latest_decision.primitive_variant_id if latest_decision is not None else ""
                            ),
                            action="continue_streaming_active_primitive_until_prepared_decision_ready",
                        )

            action_elapsed_s = time.perf_counter() - started
            if action_elapsed_s + 1e-12 >= next_serial_s:
                if config.controller_mode == "open_loop_neutral":
                    packet = controller.neutral_packet()
                    event = "active_open_loop_neutral_packet"
                    summary["open_loop_neutral_packet_count"] = int(summary["open_loop_neutral_packet_count"]) + 1
                else:
                    if pending_slot_packet is not None:
                        packet = pending_slot_packet
                        pending_slot_packet = None
                        event = "active_command_packet_primitive_entry_slot"
                    else:
                        packet = controller.packet_for_active_slot_command(latest_state)
                        event = "active_command_packet_slot_update"
                    summary["slot_command_update_count"] = int(summary["slot_command_update_count"]) + 1
                if float(summary.get("first_active_command_elapsed_s", 0.0)) <= 0.0:
                    summary["first_active_command_elapsed_s"] = float(action_elapsed_s)
                    _append_runtime_event(
                        logger,
                        "first_active_command",
                        metric_buffer=deferred_active_metric_rows,
                        elapsed_s=float(action_elapsed_s),
                        source_event=event,
                        controller_mode=config.controller_mode,
                    )
                _write_packet_safe(
                    tx,
                    packet,
                    logger=logger,
                    summary=summary,
                    event=event,
                )
                next_serial_s += float(config.serial_period_s)

            _append_metric_row(
                logger,
                "state_samples.csv",
                state_row,
                metric_buffer=deferred_active_metric_rows,
            )

            if mode in {"armed", "vicon-smoke"}:
                _sleep_until_next_runtime_poll(
                    config=config,
                    started=started,
                    next_serial_s=next_serial_s,
                    next_governor_s=next_governor_s,
                )

        if bool(summary["launch_gate_approved"]) and config.controller_mode == "closed_loop":
            if latest_decision is not None and latest_state is not None and not terminal_record_appended:
                decision_records.append(
                    {
                        "t_s": float(time.perf_counter() - started),
                        "state": latest_state.copy(),
                        "expected_energy_residual_m": float(latest_decision.expected_energy_residual_m),
                        "primitive_variant_id": latest_decision.primitive_variant_id,
                    }
                )
            memory_summary = controller.update_memory_from_decision_records(decision_records)
            summary["memory_update_observation_count"] = int(memory_summary.observation_count)
            summary["memory_cell_count"] = int(memory_summary.updated_cell_count)
            logger.append_metric_row("memory_update_summary.csv", asdict(memory_summary))
        elif bool(summary["launch_gate_approved"]):
            logger.append_metric_row(
                "memory_update_summary.csv",
                {
                    "controller_mode": config.controller_mode,
                    "memory_update_status": "skipped_open_loop_neutral",
                    "observation_count": 0,
                    "updated_cell_count": 0,
                },
            )
        summary["completed"] = True
        return summary
    finally:
        if latest_state is not None:
            _write_packet_safe(
                tx,
                controller.neutral_packet(),
                logger=logger,
                summary=summary,
                event="final_neutral_packet",
                quiet=True,
            )
        if continuation_executor is not None:
            continuation_executor.shutdown(wait=True, cancel_futures=True)
        tx.close()
        vicon.close()
        if deferred_active_metric_rows:
            buffered_count = len(deferred_active_metric_rows)
            summary["active_metric_buffered_row_count"] = int(buffered_count)
            flushed_count = _flush_metric_buffer(logger, deferred_active_metric_rows)
            summary["active_metric_buffer_flush_count"] = (
                int(summary.get("active_metric_buffer_flush_count", 0)) + int(flushed_count)
            )
        posthoc_score = _real_flight_posthoc_score_row(
            config=config,
            summary=summary,
            controller_decision_rows=controller_decision_rows,
            latest_state=latest_state,
        )
        summary.update(
            {
                "posthoc_accumulated_selected_score": posthoc_score["accumulated_selected_score"],
                "posthoc_executed_selected_decision_count": posthoc_score[
                    "executed_selected_decision_count"
                ],
                "posthoc_memory_history_bucket": posthoc_score["memory_history_bucket"],
                "posthoc_score_source": posthoc_score["posthoc_score_source"],
                "posthoc_final_observable_specific_energy_m": posthoc_score[
                    "final_observable_specific_energy_m"
                ],
            }
        )
        logger.append_metric_row("posthoc_throw.csv", posthoc_score)
        logger.write_manifest("real_flight_runtime_summary.json", summary)
        logger.write_report(
            "real_flight_runtime_report.md",
            [
                "# Real Flight Runtime Report",
                f"- Mode: `{mode}`",
                f"- Run root: `{summary['run_root']}`",
                f"- Experiment case: `{summary['experiment_case_id']}`",
                f"- Controller mode: `{summary['controller_mode']}`",
                f"- Valid throw: `{summary['valid_throw']}`",
                f"- Launch gate approved: `{summary['launch_gate_approved']}`",
                f"- Launch handoff policy: `{summary['launch_handoff_policy_version']}`",
                f"- Launch handoff duration (s): `{float(summary['launch_handoff_duration_s']):.3f}`",
                f"- Launch handoff completed: `{summary['launch_handoff_completed']}`",
                f"- Launch handoff neutral packets: `{summary['launch_handoff_neutral_packet_count']}`",
                f"- First active command elapsed (s): `{float(summary['first_active_command_elapsed_s']):.3f}`",
                f"- Launch speed (m/s): `{float(summary['launch_speed_m_s']):.3f}`",
                f"- Flight cancelled: `{summary['flight_cancelled']}`",
                f"- Cancellation reason: `{summary['cancellation_reason']}`",
                f"- Exit gate triggered: `{summary['exit_gate_triggered']}`",
                f"- Termination reason: `{summary['termination_reason']}`",
                f"- State samples: `{summary['state_sample_count']}`",
                f"- Controller decisions: `{summary['controller_decision_count']}`",
                f"- Packets sent: `{summary['packet_count']}`",
                f"- Neutral failsafe commands: `{summary['neutral_failsafe_count']}`",
                f"- Open-loop neutral packets: `{summary['open_loop_neutral_packet_count']}`",
                f"- Closed-loop slot command updates: `{summary['slot_command_update_count']}`",
                f"- Active metric logging policy: `{summary['active_metric_logging_policy']}`",
                f"- Active metric buffered rows: `{summary['active_metric_buffered_row_count']}`",
                f"- Active fan logging policy: `{summary['active_fan_logging_policy']}`",
                f"- Active runtime wake-ahead (s): `{float(summary['active_runtime_wake_ahead_s']):.3f}`",
                f"- Serial write errors: `{summary['serial_write_error_count']}`",
                f"- Serial write timeouts: `{summary['serial_write_timeout_count']}`",
                f"- Post-exit neutral packets: `{summary['post_exit_neutral_packets']}`",
                f"- Latest visible fan count: `{summary['fan_visible_count_latest']}`",
                f"- Fan expected count OK: `{summary['fan_expected_count_ok_latest']}`",
                f"- Memory update observations: `{summary['memory_update_observation_count']}`",
                f"- Memory cells: `{summary['memory_cell_count']}`",
                f"- Max decision time (s): `{float(summary['max_decision_time_s']):.6f}`",
                f"- Posthoc accumulated selected score: `{float(summary['posthoc_accumulated_selected_score']):.6f}`",
                f"- Posthoc executed selected decisions: `{summary['posthoc_executed_selected_decision_count']}`",
                f"- Posthoc memory history bucket: `{summary['posthoc_memory_history_bucket']}`",
                f"- Posthoc score source: `{summary['posthoc_score_source']}`",
            ],
        )
        logger.close()


def _real_flight_posthoc_score_row(
    *,
    config: FlightRuntimeConfig,
    summary: dict[str, object],
    controller_decision_rows: list[dict[str, object]],
    latest_state: np.ndarray | None,
) -> dict[str, object]:
    selected_rows = [row for row in controller_decision_rows if _truthy(row.get("selected", False))]
    final_energy = _real_flight_specific_energy_m(latest_state) if latest_state is not None else float("nan")
    final_z = float(latest_state[STATE_INDEX["z_w"]]) if latest_state is not None else float("nan")
    return {
        "experiment_case_id": config.experiment_case_id,
        "experiment_case_name": config.experiment_case_name,
        "experiment_layout_id": config.experiment_layout_id,
        "controller_mode": config.controller_mode,
        "memory_enabled": bool(config.experiment_memory_enabled),
        "memory_history_bucket": _real_flight_memory_history_bucket(config),
        "throw_index": int(config.throw_index),
        "attempt_index": int(config.attempt_index),
        "valid_throw": bool(summary.get("valid_throw", False)),
        "launch_gate_approved": bool(summary.get("launch_gate_approved", False)),
        "termination_reason": str(summary.get("termination_reason", "")),
        "cancellation_reason": str(summary.get("cancellation_reason", "")),
        "launch_speed_m_s": _safe_float(summary.get("launch_speed_m_s", 0.0)),
        "final_observable_specific_energy_m": float(final_energy),
        "final_observable_z_w_m": float(final_z),
        "controller_decision_count": int(summary.get("controller_decision_count", 0)),
        "executed_selected_decision_count": int(len(selected_rows)),
        "blocked_or_neutral_decision_count": int(len(controller_decision_rows) - len(selected_rows)),
        "accumulated_selected_score": _sum_decision_field(selected_rows, "selected_score"),
        "accumulated_base_library_score_component": _sum_decision_field(
            selected_rows,
            "selected_base_library_score_component",
        ),
        "accumulated_mission_score_component": _sum_decision_field(
            selected_rows,
            "selected_mission_score_component",
        ),
        "accumulated_exploration_score_component": _sum_decision_field(
            selected_rows,
            "selected_exploration_score_component",
        ),
        "accumulated_memory_score_component": _sum_decision_field(
            selected_rows,
            "selected_memory_score_component",
        ),
        "accumulated_calibrated_regime_mismatch_score_component": _sum_decision_field(
            selected_rows,
            "selected_calibrated_regime_mismatch_score_component",
        ),
        "posthoc_score_source": (
            "controller_decisions_selected_rows"
            if config.controller_mode == "closed_loop"
            else "open_loop_neutral_zero_controller_decisions"
        ),
        "claim_status": "real_flight_posthoc_score_audit_not_runtime_control",
    }


def _real_flight_memory_history_bucket(config: FlightRuntimeConfig) -> str:
    if config.controller_mode == "open_loop_neutral":
        return "open_loop"
    if not bool(config.experiment_memory_enabled):
        return "no_memory"
    prior_valid_memory_throws = max(0, int(config.throw_index) - 1)
    if prior_valid_memory_throws <= 0:
        return "h0"
    if prior_valid_memory_throws <= 3:
        return "h1_3"
    if prior_valid_memory_throws <= 10:
        return "h4_10"
    return "h11_30_plus"


def _real_flight_specific_energy_m(state: np.ndarray) -> float:
    speed = float(
        np.linalg.norm(
            state[
                [
                    STATE_INDEX["u"],
                    STATE_INDEX["v"],
                    STATE_INDEX["w"],
                ]
            ]
        )
    )
    return float(state[STATE_INDEX["z_w"]] + speed * speed / (2.0 * 9.80665))


def _sum_decision_field(rows: list[dict[str, object]], field: str) -> float:
    return float(sum(_safe_float(row.get(field, 0.0)) for row in rows))


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def _truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _await_launch_gate(
    *,
    config: FlightRuntimeConfig,
    tx: NanoSerialTx | FakeNanoSerialTx,
    vicon: LiveNausicaaViconRigidBody | ReplayNausicaaViconRigidBody,
    adapter: NausicaaViconStateAdapter,
    controller: FrozenFlightController,
    logger: FlightLogger,
    mode: str,
    summary: dict[str, object],
    expected_visible_fan_range: tuple[int, int] | None,
) -> np.ndarray | None:
    started = time.perf_counter()
    next_serial_s = 0.0
    consecutive_approved = 0
    required_consecutive = max(1, int(config.launch_gate_required_consecutive_frames))
    latest_gate_reason = "not_evaluated"
    next_invalid_log_s = 0.0
    next_console_status_s = 0.0
    previous_state: np.ndarray | None = None
    rate_confidence_min = float(config.launch_gate_rate_confidence_min)
    fan_snapshot_logged = False

    while (time.perf_counter() - started) <= float(config.launch_wait_timeout_s):
        elapsed_s = time.perf_counter() - started
        sample, status = vicon.read_latest()
        if sample is None or not status.valid:
            latest_gate_reason = status.reason
            if elapsed_s + 1e-12 >= next_console_status_s:
                print(
                    f"[LAUNCH_WAIT] t={elapsed_s:.1f}s "
                    f"Vicon invalid: {status.reason}; holding neutral"
                )
                next_console_status_s = elapsed_s + 1.0
            next_serial_s = _send_neutral_if_due(
                tx=tx,
                controller=controller,
                logger=logger,
                summary=summary,
                elapsed_s=elapsed_s,
                next_serial_s=next_serial_s,
                serial_period_s=float(config.serial_period_s),
            )
            if elapsed_s + 1e-12 >= next_invalid_log_s:
                logger.append_metric_row(
                    "prelaunch_events.csv",
                    {
                        "t_host_s": time.perf_counter(),
                        "event": "vicon_invalid_waiting_for_launch",
                        "reason": status.reason,
                    },
                )
                next_invalid_log_s = elapsed_s + 0.50
            if mode in {"armed", "vicon-smoke"}:
                time.sleep(float(config.vicon_poll_period_s))
            continue

        state = adapter.update(sample, command_norm=controller.last_command_norm())
        estimator = adapter.estimator_status()
        if not fan_snapshot_logged:
            _append_fan_positions(
                logger=logger,
                vicon=vicon,
                adapter=adapter,
                phase="prelaunch",
                expected_visible_fan_range=expected_visible_fan_range,
                summary=summary,
            )
            fan_snapshot_logged = True
        full_window_gate = evaluate_launch_gate(
            state,
            body_rate_limits_rad_s=config.launch_gate_body_rate_limits_rad_s,
        )
        launch_plane_state = interpolate_launch_plane_state(previous_state, state)
        crossed_launch_plane = launch_plane_state is not None
        plane_gate = (
            evaluate_launch_plane_gate(
                launch_plane_state,
                body_rate_limits_rad_s=config.launch_gate_body_rate_limits_rad_s,
            )
            if crossed_launch_plane and launch_plane_state is not None
            else None
        )
        if full_window_gate.approved:
            gate = full_window_gate
            trigger_approved = True
            trigger_source = "valid_r5_launch_window"
            launch_state = state
        elif plane_gate is not None:
            gate = plane_gate
            trigger_approved = bool(plane_gate.approved)
            trigger_source = "valid_interpolated_launch_plane" if trigger_approved else "crossed_plane_rejected"
            launch_state = launch_plane_state if trigger_approved else None
        else:
            gate = full_window_gate
            trigger_approved = False
            trigger_source = "tracking"
            launch_state = None
        rate_confidence = float(estimator.get("rate_confidence", 0.0))
        rate_confidence_ok = bool(rate_confidence >= rate_confidence_min)
        base_trigger_approved = bool(trigger_approved)
        previous_consecutive_approved = int(consecutive_approved)
        if trigger_approved and not rate_confidence_ok:
            trigger_approved = False
            trigger_source = f"{trigger_source}_rate_confidence_wait"
            launch_state = None
            latest_gate_reason = "rate_confidence_below_launch_gate"
        else:
            latest_gate_reason = gate.reason
        consecutive_approved = consecutive_approved + 1 if trigger_approved else 0
        if elapsed_s + 1e-12 >= next_console_status_s:
            speed_m_s = float(np.linalg.norm(state[6:9]))
            print(
                f"[LAUNCH_WAIT] t={elapsed_s:.1f}s trigger={trigger_source} gate={gate.reason} "
                f"approved={trigger_approved} consecutive={consecutive_approved}/{required_consecutive} "
                f"rate_conf={rate_confidence:.2f}/{rate_confidence_min:.2f} "
                f"x={state[0]:.2f} y={state[1]:.2f} z={state[2]:.2f} total_speed={speed_m_s:.2f}m/s "
                f"u={gate.u_m_s:.2f} v={gate.v_m_s:.2f} w={gate.w_m_s:.2f}"
            )
            next_console_status_s = elapsed_s + 1.0
        logger.append_metric_row(
            "prelaunch_state_samples.csv",
            {
                "t_host_s": time.perf_counter(),
                "frame_number": status.frame_number,
                "vicon_frame_rate_hz": status.frame_rate_hz,
                "vicon_latency_s": status.vicon_latency_s,
                **{f"estimator_{key}": value for key, value in estimator.items()},
                "launch_gate_consecutive_approved": consecutive_approved,
                "trigger_policy": "positive_x_launch_plane_crossing",
                "launch_trigger_x_w_m": float(LAUNCH_TRIGGER_X_W_M),
                "crossed_launch_plane": bool(crossed_launch_plane),
                "base_trigger_approved_before_rate_confidence": bool(base_trigger_approved),
                "rate_confidence_ok": bool(rate_confidence_ok),
                "launch_gate_rate_confidence_min": float(rate_confidence_min),
                "trigger_approved": bool(trigger_approved),
                "trigger_source": trigger_source,
                "full_window_reason": full_window_gate.reason,
                "full_window_approved": bool(full_window_gate.approved),
                "interpolated_plane_reason": plane_gate.reason if plane_gate is not None else "",
                "interpolated_plane_approved": bool(plane_gate.approved) if plane_gate is not None else False,
                "diagnostic_box_reason": full_window_gate.reason,
                "diagnostic_box_approved": bool(full_window_gate.approved),
                **asdict(gate),
                **state_dataframe_row(state),
            },
        )

        next_serial_s = _send_neutral_if_due(
            tx=tx,
            controller=controller,
            logger=logger,
            summary=summary,
            elapsed_s=elapsed_s,
            next_serial_s=next_serial_s,
            serial_period_s=float(config.serial_period_s),
        )

        if trigger_approved and consecutive_approved >= required_consecutive:
            summary["launch_gate_approved"] = True
            _append_runtime_event(
                logger,
                "launch_gate_approved_start_active_record",
                required_consecutive_frames=required_consecutive,
                launch_trigger_x_w_m=float(LAUNCH_TRIGGER_X_W_M),
                trigger_policy="first_valid_r5_launch_window_with_interpolated_plane_fallback",
                trigger_source=trigger_source,
                rate_confidence=rate_confidence,
                launch_gate_rate_confidence_min=rate_confidence_min,
                **asdict(gate),
            )
            return launch_state.copy() if launch_state is not None else state.copy()

        rejected_crossing_speed_m_s = (
            float(np.linalg.norm(launch_plane_state[6:9])) if launch_plane_state is not None else 0.0
        )
        passed_launch_window_after_partial_approval = bool(
            previous_consecutive_approved > 0
            and not trigger_approved
            and float(state[STATE_INDEX["x_w"]]) > float(LAUNCH_GATE_X_W_M[1])
        )
        rejected_launch_attempt = bool(
            launch_plane_state is not None
            and not trigger_approved
            and rejected_crossing_speed_m_s >= float(config.rejected_launch_attempt_min_speed_m_s)
        )
        if rejected_launch_attempt or passed_launch_window_after_partial_approval:
            speed_for_reason = (
                rejected_crossing_speed_m_s if launch_plane_state is not None else float(np.linalg.norm(state[6:9]))
            )
            summary["cancellation_reason"] = f"rejected_launch_attempt:{latest_gate_reason}"
            gate_details = asdict(gate)
            gate_reason = gate_details.pop("reason", latest_gate_reason)
            _append_runtime_event(
                logger,
                "flight_record_rejected_launch_attempt",
                cancellation_reason=summary["cancellation_reason"],
                launch_gate_reason=gate_reason,
                launch_attempt_speed_m_s=speed_for_reason,
                rejected_launch_attempt_min_speed_m_s=float(config.rejected_launch_attempt_min_speed_m_s),
                trigger_source=trigger_source,
                previous_consecutive_approved=previous_consecutive_approved,
                required_consecutive_frames=required_consecutive,
                **gate_details,
            )
            return None

        previous_state = state.copy()

        if mode in {"armed", "vicon-smoke"}:
            time.sleep(float(config.vicon_poll_period_s))

    summary["cancellation_reason"] = f"launch_gate_timeout:{latest_gate_reason}"
    _append_runtime_event(
        logger,
        "flight_record_cancelled_before_launch",
        reason=summary["cancellation_reason"],
        launch_wait_timeout_s=float(config.launch_wait_timeout_s),
    )
    return None


def _append_fan_positions(
    *,
    logger: FlightLogger,
    vicon: LiveNausicaaViconRigidBody | ReplayNausicaaViconRigidBody,
    adapter: NausicaaViconStateAdapter,
    phase: str,
    expected_visible_fan_range: tuple[int, int] | None,
    summary: dict[str, object],
    metric_buffer: list[tuple[str, dict[str, object]]] | None = None,
) -> None:
    if not hasattr(vicon, "read_fans"):
        return
    try:
        fans = vicon.read_fans()
    except Exception as exc:
        _append_metric_row(
            logger,
            "fan_positions.csv",
            {
                "t_host_s": time.perf_counter(),
                "phase": phase,
                "fan_subject": "",
                "visible": False,
                "reason": f"fan_tracker_failed:{type(exc).__name__}",
                "x_w": "",
                "y_w": "",
                "z_w": "",
                "visible_count": 0,
                "expected_count_ok": False,
            },
            metric_buffer=metric_buffer,
        )
        return
    visible_count = sum(1 for fan in fans if fan.visible)
    if expected_visible_fan_range is None:
        expected_ok = True
    else:
        expected_ok = int(expected_visible_fan_range[0]) <= visible_count <= int(expected_visible_fan_range[1])
    summary["fan_visible_count_latest"] = int(visible_count)
    summary["fan_expected_count_ok_latest"] = bool(expected_ok)
    for fan in fans:
        world = ("", "", "")
        if fan.visible and fan.position_m is not None:
            position = adapter.arena_transform.position_to_world(np.asarray(fan.position_m, dtype=float))
            world = tuple(float(value) for value in position)
        _append_metric_row(
            logger,
            "fan_positions.csv",
            {
                "t_host_s": time.perf_counter(),
                "phase": phase,
                "fan_subject": fan.subject_name,
                "visible": bool(fan.visible),
                "reason": fan.reason,
                "x_w": world[0],
                "y_w": world[1],
                "z_w": world[2],
                "visible_count": int(visible_count),
                "expected_count_ok": bool(expected_ok),
            },
            metric_buffer=metric_buffer,
        )


def _send_neutral_if_due(
    *,
    tx: NanoSerialTx | FakeNanoSerialTx,
    controller: FrozenFlightController,
    logger: FlightLogger,
    summary: dict[str, object],
    elapsed_s: float,
    next_serial_s: float,
    serial_period_s: float,
) -> float:
    if elapsed_s + 1e-12 < float(next_serial_s):
        return float(next_serial_s)
    if _write_packet_safe(
        tx,
        controller.neutral_packet(),
        logger=logger,
        summary=summary,
        event="launch_wait_neutral_packet",
    ):
        summary["neutral_failsafe_count"] = int(summary["neutral_failsafe_count"]) + 1
    return float(next_serial_s) + float(serial_period_s)


def _send_neutral_tail(
    *,
    config: FlightRuntimeConfig,
    tx: NanoSerialTx | FakeNanoSerialTx,
    controller: FrozenFlightController,
    logger: FlightLogger,
    mode: str,
    summary: dict[str, object],
    reason: str,
    metric_buffer: list[tuple[str, dict[str, object]]] | None = None,
) -> None:
    packet_count = max(1, int(np.ceil(float(config.post_exit_neutral_tail_s) / float(config.serial_period_s))))
    for _ in range(packet_count):
        if _write_packet_safe(
            tx,
            controller.neutral_packet(),
            logger=logger,
            summary=summary,
            event="post_exit_neutral_tail_packet",
        ):
            summary["neutral_failsafe_count"] = int(summary["neutral_failsafe_count"]) + 1
            summary["post_exit_neutral_packets"] = int(summary["post_exit_neutral_packets"]) + 1
        if mode in {"armed", "vicon-smoke"}:
            time.sleep(float(config.serial_period_s))
    _append_runtime_event(
        logger,
        "post_exit_neutral_tail_sent",
        metric_buffer=metric_buffer,
        reason=str(reason),
        neutral_packet_count=packet_count,
        post_exit_neutral_tail_s=float(config.post_exit_neutral_tail_s),
    )


def _predict_boundary_state(state_vector: np.ndarray, dt_s: float) -> np.ndarray:
    """Short-horizon boundary-state predictor for governor selection only."""

    state = np.asarray(state_vector, dtype=float).reshape(STATE_SIZE).copy()
    dt = max(0.0, min(float(dt_s), 0.25))
    if dt <= 0.0:
        return state
    phi = float(state[STATE_INDEX["phi"]])
    theta = float(state[STATE_INDEX["theta"]])
    psi = float(state[STATE_INDEX["psi"]])
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    c_theta, s_theta = np.cos(theta), np.sin(theta)
    c_psi, s_psi = np.cos(psi), np.sin(psi)
    c_wb = np.asarray(
        [
            [
                c_theta * c_psi,
                s_phi * s_theta * c_psi - c_phi * s_psi,
                c_phi * s_theta * c_psi + s_phi * s_psi,
            ],
            [
                c_theta * s_psi,
                s_phi * s_theta * s_psi + c_phi * c_psi,
                c_phi * s_theta * s_psi - s_phi * c_psi,
            ],
            [-s_theta, s_phi * c_theta, c_phi * c_theta],
        ],
        dtype=float,
    )
    body_velocity = state[[STATE_INDEX["u"], STATE_INDEX["v"], STATE_INDEX["w"]]]
    world_internal_velocity = c_wb @ body_velocity
    state[STATE_INDEX["x_w"]] += float(world_internal_velocity[0]) * dt
    state[STATE_INDEX["y_w"]] += float(world_internal_velocity[1]) * dt
    state[STATE_INDEX["z_w"]] -= float(world_internal_velocity[2]) * dt
    state[STATE_INDEX["phi"]] += float(state[STATE_INDEX["p"]]) * dt
    state[STATE_INDEX["theta"]] += float(state[STATE_INDEX["q"]]) * dt
    state[STATE_INDEX["psi"]] = _wrap_to_pi(float(state[STATE_INDEX["psi"]]) + float(state[STATE_INDEX["r"]]) * dt)
    return state


def _wrap_to_pi(value: float) -> float:
    return float((float(value) + np.pi) % (2.0 * np.pi) - np.pi)


def _append_metric_row(
    logger: FlightLogger,
    name: str,
    row: dict[str, object],
    *,
    metric_buffer: list[tuple[str, dict[str, object]]] | None = None,
) -> None:
    if metric_buffer is not None:
        metric_buffer.append((name, row))
        return
    logger.append_metric_row(name, row)


def _flush_metric_buffer(
    logger: FlightLogger,
    metric_buffer: list[tuple[str, dict[str, object]]],
) -> int:
    flushed = 0
    for name, row in metric_buffer:
        logger.append_metric_row(name, row)
        flushed += 1
    metric_buffer.clear()
    return flushed


def _append_runtime_event(
    logger: FlightLogger,
    event: str,
    *,
    metric_buffer: list[tuple[str, dict[str, object]]] | None = None,
    **details: object,
) -> None:
    _append_metric_row(
        logger,
        "runtime_events.csv",
        {
            "t_host_s": time.perf_counter(),
            "event": str(event),
            "details_json": json.dumps(details, sort_keys=True, separators=(",", ":")),
        },
        metric_buffer=metric_buffer,
    )


def _deployment_guard_status(*, config: FlightRuntimeConfig, mode: str) -> str:
    if mode != "armed" or config.controller_mode != "closed_loop":
        return "not_required_for_this_mode"
    return "passed"


def _tuple_or_empty(payload: object) -> tuple[float, ...]:
    if not isinstance(payload, (list, tuple)):
        return ()
    try:
        return tuple(float(value) for value in payload)
    except (TypeError, ValueError):
        return ()


def _calibration_convention_mismatches(
    *,
    manifest_profile: dict[str, object],
    config: FlightRuntimeConfig,
) -> list[str]:
    """Return mismatches that affect the frozen evidence/runtime convention.

    The Vicon position offset is intentionally excluded: it maps raw lab Vicon
    coordinates into the same arena frame used by simulation evidence and may
    be remeasured during preflight without invalidating R5/R8/R10/R11 artifacts.
    """

    checks: tuple[tuple[str, object, float], ...] = (
        ("vicon_yaw_alignment_deg", config.vicon_yaw_alignment_deg, 1e-12),
        ("vicon_attitude_signs", config.vicon_attitude_signs, 1e-12),
        ("vicon_attitude_offset_rad", config.vicon_attitude_offset_rad, 1e-12),
        ("requested_vicon_tracking_rate_hz", 1.0 / float(config.vicon_poll_period_s), 1e-9),
        ("derivative_cutoff_hz", config.derivative_cutoff_hz, 1e-12),
        ("body_rate_limit_rad_s", config.body_rate_limit_rad_s, 1e-12),
        ("body_rate_observer_window_frames", config.body_rate_observer_window_frames, 0.0),
        ("launch_gate_required_consecutive_frames", config.launch_gate_required_consecutive_frames, 0.0),
        ("launch_gate_rate_confidence_min", config.launch_gate_rate_confidence_min, 1e-12),
        ("launch_gate_body_rate_limits_rad_s", config.launch_gate_body_rate_limits_rad_s, 1e-12),
        ("rejected_launch_attempt_min_speed_m_s", config.rejected_launch_attempt_min_speed_m_s, 1e-12),
    )
    mismatches: list[str] = []
    for key, runtime_value, atol in checks:
        if key not in manifest_profile:
            mismatches.append(f"{key}:missing_in_manifest")
            continue
        manifest_value = manifest_profile[key]
        if isinstance(runtime_value, tuple):
            manifest_tuple = _tuple_or_empty(manifest_value)
            runtime_tuple = tuple(float(value) for value in runtime_value)
            if len(manifest_tuple) != len(runtime_tuple) or not np.allclose(
                manifest_tuple,
                runtime_tuple,
                rtol=0.0,
                atol=float(atol),
            ):
                mismatches.append(f"{key}:manifest={manifest_tuple},runtime={runtime_tuple}")
            continue
        if isinstance(runtime_value, int):
            try:
                manifest_int = int(manifest_value)
            except (TypeError, ValueError):
                mismatches.append(f"{key}:manifest={manifest_value},runtime={runtime_value}")
                continue
            if manifest_int != int(runtime_value):
                mismatches.append(f"{key}:manifest={manifest_int},runtime={int(runtime_value)}")
            continue
        try:
            manifest_float = float(manifest_value)
            runtime_float = float(runtime_value)
        except (TypeError, ValueError):
            mismatches.append(f"{key}:manifest={manifest_value},runtime={runtime_value}")
            continue
        if not np.isclose(manifest_float, runtime_float, rtol=0.0, atol=float(atol)):
            mismatches.append(f"{key}:manifest={manifest_float},runtime={runtime_float}")
    return mismatches


def _validate_closed_loop_deployment_evidence(*, config: FlightRuntimeConfig, mode: str) -> None:
    if (
        mode != "armed"
        or config.controller_mode != "closed_loop"
        or not bool(config.deployment_evidence_required_for_armed_closed_loop)
    ):
        return
    manifest_path = Path(config.deployment_evidence_manifest_path)
    if not manifest_path.exists():
        raise RuntimeError(
            "Refusing armed closed-loop flight: deployment evidence manifest is missing. "
            f"Expected {manifest_path}. Regenerate/freeze R5/R7/R8/R10/R11 for the active "
            "calibration profile, or run open-loop/calibration mode only."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    except Exception as exc:
        raise RuntimeError(
            f"Refusing armed closed-loop flight: deployment evidence manifest could not be read: {manifest_path}"
        ) from exc
    manifest_hash = str(manifest.get("calibration_profile_hash", ""))
    if manifest_hash != str(config.calibration_profile_hash):
        manifest_profile = manifest.get("active_calibration_profile", {})
        if not isinstance(manifest_profile, dict):
            raise RuntimeError(
                "Refusing armed closed-loop flight: frozen evidence manifest has a calibration hash mismatch "
                "and no active calibration profile object to compare evidence-sensitive conventions."
            )
        mismatches = _calibration_convention_mismatches(manifest_profile=manifest_profile, config=config)
        if mismatches:
            raise RuntimeError(
                "Refusing armed closed-loop flight: frozen evidence calibration conventions do not match "
                "the active runtime profile. Position-offset-only Vicon frame updates are allowed, but "
                f"these evidence-sensitive fields differ: {'; '.join(mismatches)}. "
                f"manifest_hash={manifest_hash or '<missing>'} runtime_hash={config.calibration_profile_hash}"
            )
    if manifest.get("evidence_regenerated_after_calibration") is not True:
        raise RuntimeError(
            "Refusing armed closed-loop flight: deployment evidence manifest does not confirm "
            "R5/R7/R8/R10/R11 regeneration after calibration."
        )
    required_keys = ("r5_label", "r7_label", "r8_label", "r10_label", "r11_label")
    missing = [key for key in required_keys if not str(manifest.get(key, "")).strip()]
    if missing:
        raise RuntimeError(
            "Refusing armed closed-loop flight: deployment evidence manifest is incomplete; "
            f"missing {', '.join(missing)}."
        )


def _sleep_until_next_runtime_poll(
    *,
    config: FlightRuntimeConfig,
    started: float,
    next_serial_s: float,
    next_governor_s: float,
) -> None:
    elapsed_s = time.perf_counter() - float(started)
    waits = [float(config.vicon_poll_period_s)]
    for next_event_s in (float(next_serial_s), float(next_governor_s)):
        wait_s = next_event_s - elapsed_s
        if wait_s > 0.0:
            waits.append(wait_s)
    wait_s = max(0.0, min(waits))
    if wait_s > ACTIVE_RUNTIME_WAKE_AHEAD_S:
        wait_s -= ACTIVE_RUNTIME_WAKE_AHEAD_S
    time.sleep(wait_s)


def _write_packet_safe(
    tx: NanoSerialTx | FakeNanoSerialTx,
    packet: bytes,
    *,
    logger: FlightLogger,
    summary: dict[str, object],
    event: str,
    quiet: bool = False,
) -> bool:
    try:
        tx.write_packet(packet)
    except Exception as exc:
        summary["serial_write_error_count"] = int(summary.get("serial_write_error_count", 0)) + 1
        if "Timeout" in type(exc).__name__:
            summary["serial_write_timeout_count"] = int(summary.get("serial_write_timeout_count", 0)) + 1
        _append_runtime_event(
            logger,
            "serial_write_failed",
            source_event=str(event),
            error_type=type(exc).__name__,
            error=str(exc),
        )
        if not quiet:
            print(f"[SERIAL] write failed during {event}: {type(exc).__name__}: {exc}")
        return False
    summary["packet_count"] = int(summary["packet_count"]) + 1
    return True


def _run_packet_smoke(
    *,
    config: FlightRuntimeConfig,
    tx: NanoSerialTx | FakeNanoSerialTx,
    controller: FrozenFlightController,
    logger: FlightLogger,
) -> None:
    del config
    commands = (
        np.zeros(3),
        np.asarray([0.2, 0.0, 0.0], dtype=float),
        np.asarray([-0.2, 0.0, 0.0], dtype=float),
        np.zeros(3),
    )
    for index, command in enumerate(commands):
        controller._last_command_norm = command.copy()  # packet smoke deliberately bypasses selection.
        packet = controller.packet_for_last_command()
        tx.write_packet(packet)
        logger.append_metric_row(
            "packet_smoke.csv",
            {
                "packet_index": index,
                "command_norm": tuple(float(value) for value in command),
                "packet_hex": packet.hex(),
            },
        )
        time.sleep(0.20)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the self-contained Nausicaa real-flight controller runtime.")
    parser.add_argument("--mode", choices=("dry-run", "packet-smoke", "vicon-smoke", "armed"), default="dry-run")
    parser.add_argument("--run-label", default="")
    parser.add_argument("--library-tier", choices=("balanced_cluster", "heavy_cluster"), default=DEFAULT_REAL_FLIGHT_LIBRARY_TIER)
    parser.add_argument("--controller-mode", choices=("closed_loop", "open_loop_neutral"), default="closed_loop")
    parser.add_argument("--serial-port", default="COM11")
    parser.add_argument("--vicon-host", default="192.168.0.100:801")
    parser.add_argument("--calibration-profile", choices=("active",), default=None)
    parser.add_argument("--vicon-offset-m", nargs=3, type=float, default=None)
    parser.add_argument("--vicon-yaw-deg", type=float, default=None)
    parser.add_argument("--vicon-attitude-signs", nargs=3, type=float, default=None)
    parser.add_argument("--vicon-attitude-offset-deg", nargs=3, type=float, default=None)
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--launch-wait-timeout-s", type=float, default=8.0)
    parser.add_argument("--launch-gate-frames", type=int, default=FlightRuntimeConfig.launch_gate_required_consecutive_frames)
    parser.add_argument("--post-exit-neutral-tail-s", type=float, default=0.30)
    parser.add_argument("--vicon-tracking-rate-hz", type=float, default=200.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.mode == "armed" and args.vicon_offset_m is None and args.calibration_profile != "active":
        raise SystemExit(
            "Refusing armed flight without an explicit calibrated Vicon profile. "
            "Use --calibration-profile active, or pass --vicon-offset-m X Y Z explicitly."
        )
    if args.calibration_profile == "active" and (
        args.vicon_offset_m is not None
        or args.vicon_yaw_deg is not None
        or args.vicon_attitude_signs is not None
        or args.vicon_attitude_offset_deg is not None
    ):
        raise SystemExit(
            "--calibration-profile active already supplies Vicon offset/yaw/signs/attitude offset; "
            "remove manual Vicon transform arguments or omit --calibration-profile."
        )
    if args.calibration_profile == "active":
        calibration_profile = ACTIVE_CALIBRATION_PROFILE
    else:
        offset = tuple(args.vicon_offset_m) if args.vicon_offset_m is not None else DEFAULT_VICON_POSITION_OFFSET_M
        yaw_deg = float(args.vicon_yaw_deg) if args.vicon_yaw_deg is not None else 0.0
        attitude_signs = (
            tuple(float(value) for value in args.vicon_attitude_signs)
            if args.vicon_attitude_signs is not None
            else DEFAULT_VICON_ATTITUDE_SIGNS
        )
        attitude_offset_rad = (
            tuple(float(np.deg2rad(value)) for value in args.vicon_attitude_offset_deg)
            if args.vicon_attitude_offset_deg is not None
            else DEFAULT_VICON_ATTITUDE_OFFSET_RAD
        )
        profile_id = "manual_cli_vicon_transform" if args.vicon_offset_m is not None else "default_runtime_transform"
        calibration_profile = calibration_profile_for_runtime_values(
            profile_id=profile_id,
            vicon_position_offset_m=offset,
            vicon_yaw_alignment_deg=yaw_deg,
            vicon_attitude_signs=attitude_signs,
            vicon_attitude_offset_rad=attitude_offset_rad,
            requested_vicon_tracking_rate_hz=float(args.vicon_tracking_rate_hz),
            launch_gate_required_consecutive_frames=int(args.launch_gate_frames),
        )
    config = FlightRuntimeConfig(
        run_label=args.run_label or default_run_label(),
        library_tier=args.library_tier,
        controller_mode=args.controller_mode,
        serial_port=args.serial_port,
        vicon_host=args.vicon_host,
        vicon_position_offset_m=calibration_profile.vicon_position_offset_m,
        vicon_yaw_alignment_deg=float(calibration_profile.vicon_yaw_alignment_deg),
        vicon_attitude_signs=calibration_profile.vicon_attitude_signs,
        vicon_attitude_offset_rad=calibration_profile.vicon_attitude_offset_rad,
        calibration_profile_id=calibration_profile.profile_id,
        calibration_profile_hash=calibration_profile.profile_hash(),
        vicon_calibration_source=(
            "active_calibration_profile" if args.calibration_profile == "active" else calibration_profile.profile_id
        ),
        max_duration_s=float(args.duration_s),
        launch_wait_timeout_s=float(args.launch_wait_timeout_s),
        launch_gate_required_consecutive_frames=int(args.launch_gate_frames),
        post_exit_neutral_tail_s=float(args.post_exit_neutral_tail_s),
        vicon_poll_period_s=1.0 / float(args.vicon_tracking_rate_hz),
    )
    summary = run_real_flight(config, mode=str(args.mode))
    print(f"run_root={summary['run_root']}")
    print(f"completed={summary['completed']}")
    print(f"controller_decisions={summary['controller_decision_count']}")
    print(f"packets={summary['packet_count']}")


if __name__ == "__main__":
    main()
