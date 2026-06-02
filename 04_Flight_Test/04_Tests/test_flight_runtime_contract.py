from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "01_Runtime"
CONTROLLER = ROOT / "02_Controller"
for path in (RUNTIME, CONTROLLER):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from flight_config import FlightRuntimeConfig  # noqa: E402
from frozen_flight_controller import FrozenFlightController  # noqa: E402
from frozen_flight_controller import (  # noqa: E402
    _governor_mode_for_route,
    _live_context_row,
    _validation_route_for_primitive_step,
)
from experiment_cases import EXPERIMENT_CASES, get_experiment_case  # noqa: E402
from exit_gate import evaluate_exit_gate  # noqa: E402
from launch_gate import (  # noqa: E402
    LAUNCH_GATE_ROLL_RATE_LIMIT_RAD_S,
    LAUNCH_TRIGGER_X_W_M,
    evaluate_launch_gate,
    evaluate_launch_plane_gate,
    interpolate_launch_plane_state,
)
from real_flight_io import (  # noqa: E402
    NausicaaViconSample,
    NausicaaViconStateAdapter,
    ViconArenaFrameTransform,
    aggregate_to_physical_surface_norm,
    encode_arduino_command_packet,
)
from run_real_flight import run_real_flight  # noqa: E402
from run_experiment_sequence import run_experiment_sequence  # noqa: E402
from run_surface_sign_check import SURFACE_CHECK_SEQUENCE, run_surface_sign_check  # noqa: E402
from state_contract import STATE_INDEX  # noqa: E402
from episode_selector import select_compact_representative  # noqa: E402
from transition_labels import classify_state  # noqa: E402
from vicon_rigid_body import ReplayNausicaaViconRigidBody  # noqa: E402


def _code(value: float) -> int:
    return int(np.rint((float(value) + 1.0) * 0.5 * 65535.0))


def _state(
    *,
    x_w: float = 2.0,
    y_w: float = 2.2,
    z_w: float = 1.6,
    phi: float = 0.0,
    theta: float = 0.0,
    psi: float = 0.0,
    u: float = 6.5,
    v: float = 0.0,
    w: float = 0.0,
    p: float = 0.0,
    q: float = 0.0,
    r: float = 0.0,
) -> np.ndarray:
    state = np.zeros(15, dtype=float)
    values = {
        "x_w": x_w,
        "y_w": y_w,
        "z_w": z_w,
        "phi": phi,
        "theta": theta,
        "psi": psi,
        "u": u,
        "v": v,
        "w": w,
        "p": p,
        "q": q,
        "r": r,
    }
    for name, value in values.items():
        state[STATE_INDEX[name]] = float(value)
    return state


def test_boundary_near_uses_side_closing_speed_not_total_forward_speed() -> None:
    central_forward = _state(x_w=2.0, y_w=2.1, z_w=1.55, psi=0.0, u=6.6, v=0.0)

    assert classify_state(central_forward, primitive_step_index=1, allow_post_launch_degraded=True) == "inflight_stable"

    side_closing = _state(x_w=2.0, y_w=2.1, z_w=1.55, psi=0.0, u=6.6, v=-6.6)

    assert classify_state(side_closing, primitive_step_index=1, allow_post_launch_degraded=True) == "boundary_near"


def test_launch_selector_uses_neighbouring_speed_bins_for_safe_fallbacks(tmp_path: Path) -> None:
    config = FlightRuntimeConfig(run_label="selector_regression", output_root=tmp_path)
    controller = FrozenFlightController(config)
    launch_state = _state(x_w=1.3, y_w=2.05, z_w=1.65, psi=0.0, u=6.6)
    route = _validation_route_for_primitive_step(0, state=launch_state)
    context = _live_context_row(
        launch_state,
        library_tier=config.library_tier,
        primitive_step_index=0,
        route=route,
        memory_enabled=False,
        memory_launch_index=0,
    )

    selected, candidates = select_compact_representative(
        representatives=controller.representatives,
        outcome_rows_by_variant_id=controller.outcomes,
        context=context,
        governor_mode=_governor_mode_for_route(route),
        policy_id="selector_regression",
        belief_features=None,
        candidate_belief_features=None,
        adaptive_memory_active=False,
        governor_config=controller.governor_config,
        candidate_row_mode="diagnostic",
    )
    viable_ids = {str(row.get("primitive_id", "")) for row in candidates if bool(row.get("viable", False))}

    assert selected is not None
    assert len(viable_ids) > 1
    assert "glide" in viable_ids


def test_full_authority_packet_has_no_0p70_cap() -> None:
    physical = aggregate_to_physical_surface_norm([1.0, -1.0, 1.0])
    np.testing.assert_allclose(physical, [1.0, -1.0, 1.0, -1.0])

    packet = encode_arduino_command_packet([1.0, -1.0, 1.0], sequence=7)

    assert packet.receiver_channel_codes == (_code(1.0), _code(1.0), _code(1.0), _code(1.0))
    assert max(packet.receiver_channel_codes) == 65535
    assert int.from_bytes(packet.packet_bytes[3:7], byteorder="little") == 7


def test_vicon_rigid_body_adapter_uses_command_history_surfaces() -> None:
    adapter = NausicaaViconStateAdapter(derivative_cutoff_hz=0.0, actuator_tau_s=(0.1, 0.1, 0.1))
    adapter.update(
        NausicaaViconSample(0.0, (1.2, 2.2, 1.5), euler_rad=(0.0, 0.0, 0.0)),
        command_norm=[0.0, 0.0, 0.0],
    )
    state = adapter.update(
        NausicaaViconSample(0.1, (1.8, 2.2, 1.5), euler_rad=(0.0, 0.0, 0.0)),
        command_norm=[1.0, 0.0, 0.0],
    )

    assert np.isclose(state[STATE_INDEX["u"]], 6.0)
    assert state[STATE_INDEX["delta_a"]] > 0.0


def test_vicon_transform_applies_full_xyz_offset() -> None:
    adapter = NausicaaViconStateAdapter(
        derivative_cutoff_hz=0.0,
        arena_transform=ViconArenaFrameTransform(position_offset_m=(3.9, 2.2, 1.95)),
    )

    state = adapter.update(
        NausicaaViconSample(0.0, (0.0, 0.0, 0.42), euler_rad=(0.0, 0.0, 0.0)),
        command_norm=[0.0, 0.0, 0.0],
    )

    assert np.isclose(state[STATE_INDEX["x_w"]], 3.9)
    assert np.isclose(state[STATE_INDEX["y_w"]], 2.2)
    assert np.isclose(state[STATE_INDEX["z_w"]], 2.37)


def test_vicon_transform_applies_recovered_pitch_yaw_signs() -> None:
    adapter = NausicaaViconStateAdapter(
        derivative_cutoff_hz=0.0,
        arena_transform=ViconArenaFrameTransform(attitude_signs=(1.0, -1.0, -1.0)),
    )

    state = adapter.update(
        NausicaaViconSample(0.0, (0.0, 0.0, 0.0), euler_rad=(0.10, -0.20, -0.30)),
        command_norm=[0.0, 0.0, 0.0],
    )

    assert np.isclose(state[STATE_INDEX["phi"]], 0.10)
    assert np.isclose(state[STATE_INDEX["theta"]], 0.20)
    assert np.isclose(state[STATE_INDEX["psi"]], 0.30)


def test_vicon_adapter_body_rates_are_rotation_delta_based_and_bounded() -> None:
    adapter = NausicaaViconStateAdapter(
        derivative_cutoff_hz=0.0,
        body_rate_limit_rad_s=2.0,
    )
    adapter.update(
        NausicaaViconSample(0.0, (0.0, 0.0, 0.0), euler_rad=(0.0, 0.0, 0.0)),
        command_norm=[0.0, 0.0, 0.0],
    )

    state = adapter.update(
        NausicaaViconSample(0.1, (0.0, 0.0, 0.0), euler_rad=(0.1, 0.0, 0.0)),
        command_norm=[0.0, 0.0, 0.0],
    )

    assert np.isclose(state[STATE_INDEX["p"]], 1.0, atol=1e-6)
    assert np.isclose(state[STATE_INDEX["q"]], 0.0, atol=1e-6)
    assert np.isclose(state[STATE_INDEX["r"]], 0.0, atol=1e-6)

    clipped = adapter.update(
        NausicaaViconSample(0.11, (0.0, 0.0, 0.0), euler_rad=(1.1, 0.0, 0.0)),
        command_norm=[0.0, 0.0, 0.0],
    )

    assert abs(float(clipped[STATE_INDEX["p"]])) <= 2.0 + 1e-9
    adapter.reset_angular_rate_filter()
    assert np.isfinite(adapter.update(
        NausicaaViconSample(0.12, (0.0, 0.0, 0.0), euler_rad=(1.1, 0.0, 0.0)),
        command_norm=[0.0, 0.0, 0.0],
    )[STATE_INDEX["p"]])


def test_vicon_adapter_prefers_frame_synchronous_timing_over_host_jitter() -> None:
    adapter = NausicaaViconStateAdapter(derivative_cutoff_hz=0.0)
    adapter.update(
        NausicaaViconSample(
            1000.0,
            (0.0, 0.0, 0.0),
            euler_rad=(0.0, 0.0, 0.0),
            frame_number=10,
            frame_rate_hz=200.0,
        ),
        command_norm=[0.0, 0.0, 0.0],
    )

    state = adapter.update(
        NausicaaViconSample(
            1000.001,
            (1.0, 0.0, 0.0),
            euler_rad=(0.0, 0.0, 0.0),
            frame_number=30,
            frame_rate_hz=200.0,
        ),
        command_norm=[0.0, 0.0, 0.0],
    )
    status = adapter.estimator_status()

    assert np.isclose(state[STATE_INDEX["u"]], 10.0)
    assert status["dt_source"] == "vicon_frame_time"
    assert status["frame_delta"] == 20
    assert np.isclose(status["dt_s"], 0.1)

    duplicate = adapter.update(
        NausicaaViconSample(
            1000.002,
            (99.0, 0.0, 0.0),
            euler_rad=(0.0, 0.0, 0.0),
            frame_number=30,
            frame_rate_hz=200.0,
        ),
        command_norm=[0.0, 0.0, 0.0],
    )
    duplicate_status = adapter.estimator_status()

    assert duplicate_status["dt_source"] == "duplicate_or_reordered_vicon_frame"
    assert np.isclose(duplicate[STATE_INDEX["u"]], 10.0)


def test_launch_gate_uses_r5_release_bounds() -> None:
    state = np.zeros(15)
    state[STATE_INDEX["x_w"]] = 1.3
    state[STATE_INDEX["y_w"]] = 2.0
    state[STATE_INDEX["z_w"]] = 1.7
    state[STATE_INDEX["u"]] = 5.0

    assert evaluate_launch_gate(state).approved is True

    state[STATE_INDEX["x_w"]] = 1.5
    rejected = evaluate_launch_gate(state)
    assert rejected.approved is False
    assert rejected.reason == "x_w_outside_launch_gate"

    state[STATE_INDEX["x_w"]] = 1.3
    state[STATE_INDEX["p"]] = LAUNCH_GATE_ROLL_RATE_LIMIT_RAD_S + 0.01
    rejected = evaluate_launch_gate(state)
    assert rejected.approved is False
    assert rejected.reason == "roll_rate_outside_launch_gate"


def test_launch_plane_crossing_interpolates_entry_state() -> None:
    previous = np.zeros(15)
    current = np.zeros(15)
    previous[STATE_INDEX["x_w"]] = 1.1
    current[STATE_INDEX["x_w"]] = 1.5
    previous[STATE_INDEX["y_w"]] = 2.0
    current[STATE_INDEX["y_w"]] = 2.0
    previous[STATE_INDEX["z_w"]] = 1.7
    current[STATE_INDEX["z_w"]] = 1.7
    previous[STATE_INDEX["u"]] = 5.0
    current[STATE_INDEX["u"]] = 5.0

    launch_state = interpolate_launch_plane_state(previous, current)

    assert launch_state is not None
    assert np.isclose(launch_state[STATE_INDEX["x_w"]], LAUNCH_TRIGGER_X_W_M)
    assert evaluate_launch_plane_gate(launch_state).approved is True

    current[STATE_INDEX["q"]] = 1.0
    launch_state = interpolate_launch_plane_state(previous, current)
    assert launch_state is not None
    rejected = evaluate_launch_plane_gate(launch_state)
    assert rejected.approved is False
    assert rejected.reason == "pitch_rate_outside_launch_gate"


def test_exit_gate_uses_true_operational_region() -> None:
    state = np.zeros(15)
    state[STATE_INDEX["x_w"]] = 3.0
    state[STATE_INDEX["y_w"]] = 2.0
    state[STATE_INDEX["z_w"]] = 1.7

    assert evaluate_exit_gate(state).inside is True

    state[STATE_INDEX["x_w"]] = 6.7
    exited = evaluate_exit_gate(state)
    assert exited.inside is False
    assert exited.reason == "exit_gate_front_wall"


def test_frozen_controller_loads_and_returns_quantised_command() -> None:
    controller = FrozenFlightController(FlightRuntimeConfig(run_label="pytest_contract"))
    state = np.zeros(15)
    state[STATE_INDEX["x_w"]] = 1.2
    state[STATE_INDEX["y_w"]] = 2.2
    state[STATE_INDEX["z_w"]] = 1.6
    state[STATE_INDEX["u"]] = 6.0

    decision = controller.decide(state, primitive_step_index=0)

    assert decision.candidate_count > 0
    assert all(value in {-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0} for value in decision.command_norm)
    assert len(decision.packet_bytes) == 15


def test_dry_run_writes_local_result_tree(tmp_path: Path) -> None:
    config = FlightRuntimeConfig(run_label="T01", output_root=tmp_path, max_duration_s=0.12)

    summary = run_real_flight(config, mode="dry-run")

    assert summary["completed"] is True
    assert (tmp_path / "T01" / "manifests" / "real_flight_runtime_manifest.json").exists()
    assert (tmp_path / "T01" / "metrics" / "state_samples.csv").exists()


def test_flight_record_cancels_when_launch_gate_never_passes(tmp_path: Path) -> None:
    config = FlightRuntimeConfig(
        run_label="T_cancel",
        output_root=tmp_path,
        max_duration_s=0.12,
        launch_wait_timeout_s=0.001,
        vicon_position_offset_m=(1000.0, 0.0, 0.0),
    )

    summary = run_real_flight(config, mode="dry-run")

    assert summary["flight_cancelled"] is True
    assert summary["launch_gate_approved"] is False
    assert summary["controller_decision_count"] == 0
    assert not (tmp_path / "T_cancel" / "metrics" / "state_samples.csv").exists()


def test_active_record_terminates_at_exit_gate_and_sends_neutral_tail(tmp_path: Path) -> None:
    config = FlightRuntimeConfig(
        run_label="T_exit",
        output_root=tmp_path,
        max_duration_s=1.2,
        post_exit_neutral_tail_s=0.04,
    )

    summary = run_real_flight(config, mode="dry-run")

    assert summary["launch_gate_approved"] is True
    assert summary["exit_gate_triggered"] is True
    assert summary["termination_reason"] == "exit_gate_front_wall"
    assert summary["post_exit_neutral_packets"] >= 1
    assert (tmp_path / "T_exit" / "metrics" / "state_samples.csv").exists()


def test_launch_gate_default_debounces_three_consecutive_approved_frames(tmp_path: Path) -> None:
    config = FlightRuntimeConfig(
        run_label="T_launch_debounce",
        output_root=tmp_path,
        max_duration_s=0.12,
        post_exit_neutral_tail_s=0.0,
    )

    assert config.launch_gate_required_consecutive_frames == 3

    summary = run_real_flight(config, mode="dry-run")

    assert summary["launch_gate_approved"] is True
    prelaunch_path = tmp_path / "T_launch_debounce" / "metrics" / "prelaunch_state_samples.csv"
    with prelaunch_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    approved_counts = [
        int(row["launch_gate_consecutive_approved"])
        for row in rows
        if row["trigger_approved"].lower() == "true"
    ]

    assert 1 in approved_counts
    assert 2 in approved_counts
    assert max(approved_counts) == 3
    assert (tmp_path / "T_launch_debounce" / "metrics" / "state_samples.csv").exists()


def test_experiment_case_registry_contains_requested_cases() -> None:
    requested = {
        "E0.1",
        "E0.2",
        "E1.1",
        "E1.2",
        "E2.1",
        "E2.2",
        "E3.1",
        "E3.2",
        "E4a.1",
        "E4a.2",
        "E4b.1",
        "E4b.2",
        "E4c.1",
        "E4c.2",
        "E5a.1",
        "E5a.2",
        "E5b.1",
        "E5b.2",
        "E5c.1",
        "E5c.2",
        "E5d.1",
        "E5d.2",
    }
    assert requested.issubset(EXPERIMENT_CASES)
    assert get_experiment_case("E2.2").memory_enabled is True
    assert get_experiment_case("E1.1").memory_enabled is False


def test_replay_fan_tracker_handles_zero_one_and_four_subjects() -> None:
    replay = ReplayNausicaaViconRigidBody().open()
    replay.read_latest()
    assert sum(1 for fan in replay.read_fans(("Missing_Fan",)) if fan.visible) == 0
    assert sum(1 for fan in replay.read_fans(("Fan_1",)) if fan.visible) == 1
    assert sum(1 for fan in replay.read_fans(("Fan_1", "Fan_2", "Fan_3", "Fan_4")) if fan.visible) == 4


def test_experiment_sequence_counts_only_valid_throws_and_persists_memory(tmp_path: Path, monkeypatch) -> None:
    import run_experiment_sequence as sequence_module

    monkeypatch.setattr(sequence_module, "RESULT_ROOT", tmp_path)
    result = run_experiment_sequence(
        case_id="E2.2",
        session_label="pytest_memory",
        mode="dry-run",
        serial_port="COM_TEST",
        vicon_host="mock",
        target_valid_throws=2,
        cooldown_s=0.0,
        retry_cooldown_s=0.0,
        max_invalid_attempts=2,
        max_duration_s=0.22,
        launch_wait_timeout_s=0.20,
        post_exit_neutral_tail_s=0.0,
        vicon_poll_period_s=0.005,
        vicon_position_offset_m=(3.9, 2.2, 1.95),
        vicon_yaw_alignment_deg=0.0,
        vicon_attitude_signs=(1.0, -1.0, -1.0),
    )
    assert result["valid_throw_count"] == 2
    assert result["invalid_attempt_count"] == 0
    assert int(result["memory"]["memory_launch_index"]) >= 1
    assert (tmp_path / "E2.2" / "pytest_memory" / "throw_001" / "metrics" / "memory_update_summary.csv").exists()


def test_experiment_sequence_invalid_start_does_not_count_or_update_memory(tmp_path: Path, monkeypatch) -> None:
    import run_experiment_sequence as sequence_module

    monkeypatch.setattr(sequence_module, "RESULT_ROOT", tmp_path)
    result = run_experiment_sequence(
        case_id="E2.2",
        session_label="pytest_invalid",
        mode="dry-run",
        serial_port="COM_TEST",
        vicon_host="mock",
        target_valid_throws=1,
        cooldown_s=0.0,
        retry_cooldown_s=0.0,
        max_invalid_attempts=1,
        max_duration_s=0.05,
        launch_wait_timeout_s=0.001,
        post_exit_neutral_tail_s=0.0,
        vicon_poll_period_s=0.005,
        vicon_position_offset_m=(1000.0, 2.2, 1.95),
        vicon_yaw_alignment_deg=0.0,
        vicon_attitude_signs=(1.0, -1.0, -1.0),
    )
    assert result["valid_throw_count"] == 0
    assert result["invalid_attempt_count"] >= 1
    assert int(result["memory"]["memory_launch_index"]) == 0
    assert (tmp_path / "E2.2" / "pytest_invalid" / "invalid_attempts" / "attempt_001").exists()


def test_surface_sign_check_uses_same_packet_contract(tmp_path: Path, monkeypatch) -> None:
    import run_surface_sign_check as sign_module

    monkeypatch.setattr(sign_module, "RESULT_ROOT", tmp_path)
    result = run_surface_sign_check(
        serial_port="COM_TEST",
        mode="dry-run",
        dwell_s=0.0,
        run_label="pytest_surface",
    )
    assert result["packet_count"] >= len(SURFACE_CHECK_SEQUENCE) + 1
    assert (tmp_path / "surface_sign_check" / "pytest_surface" / "metrics" / "surface_sign_check.csv").exists()
