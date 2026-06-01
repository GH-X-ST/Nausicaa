from __future__ import annotations

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
from experiment_cases import EXPERIMENT_CASES, get_experiment_case  # noqa: E402
from exit_gate import evaluate_exit_gate  # noqa: E402
from launch_gate import (  # noqa: E402
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
from vicon_rigid_body import ReplayNausicaaViconRigidBody  # noqa: E402


def _code(value: float) -> int:
    return int(np.rint((float(value) + 1.0) * 0.5 * 65535.0))


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
