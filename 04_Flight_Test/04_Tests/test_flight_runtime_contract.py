from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "01_Runtime"
CONTROLLER = ROOT / "02_Controller"
SCENARIOS = ROOT.parent / "03_Control" / "04_Scenarios"
for path in (RUNTIME, CONTROLLER, SCENARIOS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from calibration_profile import ACTIVE_CALIBRATION_PROFILE  # noqa: E402
from command_contract import normalised_command_to_surface_rad  # noqa: E402
from flight_config import DEFAULT_REAL_FLIGHT_LIBRARY_TIER, LAUNCH_HANDOFF_DURATION_S, FlightRuntimeConfig  # noqa: E402
from frozen_flight_controller import FrozenFlightController  # noqa: E402
from frozen_flight_controller import (  # noqa: E402
    _governor_mode_for_route,
    _live_context_row,
    _validation_route_for_primitive_step,
)
from experiment_cases import EXPERIMENT_CASES, get_experiment_case  # noqa: E402
from exit_gate import evaluate_exit_gate  # noqa: E402
from launch_gate import (  # noqa: E402
    LAUNCH_GATE_FORWARD_SPEED_MAX_M_S,
    LAUNCH_GATE_FORWARD_SPEED_MIN_M_S,
    LAUNCH_GATE_SIDE_VELOCITY_LIMIT_M_S,
    LAUNCH_GATE_ROLL_RATE_LIMIT_RAD_S,
    LAUNCH_TRIGGER_X_W_M,
    LAUNCH_GATE_VERTICAL_BODY_VELOCITY_LIMIT_M_S,
    evaluate_launch_gate,
    evaluate_launch_plane_gate,
    interpolate_launch_plane_state,
)
from latency import latency_case_config  # noqa: E402
from state_sampling import (  # noqa: E402
    LAUNCH_GATE_FORWARD_SPEED_MAX_M_S as SIM_LAUNCH_GATE_FORWARD_SPEED_MAX_M_S,
    LAUNCH_GATE_FORWARD_SPEED_MIN_M_S as SIM_LAUNCH_GATE_FORWARD_SPEED_MIN_M_S,
    LAUNCH_GATE_SIDE_VELOCITY_LIMIT_M_S as SIM_LAUNCH_GATE_SIDE_VELOCITY_LIMIT_M_S,
    LAUNCH_GATE_VERTICAL_BODY_VELOCITY_LIMIT_M_S as SIM_LAUNCH_GATE_VERTICAL_BODY_VELOCITY_LIMIT_M_S,
)
from real_flight_io import (  # noqa: E402
    NausicaaViconSample,
    NausicaaViconStateAdapter,
    SO3AngularRateObserver,
    ViconArenaFrameTransform,
    aggregate_to_physical_surface_norm,
    encode_arduino_command_packet,
)
from run_real_flight import _predict_boundary_state, _validate_closed_loop_deployment_evidence, run_real_flight  # noqa: E402
from run_experiment_sequence import run_experiment_case_repeats, run_experiment_sequence  # noqa: E402
from run_glider_calibration_sequence import (  # noqa: E402
    PULSE_DURATION_BY_ABS_COMMAND,
    PULSE_START_DELAY_S,
    SUSTAINED_CONTROL_EFFECT_DURATION_S,
    _block_storage_id,
    _case_storage_id,
    _command_for_case,
    calibration_cases_for_block,
    run_calibration_sequence,
)
from run_vicon_orientation_check import _evaluate_steps, _sample_rate_summary  # noqa: E402
from run_surface_sign_check import SURFACE_CHECK_SEQUENCE, run_surface_sign_check  # noqa: E402
from run_vicon_frame_calibration import (  # noqa: E402
    DEFAULT_FAN_POSITION_TOLERANCE_M,
    _fan_position_error_m,
    _update_active_vicon_calibration_files,
)
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


def _fifo_order_sensitive_payload() -> tuple[dict[str, object], np.ndarray, list[np.ndarray]]:
    controller = FrozenFlightController(FlightRuntimeConfig(run_label="pytest_fifo_probe"))
    fifo_history = [
        normalised_command_to_surface_rad(np.asarray(row, dtype=float))
        for row in (
            (-0.8, -0.4, 0.2),
            (-0.2, 0.0, 0.4),
            (0.2, 0.4, -0.2),
            (0.8, 0.6, -0.4),
        )
    ]
    for payload in sorted(controller.controllers.values(), key=lambda item: str(item.get("controller_id", ""))):
        delay_steps = int(float(payload.get("command_delay_steps", 0)))
        if delay_steps != len(fifo_history):
            continue
        controller_id = str(payload.get("controller_id", ""))
        state = np.asarray(payload["reference_state_vector"], dtype=float).reshape(15).copy()
        state[STATE_INDEX["phi"]] += 0.20
        state[STATE_INDEX["theta"]] += -0.10
        state[STATE_INDEX["p"]] += 0.30
        state[STATE_INDEX["q"]] += -0.20

        controller._command_fifo_by_controller_id[controller_id] = [row.copy() for row in fifo_history]
        old_to_new_norm, _ = controller._command_for_payload(payload, state)
        controller._command_fifo_by_controller_id[controller_id] = [row.copy() for row in reversed(fifo_history)]
        reversed_norm, _ = controller._command_for_payload(payload, state)
        if not np.allclose(old_to_new_norm, reversed_norm):
            controller._command_fifo_by_controller_id.pop(controller_id, None)
            return payload, state, fifo_history
    pytest.fail("frozen controller bundle has no FIFO-order-sensitive payload")


def test_real_flight_default_library_tier_is_balanced_cluster() -> None:
    config = FlightRuntimeConfig(run_label="default_tier_regression")

    assert DEFAULT_REAL_FLIGHT_LIBRARY_TIER == "balanced_cluster"
    assert config.library_tier == "balanced_cluster"
    assert config.library_manifest_path.name == "balanced_cluster_primitive_library.json"
    assert config.vicon_position_offset_m == ACTIVE_CALIBRATION_PROFILE.vicon_position_offset_m
    assert config.calibration_profile_hash == ACTIVE_CALIBRATION_PROFILE.profile_hash()


def test_vicon_frame_calibration_can_update_active_position_only(tmp_path: Path) -> None:
    position_path = tmp_path / "active_vicon_position_calibration.json"
    attitude_path = tmp_path / "active_vicon_attitude_calibration.json"

    update = _update_active_vicon_calibration_files(
        position_calibration_path=position_path,
        attitude_calibration_path=attitude_path,
        recommended_offset_m=(1.0, 2.0, 3.0),
        recommended_attitude_offset_rad=(0.1, -0.2, 0.3),
        yaw_alignment_deg=0.0,
        attitude_signs=(1.0, -1.0, -1.0),
        update_position=True,
        update_attitude=False,
        profile_id="pytest_vicon_calibration",
        profile_version="9.0",
    )
    position_payload = json.loads(position_path.read_text(encoding="utf-8"))

    assert position_payload["profile_id"] == "pytest_vicon_calibration_position"
    assert position_payload["profile_version"] == "9.0"
    assert position_payload["vicon_position_offset_m"] == [1.0, 2.0, 3.0]
    assert not attitude_path.exists()
    assert update["updated_vicon_position_offset_m"] is True
    assert update["updated_vicon_attitude_offset_rad"] is False
    assert update["profile_id"] == "pytest_vicon_calibration"
    assert isinstance(update["profile_hash"], str)
    assert len(str(update["profile_hash"])) == 64


def test_vicon_frame_calibration_can_update_active_attitude_only(tmp_path: Path) -> None:
    position_path = tmp_path / "active_vicon_position_calibration.json"
    attitude_path = tmp_path / "active_vicon_attitude_calibration.json"

    update = _update_active_vicon_calibration_files(
        position_calibration_path=position_path,
        attitude_calibration_path=attitude_path,
        recommended_offset_m=(1.0, 2.0, 3.0),
        recommended_attitude_offset_rad=(0.1, -0.2, 0.3),
        yaw_alignment_deg=4.0,
        attitude_signs=(1.0, -1.0, -1.0),
        update_position=False,
        update_attitude=True,
        profile_id="pytest_vicon_attitude_calibration",
        profile_version="9.1",
    )
    attitude_payload = json.loads(attitude_path.read_text(encoding="utf-8"))

    assert not position_path.exists()
    assert attitude_payload["profile_id"] == "pytest_vicon_attitude_calibration_attitude"
    assert attitude_payload["profile_version"] == "9.1"
    assert attitude_payload["vicon_yaw_alignment_deg"] == 4.0
    assert attitude_payload["vicon_attitude_signs"] == [1.0, -1.0, -1.0]
    assert attitude_payload["vicon_attitude_offset_rad"] == [0.1, -0.2, 0.3]
    assert update["updated_vicon_position_offset_m"] is False
    assert update["updated_vicon_attitude_offset_rad"] is True
    assert update["profile_id"] == "pytest_vicon_attitude_calibration"
    assert isinstance(update["profile_hash"], str)
    assert len(str(update["profile_hash"])) == 64


def test_direct_armed_cli_refuses_without_calibrated_profile(monkeypatch) -> None:
    import run_real_flight as runtime_module

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_real_flight.py",
            "--mode",
            "armed",
            "--controller-mode",
            "open_loop_neutral",
            "--duration-s",
            "0.01",
        ],
    )

    with pytest.raises(SystemExit, match="explicit calibrated Vicon profile"):
        runtime_module.main()


def test_armed_closed_loop_requires_matching_deployment_evidence_manifest(tmp_path: Path) -> None:
    config = FlightRuntimeConfig(
        run_label="T_guard",
        output_root=tmp_path,
        deployment_evidence_manifest_path=tmp_path / "missing_deployment_manifest.json",
    )

    with pytest.raises(RuntimeError, match="deployment evidence manifest is missing"):
        run_real_flight(config, mode="armed")


def test_armed_closed_loop_allows_position_only_calibration_hash_drift(tmp_path: Path) -> None:
    profile = ACTIVE_CALIBRATION_PROFILE.to_manifest()
    profile["profile_id"] = f"{profile['profile_id']}+pytest_position_recheck"
    profile["profile_hash"] = "pytest_position_only_profile_hash"
    profile["vicon_position_offset_m"] = [
        float(ACTIVE_CALIBRATION_PROFILE.vicon_position_offset_m[0]) + 0.1,
        float(ACTIVE_CALIBRATION_PROFILE.vicon_position_offset_m[1]) - 0.1,
        float(ACTIVE_CALIBRATION_PROFILE.vicon_position_offset_m[2]) + 0.02,
    ]
    manifest_path = tmp_path / "deployment_evidence_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "calibration_profile_hash": "pytest_position_only_manifest_hash",
                "active_calibration_profile": profile,
                "evidence_regenerated_after_calibration": True,
                "r5_label": "E03",
                "r7_label": "E03",
                "r8_label": "E03",
                "r10_label": "E03",
                "r11_label": "E03.1",
            },
            sort_keys=True,
        ),
        encoding="ascii",
    )
    config = FlightRuntimeConfig(
        run_label="T_position_only_guard",
        output_root=tmp_path,
        deployment_evidence_manifest_path=manifest_path,
    )

    _validate_closed_loop_deployment_evidence(config=config, mode="armed")


def test_armed_closed_loop_rejects_evidence_sensitive_calibration_drift(tmp_path: Path) -> None:
    profile = ACTIVE_CALIBRATION_PROFILE.to_manifest()
    profile["profile_hash"] = "pytest_bad_attitude_profile_hash"
    profile["vicon_attitude_signs"] = [1.0, 1.0, -1.0]
    manifest_path = tmp_path / "deployment_evidence_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "calibration_profile_hash": "pytest_bad_attitude_manifest_hash",
                "active_calibration_profile": profile,
                "evidence_regenerated_after_calibration": True,
                "r5_label": "E03",
                "r7_label": "E03",
                "r8_label": "E03",
                "r10_label": "E03",
                "r11_label": "E03.1",
            },
            sort_keys=True,
        ),
        encoding="ascii",
    )
    config = FlightRuntimeConfig(
        run_label="T_attitude_guard",
        output_root=tmp_path,
        deployment_evidence_manifest_path=manifest_path,
    )

    with pytest.raises(RuntimeError, match="vicon_attitude_signs"):
        _validate_closed_loop_deployment_evidence(config=config, mode="armed")


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


def test_real_flight_surface_state_estimator_matches_nominal_latency_contract() -> None:
    nominal = latency_case_config("nominal")
    config = FlightRuntimeConfig(run_label="latency_contract")
    assert config.surface_state_estimator_latency_case == "nominal"
    assert config.surface_command_delay_s == pytest.approx(
        nominal.command_onset_delay_s + nominal.command_transport_delay_s
    )
    assert config.actuator_tau_s == pytest.approx(nominal.actuator_tau_s)

    adapter = NausicaaViconStateAdapter(
        derivative_cutoff_hz=0.0,
        actuator_tau_s=config.actuator_tau_s,
        command_delay_s=config.surface_command_delay_s,
    )
    adapter.update(
        NausicaaViconSample(0.0, (1.2, 2.2, 1.5), euler_rad=(0.0, 0.0, 0.0)),
        command_norm=[0.0, 0.0, 0.0],
    )
    delayed_state = adapter.update(
        NausicaaViconSample(0.05, (1.4, 2.2, 1.5), euler_rad=(0.0, 0.0, 0.0)),
        command_norm=[1.0, 0.0, 0.0],
    )
    active_state = adapter.update(
        NausicaaViconSample(0.13, (1.6, 2.2, 1.5), euler_rad=(0.0, 0.0, 0.0)),
        command_norm=[1.0, 0.0, 0.0],
    )

    assert delayed_state[STATE_INDEX["delta_a"]] == 0.0
    assert active_state[STATE_INDEX["delta_a"]] > 0.0
    assert adapter.estimator_status()["surface_command_delay_s"] == pytest.approx(
        config.surface_command_delay_s
    )


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
        arena_transform=ViconArenaFrameTransform(
            attitude_signs=(1.0, -1.0, -1.0),
            attitude_offset_rad=(0.01, -0.02, 0.03),
        ),
    )

    state = adapter.update(
        NausicaaViconSample(0.0, (0.0, 0.0, 0.0), euler_rad=(0.10, -0.20, -0.30)),
        command_norm=[0.0, 0.0, 0.0],
    )

    assert np.isclose(state[STATE_INDEX["phi"]], 0.11)
    assert np.isclose(state[STATE_INDEX["theta"]], 0.18)
    assert np.isclose(state[STATE_INDEX["psi"]], 0.33)


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
    status = adapter.estimator_status()

    assert np.isclose(state[STATE_INDEX["p"]], 1.0, atol=1e-6)
    assert np.isclose(state[STATE_INDEX["q"]], 0.0, atol=1e-6)
    assert np.isclose(state[STATE_INDEX["r"]], 0.0, atol=1e-6)
    assert status["observer_mode"] == "so3_window"
    assert status["rate_confidence"] == 0.25

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


def test_so3_observer_rejects_one_frame_spike_but_keeps_sustained_rate() -> None:
    observer = SO3AngularRateObserver(window_frames=7, cutoff_hz=0.0, rate_limit_rad_s=6.0)
    for _ in range(6):
        rate, status = observer.update(np.eye(3), dt_s=0.01)

    assert np.isclose(rate[0], 0.0)
    assert status["rate_confidence"] >= 0.65

    spike_rotation = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(0.5), -np.sin(0.5)],
            [0.0, np.sin(0.5), np.cos(0.5)],
        ],
        dtype=float,
    )
    rate, status = observer.update(spike_rotation, dt_s=0.01)

    assert status["spike_rejected"] is True
    assert status["rate_confidence"] < 0.65
    assert abs(float(rate[0])) < 1.0

    sustained = SO3AngularRateObserver(window_frames=7, cutoff_hz=0.0, rate_limit_rad_s=6.0)
    for index in range(10):
        angle = 0.02 * index
        rotation = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(angle), -np.sin(angle)],
                [0.0, np.sin(angle), np.cos(angle)],
            ],
            dtype=float,
        )
        rate, status = sustained.update(rotation, dt_s=0.01)

    assert status["spike_rejected"] is False
    assert status["rate_confidence"] >= 0.65
    assert np.isclose(rate[0], 2.0, atol=0.25)


def test_vicon_rate_observer_uses_corrected_attitude_before_so3_rate() -> None:
    adapter = NausicaaViconStateAdapter(
        derivative_cutoff_hz=0.0,
        body_rate_limit_rad_s=6.0,
        arena_transform=ViconArenaFrameTransform(attitude_signs=(1.0, -1.0, -1.0)),
    )
    for index in range(6):
        raw_yaw = -0.02 * index
        state = adapter.update(
            NausicaaViconSample(
                0.01 * index,
                (0.0, 0.0, 0.0),
                euler_rad=(0.0, 0.0, raw_yaw),
                frame_number=index + 1,
                frame_rate_hz=100.0,
            ),
            command_norm=[0.0, 0.0, 0.0],
        )
    status = adapter.estimator_status()

    assert np.isclose(state[STATE_INDEX["psi"]], 0.10)
    assert state[STATE_INDEX["r"]] > 1.0
    assert status["rate_confidence"] >= 0.65
    assert status["attitude_transform_applied"] is True


def test_vicon_orientation_check_summary_includes_rate_sign_and_confidence() -> None:
    neutral = [_state(p=0.0, q=0.0, r=0.0) for _ in range(12)]
    roll_right = [_state(phi=0.25, p=0.25, q=0.0, r=0.0) for _ in range(12)]
    statuses = [
        {
            "rate_confidence": 0.9,
            "spike_rejected": False,
            "observer_mode": "so3_window",
            "rate_window_frames": 7,
        }
        for _ in range(12)
    ]

    rows = _evaluate_steps(
        {"neutral": neutral, "roll_right": roll_right},
        {"neutral": statuses, "roll_right": statuses},
        {
            "neutral": [{"frame_number": index, "frame_rate_hz": 200.0, "t_host_s": index / 200.0} for index in range(12)],
            "roll_right": [{"frame_number": index, "frame_rate_hz": 200.0, "t_host_s": index / 200.0} for index in range(12)],
        },
        requested_sample_rate_hz=200.0,
        step_duration_s=0.06,
    )
    roll_row = next(row for row in rows if row["step_id"] == "roll_right")
    neutral_row = next(row for row in rows if row["step_id"] == "neutral")

    assert neutral_row["passed"] is True
    assert roll_row["passed"] is True
    assert roll_row["rate_signal_name"] == "p"
    assert float(roll_row["observed_rate"]) > 0.0
    assert roll_row["rate_passed"] is True
    assert roll_row["rate_confidence_passed"] is True
    assert roll_row["sample_rate_passed"] is True


def test_vicon_orientation_check_sample_rate_summary_flags_slow_stream() -> None:
    fast = _sample_rate_summary(
        [{"frame_number": index, "frame_rate_hz": 200.0, "t_host_s": index / 200.0} for index in range(20)],
        requested_sample_rate_hz=200.0,
        step_duration_s=0.10,
    )
    slow = _sample_rate_summary(
        [{"frame_number": index, "frame_rate_hz": 50.0, "t_host_s": index / 50.0} for index in range(20)],
        requested_sample_rate_hz=200.0,
        step_duration_s=0.40,
    )

    assert fast["sample_rate_passed"] is True
    assert slow["sample_rate_passed"] is False
    assert float(slow["measured_sample_rate_hz"]) < 160.0


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
    state[STATE_INDEX["v"]] = LAUNCH_GATE_SIDE_VELOCITY_LIMIT_M_S
    state[STATE_INDEX["w"]] = LAUNCH_GATE_VERTICAL_BODY_VELOCITY_LIMIT_M_S

    assert LAUNCH_GATE_SIDE_VELOCITY_LIMIT_M_S == 1.5
    assert LAUNCH_GATE_VERTICAL_BODY_VELOCITY_LIMIT_M_S == 0.9
    assert LAUNCH_GATE_FORWARD_SPEED_MIN_M_S == 4.0
    assert LAUNCH_GATE_FORWARD_SPEED_MAX_M_S == 8.0
    assert SIM_LAUNCH_GATE_FORWARD_SPEED_MIN_M_S == LAUNCH_GATE_FORWARD_SPEED_MIN_M_S
    assert SIM_LAUNCH_GATE_FORWARD_SPEED_MAX_M_S == LAUNCH_GATE_FORWARD_SPEED_MAX_M_S
    assert SIM_LAUNCH_GATE_SIDE_VELOCITY_LIMIT_M_S == LAUNCH_GATE_SIDE_VELOCITY_LIMIT_M_S
    assert SIM_LAUNCH_GATE_VERTICAL_BODY_VELOCITY_LIMIT_M_S == LAUNCH_GATE_VERTICAL_BODY_VELOCITY_LIMIT_M_S
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

    state[STATE_INDEX["p"]] = 0.0
    state[STATE_INDEX["v"]] = LAUNCH_GATE_SIDE_VELOCITY_LIMIT_M_S + 0.01
    rejected = evaluate_launch_gate(state)
    assert rejected.approved is False
    assert rejected.reason == "side_velocity_outside_launch_gate"

    state[STATE_INDEX["v"]] = 0.0
    state[STATE_INDEX["w"]] = -(LAUNCH_GATE_VERTICAL_BODY_VELOCITY_LIMIT_M_S + 0.01)
    rejected = evaluate_launch_gate(state)
    assert rejected.approved is False
    assert rejected.reason == "vertical_body_velocity_outside_launch_gate"


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

    current[STATE_INDEX["q"]] = 2.6
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


def test_frozen_controller_prepares_continuation_without_emitting_packet() -> None:
    controller = FrozenFlightController(FlightRuntimeConfig(run_label="pytest_continuation_prepare"))
    selection_state = _state(x_w=1.8, y_w=2.2, z_w=1.6, u=6.0)
    commit_state = selection_state.copy()
    commit_state[STATE_INDEX["theta"]] += 0.04
    sequence_before = controller.sequence

    prepared = controller.prepare_continuation_decision(
        selection_state,
        primitive_step_index=1,
        target_boundary_s=0.14,
        prepare_started_elapsed_s=0.04,
        prediction_dt_s=0.10,
    )

    assert prepared["ready"] is True
    assert prepared["primitive_step_index"] == 1
    assert controller.sequence == sequence_before

    decision = controller.commit_prepared_continuation_decision(commit_state, primitive_step_index=1)

    assert decision.selected is True
    assert len(decision.packet_bytes) == 15
    assert controller.sequence == sequence_before + 1
    assert decision.decision_time_s >= float(prepared["decision_time_s"])


def test_runtime_boundary_predictor_propagates_short_horizon_state() -> None:
    state = _state(x_w=1.0, y_w=2.0, z_w=1.5, psi=0.0, u=6.0, v=0.5, w=-0.4, r=0.2)

    predicted = _predict_boundary_state(state, 0.10)

    assert predicted[STATE_INDEX["x_w"]] == pytest.approx(1.6)
    assert predicted[STATE_INDEX["y_w"]] == pytest.approx(2.05)
    assert predicted[STATE_INDEX["z_w"]] == pytest.approx(1.54)
    assert predicted[STATE_INDEX["psi"]] == pytest.approx(0.02)


def test_frozen_controller_pushes_command_fifo_old_to_new() -> None:
    controller = FrozenFlightController(FlightRuntimeConfig(run_label="pytest_fifo_order"))
    commands = [
        np.asarray([float(index), float(index) + 0.1, -float(index)], dtype=float)
        for index in range(5)
    ]

    for command in commands:
        controller._push_command_fifo("fifo_contract", command, delay_steps=4)

    fifo = np.vstack(controller._command_fifo_by_controller_id["fifo_contract"])
    np.testing.assert_allclose(fifo, np.vstack(commands[1:]))
    assert not np.allclose(fifo, np.vstack(list(reversed(commands[1:]))))


def test_frozen_controller_live_fifo_push_matches_explicit_old_to_new_command() -> None:
    payload, state, fifo_history = _fifo_order_sensitive_payload()
    controller_id = str(payload.get("controller_id", ""))

    explicit = FrozenFlightController(FlightRuntimeConfig(run_label="pytest_fifo_explicit"))
    explicit._command_fifo_by_controller_id[controller_id] = [row.copy() for row in fifo_history]
    expected_norm, expected_rad = explicit._command_for_payload(payload, state)

    live = FrozenFlightController(FlightRuntimeConfig(run_label="pytest_fifo_live"))
    for row in fifo_history:
        live._push_command_fifo(controller_id, row, delay_steps=len(fifo_history))
    live_norm, live_rad = live._command_for_payload(payload, state)

    reversed_controller = FrozenFlightController(FlightRuntimeConfig(run_label="pytest_fifo_reversed"))
    reversed_controller._command_fifo_by_controller_id[controller_id] = [
        row.copy() for row in reversed(fifo_history)
    ]
    reversed_norm, _ = reversed_controller._command_for_payload(payload, state)

    np.testing.assert_allclose(live_norm, expected_norm)
    np.testing.assert_allclose(live_rad, expected_rad)
    assert not np.allclose(reversed_norm, expected_norm)


def test_dry_run_writes_local_result_tree(tmp_path: Path) -> None:
    config = FlightRuntimeConfig(run_label="T01", output_root=tmp_path, max_duration_s=0.12)

    summary = run_real_flight(config, mode="dry-run")

    assert summary["completed"] is True
    assert summary["launch_handoff_duration_s"] == pytest.approx(LAUNCH_HANDOFF_DURATION_S)
    assert summary["launch_handoff_completed"] is True
    assert (tmp_path / "T01" / "manifests" / "real_flight_runtime_manifest.json").exists()
    assert (tmp_path / "T01" / "metrics" / "state_samples.csv").exists()


def test_closed_loop_dry_run_holds_neutral_during_launch_handoff(tmp_path: Path) -> None:
    config = FlightRuntimeConfig(
        run_label="T_handoff",
        output_root=tmp_path,
        max_duration_s=0.12,
        post_exit_neutral_tail_s=0.0,
    )

    summary = run_real_flight(config, mode="dry-run")

    assert summary["launch_handoff_completed"] is True
    assert summary["launch_handoff_neutral_packet_count"] == 2
    assert summary["first_launch_decision_ready_before_handoff"] is True
    assert summary["controller_decision_count"] >= 1
    assert float(summary["first_active_command_elapsed_s"]) >= LAUNCH_HANDOFF_DURATION_S
    events_path = tmp_path / "T_handoff" / "metrics" / "runtime_events.csv"
    with events_path.open(newline="") as handle:
        events = [row["event"] for row in csv.DictReader(handle)]

    assert "launch_handoff_start" in events
    assert "first_launch_decision_prepared" in events
    assert "launch_handoff_complete" in events
    assert "first_active_command" in events
    with (tmp_path / "T_handoff" / "metrics" / "controller_decisions.csv").open(newline="") as handle:
        decisions = list(csv.DictReader(handle))
    assert decisions
    assert "selected_score" in decisions[0]
    assert "selected_base_library_score_component" in decisions[0]
    assert "selected_mission_score_component" in decisions[0]
    with (tmp_path / "T_handoff" / "metrics" / "posthoc_throw.csv").open(newline="") as handle:
        posthoc = list(csv.DictReader(handle))
    assert len(posthoc) == 1
    assert int(posthoc[0]["executed_selected_decision_count"]) >= 1


def test_missed_first_launch_decision_is_not_counted_valid_throw(tmp_path: Path, monkeypatch) -> None:
    config = FlightRuntimeConfig(
        run_label="T_missed_handoff",
        output_root=tmp_path,
        max_duration_s=0.12,
        post_exit_neutral_tail_s=0.0,
    )
    controller = FrozenFlightController(config)

    def slow_launch_decision(*_args, **_kwargs) -> dict[str, object]:
        return {
            "ready": True,
            "reason": "selected",
            "decision_time_s": LAUNCH_HANDOFF_DURATION_S + 0.010,
            "candidate_count": 1,
            "viable_count": 1,
            "primitive_variant_id": "pytest_slow_handoff",
            "primitive_id": "pytest",
            "controller_id": "pytest",
            "expected_energy_residual_m": 0.0,
        }

    monkeypatch.setattr(controller, "prepare_launch_handoff_decision", slow_launch_decision)

    summary = run_real_flight(config, mode="dry-run", controller=controller)

    assert summary["launch_gate_approved"] is True
    assert summary["valid_throw"] is False
    assert summary["flight_cancelled"] is True
    assert summary["cancellation_reason"] == "first_launch_decision_missed_handoff_budget"
    assert summary["controller_decision_count"] == 0
    assert summary["slot_command_update_count"] == 0


def test_memory_candidate_feature_evaluator_reuses_decision_lookup_context(monkeypatch) -> None:
    import real_flight_memory as memory_module

    memory = memory_module.RealFlightMemoryState(enabled=True)
    memory.update_from_decision_records(
        [
            {"state": _state(x_w=1.4, y_w=2.1, z_w=1.5), "t_s": 0.0, "expected_energy_residual_m": 0.0},
            {"state": _state(x_w=2.0, y_w=2.1, z_w=1.55), "t_s": 0.1, "expected_energy_residual_m": 0.0},
            {"state": _state(x_w=2.6, y_w=2.15, z_w=1.6), "t_s": 0.2, "expected_energy_residual_m": 0.0},
        ]
    )
    assert memory.cell_count() > 0

    original_cell_lookup = memory_module.directional_residual_lift_cell_lookup
    original_spatial_lookup = memory_module.directional_residual_lift_spatial_cell_lookup
    original_query = memory_module.query_spatial_flow_belief_features_fast
    lookup_counts = {"cell": 0, "spatial": 0, "query": 0}

    def counted_cell_lookup(*args, **kwargs):
        lookup_counts["cell"] += 1
        return original_cell_lookup(*args, **kwargs)

    def counted_spatial_lookup(*args, **kwargs):
        lookup_counts["spatial"] += 1
        return original_spatial_lookup(*args, **kwargs)

    def counted_query(*args, **kwargs):
        lookup_counts["query"] += 1
        return original_query(*args, **kwargs)

    monkeypatch.setattr(memory_module, "directional_residual_lift_cell_lookup", counted_cell_lookup)
    monkeypatch.setattr(memory_module, "directional_residual_lift_spatial_cell_lookup", counted_spatial_lookup)
    monkeypatch.setattr(memory_module, "query_spatial_flow_belief_features_fast", counted_query)

    evaluator = memory.candidate_feature_evaluator(current_state=_state(x_w=1.35, y_w=2.1, z_w=1.5))
    representatives = [
        {"primitive_id": "recovery", "local_lqr_reference_speed_m_s": 5.8, "__memory_query_mode": "geometry_only"},
        {"primitive_id": "energy_retaining_bank", "local_lqr_reference_speed_m_s": 5.9, "__memory_query_mode": "geometry_only"},
        {"primitive_id": "mild_turn_left", "local_lqr_reference_speed_m_s": 6.0, "__memory_query_mode": "geometry_only"},
    ]
    for representative in representatives:
        features = evaluator(representative, {})
        assert features is not None
        assert features["belief_candidate_path_residual_memory_active"] is False
    assert lookup_counts == {"cell": 1, "spatial": 1, "query": 0}

    features = evaluator({"primitive_id": "recovery", "local_lqr_reference_speed_m_s": 5.8}, {})
    assert features is not None
    assert features["belief_candidate_path_residual_memory_active"] is True
    assert lookup_counts["cell"] == 1
    assert lookup_counts["spatial"] == 1
    assert lookup_counts["query"] > 0


def test_closed_loop_dry_run_buffers_active_metrics_off_timing_path(tmp_path: Path) -> None:
    config = FlightRuntimeConfig(
        run_label="T_buffered_active_metrics",
        output_root=tmp_path,
        max_duration_s=0.12,
        post_exit_neutral_tail_s=0.0,
    )

    summary = run_real_flight(config, mode="dry-run")

    assert summary["active_metric_logging_policy"] == "buffer_active_rows_flush_after_active_record"
    assert int(summary["active_metric_buffered_row_count"]) > 0
    assert int(summary["active_metric_buffer_flush_count"]) == int(summary["active_metric_buffered_row_count"])
    assert summary["active_fan_logging_policy"] == "single_prelaunch_snapshot_only"
    assert float(summary["active_runtime_wake_ahead_s"]) > 0.0
    assert (tmp_path / "T_buffered_active_metrics" / "metrics" / "controller_decisions.csv").exists()
    assert (tmp_path / "T_buffered_active_metrics" / "metrics" / "runtime_events.csv").exists()
    assert (tmp_path / "T_buffered_active_metrics" / "metrics" / "state_samples.csv").exists()


def test_fan_positions_are_single_prelaunch_snapshot_per_throw(tmp_path: Path) -> None:
    config = FlightRuntimeConfig(
        run_label="T_fan_snapshot",
        output_root=tmp_path,
        experiment_case_id="E3.1",
        experiment_case_name="Fixed four-fan snapshot contract",
        experiment_layout_id="four_fan_fixed",
        max_duration_s=0.12,
        post_exit_neutral_tail_s=0.0,
    )

    summary = run_real_flight(config, mode="dry-run", expected_visible_fan_range=(4, 4))

    assert summary["completed"] is True
    assert summary["active_fan_logging_policy"] == "single_prelaunch_snapshot_only"
    assert summary["fan_visible_count_latest"] == 4
    assert summary["fan_expected_count_ok_latest"] is True
    fan_path = tmp_path / "T_fan_snapshot" / "metrics" / "fan_positions.csv"
    with fan_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 4
    assert {row["phase"] for row in rows} == {"prelaunch"}
    assert {row["fan_subject"] for row in rows} == {"Fan_1", "Fan_2", "Fan_3", "Fan_4"}
    assert all(row["visible"] == "True" for row in rows)
    assert all(row["visible_count"] == "4" for row in rows)
    assert all(row["expected_count_ok"] == "True" for row in rows)
    assert {"t_host_s", "phase", "fan_subject", "visible", "reason", "x_w", "y_w", "z_w"}.issubset(
        rows[0].keys()
    )


def test_vicon_fan_position_check_uses_five_cm_independent_xy_tolerance() -> None:
    assert DEFAULT_FAN_POSITION_TOLERANCE_M == pytest.approx(0.05)
    target = np.asarray([3.0, 1.2, 0.75], dtype=float)

    delta, error = _fan_position_error_m(target + np.asarray([0.05, -0.05, 0.40]), target)
    np.testing.assert_allclose(delta, [0.05, -0.05, 0.40])
    assert error == pytest.approx(0.05)

    _, just_outside_x = _fan_position_error_m(target + np.asarray([0.051, 0.0, 0.0]), target)
    _, just_outside_y = _fan_position_error_m(target + np.asarray([0.0, -0.051, 0.0]), target)
    _, z_only = _fan_position_error_m(target + np.asarray([0.0, 0.0, 1.0]), target)
    assert just_outside_x > DEFAULT_FAN_POSITION_TOLERANCE_M
    assert just_outside_y > DEFAULT_FAN_POSITION_TOLERANCE_M
    assert z_only == pytest.approx(0.0)


def test_open_loop_neutral_dry_run_records_state_without_controller_or_memory(tmp_path: Path) -> None:
    config = FlightRuntimeConfig(
        run_label="T_open_loop",
        output_root=tmp_path,
        controller_mode="open_loop_neutral",
        experiment_case_id="E1.0",
        experiment_case_name="Dry air, open-loop neutral",
        max_duration_s=0.12,
        post_exit_neutral_tail_s=0.0,
    )

    summary = run_real_flight(config, mode="dry-run")

    assert summary["completed"] is True
    assert summary["valid_throw"] is True
    assert summary["controller_mode"] == "open_loop_neutral"
    assert summary["launch_handoff_completed"] is True
    assert summary["controller_decision_count"] == 0
    assert summary["open_loop_neutral_packet_count"] > 0
    assert summary["memory_update_observation_count"] == 0
    assert not (tmp_path / "T_open_loop" / "metrics" / "controller_decisions.csv").exists()
    assert (tmp_path / "T_open_loop" / "metrics" / "state_samples.csv").exists()
    with (tmp_path / "T_open_loop" / "metrics" / "posthoc_throw.csv").open(newline="") as handle:
        posthoc = list(csv.DictReader(handle))
    assert len(posthoc) == 1
    assert posthoc[0]["memory_history_bucket"] == "open_loop"
    assert int(posthoc[0]["executed_selected_decision_count"]) == 0


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


def test_flight_record_rejects_crossed_launch_attempt_without_waiting_for_timeout(tmp_path: Path) -> None:
    config = FlightRuntimeConfig(
        run_label="T_rejected_crossing",
        output_root=tmp_path,
        max_duration_s=0.12,
        launch_wait_timeout_s=10.0,
        post_exit_neutral_tail_s=0.0,
        vicon_position_offset_m=(3.7, 2.2, 1.0),
    )

    started = time.perf_counter()
    summary = run_real_flight(config, mode="dry-run")
    elapsed_s = time.perf_counter() - started

    assert summary["flight_cancelled"] is True
    assert summary["valid_throw"] is False
    assert str(summary["cancellation_reason"]).startswith("rejected_launch_attempt:")
    assert elapsed_s < 1.0
    assert not (tmp_path / "T_rejected_crossing" / "metrics" / "state_samples.csv").exists()


def test_glider_calibration_case_registry_uses_neutral_and_0p2_lattice() -> None:
    neutral_cases = calibration_cases_for_block("neutral_30")
    pulse_blocks = {
        "pulse_ladder_elevator_30": "delta_e",
        "pulse_ladder_aileron_30": "delta_a",
        "pulse_ladder_rudder_30": "delta_r",
    }

    assert PULSE_START_DELAY_S == 0.15
    assert set(PULSE_DURATION_BY_ABS_COMMAND) == {0.2, 0.4, 0.6, 0.8, 1.0}
    assert set(PULSE_DURATION_BY_ABS_COMMAND.values()) == {SUSTAINED_CONTROL_EFFECT_DURATION_S}

    assert len(neutral_cases) == 1
    assert neutral_cases[0].case_id == "C0_neutral"
    assert neutral_cases[0].target_valid_throws == 30
    assert neutral_cases[0].is_neutral is True

    for block_id, axis in pulse_blocks.items():
        axis_cases = calibration_cases_for_block(block_id)
        assert len(axis_cases) == 10
        assert {case.command_axis for case in axis_cases} == {axis}
        assert [case.command_value for case in axis_cases] == [
            0.2,
            -0.2,
            0.4,
            -0.4,
            0.6,
            -0.6,
            0.8,
            -0.8,
            1.0,
            -1.0,
        ]
        for case in axis_cases:
            assert case.pulse_start_s == PULSE_START_DELAY_S
            assert case.pulse_duration_s == PULSE_DURATION_BY_ABS_COMMAND[round(abs(case.command_value), 1)]
            assert case.target_valid_throws == 3


def test_glider_calibration_control_effect_command_is_single_axis_and_sustained() -> None:
    case = next(
        item
        for item in calibration_cases_for_block("pulse_ladder_elevator_30")
        if item.command_axis == "delta_e" and np.isclose(item.command_value, 0.6)
    )

    before = _command_for_case(case, case.pulse_start_s - 1e-3)
    onset = _command_for_case(case, case.pulse_start_s)
    sustained = _command_for_case(case, case.pulse_start_s + 5.0)
    after = _command_for_case(case, case.pulse_start_s + case.pulse_duration_s + 1e-3)

    np.testing.assert_allclose(before, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(onset, [0.0, 0.6, 0.0])
    np.testing.assert_allclose(sustained, [0.0, 0.6, 0.0])
    np.testing.assert_allclose(after, [0.0, 0.0, 0.0])

    neutral = calibration_cases_for_block("neutral_30")[0]
    np.testing.assert_allclose(_command_for_case(neutral, 10.0), [0.0, 0.0, 0.0])


def test_glider_calibration_logs_profile_hash_schema_and_continuous_sequence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import run_glider_calibration_sequence as calibration_module

    monkeypatch.setattr(calibration_module, "RESULT_ROOT", tmp_path)
    result = run_calibration_sequence(
        block_id="neutral_30",
        session_label="pytest_cal_schema",
        mode="dry-run",
        serial_port="COM_TEST",
        vicon_host="mock",
        pre_arm_delay_s=0.0,
        cooldown_s=0.0,
        retry_cooldown_s=0.0,
        launch_wait_timeout_s=0.20,
        max_duration_s=0.12,
        post_exit_neutral_tail_s=0.0,
        vicon_tracking_rate_hz=200.0,
        vicon_position_offset_m=ACTIVE_CALIBRATION_PROFILE.vicon_position_offset_m,
        vicon_yaw_alignment_deg=ACTIVE_CALIBRATION_PROFILE.vicon_yaw_alignment_deg,
        vicon_attitude_signs=ACTIVE_CALIBRATION_PROFILE.vicon_attitude_signs,
        vicon_attitude_offset_rad=ACTIVE_CALIBRATION_PROFILE.vicon_attitude_offset_rad,
        target_valid_throws=1,
    )

    assert result["total_valid_throw_count"] == 1
    run_root = tmp_path / "cal" / "n30" / "pytest_cal_schema" / "c0_neu" / "v001"
    manifest = json.loads((run_root / "manifests" / "glider_calibration_throw_manifest.json").read_text())
    summary = json.loads((run_root / "manifests" / "glider_calibration_throw_summary.json").read_text())
    with (run_root / "metrics" / "command_schedule.csv").open(newline="") as handle:
        command_rows = list(csv.DictReader(handle))
    with (run_root / "metrics" / "state_samples.csv").open(newline="") as handle:
        state_rows = list(csv.DictReader(handle))

    assert manifest["calibration_profile"]["profile_hash"] == ACTIVE_CALIBRATION_PROFILE.profile_hash()
    assert manifest["case_storage_id"] == "c0_neu"
    assert result["session_root"].endswith("/cal/n30/pytest_cal_schema")
    assert "calibration_csv_schema" in manifest
    assert int(summary["active_packet_sequence_start"]) > 0
    assert int(command_rows[0]["packet_sequence"]) >= int(summary["active_packet_sequence_start"])
    assert command_rows[0]["aggregate_command_units"] == "normalized_aggregate_command"
    assert command_rows[0]["physical_surface_units"] == "normalized_physical_surface_command"
    assert command_rows[0]["packet_surface_units"] == "normalized_servo_signed_packet_surface_command"
    assert command_rows[0]["receiver_channel_units"] == "uint16_after_receiver_channel_order"
    assert state_rows[0]["surface_state_source"] == "estimated_actuator_state_not_measured_surface_marker"


def test_glider_calibration_storage_ids_keep_git_paths_short() -> None:
    block_id = "pulse_ladder_aileron_30"
    case = next(item for item in calibration_cases_for_block(block_id) if np.isclose(item.command_value, 1.0))
    relative_path = (
        Path("04_Flight_Test")
        / "05_Results"
        / "cal"
        / _block_storage_id(block_id)
        / "20260603_000000"
        / _case_storage_id(case)
        / "bad"
        / "i001"
        / "manifests"
        / "glider_calibration_throw_manifest.json"
    )

    assert relative_path.as_posix() == (
        "04_Flight_Test/05_Results/cal/pa30/20260603_000000/"
        "c1_a_p10/bad/i001/manifests/glider_calibration_throw_manifest.json"
    )
    assert len(relative_path.as_posix()) < 130


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


def test_launch_gate_default_debounces_two_consecutive_approved_frames(tmp_path: Path) -> None:
    config = FlightRuntimeConfig(
        run_label="T_launch_debounce",
        output_root=tmp_path,
        max_duration_s=0.12,
        post_exit_neutral_tail_s=0.0,
    )

    assert config.launch_gate_required_consecutive_frames == 2

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
    assert max(approved_counts) == 2
    assert (tmp_path / "T_launch_debounce" / "metrics" / "state_samples.csv").exists()


def test_launch_gate_rejects_low_rate_confidence_even_when_bounds_pass(tmp_path: Path) -> None:
    config = FlightRuntimeConfig(
        run_label="T_launch_confidence",
        output_root=tmp_path,
        max_duration_s=0.12,
        launch_wait_timeout_s=0.08,
        post_exit_neutral_tail_s=0.0,
        launch_gate_rate_confidence_min=0.99,
    )

    summary = run_real_flight(config, mode="dry-run")

    assert summary["flight_cancelled"] is True
    assert summary["launch_gate_approved"] is False
    prelaunch_path = tmp_path / "T_launch_confidence" / "metrics" / "prelaunch_state_samples.csv"
    with prelaunch_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert any(row["base_trigger_approved_before_rate_confidence"].lower() == "true" for row in rows)
    assert any(row["rate_confidence_ok"].lower() == "false" for row in rows)


def test_launch_gate_uses_formal_realistic_rate_limits() -> None:
    state = _state(x_w=1.3, y_w=2.0, z_w=1.65, u=6.0, p=1.0, q=0.8, r=1.4)

    assert FlightRuntimeConfig(run_label="formal").launch_gate_body_rate_limits_rad_s == (1.2, 1.2, 1.8)
    assert evaluate_launch_gate(state).approved is True
    assert evaluate_launch_plane_gate(state).approved is True

    rejected = _state(x_w=1.3, y_w=2.0, z_w=1.65, u=6.0, p=1.21, q=0.8, r=1.4)
    assert evaluate_launch_gate(rejected).approved is False
    assert evaluate_launch_gate(rejected).reason == "roll_rate_outside_launch_gate"


def test_experiment_case_registry_contains_requested_cases() -> None:
    requested = {
        "E0.1",
        "E0.2",
        "E1.0",
        "E1.1",
        "E1.2",
        "E2.0",
        "E2.1",
        "E2.2",
        "E3.0",
        "E3.1",
        "E3.2",
        "E4a.0",
        "E4a.1",
        "E4a.2",
        "E4b.0",
        "E4b.1",
        "E4b.2",
        "E4c.0",
        "E4c.1",
        "E4c.2",
        "E5a.0",
        "E5a.1",
        "E5a.2",
        "E5b.0",
        "E5b.1",
        "E5b.2",
        "E5c.0",
        "E5c.1",
        "E5c.2",
        "E5d.0",
        "E5d.1",
        "E5d.2",
    }
    assert requested.issubset(EXPERIMENT_CASES)
    assert get_experiment_case("E2.2").memory_enabled is True
    assert get_experiment_case("E1.1").memory_enabled is False
    assert get_experiment_case("E3.0").controller_mode == "open_loop_neutral"
    for case_id in ("E2.2", "E3.2"):
        case = get_experiment_case(case_id)
        assert case.target_valid_throws == 30
        assert case.target_session_repeats == 3
    for case_id in ("E4a.2", "E4b.2", "E4c.2", "E5a.2", "E5b.2", "E5c.2", "E5d.2"):
        case = get_experiment_case(case_id)
        assert case.target_valid_throws == 30
        assert case.target_session_repeats == 2


def test_replay_fan_tracker_handles_zero_one_and_four_subjects() -> None:
    replay = ReplayNausicaaViconRigidBody().open()
    replay.read_latest()
    assert sum(1 for fan in replay.read_fans(("Missing_Fan",)) if fan.visible) == 0
    assert sum(1 for fan in replay.read_fans(("Fan_1",)) if fan.visible) == 1
    assert sum(1 for fan in replay.read_fans(("Fan_1", "Fan_2", "Fan_3", "Fan_4")) if fan.visible) == 4


def test_experiment_sequence_propagates_formal_rate_gate_to_throw_runtime(tmp_path: Path, monkeypatch) -> None:
    import run_experiment_sequence as sequence_module

    monkeypatch.setattr(sequence_module, "RESULT_ROOT", tmp_path)
    result = run_experiment_sequence(
        case_id="E0.1",
        session_label="pytest_e0_gate",
        mode="dry-run",
        serial_port="COM_TEST",
        vicon_host="mock",
        target_valid_throws=1,
        cooldown_s=0.0,
        retry_cooldown_s=0.0,
        max_invalid_attempts=1,
        max_duration_s=0.12,
        launch_wait_timeout_s=0.20,
        post_exit_neutral_tail_s=0.0,
        vicon_poll_period_s=0.005,
        vicon_position_offset_m=ACTIVE_CALIBRATION_PROFILE.vicon_position_offset_m,
        vicon_yaw_alignment_deg=0.0,
        vicon_attitude_signs=(1.0, -1.0, -1.0),
        vicon_attitude_offset_rad=ACTIVE_CALIBRATION_PROFILE.vicon_attitude_offset_rad,
    )

    assert result["valid_throw_count"] == 1
    session_manifest = json.loads(
        (tmp_path / "E0.1" / "pytest_e0_gate" / "manifests" / "experiment_sequence_manifest.json").read_text()
    )
    throw_manifest = json.loads(
        (
            tmp_path
            / "E0.1"
            / "pytest_e0_gate"
            / "throw_001"
            / "manifests"
            / "real_flight_runtime_manifest.json"
        ).read_text()
    )

    assert session_manifest["launch_gate_body_rate_limits_rad_s"] == [1.2, 1.2, 1.8]
    assert throw_manifest["config"]["launch_gate_body_rate_limits_rad_s"] == [1.2, 1.2, 1.8]
    assert throw_manifest["launch_gate_bounds"]["p_rad_s"] == [-1.2, 1.2]
    assert throw_manifest["launch_gate_bounds"]["q_rad_s"] == [-1.2, 1.2]
    assert throw_manifest["launch_gate_bounds"]["r_rad_s"] == [-1.8, 1.8]


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
        vicon_position_offset_m=ACTIVE_CALIBRATION_PROFILE.vicon_position_offset_m,
        vicon_yaw_alignment_deg=0.0,
        vicon_attitude_signs=(1.0, -1.0, -1.0),
        vicon_attitude_offset_rad=ACTIVE_CALIBRATION_PROFILE.vicon_attitude_offset_rad,
    )
    assert result["valid_throw_count"] == 2
    assert result["invalid_attempt_count"] == 0
    assert int(result["memory"]["memory_launch_index"]) >= 1
    assert (tmp_path / "E2.2" / "pytest_memory" / "throw_001" / "metrics" / "memory_update_summary.csv").exists()


def test_experiment_case_repeat_wrapper_resets_memory_between_sessions(tmp_path: Path, monkeypatch) -> None:
    import run_experiment_sequence as sequence_module

    monkeypatch.setattr(sequence_module, "RESULT_ROOT", tmp_path)
    result = run_experiment_case_repeats(
        case_id="E2.2",
        session_label="pytest_repeat",
        mode="dry-run",
        serial_port="COM_TEST",
        vicon_host="mock",
        target_valid_throws=1,
        repeat_sessions=2,
        cooldown_s=0.0,
        retry_cooldown_s=0.0,
        max_invalid_attempts=2,
        max_duration_s=0.12,
        launch_wait_timeout_s=0.20,
        post_exit_neutral_tail_s=0.0,
        vicon_poll_period_s=0.005,
        vicon_position_offset_m=ACTIVE_CALIBRATION_PROFILE.vicon_position_offset_m,
        vicon_yaw_alignment_deg=0.0,
        vicon_attitude_signs=(1.0, -1.0, -1.0),
        vicon_attitude_offset_rad=ACTIVE_CALIBRATION_PROFILE.vicon_attitude_offset_rad,
    )

    assert result["session_repeat_count"] == 2
    assert result["target_protocol_valid_throws"] == 2
    assert result["valid_throw_count"] == 2
    assert len(result["session_results"]) == 2
    assert (tmp_path / "E2.2" / "pytest_repeat_r01" / "throw_001").exists()
    assert (tmp_path / "E2.2" / "pytest_repeat_r02" / "throw_001").exists()
    for session_result in result["session_results"]:
        assert session_result["valid_throw_count"] == 1
        assert int(session_result["memory"]["memory_launch_index"]) == 1


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
        vicon_position_offset_m=(
            1000.0,
            ACTIVE_CALIBRATION_PROFILE.vicon_position_offset_m[1],
            ACTIVE_CALIBRATION_PROFILE.vicon_position_offset_m[2],
        ),
        vicon_yaw_alignment_deg=0.0,
        vicon_attitude_signs=(1.0, -1.0, -1.0),
        vicon_attitude_offset_rad=ACTIVE_CALIBRATION_PROFILE.vicon_attitude_offset_rad,
    )
    assert result["valid_throw_count"] == 0
    assert result["invalid_attempt_count"] >= 1
    assert int(result["memory"]["memory_launch_index"]) == 0
    assert (tmp_path / "E2.2" / "pytest_invalid" / "invalid_attempts" / "attempt_001").exists()


def test_surface_sign_check_uses_same_packet_contract(tmp_path: Path, monkeypatch) -> None:
    import run_surface_sign_check as sign_module

    monkeypatch.setattr(sign_module, "RESULT_ROOT", tmp_path)
    sequence_by_name = {
        step_name: command
        for step_name, command, _expected_motion in SURFACE_CHECK_SEQUENCE
    }
    assert sequence_by_name["aileron_+1.0"] == (1.0, 0.0, 0.0)
    assert sequence_by_name["elevator_+1.0"] == (0.0, 1.0, 0.0)
    assert sequence_by_name["rudder_+1.0"] == (0.0, 0.0, 1.0)
    assert encode_arduino_command_packet(np.asarray(sequence_by_name["aileron_+1.0"])).physical_surface_norm == pytest.approx(
        (1.0, -1.0, 0.0, 0.0)
    )
    assert encode_arduino_command_packet(np.asarray(sequence_by_name["elevator_+1.0"])).physical_surface_norm == pytest.approx(
        (0.0, 0.0, 0.0, 1.0)
    )
    assert encode_arduino_command_packet(np.asarray(sequence_by_name["rudder_+1.0"])).physical_surface_norm == pytest.approx(
        (0.0, 0.0, 1.0, 0.0)
    )
    result = run_surface_sign_check(
        serial_port="COM_TEST",
        mode="dry-run",
        dwell_s=0.0,
        run_label="pytest_surface",
    )
    assert result["packet_count"] >= len(SURFACE_CHECK_SEQUENCE) + 1
    assert (tmp_path / "surface_sign_check" / "pytest_surface" / "metrics" / "surface_sign_check.csv").exists()
