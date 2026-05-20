from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTROL_DIR = REPO_ROOT / "03_Control"
PRIMITIVES_DIR = CONTROL_DIR / "03_Primitives"
SCENARIOS_DIR = CONTROL_DIR / "04_Scenarios"
for path in (PRIMITIVES_DIR, SCENARIOS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dense_archive_trial_logging import (  # noqa: E402
    DENSE_TRIAL_DESCRIPTOR_COLUMNS,
    DenseTrialDescriptorConfig,
    dense_trial_descriptor_row,
    dense_trial_match_key,
)
from wing_wind_descriptors import WING_WIND_DESCRIPTOR_COLUMNS  # noqa: E402


EXPECTED_COLUMNS = (
    "trial_descriptor_id",
    "layout_branch_id",
    "fan_layout",
    "fan_config_id",
    "test_environment_mode",
    "paired_environment_mode",
    "environment_role",
    "validity_gate_role",
    "acceptance_interpretation",
    "candidate_id",
    "sample_id",
    "paired_sample_key",
    "seed",
    "replay_seed",
    "sampling_round",
    "updraft_model_id",
    "family",
    "target_heading_deg",
    "direction_sign",
    "start_class",
    "x0_w_m",
    "y0_w_m",
    "z0_w_m",
    "speed0_m_s",
    "phi0_rad",
    "theta0_rad",
    "psi0_rad",
    "u0_m_s",
    "v0_m_s",
    "w0_m_s",
    "p0_rad_s",
    "q0_rad_s",
    "r0_rad_s",
    "updraft_center_x_m",
    "updraft_center_y_m",
    "updraft_relative_radius_m",
    "updraft_relative_azimuth_rad",
    "updraft_relative_height_m",
    "updraft_sector_label",
    *WING_WIND_DESCRIPTOR_COLUMNS,
    "duration_s",
    "heading_initial_deg",
    "heading_terminal_deg",
    "heading_change_deg",
    "heading_error_deg",
    "path_length_xy_m",
    "path_length_3d_m",
    "forward_displacement_m",
    "lateral_displacement_m",
    "xy_bounding_box_area_m2",
    "turn_footprint_proxy_m2",
    "entry_clearance_required_x_plus_m",
    "entry_clearance_required_x_minus_m",
    "entry_clearance_required_y_plus_m",
    "entry_clearance_required_y_minus_m",
    "floor_margin_required_m",
    "ceiling_margin_required_m",
    "min_true_margin_m",
    "floor_margin_min_m",
    "ceiling_margin_min_m",
    "speed_min_m_s",
    "terminal_speed_m_s",
    "specific_energy_initial_m",
    "specific_energy_terminal_m",
    "energy_residual_m",
    "lift_dwell_fraction",
    "lift_dwell_fraction_status",
    "wind_query_region",
    "saturation_fraction",
    "latency_case",
    "latency_acceptance_scope",
    "latency_pass_label",
    "state_feedback_delay_s",
    "command_onset_delay_s",
    "command_transport_delay_s",
    "actuator_tau_s",
    "actuator_t50_s",
    "actuator_t90_s",
    "latency_jitter_s",
    "timing_model_version",
    "state_feedback_delay_applied",
    "success_flag",
    "failure_label",
    "governor_rejection_cause",
    "robustness_label",
    "physics_priority_level",
    "sim_real_match_key",
    "sim_real_match_key_version",
    "sim_real_transfer_result",
    "descriptor_status",
)


def test_schema_order_and_csv_ready_scalars() -> None:
    row = _descriptor_row()

    assert DENSE_TRIAL_DESCRIPTOR_COLUMNS == EXPECTED_COLUMNS
    assert tuple(row) == DENSE_TRIAL_DESCRIPTOR_COLUMNS
    for value in row.values():
        assert not isinstance(value, (list, tuple, dict, np.ndarray))


def test_shape_validation_and_nonfinite_status_contract() -> None:
    time_s, x_ref, requested, effective, applied, delta = _straight_arrays()

    with pytest.raises(ValueError, match="must share the same N"):
        _descriptor_row(time_s=time_s, u_norm_applied=applied[:-1])
    with pytest.raises(ValueError, match="lift_exposure_m_s"):
        _descriptor_row(lift_exposure_m_s=np.array([0.1, 0.1]))

    nonfinite = x_ref.copy()
    nonfinite[1, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite x_ref"):
        _descriptor_row(x_ref=nonfinite)
    row = _descriptor_row(
        x_ref=nonfinite,
        descriptor_status="nonfinite_state",
        time_s=time_s,
        u_norm_requested=requested,
        u_norm_effective_target=effective,
        u_norm_applied=applied,
        delta_cmd_rad=delta,
    )
    assert row["descriptor_status"] == "nonfinite_state"


def test_match_key_replay_seed_and_trial_descriptor_id_are_deterministic() -> None:
    key = _match_key(replay_seed=99)
    row_a = _descriptor_row(sim_real_match_key=key, replay_seed=99)
    row_b = _descriptor_row(sim_real_match_key=key, replay_seed=99)
    row_c = _descriptor_row(sim_real_match_key=_match_key(replay_seed=100), replay_seed=100)

    assert row_a["sim_real_match_key"] == key
    assert row_a["sim_real_match_key_version"] == "dense_trial_match_key_v1"
    assert row_a["trial_descriptor_id"].startswith("trial_descriptor_id_v1|")
    assert row_a["trial_descriptor_id"] != row_a["sim_real_match_key"]
    assert row_a["trial_descriptor_id"] == row_b["trial_descriptor_id"]
    assert row_a["trial_descriptor_id"] != row_c["trial_descriptor_id"]
    assert row_a["replay_seed"] == 99
    assert "replay_seed=99" in row_a["trial_descriptor_id"]
    assert "replay_seed=99" in key

    changed = _match_key(test_environment_mode="W0_single_fan_branch")
    assert changed != _match_key(test_environment_mode="W1_single_fan")


def test_success_flag_physics_priority_and_sim_real_defaults() -> None:
    success = _descriptor_row(failure_label="success", governor_rejection_cause="none")
    assert success["success_flag"] is True
    assert success["sim_real_transfer_result"] == "not_evaluated"
    assert success["physics_priority_level"] == "target_steering_30deg_priority"

    assert _descriptor_row(robustness_label="stress_failed")["success_flag"] is False
    assert _descriptor_row(failure_label="target_miss")["success_flag"] is False
    assert _descriptor_row(governor_rejection_cause="entry_reject")["success_flag"] is False
    assert _descriptor_row(target_heading_deg=15.0)["physics_priority_level"] == (
        "target_steering_15deg_priority"
    )
    assert _descriptor_row(target_heading_deg=45.0)["physics_priority_level"] == (
        "target_steering_future_target_priority"
    )
    assert _descriptor_row(family="glide", target_heading_deg="")["physics_priority_level"] == (
        "not_priority_ranked"
    )
    assert _descriptor_row(sim_real_transfer_result="matched")["sim_real_transfer_result"] == "matched"


def test_lift_dwell_interval_and_proxy_semantics() -> None:
    time_s = np.array([0.0, 1.0, 3.0, 6.0])
    x_ref = _state_from_points(
        np.array(
            [
                [2.0, 1.0, 1.0],
                [3.0, 1.0, 1.0],
                [4.0, 1.0, 1.0],
                [5.0, 1.0, 1.0],
            ]
        ),
        np.zeros(4),
    )
    commands = np.zeros((4, 3))
    row = _descriptor_row(
        time_s=time_s,
        x_ref=x_ref,
        u_norm_requested=commands,
        u_norm_effective_target=commands,
        u_norm_applied=commands,
        delta_cmd_rad=commands,
        lift_exposure_m_s=np.array([0.10, 0.00, 0.10, 0.00]),
    )
    assert row["lift_dwell_fraction_status"] == "trajectory_exposure"
    assert row["lift_dwell_fraction"] == pytest.approx(4.0 / 6.0)

    assert _descriptor_row(w_wing_mean_m_s=0.06)["lift_dwell_fraction"] == 1.0
    assert _descriptor_row(w_wing_mean_m_s=0.04)["lift_dwell_fraction"] == 0.0
    unavailable = _descriptor_row(w_wing_mean_m_s=np.nan)
    assert unavailable["lift_dwell_fraction_status"] == "not_available"
    assert np.isnan(unavailable["lift_dwell_fraction"])


def test_straight_toy_trajectory_metrics_and_blank_target_heading() -> None:
    row = _descriptor_row(target_heading_deg="")

    assert row["duration_s"] == pytest.approx(2.0)
    assert row["heading_initial_deg"] == pytest.approx(0.0)
    assert row["heading_terminal_deg"] == pytest.approx(0.0)
    assert row["heading_change_deg"] == pytest.approx(0.0)
    assert np.isnan(row["heading_error_deg"])
    assert row["path_length_xy_m"] == pytest.approx(2.0)
    assert row["path_length_3d_m"] == pytest.approx(2.0)
    assert row["forward_displacement_m"] == pytest.approx(2.0)
    assert row["lateral_displacement_m"] == pytest.approx(0.0)
    assert row["xy_bounding_box_area_m2"] == pytest.approx(0.0)
    assert row["turn_footprint_proxy_m2"] == pytest.approx(0.1)
    assert row["entry_clearance_required_x_plus_m"] == pytest.approx(2.0)
    assert row["entry_clearance_required_x_minus_m"] == pytest.approx(0.0)
    assert row["floor_margin_min_m"] == pytest.approx(0.6)
    assert row["ceiling_margin_min_m"] == pytest.approx(2.5)
    assert row["min_true_margin_m"] == pytest.approx(0.6)
    assert row["speed_min_m_s"] == pytest.approx(6.0)
    assert row["terminal_speed_m_s"] == pytest.approx(6.0)
    assert row["energy_residual_m"] == pytest.approx(0.0)
    assert row["wind_query_region"] == "unknown"


def test_turning_toy_trajectory_metrics() -> None:
    points = np.array(
        [
            [2.0, 1.0, 1.0],
            [2.5, 1.5, 1.0],
            [3.0, 2.0, 1.0],
        ]
    )
    yaw = np.deg2rad(np.array([0.0, 15.0, 30.0]))
    row = _descriptor_row(x_ref=_state_from_points(points, yaw), target_heading_deg=30.0)

    assert row["heading_change_deg"] == pytest.approx(30.0)
    assert row["heading_error_deg"] == pytest.approx(0.0)
    assert row["path_length_xy_m"] == pytest.approx(2.0 * np.sqrt(0.5))
    assert row["path_length_3d_m"] == pytest.approx(2.0 * np.sqrt(0.5))
    assert row["forward_displacement_m"] == pytest.approx(1.0)
    assert row["lateral_displacement_m"] == pytest.approx(1.0)
    assert row["xy_bounding_box_area_m2"] == pytest.approx(1.0)
    assert row["turn_footprint_proxy_m2"] == pytest.approx(1.0)


def test_saturation_uses_effective_target_and_latency_fields_are_copied() -> None:
    time_s, x_ref, requested, effective, applied, delta = _straight_arrays()
    requested[0] = np.array([1.0, 1.0, 1.0])
    row = _descriptor_row(
        time_s=time_s,
        x_ref=x_ref,
        u_norm_requested=requested,
        u_norm_effective_target=effective,
        u_norm_applied=applied,
        delta_cmd_rad=delta,
    )
    assert row["saturation_fraction"] == 0.0

    applied[1, 0] = 0.5
    row = _descriptor_row(
        time_s=time_s,
        x_ref=x_ref,
        u_norm_requested=requested,
        u_norm_effective_target=effective,
        u_norm_applied=applied,
        delta_cmd_rad=delta,
    )
    assert row["saturation_fraction"] == pytest.approx(1.0 / 3.0)
    assert row["latency_case"] == "nominal"
    assert row["latency_acceptance_scope"] == "command_path_nominal_no_feedback_controller"
    assert row["latency_pass_label"] == "nominal_pass"
    assert row["state_feedback_delay_applied"] is False


def _descriptor_row(**overrides: object) -> dict[str, object]:
    time_s, x_ref, requested, effective, applied, delta = _straight_arrays()
    start_row = _start_row(
        family=str(overrides.pop("family", "mild_bank")),
        target_heading_deg=overrides.pop("target_heading_deg", 30.0),
        w_wing_mean_m_s=overrides.pop("w_wing_mean_m_s", 0.08),
    )
    candidate_row = _candidate_row(start_row)
    latency_fields = _latency_fields()
    replay_seed = overrides.pop("replay_seed", 17)
    sim_real_match_key = overrides.pop("sim_real_match_key", _match_key(replay_seed=replay_seed))
    return dense_trial_descriptor_row(
        start_row=start_row,
        candidate_row=candidate_row,
        time_s=overrides.pop("time_s", time_s),
        x_ref=overrides.pop("x_ref", x_ref),
        u_norm_requested=overrides.pop("u_norm_requested", requested),
        u_norm_effective_target=overrides.pop("u_norm_effective_target", effective),
        u_norm_applied=overrides.pop("u_norm_applied", applied),
        delta_cmd_rad=overrides.pop("delta_cmd_rad", delta),
        latency_fields=latency_fields,
        failure_label=str(overrides.pop("failure_label", "success")),
        governor_rejection_cause=str(overrides.pop("governor_rejection_cause", "none")),
        robustness_label=str(overrides.pop("robustness_label", "not_evaluated")),
        sim_real_match_key=str(sim_real_match_key),
        descriptor_status=str(overrides.pop("descriptor_status", "synthetic_descriptor_only")),
        replay_seed=replay_seed,
        lift_exposure_m_s=overrides.pop("lift_exposure_m_s", None),
        sim_real_transfer_result=str(overrides.pop("sim_real_transfer_result", "not_evaluated")),
        config=overrides.pop("config", DenseTrialDescriptorConfig()),
    )


def _match_key(**overrides: object) -> str:
    fields = {
        "layout_branch_id": "single_fan_branch",
        "fan_layout": "single_fan",
        "fan_config_id": "single_fan_nominal_updraft",
        "test_environment_mode": "W1_single_fan",
        "paired_environment_mode": "W0_single_fan_branch",
        "candidate_id": "candidate_001",
        "sample_id": "sample_001",
        "paired_sample_key": "pair_001",
        "seed": 101,
        "replay_seed": 17,
        "latency_case": "nominal",
    }
    fields.update(overrides)
    return dense_trial_match_key(**fields)


def _straight_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time_s = np.array([0.0, 1.0, 2.0])
    x_ref = _state_from_points(
        np.array(
            [
                [2.0, 1.0, 1.0],
                [3.0, 1.0, 1.0],
                [4.0, 1.0, 1.0],
            ]
        ),
        np.zeros(3),
    )
    commands = np.zeros((3, 3))
    return time_s, x_ref, commands.copy(), commands.copy(), commands.copy(), commands.copy()


def _state_from_points(points: np.ndarray, yaw_rad: np.ndarray) -> np.ndarray:
    state = np.zeros((points.shape[0], 15))
    state[:, 0:3] = points
    state[:, 5] = yaw_rad
    state[:, 6] = 6.0
    return state


def _start_row(
    *,
    family: str,
    target_heading_deg: object,
    w_wing_mean_m_s: float,
) -> dict[str, object]:
    row: dict[str, object] = {
        "sample_id": "sample_001",
        "paired_sample_key": "pair_001",
        "seed": 101,
        "sampling_round": 0,
        "fan_layout": "single_fan",
        "layout_branch_id": "single_fan_branch",
        "fan_config_id": "single_fan_nominal_updraft",
        "updraft_model_id": "single_gaussian_var",
        "start_class": "favourable",
        "family": family,
        "target_heading_deg": target_heading_deg,
        "direction_sign": 1,
        "environment_role": "dry_air_capable",
        "x_w_m": 2.0,
        "y_w_m": 1.0,
        "z_w_m": 1.0,
        "speed_m_s": 6.0,
        "phi_rad": 0.0,
        "theta_rad": 0.0,
        "psi_rad": 0.0,
        "u_m_s": 6.0,
        "v_m_s": 0.0,
        "w_m_s": 0.0,
        "p_rad_s": 0.0,
        "q_rad_s": 0.0,
        "r_rad_s": 0.0,
        "updraft_center_x_m": 3.0,
        "updraft_center_y_m": 2.0,
        "updraft_relative_radius_m": 1.1,
        "updraft_relative_azimuth_rad": 0.25,
        "updraft_relative_height_m": 0.67,
        "updraft_sector_label": "core",
    }
    row.update(_wing_fields(w_wing_mean_m_s))
    return row


def _candidate_row(start_row: dict[str, object]) -> dict[str, object]:
    row = dict(start_row)
    row.update(
        {
            "candidate_id": "candidate_001",
            "test_environment_mode": "W1_single_fan",
            "paired_environment_mode": "W0_single_fan_branch",
            "validity_gate_role": "first_validity_gate",
            "acceptance_interpretation": "first_validity_gate",
        }
    )
    return row


def _wing_fields(w_wing_mean_m_s: float) -> dict[str, object]:
    return {
        "wind_descriptor_status": "wind_model_evaluated",
        "wind_descriptor_environment_mode": "W1_single_fan",
        "wind_descriptor_model_id": "single_gaussian_var",
        "wind_descriptor_model_source": "synthetic_test_model",
        "w_cg_m_s": w_wing_mean_m_s,
        "w_wing_mean_m_s": w_wing_mean_m_s,
        "w_left_m_s": w_wing_mean_m_s + 0.01,
        "w_right_m_s": w_wing_mean_m_s - 0.01,
        "delta_w_lr_m_s": 0.02,
        "w_panel_max_m_s": w_wing_mean_m_s + 0.02,
        "w_panel_min_m_s": w_wing_mean_m_s - 0.02,
        "spanwise_w_gradient_m_s_per_m": 0.01,
        "local_updraft_uncertainty_m_s": np.nan,
        "local_updraft_uncertainty_status": "not_available_in_model",
        "wing_panel_sample_count": 11,
    }


def _latency_fields() -> dict[str, object]:
    return {
        "latency_case": "nominal",
        "latency_acceptance_scope": "command_path_nominal_no_feedback_controller",
        "latency_pass_label": "nominal_pass",
        "state_feedback_delay_s": 0.0,
        "command_onset_delay_s": 0.02,
        "command_transport_delay_s": 0.01,
        "actuator_tau_s": "0.060000000;0.060000000;0.060000000",
        "actuator_t50_s": 0.04159,
        "actuator_t90_s": 0.13816,
        "latency_jitter_s": 0.0,
        "timing_model_version": "test_model_v1",
        "state_feedback_delay_applied": False,
    }
