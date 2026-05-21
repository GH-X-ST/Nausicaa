from __future__ import annotations

import numpy as np
import pytest

import agile_tvlqr as tvlqr
from agile_trajectory_optimisation import (
    AGILE_TRAJECTORY_GENERATION_METHOD,
    AgileTrajectoryRequest,
    AgileTrajectoryResult,
)


def test_tvlqr_success_uses_nominal_latency_closed_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    trajectory = _trajectory()
    gains = np.zeros((trajectory.time_s.size, 3, 15), dtype=float)

    monkeypatch.setattr(tvlqr, "_finite_difference_tvlqr", lambda *args, **kwargs: gains)
    monkeypatch.setattr(tvlqr, "rk4_step", _tracking_step)

    result = tvlqr.synthesize_tvlqr_for_trajectory(
        trajectory,
        aircraft=object(),
        wind_model=None,
        wind_mode="none",
        dt_s=0.02,
    )

    assert result.tvlqr_status == tvlqr.TVLQR_STATUS_SUCCESS
    assert result.agile_evidence_class == tvlqr.EVIDENCE_TVLQR_NOMINAL
    assert result.latency_case == "nominal"
    assert result.latency_pass_label in {"nominal_pass", "nominal_fail"}
    assert result.k_feedback.shape == (trajectory.time_s.size, 3, 15)


def test_failed_riccati_is_labelled_local_feedback_not_full_tvlqr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = _trajectory()

    def fail(*args, **kwargs):  # noqa: ANN001
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(tvlqr, "_finite_difference_tvlqr", fail)
    monkeypatch.setattr(tvlqr, "rk4_step", _tracking_step)

    result = tvlqr.synthesize_tvlqr_for_trajectory(
        trajectory,
        aircraft=object(),
        wind_model=None,
        wind_mode="none",
        dt_s=0.02,
    )

    assert result.tvlqr_status == tvlqr.TVLQR_STATUS_FALLBACK
    assert result.agile_evidence_class == tvlqr.EVIDENCE_LOCAL_FEEDBACK_NOMINAL


def test_command_template_ablation_is_not_tvlqr_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    trajectory = _trajectory(method="command_template_initial_guess_or_ablation")
    monkeypatch.setattr(tvlqr, "rk4_step", _tracking_step)

    result = tvlqr.synthesize_tvlqr_for_trajectory(
        trajectory,
        aircraft=object(),
        wind_model=None,
        wind_mode="none",
        dt_s=0.02,
    )

    assert result.tvlqr_status == tvlqr.TVLQR_STATUS_NOT_APPLICABLE
    assert result.agile_evidence_class == tvlqr.EVIDENCE_COMMAND_TEMPLATE_BASELINE


def _trajectory(method: str = AGILE_TRAJECTORY_GENERATION_METHOD) -> AgileTrajectoryResult:
    time_s = np.array([0.0, 0.02, 0.04], dtype=float)
    x_ref = np.zeros((3, 15), dtype=float)
    x_ref[:, 0] = [2.0, 2.02, 2.04]
    x_ref[:, 1] = 2.0
    x_ref[:, 2] = 1.4
    x_ref[:, 5] = np.deg2rad([0.0, 7.5, 15.0])
    x_ref[:, 6] = 6.0
    u_ref = np.zeros((3, 3), dtype=float)
    delta = np.zeros((3, 3), dtype=float)
    request = AgileTrajectoryRequest(
        trajectory_id="traj",
        family="canyon_steep_bank",
        target_heading_deg=15.0,
        direction_sign=1,
        x0=x_ref[0],
        test_environment_mode="W0_single_fan_branch",
        dt_s=0.02,
        horizon_s=0.04,
    )
    return AgileTrajectoryResult(
        request=request,
        trajectory_id="traj",
        trajectory_generation_method=method,
        optimizer_status="success",
        optimizer_message="ok",
        optimizer_success=True,
        optimizer_iterations=1,
        optimizer_wall_time_s=0.01,
        objective_cost=0.0,
        heading_cost=0.0,
        speed_loss_cost=0.0,
        height_loss_cost=0.0,
        saturation_cost=0.0,
        safety_status="pass",
        time_s=time_s,
        x_ref=x_ref,
        u_norm_ref=u_ref,
        delta_cmd_ref_rad=delta,
        achieved_heading_deg=15.0,
        terminal_heading_error_deg=0.0,
        terminal_speed_m_s=6.0,
        height_loss_m=0.0,
        minimum_safety_margin_m=1.0,
        actuator_saturation_fraction=0.0,
        open_loop_success_flag=True,
        failure_label="success",
    )


def _tracking_step(
    x: np.ndarray,
    delta_cmd_rad: np.ndarray,
    dt_s: float,
    aircraft: object,
    wind_model: object,
    wind_mode: str,
    actuator_tau_s: tuple[float, float, float],
) -> np.ndarray:
    state = np.asarray(x, dtype=float).copy()
    state[0] += 0.02
    state[5] += np.deg2rad(7.5)
    state[12:15] = np.asarray(delta_cmd_rad, dtype=float)
    return state
