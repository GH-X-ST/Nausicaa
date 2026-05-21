from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import agile_trajectory_optimisation as agile
import run_agile_trajectory_optimisation as runner


def test_optimiser_result_is_stored_as_reference_trajectory(monkeypatch: pytest.MonkeyPatch) -> None:
    request = _request()

    def fake_minimize(objective, x0, **kwargs):  # noqa: ANN001
        objective(x0)
        return SimpleNamespace(x=np.asarray(x0), success=True, message="ok", nit=1)

    monkeypatch.setattr(agile, "minimize", fake_minimize)
    monkeypatch.setattr(agile, "rk4_step", _heading_step)

    result = agile.optimise_agile_trajectory(
        request,
        aircraft=object(),
        wind_model=None,
        wind_mode="none",
    )

    assert result.trajectory_generation_method == agile.AGILE_TRAJECTORY_GENERATION_METHOD
    assert result.time_s.ndim == 1
    assert result.x_ref.shape[0] == result.time_s.size
    assert result.u_norm_ref.shape == (result.time_s.size, 3)
    assert result.delta_cmd_ref_rad.shape == (result.time_s.size, 3)
    row = agile.result_index_row(result)
    assert row["trajectory_id"] == request.trajectory_id
    assert row["terminal_heading_target_deg"] == request.target_heading_deg


def test_command_template_reference_is_ablation_not_d2_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agile, "rk4_step", _heading_step)

    result = agile.command_template_reference(
        _request(),
        aircraft=object(),
        wind_model=None,
        wind_mode="none",
    )

    assert result.trajectory_generation_method == agile.COMMAND_TEMPLATE_ABLATION_METHOD
    assert result.optimizer_status == "not_optimised_command_template_ablation"
    assert result.open_loop_success_flag is False


def test_unsafe_initial_state_is_rejected() -> None:
    request = _request()
    bad = agile.AgileTrajectoryRequest(
        **{
            **request.__dict__,
            "x0": np.array([0.1, 1.0, 1.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        }
    )

    with pytest.raises(ValueError, match="true safe bounds"):
        agile.optimise_agile_trajectory(
            bad,
            aircraft=object(),
            wind_model=None,
            wind_mode="none",
        )


def test_runtime_gate_selects_largest_feasible_scale() -> None:
    profile = pd.DataFrame(
        {
            "seconds_per_reference": [5.0, 5.0],
            "optimizer_success": [True, True],
            "tvlqr_status": ["tvlqr_synthesised", "local_feedback_approx"],
            "closed_loop_success_flag": [True, False],
        }
    )

    gate = runner.profile_runtime_gate(
        profile,
        budget_hours=5.0,
        buffer_fraction=0.20,
        elapsed_s=0.0,
    )

    assert gate["runtime_budget_status"] == "pass"
    assert gate["selected_reference_library_size"] == 512
    assert gate["selected_d2_scale_class"] == "primary"
    assert {
        (item["reference_library_size"], item["d2_scale_class"])
        for item in gate["projections"]
    } == {(512, "primary"), (256, "primary"), (128, "primary"), (128, "fast")}


def test_runtime_gate_blocks_when_even_fast_fallback_exceeds_budget() -> None:
    profile = pd.DataFrame(
        {
            "seconds_per_reference": [200.0, 220.0],
            "optimizer_success": [False, False],
            "tvlqr_status": ["local_feedback_approx", "local_feedback_approx"],
            "closed_loop_success_flag": [False, False],
        }
    )

    gate = runner.profile_runtime_gate(
        profile,
        budget_hours=1.0,
        buffer_fraction=0.20,
        elapsed_s=0.0,
    )

    assert gate["runtime_budget_status"] == "D2_execution_blocked_by_runtime_budget"
    assert gate["selected_reference_library_size"] == 0


def test_profile_request_selector_covers_all_agile_families() -> None:
    trials = pd.DataFrame(
        [
            _trial_row("bank_yaw_energy_retaining", 15.0, index=0),
            _trial_row("canyon_steep_bank", 15.0, index=1),
            _trial_row("wingover_lite", 45.0, index=2),
        ]
    )

    requests = runner.build_profile_requests(
        trials,
        profile_reference_count=16,
        random_seed=1,
    )

    assert {"bank_yaw_energy_retaining", "canyon_steep_bank", "wingover_lite"}.issubset(
        {request.family for request in requests}
    )


def _request() -> agile.AgileTrajectoryRequest:
    x0 = np.zeros(15, dtype=float)
    x0[0:3] = (2.0, 2.0, 1.4)
    x0[6] = 6.0
    return agile.AgileTrajectoryRequest(
        trajectory_id="test_ref",
        family="canyon_steep_bank",
        target_heading_deg=15.0,
        direction_sign=1,
        x0=x0,
        test_environment_mode="W0_single_fan_branch",
        dt_s=0.02,
        horizon_s=0.08,
        command_knot_count=3,
        max_iterations=1,
    )


def _trial_row(family: str, target: float, *, index: int) -> dict[str, object]:
    return {
        "layout_branch_id": "single_fan_branch",
        "fan_layout": "single_fan",
        "test_environment_mode": "W0_single_fan_branch",
        "updraft_model_id": "",
        "sample_id": f"s{index}",
        "family": family,
        "target_heading_deg": target,
        "direction_sign": 1,
        "x0_w_m": 2.0,
        "y0_w_m": 2.0,
        "z0_w_m": 1.4,
        "phi0_rad": 0.0,
        "theta0_rad": 0.0,
        "psi0_rad": 0.0,
        "u0_m_s": 6.0,
        "v0_m_s": 0.0,
        "w0_m_s": 0.0,
        "p0_rad_s": 0.0,
        "q0_rad_s": 0.0,
        "r0_rad_s": 0.0,
    }


def _heading_step(
    x: np.ndarray,
    delta_cmd_rad: np.ndarray,
    dt_s: float,
    aircraft: object,
    wind_model: object,
    wind_mode: str,
    actuator_tau_s: tuple[float, float, float],
) -> np.ndarray:
    state = np.asarray(x, dtype=float).copy()
    state[0] += 0.05
    state[5] += float(delta_cmd_rad[0]) * float(dt_s) * 2.0
    state[12:15] = np.asarray(delta_cmd_rad, dtype=float)
    return state
