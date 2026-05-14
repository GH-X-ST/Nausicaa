from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from flight_dynamics import adapt_glider
from glider import build_nausicaa_glider
from latency import CommandToSurfaceLayer
from linearisation import STATE_INDEX
from linearisation import linearise_trim
from primitive import build_primitive_context
from run_agile_trajectory_optimisation import _phase2_gate_summary
from scenarios import arena_feasible_entry_state
from trajectory_primitive import trajectory_error
from turn_trajectory_optimisation import (
    OptimisedTurnResult,
    TurnOptimisationConfig,
    TurnTarget,
    build_turn_trajectory_primitive,
    load_selected_turn_primitive,
    primitive_open_loop_copy,
    save_turn_result,
)


def _synthetic_result() -> tuple[OptimisedTurnResult, object, object]:
    aircraft = adapt_glider(build_nausicaa_glider())
    linear_model = linearise_trim(aircraft=aircraft)
    x0 = arena_feasible_entry_state(linear_model.x_trim, altitude_m=2.7)
    times = np.array([0.0, 0.02, 0.04], dtype=float)
    x_ref = np.vstack([x0, x0, x0])
    x_ref[:, 0] += np.array([0.0, 0.13, 0.26])
    u_ff = np.vstack([linear_model.u_trim, linear_model.u_trim, linear_model.u_trim])
    config = TurnOptimisationConfig(n_intervals=2, t_min_s=0.04, t_max_s=0.04)
    result = OptimisedTurnResult(
        success=True,
        failure_reason="",
        feasibility_label="accepted_low_alpha",
        target=TurnTarget(target_heading_deg=0.0, direction="left"),
        config=config,
        times_s=times,
        x_ref=x_ref,
        u_ff=u_ff,
        objective_value=0.0,
        metrics={
            "target_heading_deg": 0.0,
            "dynamics_defect_max": 0.0,
            "slack_max": 0.0,
        },
        solver_stats={"initial_guess_name": "unit"},
        nu_ff=np.zeros((3, 3), dtype=float),
    )
    context = build_primitive_context(linear_model.x_trim, linear_model.u_trim)
    return result, aircraft, context


def test_turn_result_converts_to_finite_trajectory_primitive(tmp_path: Path) -> None:
    result, aircraft, context = _synthetic_result()

    primitive = build_turn_trajectory_primitive(
        result=result,
        context=context,
        aircraft=aircraft,
        wind_model=None,
        wind_mode="none",
        command_layer=CommandToSurfaceLayer(),
    )
    paths = save_turn_result(result, tmp_path, "unit_turn", primitive=primitive)
    loaded = load_selected_turn_primitive(paths["trajectory"])

    assert primitive.times_s.shape == (3,)
    assert primitive.x_ref.shape == (3, 15)
    assert primitive.u_ff.shape == (3, 3)
    assert primitive.a_mats is not None and primitive.a_mats.shape == (3, 15, 15)
    assert primitive.b_mats is not None and primitive.b_mats.shape == (3, 15, 3)
    assert primitive.k_lqr.shape == (3, 3, 15)
    assert primitive.s_mats is not None and primitive.s_mats.shape == (3, 15, 15)
    assert np.all(np.isfinite(primitive.k_lqr))
    assert loaded.k_lqr.shape == primitive.k_lqr.shape


def test_open_loop_copy_preserves_reference_and_removes_feedback() -> None:
    result, aircraft, context = _synthetic_result()
    primitive = build_turn_trajectory_primitive(
        result=result,
        context=context,
        aircraft=aircraft,
        wind_model=None,
        wind_mode="none",
        command_layer=CommandToSurfaceLayer(),
    )

    open_loop = primitive_open_loop_copy(primitive)

    np.testing.assert_allclose(open_loop.times_s, primitive.times_s)
    np.testing.assert_allclose(open_loop.u_ff, primitive.u_ff)
    np.testing.assert_allclose(open_loop.k_lqr, np.zeros_like(primitive.k_lqr))


def test_trajectory_error_wraps_euler_angles() -> None:
    x_ref = np.zeros(15, dtype=float)
    x = np.zeros(15, dtype=float)
    x_ref[STATE_INDEX["psi"]] = np.pi - 0.01
    x[STATE_INDEX["psi"]] = -np.pi + 0.01

    error = trajectory_error(x, x_ref)

    assert abs(error[STATE_INDEX["psi"]] - 0.02) < 1e-12


def test_phase2_summary_does_not_promote_latency_or_recovery_failure() -> None:
    best_30 = {
        "success": True,
        "feasibility_label": "accepted_low_alpha",
        "slack_max": 0.0,
        "heading_gate_passed": True,
        "inside_true_safety_volume": True,
        "exit_recoverable_gate": True,
    }
    replay_rows = [
        {"replay_kind": "open_loop_no_latency", "success": True},
        {"replay_kind": "closed_loop_no_latency", "success": True},
        {
            "replay_kind": "closed_loop_nominal_latency",
            "success": False,
            "termination_reason": "angle of attack exceeded bound",
            "failure_class": "model",
            "feasibility_label": "model_limited_high_alpha",
        },
        {
            "replay_kind": "terminal_altitude_sensitivity",
            "terminal_altitude_min_m": 0.75,
            "success": False,
            "termination_reason": "angle of attack exceeded bound",
        },
        {
            "replay_kind": "terminal_altitude_sensitivity",
            "terminal_altitude_min_m": 1.0,
            "success": False,
            "termination_reason": "angle of attack exceeded bound",
        },
        {
            "replay_kind": "terminal_altitude_sensitivity",
            "terminal_altitude_min_m": 1.2,
            "success": False,
            "termination_reason": "angle of attack exceeded bound",
        },
    ]

    summary = _phase2_gate_summary(best_30, replay_rows)

    assert summary["phase2_status"] == "boundary_only"
    assert summary["active_failure_class"] == "latency_limited_high_alpha"
    assert "terminal_recovery_limited" in summary["all_failure_classes"]


def test_phase2_report_paths_are_not_ignored() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            "git",
            "check-ignore",
            "docs/control/phase2_tvlqr_ocp30_report.md",
            "docs/control/phase2_tvlqr_ocp30_boundary.md",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1, result.stdout + result.stderr
