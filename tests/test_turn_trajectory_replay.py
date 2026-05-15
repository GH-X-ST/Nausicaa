from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

import casadi as ca
import numpy as np

from flight_dynamics import adapt_glider
from glider import build_nausicaa_glider
from latency import CommandToSurfaceLayer
from linearisation import STATE_INDEX
from linearisation import linearise_trim
from primitive import build_primitive_context
from run_agile_trajectory_optimisation import _phase2_gate_summary
from run_agile_trajectory_optimisation import _phase2_candidate_config_variants
from run_agile_trajectory_optimisation import _phase2_selection_score
from run_agile_trajectory_optimisation import _phase2_tvlqr_variant_configs
from run_agile_trajectory_optimisation import _latency_ablation_specs
from run_agile_trajectory_optimisation import _strict_replay_forbidden_reason
from scenarios import arena_feasible_entry_state
from trajectory_primitive import trajectory_error
from turn_trajectory_optimisation import (
    OptimisedTurnResult,
    TurnOptimisationConfig,
    TurnTarget,
    _delayed_alpha_margin_cost,
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
        {"replay_kind": "open_loop_nominal_latency", "success": True},
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


def test_strict_phase2_gate_refuses_successful_high_alpha_label() -> None:
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
        {"replay_kind": "open_loop_nominal_latency", "success": True},
        {
            "replay_kind": "closed_loop_nominal_latency",
            "success": True,
            "actual_heading_change_deg": -25.0,
            "exit_recoverable": True,
            "feasibility_label": "model_limited_high_alpha",
            "termination_reason": "",
        },
        {
            "replay_kind": "terminal_altitude_sensitivity",
            "terminal_altitude_min_m": 0.75,
            "success": True,
        },
        {
            "replay_kind": "terminal_altitude_sensitivity",
            "terminal_altitude_min_m": 1.0,
            "success": True,
        },
        {
            "replay_kind": "terminal_altitude_sensitivity",
            "terminal_altitude_min_m": 1.2,
            "success": True,
        },
    ]

    summary = _phase2_gate_summary(best_30, replay_rows)

    assert summary["phase2_status"] == "boundary_only"
    assert summary["closed_loop_nominal_latency"] is False
    assert _strict_replay_forbidden_reason(replay_rows[3]) == "high_alpha_exposure"


def test_terminal_sensitivity_refuses_unrecoverable_success_row() -> None:
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
        {"replay_kind": "open_loop_nominal_latency", "success": True},
        {"replay_kind": "closed_loop_nominal_latency", "success": True},
        {
            "replay_kind": "terminal_altitude_sensitivity",
            "terminal_altitude_min_m": 0.75,
            "success": True,
            "exit_recoverable": False,
        },
        {
            "replay_kind": "terminal_altitude_sensitivity",
            "terminal_altitude_min_m": 1.0,
            "success": True,
        },
        {
            "replay_kind": "terminal_altitude_sensitivity",
            "terminal_altitude_min_m": 1.2,
            "success": True,
        },
    ]

    summary = _phase2_gate_summary(best_30, replay_rows)

    assert summary["phase2_status"] == "boundary_only"
    assert summary["terminal_altitude_sensitivity"] is False
    assert "terminal_recovery_limited" in summary["all_failure_classes"]


def test_phase2_summary_filters_to_selected_candidate_variant() -> None:
    best_30 = {
        "success": True,
        "candidate_variant": "latency090",
        "feasibility_label": "accepted_low_alpha",
        "slack_max": 0.0,
        "heading_gate_passed": True,
        "inside_true_safety_volume": True,
        "exit_recoverable_gate": True,
    }
    replay_rows = [
        {"candidate_variant": "baseline", "replay_kind": "open_loop_no_latency", "success": True},
        {"candidate_variant": "baseline", "replay_kind": "closed_loop_no_latency", "success": True},
        {
            "candidate_variant": "baseline",
            "replay_kind": "closed_loop_nominal_latency",
            "success": False,
            "termination_reason": "angle of attack exceeded bound",
        },
        {
            "candidate_variant": "latency090",
            "replay_kind": "open_loop_no_latency",
            "success": True,
        },
        {
            "candidate_variant": "latency090",
            "replay_kind": "closed_loop_no_latency",
            "success": True,
        },
        {
            "candidate_variant": "latency090",
            "replay_kind": "closed_loop_nominal_latency",
            "success": True,
        },
        {
            "candidate_variant": "latency090",
            "replay_kind": "open_loop_nominal_latency",
            "success": True,
        },
        {
            "candidate_variant": "latency090",
            "replay_kind": "terminal_altitude_sensitivity",
            "terminal_altitude_min_m": 0.75,
            "success": True,
        },
        {
            "candidate_variant": "latency090",
            "replay_kind": "terminal_altitude_sensitivity",
            "terminal_altitude_min_m": 1.0,
            "success": True,
        },
        {
            "candidate_variant": "latency090",
            "replay_kind": "terminal_altitude_sensitivity",
            "terminal_altitude_min_m": 1.2,
            "success": True,
        },
    ]

    summary = _phase2_gate_summary(best_30, replay_rows)

    assert summary["phase2_status"] == "promoted_phase2"
    assert summary["active_failure_class"] == ""


def test_phase2_summary_does_not_promote_under_turning_replay() -> None:
    best_30 = {
        "success": True,
        "candidate_variant": "recovery065",
        "feasibility_label": "accepted_low_alpha",
        "slack_max": 0.0,
        "heading_gate_passed": True,
        "inside_true_safety_volume": True,
        "exit_recoverable_gate": True,
    }
    replay_rows = [
        {
            "candidate_variant": "recovery065",
            "replay_kind": "open_loop_no_latency",
            "success": True,
            "actual_heading_change_deg": -26.0,
            "exit_recoverable": True,
        },
        {
            "candidate_variant": "recovery065",
            "replay_kind": "closed_loop_no_latency",
            "success": True,
            "actual_heading_change_deg": -21.0,
            "exit_recoverable": True,
        },
        {
            "candidate_variant": "recovery065",
            "replay_kind": "open_loop_nominal_latency",
            "success": True,
            "actual_heading_change_deg": -20.0,
            "exit_recoverable": True,
        },
        {
            "candidate_variant": "recovery065",
            "replay_kind": "closed_loop_nominal_latency",
            "success": True,
            "actual_heading_change_deg": -21.0,
            "exit_recoverable": True,
        },
    ]

    summary = _phase2_gate_summary(best_30, replay_rows)

    assert summary["phase2_status"] == "boundary_only"
    assert summary["closed_loop_no_latency"] is False
    assert summary["open_loop_nominal_latency"] is False


def test_phase2_selection_score_prefers_replay_gates_over_ocp_score() -> None:
    result, _, _ = _synthetic_result()
    baseline = replace(
        result,
        solver_stats={**result.solver_stats, "candidate_variant": "baseline"},
        metrics={
            **result.metrics,
            "directed_heading_change_deg": 30.0,
            "terminal_speed_m_s": 6.3,
            "saturation_fraction": 0.0,
            "slack_max": 0.0,
            "heading_gate_passed": True,
            "inside_true_safety_volume": True,
            "exit_recoverable_gate": True,
        },
    )
    robust = replace(
        baseline,
        solver_stats={**baseline.solver_stats, "candidate_variant": "latency090"},
        metrics={**baseline.metrics, "directed_heading_change_deg": 25.0},
    )
    failing_rows = [
        {"replay_kind": "open_loop_no_latency", "success": True},
        {"replay_kind": "closed_loop_no_latency", "success": True},
        {
            "replay_kind": "closed_loop_nominal_latency",
            "success": False,
            "max_alpha_deg": 29.0,
            "termination_reason": "angle of attack exceeded bound",
        },
    ]
    passing_rows = [
        {
            "replay_kind": "open_loop_no_latency",
            "success": True,
            "actual_heading_change_deg": -25.0,
            "exit_recoverable": True,
        },
        {
            "replay_kind": "closed_loop_no_latency",
            "success": True,
            "actual_heading_change_deg": -25.0,
            "exit_recoverable": True,
        },
        {
            "replay_kind": "open_loop_nominal_latency",
            "success": True,
            "actual_heading_change_deg": -25.0,
            "exit_recoverable": True,
            "max_alpha_deg": 5.5,
        },
        {
            "replay_kind": "closed_loop_nominal_latency",
            "success": True,
            "actual_heading_change_deg": -25.0,
            "exit_recoverable": True,
            "max_alpha_deg": 6.0,
        },
        {
            "replay_kind": "terminal_altitude_sensitivity",
            "terminal_altitude_min_m": 0.75,
            "success": True,
        },
        {
            "replay_kind": "terminal_altitude_sensitivity",
            "terminal_altitude_min_m": 1.0,
            "success": True,
        },
        {
            "replay_kind": "terminal_altitude_sensitivity",
            "terminal_altitude_min_m": 1.2,
            "success": True,
        },
    ]

    assert _phase2_selection_score(robust, passing_rows) > _phase2_selection_score(
        baseline,
        failing_rows,
    )


def test_phase2_robust_configs_tighten_terminal_recovery_without_command_cap() -> None:
    base = TurnOptimisationConfig()
    variants = dict(_phase2_candidate_config_variants(base))

    assert set(variants) == {
        "baseline",
        "recovery065",
        "recovery080",
        "latency075",
        "latency090",
        "latency105",
        "h070_terminal_buffer",
        "h080_terminal_buffer",
        "h095_recovery_buffer",
        "h110_conservative",
    }
    assert variants["baseline"] == base
    for name in (
        "recovery065",
        "recovery080",
        "latency075",
        "latency090",
        "latency105",
        "h070_terminal_buffer",
        "h080_terminal_buffer",
        "h095_recovery_buffer",
        "h110_conservative",
    ):
        variant = variants[name]
        assert variant.t_min_s >= base.t_min_s
        assert variant.t_max_s <= base.t_max_s
        assert variant.terminal_alpha_deg is not None
        assert variant.terminal_alpha_deg < base.max_alpha_deg
        assert variant.terminal_beta_deg is not None
        assert variant.terminal_beta_deg < base.max_beta_deg
        assert variant.terminal_rate_max_rad_s is not None
        assert variant.final_third_smoothness_weight > 0.0
        assert variant.late_command_reversal_weight > 0.0
        assert variant.delayed_alpha_weight > 0.0


def test_latency_ablation_schema_and_toggles_are_explicit() -> None:
    specs = {name: (open_loop, config) for name, open_loop, config in _latency_ablation_specs()}

    assert set(specs) == {
        "no_latency_no_feedback_delay",
        "actuator_onset_only",
        "state_feedback_delay_only",
        "actuator_state_delay_no_quantisation",
        "actuator_state_delay_quantisation_on",
        "open_loop_feedforward_nominal_latency",
        "closed_loop_tvlqr_nominal_latency",
        "final_recovery_feedback_disabled",
    }
    assert specs["open_loop_feedforward_nominal_latency"][0] is True
    assert specs["closed_loop_tvlqr_nominal_latency"][0] is False
    assert specs["actuator_onset_only"][1].use_onset_delay is True
    assert specs["actuator_onset_only"][1].use_state_feedback_delay is False
    assert specs["state_feedback_delay_only"][1].use_onset_delay is False
    assert specs["state_feedback_delay_only"][1].use_state_feedback_delay is True
    assert specs["actuator_state_delay_quantisation_on"][1].quantise is True


def test_tvlqr_variant_configs_are_finite_and_named() -> None:
    variants = dict(_phase2_tvlqr_variant_configs())

    assert set(variants) == {
        "baseline",
        "r110",
        "yaw_light_r110",
        "recovery_heavy_r90",
        "late_feedback_half_r110",
        "k_smooth3_r110",
    }
    for config in variants.values():
        assert np.all(np.isfinite(config.q_diag))
        assert np.all(np.isfinite(config.r_diag))
        assert config.qf_diag is not None
        assert np.all(np.isfinite(config.qf_diag))


def test_phase2_default_config_disables_v2_objective_terms() -> None:
    config = TurnOptimisationConfig()

    assert config.terminal_surface_weight == 0.0
    assert config.final_third_smoothness_weight == 0.0
    assert config.late_command_reversal_weight == 0.0
    assert config.delayed_alpha_weight == 0.0
    assert config.delayed_alpha_margin_deg is None


def test_delayed_alpha_margin_cost_is_finite() -> None:
    x = ca.MX.sym("x", 15)
    u_cmd = ca.MX.sym("u_cmd", 3)
    x_dot = ca.MX.zeros(15, 1)
    x_dot[STATE_INDEX["u"]] = -0.1 * u_cmd[1]
    x_dot[STATE_INDEX["w"]] = 0.2 * u_cmd[1]
    dyn_fun = ca.Function("unit_dyn", [x, u_cmd], [x_dot])
    cost_fun = ca.Function(
        "delayed_alpha_cost",
        [x, u_cmd],
        [
            _delayed_alpha_margin_cost(
                dyn_fun,
                x,
                u_cmd,
                0.05,
                TurnOptimisationConfig(
                    delayed_alpha_weight=1.0,
                    delayed_alpha_margin_deg=5.0,
                ),
            )
        ],
    )
    x_value = np.zeros(15, dtype=float)
    x_value[STATE_INDEX["u"]] = 6.0

    value = float(cost_fun(x_value, np.array([0.0, -0.5, 0.0])).full().item())

    assert np.isfinite(value)
    assert value >= 0.0


def test_phase2_report_paths_are_not_ignored() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            "git",
            "check-ignore",
            "docs/control/phase2_latency_recovery_ocp30_report.md",
            "docs/control/phase2_latency_recovery_ocp30_boundary.md",
            "docs/control/phase2_overnight_latency_recovery_report.md",
            "docs/control/phase2_overnight_latency_recovery_boundary.md",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1, result.stdout + result.stderr
