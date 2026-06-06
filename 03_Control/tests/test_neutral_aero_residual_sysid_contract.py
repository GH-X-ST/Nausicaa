from __future__ import annotations

import json
import math
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
INNER_LOOP = ROOT / "03_Control" / "02_Inner_Loop"
if str(INNER_LOOP) not in sys.path:
    sys.path.insert(0, str(INNER_LOOP))

import run_fit_neutral_aero_residual_calibration as sysid  # noqa: E402
import run_neutral_aero_residual_local_pareto_audit as local_audit  # noqa: E402
from A_model_parameters import mass_properties_estimate  # noqa: E402
from A_model_parameters import neutral_dry_air_calibration as active_calibration  # noqa: E402


def test_default_neutral_sysid_is_longitudinal_primary_with_lateral_diagnostic() -> None:
    assert sysid.DEFAULT_FIT_WORKFLOW == "cm_regime_staged"
    assert sysid.DEFAULT_FIT_ATTACHED_LATERAL_COUPLING is False
    assert sysid.DEFAULT_FIT_TRANSITION_LATERAL_COUPLING is False
    assert sysid.DEFAULT_FIT_LATERAL_SURFACES is False
    assert sysid.DEFAULT_FIT_POST_STALL_SURFACES is False
    assert sysid.DEFAULT_FIT_SECONDARY_LATERAL_DIAGNOSTIC is True
    assert sysid.DEFAULT_ALIGNMENT_WINDOW_S == pytest.approx(0.040)
    assert sysid.DEFAULT_SENSITIVITY_ALIGNMENT_WINDOWS_S == ()
    assert sysid.prep.DEFAULT_ALIGNMENT_WINDOW_S == pytest.approx(0.040)
    assert sysid.replay_fit.DEFAULT_ALIGNMENT_WINDOW_S == pytest.approx(0.040)
    assert sysid.DEFAULT_JOINT_PARETO_AUDIT is True
    assert sysid.DEFAULT_JOINT_PARETO_AUDIT_ALIGNMENT_WINDOW_S == pytest.approx(0.040)
    assert sysid.DEFAULT_ALIGNED_U_MIN_M_S == pytest.approx(3.0)
    assert sysid.DEFAULT_ALIGNED_U_MAX_M_S == pytest.approx(8.0)
    parser = sysid.build_arg_parser()
    args = parser.parse_args(["--fit-workflow", "compact_joint_sweep", "--workers", "8"])
    assert args.fit_workflow == "compact_joint_sweep"
    assert args.workers == 8


def test_alignment_sensitivity_is_opt_in_after_40ms_becomes_primary_default() -> None:
    parser = sysid.build_arg_parser()
    args = parser.parse_args([])
    requested = (
        ()
        if args.no_sensitivity_alignment
        else tuple(args.sensitivity_alignment_window_s or sysid.DEFAULT_SENSITIVITY_ALIGNMENT_WINDOWS_S)
    )
    assert requested == ()
    assert sysid.normalized_sensitivity_alignment_windows(0.040, requested) == ()
    assert sysid.normalized_sensitivity_alignment_windows(0.040, (0.040,)) == ()
    assert sysid.normalized_sensitivity_alignment_windows(0.040, (0.100, 0.040, 0.100)) == pytest.approx((0.100,))


def test_joint_pareto_audit_parser_defaults_to_40ms_diagnostic() -> None:
    parser = sysid.build_arg_parser()
    args = parser.parse_args([])
    assert args.joint_pareto_audit is True
    assert args.joint_pareto_audit_alignment_window_s == pytest.approx(0.040)
    assert args.joint_pareto_profile == "small"

    disabled = parser.parse_args(["--no-joint-pareto-audit"])
    assert disabled.joint_pareto_audit is False


def test_heavy_joint_pareto_profile_defaults_are_bounded_40ms_contract() -> None:
    parser = sysid.build_arg_parser()
    args = parser.parse_args(
        [
            "--joint-pareto-profile",
            "heavy",
            "--joint-pareto-top-longitudinal",
            "6",
            "--joint-pareto-top-lateral",
            "10",
            "--joint-pareto-max-lateral-order",
            "3",
            "--joint-pareto-top-triples",
            "80",
            "--joint-pareto-max-candidates",
            "900",
            "--joint-pareto-selected-limit",
            "8",
            "--workers",
            "8",
        ]
    )
    config = sysid.resolved_joint_pareto_profile_config(
        profile=args.joint_pareto_profile,
        top_longitudinal=args.joint_pareto_top_longitudinal,
        top_lateral=args.joint_pareto_top_lateral,
        max_lateral_order=args.joint_pareto_max_lateral_order,
        top_triples=args.joint_pareto_top_triples,
        max_candidates=args.joint_pareto_max_candidates,
        selected_limit=args.joint_pareto_selected_limit,
    )

    assert args.joint_pareto_audit_alignment_window_s == pytest.approx(0.040)
    assert args.workers == 8
    assert config["profile"] == "heavy"
    assert config["top_longitudinal"] == 6
    assert config["top_lateral"] == 10
    assert config["max_lateral_order"] == 3
    assert config["top_triples"] == 80
    assert config["max_candidates"] == 900
    assert config["selected_limit"] == 8
    assert config["scale_grid"] == pytest.approx(sysid.JOINT_PARETO_HEAVY_SCALE_GRID)
    assert config["reference_policy"] == "per_base"
    assert config["include_rejected_stage_candidates"] is True


def test_local_promising_is_not_a_full_calibration_runner_profile() -> None:
    parser = sysid.build_arg_parser()

    assert "local_promising" not in sysid.JOINT_PARETO_PROFILE_DEFAULTS
    with pytest.raises(SystemExit):
        parser.parse_args(["--joint-pareto-profile", "local_promising"])


def test_local_pareto_audit_script_is_audit_only_entrypoint() -> None:
    source = inspect.getsource(local_audit)
    forbidden_calls = (
        "residual_rows(",
        "fit_pitch_residual_coefficients(",
        "cm_regime_staged_refinement(",
        "lateral_ablation_diagnostic_rows(",
    )

    assert local_audit.DEFAULT_RUN_LABEL == "n30_joint_pareto_040_local_promising_v1"
    assert local_audit.LONGITUDINAL_SOURCE_IDS == (
        "proposal_stage_5_stage5_transition_blend",
        "proposal_stage_9_stage9_post_blend_post_stall_lift_dr",
        "active_baseline",
    )
    assert all(call not in source for call in forbidden_calls)


def test_replay_alignment_accepts_synchronized_40ms_handoff_window() -> None:
    def sample(t_s: float) -> dict[str, float]:
        return {
            "t_s": t_s,
            "x_w": t_s,
            "y_w": 0.0,
            "z_w": 1.0,
            "phi": 0.0,
            "theta": 0.0,
            "psi": 0.0,
            "u": 1.0,
            "v": 0.0,
            "w": 0.0,
            "p": 0.0,
            "q": 0.0,
            "r": 0.0,
            "delta_a": 0.0,
            "delta_e": 0.0,
            "delta_r": 0.0,
        }

    rows = [sample(0.0), sample(0.02), sample(0.04)]
    aligned = sysid.prep._aligned_state_from_sample_rows(rows, 0.040)
    assert aligned["status"] == "ok"
    assert aligned["alignment_elapsed_s"] == pytest.approx(0.040)

    too_short = sysid.prep._aligned_state_from_sample_rows(rows, 0.020)
    assert too_short["status"] == "alignment_window_too_short"


def test_pitch_moment_regime_weights_are_normalized_and_localized() -> None:
    assert sysid.pitch_moment_regime_weights_from_activation(0.0) == pytest.approx((1.0, 0.0, 0.0))
    assert sysid.pitch_moment_regime_weights_from_activation(1.0) == pytest.approx((0.0, 0.0, 1.0))
    attached, transition, post = sysid.pitch_moment_regime_weights_from_activation(0.5)
    assert attached + transition + post == pytest.approx(1.0)
    assert transition > attached
    assert transition > post


def test_launch_confidence_uses_only_lateral_launch_contamination() -> None:
    state = np.zeros(12, dtype=float)
    base_score = sysid.launch_quality_score_from_state(state)
    assert base_score == pytest.approx(0.0)

    longitudinally_messy = state.copy()
    longitudinally_messy[4] = math.radians(15.0)  # theta
    longitudinally_messy[6] = 7.5  # u
    longitudinally_messy[8] = 0.9  # w
    longitudinally_messy[10] = 1.2  # q
    assert sysid.launch_quality_score_from_state(longitudinally_messy) == pytest.approx(base_score)

    laterally_messy = state.copy()
    laterally_messy[3] = math.radians(20.0)  # phi
    laterally_messy[5] = math.radians(20.0)  # psi
    laterally_messy[7] = 1.5  # v
    laterally_messy[9] = 1.2  # p
    laterally_messy[11] = 1.8  # r
    assert sysid.launch_quality_score_from_state(laterally_messy) == pytest.approx(1.0)
    assert sysid.launch_confidence_weight_from_state(laterally_messy) < 1.0


def test_lateral_excitation_confidence_keeps_moderate_excitation_and_downweights_extreme_launch() -> None:
    clean_low_excitation = {
        "beta_rad": 0.0,
        "p_hat": 0.0,
        "r_hat": 0.0,
        "launch_lateral_score": 0.0,
    }
    clean_moderate_excitation = {
        "beta_rad": math.radians(8.0),
        "p_hat": 0.12,
        "r_hat": 0.12,
        "launch_lateral_score": 0.2,
    }
    contaminated_extreme = {
        "beta_rad": math.radians(18.0),
        "p_hat": 0.35,
        "r_hat": 0.35,
        "launch_lateral_score": 1.4,
    }
    weights = sysid.lateral_excitation_confidence_weights(
        [clean_low_excitation, clean_moderate_excitation, contaminated_extreme]
    )
    assert weights[1] > weights[0]
    assert weights[2] < weights[1]


def test_resolved_heldout_count_uses_fraction_after_filtering() -> None:
    assert sysid.resolved_heldout_count(filtered_valid_count=74, heldout_count=0, heldout_fraction=0.15) == 11
    assert sysid.resolved_heldout_count(filtered_valid_count=74, heldout_count=9, heldout_fraction=0.15) == 9
    assert sysid.resolved_heldout_count(filtered_valid_count=4, heldout_count=99, heldout_fraction=0.15) == 3


def test_compact_joint_sweep_parameters_are_deltas_from_active_except_blend() -> None:
    base = sysid.active_parameter_dict()
    base["attached_pitch_moment_bias_coeff"] = 0.01
    params = sysid.compact_joint_sweep_parameters(
        base,
        {
            "attached_pitch_moment_bias_coeff": 0.02,
            "post_stall_residual_blend_start_alpha_deg": 14.0,
            "post_stall_residual_blend_full_alpha_deg": 22.0,
        },
    )
    assert params["attached_pitch_moment_bias_coeff"] == pytest.approx(0.03)
    assert params["post_stall_residual_blend_start_alpha_deg"] == pytest.approx(14.0)
    assert params["post_stall_residual_blend_full_alpha_deg"] == pytest.approx(22.0)


def test_compact_joint_sweep_selected_rows_have_required_classes() -> None:
    base = sysid.active_parameter_dict()
    states = []
    for idx, (candidate_id, dy, roll, yaw, pitch, lateral_count) in enumerate(
        [
            ("longitudinal", 1.0, 20.0, 20.0, 5.0, 0),
            ("strict", 0.8, 18.0, 15.0, 5.2, 1),
            ("diagnostic", 0.4, 10.0, 8.0, 8.0, 2),
        ]
    ):
        params = dict(base)
        if lateral_count:
            params["side_force_beta_coeff"] = -0.1 * lateral_count
        states.append(
            {
                "candidate_id": candidate_id,
                "parameters": params,
                "updates": sysid.parameter_updates(base, params),
                "sweep_stage": "test",
                "split": "heldout",
                "summary": {
                    "dx_mae_m": 0.2,
                    "dy_mae_m": dy,
                    "altitude_loss_mae_m": 0.2,
                    "sink_mae_m_s": 0.2,
                    "final_phi_mae_deg": roll,
                    "final_theta_mae_deg": pitch,
                    "final_psi_mae_deg": yaw,
                },
                "score": float(idx + 1),
                "longitudinal_score": 5.0 + idx,
                "lateral_score": dy + roll / 12.0 + yaw / 18.0,
            }
        )
    rows, models = sysid.compact_joint_sweep_selected_rows(states, base)
    classes = {row["selection_class"] for row in rows}
    assert {"strict_best", "balanced_best", "diagnostic_best"} <= classes
    assert {name for name, _ in models} == classes


def test_joint_pareto_combination_overlays_only_lateral_terms() -> None:
    base = sysid.active_parameter_dict()
    longitudinal = dict(base)
    longitudinal["attached_pitch_moment_bias_coeff"] = base["attached_pitch_moment_bias_coeff"] + 0.05
    lateral = dict(base)
    lateral["side_force_beta_coeff"] = base["side_force_beta_coeff"] - 0.25
    lateral["transition_yaw_moment_p_hat_coeff"] = base["transition_yaw_moment_p_hat_coeff"] - 0.10
    lateral["post_stall_pitch_moment_coeff"] = base["post_stall_pitch_moment_coeff"] + 0.40

    combined = sysid.joint_pareto_combined_parameters(
        longitudinal,
        base_parameters=base,
        lateral_parameters=lateral,
    )

    assert combined["attached_pitch_moment_bias_coeff"] == pytest.approx(longitudinal["attached_pitch_moment_bias_coeff"])
    assert combined["side_force_beta_coeff"] == pytest.approx(lateral["side_force_beta_coeff"])
    assert combined["transition_yaw_moment_p_hat_coeff"] == pytest.approx(lateral["transition_yaw_moment_p_hat_coeff"])
    assert combined["post_stall_pitch_moment_coeff"] == pytest.approx(longitudinal["post_stall_pitch_moment_coeff"])


def test_joint_pareto_acceptance_requires_lateral_improvement_with_longitudinal_tolerance() -> None:
    reference = {
        "dx_mae_m": 0.20,
        "altitude_loss_mae_m": 0.08,
        "sink_mae_m_s": 0.07,
        "final_theta_mae_deg": 5.0,
        "dy_mae_m": 0.45,
        "final_phi_mae_deg": 10.0,
        "final_psi_mae_deg": 12.0,
    }
    accepted_candidate = {
        "dx_mae_m": 0.24,
        "altitude_loss_mae_m": 0.10,
        "sink_mae_m_s": 0.09,
        "final_theta_mae_deg": 5.8,
        "dy_mae_m": 0.40,
        "final_phi_mae_deg": 9.5,
        "final_psi_mae_deg": 11.5,
    }
    accepted, reason = sysid.joint_pareto_audit_acceptance(reference, accepted_candidate)
    assert accepted is True
    assert reason == "accepted_lateral_metrics_improved_with_longitudinal_tolerance"

    lateral_regression = dict(accepted_candidate)
    lateral_regression["final_psi_mae_deg"] = reference["final_psi_mae_deg"]
    accepted, reason = sysid.joint_pareto_audit_acceptance(reference, lateral_regression)
    assert accepted is False
    assert reason.startswith("rejected_lateral_metrics_not_all_improved")

    longitudinal_regression = dict(accepted_candidate)
    longitudinal_regression["dx_mae_m"] = reference["dx_mae_m"] + 0.10
    accepted, reason = sysid.joint_pareto_audit_acceptance(reference, longitudinal_regression)
    assert accepted is False
    assert reason.startswith("rejected_longitudinal_metrics_degraded")


def test_heavy_joint_pareto_candidate_generation_respects_limits_and_lateral_family(monkeypatch: pytest.MonkeyPatch) -> None:
    base = sysid.active_parameter_dict()
    long_a = dict(base)
    long_a["attached_pitch_moment_bias_coeff"] = base["attached_pitch_moment_bias_coeff"] + 0.01
    long_b = dict(base)
    long_b["transition_pitch_moment_bias_coeff"] = base["transition_pitch_moment_bias_coeff"] + 0.01
    top_longitudinal = [
        {"candidate_id": "long_a", "parameters": long_a, "longitudinal_score": 1.0},
        {"candidate_id": "long_b", "parameters": long_b, "longitudinal_score": 2.0},
    ]
    lateral_sources = [
        {"candidate_id": "no_lateral_update", "parameters": dict(base), "updates": {}, "source_priority": 0.0},
        {
            "candidate_id": "source_attached",
            "parameters": {**base, "side_force_beta_coeff": base["side_force_beta_coeff"] - 0.2},
            "updates": {"side_force_beta_coeff": base["side_force_beta_coeff"] - 0.2},
            "source_priority": 0.0,
        },
        {
            "candidate_id": "source_transition",
            "parameters": {**base, "transition_yaw_moment_p_hat_coeff": base["transition_yaw_moment_p_hat_coeff"] - 0.1},
            "updates": {"transition_yaw_moment_p_hat_coeff": base["transition_yaw_moment_p_hat_coeff"] - 0.1},
            "source_priority": 0.0,
        },
        {
            "candidate_id": "source_post",
            "parameters": {
                **base,
                sysid.lateral_surface_parameter_name("post_stall_yaw_moment", "p_hat", sysid.SURFACE_RBF_ALPHA_CENTERS_DEG[0]): -0.05,
            },
            "updates": {
                sysid.lateral_surface_parameter_name("post_stall_yaw_moment", "p_hat", sysid.SURFACE_RBF_ALPHA_CENTERS_DEG[0]): -0.05,
            },
            "source_priority": 0.0,
        },
    ]

    def fake_evaluate(states: list[dict[str, object]], **_: object) -> list[dict[str, object]]:
        out = []
        for index, state in enumerate(states):
            updated = dict(state)
            updated["summary"] = {
                "dx_mae_m": 0.2,
                "dy_mae_m": 0.5 - 0.01 * index,
                "altitude_loss_mae_m": 0.1,
                "sink_mae_m_s": 0.1,
                "final_phi_mae_deg": 10.0 - 0.1 * index,
                "final_theta_mae_deg": 5.0,
                "final_psi_mae_deg": 12.0 - 0.1 * index,
            }
            updated["split"] = "heldout"
            updated["score"] = float(index)
            updated["longitudinal_score"] = 1.0
            updated["lateral_score"] = float(index)
            out.append(updated)
        return out

    monkeypatch.setattr(sysid, "compact_joint_sweep_evaluate_states", fake_evaluate)
    states = sysid.joint_pareto_heavy_combination_states(
        top_longitudinal_sources=top_longitudinal,
        lateral_sources=lateral_sources,
        base_parameters=base,
        best_longitudinal=top_longitudinal[0],
        heldout_rows=[{}],
        replay_dt_s=0.005,
        alignment_window_s=0.040,
        workers=1,
        config={
                **sysid.resolved_joint_pareto_profile_config(
                    profile="heavy",
                    top_longitudinal=2,
                    top_lateral=24,
                    max_lateral_order=3,
                    top_triples=2,
                    max_candidates=900,
                    selected_limit=8,
                )
            },
        )

    assert len(states) <= 900
    assert any(state["lateral_source_id"] == "no_lateral_update" for state in states)
    assert any(int(state["bundle_order"]) == 3 for state in states)
    for state in states:
        lateral_keys = [
            key for key in sysid.parameter_updates(base, state["parameters"]) if sysid.joint_sweep_is_lateral_parameter(key)
        ]
        assert all(sysid.joint_pareto_heavy_allowed_lateral_key(key) for key in lateral_keys)


def test_local_audit_candidate_generation_is_exact_dense_two_term_grid() -> None:
    base = sysid.active_parameter_dict()
    top_longitudinal = []
    for index, source_id in enumerate(local_audit.LONGITUDINAL_SOURCE_IDS):
        params = dict(base)
        params["post_stall_residual_blend_start_alpha_deg"] = 14.0 + index
        params["post_stall_residual_blend_full_alpha_deg"] = 18.0 + index
        top_longitudinal.append({"candidate_id": source_id, "parameters": params, "longitudinal_score": float(index)})
    source_values = {
        local_audit.YAW_BETA_KEY: base[local_audit.YAW_BETA_KEY] - 0.05,
        local_audit.POST_STALL_CLR_KEY: base[local_audit.POST_STALL_CLR_KEY] - 0.4,
    }

    states = local_audit.build_local_candidate_states(
        longitudinal_sources=top_longitudinal,
        base_parameters=base,
        source_values=source_values,
        alignment_window_s=0.040,
    )

    assert len(states) == 243
    assert sum(state["lateral_source_id"] == "no_lateral_update" for state in states) == 3
    assert sum(int(state["bundle_order"]) == 1 for state in states) == 48
    assert sum(int(state["bundle_order"]) == 2 for state in states) == 192
    allowed_lateral = {
        local_audit.YAW_BETA_KEY,
        local_audit.POST_STALL_CLR_KEY,
    }
    for state in states:
        lateral_keys = {
            key
            for key in sysid.parameter_updates(base, state["parameters"])
            if sysid.joint_sweep_is_lateral_parameter(key)
        }
        assert lateral_keys <= allowed_lateral


def test_local_audit_infers_scaled_source_values_from_heavy_candidate_rows() -> None:
    base = sysid.active_parameter_dict()
    source_value = base[local_audit.POST_STALL_CLR_KEY] - 0.4
    rows = [
        {
            "lateral_source_id": "ablation_post_stall_Cl_r__post_stall_roll_moment_r_hat_r__s0p5",
            "parameter_updates_json": json.dumps(
                {
                    local_audit.POST_STALL_CLR_KEY: base[local_audit.POST_STALL_CLR_KEY]
                    + 0.5 * (source_value - base[local_audit.POST_STALL_CLR_KEY])
                }
            ),
        },
        {
            "lateral_source_id": "ablation_post_stall_Cl_r__post_stall_roll_moment_r_hat_r__s0p75",
            "parameter_updates_json": json.dumps(
                {
                    local_audit.POST_STALL_CLR_KEY: base[local_audit.POST_STALL_CLR_KEY]
                    + 0.75 * (source_value - base[local_audit.POST_STALL_CLR_KEY])
                }
            ),
        },
    ]

    inferred = local_audit.infer_scaled_source_value(
        rows,
        base_parameters=base,
        key=local_audit.POST_STALL_CLR_KEY,
        source_pattern=local_audit.CLR_SOURCE_PATTERN,
    )

    assert inferred == pytest.approx(source_value)


def test_local_audit_acceptance_uses_stricter_per_base_longitudinal_gate() -> None:
    reference = {
        "dx_mae_m": 0.20,
        "altitude_loss_mae_m": 0.10,
        "sink_mae_m_s": 0.10,
        "final_theta_mae_deg": 5.0,
        "dy_mae_m": 0.60,
        "final_phi_mae_deg": 12.0,
        "final_psi_mae_deg": 15.0,
    }
    accepted_candidate = {
        "dx_mae_m": 0.224,
        "altitude_loss_mae_m": 0.109,
        "sink_mae_m_s": 0.109,
        "final_theta_mae_deg": 5.4,
        "dy_mae_m": 0.59,
        "final_phi_mae_deg": 11.9,
        "final_psi_mae_deg": 14.9,
    }
    accepted, reason = sysid.joint_pareto_audit_acceptance(
        reference,
        accepted_candidate,
        longitudinal_tolerances=local_audit.STRICT_LONGITUDINAL_TOLERANCES,
    )
    assert accepted is True
    assert reason == "accepted_lateral_metrics_improved_with_longitudinal_tolerance"

    rejected_candidate = dict(accepted_candidate)
    rejected_candidate["dx_mae_m"] = 0.226
    accepted, reason = sysid.joint_pareto_audit_acceptance(
        reference,
        rejected_candidate,
        longitudinal_tolerances=local_audit.STRICT_LONGITUDINAL_TOLERANCES,
    )
    assert accepted is False
    assert reason == "rejected_longitudinal_metrics_degraded:dx_mae_m"


def test_joint_pareto_rows_accept_against_each_longitudinal_base() -> None:
    base = sysid.active_parameter_dict()
    reference_good = {
        "candidate_id": "ref_good",
        "summary": {
            "dx_mae_m": 0.20,
            "altitude_loss_mae_m": 0.10,
            "sink_mae_m_s": 0.10,
            "final_theta_mae_deg": 5.0,
            "dy_mae_m": 0.60,
            "final_phi_mae_deg": 12.0,
            "final_psi_mae_deg": 15.0,
        },
        "longitudinal_score": 1.0,
        "lateral_score": 3.0,
    }
    reference_bad = {
        "candidate_id": "ref_bad",
        "summary": {
            "dx_mae_m": 0.50,
            "altitude_loss_mae_m": 0.20,
            "sink_mae_m_s": 0.20,
            "final_theta_mae_deg": 9.0,
            "dy_mae_m": 0.70,
            "final_phi_mae_deg": 14.0,
            "final_psi_mae_deg": 17.0,
        },
        "longitudinal_score": 2.0,
        "lateral_score": 4.0,
    }
    candidate_params = dict(base)
    candidate_params["side_force_beta_coeff"] = base["side_force_beta_coeff"] - 0.2
    candidate = {
        "candidate_id": "candidate",
        "longitudinal_source_id": "bad_base",
        "lateral_source_id": "lat",
        "parameters": candidate_params,
        "summary": {
            "dx_mae_m": 0.54,
            "altitude_loss_mae_m": 0.22,
            "sink_mae_m_s": 0.22,
            "final_theta_mae_deg": 9.5,
            "dy_mae_m": 0.60,
            "final_phi_mae_deg": 13.0,
            "final_psi_mae_deg": 16.0,
        },
        "score": 1.0,
        "longitudinal_score": 2.2,
        "lateral_score": 3.5,
        "split": "heldout",
    }

    rows = sysid.joint_pareto_audit_candidate_rows(
        [candidate],
        base_parameters=base,
        reference_state=reference_good,
        reference_by_longitudinal={"bad_base": reference_bad},
        global_reference_state=reference_good,
        alignment_window_s=0.040,
        profile="heavy",
    )

    assert rows[0]["accepted"] is True
    assert rows[0]["reference_candidate_id"] == "ref_bad"
    assert rows[0]["global_reference_candidate_id"] == "ref_good"
    assert rows[0]["delta_dx_mae_m"] == pytest.approx(0.04)
    assert rows[0]["global_delta_dx_mae_m"] == pytest.approx(0.34)


def test_joint_pareto_selected_rows_are_capped_accepted_pareto_survivors() -> None:
    rows = [
        {
            "candidate_id": f"c{idx}",
            "accepted": True,
            "pareto_member": True,
            "lateral_score": float(idx),
            "longitudinal_score": float(idx),
        }
        for idx in range(5)
    ]
    rows.append({"candidate_id": "rejected", "accepted": False, "pareto_member": True, "lateral_score": -1.0, "longitudinal_score": -1.0})
    selected = sysid.joint_pareto_audit_selected_rows(rows, selected_limit=3)

    assert [row["candidate_id"] for row in selected] == ["c0", "c1", "c2"]
    assert all(row["selection_class"] == "accepted_pareto" for row in selected)


def test_heavy_stage_replay_rows_use_normal_report_label(monkeypatch: pytest.MonkeyPatch) -> None:
    base = sysid.active_parameter_dict()
    selected = {
        "candidate_id": "selected",
        "reference_candidate_id": "reference",
        "global_reference_candidate_id": "reference",
        "parameter_updates_json": "{}",
    }
    reference = {"candidate_id": "reference", "parameter_updates_json": "{}"}

    def fake_sample_payload(payload: tuple[str, str, dict[str, object], dict[str, float], float, float]) -> list[dict[str, object]]:
        model_id, split, *_ = payload
        return [
            {
                "model_id": model_id,
                "split": split,
                "session_label": "s",
                "throw_id": "same_throw",
                "regime": regime,
                "dx_residual_actual_minus_sim_m": 0.1,
                "dy_residual_actual_minus_sim_m": 0.2,
                "altitude_loss_residual_actual_minus_sim_m": 0.3,
                "sink_rate_residual_actual_minus_sim_m_s": 0.4,
                "roll_residual_actual_minus_sim_deg": 1.0,
                "pitch_residual_actual_minus_sim_deg": 2.0,
                "yaw_residual_actual_minus_sim_deg": 3.0,
            }
            for regime in ("attached", "transition", "post_stall")
        ]

    monkeypatch.setattr(sysid, "stage_replay_sample_rows_payload", fake_sample_payload)
    rows = sysid.joint_pareto_heavy_stage_replay_summary_rows(
        candidate_rows=[selected, reference],
        selected_rows=[selected],
        base_parameters=base,
        heldout_rows=[{}],
        replay_dt_s=0.005,
        alignment_window_s=0.040,
        workers=1,
    )

    assert {row["report_regime"] for row in rows} == {"normal", "transition", "post_stall"}
    assert all(row["split"] == "heldout" for row in rows)
    assert all(int(row["throw_count"]) == 1 for row in rows)
    assert any(row["model_role"] == "selected_candidate" for row in rows)
    assert any(row["model_role"] == "matched_longitudinal_reference" for row in rows)


def test_compact_joint_sweep_replay_path_does_not_hardcode_single_worker() -> None:
    source = inspect.getsource(sysid.compact_joint_sweep_evaluate_states)
    assert "max_workers=int(workers)" in source
    assert "workers=1" not in source


def test_lateral_candidate_application_is_limited_to_minimal_terms() -> None:
    base = sysid.active_parameter_dict()
    coeffs = sysid.zero_coefficients()
    coeffs.update(
        {
            "side_force_bias_coeff": 9.0,
            "side_force_beta_coeff": 0.1,
            "roll_moment_beta_coeff": 9.0,
            "roll_moment_p_hat_coeff": 0.2,
            "yaw_moment_beta_coeff": 9.0,
            "yaw_moment_r_hat_coeff": 0.3,
            "post_stall_lift_residual_coeff": 9.0,
        }
    )
    candidate = sysid.candidate_from_fit(
        base,
        {"coefficients": coeffs},
        apply_attached_cm_bias=False,
        fit_post_stall_longitudinal=False,
        fit_transition_blender=False,
        fit_post_stall_surfaces=False,
        fit_post_stall_damping=False,
        fit_attached_lateral_coupling=True,
        fit_transition_lateral_coupling=False,
        fit_lateral_surfaces=False,
        group_scales={
            "attached_lateral": 1.0,
            "post_stall_longitudinal": 0.0,
            "transition_blender": 0.0,
        },
    )

    assert candidate["side_force_beta_coeff"] == pytest.approx(base["side_force_beta_coeff"] + 0.1)
    assert candidate["roll_moment_p_hat_coeff"] == pytest.approx(base["roll_moment_p_hat_coeff"] + 0.2)
    assert candidate["yaw_moment_r_hat_coeff"] == pytest.approx(base["yaw_moment_r_hat_coeff"] + 0.3)
    assert candidate["side_force_bias_coeff"] == pytest.approx(base["side_force_bias_coeff"])
    assert candidate["roll_moment_beta_coeff"] == pytest.approx(base["roll_moment_beta_coeff"])
    assert candidate["yaw_moment_beta_coeff"] == pytest.approx(base["yaw_moment_beta_coeff"])
    assert candidate["post_stall_lift_residual_coeff"] == pytest.approx(base["post_stall_lift_residual_coeff"])


def test_attached_cm_bias_maps_to_attached_regime_parameter_only() -> None:
    base = sysid.active_parameter_dict()
    coeffs = sysid.zero_coefficients()
    coeffs["attached_cm_bias_coeff"] = 0.08
    candidate = sysid.candidate_from_fit(
        base,
        {"coefficients": coeffs},
        apply_attached_cm_bias=True,
        fit_post_stall_longitudinal=False,
        fit_transition_blender=False,
        fit_post_stall_surfaces=False,
        fit_post_stall_damping=False,
        fit_attached_lateral_coupling=False,
        fit_transition_lateral_coupling=False,
        fit_lateral_surfaces=False,
    )

    assert candidate["attached_pitch_moment_bias_coeff"] == pytest.approx(
        base["attached_pitch_moment_bias_coeff"] + 0.08
    )
    assert candidate["pitch_moment_bias_coeff"] == pytest.approx(base["pitch_moment_bias_coeff"])
    assert candidate["transition_pitch_moment_bias_coeff"] == pytest.approx(base["transition_pitch_moment_bias_coeff"])


def test_post_stall_and_transition_groups_can_be_frozen() -> None:
    base = sysid.active_parameter_dict()
    coeffs = sysid.zero_coefficients()
    coeffs.update(
        {
            "post_stall_lift_residual_coeff": 0.4,
            "post_stall_drag_residual_coeff": 0.5,
            "post_stall_pitch_moment_coeff": 0.6,
            "post_stall_pitch_damping_coeff": 0.7,
            "post_stall_residual_blend_start_alpha_deg": 9.0,
            "post_stall_residual_blend_full_alpha_deg": 18.0,
        }
    )
    candidate = sysid.candidate_from_fit(
        base,
        {"coefficients": coeffs},
        apply_attached_cm_bias=False,
        fit_post_stall_longitudinal=False,
        fit_transition_blender=False,
        fit_post_stall_surfaces=False,
        fit_post_stall_damping=True,
        fit_attached_lateral_coupling=False,
        fit_transition_lateral_coupling=False,
        fit_lateral_surfaces=False,
    )

    assert candidate["post_stall_lift_residual_coeff"] == pytest.approx(base["post_stall_lift_residual_coeff"])
    assert candidate["post_stall_drag_residual_coeff"] == pytest.approx(base["post_stall_drag_residual_coeff"])
    assert candidate["post_stall_pitch_moment_coeff"] == pytest.approx(base["post_stall_pitch_moment_coeff"])
    assert candidate["post_stall_pitch_damping_coeff"] == pytest.approx(base["post_stall_pitch_damping_coeff"])
    assert candidate["post_stall_residual_blend_start_alpha_deg"] == pytest.approx(
        base["post_stall_residual_blend_start_alpha_deg"]
    )
    assert candidate["post_stall_residual_blend_full_alpha_deg"] == pytest.approx(
        base["post_stall_residual_blend_full_alpha_deg"]
    )


def test_compact_joint_sweep_tests_delayed_full_stall_blend() -> None:
    assert (12.0, 24.0) in sysid.COMPACT_JOINT_SWEEP_BLEND_GRID_DEG
    assert (14.0, 24.0) in sysid.COMPACT_JOINT_SWEEP_BLEND_GRID_DEG


def test_post_stall_pitch_damping_upper_bound_is_released_for_diagnostics() -> None:
    assert sysid.replay_fit.bounded_parameter_value("post_stall_pitch_damping_coeff", 9.0) == pytest.approx(8.0)
    assert sysid.replay_fit.bounded_parameter_value("post_stall_pitch_damping_coeff", -9.0) == pytest.approx(-4.0)


def test_active_calibration_is_promoted_compact_coupled_replay_model() -> None:
    assert active_calibration.CALIBRATION_ID == (
        "neutral_dry_air_residual_calibrated_replay_n30_joint_pareto_040_local_s5_yaw0p75_clr0p60_elevator_rudder_effectiveness_v1"
    )
    assert active_calibration.CLAIM_BOUNDARY == (
        "compact_residual_calibrated_replay_alignment_with_40ms_local_pareto_transition_blend_yaw_beta_and_post_stall_clr_corrections_conservative_elevator_and_rudder_effectiveness"
    )
    assert active_calibration.SOURCE_PREP_RUN.endswith("n30_joint_pareto_040_local_promising_v1")
    assert active_calibration.SOURCE_HEAVY_JOINT_PARETO_RUN.endswith("n30_joint_pareto_040_heavy_v1")
    assert active_calibration.SOURCE_SELECTED_CANDIDATE == (
        "jp040local_L00_proposal_stage_5_stage5_tran_local_yaw_beta_s0p75_local_post_stall_Cl_r_s0p6"
    )
    assert active_calibration.SOURCE_SELECTED_LONGITUDINAL_BASE == "proposal_stage_5_stage5_transition_blend"
    assert active_calibration.SOURCE_SELECTED_LATERAL_BUNDLE == (
        "local_yaw_beta_s0p75+local_post_stall_Cl_r_s0p6"
    )
    assert active_calibration.ALIGNMENT_WINDOW_S == pytest.approx(0.040)
    assert active_calibration.ATTACHED_PITCH_MOMENT_BIAS_COEFF == pytest.approx(0.11309832420327923)
    assert active_calibration.TRANSITION_PITCH_MOMENT_BIAS_COEFF == pytest.approx(0.05711558897899738)
    assert active_calibration.POST_STALL_PITCH_MOMENT_COEFF == pytest.approx(0.07585874586245771)
    assert active_calibration.POST_STALL_PITCH_DAMPING_COEFF == pytest.approx(4.0)
    assert active_calibration.POST_STALL_RESIDUAL_BLEND_START_ALPHA_DEG == pytest.approx(14.0)
    assert active_calibration.POST_STALL_RESIDUAL_BLEND_FULL_ALPHA_DEG == pytest.approx(18.0)
    assert active_calibration.SIDE_FORCE_BETA_COEFF == pytest.approx(-1.9802669025044202)
    assert active_calibration.YAW_MOMENT_BETA_COEFF == pytest.approx(-0.034325897691662534)
    assert active_calibration.TRANSITION_SIDE_FORCE_R_HAT_COEFF == pytest.approx(-3.0)
    assert active_calibration.TRANSITION_YAW_MOMENT_BETA_COEFF == pytest.approx(-0.01467582447)
    assert active_calibration.TRANSITION_YAW_MOMENT_P_HAT_COEFF == pytest.approx(-0.1461483136170422)
    assert np.asarray(active_calibration.POST_STALL_ROLL_MOMENT_RBF_COEFFS, dtype=float)[3, 0] == pytest.approx(
        -0.46003383312128726
    )
    assert np.asarray(active_calibration.POST_STALL_YAW_MOMENT_RBF_COEFFS, dtype=float)[1, 0] == pytest.approx(
        -0.01870860145
    )
    assert np.asarray(active_calibration.POST_STALL_YAW_MOMENT_RBF_COEFFS, dtype=float)[2, 0] == pytest.approx(
        -0.07119920085255141
    )
    assert active_calibration.DELTA_A_AERO_EFFECTIVENESS_SCALE == pytest.approx(1.0)
    assert active_calibration.DELTA_E_AERO_EFFECTIVENESS_SCALE == pytest.approx(0.60)
    assert active_calibration.DELTA_R_AERO_EFFECTIVENESS_SCALE == pytest.approx(0.531)


def test_theory_baseline_is_comparison_only_data_artifact() -> None:
    comparison_path = (
        INNER_LOOP / "A_model_parameters" / "neutral_dry_air_theory_baseline_comparison.json"
    )
    with comparison_path.open("r", encoding="utf-8") as handle:
        comparison = json.load(handle)

    assert comparison_path.suffix == ".json"
    assert comparison["artifact_role"] == "comparison_only_not_runtime_imported"
    assert comparison["calibration_active"] is False
    assert comparison["claim_boundary"] == "reference_baseline_only_no_sim_or_real_runtime_call"
    assert comparison["runtime_model_module"].endswith("neutral_dry_air_calibration.py")
    assert comparison["constants"]["attached_pitch_moment_bias_coeff"] == pytest.approx(0.0)
    assert comparison["constants"]["post_stall_pitch_damping_coeff"] == pytest.approx(0.0)


def test_mass_properties_match_ballasted_glider_measurement() -> None:
    assert mass_properties_estimate.MASS_KG == pytest.approx(0.14856)
    assert mass_properties_estimate.R_CG_BUILD_M[0] == pytest.approx(0.1055)
    assert mass_properties_estimate.R_CG_BUILD_M[1] == pytest.approx(0.0)
    inertia = np.asarray(mass_properties_estimate.INERTIA_B, dtype=float)
    assert inertia.shape == (3, 3)
    assert np.allclose(inertia, inertia.T)
