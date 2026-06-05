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
from A_model_parameters import mass_properties_estimate  # noqa: E402
from A_model_parameters import neutral_dry_air_calibration as active_calibration  # noqa: E402


def test_default_neutral_sysid_is_longitudinal_primary_with_lateral_diagnostic() -> None:
    assert sysid.DEFAULT_FIT_WORKFLOW == "cm_regime_staged"
    assert sysid.DEFAULT_FIT_ATTACHED_LATERAL_COUPLING is False
    assert sysid.DEFAULT_FIT_TRANSITION_LATERAL_COUPLING is False
    assert sysid.DEFAULT_FIT_LATERAL_SURFACES is False
    assert sysid.DEFAULT_FIT_POST_STALL_SURFACES is False
    assert sysid.DEFAULT_FIT_SECONDARY_LATERAL_DIAGNOSTIC is True
    assert sysid.DEFAULT_ALIGNED_U_MIN_M_S == pytest.approx(3.0)
    assert sysid.DEFAULT_ALIGNED_U_MAX_M_S == pytest.approx(8.0)
    parser = sysid.build_arg_parser()
    args = parser.parse_args(["--fit-workflow", "compact_joint_sweep", "--workers", "8"])
    assert args.fit_workflow == "compact_joint_sweep"
    assert args.workers == 8


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
    assert active_calibration.CALIBRATION_ID == "neutral_dry_air_residual_calibrated_replay_n30_compact_coupled_v1"
    assert active_calibration.CLAIM_BOUNDARY == (
        "compact_residual_calibrated_replay_alignment_with_selected_coupling_terms"
    )
    assert active_calibration.SOURCE_SELECTED_CANDIDATE == "joint_0270_post_stall_Cn_p_1.5"
    assert active_calibration.ATTACHED_PITCH_MOMENT_BIAS_COEFF == pytest.approx(0.11309832420327923)
    assert active_calibration.TRANSITION_PITCH_MOMENT_BIAS_COEFF == pytest.approx(0.05711558897899738)
    assert active_calibration.POST_STALL_PITCH_MOMENT_COEFF == pytest.approx(0.07585874586245771)
    assert active_calibration.POST_STALL_PITCH_DAMPING_COEFF == pytest.approx(4.0)
    assert active_calibration.POST_STALL_RESIDUAL_BLEND_FULL_ALPHA_DEG == pytest.approx(22.0)
    assert active_calibration.SIDE_FORCE_BETA_COEFF == pytest.approx(-1.9802669025044202)
    assert active_calibration.TRANSITION_SIDE_FORCE_R_HAT_COEFF == pytest.approx(-3.0)
    assert active_calibration.TRANSITION_YAW_MOMENT_P_HAT_COEFF == pytest.approx(-0.1461483136170422)
    assert np.asarray(active_calibration.POST_STALL_YAW_MOMENT_RBF_COEFFS, dtype=float)[2, 0] == pytest.approx(
        -0.07119920085255141
    )


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
