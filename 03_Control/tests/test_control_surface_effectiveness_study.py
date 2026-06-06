from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
INNER_LOOP = ROOT / "03_Control" / "02_Inner_Loop"
if str(INNER_LOOP) not in sys.path:
    sys.path.insert(0, str(INNER_LOOP))

import run_control_surface_effectiveness_study as study  # noqa: E402
from glider import AILERON, ELEVATOR, RUDDER, build_nausicaa_glider  # noqa: E402
from A_model_parameters import neutral_dry_air_calibration as active_calibration  # noqa: E402


@pytest.fixture
def local_tmp_path(request: pytest.FixtureRequest) -> Path:
    root = ROOT / ".tmp_cse"
    path = root / f"t{abs(hash(request.node.name)) % 100000:05d}"
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_inventory_parser_reads_manifest_and_keeps_valid_lattice_throw(local_tmp_path: Path) -> None:
    throw_dir = write_throw(local_tmp_path, "pa30", "delta_a", -0.4, valid=True)

    row = study.inventory_row_from_throw(local_tmp_path / "pa30", throw_dir)

    assert row["surface_axis"] == "aileron"
    assert row["command_axis"] == "delta_a"
    assert row["command_value"] == pytest.approx(-0.4)
    assert row["command_lattice_percent"] == pytest.approx(-40.0)
    assert row["command_schedule_status"] == "ok"
    assert row["filter_status"] == "kept"


def test_inventory_keeps_invalid_throw_as_filtered_row(local_tmp_path: Path) -> None:
    throw_dir = write_throw(local_tmp_path, "pr30", "delta_r", 0.2, valid=False)

    row = study.inventory_row_from_throw(local_tmp_path / "pr30", throw_dir)

    assert row["surface_axis"] == "rudder"
    assert row["filter_status"] == "filtered"
    assert "invalid_or_cancelled_throw" in row["filter_reasons"]


def test_command_schedule_audit_rejects_off_axis_or_wrong_sign() -> None:
    rows = [
        {
            "t_s": "0.15",
            "pulse_active": "True",
            "delta_a_cmd_norm": "0.0",
            "delta_e_cmd_norm": "0.2",
            "delta_r_cmd_norm": "0.0",
        }
    ]

    status, notes = study.command_schedule_audit(rows, "delta_a", 0.2, 0.15)

    assert status == "command_value_mismatch"
    assert "manifest" in notes


def test_20_percent_lattice_parser_accepts_only_fixed_ladder() -> None:
    assert study.is_lattice_20_percent(0.2)
    assert study.is_lattice_20_percent(-1.0)
    assert not study.is_lattice_20_percent(0.3)
    assert not study.is_lattice_20_percent(1.2)


def test_launch_level_heldout_split_uses_surface_magnitude_sign_pairs() -> None:
    rows = []
    for command in (-0.2, 0.2):
        for index in range(3):
            rows.append(
                {
                    "filter_status": "kept",
                    "surface_axis": "aileron",
                    "command_value": command,
                    "command_abs": abs(command),
                    "trial_id": f"{command}/{index}",
                }
            )

    study.assign_launch_level_split(rows, heldout_seed=606)

    heldout = [row for row in rows if row["split"] == "heldout"]
    assert len(heldout) == 2
    assert {math.copysign(1.0, row["command_value"]) for row in heldout} == {-1.0, 1.0}
    assert all(row["split"] in {"train", "heldout"} for row in rows)


def test_antisymmetric_and_symmetric_formulas_are_pairwise() -> None:
    positive = [{"actual_peak_p_rad_s": 4.0, "sim_peak_p_rad_s": 3.0}]
    negative = [{"actual_peak_p_rad_s": -2.0, "sim_peak_p_rad_s": -1.0}]

    row = study.effectiveness_metric_row("all", "aileron", 0.2, "peak_p_rad_s", positive, negative)

    assert row["actual_antisymmetric_response"] == pytest.approx(3.0)
    assert row["actual_symmetric_response"] == pytest.approx(1.0)
    assert row["sim_antisymmetric_response"] == pytest.approx(2.0)
    assert row["sim_symmetric_response"] == pytest.approx(1.0)


def test_confidence_weighted_effectiveness_uses_launch_weights() -> None:
    positive = [
        {"actual_peak_p_rad_s": 10.0, "sim_peak_p_rad_s": 8.0, "launch_confidence_weight": 0.1},
        {"actual_peak_p_rad_s": 0.0, "sim_peak_p_rad_s": 0.0, "launch_confidence_weight": 0.9},
    ]
    negative = [{"actual_peak_p_rad_s": -2.0, "sim_peak_p_rad_s": -1.0, "launch_confidence_weight": 1.0}]

    row = study.effectiveness_metric_row(
        "confidence_weighted_all",
        "aileron",
        0.2,
        "peak_p_rad_s",
        positive,
        negative,
        confidence_weighted=True,
    )

    assert row["actual_positive_mean"] == pytest.approx(1.0)
    assert row["actual_antisymmetric_response"] == pytest.approx(1.5)


def test_launch_confidence_scores_clean_and_asymmetric_starts() -> None:
    clean = study.launch_confidence_from_inventory_row(
        {
            "u0_m_s": 5.0,
            "v0_m_s": 0.0,
            "w0_m_s": 0.0,
            "phi0_deg": 0.0,
            "theta0_deg": 0.0,
            "psi0_deg": 0.0,
            "p0_rad_s": 0.0,
            "q0_rad_s": 0.0,
            "r0_rad_s": 0.0,
            "mean_rate_confidence": 0.95,
            "response_spike_fraction": 0.0,
        }
    )
    asymmetric = study.launch_confidence_from_inventory_row(
        {
            "u0_m_s": 3.2,
            "v0_m_s": 1.4,
            "w0_m_s": 0.8,
            "phi0_deg": 14.0,
            "theta0_deg": 20.0,
            "psi0_deg": 14.0,
            "p0_rad_s": 1.1,
            "q0_rad_s": 1.0,
            "r0_rad_s": 1.7,
            "mean_rate_confidence": 0.55,
            "response_spike_fraction": 0.2,
        }
    )

    assert clean["launch_confidence_label"] == "high"
    assert asymmetric["launch_confidence_label"] == "low"


def test_surface_derivative_fit_uses_launch_confidence_weights() -> None:
    samples = [
        {"state_12": 1.0, "moment_coeff_residual_0": 4.0, "launch_confidence_weight": 0.1},
        {"state_12": 1.0, "moment_coeff_residual_0": 2.0, "launch_confidence_weight": 0.9},
    ]

    coeff = study.weighted_surface_derivative_fit(samples, state_index=12, moment_index=0)
    baseline = study.weighted_derivative_mae(samples, state_index=12, moment_index=0, coefficient=0.0)
    candidate = study.weighted_derivative_mae(samples, state_index=12, moment_index=0, coefficient=coeff)

    assert coeff == pytest.approx(2.2)
    assert candidate < baseline


def test_surface_derivative_fit_accepts_side_force_residual_key() -> None:
    samples = [
        {"state_12": 1.0, "force_coeff_residual_y": 6.0, "launch_confidence_weight": 0.25},
        {"state_12": 1.0, "force_coeff_residual_y": 2.0, "launch_confidence_weight": 0.75},
    ]

    coeff = study.weighted_surface_derivative_fit(samples, state_index=12, residual_key="force_coeff_residual_y")
    baseline = study.weighted_derivative_mae(samples, state_index=12, residual_key="force_coeff_residual_y", coefficient=0.0)
    candidate = study.weighted_derivative_mae(samples, state_index=12, residual_key="force_coeff_residual_y", coefficient=coeff)

    assert coeff == pytest.approx(3.0)
    assert candidate < baseline


def test_surface_aero_coupling_candidate_families_are_constrained() -> None:
    coefficients = {
        "Cl_delta_a_residual": 0.1,
        "Cm_delta_e_residual": -0.2,
        "Cn_delta_r_residual": 0.3,
        "Cn_delta_a_residual": -0.4,
        "CY_delta_a_residual": 0.5,
        "CY_delta_r_residual": -0.6,
    }

    families = study.surface_aero_coupling_candidate_coefficients(coefficients)

    assert families["C0_frozen_neutral"] == {}
    assert set(families["C1_primary_moment_derivatives"]) == {
        "Cl_delta_a_residual",
        "Cm_delta_e_residual",
        "Cn_delta_r_residual",
    }
    assert "Cn_delta_a_residual" in families["C2_c1_plus_aileron_adverse_yaw"]
    assert "CY_delta_a_residual" in families["C4_c1_plus_surface_side_force"]
    assert "Cl_delta_r_residual" not in families["C5_c2_plus_surface_side_force"]


def test_scaled_surface_command_applies_pairwise_gain_by_surface() -> None:
    command = np.asarray([1.0, 2.0, 3.0], dtype=float)

    scaled = study.scaled_surface_command(command, {"elevator": 0.5, "rudder": 1.25})

    assert scaled.tolist() == pytest.approx([1.0, 1.0, 3.75])
    assert command.tolist() == pytest.approx([1.0, 2.0, 3.0])


def test_surface_scale_candidate_grid_includes_exact_aileron_active_baseline() -> None:
    candidates = study.single_axis_surface_scale_candidates()

    aileron_ids = [
        candidate_id
        for candidate_id, scale_by_surface in candidates.items()
        if "aileron" in scale_by_surface
    ]

    assert len(candidates) == 22
    assert len(aileron_ids) == 8
    assert study.surface_scale_candidate_id("aileron", 1.0) in candidates
    assert candidates[study.surface_scale_candidate_id("aileron", 1.0)] == {"aileron": pytest.approx(1.0)}
    assert tuple(candidates[study.surface_scale_candidate_id("aileron", scale)]["aileron"] for scale in study.SURFACE_SCALE_CANDIDATE_GRIDS["aileron"]) == pytest.approx(
        study.SURFACE_SCALE_CANDIDATE_GRIDS["aileron"]
    )


def test_surface_scale_candidate_scales_control_mix_without_changing_command_values() -> None:
    aircraft = build_nausicaa_glider()
    control_mix = np.asarray(aircraft.control_mix, dtype=float)
    active_scales = study.active_surface_effectiveness_scales_by_surface()
    command = np.asarray([0.2, -0.4, 0.6], dtype=float)

    candidate = study.aircraft_with_surface_effectiveness_scales(
        aircraft,
        {"aileron": 0.75, "elevator": active_scales["elevator"], "rudder": active_scales["rudder"]},
    )
    unchanged_command = study.scaled_surface_command(command, None)

    assert np.asarray(candidate.control_mix)[:, AILERON].tolist() == pytest.approx(
        (control_mix[:, AILERON] * 0.75).tolist()
    )
    assert np.asarray(candidate.control_mix)[:, ELEVATOR].tolist() == pytest.approx(
        (control_mix[:, ELEVATOR] * active_scales["elevator"]).tolist()
    )
    assert np.asarray(candidate.control_mix)[:, RUDDER].tolist() == pytest.approx(
        (control_mix[:, RUDDER] * active_scales["rudder"]).tolist()
    )
    assert np.allclose(np.asarray(candidate.control_effectiveness_regime_scales), np.ones((3, 3)))
    assert unchanged_command.tolist() == pytest.approx(command.tolist())


def test_stage_surface_scale_candidate_generation_is_exact_grid() -> None:
    candidates = study.stage_surface_scale_candidates()

    assert len(candidates) == 117
    assert {
        (row["surface_axis"], row["alpha_regime"])
        for row in candidates
    } == {
        (surface, regime)
        for surface in ("aileron", "elevator", "rudder")
        for regime in study.SURFACE_AERO_ALPHA_REGIMES
    }
    assert tuple(row["candidate_scale"] for row in candidates[:13]) == pytest.approx(study.STAGE_SURFACE_SCALE_GRID)


def test_stage_selection_uses_all_data_and_one_sign_evidence() -> None:
    evidence_rows = [
        {"filter_status": "kept", "surface_axis": "aileron", "max_abs_alpha_deg": 10.0, "command_value": 0.4}
    ]
    candidate_rows = [
        study.stage_candidate_score_row(
            {"candidate_id": "STG_aileron_normal_s0p4", "surface_axis": "aileron", "alpha_regime": "normal", "candidate_scale": 0.4},
            [
                {
                    "replay_status": "ok",
                    "peak_p_rad_s_residual_actual_minus_sim": 0.8,
                    "final_phi_residual_actual_minus_sim_deg": 5.0,
                    "dx_residual_actual_minus_sim_m": 0.3,
                    "dy_residual_actual_minus_sim_m": 0.2,
                    "altitude_loss_residual_actual_minus_sim_m": 0.1,
                }
            ],
            evidence_rows,
        ),
        study.stage_candidate_score_row(
            {"candidate_id": "STG_aileron_normal_s0p8", "surface_axis": "aileron", "alpha_regime": "normal", "candidate_scale": 0.8},
            [
                {
                    "replay_status": "ok",
                    "peak_p_rad_s_residual_actual_minus_sim": 0.2,
                    "final_phi_residual_actual_minus_sim_deg": 6.0,
                    "dx_residual_actual_minus_sim_m": 0.3,
                    "dy_residual_actual_minus_sim_m": 0.2,
                    "altitude_loss_residual_actual_minus_sim_m": 0.1,
                }
            ],
            evidence_rows,
        ),
    ]

    selected = study.select_stage_fit_rows(candidate_rows, evidence_rows)
    aileron_normal = next(row for row in selected if row["surface_axis"] == "aileron" and row["alpha_regime"] == "normal")

    assert aileron_normal["selected_scale"] == pytest.approx(0.8)
    assert aileron_normal["positive_count"] == 1
    assert aileron_normal["negative_count"] == 0
    assert aileron_normal["fit_split"] == "all"


def test_stage_zero_support_uses_1p0_prior() -> None:
    selected = study.select_stage_fit_rows([], [])
    rudder_normal = next(row for row in selected if row["surface_axis"] == "rudder" and row["alpha_regime"] == "normal")

    assert rudder_normal["selected_scale"] == pytest.approx(1.0)
    assert rudder_normal["fit_status"] == study.STAGE_ZERO_SUPPORT_STATUS
    assert rudder_normal["zero_support_policy"] == "absolute_scale_1p0_prior"


def test_stage_schedule_scales_control_mix_by_alpha_regime_without_changing_command() -> None:
    aircraft = build_nausicaa_glider()
    control_mix = np.asarray(aircraft.control_mix, dtype=float)
    active_scales = study.active_surface_effectiveness_scales_by_surface()
    schedule = study.active_stage_surface_schedule()
    schedule["aileron"]["normal"] = 1.0
    schedule["aileron"]["post_stall"] = 0.4
    normal_state = np.zeros(15)
    normal_state[6] = 4.0
    normal_state[8] = 0.1
    post_stall_state = np.zeros(15)
    post_stall_state[6] = 1.0
    post_stall_state[8] = 1.0
    command = np.asarray([0.2, -0.4, 0.6], dtype=float)

    normal_aircraft = study.aircraft_with_surface_effectiveness_schedule_for_state(aircraft, schedule, normal_state)
    post_stall_aircraft = study.aircraft_with_surface_effectiveness_schedule_for_state(aircraft, schedule, post_stall_state)
    unchanged_command = study.scaled_surface_command(command, None)

    assert np.asarray(normal_aircraft.control_mix)[:, AILERON].tolist() == pytest.approx(
        (control_mix[:, AILERON] * 1.0).tolist()
    )
    assert np.asarray(post_stall_aircraft.control_mix)[:, AILERON].tolist() == pytest.approx(
        (control_mix[:, AILERON] * 0.4).tolist()
    )
    assert np.asarray(normal_aircraft.control_mix)[:, ELEVATOR].tolist() == pytest.approx(
        (control_mix[:, ELEVATOR] * active_scales["elevator"]).tolist()
    )
    assert unchanged_command.tolist() == pytest.approx(command.tolist())


def test_combined_stage_schedule_contains_all_surface_regime_entries() -> None:
    rows = [
        {
            "surface_axis": surface,
            "alpha_regime": regime,
            "selected_scale": 0.5,
        }
        for surface in ("aileron", "elevator", "rudder")
        for regime in study.SURFACE_AERO_ALPHA_REGIMES
    ]

    schedule = study.schedule_from_stage_fit_rows(rows)

    assert set(schedule) == {"aileron", "elevator", "rudder"}
    assert all(set(regimes) == set(study.SURFACE_AERO_ALPHA_REGIMES) for regimes in schedule.values())
    assert all(value == pytest.approx(0.5) for regimes in schedule.values() for value in regimes.values())


def test_constrained_stage_candidates_are_monotonic_and_complete() -> None:
    assert len(study.CONSTRAINED_STAGE_CANDIDATES) == 8
    for schedule in study.CONSTRAINED_STAGE_CANDIDATES.values():
        assert set(schedule) == {"aileron", "elevator", "rudder"}
        for regimes in schedule.values():
            assert set(regimes) == set(study.SURFACE_AERO_ALPHA_REGIMES)
            assert regimes["normal"] >= regimes["transition"] >= regimes["post_stall"]


def test_constrained_stage_summary_selects_lowest_joint_score() -> None:
    baseline = {
        "candidate_id": "C0_frozen_neutral",
        "split": "all",
        "surface_axis": "all",
        "replay_count": 2,
        "dx_mae_m": 0.2,
        "dy_mae_m": 0.2,
        "altitude_loss_mae_m": 0.2,
        "final_phi_mae_deg": 6.0,
        "final_theta_mae_deg": 6.0,
        "final_psi_mae_deg": 6.0,
        "primary_antisym_residual": 0.2,
    }
    rows = [baseline]
    for index, candidate_id in enumerate(study.CONSTRAINED_STAGE_CANDIDATES):
        rows.append(
            {
                **baseline,
                "candidate_id": candidate_id,
                "dx_mae_m": 0.1 + 0.01 * index,
                "dy_mae_m": 0.1 + 0.01 * index,
                "altitude_loss_mae_m": 0.1 + 0.01 * index,
                "primary_antisym_residual": 0.1 + 0.01 * index,
            }
        )

    summary = study.constrained_stage_candidate_summary_rows(rows)
    selected = [row for row in summary if row["selected"]]

    assert len(selected) == 1
    assert selected[0]["candidate_id"] == next(iter(study.CONSTRAINED_STAGE_CANDIDATES))
    assert selected[0]["accepted"] is True


def test_surface_scale_gate_rejects_aileron_when_primary_residual_worsens() -> None:
    candidate_id = study.surface_scale_candidate_id("aileron", 0.75)
    rows = synthetic_surface_replay_summary(candidate_id, primary=2.1, same_attitude=4.5)

    gate = study.surface_scale_promotion_gate_rows(
        rows,
        {candidate_id: {"aileron": 0.75}},
        study.active_surface_effectiveness_scales_by_surface(),
    )

    assert gate[0]["surface_axis"] == "aileron"
    assert gate[0]["primary_improved"] is False
    assert gate[0]["accepted"] is False
    assert "primary_response_not_improved" in gate[0]["promotion_gate_status"]


def test_surface_scale_gate_accepts_aileron_when_response_improves_and_limits_hold() -> None:
    candidate_id = study.surface_scale_candidate_id("aileron", 0.75)
    rows = synthetic_surface_replay_summary(candidate_id, primary=1.5, same_attitude=4.5)

    gate = study.surface_scale_promotion_gate_rows(
        rows,
        {candidate_id: {"aileron": 0.75}},
        study.active_surface_effectiveness_scales_by_surface(),
    )
    selected = study.selected_surface_scales_from_gate(gate, study.active_surface_effectiveness_scales_by_surface())

    assert gate[0]["primary_improved"] is True
    assert gate[0]["same_attitude_non_regression"] is True
    assert gate[0]["trajectory_gate"] is True
    assert gate[0]["cross_axis_gate"] is True
    assert gate[0]["accepted"] is True
    assert selected["aileron"] == pytest.approx(0.75)


def test_pairwise_surface_gain_fit_uses_confidence_weighted_train_pairs() -> None:
    rows = [
        pairwise_effectiveness_row("confidence_weighted_train", "elevator", actual=0.5, sim=1.0),
        pairwise_effectiveness_row("train", "elevator", actual=0.5, sim=1.0),
        pairwise_effectiveness_row("heldout", "elevator", actual=0.6, sim=1.0),
    ]

    fit_rows = study.pairwise_surface_gain_fit(rows)
    gains = study.pairwise_surface_gain_values(fit_rows)
    candidates = study.pairwise_gain_candidate_gains(gains)
    elevator_row = next(row for row in fit_rows if row["surface_axis"] == "elevator")

    assert elevator_row["raw_gain"] == pytest.approx(0.5)
    assert elevator_row["bounded_gain"] == pytest.approx(0.6)
    assert elevator_row["heldout_improved"] is True
    assert candidates["P2_pairwise_elevator_gain"] == {"elevator": pytest.approx(0.6)}
    assert candidates["P4_pairwise_all_surface_gains"] == {"elevator": pytest.approx(0.6)}


def test_control_surface_replay_is_hardcoded_to_eight_workers() -> None:
    source = Path(study.__file__).read_text(encoding="utf-8")

    assert study.DEFAULT_WORKERS == 8
    assert study.DEFAULT_MAX_WORKERS == 8
    assert "ProcessPoolExecutor(max_workers=worker_count)" in source
    assert "worker_count = min(DEFAULT_WORKERS, DEFAULT_MAX_WORKERS)" in source


def test_active_surface_effectiveness_promotes_conservative_regime_schedule() -> None:
    glider = build_nausicaa_glider()
    control_mix = np.asarray(glider.control_mix, dtype=float)

    assert active_calibration.CONTROL_SURFACE_EFFECTIVENESS_MODEL == "alpha_regime_scheduled_v1"
    assert np.allclose(
        np.asarray(active_calibration.CONTROL_SURFACE_EFFECTIVENESS_SCHEDULE, dtype=float),
        np.asarray(
            [
                [0.85, 0.75, 0.85],
                [0.55, 0.55, 0.55],
                [0.45, 0.45, 0.40],
            ],
            dtype=float,
        ),
    )
    assert active_calibration.DELTA_A_AERO_EFFECTIVENESS_SCALE == pytest.approx(0.65)
    assert active_calibration.DELTA_E_AERO_EFFECTIVENESS_SCALE == pytest.approx(0.70)
    assert active_calibration.DELTA_R_AERO_EFFECTIVENESS_SCALE == pytest.approx(0.45)
    assert np.max(np.abs(control_mix[:, AILERON])) == pytest.approx(1.0)
    assert np.max(np.abs(control_mix[:, ELEVATOR])) == pytest.approx(1.0)
    assert np.max(np.abs(control_mix[:, RUDDER])) == pytest.approx(1.0)
    assert np.allclose(
        np.asarray(glider.control_effectiveness_regime_scales, dtype=float),
        np.asarray(active_calibration.CONTROL_SURFACE_EFFECTIVENESS_SCHEDULE, dtype=float),
    )


def test_alpha_regime_candidates_share_rudder_post_stall_with_transition() -> None:
    families = study.surface_aero_coupling_candidate_coefficients(
        {
            "Cl_delta_a_residual@post_stall": 0.1,
            "Cm_delta_e_residual@post_stall": -0.2,
            "Cn_delta_r_residual@transition": 0.3,
        }
    )
    state = np.zeros(15)
    state[6] = 1.0
    state[8] = 1.0

    assert "Cn_delta_r_residual@post_stall" not in families["C6_alpha_regime_primary_derivatives"]
    assert study.alpha_regime_from_state(state) == "post_stall"
    assert study.surface_aero_coefficient(
        families["C6_alpha_regime_primary_derivatives"],
        "Cn_delta_r_residual",
        state,
    ) == pytest.approx(0.3)


def test_regime_ladder_error_summary_reports_surface_regime_ladder() -> None:
    replay_rows = [
        {
            "replay_status": "ok",
            "split": "heldout",
            "surface_axis": "aileron",
            "command_abs": 0.2,
            "command_value": 0.2,
            "actual_max_abs_alpha_deg": 15.0,
            "actual_alpha_gt_20_s": 0.0,
            "actual_alpha_gt_30_s": 0.0,
            "dx_residual_actual_minus_sim_m": 0.2,
            "dy_residual_actual_minus_sim_m": -0.4,
            "altitude_loss_residual_actual_minus_sim_m": 0.1,
            "final_phi_residual_actual_minus_sim_deg": -5.0,
            "final_theta_residual_actual_minus_sim_deg": 2.0,
            "final_psi_residual_actual_minus_sim_deg": 3.0,
            "actual_peak_p_rad_s": 4.0,
            "sim_peak_p_rad_s": 3.0,
        },
        {
            "replay_status": "ok",
            "split": "heldout",
            "surface_axis": "aileron",
            "command_abs": 0.2,
            "command_value": -0.2,
            "actual_max_abs_alpha_deg": 16.0,
            "actual_alpha_gt_20_s": 0.0,
            "actual_alpha_gt_30_s": 0.0,
            "dx_residual_actual_minus_sim_m": -0.4,
            "dy_residual_actual_minus_sim_m": 0.2,
            "altitude_loss_residual_actual_minus_sim_m": -0.3,
            "final_phi_residual_actual_minus_sim_deg": 7.0,
            "final_theta_residual_actual_minus_sim_deg": -4.0,
            "final_psi_residual_actual_minus_sim_deg": -1.0,
            "actual_peak_p_rad_s": -2.0,
            "sim_peak_p_rad_s": -1.0,
        },
    ]

    rows = study.regime_ladder_error_summary({"C0_frozen_neutral": replay_rows})
    row = next(
        item
        for item in rows
        if item["split"] == "heldout"
        and item["surface_axis"] == "aileron"
        and item["alpha_regime"] == "transition"
        and item["command_abs"] == pytest.approx(0.2)
    )

    assert row["replay_count"] == 2
    assert row["positive_count"] == 1
    assert row["negative_count"] == 1
    assert row["dx_mae_m"] == pytest.approx(0.3)
    assert row["dy_mae_m"] == pytest.approx(0.3)
    assert row["altitude_loss_mae_m"] == pytest.approx(0.2)
    assert row["final_phi_mae_deg"] == pytest.approx(6.0)
    assert row["primary_metric"] == "peak_p_rad_s"
    assert row["primary_antisym_residual"] == pytest.approx(1.0)


def test_short_response_window_returns_nan_metrics_without_crashing() -> None:
    rows = [{"t_s": "0.0", "x_w": "0", "y_w": "0", "z_w": "1", "phi": "0", "theta": "0", "psi": "0", "u": "5", "v": "0", "w": "0", "p": "0", "q": "0", "r": "0"}]

    metrics = study.response_metrics_from_rows(rows, 0.15, 0.2)

    assert math.isnan(metrics["peak_p_rad_s"])


def test_replay_csv_fields_are_unique_for_standard_csv_readers() -> None:
    assert len(study.REPLAY_FIELDS) == len(set(study.REPLAY_FIELDS))


def test_optional_surface_fit_is_diagnostic_and_non_promoting() -> None:
    effectiveness_rows = [
        {
            "split": "train",
            "surface_axis": "aileron",
            "metric": "peak_p_rad_s",
            "actual_antisymmetric_response": 2.0,
            "sim_antisymmetric_response": 1.0,
        },
        {
            "split": "heldout",
            "surface_axis": "aileron",
            "metric": "peak_p_rad_s",
            "positive_count": 1,
            "negative_count": 1,
            "actual_antisymmetric_response": 2.0,
            "sim_antisymmetric_response": 1.0,
        },
    ]

    before = active_calibration.CALIBRATION_ID
    candidates, heldout = study.optional_surface_fit_diagnostics(effectiveness_rows, run_diagnostics=True)

    assert active_calibration.CALIBRATION_ID == before
    assert {row["candidate_id"] for row in candidates} == {
        "S0_frozen_neutral",
        "S1_surface_effectiveness_scales",
        "S2_scales_plus_neutral_biases",
    }
    assert all(row["promoted"] is False for row in candidates)
    assert heldout[0]["candidate_id"] == "S1_surface_effectiveness_scales"
    assert heldout[0]["promotion_gate_status"].startswith("not_promoted")


def write_throw(tmp_path: Path, dataset: str, command_axis: str, command_value: float, *, valid: bool) -> Path:
    throw_dir = tmp_path / dataset / "20260605_000000" / case_storage_id(command_axis, command_value) / "v001"
    (throw_dir / "manifests").mkdir(parents=True)
    (throw_dir / "metrics").mkdir(parents=True)
    manifest = {
        "case_storage_id": case_storage_id(command_axis, command_value),
        "calibration_case": {
            "case_id": f"case_{command_axis}_{command_value}",
            "case_name": "test",
            "command_axis": command_axis,
            "command_value": command_value,
            "pulse_start_s": 0.15,
            "pulse_duration_s": 60.0,
        },
        "config": {"actuator_tau_s": [0.06, 0.06, 0.06], "controller_mode": "open_loop_neutral"},
    }
    summary = {
        "case_id": manifest["calibration_case"]["case_id"],
        "case_storage_id": manifest["case_storage_id"],
        "valid_throw": valid,
        "completed": valid,
        "termination_reason": "exit_gate_floor",
        "launch_speed_m_s": 5.0,
    }
    (throw_dir / "manifests" / "glider_calibration_throw_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (throw_dir / "manifests" / "glider_calibration_throw_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    write_csv(
        throw_dir / "metrics" / "command_schedule.csv",
        [
            command_row(0.0, command_axis, 0.0, False),
            command_row(0.15, command_axis, command_value, True),
            command_row(0.4, command_axis, command_value, True),
        ],
    )
    write_csv(
        throw_dir / "metrics" / "state_samples.csv",
        [
            state_row(0.0, x=1.0),
            state_row(0.15, x=1.7),
            state_row(0.4, x=2.8),
            state_row(0.85, x=4.8),
        ],
    )
    return throw_dir


def command_row(t_s: float, command_axis: str, command_value: float, pulse_active: bool) -> dict[str, object]:
    values = {"delta_a": 0.0, "delta_e": 0.0, "delta_r": 0.0}
    values[command_axis] = command_value
    return {
        "t_s": t_s,
        "pulse_active": pulse_active,
        "delta_a_cmd_norm": values["delta_a"],
        "delta_e_cmd_norm": values["delta_e"],
        "delta_r_cmd_norm": values["delta_r"],
    }


def pairwise_effectiveness_row(split: str, surface: str, *, actual: float, sim: float) -> dict[str, object]:
    return {
        "split": split,
        "surface_axis": surface,
        "command_abs": 0.4,
        "metric": {"aileron": "peak_p_rad_s", "elevator": "peak_q_rad_s", "rudder": "peak_r_rad_s"}[surface],
        "positive_count": 1,
        "negative_count": 1,
        "actual_antisymmetric_response": actual,
        "sim_antisymmetric_response": sim,
    }


def synthetic_surface_replay_summary(candidate_id: str, *, primary: float, same_attitude: float) -> list[dict[str, object]]:
    baseline = {
        "candidate_id": "C0_frozen_neutral",
        "split": "heldout",
        "surface_axis": "aileron",
        "replay_count": 4,
        "dx_mae_m": 0.20,
        "dy_mae_m": 0.30,
        "altitude_loss_mae_m": 0.40,
        "final_phi_mae_deg": 5.0,
        "final_theta_mae_deg": 3.0,
        "final_psi_mae_deg": 4.0,
        "primary_antisym_residual": 2.0,
    }
    candidate = {
        "candidate_id": candidate_id,
        "split": "heldout",
        "surface_axis": "aileron",
        "replay_count": 4,
        "dx_mae_m": 0.22,
        "dy_mae_m": 0.31,
        "altitude_loss_mae_m": 0.41,
        "final_phi_mae_deg": same_attitude,
        "final_theta_mae_deg": 4.0,
        "final_psi_mae_deg": 5.0,
        "primary_antisym_residual": primary,
    }
    return [baseline, candidate]


def state_row(t_s: float, *, x: float) -> dict[str, object]:
    return {
        "t_s": t_s,
        "x_w": x,
        "y_w": 2.0,
        "z_w": 1.5 - 0.1 * t_s,
        "phi": 0.0,
        "theta": 0.0,
        "psi": 0.0,
        "u": 5.0,
        "v": 0.0,
        "w": 0.0,
        "p": 0.0,
        "q": 0.0,
        "r": 0.0,
        "delta_a": 0.0,
        "delta_e": 0.0,
        "delta_r": 0.0,
        "estimator_rate_confidence": 0.95,
        "estimator_spike_rejected": False,
        "estimator_body_rate_limited": False,
    }


def case_storage_id(command_axis: str, command_value: float) -> str:
    prefix = {"delta_a": "a", "delta_e": "e", "delta_r": "r"}[command_axis]
    sign = "p" if command_value >= 0.0 else "n"
    value = int(round(abs(command_value) * 10))
    return f"c1_{prefix}_{sign}{value:02d}"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
