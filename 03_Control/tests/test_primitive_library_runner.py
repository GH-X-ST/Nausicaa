from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import run_primitive_library_pass as runner
from latency import actuator_tau_for_case, format_actuator_tau_s, latency_case_config
from primitive_library_schema import PrimitiveLibraryConfig


@pytest.fixture()
def runner_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    result_root = tmp_path / "results"
    archive = result_root / "07_aggressive_reversal_ocp" / "002"
    archive.mkdir(parents=True)
    (archive / "boundary_reference.txt").write_text("preserved\n", encoding="ascii")
    monkeypatch.setattr(runner, "RESULT_ROOT", result_root)
    monkeypatch.setattr(runner, "_obsolete_files", lambda: tuple())
    monkeypatch.setattr(runner, "_obsolete_result_dirs", lambda: tuple())
    monkeypatch.setattr(runner, "_negative_grep", lambda: (1, ""))

    paths = runner.run_primitive_library_pass(
        run_id=12,
        targets_deg=(15.0,),
        wind_fidelities=("W0",),
        updraft_configs=("none",),
        latency_case="nominal",
    )
    manifest = json.loads(paths["manifest"].read_text(encoding="ascii"))
    evidence = pd.read_csv(paths["library_csv"])
    library_summary = pd.read_csv(paths["summary_csv"])
    group_summary = pd.read_csv(paths["group_summary_csv"])
    coverage = pd.read_csv(paths["coverage_summary_csv"])
    return paths, manifest, evidence, library_summary, group_summary, coverage


def test_primitive_library_outputs_and_manifest_flags(runner_outputs) -> None:
    paths, manifest, evidence, library_summary, group_summary, coverage = runner_outputs

    assert paths["report"].exists()
    assert paths["coverage_summary_csv"].exists()
    assert manifest["central_research_question"] == "widen primitive envelopes before library growth"
    assert manifest["classification_semantics_fixed"] is True
    assert manifest["w1_w2_dry_recoverable_not_boundary_by_default"] is True
    assert manifest["library_growth_trigger_is_group_level"] is True
    assert manifest["entry_envelope_failures_are_governor_rejections"] is True
    assert manifest["evidence_source_field_present"] is True
    assert manifest["dry_air_agile_turn_recovery_loop_closed"] is True
    assert manifest["preserved_boundary_reference"] is True
    assert manifest["ocp_implemented"] is False
    assert manifest["tvlqr_implemented"] is False
    assert manifest["governor_implemented"] is False
    assert manifest["outer_loop_implemented"] is False
    assert manifest["real_flight_validation_claim"] is False
    assert manifest["high_incidence_validation_claim"] is False
    assert manifest["latency_case"] == "nominal"
    assert manifest["latency_acceptance_scope"] == (
        "command_path_nominal_no_feedback_controller"
    )
    assert manifest["latency_pass_label_policy"] == (
        "computed_per_evidence_row_after_final_primitive_acceptance"
    )
    assert manifest["state_feedback_delay_applied"] is False
    assert manifest["closed_loop_delayed_state_feedback_applied"] is False
    nominal_config = latency_case_config("nominal")
    assert manifest["actuator_tau_s"] == format_actuator_tau_s(
        actuator_tau_for_case(nominal_config)
    )
    assert manifest["actuator_t50_s"] == pytest.approx(nominal_config.actuator_t50_s)
    assert manifest["actuator_t90_s"] == pytest.approx(nominal_config.actuator_t90_s)

    assert not evidence.empty
    assert not library_summary.empty
    assert not group_summary.empty
    assert not coverage.empty


def test_evidence_schema_start_conditions_growth_and_latency_fields(runner_outputs) -> None:
    _, _, evidence, _, group_summary, _ = runner_outputs
    expected_families = {
        "glide",
        "recovery",
        "mild_bank",
        "canyon_steep_bank",
        "wingover_lite",
        "bank_yaw_energy_retaining",
    }
    growth_fields = {
        "evidence_source",
        "recovery_basis",
        "entry_envelope_status",
        "envelope_status",
        "coverage_status",
        "parent_primitive_id",
        "variant_id",
        "envelope_group_id",
        "within_existing_envelope",
        "nearest_existing_primitive_id",
        "coverage_region_id",
        "marginal_coverage_gain",
        "library_growth_trigger",
        "growth_reason",
    }
    metric_fields = {
        "path_length_xy_m",
        "turn_footprint_proxy_m2",
        "entry_clearance_required_x_plus_m",
        "entry_clearance_required_y_plus_m",
        "floor_margin_required_m",
        "ceiling_margin_required_m",
        "margin_consumption_x_m",
        "margin_consumption_y_m",
        "margin_consumption_z_m",
    }
    latency_fields = {
        "latency_case",
        "actuator_tau_s",
        "latency_pass_label",
        "latency_acceptance_scope",
        "state_feedback_delay_applied",
        "command_delay_applied",
        "actuator_lag_applied",
    }

    assert set(evidence["family"].unique()) == expected_families
    assert growth_fields.issubset(evidence.columns)
    assert metric_fields.issubset(evidence.columns)
    assert latency_fields.issubset(evidence.columns)
    assert set(evidence["latency_case"].unique()) == {"nominal"}
    assert evidence["actuator_tau_s"].str.match(
        r"^\d+\.\d{9};\d+\.\d{9};\d+\.\d{9}$"
    ).all()

    starts_by_family = evidence.groupby("family")["start_condition"].unique().to_dict()
    for family in expected_families:
        assert {"favourable", "mid_arena"}.issubset(set(starts_by_family[family]))

    assert set(group_summary["group_status"]).issubset(
        {
            "widening_existing_envelope",
            "requires_library_growth",
            "outside_entry_envelope_governor_reject",
            "candidate_family_needs_refinement",
            "candidate_family_boundary",
            "not_evaluated_model_unavailable",
        }
    )
    assert set(evidence["evidence_source"].unique()) == {"deterministic_seed_replay"}
    assert evidence["library_growth_trigger"].astype(bool).sum() == 0


def test_wind_fidelity_rows_and_missing_model_semantics(runner_outputs) -> None:
    manifest, evidence = runner_outputs[1], runner_outputs[2]
    w0 = evidence[evidence["wind_fidelity"] == "W0"]

    assert not w0.empty
    assert (w0["evaluation_status"] == "evaluated").all()
    assert manifest["w1_complete"] is True
    assert manifest["w2_complete"] is True


def test_coverage_summary_and_report_record_latency_scope(runner_outputs) -> None:
    paths, manifest, evidence, _, _, coverage = runner_outputs
    report = paths["report"].read_text(encoding="ascii")

    assert not coverage.empty
    assert "coverage_status" in coverage.columns
    assert "library_growth_trigger" in coverage.columns
    assert coverage["library_growth_trigger"].astype(bool).sum() == 0
    assert manifest["baseline_run_001_diagnosis"]["baseline_available"] is False
    assert "Latency Replay Scope" in report
    assert "command_path_nominal_no_feedback_controller" in report
    assert "computed_per_evidence_row_after_final_primitive_acceptance" in report
    assert f"Actuator tau s: `{manifest['actuator_tau_s']}`" in report
    assert f"Actuator t50 s: `{manifest['actuator_t50_s']}`" in report
    assert f"Actuator t90 s: `{manifest['actuator_t90_s']}`" in report
    assert "coverage_status" in evidence.columns


def test_latency_manifest_fields_use_active_tau_semantics() -> None:
    default_fields = runner._latency_manifest_fields(PrimitiveLibraryConfig())
    active_tau = actuator_tau_for_case(latency_case_config("actuator_lag_only"))

    assert default_fields["latency_case"] == "actuator_lag_only"
    assert default_fields["actuator_tau_s"] == "0.060000000;0.060000000;0.060000000"
    assert default_fields["actuator_tau_s"] == format_actuator_tau_s(active_tau)
    assert default_fields["actuator_t50_s"] == pytest.approx(max(active_tau) * np.log(2.0))
    assert default_fields["actuator_t90_s"] == pytest.approx(max(active_tau) * np.log(10.0))
    assert default_fields["command_delay_applied"] is False
    assert default_fields["actuator_lag_applied"] is True

    for latency_case in ("nominal", "conservative"):
        config = latency_case_config(latency_case)
        fields = runner._latency_manifest_fields(
            PrimitiveLibraryConfig(latency_case=latency_case)
        )
        assert fields["actuator_tau_s"] == format_actuator_tau_s(
            actuator_tau_for_case(config)
        )
        assert fields["actuator_t50_s"] == pytest.approx(config.actuator_t50_s)
        assert fields["actuator_t90_s"] == pytest.approx(config.actuator_t90_s)

    none_fields = runner._latency_manifest_fields(
        PrimitiveLibraryConfig(latency_case="none")
    )
    assert none_fields["actuator_tau_s"] == "0.000000000;0.000000000;0.000000000"
    assert none_fields["actuator_t50_s"] == 0.0
    assert none_fields["actuator_t90_s"] == 0.0
    assert none_fields["actuator_lag_applied"] is False


def test_representative_logs_preserve_raw_command_bridge_columns(runner_outputs) -> None:
    _, manifest, _, _, _, _ = runner_outputs
    output_files = manifest["output_files"]
    command_keys = [key for key in output_files if key.endswith("_commands_csv")]

    assert command_keys
    command_path = Path(output_files[command_keys[0]])
    commands = pd.read_csv(command_path)

    assert "u_norm_requested_delta_a_norm" in commands.columns
    assert "u_norm_effective_target_delta_a_norm" in commands.columns
    assert "u_norm_effective_target_delta_e_norm" in commands.columns
    assert "u_norm_effective_target_delta_r_norm" in commands.columns
    assert "u_norm_applied_delta_a_norm" in commands.columns
    assert "delta_cmd_rad_delta_a_cmd" in commands.columns
    assert "delta_cmd_rad_delta_e_cmd" in commands.columns
    assert "delta_cmd_rad_delta_r_cmd" in commands.columns
