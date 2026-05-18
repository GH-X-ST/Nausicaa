from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTROL_ROOT = REPO_ROOT / "03_Control"
RESULT_ROOT = CONTROL_ROOT / "05_Results" / "09_primitive_library" / "001"


def _load_outputs() -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest_path = RESULT_ROOT / "manifests" / "primitive_library_manifest_s001.json"
    evidence_path = RESULT_ROOT / "metrics" / "primitive_evidence_library_s001.csv"
    library_summary_path = RESULT_ROOT / "metrics" / "primitive_library_summary_s001.csv"
    group_summary_path = RESULT_ROOT / "metrics" / "primitive_envelope_group_summary_s001.csv"

    assert manifest_path.exists()
    assert evidence_path.exists()
    assert library_summary_path.exists()
    assert group_summary_path.exists()

    return (
        json.loads(manifest_path.read_text(encoding="ascii")),
        pd.read_csv(evidence_path),
        pd.read_csv(library_summary_path),
        pd.read_csv(group_summary_path),
    )


def test_primitive_library_outputs_and_manifest_flags() -> None:
    manifest, evidence, library_summary, group_summary = _load_outputs()
    report_path = RESULT_ROOT / "reports" / "primitive_library_report_s001.md"

    assert report_path.exists()
    assert manifest["central_research_question"] == "widen primitive envelopes before library growth"
    assert manifest["dry_air_agile_turn_recovery_loop_closed"] is True
    assert manifest["preserved_boundary_reference"] is True
    assert manifest["ocp_implemented"] is False
    assert manifest["tvlqr_implemented"] is False
    assert manifest["governor_implemented"] is False
    assert manifest["outer_loop_implemented"] is False
    assert manifest["real_flight_validation_claim"] is False
    assert manifest["high_incidence_validation_claim"] is False
    assert manifest["w1_complete"] in (True, False)
    assert manifest["w2_complete"] in (True, False)

    assert not evidence.empty
    assert not library_summary.empty
    assert not group_summary.empty


def test_evidence_schema_start_conditions_and_growth_fields() -> None:
    _, evidence, _, group_summary = _load_outputs()
    expected_families = {
        "glide",
        "recovery",
        "mild_bank",
        "canyon_steep_bank",
        "wingover_lite",
        "bank_yaw_energy_retaining",
    }
    growth_fields = {
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

    assert set(evidence["family"].unique()) == expected_families
    assert growth_fields.issubset(evidence.columns)
    assert metric_fields.issubset(evidence.columns)

    starts_by_family = evidence.groupby("family")["start_condition"].unique().to_dict()
    for family in expected_families:
        assert {"favourable", "mid_arena"}.issubset(set(starts_by_family[family]))

    assert set(group_summary["group_status"]).issubset(
        {
            "widening_existing_envelope",
            "requires_library_growth",
            "remaining_boundary_evidence",
            "not_evaluated_model_unavailable",
        }
    )


def test_wind_fidelity_rows_and_missing_model_semantics() -> None:
    manifest, evidence, _, _ = _load_outputs()
    w0 = evidence[evidence["wind_fidelity"] == "W0"]
    w1_w2 = evidence[evidence["wind_fidelity"].isin(["W1", "W2"])]

    assert not w0.empty
    assert (w0["evaluation_status"] == "evaluated").all()
    assert set(w1_w2["evaluation_status"].unique()).issubset(
        {"evaluated", "not_evaluated_model_missing", "model_unavailable"}
    )

    missing = evidence[evidence["evaluation_status"] != "evaluated"]
    if not missing.empty:
        assert (missing["candidate_class"] != "boundary_evidence").all()
        assert set(missing["candidate_class"].unique()) == {"not_evaluated"}
        assert manifest["w1_complete"] is False or manifest["w2_complete"] is False


def test_representative_logs_preserve_command_bridge_columns() -> None:
    manifest, _, _, _ = _load_outputs()
    output_files = manifest["output_files"]
    command_keys = [key for key in output_files if key.endswith("_commands_csv")]

    assert command_keys
    command_path = REPO_ROOT / output_files[command_keys[0]]
    commands = pd.read_csv(command_path)

    assert "u_norm_requested_delta_a_norm" in commands.columns
    assert "u_norm_applied_delta_a_norm" in commands.columns
    assert "delta_cmd_rad_delta_a_cmd" in commands.columns
    assert "delta_cmd_rad_delta_e_cmd" in commands.columns
    assert "delta_cmd_rad_delta_r_cmd" in commands.columns

    for relative_path in output_files.values():
        assert not Path(relative_path).is_absolute()
