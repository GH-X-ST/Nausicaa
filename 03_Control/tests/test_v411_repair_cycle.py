from __future__ import annotations

import json
from pathlib import Path

from directional_residual_lift_belief import (
    DirectionalResidualObservation,
    initial_directional_residual_lift_belief,
    query_directional_residual_lift_features,
    update_directional_residual_lift_belief,
)
from episode_selector import select_compact_representative
from primitive_timing_contract import (
    CONTROLLER_INPUT_SLOTS_PER_PRIMITIVE,
    CONTROLLER_INPUT_UPDATE_PERIOD_S,
    PRIMITIVE_FINITE_HORIZON_S,
    PRIMITIVE_TIMING_CONTRACT_VERSION,
    primitive_step_count,
)
from primitive_variant_registry import _variant_id
from run_post_w3_cluster_merge import run_post_w3_cluster_merge
from run_post_w3_library_size_study import LIBRARY_SIZE_CASE_IDS
from run_repeated_launch_learning_curve import HISTORY_LENGTHS
from run_v411_source_audit import (
    SourceAuditConfig,
    build_control_inventory,
    classify_control_file,
    discover_superseded_result_roots,
    inspect_required_docs,
    run_v411_source_audit,
    write_diagnostic_not_passed_archive,
)


def test_v411_docs_gate_inventory_and_audit_are_ready() -> None:
    docs = inspect_required_docs(Path("docs"))
    assert docs
    assert all(row["readable"] for row in docs)
    assert {row["doc_name"] for row in docs} >= {
        "Glider_Control_Project_Plan.md",
        "Python Coding to CODEX.txt",
        "PR.txt",
    }

    assert classify_control_file("03_Primitives/prim_cat.py") == "active_repair_cycle"
    assert classify_control_file("04_Scenarios/run_v410_source_audit.py") == "retired_not_active"
    inventory = build_control_inventory(Path("03_Control"))
    assert any(row["classification"] == "active_repair_cycle" for row in inventory)

    result = run_v411_source_audit(SourceAuditConfig(dry_run=True))
    assert result["status"] == "ready"
    assert result["blockers"] == []


def test_v411_timing_contract_exact_five_slots_and_variant_id_sensitivity() -> None:
    assert PRIMITIVE_FINITE_HORIZON_S == 0.100
    assert CONTROLLER_INPUT_UPDATE_PERIOD_S == 0.020
    assert CONTROLLER_INPUT_SLOTS_PER_PRIMITIVE == 5
    assert primitive_step_count() == 5

    common = {
        "primitive_id": "glide",
        "entry_role": "launch_capable",
        "reference_state_vector": "[0,0,0]",
        "reference_command_vector": "[0,0,0]",
        "controller_id": "ctrl",
        "linearisation_id": "lin",
        "q_json": "{}",
        "r_json": "{}",
        "gain_checksum": "k",
        "timing_augmentation_type": "aug",
        "timing_design_version": "design",
        "sample_time_s": CONTROLLER_INPUT_UPDATE_PERIOD_S,
        "latency_case": "nominal",
        "command_delay_steps": 1,
        "predictor_horizon_steps": 1,
        "augmented_a_checksum": "a",
        "augmented_b_checksum": "b",
        "augmented_q_json": "{}",
        "augmented_r_json": "{}",
        "augmented_gain_checksum": "g",
        "exit_check_version": "exit",
    }
    v411_id = _variant_id(
        **common,
        finite_horizon_s=0.100,
        controller_input_slots_per_primitive=5,
        controller_input_update_period_s=0.020,
        primitive_timing_contract_version=PRIMITIVE_TIMING_CONTRACT_VERSION,
    )
    old_id = _variant_id(
        **common,
        finite_horizon_s=0.800,
        controller_input_slots_per_primitive=40,
        controller_input_update_period_s=0.020,
        primitive_timing_contract_version="legacy",
    )
    assert v411_id != old_id


def test_v411_dynamic_archive_marks_old_roots_diagnostic_not_passed(tmp_path: Path) -> None:
    old_root = tmp_path / "results" / "w01_dense" / "001"
    (old_root / "manifests").mkdir(parents=True)
    (old_root / "metrics").mkdir()
    (old_root / "manifests" / "run_manifest.json").write_text(
        json.dumps({"project_title_version": "LQR-Stabilised Contextual Primitive v4.10"}, indent=2),
        encoding="ascii",
    )
    (old_root / "metrics" / "primitive_variant_registry.csv").write_text(
        "primitive_variant_id,finite_horizon_s\nold,0.8\n",
        encoding="ascii",
    )

    roots = discover_superseded_result_roots(tmp_path / "results")
    assert len(roots) == 1
    assert roots[0]["diagnostic_archive_status"] == "diagnostic_not_passed"
    manifest = write_diagnostic_not_passed_archive(
        archive_root=tmp_path / "archive" / "diagnostic_not_passed_v410" / "001",
        docs=[],
        inventory=[],
        superseded_roots=roots,
    )
    archive_path = Path(manifest["manifest_path"])
    assert archive_path.is_file()
    assert "diagnostic_not_passed" in archive_path.read_text(encoding="ascii")


def test_v411_directional_memory_and_safe_exploration_after_filtering() -> None:
    belief = initial_directional_residual_lift_belief()
    belief = update_directional_residual_lift_belief(
        belief,
        DirectionalResidualObservation(
            x_m=0.0,
            y_m=0.0,
            z_m=1.5,
            direction_rad=0.0,
            lift_residual_m_s=0.2,
            energy_residual_m=0.1,
            dwell_residual_s=0.05,
        ),
    )
    features = query_directional_residual_lift_features(
        belief,
        x_m=0.0,
        y_m=0.0,
        z_m=1.5,
        direction_rad=0.0,
    )
    assert features["belief_local_lift_residual_m_s"] > 0.0
    assert features["belief_direction_bin"] == 0

    timing_payload = {
        "finite_horizon_s": 0.1,
        "controller_input_slots_per_primitive": 5,
        "controller_input_update_period_s": 0.02,
        "primitive_timing_contract_version": PRIMITIVE_TIMING_CONTRACT_VERSION,
    }
    representatives = [
        {
            "compact_library_id": "good",
            "primitive_variant_id": "good",
            "primitive_id": "glide",
            "entry_role": "launch_capable",
            "controller_id": "ctrl",
            "K_gain_checksum": "k",
            "augmented_A_checksum": "a",
            "augmented_B_checksum": "b",
            "augmented_gain_checksum": "g",
            "library_size_case_id": "balanced_cluster",
            **timing_payload,
        },
        {
            "compact_library_id": "bad",
            "primitive_variant_id": "bad",
            "primitive_id": "lift_entry",
            "entry_role": "inflight_only",
            "controller_id": "ctrl",
            "K_gain_checksum": "k",
            "augmented_A_checksum": "a",
            "augmented_B_checksum": "b",
            "augmented_gain_checksum": "g",
            "library_size_case_id": "balanced_cluster",
            **timing_payload,
        },
    ]
    outcome_rows = {
        "good": {"continuation_probability": 0.8, "hard_failure_risk": 0.1},
        "bad": {"continuation_probability": 0.9, "hard_failure_risk": 0.1},
    }
    selected, rows = select_compact_representative(
        representatives=representatives,
        outcome_rows_by_variant_id=outcome_rows,
        context={
            "start_state_family": "launch_gate",
            "wall_margin_m": 0.5,
            "floor_margin_m": 0.5,
            "ceiling_margin_m": 0.5,
            "speed_margin_m_s": 1.0,
            "latency_case": "nominal",
            "history_length": 5,
            "library_size_case_id": "balanced_cluster",
        },
        governor_mode="continuation_mode",
        belief_features=features,
    )
    assert selected is not None
    viable = {row["primitive_variant_id"]: row for row in rows}
    assert viable["good"]["safe_exploration_status"] == "applied_after_viability_filter"
    assert viable["good"]["exploration_score_component"] > 0.0
    assert viable["bad"]["safe_exploration_status"] == "not_applied_rejected_before_exploration"
    assert viable["bad"]["exploration_score_component"] == 0.0


def test_v411_case_ids_histories_and_retired_gate(tmp_path: Path) -> None:
    assert LIBRARY_SIZE_CASE_IDS == (
        "heavy_cluster",
        "balanced_cluster",
        "light_cluster",
        "no_cluster_no_merge",
    )
    assert HISTORY_LENGTHS == (0, 5, 10, 20, 50, 100)
    result = run_post_w3_cluster_merge(
        input_root=tmp_path / "missing_w3",
        output_root=tmp_path / "post_w3_cluster",
        run_id=1,
    )
    assert result["status"] == "blocked"
    assert result["blocked_reason"] == "retired_diagnostic_requires_explicit_allow_retired_diagnostic"

