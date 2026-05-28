from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dense_archive_table_io import (
    TableManifest,
    load_table_manifest,
    read_table_partition,
    write_table_manifest,
    write_table_partition,
)
from lqr_linearisation import lqr_speed_bin_id
from prim_cat import ACTIVE_PRIMITIVE_IDS
from run_changed_case_validation import (
    HeldoutChangedCaseValidationConfig,
    R10_EXPECTED_FINAL_HELDOUT_LAUNCHES,
    R10_EXPECTED_HISTORY_LAUNCHES,
    ChangedCaseValidationConfig,
    run_changed_case_validation,
    run_heldout_changed_case_validation,
)
from run_outcome_model_build import OutcomeModelBuildConfig, run_outcome_model_build
from run_post_w3_cluster_merge import run_post_w3_cluster_merge
from run_post_w3_library_size_study import (
    LIBRARY_SIZE_CASE_IDS,
    PROJECT_TITLE_VERSION as POST_W3_PROJECT_TITLE_VERSION,
    PostW3LibrarySizeStudyConfig,
    _coverage_medoid_selection,
    run_post_w3_library_size_study,
)
from run_repeated_launch_learning_curve import (
    HISTORY_LENGTHS,
    POLICY_HISTORY_CONDITIONS,
    R9_EXPECTED_FINAL_HELDOUT_LAUNCHES,
    R9_EXPECTED_HISTORY_LAUNCHES,
    RepeatedLaunchValidationConfig,
    run_repeated_launch_learning_curve,
)
from run_w3_survival_analysis import W3SurvivalAnalysisConfig, run_w3_survival_analysis


def test_w3_analysis_separates_terminal_and_continuation_evidence(tmp_path: Path) -> None:
    w3_root = _write_tiny_w3_root(tmp_path / "w3_survival" / "013")

    result = run_w3_survival_analysis(W3SurvivalAnalysisConfig(input_root=w3_root))
    summary = pd.read_csv(w3_root / "metrics" / "w3_variant_survival_summary.csv")
    registry = json.loads((w3_root / "manifests" / "w3_survivor_registry.json").read_text(encoding="ascii"))

    assert result["status"] == "w3_survivors_available"
    assert registry["survivor_count"] == 11
    survived = summary[summary["primitive_variant_id"] == "primvar_glide_inflight"]
    terminal_safe = summary[summary["primitive_variant_id"] == "primvar_lift_terminal"]
    assert survived["w3_variant_status"].iloc[0] == "survived"
    assert terminal_safe["w3_variant_status"].iloc[0] == "survived"
    assert int(terminal_safe["incompatible_row_count"].iloc[0]) == 1
    assert int(terminal_safe["continuation_valid_count"].iloc[0]) == 0
    assert int(terminal_safe["episode_terminal_useful_count"].iloc[0]) == 2
    assert int(terminal_safe["transition_chain_compatible_count"].iloc[0]) == 2


def test_w3_analysis_blocks_when_launch_gate_entry_has_no_survivors(tmp_path: Path) -> None:
    w3_root = _write_tiny_w3_root(tmp_path / "w3_survival" / "013")
    table_path = w3_root / "tables" / "w3_survival_rows" / "c00000.csv.gz"
    frame = pd.read_csv(table_path)
    launch = frame["transition_entry_class"].astype(str) == "launch_gate"
    frame.loc[launch, "outcome_class"] = "failed"
    frame.loc[launch, "continuation_valid"] = False
    frame.loc[launch, "episode_terminal_useful"] = False
    frame.loc[launch, "boundary_use_class"] = "hard_failure"
    frame.loc[launch, "failure_label"] = "floor_violation"
    frame.loc[launch, "transition_exit_class"] = "hard_failure"
    frame.loc[launch, "transition_chain_compatible"] = False
    frame.to_csv(table_path, index=False, compression="gzip")

    result = run_w3_survival_analysis(W3SurvivalAnalysisConfig(input_root=w3_root))
    registry = json.loads((w3_root / "manifests" / "w3_survivor_registry.json").read_text(encoding="ascii"))

    assert result["status"] == "blocked_no_launch_gate_entry_w3_survivors"
    assert registry["survivor_count"] > 0
    assert registry["launch_gate_entry_survivor_count"] == 0
    assert set(registry["missing_launch_gate_entry_primitive_ids"]) == set(ACTIVE_PRIMITIVE_IDS)


def test_post_w3_library_size_study_writes_five_cases_without_mutation(tmp_path: Path) -> None:
    w3_root = _write_tiny_w3_root(tmp_path / "w3_survival" / "013")
    run_w3_survival_analysis(W3SurvivalAnalysisConfig(input_root=w3_root))

    result = run_post_w3_library_size_study(
        PostW3LibrarySizeStudyConfig(
            input_root=w3_root,
            output_root=tmp_path / "post_w3_library_size_study",
            run_id=1,
        )
    )
    run_root = Path(result["run_root"])
    summary = pd.read_csv(run_root / "metrics" / "library_size_case_summary.csv")
    library = json.loads((run_root / "manifests" / "no_cluster_no_merge_primitive_library.json").read_text(encoding="ascii"))
    representatives = library["representatives"]
    representative_ids = {row["primitive_variant_id"] for row in representatives}

    assert result["status"] == "complete"
    assert tuple(summary["library_size_case_id"]) == LIBRARY_SIZE_CASE_IDS
    assert library["library_size_human_label"] == "no-clustering/no-merging"
    assert library["entry_role_regime_separation_policy"] == "representatives_grouped_by_primitive_id_and_transition_entry_class_no_cross_entry_merge"
    assert library["selection_algorithm"] == "coverage_aware_behavior_qr_medoid_greedy_marginal"
    assert library["no_controller_mutation"] is True
    assert representative_ids == {
        "primvar_glide_inflight",
        "primvar_lift_terminal",
        "primvar_lift_survived",
        *(f"primvar_{primitive_id}_launch_gate" for primitive_id in ACTIVE_PRIMITIVE_IDS),
    }
    for representative in representatives:
        assert representative["w3_variant_status"] == "survived"
        assert representative["mutation_status"].startswith("references_existing_frozen_transition_object")
        assert representative["library_size_case_id"] == "no_cluster_no_merge"
        assert representative["selection_algorithm"] == "coverage_aware_behavior_qr_medoid_greedy_marginal"
    glide = [row for row in representatives if row["primitive_variant_id"] == "primvar_glide_inflight"][0]
    assert glide["controller_id"] == "ctrl_glide"
    assert glide["K_gain_checksum"] == "k_glide"
    availability = pd.read_csv(run_root / "metrics" / "launch_gate_candidate_availability.csv")
    assert set(availability["library_size_case_id"]) == set(LIBRARY_SIZE_CASE_IDS)
    assert (availability["launch_gate_entry_primitive_family_count"] == len(ACTIVE_PRIMITIVE_IDS)).all()
    assert (availability["missing_launch_gate_entry_primitive_ids"].fillna("") == "").all()
    for case_id in LIBRARY_SIZE_CASE_IDS:
        case_library = json.loads((run_root / "manifests" / f"{case_id}_primitive_library.json").read_text(encoding="ascii"))
        case_launch = {
            row["primitive_id"]
            for row in case_library["representatives"]
            if row["transition_entry_class"] == "launch_gate"
        }
        assert case_launch == set(ACTIVE_PRIMITIVE_IDS)


def test_post_w3_medoid_selection_keeps_existing_broad_coverage_variant() -> None:
    frame = pd.DataFrame(
        [
            {
                "primitive_variant_id": "narrow_high_average",
                "primitive_id": "glide",
                "entry_role": "transition_object",
                "transition_entry_class": "inflight_stable",
                "continuation_valid_rate": 0.95,
                "episode_terminal_useful_rate": 0.1,
                "hard_failure_rate": 0.01,
                "robustness_coverage_labels_json": '["env:single","env:four","start:inflight_nominal","active_fan_count:4"]',
                "robustness_coverage_rates_json": "[1.0,1.0,1.0,0.0]",
                "Q_weight_json": '{"q":1.0}',
                "R_weight_json": '{"r":1.0}',
                "reference_state_vector": "[0,0,0]",
                "reference_command_vector": "[0,0,0]",
            },
            {
                "primitive_variant_id": "broad_medoid",
                "primitive_id": "glide",
                "entry_role": "transition_object",
                "transition_entry_class": "inflight_stable",
                "continuation_valid_rate": 0.75,
                "episode_terminal_useful_rate": 0.1,
                "hard_failure_rate": 0.02,
                "robustness_coverage_labels_json": '["env:single","env:four","start:inflight_nominal","active_fan_count:4"]',
                "robustness_coverage_rates_json": "[0.7,0.7,0.7,0.7]",
                "Q_weight_json": '{"q":1.1}',
                "R_weight_json": '{"r":1.0}',
                "reference_state_vector": "[0,0,0]",
                "reference_command_vector": "[0,0,0]",
            },
            {
                "primitive_variant_id": "unsafe_high_coverage",
                "primitive_id": "glide",
                "entry_role": "transition_object",
                "transition_entry_class": "inflight_stable",
                "continuation_valid_rate": 1.0,
                "episode_terminal_useful_rate": 0.5,
                "hard_failure_rate": 0.90,
                "robustness_coverage_labels_json": '["env:single","env:four","start:inflight_nominal","active_fan_count:4"]',
                "robustness_coverage_rates_json": "[1.0,1.0,1.0,1.0]",
                "Q_weight_json": '{"q":1.0}',
                "R_weight_json": '{"r":1.0}',
                "reference_state_vector": "[0,0,0]",
                "reference_command_vector": "[0,0,0]",
            },
        ]
    )

    selected = _coverage_medoid_selection(frame, max_representatives=1, case_id="heavy_cluster")

    assert list(selected["primitive_variant_id"]) == ["broad_medoid"]
    assert selected["_selection_algorithm"].iloc[0] == "coverage_aware_behavior_qr_medoid_greedy_marginal"


def test_post_w3_medoid_selection_preserves_speed_bin_coverage_when_budget_allows() -> None:
    frame = pd.DataFrame(
        [
            {
                "primitive_variant_id": "strong_speed_5p0",
                "primitive_id": "glide",
                "entry_role": "transition_object",
                "transition_entry_class": "inflight_stable",
                "local_lqr_reference_speed_m_s": 5.0,
                "local_lqr_speed_bin_id": lqr_speed_bin_id(5.0),
                "continuation_valid_rate": 0.95,
                "episode_terminal_useful_rate": 0.2,
                "hard_failure_rate": 0.01,
                "robustness_coverage_labels_json": '["env:single","speed_bin:speed_bin_5p0_m_s"]',
                "robustness_coverage_rates_json": "[0.95,0.95]",
                "Q_weight_json": '{"q":1.0}',
                "R_weight_json": '{"r":1.0}',
                "reference_state_vector": "[0,0,0]",
                "reference_command_vector": "[0,0,0]",
            },
            {
                "primitive_variant_id": "redundant_speed_5p0",
                "primitive_id": "glide",
                "entry_role": "transition_object",
                "transition_entry_class": "inflight_stable",
                "local_lqr_reference_speed_m_s": 5.0,
                "local_lqr_speed_bin_id": lqr_speed_bin_id(5.0),
                "continuation_valid_rate": 0.93,
                "episode_terminal_useful_rate": 0.2,
                "hard_failure_rate": 0.01,
                "robustness_coverage_labels_json": '["env:single","speed_bin:speed_bin_5p0_m_s"]',
                "robustness_coverage_rates_json": "[0.93,0.93]",
                "Q_weight_json": '{"q":1.01}',
                "R_weight_json": '{"r":1.0}',
                "reference_state_vector": "[0,0,0]",
                "reference_command_vector": "[0,0,0]",
            },
            {
                "primitive_variant_id": "needed_speed_7p0",
                "primitive_id": "glide",
                "entry_role": "transition_object",
                "transition_entry_class": "inflight_stable",
                "local_lqr_reference_speed_m_s": 7.0,
                "local_lqr_speed_bin_id": lqr_speed_bin_id(7.0),
                "continuation_valid_rate": 0.70,
                "episode_terminal_useful_rate": 0.1,
                "hard_failure_rate": 0.02,
                "robustness_coverage_labels_json": '["env:single","speed_bin:speed_bin_7p0_m_s"]',
                "robustness_coverage_rates_json": "[0.70,0.70]",
                "Q_weight_json": '{"q":1.3}',
                "R_weight_json": '{"r":1.0}',
                "reference_state_vector": "[0,0,0]",
                "reference_command_vector": "[0,0,0]",
            },
        ]
    )

    selected = _coverage_medoid_selection(frame, max_representatives=2, case_id="balanced_cluster")

    assert set(selected["local_lqr_speed_bin_id"]) == {lqr_speed_bin_id(5.0), lqr_speed_bin_id(7.0)}
    assert "greedy_speed_bin_marginal_coverage_medoid" in set(selected["_medoid_selection_reason"])


def test_outcome_and_repeated_launch_validation_use_case_ids_histories_and_counts(tmp_path: Path) -> None:
    w3_root = _write_tiny_w3_root(tmp_path / "w3_survival" / "013")
    run_w3_survival_analysis(W3SurvivalAnalysisConfig(input_root=w3_root))
    study = run_post_w3_library_size_study(
        PostW3LibrarySizeStudyConfig(
            input_root=w3_root,
            output_root=tmp_path / "post_w3_library_size_study",
            run_id=1,
        )
    )

    outcome_result = run_outcome_model_build(
        OutcomeModelBuildConfig(
            compact_library_path=Path(study["manifest"]),
            output_root=tmp_path / "outcome_model",
            run_id=2,
        )
    )
    validation = run_repeated_launch_learning_curve(
        RepeatedLaunchValidationConfig(
            library_root=Path(study["run_root"]),
            outcome_root=Path(outcome_result["run_root"]),
            output_root=tmp_path / "repeated_launch_validation",
            run_id=1,
            dry_run_schedule=True,
        )
    )
    r10_validation = run_changed_case_validation(
        ChangedCaseValidationConfig(
            library_root=Path(study["run_root"]),
            outcome_root=Path(outcome_result["run_root"]),
            output_root=tmp_path / "changed_case_validation",
            run_id=1,
            dry_run_schedule=True,
        )
    )
    r11_validation = run_heldout_changed_case_validation(
        HeldoutChangedCaseValidationConfig(
            library_root=Path(study["run_root"]),
            outcome_root=Path(outcome_result["run_root"]),
            output_root=tmp_path / "heldout_changed_case_validation",
            run_id=1,
            dry_run_schedule=True,
        )
    )
    outcome = pd.read_csv(Path(outcome_result["run_root"]) / "metrics" / "outcome_model_summary.csv")
    r9_final = pd.read_csv(Path(validation["run_root"]) / "metrics" / "final_heldout_launch_schedule.csv")
    r9_history = pd.read_csv(Path(validation["run_root"]) / "metrics" / "history_launch_schedule.csv")
    r10_final = pd.read_csv(Path(r10_validation["run_root"]) / "metrics" / "final_heldout_launch_schedule.csv")
    r10_history = _read_schedule_table(Path(r10_validation["run_root"]), "history_launch_schedule")
    r11_final = pd.read_csv(Path(r11_validation["run_root"]) / "metrics" / "final_heldout_launch_schedule.csv")
    r11_history = _read_schedule_table(Path(r11_validation["run_root"]), "history_launch_schedule")
    r10_active_fan_audit = pd.read_csv(
        Path(r10_validation["run_root"]) / "metrics" / "active_fan_count_schedule_audit.csv"
    )

    assert outcome_result["status"] == "complete"
    assert validation["status"] == "dry_run_schedule"
    assert r10_validation["status"] == "dry_run_schedule"
    assert r11_validation["status"] == "dry_run_schedule"
    assert {
        "continuation_probability",
        "terminal_useful_probability",
        "hard_failure_risk",
        "finite_horizon_s",
        "controller_input_slots_per_primitive",
        "controller_input_update_period_s",
        "primitive_timing_contract_version",
    }.issubset(outcome.columns)
    assert set(outcome["library_size_case_id"]) == set(LIBRARY_SIZE_CASE_IDS)
    assert len(r9_final) == R9_EXPECTED_FINAL_HELDOUT_LAUNCHES
    assert len(r9_history) == R9_EXPECTED_HISTORY_LAUNCHES
    assert len(r10_final) == R10_EXPECTED_FINAL_HELDOUT_LAUNCHES
    assert len(r10_history) == R10_EXPECTED_HISTORY_LAUNCHES
    assert len(r11_final) == R10_EXPECTED_FINAL_HELDOUT_LAUNCHES
    assert len(r11_history) == R10_EXPECTED_HISTORY_LAUNCHES
    assert set(r9_final["library_size_case_id"]) == set(LIBRARY_SIZE_CASE_IDS)
    assert set(r9_final["policy_id"]) == set(POLICY_HISTORY_CONDITIONS)
    assert set(r9_final["history_length"]) == {0, *HISTORY_LENGTHS}
    assert set(r10_final["environment_block_id"]) == {
        "nominal_single_fan_perturbations",
        "nominal_four_fan_perturbations",
        "shifted_single_fan_positions",
        "shifted_four_fan_positions",
        "active_fan_number_variation",
        "arena_wide_fan_position_generalisation",
    }
    r10_active_final = r10_final[
        r10_final["environment_block_id"].eq("active_fan_number_variation")
    ].copy()
    r10_nominal_final = r10_final[
        r10_final["environment_block_id"].isin(
            {"nominal_single_fan_perturbations", "nominal_four_fan_perturbations"}
        )
    ].copy()
    r10_shifted_final = r10_final[
        r10_final["environment_block_id"].isin(
            {"shifted_single_fan_positions", "shifted_four_fan_positions"}
        )
    ].copy()
    r10_non_active_four = r10_final[
        r10_final["environment_block_id"].isin(
            {
                "nominal_four_fan_perturbations",
                "shifted_four_fan_positions",
            }
        )
    ].copy()
    assert set(r10_nominal_final["fan_position_policy"]) == {"fixed_base_positions"}
    assert set(r10_nominal_final["fan_position_xy_bounds_m"]) == {"fixed_base_positions_no_shift"}
    assert set(r10_active_final["fan_position_policy"]) == {"fixed_base_positions"}
    assert set(r10_shifted_final["fan_position_policy"]) == {"common_shift"}
    assert set(r10_non_active_four["scheduled_active_fan_count"].astype(int)) == {4}
    assert set(r10_non_active_four["fan_layout_policy"]) == {"four_fan_geometry"}
    assert set(r10_final[r10_final["environment_block_id"].str.contains("single_fan")]["fan_layout_policy"]) == {
        "single_fan_geometry"
    }
    assert set(r10_active_final["scheduled_active_fan_count"].astype(int)) == {1, 2, 3, 4}
    assert r10_active_final.groupby("scheduled_active_fan_count").size().to_dict() == {
        1: 125,
        2: 125,
        3: 125,
        4: 125,
    }
    assert set(r10_active_fan_audit["environment_block_id"]) == {
        "active_fan_number_variation",
        "arena_wide_fan_position_generalisation",
    }
    assert set(r10_active_fan_audit["scheduled_active_fan_count"].astype(int)) == {1, 2, 3, 4}
    assert set(r10_active_fan_audit["outer_case_count"].astype(int)) == {5}
    assert r10_active_fan_audit["audit_passed"].all()
    r10_arena_wide = r10_final[
        r10_final["environment_block_id"].eq("arena_wide_fan_position_generalisation")
    ].copy()
    assert len(r10_arena_wide) == 500
    assert set(r10_arena_wide["fan_position_policy"]) == {"independent_uniform_xy_bounds"}
    assert set(r10_arena_wide["scheduled_active_fan_count"].astype(int)) == {1, 2, 3, 4}
    assert r10_arena_wide.groupby("scheduled_active_fan_count").size().to_dict() == {
        1: 125,
        2: 125,
        3: 125,
        4: 125,
    }


def test_post_w3_compression_refuses_roots_without_w3_registry(tmp_path: Path) -> None:
    w3_root = _write_tiny_w3_root(tmp_path / "w3_survival" / "013")

    result = run_post_w3_library_size_study(
        PostW3LibrarySizeStudyConfig(
            input_root=w3_root,
            output_root=tmp_path / "post_w3_library_size_study",
            run_id=1,
        )
    )

    assert result["status"] == "blocked"
    assert result["blocked_reason"] == "missing_w3_survivor_registry"


def test_retired_single_compact_wrapper_requires_explicit_gate(tmp_path: Path) -> None:
    result = run_post_w3_cluster_merge(
        input_root=tmp_path / "w3_survival" / "013",
        output_root=tmp_path / "post_w3_cluster",
        run_id=1,
    )

    assert result["status"] == "blocked"
    assert result["blocked_reason"] == "retired_diagnostic_requires_explicit_allow_retired_diagnostic"


def _read_schedule_table(run_root: Path, table_name: str) -> pd.DataFrame:
    manifest_path = run_root / "manifests" / f"{table_name}_manifest.json"
    if manifest_path.is_file():
        manifest = load_table_manifest(manifest_path)
        frames = [
            read_table_partition(
                run_root / "tables" / partition.relative_path,
                storage_format=partition.storage_format,
            )
            for partition in manifest.tables
        ]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return pd.read_csv(run_root / "metrics" / f"{table_name}.csv")


def _write_tiny_w3_root(root: Path) -> Path:
    rows = [
        _w3_row("primvar_glide_inflight", "glide", "transition_object", "ctrl_glide", "w3_randomised_single", True, "accepted", True, False, "continuation_valid", "success", 0, transition_entry_class="inflight_stable"),
        _w3_row("primvar_glide_inflight", "glide", "transition_object", "ctrl_glide", "w3_randomised_four", True, "accepted", True, False, "continuation_valid", "success", 0, transition_entry_class="inflight_stable"),
        _w3_row("primvar_glide_inflight", "glide", "transition_object", "ctrl_glide", "w3_randomised_four", True, "failed", False, False, "hard_failure", "floor_violation", 0, transition_entry_class="inflight_stable"),
        _w3_row("primvar_lift_terminal", "lift_entry", "transition_object", "ctrl_lift", "w3_randomised_single", True, "weak", False, True, "episode_terminal_useful", "xy_boundary_terminal", 1, transition_entry_class="inflight_stable", transition_exit_class="safe_terminal"),
        _w3_row("primvar_lift_terminal", "lift_entry", "transition_object", "ctrl_lift", "w3_randomised_four", True, "weak", False, True, "episode_terminal_useful", "xy_boundary_terminal", 1, transition_entry_class="inflight_stable", transition_exit_class="safe_terminal"),
        _w3_row("primvar_lift_terminal", "lift_entry", "transition_object", "ctrl_lift", "w3_randomised_four", False, "rejected", False, False, "blocked", "entry_class_incompatible_start_family", 1, transition_entry_class="inflight_stable", transition_exit_class="hard_failure", transition_chain_compatible=False),
        _w3_row("primvar_lift_survived", "lift_entry", "transition_object", "ctrl_lift_survived", "w3_randomised_single", True, "accepted", True, False, "continuation_valid", "success", 2, transition_entry_class="inflight_stable"),
        _w3_row("primvar_lift_survived", "lift_entry", "transition_object", "ctrl_lift_survived", "w3_randomised_four", True, "accepted", True, False, "continuation_valid", "success", 2, transition_entry_class="inflight_stable"),
    ]
    for offset, primitive_id in enumerate(ACTIVE_PRIMITIVE_IDS, start=3):
        rows.extend(
            [
                _w3_row(
                    f"primvar_{primitive_id}_launch_gate",
                    primitive_id,
                    "transition_object",
                    f"ctrl_{primitive_id}_launch_gate",
                    "w3_randomised_single",
                    True,
                    "accepted",
                    True,
                    False,
                    "continuation_valid",
                    "success",
                    offset,
                    transition_entry_class="launch_gate",
                    transition_exit_class="post_launch_degraded",
                ),
                _w3_row(
                    f"primvar_{primitive_id}_launch_gate",
                    primitive_id,
                    "transition_object",
                    f"ctrl_{primitive_id}_launch_gate",
                    "w3_randomised_four",
                    True,
                    "accepted",
                    True,
                    False,
                    "continuation_valid",
                    "success",
                    offset,
                    transition_entry_class="launch_gate",
                    transition_exit_class="inflight_stable",
                ),
            ]
        )
    table_root = root / "tables" / "w3_survival_rows"
    table_root.mkdir(parents=True, exist_ok=True)
    partition = write_table_partition(pd.DataFrame(rows), table_root / "c00000.csv.gz", storage_format="csv_gz", compression_level=1)
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / "w3_survival_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "input_root": (root.parent.parent / "w2_survival" / "015").as_posix(),
                "row_count": len(rows),
                "project_title_version": POST_W3_PROJECT_TITLE_VERSION,
                "primitive_timing_contract": {
                    "finite_horizon_s": 0.1,
                    "controller_input_slots_per_primitive": 5,
                    "controller_input_update_period_s": 0.02,
                    "primitive_timing_contract_version": "v411_0p10s_5slot_20ms",
                },
                "method_evidence_level": "w3_dense_survival_pass",
                "test_fixture_not_method_evidence": False,
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    write_table_manifest(
        manifests / "table_manifest.json",
        TableManifest(run_id=13, root=root.as_posix(), storage_format="csv_gz", tables=(partition,)),
    )
    w2_manifests = root.parent.parent / "w2_survival" / "015" / "manifests"
    w2_manifests.mkdir(parents=True, exist_ok=True)
    (w2_manifests / "w2_survival_manifest.json").write_text(
        json.dumps(
            {
                "source_w01_root": (root.parent.parent / "w01_dense" / "015").as_posix(),
                "project_title_version": POST_W3_PROJECT_TITLE_VERSION,
                "status": "w2_dense_survival_pass",
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    return root


def _w3_row(
    variant_id: str,
    primitive_id: str,
    entry_role: str,
    controller_id: str,
    environment_mode: str,
    compatible: bool,
    outcome_class: str,
    continuation_valid: bool,
    terminal_useful: bool,
    boundary_use_class: str,
    failure_label: str,
    candidate_index: int,
    *,
    transition_entry_class: str = "inflight_stable",
    transition_exit_class: str = "",
    transition_chain_compatible: bool | None = None,
    local_speed_m_s: float = 5.0,
) -> dict[str, object]:
    prefix = "glide" if primitive_id == "glide" else "lift"
    if not transition_exit_class:
        if str(boundary_use_class) == "hard_failure" or str(outcome_class) == "failed":
            transition_exit_class = "hard_failure"
        elif bool(terminal_useful):
            transition_exit_class = "safe_terminal"
        elif transition_entry_class == "launch_gate":
            transition_exit_class = "post_launch_degraded"
        else:
            transition_exit_class = "inflight_stable"
    if transition_chain_compatible is None:
        transition_chain_compatible = str(transition_exit_class) != "hard_failure" and bool(compatible)
    start_state_family_by_entry = {
        "launch_gate": "launch_gate",
        "inflight_stable": "inflight_nominal",
        "boundary_near": "inflight_boundary_near",
        "recoverable_degraded": "inflight_recovery_edge",
    }
    return {
        "primitive_variant_id": variant_id,
        "primitive_id": primitive_id,
        "entry_role": entry_role,
        "controller_id": controller_id,
        "candidate_index": candidate_index,
        "candidate_weight_label": f"{prefix}_weights",
        "start_state_family": start_state_family_by_entry.get(transition_entry_class, "inflight_nominal"),
        "environment_mode": environment_mode,
        "entry_role_compatible": compatible,
        "transition_entry_class": transition_entry_class,
        "transition_exit_class": transition_exit_class,
        "transition_pair": f"{transition_entry_class}->{transition_exit_class}",
        "transition_chain_compatible": bool(transition_chain_compatible),
        "transition_success": bool(transition_chain_compatible),
        "continuation_valid": continuation_valid,
        "episode_terminal_useful": terminal_useful,
        "boundary_use_class": boundary_use_class,
        "outcome_class": outcome_class,
        "failure_label": failure_label,
        "minimum_wall_margin_m": 0.4,
        "floor_margin_m": 0.5,
        "ceiling_margin_m": 0.6,
        "energy_residual_m": 0.2,
        "lift_dwell_time_s": 0.8,
        "saturation_fraction": 0.01,
        "variant_K_gain_checksum": f"k_{prefix}",
        "variant_augmented_A_checksum": f"a_{prefix}",
        "variant_augmented_B_checksum": f"b_{prefix}",
        "variant_augmented_gain_checksum": f"g_{prefix}",
        "variant_Q_weight_json": "{\"q\":1}",
        "variant_R_weight_json": "{\"r\":1}",
        "variant_reference_state_vector": "[0,0,0]",
        "variant_reference_command_vector": "[0,0,0]",
        "variant_local_lqr_reference_speed_m_s": float(local_speed_m_s),
        "variant_local_lqr_speed_bin_id": lqr_speed_bin_id(float(local_speed_m_s)),
        "variant_finite_horizon_s": 0.1,
        "variant_controller_input_slots_per_primitive": 5,
        "variant_controller_input_update_period_s": 0.02,
        "variant_primitive_timing_contract_version": "v411_0p10s_5slot_20ms",
        "variant_timing_augmentation_type": "predictor_compensated_augmented_discrete_lqr_v1",
    }
