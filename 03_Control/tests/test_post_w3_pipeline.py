from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dense_archive_table_io import TableManifest, write_table_manifest, write_table_partition
from run_outcome_model_build import OutcomeModelBuildConfig, run_outcome_model_build
from run_post_w3_cluster_merge import run_post_w3_cluster_merge
from run_post_w3_library_size_study import (
    LIBRARY_SIZE_CASE_IDS,
    PostW3LibrarySizeStudyConfig,
    run_post_w3_library_size_study,
)
from run_repeated_launch_learning_curve import (
    HISTORY_LENGTHS,
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
    assert registry["survivor_count"] == 2
    survived = summary[summary["primitive_variant_id"] == "primvar_glide_launch"]
    downgraded = summary[summary["primitive_variant_id"] == "primvar_lift_terminal"]
    assert survived["w3_variant_status"].iloc[0] == "survived"
    assert downgraded["w3_variant_status"].iloc[0] == "downgraded"
    assert int(downgraded["incompatible_row_count"].iloc[0]) == 1
    assert int(downgraded["continuation_valid_count"].iloc[0]) == 0
    assert int(downgraded["episode_terminal_useful_count"].iloc[0]) == 2


def test_post_w3_library_size_study_writes_four_cases_without_mutation(tmp_path: Path) -> None:
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
    assert representative_ids == {"primvar_glide_launch", "primvar_lift_survived"}
    for representative in representatives:
        assert representative["w3_variant_status"] == "survived"
        assert representative["mutation_status"].startswith("references_existing_frozen_variant")
        assert representative["library_size_case_id"] == "no_cluster_no_merge"
    glide = [row for row in representatives if row["primitive_variant_id"] == "primvar_glide_launch"][0]
    assert glide["controller_id"] == "ctrl_glide"
    assert glide["K_gain_checksum"] == "k_glide"


def test_outcome_and_repeated_launch_validation_use_case_ids_and_histories(tmp_path: Path) -> None:
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
        )
    )
    outcome = pd.read_csv(Path(outcome_result["run_root"]) / "metrics" / "outcome_model_summary.csv")
    learning_curve = pd.read_csv(Path(validation["run_root"]) / "metrics" / "validation_learning_curve.csv")

    assert outcome_result["status"] == "complete"
    assert validation["status"] == "complete"
    assert {"continuation_probability", "terminal_useful_probability", "hard_failure_risk"}.issubset(outcome.columns)
    assert set(outcome["library_size_case_id"]) == set(LIBRARY_SIZE_CASE_IDS)
    assert set(learning_curve["library_size_case_id"]) == set(LIBRARY_SIZE_CASE_IDS)
    assert set(learning_curve["history_length"]) == set(HISTORY_LENGTHS)


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


def _write_tiny_w3_root(root: Path) -> Path:
    rows = [
        _w3_row("primvar_glide_launch", "glide", "launch_capable", "ctrl_glide", "w3_randomised_single", True, "accepted", True, False, "continuation_valid", "success", 0),
        _w3_row("primvar_glide_launch", "glide", "launch_capable", "ctrl_glide", "w3_randomised_four", True, "accepted", True, False, "continuation_valid", "success", 0),
        _w3_row("primvar_glide_launch", "glide", "launch_capable", "ctrl_glide", "w3_randomised_four", True, "failed", False, False, "hard_failure", "floor_violation", 0),
        _w3_row("primvar_lift_terminal", "lift_entry", "inflight_only", "ctrl_lift", "w3_randomised_single", True, "weak", False, True, "episode_terminal_useful", "xy_boundary_terminal", 1),
        _w3_row("primvar_lift_terminal", "lift_entry", "inflight_only", "ctrl_lift", "w3_randomised_four", True, "weak", False, True, "episode_terminal_useful", "xy_boundary_terminal", 1),
        _w3_row("primvar_lift_terminal", "lift_entry", "inflight_only", "ctrl_lift", "w3_randomised_four", False, "rejected", False, False, "blocked", "entry_role_not_launch_capable", 1),
        _w3_row("primvar_lift_survived", "lift_entry", "inflight_only", "ctrl_lift_survived", "w3_randomised_single", True, "accepted", True, False, "continuation_valid", "success", 2),
        _w3_row("primvar_lift_survived", "lift_entry", "inflight_only", "ctrl_lift_survived", "w3_randomised_four", True, "accepted", True, False, "continuation_valid", "success", 2),
    ]
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
                "project_title_version": "LQR-Stabilised Contextual Primitive v4.7",
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
        json.dumps({"source_w01_root": (root.parent.parent / "w01_dense" / "015").as_posix()}, indent=2) + "\n",
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
) -> dict[str, object]:
    prefix = "glide" if primitive_id == "glide" else "lift"
    return {
        "primitive_variant_id": variant_id,
        "primitive_id": primitive_id,
        "entry_role": entry_role,
        "controller_id": controller_id,
        "candidate_index": candidate_index,
        "candidate_weight_label": f"{prefix}_weights",
        "environment_mode": environment_mode,
        "entry_role_compatible": compatible,
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
        "variant_finite_horizon_s": 0.1,
        "variant_controller_input_slots_per_primitive": 5,
        "variant_controller_input_update_period_s": 0.02,
        "variant_primitive_timing_contract_version": "v411_0p10s_5slot_20ms",
        "variant_timing_augmentation_type": "predictor_compensated_augmented_discrete_lqr_v1",
    }
