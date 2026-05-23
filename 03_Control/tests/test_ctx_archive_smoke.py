from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from dense_archive_table_io import load_table_manifest, read_table_partition
from run_ctx_archive import ContextArchiveConfig, run_contextual_archive_preflight
from run_lqr_contextual_archive import parse_args as parse_official_archive_args


def _results_entries_are_placeholder_or_allowed(entries: list[str]) -> bool:
    allowed_root = os.environ.get("NAUSICAA_ALLOW_LOCAL_EVIDENCE_ROOT", "").strip()
    if not allowed_root:
        return entries == [".gitkeep"]

    root = Path(allowed_root)
    if root.is_absolute():
        try:
            allowed_prefix = root.resolve().relative_to(
                Path.cwd() / "03_Control" / "05_Results"
            )
        except ValueError:
            return False
    else:
        root_text = root.as_posix().rstrip("/")
        prefix = "03_Control/05_Results/"
        if root_text.startswith(prefix):
            root_text = root_text[len(prefix) :]
        allowed_prefix = Path(root_text)

    allowed_text = allowed_prefix.as_posix().rstrip("/")
    allowed_parents = {
        parent.as_posix()
        for parent in Path(allowed_text).parents
        if parent.as_posix() != "."
    }
    return all(
        entry == ".gitkeep"
        or entry == allowed_text
        or entry in allowed_parents
        or entry.startswith(f"{allowed_text}/")
        for entry in entries
    )


def test_contextual_archive_preflight_writes_temp_chunked_evidence(tmp_path: Path) -> None:
    result = run_contextual_archive_preflight(
        ContextArchiveConfig(
            run_id=24,
            rows=500,
            seed=11,
            w_layers=("W0", "W1"),
            env_modes=("dry_air", "gaussian_single"),
            candidate_chunk_size=125,
            workers=8,
            max_workers=8,
            storage_format="csv_gz",
            compression_level=1,
            resume=True,
            repair_incomplete=False,
            dry_run_schedule=False,
            stop_after_chunks=None,
            continue_on_chunk_failure=False,
            output_root=tmp_path,
        )
    )
    run_root = Path(result["run_root"])
    table_manifest = load_table_manifest(run_root / "manifests" / "table_manifest.json")
    run_manifest = json.loads((run_root / "manifests" / "run_manifest.json").read_text())
    runtime_summary = pd.read_csv(run_root / "metrics" / "runtime_summary.csv")
    outcome_summary = pd.read_csv(run_root / "metrics" / "outcome_summary.csv")
    file_size_audit = pd.read_csv(run_root / "metrics" / "file_size_audit.csv")
    coverage_summary = pd.read_csv(run_root / "metrics" / "coverage_summary.csv")
    ratio_summary = pd.read_csv(run_root / "metrics" / "blocked_or_approximate_ratio_summary.csv")

    assert run_manifest["claim_status"] == "simulation_only_lqr_backed_preflight"
    assert run_manifest["rollout_backend"] == "model_backed_lqr"
    assert run_manifest["selected_controller_registry_required"] is True
    assert run_manifest["archive_evidence_status"] == "blocked"
    assert run_manifest["evidence_eligibility_reason"] == "blocked_missing_selected_registry"
    assert run_manifest["chunk_execution_backend"] == "process_pool"
    assert run_manifest["worker_enabled"] is True
    assert run_manifest["rows_requested"] == 500
    assert len(table_manifest.tables) == 4
    assert int(runtime_summary["row_count"].iloc[0]) == 500
    assert int(outcome_summary["row_count"].sum()) == 500
    assert file_size_audit["under_100mb"].all()
    assert {"start_state_family", "primitive_id", "environment_id", "W_layer"}.issubset(
        set(coverage_summary["coverage_axis"])
    )
    assert "blocked_ratio" in ratio_summary.columns
    assert (run_root / "reports" / "run_report.md").is_file()
    assert (run_root / "reports" / "claim_boundary_report.md").is_file()

    first_partition = run_root / "tables" / table_manifest.tables[0].relative_path
    frame = read_table_partition(first_partition, storage_format="csv_gz")
    assert {
        "rollout_id",
        "context_w_cg_m_s",
        "outcome_class",
        "rollout_backend",
        "evidence_role",
        "continuation_valid",
        "episode_terminal_useful",
        "continuation_status",
        "episode_terminal_status",
        "trajectory_integrity_status",
        "surrogate_binding_status",
        "state_sample_source",
        "start_state_family",
        "paired_start_key",
        "state_envelope_label",
        "previous_primitive_status",
        "launch_gate_compliant",
        "state_sampling_version",
        "primitive_feature_vector",
        "boundary_use_class",
        "implementation_instance_implementation_instance_id",
        "plant_instance_plant_instance_id",
        "environment_adjustment_status",
        "state_feedback_delay_applied",
        "command_delay_applied",
        "actuator_lag_applied",
        "saturation_fraction",
        "controller_family",
        "controller_id",
        "controller_selection_status",
        "controller_executable",
        "controller_evidence_status",
        "archive_evidence_status",
        "evidence_eligibility_reason",
        "registry_status",
        "registry_claim_status",
        "registry_path",
        "lqr_synthesis_status",
        "reduced_order_lqr",
        "sampled_data_check_status",
        "latency_actuator_survival_status",
    }.issubset(frame.columns)
    for name in ("x_w", "y_w", "z_w", "phi", "theta", "psi", "u", "v", "w", "p", "q", "r", "delta_a", "delta_e", "delta_r"):
        assert f"initial_{name}" in frame.columns
    assert set(frame["start_state_family"]).issubset(
        {
            "launch_gate",
            "inflight_nominal",
            "inflight_lift_region",
            "inflight_boundary_near",
            "inflight_recovery_edge",
        }
    )
    assert set(frame["rollout_backend"]) == {"model_backed_lqr"}
    assert set(frame["evidence_role"]).issubset({"lqr_rollout_candidate", "blocked_lqr_synthesis"})
    assert set(frame["controller_family"]) == {"lqr"}
    assert set(frame["W_layer"]).issubset({"W0", "W1"})
    assert set(frame["controller_selection_status"]) == {"missing_selected_registry_entry"}
    assert set(frame["controller_executable"].astype(str).str.lower()) == {"false"}
    assert set(frame["archive_evidence_status"]) == {"blocked"}
    assert set(frame["evidence_eligibility_reason"]) == {"blocked_missing_selected_registry"}
    assert set(frame["outcome_class"]) == {"blocked"}
    assert "boundary_terminal" not in set(frame["outcome_class"].astype(str))


def test_contextual_archive_smoke_backend_is_explicit_opt_in(tmp_path: Path) -> None:
    result = run_contextual_archive_preflight(
        ContextArchiveConfig(
            run_id=26,
            rows=16,
            seed=13,
            w_layers=("W0", "W1"),
            env_modes=("dry_air", "gaussian_single"),
            candidate_chunk_size=8,
            workers=2,
            max_workers=2,
            storage_format="csv_gz",
            compression_level=1,
            resume=True,
            repair_incomplete=False,
            dry_run_schedule=False,
            stop_after_chunks=None,
            continue_on_chunk_failure=False,
            output_root=tmp_path,
            rollout_backend="smoke_only",
        )
    )
    run_root = Path(result["run_root"])
    table_manifest = load_table_manifest(run_root / "manifests" / "table_manifest.json")
    first_partition = run_root / "tables" / table_manifest.tables[0].relative_path
    frame = read_table_partition(first_partition, storage_format="csv_gz")

    assert set(frame["rollout_backend"]) == {"smoke_only"}
    assert set(frame["evidence_role"]) == {"interface_smoke"}


def test_official_contextual_archive_passes_selected_registry_path() -> None:
    config = parse_official_archive_args(
        [
            "--run-id",
            "31",
            "--rows",
            "16",
            "--selected-controller-registry",
            "03_Control/05_Results/lqr_contextual_v1_0/r6/tune_100/metrics/selected_lqr_controllers.csv",
        ]
    )
    assert config.selected_controller_registry is not None
    assert config.selected_controller_registry.as_posix().endswith("selected_lqr_controllers.csv")


def test_contextual_archive_rejects_w2_w3_for_r6_stage(tmp_path: Path) -> None:
    config = ContextArchiveConfig(
        run_id=27,
        rows=16,
        seed=14,
        w_layers=("W0", "W1", "W2"),
        env_modes=("dry_air", "gaussian_single"),
        candidate_chunk_size=8,
        workers=2,
        max_workers=2,
        storage_format="csv_gz",
        compression_level=1,
        resume=True,
        repair_incomplete=False,
        dry_run_schedule=False,
        stop_after_chunks=None,
        continue_on_chunk_failure=False,
        output_root=tmp_path,
    )
    try:
        run_contextual_archive_preflight(config)
    except ValueError as exc:
        assert "W0/W1 only" in str(exc)
    else:
        raise AssertionError("R6 archive accepted W2/W3 coverage")


def test_contextual_archive_preflight_does_not_touch_active_results_root(tmp_path: Path) -> None:
    run_contextual_archive_preflight(
        ContextArchiveConfig(
            run_id=25,
            rows=500,
            seed=12,
            w_layers=("W0", "W1"),
            env_modes=("dry_air", "gaussian_single"),
            candidate_chunk_size=125,
            workers=8,
            max_workers=8,
            storage_format="csv_gz",
            compression_level=1,
            resume=True,
            repair_incomplete=False,
            dry_run_schedule=False,
            stop_after_chunks=None,
            continue_on_chunk_failure=False,
            output_root=tmp_path,
        )
    )
    result_entries = [
        path.relative_to("03_Control/05_Results").as_posix()
        for path in Path("03_Control/05_Results").rglob("*")
    ]

    assert _results_entries_are_placeholder_or_allowed(result_entries)
