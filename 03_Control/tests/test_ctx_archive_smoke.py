from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dense_archive_table_io import load_table_manifest, read_table_partition
from run_ctx_archive import ContextArchiveConfig, run_contextual_archive_preflight


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

    assert run_manifest["claim_status"] == "simulation_only_feedback_backed_preflight"
    assert run_manifest["rollout_backend"] == "model_backed_feedback"
    assert run_manifest["chunk_execution_backend"] == "process_pool"
    assert run_manifest["worker_enabled"] is True
    assert run_manifest["rows_requested"] == 500
    assert len(table_manifest.tables) == 4
    assert int(runtime_summary["row_count"].iloc[0]) == 500
    assert int(outcome_summary["row_count"].sum()) == 500
    assert file_size_audit["under_100mb"].all()
    assert (run_root / "reports" / "run_report.md").is_file()

    first_partition = run_root / "tables" / table_manifest.tables[0].relative_path
    frame = read_table_partition(first_partition, storage_format="csv_gz")
    assert {
        "rollout_id",
        "context_w_cg_m_s",
        "outcome_class",
        "rollout_backend",
        "evidence_role",
        "continuation_status",
        "episode_terminal_status",
        "trajectory_integrity_status",
        "surrogate_binding_status",
    }.issubset(frame.columns)
    assert set(frame["rollout_backend"]) == {"model_backed_feedback"}
    assert set(frame["evidence_role"]) == {"feedback_rollout_candidate"}


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

    assert result_entries == [".gitkeep"]
