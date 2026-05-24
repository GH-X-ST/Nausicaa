from __future__ import annotations

from pathlib import Path

import pandas as pd

from dense_archive_table_io import load_table_manifest
from prim_cat import ACTIVE_PRIMITIVE_IDS
from run_lqr_w01_dense_chunked import W01DenseRunConfig, _l6_move_on_status, run_lqr_w01_dense_chunked


def test_w01_dry_run_writes_compact_manifests_and_no_partitions(tmp_path: Path) -> None:
    result = run_lqr_w01_dense_chunked(
        W01DenseRunConfig(
            run_id=1,
            output_root=tmp_path,
            rows=60,
            seed=1,
            candidate_chunk_size=20,
            workers=1,
            max_workers=1,
            dry_run_schedule=True,
            candidate_count=1,
        )
    )
    run_root = Path(result["run_root"])
    table_manifest = load_table_manifest(run_root / "manifests" / "table_manifest.json")
    chunk_summary = pd.read_csv(run_root / "metrics" / "chunk_summary.csv")

    assert table_manifest.tables == ()
    assert not (run_root / "tables" / "w01_primitive_rows").exists()
    assert set(chunk_summary["status"]) == {"scheduled"}
    assert (run_root / "manifests" / "primitive_variant_registry.json").is_file()
    assert (run_root / "reports" / "l6_move_on_check.md").is_file()
    assert (run_root / "reports" / "timing_synthesis_boundary.md").is_file()
    l6_report = (run_root / "reports" / "l6_move_on_check.md").read_text(encoding="ascii")
    assert "predictor_compensated_augmented_discrete_lqr_v1" in l6_report
    assert "no_rollout_evidence_written" in l6_report
    file_audit = pd.read_csv(run_root / "metrics" / "file_size_audit.csv")
    assert {
        "relative_path",
        "byte_count",
        "size_mb",
        "above_75mb",
        "above_100mb",
        "push_allowed",
        "dense_table_partition",
    }.issubset(file_audit.columns)


def test_l6_move_on_blocks_baseline_rows_and_missing_timing_coverage(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    pd.DataFrame(
        [
            {
                "primitive_id": primitive_id,
                "controller_design_role": "active_timing_aware_w01",
                "lqr_synthesis_status": "solved",
            }
            for primitive_id in ACTIVE_PRIMITIVE_IDS[:-1]
        ]
    ).to_csv(metrics / "primitive_variant_registry.csv", index=False)
    pd.DataFrame(
        [
            {
                "primitive_id": "glide",
                "entry_role": "launch_capable",
                "candidate_index": 0,
                "controller_design_role": "superseded_baseline_not_active_w01",
                "timing_augmentation_type": "none",
                "timing_design_version": "time_invariant_reduced_order_lqr_v1",
                "lqr_synthesis_status": "solved",
                "sampled_data_check_status": "sampled_stable",
                "sampled_data_timing_audit_status": "sampled_stable_with_nominal_timing_smoke",
                "row_count": 1000,
            }
        ]
    ).to_csv(metrics / "variant_synthesis_summary.csv", index=False)

    run_class, blockers = _l6_move_on_status(run_root=tmp_path, status="complete", row_count=1000)

    assert run_class == "preflight"
    assert "missing_timing_aware_solved_or_blocked_variant_for_safe_exit_or_recovery_handoff" in blockers
    assert "w01_rows_missing_active_timing_aware_controller_ids" in blockers
    assert "w01_rows_include_superseded_baseline_controller_ids" in blockers
