from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from dense_archive_table_io import load_table_manifest
from fixed_gate_table_sources import read_fixed_gate_table_source
from run_fixed_gate_w0_w1_archive import run_fixed_gate_w0_w1_archive
from run_fixed_gate_w0_w1_archive_chunked import run_fixed_gate_w0_w1_archive_chunked


def test_simple_fixed_gate_runner_refuses_dense_or_archive_scale(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="smoke-scale only.*run_fixed_gate_w0_w1_archive_chunked.py"):
        run_fixed_gate_w0_w1_archive(
            run_id=411,
            rows_per_branch=201,
            latency_case="none",
            run_purpose="archive",
            result_root=tmp_path,
            storage_format="csv",
            overwrite=True,
        )


def test_chunked_archive_writes_compressed_partitions_and_no_full_rollout_csv(tmp_path: Path) -> None:
    paths = run_fixed_gate_w0_w1_archive_chunked(
        run_id=421,
        rows_per_branch=4,
        seed=20260522,
        latency_case="none",
        candidate_chunk_size=4,
        workers=1,
        max_workers=1,
        result_root=tmp_path,
        storage_format="csv_gz",
        compression_level=1,
        overwrite=True,
    )

    root = paths["root"]
    manifest = json.loads(paths["manifest_json"].read_text(encoding="ascii"))
    table_manifest = load_table_manifest(paths["table_manifest_json"])
    table_names = {partition.table_name for partition in table_manifest.tables}
    rollout_rows = read_fixed_gate_table_source(root, table_name="primitive_rollout_rows")

    assert manifest["worker_count_decision"]["selected_worker_count"] == 1
    assert manifest["full_rollout_metrics_csv_written"] is False
    assert not (root / "metrics" / "fixed_gate_w0_w1_primitive_rollout_rows.csv").exists()
    assert {"candidate_index", "primitive_rollout_rows", "partial_feedback_rows", "chunk_branch_coverage"}.issubset(table_names)
    assert {path.suffix for path in (root / "chunk_manifests").glob("chunk_*.json")} == {".json"}
    assert {"single_fan_branch", "four_fan_branch"} == set(rollout_rows["fan_branch"])
    assert {"W0", "W1"} == set(rollout_rows["W_layer"])
    assert "archive_chunk_index" in rollout_rows.columns


def test_chunked_archive_resume_skips_and_repair_reruns_corrupt_chunk(tmp_path: Path) -> None:
    paths = run_fixed_gate_w0_w1_archive_chunked(
        run_id=422,
        rows_per_branch=4,
        seed=20260522,
        latency_case="none",
        candidate_chunk_size=4,
        workers=1,
        max_workers=1,
        result_root=tmp_path,
        storage_format="csv_gz",
        compression_level=1,
        overwrite=True,
    )
    resumed = run_fixed_gate_w0_w1_archive_chunked(
        run_id=422,
        rows_per_branch=4,
        seed=20260522,
        latency_case="none",
        candidate_chunk_size=4,
        workers=1,
        max_workers=1,
        result_root=tmp_path,
        storage_format="csv_gz",
        compression_level=1,
        resume=True,
    )
    runtime = pd.read_csv(resumed["runtime_summary_csv"])
    assert int(runtime.loc[0, "skipped_chunk_count"]) > 0

    table_manifest = load_table_manifest(paths["table_manifest_json"])
    first_partition = table_manifest.tables[0]
    corrupt_path = paths["root"] / "tables" / first_partition.relative_path
    corrupt_path.write_text("corrupt", encoding="ascii")

    repaired = run_fixed_gate_w0_w1_archive_chunked(
        run_id=422,
        rows_per_branch=4,
        seed=20260522,
        latency_case="none",
        candidate_chunk_size=4,
        workers=1,
        max_workers=1,
        result_root=tmp_path,
        storage_format="csv_gz",
        compression_level=1,
        resume=True,
        repair_incomplete=True,
    )
    repaired_rows = read_fixed_gate_table_source(repaired["root"], table_name="primitive_rollout_rows")
    assert not repaired_rows.empty


def test_chunked_archive_overwrite_is_preflight_only(tmp_path: Path) -> None:
    run_fixed_gate_w0_w1_archive_chunked(
        run_id=423,
        rows_per_branch=2,
        latency_case="none",
        candidate_chunk_size=4,
        workers=1,
        max_workers=1,
        result_root=tmp_path,
        storage_format="csv_gz",
        overwrite=True,
    )

    with pytest.raises(RuntimeError, match="overwrite is allowed only for scratch/preflight"):
        run_fixed_gate_w0_w1_archive_chunked(
            run_id=423,
            rows_per_branch=2,
            latency_case="none",
            candidate_chunk_size=4,
            workers=1,
            max_workers=1,
            result_root=tmp_path,
            storage_format="csv_gz",
            overwrite=True,
        )
