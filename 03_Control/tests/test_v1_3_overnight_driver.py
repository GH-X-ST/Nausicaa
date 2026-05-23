from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from run_feedback_contextual_v1_3_overnight import (
    OvernightEvidenceConfig,
    run_feedback_contextual_v1_3_overnight,
)


def test_overnight_driver_writes_independent_stage_statuses(tmp_path: Path) -> None:
    result = run_feedback_contextual_v1_3_overnight(
        OvernightEvidenceConfig(
            run_id=91,
            output_root=tmp_path,
            r6_target_rows=8,
            r6_fallback_rows=4,
            r8_target_rows=4,
            r8_fallback_rows=2,
            r9_target_rows=4,
            r9_fallback_rows=2,
            candidate_chunk_size=4,
            workers=2,
            max_workers=2,
            storage_format="csv_gz",
            compression_level=1,
            run_preflight_checks=False,
            run_r9=True,
            r6_time_budget_s=3600.0,
            r8_time_budget_s=3600.0,
            r9_time_budget_s=3600.0,
        )
    )
    manifest = json.loads(Path(result["status_manifest"]).read_text())
    statuses = {row["stage"]: row for row in manifest["stage_statuses"]}

    assert set(statuses) == {"R6", "R7", "R8", "R9"}
    assert statuses["R6"]["status"] in {"complete", "fallback"}
    assert statuses["R6"]["table_manifest_path"]
    assert statuses["R8"]["status"] in {"complete", "fallback", "partial", "blocked"}
    assert "No controller-performance" in manifest["claim_boundary"]


def test_r8_replay_reuses_source_rows_with_limit(tmp_path: Path) -> None:
    result = run_feedback_contextual_v1_3_overnight(
        OvernightEvidenceConfig(
            run_id=92,
            output_root=tmp_path,
            r6_target_rows=20,
            r6_fallback_rows=20,
            r8_target_rows=12,
            r8_fallback_rows=4,
            r9_target_rows=2,
            r9_fallback_rows=2,
            candidate_chunk_size=5,
            workers=2,
            max_workers=2,
            storage_format="csv_gz",
            compression_level=1,
            run_preflight_checks=False,
            run_r9=False,
        )
    )
    statuses = {status.stage: status for status in result["statuses"]}
    if statuses["R8"].table_manifest_path:
        manifest_path = Path(statuses["R8"].run_root) / "manifests" / "table_manifest.json"
        table_manifest = json.loads(manifest_path.read_text())
        first = Path(table_manifest["root"]) / "tables" / table_manifest["tables"][0]["relative_path"]
        frame = pd.read_csv(first, compression="gzip")
        assert frame["source_reuse_count"].max() <= 8
        assert set(frame["replay_generation_path"]) == {"simulate_primitive_rollout"}
        assert not frame["source_label_copied_as_evidence"].astype(bool).any()
