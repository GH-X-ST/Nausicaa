from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

import run_feedback_contextual_v1_4_overnight as driver
from run_feedback_contextual_v1_4_overnight import (
    OvernightV14Config,
    run_feedback_contextual_v1_4_overnight,
)


def _status_rows(path: Path) -> dict[str, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="ascii"))
    return {str(row["stage"]): row for row in payload["stage_statuses"]}


def test_v1_4_preflight_failure_blocks_before_projection(tmp_path: Path) -> None:
    result = run_feedback_contextual_v1_4_overnight(
        OvernightV14Config(
            run_id="141",
            output_root=tmp_path,
            preflight_commands=((sys.executable, "-c", "import sys; sys.exit(7)"),),
            run_preflight_checks=True,
        )
    )

    statuses = _status_rows(result.status_manifest_path)
    assert statuses["R6"]["status"] == "blocked"
    assert statuses["R7"]["status"] == "deferred"
    assert not (result.run_root / "first_chunk_projection").exists()


def test_v1_4_dry_run_schedule_writes_deferred_status_only(tmp_path: Path) -> None:
    result = run_feedback_contextual_v1_4_overnight(
        OvernightV14Config(
            run_id="142",
            output_root=tmp_path,
            r6_target_rows=12,
            r6_fallback_rows=6,
            candidate_chunk_size=4,
            workers=1,
            max_workers=1,
            storage_format="csv_gz",
            run_preflight_checks=False,
            dry_run_schedule=True,
        )
    )

    statuses = _status_rows(result.status_manifest_path)
    assert set(statuses) == {"R6", "R7", "R8", "R9"}
    assert all(row["status"] == "deferred" for row in statuses.values())
    assert (result.run_root / "dry_run_schedule").exists()
    assert not (result.run_root / "r6_w0_w1_archive").exists()


def test_v1_4_stage_local_r8_block_preserves_r6_r7(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_surrogate_status(layers) -> str:
        layer_tuple = tuple(layers)
        if layer_tuple == ("W2",):
            return "blocked:forced_missing_w2_for_stage_local_test"
        return "ready:forced_test"

    monkeypatch.setattr(driver, "_surrogate_status", fake_surrogate_status)
    result = run_feedback_contextual_v1_4_overnight(
        OvernightV14Config(
            run_id="143",
            output_root=tmp_path,
            r6_target_rows=8,
            r6_fallback_rows=4,
            r8_target_rows=4,
            r8_fallback_rows=2,
            candidate_chunk_size=4,
            workers=1,
            max_workers=1,
            storage_format="csv_gz",
            run_preflight_checks=False,
            skip_r9=True,
        )
    )

    statuses = _status_rows(result.status_manifest_path)
    assert statuses["R6"]["status"] in {"complete", "fallback"}
    assert statuses["R7"]["status"] in {"complete", "fallback", "partial"}
    assert statuses["R8"]["status"] == "blocked"
    assert "forced_missing_w2" in statuses["R8"]["surrogate_status"]


def test_v1_4_uses_stage_specific_r6_and_r8_projections(tmp_path: Path) -> None:
    result = run_feedback_contextual_v1_4_overnight(
        OvernightV14Config(
            run_id="144",
            output_root=tmp_path,
            r6_target_rows=12,
            r6_fallback_rows=6,
            r8_target_rows=4,
            r8_fallback_rows=2,
            candidate_chunk_size=4,
            workers=1,
            max_workers=1,
            storage_format="csv_gz",
            run_preflight_checks=False,
            stop_after_stage="r8_projection",
            skip_r9=True,
        )
    )
    payload = json.loads(result.status_manifest_path.read_text(encoding="ascii"))
    projection_stages = [row["stage"] for row in payload["metadata"]["projection_records"]]
    assert projection_stages == ["R6", "R8"]


def test_v1_4_status_manifest_updates_include_claim_boundary(tmp_path: Path) -> None:
    result = run_feedback_contextual_v1_4_overnight(
        OvernightV14Config(
            run_id="145",
            output_root=tmp_path,
            r6_target_rows=8,
            r6_fallback_rows=4,
            candidate_chunk_size=4,
            workers=1,
            max_workers=1,
            storage_format="csv_gz",
            run_preflight_checks=False,
            stop_after_stage="r7",
            skip_r9=True,
        )
    )
    payload = json.loads(result.status_manifest_path.read_text(encoding="ascii"))
    statuses = {str(row["stage"]): row for row in payload["stage_statuses"]}
    assert payload["metadata"]["version"] == "feedback_contextual_primitive_v1_4"
    assert "No controller-performance" in payload["claim_boundary"]
    assert statuses["R6"]["table_manifest_path"]
    assert statuses["R8"]["status"] == "deferred"
    assert pd.notna(statuses["R6"]["row_count"])
