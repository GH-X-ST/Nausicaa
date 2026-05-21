from __future__ import annotations

import json
from pathlib import Path

import run_paired_w0_w1_archive_chunked as chunked
import run_paired_w0_w1_partitioned_planning as planning


def test_w1_only_active_schedule_ignores_missing_w0_planning_chunks(tmp_path: Path) -> None:
    root = _short_root(tmp_path)
    planning_root = root / "10_dense_archive_planning"
    archive_root = root / "12_paired_w0_w1_archive"
    active_modes = ("W1_single_fan", "W1_four_fan")
    planning.run_paired_w0_w1_partitioned_planning(
        run_id=13,
        result_root=planning_root,
        paired_scale_mode="proof",
        active_environment_modes=active_modes,
        proof_target_trials_per_environment=2,
        partition_rows=1,
        storage_format="csv_gz",
    )

    paths = chunked.run_paired_w0_w1_archive_chunked(
        run_id=14,
        planning_run_id=13,
        result_root=archive_root,
        active_environment_modes=active_modes,
        paired_scale_mode="proof",
        workers=1,
        max_workers=1,
        chunk_size=1,
        storage_format="csv_gz",
        dry_run_schedule=True,
        resume=True,
    )

    manifest = json.loads(paths["progress_manifest_json"].read_text(encoding="ascii"))
    assert manifest["paired_scale_mode"] == "proof"
    assert manifest["active_environment_modes"] == list(active_modes)
    assert manifest["scheduled_chunk_count"] == 4
    assert {chunk["test_environment_mode"] for chunk in manifest["chunks"]} == set(active_modes)


def test_existing_unrelated_archive_root_is_not_resumable(tmp_path: Path) -> None:
    root = _short_root(tmp_path)
    planning_root = root / "10_dense_archive_planning"
    archive_root = root / "12_paired_w0_w1_archive"
    planning.run_paired_w0_w1_partitioned_planning(
        run_id=13,
        result_root=planning_root,
        paired_scale_mode="proof",
        proof_target_trials_per_environment=1,
        partition_rows=1,
        storage_format="csv_gz",
    )
    unrelated = archive_root / "014" / "unrelated.txt"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("not a chunked archive\n", encoding="ascii")

    try:
        chunked.run_paired_w0_w1_archive_chunked(
            run_id=14,
            planning_run_id=13,
            result_root=archive_root,
            paired_scale_mode="proof",
            workers=1,
            max_workers=1,
            chunk_size=1,
            storage_format="csv_gz",
            dry_run_schedule=True,
            resume=True,
        )
    except RuntimeError as exc:
        assert "compatible chunked resume root" in str(exc)
    else:  # pragma: no cover - explicit failure branch.
        raise AssertionError("unrelated archive root was accepted as resumable")


def test_dry_run_progress_carries_d1a_metadata_from_planning(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _short_root(tmp_path)
    planning_root = root / "10_dense_archive_planning"
    archive_root = root / "12_paired_w0_w1_archive"
    _set_d1a_contract(monkeypatch, "thesis_primary", w0=1, w1=2)
    planning.run_paired_w0_w1_partitioned_planning(
        run_id=15,
        result_root=planning_root,
        paired_scale_mode="production",
        w0_target_trials_per_branch=1,
        w1_floor_trials_per_branch=2,
        w1_target_trials_per_branch=2,
        partition_rows=1,
        storage_format="csv_gz",
        d1a_evidence_class="thesis_primary",
    )

    paths = chunked.run_paired_w0_w1_archive_chunked(
        run_id=16,
        planning_run_id=15,
        result_root=archive_root,
        paired_scale_mode="production",
        workers=1,
        max_workers=1,
        chunk_size=1,
        storage_format="csv_gz",
        dry_run_schedule=True,
        resume=True,
    )

    manifest = json.loads(paths["progress_manifest_json"].read_text(encoding="ascii"))
    assert manifest["d1a_evidence_class"] == "thesis_primary"
    assert manifest["d1a_target_contract"] == "updated_thesis_scale_v1"
    assert manifest["d1a_w0_trials_per_environment"] == 1
    assert manifest["d1a_w1_trials_per_environment"] == 2


def _short_root(tmp_path: Path) -> Path:
    return tmp_path.parent / f"p{abs(hash(tmp_path.name)) % 100000}"


def _set_d1a_contract(monkeypatch, evidence_class: str, *, w0: int, w1: int) -> None:
    updated = dict(planning.D1A_EVIDENCE_CONTRACTS)
    updated[evidence_class] = {
        "w0_trials_per_environment": int(w0),
        "w1_trials_per_environment": int(w1),
    }
    monkeypatch.setattr(planning, "D1A_EVIDENCE_CONTRACTS", updated)
