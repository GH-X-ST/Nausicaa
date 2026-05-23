from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from controller_registry import controller_from_evidence_row
from dense_archive_table_io import load_table_manifest, read_table_partition
from prim_cat import ACTIVE_PRIMITIVE_IDS
from run_lqr_tuning_sweep import LQRTuningSweepConfig, run_lqr_tuning_sweep


def test_lqr_tuning_rolls_out_candidate_controller_ids_and_writes_registry(tmp_path: Path) -> None:
    config = LQRTuningSweepConfig(
        run_id=91,
        output_root=tmp_path,
        rows=32,
        seed=91,
        candidate_count=2,
        paired_tests_per_candidate=1,
        candidate_chunk_size=8,
        workers=1,
        max_workers=1,
        storage_format="csv_gz",
        compression_level=1,
    )
    result = run_lqr_tuning_sweep(config)

    run_root = Path(result["run_root"])
    table_manifest = load_table_manifest(run_root / "manifests" / "table_manifest.json")
    first_chunk_manifest = run_root / "chunk_manifests" / "lqr_tuning_rows" / "c00000.json"
    last_chunk_manifest = run_root / "chunk_manifests" / "lqr_tuning_rows" / "c00003.json"
    first_partition = run_root / "tables" / "lqr_tuning_rows" / "c00000.csv.gz"
    frame = pd.concat(
        [
            read_table_partition(
                run_root / "tables" / partition.relative_path,
                storage_format="csv_gz",
            )
            for partition in table_manifest.tables
        ],
        ignore_index=True,
    )
    registry = pd.read_csv(result["selected_controller_registry"])
    chunk_summary = pd.read_csv(run_root / "metrics" / "chunk_summary.csv")
    runtime_summary = pd.read_csv(run_root / "metrics" / "runtime_summary.csv")
    file_size_audit = pd.read_csv(run_root / "metrics" / "file_size_audit.csv")

    assert first_chunk_manifest.is_file()
    assert last_chunk_manifest.is_file()
    assert table_manifest.tables[0].table_name == "lqr_tuning_rows"
    assert len(table_manifest.tables) == 4
    assert set(chunk_summary["status"]) == {"complete"}
    assert int(runtime_summary["row_count"].iloc[0]) == 32
    assert {
        "relative_path",
        "filename_stem_length",
        "relative_path_length",
        "stem_under_64",
        "path_under_140",
        "push_allowed",
    }.issubset(file_size_audit.columns)
    assert file_size_audit["push_allowed"].all()
    assert "candidate_weight_label" in frame.columns
    assert "controller_selection_status" in frame.columns
    assert "entry_rejection_class" in frame.columns
    assert "continuation_valid" in frame.columns
    assert "episode_terminal_useful" in frame.columns
    assert "boundary_terminal" not in set(frame["outcome_class"].astype(str))
    assert set(frame["hard_gate_status"].astype(str)).issubset({"passed", "blocked"})
    assert set(frame["controller_selection_status"]) == {"W0_W1_candidate_rollout"}
    assert frame.groupby("primitive_id")["controller_id"].nunique().ge(2).all()
    paired_layers = frame.groupby(["primitive_id", "candidate_index", "paired_start_key"])["W_layer"].agg(set)
    assert paired_layers.map(lambda layers: {"W0", "W1"}.issubset(layers)).all()
    assert set(registry["primitive_id"]) == set(ACTIVE_PRIMITIVE_IDS)
    assert set(registry["selected_controller_status"]).issuperset(
        {"smoke_selected_not_thesis_evidence", "rejected"}
    )
    assert set(registry["registry_status"]) == {"smoke_incomplete"}
    assert set(registry["registry_claim_status"]) == {"simulation_only_smoke_incomplete"}
    selected = registry[registry["selected_controller_status"] == "smoke_selected_not_thesis_evidence"]
    assert set(selected["primitive_id"]) == set(ACTIVE_PRIMITIVE_IDS)
    controller = controller_from_evidence_row(selected.iloc[0].to_dict())
    assert controller.controller_id == str(selected.iloc[0]["controller_id"])
    with pytest.raises(ValueError, match="selected-controller registry"):
        controller_from_evidence_row(frame.iloc[0].to_dict())

    before = (first_partition.stat().st_mtime_ns, first_chunk_manifest.stat().st_mtime_ns)
    resumed = run_lqr_tuning_sweep(config)
    after = (first_partition.stat().st_mtime_ns, first_chunk_manifest.stat().st_mtime_ns)
    resumed_chunk_summary = pd.read_csv(Path(resumed["run_root"]) / "metrics" / "chunk_summary.csv")
    assert before == after
    assert set(resumed_chunk_summary["status"]) == {"skipped"}

    with first_partition.open("ab") as handle:
        handle.write(b"corrupt")
    repaired = run_lqr_tuning_sweep(
        LQRTuningSweepConfig(
            run_id=91,
            output_root=tmp_path,
            rows=32,
            seed=91,
            candidate_count=2,
            paired_tests_per_candidate=1,
            candidate_chunk_size=8,
            workers=1,
            max_workers=1,
            storage_format="csv_gz",
            compression_level=1,
            repair_incomplete=True,
        )
    )
    repaired_manifest = load_table_manifest(Path(repaired["run_root"]) / "manifests" / "table_manifest.json")
    assert len(repaired_manifest.tables) == 4
    repaired_summary = pd.read_csv(Path(repaired["run_root"]) / "metrics" / "chunk_summary.csv")
    assert "failed" not in set(repaired_summary["status"])
