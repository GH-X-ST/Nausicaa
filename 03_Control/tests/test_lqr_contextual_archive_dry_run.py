from __future__ import annotations

from pathlib import Path

import pandas as pd

from dense_archive_table_io import load_table_manifest, read_table_partition
from prim_cat import ACTIVE_PRIMITIVE_IDS
from run_lqr_w01_dense_chunked import W01DenseRunConfig, run_lqr_w01_dense_chunked


def test_w01_tiny_smoke_covers_primitives_start_families_and_layers(tmp_path: Path) -> None:
    result = run_lqr_w01_dense_chunked(
        W01DenseRunConfig(
            run_id=2,
            output_root=tmp_path,
            rows=120,
            seed=2,
            candidate_chunk_size=40,
            workers=1,
            max_workers=1,
            storage_format="csv_gz",
            compression_level=1,
            candidate_count=1,
        )
    )
    run_root = Path(result["run_root"])
    manifest = load_table_manifest(run_root / "manifests" / "table_manifest.json")
    frame = pd.concat(
        [
            read_table_partition(run_root / "tables" / partition.relative_path, storage_format=partition.storage_format)
            for partition in manifest.tables
        ],
        ignore_index=True,
    )

    assert set(frame["primitive_id"]) == set(ACTIVE_PRIMITIVE_IDS)
    assert set(frame["start_state_family"]) == {
        "launch_gate",
        "inflight_nominal",
        "inflight_lift_region",
        "inflight_boundary_near",
        "inflight_recovery_edge",
    }
    assert {"W0", "W1"}.issubset(set(frame["W_layer"]))
    assert {"dry_air", "gaussian_single", "gaussian_four"}.issubset(set(frame["environment_mode"]))
    assert set(frame["small_library_selection_allowed"]) == {False}
    assert set(frame["pd_pid_fallback_allowed"]) == {False}


def test_w01_launch_gate_rejections_are_not_controller_failures(tmp_path: Path) -> None:
    result = run_lqr_w01_dense_chunked(
        W01DenseRunConfig(
            run_id=3,
            output_root=tmp_path,
            rows=24,
            seed=3,
            candidate_chunk_size=24,
            workers=1,
            max_workers=1,
            candidate_count=1,
        )
    )
    run_root = Path(result["run_root"])
    manifest = load_table_manifest(run_root / "manifests" / "table_manifest.json")
    frame = read_table_partition(
        run_root / "tables" / manifest.tables[0].relative_path,
        storage_format=manifest.tables[0].storage_format,
    )
    rejected = frame[
        frame["start_state_family"].eq("launch_gate")
        & frame["entry_role"].isin(["inflight_only", "terminal_or_recovery"])
    ]

    assert not rejected.empty
    assert set(rejected["outcome_class"]) == {"rejected"}
    assert set(rejected["entry_check_status"]) == {"entry_role_incompatible_start"}
    assert set(rejected["failure_label"]) == {"entry_role_not_launch_capable"}
    assert not rejected["lqr_synthesis_status"].astype(str).str.contains("entry_role").any()
