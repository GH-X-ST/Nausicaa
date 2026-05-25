from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dense_archive_table_io import TableManifest, load_table_manifest, read_table_partition, write_table_manifest
from frozen_w01_controller_bundle import (
    FROZEN_CONTROLLER_BLOCKED,
    FROZEN_CONTROLLER_READY,
    load_frozen_w01_controller_bundle,
    materialize_frozen_w01_controller_bundle,
)
from lqr_controller import lqr_controller_for_primitive_id
from prim_cat import primitive_by_id
from primitive_variant_registry import (
    ENTRY_ROLE_REJECTION_LABEL,
    ENTRY_ROLE_REJECTION_STATUS,
    primitive_controller_variant,
    variant_row,
)
from run_w2_survival import W2SurvivalConfig, run_w2_survival


def test_frozen_w01_bundle_restores_complete_timing_controller(tmp_path: Path) -> None:
    source_root = _write_w01_source_fixture(tmp_path / "w01", "glide", include_augmented_payload=True)
    bundle_path = tmp_path / "bundle.json"

    materialize_frozen_w01_controller_bundle(input_root=source_root, bundle_path=bundle_path)
    bundle = load_frozen_w01_controller_bundle(bundle_path)
    record = bundle.records[0]

    assert record.bundle_status == FROZEN_CONTROLLER_READY
    assert record.controller.controller_id == record.variant.controller_id
    assert record.controller.lqr_gain_checksum == record.variant.K_gain_checksum
    assert record.controller.augmented_gain_checksum == record.variant.augmented_gain_checksum
    assert record.controller.augmented_gain_matrix_json
    assert record.controller.predictor_A_reduced_json


def test_missing_augmented_payload_blocks_instead_of_using_physical_k_only(tmp_path: Path) -> None:
    source_root = _write_w01_source_fixture(tmp_path / "w01", "glide", include_augmented_payload=False)
    bundle_path = tmp_path / "bundle.json"

    materialize_frozen_w01_controller_bundle(input_root=source_root, bundle_path=bundle_path)
    bundle = load_frozen_w01_controller_bundle(bundle_path)
    record = bundle.records[0]

    assert record.bundle_status == FROZEN_CONTROLLER_BLOCKED
    assert "missing_augmented_gain_matrix_json" in record.blocked_reason
    assert "missing_predictor_A_reduced_json" in record.blocked_reason
    assert record.controller.k_gain_matrix
    assert not record.controller.augmented_gain_matrix_json


def test_w2_small_replay_uses_frozen_history_backed_timing_state(tmp_path: Path) -> None:
    source_root = _write_w01_source_fixture(tmp_path / "w01", "glide", include_augmented_payload=True)
    result = run_w2_survival(
        W2SurvivalConfig(
            run_id=10,
            input_root=source_root,
            output_root=tmp_path / "w2",
            paired_tests_per_variant=1,
            candidate_chunk_size=2,
            workers=1,
            max_workers=1,
            storage_format="csv_gz",
            compression_level=1,
            resume=True,
            repair_incomplete=True,
        )
    )
    run_root = Path(result["run_root"])
    frame = _read_w2_frame(run_root)

    assert len(frame) == 2
    assert set(frame["environment_mode"]) == {"annular_gp_single", "annular_gp_four"}
    assert set(frame["controller_bundle_status"]) == {FROZEN_CONTROLLER_READY}
    assert set(frame["timing_state_source"]) == {"history_backed_fifo"}
    assert frame["fixed_lqr_replay_only"].astype(bool).all()
    assert not frame["baseline_controller_active"].astype(bool).any()


def test_w2_entry_role_rejections_are_preserved_but_not_survival_scored(tmp_path: Path) -> None:
    source_root = _write_w01_source_fixture(tmp_path / "w01", "lift_entry", include_augmented_payload=True)
    result = run_w2_survival(
        W2SurvivalConfig(
            run_id=11,
            input_root=source_root,
            output_root=tmp_path / "w2",
            paired_tests_per_variant=1,
            candidate_chunk_size=2,
            workers=1,
            max_workers=1,
            storage_format="csv_gz",
            compression_level=1,
            resume=True,
            repair_incomplete=True,
        )
    )
    run_root = Path(result["run_root"])
    frame = _read_w2_frame(run_root)
    variant_summary = pd.read_csv(run_root / "metrics" / "w2_variant_survival_summary.csv")

    assert set(frame["entry_check_status"]) == {ENTRY_ROLE_REJECTION_STATUS}
    assert set(frame["failure_label"]) == {ENTRY_ROLE_REJECTION_LABEL}
    assert set(frame["outcome_class"]) == {"rejected"}
    assert set(frame["w2_survival_status"]) == {"blocked"}
    assert int(variant_summary["compatible_row_count"].iloc[0]) == 0
    assert variant_summary["w2_variant_status"].iloc[0] == "not_run"


def test_w2_default_schedule_dimensions() -> None:
    assert 256 * 2 * 100 == 51200


def _write_w01_source_fixture(
    root: Path,
    primitive_id: str,
    *,
    include_augmented_payload: bool,
) -> Path:
    controller = lqr_controller_for_primitive_id(primitive_id)
    primitive = primitive_by_id(primitive_id)
    variant = primitive_controller_variant(
        primitive=primitive,
        controller=controller,
        candidate_index=0,
        candidate_weight_label=f"{primitive_id}_fixture",
    )
    row = variant_row(variant)
    if include_augmented_payload:
        row["augmented_gain_matrix_json"] = controller.augmented_gain_matrix_json
        row["predictor_A_reduced_json"] = controller.predictor_A_reduced_json
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / "primitive_variant_registry.json").write_text(
        json.dumps(
            {
                "registry_version": "test_registry",
                "variant_count": 1,
                "primitive_count": 1,
                "entry_roles": {primitive_id: variant.entry_role},
                "variants": [row],
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    (manifests / "run_manifest.json").write_text(
        json.dumps({"run_id": 8, "status": "fixture"}, indent=2) + "\n",
        encoding="ascii",
    )
    write_table_manifest(
        manifests / "table_manifest.json",
        TableManifest(run_id=8, root=root.as_posix(), storage_format="csv_gz", tables=()),
    )
    return root


def _read_w2_frame(run_root: Path) -> pd.DataFrame:
    manifest = load_table_manifest(run_root / "manifests" / "table_manifest.json")
    return pd.concat(
        [
            read_table_partition(run_root / "tables" / partition.relative_path, storage_format=partition.storage_format)
            for partition in manifest.tables
        ],
        ignore_index=True,
    )
