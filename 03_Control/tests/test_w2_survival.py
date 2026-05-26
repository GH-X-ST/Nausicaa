from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from dense_archive_table_io import TableManifest, load_table_manifest, read_table_partition, write_table_manifest
from frozen_w01_controller_bundle import (
    FROZEN_CONTROLLER_BLOCKED,
    FROZEN_CONTROLLER_READY,
    load_frozen_w01_controller_bundle,
    write_frozen_w01_controller_bundle,
)
from lqr_controller import lqr_controller_for_primitive_id
from prim_cat import primitive_by_id
from primitive_timing_contract import primitive_timing_contract_row
from primitive_variant_registry import (
    ENTRY_ROLE_REJECTION_LABEL,
    ENTRY_ROLE_REJECTION_STATUS,
    primitive_controller_variant,
    variant_row,
)
from run_w2_survival import W2_DENSE_ROW_COUNT, W2_DENSE_VARIANT_COUNT, W2SurvivalConfig, discover_latest_w01_root_for_w2, run_w2_survival
from run_w3_survival import TEST_FIXTURE_LABEL, W3SurvivalConfig, run_w3_survival, write_w3_fixture_survivor_root


def test_frozen_w01_bundle_restores_complete_timing_controller(tmp_path: Path) -> None:
    source_root = _write_w01_source_fixture(tmp_path / "w01", "glide", include_augmented_payload=True)

    bundle = load_frozen_w01_controller_bundle(source_root / "manifests" / "frozen_w01_controller_bundle.json")
    record = bundle.records[0]

    assert record.bundle_status == FROZEN_CONTROLLER_READY
    assert record.controller.controller_id == record.variant.controller_id
    assert record.controller.lqr_gain_checksum == record.variant.K_gain_checksum
    assert record.controller.augmented_gain_checksum == record.variant.augmented_gain_checksum
    assert record.controller.augmented_gain_matrix_json
    assert record.controller.predictor_A_reduced_json
    assert record.controller.augmented_A_matrix_json
    assert record.controller.augmented_B_matrix_json


def test_missing_augmented_payload_blocks_instead_of_using_physical_k_only(tmp_path: Path) -> None:
    source_root = _write_w01_source_fixture(tmp_path / "w01", "glide", include_augmented_payload=False)

    bundle = load_frozen_w01_controller_bundle(source_root / "manifests" / "frozen_w01_controller_bundle.json")
    record = bundle.records[0]

    assert record.bundle_status == FROZEN_CONTROLLER_BLOCKED
    assert "missing_augmented_gain_matrix_json" in record.blocked_reason
    assert "missing_predictor_A_reduced_json" in record.blocked_reason
    assert "missing_augmented_A_matrix_json" in record.blocked_reason
    assert "missing_augmented_B_matrix_json" in record.blocked_reason
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


def test_w2_role_aware_schedule_avoids_inflight_launch_gate_rejections(tmp_path: Path) -> None:
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

    assert set(frame["start_state_family"]) == {"inflight_nominal"}
    assert not frame["entry_check_status"].astype(str).eq(ENTRY_ROLE_REJECTION_STATUS).any()
    assert not frame["failure_label"].astype(str).eq(ENTRY_ROLE_REJECTION_LABEL).any()
    assert int(variant_summary["compatible_row_count"].iloc[0]) == 2


def test_w2_blocks_w01_roots_without_emitted_frozen_bundle(tmp_path: Path) -> None:
    source_root = tmp_path / "w01_without_bundle"
    (source_root / "manifests").mkdir(parents=True)
    (source_root / "manifests" / "run_manifest.json").write_text('{"run_id": 8}\n', encoding="ascii")

    result = run_w2_survival(
        W2SurvivalConfig(
            run_id=12,
            input_root=source_root,
            output_root=tmp_path / "w2",
            workers=1,
            max_workers=1,
            storage_format="csv_gz",
        )
    )
    manifest = (Path(result["run_root"]) / "manifests" / "w2_survival_manifest.json").read_text(encoding="ascii")

    assert result["status"] == "blocked"
    assert "missing_W01_frozen_controller_bundle" in manifest


def test_w3_executes_from_valid_w2_survivor_registry_and_frozen_bundle(tmp_path: Path) -> None:
    source_root = _write_w01_source_fixture(tmp_path / "w01", "glide", include_augmented_payload=True)
    bundle = load_frozen_w01_controller_bundle(source_root / "manifests" / "frozen_w01_controller_bundle.json")
    record = bundle.records[0]
    w2_root = tmp_path / "w2_survivor_source"
    manifests = w2_root / "manifests"
    manifests.mkdir(parents=True)
    (manifests / "frozen_w01_controller_bundle.json").write_text(
        (source_root / "manifests" / "frozen_w01_controller_bundle.json").read_text(encoding="ascii"),
        encoding="ascii",
    )
    (manifests / "w2_survival_manifest.json").write_text(
        json.dumps(
            {
                "status": "w2_dense_survival_pass",
                "project_title_version": "LQR-Stabilised Contextual Primitive v5.3",
                "primitive_timing_contract": primitive_timing_contract_row(),
                "method_evidence_level": "w2_dense_survival_pass",
                "w2_dense_survival_evidence_complete": True,
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    survivor = variant_row(record.variant)
    survivor.update({"w2_variant_status": "survived", "eligible_for_w3": True})
    (manifests / "w2_survivor_registry.json").write_text(
        json.dumps(
            {
                "survivor_registry_version": "w2_survivor_registry_v1",
                "status": "survived_variants_available",
                "survivor_count": 1,
                "survivors": [survivor],
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )

    result = run_w3_survival(
        W3SurvivalConfig(
            run_id=13,
            input_root=w2_root,
            output_root=tmp_path / "w3",
            paired_tests_per_variant=1,
            storage_format="csv_gz",
        )
    )
    frame = _read_frame(Path(result["run_root"]))

    assert result["status"] == "complete"
    assert len(frame) == 2
    assert set(frame["environment_mode"]) == {"w3_randomised_single", "w3_randomised_four"}
    assert frame["fixed_lqr_replay_only"].astype(bool).all()


def test_w2_default_discovery_requires_dense_method_evidence(tmp_path: Path) -> None:
    discovery_root = tmp_path / "w01_dense"
    sparse_root = _write_w01_source_fixture(discovery_root / "012", "glide", include_augmented_payload=True)
    ready_root = _write_w01_source_fixture(discovery_root / "014", "glide", include_augmented_payload=True)
    (sparse_root / "manifests" / "run_manifest.json").write_text(
        json.dumps(
            {
                    "run_id": 12,
                    "rows_requested": 240,
                    "cross_layer_smoke_status": "artifact_smoke_only_start_family_incomplete",
                    "project_title_version": "LQR-Stabilised Contextual Primitive v5.3",
                    "primitive_timing_contract": primitive_timing_contract_row(),
                    "method_evidence_level": "w01_smoke_or_preflight_only",
                    "w01_dense_evidence_complete": False,
                },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    (ready_root / "manifests" / "run_manifest.json").write_text(
        json.dumps(
            {
                    "run_id": 14,
                    "rows_requested": 960,
                    "cross_layer_smoke_status": "cross_layer_smoke_start_family_complete",
                    "project_title_version": "LQR-Stabilised Contextual Primitive v5.3",
                    "primitive_timing_contract": primitive_timing_contract_row(),
                    "method_evidence_level": "w01_dense_evidence_complete",
                    "w01_dense_evidence_complete": True,
                    "W2_W3_replay_only": True,
                    "no_clustering_before_W2_W3": True,
                },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )

    discovered = discover_latest_w01_root_for_w2(discovery_root)
    assert discovered is not None
    assert _normal_path_text(discovered) == _normal_path_text(ready_root)


def test_w3_fixture_plumbing_is_labelled_non_method_evidence(tmp_path: Path) -> None:
    source_root = _write_w01_source_fixture(tmp_path / "w01", "glide", include_augmented_payload=True)
    fixture_root = write_w3_fixture_survivor_root(
        fixture_root=tmp_path / "fixture_w2_survivor",
        source_w01_root=source_root,
    )

    result = run_w3_survival(
        W3SurvivalConfig(
            run_id=14,
            input_root=fixture_root,
            output_root=tmp_path / "w3",
            paired_tests_per_variant=1,
            storage_format="csv_gz",
        )
    )
    run_root = Path(result["run_root"])
    manifest = json.loads((run_root / "manifests" / "w3_survival_manifest.json").read_text(encoding="ascii"))
    frame = _read_frame(run_root)

    assert result["status"] == "complete"
    assert manifest["test_fixture_not_method_evidence"] is True
    assert set(frame["source_evidence_label"]) == {TEST_FIXTURE_LABEL}
    assert set(frame["claim_boundary"]) == {TEST_FIXTURE_LABEL}


def test_w3_balances_active_fan_count_for_four_fan_randomisation(tmp_path: Path) -> None:
    w01_root = _write_w01_source_fixture(tmp_path / "w01" / "024", "glide", include_augmented_payload=True)
    w2_root = tmp_path / "w2" / "017"
    manifests = w2_root / "manifests"
    manifests.mkdir(parents=True)
    shutil.copyfile(
        w01_root / "manifests" / "frozen_w01_controller_bundle.json",
        manifests / "frozen_w01_controller_bundle.json",
    )
    (manifests / "w2_survival_manifest.json").write_text(
        json.dumps(
            {
                "status": "w2_dense_survival_pass",
                "project_title_version": "LQR-Stabilised Contextual Primitive v5.3",
                "primitive_timing_contract": primitive_timing_contract_row(),
                "method_evidence_level": "w2_dense_survival_pass",
                "w2_dense_survival_evidence_complete": True,
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    bundle = load_frozen_w01_controller_bundle(manifests / "frozen_w01_controller_bundle.json")
    survivor = variant_row(bundle.records[0].variant)
    survivor.update({"w2_variant_status": "survived", "eligible_for_w3": True})
    (manifests / "w2_survivor_registry.json").write_text(
        json.dumps(
            {
                "survivor_registry_version": "w2_survivor_registry_v1",
                "status": "survived_variants_available",
                "survivor_count": 1,
                "survivors": [survivor],
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )

    result = run_w3_survival(
        W3SurvivalConfig(
            run_id=15,
            input_root=w2_root,
            output_root=tmp_path / "w3",
            paired_tests_per_variant=4,
            storage_format="csv_gz",
        )
    )
    audit = pd.read_csv(Path(result["run_root"]) / "metrics" / "w3_active_fan_count_audit.csv")

    four = audit[audit["environment_mode"].eq("w3_randomised_four")]
    assert set(four["active_fan_count"].astype(int)) == {1, 2, 3, 4}
    assert set(four["row_count"].astype(int)) == {1}


def test_w2_default_schedule_dimensions() -> None:
    assert W2_DENSE_VARIANT_COUNT == 14 * 32
    assert W2_DENSE_ROW_COUNT == 14 * 32 * 2 * 100


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
        json.dumps(
            {
                "run_id": 8,
                "status": "fixture",
                "project_title_version": "LQR-Stabilised Contextual Primitive v5.3",
                "cross_layer_smoke_status": "cross_layer_smoke_start_family_complete",
                "primitive_timing_contract": primitive_timing_contract_row(),
                "method_evidence_level": "w01_dense_evidence_complete",
                "w01_dense_evidence_complete": True,
                "W2_W3_replay_only": True,
                "no_clustering_before_W2_W3": True,
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    write_table_manifest(
        manifests / "table_manifest.json",
        TableManifest(run_id=8, root=root.as_posix(), storage_format="csv_gz", tables=()),
    )
    write_frozen_w01_controller_bundle(run_root=root, source_records=((variant, controller),))
    if not include_augmented_payload:
        bundle_path = manifests / "frozen_w01_controller_bundle.json"
        payload = json.loads(bundle_path.read_text(encoding="ascii"))
        controller_payload = payload["records"][0]["controller_payload"]
        controller_payload["augmented_gain_matrix_json"] = ""
        controller_payload["predictor_A_reduced_json"] = ""
        controller_payload["augmented_A_matrix_json"] = ""
        controller_payload["augmented_B_matrix_json"] = ""
        bundle_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    return root


def _read_w2_frame(run_root: Path) -> pd.DataFrame:
    return _read_frame(run_root)


def _read_frame(run_root: Path) -> pd.DataFrame:
    manifest = load_table_manifest(run_root / "manifests" / "table_manifest.json")
    return pd.concat(
        [
            read_table_partition(run_root / "tables" / partition.relative_path, storage_format=partition.storage_format)
            for partition in manifest.tables
        ],
        ignore_index=True,
    )


def _normal_path_text(path: Path) -> str:
    return str(Path(path)).replace("\\\\?\\", "").replace("\\", "/")
