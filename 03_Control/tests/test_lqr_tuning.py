from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from controller_registry import controller_from_evidence_row
from dense_archive_table_io import load_table_manifest, read_table_partition
from prim_cat import ACTIVE_PRIMITIVE_IDS
from run_lqr_tuning_sweep import LQRTuningSweepConfig, run_lqr_tuning_sweep


def test_lqr_tuning_rolls_out_candidate_controller_ids_and_writes_registry(tmp_path: Path) -> None:
    result = run_lqr_tuning_sweep(
        LQRTuningSweepConfig(
            run_id=91,
            output_root=tmp_path,
            rows=32,
            seed=91,
            candidate_count=2,
            paired_tests_per_candidate=1,
            candidate_chunk_size=32,
            workers=1,
            max_workers=1,
            storage_format="csv_gz",
            compression_level=1,
        )
    )

    run_root = Path(result["run_root"])
    table_manifest = load_table_manifest(run_root / "manifests" / "table_manifest.json")
    frame = read_table_partition(
        run_root / "tables" / table_manifest.tables[0].relative_path,
        storage_format="csv_gz",
    )
    registry = pd.read_csv(result["selected_controller_registry"])

    assert "candidate_weight_label" in frame.columns
    assert "controller_selection_status" in frame.columns
    assert set(frame["controller_selection_status"]) == {"W0_W1_candidate_rollout"}
    assert frame.groupby("primitive_id")["controller_id"].nunique().ge(2).all()
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
