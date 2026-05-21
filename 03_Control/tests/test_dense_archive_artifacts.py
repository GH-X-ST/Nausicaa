from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dense_archive_artifacts import ensure_no_raw_tables, write_governor_branch_package


def test_governor_branch_package_metadata_is_compact_w1_only_contract(tmp_path: Path) -> None:
    metadata = write_governor_branch_package(
        root=tmp_path,
        fan_layout="single_fan",
        environment_mode="W1_single_fan",
        envelope_cells=pd.DataFrame([{"envelope_cell_id": "cell_1"}]),
        candidate_representatives=pd.DataFrame(
            [{"candidate_id": "w1_candidate", "test_environment_mode": "W1_single_fan"}]
        ),
        viability_thresholds=pd.DataFrame([{"required_latency_case": "nominal"}]),
        latency_metadata=pd.DataFrame([{"latency_case": "nominal"}]),
        model_ids=pd.DataFrame([{"updraft_model_id": "single_gaussian_var"}]),
        worker_profile_metadata={"selected_worker_count": 8},
        storage_format="csv_gz",
    )

    path = tmp_path / "single_fan" / "W1_single_fan_governor_metadata.json"
    payload = json.loads(path.read_text(encoding="ascii"))
    assert metadata["raw_tables_included"] is False
    assert payload["governor_artifacts_scan_raw_tables"] is False
    assert payload["governor_package_contains_w0_candidates"] is False
    assert payload["governor_package_branch_local_only"] is True
    ensure_no_raw_tables(tmp_path)
