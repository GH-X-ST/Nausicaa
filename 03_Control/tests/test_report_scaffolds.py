from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from env_ctx import EnvironmentMetadata, build_environment_context, environment_context_row
from prim_cat import primitive_by_id
from prim_features import primitive_feature_record, primitive_feature_row
from run_primitive_selector_report import SelectorReportConfig, run_primitive_selector_report
from run_w2_replay import W2ReplayConfig, run_w2_replay_scaffold
from run_w3_generalisation import W3GeneralisationConfig, run_w3_generalisation_scaffold
from state_contract import STATE_INDEX, STATE_NAMES, STATE_SIZE
from state_sampling import archive_state_sample_for_row, archive_state_sample_row


def _archive_row(
    *,
    outcome_class: str = "accepted",
    start_row: int = 0,
    primitive_id: str = "glide",
) -> dict[str, object]:
    state = np.zeros(STATE_SIZE)
    state[STATE_INDEX["x_w"]] = 2.0
    state[STATE_INDEX["y_w"]] = 2.0
    state[STATE_INDEX["z_w"]] = 1.6
    state[STATE_INDEX["u"]] = 5.8
    context = build_environment_context(
        state,
        wind_field=None,
        metadata=EnvironmentMetadata(
            environment_id="W1_report",
            fan_count=1,
            W_layer="W1",
            wind_mode="panel",
            environment_mode="gaussian_single",
            environment_instance_id="W1_report",
            updraft_model_id="single_gaussian_var",
        ),
        latency_case="nominal",
    )
    primitive = primitive_by_id(primitive_id)
    row = {
        "rollout_id": f"row_{start_row}_{outcome_class}",
        "primitive_id": primitive_id,
        "evidence_role": "feedback_rollout_candidate",
        "outcome_class": outcome_class,
        "continuation_status": "not_continuation_valid" if outcome_class == "boundary_terminal" else "continuation_success",
        "episode_terminal_status": "boundary_terminal" if outcome_class == "boundary_terminal" else "not_terminal",
        "episode_utility_label": "terminal_useful" if outcome_class == "boundary_terminal" else "continuation_useful",
        "terminal_use_trainable": outcome_class == "boundary_terminal",
        "energy_residual_m": 0.1,
        "lift_dwell_time_s": 0.1,
        "minimum_wall_margin_m": 1.0,
        "termination_cause": "controlled_finish",
        "W_layer": "W1",
        "latency_case": "nominal",
        "boundary_use_class": "episode_terminal_useful_or_trainable" if outcome_class == "boundary_terminal" else "continuation_valid",
        "environment_mode": "gaussian_single",
    }
    row.update({f"context_{key}": value for key, value in environment_context_row(context).items()})
    sample = archive_state_sample_for_row(start_row, seed=1, W_layer="W1", environment_mode="gaussian_single")
    row.update(archive_state_sample_row(sample))
    row.update({f"initial_{name}": float(state[index]) for index, name in enumerate(STATE_NAMES)})
    row.update(primitive_feature_row(primitive_feature_record(state=state, context=context, primitive=primitive)))
    return row


def test_selector_and_replay_scaffolds_write_temp_manifests(tmp_path: Path) -> None:
    archive_table = tmp_path / "archive_rows.csv"
    rows = [
        _archive_row(outcome_class=outcome, start_row=index, primitive_id="glide")
        for index, outcome in enumerate(("accepted", "weak", "boundary_terminal", "failed", "rejected"))
    ]
    pd.DataFrame(rows).to_csv(archive_table, index=False)

    selector = run_primitive_selector_report(
        SelectorReportConfig(
            run_id=81,
            archive_table=archive_table,
            output_root=tmp_path,
        )
    )
    w2 = run_w2_replay_scaffold(
        W2ReplayConfig(run_id=82, output_root=tmp_path, source_archive=archive_table)
    )
    w3 = run_w3_generalisation_scaffold(
        W3GeneralisationConfig(run_id=83, output_root=tmp_path, source_replay=archive_table)
    )

    selector_manifest = json.loads(Path(selector["manifest"]).read_text())
    w2_manifest = json.loads(Path(w2["manifest"]).read_text())
    w3_manifest = json.loads(Path(w3["manifest"]).read_text())

    assert selector_manifest["claim_status"] == "simulation_only_selector_report_smoke_or_subset"
    assert w2_manifest["R8_W2_replay_complete"] is False
    assert w2_manifest["actual_model_backed_replay"] is True
    assert w2_manifest["replayed_row_count"] > 0
    assert w3_manifest["R9_W3_generalisation_complete"] is False
    assert w3_manifest["actual_model_backed_replay"] is True

    selector_rows = pd.read_csv(selector["decision_table"])
    w2_rows = pd.read_csv(w2["replay_table"])
    w3_rows = pd.read_csv(w3["case_table"])
    for frame in (selector_rows, w2_rows, w3_rows):
        for name in STATE_NAMES:
            assert f"entry_{name}" in frame.columns
    assert "validation_split_columns" in selector_manifest
    assert set(w2_rows["replay_generation_path"]) == {"simulate_primitive_rollout"}
    assert not w2_rows["source_label_copied_as_evidence"].astype(bool).any()
    assert set(w3_rows["replay_generation_path"]) == {"simulate_primitive_rollout"}
    assert "approximate_limitation_label" in w3_rows.columns


def test_w2_w3_replay_scaffolds_write_chunked_partitions(tmp_path: Path) -> None:
    archive_table = tmp_path / "archive_rows.csv"
    rows = [
        _archive_row(outcome_class=outcome, start_row=index, primitive_id="glide")
        for index, outcome in enumerate(("accepted", "weak", "boundary_terminal", "failed", "rejected"))
    ]
    pd.DataFrame(rows).to_csv(archive_table, index=False)

    w2 = run_w2_replay_scaffold(
        W2ReplayConfig(
            run_id=84,
            output_root=tmp_path,
            source_archive=archive_table,
            target_rows=6,
            fallback_rows=2,
            chunk_size=2,
            storage_format="csv_gz",
        )
    )
    w3 = run_w3_generalisation_scaffold(
        W3GeneralisationConfig(
            run_id=85,
            output_root=tmp_path,
            source_replay=w2["table_manifest"],
            target_rows=6,
            fallback_rows=2,
            chunk_size=2,
            storage_format="csv_gz",
        )
    )

    w2_table_manifest = json.loads(Path(w2["table_manifest"]).read_text())
    w3_table_manifest = json.loads(Path(w3["table_manifest"]).read_text())
    assert len(w2_table_manifest["tables"]) >= 2
    assert len(w3_table_manifest["tables"]) >= 2
    assert all(row["checksum_sha256"] for row in w2_table_manifest["tables"])
    assert all(row["checksum_sha256"] for row in w3_table_manifest["tables"])
    assert (Path(w2["run_root"]) / "metrics" / "chunk_summary.csv").is_file()
    assert (Path(w3["run_root"]) / "metrics" / "chunk_summary.csv").is_file()
