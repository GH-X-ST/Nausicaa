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


def _archive_row() -> dict[str, object]:
    state = np.zeros(STATE_SIZE)
    state[STATE_INDEX["x_w"]] = 2.0
    state[STATE_INDEX["y_w"]] = 2.0
    state[STATE_INDEX["z_w"]] = 1.6
    state[STATE_INDEX["u"]] = 5.8
    context = build_environment_context(
        state,
        wind_field=None,
        metadata=EnvironmentMetadata(environment_id="W0_report", fan_count=0),
        latency_case="none",
    )
    primitive = primitive_by_id("glide")
    row = {
        "rollout_id": "row_0",
        "primitive_id": "glide",
        "evidence_role": "feedback_rollout_candidate",
        "outcome_class": "accepted",
        "continuation_status": "continuation_success",
        "episode_terminal_status": "not_terminal",
        "episode_utility_label": "continuation_useful",
        "terminal_use_trainable": False,
        "energy_residual_m": 0.1,
        "lift_dwell_time_s": 0.1,
        "minimum_wall_margin_m": 1.0,
        "termination_cause": "controlled_finish",
    }
    row.update({f"context_{key}": value for key, value in environment_context_row(context).items()})
    sample = archive_state_sample_for_row(0, seed=1, W_layer="W0", environment_mode="dry_air")
    row.update(archive_state_sample_row(sample))
    row.update({f"initial_{name}": float(state[index]) for index, name in enumerate(STATE_NAMES)})
    row.update(primitive_feature_row(primitive_feature_record(state=state, context=context, primitive=primitive)))
    return row


def test_selector_and_replay_scaffolds_write_temp_manifests(tmp_path: Path) -> None:
    archive_table = tmp_path / "archive_rows.csv"
    pd.DataFrame([_archive_row()]).to_csv(archive_table, index=False)

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

    assert selector_manifest["claim_status"] == "simulation_only_selector_report_smoke"
    assert w2_manifest["R8_W2_replay_complete"] is False
    assert w2_manifest["replay_status"] == "mixed_start_w2_replay_smoke_from_source"
    assert w3_manifest["R9_W3_generalisation_complete"] is False
    assert w3_manifest["generalisation_status"] == "partial_smoke_from_source_no_robustness_claim"

    selector_rows = pd.read_csv(selector["decision_table"])
    w2_rows = pd.read_csv(w2["replay_table"])
    w3_rows = pd.read_csv(w3["case_table"])
    for frame in (selector_rows, w2_rows, w3_rows):
        for name in STATE_NAMES:
            assert f"entry_{name}" in frame.columns
    assert "validation_split_columns" in selector_manifest
    assert "environment_adjustment_status" in w3_manifest
