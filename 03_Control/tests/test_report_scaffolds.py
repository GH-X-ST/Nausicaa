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
from state_contract import STATE_INDEX, STATE_SIZE


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
    row.update({f"initial_{name}": float(state[index]) for index, name in enumerate(("x_w", "y_w", "z_w", "phi", "theta", "psi", "u", "v", "w", "p", "q", "r", "delta_a", "delta_e", "delta_r"))})
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
    w2 = run_w2_replay_scaffold(W2ReplayConfig(run_id=82, output_root=tmp_path))
    w3 = run_w3_generalisation_scaffold(W3GeneralisationConfig(run_id=83, output_root=tmp_path))

    selector_manifest = json.loads(Path(selector["manifest"]).read_text())
    w2_manifest = json.loads(Path(w2["manifest"]).read_text())
    w3_manifest = json.loads(Path(w3["manifest"]).read_text())

    assert selector_manifest["claim_status"] == "simulation_only_selector_report_smoke"
    assert w2_manifest["replay_status"] == "blocked_until_approved_R6_archive_exists"
    assert w3_manifest["generalisation_status"] == "blocked_until_W2_supported_cases_exist"
