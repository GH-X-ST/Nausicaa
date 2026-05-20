from __future__ import annotations

import pandas as pd

from dense_archive_clustering import (
    CLUSTER_REPRESENTATIVE_COLUMNS,
    DenseClusterConfig,
    cluster_key,
    select_cluster_representatives,
)
from dense_archive_envelope_maps import (
    ENVELOPE_MAP_COLUMNS,
    EnvelopeMapConfig,
    build_envelope_map,
    envelope_cell_id,
)


EXPECTED_COLUMNS = (
    "cluster_key",
    "representative_rank",
    "trial_descriptor_id",
    "sim_real_match_key",
    "layout_branch_id",
    "fan_layout",
    "test_environment_mode",
    "family",
    "target_heading_deg",
    "direction_sign",
    "start_class",
    "latency_case",
    "success_flag",
    "failure_label",
    "governor_rejection_cause",
    "robustness_label",
    "heading_error_deg",
    "energy_residual_m",
    "lift_dwell_fraction",
    "min_true_margin_m",
    "saturation_fraction",
    "w_wing_mean_m_s",
    "delta_w_lr_m_s",
    "physics_priority_level",
    "candidate_role",
    "branch_decision_scope",
)


def test_empty_input_returns_exact_schema() -> None:
    result = select_cluster_representatives(pd.DataFrame())

    assert CLUSTER_REPRESENTATIVE_COLUMNS == EXPECTED_COLUMNS
    assert tuple(result.columns) == EXPECTED_COLUMNS
    assert result.empty


def test_cluster_key_reuses_envelope_cell_bins() -> None:
    row = _row()

    assert cluster_key(row) == envelope_cell_id(row).replace("cell|", "cluster|", 1)


def test_cluster_key_reuses_custom_envelope_cell_bins() -> None:
    row = _row()
    config = EnvelopeMapConfig(radius_bin_width_m=1.0)

    assert cluster_key(row, config) == envelope_cell_id(row, config).replace(
        "cell|",
        "cluster|",
        1,
    )
    assert cluster_key(row, config) != cluster_key(row)


def test_representative_roles_and_sort_order_are_deterministic() -> None:
    rows = [
        _row(
            trial_descriptor_id="mixed_success",
            success_flag=True,
            failure_label="success",
            heading_error_deg=4.0,
        ),
        _row(
            trial_descriptor_id="mixed_failure",
            success_flag=False,
            failure_label="target_miss",
            heading_error_deg=1.0,
        ),
        _row(
            trial_descriptor_id="all_success",
            start_class="mid_arena",
            success_flag=True,
            failure_label="success",
            heading_error_deg=2.0,
            energy_residual_m=0.5,
            min_true_margin_m=0.7,
            saturation_fraction=0.0,
            lift_dwell_fraction=0.9,
        ),
        _row(
            trial_descriptor_id="all_failure",
            start_class="random_stress",
            success_flag=False,
            failure_label="true_safety_violation",
            heading_error_deg=9.0,
        ),
    ]
    frame = pd.DataFrame(rows)
    envelope = build_envelope_map(frame)
    result = select_cluster_representatives(frame, envelope)

    assert tuple(result.columns) == EXPECTED_COLUMNS
    by_id = {row["trial_descriptor_id"]: row for row in result.to_dict(orient="records")}
    assert by_id["mixed_success"]["candidate_role"] == "boundary_representative"
    assert by_id["mixed_failure"]["candidate_role"] == "boundary_representative"
    assert by_id["all_success"]["candidate_role"] == "success_representative"
    assert by_id["all_failure"]["candidate_role"] == "failure_representative"

    mixed = result[result["cluster_key"] == cluster_key(rows[0])]
    assert list(mixed["trial_descriptor_id"]) == ["mixed_success", "mixed_failure"]
    assert list(mixed["representative_rank"]) == [1, 2]


def test_failure_representatives_can_be_excluded() -> None:
    frame = pd.DataFrame(
        [
            _row(
                trial_descriptor_id="failure",
                success_flag=False,
                failure_label="target_miss",
            )
        ]
    )
    result = select_cluster_representatives(
        frame,
        build_envelope_map(frame),
        DenseClusterConfig(include_failure_representatives=False),
    )

    assert result.empty
    assert tuple(result.columns) == EXPECTED_COLUMNS


def test_not_replayed_representatives_are_explicit_diagnostic_role() -> None:
    frame = pd.DataFrame(
        [
            _row(
                trial_descriptor_id="diagnostic",
                descriptor_status="not_replayed",
                success_flag=False,
                failure_label="not_replayed",
            )
        ]
    )
    result = select_cluster_representatives(frame, build_envelope_map(frame))

    assert list(result["candidate_role"]) == ["not_replayed_representative"]


def test_not_replayed_representatives_can_be_excluded() -> None:
    frame = pd.DataFrame(
        [
            _row(
                trial_descriptor_id="diagnostic",
                descriptor_status="synthetic_descriptor_only",
                success_flag=False,
                failure_label="not_replayed",
            )
        ]
    )
    result = select_cluster_representatives(
        frame,
        build_envelope_map(frame),
        DenseClusterConfig(include_failure_representatives=False),
    )

    assert result.empty
    assert tuple(result.columns) == EXPECTED_COLUMNS


def test_not_replayed_representatives_sort_after_evaluated_failures() -> None:
    rows = [
        _row(
            trial_descriptor_id="failure",
            descriptor_status="replay_evaluated",
            success_flag=False,
            failure_label="target_miss",
        ),
        _row(
            trial_descriptor_id="diagnostic",
            descriptor_status="not_replayed",
            success_flag=False,
            failure_label="not_replayed",
        ),
    ]
    frame = pd.DataFrame(rows)
    envelope = _manual_envelope_for_cell(
        envelope_cell_id(rows[0]),
        cell_status="all_failure",
        success_fraction=0.0,
    )
    result = select_cluster_representatives(frame, envelope)

    assert list(result["trial_descriptor_id"]) == ["failure", "diagnostic"]
    assert list(result["candidate_role"]) == [
        "failure_representative",
        "not_replayed_representative",
    ]
    assert list(result["representative_rank"]) == [1, 2]


def test_custom_envelope_config_controls_internal_maps_and_joins() -> None:
    rows = [
        _row(
            trial_descriptor_id="near",
            updraft_relative_radius_m=0.2,
            success_flag=True,
            failure_label="success",
        ),
        _row(
            trial_descriptor_id="far",
            updraft_relative_radius_m=0.8,
            success_flag=False,
            failure_label="target_miss",
        ),
    ]
    frame = pd.DataFrame(rows)
    config = EnvelopeMapConfig(radius_bin_width_m=1.0)

    default_keys = {cluster_key(row) for row in rows}
    custom_keys = {cluster_key(row, config) for row in rows}
    internal = select_cluster_representatives(frame, envelope_config=config)
    supplied = select_cluster_representatives(
        frame,
        build_envelope_map(frame, config),
        envelope_config=config,
    )

    assert len(default_keys) == 2
    assert len(custom_keys) == 1
    assert list(internal["cluster_key"].unique()) == list(custom_keys)
    assert list(supplied["cluster_key"].unique()) == list(custom_keys)
    assert set(internal["candidate_role"]) == {"boundary_representative"}
    assert set(supplied["candidate_role"]) == {"boundary_representative"}


def _manual_envelope_for_cell(
    cell_id: str,
    *,
    cell_status: str,
    success_fraction: float,
) -> pd.DataFrame:
    row = {column: "" for column in ENVELOPE_MAP_COLUMNS}
    row.update(
        {
            "envelope_cell_id": cell_id,
            "trial_count": 2,
            "evaluated_trial_count": 2,
            "success_count": 0,
            "success_fraction": success_fraction,
            "cell_status": cell_status,
        }
    )
    return pd.DataFrame([row], columns=ENVELOPE_MAP_COLUMNS)


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "trial_descriptor_id": "trial_001",
        "sim_real_match_key": "match_001",
        "layout_branch_id": "single_fan_branch",
        "fan_layout": "single_fan",
        "test_environment_mode": "W1_single_fan",
        "family": "mild_bank",
        "target_heading_deg": 30.0,
        "direction_sign": 1,
        "start_class": "favourable",
        "latency_case": "nominal",
        "updraft_relative_radius_m": 0.55,
        "speed0_m_s": 6.1,
        "w_wing_mean_m_s": 0.12,
        "delta_w_lr_m_s": 0.02,
        "min_true_margin_m": 0.45,
        "descriptor_status": "replay_evaluated",
        "success_flag": False,
        "failure_label": "target_miss",
        "governor_rejection_cause": "none",
        "robustness_label": "not_evaluated",
        "heading_error_deg": 1.0,
        "energy_residual_m": 0.2,
        "lift_dwell_fraction": 0.4,
        "saturation_fraction": 0.1,
        "latency_pass_label": "nominal_pass",
        "physics_priority_level": "target_steering_30deg_priority",
        "branch_decision_scope": "branch_local_only_no_cross_layout_decision_transfer",
    }
    row.update(overrides)
    return row
