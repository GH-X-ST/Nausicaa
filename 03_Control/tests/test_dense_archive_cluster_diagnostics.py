from __future__ import annotations

import pandas as pd

from dense_archive_cluster_diagnostics import (
    CLUSTER_DIAGNOSTIC_COLUMNS,
    DenseClusterDiagnosticConfig,
    build_cluster_diagnostics,
)


def test_empty_input_returns_exact_schema() -> None:
    result = build_cluster_diagnostics(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    assert tuple(result.columns) == CLUSTER_DIAGNOSTIC_COLUMNS
    assert result.empty


def test_balanced_fake_clusters_are_baseline_sufficient() -> None:
    result = build_cluster_diagnostics(
        _trials({"single_fan_branch": 60, "four_fan_branch": 60}),
        _envelope({"single_fan_branch": [30, 30], "four_fan_branch": [30, 30]}),
        _representatives({"single_fan_branch": 2, "four_fan_branch": 2}, role="success_representative"),
        DenseClusterDiagnosticConfig(min_trials_total=100),
    )

    assert set(result["clustering_strategy_status"]) == {
        "baseline_sufficient_for_w0_inspection"
    }


def test_imbalanced_branch_fake_data_needs_cluster_quota_strategy() -> None:
    result = build_cluster_diagnostics(
        _trials({"single_fan_branch": 100, "four_fan_branch": 20}),
        _envelope({"single_fan_branch": [50, 50], "four_fan_branch": [10, 10]}),
        _representatives({"single_fan_branch": 2, "four_fan_branch": 2}, role="success_representative"),
        DenseClusterDiagnosticConfig(min_trials_total=100),
    )

    assert "needs_cluster_quota_strategy" in set(result["clustering_strategy_status"])


def test_mixed_boundary_cells_with_too_few_boundary_representatives_are_flagged() -> None:
    result = build_cluster_diagnostics(
        _trials({"single_fan_branch": 60, "four_fan_branch": 60}),
        _envelope(
            {"single_fan_branch": [30, 30], "four_fan_branch": [30, 30]},
            cell_status="mixed_boundary",
        ),
        _representatives({"single_fan_branch": 2, "four_fan_branch": 2}, role="success_representative"),
        DenseClusterDiagnosticConfig(min_trials_total=100),
    )

    assert set(result["clustering_strategy_status"]) == {
        "needs_boundary_augmented_strategy"
    }


def test_large_cluster_fake_data_needs_adaptive_binning_strategy() -> None:
    result = build_cluster_diagnostics(
        _trials({"single_fan_branch": 600, "four_fan_branch": 600}),
        _envelope({"single_fan_branch": [300, 300], "four_fan_branch": [300, 300]}),
        _representatives({"single_fan_branch": 2, "four_fan_branch": 2}, role="success_representative"),
        DenseClusterDiagnosticConfig(min_trials_total=100),
    )

    assert set(result["clustering_strategy_status"]) == {
        "needs_adaptive_binning_strategy"
    }


def test_reduced_w0_fake_run_is_insufficient_for_clustering_decision() -> None:
    result = build_cluster_diagnostics(
        _trials({"single_fan_branch": 10, "four_fan_branch": 10}),
        _envelope({"single_fan_branch": [10], "four_fan_branch": [10]}),
        _representatives({"single_fan_branch": 1, "four_fan_branch": 1}, role="success_representative"),
        DenseClusterDiagnosticConfig(min_trials_total=100),
    )

    assert set(result["clustering_strategy_status"]) == {
        "insufficient_data_for_clustering_decision"
    }


def _trials(counts_by_branch: dict[str, int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for branch, count in counts_by_branch.items():
        fan, mode = _branch_labels(branch)
        for index in range(int(count)):
            rows.append(
                {
                    "trial_descriptor_id": f"{branch}_trial_{index:04d}",
                    "layout_branch_id": branch,
                    "fan_layout": fan,
                    "test_environment_mode": mode,
                    "family": "mild_bank",
                    "target_heading_deg": 30.0,
                    "direction_sign": 1,
                    "start_class": "favourable",
                    "success_flag": True,
                    "descriptor_status": "replay_evaluated",
                }
            )
    return pd.DataFrame(rows)


def _envelope(
    sizes_by_branch: dict[str, list[int]],
    *,
    cell_status: str = "all_success",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for branch, sizes in sizes_by_branch.items():
        fan, mode = _branch_labels(branch)
        for index, size in enumerate(sizes):
            rows.append(
                {
                    "envelope_cell_id": f"cell|branch={branch}|idx={index}",
                    "layout_branch_id": branch,
                    "fan_layout": fan,
                    "test_environment_mode": mode,
                    "trial_count": int(size),
                    "success_count": int(size) if cell_status == "all_success" else 0,
                    "success_fraction": 1.0 if cell_status == "all_success" else 0.0,
                    "cell_status": cell_status,
                }
            )
    return pd.DataFrame(rows)


def _representatives(
    counts_by_branch: dict[str, int],
    *,
    role: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for branch, count in counts_by_branch.items():
        fan, mode = _branch_labels(branch)
        for index in range(int(count)):
            rows.append(
                {
                    "cluster_key": f"cluster|branch={branch}|idx={index}",
                    "trial_descriptor_id": f"{branch}_rep_{index:04d}",
                    "layout_branch_id": branch,
                    "fan_layout": fan,
                    "test_environment_mode": mode,
                    "candidate_role": role,
                    "start_class": "favourable",
                }
            )
    return pd.DataFrame(rows)


def _branch_labels(branch: str) -> tuple[str, str]:
    if branch == "single_fan_branch":
        return "single_fan", "W0_single_fan_branch"
    return "four_fan", "W0_four_fan_branch"
