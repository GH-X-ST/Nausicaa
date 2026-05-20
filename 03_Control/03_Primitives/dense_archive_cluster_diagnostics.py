from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Diagnostics Constants and Configuration
# 2) Public Diagnostics Builder
# 3) Aggregation Helpers
# 4) Conversion Helpers
# =============================================================================


# =============================================================================
# 1) Diagnostics Constants and Configuration
# =============================================================================
CLUSTER_DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "layout_branch_id",
    "fan_layout",
    "test_environment_mode",
    "trial_count",
    "cluster_count",
    "representative_count",
    "success_representative_count",
    "boundary_representative_count",
    "failure_representative_count",
    "not_replayed_representative_count",
    "success_trial_count",
    "failure_trial_count",
    "not_replayed_trial_count",
    "cluster_size_min",
    "cluster_size_median",
    "cluster_size_max",
    "large_cluster_count",
    "large_cluster_fraction",
    "branch_balance_fraction",
    "family_count",
    "target_count",
    "direction_count",
    "duplicated_representative_id_count",
    "start_class_fraction_json",
    "boundary_representative_fraction",
    "clusters_without_success_representative_fraction",
    "top_largest_clusters_json",
    "top_mixed_boundary_clusters_json",
    "clustering_strategy_status",
)

CLUSTERING_STRATEGY_STATUS_VALUES: tuple[str, ...] = (
    "baseline_sufficient_for_w0_inspection",
    "needs_boundary_augmented_strategy",
    "needs_adaptive_binning_strategy",
    "needs_cluster_quota_strategy",
    "insufficient_data_for_clustering_decision",
)


@dataclass(frozen=True)
class DenseClusterDiagnosticConfig:
    min_trials_total: int = 300000
    max_representatives_per_group: int = 12
    boundary_representative_fraction_min: float = 0.05
    branch_balance_min_fraction: float = 0.45
    large_cluster_multiplier: int = 10
    large_cluster_fraction_max: float = 0.30
    top_cluster_count: int = 20


# =============================================================================
# 2) Public Diagnostics Builder
# =============================================================================
def build_cluster_diagnostics(
    trial_rows: pd.DataFrame,
    envelope_map: pd.DataFrame,
    cluster_representatives: pd.DataFrame,
    config: DenseClusterDiagnosticConfig | None = None,
) -> pd.DataFrame:
    """Return deterministic branch/environment diagnostics for cluster inspection."""

    cfg = DenseClusterDiagnosticConfig() if config is None else config
    if trial_rows.empty and envelope_map.empty and cluster_representatives.empty:
        return pd.DataFrame(columns=CLUSTER_DIAGNOSTIC_COLUMNS)

    branch_counts = _branch_counts(trial_rows)
    keys = _diagnostic_keys(trial_rows, envelope_map, cluster_representatives)
    rows = [
        _diagnostic_row(
            key,
            trial_rows,
            envelope_map,
            cluster_representatives,
            branch_counts,
            cfg,
        )
        for key in keys
    ]
    return pd.DataFrame(rows, columns=CLUSTER_DIAGNOSTIC_COLUMNS)


# =============================================================================
# 3) Aggregation Helpers
# =============================================================================
def _diagnostic_keys(
    trial_rows: pd.DataFrame,
    envelope_map: pd.DataFrame,
    cluster_representatives: pd.DataFrame,
) -> list[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    for frame in (trial_rows, envelope_map, cluster_representatives):
        if frame.empty:
            continue
        for row in frame.to_dict(orient="records"):
            keys.add(
                (
                    _text(row.get("layout_branch_id", "")),
                    _text(row.get("fan_layout", "")),
                    _text(row.get("test_environment_mode", "")),
                )
            )
    return sorted(keys)


def _diagnostic_row(
    key: tuple[str, str, str],
    trial_rows: pd.DataFrame,
    envelope_map: pd.DataFrame,
    cluster_representatives: pd.DataFrame,
    branch_counts: dict[str, int],
    config: DenseClusterDiagnosticConfig,
) -> dict[str, object]:
    branch_id, fan_layout, environment_mode = key
    trials = _subset(trial_rows, key)
    env = _subset(envelope_map, key)
    reps = _subset(cluster_representatives, key)
    cluster_sizes = _cluster_sizes(trials, env)
    role_counts = _role_counts(reps)
    trial_status_counts = _trial_status_counts(trials)
    boundary_fraction = _fraction(
        role_counts["boundary_representative_count"],
        int(len(reps)),
    )
    no_success_fraction = _clusters_without_success_fraction(reps, env)
    large_threshold = int(config.max_representatives_per_group) * int(
        config.large_cluster_multiplier
    )
    large_cluster_count = int(sum(size > large_threshold for size in cluster_sizes))
    large_cluster_fraction = _fraction(large_cluster_count, len(cluster_sizes))
    branch_balance_fraction = _fraction(
        branch_counts.get(branch_id, 0),
        sum(branch_counts.values()),
    )
    mixed_boundary_count = _cell_status_count(env, "mixed_boundary")
    status = _strategy_status(
        trial_total=sum(branch_counts.values()),
        branch_balance_fraction=branch_balance_fraction,
        mixed_boundary_count=mixed_boundary_count,
        boundary_fraction=boundary_fraction,
        large_cluster_fraction=large_cluster_fraction,
        config=config,
    )
    return {
        "layout_branch_id": branch_id,
        "fan_layout": fan_layout,
        "test_environment_mode": environment_mode,
        "trial_count": int(len(trials)),
        "cluster_count": int(len(cluster_sizes)),
        "representative_count": int(len(reps)),
        **role_counts,
        **trial_status_counts,
        "cluster_size_min": _nan_if_empty(np.min, cluster_sizes),
        "cluster_size_median": _nan_if_empty(np.median, cluster_sizes),
        "cluster_size_max": _nan_if_empty(np.max, cluster_sizes),
        "large_cluster_count": large_cluster_count,
        "large_cluster_fraction": large_cluster_fraction,
        "branch_balance_fraction": branch_balance_fraction,
        "family_count": _nunique(trials, "family"),
        "target_count": _nunique(trials, "target_heading_deg"),
        "direction_count": _nunique(trials, "direction_sign"),
        "duplicated_representative_id_count": _duplicated_representative_count(reps),
        "start_class_fraction_json": _start_class_fraction_json(trials),
        "boundary_representative_fraction": boundary_fraction,
        "clusters_without_success_representative_fraction": no_success_fraction,
        "top_largest_clusters_json": _top_largest_clusters_json(cluster_sizes, env, config),
        "top_mixed_boundary_clusters_json": _top_mixed_boundary_clusters_json(env, config),
        "clustering_strategy_status": status,
    }


def _strategy_status(
    *,
    trial_total: int,
    branch_balance_fraction: float,
    mixed_boundary_count: int,
    boundary_fraction: float,
    large_cluster_fraction: float,
    config: DenseClusterDiagnosticConfig,
) -> str:
    if int(trial_total) < int(config.min_trials_total):
        return "insufficient_data_for_clustering_decision"
    if branch_balance_fraction < float(config.branch_balance_min_fraction):
        return "needs_cluster_quota_strategy"
    if mixed_boundary_count > 0 and boundary_fraction < float(
        config.boundary_representative_fraction_min
    ):
        return "needs_boundary_augmented_strategy"
    if large_cluster_fraction > float(config.large_cluster_fraction_max):
        return "needs_adaptive_binning_strategy"
    return "baseline_sufficient_for_w0_inspection"


def _subset(frame: pd.DataFrame, key: tuple[str, str, str]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    branch_id, fan_layout, environment_mode = key
    mask = (
        frame["layout_branch_id"].astype(str).eq(branch_id)
        & frame["fan_layout"].astype(str).eq(fan_layout)
        & frame["test_environment_mode"].astype(str).eq(environment_mode)
    )
    return frame[mask].copy()


def _branch_counts(trial_rows: pd.DataFrame) -> dict[str, int]:
    if trial_rows.empty:
        return {}
    return {
        str(key): int(value)
        for key, value in trial_rows["layout_branch_id"].value_counts().to_dict().items()
    }


def _cluster_sizes(trial_rows: pd.DataFrame, envelope_map: pd.DataFrame) -> list[int]:
    if not envelope_map.empty and "trial_count" in envelope_map.columns:
        return [
            int(value)
            for value in pd.to_numeric(envelope_map["trial_count"], errors="coerce")
            .fillna(0)
            .to_list()
        ]
    if trial_rows.empty or "cluster_key" not in trial_rows.columns:
        return []
    return [int(value) for value in trial_rows["cluster_key"].value_counts().to_list()]


def _role_counts(reps: pd.DataFrame) -> dict[str, int]:
    counts = (
        reps["candidate_role"].astype(str).value_counts().to_dict()
        if not reps.empty and "candidate_role" in reps.columns
        else {}
    )
    return {
        "success_representative_count": int(counts.get("success_representative", 0)),
        "boundary_representative_count": int(counts.get("boundary_representative", 0)),
        "failure_representative_count": int(counts.get("failure_representative", 0)),
        "not_replayed_representative_count": int(
            counts.get("not_replayed_representative", 0)
        ),
    }


def _trial_status_counts(trials: pd.DataFrame) -> dict[str, int]:
    if trials.empty:
        return {
            "success_trial_count": 0,
            "failure_trial_count": 0,
            "not_replayed_trial_count": 0,
        }
    success = trials["success_flag"].astype(bool) if "success_flag" in trials.columns else []
    descriptor_status = (
        trials["descriptor_status"].astype(str)
        if "descriptor_status" in trials.columns
        else pd.Series([], dtype=str)
    )
    not_replayed = descriptor_status.isin({"not_replayed", "synthetic_descriptor_only"})
    success_count = int(pd.Series(success).sum())
    return {
        "success_trial_count": success_count,
        "failure_trial_count": int(len(trials) - success_count - int(not_replayed.sum())),
        "not_replayed_trial_count": int(not_replayed.sum()),
    }


def _clusters_without_success_fraction(reps: pd.DataFrame, envelope_map: pd.DataFrame) -> float:
    if envelope_map.empty:
        return 0.0
    if not reps.empty and "cluster_key" in reps.columns:
        success_clusters = set(
            reps[reps["candidate_role"].astype(str).eq("success_representative")][
                "cluster_key"
            ].astype(str)
        )
        cluster_ids = (
            envelope_map["envelope_cell_id"].astype(str).str.replace("cell|", "cluster|", n=1)
        )
        missing = sum(cluster_id not in success_clusters for cluster_id in cluster_ids)
        return _fraction(missing, int(len(envelope_map)))
    success_count = pd.to_numeric(
        envelope_map.get("success_count", pd.Series([], dtype=float)),
        errors="coerce",
    ).fillna(0)
    return _fraction(int((success_count <= 0).sum()), int(len(envelope_map)))


def _cell_status_count(envelope_map: pd.DataFrame, status: str) -> int:
    if envelope_map.empty or "cell_status" not in envelope_map.columns:
        return 0
    return int(envelope_map["cell_status"].astype(str).eq(status).sum())


def _top_largest_clusters_json(
    cluster_sizes: list[int],
    envelope_map: pd.DataFrame,
    config: DenseClusterDiagnosticConfig,
) -> str:
    if envelope_map.empty:
        rows = [
            {"cluster_rank": index + 1, "trial_count": int(size)}
            for index, size in enumerate(sorted(cluster_sizes, reverse=True)[: int(config.top_cluster_count)])
        ]
        return json.dumps(rows, separators=(",", ":"))
    columns = ["envelope_cell_id", "trial_count", "cell_status"]
    available = [column for column in columns if column in envelope_map.columns]
    rows = (
        envelope_map[available]
        .sort_values("trial_count", ascending=False)
        .head(int(config.top_cluster_count))
        .to_dict(orient="records")
    )
    return json.dumps(rows, separators=(",", ":"))


def _top_mixed_boundary_clusters_json(
    envelope_map: pd.DataFrame,
    config: DenseClusterDiagnosticConfig,
) -> str:
    if envelope_map.empty or "cell_status" not in envelope_map.columns:
        return "[]"
    columns = ["envelope_cell_id", "trial_count", "cell_status", "success_fraction"]
    available = [column for column in columns if column in envelope_map.columns]
    rows = (
        envelope_map[envelope_map["cell_status"].astype(str).eq("mixed_boundary")][available]
        .sort_values("trial_count", ascending=False)
        .head(int(config.top_cluster_count))
        .to_dict(orient="records")
    )
    return json.dumps(rows, separators=(",", ":"))


# =============================================================================
# 4) Conversion Helpers
# =============================================================================
def _nunique(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame.columns:
        return 0
    return int(frame[column].nunique(dropna=True))


def _duplicated_representative_count(reps: pd.DataFrame) -> int:
    if reps.empty or "trial_descriptor_id" not in reps.columns:
        return 0
    return int(reps["trial_descriptor_id"].astype(str).duplicated().sum())


def _start_class_fraction_json(trials: pd.DataFrame) -> str:
    if trials.empty or "start_class" not in trials.columns:
        return "{}"
    counts = trials["start_class"].astype(str).value_counts(normalize=True).sort_index()
    return json.dumps({key: float(value) for key, value in counts.items()}, separators=(",", ":"))


def _fraction(numerator: int, denominator: int) -> float:
    if int(denominator) <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _nan_if_empty(func: object, values: list[int]) -> float:
    if not values:
        return float("nan")
    return float(func(values))


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value)
