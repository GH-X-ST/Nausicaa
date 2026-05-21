from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from episode_schema import (
    ENTRY_SOURCE_VALUES,
    validate_primitive_rollout_evidence_frame,
)


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Clustering constants and configuration
# 2) Primitive-row clustering
# 3) Medoid and governor package builders
# 4) Validation and conversion helpers
# =============================================================================


STRATIFY_COLUMNS = (
    "fan_branch",
    "W_layer",
    "primitive_family",
    "latency_case",
    "entry_source",
    "outcome_class",
    "evidence_role",
    "feedback_mode",
)

FEATURE_COLUMNS = (
    "x0_w_m",
    "y0_w_m",
    "z0_w_m",
    "phi0_rad",
    "theta0_rad",
    "psi0_rad",
    "speed0_m_s",
    "minimum_margin_m",
    "w_wing_mean_m_s",
    "delta_w_lr_m_s",
    "spanwise_gradient_m_s_m",
    "dwell_time_s",
    "energy_residual_m",
    "exit_speed_m_s",
    "control_saturation",
)

MEDOID_COLUMNS = (
    "cluster_id",
    "medoid_sample_id",
    "primitive_id",
    "fan_branch",
    "W_layer",
    "latency_case",
    "entry_source",
    "outcome_class",
    "primitive_family",
    "dwell_time_s",
    "energy_residual_m",
    "minimum_margin_m",
    "exit_speed_m_s",
    "failure_label",
    "recommended_use",
    "is_medoid",
    "controller_mode",
    "feedback_mode",
    "claim_status",
    "evidence_role",
)


@dataclass(frozen=True)
class PrimitiveEnvelopeClusterConfig:
    epsilon: float = 1e-9
    max_medoid_count: int = 8


def validate_primitive_rollout_rows(frame: pd.DataFrame) -> None:
    missing = sorted(set(STRATIFY_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"primitive rollout rows missing stratification columns: {missing}")
    invalid_entry = sorted(set(frame["entry_source"].astype(str)).difference(ENTRY_SOURCE_VALUES))
    if invalid_entry:
        raise ValueError(f"unsupported entry_source values: {invalid_entry}")
    evidence_columns = {"controller_mode", "feedback_mode", "claim_status", "evidence_role"}
    if evidence_columns.issubset(frame.columns):
        validate_primitive_rollout_evidence_frame(frame)


def build_primitive_envelope_clusters(
    primitive_rows: pd.DataFrame,
    config: PrimitiveEnvelopeClusterConfig | None = None,
) -> dict[str, pd.DataFrame | dict[str, object]]:
    """Cluster primitive rollout rows before any whole-episode clustering."""

    cfg = PrimitiveEnvelopeClusterConfig() if config is None else config
    if primitive_rows.empty:
        return {
            "cluster_input_table": primitive_rows.copy(),
            "cluster_feature_scaling": {},
            "cluster_assignments": pd.DataFrame(columns=[*primitive_rows.columns, "cluster_id", "cluster_strategy"]),
            "cluster_medoids": pd.DataFrame(columns=MEDOID_COLUMNS),
            "cluster_summary": pd.DataFrame(columns=[*STRATIFY_COLUMNS, "row_count", "cluster_count"]),
            "governor_candidate_package": pd.DataFrame(columns=MEDOID_COLUMNS),
            "mission_medoids": pd.DataFrame(columns=MEDOID_COLUMNS),
            "partial_feedback_medoids": pd.DataFrame(columns=MEDOID_COLUMNS),
            "diagnostic_medoids": pd.DataFrame(columns=MEDOID_COLUMNS),
            "rejected_or_blocked_medoids": pd.DataFrame(columns=MEDOID_COLUMNS),
        }
    validate_primitive_rollout_rows(primitive_rows)
    frame = primitive_rows.copy()
    frame["_source_row_index"] = np.arange(len(frame), dtype=int)
    feature_frame, scaling = robust_scale_features(frame, cfg)
    assignments: list[pd.DataFrame] = []
    medoids: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for key, group in frame.groupby(list(STRATIFY_COLUMNS), sort=True, dropna=False):
        group_features = feature_frame.loc[group.index]
        cluster_count = suggested_cluster_count(len(group))
        labels, strategy = _assign_clusters(group_features, cluster_count, cfg)
        assigned = group.copy()
        stratum_prefix = _cluster_prefix(key)
        assigned["cluster_id"] = [f"{stratum_prefix}__c{int(label):02d}" for label in labels]
        assigned["cluster_strategy"] = strategy
        assignments.append(assigned)
        for cluster_id, cluster_group in assigned.groupby("cluster_id", sort=True):
            medoids.append(_medoid_row(cluster_id, cluster_group, feature_frame.loc[cluster_group.index]))
        summary = {column: value for column, value in zip(STRATIFY_COLUMNS, key, strict=True)}
        summary.update({"row_count": int(len(group)), "cluster_count": int(len(set(labels))), "cluster_strategy": strategy})
        summaries.append(summary)
    assignment_frame = pd.concat(assignments, ignore_index=True) if assignments else pd.DataFrame()
    medoid_frame = pd.DataFrame(medoids, columns=MEDOID_COLUMNS)
    package = build_governor_candidate_package(medoid_frame)
    return {
        "cluster_input_table": primitive_rows.copy(),
        "cluster_feature_scaling": scaling,
        "cluster_assignments": assignment_frame,
        "cluster_medoids": medoid_frame,
        "cluster_summary": pd.DataFrame(summaries),
        "governor_candidate_package": package,
        "mission_medoids": _role_medoids(medoid_frame, {"mission_candidate"}),
        "partial_feedback_medoids": _role_medoids(medoid_frame, {"partial_feedback"}),
        "diagnostic_medoids": _role_medoids(medoid_frame, {"ablation_diagnostic", "boundary_diagnostic"}),
        "rejected_or_blocked_medoids": _role_medoids(medoid_frame, {"blocked_partial", "schema_only"}),
    }


def robust_scale_features(
    frame: pd.DataFrame,
    config: PrimitiveEnvelopeClusterConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    cfg = PrimitiveEnvelopeClusterConfig() if config is None else config
    scaled: dict[str, pd.Series] = {}
    scaling: dict[str, object] = {}
    for column in FEATURE_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce") if column in frame.columns else pd.Series(np.nan, index=frame.index)
        median = float(values.median()) if values.notna().any() else 0.0
        q75 = float(values.quantile(0.75)) if values.notna().any() else 0.0
        q25 = float(values.quantile(0.25)) if values.notna().any() else 0.0
        iqr = q75 - q25
        filled = values.fillna(median)
        scaled[column] = (filled - median) / (iqr + float(cfg.epsilon))
        scaling[column] = {"median": median, "iqr": float(iqr), "epsilon": float(cfg.epsilon)}
    return pd.DataFrame(scaled, index=frame.index), scaling


def suggested_cluster_count(row_count: int) -> int:
    n = int(row_count)
    if n < 100:
        return 1
    if n <= 1000:
        return 3
    if n <= 5000:
        return 5
    return 8


def build_governor_candidate_package(medoids: pd.DataFrame) -> pd.DataFrame:
    if medoids.empty:
        return medoids.copy()
    package = medoids.copy()
    if "evidence_role" in package.columns:
        package = package[package["evidence_role"].astype(str).isin({"mission_candidate", "partial_feedback"})].copy()
    package = package[
        package["recommended_use"].astype(str).isin(
            {"simulation_candidate", "hardware_candidate", "thesis", "hardware"}
        )
    ].copy()
    package["governor_package_status"] = "candidate_summary_only_governor_still_required"
    return package.reset_index(drop=True)


def write_cluster_feature_scaling(path: Path, scaling: dict[str, object]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(scaling, indent=2, sort_keys=False) + "\n", encoding="ascii")


def _assign_clusters(
    features: pd.DataFrame,
    cluster_count: int,
    config: PrimitiveEnvelopeClusterConfig,
) -> tuple[np.ndarray, str]:
    if features.empty:
        return np.array([], dtype=int), "empty"
    if int(cluster_count) <= 1:
        return np.zeros(len(features), dtype=int), "rank_cases_no_clustering"
    values = features.to_numpy(dtype=float)
    medoid_indices = _farthest_first_indices(values, min(int(cluster_count), int(config.max_medoid_count), len(values)))
    distances = np.stack([np.linalg.norm(values - values[index], axis=1) for index in medoid_indices], axis=1)
    labels = np.argmin(distances, axis=1).astype(int)
    return labels, "deterministic_robust_scaled_medoid_fallback"


def _farthest_first_indices(values: np.ndarray, count: int) -> list[int]:
    centre = np.nanmedian(values, axis=0)
    first = int(np.argmin(np.linalg.norm(values - centre, axis=1)))
    selected = [first]
    while len(selected) < int(count):
        distances = np.stack([np.linalg.norm(values - values[index], axis=1) for index in selected], axis=1)
        min_distance = np.min(distances, axis=1)
        next_index = int(np.argmax(min_distance))
        if next_index in selected:
            break
        selected.append(next_index)
    return selected


def _medoid_row(cluster_id: str, cluster_group: pd.DataFrame, feature_group: pd.DataFrame) -> dict[str, object]:
    values = feature_group.to_numpy(dtype=float)
    centre = np.nanmedian(values, axis=0)
    local_index = int(np.argmin(np.linalg.norm(values - centre, axis=1)))
    row = cluster_group.iloc[local_index].to_dict()
    sample_id = _text(_first(row, ("sample_id", "trial_descriptor_id", "primitive_rollout_id"), "unknown_sample"))
    primitive_id = _text(_first(row, ("primitive_id", "candidate_id", "source_primitive_id"), sample_id))
    outcome_class = _text(row.get("outcome_class", "unknown"))
    return {
        "cluster_id": str(cluster_id),
        "medoid_sample_id": sample_id,
        "primitive_id": primitive_id,
        "fan_branch": _text(row.get("fan_branch")),
        "W_layer": _text(row.get("W_layer")),
        "latency_case": _text(row.get("latency_case")),
        "entry_source": _text(row.get("entry_source")),
        "outcome_class": outcome_class,
        "primitive_family": _text(row.get("primitive_family")),
        "dwell_time_s": _float(row.get("dwell_time_s", row.get("lift_dwell_time_s", np.nan))),
        "energy_residual_m": _float(row.get("energy_residual_m", np.nan)),
        "minimum_margin_m": _float(row.get("minimum_margin_m", row.get("min_true_margin_m", np.nan))),
        "exit_speed_m_s": _float(row.get("exit_speed_m_s", row.get("terminal_speed_m_s", np.nan))),
        "failure_label": _text(row.get("failure_label", "none")),
        "recommended_use": _recommended_use(row, outcome_class),
        "is_medoid": True,
        "controller_mode": _text(row.get("controller_mode")),
        "feedback_mode": _text(row.get("feedback_mode")),
        "claim_status": _text(row.get("claim_status")),
        "evidence_role": _text(row.get("evidence_role")),
    }


def _recommended_use(row: dict[str, object], outcome_class: str) -> str:
    if _text(row.get("evidence_role")) in {"ablation_diagnostic", "boundary_diagnostic"}:
        return "diagnostic_only"
    if _text(row.get("evidence_role")) in {"blocked_partial", "schema_only"}:
        return "blocked"
    if _text(row.get("entry_source")) == "diagnostic_broad_only":
        return "diagnostic_only"
    if outcome_class in {"accepted", "success", "weak"}:
        margin = _float(row.get("minimum_margin_m", row.get("min_true_margin_m", 0.0)))
        if _text(row.get("evidence_role")) == "mission_candidate" and _text(row.get("feedback_mode")) == "delayed_state_feedback":
            return "hardware_candidate" if margin > 0.25 else "simulation_candidate"
        return "simulation_candidate"
    if outcome_class in {"failed", "rejected"}:
        return "reject"
    return "diagnostic_only"


def _role_medoids(medoids: pd.DataFrame, roles: set[str]) -> pd.DataFrame:
    if medoids.empty or "evidence_role" not in medoids.columns:
        return pd.DataFrame(columns=medoids.columns)
    return medoids[medoids["evidence_role"].astype(str).isin(roles)].copy().reset_index(drop=True)


def _cluster_prefix(key: tuple[object, ...]) -> str:
    parts = []
    for column, value in zip(STRATIFY_COLUMNS, key, strict=True):
        parts.append(f"{column}={_safe_key(value)}")
    return "|".join(parts)


def _safe_key(value: object) -> str:
    return str(value).replace(" ", "_").replace("/", "_").replace("\\", "_").replace("|", "_")


def _first(row: dict[str, object], names: tuple[str, ...], default: object) -> object:
    for name in names:
        if name in row and str(row[name]) != "":
            return row[name]
    return default


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _text(value: object) -> str:
    if value is None:
        return ""
    return str(value)
