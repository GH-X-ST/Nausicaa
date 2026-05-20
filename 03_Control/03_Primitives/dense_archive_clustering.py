from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from dense_archive_envelope_maps import (
    EnvelopeMapConfig,
    build_envelope_map,
    dense_cell_key,
    envelope_cell_id,
)
from dense_archive_schema import BRANCH_DECISION_SCOPE


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Cluster Constants and Configuration
# 2) Public Cluster Builders
# 3) Representative Selection Helpers
# 4) Conversion Helpers
# =============================================================================


# =============================================================================
# 1) Cluster Constants and Configuration
# =============================================================================
CLUSTER_REPRESENTATIVE_COLUMNS: tuple[str, ...] = (
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


@dataclass(frozen=True)
class DenseClusterConfig:
    max_representatives_per_group: int = 12
    min_success_fraction_for_candidate: float = 0.50
    include_failure_representatives: bool = True
    branch_selection_rule: str = "branch_local_only_no_cross_layout_decision_transfer"
    environment_selection_rule: str = "preserve_w0_w1_environment_modes"


# =============================================================================
# 2) Public Cluster Builders
# =============================================================================
def cluster_key(row: Mapping[str, object]) -> str:
    """Return the branch-local cluster key using envelope-map bin semantics."""

    return dense_cell_key(row, prefix="cluster", config=EnvelopeMapConfig())


def select_cluster_representatives(
    trial_rows: pd.DataFrame,
    envelope_map: pd.DataFrame | None = None,
    config: DenseClusterConfig | None = None,
) -> pd.DataFrame:
    """Select deterministic branch-local representatives from descriptor rows."""

    cfg = DenseClusterConfig() if config is None else config
    if trial_rows.empty:
        return pd.DataFrame(columns=CLUSTER_REPRESENTATIVE_COLUMNS)

    env = build_envelope_map(trial_rows) if envelope_map is None else envelope_map.copy()
    frame = _attach_envelope_status(trial_rows.copy(), env)
    frame["_cluster_key"] = [cluster_key(row) for row in frame.to_dict(orient="records")]

    rows: list[dict[str, object]] = []
    for key, group in frame.groupby("_cluster_key", sort=True):
        sorted_group = sorted(group.to_dict(orient="records"), key=_representative_sort_key)
        accepted = [
            row for row in sorted_group
            if _include_role(_candidate_role(row, cfg), cfg)
        ][: int(cfg.max_representatives_per_group)]
        for rank, row in enumerate(accepted, start=1):
            rows.append(_representative_row(key, rank, row, cfg))
    return pd.DataFrame(rows, columns=CLUSTER_REPRESENTATIVE_COLUMNS)


# =============================================================================
# 3) Representative Selection Helpers
# =============================================================================
def _attach_envelope_status(
    trial_rows: pd.DataFrame,
    envelope_map: pd.DataFrame,
) -> pd.DataFrame:
    frame = trial_rows.copy()
    frame["_envelope_cell_id"] = [
        envelope_cell_id(row) for row in frame.to_dict(orient="records")
    ]
    if envelope_map.empty:
        frame["_cell_status"] = "no_trials"
        frame["_cell_success_fraction"] = 0.0
        return frame
    env = envelope_map[["envelope_cell_id", "cell_status", "success_fraction"]].copy()
    env = env.rename(
        columns={
            "envelope_cell_id": "_envelope_cell_id",
            "cell_status": "_cell_status",
            "success_fraction": "_cell_success_fraction",
        }
    )
    joined = frame.merge(env, on="_envelope_cell_id", how="left")
    joined["_cell_status"] = joined["_cell_status"].fillna("no_trials")
    joined["_cell_success_fraction"] = pd.to_numeric(
        joined["_cell_success_fraction"],
        errors="coerce",
    ).fillna(0.0)
    return joined


def _candidate_role(
    row: Mapping[str, object],
    config: DenseClusterConfig,
) -> str:
    success = _bool_value(_value(row, "success_flag"))
    status = _text(_value(row, "_cell_status", "no_trials"))
    success_fraction = _float_or_nan(_value(row, "_cell_success_fraction", 0.0))
    if status == "mixed_boundary":
        return "boundary_representative"
    if success and success_fraction >= float(config.min_success_fraction_for_candidate):
        return "success_representative"
    return "failure_representative"


def _include_role(role: str, config: DenseClusterConfig) -> bool:
    if role == "failure_representative":
        return bool(config.include_failure_representatives)
    return True


def _representative_sort_key(row: Mapping[str, object]) -> tuple[object, ...]:
    status_rank = {
        "all_success": 0,
        "mixed_boundary": 1,
        "all_failure": 2,
        "not_replayed": 3,
        "no_trials": 4,
    }
    return (
        -int(_bool_value(_value(row, "success_flag"))),
        status_rank.get(_text(_value(row, "_cell_status", "no_trials")), 5),
        _ascending_nan_last(_value(row, "heading_error_deg")),
        _descending_nan_last(_value(row, "energy_residual_m")),
        _descending_nan_last(_value(row, "min_true_margin_m")),
        _ascending_nan_last(_value(row, "saturation_fraction")),
        _descending_nan_last(_value(row, "lift_dwell_fraction")),
        _text(_value(row, "trial_descriptor_id")),
    )


def _representative_row(
    key: str,
    rank: int,
    row: Mapping[str, object],
    config: DenseClusterConfig,
) -> dict[str, object]:
    return {
        "cluster_key": key,
        "representative_rank": int(rank),
        "trial_descriptor_id": _text(_value(row, "trial_descriptor_id")),
        "sim_real_match_key": _text(_value(row, "sim_real_match_key")),
        "layout_branch_id": _text(_value(row, "layout_branch_id")),
        "fan_layout": _text(_value(row, "fan_layout")),
        "test_environment_mode": _text(_value(row, "test_environment_mode")),
        "family": _text(_value(row, "family")),
        "target_heading_deg": _target_value(_value(row, "target_heading_deg")),
        "direction_sign": _direction_int(_value(row, "direction_sign")),
        "start_class": _text(_value(row, "start_class")),
        "latency_case": _text(_value(row, "latency_case")),
        "success_flag": _bool_value(_value(row, "success_flag")),
        "failure_label": _text(_value(row, "failure_label")),
        "governor_rejection_cause": _text(_value(row, "governor_rejection_cause")),
        "robustness_label": _text(_value(row, "robustness_label")),
        "heading_error_deg": _float_or_nan(_value(row, "heading_error_deg")),
        "energy_residual_m": _float_or_nan(_value(row, "energy_residual_m")),
        "lift_dwell_fraction": _float_or_nan(_value(row, "lift_dwell_fraction")),
        "min_true_margin_m": _float_or_nan(_value(row, "min_true_margin_m")),
        "saturation_fraction": _float_or_nan(_value(row, "saturation_fraction")),
        "w_wing_mean_m_s": _float_or_nan(_value(row, "w_wing_mean_m_s")),
        "delta_w_lr_m_s": _float_or_nan(_value(row, "delta_w_lr_m_s")),
        "physics_priority_level": _text(_value(row, "physics_priority_level")),
        "candidate_role": _candidate_role(row, config),
        "branch_decision_scope": _text(
            _value(row, "branch_decision_scope", BRANCH_DECISION_SCOPE)
        ),
    }


# =============================================================================
# 4) Conversion Helpers
# =============================================================================
def _value(
    row: Mapping[str, object],
    key: str,
    default: object = "",
) -> object:
    if key in row:
        return row[key]
    return default


def _float_or_nan(value: object) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, str) and value.strip() == "":
        return float("nan")
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _target_value(value: object) -> object:
    numeric = _float_or_nan(value)
    if np.isfinite(numeric):
        return float(numeric)
    return ""


def _direction_int(value: object) -> int:
    numeric = _float_or_nan(value)
    if not np.isfinite(numeric) or numeric == 0.0:
        return 1
    return -1 if numeric < 0.0 else 1


def _ascending_nan_last(value: object) -> tuple[int, float]:
    numeric = _float_or_nan(value)
    if not np.isfinite(numeric):
        return 1, 0.0
    return 0, numeric


def _descending_nan_last(value: object) -> tuple[int, float]:
    numeric = _float_or_nan(value)
    if not np.isfinite(numeric):
        return 1, 0.0
    return 0, -numeric


def _bool_value(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
        return ""
    return str(value)
