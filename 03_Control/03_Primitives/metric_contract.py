from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Metric schema constants
# 2) Metric row helpers
# 3) Metric schema audit table
# =============================================================================


# =============================================================================
# 1) Metric Schema Constants
# =============================================================================
FAILURE_LABELS = (
    "success",
    "not_run",
    "nonfinite_state",
    "model_audit_failed",
    "entry_set_violation",
    "true_safety_violation",
    "tracker_limit_violation",
    "floor_violation",
    "ceiling_violation",
    "wall_violation",
    "speed_low",
    "speed_high",
    "alpha_boundary",
    "beta_boundary",
    "rate_boundary",
    "actuator_saturation_limited",
    "latency_limited",
    "terminal_recovery_limited",
    "under_turning",
    "solver_failure",
    "model_boundary_only",
)

REQUIRED_METRIC_COLUMNS = (
    "run_id",
    "seed",
    "primitive_name",
    "primitive_family",
    "scenario_name",
    "wind_mode",
    "latency_case",
    "success",
    "failure_label",
    "duration_s",
    "initial_speed_m_s",
    "terminal_speed_m_s",
    "height_change_m",
    "min_true_wall_margin_m",
    "min_floor_margin_m",
    "min_ceiling_margin_m",
    "max_alpha_deg",
    "max_beta_deg",
    "max_bank_deg",
    "max_pitch_deg",
    "max_rate_rad_s",
    "saturation_fraction",
    "notes",
)
AGILE_METRIC_COLUMNS = (
    "target_heading_deg",
    "actual_heading_change_deg",
    "forward_travel_m",
    "turn_volume_proxy_m2",
    "exit_recoverable",
)


# =============================================================================
# 2) Metric Row Helpers
# =============================================================================
def empty_metric_row(include_agile: bool = False) -> dict[str, object]:
    """Return a metric row with all required keys."""

    row: dict[str, object] = {
        "run_id": "",
        "seed": 0,
        "primitive_name": "",
        "primitive_family": "",
        "scenario_name": "",
        "wind_mode": "",
        "latency_case": "",
        "success": False,
        "failure_label": "not_run",
        "duration_s": np.nan,
        "initial_speed_m_s": np.nan,
        "terminal_speed_m_s": np.nan,
        "height_change_m": np.nan,
        "min_true_wall_margin_m": np.nan,
        "min_floor_margin_m": np.nan,
        "min_ceiling_margin_m": np.nan,
        "max_alpha_deg": np.nan,
        "max_beta_deg": np.nan,
        "max_bank_deg": np.nan,
        "max_pitch_deg": np.nan,
        "max_rate_rad_s": np.nan,
        "saturation_fraction": np.nan,
        "notes": "",
    }
    if include_agile:
        row.update(
            {
                "target_heading_deg": np.nan,
                "actual_heading_change_deg": np.nan,
                "forward_travel_m": np.nan,
                "turn_volume_proxy_m2": np.nan,
                "exit_recoverable": False,
            }
        )
    return row


def validate_metric_row(row: Mapping[str, object], allow_agile: bool = True) -> None:
    """Check required keys and allowed failure labels."""

    supplied = set(row.keys())
    required = set(REQUIRED_METRIC_COLUMNS)
    agile = set(AGILE_METRIC_COLUMNS)
    missing = required - supplied
    if missing:
        raise ValueError(f"metric row missing required keys: {sorted(missing)}.")

    allowed = required | agile if allow_agile else required
    extra = supplied - allowed
    if extra:
        raise ValueError(f"metric row contains unknown keys: {sorted(extra)}.")
    if row["failure_label"] not in FAILURE_LABELS:
        raise ValueError(f"unknown failure_label: {row['failure_label']}.")
    if not isinstance(row["success"], bool):
        raise ValueError("metric row success must be a bool.")


# =============================================================================
# 3) Metric Schema Audit Table
# =============================================================================
def metric_schema_dataframe() -> pd.DataFrame:
    """Return the metric schema as a table for audit output."""

    rows = [
        {"column": name, "required": True, "agile_only": False}
        for name in REQUIRED_METRIC_COLUMNS
    ]
    rows.extend(
        {"column": name, "required": False, "agile_only": True}
        for name in AGILE_METRIC_COLUMNS
    )
    return pd.DataFrame(rows)
