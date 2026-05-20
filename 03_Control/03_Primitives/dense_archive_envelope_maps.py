from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from dense_archive_schema import BRANCH_DECISION_SCOPE


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Envelope Map Constants and Configuration
# 2) Public Cell and Map Builders
# 3) Cell Aggregation Helpers
# 4) Conversion and Binning Helpers
# =============================================================================


# =============================================================================
# 1) Envelope Map Constants and Configuration
# =============================================================================
ENVELOPE_MAP_COLUMNS: tuple[str, ...] = (
    "envelope_cell_id",
    "layout_branch_id",
    "fan_layout",
    "test_environment_mode",
    "family",
    "target_heading_deg",
    "direction_sign",
    "start_class",
    "latency_case",
    "radius_bin_m",
    "speed_bin_m_s",
    "wing_mean_bin_m_s",
    "margin_bin_m",
    "trial_count",
    "evaluated_trial_count",
    "success_count",
    "success_fraction",
    "nominal_pass_count",
    "conservative_pass_count",
    "dominant_failure_label",
    "dominant_governor_rejection_cause",
    "mean_heading_error_deg",
    "median_heading_error_deg",
    "mean_energy_residual_m",
    "mean_lift_dwell_fraction",
    "min_true_margin_m_min",
    "saturation_fraction_mean",
    "cell_status",
    "branch_decision_scope",
)

ENVELOPE_CELL_STATUS_VALUES: tuple[str, ...] = (
    "no_trials",
    "all_success",
    "mixed_boundary",
    "all_failure",
    "not_replayed",
)

EVALUATED_DESCRIPTOR_STATUSES = frozenset(
    {"replay_evaluated", "entry_invalid", "nonfinite_state"}
)
NON_EVALUATED_DESCRIPTOR_STATUSES = frozenset(
    {"not_replayed", "synthetic_descriptor_only"}
)


@dataclass(frozen=True)
class EnvelopeMapConfig:
    min_trials_per_cell: int = 1
    radius_bin_width_m: float = 0.25
    speed_bin_width_m_s: float = 0.5
    wing_mean_bin_width_m_s: float = 0.10
    margin_bin_width_m: float = 0.25
    heading_error_success_deg: float = 5.0
    branch_selection_rule: str = "branch_local_only_no_cross_layout_decision_transfer"
    environment_selection_rule: str = "preserve_w0_w1_environment_modes"


# =============================================================================
# 2) Public Cell and Map Builders
# =============================================================================
def envelope_cell_id(
    row: Mapping[str, object],
    config: EnvelopeMapConfig | None = None,
) -> str:
    """Return the branch-local dense-envelope cell key for one descriptor row."""

    return dense_cell_key(row, prefix="cell", config=config)


def build_envelope_map(
    trial_rows: pd.DataFrame,
    config: EnvelopeMapConfig | None = None,
) -> pd.DataFrame:
    """Aggregate dense-trial descriptors into branch-local envelope cells."""

    cfg = EnvelopeMapConfig() if config is None else config
    if trial_rows.empty:
        return pd.DataFrame(columns=ENVELOPE_MAP_COLUMNS)

    frame = trial_rows.copy()
    frame["_envelope_cell_id"] = [
        envelope_cell_id(row, cfg) for row in frame.to_dict(orient="records")
    ]
    rows = [
        _envelope_row(cell_id, group, cfg)
        for cell_id, group in frame.groupby("_envelope_cell_id", sort=True)
    ]
    return pd.DataFrame(rows, columns=ENVELOPE_MAP_COLUMNS)


def dense_cell_key(
    row: Mapping[str, object],
    *,
    prefix: str,
    config: EnvelopeMapConfig | None = None,
) -> str:
    """Return the shared envelope/cluster key suffix with a caller-chosen prefix."""

    cfg = EnvelopeMapConfig() if config is None else config
    bins = envelope_bins(row, cfg)
    return "|".join(
        (
            str(prefix),
            f"branch={_key_text(_value(row, 'layout_branch_id'))}",
            f"fan={_key_text(_value(row, 'fan_layout'))}",
            f"env={_key_text(_value(row, 'test_environment_mode'))}",
            f"family={_key_text(_value(row, 'family'))}",
            f"target={_target_label(_value(row, 'target_heading_deg'))}",
            f"dir={_direction_label(_value(row, 'direction_sign'))}",
            f"start={_key_text(_value(row, 'start_class'))}",
            f"latency={_key_text(_value(row, 'latency_case'))}",
            f"radius={bins['radius_bin_m']}",
            f"speed={bins['speed_bin_m_s']}",
            f"wing_mean={bins['wing_mean_bin_m_s']}",
            f"margin={bins['margin_bin_m']}",
        )
    )


def envelope_bins(
    row: Mapping[str, object],
    config: EnvelopeMapConfig | None = None,
) -> dict[str, str]:
    """Return deterministic bin labels shared by envelope maps and clusters."""

    cfg = EnvelopeMapConfig() if config is None else config
    return {
        "radius_bin_m": _existing_or_bin(
            row,
            "radius_bin_m",
            _value(row, "updraft_relative_radius_m"),
            float(cfg.radius_bin_width_m),
            "r",
            "r_nan",
        ),
        "speed_bin_m_s": _existing_or_bin(
            row,
            "speed_bin_m_s",
            _first_value(row, ("speed0_m_s", "speed_m_s")),
            float(cfg.speed_bin_width_m_s),
            "v",
            "v_nan",
        ),
        "wing_mean_bin_m_s": _existing_or_bin(
            row,
            "wing_mean_bin_m_s",
            _value(row, "w_wing_mean_m_s"),
            float(cfg.wing_mean_bin_width_m_s),
            "ww",
            "ww_nan",
        ),
        "margin_bin_m": _existing_or_bin(
            row,
            "margin_bin_m",
            _value(row, "min_true_margin_m"),
            float(cfg.margin_bin_width_m),
            "m",
            "m_nan",
        ),
    }


# =============================================================================
# 3) Cell Aggregation Helpers
# =============================================================================
def _envelope_row(
    cell_id: str,
    group: pd.DataFrame,
    config: EnvelopeMapConfig,
) -> dict[str, object]:
    first = group.iloc[0].to_dict()
    bins = envelope_bins(first, config)
    evaluated = group["descriptor_status"].astype(str).isin(EVALUATED_DESCRIPTOR_STATUSES)
    success = _success_series(group) & evaluated
    trial_count = int(len(group))
    evaluated_count = int(evaluated.sum())
    success_count = int(success.sum())
    return {
        "envelope_cell_id": cell_id,
        "layout_branch_id": _text(_value(first, "layout_branch_id")),
        "fan_layout": _text(_value(first, "fan_layout")),
        "test_environment_mode": _text(_value(first, "test_environment_mode")),
        "family": _text(_value(first, "family")),
        "target_heading_deg": _target_value(_value(first, "target_heading_deg")),
        "direction_sign": _direction_int(_value(first, "direction_sign")),
        "start_class": _text(_value(first, "start_class")),
        "latency_case": _text(_value(first, "latency_case")),
        "radius_bin_m": bins["radius_bin_m"],
        "speed_bin_m_s": bins["speed_bin_m_s"],
        "wing_mean_bin_m_s": bins["wing_mean_bin_m_s"],
        "margin_bin_m": bins["margin_bin_m"],
        "trial_count": trial_count,
        "evaluated_trial_count": evaluated_count,
        "success_count": success_count,
        "success_fraction": _success_fraction(success_count, evaluated_count, trial_count),
        "nominal_pass_count": _label_count(group, "latency_pass_label", "nominal_pass"),
        "conservative_pass_count": _label_count(
            group,
            "latency_pass_label",
            "conservative_pass",
        ),
        "dominant_failure_label": _dominant_text(group, "failure_label"),
        "dominant_governor_rejection_cause": _dominant_text(
            group,
            "governor_rejection_cause",
        ),
        "mean_heading_error_deg": _nanmean(group, "heading_error_deg"),
        "median_heading_error_deg": _nanmedian(group, "heading_error_deg"),
        "mean_energy_residual_m": _nanmean(group, "energy_residual_m"),
        "mean_lift_dwell_fraction": _nanmean(group, "lift_dwell_fraction"),
        "min_true_margin_m_min": _nanmin(group, "min_true_margin_m"),
        "saturation_fraction_mean": _nanmean(group, "saturation_fraction"),
        "cell_status": _cell_status(group, evaluated, success),
        "branch_decision_scope": _text(
            _value(first, "branch_decision_scope", BRANCH_DECISION_SCOPE)
        ),
    }


def _cell_status(
    group: pd.DataFrame,
    evaluated: pd.Series,
    success: pd.Series,
) -> str:
    trial_count = int(len(group))
    if trial_count == 0:
        return "no_trials"
    evaluated_count = int(evaluated.sum())
    success_count = int(success.sum())
    non_evaluated = group["descriptor_status"].astype(str).isin(
        NON_EVALUATED_DESCRIPTOR_STATUSES
    )
    raw_success_count = int(_success_series(group).sum())
    if evaluated_count == 0 and raw_success_count == 0 and bool(non_evaluated.all()):
        return "not_replayed"
    if evaluated_count == trial_count and success_count == evaluated_count:
        return "all_success"
    if evaluated_count == trial_count and success_count == 0:
        return "all_failure"
    return "mixed_boundary"


def _success_series(group: pd.DataFrame) -> pd.Series:
    if "success_flag" not in group:
        return pd.Series(False, index=group.index)
    return group["success_flag"].map(_bool_value).astype(bool)


def _success_fraction(
    success_count: int,
    evaluated_count: int,
    trial_count: int,
) -> float:
    if int(evaluated_count) > 0:
        return float(success_count) / float(evaluated_count)
    if int(trial_count) > 0:
        return 0.0
    return float("nan")


def _label_count(group: pd.DataFrame, column: str, label: str) -> int:
    if column not in group:
        return 0
    return int((group[column].astype(str) == str(label)).sum())


def _dominant_text(group: pd.DataFrame, column: str) -> str:
    if column not in group:
        return ""
    values = [_text(value) for value in group[column].to_list()]
    values = [value for value in values if value != ""]
    if not values:
        return ""
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return sorted(counts, key=lambda value: (-counts[value], value))[0]


def _nanmean(group: pd.DataFrame, column: str) -> float:
    values = _finite_column(group, column)
    return float(np.mean(values)) if values.size else float("nan")


def _nanmedian(group: pd.DataFrame, column: str) -> float:
    values = _finite_column(group, column)
    return float(np.median(values)) if values.size else float("nan")


def _nanmin(group: pd.DataFrame, column: str) -> float:
    values = _finite_column(group, column)
    return float(np.min(values)) if values.size else float("nan")


# =============================================================================
# 4) Conversion and Binning Helpers
# =============================================================================
def _existing_or_bin(
    row: Mapping[str, object],
    bin_key: str,
    value: object,
    width: float,
    prefix: str,
    nan_label: str,
) -> str:
    existing = _text(_value(row, bin_key, ""))
    if existing:
        return existing
    return _bin_label(value, width, prefix, nan_label)


def _bin_label(value: object, width: float, prefix: str, nan_label: str) -> str:
    numeric = _float_or_nan(value)
    if not np.isfinite(numeric):
        return nan_label
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("bin widths must be finite and positive.")
    lower = np.floor(numeric / width) * width
    upper = lower + width
    return f"{prefix}[{lower:.2f},{upper:.2f})"


def _finite_column(group: pd.DataFrame, column: str) -> np.ndarray:
    if column not in group:
        return np.array([], dtype=float)
    values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _value(
    row: Mapping[str, object],
    key: str,
    default: object = "",
) -> object:
    if key in row:
        return row[key]
    return default


def _first_value(row: Mapping[str, object], keys: tuple[str, ...]) -> object:
    for key in keys:
        value = _value(row, key, "")
        if _text(value) != "":
            return value
    return ""


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


def _target_label(value: object) -> str:
    numeric = _float_or_nan(value)
    if not np.isfinite(numeric):
        return "none"
    return f"{numeric:.1f}".replace(".", "p")


def _direction_int(value: object) -> int:
    numeric = _float_or_nan(value)
    if not np.isfinite(numeric) or numeric == 0.0:
        return 1
    return -1 if numeric < 0.0 else 1


def _direction_label(value: object) -> str:
    return str(_direction_int(value))


def _key_text(value: object) -> str:
    text = _text(value)
    return text.replace("|", "_").replace("\n", "_").replace("\r", "_")


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
        return ""
    return str(value)


def _bool_value(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)
