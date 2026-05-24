from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from lqr_controller import LQR_SYNTHESIS_SOLVED, LQRWeightSpec, default_lqr_weight_spec, synthesize_lqr_controller
from prim_cat import PrimitiveDefinition, active_primitive_catalogue


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Tuning contracts
# 2) Candidate generation
# 3) Ranking helpers
# =============================================================================


TUNING_METHOD_VERSION = "grouped_log_scaled_diagonal_qr_v1"
R61_TUNING_METHOD_VERSION = "r6_1_empirical_envelope_lqr_qr_v1"
PREFERRED_CANDIDATES_PER_PRIMITIVE = (16, 32)
FALLBACK_CANDIDATES_PER_PRIMITIVE = 8
PREFERRED_PAIRED_TESTS_PER_CANDIDATE = (50, 100)
FALLBACK_PAIRED_TESTS_PER_CANDIDATE = 25
R61_CANDIDATES_PER_PRIMITIVE = 32
R61_FINALISTS_PER_PRIMITIVE = 6
R61_STAGE_C_PAIRED_STARTS = 12
R61_STAGE_D_PAIRED_STARTS = 100
R61_STAGE_E_PAIRED_STARTS = 100
R61_STAGE_E_MAX_EXTRA_ROWS = 3200
R61_STAGE_E_MAX_CANDIDATES_PER_PRIMITIVE = 2
R61_SCORE_TIE_MARGIN = 0.05
HARD_GATE_LABELS = (
    "finite_state",
    "safe_volume",
    "floor_ceiling_margin",
    "minimum_speed",
    "surface_limits",
    "valid_lqr_synthesis",
    "sampled_data_stability",
    "surrogate_ready",
)
SOFT_OBJECTIVE_TERMS = (
    "energy_residual_m",
    "lift_dwell_time_s",
    "wall_margin_m",
    "saturation_fraction",
    "uncertainty_m_s",
    "latency_degradation",
    "terminal_use_label",
)
R61_CANDIDATE_FAMILIES = (
    ("nominal", 1),
    ("local", 7),
    ("conservative_high_r", 8),
    ("aggressive_low_r", 8),
    ("diagnostic_exploit", 8),
)
R61_SELECTION_REASON_SYSTEMIC = (
    "unsolved_lqr",
    "unstable_sampled_data",
    "missing_reconstructability_metadata",
    "command_reference_audit_failure",
    "surrogate_unavailable",
    "nonfinite_corrupt_rollout_ratio_high",
)


@dataclass(frozen=True)
class LQRTuningCandidate:
    primitive_id: str
    candidate_index: int
    candidate_weight_label: str
    controller_id: str
    lqr_Q_weights_json: str
    lqr_R_weights_json: str
    lqr_gain_checksum: str
    lqr_synthesis_status: str
    tuning_stage: str
    tuning_method: str = TUNING_METHOD_VERSION
    raw_K_tuning_allowed: bool = False


@dataclass(frozen=True)
class LQRTuningSchedule:
    primitive_count: int
    candidate_count: int
    paired_tests_per_candidate: int
    planned_rows: int
    fallback_mode: bool
    hard_gates_json: str
    soft_objective_terms_json: str
    method_version: str = TUNING_METHOD_VERSION


@dataclass(frozen=True)
class R61SelectionThresholds:
    launch_gate_min_pass_rate: float = 0.50
    nominal_inflight_min_pass_rate: float = 0.35
    systemic_blocker_max_ratio: float = 0.02
    hard_failure_max_ratio: float = 0.75
    fallback_launch_gate_min_pass_rate: float = 0.40
    fallback_nominal_inflight_min_pass_rate: float = 0.25
    fallback_hard_failure_max_ratio: float = 0.85
    score_tie_margin: float = R61_SCORE_TIE_MARGIN


def candidate_weight_specs(
    *,
    primitive_id: str,
    candidate_count: int = 16,
    tuning_stage: str = "W0_W1",
) -> tuple[LQRWeightSpec, ...]:
    """Return deterministic grouped log-scaled diagonal Q/R candidates."""

    if int(candidate_count) <= 0:
        raise ValueError("candidate_count must be positive.")
    scales = np.asarray([0.5, 1.0, 2.0, 4.0], dtype=float)
    candidates: list[LQRWeightSpec] = []
    cursor = 0
    for q_att in scales:
        for q_vel in scales:
            for q_rate in (1.0, 2.0):
                r_scale = scales[cursor % len(scales)]
                cursor += 1
                candidates.append(
                    LQRWeightSpec(
                        q_attitude=float(4.0 * q_att),
                        q_velocity=float(2.0 * q_vel),
                        q_rates=float(1.6 * q_rate),
                        q_surfaces=0.15,
                        r_aileron=float(1.0 * r_scale),
                        r_elevator=float(0.9 * r_scale),
                        r_rudder=float(1.1 * r_scale),
                        tuning_stage=tuning_stage,
                        weight_label=f"{primitive_id}_qr_{len(candidates):03d}",
                    )
                )
                if len(candidates) >= int(candidate_count):
                    return tuple(candidates)
    return tuple(candidates[: int(candidate_count)])


def _clamped_positive(value: float, *, lower: float = 0.02, upper: float = 128.0) -> float:
    return float(min(max(float(value), lower), upper))


def _scaled_r61_weight_spec(
    base: LQRWeightSpec,
    *,
    primitive_id: str,
    candidate_index: int,
    family: str,
    q_attitude: float = 1.0,
    q_velocity: float = 1.0,
    q_rates: float = 1.0,
    q_surfaces: float = 1.0,
    r_aileron: float = 1.0,
    r_elevator: float = 1.0,
    r_rudder: float = 1.0,
    tuning_stage: str = "W0_W1",
) -> LQRWeightSpec:
    return LQRWeightSpec(
        q_attitude=_clamped_positive(base.q_attitude * q_attitude),
        q_velocity=_clamped_positive(base.q_velocity * q_velocity),
        q_rates=_clamped_positive(base.q_rates * q_rates),
        q_surfaces=_clamped_positive(base.q_surfaces * q_surfaces),
        r_aileron=_clamped_positive(base.r_aileron * r_aileron),
        r_elevator=_clamped_positive(base.r_elevator * r_elevator),
        r_rudder=_clamped_positive(base.r_rudder * r_rudder),
        tuning_stage=tuning_stage,
        weight_label=f"{primitive_id}_r61_{family}_{candidate_index:03d}",
    )


def _json_float_map(value: object) -> dict[str, float]:
    if not isinstance(value, str) or not value:
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    out: dict[str, float] = {}
    for key, item in payload.items():
        try:
            out[str(key)] = float(item)
        except (TypeError, ValueError):
            continue
    return out


def _weight_spec_from_diagnostic_row(
    row: pd.Series,
    *,
    primitive_id: str,
    candidate_index: int,
    family: str,
    tuning_stage: str = "W0_W1",
) -> LQRWeightSpec | None:
    q_weights = _json_float_map(row.get("lqr_Q_weights_json"))
    r_weights = _json_float_map(row.get("lqr_R_weights_json"))
    required_q = ("q_attitude", "q_velocity", "q_rates", "q_surfaces")
    required_r = ("r_aileron", "r_elevator", "r_rudder")
    if any(key not in q_weights for key in required_q) or any(key not in r_weights for key in required_r):
        return None
    return LQRWeightSpec(
        q_attitude=_clamped_positive(q_weights["q_attitude"]),
        q_velocity=_clamped_positive(q_weights["q_velocity"]),
        q_rates=_clamped_positive(q_weights["q_rates"]),
        q_surfaces=_clamped_positive(q_weights["q_surfaces"]),
        r_aileron=_clamped_positive(r_weights["r_aileron"]),
        r_elevator=_clamped_positive(r_weights["r_elevator"]),
        r_rudder=_clamped_positive(r_weights["r_rudder"]),
        tuning_stage=tuning_stage,
        weight_label=f"{primitive_id}_r61_{family}_{candidate_index:03d}",
    )


def r6_1_candidate_family(candidate_weight_label: str) -> str:
    marker = "_r61_"
    if marker not in str(candidate_weight_label):
        return "unknown"
    tail = str(candidate_weight_label).split(marker, 1)[1]
    if tail.endswith(tuple(f"_{index:03d}" for index in range(1000))):
        return "_".join(tail.split("_")[:-1])
    return tail


def _diagnostic_metric_frame(diagnostic_frame: pd.DataFrame | None) -> pd.DataFrame:
    if diagnostic_frame is None or diagnostic_frame.empty:
        return pd.DataFrame()
    return r6_1_candidate_summary(diagnostic_frame)


def _diagnostic_exploit_bases(
    *,
    primitive_id: str,
    diagnostic_frame: pd.DataFrame | None,
    tuning_stage: str,
) -> tuple[LQRWeightSpec, ...]:
    summary = _diagnostic_metric_frame(diagnostic_frame)
    if summary.empty or "primitive_id" not in summary.columns:
        return ()
    rows = summary.loc[summary["primitive_id"] == primitive_id].copy()
    if rows.empty:
        return ()
    rows = rows.sort_values("robust_score", ascending=False).head(2)
    bases: list[LQRWeightSpec] = []
    for _, row in rows.iterrows():
        spec = _weight_spec_from_diagnostic_row(
            row,
            primitive_id=primitive_id,
            candidate_index=0,
            family="diagnostic_exploit",
            tuning_stage=tuning_stage,
        )
        if spec is not None:
            bases.append(spec)
    return tuple(bases)


def r6_1_candidate_weight_specs(
    *,
    primitive_id: str,
    diagnostic_frame: pd.DataFrame | None = None,
    candidate_count: int = R61_CANDIDATES_PER_PRIMITIVE,
    tuning_stage: str = "W0_W1",
) -> tuple[LQRWeightSpec, ...]:
    """Return deterministic R6.1 empirical-envelope Q/R candidates.

    The default output is exactly 32 candidates per active primitive:
    1 primitive-family nominal, 7 local perturbations, 8 conservative
    high-R, 8 aggressive low-R, and 8 diagnostic exploit candidates.
    """

    if int(candidate_count) <= 0:
        raise ValueError("candidate_count must be positive.")
    base = default_lqr_weight_spec(primitive_id, tuning_stage=tuning_stage)
    sqrt2 = float(np.sqrt(2.0))
    candidates: list[LQRWeightSpec] = [
        _scaled_r61_weight_spec(
            base,
            primitive_id=primitive_id,
            candidate_index=0,
            family="nominal",
            tuning_stage=tuning_stage,
        )
    ]

    local_specs = (
        {"q_attitude": sqrt2},
        {"q_attitude": 1.0 / sqrt2},
        {"q_velocity": sqrt2},
        {"q_velocity": 1.0 / sqrt2},
        {"q_rates": sqrt2},
        {"q_rates": 1.0 / sqrt2},
        {"r_aileron": sqrt2, "r_elevator": sqrt2, "r_rudder": sqrt2},
    )
    for params in local_specs:
        candidates.append(
            _scaled_r61_weight_spec(
                base,
                primitive_id=primitive_id,
                candidate_index=len(candidates),
                family="local",
                tuning_stage=tuning_stage,
                **params,
            )
        )

    for r_scale in (1.5, 2.0, 3.0, 4.0):
        for q_rate_scale in (1.0, 1.35):
            candidates.append(
                _scaled_r61_weight_spec(
                    base,
                    primitive_id=primitive_id,
                    candidate_index=len(candidates),
                    family="conservative_high_r",
                    q_rates=q_rate_scale,
                    r_aileron=r_scale,
                    r_elevator=r_scale,
                    r_rudder=r_scale,
                    tuning_stage=tuning_stage,
                )
            )

    for r_scale in (0.35, 0.50, 0.70, 0.85):
        for q_rate_scale in (1.0, 1.35):
            candidates.append(
                _scaled_r61_weight_spec(
                    base,
                    primitive_id=primitive_id,
                    candidate_index=len(candidates),
                    family="aggressive_low_r",
                    q_rates=q_rate_scale,
                    r_aileron=r_scale,
                    r_elevator=r_scale,
                    r_rudder=r_scale,
                    tuning_stage=tuning_stage,
                )
            )

    exploit_bases = _diagnostic_exploit_bases(
        primitive_id=primitive_id,
        diagnostic_frame=diagnostic_frame,
        tuning_stage=tuning_stage,
    )
    exploit_transforms = (
        {"r_aileron": 0.70, "r_elevator": 0.70, "r_rudder": 0.70},
        {"r_aileron": 1.40, "r_elevator": 1.40, "r_rudder": 1.40},
        {"q_rates": 1.40},
        {"q_velocity": 1.30, "r_aileron": 0.85, "r_elevator": 0.85, "r_rudder": 0.85},
    )
    if exploit_bases:
        exploit_seed_specs: list[tuple[LQRWeightSpec, dict[str, float]]] = []
        for exploit_base in exploit_bases[:2]:
            for transform in exploit_transforms:
                exploit_seed_specs.append((exploit_base, transform))
    else:
        fallback_transforms = (
            {"q_velocity": 1.40, "r_aileron": 0.70, "r_elevator": 0.70, "r_rudder": 0.70},
            {"q_velocity": 1.40, "r_aileron": 1.40, "r_elevator": 1.40, "r_rudder": 1.40},
            {"q_velocity": 0.75, "r_aileron": 0.70, "r_elevator": 0.70, "r_rudder": 0.70},
            {"q_velocity": 0.75, "r_aileron": 1.40, "r_elevator": 1.40, "r_rudder": 1.40},
            {"q_velocity": 1.80, "r_aileron": 0.50, "r_elevator": 0.50, "r_rudder": 0.50},
            {"q_velocity": 1.80, "r_aileron": 2.00, "r_elevator": 2.00, "r_rudder": 2.00},
            {"q_velocity": 0.60, "r_aileron": 0.50, "r_elevator": 0.50, "r_rudder": 0.50},
            {"q_velocity": 0.60, "r_aileron": 2.00, "r_elevator": 2.00, "r_rudder": 2.00},
        )
        exploit_seed_specs = [(base, params) for params in fallback_transforms]

    for exploit_base, params in exploit_seed_specs[:8]:
        candidates.append(
            _scaled_r61_weight_spec(
                exploit_base,
                primitive_id=primitive_id,
                candidate_index=len(candidates),
                family="diagnostic_exploit",
                tuning_stage=tuning_stage,
                **params,
            )
        )

    return tuple(candidates[: int(candidate_count)])


def r6_1_candidate_family_counts(candidates: tuple[LQRWeightSpec, ...]) -> dict[str, int]:
    counts = {family: 0 for family, _ in R61_CANDIDATE_FAMILIES}
    for candidate in candidates:
        family = r6_1_candidate_family(candidate.weight_label)
        counts[family] = counts.get(family, 0) + 1
    return counts


def _present_series(frame: pd.DataFrame, column: str, default: object) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series([default] * len(frame), index=frame.index)


def _bool_series(frame: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    series = _present_series(frame, column, default)
    if series.dtype == bool:
        return series.fillna(default)
    lowered = series.astype(str).str.lower()
    return lowered.isin(("1", "true", "yes", "y"))


def _numeric_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(_present_series(frame, column, default), errors="coerce").fillna(default)


def _rate(mask: pd.Series) -> float:
    if len(mask) == 0:
        return 0.0
    return float(mask.astype(bool).mean())


def _percentile(values: pd.Series, percentile: float, default: float = 0.0) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return float(default)
    return float(np.percentile(numeric.to_numpy(dtype=float), percentile))


def _dominant_label(series: pd.Series, default: str = "none") -> str:
    if series.empty:
        return default
    counts = series.astype(str).replace("", default).value_counts()
    if counts.empty:
        return default
    return str(counts.index[0])


def _systemic_blocker_mask(frame: pd.DataFrame) -> pd.Series:
    reason = _present_series(frame, "hard_gate_reason", "").astype(str).str.lower()
    termination = _present_series(frame, "termination_cause", "").astype(str).str.lower()
    synthesis = _present_series(frame, "lqr_synthesis_status", LQR_SYNTHESIS_SOLVED).astype(str)
    sampled = _present_series(frame, "sampled_data_check", "sampled_stable").astype(str)
    controller_executable = _bool_series(frame, "controller_executable", default=True)
    systemic_reason = reason.str.contains(
        "nonfinite|corrupt|impossible|surrogate|controller|metadata|sampled|unstable|unsolved",
        regex=True,
    )
    systemic_termination = termination.str.contains(
        "nonfinite|corrupt|impossible|surrogate|controller|metadata|sampled|unstable|unsolved",
        regex=True,
    )
    return (
        systemic_reason
        | systemic_termination
        | synthesis.ne(LQR_SYNTHESIS_SOLVED)
        | sampled.ne("sampled_stable")
        | ~controller_executable
    )


def _metadata_complete_mask(frame: pd.DataFrame) -> pd.Series:
    required = ("candidate_weight_label", "lqr_Q_weights_json", "lqr_R_weights_json", "lqr_gain_checksum")
    mask = pd.Series([True] * len(frame), index=frame.index)
    for column in required:
        mask &= _present_series(frame, column, "").astype(str).str.len().gt(0)
    return mask


def r6_1_candidate_summary(
    frame: pd.DataFrame,
    thresholds: R61SelectionThresholds | None = None,
) -> pd.DataFrame:
    """Summarise rollout rows into R6.1 robust-scoring records."""

    thresholds = thresholds or R61SelectionThresholds()
    if frame is None or frame.empty:
        return pd.DataFrame(
            columns=(
                "primitive_id",
                "candidate_index",
                "candidate_weight_label",
                "row_count",
                "robust_score",
                "eligibility_status",
                "selection_reason",
            )
        )
    work = frame.copy()
    group_columns = [column for column in ("primitive_id", "candidate_index", "candidate_weight_label") if column in work]
    if "primitive_id" not in group_columns:
        raise ValueError("R6.1 candidate summary requires primitive_id.")
    if "candidate_index" not in group_columns:
        work["candidate_index"] = 0
        group_columns.append("candidate_index")
    if "candidate_weight_label" not in group_columns:
        work["candidate_weight_label"] = ""
        group_columns.append("candidate_weight_label")

    rows: list[dict[str, object]] = []
    for keys, subset in work.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = dict(zip(group_columns, keys, strict=True))
        hard_gate = _present_series(subset, "hard_gate_status", "").astype(str)
        outcome = _present_series(subset, "outcome_class", "").astype(str)
        boundary_use = _present_series(subset, "boundary_use_class", "").astype(str)
        start_family = _present_series(subset, "start_state_family", "").astype(str)
        continuation_valid = _bool_series(subset, "continuation_valid")
        terminal_useful = _bool_series(subset, "episode_terminal_useful")
        accepted_or_weak = outcome.isin(("accepted", "weak"))
        hard_failure = boundary_use.eq("hard_failure") | hard_gate.eq("blocked")
        systemic_mask = _systemic_blocker_mask(subset)
        metadata_complete = _metadata_complete_mask(subset)
        launch_rows = subset.loc[start_family.eq("launch_gate")]
        nominal_rows = subset.loc[start_family.eq("inflight_nominal")]
        launch_pass_rate = _rate(_present_series(launch_rows, "hard_gate_status", "").astype(str).eq("passed"))
        nominal_pass_rate = _rate(_present_series(nominal_rows, "hard_gate_status", "").astype(str).eq("passed"))
        systemic_ratio = _rate(systemic_mask)
        hard_failure_ratio = _rate(hard_failure)
        continuation_rate = _rate(continuation_valid)
        accepted_or_weak_rate = _rate(accepted_or_weak)
        terminal_useful_rate = _rate(terminal_useful)
        robust_lift_dwell_p20 = _percentile(_numeric_series(subset, "lift_dwell_time_s"), 20.0)
        robust_energy_p20 = _percentile(_numeric_series(subset, "energy_residual_m"), 20.0)
        wall_margin_p10 = _percentile(_numeric_series(subset, "minimum_wall_margin_m"), 10.0)
        saturation_p95 = _percentile(_numeric_series(subset, "saturation_fraction"), 95.0)
        uncertainty_p95 = _percentile(_numeric_series(subset, "uncertainty_m_s"), 95.0)
        speed_low_ratio = _rate(_present_series(subset, "hard_gate_reason", "").astype(str).str.contains("speed", case=False))
        floor_ceiling_ratio = _rate(
            _present_series(subset, "hard_gate_reason", "").astype(str).str.contains("floor|ceiling|z_boundary", case=False)
        )
        missing_metadata_ratio = _rate(~metadata_complete)
        synthesis_status = _dominant_label(_present_series(subset, "lqr_synthesis_status", LQR_SYNTHESIS_SOLVED))
        sampled_data_check = _dominant_label(_present_series(subset, "sampled_data_check", "sampled_stable"))
        if missing_metadata_ratio > 0.0:
            eligibility = "blocked"
            reason = "missing_reconstructability_metadata"
        elif synthesis_status != LQR_SYNTHESIS_SOLVED:
            eligibility = "blocked"
            reason = "unsolved_lqr"
        elif sampled_data_check != "sampled_stable":
            eligibility = "blocked"
            reason = "unstable_sampled_data"
        elif systemic_ratio > thresholds.systemic_blocker_max_ratio:
            eligibility = "blocked"
            reason = "nonfinite_corrupt_rollout_ratio_high"
        elif (
            launch_pass_rate >= thresholds.launch_gate_min_pass_rate
            and nominal_pass_rate >= thresholds.nominal_inflight_min_pass_rate
            and hard_failure_ratio <= thresholds.hard_failure_max_ratio
        ):
            eligibility = "eligible"
            reason = "eligible_preferred_thresholds"
        elif (
            launch_pass_rate >= thresholds.fallback_launch_gate_min_pass_rate
            and nominal_pass_rate >= thresholds.fallback_nominal_inflight_min_pass_rate
            and hard_failure_ratio <= thresholds.fallback_hard_failure_max_ratio
        ):
            eligibility = "accepted_fallback"
            reason = "accepted_fallback_below_preferred_threshold"
        else:
            eligibility = "rejected"
            reason = "below_minimum_empirical_gate"
        robust_score = (
            1.50 * continuation_rate
            + 1.00 * accepted_or_weak_rate
            + 0.55 * min(max(robust_lift_dwell_p20, 0.0), 5.0) / 5.0
            + 0.45 * min(max(robust_energy_p20, -5.0) + 5.0, 10.0) / 10.0
            + 0.25 * terminal_useful_rate
            + 0.20 * min(max(wall_margin_p10, -5.0) + 5.0, 20.0) / 20.0
            - 1.60 * hard_failure_ratio
            - 0.45 * speed_low_ratio
            - 0.55 * floor_ceiling_ratio
            - 0.40 * min(max(saturation_p95, 0.0), 1.0)
            - 0.20 * min(max(uncertainty_p95, 0.0), 5.0) / 5.0
        )
        rows.append(
            {
                **values,
                "candidate_family": r6_1_candidate_family(str(values.get("candidate_weight_label", ""))),
                "row_count": int(len(subset)),
                "accepted_or_weak_rate": accepted_or_weak_rate,
                "continuation_valid_rate": continuation_rate,
                "episode_terminal_useful_rate": terminal_useful_rate,
                "launch_gate_pass_rate": launch_pass_rate,
                "nominal_inflight_pass_rate": nominal_pass_rate,
                "systemic_blocker_ratio": systemic_ratio,
                "hard_failure_ratio": hard_failure_ratio,
                "missing_metadata_ratio": missing_metadata_ratio,
                "speed_low_ratio": speed_low_ratio,
                "floor_ceiling_ratio": floor_ceiling_ratio,
                "robust_lift_dwell_p20": robust_lift_dwell_p20,
                "robust_energy_residual_p20": robust_energy_p20,
                "wall_margin_p10": wall_margin_p10,
                "saturation_fraction_p95": saturation_p95,
                "uncertainty_m_s_p95": uncertainty_p95,
                "dominant_failure_label": _dominant_label(_present_series(subset, "hard_gate_reason", "none")),
                "lqr_Q_weights_json": _dominant_label(_present_series(subset, "lqr_Q_weights_json", "")),
                "lqr_R_weights_json": _dominant_label(_present_series(subset, "lqr_R_weights_json", "")),
                "lqr_gain_checksum": _dominant_label(_present_series(subset, "lqr_gain_checksum", "")),
                "lqr_synthesis_status": synthesis_status,
                "sampled_data_check": sampled_data_check,
                "robust_score": float(robust_score),
                "eligibility_status": eligibility,
                "selection_reason": reason,
            }
        )
    return pd.DataFrame(rows).sort_values(["primitive_id", "robust_score"], ascending=[True, False]).reset_index(drop=True)


def r6_1_strata_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    group_columns = [
        column
        for column in ("primitive_id", "candidate_index", "candidate_weight_label", "stage_label", "start_state_family")
        if column in frame.columns
    ]
    if not group_columns:
        return pd.DataFrame()
    work = frame.copy()
    work["_hard_gate_passed"] = _present_series(work, "hard_gate_status", "").astype(str).eq("passed")
    work["_accepted_or_weak"] = _present_series(work, "outcome_class", "").astype(str).isin(("accepted", "weak"))
    work["_continuation_valid"] = _bool_series(work, "continuation_valid")
    work["_episode_terminal_useful"] = _bool_series(work, "episode_terminal_useful")
    return (
        work.groupby(group_columns, dropna=False)
        .agg(
            row_count=("primitive_id", "size"),
            pass_rate=("_hard_gate_passed", "mean"),
            accepted_or_weak_rate=("_accepted_or_weak", "mean"),
            continuation_valid_rate=("_continuation_valid", "mean"),
            episode_terminal_useful_rate=("_episode_terminal_useful", "mean"),
        )
        .reset_index()
    )


def r6_1_failure_taxonomy(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=("primitive_id", "candidate_index", "failure_label", "row_count"))
    work = frame.copy()
    if "candidate_index" not in work.columns:
        work["candidate_index"] = 0
    reason = _present_series(work, "hard_gate_reason", "").astype(str)
    termination = _present_series(work, "termination_cause", "").astype(str)
    outcome = _present_series(work, "outcome_class", "").astype(str)
    work["failure_label"] = reason.where(reason.str.len().gt(0), termination)
    work["failure_label"] = work["failure_label"].where(work["failure_label"].astype(str).str.len().gt(0), outcome)
    work["failure_label"] = work["failure_label"].replace("", "none")
    return (
        work.groupby(["primitive_id", "candidate_index", "failure_label"], dropna=False)
        .size()
        .reset_index(name="row_count")
        .sort_values(["primitive_id", "candidate_index", "row_count"], ascending=[True, True, False])
    )


def r6_1_stage_e_decision(
    finalist_summary: pd.DataFrame,
    *,
    thresholds: R61SelectionThresholds | None = None,
) -> dict[str, dict[str, object]]:
    """Return per-primitive Stage E closure decisions after Stage D."""

    thresholds = thresholds or R61SelectionThresholds()
    decisions: dict[str, dict[str, object]] = {}
    if finalist_summary is None or finalist_summary.empty:
        return decisions
    for primitive_id, subset in finalist_summary.groupby("primitive_id", dropna=False):
        ordered = subset.sort_values("robust_score", ascending=False).reset_index(drop=True)
        viable = ordered[ordered["eligibility_status"].isin(("eligible", "accepted_fallback"))]
        if viable.empty:
            if ordered["eligibility_status"].eq("blocked").all():
                reason = "stage_e_not_applicable_all_systemic_blocked"
                required = False
            elif ordered["selection_reason"].astype(str).eq("missing_reconstructability_metadata").all():
                reason = "stage_e_not_applicable_no_metadata_complete_candidate"
                required = False
            else:
                reason = "stage_e_not_applicable_no_candidate_above_minimum_gate"
                required = False
            candidate_indices: list[int] = []
        else:
            top = viable.iloc[0]
            second_score = float(viable.iloc[1]["robust_score"]) if len(viable) > 1 else float("-inf")
            gap = float(top["robust_score"]) - second_score
            below_preferred = str(top["eligibility_status"]) == "accepted_fallback"
            tied_or_marginal = gap <= thresholds.score_tie_margin
            required = bool(below_preferred or tied_or_marginal)
            reason = (
                "stage_e_required_accepted_fallback_below_preferred"
                if below_preferred
                else ("stage_e_required_tied_or_marginal_finalists" if tied_or_marginal else "stage_e_not_required_clear_winner")
            )
            candidate_indices = [int(index) for index in viable.head(R61_STAGE_E_MAX_CANDIDATES_PER_PRIMITIVE)["candidate_index"]]
        decisions[str(primitive_id)] = {
            "primitive_id": str(primitive_id),
            "stage_e_required": required,
            "stage_e_reason": reason,
            "candidate_indices": candidate_indices,
        }
    return decisions


def lqr_tuning_schedule(
    *,
    candidate_count: int = 16,
    paired_tests_per_candidate: int = 50,
    fallback_mode: bool = False,
) -> LQRTuningSchedule:
    primitive_count = len(active_primitive_catalogue())
    rows = primitive_count * int(candidate_count) * int(paired_tests_per_candidate) * 2
    return LQRTuningSchedule(
        primitive_count=primitive_count,
        candidate_count=int(candidate_count),
        paired_tests_per_candidate=int(paired_tests_per_candidate),
        planned_rows=int(rows),
        fallback_mode=bool(fallback_mode),
        hard_gates_json=json.dumps(HARD_GATE_LABELS, separators=(",", ":")),
        soft_objective_terms_json=json.dumps(SOFT_OBJECTIVE_TERMS, separators=(",", ":")),
    )


def tuning_candidates_for_primitive(
    primitive: PrimitiveDefinition,
    *,
    candidate_count: int = 16,
) -> tuple[LQRTuningCandidate, ...]:
    rows = []
    for index, weight_spec in enumerate(
        candidate_weight_specs(
            primitive_id=primitive.primitive_id,
            candidate_count=candidate_count,
        )
    ):
        controller = synthesize_lqr_controller(primitive, weight_spec=weight_spec)
        rows.append(
            LQRTuningCandidate(
                primitive_id=primitive.primitive_id,
                candidate_index=int(index),
                candidate_weight_label=weight_spec.weight_label,
                controller_id=controller.controller_id,
                lqr_Q_weights_json=controller.lqr_Q_weights_json,
                lqr_R_weights_json=controller.lqr_R_weights_json,
                lqr_gain_checksum=controller.lqr_gain_checksum,
                lqr_synthesis_status=controller.lqr_synthesis_status,
                tuning_stage=controller.tuning_stage,
            )
        )
    return tuple(rows)


def tuning_candidate_row(candidate: LQRTuningCandidate) -> dict[str, object]:
    return asdict(candidate)


def tuning_schedule_row(schedule: LQRTuningSchedule) -> dict[str, object]:
    return asdict(schedule)
