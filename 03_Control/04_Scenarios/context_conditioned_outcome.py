from __future__ import annotations

from primitive_timing_contract import PRIMITIVE_FINITE_HORIZON_S


CONTEXT_CONDITIONED_OUTCOME_MODEL_VERSION = "robust_context_conditioned_outcome_v1"
R10_CHANGED_CASE_BLOCK_IDS = {
    "nominal_single_fan_perturbations",
    "nominal_four_fan_perturbations",
    "shifted_single_fan_positions",
    "shifted_four_fan_positions",
    "active_fan_number_variation",
    "arena_wide_fan_position_generalisation",
}


def lookup_outcome_for_identity(
    *,
    identity: dict[str, object],
    outcome_rows_by_variant_id: dict[str, dict[str, object]],
) -> dict[str, object]:
    case_id = str(identity.get("library_size_case_id", ""))
    compact_id = str(identity.get("compact_library_id", ""))
    variant_id = str(identity.get("primitive_variant_id", ""))
    for key in (f"{case_id}|{compact_id}", f"{case_id}|{variant_id}|{compact_id}", compact_id, variant_id):
        if key and key in outcome_rows_by_variant_id:
            return outcome_rows_by_variant_id[key]
    return {}


def context_conditioned_outcome(
    *,
    representative: dict[str, object],
    base_outcome: dict[str, object],
    context: dict[str, object],
    governor_mode: str,
) -> dict[str, object]:
    """Return a robust, context-conditioned outcome row.

    The adapter is deliberately one-sided: it may reduce probabilities, cap soft
    reward proxies, or increase risk, but it must not make a representative look
    better than its base post-W3 evidence.
    """

    if not base_outcome:
        return {}

    row = dict(base_outcome)
    reasons: list[str] = []
    base_continuation = _probability(row.get("continuation_probability", 0.0))
    base_terminal = _probability(row.get("terminal_useful_probability", 0.0))
    base_risk = _probability(row.get("hard_failure_risk", 1.0))
    continuation = base_continuation
    terminal = base_terminal
    risk = base_risk
    expected_updraft = max(_float(row.get("expected_updraft_gain_proxy_m", 0.0)), 0.0)
    expected_dwell = max(_float(row.get("expected_lift_dwell_time_s", 0.0)), 0.0)

    sample_count = _sample_count(row)
    if sample_count is not None:
        if sample_count <= 0:
            return {}
        if sample_count < 5:
            continuation *= 0.35
            terminal *= 0.35
            risk = max(risk, 0.85)
            reasons.append("very_sparse_outcome_evidence")
        elif sample_count < 20:
            continuation *= 0.70
            terminal *= 0.70
            risk = max(risk, 0.55)
            reasons.append("sparse_outcome_evidence")
    else:
        reasons.append("sample_count_unavailable")

    environment_status = _environment_match_status(row, context)
    if environment_status == "mismatch":
        continuation *= 0.55
        terminal *= 0.60
        risk = max(risk, 0.55)
        reasons.append("environment_class_mismatch")
    elif environment_status == "unknown":
        continuation *= 0.85
        terminal *= 0.85
        risk = max(risk, 0.35)
        reasons.append("environment_coverage_unknown")

    if str(context.get("environment_block_id", "")) in R10_CHANGED_CASE_BLOCK_IDS:
        continuation *= 0.85
        terminal *= 0.85
        risk = max(risk, 0.35)
        reasons.append("r10_changed_case_conservative_penalty")

    local_uncertainty = max(_float(context.get("w_local_uncertainty_m_s", 0.0)), 0.0)
    if local_uncertainty > 0.0:
        factor = max(0.60, 1.0 - 0.12 * local_uncertainty)
        continuation *= factor
        terminal *= factor
        risk = max(risk, min(0.95, base_risk + 0.08 * local_uncertainty))
        reasons.append("local_uncertainty_conservative_penalty")

    governor_wall_margin = _float(context.get("governor_wall_margin_m", context.get("wall_margin_m", 0.0)))
    floor_margin = _float(context.get("floor_margin_m", 0.0))
    ceiling_margin = _float(context.get("ceiling_margin_m", 0.0))
    if min(floor_margin, ceiling_margin) < 0.0:
        continuation = 0.0
        terminal = 0.0
        risk = 1.0
        reasons.append("vertical_margin_negative")
    elif min(floor_margin, ceiling_margin) < 0.05:
        continuation *= 0.40
        terminal *= 0.40
        risk = max(risk, 0.85)
        reasons.append("vertical_margin_tight")
    if governor_wall_margin < 0.05 and str(governor_mode) != "terminal_episode_mode":
        continuation *= 0.40
        terminal *= 0.40
        risk = max(risk, 0.85)
        reasons.append("governor_wall_margin_tight")

    local_updraft_proxy = max(_float(context.get("w_wing_mean_m_s", 0.0)), 0.0) * float(PRIMITIVE_FINITE_HORIZON_S)
    if local_updraft_proxy <= 0.0:
        expected_updraft = 0.0
        expected_dwell = 0.0
        reasons.append("no_local_updraft_for_soft_reward")
    else:
        expected_updraft = min(expected_updraft, local_updraft_proxy + 0.05)
        expected_dwell = min(expected_dwell, float(PRIMITIVE_FINITE_HORIZON_S))
        reasons.append("local_updraft_soft_reward_capped")

    row.update(
        {
            "base_continuation_probability": base_continuation,
            "base_terminal_useful_probability": base_terminal,
            "base_hard_failure_risk": base_risk,
            "base_expected_updraft_gain_proxy_m": max(_float(base_outcome.get("expected_updraft_gain_proxy_m", 0.0)), 0.0),
            "base_expected_lift_dwell_time_s": max(_float(base_outcome.get("expected_lift_dwell_time_s", 0.0)), 0.0),
            "continuation_probability": _probability(continuation),
            "terminal_useful_probability": _probability(terminal),
            "hard_failure_risk": _probability(risk),
            "expected_updraft_gain_proxy_m": max(float(expected_updraft), 0.0),
            "expected_lift_dwell_time_s": max(float(expected_dwell), 0.0),
            "context_conditioned_outcome_model_version": CONTEXT_CONDITIONED_OUTCOME_MODEL_VERSION,
            "context_conditioning_policy": "robust_downgrade_only_never_optimistic",
            "context_conditioning_status": "applied",
            "context_conditioning_reasons": ";".join(reasons) if reasons else "matched_context_no_adjustment",
            "context_environment_mode": str(context.get("environment_mode", "")),
            "context_environment_block_id": str(context.get("environment_block_id", "")),
            "context_start_state_family": str(context.get("start_state_family", "")),
            "context_w_wing_mean_m_s": _float(context.get("w_wing_mean_m_s", 0.0)),
            "context_w_local_uncertainty_m_s": local_uncertainty,
            "context_actual_active_fan_count": int(_float(context.get("actual_active_fan_count", context.get("fan_count", 0)))),
            "context_representative_id": str(representative.get("compact_library_id", "")),
        }
    )
    return row


def _environment_match_status(outcome: dict[str, object], context: dict[str, object]) -> str:
    coverage = str(
        outcome.get(
            "environment_coverage",
            outcome.get("w3_environment_modes_seen", outcome.get("environment_mode_coverage", "")),
        )
    ).strip()
    if not coverage:
        return "unknown"
    context_mode = str(context.get("environment_mode", "")).strip()
    if context_mode and context_mode in coverage:
        return "match"
    context_class = _environment_class(context_mode)
    coverage_classes = _environment_classes(coverage)
    if not coverage_classes or context_class == "unknown":
        return "unknown"
    return "match" if context_class in coverage_classes else "mismatch"


def _environment_classes(value: str) -> set[str]:
    text = str(value).lower()
    classes: set[str] = set()
    if "dry" in text or "no_updraft" in text:
        classes.add("dry")
    if "four" in text or "4fan" in text or "multi" in text:
        classes.add("four")
    if "single" in text or "1fan" in text:
        classes.add("single")
    return classes


def _environment_class(value: str) -> str:
    text = str(value).lower()
    if "dry" in text or "no_updraft" in text:
        return "dry"
    if "four" in text or "4fan" in text or "multi" in text:
        return "four"
    if "single" in text or "1fan" in text:
        return "single"
    return "unknown"


def _sample_count(row: dict[str, object]) -> float | None:
    if "sample_count" in row:
        return _float(row.get("sample_count", 0.0))
    return None


def _probability(value: object) -> float:
    return float(max(0.0, min(1.0, _float(value, 0.0))))


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)
