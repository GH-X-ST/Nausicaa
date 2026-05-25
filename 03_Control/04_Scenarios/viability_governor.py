from __future__ import annotations

from dataclasses import dataclass

from primitive_variant_registry import start_family_is_compatible


GOVERNOR_MODES = ("continuation_mode", "terminal_episode_mode")
REJECTION_REASONS = (
    "entry_role_incompatible_start_family",
    "context_wall_margin_low",
    "context_vertical_safety_violation",
    "context_speed_margin_low",
    "timing_payload_checksum_missing",
    "known_hard_failure_boundary_high",
    "continuation_probability_zero",
    "terminal_and_continuation_probability_zero",
    "primitive_not_in_compact_library",
    "unsupported_feedback_or_latency_case",
)


@dataclass(frozen=True)
class GovernorThresholds:
    minimum_wall_margin_m: float = 0.05
    minimum_speed_margin_m_s: float = 0.0
    maximum_hard_failure_risk: float = 0.75


def governor_candidate_row(
    *,
    representative: dict[str, object],
    outcome: dict[str, object],
    context: dict[str, object],
    governor_mode: str,
    policy_id: str = "",
    belief_features: dict[str, float] | None = None,
    thresholds: GovernorThresholds | None = None,
) -> dict[str, object]:
    """Evaluate one compact-library representative in one local context."""

    if governor_mode not in GOVERNOR_MODES:
        raise ValueError("governor_mode must be continuation_mode or terminal_episode_mode.")
    cfg = thresholds or GovernorThresholds()
    rejection_reason = governor_rejection_reason(
        representative=representative,
        outcome=outcome,
        context=context,
        governor_mode=governor_mode,
        thresholds=cfg,
    )
    continuation_probability = _float(outcome.get("continuation_probability", 0.0))
    terminal_probability = _float(outcome.get("terminal_useful_probability", 0.0))
    hard_failure_risk = _float(outcome.get("hard_failure_risk", 1.0))
    energy = _float(outcome.get("expected_energy_residual_m", 0.0))
    dwell = _float(outcome.get("expected_lift_dwell_time_s", 0.0))
    wall_margin = _float(context.get("wall_margin_m", 0.0))
    belief = 0.0 if belief_features is None else _float(belief_features.get("belief_local_lift_m_s", 0.0))
    score = governor_score(
        governor_mode=governor_mode,
        continuation_probability=continuation_probability,
        terminal_useful_probability=terminal_probability,
        hard_failure_risk=hard_failure_risk,
        expected_energy_residual_m=energy,
        expected_lift_dwell_time_s=dwell,
        wall_margin_m=wall_margin,
        belief_local_lift_m_s=belief,
    )
    return {
        "policy_id": str(policy_id),
        "context_id": str(context.get("context_id", "")),
        "W_layer": str(context.get("W_layer", "")),
        "environment_mode": str(context.get("environment_mode", "")),
        "start_state_family": str(context.get("start_state_family", "")),
        "governor_mode": str(governor_mode),
        "compact_library_id": str(representative.get("compact_library_id", "")),
        "primitive_variant_id": str(representative.get("primitive_variant_id", "")),
        "primitive_id": str(representative.get("primitive_id", "")),
        "entry_role": str(representative.get("entry_role", "")),
        "controller_id": str(representative.get("controller_id", "")),
        "viable": bool(rejection_reason == ""),
        "rejection_reason": rejection_reason,
        "score": float(score if rejection_reason == "" else float("-inf")),
        "continuation_probability": continuation_probability,
        "terminal_useful_probability": terminal_probability,
        "hard_failure_risk": hard_failure_risk,
        "expected_energy_residual_m": energy,
        "expected_lift_dwell_time_s": dwell,
        "wall_margin_m": wall_margin,
        "floor_margin_m": _float(context.get("floor_margin_m", 0.0)),
        "ceiling_margin_m": _float(context.get("ceiling_margin_m", 0.0)),
        "speed_margin_m_s": _float(context.get("speed_margin_m_s", 0.0)),
        "belief_local_lift_m_s": belief,
        "claim_status": "simulation_only_viability_governor_candidate",
    }


def governor_rejection_reason(
    *,
    representative: dict[str, object],
    outcome: dict[str, object],
    context: dict[str, object],
    governor_mode: str,
    thresholds: GovernorThresholds | None = None,
) -> str:
    """Return the first claim-safe rejection reason for one candidate."""

    cfg = thresholds or GovernorThresholds()
    if not representative.get("compact_library_id") or not representative.get("primitive_variant_id"):
        return "primitive_not_in_compact_library"
    if not start_family_is_compatible(
        entry_role=str(representative.get("entry_role", "")),
        start_state_family=str(context.get("start_state_family", "")),
    ):
        return "entry_role_incompatible_start_family"
    if _float(context.get("wall_margin_m", 0.0)) < float(cfg.minimum_wall_margin_m):
        return "context_wall_margin_low"
    if _float(context.get("floor_margin_m", 0.0)) < 0.0 or _float(context.get("ceiling_margin_m", 0.0)) < 0.0:
        return "context_vertical_safety_violation"
    if _float(context.get("speed_margin_m_s", 0.0)) < float(cfg.minimum_speed_margin_m_s):
        return "context_speed_margin_low"
    if not _has_timing_payload(representative):
        return "timing_payload_checksum_missing"
    latency_case = str(context.get("latency_case", "nominal"))
    if latency_case not in {"none", "nominal", "conservative"}:
        return "unsupported_feedback_or_latency_case"
    hard_failure_risk = _float(outcome.get("hard_failure_risk", 1.0))
    if hard_failure_risk > float(cfg.maximum_hard_failure_risk):
        return "known_hard_failure_boundary_high"
    continuation_probability = _float(outcome.get("continuation_probability", 0.0))
    terminal_probability = _float(outcome.get("terminal_useful_probability", 0.0))
    if governor_mode == "continuation_mode" and continuation_probability <= 0.0:
        return "continuation_probability_zero"
    if governor_mode == "terminal_episode_mode" and max(terminal_probability, continuation_probability) <= 0.0:
        return "terminal_and_continuation_probability_zero"
    return ""


def governor_score(
    *,
    governor_mode: str,
    continuation_probability: float,
    terminal_useful_probability: float,
    hard_failure_risk: float,
    expected_energy_residual_m: float,
    expected_lift_dwell_time_s: float,
    wall_margin_m: float,
    belief_local_lift_m_s: float = 0.0,
) -> float:
    """Return an interpretable deterministic selector score."""

    if governor_mode == "terminal_episode_mode":
        return (
            1.10 * float(terminal_useful_probability)
            + 0.25 * float(continuation_probability)
            - 0.75 * float(hard_failure_risk)
            + 0.05 * float(expected_energy_residual_m)
            + 0.04 * float(expected_lift_dwell_time_s)
            + 0.01 * float(wall_margin_m)
            + 0.05 * float(belief_local_lift_m_s)
        )
    return (
        1.00 * float(continuation_probability)
        - 0.30 * float(terminal_useful_probability)
        - 0.80 * float(hard_failure_risk)
        + 0.04 * float(expected_energy_residual_m)
        + 0.03 * float(expected_lift_dwell_time_s)
        + 0.02 * float(wall_margin_m)
        + 0.05 * float(belief_local_lift_m_s)
    )


def _has_timing_payload(representative: dict[str, object]) -> bool:
    required = (
        "controller_id",
        "primitive_variant_id",
        "K_gain_checksum",
        "augmented_A_checksum",
        "augmented_B_checksum",
        "augmented_gain_checksum",
    )
    return all(bool(str(representative.get(key, ""))) for key in required)


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)
