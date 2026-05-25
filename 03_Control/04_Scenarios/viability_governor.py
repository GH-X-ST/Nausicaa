from __future__ import annotations

from dataclasses import asdict, dataclass

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
class GovernorConfig:
    config_id: str
    minimum_wall_margin_m: float
    minimum_speed_margin_m_s: float
    maximum_hard_failure_risk: float
    continuation_weight: float
    terminal_weight: float
    hard_failure_weight: float
    energy_weight: float
    lift_dwell_weight: float
    wall_margin_weight: float
    belief_weight: float
    exploration_bonus_weight: float
    no_viable_penalty: float
    terminal_mode_bias: float
    continuation_mode_bias: float
    terminal_continuation_weight: float
    terminal_terminal_weight: float
    terminal_hard_failure_weight: float
    terminal_energy_weight: float
    terminal_lift_dwell_weight: float
    terminal_wall_margin_weight: float


DEFAULT_GOVERNOR_CONFIG = GovernorConfig(
    config_id="v49_default_governor",
    minimum_wall_margin_m=0.05,
    minimum_speed_margin_m_s=0.0,
    maximum_hard_failure_risk=0.75,
    continuation_weight=1.00,
    terminal_weight=-0.30,
    hard_failure_weight=-0.80,
    energy_weight=0.04,
    lift_dwell_weight=0.03,
    wall_margin_weight=0.02,
    belief_weight=0.05,
    exploration_bonus_weight=0.0,
    no_viable_penalty=-1.0,
    terminal_mode_bias=0.0,
    continuation_mode_bias=0.0,
    terminal_continuation_weight=0.25,
    terminal_terminal_weight=1.10,
    terminal_hard_failure_weight=-0.75,
    terminal_energy_weight=0.05,
    terminal_lift_dwell_weight=0.04,
    terminal_wall_margin_weight=0.01,
)


@dataclass(frozen=True)
class GovernorThresholds:
    minimum_wall_margin_m: float = DEFAULT_GOVERNOR_CONFIG.minimum_wall_margin_m
    minimum_speed_margin_m_s: float = DEFAULT_GOVERNOR_CONFIG.minimum_speed_margin_m_s
    maximum_hard_failure_risk: float = DEFAULT_GOVERNOR_CONFIG.maximum_hard_failure_risk


def governor_config_to_row(config: GovernorConfig) -> dict[str, object]:
    row = asdict(config)
    row["claim_status"] = "simulation_only_frozen_governor_config"
    return row


def governor_config_from_row(row: dict[str, object]) -> GovernorConfig:
    values = asdict(DEFAULT_GOVERNOR_CONFIG)
    values.update({key: row[key] for key in values if key in row})
    values["config_id"] = str(values["config_id"])
    for key, value in list(values.items()):
        if key != "config_id":
            values[key] = float(value)
    return GovernorConfig(**values)


def governor_candidate_row(
    *,
    representative: dict[str, object],
    outcome: dict[str, object],
    context: dict[str, object],
    governor_mode: str,
    policy_id: str = "",
    belief_features: dict[str, float] | None = None,
    thresholds: GovernorThresholds | None = None,
    governor_config: GovernorConfig | None = None,
) -> dict[str, object]:
    """Evaluate one compact-library representative in one local context."""

    if governor_mode not in GOVERNOR_MODES:
        raise ValueError("governor_mode must be continuation_mode or terminal_episode_mode.")
    cfg = governor_config or _config_from_thresholds(thresholds)
    rejection_reason = governor_rejection_reason(
        representative=representative,
        outcome=outcome,
        context=context,
        governor_mode=governor_mode,
        governor_config=cfg,
    )
    continuation_probability = _float(outcome.get("continuation_probability", 0.0))
    terminal_probability = _float(outcome.get("terminal_useful_probability", 0.0))
    hard_failure_risk = _float(outcome.get("hard_failure_risk", 1.0))
    energy = _float(outcome.get("expected_energy_residual_m", 0.0))
    dwell = _float(outcome.get("expected_lift_dwell_time_s", 0.0))
    wall_margin = _float(context.get("wall_margin_m", 0.0))
    belief_local = 0.0 if belief_features is None else _float(belief_features.get("belief_local_lift_m_s", 0.0))
    belief_mean = 0.0 if belief_features is None else _float(belief_features.get("belief_mean_lift_m_s", 0.0))
    belief_max = 0.0 if belief_features is None else _float(belief_features.get("belief_max_lift_m_s", 0.0))
    base_score = governor_score(
        governor_mode=governor_mode,
        continuation_probability=continuation_probability,
        terminal_useful_probability=terminal_probability,
        hard_failure_risk=hard_failure_risk,
        expected_energy_residual_m=energy,
        expected_lift_dwell_time_s=dwell,
        wall_margin_m=wall_margin,
        belief_local_lift_m_s=0.0,
        governor_config=cfg,
    )
    score_with_memory = governor_score(
        governor_mode=governor_mode,
        continuation_probability=continuation_probability,
        terminal_useful_probability=terminal_probability,
        hard_failure_risk=hard_failure_risk,
        expected_energy_residual_m=energy,
        expected_lift_dwell_time_s=dwell,
        wall_margin_m=wall_margin,
        belief_local_lift_m_s=belief_local,
        governor_config=cfg,
    )
    memory_component = float(score_with_memory) - float(base_score)
    return {
        "policy_id": str(policy_id),
        "context_id": str(context.get("context_id", "")),
        "W_layer": str(context.get("W_layer", "")),
        "environment_mode": str(context.get("environment_mode", "")),
        "start_state_family": str(context.get("start_state_family", "")),
        "governor_mode": str(governor_mode),
        "governor_config_id": cfg.config_id,
        **{f"governor_{key}": value for key, value in asdict(cfg).items() if key != "config_id"},
        "compact_library_id": str(representative.get("compact_library_id", "")),
        "primitive_variant_id": str(representative.get("primitive_variant_id", "")),
        "primitive_id": str(representative.get("primitive_id", "")),
        "entry_role": str(representative.get("entry_role", "")),
        "controller_id": str(representative.get("controller_id", "")),
        "viable": bool(rejection_reason == ""),
        "rejection_reason": rejection_reason,
        "score": float(score_with_memory if rejection_reason == "" else float("-inf")),
        "base_score_without_memory": float(base_score if rejection_reason == "" else float("-inf")),
        "memory_score_component": float(memory_component if rejection_reason == "" else 0.0),
        "score_with_memory": float(score_with_memory if rejection_reason == "" else float("-inf")),
        "score_margin_to_selected": 0.0,
        "rank_without_memory": 0,
        "rank_with_memory": 0,
        "rank_change_due_to_memory": 0,
        "belief_local_lift_m_s": belief_local,
        "belief_mean_lift_m_s": belief_mean,
        "belief_max_lift_m_s": belief_max,
        "continuation_probability": continuation_probability,
        "terminal_useful_probability": terminal_probability,
        "hard_failure_risk": hard_failure_risk,
        "expected_energy_residual_m": energy,
        "expected_lift_dwell_time_s": dwell,
        "wall_margin_m": wall_margin,
        "floor_margin_m": _float(context.get("floor_margin_m", 0.0)),
        "ceiling_margin_m": _float(context.get("ceiling_margin_m", 0.0)),
        "speed_margin_m_s": _float(context.get("speed_margin_m_s", 0.0)),
        "claim_status": "simulation_only_viability_governor_candidate",
    }


def governor_rejection_reason(
    *,
    representative: dict[str, object],
    outcome: dict[str, object],
    context: dict[str, object],
    governor_mode: str,
    thresholds: GovernorThresholds | None = None,
    governor_config: GovernorConfig | None = None,
) -> str:
    """Return the first claim-safe rejection reason for one candidate."""

    cfg = governor_config or _config_from_thresholds(thresholds)
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
    governor_config: GovernorConfig | None = None,
) -> float:
    """Return an interpretable deterministic selector score."""

    cfg = governor_config or DEFAULT_GOVERNOR_CONFIG
    if governor_mode == "terminal_episode_mode":
        return (
            cfg.terminal_mode_bias
            + cfg.terminal_terminal_weight * float(terminal_useful_probability)
            + cfg.terminal_continuation_weight * float(continuation_probability)
            + cfg.terminal_hard_failure_weight * float(hard_failure_risk)
            + cfg.terminal_energy_weight * float(expected_energy_residual_m)
            + cfg.terminal_lift_dwell_weight * float(expected_lift_dwell_time_s)
            + cfg.terminal_wall_margin_weight * float(wall_margin_m)
            + cfg.belief_weight * float(belief_local_lift_m_s)
        )
    return (
        cfg.continuation_mode_bias
        + cfg.continuation_weight * float(continuation_probability)
        + cfg.terminal_weight * float(terminal_useful_probability)
        + cfg.hard_failure_weight * float(hard_failure_risk)
        + cfg.energy_weight * float(expected_energy_residual_m)
        + cfg.lift_dwell_weight * float(expected_lift_dwell_time_s)
        + cfg.wall_margin_weight * float(wall_margin_m)
        + cfg.belief_weight * float(belief_local_lift_m_s)
    )


def _config_from_thresholds(thresholds: GovernorThresholds | None) -> GovernorConfig:
    if thresholds is None:
        return DEFAULT_GOVERNOR_CONFIG
    values = asdict(DEFAULT_GOVERNOR_CONFIG)
    values["config_id"] = "legacy_threshold_override"
    values["minimum_wall_margin_m"] = float(thresholds.minimum_wall_margin_m)
    values["minimum_speed_margin_m_s"] = float(thresholds.minimum_speed_margin_m_s)
    values["maximum_hard_failure_risk"] = float(thresholds.maximum_hard_failure_risk)
    return GovernorConfig(**values)


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
