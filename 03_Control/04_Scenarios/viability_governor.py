from __future__ import annotations

from dataclasses import asdict, dataclass

from lqr_linearisation import lqr_speed_bin_id
from primitive_timing_contract import PRIMITIVE_FINITE_HORIZON_S
from transition_labels import classify_state, entry_classes_for_state_class, transition_is_chain_compatible


GOVERNOR_MODES = ("continuation_mode", "terminal_episode_mode")
REJECTION_REASONS = (
    "entry_role_incompatible_start_family",
    "transition_entry_class_incompatible",
    "transition_success_probability_zero",
    "transition_predicted_exit_class_incompatible",
    "context_vertical_safety_violation",
    "timing_payload_checksum_missing",
    "known_hard_failure_boundary_high",
    "missing_outcome_evidence_for_candidate",
    "continuation_probability_zero",
    "terminal_and_continuation_probability_zero",
    "primitive_not_in_compact_library",
    "unsupported_feedback_or_latency_case",
)


@dataclass(frozen=True)
class GovernorConfig:
    config_id: str
    minimum_wall_margin_m: float
    maximum_hard_failure_risk: float
    continuation_weight: float
    terminal_weight: float
    hard_failure_weight: float
    updraft_gain_weight: float
    lift_dwell_weight: float
    belief_weight: float
    exploration_bonus_weight: float
    no_viable_penalty: float
    terminal_mode_bias: float
    continuation_mode_bias: float
    terminal_continuation_weight: float
    terminal_terminal_weight: float
    terminal_hard_failure_weight: float
    terminal_updraft_gain_weight: float
    terminal_lift_dwell_weight: float


DEFAULT_GOVERNOR_CONFIG = GovernorConfig(
    config_id="v53_viability_filtered_safe_exploration_governor_wall_0p10cm",
    minimum_wall_margin_m=0.001,
    maximum_hard_failure_risk=0.75,
    continuation_weight=1.00,
    terminal_weight=-0.30,
    hard_failure_weight=-0.80,
    updraft_gain_weight=0.04,
    lift_dwell_weight=0.03,
    belief_weight=0.05,
    exploration_bonus_weight=0.02,
    no_viable_penalty=-1.0,
    terminal_mode_bias=0.0,
    continuation_mode_bias=0.0,
    terminal_continuation_weight=0.25,
    terminal_terminal_weight=1.10,
    terminal_hard_failure_weight=-0.75,
    terminal_updraft_gain_weight=0.05,
    terminal_lift_dwell_weight=0.04,
)


@dataclass(frozen=True)
class GovernorThresholds:
    minimum_wall_margin_m: float = DEFAULT_GOVERNOR_CONFIG.minimum_wall_margin_m
    maximum_hard_failure_risk: float = DEFAULT_GOVERNOR_CONFIG.maximum_hard_failure_risk


def governor_config_to_row(config: GovernorConfig) -> dict[str, object]:
    row = asdict(config)
    row["claim_status"] = "simulation_only_frozen_governor_config"
    return row


def governor_config_from_row(row: dict[str, object]) -> GovernorConfig:
    values = asdict(DEFAULT_GOVERNOR_CONFIG)
    if "updraft_gain_weight" not in row and "energy_weight" in row:
        values["updraft_gain_weight"] = row["energy_weight"]
    if "terminal_updraft_gain_weight" not in row and "terminal_energy_weight" in row:
        values["terminal_updraft_gain_weight"] = row["terminal_energy_weight"]
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
    transition_success_probability = _float(
        outcome.get("transition_success_probability", outcome.get("transition_chain_compatible_rate", outcome.get("continuation_probability", 0.0)))
    )
    continuation_probability = transition_success_probability
    terminal_probability = _float(outcome.get("terminal_useful_probability", 0.0))
    hard_failure_risk = _float(outcome.get("hard_failure_risk", 1.0))
    expected_net_specific_energy_delta_m = _float(outcome.get("expected_energy_residual_m", 0.0))
    updraft_gain = _float(outcome.get("expected_updraft_gain_proxy_m", 0.0))
    score_updraft_gain = _contextual_updraft_gain_proxy_m(
        expected_updraft_gain_proxy_m=updraft_gain,
        context=context,
    )
    dwell = _float(outcome.get("expected_lift_dwell_time_s", 0.0))
    wall_margin = _governor_wall_margin(context)
    belief_features = belief_features or {}
    belief_local = _float(
        belief_features.get(
            "belief_local_lift_residual_m_s",
            belief_features.get("belief_local_lift_m_s", 0.0),
        )
    )
    belief_updraft_gain_residual = _float(
        belief_features.get(
            "belief_local_updraft_gain_residual_m",
            belief_features.get("belief_local_energy_residual_m", 0.0),
        )
    )
    belief_updraft_gain = _float(
        belief_features.get(
            "belief_local_updraft_gain_proxy_m",
            max(belief_updraft_gain_residual, 0.0),
        )
    )
    belief_dwell = _float(belief_features.get("belief_local_dwell_residual_s", 0.0))
    belief_mean = _float(belief_features.get("belief_mean_lift_m_s", belief_local))
    belief_max = _float(belief_features.get("belief_max_lift_m_s", belief_local))
    belief_uncertainty = _float(belief_features.get("belief_uncertainty", 1.0), default=1.0)
    belief_observation_count = int(_float(belief_features.get("belief_observation_count", 0.0)))
    history_length = int(_float(context.get("history_length", belief_features.get("history_length", belief_observation_count))))
    base_score = governor_score(
        governor_mode=governor_mode,
        continuation_probability=continuation_probability,
        terminal_useful_probability=terminal_probability,
        hard_failure_risk=hard_failure_risk,
        expected_updraft_gain_proxy_m=score_updraft_gain,
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
        expected_updraft_gain_proxy_m=score_updraft_gain,
        expected_lift_dwell_time_s=dwell,
        wall_margin_m=wall_margin,
        belief_local_lift_m_s=belief_updraft_gain_residual,
        governor_config=cfg,
    )
    memory_component = float(score_with_memory) - float(base_score)
    viable = bool(rejection_reason == "")
    exploration_component = _safe_exploration_bonus(
        viable=viable,
        belief_uncertainty=belief_uncertainty,
        history_length=history_length,
        governor_config=cfg,
    )
    total_score = float(score_with_memory) + float(exploration_component)
    library_size_case_id = str(
        representative.get(
            "library_size_case_id",
            outcome.get("library_size_case_id", context.get("library_size_case_id", "unknown_library_size_case")),
        )
    )
    return {
        "policy_id": str(policy_id),
        "context_id": str(context.get("context_id", "")),
        "W_layer": str(context.get("W_layer", "")),
        "environment_mode": str(context.get("environment_mode", "")),
        "start_state_family": str(context.get("start_state_family", "")),
        "launch_sequence_policy": str(context.get("launch_sequence_policy", "")),
        "launch_sequence_phase": str(context.get("launch_sequence_phase", "")),
        "route_required_entry_role": str(context.get("route_required_entry_role", "")),
        "route_required_entry_class": str(context.get("route_required_entry_class", "")),
        "route_reason": str(context.get("route_reason", "")),
        "governor_mode": str(governor_mode),
        "governor_config_id": cfg.config_id,
        **{f"governor_{key}": value for key, value in asdict(cfg).items() if key != "config_id"},
        "compact_library_id": str(representative.get("compact_library_id", "")),
        "primitive_variant_id": str(representative.get("primitive_variant_id", "")),
        "primitive_id": str(representative.get("primitive_id", "")),
        "entry_role": str(representative.get("entry_role", "")),
        "transition_entry_class": str(
            representative.get(
                "transition_entry_class",
                outcome.get("transition_entry_class", ""),
            )
        ),
        "candidate_local_lqr_speed_bin_id": _candidate_speed_bin(representative=representative, outcome=outcome),
        "context_local_lqr_speed_bin_id": _context_speed_bin(context),
        "controller_id": str(representative.get("controller_id", "")),
        "library_size_case_id": library_size_case_id,
        "library_size_human_label": str(
            representative.get("library_size_human_label", outcome.get("library_size_human_label", ""))
        ),
        "viable": viable,
        "rejection_reason": rejection_reason,
        "score": float(total_score if viable else float("-inf")),
        "exploit_score_component": float(base_score if viable else float("-inf")),
        "base_score_without_memory": float(base_score if rejection_reason == "" else float("-inf")),
        "memory_score_component": float(memory_component if viable else 0.0),
        "memory_residual_score_component": float(memory_component if viable else 0.0),
        "exploration_score_component": float(exploration_component if viable else 0.0),
        "score_with_memory": float(score_with_memory if rejection_reason == "" else float("-inf")),
        "total_score_with_memory_and_exploration": float(total_score if viable else float("-inf")),
        "safe_exploration_status": "applied_after_viability_filter" if viable else "not_applied_rejected_before_exploration",
        "score_margin_to_selected": 0.0,
        "rank_without_memory": 0,
        "rank_with_memory": 0,
        "rank_with_memory_and_exploration": 0,
        "rank_change_due_to_memory": 0,
        "rank_change_due_to_exploration": 0,
        "history_length": history_length,
        "belief_version": str(belief_features.get("belief_version", "")),
        "belief_local_lift_m_s": belief_local,
        "belief_local_lift_residual_m_s": belief_local,
        "belief_local_updraft_gain_proxy_m": belief_updraft_gain,
        "belief_local_updraft_gain_residual_m": belief_updraft_gain_residual,
        "belief_local_energy_residual_m": belief_updraft_gain_residual,
        "belief_energy_residual_alias_status": "legacy_alias_for_updraft_gain_residual_not_total_energy",
        "belief_local_dwell_residual_s": belief_dwell,
        "belief_mean_lift_m_s": belief_mean,
        "belief_max_lift_m_s": belief_max,
        "belief_uncertainty": belief_uncertainty,
        "belief_observation_count": belief_observation_count,
        "belief_effective_observation_count": _float(belief_features.get("belief_effective_observation_count", belief_observation_count)),
        "belief_recency_weight": _float(belief_features.get("belief_recency_weight", 0.0)),
        "belief_observation_age": int(_float(belief_features.get("belief_observation_age", 0.0))),
        "belief_direction_bin": int(_float(belief_features.get("belief_direction_bin", 0.0))),
        "belief_z_bin": int(_float(belief_features.get("belief_z_bin", 0.0))),
        "belief_update_count": int(_float(belief_features.get("belief_update_count", 0.0))),
        "belief_memory_policy_version": str(belief_features.get("belief_memory_policy_version", "")),
        "belief_candidate_path_probe_count": int(_float(belief_features.get("belief_candidate_path_probe_count", 0.0))),
        "belief_candidate_path_lookahead_s": _float(belief_features.get("belief_candidate_path_lookahead_s", 0.0)),
        "belief_candidate_path_confidence": _float(belief_features.get("belief_candidate_path_confidence", 0.0)),
        "belief_candidate_path_updraft_residual_uncapped_m": _float(
            belief_features.get("belief_candidate_path_updraft_residual_uncapped_m", belief_updraft_gain_residual)
        ),
        "belief_candidate_path_updraft_residual_cap_m": _float(belief_features.get("belief_candidate_path_updraft_residual_cap_m", 0.0)),
        "belief_candidate_path_reference_bank_rad": _float(belief_features.get("belief_candidate_path_reference_bank_rad", 0.0)),
        "belief_candidate_path_heading_offset_rad": _float(belief_features.get("belief_candidate_path_heading_offset_rad", 0.0)),
        "belief_candidate_path_speed_m_s": _float(belief_features.get("belief_candidate_path_speed_m_s", 0.0)),
        "belief_candidate_path_exit_x_w_m": _float(belief_features.get("belief_candidate_path_exit_x_w_m", 0.0)),
        "belief_candidate_path_exit_y_w_m": _float(belief_features.get("belief_candidate_path_exit_y_w_m", 0.0)),
        "belief_candidate_path_exit_z_w_m": _float(belief_features.get("belief_candidate_path_exit_z_w_m", 0.0)),
        "belief_candidate_path_exit_direction_rad": _float(belief_features.get("belief_candidate_path_exit_direction_rad", 0.0)),
        "continuation_probability": continuation_probability,
        "transition_success_probability": transition_success_probability,
        "transition_chain_compatible_rate": _float(
            outcome.get("transition_chain_compatible_rate", transition_success_probability)
        ),
        "transition_exit_classes_seen": str(outcome.get("transition_exit_classes_seen", "")),
        "transition_pairs_seen": str(outcome.get("transition_pairs_seen", "")),
        "terminal_useful_probability": terminal_probability,
        "hard_failure_risk": hard_failure_risk,
        "expected_net_specific_energy_delta_m": expected_net_specific_energy_delta_m,
        "expected_energy_residual_alias_status": "legacy_outcome_column_not_used_for_governor_soft_score",
        "expected_updraft_gain_proxy_m": updraft_gain,
        "score_updraft_gain_proxy_m": score_updraft_gain,
        "context_conditioned_outcome_score_version": "context_limited_updraft_gain_proxy_v2_no_energy_residual_score",
        "expected_lift_dwell_time_s": dwell,
        "wall_margin_m": _float(context.get("wall_margin_m", wall_margin)),
        "all_wall_margin_m": _float(context.get("all_wall_margin_m", context.get("wall_margin_m", 0.0))),
        "front_wall_margin_m": _float(context.get("front_wall_margin_m", context.get("wall_margin_m", 0.0))),
        "left_wall_margin_m": _float(context.get("left_wall_margin_m", context.get("wall_margin_m", 0.0))),
        "right_wall_margin_m": _float(context.get("right_wall_margin_m", context.get("wall_margin_m", 0.0))),
        "rear_wall_margin_m": _float(context.get("rear_wall_margin_m", context.get("wall_margin_m", 0.0))),
        "governor_wall_margin_m": wall_margin,
        "floor_margin_m": _float(context.get("floor_margin_m", 0.0)),
        "ceiling_margin_m": _float(context.get("ceiling_margin_m", 0.0)),
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
    entry_role = str(representative.get("entry_role", ""))
    candidate_entry_class = _candidate_entry_class(representative=representative, outcome=outcome)
    start_state_family = str(context.get("start_state_family", ""))
    entry_class = str(context.get("current_state_class", context.get("transition_current_state_class", "")))
    if not entry_class:
        entry_class = classify_state(start_state_family=start_state_family)
    allowed_entry_classes = set(entry_classes_for_state_class(entry_class))
    if candidate_entry_class not in allowed_entry_classes:
        return "transition_entry_class_incompatible"
    if str(context.get("start_state_family", "")) and entry_class == "launch_gate" and start_state_family != "launch_gate":
        return "entry_role_incompatible_start_family"
    if _float(context.get("floor_margin_m", 0.0)) < 0.0 or _float(context.get("ceiling_margin_m", 0.0)) < 0.0:
        return "context_vertical_safety_violation"
    if not _has_timing_payload(representative):
        return "timing_payload_checksum_missing"
    latency_case = str(context.get("latency_case", "nominal"))
    if latency_case not in {"none", "nominal", "conservative"}:
        return "unsupported_feedback_or_latency_case"
    if not _has_outcome_evidence(outcome):
        return "missing_outcome_evidence_for_candidate"
    hard_failure_risk = _float(outcome.get("hard_failure_risk", 1.0))
    if hard_failure_risk > float(cfg.maximum_hard_failure_risk):
        return "known_hard_failure_boundary_high"
    transition_success_probability = _float(
        outcome.get("transition_success_probability", outcome.get("transition_chain_compatible_rate", outcome.get("continuation_probability", 0.0)))
    )
    if transition_success_probability <= 0.0:
        return "transition_success_probability_zero"
    if not transition_is_chain_compatible(
        entry_role=entry_role,
        entry_class=candidate_entry_class,
        exit_class=_predicted_exit_class(outcome, entry_role, candidate_entry_class),
    ):
        return "transition_predicted_exit_class_incompatible"
    continuation_probability = transition_success_probability
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
    expected_updraft_gain_proxy_m: float,
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
            + cfg.terminal_updraft_gain_weight * float(expected_updraft_gain_proxy_m)
            + cfg.terminal_lift_dwell_weight * float(expected_lift_dwell_time_s)
            + cfg.belief_weight * float(belief_local_lift_m_s)
        )
    return (
        cfg.continuation_mode_bias
        + cfg.continuation_weight * float(continuation_probability)
        + cfg.terminal_weight * float(terminal_useful_probability)
        + cfg.hard_failure_weight * float(hard_failure_risk)
        + cfg.updraft_gain_weight * float(expected_updraft_gain_proxy_m)
        + cfg.lift_dwell_weight * float(expected_lift_dwell_time_s)
        + cfg.belief_weight * float(belief_local_lift_m_s)
    )


def _config_from_thresholds(thresholds: GovernorThresholds | None) -> GovernorConfig:
    if thresholds is None:
        return DEFAULT_GOVERNOR_CONFIG
    values = asdict(DEFAULT_GOVERNOR_CONFIG)
    values["config_id"] = "legacy_threshold_override"
    values["minimum_wall_margin_m"] = float(thresholds.minimum_wall_margin_m)
    values["maximum_hard_failure_risk"] = float(thresholds.maximum_hard_failure_risk)
    return GovernorConfig(**values)


def _governor_wall_margin(context: dict[str, object]) -> float:
    return _float(context.get("governor_wall_margin_m", context.get("wall_margin_m", 0.0)))


def _has_timing_payload(representative: dict[str, object]) -> bool:
    required = (
        "controller_id",
        "primitive_variant_id",
        "K_gain_checksum",
        "augmented_A_checksum",
        "augmented_B_checksum",
        "augmented_gain_checksum",
        "finite_horizon_s",
        "controller_input_slots_per_primitive",
        "controller_input_update_period_s",
        "primitive_timing_contract_version",
    )
    return all(bool(str(representative.get(key, ""))) for key in required)


def _has_outcome_evidence(outcome: dict[str, object]) -> bool:
    if not outcome:
        return False
    if "sample_count" in outcome and _float(outcome.get("sample_count", 0.0)) <= 0.0:
        return False
    evidence_keys = (
        "transition_success_probability",
        "transition_chain_compatible_rate",
        "continuation_probability",
        "terminal_useful_probability",
        "hard_failure_risk",
        "expected_updraft_gain_proxy_m",
        "expected_lift_dwell_time_s",
    )
    return any(key in outcome for key in evidence_keys)


def _candidate_entry_class(*, representative: dict[str, object], outcome: dict[str, object]) -> str:
    for source in (representative, outcome):
        value = str(source.get("transition_entry_class", "")).strip()
        if value:
            return value
        pair = str(source.get("transition_pair", "")).strip()
        if "->" in pair:
            return pair.split("->", 1)[0].strip()
    return ""


def _candidate_speed_bin(*, representative: dict[str, object], outcome: dict[str, object]) -> str:
    for source in (representative, outcome):
        value = str(source.get("local_lqr_speed_bin_id", source.get("variant_local_lqr_speed_bin_id", ""))).strip()
        if value:
            return value
        speed = source.get("local_lqr_reference_speed_m_s", source.get("variant_local_lqr_reference_speed_m_s", ""))
        if str(speed).strip() and str(speed).strip().lower() != "nan":
            return lqr_speed_bin_id(_float(speed, default=0.0))
    return ""


def _context_speed_bin(context: dict[str, object]) -> str:
    value = str(context.get("local_lqr_speed_bin_id", context.get("current_local_lqr_speed_bin_id", ""))).strip()
    if value:
        return value
    for key in ("current_speed_m_s", "flight_speed_m_s", "speed_m_s"):
        raw = context.get(key, "")
        if str(raw).strip() and str(raw).strip().lower() != "nan":
            return lqr_speed_bin_id(_float(raw, default=0.0))
    return ""


def _predicted_exit_class(outcome: dict[str, object], entry_role: str, entry_class: str = "") -> str:
    classes = str(outcome.get("transition_exit_classes_seen", "")).replace(",", ";").split(";")
    classes = [item.strip() for item in classes if item.strip()]
    if classes:
        for candidate in classes:
            if transition_is_chain_compatible(entry_role=entry_role, entry_class=entry_class, exit_class=candidate):
                return candidate
        return classes[0]
    if _float(outcome.get("transition_success_probability", outcome.get("continuation_probability", 0.0))) > 0.0:
        return "post_launch_degraded" if str(entry_class) == "launch_gate" else "inflight_stable"
    if _float(outcome.get("terminal_useful_probability", 0.0)) > 0.0:
        return "safe_terminal"
    return "hard_failure"


def _contextual_updraft_gain_proxy_m(
    *,
    expected_updraft_gain_proxy_m: float,
    context: dict[str, object],
) -> float:
    local_wing_lift = max(_float(context.get("w_wing_mean_m_s", 0.0)), 0.0)
    local_one_primitive_proxy = local_wing_lift * float(PRIMITIVE_FINITE_HORIZON_S)
    expected = max(float(expected_updraft_gain_proxy_m), 0.0)
    if expected <= 0.0:
        return float(local_one_primitive_proxy)
    context_tolerance = 0.05 if local_one_primitive_proxy > 0.0 else 0.0
    return float(min(expected, local_one_primitive_proxy + context_tolerance))


def _safe_exploration_bonus(
    *,
    viable: bool,
    belief_uncertainty: float,
    history_length: int,
    governor_config: GovernorConfig,
) -> float:
    if not viable:
        return 0.0
    attenuation = 1.0 / max(1.0, float(history_length + 1) ** 0.5)
    return float(governor_config.exploration_bonus_weight) * max(0.0, float(belief_uncertainty)) * attenuation


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)
