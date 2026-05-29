from __future__ import annotations

from collections.abc import Callable

from context_conditioned_outcome import context_conditioned_outcome, lookup_outcome_for_identity
from viability_governor import DEFAULT_GOVERNOR_CONFIG, GOVERNOR_MODES, GovernorConfig, governor_candidate_row

CandidateBeliefFeaturesFn = Callable[[dict[str, object], dict[str, object]], dict[str, object] | None]
BASELINE_SHIELDED_MEMORY_POLICY_VERSION = "baseline_shielded_candidate_path_memory_safe_exploration_v1_4"
MEMORY_SWITCH_MIN_CONFIDENCE = 0.45
MEMORY_SWITCH_MIN_SCORE_MARGIN = 0.005
MEMORY_SWITCH_MAX_BASE_SCORE_DROP = 0.03
MEMORY_SWITCH_MAX_TRANSITION_SUCCESS_DROP = 0.02
MEMORY_SWITCH_MAX_HARD_FAILURE_RISK_INCREASE = 0.0
EXPLORATION_SWITCH_MIN_UNCERTAINTY = 0.55
EXPLORATION_SWITCH_MIN_SCORE_MARGIN = 0.0
EXPLORATION_SWITCH_MAX_BASE_SCORE_DROP = 0.01
EXPLORATION_SWITCH_MAX_TRANSITION_SUCCESS_DROP = 0.0
EXPLORATION_SWITCH_MAX_HARD_FAILURE_RISK_INCREASE = 0.0
EXPLORATION_SWITCH_ALLOW_CROSS_FAMILY = False
ADAPTIVE_SWITCH_MAX_PATH_EXIT_MARGIN_DROP_M = 0.05
ADAPTIVE_SWITCH_MIN_PATH_EXIT_MARGIN_M = 0.02


def select_compact_representative(
    *,
    representatives: list[dict[str, object]],
    outcome_rows_by_variant_id: dict[str, dict[str, object]],
    context: dict[str, object],
    governor_mode: str,
    policy_id: str = "",
    belief_features: dict[str, float] | None = None,
    candidate_belief_features: CandidateBeliefFeaturesFn | None = None,
    governor_config: GovernorConfig | None = None,
) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    """Return the highest-scoring viable compact representative and all candidate rows."""

    if governor_mode not in GOVERNOR_MODES:
        raise ValueError("governor_mode must be continuation_mode or terminal_episode_mode.")
    cfg = governor_config or DEFAULT_GOVERNOR_CONFIG
    candidate_rows = []
    for representative in representatives:
        outcome = _outcome_for_representative(
            representative,
            outcome_rows_by_variant_id,
            context=context,
            governor_mode=governor_mode,
        )
        features = belief_features
        if candidate_belief_features is not None:
            features = candidate_belief_features(representative, outcome) or belief_features
        candidate_rows.append(
            governor_candidate_row(
                representative=representative,
                outcome=outcome,
                context=context,
                governor_mode=governor_mode,
                policy_id=policy_id,
                belief_features=features,
                governor_config=cfg,
            )
        )
    _add_rank_diagnostics(candidate_rows)
    viable = [row for row in candidate_rows if bool(row.get("viable", False))]
    if not viable:
        return None, candidate_rows
    baseline_selected = sorted(
        viable,
        key=lambda row: (
            -float(row.get("base_score_without_memory", float("-inf"))),
            str(row.get("primitive_id", "")),
            str(row.get("primitive_variant_id", "")),
        ),
    )[0]
    memory_selected = sorted(
        viable,
        key=lambda row: (
            -float(row.get("total_score_with_memory_and_exploration", row.get("score", float("-inf")))),
            str(row.get("primitive_id", "")),
            str(row.get("primitive_variant_id", "")),
        ),
    )[0]
    selected = _baseline_shielded_memory_selection(
        viable_rows=viable,
        baseline_selected=baseline_selected,
        memory_selected=memory_selected,
        memory_active=candidate_belief_features is not None,
    )
    selected_score = float(selected.get("total_score_with_memory_and_exploration", selected.get("score", float("-inf"))))
    for row in candidate_rows:
        row["score_margin_to_selected"] = (
            selected_score - float(row.get("total_score_with_memory_and_exploration", row.get("score", float("-inf"))))
            if bool(row.get("viable", False))
            else float("inf")
        )
    return selected, candidate_rows


def _baseline_shielded_memory_selection(
    *,
    viable_rows: list[dict[str, object]],
    baseline_selected: dict[str, object],
    memory_selected: dict[str, object],
    memory_active: bool,
) -> dict[str, object]:
    baseline_variant_id = str(baseline_selected.get("primitive_variant_id", ""))
    memory_variant_id = str(memory_selected.get("primitive_variant_id", ""))
    if not memory_active:
        _mark_memory_shield_rows(
            viable_rows,
            baseline_selected=baseline_selected,
            memory_selected=memory_selected,
            accepted=True,
            status="not_active_no_candidate_path_memory",
        )
        return memory_selected
    if memory_variant_id == baseline_variant_id:
        _mark_memory_shield_rows(
            viable_rows,
            baseline_selected=baseline_selected,
            memory_selected=memory_selected,
            accepted=True,
            status="baseline_and_memory_winner_match",
        )
        return memory_selected
    accepted, status = _memory_switch_acceptance_status(
        baseline_selected=baseline_selected,
        memory_selected=memory_selected,
    )
    _mark_memory_shield_rows(
        viable_rows,
        baseline_selected=baseline_selected,
        memory_selected=memory_selected,
        accepted=accepted,
        status=status,
    )
    return memory_selected if accepted else baseline_selected


def _memory_switch_acceptance_status(
    *,
    baseline_selected: dict[str, object],
    memory_selected: dict[str, object],
) -> tuple[bool, str]:
    confidence = _float(memory_selected.get("belief_candidate_path_confidence", 0.0))
    uncertainty = _float(memory_selected.get("belief_uncertainty", 1.0), default=1.0)
    exploration_component = _float(memory_selected.get("exploration_score_component", 0.0))
    score_margin = _score(memory_selected, "total_score_with_memory_and_exploration") - _score(
        baseline_selected,
        "total_score_with_memory_and_exploration",
    )
    base_score_delta = _score(memory_selected, "base_score_without_memory") - _score(
        baseline_selected,
        "base_score_without_memory",
    )
    transition_delta = _float(memory_selected.get("transition_success_probability", 0.0)) - _float(
        baseline_selected.get("transition_success_probability", 0.0)
    )
    hard_failure_delta = _float(memory_selected.get("hard_failure_risk", 1.0)) - _float(
        baseline_selected.get("hard_failure_risk", 1.0)
    )
    path_margin_non_regressive = _candidate_path_margin_non_regressive(
        baseline_selected=baseline_selected,
        memory_selected=memory_selected,
    )
    exploration_family_safe = _exploration_family_switch_allowed(
        baseline_selected=baseline_selected,
        memory_selected=memory_selected,
    )
    memory_non_regressive = (
        base_score_delta >= -MEMORY_SWITCH_MAX_BASE_SCORE_DROP
        and transition_delta >= -MEMORY_SWITCH_MAX_TRANSITION_SUCCESS_DROP
        and hard_failure_delta <= MEMORY_SWITCH_MAX_HARD_FAILURE_RISK_INCREASE
        and path_margin_non_regressive
    )
    if (
        confidence >= MEMORY_SWITCH_MIN_CONFIDENCE
        and score_margin >= MEMORY_SWITCH_MIN_SCORE_MARGIN
        and memory_non_regressive
    ):
        return True, "accepted_confident_non_regressive_memory_switch"
    exploration_non_regressive = (
        base_score_delta >= -EXPLORATION_SWITCH_MAX_BASE_SCORE_DROP
        and transition_delta >= -EXPLORATION_SWITCH_MAX_TRANSITION_SUCCESS_DROP
        and hard_failure_delta <= EXPLORATION_SWITCH_MAX_HARD_FAILURE_RISK_INCREASE
        and path_margin_non_regressive
    )
    if (
        exploration_component > 0.0
        and uncertainty >= EXPLORATION_SWITCH_MIN_UNCERTAINTY
        and score_margin >= EXPLORATION_SWITCH_MIN_SCORE_MARGIN
        and exploration_non_regressive
        and exploration_family_safe
    ):
        return True, "accepted_shielded_uncertainty_directed_exploration_switch"
    if confidence < MEMORY_SWITCH_MIN_CONFIDENCE and exploration_component <= 0.0:
        return False, "rejected_low_candidate_path_memory_confidence"
    if exploration_component > 0.0 and uncertainty < EXPLORATION_SWITCH_MIN_UNCERTAINTY:
        return False, "rejected_exploration_uncertainty_too_low"
    if score_margin < min(MEMORY_SWITCH_MIN_SCORE_MARGIN, EXPLORATION_SWITCH_MIN_SCORE_MARGIN):
        return False, "rejected_adaptive_score_margin_too_small"
    if base_score_delta < -MEMORY_SWITCH_MAX_BASE_SCORE_DROP:
        return False, "rejected_base_score_drop_too_large"
    if transition_delta < -MEMORY_SWITCH_MAX_TRANSITION_SUCCESS_DROP:
        return False, "rejected_transition_success_regression"
    if hard_failure_delta > MEMORY_SWITCH_MAX_HARD_FAILURE_RISK_INCREASE:
        return False, "rejected_hard_failure_risk_regression"
    if not path_margin_non_regressive:
        return False, "rejected_candidate_path_exit_margin_regression"
    if exploration_component > 0.0 and not exploration_family_safe:
        return False, "rejected_exploration_cross_family_without_memory_confidence"
    return False, "rejected_adaptive_switch_guard_not_satisfied"


def _mark_memory_shield_rows(
    rows: list[dict[str, object]],
    *,
    baseline_selected: dict[str, object],
    memory_selected: dict[str, object],
    accepted: bool,
    status: str,
) -> None:
    baseline_variant_id = str(baseline_selected.get("primitive_variant_id", ""))
    memory_variant_id = str(memory_selected.get("primitive_variant_id", ""))
    score_margin = _score(memory_selected, "total_score_with_memory_and_exploration") - _score(
        baseline_selected,
        "total_score_with_memory_and_exploration",
    )
    base_score_delta = _score(memory_selected, "base_score_without_memory") - _score(
        baseline_selected,
        "base_score_without_memory",
    )
    transition_delta = _float(memory_selected.get("transition_success_probability", 0.0)) - _float(
        baseline_selected.get("transition_success_probability", 0.0)
    )
    hard_failure_delta = _float(memory_selected.get("hard_failure_risk", 1.0)) - _float(
        baseline_selected.get("hard_failure_risk", 1.0)
    )
    baseline_path_margin = _candidate_path_exit_margin(baseline_selected)
    memory_path_margin = _candidate_path_exit_margin(memory_selected)
    path_margin_delta = memory_path_margin - baseline_path_margin
    confidence = _float(memory_selected.get("belief_candidate_path_confidence", 0.0))
    uncertainty = _float(memory_selected.get("belief_uncertainty", 1.0), default=1.0)
    exploration_component = _float(memory_selected.get("exploration_score_component", 0.0))
    exploration_family_safe = _exploration_family_switch_allowed(
        baseline_selected=baseline_selected,
        memory_selected=memory_selected,
    )
    for row in rows:
        row["memory_shield_policy_version"] = BASELINE_SHIELDED_MEMORY_POLICY_VERSION
        row["memory_shield_status"] = str(status)
        row["memory_shield_accepted"] = bool(accepted)
        row["memory_shield_baseline_variant_id"] = baseline_variant_id
        row["memory_shield_memory_variant_id"] = memory_variant_id
        row["memory_shield_selected_variant_id"] = memory_variant_id if accepted else baseline_variant_id
        row["memory_shield_score_margin"] = float(score_margin)
        row["memory_shield_base_score_delta"] = float(base_score_delta)
        row["memory_shield_transition_success_delta"] = float(transition_delta)
        row["memory_shield_hard_failure_risk_delta"] = float(hard_failure_delta)
        row["memory_shield_baseline_path_exit_margin_m"] = float(baseline_path_margin)
        row["memory_shield_memory_path_exit_margin_m"] = float(memory_path_margin)
        row["memory_shield_path_exit_margin_delta_m"] = float(path_margin_delta)
        row["memory_shield_max_path_exit_margin_drop_m"] = float(ADAPTIVE_SWITCH_MAX_PATH_EXIT_MARGIN_DROP_M)
        row["memory_shield_min_path_exit_margin_m"] = float(ADAPTIVE_SWITCH_MIN_PATH_EXIT_MARGIN_M)
        row["memory_shield_candidate_path_confidence"] = float(confidence)
        row["memory_shield_candidate_path_uncertainty"] = float(uncertainty)
        row["memory_shield_exploration_score_component"] = float(exploration_component)
        row["memory_shield_exploration_cross_family_allowed"] = bool(exploration_family_safe)
        row["memory_shield_min_confidence"] = float(MEMORY_SWITCH_MIN_CONFIDENCE)
        row["memory_shield_min_score_margin"] = float(MEMORY_SWITCH_MIN_SCORE_MARGIN)
        row["memory_shield_exploration_min_uncertainty"] = float(EXPLORATION_SWITCH_MIN_UNCERTAINTY)


def _candidate_path_margin_non_regressive(
    *,
    baseline_selected: dict[str, object],
    memory_selected: dict[str, object],
) -> bool:
    baseline_margin = _candidate_path_exit_margin(baseline_selected)
    memory_margin = _candidate_path_exit_margin(memory_selected)
    if baseline_margin >= ADAPTIVE_SWITCH_MIN_PATH_EXIT_MARGIN_M and memory_margin < ADAPTIVE_SWITCH_MIN_PATH_EXIT_MARGIN_M:
        return False
    return bool(memory_margin + ADAPTIVE_SWITCH_MAX_PATH_EXIT_MARGIN_DROP_M >= baseline_margin)


def _exploration_family_switch_allowed(
    *,
    baseline_selected: dict[str, object],
    memory_selected: dict[str, object],
) -> bool:
    if EXPLORATION_SWITCH_ALLOW_CROSS_FAMILY:
        return True
    return _primitive_family(baseline_selected) == _primitive_family(memory_selected)


def _primitive_family(row: dict[str, object]) -> str:
    primitive_id = str(row.get("primitive_id", ""))
    if primitive_id:
        return primitive_id
    variant_id = str(row.get("primitive_variant_id", ""))
    if variant_id.startswith("primvar_"):
        variant_id = variant_id[len("primvar_") :]
    marker = "_transition_object"
    if marker in variant_id:
        return variant_id.split(marker, maxsplit=1)[0]
    return variant_id


def _candidate_path_exit_margin(row: dict[str, object]) -> float:
    wall_margin = _float(row.get("belief_candidate_path_exit_wall_margin_m", float("nan")), default=float("nan"))
    min_margin = _float(row.get("belief_candidate_path_exit_min_margin_m", float("nan")), default=float("nan"))
    if wall_margin != wall_margin and min_margin != min_margin:
        return float(row.get("governor_wall_margin_m", row.get("wall_margin_m", 0.0)) or 0.0)
    if wall_margin != wall_margin:
        return float(min_margin)
    if min_margin != min_margin:
        return float(wall_margin)
    return float(min(wall_margin, min_margin))


def _outcome_for_representative(
    representative: dict[str, object],
    outcome_rows_by_variant_id: dict[str, dict[str, object]],
    *,
    context: dict[str, object] | None = None,
    governor_mode: str = "",
) -> dict[str, object]:
    base = lookup_outcome_for_identity(
        identity=representative,
        outcome_rows_by_variant_id=outcome_rows_by_variant_id,
    )
    if context is None:
        return base
    return context_conditioned_outcome(
        representative=representative,
        base_outcome=base,
        context=context,
        governor_mode=governor_mode,
    )


def selector_decision_row(
    *,
    episode_id: str,
    primitive_step_index: int,
    policy_id: str,
    governor_mode: str,
    context: dict[str, object],
    selected: dict[str, object] | None,
    candidate_count: int,
    viable_count: int,
    governor_config: GovernorConfig | None = None,
) -> dict[str, object]:
    """Return a compact selector audit row."""

    cfg = governor_config or DEFAULT_GOVERNOR_CONFIG
    return {
        "episode_id": str(episode_id),
        "primitive_step_index": int(primitive_step_index),
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
        "governor_belief_weight": float(cfg.belief_weight),
        "governor_maximum_hard_failure_risk": float(cfg.maximum_hard_failure_risk),
        "wall_margin_m": float(context.get("wall_margin_m", 0.0)),
        "governor_wall_margin_m": float(context.get("governor_wall_margin_m", context.get("wall_margin_m", 0.0))),
        "candidate_count": int(candidate_count),
        "viable_count": int(viable_count),
        "decision_status": "selected_compact_representative" if selected else "blocked_no_viable_representative",
        "selected_compact_library_id": "" if selected is None else str(selected.get("compact_library_id", "")),
        "selected_primitive_variant_id": "" if selected is None else str(selected.get("primitive_variant_id", "")),
        "selected_primitive_id": "" if selected is None else str(selected.get("primitive_id", "")),
        "selected_entry_role": "" if selected is None else str(selected.get("entry_role", "")),
        "selected_transition_entry_class": "" if selected is None else str(selected.get("transition_entry_class", "")),
        "selected_controller_id": "" if selected is None else str(selected.get("controller_id", "")),
        "selected_score": float("-inf") if selected is None else float(selected.get("score", float("-inf"))),
        "selected_memory_shield_policy_version": "" if selected is None else str(selected.get("memory_shield_policy_version", "")),
        "selected_memory_shield_status": "" if selected is None else str(selected.get("memory_shield_status", "")),
        "selected_memory_shield_accepted": False if selected is None else bool(selected.get("memory_shield_accepted", False)),
        "selected_memory_shield_baseline_variant_id": "" if selected is None else str(selected.get("memory_shield_baseline_variant_id", "")),
        "selected_memory_shield_memory_variant_id": "" if selected is None else str(selected.get("memory_shield_memory_variant_id", "")),
        "selected_memory_shield_path_exit_margin_delta_m": (
            0.0 if selected is None else float(selected.get("memory_shield_path_exit_margin_delta_m", 0.0))
        ),
        "selected_memory_shield_exploration_cross_family_allowed": (
            False if selected is None else bool(selected.get("memory_shield_exploration_cross_family_allowed", False))
        ),
        "claim_status": "simulation_only_selector_decision",
    }


def _add_rank_diagnostics(rows: list[dict[str, object]]) -> None:
    viable = [row for row in rows if bool(row.get("viable", False))]
    with_memory = sorted(
        viable,
        key=lambda row: (
            -float(row.get("score_with_memory", float("-inf"))),
            str(row.get("primitive_id", "")),
            str(row.get("primitive_variant_id", "")),
        ),
    )
    without_memory = sorted(
        viable,
        key=lambda row: (
            -float(row.get("base_score_without_memory", float("-inf"))),
            str(row.get("primitive_id", "")),
            str(row.get("primitive_variant_id", "")),
        ),
    )
    with_exploration = sorted(
        viable,
        key=lambda row: (
            -float(row.get("total_score_with_memory_and_exploration", row.get("score", float("-inf")))),
            str(row.get("primitive_id", "")),
            str(row.get("primitive_variant_id", "")),
        ),
    )
    rank_with = {str(row.get("primitive_variant_id", "")): rank for rank, row in enumerate(with_memory, start=1)}
    rank_without = {str(row.get("primitive_variant_id", "")): rank for rank, row in enumerate(without_memory, start=1)}
    rank_explore = {str(row.get("primitive_variant_id", "")): rank for rank, row in enumerate(with_exploration, start=1)}
    for row in rows:
        variant_id = str(row.get("primitive_variant_id", ""))
        row["rank_with_memory"] = int(rank_with.get(variant_id, 0))
        row["rank_without_memory"] = int(rank_without.get(variant_id, 0))
        row["rank_with_memory_and_exploration"] = int(rank_explore.get(variant_id, 0))
        row["rank_change_due_to_memory"] = int(rank_without.get(variant_id, 0) - rank_with.get(variant_id, 0))
        row["rank_change_due_to_exploration"] = int(rank_with.get(variant_id, 0) - rank_explore.get(variant_id, 0))


def _score(row: dict[str, object], column: str) -> float:
    return _float(row.get(column, float("-inf")), default=float("-inf"))


def _float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result
