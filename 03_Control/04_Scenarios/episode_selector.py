from __future__ import annotations

from collections.abc import Callable

from context_conditioned_outcome import context_conditioned_outcome, lookup_outcome_for_identity
from lqr_linearisation import LQR_LOCAL_OPERATING_SPEED_GRID_M_S, lqr_speed_bin_id
from transition_labels import entry_classes_for_state_class
from viability_governor import DEFAULT_GOVERNOR_CONFIG, GOVERNOR_MODES, GovernorConfig, governor_candidate_row

CandidateBeliefFeaturesFn = Callable[[dict[str, object], dict[str, object]], dict[str, object] | None]
BASELINE_SHIELDED_MEMORY_POLICY_VERSION = "baseline_shielded_candidate_path_memory_safe_exploration_v1_5"
REAL_TIME_COMPATIBILITY_PREFILTER_VERSION = "transition_entry_and_speed_bin_shortlist_v2"
SPEED_BIN_NEIGHBOUR_WINDOW = 0
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
    adaptive_memory_active: bool | None = None,
    governor_config: GovernorConfig | None = None,
) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    """Return the highest-scoring viable compact representative and all candidate rows."""

    if governor_mode not in GOVERNOR_MODES:
        raise ValueError("governor_mode must be continuation_mode or terminal_episode_mode.")
    cfg = governor_config or DEFAULT_GOVERNOR_CONFIG
    prefilter = _prefilter_representatives(representatives, context)
    candidate_rows = []
    for representative in prefilter["representatives"]:
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
    for row in candidate_rows:
        row.update(_prefilter_audit_fields(prefilter))
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
        memory_active=bool(candidate_belief_features is not None if adaptive_memory_active is None else adaptive_memory_active),
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
    baseline_memory_component = _float(baseline_selected.get("memory_score_component", 0.0))
    memory_candidate_memory_component = _float(memory_selected.get("memory_score_component", 0.0))
    memory_correction_delta = memory_candidate_memory_component - baseline_memory_component
    base_score_gap = _score(baseline_selected, "base_score_without_memory") - _score(
        memory_selected,
        "base_score_without_memory",
    )
    opportunity_ratio = memory_correction_delta / max(abs(base_score_gap), 1e-9)
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
        row["memory_shield_memory_correction_delta"] = float(memory_correction_delta)
        row["memory_shield_base_score_gap_to_baseline"] = float(base_score_gap)
        row["memory_shield_memory_opportunity_ratio"] = float(opportunity_ratio)
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
        row["memory_shield_baseline_memory_score_component"] = float(baseline_memory_component)
        row["memory_shield_memory_candidate_memory_score_component"] = float(memory_candidate_memory_component)
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


def _prefilter_representatives(
    representatives: list[dict[str, object]],
    context: dict[str, object],
) -> dict[str, object]:
    allowed_entry_classes = _allowed_entry_classes_for_context(context)
    entry_filtered = [
        representative
        for representative in representatives
        if _entry_class_allowed(representative, allowed_entry_classes)
    ]
    allowed_speed_bins = _allowed_speed_bins_for_context(context)
    speed_filtered = [
        representative
        for representative in entry_filtered
        if _speed_bin_allowed(representative, allowed_speed_bins)
    ]
    if allowed_speed_bins and entry_filtered and not speed_filtered:
        nearest_speed_bins = _nearest_populated_speed_bins(entry_filtered, allowed_speed_bins)
        selected = [
            representative
            for representative in entry_filtered
            if _speed_bin_allowed(representative, nearest_speed_bins)
        ]
        status = "active_entry_class_nearest_populated_speed_bin_fallback"
        allowed_speed_bins = nearest_speed_bins
    else:
        selected = speed_filtered if allowed_speed_bins else entry_filtered
        status = "active_entry_class_and_speed_window" if allowed_speed_bins else "active_entry_class_only"
    return {
        "representatives": selected,
        "total_candidate_count": int(len(representatives)),
        "entry_filtered_candidate_count": int(len(entry_filtered)),
        "evaluated_candidate_count": int(len(selected)),
        "skipped_candidate_count": int(max(0, len(representatives) - len(selected))),
        "allowed_entry_classes": ";".join(sorted(allowed_entry_classes)),
        "allowed_speed_bins": ";".join(sorted(allowed_speed_bins)),
        "status": status,
    }


def _allowed_entry_classes_for_context(context: dict[str, object]) -> set[str]:
    state_class = str(context.get("current_state_class", context.get("transition_current_state_class", ""))).strip()
    if not state_class:
        state_class = str(context.get("route_required_entry_class", "")).strip()
    if not state_class:
        return set()
    return set(entry_classes_for_state_class(state_class))


def _entry_class_allowed(representative: dict[str, object], allowed_entry_classes: set[str]) -> bool:
    if not allowed_entry_classes:
        return True
    entry_class = _representative_entry_class(representative)
    return not entry_class or entry_class in allowed_entry_classes


def _representative_entry_class(representative: dict[str, object]) -> str:
    value = str(representative.get("transition_entry_class", "")).strip()
    if value:
        return value
    pair = str(representative.get("transition_pair", "")).strip()
    if "->" in pair:
        return pair.split("->", 1)[0].strip()
    return ""


def _allowed_speed_bins_for_context(context: dict[str, object]) -> set[str]:
    context_speed_bin = str(context.get("current_local_lqr_speed_bin_id", context.get("local_lqr_speed_bin_id", ""))).strip()
    if not context_speed_bin:
        speed = context.get("current_speed_m_s", context.get("flight_speed_m_s", context.get("speed_m_s", "")))
        if str(speed).strip():
            context_speed_bin = lqr_speed_bin_id(_float(speed))
    if not context_speed_bin:
        return set()
    grid = [lqr_speed_bin_id(float(value)) for value in LQR_LOCAL_OPERATING_SPEED_GRID_M_S]
    if context_speed_bin not in grid:
        return {context_speed_bin}
    index = grid.index(context_speed_bin)
    lower = max(0, index - int(SPEED_BIN_NEIGHBOUR_WINDOW))
    upper = min(len(grid), index + int(SPEED_BIN_NEIGHBOUR_WINDOW) + 1)
    return set(grid[lower:upper])


def _speed_bin_allowed(representative: dict[str, object], allowed_speed_bins: set[str]) -> bool:
    if not allowed_speed_bins:
        return True
    candidate_speed_bin = _representative_speed_bin(representative)
    return not candidate_speed_bin or candidate_speed_bin in allowed_speed_bins


def _nearest_populated_speed_bins(
    representatives: list[dict[str, object]],
    allowed_speed_bins: set[str],
) -> set[str]:
    target_speeds = [_speed_from_bin_id(speed_bin) for speed_bin in allowed_speed_bins]
    target_speeds = [speed for speed in target_speeds if speed is not None]
    populated = sorted(
        {
            speed
            for speed in (_speed_from_bin_id(_representative_speed_bin(representative)) for representative in representatives)
            if speed is not None
        }
    )
    if not populated or not target_speeds:
        return set()
    target = sum(target_speeds) / float(len(target_speeds))
    nearest = min(populated, key=lambda speed: abs(float(speed) - float(target)))
    return {lqr_speed_bin_id(float(nearest))}


def _representative_speed_bin(representative: dict[str, object]) -> str:
    value = str(representative.get("local_lqr_speed_bin_id", representative.get("variant_local_lqr_speed_bin_id", ""))).strip()
    if value:
        return value
    speed = representative.get("local_lqr_reference_speed_m_s", representative.get("variant_local_lqr_reference_speed_m_s", ""))
    if str(speed).strip():
        return lqr_speed_bin_id(_float(speed))
    return ""


def _speed_from_bin_id(speed_bin_id: str) -> float | None:
    text = str(speed_bin_id).strip()
    prefix = "speed_bin_"
    suffix = "_m_s"
    if not text.startswith(prefix) or not text.endswith(suffix):
        return None
    value = text[len(prefix) : -len(suffix)].replace("p", ".")
    try:
        return _float(value)
    except Exception:
        return None


def _prefilter_audit_fields(prefilter: dict[str, object]) -> dict[str, object]:
    return {
        "selector_prefilter_version": REAL_TIME_COMPATIBILITY_PREFILTER_VERSION,
        "selector_prefilter_status": str(prefilter.get("status", "")),
        "selector_total_candidate_count": int(prefilter.get("total_candidate_count", 0)),
        "selector_entry_filtered_candidate_count": int(prefilter.get("entry_filtered_candidate_count", 0)),
        "selector_evaluated_candidate_count": int(prefilter.get("evaluated_candidate_count", 0)),
        "selector_skipped_candidate_count": int(prefilter.get("skipped_candidate_count", 0)),
        "selector_allowed_entry_classes": str(prefilter.get("allowed_entry_classes", "")),
        "selector_allowed_speed_bins": str(prefilter.get("allowed_speed_bins", "")),
    }


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
        "selector_prefilter_version": "" if selected is None else str(selected.get("selector_prefilter_version", "")),
        "selector_prefilter_status": "" if selected is None else str(selected.get("selector_prefilter_status", "")),
        "selector_total_candidate_count": int(candidate_count) if selected is None else int(selected.get("selector_total_candidate_count", candidate_count)),
        "selector_entry_filtered_candidate_count": int(candidate_count) if selected is None else int(selected.get("selector_entry_filtered_candidate_count", candidate_count)),
        "selector_evaluated_candidate_count": int(candidate_count) if selected is None else int(selected.get("selector_evaluated_candidate_count", candidate_count)),
        "selector_skipped_candidate_count": 0 if selected is None else int(selected.get("selector_skipped_candidate_count", 0)),
        "selector_allowed_entry_classes": "" if selected is None else str(selected.get("selector_allowed_entry_classes", "")),
        "selector_allowed_speed_bins": "" if selected is None else str(selected.get("selector_allowed_speed_bins", "")),
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
        "selected_memory_shield_score_margin": (
            0.0 if selected is None else float(selected.get("memory_shield_score_margin", 0.0))
        ),
        "selected_memory_shield_memory_correction_delta": (
            0.0 if selected is None else float(selected.get("memory_shield_memory_correction_delta", 0.0))
        ),
        "selected_memory_shield_base_score_gap_to_baseline": (
            0.0 if selected is None else float(selected.get("memory_shield_base_score_gap_to_baseline", 0.0))
        ),
        "selected_memory_shield_memory_opportunity_ratio": (
            0.0 if selected is None else float(selected.get("memory_shield_memory_opportunity_ratio", 0.0))
        ),
        "selected_memory_shield_base_score_delta": (
            0.0 if selected is None else float(selected.get("memory_shield_base_score_delta", 0.0))
        ),
        "selected_memory_shield_transition_success_delta": (
            0.0 if selected is None else float(selected.get("memory_shield_transition_success_delta", 0.0))
        ),
        "selected_memory_shield_hard_failure_risk_delta": (
            0.0 if selected is None else float(selected.get("memory_shield_hard_failure_risk_delta", 0.0))
        ),
        "selected_memory_shield_path_exit_margin_delta_m": (
            0.0 if selected is None else float(selected.get("memory_shield_path_exit_margin_delta_m", 0.0))
        ),
        "selected_memory_shield_candidate_path_confidence": (
            0.0 if selected is None else float(selected.get("memory_shield_candidate_path_confidence", 0.0))
        ),
        "selected_memory_shield_candidate_path_uncertainty": (
            1.0 if selected is None else float(selected.get("memory_shield_candidate_path_uncertainty", 1.0))
        ),
        "selected_memory_shield_exploration_score_component": (
            0.0 if selected is None else float(selected.get("memory_shield_exploration_score_component", 0.0))
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
