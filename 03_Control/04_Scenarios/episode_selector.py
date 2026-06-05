from __future__ import annotations

from collections.abc import Callable

from context_conditioned_outcome import context_conditioned_outcome, lookup_outcome_for_identity
from lqr_linearisation import LQR_LOCAL_OPERATING_SPEED_GRID_M_S, lqr_speed_bin_id
from transition_labels import entry_classes_for_state_class
from viability_governor import DEFAULT_GOVERNOR_CONFIG, GOVERNOR_MODES, GovernorConfig, governor_candidate_row

CandidateBeliefFeaturesFn = Callable[[dict[str, object], dict[str, object]], dict[str, object] | None]
BASELINE_SHIELDED_MEMORY_POLICY_VERSION = "cost_benefit_spatial_flow_memory_governor_v4_1"
REAL_TIME_COMPATIBILITY_PREFILTER_VERSION = "transition_entry_and_speed_bin_shortlist_v3_neighbour_bin"
SPEED_BIN_NEIGHBOUR_WINDOW = 1
MEMORY_COST_BENEFIT_SHORTLIST_TOP_K = 4
MEMORY_COST_BENEFIT_SHORTLIST_SCORE_GAP = 0.35
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
    candidate_row_mode: str = "diagnostic",
) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    """Return the highest-scoring viable compact representative and all candidate rows."""

    if governor_mode not in GOVERNOR_MODES:
        raise ValueError("governor_mode must be continuation_mode or terminal_episode_mode.")
    if candidate_row_mode not in {"diagnostic", "controller"}:
        raise ValueError("candidate_row_mode must be diagnostic or controller.")
    cfg = governor_config or DEFAULT_GOVERNOR_CONFIG
    include_diagnostics = str(candidate_row_mode) == "diagnostic"
    prefilter = _prefilter_representatives(representatives, context)
    candidate_rows = []
    candidate_sources: dict[str, tuple[dict[str, object], dict[str, object]]] = {}
    for representative in prefilter["representatives"]:
        outcome = _outcome_for_representative(
            representative,
            outcome_rows_by_variant_id,
            context=context,
            governor_mode=governor_mode,
        )
        variant_id = str(representative.get("primitive_variant_id", outcome.get("primitive_variant_id", "")))
        candidate_sources[variant_id] = (representative, outcome)
        features = belief_features
        if candidate_belief_features is not None:
            features = _candidate_memory_features(
                candidate_belief_features,
                representative,
                outcome,
                memory_query_mode="geometry_only",
            ) or belief_features
        candidate_rows.append(
            governor_candidate_row(
                representative=representative,
                outcome=outcome,
                context=context,
                governor_mode=governor_mode,
                policy_id=policy_id,
                belief_features=features,
                governor_config=cfg,
                include_diagnostics=include_diagnostics,
            )
        )
    for row in candidate_rows:
        row.update(_prefilter_audit_fields(prefilter))
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
    memory_active = bool(candidate_belief_features is not None if adaptive_memory_active is None else adaptive_memory_active)
    if memory_active and candidate_belief_features is not None:
        _refresh_memory_shortlist_rows(
            candidate_rows=candidate_rows,
            candidate_sources=candidate_sources,
            candidate_belief_features=candidate_belief_features,
            belief_features=belief_features,
            context=context,
            governor_mode=governor_mode,
            policy_id=policy_id,
            prefilter=prefilter,
            baseline_selected=baseline_selected,
            governor_config=cfg,
            include_diagnostics=include_diagnostics,
        )
        viable = [row for row in candidate_rows if bool(row.get("viable", False))]
        baseline_selected = sorted(
            viable,
            key=lambda row: (
                -float(row.get("base_score_without_memory", float("-inf"))),
                str(row.get("primitive_id", "")),
                str(row.get("primitive_variant_id", "")),
            ),
        )[0]
    else:
        for row in candidate_rows:
            row["memory_shortlist_selected"] = False
            row["memory_shortlist_policy"] = "not_active"
            row["memory_shortlist_reason"] = "memory_not_active"
    _apply_bounded_memory_objective(
        viable_rows=viable,
        baseline_selected=baseline_selected,
        memory_active=memory_active,
        governor_config=cfg,
    )
    _add_rank_diagnostics(candidate_rows)
    memory_ranked = sorted(
        viable,
        key=lambda row: (
            -float(row.get("total_score_with_memory_and_exploration", row.get("score", float("-inf")))),
            str(row.get("primitive_id", "")),
            str(row.get("primitive_variant_id", "")),
        ),
    )
    selected = _baseline_shielded_memory_selection(
        viable_rows=viable,
        baseline_selected=baseline_selected,
        memory_ranked=memory_ranked,
        memory_active=memory_active,
        governor_config=cfg,
    )
    selected_score = float(selected.get("total_score_with_memory_and_exploration", selected.get("score", float("-inf"))))
    for row in candidate_rows:
        row["score_margin_to_selected"] = (
            selected_score - float(row.get("total_score_with_memory_and_exploration", row.get("score", float("-inf"))))
            if bool(row.get("viable", False))
            else float("inf")
        )
    return selected, candidate_rows


def _candidate_memory_features(
    candidate_belief_features: CandidateBeliefFeaturesFn,
    representative: dict[str, object],
    outcome: dict[str, object],
    *,
    memory_query_mode: str,
) -> dict[str, object] | None:
    representative_with_mode = dict(representative)
    representative_with_mode["__memory_query_mode"] = str(memory_query_mode)
    return candidate_belief_features(representative_with_mode, outcome)


def _refresh_memory_shortlist_rows(
    *,
    candidate_rows: list[dict[str, object]],
    candidate_sources: dict[str, tuple[dict[str, object], dict[str, object]]],
    candidate_belief_features: CandidateBeliefFeaturesFn,
    belief_features: dict[str, float] | None,
    context: dict[str, object],
    governor_mode: str,
    policy_id: str,
    prefilter: dict[str, object],
    baseline_selected: dict[str, object],
    governor_config: GovernorConfig,
    include_diagnostics: bool = True,
) -> None:
    shortlist = _memory_shortlist_variant_ids(
        candidate_rows,
        baseline_selected=baseline_selected,
        governor_config=governor_config,
    )
    for row in candidate_rows:
        variant_id = str(row.get("primitive_variant_id", ""))
        selected = variant_id in shortlist
        row["memory_shortlist_selected"] = bool(selected)
        row["memory_shortlist_policy"] = "top_k_or_recoverable_score_gap_before_expensive_flow_queries"
        row["memory_shortlist_top_k"] = int(MEMORY_COST_BENEFIT_SHORTLIST_TOP_K)
        row["memory_shortlist_score_gap"] = float(MEMORY_COST_BENEFIT_SHORTLIST_SCORE_GAP)
        row["memory_shortlist_reason"] = "selected_for_full_memory_query" if selected else "skipped_full_memory_query"
    refreshed: dict[str, dict[str, object]] = {}
    for variant_id in shortlist:
        source = candidate_sources.get(variant_id)
        if source is None:
            continue
        representative, outcome = source
        features = _candidate_memory_features(
            candidate_belief_features,
            representative,
            outcome,
            memory_query_mode="full",
        ) or belief_features
        refreshed_row = governor_candidate_row(
            representative=representative,
            outcome=outcome,
            context=context,
            governor_mode=governor_mode,
            policy_id=policy_id,
            belief_features=features,
            governor_config=governor_config,
            include_diagnostics=include_diagnostics,
        )
        refreshed_row.update(_prefilter_audit_fields(prefilter))
        refreshed_row["memory_shortlist_selected"] = True
        refreshed_row["memory_shortlist_policy"] = "top_k_or_recoverable_score_gap_before_expensive_flow_queries"
        refreshed_row["memory_shortlist_top_k"] = int(MEMORY_COST_BENEFIT_SHORTLIST_TOP_K)
        refreshed_row["memory_shortlist_score_gap"] = float(MEMORY_COST_BENEFIT_SHORTLIST_SCORE_GAP)
        refreshed_row["memory_shortlist_reason"] = "selected_for_full_memory_query"
        refreshed[variant_id] = refreshed_row
    for index, row in enumerate(candidate_rows):
        variant_id = str(row.get("primitive_variant_id", ""))
        if variant_id in refreshed:
            candidate_rows[index] = refreshed[variant_id]


def _memory_shortlist_variant_ids(
    candidate_rows: list[dict[str, object]],
    *,
    baseline_selected: dict[str, object],
    governor_config: GovernorConfig,
) -> set[str]:
    viable = [row for row in candidate_rows if bool(row.get("viable", False))]
    ranked = sorted(
        viable,
        key=lambda row: (
            -float(row.get("base_score_without_memory", float("-inf"))),
            str(row.get("primitive_id", "")),
            str(row.get("primitive_variant_id", "")),
        ),
    )
    baseline_score = _score(baseline_selected, "base_score_without_memory")
    allowed_gap = max(
        float(MEMORY_COST_BENEFIT_SHORTLIST_SCORE_GAP),
        float(governor_config.memory_cost_benefit_score_cap),
    )
    selected = {
        str(row.get("primitive_variant_id", ""))
        for row in ranked[: int(MEMORY_COST_BENEFIT_SHORTLIST_TOP_K)]
    }
    selected.add(str(baseline_selected.get("primitive_variant_id", "")))
    for row in ranked:
        if float(baseline_score) - _score(row, "base_score_without_memory") <= allowed_gap:
            selected.add(str(row.get("primitive_variant_id", "")))
    return {variant_id for variant_id in selected if variant_id}


def _baseline_shielded_memory_selection(
    *,
    viable_rows: list[dict[str, object]],
    baseline_selected: dict[str, object],
    memory_ranked: list[dict[str, object]],
    memory_active: bool,
    governor_config: GovernorConfig | None = None,
) -> dict[str, object]:
    cfg = governor_config or DEFAULT_GOVERNOR_CONFIG
    baseline_variant_id = str(baseline_selected.get("primitive_variant_id", ""))
    memory_selected = memory_ranked[0] if memory_ranked else baseline_selected
    memory_variant_id = str(memory_selected.get("primitive_variant_id", ""))
    if not memory_active:
        _mark_memory_shield_rows(
            viable_rows,
            baseline_selected=baseline_selected,
            memory_selected=memory_selected,
            accepted=True,
            status="not_active_no_candidate_path_memory",
            governor_config=cfg,
        )
        return memory_selected
    first_rejected: tuple[dict[str, object], str] | None = None
    for candidate in memory_ranked:
        memory_variant_id = str(candidate.get("primitive_variant_id", ""))
        if memory_variant_id == baseline_variant_id:
            if first_rejected is not None:
                rejected_candidate, rejected_status = first_rejected
                _mark_memory_shield_rows(
                    viable_rows,
                    baseline_selected=baseline_selected,
                    memory_selected=rejected_candidate,
                    accepted=False,
                    status=rejected_status,
                    governor_config=cfg,
                )
                return baseline_selected
            _mark_memory_shield_rows(
                viable_rows,
                baseline_selected=baseline_selected,
                memory_selected=baseline_selected,
                accepted=True,
                status="baseline_and_memory_winner_match",
                governor_config=cfg,
            )
            return baseline_selected
        accepted, status = _memory_switch_acceptance_status(
            baseline_selected=baseline_selected,
            memory_selected=candidate,
            governor_config=cfg,
        )
        if accepted:
            _mark_memory_shield_rows(
                viable_rows,
                baseline_selected=baseline_selected,
                memory_selected=candidate,
                accepted=True,
                status=status,
                governor_config=cfg,
            )
            return candidate
        if first_rejected is None:
            first_rejected = (candidate, status)
    memory_selected, status = first_rejected or (baseline_selected, "baseline_and_memory_winner_match")
    _mark_memory_shield_rows(
        viable_rows,
        baseline_selected=baseline_selected,
        memory_selected=memory_selected,
        accepted=False,
        status=status,
        governor_config=cfg,
    )
    return baseline_selected


def _apply_bounded_memory_objective(
    *,
    viable_rows: list[dict[str, object]],
    baseline_selected: dict[str, object],
    memory_active: bool,
    governor_config: GovernorConfig | None = None,
) -> None:
    """Apply one bounded cost-benefit memory value after viability filtering."""

    cfg = governor_config or DEFAULT_GOVERNOR_CONFIG
    baseline_score = _score(baseline_selected, "base_score_without_memory")
    for row in viable_rows:
        base_score = _score(row, "base_score_without_memory")
        raw_memory_component = _float(row.get("memory_score_component", 0.0))
        base_gap = max(0.0, float(baseline_score) - float(base_score))
        cost_benefit_component, cost_benefit = _memory_cost_benefit_component(
            row,
            baseline_selected=baseline_selected,
            base_gap=base_gap,
            memory_active=memory_active,
            governor_config=cfg,
        )
        effective_memory_component = float(cost_benefit_component)
        exploration_component = 0.0
        adjusted_score_with_memory = float(base_score) + float(effective_memory_component)
        adjusted_total_score = float(adjusted_score_with_memory) + float(exploration_component)
        row["raw_memory_score_component"] = float(raw_memory_component)
        row["raw_residual_memory_score_component"] = float(cost_benefit["known_flow_benefit_m"])
        row["memory_score_component"] = float(effective_memory_component)
        row["memory_residual_score_component"] = float(effective_memory_component)
        row["memory_near_tie_policy_version"] = BASELINE_SHIELDED_MEMORY_POLICY_VERSION
        row["memory_near_tie_active"] = bool(memory_active)
        row["memory_near_tie_base_score_margin"] = 0.0
        row["memory_near_tie_base_score_gap_to_best"] = float(base_gap)
        row["memory_near_tie_factor"] = 1.0 if memory_active else 0.0
        row["memory_near_tie_raw_memory_component"] = float(raw_memory_component)
        row["memory_near_tie_raw_residual_component"] = float(cost_benefit["known_flow_benefit_m"])
        row["memory_near_tie_effective_memory_component"] = float(effective_memory_component)
        row["memory_objective_policy"] = "single_cost_benefit_spatial_flow_memory_after_viability_filter"
        row["memory_objective_residual_confidence_gate"] = float(cost_benefit["confidence"])
        row["memory_objective_score_cap"] = float(cfg.memory_cost_benefit_score_cap)
        row["memory_objective_min_confidence"] = 0.0
        row["memory_objective_max_base_score_drop"] = 0.0
        row["memory_flow_region_attraction_score_component"] = 0.0
        row["memory_flow_region_attraction_gate"] = 0.0
        row["memory_flow_region_attraction_policy"] = "deprecated_folded_into_cost_benefit_memory_value"
        row["memory_flow_region_attraction_max_base_score_drop"] = 0.0
        row["memory_flow_region_attraction_min_confidence"] = 0.0
        row["memory_flow_region_attraction_score_cap"] = 0.0
        row["memory_flow_region_attraction_min_front_progress_ratio"] = 0.0
        row["memory_information_gain_score_component"] = float(cost_benefit["information_benefit"])
        row["memory_information_gain_gate"] = 1.0 if float(cost_benefit["information_benefit"]) > 0.0 else 0.0
        row["memory_information_gain_policy"] = "deprecated_folded_into_cost_benefit_memory_value"
        row["memory_information_gain_weight"] = float(cfg.memory_cost_benefit_information_gain_weight)
        row["memory_information_gain_score_cap"] = float(cfg.memory_cost_benefit_score_cap)
        row["memory_information_gain_min_uncertainty"] = 0.0
        row["memory_information_gain_max_base_score_drop"] = 0.0
        row["memory_information_gain_min_front_progress_ratio"] = 0.0
        row["memory_route_score_component"] = float(effective_memory_component)
        row["memory_route_gate"] = 1.0 if float(effective_memory_component) != 0.0 else 0.0
        row["memory_route_policy"] = "cost_benefit_known_flow_benefit_minus_mission_detour_risk_cost"
        row["memory_route_planning_weight"] = float(cfg.memory_route_planning_weight)
        row["memory_route_information_gain_weight"] = float(cfg.memory_cost_benefit_information_gain_weight)
        row["memory_route_score_cap"] = float(cfg.memory_cost_benefit_score_cap)
        row["memory_route_min_confidence"] = 0.0
        row["memory_route_max_base_score_drop"] = 0.0
        row["memory_route_min_front_progress_ratio"] = 0.0
        row["memory_route_horizon_primitives"] = int(round(float(cfg.memory_route_horizon_primitives)))
        row["memory_route_discount"] = float(cfg.memory_route_discount)
        row["memory_cost_benefit_known_flow_benefit_m"] = float(cost_benefit["known_flow_benefit_m"])
        row["memory_cost_benefit_information_benefit"] = float(cost_benefit["information_benefit"])
        row["memory_cost_benefit_total_benefit"] = float(cost_benefit["total_benefit"])
        row["memory_cost_benefit_base_score_cost"] = float(cost_benefit["base_score_cost"])
        row["memory_cost_benefit_front_progress_cost"] = float(cost_benefit["front_progress_cost"])
        row["memory_cost_benefit_risk_cost"] = float(cost_benefit["risk_cost"])
        row["memory_cost_benefit_margin_cost"] = float(cost_benefit["margin_cost"])
        row["memory_cost_benefit_total_cost"] = float(cost_benefit["total_cost"])
        row["memory_cost_benefit_net_value"] = float(cost_benefit["net_value"])
        row["memory_cost_benefit_confidence"] = float(cost_benefit["confidence"])
        row["memory_cost_benefit_weight"] = float(cfg.memory_cost_benefit_weight)
        row["memory_cost_benefit_score_cap"] = float(cfg.memory_cost_benefit_score_cap)
        row["score_with_memory"] = float(adjusted_score_with_memory)
        row["total_score_with_memory_and_exploration"] = float(adjusted_total_score)
        row["score"] = float(adjusted_total_score)


def _memory_cost_benefit_component(
    row: dict[str, object],
    *,
    baseline_selected: dict[str, object],
    base_gap: float,
    memory_active: bool,
    governor_config: GovernorConfig,
) -> tuple[float, dict[str, float]]:
    """Return the simple memory value: useful remembered flow minus cost to reach it."""

    empty = {
        "known_flow_benefit_m": 0.0,
        "information_benefit": 0.0,
        "total_benefit": 0.0,
        "base_score_cost": 0.0,
        "front_progress_cost": 0.0,
        "risk_cost": 0.0,
        "margin_cost": 0.0,
        "total_cost": 0.0,
        "net_value": 0.0,
        "confidence": 0.0,
    }
    if not memory_active:
        return 0.0, empty
    if not bool(row.get("memory_shortlist_selected", False)):
        return 0.0, empty

    path_utility = _float(
        row.get(
            "belief_candidate_path_memory_utility_without_attraction_m",
            row.get("belief_local_specific_energy_residual_m", 0.0),
        )
    )
    path_confidence = _clip(_float(row.get("belief_candidate_path_confidence", 0.0)), 0.0, 1.0)
    reachable_utility = max(0.0, _float(row.get("belief_flow_map_reachable_attraction_m", 0.0)))
    reachable_confidence = _clip(
        _float(row.get("belief_flow_map_reachable_attraction_confidence", 0.0)),
        0.0,
        1.0,
    )
    route_utility = _float(row.get("belief_flow_map_route_exploitation_m", 0.0))
    route_confidence = _clip(_float(row.get("belief_flow_map_route_confidence", 0.0)), 0.0, 1.0)
    route_safe_fraction = _clip(_float(row.get("belief_flow_map_route_safe_fraction", 1.0), default=1.0), 0.0, 1.0)
    route_information_gain = max(
        0.0,
        _float(
            row.get(
                "belief_flow_map_route_information_gain",
                row.get("belief_flow_map_information_gain", 0.0),
            )
        ),
    )
    map_information_gain = max(0.0, _float(row.get("belief_flow_map_information_gain", 0.0)))

    known_flow_benefit = (
        float(path_utility) * max(0.25, float(path_confidence))
        + float(reachable_utility) * max(0.25, float(reachable_confidence))
        + float(route_utility) * max(0.25, float(route_confidence)) * float(route_safe_fraction)
    )
    information_benefit = float(governor_config.memory_cost_benefit_information_gain_weight) * max(
        float(route_information_gain),
        float(map_information_gain),
    )
    total_benefit = float(known_flow_benefit) + float(information_benefit)

    baseline_progress = _float(baseline_selected.get("mission_front_wall_progress_fraction", 0.0))
    candidate_progress = _float(row.get("mission_front_wall_progress_fraction", 0.0))
    route_progress = _float(row.get("belief_flow_map_route_front_progress", candidate_progress))
    effective_progress = max(float(candidate_progress), float(route_progress))
    front_progress_drop = max(0.0, float(baseline_progress) - float(effective_progress))

    transition_drop = max(
        0.0,
        _float(baseline_selected.get("transition_success_probability", 0.0))
        - _float(row.get("transition_success_probability", 0.0)),
    )
    hard_failure_increase = max(
        0.0,
        _float(row.get("hard_failure_risk", 1.0)) - _float(baseline_selected.get("hard_failure_risk", 1.0)),
    )
    margin_drop = max(0.0, _candidate_path_exit_margin(baseline_selected) - _candidate_path_exit_margin(row))

    base_score_cost = max(0.0, float(base_gap))
    front_progress_cost = float(governor_config.memory_cost_benefit_progress_cost_weight) * float(front_progress_drop)
    risk_cost = float(governor_config.memory_cost_benefit_risk_cost_weight) * (
        float(transition_drop) + float(hard_failure_increase)
    )
    margin_cost = float(governor_config.memory_cost_benefit_margin_cost_weight) * float(margin_drop)
    total_cost = float(base_score_cost) + float(front_progress_cost) + float(risk_cost) + float(margin_cost)
    net_value = float(total_benefit) - float(total_cost)
    score = _clip(
        float(governor_config.memory_cost_benefit_weight) * float(net_value),
        -float(governor_config.memory_cost_benefit_score_cap),
        float(governor_config.memory_cost_benefit_score_cap),
    )
    details = {
        "known_flow_benefit_m": float(known_flow_benefit),
        "information_benefit": float(information_benefit),
        "total_benefit": float(total_benefit),
        "base_score_cost": float(base_score_cost),
        "front_progress_cost": float(front_progress_cost),
        "risk_cost": float(risk_cost),
        "margin_cost": float(margin_cost),
        "total_cost": float(total_cost),
        "net_value": float(net_value),
        "confidence": float(max(path_confidence, reachable_confidence, route_confidence)),
    }
    return float(score), details


def _memory_switch_acceptance_status(
    *,
    baseline_selected: dict[str, object],
    memory_selected: dict[str, object],
    governor_config: GovernorConfig | None = None,
) -> tuple[bool, str]:
    cfg = governor_config or DEFAULT_GOVERNOR_CONFIG
    memory_component = _float(memory_selected.get("memory_score_component", 0.0))
    baseline_memory_component = _float(baseline_selected.get("memory_score_component", 0.0))
    memory_correction_delta = float(memory_component) - float(baseline_memory_component)
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
    calibrated_regime_risk_delta = _float(memory_selected.get("calibrated_regime_mismatch_risk", 0.0)) - _float(
        baseline_selected.get("calibrated_regime_mismatch_risk", 0.0)
    )
    path_margin_non_regressive = _candidate_path_margin_non_regressive(
        baseline_selected=baseline_selected,
        memory_selected=memory_selected,
        governor_config=cfg,
    )
    if (
        memory_correction_delta > 0.0
        and score_margin >= float(cfg.memory_switch_min_score_margin)
        and transition_delta >= -float(cfg.memory_switch_max_transition_success_drop)
        and hard_failure_delta <= float(cfg.memory_switch_max_hard_failure_risk_increase)
        and calibrated_regime_risk_delta <= float(cfg.memory_switch_max_calibrated_regime_risk_increase) + 1e-12
        and path_margin_non_regressive
    ):
        return True, "accepted_cost_benefit_spatial_flow_memory_switch"
    if memory_correction_delta <= 0.0:
        return False, "rejected_memory_cost_benefit_not_better_than_baseline"
    if score_margin < float(cfg.memory_switch_min_score_margin):
        return False, "rejected_adaptive_score_margin_too_small"
    if transition_delta < -float(cfg.memory_switch_max_transition_success_drop):
        return False, "rejected_transition_success_regression"
    if hard_failure_delta > float(cfg.memory_switch_max_hard_failure_risk_increase):
        return False, "rejected_hard_failure_risk_regression"
    if calibrated_regime_risk_delta > float(cfg.memory_switch_max_calibrated_regime_risk_increase) + 1e-12:
        return False, "rejected_calibrated_regime_mismatch_risk_regression"
    if not path_margin_non_regressive:
        return False, "rejected_candidate_path_exit_margin_regression"
    return False, "rejected_adaptive_switch_guard_not_satisfied"


def _mark_memory_shield_rows(
    rows: list[dict[str, object]],
    *,
    baseline_selected: dict[str, object],
    memory_selected: dict[str, object],
    accepted: bool,
    status: str,
    governor_config: GovernorConfig | None = None,
) -> None:
    cfg = governor_config or DEFAULT_GOVERNOR_CONFIG
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
    baseline_calibrated_regime_risk = _float(baseline_selected.get("calibrated_regime_mismatch_risk", 0.0))
    memory_calibrated_regime_risk = _float(memory_selected.get("calibrated_regime_mismatch_risk", 0.0))
    calibrated_regime_risk_delta = memory_calibrated_regime_risk - baseline_calibrated_regime_risk
    baseline_path_margin = _candidate_path_exit_margin(baseline_selected)
    memory_path_margin = _candidate_path_exit_margin(memory_selected)
    path_margin_delta = memory_path_margin - baseline_path_margin
    flow_region_component = _float(memory_selected.get("memory_flow_region_attraction_score_component", 0.0))
    flow_region_confidence = _float(memory_selected.get("belief_flow_map_reachable_attraction_confidence", 0.0))
    information_gain_component = _float(memory_selected.get("memory_information_gain_score_component", 0.0))
    information_gain = _float(memory_selected.get("belief_flow_map_information_gain", 0.0))
    route_component = _float(memory_selected.get("memory_route_score_component", 0.0))
    route_confidence = _float(memory_selected.get("belief_flow_map_route_confidence", 0.0))
    route_information_gain = _float(memory_selected.get("belief_flow_map_route_information_gain", 0.0))
    route_exploitation = _float(memory_selected.get("belief_flow_map_route_exploitation_m", 0.0))
    confidence = max(
        _float(memory_selected.get("belief_candidate_path_confidence", 0.0)),
        flow_region_confidence if flow_region_component > 0.0 else 0.0,
        route_confidence if route_component > 0.0 else 0.0,
    )
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
        row["memory_shield_baseline_calibrated_regime_mismatch_risk"] = float(baseline_calibrated_regime_risk)
        row["memory_shield_memory_calibrated_regime_mismatch_risk"] = float(memory_calibrated_regime_risk)
        row["memory_shield_calibrated_regime_mismatch_risk_delta"] = float(calibrated_regime_risk_delta)
        row["memory_shield_memory_max_calibrated_regime_risk_increase"] = float(
            cfg.memory_switch_max_calibrated_regime_risk_increase
        )
        row["memory_shield_baseline_path_exit_margin_m"] = float(baseline_path_margin)
        row["memory_shield_memory_path_exit_margin_m"] = float(memory_path_margin)
        row["memory_shield_path_exit_margin_delta_m"] = float(path_margin_delta)
        row["memory_shield_max_path_exit_margin_drop_m"] = float(cfg.adaptive_switch_max_path_exit_margin_drop_m)
        row["memory_shield_min_path_exit_margin_m"] = float(cfg.adaptive_switch_min_path_exit_margin_m)
        row["memory_shield_candidate_path_confidence"] = float(confidence)
        row["memory_shield_candidate_path_uncertainty"] = float(uncertainty)
        row["memory_shield_exploration_score_component"] = float(exploration_component)
        row["memory_shield_baseline_memory_score_component"] = float(baseline_memory_component)
        row["memory_shield_memory_candidate_memory_score_component"] = float(memory_candidate_memory_component)
        row["memory_shield_flow_region_attraction_score_component"] = float(flow_region_component)
        row["memory_shield_flow_region_attraction_confidence"] = float(flow_region_confidence)
        row["memory_shield_information_gain_score_component"] = float(information_gain_component)
        row["memory_shield_information_gain"] = float(information_gain)
        row["memory_shield_route_score_component"] = float(route_component)
        row["memory_shield_route_exploitation_m"] = float(route_exploitation)
        row["memory_shield_route_information_gain"] = float(route_information_gain)
        row["memory_shield_route_confidence"] = float(route_confidence)
        row["memory_shield_memory_objective_score_cap"] = float(cfg.memory_objective_score_cap)
        row["memory_shield_memory_objective_min_confidence"] = float(cfg.memory_objective_min_confidence)
        row["memory_shield_memory_objective_max_base_score_drop"] = float(cfg.memory_objective_max_base_score_drop)
        row["memory_shield_cost_benefit_known_flow_benefit_m"] = float(
            memory_selected.get("memory_cost_benefit_known_flow_benefit_m", 0.0)
        )
        row["memory_shield_cost_benefit_information_benefit"] = float(
            memory_selected.get("memory_cost_benefit_information_benefit", 0.0)
        )
        row["memory_shield_cost_benefit_total_benefit"] = float(
            memory_selected.get("memory_cost_benefit_total_benefit", 0.0)
        )
        row["memory_shield_cost_benefit_total_cost"] = float(
            memory_selected.get("memory_cost_benefit_total_cost", 0.0)
        )
        row["memory_shield_cost_benefit_net_value"] = float(
            memory_selected.get("memory_cost_benefit_net_value", 0.0)
        )
        row["memory_shield_cost_benefit_score_cap"] = float(cfg.memory_cost_benefit_score_cap)
        row["memory_shield_flow_region_max_base_score_drop"] = float(cfg.flow_region_attraction_max_base_score_drop)
        row["memory_shield_route_max_base_score_drop"] = float(cfg.memory_route_max_base_score_drop)
        row["memory_shield_route_min_confidence"] = float(cfg.memory_route_min_confidence)
        row["memory_shield_route_score_cap"] = float(cfg.memory_route_score_cap)
        row["memory_shield_exploration_cross_family_allowed"] = False
        row["memory_shield_information_gain_cross_family_allowed"] = False
        row["memory_shield_information_gain_min_uncertainty"] = float(cfg.memory_information_gain_min_uncertainty)
        row["memory_shield_information_gain_max_base_score_drop"] = float(
            cfg.memory_information_gain_max_base_score_drop
        )
        row["memory_shield_min_confidence"] = float(cfg.memory_switch_min_confidence)
        row["memory_shield_min_score_margin"] = float(cfg.memory_switch_min_score_margin)
        row["memory_shield_memory_max_base_score_drop"] = float(cfg.memory_switch_max_base_score_drop)
        row["memory_shield_memory_max_transition_success_drop"] = float(cfg.memory_switch_max_transition_success_drop)
        row["memory_shield_memory_max_hard_failure_risk_increase"] = float(
            cfg.memory_switch_max_hard_failure_risk_increase
        )
        row["memory_shield_exploration_min_uncertainty"] = float(cfg.exploration_switch_min_uncertainty)
        row["memory_shield_exploration_min_score_margin"] = float(cfg.exploration_switch_min_score_margin)
        row["memory_shield_exploration_max_base_score_drop"] = float(cfg.exploration_switch_max_base_score_drop)
        row["memory_shield_exploration_max_transition_success_drop"] = float(
            cfg.exploration_switch_max_transition_success_drop
        )
        row["memory_shield_exploration_max_hard_failure_risk_increase"] = float(
            cfg.exploration_switch_max_hard_failure_risk_increase
        )


def _candidate_path_margin_non_regressive(
    *,
    baseline_selected: dict[str, object],
    memory_selected: dict[str, object],
    governor_config: GovernorConfig | None = None,
) -> bool:
    cfg = governor_config or DEFAULT_GOVERNOR_CONFIG
    baseline_margin = _candidate_path_exit_margin(baseline_selected)
    memory_margin = _candidate_path_exit_margin(memory_selected)
    if baseline_margin >= float(cfg.adaptive_switch_min_path_exit_margin_m) and memory_margin < float(
        cfg.adaptive_switch_min_path_exit_margin_m
    ):
        return False
    return bool(memory_margin + float(cfg.adaptive_switch_max_path_exit_margin_drop_m) >= baseline_margin)


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
        "governor_calibrated_regime_mismatch_risk_weight": float(cfg.calibrated_regime_mismatch_risk_weight),
        "governor_calibrated_regime_mismatch_score_cap": float(cfg.calibrated_regime_mismatch_score_cap),
        "governor_memory_switch_max_calibrated_regime_risk_increase": float(
            cfg.memory_switch_max_calibrated_regime_risk_increase
        ),
        "governor_memory_switch_min_confidence": float(cfg.memory_switch_min_confidence),
        "governor_memory_switch_min_score_margin": float(cfg.memory_switch_min_score_margin),
        "governor_exploration_switch_min_uncertainty": float(cfg.exploration_switch_min_uncertainty),
        "governor_adaptive_switch_max_path_exit_margin_drop_m": float(
            cfg.adaptive_switch_max_path_exit_margin_drop_m
        ),
        "governor_candidate_path_memory_residual_cap_m": float(cfg.candidate_path_memory_residual_cap_m),
        "governor_candidate_path_memory_specific_energy_residual_cap_m": float(
            cfg.candidate_path_memory_specific_energy_residual_cap_m
        ),
        "governor_residual_memory_launch_recency_half_life": float(cfg.residual_memory_launch_recency_half_life),
        "governor_memory_objective_score_cap": float(cfg.memory_objective_score_cap),
        "governor_memory_objective_min_confidence": float(cfg.memory_objective_min_confidence),
        "governor_memory_objective_max_base_score_drop": float(cfg.memory_objective_max_base_score_drop),
        "governor_flow_region_attraction_weight": float(cfg.flow_region_attraction_weight),
        "governor_flow_region_attraction_score_cap": float(cfg.flow_region_attraction_score_cap),
        "governor_flow_region_attraction_min_confidence": float(cfg.flow_region_attraction_min_confidence),
        "governor_flow_region_attraction_max_base_score_drop": float(
            cfg.flow_region_attraction_max_base_score_drop
        ),
        "governor_flow_region_attraction_min_front_progress_ratio": float(
            cfg.flow_region_attraction_min_front_progress_ratio
        ),
        "governor_memory_information_gain_weight": float(cfg.memory_information_gain_weight),
        "governor_memory_information_gain_score_cap": float(cfg.memory_information_gain_score_cap),
        "governor_memory_information_gain_min_uncertainty": float(
            cfg.memory_information_gain_min_uncertainty
        ),
        "governor_memory_information_gain_max_base_score_drop": float(
            cfg.memory_information_gain_max_base_score_drop
        ),
        "governor_memory_information_gain_min_front_progress_ratio": float(
            cfg.memory_information_gain_min_front_progress_ratio
        ),
        "governor_memory_information_gain_allow_cross_family": bool(
            cfg.memory_information_gain_allow_cross_family
        ),
        "governor_memory_route_planning_weight": float(cfg.memory_route_planning_weight),
        "governor_memory_route_information_gain_weight": float(cfg.memory_route_information_gain_weight),
        "governor_memory_route_score_cap": float(cfg.memory_route_score_cap),
        "governor_memory_route_min_confidence": float(cfg.memory_route_min_confidence),
        "governor_memory_route_max_base_score_drop": float(cfg.memory_route_max_base_score_drop),
        "governor_memory_route_min_front_progress_ratio": float(cfg.memory_route_min_front_progress_ratio),
        "governor_memory_route_horizon_primitives": int(round(float(cfg.memory_route_horizon_primitives))),
        "governor_memory_route_discount": float(cfg.memory_route_discount),
        "governor_memory_cost_benefit_weight": float(cfg.memory_cost_benefit_weight),
        "governor_memory_cost_benefit_score_cap": float(cfg.memory_cost_benefit_score_cap),
        "governor_memory_cost_benefit_information_gain_weight": float(
            cfg.memory_cost_benefit_information_gain_weight
        ),
        "governor_memory_cost_benefit_progress_cost_weight": float(cfg.memory_cost_benefit_progress_cost_weight),
        "governor_memory_cost_benefit_risk_cost_weight": float(cfg.memory_cost_benefit_risk_cost_weight),
        "governor_memory_cost_benefit_margin_cost_weight": float(cfg.memory_cost_benefit_margin_cost_weight),
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
        "selected_calibrated_regime_alpha_abs_deg": (
            0.0 if selected is None else float(selected.get("calibrated_regime_alpha_abs_deg", 0.0))
        ),
        "selected_calibrated_regime_label": (
            "" if selected is None else str(selected.get("calibrated_regime_label", ""))
        ),
        "selected_calibrated_transition_activation": (
            0.0 if selected is None else float(selected.get("calibrated_transition_activation", 0.0))
        ),
        "selected_calibrated_post_stall_activation": (
            0.0 if selected is None else float(selected.get("calibrated_post_stall_activation", 0.0))
        ),
        "selected_calibrated_regime_mismatch_risk": (
            0.0 if selected is None else float(selected.get("calibrated_regime_mismatch_risk", 0.0))
        ),
        "selected_calibrated_regime_mismatch_score_component": (
            0.0 if selected is None else float(selected.get("calibrated_regime_mismatch_score_component", 0.0))
        ),
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
        "selected_memory_near_tie_base_score_margin": (
            0.0 if selected is None else float(selected.get("memory_near_tie_base_score_margin", 0.0))
        ),
        "selected_memory_near_tie_base_score_gap_to_best": (
            0.0 if selected is None else float(selected.get("memory_near_tie_base_score_gap_to_best", 0.0))
        ),
        "selected_memory_near_tie_factor": (
            0.0 if selected is None else float(selected.get("memory_near_tie_factor", 0.0))
        ),
        "selected_memory_objective_residual_confidence_gate": (
            0.0 if selected is None else float(selected.get("memory_objective_residual_confidence_gate", 0.0))
        ),
        "selected_memory_objective_score_cap": (
            0.0 if selected is None else float(selected.get("memory_objective_score_cap", 0.0))
        ),
        "selected_memory_objective_max_base_score_drop": (
            0.0 if selected is None else float(selected.get("memory_objective_max_base_score_drop", 0.0))
        ),
        "selected_raw_memory_score_component": (
            0.0 if selected is None else float(selected.get("raw_memory_score_component", 0.0))
        ),
        "selected_effective_memory_score_component": (
            0.0 if selected is None else float(selected.get("memory_score_component", 0.0))
        ),
        "selected_memory_flow_region_attraction_score_component": (
            0.0 if selected is None else float(selected.get("memory_flow_region_attraction_score_component", 0.0))
        ),
        "selected_memory_flow_region_attraction_gate": (
            0.0 if selected is None else float(selected.get("memory_flow_region_attraction_gate", 0.0))
        ),
        "selected_memory_information_gain_score_component": (
            0.0 if selected is None else float(selected.get("memory_information_gain_score_component", 0.0))
        ),
        "selected_memory_information_gain_gate": (
            0.0 if selected is None else float(selected.get("memory_information_gain_gate", 0.0))
        ),
        "selected_memory_route_score_component": (
            0.0 if selected is None else float(selected.get("memory_route_score_component", 0.0))
        ),
        "selected_memory_route_gate": (
            0.0 if selected is None else float(selected.get("memory_route_gate", 0.0))
        ),
        "selected_memory_route_horizon_primitives": (
            0 if selected is None else int(float(selected.get("memory_route_horizon_primitives", 0)))
        ),
        "selected_memory_cost_benefit_known_flow_benefit_m": (
            0.0 if selected is None else float(selected.get("memory_cost_benefit_known_flow_benefit_m", 0.0))
        ),
        "selected_memory_cost_benefit_information_benefit": (
            0.0 if selected is None else float(selected.get("memory_cost_benefit_information_benefit", 0.0))
        ),
        "selected_memory_cost_benefit_total_benefit": (
            0.0 if selected is None else float(selected.get("memory_cost_benefit_total_benefit", 0.0))
        ),
        "selected_memory_cost_benefit_total_cost": (
            0.0 if selected is None else float(selected.get("memory_cost_benefit_total_cost", 0.0))
        ),
        "selected_memory_cost_benefit_net_value": (
            0.0 if selected is None else float(selected.get("memory_cost_benefit_net_value", 0.0))
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
        "selected_memory_shield_baseline_calibrated_regime_mismatch_risk": (
            0.0
            if selected is None
            else float(selected.get("memory_shield_baseline_calibrated_regime_mismatch_risk", 0.0))
        ),
        "selected_memory_shield_memory_calibrated_regime_mismatch_risk": (
            0.0
            if selected is None
            else float(selected.get("memory_shield_memory_calibrated_regime_mismatch_risk", 0.0))
        ),
        "selected_memory_shield_calibrated_regime_mismatch_risk_delta": (
            0.0
            if selected is None
            else float(selected.get("memory_shield_calibrated_regime_mismatch_risk_delta", 0.0))
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
        "selected_memory_shield_flow_region_attraction_score_component": (
            0.0
            if selected is None
            else float(selected.get("memory_shield_flow_region_attraction_score_component", 0.0))
        ),
        "selected_memory_shield_flow_region_attraction_confidence": (
            0.0 if selected is None else float(selected.get("memory_shield_flow_region_attraction_confidence", 0.0))
        ),
        "selected_memory_shield_information_gain_score_component": (
            0.0
            if selected is None
            else float(selected.get("memory_shield_information_gain_score_component", 0.0))
        ),
        "selected_memory_shield_information_gain": (
            0.0 if selected is None else float(selected.get("memory_shield_information_gain", 0.0))
        ),
        "selected_memory_shield_route_score_component": (
            0.0 if selected is None else float(selected.get("memory_shield_route_score_component", 0.0))
        ),
        "selected_memory_shield_route_exploitation_m": (
            0.0 if selected is None else float(selected.get("memory_shield_route_exploitation_m", 0.0))
        ),
        "selected_memory_shield_route_information_gain": (
            0.0 if selected is None else float(selected.get("memory_shield_route_information_gain", 0.0))
        ),
        "selected_memory_shield_route_confidence": (
            0.0 if selected is None else float(selected.get("memory_shield_route_confidence", 0.0))
        ),
        "selected_flow_map_grid_resolution_m": (
            0.0 if selected is None else float(selected.get("belief_flow_map_grid_resolution_m", 0.0))
        ),
        "selected_flow_map_query_radius_m": (
            0.0 if selected is None else float(selected.get("belief_flow_map_query_radius_m", 0.0))
        ),
        "selected_flow_map_controller_query_mode": (
            "" if selected is None else str(selected.get("belief_flow_map_controller_query_mode", ""))
        ),
        "selected_flow_map_controller_path_probe_count": (
            0 if selected is None else int(float(selected.get("belief_flow_map_controller_path_probe_count", 0)))
        ),
        "selected_flow_map_controller_reachable_probe_count": (
            0 if selected is None else int(float(selected.get("belief_flow_map_controller_reachable_probe_count", 0)))
        ),
        "selected_flow_map_controller_route_probe_count": (
            0 if selected is None else int(float(selected.get("belief_flow_map_controller_route_probe_count", 0)))
        ),
        "selected_flow_map_reachable_attraction_m": (
            0.0 if selected is None else float(selected.get("belief_flow_map_reachable_attraction_m", 0.0))
        ),
        "selected_flow_map_reachable_attraction_confidence": (
            0.0 if selected is None else float(selected.get("belief_flow_map_reachable_attraction_confidence", 0.0))
        ),
        "selected_flow_map_reachable_attraction_query_count": (
            0 if selected is None else int(float(selected.get("belief_flow_map_reachable_attraction_query_count", 0)))
        ),
        "selected_flow_map_reachable_attraction_observation_count": (
            0 if selected is None else int(float(selected.get("belief_flow_map_reachable_attraction_observation_count", 0)))
        ),
        "selected_flow_map_reachable_attraction_geometry": (
            "" if selected is None else str(selected.get("belief_flow_map_reachable_attraction_geometry", ""))
        ),
        "selected_flow_map_candidate_path_uncertainty": (
            0.0 if selected is None else float(selected.get("belief_flow_map_candidate_path_uncertainty", 0.0))
        ),
        "selected_flow_map_memory_guided_exploration_uncertainty": (
            0.0
            if selected is None
            else float(selected.get("belief_flow_map_memory_guided_exploration_uncertainty", 0.0))
        ),
        "selected_flow_map_information_gain": (
            0.0 if selected is None else float(selected.get("belief_flow_map_information_gain", 0.0))
        ),
        "selected_flow_map_information_gain_path_uncertainty": (
            0.0
            if selected is None
            else float(selected.get("belief_flow_map_information_gain_path_uncertainty", 0.0))
        ),
        "selected_flow_map_information_gain_reachable_uncertainty": (
            0.0
            if selected is None
            else float(selected.get("belief_flow_map_information_gain_reachable_uncertainty", 0.0))
        ),
        "selected_flow_map_information_gain_query_count": (
            0
            if selected is None
            else int(float(selected.get("belief_flow_map_information_gain_query_count", 0)))
        ),
        "selected_flow_map_information_gain_low_confidence_query_count": (
            0
            if selected is None
            else int(float(selected.get("belief_flow_map_information_gain_low_confidence_query_count", 0)))
        ),
        "selected_flow_map_route_policy": (
            "" if selected is None else str(selected.get("belief_flow_map_route_policy", ""))
        ),
        "selected_flow_map_route_exploitation_m": (
            0.0 if selected is None else float(selected.get("belief_flow_map_route_exploitation_m", 0.0))
        ),
        "selected_flow_map_route_information_gain": (
            0.0 if selected is None else float(selected.get("belief_flow_map_route_information_gain", 0.0))
        ),
        "selected_flow_map_route_confidence": (
            0.0 if selected is None else float(selected.get("belief_flow_map_route_confidence", 0.0))
        ),
        "selected_flow_map_route_front_progress": (
            0.0 if selected is None else float(selected.get("belief_flow_map_route_front_progress", 0.0))
        ),
        "selected_flow_map_route_safe_fraction": (
            0.0 if selected is None else float(selected.get("belief_flow_map_route_safe_fraction", 0.0))
        ),
        "selected_memory_shield_exploration_cross_family_allowed": (
            False if selected is None else bool(selected.get("memory_shield_exploration_cross_family_allowed", False))
        ),
        "selected_memory_shield_information_gain_cross_family_allowed": (
            False
            if selected is None
            else bool(selected.get("memory_shield_information_gain_cross_family_allowed", False))
        ),
        "selected_memory_shield_min_confidence": (
            0.0 if selected is None else float(selected.get("memory_shield_min_confidence", 0.0))
        ),
        "selected_memory_shield_min_score_margin": (
            0.0 if selected is None else float(selected.get("memory_shield_min_score_margin", 0.0))
        ),
        "selected_memory_shield_exploration_min_uncertainty": (
            0.0 if selected is None else float(selected.get("memory_shield_exploration_min_uncertainty", 0.0))
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


def _clip(value: float, lower: float, upper: float) -> float:
    return float(min(max(float(value), float(lower)), float(upper)))


def _float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result
