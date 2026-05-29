from __future__ import annotations

from collections.abc import Callable

from context_conditioned_outcome import context_conditioned_outcome, lookup_outcome_for_identity
from viability_governor import DEFAULT_GOVERNOR_CONFIG, GOVERNOR_MODES, GovernorConfig, governor_candidate_row

CandidateBeliefFeaturesFn = Callable[[dict[str, object], dict[str, object]], dict[str, object] | None]


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
    selected = sorted(
        viable,
        key=lambda row: (
            -float(row.get("total_score_with_memory_and_exploration", row.get("score", float("-inf")))),
            str(row.get("primitive_id", "")),
            str(row.get("primitive_variant_id", "")),
        ),
    )[0]
    selected_score = float(selected.get("total_score_with_memory_and_exploration", selected.get("score", float("-inf"))))
    for row in candidate_rows:
        row["score_margin_to_selected"] = (
            selected_score - float(row.get("total_score_with_memory_and_exploration", row.get("score", float("-inf"))))
            if bool(row.get("viable", False))
            else float("inf")
        )
    return selected, candidate_rows


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
