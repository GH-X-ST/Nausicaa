from __future__ import annotations

from viability_governor import GOVERNOR_MODES, governor_candidate_row


def select_compact_representative(
    *,
    representatives: list[dict[str, object]],
    outcome_rows_by_variant_id: dict[str, dict[str, object]],
    context: dict[str, object],
    governor_mode: str,
    policy_id: str = "",
    belief_features: dict[str, float] | None = None,
) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    """Return the highest-scoring viable compact representative and all candidate rows."""

    if governor_mode not in GOVERNOR_MODES:
        raise ValueError("governor_mode must be continuation_mode or terminal_episode_mode.")
    candidate_rows = []
    for representative in representatives:
        outcome = outcome_rows_by_variant_id.get(str(representative.get("primitive_variant_id", "")), {})
        candidate_rows.append(
            governor_candidate_row(
                representative=representative,
                outcome=outcome,
                context=context,
                governor_mode=governor_mode,
                policy_id=policy_id,
                belief_features=belief_features,
            )
        )
    viable = [row for row in candidate_rows if bool(row.get("viable", False))]
    if not viable:
        return None, candidate_rows
    selected = sorted(
        viable,
        key=lambda row: (
            -float(row.get("score", float("-inf"))),
            str(row.get("primitive_id", "")),
            str(row.get("primitive_variant_id", "")),
        ),
    )[0]
    return selected, candidate_rows


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
) -> dict[str, object]:
    """Return a compact selector audit row."""

    return {
        "episode_id": str(episode_id),
        "primitive_step_index": int(primitive_step_index),
        "policy_id": str(policy_id),
        "context_id": str(context.get("context_id", "")),
        "W_layer": str(context.get("W_layer", "")),
        "environment_mode": str(context.get("environment_mode", "")),
        "start_state_family": str(context.get("start_state_family", "")),
        "governor_mode": str(governor_mode),
        "candidate_count": int(candidate_count),
        "viable_count": int(viable_count),
        "decision_status": "selected_compact_representative" if selected else "blocked_no_viable_representative",
        "selected_compact_library_id": "" if selected is None else str(selected.get("compact_library_id", "")),
        "selected_primitive_variant_id": "" if selected is None else str(selected.get("primitive_variant_id", "")),
        "selected_primitive_id": "" if selected is None else str(selected.get("primitive_id", "")),
        "selected_controller_id": "" if selected is None else str(selected.get("controller_id", "")),
        "selected_score": float("-inf") if selected is None else float(selected.get("score", float("-inf"))),
        "claim_status": "simulation_only_selector_decision",
    }
