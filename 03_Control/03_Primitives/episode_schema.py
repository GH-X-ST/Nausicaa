from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Episode schema constants
# 2) Validation helpers
# 3) Claim-boundary checks
# =============================================================================


EPISODE_SUMMARY_COLUMNS = (
    "episode_id",
    "policy_id",
    "fan_branch",
    "W_layer",
    "launch_gate_id",
    "initial_state_vector",
    "initial_state_admission_status",
    "belief_id_before",
    "belief_id_after",
    "primitive_sequence",
    "candidate_query_count",
    "governor_accept_count",
    "governor_reject_count",
    "lift_capture_success",
    "lift_capture_time_s",
    "lift_dwell_time_s",
    "energy_initial_m",
    "energy_final_m",
    "energy_residual_m",
    "minimum_margin_m",
    "minimum_speed_m_s",
    "maximum_abs_phi_deg",
    "maximum_abs_theta_deg",
    "termination_cause",
    "simulation_or_real",
    "matched_replay_id",
    "claim_status",
)

EPISODE_PRIMITIVE_STEP_COLUMNS = (
    "episode_id",
    "step_index",
    "primitive_id",
    "primitive_family",
    "entry_source",
    "W_layer",
    "fan_branch",
    "governor_decision_status",
    "accepted",
    "start_state_vector",
    "terminal_state_vector",
    "duration_s",
    "energy_residual_m",
    "minimum_margin_m",
    "minimum_speed_m_s",
    "failure_label",
)

EPISODE_GOVERNOR_QUERY_COLUMNS = (
    "episode_id",
    "step_index",
    "candidate_id",
    "primitive_family",
    "entry_source",
    "W_layer",
    "fan_branch",
    "accepted",
    "primary_rejection_reason",
    "all_rejection_reasons",
    "candidate_score",
    "lift_confidence",
)

EPISODE_BELIEF_UPDATE_COLUMNS = (
    "episode_id",
    "fan_branch",
    "belief_grid_id",
    "belief_before_hash",
    "belief_after_hash",
    "memory_lambda",
    "observation_count",
    "source_episode_id",
    "belief_update_status",
)

TERMINATION_CAUSES = (
    "safe_exit",
    "wall_margin_stop",
    "floor_or_ceiling_stop",
    "low_speed_stop",
    "vicon_lost",
    "controller_no_go",
    "max_duration",
    "numerical_failure",
    "operator_abort",
)

CLAIM_STATUS_VALUES = (
    "simulation_only",
    "hardware_shakedown",
    "real_flight_evidence",
    "partial_transfer",
    "negative_transfer",
    "instrumentation_limited",
    "not_tested",
)

ENTRY_SOURCE_VALUES = (
    "launch_gate_main",
    "launch_gate_tolerance_shell",
    "reachable_downstream",
    "local_robustness_shell",
    "diagnostic_broad_only",
)

W_LAYER_VALUES = ("W0", "W1", "W2", "W3", "real")
FAN_BRANCH_VALUES = ("single_fan_branch", "four_fan_branch")

FORBIDDEN_UNSUPPORTED_CLAIMS = (
    "real-flight transfer",
    "sim-to-real demonstrated",
    "mission success",
    "same-flight recapture",
    "perching",
    "all-arena validity",
    "hardware-ready agile",
)


@dataclass(frozen=True)
class EpisodeSchemaValidation:
    table_name: str
    row_count: int
    required_columns: tuple[str, ...]


def empty_episode_summary_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=EPISODE_SUMMARY_COLUMNS)


def empty_primitive_step_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=EPISODE_PRIMITIVE_STEP_COLUMNS)


def empty_governor_query_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=EPISODE_GOVERNOR_QUERY_COLUMNS)


def empty_belief_update_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=EPISODE_BELIEF_UPDATE_COLUMNS)


def validate_episode_summary_frame(frame: pd.DataFrame) -> EpisodeSchemaValidation:
    _require_columns(frame, EPISODE_SUMMARY_COLUMNS, "episode_summary")
    _require_values(frame, "termination_cause", TERMINATION_CAUSES, "episode_summary")
    _require_values(frame, "claim_status", CLAIM_STATUS_VALUES, "episode_summary")
    _require_values(frame, "W_layer", W_LAYER_VALUES, "episode_summary")
    _require_values(frame, "fan_branch", FAN_BRANCH_VALUES, "episode_summary")
    return EpisodeSchemaValidation("episode_summary", int(len(frame)), EPISODE_SUMMARY_COLUMNS)


def validate_primitive_step_frame(frame: pd.DataFrame) -> EpisodeSchemaValidation:
    _require_columns(frame, EPISODE_PRIMITIVE_STEP_COLUMNS, "episode_primitive_steps")
    _require_values(frame, "entry_source", ENTRY_SOURCE_VALUES, "episode_primitive_steps")
    _require_values(frame, "W_layer", W_LAYER_VALUES, "episode_primitive_steps")
    return EpisodeSchemaValidation("episode_primitive_steps", int(len(frame)), EPISODE_PRIMITIVE_STEP_COLUMNS)


def validate_governor_query_frame(frame: pd.DataFrame) -> EpisodeSchemaValidation:
    _require_columns(frame, EPISODE_GOVERNOR_QUERY_COLUMNS, "episode_governor_queries")
    _require_values(frame, "entry_source", ENTRY_SOURCE_VALUES, "episode_governor_queries")
    _require_values(frame, "W_layer", W_LAYER_VALUES, "episode_governor_queries")
    return EpisodeSchemaValidation("episode_governor_queries", int(len(frame)), EPISODE_GOVERNOR_QUERY_COLUMNS)


def validate_belief_update_frame(frame: pd.DataFrame) -> EpisodeSchemaValidation:
    _require_columns(frame, EPISODE_BELIEF_UPDATE_COLUMNS, "episode_belief_update")
    return EpisodeSchemaValidation("episode_belief_update", int(len(frame)), EPISODE_BELIEF_UPDATE_COLUMNS)


def validate_episode_tables(
    episode_summary: pd.DataFrame,
    primitive_steps: pd.DataFrame,
    governor_queries: pd.DataFrame,
    belief_updates: pd.DataFrame,
) -> tuple[EpisodeSchemaValidation, ...]:
    return (
        validate_episode_summary_frame(episode_summary),
        validate_primitive_step_frame(primitive_steps),
        validate_governor_query_frame(governor_queries),
        validate_belief_update_frame(belief_updates),
    )


def unsupported_claim_errors(text: str) -> list[str]:
    """Return unqualified claim phrases found in `text`.

    A small negation window is intentional here. Reports often need to name a
    forbidden claim to deny it, and those explicit denials should pass.
    """

    lower = str(text).lower()
    errors: list[str] = []
    for phrase in FORBIDDEN_UNSUPPORTED_CLAIMS:
        start = 0
        while True:
            index = lower.find(phrase, start)
            if index < 0:
                break
            window = lower[max(0, index - 80): index]
            sentence = _sentence_window(lower, index)
            negated = any(
                marker in window
                for marker in (
                    "no ",
                    "not ",
                    "without direct evidence",
                    "does not claim",
                    "must not claim",
                    "blocked",
                    "claim boundary",
                )
            ) or any(
                marker in sentence
                for marker in (
                    "no ",
                    "not ",
                    "without direct evidence",
                    "does not claim",
                    "must not claim",
                    "blocked",
                    "claim boundary",
                )
            )
            if not negated:
                errors.append(phrase)
            start = index + len(phrase)
    return errors


def assert_claim_safe_text(text: str) -> None:
    errors = unsupported_claim_errors(text)
    if errors:
        raise ValueError(f"unsupported claim phrases require explicit denial: {sorted(set(errors))}")


def _sentence_window(text: str, index: int) -> str:
    left_candidates = [text.rfind(separator, 0, index) for separator in (".", "\n")]
    left = max(left_candidates)
    right_candidates = [pos for pos in (text.find(".", index), text.find("\n", index)) if pos >= 0]
    right = min(right_candidates) if right_candidates else len(text)
    return text[left + 1:right]


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...], table_name: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def _require_values(
    frame: pd.DataFrame,
    column: str,
    allowed: tuple[str, ...],
    table_name: str,
) -> None:
    if frame.empty:
        return
    values = set(frame[column].dropna().astype(str))
    invalid = sorted(values.difference(allowed))
    if invalid:
        raise ValueError(f"{table_name}.{column} contains unsupported values: {invalid}")
