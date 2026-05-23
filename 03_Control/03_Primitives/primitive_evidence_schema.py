from __future__ import annotations

from dataclasses import dataclass


OUTCOME_CLASSES = ("accepted", "weak", "failed", "rejected", "blocked")
BOUNDARY_USE_CLASSES = ("continuation_valid", "episode_terminal_useful", "hard_failure", "blocked")

HARD_FAILURE_LABELS = {
    "floor_violation",
    "ceiling_violation",
    "z_boundary_exit",
    "initial_floor_violation",
    "initial_ceiling_violation",
    "nonfinite_initial_state",
    "nonfinite_trajectory",
    "corrupt_integration",
    "physically_impossible_initial_state",
    "true_safety_violation",
}


@dataclass(frozen=True)
class EvidenceUseLabels:
    continuation_valid: bool
    episode_terminal_useful: bool
    continuation_status: str
    episode_terminal_status: str
    episode_utility_label: str
    terminal_use_trainable: bool
    boundary_use_class: str
    exit_check_status: str


def terminal_evidence_is_useful(*, energy_residual_m: float, lift_dwell_time_s: float) -> bool:
    """Return whether a retained x-y boundary exit is useful terminal episode evidence."""

    return bool(float(energy_residual_m) >= 0.0 or float(lift_dwell_time_s) >= 0.20)


def evidence_use_labels(
    *,
    outcome_class: str,
    failure_label: str,
    termination_cause: str,
    energy_residual_m: float,
    lift_dwell_time_s: float,
    trajectory_status: str,
) -> EvidenceUseLabels:
    """Derive canonical continuation and terminal-use labels for one evidence row."""

    outcome = str(outcome_class)
    failure = str(failure_label)
    termination = str(termination_cause)
    is_xy_terminal = failure == "xy_boundary_terminal" or termination in {
        "wall_boundary_exit_retained",
        "lateral_boundary_exit_retained",
    }
    is_blocked = outcome == "blocked"
    is_hard_failure = failure in HARD_FAILURE_LABELS or "nonfinite" in failure or "corrupt" in failure
    terminal_useful = (
        is_xy_terminal
        and trajectory_status == "finite_model_backed"
        and terminal_evidence_is_useful(
            energy_residual_m=energy_residual_m,
            lift_dwell_time_s=lift_dwell_time_s,
        )
    )
    continuation_valid = outcome in {"accepted", "weak"} and not is_xy_terminal

    if is_blocked and is_hard_failure:
        boundary_use_class = "hard_failure"
        continuation_status = "continuation_failed"
        episode_utility_label = "not_useful"
        exit_check_status = "hard_failure"
    elif is_blocked:
        boundary_use_class = "blocked"
        continuation_status = "blocked"
        episode_utility_label = "blocked"
        exit_check_status = "blocked"
    elif terminal_useful:
        boundary_use_class = "episode_terminal_useful"
        continuation_status = "not_continuation_valid"
        episode_utility_label = "terminal_useful"
        exit_check_status = "episode_terminal_useful_retained"
    elif continuation_valid:
        boundary_use_class = "continuation_valid"
        continuation_status = "continuation_success" if outcome == "accepted" else "continuation_weak"
        episode_utility_label = "continuation_useful"
        exit_check_status = "continuation_exit_valid"
    else:
        boundary_use_class = "hard_failure"
        continuation_status = "continuation_failed"
        episode_utility_label = "not_useful"
        exit_check_status = "hard_failure" if is_hard_failure else "failed_or_rejected"

    return EvidenceUseLabels(
        continuation_valid=bool(continuation_valid),
        episode_terminal_useful=bool(terminal_useful),
        continuation_status=continuation_status,
        episode_terminal_status="episode_terminal_useful" if terminal_useful else "not_terminal",
        episode_utility_label=episode_utility_label,
        terminal_use_trainable=bool(terminal_useful),
        boundary_use_class=boundary_use_class,
        exit_check_status=exit_check_status,
    )


def canonical_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}
