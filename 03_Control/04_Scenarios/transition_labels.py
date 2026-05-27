from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from arena_contract import TRUE_SAFE_BOUNDS, position_margin_m
from state_contract import STATE_INDEX, STATE_SIZE, as_state_vector


STATE_CLASSES = (
    "launch_gate",
    "post_launch_degraded",
    "inflight_stable",
    "boundary_near",
    "recoverable_degraded",
    "safe_terminal",
    "hard_failure",
)

ENTRY_CLASSES_BY_ROLE = {
    "launch_capable": ("launch_gate",),
    "inflight_only": ("inflight_stable",),
    "terminal_or_recovery": ("boundary_near", "recoverable_degraded"),
}

# The online route lets the first post-launch handoff use the same in-flight
# primitive set while still recording the state as a distinct transition class.
GOVERNOR_ENTRY_CLASSES_BY_ROLE = {
    "launch_capable": ("launch_gate",),
    "inflight_only": ("post_launch_degraded", "inflight_stable"),
    "terminal_or_recovery": ("boundary_near", "recoverable_degraded"),
}

REQUIRED_EXIT_CLASSES_BY_ROLE = {
    "launch_capable": ("post_launch_degraded", "inflight_stable"),
    "inflight_only": ("inflight_stable", "boundary_near", "safe_terminal"),
    "terminal_or_recovery": ("inflight_stable", "safe_terminal"),
}

START_FAMILY_ENTRY_CLASS = {
    "launch_gate": "launch_gate",
    "inflight_nominal": "inflight_stable",
    "inflight_lift_region": "inflight_stable",
    "inflight_boundary_near": "boundary_near",
    "inflight_recovery_edge": "recoverable_degraded",
}

STATE_CLASS_START_FAMILY = {
    "launch_gate": "launch_gate",
    "post_launch_degraded": "inflight_nominal",
    "inflight_stable": "inflight_nominal",
    "boundary_near": "inflight_boundary_near",
    "recoverable_degraded": "inflight_recovery_edge",
    "safe_terminal": "inflight_recovery_edge",
    "hard_failure": "inflight_recovery_edge",
}

TRANSITION_LABEL_VERSION = "transition_labels_v1_transition_aware_primitives"
BOUNDARY_NEAR_MARGIN_M = 0.25
RECOVERABLE_MAX_ABS_ROLL_RAD = math.radians(35.0)
RECOVERABLE_MAX_ABS_PITCH_RAD = math.radians(22.0)
RECOVERABLE_MAX_BODY_RATE_RAD_S = 0.65


def classify_state(
    state: Any | None = None,
    *,
    start_state_family: str = "",
    primitive_step_index: int | None = None,
    allow_post_launch_degraded: bool = False,
) -> str:
    """Return the compact transition state class for a start or exit state."""

    if state is None:
        family_class = START_FAMILY_ENTRY_CLASS.get(str(start_state_family), "")
        return family_class if family_class else "recoverable_degraded"
    try:
        x = _state_vector(state)
        margins = position_margin_m(
            x[[STATE_INDEX["x_w"], STATE_INDEX["y_w"], STATE_INDEX["z_w"]]],
            TRUE_SAFE_BOUNDS,
        )
    except Exception:
        return "hard_failure"

    if min(float(margins["floor_margin_m"]), float(margins["ceiling_margin_m"])) < 0.0:
        return "hard_failure"
    if float(margins["min_wall_margin_m"]) <= BOUNDARY_NEAR_MARGIN_M:
        return "boundary_near"

    max_body_rate = max(
        abs(float(x[STATE_INDEX["p"]])),
        abs(float(x[STATE_INDEX["q"]])),
        abs(float(x[STATE_INDEX["r"]])),
    )
    degraded = (
        abs(float(x[STATE_INDEX["phi"]])) > RECOVERABLE_MAX_ABS_ROLL_RAD
        or abs(float(x[STATE_INDEX["theta"]])) > RECOVERABLE_MAX_ABS_PITCH_RAD
        or max_body_rate > RECOVERABLE_MAX_BODY_RATE_RAD_S
    )
    if degraded:
        if allow_post_launch_degraded or primitive_step_index == 1:
            return "post_launch_degraded"
        return "recoverable_degraded"
    return "inflight_stable"


def classify_transition(
    row: dict[str, Any],
    *,
    entry_role: str | None = None,
    start_state_family: str | None = None,
    primitive_step_index: int | None = None,
) -> dict[str, object]:
    """Classify one primitive rollout as an entry-class to exit-class object."""

    role = str(entry_role if entry_role is not None else row.get("entry_role", row.get("variant_entry_role", "")))
    family = str(
        start_state_family
        if start_state_family is not None
        else row.get("start_state_family", row.get("context_start_state_family", ""))
    )
    entry_class = classify_state(start_state_family=family)
    exit_class = _exit_class_from_row(row, entry_role=role, primitive_step_index=primitive_step_index)
    compatible = transition_is_chain_compatible(entry_role=role, entry_class=entry_class, exit_class=exit_class)
    hard_failure = exit_class == "hard_failure"
    transition_success_probability = 1.0 if compatible else 0.0
    hard_failure_probability = 1.0 if hard_failure else 0.0
    return {
        "transition_label_version": TRANSITION_LABEL_VERSION,
        "entry_class": entry_class,
        "exit_class": exit_class,
        "transition_pair": f"{entry_class}->{exit_class}",
        "transition_chain_compatible": bool(compatible),
        "transition_failure_reason": "" if compatible else _transition_failure_reason(role, entry_class, exit_class),
        "transition_success_probability": float(transition_success_probability),
        "hard_failure_probability": float(hard_failure_probability),
        "updraft_gain_proxy_m": _updraft_gain_proxy(row),
        "flight_time_s": _float(row.get("rollout_duration_s", row.get("flight_time_s", 0.0))),
        "next_allowed_entry_roles": ";".join(entry_roles_for_state_class(exit_class)),
    }


def transition_row_fields(
    row: dict[str, Any],
    *,
    entry_role: str | None = None,
    start_state_family: str | None = None,
    primitive_step_index: int | None = None,
) -> dict[str, object]:
    """Return CSV-ready transition fields for dense evidence rows."""

    transition = classify_transition(
        row,
        entry_role=entry_role,
        start_state_family=start_state_family,
        primitive_step_index=primitive_step_index,
    )
    return {
        "transition_label_version": transition["transition_label_version"],
        "transition_entry_class": transition["entry_class"],
        "transition_exit_class": transition["exit_class"],
        "transition_pair": transition["transition_pair"],
        "transition_chain_compatible": transition["transition_chain_compatible"],
        "transition_failure_reason": transition["transition_failure_reason"],
        "transition_success_probability": transition["transition_success_probability"],
        "hard_failure_probability": transition["hard_failure_probability"],
        "transition_updraft_gain_proxy_m": transition["updraft_gain_proxy_m"],
        "transition_flight_time_s": transition["flight_time_s"],
        "transition_next_allowed_entry_roles": transition["next_allowed_entry_roles"],
    }


def transition_is_chain_compatible(*, entry_role: str, exit_class: str, entry_class: str = "") -> bool:
    role = str(entry_role)
    if role not in REQUIRED_EXIT_CLASSES_BY_ROLE:
        return False
    if entry_class:
        allowed_entries = GOVERNOR_ENTRY_CLASSES_BY_ROLE.get(role, ())
        if str(entry_class) not in allowed_entries:
            return False
    return str(exit_class) in REQUIRED_EXIT_CLASSES_BY_ROLE[role]


def entry_roles_for_state_class(state_class: str) -> tuple[str, ...]:
    state = str(state_class)
    if state in {"safe_terminal", "hard_failure"}:
        return ()
    return tuple(role for role, classes in GOVERNOR_ENTRY_CLASSES_BY_ROLE.items() if state in classes)


def required_entry_role_for_state_class(state_class: str) -> str:
    roles = entry_roles_for_state_class(state_class)
    return roles[0] if roles else ""


def start_family_for_state_class(state_class: str) -> str:
    return STATE_CLASS_START_FAMILY.get(str(state_class), "inflight_recovery_edge")


def transition_contract_row() -> dict[str, object]:
    return {
        "transition_label_version": TRANSITION_LABEL_VERSION,
        "state_classes": ";".join(STATE_CLASSES),
        "launch_capable_required_exit_classes": ";".join(REQUIRED_EXIT_CLASSES_BY_ROLE["launch_capable"]),
        "inflight_only_required_exit_classes": ";".join(REQUIRED_EXIT_CLASSES_BY_ROLE["inflight_only"]),
        "terminal_or_recovery_required_exit_classes": ";".join(REQUIRED_EXIT_CLASSES_BY_ROLE["terminal_or_recovery"]),
        "boundary_near_is_route_state_not_failure": True,
        "hard_failure_is_only_failure_class": True,
    }


def _exit_class_from_row(row: dict[str, Any], *, entry_role: str, primitive_step_index: int | None) -> str:
    outcome = str(row.get("outcome_class", "")).lower()
    boundary = str(row.get("boundary_use_class", "")).lower()
    failure = str(row.get("failure_label", "")).lower()
    termination = str(row.get("termination_cause", "")).lower()
    terminal = _truthy(row.get("episode_terminal_useful", False)) or boundary == "episode_terminal_useful"
    continuation = _truthy(row.get("continuation_valid", False)) or outcome == "accepted"
    if (
        outcome in {"failed", "blocked", "rejected"}
        or boundary == "hard_failure"
        or "hard_failure" in failure
        or "hard_failure" in termination
        or _float(row.get("floor_margin_m", 0.0)) < 0.0
        or _float(row.get("ceiling_margin_m", 0.0)) < 0.0
    ):
        return "hard_failure"
    if terminal:
        return "safe_terminal"

    exit_state = _json_state(row.get("exit_state_vector", ""))
    if exit_state is not None:
        state_class = classify_state(
            exit_state,
            primitive_step_index=primitive_step_index,
            allow_post_launch_degraded=str(entry_role) == "launch_capable",
        )
        if str(entry_role) == "launch_capable" and state_class == "recoverable_degraded":
            return "post_launch_degraded"
        return state_class

    if "boundary" in boundary or _float(row.get("minimum_wall_margin_m", 1.0)) <= BOUNDARY_NEAR_MARGIN_M:
        return "boundary_near"
    if continuation:
        return "post_launch_degraded" if str(entry_role) == "launch_capable" else "inflight_stable"
    if outcome == "weak":
        return "post_launch_degraded" if str(entry_role) == "launch_capable" else "recoverable_degraded"
    return "recoverable_degraded"


def _transition_failure_reason(entry_role: str, entry_class: str, exit_class: str) -> str:
    role = str(entry_role)
    if role not in REQUIRED_EXIT_CLASSES_BY_ROLE:
        return "unknown_entry_role"
    if str(entry_class) not in GOVERNOR_ENTRY_CLASSES_BY_ROLE.get(role, ()):
        return f"entry_class_{entry_class}_not_valid_for_{role}"
    return f"exit_class_{exit_class}_not_chain_compatible_for_{role}"


def _state_vector(state: Any) -> np.ndarray:
    if isinstance(state, str):
        state = json.loads(state)
    return as_state_vector(np.asarray(state, dtype=float).reshape(STATE_SIZE))


def _json_state(value: Any) -> np.ndarray | None:
    try:
        if value is None or value == "":
            return None
        return _state_vector(value)
    except Exception:
        return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _updraft_gain_proxy(row: dict[str, Any]) -> float:
    for key in (
        "trajectory_integrated_updraft_gain_m",
        "updraft_specific_energy_gain_proxy_m",
        "expected_updraft_gain_proxy_m",
        "transition_updraft_gain_proxy_m",
    ):
        if key in row:
            return max(_float(row.get(key, 0.0)), 0.0)
    return 0.0
