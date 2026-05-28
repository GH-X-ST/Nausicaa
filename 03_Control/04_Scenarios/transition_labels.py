from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from arena_contract import TRUE_SAFE_BOUNDS, heading_aligned_wall_margins_m, position_margin_m
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
    "transition_object": ("launch_gate", "inflight_stable", "boundary_near", "recoverable_degraded"),
    "launch_capable": ("launch_gate",),
    "inflight_only": ("inflight_stable",),
    "terminal_or_recovery": ("boundary_near", "recoverable_degraded"),
}

# The online route lets the first post-launch handoff use the same in-flight
# primitive set while still recording the state as a distinct transition class.
GOVERNOR_ENTRY_CLASSES_BY_ROLE = {
    "transition_object": ("launch_gate", "post_launch_degraded", "inflight_stable", "boundary_near", "recoverable_degraded"),
    "launch_capable": ("launch_gate",),
    "inflight_only": ("post_launch_degraded", "inflight_stable"),
    "terminal_or_recovery": ("boundary_near", "recoverable_degraded"),
}

REQUIRED_EXIT_CLASSES_BY_ENTRY_CLASS = {
    "launch_gate": ("post_launch_degraded", "inflight_stable"),
    "post_launch_degraded": ("inflight_stable", "boundary_near", "safe_terminal"),
    "inflight_stable": ("inflight_stable", "boundary_near", "safe_terminal"),
    "boundary_near": ("inflight_stable", "safe_terminal"),
    "recoverable_degraded": ("inflight_stable", "recoverable_degraded", "safe_terminal"),
    "safe_terminal": (),
    "hard_failure": (),
}

REQUIRED_EXIT_CLASSES_BY_ROLE = {
    "transition_object": tuple(sorted({item for values in REQUIRED_EXIT_CLASSES_BY_ENTRY_CLASS.values() for item in values})),
    "launch_capable": ("post_launch_degraded", "inflight_stable"),
    "inflight_only": ("inflight_stable", "boundary_near", "safe_terminal"),
    "terminal_or_recovery": ("inflight_stable", "recoverable_degraded", "safe_terminal"),
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

TRANSITION_LABEL_VERSION = "transition_labels_v3_recovery_progress"
TURN_INTENT_METRIC_VERSION = "signed_turn_intent_v3_rollrate_lateral_convention"
BOUNDARY_NEAR_MARGIN_M = 0.25
BOUNDARY_NEAR_FRONT_TIME_MARGIN_S = 0.45
BOUNDARY_NEAR_SIDE_TIME_MARGIN_S = 0.35
BOUNDARY_NEAR_MIN_SPEED_FOR_TIME_MARGIN_M_S = 0.50
RECOVERABLE_MAX_ABS_ROLL_RAD = math.radians(35.0)
RECOVERABLE_MAX_ABS_PITCH_RAD = math.radians(22.0)
RECOVERABLE_MAX_BODY_RATE_RAD_S = 0.65
RECOVERY_PROGRESS_MIN_RISK_REDUCTION = 0.05
RECOVERY_PROGRESS_MAX_FLOOR_MARGIN_LOSS_M = 0.03
RECOVERY_PROGRESS_MAX_BOUNDARY_TIME_LOSS_S = 0.05
RECOVERY_PROGRESS_TIME_MARGIN_CAP_S = 10.0


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
    if _time_to_boundary_is_near(x):
        return "boundary_near"
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
    exit_class = _exit_class_from_row(
        row,
        entry_role=role,
        entry_class=entry_class,
        primitive_step_index=primitive_step_index,
    )
    progress = _recovery_progress_fields(row, entry_class=entry_class, exit_class=exit_class)
    compatible = transition_is_chain_compatible(entry_role=role, entry_class=entry_class, exit_class=exit_class)
    if entry_class == "recoverable_degraded" and exit_class == "recoverable_degraded":
        compatible = bool(progress["recovery_progress_valid"])
    hard_failure = exit_class == "hard_failure"
    transition_success_probability = 1.0 if compatible else 0.0
    hard_failure_probability = 1.0 if hard_failure else 0.0
    failure_reason = "" if compatible else _transition_failure_reason(role, entry_class, exit_class)
    if entry_class == "recoverable_degraded" and exit_class == "recoverable_degraded" and not compatible:
        failure_reason = str(progress["recovery_progress_status"])
    return {
        "transition_label_version": TRANSITION_LABEL_VERSION,
        "entry_class": entry_class,
        "exit_class": exit_class,
        "transition_pair": f"{entry_class}->{exit_class}",
        "transition_chain_compatible": bool(compatible),
        "transition_failure_reason": failure_reason,
        "transition_success_probability": float(transition_success_probability),
        "hard_failure_probability": float(hard_failure_probability),
        "updraft_gain_proxy_m": _updraft_gain_proxy(row),
        "flight_time_s": _float(row.get("rollout_duration_s", row.get("flight_time_s", 0.0))),
        "next_allowed_entry_roles": ";".join(entry_roles_for_state_class(exit_class)),
        "next_allowed_entry_classes": ";".join(entry_classes_for_state_class(exit_class)),
        **progress,
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
        "transition_next_allowed_entry_classes": transition["next_allowed_entry_classes"],
        "recovery_progress_valid": transition["recovery_progress_valid"],
        "recovery_progress_status": transition["recovery_progress_status"],
        "recovery_progress_initial_risk": transition["recovery_progress_initial_risk"],
        "recovery_progress_exit_risk": transition["recovery_progress_exit_risk"],
        "recovery_progress_risk_delta": transition["recovery_progress_risk_delta"],
        "recovery_progress_floor_margin_delta_m": transition["recovery_progress_floor_margin_delta_m"],
        "recovery_progress_boundary_time_delta_s": transition["recovery_progress_boundary_time_delta_s"],
    }


def turn_intent_row_fields(row: dict[str, Any]) -> dict[str, object]:
    """Return signed turn-intent metrics for mild-turn primitive evidence rows."""

    primitive_id = str(row.get("primitive_id", row.get("variant_primitive_id", "")))
    desired_bank_sign = _turn_desired_bank_sign(primitive_id)
    desired_lateral_sign = _turn_desired_lateral_sign(primitive_id)
    label = _turn_intent_label(primitive_id)
    empty = {
        "turn_intent_metric_version": TURN_INTENT_METRIC_VERSION,
        "turn_intent_label": label,
        "turn_intent_desired_bank_sign": float(desired_bank_sign),
        "turn_intent_desired_lateral_sign": float(desired_lateral_sign),
        "turn_delta_bank_rad": 0.0,
        "turn_delta_heading_rad": 0.0,
        "turn_lateral_displacement_body_m": 0.0,
        "turn_initial_roll_rate_rad_s": 0.0,
        "turn_exit_roll_rate_rad_s": 0.0,
        "turn_signed_bank_delta_rad": 0.0,
        "turn_signed_heading_delta_rad": 0.0,
        "turn_signed_lateral_displacement_m": 0.0,
        "turn_signed_exit_roll_rate_rad_s": 0.0,
        "turn_signed_roll_rate_delta_rad_s": 0.0,
        "turn_intent_bank_score": 0.0,
        "turn_intent_roll_rate_score": 0.0,
        "turn_intent_lateral_score": 0.0,
        "turn_intent_score": 0.0,
        "turn_intent_correct_sign": False,
        "turn_intent_status": "not_turn_primitive" if desired_bank_sign == 0.0 else "unmeasurable_turn_intent",
    }
    if desired_bank_sign == 0.0:
        return empty
    initial = _state_from_row(row, prefix="initial")
    exit_state = _json_state(row.get("exit_state_vector", ""))
    if initial is None or exit_state is None:
        return empty

    delta_bank = float(exit_state[STATE_INDEX["phi"]] - initial[STATE_INDEX["phi"]])
    delta_heading = _wrap_angle_rad(float(exit_state[STATE_INDEX["psi"]] - initial[STATE_INDEX["psi"]]))
    initial_roll_rate = float(initial[STATE_INDEX["p"]])
    exit_roll_rate = float(exit_state[STATE_INDEX["p"]])
    dx = float(exit_state[STATE_INDEX["x_w"]] - initial[STATE_INDEX["x_w"]])
    dy = float(exit_state[STATE_INDEX["y_w"]] - initial[STATE_INDEX["y_w"]])
    heading0 = float(initial[STATE_INDEX["psi"]])
    lateral_body = float(dx * (-math.sin(heading0)) + dy * math.cos(heading0))
    signed_bank = float(desired_bank_sign) * delta_bank
    signed_heading = float(desired_bank_sign) * delta_heading
    signed_lateral = float(desired_lateral_sign) * lateral_body
    signed_exit_roll_rate = float(desired_bank_sign) * exit_roll_rate
    signed_roll_rate_delta = float(desired_bank_sign) * (exit_roll_rate - initial_roll_rate)
    bank_score = _clip01(signed_bank / 0.040)
    roll_rate_score = _clip01(signed_exit_roll_rate / 0.35)
    lateral_score = _clip01(signed_lateral / 0.030)
    turn_score = _clip01(0.55 * bank_score + 0.30 * roll_rate_score + 0.15 * lateral_score)
    correct_sign = bool(signed_bank > 0.0 or signed_exit_roll_rate > 0.0 or signed_lateral > 0.0)
    return {
        **empty,
        "turn_delta_bank_rad": delta_bank,
        "turn_delta_heading_rad": delta_heading,
        "turn_lateral_displacement_body_m": lateral_body,
        "turn_initial_roll_rate_rad_s": initial_roll_rate,
        "turn_exit_roll_rate_rad_s": exit_roll_rate,
        "turn_signed_bank_delta_rad": signed_bank,
        "turn_signed_heading_delta_rad": signed_heading,
        "turn_signed_lateral_displacement_m": signed_lateral,
        "turn_signed_exit_roll_rate_rad_s": signed_exit_roll_rate,
        "turn_signed_roll_rate_delta_rad_s": signed_roll_rate_delta,
        "turn_intent_bank_score": bank_score,
        "turn_intent_roll_rate_score": roll_rate_score,
        "turn_intent_lateral_score": lateral_score,
        "turn_intent_score": turn_score,
        "turn_intent_correct_sign": correct_sign,
        "turn_intent_status": "measurable_turn_intent" if turn_score > 0.0 else "weak_or_wrong_turn_intent",
    }


def transition_is_chain_compatible(*, entry_role: str, exit_class: str, entry_class: str = "") -> bool:
    if entry_class:
        return str(exit_class) in REQUIRED_EXIT_CLASSES_BY_ENTRY_CLASS.get(str(entry_class), ())
    role = str(entry_role)
    if role not in REQUIRED_EXIT_CLASSES_BY_ROLE:
        return False
    return str(exit_class) in REQUIRED_EXIT_CLASSES_BY_ROLE[role]


def entry_roles_for_state_class(state_class: str) -> tuple[str, ...]:
    state = str(state_class)
    if state in {"safe_terminal", "hard_failure"}:
        return ()
    return tuple(role for role, classes in GOVERNOR_ENTRY_CLASSES_BY_ROLE.items() if state in classes)


def entry_classes_for_state_class(state_class: str) -> tuple[str, ...]:
    state = str(state_class)
    if state in {"safe_terminal", "hard_failure"}:
        return ()
    if state == "post_launch_degraded":
        return ("inflight_stable",)
    return (state,)


def required_entry_role_for_state_class(state_class: str) -> str:
    roles = entry_roles_for_state_class(state_class)
    return roles[0] if roles else ""


def start_family_for_state_class(state_class: str) -> str:
    return STATE_CLASS_START_FAMILY.get(str(state_class), "inflight_recovery_edge")


def transition_contract_row() -> dict[str, object]:
    return {
        "transition_label_version": TRANSITION_LABEL_VERSION,
        "state_classes": ";".join(STATE_CLASSES),
        "active_primitive_role_policy": "launch_is_entry_regime_not_primitive_family",
        "launch_gate_required_exit_classes": ";".join(REQUIRED_EXIT_CLASSES_BY_ENTRY_CLASS["launch_gate"]),
        "inflight_stable_required_exit_classes": ";".join(REQUIRED_EXIT_CLASSES_BY_ENTRY_CLASS["inflight_stable"]),
        "boundary_near_required_exit_classes": ";".join(REQUIRED_EXIT_CLASSES_BY_ENTRY_CLASS["boundary_near"]),
        "recoverable_degraded_required_exit_classes": ";".join(REQUIRED_EXIT_CLASSES_BY_ENTRY_CLASS["recoverable_degraded"]),
        "recoverable_degraded_self_exit_requires_progress": True,
        "recovery_progress_min_risk_reduction": float(RECOVERY_PROGRESS_MIN_RISK_REDUCTION),
        "recovery_progress_max_floor_margin_loss_m": float(RECOVERY_PROGRESS_MAX_FLOOR_MARGIN_LOSS_M),
        "recovery_progress_max_boundary_time_loss_s": float(RECOVERY_PROGRESS_MAX_BOUNDARY_TIME_LOSS_S),
        "boundary_near_is_route_state_not_failure": True,
        "boundary_near_static_wall_margin_m": float(BOUNDARY_NEAR_MARGIN_M),
        "boundary_near_front_time_margin_s": float(BOUNDARY_NEAR_FRONT_TIME_MARGIN_S),
        "boundary_near_side_time_margin_s": float(BOUNDARY_NEAR_SIDE_TIME_MARGIN_S),
        "boundary_near_time_margin_uses_front_and_side_not_rear": True,
        "hard_failure_is_only_failure_class": True,
    }


def _exit_class_from_row(
    row: dict[str, Any],
    *,
    entry_role: str,
    entry_class: str,
    primitive_step_index: int | None,
) -> str:
    outcome = str(row.get("outcome_class", "")).lower()
    boundary = str(row.get("boundary_use_class", "")).lower()
    failure = str(row.get("failure_label", "")).lower()
    termination = str(row.get("termination_cause", "")).lower()
    terminal = (
        _truthy(row.get("episode_terminal_useful", False))
        or boundary == "episode_terminal_useful"
        or failure == "controlled_xy_boundary_terminal"
    )
    continuation = _truthy(row.get("continuation_valid", False)) or outcome == "accepted"
    if (
        outcome in {"failed", "blocked", "rejected"}
        or boundary == "hard_failure"
        or failure == "uncontrolled_xy_boundary_exit"
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
            allow_post_launch_degraded=str(entry_class) == "launch_gate" or str(entry_role) == "launch_capable",
        )
        if str(entry_class) == "launch_gate" and state_class == "recoverable_degraded":
            return "post_launch_degraded"
        return state_class

    if "boundary" in boundary or _float(row.get("minimum_wall_margin_m", 1.0)) <= BOUNDARY_NEAR_MARGIN_M:
        return "boundary_near"
    if continuation:
        return "post_launch_degraded" if str(entry_class) == "launch_gate" else "inflight_stable"
    if outcome == "weak":
        return "post_launch_degraded" if str(entry_class) == "launch_gate" else "recoverable_degraded"
    return "recoverable_degraded"


def _transition_failure_reason(entry_role: str, entry_class: str, exit_class: str) -> str:
    role = str(entry_role)
    if str(entry_class) not in REQUIRED_EXIT_CLASSES_BY_ENTRY_CLASS:
        return f"unknown_transition_entry_class_{entry_class}"
    return f"exit_class_{exit_class}_not_chain_compatible_for_entry_class_{entry_class}_role_{role}"


def _recovery_progress_fields(
    row: dict[str, Any],
    *,
    entry_class: str,
    exit_class: str,
) -> dict[str, object]:
    empty = {
        "recovery_progress_valid": False,
        "recovery_progress_status": "not_recoverable_degraded_self_transition",
        "recovery_progress_initial_risk": 0.0,
        "recovery_progress_exit_risk": 0.0,
        "recovery_progress_risk_delta": 0.0,
        "recovery_progress_floor_margin_delta_m": 0.0,
        "recovery_progress_boundary_time_delta_s": 0.0,
    }
    if str(entry_class) != "recoverable_degraded" or str(exit_class) != "recoverable_degraded":
        return empty

    initial = _state_from_row(row, prefix="initial")
    exit_state = _json_state(row.get("exit_state_vector", ""))
    if initial is None or exit_state is None:
        return {**empty, "recovery_progress_status": "recoverable_degraded_self_transition_missing_state_trace"}

    initial_risk = _recoverable_degraded_risk(initial)
    exit_risk = _recoverable_degraded_risk(exit_state)
    risk_delta = float(initial_risk - exit_risk)
    initial_floor = _floor_margin_m(initial)
    exit_floor = _floor_margin_m(exit_state)
    floor_delta = float(exit_floor - initial_floor)
    initial_time = _front_side_boundary_time_margin_s(initial)
    exit_time = _front_side_boundary_time_margin_s(exit_state)
    boundary_time_delta = float(exit_time - initial_time)

    risk_improved = risk_delta >= RECOVERY_PROGRESS_MIN_RISK_REDUCTION
    floor_not_worse = floor_delta >= -RECOVERY_PROGRESS_MAX_FLOOR_MARGIN_LOSS_M
    boundary_time_not_collapsed = boundary_time_delta >= -RECOVERY_PROGRESS_MAX_BOUNDARY_TIME_LOSS_S
    valid = bool(risk_improved and floor_not_worse and boundary_time_not_collapsed)
    if valid:
        status = "recoverable_degraded_self_transition_with_measurable_recovery_progress"
    elif not risk_improved:
        status = "recoverable_degraded_self_transition_without_attitude_rate_progress"
    elif not floor_not_worse:
        status = "recoverable_degraded_self_transition_floor_margin_collapsed"
    else:
        status = "recoverable_degraded_self_transition_boundary_time_margin_collapsed"
    return {
        "recovery_progress_valid": valid,
        "recovery_progress_status": status,
        "recovery_progress_initial_risk": float(initial_risk),
        "recovery_progress_exit_risk": float(exit_risk),
        "recovery_progress_risk_delta": risk_delta,
        "recovery_progress_floor_margin_delta_m": floor_delta,
        "recovery_progress_boundary_time_delta_s": boundary_time_delta,
    }


def _recoverable_degraded_risk(state: np.ndarray) -> float:
    roll_excess = max(abs(float(state[STATE_INDEX["phi"]])) - RECOVERABLE_MAX_ABS_ROLL_RAD, 0.0) / max(
        RECOVERABLE_MAX_ABS_ROLL_RAD,
        1e-9,
    )
    pitch_excess = max(abs(float(state[STATE_INDEX["theta"]])) - RECOVERABLE_MAX_ABS_PITCH_RAD, 0.0) / max(
        RECOVERABLE_MAX_ABS_PITCH_RAD,
        1e-9,
    )
    max_body_rate = max(
        abs(float(state[STATE_INDEX["p"]])),
        abs(float(state[STATE_INDEX["q"]])),
        abs(float(state[STATE_INDEX["r"]])),
    )
    rate_excess = max(max_body_rate - RECOVERABLE_MAX_BODY_RATE_RAD_S, 0.0) / max(
        RECOVERABLE_MAX_BODY_RATE_RAD_S,
        1e-9,
    )
    return float(roll_excess + pitch_excess + rate_excess)


def _floor_margin_m(state: np.ndarray) -> float:
    try:
        margins = position_margin_m(
            state[[STATE_INDEX["x_w"], STATE_INDEX["y_w"], STATE_INDEX["z_w"]]],
            TRUE_SAFE_BOUNDS,
        )
        return float(margins["floor_margin_m"])
    except Exception:
        return -1.0


def _front_side_boundary_time_margin_s(state: np.ndarray) -> float:
    try:
        margins = heading_aligned_wall_margins_m(
            state[[STATE_INDEX["x_w"], STATE_INDEX["y_w"], STATE_INDEX["z_w"]]],
            float(state[STATE_INDEX["psi"]]),
            TRUE_SAFE_BOUNDS,
        )
        speed_m_s = float(np.linalg.norm(state[[STATE_INDEX["u"], STATE_INDEX["v"], STATE_INDEX["w"]]]))
    except Exception:
        return 0.0
    if speed_m_s < BOUNDARY_NEAR_MIN_SPEED_FOR_TIME_MARGIN_M_S:
        return float(RECOVERY_PROGRESS_TIME_MARGIN_CAP_S)
    front_time_s = float(margins["front_wall_margin_m"]) / max(speed_m_s, 1e-9)
    side_time_s = min(
        float(margins["left_wall_margin_m"]),
        float(margins["right_wall_margin_m"]),
    ) / max(speed_m_s, 1e-9)
    return float(max(0.0, min(RECOVERY_PROGRESS_TIME_MARGIN_CAP_S, front_time_s, side_time_s)))


def _state_vector(state: Any) -> np.ndarray:
    if isinstance(state, str):
        state = json.loads(state)
    return as_state_vector(np.asarray(state, dtype=float).reshape(STATE_SIZE))


def _time_to_boundary_is_near(state: np.ndarray) -> bool:
    """Return true when front/side boundary arrival is too close for handoff."""

    try:
        margins = heading_aligned_wall_margins_m(
            state[[STATE_INDEX["x_w"], STATE_INDEX["y_w"], STATE_INDEX["z_w"]]],
            float(state[STATE_INDEX["psi"]]),
            TRUE_SAFE_BOUNDS,
        )
        speed_m_s = float(np.linalg.norm(state[[STATE_INDEX["u"], STATE_INDEX["v"], STATE_INDEX["w"]]]))
    except Exception:
        return True
    if speed_m_s < BOUNDARY_NEAR_MIN_SPEED_FOR_TIME_MARGIN_M_S:
        return False
    front_time_s = float(margins["front_wall_margin_m"]) / max(speed_m_s, 1e-9)
    side_time_s = min(
        float(margins["left_wall_margin_m"]),
        float(margins["right_wall_margin_m"]),
    ) / max(speed_m_s, 1e-9)
    return bool(
        front_time_s <= BOUNDARY_NEAR_FRONT_TIME_MARGIN_S
        or side_time_s <= BOUNDARY_NEAR_SIDE_TIME_MARGIN_S
    )


def _json_state(value: Any) -> np.ndarray | None:
    try:
        if value is None or value == "":
            return None
        return _state_vector(value)
    except Exception:
        return None


def _state_from_row(row: dict[str, Any], *, prefix: str) -> np.ndarray | None:
    vector = _json_state(row.get(f"{prefix}_state_vector", row.get(f"{prefix}_state_vector_json", "")))
    if vector is not None:
        return vector
    keys = (
        "x_w",
        "y_w",
        "z_w",
        "phi",
        "theta",
        "psi",
        "u",
        "v",
        "w",
        "p",
        "q",
        "r",
        "delta_a",
        "delta_e",
        "delta_r",
    )
    if not any(f"{prefix}_{key}" in row for key in keys):
        return None
    values = [_float(row.get(f"{prefix}_{key}", 0.0)) for key in keys]
    try:
        return as_state_vector(np.asarray(values, dtype=float).reshape(STATE_SIZE))
    except Exception:
        return None


def _wrap_angle_rad(value: float) -> float:
    return float(math.atan2(math.sin(float(value)), math.cos(float(value))))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _turn_intent_label(primitive_id: str) -> str:
    if str(primitive_id) == "mild_turn_left":
        return "left"
    if str(primitive_id) == "mild_turn_right":
        return "right"
    return "neutral"


def _turn_desired_bank_sign(primitive_id: str) -> float:
    if str(primitive_id) == "mild_turn_left":
        return -1.0
    if str(primitive_id) == "mild_turn_right":
        return 1.0
    return 0.0


def _turn_desired_lateral_sign(primitive_id: str) -> float:
    if str(primitive_id) == "mild_turn_left":
        return -1.0
    if str(primitive_id) == "mild_turn_right":
        return 1.0
    return 0.0


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
