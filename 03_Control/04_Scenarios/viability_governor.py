from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys

from lqr_linearisation import lqr_speed_bin_id
from primitive_timing_contract import PRIMITIVE_FINITE_HORIZON_S
from state_contract import STATE_INDEX
from transition_labels import classify_state, entry_classes_for_state_class, transition_is_chain_compatible


_INNER_LOOP_DIR = Path(__file__).resolve().parents[1] / "02_Inner_Loop"
if str(_INNER_LOOP_DIR) not in sys.path:
    sys.path.insert(0, str(_INNER_LOOP_DIR))
try:
    from A_model_parameters import neutral_dry_air_calibration as active_calibration
except Exception:  # pragma: no cover - fallback only for isolated docs/import tooling.
    active_calibration = None


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
    mission_front_progress_weight: float = 0.28
    mission_front_terminal_weight: float = 0.70
    mission_terminal_energy_weight: float = 0.035
    mission_wrong_boundary_penalty_weight: float = 0.35
    calibrated_regime_mismatch_risk_weight: float = 0.12
    calibrated_regime_mismatch_score_cap: float = 0.18
    memory_switch_max_calibrated_regime_risk_increase: float = 0.0
    memory_switch_min_confidence: float = 0.15
    memory_switch_min_score_margin: float = 0.001
    memory_switch_max_base_score_drop: float = 0.12
    memory_near_tie_base_score_margin: float = 0.03
    memory_switch_max_transition_success_drop: float = 0.02
    memory_switch_max_hard_failure_risk_increase: float = 0.0
    exploration_switch_min_uncertainty: float = 0.55
    exploration_switch_min_score_margin: float = 0.0
    exploration_switch_max_base_score_drop: float = 0.01
    exploration_switch_max_transition_success_drop: float = 0.0
    exploration_switch_max_hard_failure_risk_increase: float = 0.0
    exploration_switch_allow_cross_family: bool = False
    adaptive_switch_max_path_exit_margin_drop_m: float = 0.05
    adaptive_switch_min_path_exit_margin_m: float = 0.02
    candidate_path_memory_residual_cap_m: float = 0.75
    candidate_path_memory_specific_energy_residual_cap_m: float = 1.00
    candidate_path_memory_utility_specific_energy_weight: float = 0.75
    candidate_path_memory_utility_updraft_weight: float = 0.25
    candidate_path_memory_full_confidence_observations: float = 3.0
    residual_memory_launch_recency_half_life: float = 4.0
    memory_objective_score_cap: float = 0.20
    memory_objective_min_confidence: float = 0.15
    memory_objective_max_base_score_drop: float = 0.18
    flow_region_attraction_weight: float = 1.20
    flow_region_attraction_score_cap: float = 0.18
    flow_region_attraction_min_confidence: float = 0.15
    flow_region_attraction_max_base_score_drop: float = 0.18
    flow_region_attraction_min_front_progress_ratio: float = 0.45
    memory_information_gain_weight: float = 0.18
    memory_information_gain_score_cap: float = 0.12
    memory_information_gain_min_uncertainty: float = 0.35
    memory_information_gain_max_base_score_drop: float = 0.14
    memory_information_gain_min_front_progress_ratio: float = 0.45
    memory_information_gain_allow_cross_family: bool = True
    memory_route_planning_weight: float = 0.75
    memory_route_information_gain_weight: float = 0.25
    memory_route_score_cap: float = 0.26
    memory_route_min_confidence: float = 0.12
    memory_route_max_base_score_drop: float = 0.22
    memory_route_min_front_progress_ratio: float = 0.40
    memory_route_horizon_primitives: float = 4.0
    memory_route_discount: float = 0.82
    memory_cost_benefit_weight: float = 1.0
    memory_cost_benefit_score_cap: float = 0.35
    memory_cost_benefit_information_gain_weight: float = 0.08
    memory_cost_benefit_progress_cost_weight: float = 0.25
    memory_cost_benefit_risk_cost_weight: float = 0.60
    memory_cost_benefit_margin_cost_weight: float = 0.20


DEFAULT_GOVERNOR_CONFIG = GovernorConfig(
    config_id="v53_mission_aligned_safe_exploration_governor_wall_0p10cm",
    minimum_wall_margin_m=0.001,
    maximum_hard_failure_risk=0.75,
    continuation_weight=1.00,
    terminal_weight=-0.30,
    hard_failure_weight=-0.80,
    updraft_gain_weight=0.04,
    lift_dwell_weight=0.03,
    belief_weight=0.45,
    exploration_bonus_weight=0.02,
    no_viable_penalty=-1.0,
    terminal_mode_bias=0.0,
    continuation_mode_bias=0.0,
    terminal_continuation_weight=0.25,
    terminal_terminal_weight=1.10,
    terminal_hard_failure_weight=-0.75,
    terminal_updraft_gain_weight=0.05,
    terminal_lift_dwell_weight=0.04,
    mission_front_progress_weight=0.28,
    mission_front_terminal_weight=0.70,
    mission_terminal_energy_weight=0.035,
    mission_wrong_boundary_penalty_weight=0.35,
)

BOOL_GOVERNOR_CONFIG_FIELDS = {"exploration_switch_allow_cross_family", "memory_information_gain_allow_cross_family"}

MISSION_DEFAULT_X_MIN_W_M = 1.2
MISSION_DEFAULT_FRONT_WALL_X_W_M = 6.6
MISSION_DEFAULT_Y_MIN_W_M = 0.0
MISSION_DEFAULT_Y_MAX_W_M = 4.4
MISSION_DEFAULT_Z_MIN_W_M = 0.4
MISSION_DEFAULT_Z_MAX_W_M = 3.5
MISSION_FRONT_WALL_TOL_M = 0.05
MISSION_BOUNDARY_TOL_M = 0.02
MISSION_TERMINAL_ENERGY_PROXY_CAP_M = 3.0
SPECIFIC_ENERGY_GRAVITY_M_S2 = 9.80665
CALIBRATED_REGIME_POLICY_VERSION = "active_calibrated_regime_mismatch_risk_v2_active_blend_boundary"
ACTIVE_CALIBRATION_DEFAULT_TRANSITION_START_ALPHA_DEG = 14.0
ACTIVE_CALIBRATION_DEFAULT_POST_STALL_ALPHA_DEG = 18.0
CALIBRATED_REGIME_SOURCE_CALIBRATION_ID = (
    "unavailable"
    if active_calibration is None
    else str(getattr(active_calibration, "CALIBRATION_ID", "unknown_active_calibration"))
)
CALIBRATED_REGIME_TRANSITION_START_ALPHA_DEG = (
    ACTIVE_CALIBRATION_DEFAULT_TRANSITION_START_ALPHA_DEG
    if active_calibration is None
    else float(
        getattr(
            active_calibration,
            "POST_STALL_RESIDUAL_BLEND_START_ALPHA_DEG",
            ACTIVE_CALIBRATION_DEFAULT_TRANSITION_START_ALPHA_DEG,
        )
    )
)
CALIBRATED_REGIME_POST_STALL_ALPHA_DEG = (
    ACTIVE_CALIBRATION_DEFAULT_POST_STALL_ALPHA_DEG
    if active_calibration is None
    else float(
        getattr(
            active_calibration,
            "POST_STALL_RESIDUAL_BLEND_FULL_ALPHA_DEG",
            ACTIVE_CALIBRATION_DEFAULT_POST_STALL_ALPHA_DEG,
        )
    )
)
if CALIBRATED_REGIME_POST_STALL_ALPHA_DEG <= CALIBRATED_REGIME_TRANSITION_START_ALPHA_DEG:
    CALIBRATED_REGIME_POST_STALL_ALPHA_DEG = CALIBRATED_REGIME_TRANSITION_START_ALPHA_DEG + 1.0


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
            values[key] = _bool_value(value) if key in BOOL_GOVERNOR_CONFIG_FIELDS else float(value)
    return GovernorConfig(**values)


def calibrated_regime_mismatch_score_component(
    calibrated_regime_mismatch_risk: float,
    *,
    governor_config: GovernorConfig | None = None,
) -> float:
    cfg = governor_config or DEFAULT_GOVERNOR_CONFIG
    risk = _clip(float(calibrated_regime_mismatch_risk), 0.0, 1.0)
    return -min(
        float(cfg.calibrated_regime_mismatch_score_cap),
        float(cfg.calibrated_regime_mismatch_risk_weight) * risk,
    )


def calibrated_regime_risk_features(
    *,
    alpha_proxy_deg: float | None = None,
    u_m_s: float | None = None,
    w_m_s: float | None = None,
    path_speed_m_s: float | None = None,
    vertical_speed_m_s: float | None = None,
    governor_config: GovernorConfig | None = None,
    alpha_source: str = "",
) -> dict[str, object]:
    """Return calibrated aero-regime exposure used as model-mismatch risk."""

    cfg = governor_config or DEFAULT_GOVERNOR_CONFIG
    source = str(alpha_source or "")
    alpha_deg = _finite_optional_float(alpha_proxy_deg)
    if alpha_deg is None:
        u = _finite_optional_float(u_m_s)
        w = _finite_optional_float(w_m_s)
        if u is not None and w is not None and abs(float(u)) > 1e-9:
            alpha_deg = math.degrees(math.atan2(float(w), float(u)))
            source = source or "reference_state_vector_u_w"
    if alpha_deg is None:
        speed = _finite_optional_float(path_speed_m_s)
        vertical_speed = _finite_optional_float(vertical_speed_m_s)
        if speed is not None and vertical_speed is not None and float(speed) > 1e-9:
            horizontal_speed = math.sqrt(max(float(speed) * float(speed) - float(vertical_speed) * float(vertical_speed), 1e-9))
            alpha_deg = math.degrees(math.atan2(float(vertical_speed), horizontal_speed))
            source = source or "candidate_path_vertical_speed_over_speed"
    if alpha_deg is None:
        alpha_deg = 0.0
        source = source or "unavailable_assumed_attached"

    alpha_abs = abs(float(alpha_deg))
    transition_start = float(CALIBRATED_REGIME_TRANSITION_START_ALPHA_DEG)
    post_stall = float(CALIBRATED_REGIME_POST_STALL_ALPHA_DEG)
    transition_fraction = _clip(
        (float(alpha_abs) - transition_start) / max(1e-9, post_stall - transition_start),
        0.0,
        1.0,
    )
    risk = _smoothstep(transition_fraction)
    label = "normal" if alpha_abs < transition_start else "transition" if alpha_abs < post_stall else "post_stall"
    return {
        "calibrated_regime_policy_version": CALIBRATED_REGIME_POLICY_VERSION,
        "calibrated_regime_source_calibration_id": CALIBRATED_REGIME_SOURCE_CALIBRATION_ID,
        "calibrated_regime_transition_start_alpha_deg": float(transition_start),
        "calibrated_regime_post_stall_alpha_deg": float(post_stall),
        "calibrated_regime_alpha_source": source,
        "calibrated_regime_alpha_proxy_deg": float(alpha_deg),
        "calibrated_regime_alpha_abs_deg": float(alpha_abs),
        "calibrated_regime_label": label,
        "calibrated_transition_activation": float(risk),
        "calibrated_post_stall_activation": 1.0 if alpha_abs >= post_stall else 0.0,
        "calibrated_regime_mismatch_risk": float(risk),
        "calibrated_regime_mismatch_score_component": calibrated_regime_mismatch_score_component(
            risk,
            governor_config=cfg,
        ),
    }


def _candidate_calibrated_regime_risk_features(
    *,
    representative: dict[str, object],
    outcome: dict[str, object],
    belief_features: dict[str, object],
    governor_config: GovernorConfig,
) -> dict[str, object]:
    reference_state = _candidate_reference_state_vector(representative=representative, outcome=outcome)
    if reference_state:
        return calibrated_regime_risk_features(
            u_m_s=reference_state[STATE_INDEX["u"]],
            w_m_s=reference_state[STATE_INDEX["w"]],
            governor_config=governor_config,
            alpha_source="reference_state_vector",
        )
    for key in ("calibrated_regime_alpha_proxy_deg", "belief_candidate_path_alpha_proxy_deg"):
        alpha = _finite_optional_float(belief_features.get(key))
        if alpha is not None:
            return calibrated_regime_risk_features(
                alpha_proxy_deg=alpha,
                governor_config=governor_config,
                alpha_source=key,
            )
    return calibrated_regime_risk_features(
        path_speed_m_s=belief_features.get("belief_candidate_path_speed_m_s"),
        vertical_speed_m_s=belief_features.get("belief_candidate_path_vertical_speed_m_s"),
        governor_config=governor_config,
        alpha_source="candidate_path_geometry",
    )


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
    include_diagnostics: bool = True,
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
    belief_specific_energy_residual = _float(
        belief_features.get(
            "belief_candidate_path_memory_utility_m",
            belief_features.get(
                "belief_local_specific_energy_residual_m",
                belief_features.get("belief_local_energy_residual_m", belief_updraft_gain_residual),
            ),
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
    belief_exploration_scale = (
        _float(belief_features.get("belief_flow_map_exploration_scale", 0.0), default=0.0)
        if "belief_flow_map_exploration_scale" in belief_features
        else None
    )
    belief_observation_count = int(_float(belief_features.get("belief_observation_count", 0.0)))
    history_length = int(_float(context.get("history_length", belief_features.get("history_length", belief_observation_count))))
    mission = _candidate_mission_alignment_features(context=context, belief_features=belief_features)
    calibrated_regime = _candidate_calibrated_regime_risk_features(
        representative=representative,
        outcome=outcome,
        belief_features=belief_features,
        governor_config=cfg,
    )
    calibrated_regime_mismatch_risk = float(calibrated_regime["calibrated_regime_mismatch_risk"])
    base_score = governor_score(
        governor_mode=governor_mode,
        continuation_probability=continuation_probability,
        terminal_useful_probability=terminal_probability,
        hard_failure_risk=hard_failure_risk,
        expected_updraft_gain_proxy_m=score_updraft_gain,
        expected_lift_dwell_time_s=dwell,
        wall_margin_m=wall_margin,
        belief_local_lift_m_s=0.0,
        mission_front_progress_fraction=mission["mission_front_wall_progress_fraction"],
        mission_front_terminal_proxy=mission["mission_front_wall_terminal_proxy"],
        mission_terminal_energy_proxy_m=mission["mission_terminal_energy_progress_proxy_m"],
        mission_wrong_boundary_proxy=mission["mission_wrong_boundary_proxy"],
        calibrated_regime_mismatch_risk=calibrated_regime_mismatch_risk,
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
        belief_local_lift_m_s=belief_specific_energy_residual,
        mission_front_progress_fraction=mission["mission_front_wall_progress_fraction"],
        mission_front_terminal_proxy=mission["mission_front_wall_terminal_proxy"],
        mission_terminal_energy_proxy_m=mission["mission_terminal_energy_progress_proxy_m"],
        mission_wrong_boundary_proxy=mission["mission_wrong_boundary_proxy"],
        calibrated_regime_mismatch_risk=calibrated_regime_mismatch_risk,
        governor_config=cfg,
    )
    memory_component = float(score_with_memory) - float(base_score)
    mission_score_component = _mission_score_component(
        mission_front_progress_fraction=mission["mission_front_wall_progress_fraction"],
        mission_front_terminal_proxy=mission["mission_front_wall_terminal_proxy"],
        mission_terminal_energy_proxy_m=mission["mission_terminal_energy_progress_proxy_m"],
        mission_wrong_boundary_proxy=mission["mission_wrong_boundary_proxy"],
        governor_config=cfg,
    )
    viable = bool(rejection_reason == "")
    exploration_component = _safe_exploration_bonus(
        viable=viable,
        belief_uncertainty=belief_uncertainty,
        history_length=history_length,
        governor_config=cfg,
        exploration_scale=belief_exploration_scale,
    )
    total_score = float(score_with_memory) + float(exploration_component)
    library_size_case_id = str(
        representative.get(
            "library_size_case_id",
            outcome.get("library_size_case_id", context.get("library_size_case_id", "unknown_library_size_case")),
        )
    )
    if not include_diagnostics:
        row = {
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
            "base_library_score_component": float(base_score if viable else float("-inf")),
            "base_score_without_memory": float(base_score if rejection_reason == "" else float("-inf")),
            "memory_score_component": float(memory_component if viable else 0.0),
            "memory_residual_score_component": float(memory_component if viable else 0.0),
            "mission_score_component": float(mission_score_component if viable else 0.0),
            **calibrated_regime,
            **mission,
            "exploration_score_component": float(exploration_component if viable else 0.0),
            "score_with_memory": float(score_with_memory if rejection_reason == "" else float("-inf")),
            "total_score_with_memory_and_exploration": float(total_score if viable else float("-inf")),
            "safe_exploration_status": (
                "applied_after_viability_filter_uncertainty_bonus_requires_baseline_shield"
                if viable and float(exploration_component) > 0.0
                else "applied_after_viability_filter_zero_or_disabled"
                if viable
                else "not_applied_rejected_before_exploration"
            ),
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
            "belief_local_energy_residual_m": belief_specific_energy_residual,
            "belief_local_specific_energy_residual_m": belief_specific_energy_residual,
            "belief_uncertainty": belief_uncertainty,
            "belief_observation_count": belief_observation_count,
            "continuation_probability": continuation_probability,
            "transition_success_probability": transition_success_probability,
            "transition_chain_compatible_rate": _float(
                outcome.get("transition_chain_compatible_rate", transition_success_probability)
            ),
            "terminal_useful_probability": terminal_probability,
            "hard_failure_risk": hard_failure_risk,
            "expected_net_specific_energy_delta_m": expected_net_specific_energy_delta_m,
            "expected_updraft_gain_proxy_m": updraft_gain,
            "score_updraft_gain_proxy_m": score_updraft_gain,
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
            "claim_status": "simulation_only_viability_governor_candidate_controller_row",
        }
        for key, value in belief_features.items():
            if str(key).startswith("belief_") and key not in row:
                row[str(key)] = value
        return row
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
        "base_library_score_component": float(base_score if viable else float("-inf")),
        "base_score_without_memory": float(base_score if rejection_reason == "" else float("-inf")),
        "memory_score_component": float(memory_component if viable else 0.0),
        "memory_residual_score_component": float(memory_component if viable else 0.0),
        "mission_score_component": float(mission_score_component if viable else 0.0),
        **calibrated_regime,
        **mission,
        "exploration_score_component": float(exploration_component if viable else 0.0),
        "score_with_memory": float(score_with_memory if rejection_reason == "" else float("-inf")),
        "total_score_with_memory_and_exploration": float(total_score if viable else float("-inf")),
        "safe_exploration_status": (
            "applied_after_viability_filter_uncertainty_bonus_requires_baseline_shield"
            if viable and float(exploration_component) > 0.0
            else "applied_after_viability_filter_zero_or_disabled"
            if viable
            else "not_applied_rejected_before_exploration"
        ),
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
        "belief_local_energy_residual_m": belief_specific_energy_residual,
        "belief_local_specific_energy_residual_m": belief_specific_energy_residual,
        "belief_energy_residual_alias_status": "active_total_specific_energy_memory_utility",
        "belief_local_dwell_residual_s": belief_dwell,
        "belief_mean_lift_m_s": belief_mean,
        "belief_max_lift_m_s": belief_max,
        "belief_uncertainty": belief_uncertainty,
        "belief_observation_count": belief_observation_count,
        "belief_effective_observation_count": _float(belief_features.get("belief_effective_observation_count", belief_observation_count)),
        "belief_recency_weight": _float(belief_features.get("belief_recency_weight", 0.0)),
        "belief_observation_age": int(_float(belief_features.get("belief_observation_age", 0.0))),
        "belief_launch_recency_weight": _float(belief_features.get("belief_launch_recency_weight", 0.0)),
        "belief_history_launch_age": int(_float(belief_features.get("belief_history_launch_age", 0.0))),
        "belief_last_history_launch_index": int(_float(belief_features.get("belief_last_history_launch_index", -1.0))),
        "belief_current_history_launch_index": int(_float(belief_features.get("belief_current_history_launch_index", -1.0))),
        "belief_launch_recency_half_life": _float(belief_features.get("belief_launch_recency_half_life", 0.0)),
        "belief_direction_bin": int(_float(belief_features.get("belief_direction_bin", 0.0))),
        "belief_z_bin": int(_float(belief_features.get("belief_z_bin", 0.0))),
        "belief_update_count": int(_float(belief_features.get("belief_update_count", 0.0))),
        "belief_memory_policy_version": str(belief_features.get("belief_memory_policy_version", "")),
        "belief_candidate_path_probe_count": int(_float(belief_features.get("belief_candidate_path_probe_count", 0.0))),
        "belief_candidate_path_lookahead_s": _float(belief_features.get("belief_candidate_path_lookahead_s", 0.0)),
        "belief_candidate_path_residual_memory_active": bool(
            belief_features.get("belief_candidate_path_residual_memory_active", False)
        ),
        "belief_candidate_path_confidence": _float(belief_features.get("belief_candidate_path_confidence", 0.0)),
        "belief_candidate_path_updraft_residual_uncapped_m": _float(
            belief_features.get("belief_candidate_path_updraft_residual_uncapped_m", belief_updraft_gain_residual)
        ),
        "belief_candidate_path_specific_energy_residual_uncapped_m": _float(
            belief_features.get("belief_candidate_path_specific_energy_residual_uncapped_m", belief_specific_energy_residual)
        ),
        "belief_candidate_path_specific_energy_residual_cap_m": _float(
            belief_features.get("belief_candidate_path_specific_energy_residual_cap_m", 0.0)
        ),
        "belief_candidate_path_memory_utility_m": _float(
            belief_features.get("belief_candidate_path_memory_utility_m", belief_specific_energy_residual)
        ),
        "belief_candidate_path_memory_utility_without_attraction_m": _float(
            belief_features.get("belief_candidate_path_memory_utility_without_attraction_m", belief_specific_energy_residual)
        ),
        "belief_flow_map_grid_resolution_m": _float(belief_features.get("belief_flow_map_grid_resolution_m", 0.0)),
        "belief_flow_map_query_radius_m": _float(belief_features.get("belief_flow_map_query_radius_m", 0.0)),
        "belief_flow_map_reachable_attraction_m": _float(
            belief_features.get("belief_flow_map_reachable_attraction_m", 0.0)
        ),
        "belief_flow_map_reachable_attraction_raw_m": _float(
            belief_features.get("belief_flow_map_reachable_attraction_raw_m", 0.0)
        ),
        "belief_flow_map_reachable_attraction_cap_m": _float(
            belief_features.get("belief_flow_map_reachable_attraction_cap_m", 0.0)
        ),
        "belief_flow_map_reachable_attraction_confidence": _float(
            belief_features.get("belief_flow_map_reachable_attraction_confidence", 0.0)
        ),
        "belief_flow_map_reachable_attraction_query_count": int(
            _float(belief_features.get("belief_flow_map_reachable_attraction_query_count", 0.0))
        ),
        "belief_flow_map_reachable_attraction_observation_count": int(
            _float(belief_features.get("belief_flow_map_reachable_attraction_observation_count", 0.0))
        ),
        "belief_flow_map_reachable_attraction_best_x_w_m": _float(
            belief_features.get("belief_flow_map_reachable_attraction_best_x_w_m", 0.0)
        ),
        "belief_flow_map_reachable_attraction_best_y_w_m": _float(
            belief_features.get("belief_flow_map_reachable_attraction_best_y_w_m", 0.0)
        ),
        "belief_flow_map_reachable_attraction_best_z_w_m": _float(
            belief_features.get("belief_flow_map_reachable_attraction_best_z_w_m", 0.0)
        ),
        "belief_flow_map_reachable_attraction_lookahead_m": _float(
            belief_features.get("belief_flow_map_reachable_attraction_lookahead_m", 0.0)
        ),
        "belief_flow_map_reachable_attraction_half_angle_rad": _float(
            belief_features.get("belief_flow_map_reachable_attraction_half_angle_rad", 0.0)
        ),
        "belief_flow_map_reachable_attraction_azimuth_half_angle_rad": _float(
            belief_features.get(
                "belief_flow_map_reachable_attraction_azimuth_half_angle_rad",
                belief_features.get("belief_flow_map_reachable_attraction_half_angle_rad", 0.0),
            )
        ),
        "belief_flow_map_reachable_attraction_elevation_half_angle_rad": _float(
            belief_features.get("belief_flow_map_reachable_attraction_elevation_half_angle_rad", 0.0)
        ),
        "belief_flow_map_reachable_attraction_geometry": str(
            belief_features.get("belief_flow_map_reachable_attraction_geometry", "")
        ),
        "belief_flow_map_candidate_path_uncertainty": _float(
            belief_features.get("belief_flow_map_candidate_path_uncertainty", belief_uncertainty)
        ),
        "belief_flow_map_memory_guided_exploration_uncertainty": _float(
            belief_features.get("belief_flow_map_memory_guided_exploration_uncertainty", belief_uncertainty)
        ),
        "belief_flow_map_information_gain": _float(
            belief_features.get("belief_flow_map_information_gain", 0.0)
        ),
        "belief_flow_map_information_gain_path_uncertainty": _float(
            belief_features.get("belief_flow_map_information_gain_path_uncertainty", 0.0)
        ),
        "belief_flow_map_information_gain_reachable_uncertainty": _float(
            belief_features.get("belief_flow_map_information_gain_reachable_uncertainty", 0.0)
        ),
        "belief_flow_map_information_gain_progress_gate": _float(
            belief_features.get("belief_flow_map_information_gain_progress_gate", 0.0)
        ),
        "belief_flow_map_information_gain_safe_gate": _float(
            belief_features.get("belief_flow_map_information_gain_safe_gate", 0.0)
        ),
        "belief_flow_map_information_gain_query_count": int(
            _float(belief_features.get("belief_flow_map_information_gain_query_count", 0.0))
        ),
        "belief_flow_map_information_gain_low_confidence_query_count": int(
            _float(belief_features.get("belief_flow_map_information_gain_low_confidence_query_count", 0.0))
        ),
        "belief_flow_map_exploration_scale": _float(
            belief_features.get("belief_flow_map_exploration_scale", 0.0)
        ),
        "belief_flow_map_policy": str(belief_features.get("belief_flow_map_policy", "")),
        "belief_flow_map_route_policy": str(belief_features.get("belief_flow_map_route_policy", "")),
        "belief_flow_map_route_horizon_primitives": int(
            _float(belief_features.get("belief_flow_map_route_horizon_primitives", 0.0))
        ),
        "belief_flow_map_route_probe_count": int(
            _float(belief_features.get("belief_flow_map_route_probe_count", 0.0))
        ),
        "belief_flow_map_route_exploitation_m": _float(
            belief_features.get("belief_flow_map_route_exploitation_m", 0.0)
        ),
        "belief_flow_map_route_information_gain": _float(
            belief_features.get("belief_flow_map_route_information_gain", 0.0)
        ),
        "belief_flow_map_route_confidence": _float(
            belief_features.get("belief_flow_map_route_confidence", 0.0)
        ),
        "belief_flow_map_route_uncertainty": _float(
            belief_features.get("belief_flow_map_route_uncertainty", 0.0)
        ),
        "belief_flow_map_route_front_progress": _float(
            belief_features.get("belief_flow_map_route_front_progress", 0.0)
        ),
        "belief_flow_map_route_safe_fraction": _float(
            belief_features.get("belief_flow_map_route_safe_fraction", 0.0)
        ),
        "belief_flow_map_route_best_x_w_m": _float(
            belief_features.get("belief_flow_map_route_best_x_w_m", 0.0)
        ),
        "belief_flow_map_route_best_y_w_m": _float(
            belief_features.get("belief_flow_map_route_best_y_w_m", 0.0)
        ),
        "belief_flow_map_route_best_z_w_m": _float(
            belief_features.get("belief_flow_map_route_best_z_w_m", 0.0)
        ),
        "belief_candidate_path_updraft_residual_cap_m": _float(belief_features.get("belief_candidate_path_updraft_residual_cap_m", 0.0)),
        "belief_candidate_path_reference_bank_rad": _float(belief_features.get("belief_candidate_path_reference_bank_rad", 0.0)),
        "belief_candidate_path_heading_offset_rad": _float(belief_features.get("belief_candidate_path_heading_offset_rad", 0.0)),
        "belief_candidate_path_speed_m_s": _float(belief_features.get("belief_candidate_path_speed_m_s", 0.0)),
        "belief_candidate_path_vertical_speed_m_s": _float(
            belief_features.get("belief_candidate_path_vertical_speed_m_s", 0.0)
        ),
        "belief_candidate_path_alpha_proxy_deg": _float(
            belief_features.get("belief_candidate_path_alpha_proxy_deg", calibrated_regime["calibrated_regime_alpha_proxy_deg"])
        ),
        "belief_candidate_path_alpha_abs_deg": _float(
            belief_features.get("belief_candidate_path_alpha_abs_deg", calibrated_regime["calibrated_regime_alpha_abs_deg"])
        ),
        "belief_candidate_path_calibrated_regime_label": str(
            belief_features.get("belief_candidate_path_calibrated_regime_label", calibrated_regime["calibrated_regime_label"])
        ),
        "belief_candidate_path_exit_x_w_m": _float(belief_features.get("belief_candidate_path_exit_x_w_m", 0.0)),
        "belief_candidate_path_exit_y_w_m": _float(belief_features.get("belief_candidate_path_exit_y_w_m", 0.0)),
        "belief_candidate_path_exit_z_w_m": _float(belief_features.get("belief_candidate_path_exit_z_w_m", 0.0)),
        "belief_candidate_path_exit_direction_rad": _float(belief_features.get("belief_candidate_path_exit_direction_rad", 0.0)),
        "belief_candidate_path_exit_wall_margin_m": _float(
            belief_features.get("belief_candidate_path_exit_wall_margin_m", 0.0)
        ),
        "belief_candidate_path_exit_min_margin_m": _float(
            belief_features.get("belief_candidate_path_exit_min_margin_m", 0.0)
        ),
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
    mission_front_progress_fraction: float = 0.0,
    mission_front_terminal_proxy: float = 0.0,
    mission_terminal_energy_proxy_m: float = 0.0,
    mission_wrong_boundary_proxy: float = 0.0,
    calibrated_regime_mismatch_risk: float = 0.0,
    governor_config: GovernorConfig | None = None,
) -> float:
    """Return an interpretable deterministic selector score."""

    cfg = governor_config or DEFAULT_GOVERNOR_CONFIG
    mission_component = _mission_score_component(
        mission_front_progress_fraction=mission_front_progress_fraction,
        mission_front_terminal_proxy=mission_front_terminal_proxy,
        mission_terminal_energy_proxy_m=mission_terminal_energy_proxy_m,
        mission_wrong_boundary_proxy=mission_wrong_boundary_proxy,
        governor_config=cfg,
    )
    calibrated_regime_component = calibrated_regime_mismatch_score_component(
        calibrated_regime_mismatch_risk,
        governor_config=cfg,
    )
    if governor_mode == "terminal_episode_mode":
        return (
            cfg.terminal_mode_bias
            + cfg.terminal_terminal_weight * float(terminal_useful_probability)
            + cfg.terminal_continuation_weight * float(continuation_probability)
            + cfg.terminal_hard_failure_weight * float(hard_failure_risk)
            + cfg.terminal_updraft_gain_weight * float(expected_updraft_gain_proxy_m)
            + cfg.terminal_lift_dwell_weight * float(expected_lift_dwell_time_s)
            + cfg.belief_weight * float(belief_local_lift_m_s)
            + mission_component
            + calibrated_regime_component
        )
    return (
        cfg.continuation_mode_bias
        + cfg.continuation_weight * float(continuation_probability)
        + cfg.terminal_weight * float(terminal_useful_probability)
        + cfg.hard_failure_weight * float(hard_failure_risk)
        + cfg.updraft_gain_weight * float(expected_updraft_gain_proxy_m)
        + cfg.lift_dwell_weight * float(expected_lift_dwell_time_s)
        + cfg.belief_weight * float(belief_local_lift_m_s)
        + mission_component
        + calibrated_regime_component
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


def _candidate_mission_alignment_features(
    *,
    context: dict[str, object],
    belief_features: dict[str, object],
) -> dict[str, float | bool | str]:
    current_x = _finite_float(context.get("current_x_w_m", context.get("x_w_m", MISSION_DEFAULT_X_MIN_W_M)), MISSION_DEFAULT_X_MIN_W_M)
    current_y = _finite_float(context.get("current_y_w_m", context.get("y_w_m", 0.5 * (MISSION_DEFAULT_Y_MIN_W_M + MISSION_DEFAULT_Y_MAX_W_M))), 0.5 * (MISSION_DEFAULT_Y_MIN_W_M + MISSION_DEFAULT_Y_MAX_W_M))
    current_z = _finite_float(context.get("current_z_w_m", context.get("z_w_m", MISSION_DEFAULT_Z_MIN_W_M)), MISSION_DEFAULT_Z_MIN_W_M)
    front_x = _finite_float(
        context.get("front_wall_target_x_w_m", context.get("mission_front_wall_target_x_w_m", MISSION_DEFAULT_FRONT_WALL_X_W_M)),
        MISSION_DEFAULT_FRONT_WALL_X_W_M,
    )
    x_min = _finite_float(context.get("mission_x_min_w_m", MISSION_DEFAULT_X_MIN_W_M), MISSION_DEFAULT_X_MIN_W_M)
    y_min = _finite_float(context.get("mission_terminal_y_min_m", MISSION_DEFAULT_Y_MIN_W_M), MISSION_DEFAULT_Y_MIN_W_M)
    y_max = _finite_float(context.get("mission_terminal_y_max_m", MISSION_DEFAULT_Y_MAX_W_M), MISSION_DEFAULT_Y_MAX_W_M)
    z_min = _finite_float(context.get("mission_terminal_z_min_m", MISSION_DEFAULT_Z_MIN_W_M), MISSION_DEFAULT_Z_MIN_W_M)
    z_max = _finite_float(context.get("mission_terminal_z_max_m", MISSION_DEFAULT_Z_MAX_W_M), MISSION_DEFAULT_Z_MAX_W_M)
    exit_x = _finite_float(belief_features.get("belief_candidate_path_exit_x_w_m", current_x), current_x)
    exit_y = _finite_float(belief_features.get("belief_candidate_path_exit_y_w_m", current_y), current_y)
    exit_z = _finite_float(belief_features.get("belief_candidate_path_exit_z_w_m", current_z), current_z)
    path_speed = _finite_float(belief_features.get("belief_candidate_path_speed_m_s", 0.0), 0.0)
    remaining_x = max(float(front_x) - float(current_x), 1e-6)
    progress = _clip((float(exit_x) - float(current_x)) / remaining_x, 0.0, 1.0)
    front_terminal = float(
        float(exit_x) >= float(front_x) - MISSION_FRONT_WALL_TOL_M
        and float(y_min) - MISSION_FRONT_WALL_TOL_M <= float(exit_y) <= float(y_max) + MISSION_FRONT_WALL_TOL_M
        and float(z_min) - MISSION_FRONT_WALL_TOL_M <= float(exit_z) <= float(z_max) + MISSION_FRONT_WALL_TOL_M
    )
    side_or_rear_boundary = bool(
        float(exit_x) <= float(x_min) + MISSION_BOUNDARY_TOL_M
        or float(exit_y) <= float(y_min) + MISSION_BOUNDARY_TOL_M
        or float(exit_y) >= float(y_max) - MISSION_BOUNDARY_TOL_M
    )
    vertical_boundary = bool(
        float(exit_z) <= float(z_min) + MISSION_BOUNDARY_TOL_M
        or float(exit_z) >= float(z_max) - MISSION_BOUNDARY_TOL_M
    )
    wrong_boundary = float(
        (side_or_rear_boundary or vertical_boundary)
        and float(exit_x) < float(front_x) - MISSION_FRONT_WALL_TOL_M
    )
    specific_energy = (
        float(exit_z) + float(path_speed) * float(path_speed) / (2.0 * SPECIFIC_ENERGY_GRAVITY_M_S2)
        if float(path_speed) > 0.0
        else float(exit_z)
    )
    energy_reference = _finite_float(
        context.get("mission_terminal_specific_energy_reference_m", context.get("terminal_specific_energy_reference_m", z_min)),
        z_min,
    )
    energy_reserve = _clip(float(specific_energy) - float(energy_reference), 0.0, MISSION_TERMINAL_ENERGY_PROXY_CAP_M)
    energy_progress_proxy = float(energy_reserve) * max(float(progress), float(front_terminal))
    return {
        "mission_policy_version": "candidate_path_front_wall_energy_governor_v1",
        "mission_front_wall_target_x_w_m": float(front_x),
        "mission_candidate_path_current_x_w_m": float(current_x),
        "mission_candidate_path_current_y_w_m": float(current_y),
        "mission_candidate_path_current_z_w_m": float(current_z),
        "mission_candidate_path_exit_x_w_m": float(exit_x),
        "mission_candidate_path_exit_y_w_m": float(exit_y),
        "mission_candidate_path_exit_z_w_m": float(exit_z),
        "mission_candidate_path_speed_m_s": float(path_speed),
        "mission_front_wall_progress_fraction": float(progress),
        "mission_front_wall_terminal_proxy": float(front_terminal),
        "mission_wrong_boundary_proxy": float(wrong_boundary),
        "mission_candidate_total_specific_energy_proxy_m": float(specific_energy),
        "mission_terminal_energy_reserve_proxy_m": float(energy_reserve),
        "mission_terminal_energy_progress_proxy_m": float(energy_progress_proxy),
    }


def _mission_score_component(
    *,
    mission_front_progress_fraction: float,
    mission_front_terminal_proxy: float,
    mission_terminal_energy_proxy_m: float,
    mission_wrong_boundary_proxy: float,
    governor_config: GovernorConfig,
) -> float:
    return float(
        governor_config.mission_front_progress_weight * _clip(float(mission_front_progress_fraction), 0.0, 1.0)
        + governor_config.mission_front_terminal_weight * _clip(float(mission_front_terminal_proxy), 0.0, 1.0)
        + governor_config.mission_terminal_energy_weight
        * _clip(float(mission_terminal_energy_proxy_m), 0.0, MISSION_TERMINAL_ENERGY_PROXY_CAP_M)
        - governor_config.mission_wrong_boundary_penalty_weight * _clip(float(mission_wrong_boundary_proxy), 0.0, 1.0)
    )


def _clip(value: float, lower: float, upper: float) -> float:
    return float(min(max(float(value), float(lower)), float(upper)))


def _smoothstep(value: float) -> float:
    x = _clip(float(value), 0.0, 1.0)
    return float(x * x * (3.0 - 2.0 * x))


def _finite_optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    result = _float(value, default=float("nan"))
    return float(result) if math.isfinite(float(result)) else None


def _finite_float(value: object, default: float) -> float:
    result = _float(value, default)
    return float(result) if math.isfinite(float(result)) else float(default)


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


def _candidate_reference_state_vector(
    *,
    representative: dict[str, object],
    outcome: dict[str, object],
) -> list[float]:
    for source in (representative, outcome):
        raw = source.get("reference_state_vector", "")
        if raw in ("", None):
            continue
        try:
            values = json.loads(str(raw)) if isinstance(raw, str) else list(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        try:
            vector = [float(value) for value in values]
        except (TypeError, ValueError):
            continue
        if len(vector) > max(STATE_INDEX["u"], STATE_INDEX["w"]):
            return vector
    return []


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
    exploration_scale: float | None = None,
) -> float:
    if not viable:
        return 0.0
    if exploration_scale is None:
        attenuation = 1.0 / max(1.0, float(history_length + 1) ** 0.5)
    else:
        attenuation = _clip(float(exploration_scale), 0.0, 1.0)
    return float(governor_config.exploration_bonus_weight) * max(0.0, float(belief_uncertainty)) * attenuation


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", "", "none", "nan"}:
        return False
    return bool(value)


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)
