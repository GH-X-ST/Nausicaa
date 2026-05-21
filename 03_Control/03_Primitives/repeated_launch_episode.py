from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from episode_schema import (
    EPISODE_BELIEF_UPDATE_COLUMNS,
    EPISODE_GOVERNOR_QUERY_COLUMNS,
    EPISODE_PRIMITIVE_STEP_COLUMNS,
    EPISODE_SUMMARY_COLUMNS,
)
from episodic_lift_belief import (
    belief_hash,
    initialise_belief_grid,
    observe_lift_from_episode,
    update_belief,
)
from fixed_gate_contract import FIXED_LAUNCH_GATE, launch_gate_admission_status
from fixed_gate_policies import candidate_objectives, policy_definition
from state_contract import as_state_vector


@dataclass(frozen=True)
class RepeatedLaunchEpisodeConfig:
    episode_id: str
    policy_id: str = "episodic_memory_policy"
    fan_branch: str = "single_fan_branch"
    W_layer: str = "W1"
    max_steps: int = 4
    max_duration_s: float = 5.0
    memory_lambda: float = 0.5
    claim_status: str = "simulation_only"


def run_repeated_launch_episode(
    initial_state: np.ndarray,
    candidate_package: pd.DataFrame,
    config: RepeatedLaunchEpisodeConfig,
) -> dict[str, pd.DataFrame]:
    """Run a lightweight repeated-launch episode scaffold.

    Candidate rows represent primitive objectives. This function does not
    replace the primitive rollout backend; it provides the episode-level schema
    and governor-mediated sequencing needed by the fixed-gate mission path.
    """

    state = as_state_vector(initial_state)
    admission = launch_gate_admission_status(state)
    if admission not in {"admitted_main_gate", "admitted_tolerance_shell"}:
        return _no_go_episode(state, admission, config, "controller_no_go")

    policy = policy_definition(config.policy_id)
    belief_before = initialise_belief_grid(config.fan_branch, memory_lambda=float(config.memory_lambda))
    current = state.copy()
    primitive_rows: list[dict[str, object]] = []
    query_rows: list[dict[str, object]] = []
    primitive_sequence: list[str] = []
    elapsed_s = 0.0
    energy_initial = _specific_energy_height_m(current)
    minimum_margin_m = float("inf")
    minimum_speed_m_s = _speed_m_s(current)
    termination = "max_duration"

    for step_index in range(int(config.max_steps)):
        objectives = candidate_objectives(
            policy_id=config.policy_id,
            current_state=current,
            belief=belief_before,
            candidates=candidate_package,
        )
        if objectives.empty:
            termination = "controller_no_go"
            break
        scored = objectives.sort_values("candidate_objective_score", ascending=False)
        accepted = scored[scored.apply(_governor_accepts_candidate, axis=1)].copy()
        for _, row in scored.iterrows():
            accepted_flag = bool(_governor_accepts_candidate(row))
            query_rows.append(_query_row(config, step_index, row, accepted_flag))
        if accepted.empty:
            termination = "controller_no_go"
            break

        selected = accepted.iloc[0].to_dict()
        duration_s = min(_float(selected.get("duration_s", selected.get("dwell_time_s", 0.5))), 1.0)
        energy_delta_m = _float(selected.get("energy_residual_m", 0.0))
        current = current.copy()
        current[0] += _float(selected.get("forward_displacement_m", 0.35))
        current[2] += energy_delta_m
        current[6] = max(0.0, current[6] + _float(selected.get("speed_delta_m_s", 0.0)))
        elapsed_s += duration_s
        primitive_id = str(selected.get("primitive_id", selected.get("candidate_id", f"step_{step_index}")))
        primitive_sequence.append(primitive_id)
        margin_m = _float(selected.get("minimum_margin_m", 0.5))
        minimum_margin_m = min(minimum_margin_m, margin_m)
        minimum_speed_m_s = min(minimum_speed_m_s, _speed_m_s(current))
        primitive_rows.append(_primitive_step_row(config, step_index, selected, state, current, duration_s))
        if minimum_speed_m_s < 3.0:
            termination = "low_speed_stop"
            break
        if margin_m < 0.0:
            termination = "wall_margin_stop"
            break
        if elapsed_s >= float(config.max_duration_s):
            termination = "max_duration"
            break
    else:
        termination = "max_duration"

    trajectory = pd.DataFrame([{"x_w_m": current[0], "y_w_m": current[1], "z_w_m": current[2], "w_lift_m_s": current[2] - state[2]}])
    observation = observe_lift_from_episode(trajectory, energy_residual=_specific_energy_height_m(current) - energy_initial, reference_belief=belief_before)
    belief_after = update_belief(belief_before, observation, _policy_memory_lambda(policy, config))
    summary = _summary_frame(
        config,
        state,
        admission,
        belief_before,
        belief_after,
        primitive_sequence,
        query_rows,
        energy_initial,
        current,
        minimum_margin_m,
        minimum_speed_m_s,
        termination,
    )
    belief = pd.DataFrame(
        [
            {
                "episode_id": config.episode_id,
                "fan_branch": config.fan_branch,
                "belief_grid_id": belief_after.belief_grid_id,
                "belief_before_hash": belief_hash(belief_before),
                "belief_after_hash": belief_hash(belief_after),
                "memory_lambda": float(belief_after.memory_lambda),
                "observation_count": int(belief_after.observation_count),
                "source_episode_id": config.episode_id,
                "belief_update_status": "updated_from_episode_observation",
            }
        ],
        columns=EPISODE_BELIEF_UPDATE_COLUMNS,
    )
    return {
        "episode_summary": summary,
        "primitive_steps": pd.DataFrame(primitive_rows, columns=EPISODE_PRIMITIVE_STEP_COLUMNS),
        "governor_queries": pd.DataFrame(query_rows, columns=EPISODE_GOVERNOR_QUERY_COLUMNS),
        "belief_updates": belief,
    }


def _no_go_episode(
    state: np.ndarray,
    admission: str,
    config: RepeatedLaunchEpisodeConfig,
    termination: str,
) -> dict[str, pd.DataFrame]:
    belief = initialise_belief_grid(config.fan_branch, memory_lambda=float(config.memory_lambda))
    summary = _summary_frame(config, state, admission, belief, belief, [], [], _specific_energy_height_m(state), state, 0.0, _speed_m_s(state), termination)
    return {
        "episode_summary": summary,
        "primitive_steps": pd.DataFrame(columns=EPISODE_PRIMITIVE_STEP_COLUMNS),
        "governor_queries": pd.DataFrame(columns=EPISODE_GOVERNOR_QUERY_COLUMNS),
        "belief_updates": pd.DataFrame(columns=EPISODE_BELIEF_UPDATE_COLUMNS),
    }


def _summary_frame(
    config: RepeatedLaunchEpisodeConfig,
    initial: np.ndarray,
    admission: str,
    belief_before: object,
    belief_after: object,
    primitive_sequence: list[str],
    query_rows: list[dict[str, object]],
    energy_initial: float,
    final: np.ndarray,
    minimum_margin_m: float,
    minimum_speed_m_s: float,
    termination: str,
) -> pd.DataFrame:
    accept_count = sum(1 for row in query_rows if bool(row["accepted"]))
    reject_count = sum(1 for row in query_rows if not bool(row["accepted"]))
    energy_final = _specific_energy_height_m(final)
    min_margin = 0.0 if not np.isfinite(minimum_margin_m) else float(minimum_margin_m)
    row = {
        "episode_id": config.episode_id,
        "policy_id": config.policy_id,
        "fan_branch": config.fan_branch,
        "W_layer": config.W_layer,
        "launch_gate_id": FIXED_LAUNCH_GATE.launch_gate_id,
        "initial_state_vector": _state_text(initial),
        "initial_state_admission_status": admission,
        "belief_id_before": belief_before.belief_grid_id,
        "belief_id_after": belief_after.belief_grid_id,
        "primitive_sequence": ";".join(primitive_sequence),
        "candidate_query_count": int(len(query_rows)),
        "governor_accept_count": int(accept_count),
        "governor_reject_count": int(reject_count),
        "lift_capture_success": bool(accept_count > 0),
        "lift_capture_time_s": 0.0 if accept_count > 0 else float("nan"),
        "lift_dwell_time_s": float(sum(1.0 for _ in primitive_sequence)),
        "energy_initial_m": float(energy_initial),
        "energy_final_m": float(energy_final),
        "energy_residual_m": float(energy_final - energy_initial),
        "minimum_margin_m": min_margin,
        "minimum_speed_m_s": float(minimum_speed_m_s),
        "maximum_abs_phi_deg": float(abs(np.rad2deg(initial[3]))),
        "maximum_abs_theta_deg": float(abs(np.rad2deg(initial[4]))),
        "termination_cause": termination,
        "simulation_or_real": "simulation",
        "matched_replay_id": "",
        "claim_status": config.claim_status,
    }
    return pd.DataFrame([row], columns=EPISODE_SUMMARY_COLUMNS)


def _query_row(config: RepeatedLaunchEpisodeConfig, step_index: int, row: pd.Series, accepted: bool) -> dict[str, object]:
    return {
        "episode_id": config.episode_id,
        "step_index": int(step_index),
        "candidate_id": str(row.get("candidate_id", row.get("primitive_id", ""))),
        "primitive_family": str(row.get("primitive_family", "")),
        "entry_source": str(row.get("entry_source", "launch_gate_main")),
        "W_layer": config.W_layer,
        "fan_branch": config.fan_branch,
        "accepted": bool(accepted),
        "primary_rejection_reason": "none" if accepted else "governor_rejected_candidate_summary",
        "all_rejection_reasons": "none" if accepted else "candidate_not_admissible_for_episode_scaffold",
        "candidate_score": _float(row.get("candidate_objective_score", 0.0)),
        "lift_confidence": _float(row.get("lift_confidence", 1.0)),
    }


def _primitive_step_row(
    config: RepeatedLaunchEpisodeConfig,
    step_index: int,
    selected: dict[str, object],
    start: np.ndarray,
    terminal: np.ndarray,
    duration_s: float,
) -> dict[str, object]:
    return {
        "episode_id": config.episode_id,
        "step_index": int(step_index),
        "primitive_id": str(selected.get("primitive_id", selected.get("candidate_id", ""))),
        "primitive_family": str(selected.get("primitive_family", "")),
        "entry_source": str(selected.get("entry_source", "launch_gate_main")),
        "W_layer": config.W_layer,
        "fan_branch": config.fan_branch,
        "governor_decision_status": "accepted_governor_seed",
        "accepted": True,
        "start_state_vector": _state_text(start),
        "terminal_state_vector": _state_text(terminal),
        "duration_s": float(duration_s),
        "energy_residual_m": _float(selected.get("energy_residual_m", 0.0)),
        "minimum_margin_m": _float(selected.get("minimum_margin_m", 0.0)),
        "minimum_speed_m_s": _speed_m_s(terminal),
        "failure_label": str(selected.get("failure_label", "none")),
    }


def _governor_accepts_candidate(row: pd.Series) -> bool:
    if "accepted" in row and not bool(row["accepted"]):
        return False
    if "evidence_role" in row and str(row["evidence_role"]) not in {"mission_candidate", "matched_replay_evidence"}:
        return False
    if "controller_mode" in row and str(row["controller_mode"]) in {"open_loop_rollout", "command_template_replay"}:
        return False
    if str(row.get("recommended_use", "thesis")) == "reject":
        return False
    if _float(row.get("minimum_margin_m", row.get("min_true_margin_m", 0.1))) < 0.0:
        return False
    if _float(row.get("lift_confidence", 1.0)) < _float(row.get("minimum_lift_confidence", 0.0)):
        return False
    return True


def _policy_memory_lambda(policy: dict[str, object], config: RepeatedLaunchEpisodeConfig) -> float:
    value = policy.get("memory_lambda")
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        candidate = float("nan")
    return float(config.memory_lambda) if not np.isfinite(candidate) else candidate


def _specific_energy_height_m(state: np.ndarray) -> float:
    return float(state[2] + _speed_m_s(state) ** 2 / (2.0 * 9.81))


def _speed_m_s(state: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(state, dtype=float)[6:9]))


def _state_text(state: np.ndarray) -> str:
    return ";".join(f"{float(value):.12g}" for value in np.asarray(state, dtype=float).reshape(15))


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
