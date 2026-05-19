from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from arena_contract import TRUE_SAFE_BOUNDS, inside_bounds, position_margin_m
from command_contract import clip_normalised_command, normalised_command_to_surface_rad
from flight_dynamics import adapt_glider
from glider import build_nausicaa_glider
from primitive_library_generators import generate_command_profile, primitive_candidate_inventory
from primitive_library_governor import GovernorDecisionCase, evaluate_governor_case
from primitive_library_schema import PrimitiveLibraryConfig
from rollout import rk4_step
from updraft_models import load_updraft_model


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Constants and Data Containers
# 2) Source Loading and Scenario Construction
# 3) Governor Query and Candidate Selection
# 4) Primitive Replay and Scenario Execution
# 5) Summary Builders
# =============================================================================


# =============================================================================
# 1) Constants and Data Containers
# =============================================================================
CAMPAIGN = "09_primitive_library"
OUTER_LOOP_SCENARIOS = (
    "U1_lift_sector_governed_transit",
    "U4_lift_sector_governed_transit",
    "low_lift_confidence_rejection",
    "clearance_limited_no_go",
)
OUTER_LOOP_INTENT_LABELS = (
    "seek_lift",
    "retain_energy",
    "recover_energy",
    "transit_lift_sector",
    "no_go",
)
OUTER_LOOP_EVENT_LABELS = (
    "primitive_accepted",
    "governor_no_go",
    "max_steps_reached",
    "rollout_safety_or_numerical_failure",
    "low_speed_stop",
    "target_steering_unavailable",
)
ACTIVE_SEED_STATUS = "governor_seed_available"
TARGET_STEERING_ROLE = "target_steering"
MIN_SAFE_SPEED_M_S = 3.0
GRAVITY_M_S2 = 9.81


@dataclass(frozen=True)
class OuterLoopScenario:
    scenario_id: str
    updraft_config: str
    wind_fidelity_request: str
    lift_confidence: float
    initial_position_w_m: tuple[float, float, float]
    speed_m_s: float
    max_steps: int
    intent_label: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OuterLoopState:
    scenario_id: str
    step_index: int
    time_s: float
    x: np.ndarray
    lift_confidence: float
    updraft_config: str
    wind_fidelity_request: str


@dataclass(frozen=True)
class OuterLoopCandidateDecision:
    case_id: str
    scenario_id: str
    step_index: int
    source_primitive_id: str
    w3_role: str
    family: str
    outer_loop_intent: str
    candidate_score: float
    governor_decision_status: str
    accepted: bool
    primary_rejection_reason: str
    all_rejection_reasons: str
    clearance_min_margin_m: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OuterLoopStepResult:
    scenario_id: str
    step_index: int
    time_s: float
    selected_primitive_id: str
    selected_role: str
    selected_family: str
    outer_loop_intent: str
    governor_decision_status: str
    mission_event_label: str
    x_w: float
    y_w: float
    z_w: float
    speed_m_s: float
    specific_energy_height_m: float
    energy_residual_from_start_m: float
    wind_query_region: str
    lift_confidence: float
    available_clearance_x_plus_m: float
    available_clearance_y_plus_m: float
    clearance_min_margin_m: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OuterLoopMissionSummary:
    scenario_id: str
    steps_attempted: int
    steps_accepted: int
    mission_duration_s: float
    mission_success_label: str
    mission_stop_reason: str
    accepted_primitive_sequence: str
    unique_primitives_used: int
    energy_initial_m: float
    energy_final_m: float
    energy_delta_m: float
    min_true_margin_m: float
    min_speed_m_s: float
    lift_dwell_time_s: float
    lift_dwell_fraction: float
    governor_accept_count: int
    governor_reject_count: int
    no_go_count: int
    target_steering_used: bool
    higher_target_requested: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OuterLoopConfig:
    dt_s: float = 0.02
    max_steps: int = 8
    actuator_tau_s: tuple[float, float, float] = (0.06, 0.06, 0.06)
    write_step_logs: str = "selected"


# =============================================================================
# 2) Source Loading and Scenario Construction
# =============================================================================
def load_outer_loop_sources(result_root: Path, governor_run_id: int = 5) -> dict[str, object]:
    """Load run-005 as the only source of governor candidate availability."""

    root = Path(result_root)
    suffix = f"s{governor_run_id:03d}"
    run_root = root / f"{governor_run_id:03d}"
    manifest_path = run_root / "manifests" / f"governor_seed_manifest_{suffix}.json"
    seed_path = run_root / "metrics" / f"governor_seed_candidate_table_{suffix}.csv"
    decision_path = run_root / "metrics" / f"governor_accept_reject_decisions_{suffix}.csv"
    coverage_path = run_root / "metrics" / f"governor_coverage_update_{suffix}.csv"
    missing = [str(path) for path in (manifest_path, seed_path, decision_path, coverage_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing run-{governor_run_id:03d} outer-loop source files: {missing}")

    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    seed_table = pd.read_csv(seed_path)
    governor_decisions = pd.read_csv(decision_path)
    coverage_update = pd.read_csv(coverage_path)
    _validate_source_manifest(manifest)
    _require_columns(
        seed_table,
        {
            "source_primitive_id",
            "governor_seed_candidate",
            "seed_table_status",
            "w3_role",
            "family",
            "updraft_config",
            "wind_fidelity",
            "start_condition",
            "candidate_w3_status",
            "candidate_w3_recommendation",
            "requires_lift_belief",
            "requires_wind_fidelity",
            "required_clearance_x_plus_m",
            "required_clearance_x_minus_m",
            "required_clearance_y_plus_m",
            "required_clearance_y_minus_m",
            "required_floor_margin_m",
            "required_ceiling_margin_m",
            "source_wind_query_region",
        },
        "run-005 seed table",
    )
    _require_columns(
        coverage_update,
        {"source_primitive_id", "governor_coverage_status_s005"},
        "run-005 coverage update",
    )
    merged = seed_table.merge(
        coverage_update[["source_primitive_id", "governor_coverage_status_s005"]],
        on="source_primitive_id",
        how="left",
        suffixes=("", "_coverage"),
    )
    active = merged[
        merged["governor_seed_candidate"].astype(bool)
        & (merged["governor_coverage_status_s005"].astype(str) == ACTIVE_SEED_STATUS)
    ].copy()
    if len(active) != 4:
        raise ValueError(f"expected exactly four active governor seeds, found {len(active)}.")
    target = merged[merged["w3_role"].astype(str) == TARGET_STEERING_ROLE].copy()
    return {
        "manifest": manifest,
        "seed_table": merged,
        "governor_decisions": governor_decisions,
        "coverage_update": coverage_update,
        "active_seed_table": active,
        "target_steering_table": target,
    }


def build_outer_loop_scenarios(max_steps: int = 8) -> tuple[OuterLoopScenario, ...]:
    """Return compact deterministic scenarios for the run-006 evidence pass."""

    return (
        OuterLoopScenario(
            "U1_lift_sector_governed_transit",
            "U1_single_fan",
            "W2",
            0.90,
            (1.30, 2.20, 1.80),
            6.50,
            int(max_steps),
            "transit_lift_sector",
        ),
        OuterLoopScenario(
            "U4_lift_sector_governed_transit",
            "U4_four_fan",
            "candidate",
            0.90,
            (1.30, 2.20, 1.80),
            6.50,
            int(max_steps),
            "seek_lift",
        ),
        OuterLoopScenario(
            "low_lift_confidence_rejection",
            "U1_single_fan",
            "W2",
            0.20,
            (1.30, 2.20, 1.80),
            6.50,
            int(max_steps),
            "seek_lift",
        ),
        OuterLoopScenario(
            "clearance_limited_no_go",
            "U1_single_fan",
            "W2",
            0.90,
            (6.45, 2.20, 1.80),
            6.50,
            int(max_steps),
            "transit_lift_sector",
        ),
    )


def build_outer_loop_initial_state(scenario: OuterLoopScenario) -> OuterLoopState:
    """Return the canonical 15-state mission start in public z-up coordinates."""

    x = np.zeros(15, dtype=float)
    x[0:3] = np.asarray(scenario.initial_position_w_m, dtype=float)
    x[6] = float(scenario.speed_m_s)
    return OuterLoopState(
        scenario_id=scenario.scenario_id,
        step_index=0,
        time_s=0.0,
        x=x,
        lift_confidence=float(scenario.lift_confidence),
        updraft_config=scenario.updraft_config,
        wind_fidelity_request=scenario.wind_fidelity_request,
    )


# =============================================================================
# 3) Governor Query and Candidate Selection
# =============================================================================
def build_candidate_cases_for_state(
    state: OuterLoopState,
    seed_table: pd.DataFrame,
    scenario: OuterLoopScenario,
) -> pd.DataFrame:
    """Build governor-compatible query cases from current state clearances."""

    margins = position_margin_m(state.x[0:3], TRUE_SAFE_BOUNDS)
    rows: list[dict[str, object]] = []
    for _, seed in seed_table.iterrows():
        wind_request = _requested_wind_fidelity(seed, scenario)
        case = GovernorDecisionCase(
            case_id=f"{scenario.scenario_id}__step{state.step_index:03d}__{seed['source_primitive_id']}",
            case_kind="outer_loop_query_case",
            source_primitive_id=str(seed["source_primitive_id"]),
            w3_role=str(seed["w3_role"]),
            family=str(seed["family"]),
            updraft_config_requested=str(scenario.updraft_config),
            wind_fidelity_requested=wind_request,
            start_condition_label=str(seed["start_condition"]),
            x_w=float(state.x[0]),
            y_w=float(state.x[1]),
            z_w=float(state.x[2]),
            speed_m_s=_speed_m_s(state.x),
            available_clearance_x_plus_m=float(margins["x_max_margin_m"]),
            available_clearance_x_minus_m=float(margins["x_min_margin_m"]),
            available_clearance_y_plus_m=float(margins["y_max_margin_m"]),
            available_clearance_y_minus_m=float(margins["y_min_margin_m"]),
            available_floor_margin_m=float(margins["floor_margin_m"]),
            available_ceiling_margin_m=float(margins["ceiling_margin_m"]),
            required_clearance_x_plus_m=float(seed["required_clearance_x_plus_m"]),
            required_clearance_x_minus_m=float(seed["required_clearance_x_minus_m"]),
            required_clearance_y_plus_m=float(seed["required_clearance_y_plus_m"]),
            required_clearance_y_minus_m=float(seed["required_clearance_y_minus_m"]),
            required_floor_margin_m=float(seed["required_floor_margin_m"]),
            required_ceiling_margin_m=float(seed["required_ceiling_margin_m"]),
            lift_belief_available=bool(float(scenario.lift_confidence) > 0.0),
            lift_confidence=float(scenario.lift_confidence),
            wind_query_region=str(seed["source_wind_query_region"]),
            recovery_fallback_available=True,
            no_go_flag=False,
            expected_governor_decision_status="outer_loop_query",
        )
        rows.append(case.as_dict())
    return pd.DataFrame(rows)


def score_outer_loop_candidates(
    candidate_cases: pd.DataFrame,
    governor_decisions: pd.DataFrame,
    seed_table: pd.DataFrame,
    scenario: OuterLoopScenario,
) -> pd.DataFrame:
    """Evaluate candidates with the existing governor and add a rule-based score."""

    _ = governor_decisions  # Source decisions are loaded for provenance; live cases are re-evaluated.
    seed_by_id = {str(row["source_primitive_id"]): row for _, row in seed_table.iterrows()}
    rows: list[dict[str, object]] = []
    for _, case_row in candidate_cases.iterrows():
        decision = evaluate_governor_case(case_row.to_dict(), seed_table).as_dict()
        seed = seed_by_id[str(case_row["source_primitive_id"])]
        score = _candidate_score(case_row.to_dict(), seed.to_dict(), decision, scenario)
        rows.append(
            {
                **case_row.to_dict(),
                **decision,
                "scenario_id": scenario.scenario_id,
                "step_index": int(str(case_row["case_id"]).split("__step")[1][:3]),
                "outer_loop_intent": _intent_for_case(case_row.to_dict(), seed.to_dict(), scenario),
                "candidate_score": score,
            }
        )
    return pd.DataFrame(rows)


def select_governor_approved_candidate(candidate_scores: pd.DataFrame) -> dict[str, object] | None:
    """Return the highest-scoring accepted candidate, or None for a no-go step."""

    accepted = candidate_scores[candidate_scores["accepted"].astype(bool)].copy()
    if accepted.empty:
        return None
    accepted = accepted.sort_values(
        ["candidate_score", "clearance_min_margin_m", "source_primitive_id"],
        ascending=[False, False, True],
    )
    return accepted.iloc[0].to_dict()


# =============================================================================
# 4) Primitive Replay and Scenario Execution
# =============================================================================
def rollout_selected_primitive(
    state: OuterLoopState,
    selected_candidate: dict[str, object],
    scenario: OuterLoopScenario,
    config: OuterLoopConfig,
) -> dict[str, object]:
    """Replay the selected primitive through the existing plant command bridge."""

    spec = _reconstruct_spec(str(selected_candidate["source_primitive_id"]))
    time_s = _time_vector(spec.horizon_s, config.dt_s)
    u_req, phase = generate_command_profile(spec, time_s)
    u_applied = np.array([clip_normalised_command(row) for row in u_req], dtype=float)
    delta_cmd = np.array([normalised_command_to_surface_rad(row) for row in u_applied], dtype=float)
    x_log = np.empty((time_s.size, 15), dtype=float)
    x_log[0] = np.asarray(state.x, dtype=float).reshape(15)
    wind_model = _load_wind_for_seed(spec.updraft_config)
    wind_mode = _wind_mode(spec.wind_fidelity)
    aircraft = adapt_glider(build_nausicaa_glider())
    final_index = time_s.size - 1
    for index in range(time_s.size - 1):
        x_next = rk4_step(
            x_log[index],
            delta_cmd[index],
            float(config.dt_s),
            aircraft,
            wind_model,
            wind_mode,
            config.actuator_tau_s,
        )
        x_log[index + 1] = x_next
        if not np.all(np.isfinite(x_next)):
            final_index = index + 1
            break
    x_log = x_log[: final_index + 1]
    time_s = time_s[: final_index + 1]
    u_req = u_req[: final_index + 1]
    u_applied = u_applied[: final_index + 1]
    delta_cmd = delta_cmd[: final_index + 1]
    phase = phase[: final_index + 1]
    finite = bool(np.all(np.isfinite(x_log)))
    true_safe = bool(finite and all(inside_bounds(row[0:3], TRUE_SAFE_BOUNDS) for row in x_log))
    return {
        "spec": spec,
        "time_s": time_s,
        "x_ref": x_log,
        "u_norm_requested": u_req,
        "u_norm_applied": u_applied,
        "delta_cmd_rad": delta_cmd,
        "phase_labels": phase,
        "finite": finite,
        "true_safe": true_safe,
        "terminal_state": x_log[-1].copy(),
        "duration_s": float(time_s[-1] if time_s.size else 0.0),
        "wind_mode": wind_mode,
    }


def run_outer_loop_scenario(
    scenario: OuterLoopScenario,
    sources: dict[str, object],
    config: OuterLoopConfig,
) -> dict[str, pd.DataFrame]:
    """Run one deterministic mission scenario with governor-mediated selection."""

    active_seed_table = sources["active_seed_table"]
    state = build_outer_loop_initial_state(scenario)
    energy_initial = _specific_energy_height_m(state.x)
    step_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    energy_rows: list[dict[str, object]] = [_energy_row(state, energy_initial, "initial")]
    selected_logs: list[dict[str, object]] = []
    stop_reason = "max_steps_reached"

    for step_index in range(min(int(scenario.max_steps), int(config.max_steps))):
        state = OuterLoopState(
            scenario.scenario_id,
            step_index,
            state.time_s,
            state.x,
            state.lift_confidence,
            state.updraft_config,
            state.wind_fidelity_request,
        )
        cases = build_candidate_cases_for_state(state, active_seed_table, scenario)
        scored = score_outer_loop_candidates(cases, sources["governor_decisions"], active_seed_table, scenario)
        candidate_rows.extend(scored.to_dict("records"))
        selected = select_governor_approved_candidate(scored)
        if selected is None:
            step_rows.append(_step_row(state, scenario, energy_initial, None, "governor_no_go", scored))
            stop_reason = _no_go_stop_reason(scored)
            break

        rollout = rollout_selected_primitive(state, selected, scenario, config)
        step_rows.append(_step_row(state, scenario, energy_initial, selected, "primitive_accepted", scored))
        selected_logs.append(_selected_log_row(scenario, step_index, selected, rollout))
        terminal = np.asarray(rollout["terminal_state"], dtype=float)
        state = OuterLoopState(
            scenario.scenario_id,
            step_index + 1,
            state.time_s + float(rollout["duration_s"]),
            terminal,
            state.lift_confidence,
            state.updraft_config,
            state.wind_fidelity_request,
        )
        energy_rows.append(_energy_row(state, energy_initial, "post_primitive"))
        if not bool(rollout["finite"]) or not bool(rollout["true_safe"]):
            stop_reason = "rollout_safety_or_numerical_failure"
            break
        if _speed_m_s(state.x) < MIN_SAFE_SPEED_M_S:
            stop_reason = "low_speed_stop"
            break
    else:
        stop_reason = "max_steps_reached"

    return {
        "step_log": pd.DataFrame(step_rows),
        "candidate_log": pd.DataFrame(candidate_rows),
        "energy_trace": pd.DataFrame(energy_rows),
        "selected_logs": pd.DataFrame(selected_logs),
        "mission_stop_reason": pd.DataFrame([{"scenario_id": scenario.scenario_id, "mission_stop_reason": stop_reason}]),
    }


def run_outer_loop_missions(
    scenarios: tuple[OuterLoopScenario, ...],
    sources: dict[str, object],
    config: OuterLoopConfig,
) -> dict[str, pd.DataFrame]:
    """Run all requested scenarios and return combined evidence tables."""

    scenario_results = [run_outer_loop_scenario(scenario, sources, config) for scenario in scenarios]
    step_log = _concat([result["step_log"] for result in scenario_results])
    candidate_log = _concat([result["candidate_log"] for result in scenario_results])
    energy_trace = _concat([result["energy_trace"] for result in scenario_results])
    selected_logs = _concat([result["selected_logs"] for result in scenario_results])
    stop_reasons = _concat([result["mission_stop_reason"] for result in scenario_results])
    summary = build_outer_loop_summary(step_log, candidate_log, energy_trace, stop_reasons)
    lift = build_outer_loop_lift_dwell_summary(energy_trace, step_log)
    gaps = build_outer_loop_coverage_gap_summary(step_log, candidate_log, sources["seed_table"])
    rejections = candidate_log[~candidate_log["accepted"].astype(bool)].copy() if not candidate_log.empty else pd.DataFrame()
    return {
        "step_log": step_log,
        "candidate_log": candidate_log,
        "governor_rejection_log": rejections,
        "energy_trace": energy_trace,
        "selected_logs": selected_logs,
        "mission_summary": summary,
        "lift_dwell_summary": lift,
        "coverage_gap_summary": gaps,
    }


# =============================================================================
# 5) Summary Builders
# =============================================================================
def build_outer_loop_summary(
    step_log: pd.DataFrame,
    candidate_log: pd.DataFrame,
    energy_trace: pd.DataFrame,
    stop_reasons: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one mission summary row per scenario."""

    rows: list[dict[str, object]] = []
    stop_by_scenario = {} if stop_reasons is None else {
        str(row["scenario_id"]): str(row["mission_stop_reason"])
        for _, row in stop_reasons.iterrows()
    }
    for scenario_id, group in energy_trace.groupby("scenario_id", sort=False):
        steps = step_log[step_log["scenario_id"].astype(str) == str(scenario_id)]
        candidates = candidate_log[candidate_log["scenario_id"].astype(str) == str(scenario_id)]
        accepted_steps = steps[steps["mission_event_label"] == "primitive_accepted"]
        accepted_sequence = [str(value) for value in accepted_steps["selected_primitive_id"].tolist() if str(value) != "none"]
        energy_initial = float(group["specific_energy_height_m"].iloc[0])
        energy_final = float(group["specific_energy_height_m"].iloc[-1])
        min_margin = float(group["true_margin_m"].min())
        min_speed = float(group["speed_m_s"].min())
        lift_count = int(group["lift_region"].astype(bool).sum())
        sample_dt = _median_dt(group)
        duration = float(group["time_s"].iloc[-1] - group["time_s"].iloc[0])
        accept_count = int(candidates["accepted"].astype(bool).sum()) if not candidates.empty else 0
        reject_count = int((~candidates["accepted"].astype(bool)).sum()) if not candidates.empty else 0
        stop_reason = stop_by_scenario.get(str(scenario_id), "max_steps_reached")
        rows.append(
            OuterLoopMissionSummary(
                scenario_id=str(scenario_id),
                steps_attempted=int(len(steps)),
                steps_accepted=int(len(accepted_steps)),
                mission_duration_s=duration,
                mission_success_label="completed_with_governor_evidence" if len(accepted_steps) > 0 else "no_go_coverage_gap",
                mission_stop_reason=stop_reason,
                accepted_primitive_sequence=";".join(accepted_sequence) if accepted_sequence else "none",
                unique_primitives_used=len(set(accepted_sequence)),
                energy_initial_m=energy_initial,
                energy_final_m=energy_final,
                energy_delta_m=energy_final - energy_initial,
                min_true_margin_m=min_margin,
                min_speed_m_s=min_speed,
                lift_dwell_time_s=float(lift_count * sample_dt),
                lift_dwell_fraction=float(lift_count / max(len(group), 1)),
                governor_accept_count=accept_count,
                governor_reject_count=reject_count,
                no_go_count=int((steps["mission_event_label"] == "governor_no_go").sum()) if not steps.empty else 0,
                target_steering_used=False,
                higher_target_requested=False,
            ).as_dict()
        )
    return pd.DataFrame(rows)


def build_outer_loop_lift_dwell_summary(energy_trace: pd.DataFrame, step_log: pd.DataFrame) -> pd.DataFrame:
    """Summarise deterministic lift-sector dwell and energy evidence."""

    rows: list[dict[str, object]] = []
    for scenario_id, group in energy_trace.groupby("scenario_id", sort=False):
        lift = group[group["lift_region"].astype(bool)]
        sample_dt = _median_dt(group)
        rows.append(
            {
                "scenario_id": scenario_id,
                "updraft_config": str(group["updraft_config"].iloc[0]),
                "wind_fidelity": str(group["wind_fidelity"].iloc[0]),
                "lift_region_time_s": float(len(lift) * sample_dt),
                "lift_region_fraction": float(len(lift) / max(len(group), 1)),
                "mean_lift_confidence": float(group["lift_confidence"].mean()),
                "mean_energy_residual_m": float(group["energy_residual_from_start_m"].mean()),
                "energy_gain_event_count": int((group["energy_residual_from_start_m"].diff().fillna(0.0) > 0.0).sum()),
                "accepted_step_count": int((step_log[step_log["scenario_id"] == scenario_id]["mission_event_label"] == "primitive_accepted").sum()),
            }
        )
    return pd.DataFrame(rows)


def build_outer_loop_coverage_gap_summary(
    step_log: pd.DataFrame,
    candidate_log: pd.DataFrame,
    seed_table: pd.DataFrame,
) -> pd.DataFrame:
    """Classify no-go reasons without requesting higher targets automatically."""

    _ = seed_table
    rows: list[dict[str, object]] = []
    for scenario_id in sorted(set(step_log["scenario_id"].astype(str)) | set(candidate_log["scenario_id"].astype(str))):
        steps = step_log[step_log["scenario_id"].astype(str) == scenario_id]
        candidates = candidate_log[candidate_log["scenario_id"].astype(str) == scenario_id]
        statuses = set(candidates["governor_decision_status"].astype(str)) if not candidates.empty else set()
        accepted = bool(not candidates.empty and candidates["accepted"].astype(bool).any())
        if not accepted and "rejected_lift_belief" in statuses:
            action = "improve_lift_belief_or_recovery_policy"
            request = "not_requested_lift_belief_limited"
            gap_type = "lift_belief_limited"
        elif not accepted and "rejected_clearance" in statuses:
            action = "entry_envelope_or_start_state_restriction"
            request = "not_requested_clearance_limited"
            gap_type = "clearance_limited"
        elif not accepted:
            action = "refine_15_target_steering_before_higher_targets"
            request = "not_requested_refine_15_first"
            gap_type = "target_steering_unavailable"
        else:
            action = "proceed_to_ablation"
            request = "not_requested_current_library_sufficient_for_test"
            gap_type = "short_mission_supported_without_target_steering"
        rows.append(
            {
                "scenario_id": scenario_id,
                "coverage_gap_type": gap_type,
                "gap_reason": _gap_reason(steps, statuses, accepted),
                "blocked_by_target_steering_unavailable": bool(gap_type == "target_steering_unavailable"),
                "blocked_by_clearance": bool("rejected_clearance" in statuses and not accepted),
                "blocked_by_lift_confidence": bool("rejected_lift_belief" in statuses and not accepted),
                "blocked_by_wind_fidelity": bool("rejected_wind_fidelity" in statuses and not accepted),
                "blocked_by_recovery": bool("rejected_recovery_class" in statuses and not accepted),
                "recommended_next_action": action,
                "higher_target_request_status": request,
            }
        )
    return pd.DataFrame(rows)


def _validate_source_manifest(manifest: dict[str, object]) -> None:
    checks = {
        "governor_seed_implemented": True,
        "governor_query_implemented": True,
        "target_steering_governor_allowed": False,
        "outer_loop_implemented": False,
    }
    for key, expected in checks.items():
        if manifest.get(key) is not expected:
            raise ValueError(f"run-005 manifest field {key!r} is not {expected!r}.")
    if int(manifest.get("accepted_seed_candidate_count", -1)) != 4:
        raise ValueError("run-005 manifest does not report exactly four accepted governor seeds.")


def _require_columns(df: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns.difference(df.columns))
    if missing:
        raise ValueError(f"malformed {label}; missing columns: {missing}")


def _requested_wind_fidelity(seed: pd.Series, scenario: OuterLoopScenario) -> str:
    if scenario.wind_fidelity_request == "candidate":
        return str(seed["wind_fidelity"])
    return scenario.wind_fidelity_request


def _candidate_score(
    case: dict[str, object],
    seed: dict[str, object],
    decision: dict[str, object],
    scenario: OuterLoopScenario,
) -> float:
    role = str(seed["w3_role"])
    family = str(seed["family"])
    speed = float(case["speed_m_s"])
    role_weight = {
        "recovery_fallback": 4.0 if speed < 5.0 else 1.0,
        "mild_bank_updraft_encounter": 3.0,
        "glide_transit": 2.0,
        "environment_comparison": 1.0,
    }.get(role, 0.5)
    wind_match = 1.0 if str(seed["updraft_config"]) == scenario.updraft_config else -3.0
    lift_weight = float(scenario.lift_confidence) if bool(seed["requires_lift_belief"]) else 0.25
    clearance = float(decision.get("clearance_min_margin_m", -10.0))
    path_penalty = 0.3 if family == "recovery" and speed >= 5.0 else 0.0
    if not bool(decision["accepted"]):
        return -100.0 + clearance
    return role_weight + wind_match + lift_weight + 0.2 * clearance - path_penalty


def _intent_for_case(case: dict[str, object], seed: dict[str, object], scenario: OuterLoopScenario) -> str:
    _ = case
    if str(seed["w3_role"]) == "recovery_fallback":
        return "recover_energy"
    if scenario.lift_confidence >= 0.70 and str(seed["updraft_config"]) != "none":
        return "seek_lift"
    return scenario.intent_label


def _reconstruct_spec(source_primitive_id: str):
    config = PrimitiveLibraryConfig()
    inventory = {spec.primitive_id: spec for spec in primitive_candidate_inventory(config)}
    if source_primitive_id not in inventory:
        raise ValueError(f"outer-loop selected candidate cannot be reconstructed exactly: {source_primitive_id}")
    return inventory[source_primitive_id]


@lru_cache(maxsize=3)
def _load_wind_for_seed(updraft_config: str) -> object | None:
    if updraft_config == "none":
        return None
    model_name = {
        "U1_single_fan": "single_gaussian_var",
        "U4_four_fan": "four_gaussian_var",
    }.get(updraft_config)
    if model_name is None:
        raise ValueError(f"unsupported outer-loop updraft_config: {updraft_config}")
    return load_updraft_model(model_name)


def _wind_mode(wind_fidelity: str) -> str:
    return {"W0": "none", "W1": "cg", "W2": "panel"}.get(wind_fidelity, "none")


def _time_vector(horizon_s: float, dt_s: float) -> np.ndarray:
    step_count = int(round(float(horizon_s) / float(dt_s)))
    return np.arange(step_count + 1, dtype=float) * float(dt_s)


def _step_row(
    state: OuterLoopState,
    scenario: OuterLoopScenario,
    energy_initial: float,
    selected: dict[str, object] | None,
    event_label: str,
    scored: pd.DataFrame,
) -> dict[str, object]:
    margins = position_margin_m(state.x[0:3], TRUE_SAFE_BOUNDS)
    speed = _speed_m_s(state.x)
    if selected is None:
        selected_id = "none"
        selected_role = "none"
        selected_family = "none"
        status = "none"
        clearance = float(scored["clearance_min_margin_m"].max()) if not scored.empty else float("nan")
    else:
        selected_id = str(selected["source_primitive_id"])
        selected_role = str(selected["w3_role"])
        selected_family = str(selected["family"])
        status = str(selected["governor_decision_status"])
        clearance = float(selected["clearance_min_margin_m"])
    return OuterLoopStepResult(
        scenario_id=scenario.scenario_id,
        step_index=int(state.step_index),
        time_s=float(state.time_s),
        selected_primitive_id=selected_id,
        selected_role=selected_role,
        selected_family=selected_family,
        outer_loop_intent=scenario.intent_label,
        governor_decision_status=status,
        mission_event_label=event_label,
        x_w=float(state.x[0]),
        y_w=float(state.x[1]),
        z_w=float(state.x[2]),
        speed_m_s=speed,
        specific_energy_height_m=_specific_energy_height_m(state.x),
        energy_residual_from_start_m=_specific_energy_height_m(state.x) - energy_initial,
        wind_query_region="measured",
        lift_confidence=float(state.lift_confidence),
        available_clearance_x_plus_m=float(margins["x_max_margin_m"]),
        available_clearance_y_plus_m=float(margins["y_max_margin_m"]),
        clearance_min_margin_m=clearance,
    ).as_dict()


def _energy_row(state: OuterLoopState, energy_initial: float, event_label: str) -> dict[str, object]:
    energy = _specific_energy_height_m(state.x)
    margins = position_margin_m(state.x[0:3], TRUE_SAFE_BOUNDS)
    return {
        "scenario_id": state.scenario_id,
        "step_index": int(state.step_index),
        "time_s": float(state.time_s),
        "event_label": event_label,
        "x_w": float(state.x[0]),
        "y_w": float(state.x[1]),
        "z_w": float(state.x[2]),
        "speed_m_s": _speed_m_s(state.x),
        "specific_energy_height_m": energy,
        "energy_residual_from_start_m": energy - energy_initial,
        "true_margin_m": float(margins["min_margin_m"]),
        "updraft_config": state.updraft_config,
        "wind_fidelity": state.wind_fidelity_request,
        "lift_confidence": float(state.lift_confidence),
        "lift_region": bool(state.updraft_config != "none" and float(state.lift_confidence) >= 0.70),
    }


def _selected_log_row(
    scenario: OuterLoopScenario,
    step_index: int,
    selected: dict[str, object],
    rollout: dict[str, object],
) -> dict[str, object]:
    return {
        "scenario_id": scenario.scenario_id,
        "step_index": int(step_index),
        "source_primitive_id": selected["source_primitive_id"],
        "sample_count": int(len(rollout["time_s"])),
        "duration_s": float(rollout["duration_s"]),
        "finite": bool(rollout["finite"]),
        "true_safe": bool(rollout["true_safe"]),
        "command_bridge": "u_norm_requested -> u_norm_applied -> delta_cmd_rad -> rk4_step/state_derivative",
    }


def _no_go_stop_reason(scored: pd.DataFrame) -> str:
    if scored.empty:
        return "no_candidate_available"
    statuses = set(scored["governor_decision_status"].astype(str))
    if "rejected_clearance" in statuses:
        return "no_candidate_accepted_by_governor_clearance"
    if "rejected_lift_belief" in statuses:
        return "no_candidate_accepted_by_governor_lift_belief"
    if "rejected_wind_fidelity" in statuses:
        return "no_candidate_accepted_by_governor_wind_fidelity"
    return "no_candidate_accepted_by_governor"


def _gap_reason(steps: pd.DataFrame, statuses: set[str], accepted: bool) -> str:
    if accepted:
        return "accepted_baseline_or_updraft_support_primitive"
    if "rejected_clearance" in statuses:
        return "all_candidate_queries_rejected_for_clearance"
    if "rejected_lift_belief" in statuses:
        return "all_candidate_queries_rejected_for_lift_belief"
    if steps.empty:
        return "no_step_log_available"
    return "no_go_without_governor_approved_target_steering"


def _speed_m_s(x: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=float).reshape(15)[6:9]))


def _specific_energy_height_m(x: np.ndarray) -> float:
    state = np.asarray(x, dtype=float).reshape(15)
    return float(state[2] + _speed_m_s(state) ** 2 / (2.0 * GRAVITY_M_S2))


def _median_dt(group: pd.DataFrame) -> float:
    if len(group) < 2:
        return 0.0
    dt = np.diff(np.asarray(group["time_s"], dtype=float))
    positive = dt[dt > 0.0]
    return float(np.median(positive)) if positive.size else 0.0


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if not frame.empty]
    return pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()
