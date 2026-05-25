from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def _bootstrap_import_paths() -> None:
    control_root = Path(__file__).resolve().parents[1]
    for rel in ("03_Primitives", "04_Scenarios"):
        path = control_root / rel
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


_bootstrap_import_paths()

from dense_archive_runtime import MAX_GENERATED_FILE_SIZE_MB  # noqa: E402
from dense_archive_table_io import filesystem_path  # noqa: E402
from env_ctx import build_environment_context  # noqa: E402
from env_instance import environment_instance_for_mode, environment_metadata_from_instance  # noqa: E402
from env_surrogate import resolve_surrogate_binding, wind_field_for_binding  # noqa: E402
from episode_selector import select_compact_representative, selector_decision_row  # noqa: E402
from episodic_lift_belief import (  # noqa: E402
    BELIEF_LAMBDA_VALUES,
    LiftObservation,
    belief_snapshot_row,
    initial_belief,
    query_belief_features,
    update_belief,
)
from frozen_w01_controller_bundle import FROZEN_CONTROLLER_READY, load_frozen_w01_controller_bundle  # noqa: E402
from implementation_instance import implementation_instance_for_layer  # noqa: E402
from plant_instance import plant_instance_for_layer  # noqa: E402
from prim_cat import primitive_by_id  # noqa: E402
from prim_roll import RolloutConfig, rollout_evidence_row, simulate_primitive_rollout  # noqa: E402
from run_outcome_model_build import (  # noqa: E402
    DEFAULT_COMPACT_LIBRARY,
    DEFAULT_OUTPUT_ROOT as DEFAULT_OUTCOME_ROOT,
    OutcomeModelBuildConfig,
    run_outcome_model_build,
)
from run_v48_source_audit import (  # noqa: E402
    BLOCKED_CLAIMS,
    DEFAULT_GOVERNOR_SMOKE_ROOT,
    DEFAULT_OUTCOME_SMOKE_ROOT,
    DEFAULT_POST_W3_ROOT,
    DEFAULT_W01_ROOT,
    DEFAULT_W2_ROOT,
    DEFAULT_W3_ROOT,
    V48SourceAuditConfig,
    run_v48_source_audit,
)
from state_contract import as_state_vector  # noqa: E402
from state_sampling import archive_state_sample_for_family  # noqa: E402


PROJECT_TITLE_VERSION = "LQR-Stabilised Contextual Primitive v4.8"
FULL_LOOP_VERSION = "v48_full_loop_validation_and_memory_v1"
DEFAULT_OUTPUT_ROOT = Path("03_Control/05_Results/lqr_contextual_v1_0/full_loop_validation")
ENVIRONMENT_CASES = (
    ("W0", "dry_air"),
    ("W1", "gaussian_single"),
    ("W1", "gaussian_four"),
    ("W2", "annular_gp_single"),
    ("W2", "annular_gp_four"),
    ("W3", "w3_randomised_single"),
    ("W3", "w3_randomised_four"),
)
POLICIES = (
    {
        "policy_id": "no_memory_baseline",
        "lambda_value": 0.0,
        "uses_memory_features": False,
        "updates_belief": False,
        "policy_role": "baseline_no_memory",
    },
    {
        "policy_id": "static_map_baseline",
        "lambda_value": 0.8,
        "uses_memory_features": True,
        "updates_belief": False,
        "policy_role": "static_initial_map_no_episode_update",
    },
    {
        "policy_id": "context_only_without_memory",
        "lambda_value": 0.0,
        "uses_memory_features": False,
        "updates_belief": False,
        "policy_role": "context_outcome_model_no_memory",
    },
    {
        "policy_id": "context_plus_memory_lambda_0_5",
        "lambda_value": 0.5,
        "uses_memory_features": True,
        "updates_belief": True,
        "policy_role": "context_plus_episodic_memory",
    },
    {
        "policy_id": "context_plus_memory_lambda_0_8",
        "lambda_value": 0.8,
        "uses_memory_features": True,
        "updates_belief": True,
        "policy_role": "context_plus_episodic_memory",
    },
    {
        "policy_id": "context_plus_memory_lambda_0_95",
        "lambda_value": 0.95,
        "uses_memory_features": True,
        "updates_belief": True,
        "policy_role": "context_plus_episodic_memory",
    },
)


@dataclass(frozen=True)
class FullLoopValidationConfig:
    run_id: int = 1
    output_root: Path = DEFAULT_OUTPUT_ROOT
    episodes_per_policy: int = 50
    max_primitives_per_episode: int = 4
    seed: int = 48
    workers: int | str = 1
    max_workers: int | None = 1
    storage_format: str = "auto"
    compression_level: int = 1
    resume: bool = True
    repair_incomplete: bool = False
    dry_run_schedule: bool = False
    source_w01_root: Path = DEFAULT_W01_ROOT
    source_w2_root: Path = DEFAULT_W2_ROOT
    source_w3_root: Path = DEFAULT_W3_ROOT
    post_w3_root: Path = DEFAULT_POST_W3_ROOT
    outcome_smoke_root: Path = DEFAULT_OUTCOME_SMOKE_ROOT
    governor_smoke_root: Path = DEFAULT_GOVERNOR_SMOKE_ROOT
    compact_library_path: Path = DEFAULT_COMPACT_LIBRARY
    outcome_model_root: Path = DEFAULT_OUTCOME_ROOT
    outcome_model_run_id: int = 2


def run_full_loop_validation(config: FullLoopValidationConfig) -> dict[str, object]:
    """Run v4.8 simulation-only full-loop validation over the frozen compact library."""

    run_root = Path(config.output_root) / f"{int(config.run_id):03d}"
    for subdir in ("manifests", "metrics", "reports"):
        filesystem_path(run_root / subdir).mkdir(parents=True, exist_ok=True)

    source_result = run_v48_source_audit(
        V48SourceAuditConfig(
            output_root=run_root,
            w01_root=config.source_w01_root,
            w2_root=config.source_w2_root,
            w3_root=config.source_w3_root,
            post_w3_root=config.post_w3_root,
            outcome_smoke_root=config.outcome_smoke_root,
            governor_smoke_root=config.governor_smoke_root,
        )
    )
    if source_result["status"] != "source_audit_pass":
        _write_blocked_outputs(run_root, config, "source_audit_failed", source_result.get("blockers", []))
        return {"status": "blocked", "blocked_reason": "source_audit_failed", "run_root": run_root.as_posix()}

    outcome_path = Path(config.outcome_model_root) / f"{int(config.outcome_model_run_id):03d}" / "metrics" / "outcome_model_table.csv"
    if not filesystem_path(outcome_path).is_file():
        outcome_result = run_outcome_model_build(
            OutcomeModelBuildConfig(
                compact_library_path=config.compact_library_path,
                output_root=config.outcome_model_root,
                run_id=int(config.outcome_model_run_id),
            )
        )
        if outcome_result.get("status") == "blocked":
            _write_blocked_outputs(run_root, config, "outcome_model_build_blocked", [str(outcome_result.get("blocked_reason", ""))])
            return {"status": "blocked", "blocked_reason": "outcome_model_build_blocked", "run_root": run_root.as_posix()}

    library = _read_json(config.compact_library_path)
    representatives = list(library.get("representatives", []))
    outcome_rows = pd.read_csv(filesystem_path(outcome_path)).to_dict(orient="records")
    outcomes_by_variant = {str(row.get("primitive_variant_id", "")): row for row in outcome_rows}
    bundle = load_frozen_w01_controller_bundle(config.source_w2_root / "manifests" / "frozen_w01_controller_bundle.json")
    records_by_variant = {
        record.primitive_variant_id: record
        for record in bundle.records
        if record.bundle_status == FROZEN_CONTROLLER_READY
    }
    missing_records = [
        str(row.get("primitive_variant_id", ""))
        for row in representatives
        if str(row.get("primitive_variant_id", "")) not in records_by_variant
    ]
    if missing_records:
        _write_blocked_outputs(run_root, config, "compact_representative_missing_frozen_record", missing_records[:10])
        return {"status": "blocked", "blocked_reason": "compact_representative_missing_frozen_record", "run_root": run_root.as_posix()}

    policy_rows = _policy_rows()
    episode_schedule = _episode_schedule(config)
    _write_policy_set(run_root, policy_rows)
    if config.dry_run_schedule:
        _write_dry_run_outputs(run_root, config, episode_schedule, policy_rows, library)
        return {"status": "dry_run_schedule", "run_root": run_root.as_posix(), "episode_count": int(len(episode_schedule))}

    episode_rows: list[dict[str, object]] = []
    primitive_rows: list[dict[str, object]] = []
    governor_rows: list[dict[str, object]] = []
    selector_rows: list[dict[str, object]] = []
    belief_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for scheduled in episode_schedule:
        episode_result = _run_episode(
            scheduled=scheduled,
            config=config,
            representatives=representatives,
            outcomes_by_variant=outcomes_by_variant,
            records_by_variant=records_by_variant,
        )
        episode_rows.append(episode_result["episode_row"])
        primitive_rows.extend(episode_result["primitive_rows"])
        governor_rows.extend(episode_result["governor_rows"])
        selector_rows.extend(episode_result["selector_rows"])
        belief_rows.extend(episode_result["belief_rows"])
        prediction_rows.extend(episode_result["prediction_rows"])

    _write_csv(run_root / "metrics" / "episode_summary.csv", pd.DataFrame(episode_rows))
    _write_csv(run_root / "metrics" / "primitive_execution_log.csv", pd.DataFrame(primitive_rows))
    _write_csv(run_root / "metrics" / "governor_rejection_log.csv", pd.DataFrame(governor_rows))
    _write_csv(run_root / "metrics" / "governor_rejection_summary.csv", _governor_rejection_summary(governor_rows))
    _write_csv(run_root / "metrics" / "selector_choice_log.csv", pd.DataFrame(selector_rows))
    _write_csv(run_root / "metrics" / "belief_snapshot_log.csv", pd.DataFrame(belief_rows))
    _write_csv(run_root / "metrics" / "prediction_alignment_summary.csv", _prediction_alignment_summary(prediction_rows))
    _write_csv(run_root / "metrics" / "memory_ablation_summary.csv", _memory_ablation_summary(episode_rows))
    _write_csv(run_root / "metrics" / "termination_summary.csv", _termination_summary(episode_rows))
    _write_csv(run_root / "metrics" / "policy_summary.csv", _policy_summary(episode_rows, governor_rows, prediction_rows))
    _write_run_manifests(run_root, config, "complete", library, len(episode_rows), len(primitive_rows), [])
    _write_file_size_audit(run_root)
    _write_reports(run_root, status="complete", episode_rows=episode_rows, primitive_rows=primitive_rows)
    return {
        "status": "complete",
        "run_root": run_root.as_posix(),
        "episode_count": int(len(episode_rows)),
        "primitive_execution_count": int(len(primitive_rows)),
    }


def _run_episode(
    *,
    scheduled: dict[str, object],
    config: FullLoopValidationConfig,
    representatives: list[dict[str, object]],
    outcomes_by_variant: dict[str, dict[str, object]],
    records_by_variant: dict[str, object],
) -> dict[str, object]:
    policy = dict(scheduled["policy"])
    episode_id = str(scheduled["episode_id"])
    layer = str(scheduled["W_layer"])
    mode = str(scheduled["environment_mode"])
    lambda_value = float(policy["lambda_value"])
    belief = initial_belief(lambda_value=lambda_value if lambda_value in BELIEF_LAMBDA_VALUES else 0.0)
    state_sample = archive_state_sample_for_family(
        start_state_family="launch_gate",
        paired_start_key=f"v48_{episode_id}_launch",
        sample_index=int(scheduled["episode_index"]),
        seed=int(config.seed),
        W_layer=layer,
        environment_mode=mode,
    )
    state = as_state_vector(state_sample.state_vector)
    start_family = "launch_gate"
    primitive_rows: list[dict[str, object]] = []
    governor_rows: list[dict[str, object]] = []
    selector_rows: list[dict[str, object]] = []
    belief_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    termination_cause = "max_primitive_count"
    hard_failure = False
    terminal_useful = False
    continuation_valid = False
    total_duration_s = 0.0

    for primitive_step_index in range(int(config.max_primitives_per_episode)):
        context_payload = _context_payload(
            state=state,
            layer=layer,
            mode=mode,
            seed=int(config.seed) + int(scheduled["episode_index"]) * 10 + primitive_step_index,
            start_state_family=start_family,
            episode_id=episode_id,
            primitive_step_index=primitive_step_index,
        )
        belief_features = (
            query_belief_features(state, belief)
            if bool(policy["uses_memory_features"])
            else None
        )
        governor_mode = (
            "terminal_episode_mode"
            if primitive_step_index == int(config.max_primitives_per_episode) - 1
            else "continuation_mode"
        )
        selected, candidate_rows = select_compact_representative(
            representatives=representatives,
            outcome_rows_by_variant_id=outcomes_by_variant,
            context=context_payload["row"],
            governor_mode=governor_mode,
            policy_id=str(policy["policy_id"]),
            belief_features=belief_features,
        )
        if selected is None and governor_mode == "continuation_mode":
            selected, terminal_candidate_rows = select_compact_representative(
                representatives=representatives,
                outcome_rows_by_variant_id=outcomes_by_variant,
                context=context_payload["row"],
                governor_mode="terminal_episode_mode",
                policy_id=str(policy["policy_id"]),
                belief_features=belief_features,
            )
            candidate_rows.extend(terminal_candidate_rows)
            governor_mode = "terminal_episode_mode" if selected is not None else governor_mode
        governor_rows.extend(
            {
                **row,
                "episode_id": episode_id,
                "primitive_step_index": primitive_step_index,
            }
            for row in candidate_rows
        )
        viable_count = sum(1 for row in candidate_rows if bool(row.get("viable", False)))
        selector_rows.append(
            selector_decision_row(
                episode_id=episode_id,
                primitive_step_index=primitive_step_index,
                policy_id=str(policy["policy_id"]),
                governor_mode=governor_mode,
                context=context_payload["row"],
                selected=selected,
                candidate_count=len(candidate_rows),
                viable_count=viable_count,
            )
        )
        belief_rows.append(
            _belief_log_row(
                episode_id=episode_id,
                primitive_step_index=primitive_step_index,
                policy=policy,
                layer=layer,
                mode=mode,
                phase="before",
                belief=belief,
                observation=None,
                update_status="not_yet_updated",
            )
        )
        if selected is None:
            termination_cause = "no_viable_primitive"
            break

        record = records_by_variant[str(selected["primitive_variant_id"])]
        primitive = primitive_by_id(str(selected["primitive_id"]))
        rollout = simulate_primitive_rollout(
            rollout_id=f"{episode_id}_p{primitive_step_index:02d}",
            episode_id=episode_id,
            initial_state=state,
            context=context_payload["context"],
            primitive=primitive,
            config=RolloutConfig(W_layer=layer, rollout_backend="model_backed_lqr"),
            wind_field=context_payload["wind_field"],
            implementation_instance=context_payload["implementation_instance"],
            plant_instance=context_payload["plant_instance"],
            controller=record.controller,
            controller_selection_status="selected_by_v48_full_loop_governor",
            candidate_index=record.candidate_index,
            candidate_weight_label=record.candidate_weight_label,
        )
        rollout_row = rollout_evidence_row(rollout)
        representative = _representative_by_variant(representatives, str(selected["primitive_variant_id"]))
        outcome = outcomes_by_variant[str(selected["primitive_variant_id"])]
        primitive_rows.append(
            {
                "episode_id": episode_id,
                "primitive_step_index": primitive_step_index,
                "policy_id": str(policy["policy_id"]),
                "memory_lambda": lambda_value,
                "governor_mode": governor_mode,
                "selected_score": float(selected["score"]),
                "compact_library_id": str(selected["compact_library_id"]),
                "primitive_variant_id": str(selected["primitive_variant_id"]),
                "source_representative_controller_id": str(representative.get("controller_id", "")),
                "source_representative_K_gain_checksum": str(representative.get("K_gain_checksum", "")),
                "source_representative_augmented_gain_checksum": str(representative.get("augmented_gain_checksum", "")),
                **rollout_row,
            }
        )
        prediction_rows.append(
            _prediction_alignment_row(
                episode_id,
                primitive_step_index,
                policy,
                outcome,
                rollout_row,
                primitive_variant_id=str(selected["primitive_variant_id"]),
            )
        )
        observation = LiftObservation(
            x_w_m=float(state[0]),
            y_w_m=float(state[1]),
            lift_evidence_m_s=float(context_payload["row"].get("w_wing_mean_m_s", 0.0)),
            episode_id=episode_id,
        )
        if bool(policy["updates_belief"]):
            belief = update_belief(belief, observation)
            update_status = "updated"
        else:
            update_status = "not_updated_policy_does_not_use_episode_memory"
        belief_rows.append(
            _belief_log_row(
                episode_id=episode_id,
                primitive_step_index=primitive_step_index,
                policy=policy,
                layer=layer,
                mode=mode,
                phase="after",
                belief=belief,
                observation=observation,
                update_status=update_status,
            )
        )
        continuation_valid = bool(rollout_row["continuation_valid"])
        terminal_useful = bool(rollout_row["episode_terminal_useful"])
        hard_failure = str(rollout_row["boundary_use_class"]) == "hard_failure" or str(rollout_row["outcome_class"]) == "failed"
        total_duration_s += float(primitive.finite_horizon_s)
        state = as_state_vector(np.asarray(json.loads(str(rollout_row["exit_state_vector"])), dtype=float))
        start_family = "inflight_nominal"
        if terminal_useful:
            termination_cause = "terminal_useful"
            break
        if hard_failure:
            termination_cause = "hard_failure"
            break
        if not continuation_valid:
            termination_cause = str(rollout_row["termination_cause"])
            break
    else:
        termination_cause = "max_primitive_count"

    episode_row = {
        "episode_id": episode_id,
        "episode_index": int(scheduled["episode_index"]),
        "policy_id": str(policy["policy_id"]),
        "memory_lambda": lambda_value,
        "W_layer": layer,
        "environment_mode": mode,
        "episode_counted_for_claim": True,
        "primitive_count": int(len(primitive_rows)),
        "episode_duration_s": float(total_duration_s),
        "continuation_valid": bool(continuation_valid),
        "episode_terminal_useful": bool(terminal_useful),
        "hard_failure": bool(hard_failure),
        "termination_cause": termination_cause,
        "no_viable_primitive": bool(termination_cause == "no_viable_primitive"),
        "x_y_terminal": bool(terminal_useful and "terminal" in termination_cause),
        "claim_status": "simulation_only_full_loop_validation_episode",
    }
    if primitive_rows:
        last = primitive_rows[-1]
        episode_row.update(
            {
                "energy_residual_m": float(last.get("energy_residual_m", 0.0)),
                "lift_dwell_time_s": float(last.get("lift_dwell_time_s", 0.0)),
                "minimum_wall_margin_m": float(last.get("minimum_wall_margin_m", 0.0)),
                "floor_margin_m": float(last.get("floor_margin_m", 0.0)),
                "ceiling_margin_m": float(last.get("ceiling_margin_m", 0.0)),
                "minimum_speed_m_s": float(last.get("minimum_speed_m_s", 0.0)),
            }
        )
    else:
        episode_row.update(
            {
                "energy_residual_m": 0.0,
                "lift_dwell_time_s": 0.0,
                "minimum_wall_margin_m": 0.0,
                "floor_margin_m": 0.0,
                "ceiling_margin_m": 0.0,
                "minimum_speed_m_s": 0.0,
            }
        )
    return {
        "episode_row": episode_row,
        "primitive_rows": primitive_rows,
        "governor_rows": governor_rows,
        "selector_rows": selector_rows,
        "belief_rows": belief_rows,
        "prediction_rows": prediction_rows,
    }


def _context_payload(
    *,
    state: np.ndarray,
    layer: str,
    mode: str,
    seed: int,
    start_state_family: str,
    episode_id: str,
    primitive_step_index: int,
) -> dict[str, object]:
    instance = environment_instance_for_mode(layer, mode, seed)
    metadata = environment_metadata_from_instance(instance)
    binding = resolve_surrogate_binding(layer, metadata, randomisation_seed=seed)
    wind_field = wind_field_for_binding(binding)
    latency_case = "none" if str(layer).upper() == "W0" else "nominal"
    context = build_environment_context(
        state,
        wind_field=wind_field,
        metadata=metadata,
        latency_case=latency_case,
        actuator_case="nominal",
        surrogate_binding=binding,
    )
    row = {
        "context_id": f"{episode_id}_ctx{primitive_step_index:02d}",
        "W_layer": layer,
        "environment_mode": mode,
        "start_state_family": start_state_family,
        "latency_case": latency_case,
        "wall_margin_m": float(context.wall_margin_m),
        "floor_margin_m": float(context.floor_margin_m),
        "ceiling_margin_m": float(context.ceiling_margin_m),
        "speed_margin_m_s": float(context.speed_margin_m_s),
        "w_wing_mean_m_s": float(context.w_wing_mean_m_s),
        "fan_count": int(context.fan_count),
        "updraft_model_id": context.updraft_model_id,
    }
    return {
        "context": context,
        "row": row,
        "wind_field": wind_field,
        "implementation_instance": implementation_instance_for_layer(layer, seed, latency_case=latency_case),
        "plant_instance": plant_instance_for_layer(layer, seed),
    }


def _episode_schedule(config: FullLoopValidationConfig) -> list[dict[str, object]]:
    schedule = []
    index = 0
    for policy in POLICIES:
        for local_index in range(int(config.episodes_per_policy)):
            layer, mode = ENVIRONMENT_CASES[(local_index + int(config.seed)) % len(ENVIRONMENT_CASES)]
            schedule.append(
                {
                    "episode_id": f"v48_r{int(config.run_id):03d}_{policy['policy_id']}_e{local_index:04d}",
                    "episode_index": index,
                    "policy": policy,
                    "W_layer": layer,
                    "environment_mode": mode,
                    "seed": int(config.seed) + index,
                }
            )
            index += 1
    return schedule


def _policy_rows() -> list[dict[str, object]]:
    return [
        {
            **policy,
            "claim_status": "simulation_only_memory_policy",
        }
        for policy in POLICIES
    ]


def _belief_log_row(
    *,
    episode_id: str,
    primitive_step_index: int,
    policy: dict[str, object],
    layer: str,
    mode: str,
    phase: str,
    belief,
    observation: LiftObservation | None,
    update_status: str,
) -> dict[str, object]:
    snapshot = belief_snapshot_row(belief, label=f"{phase}_p{primitive_step_index:02d}")
    return {
        "belief_snapshot_id": f"{episode_id}_{phase}_p{primitive_step_index:02d}",
        "episode_id": episode_id,
        "primitive_step_index": int(primitive_step_index),
        "phase": phase,
        "policy_id": str(policy["policy_id"]),
        "lambda_value": float(policy["lambda_value"]),
        "W_layer": layer,
        "environment_mode": mode,
        "belief_before_json": json.dumps(snapshot, sort_keys=True, separators=(",", ":")) if phase == "before" else "",
        "observed_lift_evidence_json": "{}" if observation is None else json.dumps(
            {
                "x_w_m": float(observation.x_w_m),
                "y_w_m": float(observation.y_w_m),
                "lift_evidence_m_s": float(observation.lift_evidence_m_s),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        "belief_after_json": json.dumps(snapshot, sort_keys=True, separators=(",", ":")) if phase == "after" else "",
        "memory_update_status": update_status,
        "belief_values": snapshot["values"],
        "claim_status": "simulation_only_episodic_lift_belief",
    }


def _prediction_alignment_row(
    episode_id: str,
    primitive_step_index: int,
    policy: dict[str, object],
    outcome: dict[str, object],
    rollout_row: dict[str, object],
    *,
    primitive_variant_id: str,
) -> dict[str, object]:
    actual_continuation = bool(rollout_row.get("continuation_valid", False))
    actual_terminal = bool(rollout_row.get("episode_terminal_useful", False))
    actual_hard_failure = str(rollout_row.get("boundary_use_class", "")) == "hard_failure" or str(rollout_row.get("outcome_class", "")) == "failed"
    return {
        "episode_id": episode_id,
        "primitive_step_index": int(primitive_step_index),
        "policy_id": str(policy["policy_id"]),
        "primitive_variant_id": str(primitive_variant_id),
        "predicted_continuation_probability": float(outcome.get("continuation_probability", 0.0)),
        "actual_continuation_valid": actual_continuation,
        "predicted_terminal_useful_probability": float(outcome.get("terminal_useful_probability", 0.0)),
        "actual_episode_terminal_useful": actual_terminal,
        "predicted_hard_failure_risk": float(outcome.get("hard_failure_risk", 1.0)),
        "actual_hard_failure": actual_hard_failure,
        "expected_energy_residual_m": float(outcome.get("expected_energy_residual_m", 0.0)),
        "actual_energy_residual_m": float(rollout_row.get("energy_residual_m", 0.0)),
        "expected_lift_dwell_time_s": float(outcome.get("expected_lift_dwell_time_s", 0.0)),
        "actual_lift_dwell_time_s": float(rollout_row.get("lift_dwell_time_s", 0.0)),
        "prediction_actual_agree_continuation": bool((float(outcome.get("continuation_probability", 0.0)) >= 0.5) == actual_continuation),
        "prediction_actual_agree_terminal": bool((float(outcome.get("terminal_useful_probability", 0.0)) >= 0.5) == actual_terminal),
        "prediction_actual_agree_hard_failure": bool((float(outcome.get("hard_failure_risk", 1.0)) >= 0.5) == actual_hard_failure),
        "claim_status": "simulation_only_prediction_alignment",
    }


def _representative_by_variant(representatives: list[dict[str, object]], variant_id: str) -> dict[str, object]:
    for representative in representatives:
        if str(representative.get("primitive_variant_id", "")) == str(variant_id):
            return representative
    return {}


def _prediction_alignment_summary(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame()
    return (
        frame.groupby("policy_id", dropna=False)
        .agg(
            primitive_execution_count=("episode_id", "count"),
            continuation_agreement_rate=("prediction_actual_agree_continuation", "mean"),
            terminal_agreement_rate=("prediction_actual_agree_terminal", "mean"),
            hard_failure_agreement_rate=("prediction_actual_agree_hard_failure", "mean"),
            mean_energy_residual_actual_m=("actual_energy_residual_m", "mean"),
            mean_lift_dwell_actual_s=("actual_lift_dwell_time_s", "mean"),
        )
        .reset_index()
    )


def _memory_ablation_summary(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame()
    return (
        frame.groupby(["policy_id", "memory_lambda"], dropna=False)
        .agg(
            episode_count=("episode_id", "count"),
            terminal_useful_rate=("episode_terminal_useful", "mean"),
            continuation_valid_rate=("continuation_valid", "mean"),
            hard_failure_rate=("hard_failure", "mean"),
            mean_primitive_count=("primitive_count", "mean"),
            mean_episode_duration_s=("episode_duration_s", "mean"),
            no_viable_primitive_count=("no_viable_primitive", "sum"),
        )
        .reset_index()
    )


def _termination_summary(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame()
    return frame.groupby(["policy_id", "termination_cause"], dropna=False).size().reset_index(name="episode_count")


def _governor_rejection_summary(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame()
    rejected = frame[frame["rejection_reason"].astype(str) != ""]
    if rejected.empty:
        return pd.DataFrame(columns=["policy_id", "governor_mode", "rejection_reason", "governor_rejection_count"])
    return (
        rejected.groupby(["policy_id", "governor_mode", "rejection_reason"], dropna=False)
        .size()
        .reset_index(name="governor_rejection_count")
    )


def _policy_summary(
    episode_rows: list[dict[str, object]],
    governor_rows: list[dict[str, object]],
    prediction_rows: list[dict[str, object]],
) -> pd.DataFrame:
    rows = episode_rows
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame()
    summary = (
        frame.groupby(["policy_id", "memory_lambda", "W_layer", "environment_mode"], dropna=False)
        .agg(
            episode_count=("episode_id", "count"),
            mean_episode_duration_s=("episode_duration_s", "mean"),
            mean_primitive_count_per_episode=("primitive_count", "mean"),
            continuation_valid_rate=("continuation_valid", "mean"),
            terminal_useful_rate=("episode_terminal_useful", "mean"),
            hard_failure_rate=("hard_failure", "mean"),
            mean_energy_residual_m=("energy_residual_m", "mean"),
            mean_lift_dwell_time_s=("lift_dwell_time_s", "mean"),
            mean_min_wall_margin_m=("minimum_wall_margin_m", "mean"),
            floor_or_ceiling_violation_count=("termination_cause", lambda values: int(sum(str(value) in {"floor_violation", "ceiling_violation"} for value in values))),
            low_speed_count=("termination_cause", lambda values: int(sum(str(value) == "speed_low" for value in values))),
            no_viable_primitive_count=("no_viable_primitive", "sum"),
            x_y_terminal_count=("x_y_terminal", "sum"),
        )
        .reset_index()
    )
    gov = pd.DataFrame(governor_rows)
    if not gov.empty:
        gov_counts = (
            gov[gov["rejection_reason"].astype(str) != ""]
            .groupby("policy_id", dropna=False)
            .size()
            .reset_index(name="governor_rejection_count")
        )
        summary = summary.merge(gov_counts, on="policy_id", how="left")
    else:
        summary["governor_rejection_count"] = 0
    pred = pd.DataFrame(prediction_rows)
    if not pred.empty:
        pred = pred.assign(
            prediction_actual_agreement_rate=(
                pred["prediction_actual_agree_continuation"].astype(bool).astype(float)
                + pred["prediction_actual_agree_terminal"].astype(bool).astype(float)
                + pred["prediction_actual_agree_hard_failure"].astype(bool).astype(float)
            )
            / 3.0
        )
        pred_summary = pred.groupby("policy_id", dropna=False)["prediction_actual_agreement_rate"].mean().reset_index()
        summary = summary.merge(pred_summary, on="policy_id", how="left")
    else:
        summary["prediction_actual_agreement_rate"] = 0.0
    summary["governor_rejection_count"] = summary["governor_rejection_count"].fillna(0).astype(int)
    summary["prediction_actual_agreement_rate"] = summary["prediction_actual_agreement_rate"].fillna(0.0)
    return summary


def _write_policy_set(run_root: Path, policy_rows: list[dict[str, object]]) -> None:
    _write_json(
        run_root / "manifests" / "policy_set.json",
        {
            "policy_set_version": "v48_memory_policy_set_v1",
            "policies": policy_rows,
            "claim_status": "simulation_only_memory_ablation_policy_set",
        },
    )
    _write_csv(run_root / "metrics" / "policy_set.csv", pd.DataFrame(policy_rows))


def _write_dry_run_outputs(
    run_root: Path,
    config: FullLoopValidationConfig,
    episode_schedule: list[dict[str, object]],
    policy_rows: list[dict[str, object]],
    library: dict[str, object],
) -> None:
    _write_csv(run_root / "metrics" / "episode_schedule.csv", pd.DataFrame(_schedule_rows(episode_schedule)))
    _write_run_manifests(run_root, config, "dry_run_schedule", library, len(episode_schedule), 0, [])
    _write_file_size_audit(run_root)
    _write_reports(run_root, status="dry_run_schedule", episode_rows=[], primitive_rows=[])


def _schedule_rows(schedule: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for row in schedule:
        policy = dict(row["policy"])
        rows.append(
            {
                "episode_id": row["episode_id"],
                "episode_index": row["episode_index"],
                "policy_id": policy["policy_id"],
                "memory_lambda": policy["lambda_value"],
                "W_layer": row["W_layer"],
                "environment_mode": row["environment_mode"],
                "claim_status": "simulation_only_full_loop_schedule",
            }
        )
    return rows


def _write_blocked_outputs(run_root: Path, config: FullLoopValidationConfig, blocked_reason: str, blockers: list[object]) -> None:
    _write_run_manifests(run_root, config, "blocked", {}, 0, 0, [blocked_reason, *[str(item) for item in blockers if item]])
    _write_csv(run_root / "metrics" / "episode_summary.csv", pd.DataFrame())
    _write_file_size_audit(run_root)
    _write_reports(run_root, status="blocked", episode_rows=[], primitive_rows=[])


def _write_run_manifests(
    run_root: Path,
    config: FullLoopValidationConfig,
    status: str,
    library: dict[str, object],
    episode_count: int,
    primitive_execution_count: int,
    blockers: list[str],
) -> None:
    manifest = {
        "manifest_version": FULL_LOOP_VERSION,
        "project_title_version": PROJECT_TITLE_VERSION,
        "status": status,
        "run_id": int(config.run_id),
        "run_root": run_root.as_posix(),
        "episodes_per_policy": int(config.episodes_per_policy),
        "max_primitives_per_episode": int(config.max_primitives_per_episode),
        "episode_count": int(episode_count),
        "primitive_execution_count": int(primitive_execution_count),
        "seed": int(config.seed),
        "workers": str(config.workers),
        "max_workers": config.max_workers,
        "storage_format": config.storage_format,
        "compression_level": int(config.compression_level),
        "resume": bool(config.resume),
        "repair_incomplete": bool(config.repair_incomplete),
        "dry_run_schedule": bool(config.dry_run_schedule),
        "source_w01_root": config.source_w01_root.as_posix(),
        "source_w2_root": config.source_w2_root.as_posix(),
        "source_w3_root": config.source_w3_root.as_posix(),
        "post_w3_root": config.post_w3_root.as_posix(),
        "compact_library_path": config.compact_library_path.as_posix(),
        "compact_representative_count": int(library.get("representative_count", 0)) if library else 0,
        "environment_cases": [f"{layer}:{mode}" for layer, mode in ENVIRONMENT_CASES],
        "policy_ids": [str(policy["policy_id"]) for policy in POLICIES],
        "controller_mutation_allowed": False,
        "retuning_allowed": False,
        "claim_status": "simulation_only_full_loop_validation",
        "blocked_claims": list(BLOCKED_CLAIMS),
        "blockers": blockers,
    }
    _write_json(run_root / "manifests" / "full_loop_validation_manifest.json", manifest)
    _write_json(
        run_root / "manifests" / "claim_boundary.json",
        {
            "allowed_claim": "simulation_only_full_loop_validation_for_frozen_post_W3_compact_library",
            "blocked_claims": list(BLOCKED_CLAIMS),
            "controller_mutation_allowed": False,
            "hardware_or_real_flight_claim_allowed": False,
        },
    )


def _write_reports(run_root: Path, *, status: str, episode_rows: list[dict[str, object]], primitive_rows: list[dict[str, object]]) -> None:
    episode_count = len(episode_rows)
    primitive_count = len(primitive_rows)
    terminal_rate = 0.0 if not episode_rows else float(np.mean([bool(row["episode_terminal_useful"]) for row in episode_rows]))
    hard_rate = 0.0 if not episode_rows else float(np.mean([bool(row["hard_failure"]) for row in episode_rows]))
    report = [
        "# v4.8 Full-Loop Validation Report",
        "",
        f"- Status: `{status}`",
        f"- Episodes: `{episode_count}`",
        f"- Primitive executions: `{primitive_count}`",
        f"- Terminal-useful rate: `{terminal_rate:.6f}`",
        f"- Hard-failure rate: `{hard_rate:.6f}`",
        "- Continuation and terminal-useful evidence are separated.",
        "- No controller retuning or identity mutation is performed.",
        "- Claim boundary: simulation-only full-loop validation.",
        "",
    ]
    filesystem_path(run_root / "reports" / "full_loop_validation_report.md").write_text("\n".join(report), encoding="ascii")
    filesystem_path(run_root / "reports" / "memory_ablation_report.md").write_text("\n".join(report), encoding="ascii")
    filesystem_path(run_root / "reports" / "prediction_alignment_report.md").write_text("\n".join(report), encoding="ascii")
    move_on = [
        "# L11 Full-Loop Move-On Check",
        "",
        f"- Status: `{status}`",
        f"- Episodes available: `{episode_count > 0}`",
        f"- File-size audit below 100 MB: `{_file_size_gate(run_root)}`",
        "- Hardware readiness claimed: `False`",
        "- Real-flight transfer claimed: `False`",
        "- Mission success claimed: `False`",
        "",
    ]
    filesystem_path(run_root / "reports" / "l11_full_loop_move_on_check.md").write_text("\n".join(move_on), encoding="ascii")
    filesystem_path(run_root / "reports" / "v1_move_on_check.md").write_text("\n".join(move_on), encoding="ascii")


def _write_file_size_audit(root: Path) -> None:
    rows = []
    root_fs = filesystem_path(root)
    for path in sorted(root_fs.rglob("*")):
        if not path.is_file() or path.name == "file_size_audit.csv":
            continue
        rel = path.relative_to(root_fs).as_posix()
        byte_count = int(path.stat().st_size)
        size_mb = float(byte_count) / float(1024 * 1024)
        rows.append(
            {
                "relative_path": rel,
                "byte_count": byte_count,
                "size_mb": size_mb,
                "above_75mb": bool(size_mb > 75.0),
                "above_100mb": bool(size_mb > MAX_GENERATED_FILE_SIZE_MB),
                "push_allowed": bool(size_mb <= MAX_GENERATED_FILE_SIZE_MB),
                "dense_table_partition": rel.startswith("tables/"),
            }
        )
    _write_csv(root / "metrics" / "file_size_audit.csv", pd.DataFrame(rows))


def _file_size_gate(root: Path) -> bool:
    path = filesystem_path(root / "metrics" / "file_size_audit.csv")
    if not path.is_file():
        return False
    frame = pd.read_csv(path)
    return bool(frame.empty or frame["above_100mb"].astype(bool).sum() == 0)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(filesystem_path(path).read_text(encoding="ascii"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    filesystem_path(path).parent.mkdir(parents=True, exist_ok=True)
    filesystem_path(path).write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="ascii")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    filesystem_path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(filesystem_path(path), index=False)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v4.8 full-loop validation with episodic memory.")
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--episodes-per-policy", type=int, default=50)
    parser.add_argument("--max-primitives-per-episode", type=int, default=4)
    parser.add_argument("--seed", type=int, default=48)
    parser.add_argument("--workers", default=1)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--storage-format", default="auto")
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--repair-incomplete", action="store_true", default=False)
    parser.add_argument("--dry-run-schedule", action="store_true", default=False)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_full_loop_validation(
        FullLoopValidationConfig(
            run_id=args.run_id,
            output_root=args.output_root,
            episodes_per_policy=args.episodes_per_policy,
            max_primitives_per_episode=args.max_primitives_per_episode,
            seed=args.seed,
            workers=args.workers,
            max_workers=args.max_workers,
            storage_format=args.storage_format,
            compression_level=args.compression_level,
            resume=args.resume,
            repair_incomplete=args.repair_incomplete,
            dry_run_schedule=args.dry_run_schedule,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") != "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
