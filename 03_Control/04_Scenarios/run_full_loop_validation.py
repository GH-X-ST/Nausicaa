from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, replace
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
    LiftBeliefGrid,
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
    DEFAULT_OUTPUT_ROOT as DEFAULT_OUTCOME_MODEL_ROOT,
    OutcomeModelBuildConfig,
    run_outcome_model_build,
)
from run_v49_source_audit import (  # noqa: E402
    BLOCKED_CLAIMS,
    DEFAULT_GOVERNOR_SMOKE_ROOT,
    DEFAULT_OUTCOME_ROOT as DEFAULT_SOURCE_OUTCOME_ROOT,
    DEFAULT_POST_W3_ROOT,
    DEFAULT_W01_ROOT,
    DEFAULT_W2_ROOT,
    DEFAULT_W3_ROOT,
    V49SourceAuditConfig,
    run_v49_source_audit,
)
from state_contract import STATE_INDEX, STATE_SIZE, as_state_vector  # noqa: E402
from state_sampling import archive_state_sample_for_family  # noqa: E402


PROJECT_TITLE_VERSION = "LQR-Stabilised Contextual Primitive v4.9"
FULL_LOOP_VERSION = "v49_paired_full_loop_memory_validation_v1"
DEFAULT_OUTPUT_ROOT = Path("03_Control/05_Results/lqr_contextual_v1_0/full_loop_validation")
PAIRED_SCHEDULE_VERSION = "v49_common_random_paired_episode_schedule_v1"
STATIC_MAP_PRIOR_VERSION = "v49_surrogate_grid_static_lift_prior_v1"
ENVIRONMENT_CASES = (
    ("W0", "dry_air"),
    ("W1", "gaussian_single"),
    ("W1", "gaussian_four"),
    ("W2", "annular_gp_single"),
    ("W2", "annular_gp_four"),
    ("W3", "w3_randomised_single"),
    ("W3", "w3_randomised_four"),
)
STATIC_PRIOR_ENVIRONMENT_CASES = (
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
        "policy_role": "static_map_prior_no_episode_update",
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
PAIRED_COMPARISONS = (
    ("context_plus_memory_lambda_0_5", "context_only_without_memory"),
    ("context_plus_memory_lambda_0_8", "context_only_without_memory"),
    ("context_plus_memory_lambda_0_95", "context_only_without_memory"),
    ("context_plus_memory_lambda_0_8", "no_memory_baseline"),
    ("static_map_baseline", "context_plus_memory_lambda_0_8"),
)


@dataclass(frozen=True)
class FullLoopValidationConfig:
    run_id: int = 3
    output_root: Path = DEFAULT_OUTPUT_ROOT
    episodes_per_policy: int = 100
    max_primitives_per_episode: int = 4
    seed: int = 49
    workers: int | str = 1
    max_workers: int | None = 1
    storage_format: str = "auto"
    compression_level: int = 1
    resume: bool = True
    repair_incomplete: bool = False
    dry_run_schedule: bool = False
    replicate_count: int = 1
    source_audit_version: str = "v49"
    paired_schedule_version: str = PAIRED_SCHEDULE_VERSION
    static_map_prior_mode: str = "surrogate_grid_mean"
    source_w01_root: Path = DEFAULT_W01_ROOT
    source_w2_root: Path = DEFAULT_W2_ROOT
    source_w3_root: Path = DEFAULT_W3_ROOT
    post_w3_root: Path = DEFAULT_POST_W3_ROOT
    outcome_root: Path = DEFAULT_SOURCE_OUTCOME_ROOT
    governor_smoke_root: Path = DEFAULT_GOVERNOR_SMOKE_ROOT
    compact_library_path: Path = DEFAULT_COMPACT_LIBRARY
    outcome_model_root: Path = DEFAULT_OUTCOME_MODEL_ROOT
    outcome_model_run_id: int = 2


def run_full_loop_validation(config: FullLoopValidationConfig) -> dict[str, object]:
    """Run paired v4.9 full-loop validation over the frozen compact library."""

    run_root = Path(config.output_root) / f"{int(config.run_id):03d}"
    for subdir in ("manifests", "metrics", "reports"):
        filesystem_path(run_root / subdir).mkdir(parents=True, exist_ok=True)

    source_result = run_v49_source_audit(
        V49SourceAuditConfig(
            output_root=run_root,
            w01_root=config.source_w01_root,
            w2_root=config.source_w2_root,
            w3_root=config.source_w3_root,
            post_w3_root=config.post_w3_root,
            outcome_root=config.outcome_root,
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

    static_prior_result = _build_static_map_prior(config, run_root)
    policy_rows = _policy_rows(static_prior_result["status"])
    paired_schedule = _paired_episode_schedule(config)
    episode_schedule = _expand_policy_schedule(config, paired_schedule, policy_rows)
    _write_policy_set(run_root, policy_rows)
    _write_schedule_outputs(run_root, paired_schedule, episode_schedule)
    if config.dry_run_schedule:
        _write_run_manifests(run_root, config, "dry_run_schedule", library, len(episode_schedule), 0, [], static_prior_result)
        _write_file_size_audit(run_root)
        _write_reports(run_root=run_root, status="dry_run_schedule", episode_rows=[], primitive_rows=[], paired_comparison=pd.DataFrame())
        return {"status": "dry_run_schedule", "run_root": run_root.as_posix(), "episode_count": int(len(episode_schedule))}

    episode_rows: list[dict[str, object]] = []
    primitive_rows: list[dict[str, object]] = []
    governor_rows: list[dict[str, object]] = []
    selector_rows: list[dict[str, object]] = []
    belief_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    belief_state = _initial_policy_beliefs(policy_rows, static_prior_result)

    for scheduled in episode_schedule:
        key = (str(scheduled["policy_id"]), int(scheduled["replicate_id"]))
        episode_result = _run_episode(
            scheduled=scheduled,
            config=config,
            representatives=representatives,
            outcomes_by_variant=outcomes_by_variant,
            records_by_variant=records_by_variant,
            belief=belief_state[key],
        )
        belief_state[key] = episode_result["belief_after"]
        episode_rows.append(episode_result["episode_row"])
        primitive_rows.extend(episode_result["primitive_rows"])
        governor_rows.extend(episode_result["governor_rows"])
        selector_rows.extend(episode_result["selector_rows"])
        belief_rows.extend(episode_result["belief_rows"])
        prediction_rows.extend(episode_result["prediction_rows"])

    prediction_summary = _prediction_alignment_summary(prediction_rows)
    paired_comparison = _paired_policy_comparison(episode_rows, governor_rows, prediction_rows, static_prior_result["status"])
    _write_csv(run_root / "metrics" / "episode_summary.csv", pd.DataFrame(episode_rows))
    _write_csv(run_root / "metrics" / "primitive_execution_log.csv", pd.DataFrame(primitive_rows))
    _write_csv(run_root / "metrics" / "governor_rejection_log.csv", pd.DataFrame(governor_rows))
    _write_csv(run_root / "metrics" / "governor_rejection_summary.csv", _governor_rejection_summary(governor_rows))
    _write_csv(run_root / "metrics" / "selector_choice_log.csv", pd.DataFrame(selector_rows))
    _write_csv(run_root / "metrics" / "belief_snapshot_log.csv", pd.DataFrame(belief_rows))
    _write_csv(run_root / "metrics" / "belief_evolution_summary.csv", _belief_evolution_summary(belief_rows, policy_rows))
    _write_csv(run_root / "metrics" / "prediction_alignment_summary.csv", prediction_summary)
    _write_csv(run_root / "metrics" / "memory_ablation_summary.csv", _memory_ablation_summary(episode_rows))
    _write_csv(run_root / "metrics" / "termination_summary.csv", _termination_summary(episode_rows))
    _write_csv(run_root / "metrics" / "paired_policy_comparison.csv", paired_comparison)
    _write_csv(run_root / "metrics" / "policy_summary.csv", _policy_summary(episode_rows, governor_rows, prediction_rows))
    _write_run_manifests(run_root, config, "complete", library, len(episode_rows), len(primitive_rows), [], static_prior_result)
    _write_file_size_audit(run_root)
    _write_reports(run_root=run_root, status="complete", episode_rows=episode_rows, primitive_rows=primitive_rows, paired_comparison=paired_comparison)
    return {
        "status": "complete",
        "run_root": run_root.as_posix(),
        "episode_count": int(len(episode_rows)),
        "primitive_execution_count": int(len(primitive_rows)),
        "memory_effect_label": _overall_memory_effect_label(paired_comparison),
    }


def _run_episode(
    *,
    scheduled: dict[str, object],
    config: FullLoopValidationConfig,
    representatives: list[dict[str, object]],
    outcomes_by_variant: dict[str, dict[str, object]],
    records_by_variant: dict[str, object],
    belief: LiftBeliefGrid,
) -> dict[str, object]:
    policy = dict(scheduled["policy"])
    episode_id = str(scheduled["episode_id"])
    layer = str(scheduled["W_layer"])
    mode = str(scheduled["environment_mode"])
    state_sample = archive_state_sample_for_family(
        start_state_family="launch_gate",
        paired_start_key=str(scheduled["common_random_key"]),
        sample_index=int(scheduled["paired_episode_index"]),
        seed=int(scheduled["launch_state_seed"]),
        W_layer=layer,
        environment_mode=mode,
    )
    state = as_state_vector(state_sample.state_vector)
    start_family = "launch_gate"
    primitive_rows: list[dict[str, object]] = []
    governor_rows: list[dict[str, object]] = []
    selector_rows: list[dict[str, object]] = []
    belief_rows: list[dict[str, object]] = [
        _belief_log_row(
            episode_id=episode_id,
            paired_episode_index=int(scheduled["paired_episode_index"]),
            primitive_step_index=-1,
            policy=policy,
            layer=layer,
            mode=mode,
            phase="before_episode",
            belief=belief,
            observation=None,
            update_status="episode_start",
        )
    ]
    prediction_rows: list[dict[str, object]] = []
    observed_lift_values: list[float] = []
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
            seed=int(scheduled["environment_seed"]) + primitive_step_index,
            start_state_family=start_family,
            episode_id=episode_id,
            primitive_step_index=primitive_step_index,
        )
        observed_lift_values.append(float(context_payload["row"].get("w_wing_mean_m_s", 0.0)))
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
                "paired_episode_index": int(scheduled["paired_episode_index"]),
                "replicate_id": int(scheduled["replicate_id"]),
                "primitive_step_index": primitive_step_index,
                "common_random_key": str(scheduled["common_random_key"]),
            }
            for row in candidate_rows
        )
        viable_count = sum(1 for row in candidate_rows if bool(row.get("viable", False)))
        selector_rows.append(
            {
                **selector_decision_row(
                    episode_id=episode_id,
                    primitive_step_index=primitive_step_index,
                    policy_id=str(policy["policy_id"]),
                    governor_mode=governor_mode,
                    context=context_payload["row"],
                    selected=selected,
                    candidate_count=len(candidate_rows),
                    viable_count=viable_count,
                ),
                "paired_episode_index": int(scheduled["paired_episode_index"]),
                "replicate_id": int(scheduled["replicate_id"]),
                "common_random_key": str(scheduled["common_random_key"]),
            }
        )
        belief_rows.append(
            _belief_log_row(
                episode_id=episode_id,
                paired_episode_index=int(scheduled["paired_episode_index"]),
                primitive_step_index=primitive_step_index,
                policy=policy,
                layer=layer,
                mode=mode,
                phase="before_primitive",
                belief=belief,
                observation=None,
                update_status="not_updated_within_primitive",
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
            controller_selection_status="selected_by_v49_paired_full_loop_governor",
            candidate_index=record.candidate_index,
            candidate_weight_label=record.candidate_weight_label,
        )
        rollout_row = rollout_evidence_row(rollout)
        representative = _representative_by_variant(representatives, str(selected["primitive_variant_id"]))
        outcome = outcomes_by_variant[str(selected["primitive_variant_id"])]
        primitive_rows.append(
            {
                "episode_id": episode_id,
                "paired_episode_index": int(scheduled["paired_episode_index"]),
                "replicate_id": int(scheduled["replicate_id"]),
                "common_random_key": str(scheduled["common_random_key"]),
                "launch_state_seed": int(scheduled["launch_state_seed"]),
                "environment_seed": int(scheduled["environment_seed"]),
                "primitive_step_index": primitive_step_index,
                "policy_id": str(policy["policy_id"]),
                "memory_lambda": float(policy["lambda_value"]),
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
                int(scheduled["paired_episode_index"]),
                primitive_step_index,
                policy,
                outcome,
                rollout_row,
                primitive_variant_id=str(selected["primitive_variant_id"]),
            )
        )
        belief_rows.append(
            _belief_log_row(
                episode_id=episode_id,
                paired_episode_index=int(scheduled["paired_episode_index"]),
                primitive_step_index=primitive_step_index,
                policy=policy,
                layer=layer,
                mode=mode,
                phase="after_primitive",
                belief=belief,
                observation=None,
                update_status="not_updated_within_primitive",
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

    observed_lift = float(np.mean(observed_lift_values)) if observed_lift_values else 0.0
    observation = LiftObservation(
        x_w_m=float(state_sample.state_vector[STATE_INDEX["x_w"]]),
        y_w_m=float(state_sample.state_vector[STATE_INDEX["y_w"]]),
        lift_evidence_m_s=observed_lift,
        episode_id=episode_id,
    )
    if bool(policy["updates_belief"]):
        belief_after = update_belief(belief, observation)
        update_status = "updated_after_episode"
    else:
        belief_after = belief
        update_status = "not_updated_policy_does_not_use_episode_memory"
    belief_rows.append(
        _belief_log_row(
            episode_id=episode_id,
            paired_episode_index=int(scheduled["paired_episode_index"]),
            primitive_step_index=-1,
            policy=policy,
            layer=layer,
            mode=mode,
            phase="after_episode",
            belief=belief_after,
            observation=observation,
            update_status=update_status,
        )
    )

    episode_row = _episode_row(
        scheduled=scheduled,
        policy=policy,
        primitive_rows=primitive_rows,
        prediction_rows=prediction_rows,
        governor_rows=governor_rows,
        total_duration_s=total_duration_s,
        continuation_valid=continuation_valid,
        terminal_useful=terminal_useful,
        hard_failure=hard_failure,
        termination_cause=termination_cause,
        belief_before=belief,
        belief_after=belief_after,
        observed_lift=observed_lift,
    )
    return {
        "episode_row": episode_row,
        "primitive_rows": primitive_rows,
        "governor_rows": governor_rows,
        "selector_rows": selector_rows,
        "belief_rows": belief_rows,
        "prediction_rows": prediction_rows,
        "belief_after": belief_after,
    }


def _episode_row(
    *,
    scheduled: dict[str, object],
    policy: dict[str, object],
    primitive_rows: list[dict[str, object]],
    prediction_rows: list[dict[str, object]],
    governor_rows: list[dict[str, object]],
    total_duration_s: float,
    continuation_valid: bool,
    terminal_useful: bool,
    hard_failure: bool,
    termination_cause: str,
    belief_before: LiftBeliefGrid,
    belief_after: LiftBeliefGrid,
    observed_lift: float,
) -> dict[str, object]:
    if primitive_rows:
        last = primitive_rows[-1]
        metrics = {
            "energy_residual_m": float(last.get("energy_residual_m", 0.0)),
            "lift_dwell_time_s": float(last.get("lift_dwell_time_s", 0.0)),
            "minimum_wall_margin_m": float(last.get("minimum_wall_margin_m", 0.0)),
            "floor_margin_m": float(last.get("floor_margin_m", 0.0)),
            "ceiling_margin_m": float(last.get("ceiling_margin_m", 0.0)),
            "minimum_speed_m_s": float(last.get("minimum_speed_m_s", 0.0)),
        }
    else:
        metrics = {
            "energy_residual_m": 0.0,
            "lift_dwell_time_s": 0.0,
            "minimum_wall_margin_m": 0.0,
            "floor_margin_m": 0.0,
            "ceiling_margin_m": 0.0,
            "minimum_speed_m_s": 0.0,
        }
    rejected = [row for row in governor_rows if str(row.get("rejection_reason", ""))]
    prediction_agreements = [
        (
            float(row["prediction_actual_agree_continuation"])
            + float(row["prediction_actual_agree_terminal"])
            + float(row["prediction_actual_agree_hard_failure"])
        )
        / 3.0
        for row in prediction_rows
    ]
    return {
        "episode_id": str(scheduled["episode_id"]),
        "episode_index": int(scheduled["episode_index"]),
        "paired_episode_index": int(scheduled["paired_episode_index"]),
        "replicate_id": int(scheduled["replicate_id"]),
        "common_random_key": str(scheduled["common_random_key"]),
        "launch_state_seed": int(scheduled["launch_state_seed"]),
        "environment_seed": int(scheduled["environment_seed"]),
        "policy_id": str(policy["policy_id"]),
        "memory_lambda": float(policy["lambda_value"]),
        "uses_memory_features": bool(policy["uses_memory_features"]),
        "updates_belief": bool(policy["updates_belief"]),
        "W_layer": str(scheduled["W_layer"]),
        "environment_mode": str(scheduled["environment_mode"]),
        "episode_counted_for_claim": True,
        "primitive_count": int(len(primitive_rows)),
        "episode_duration_s": float(total_duration_s),
        "continuation_valid": bool(continuation_valid),
        "episode_terminal_useful": bool(terminal_useful),
        "hard_failure": bool(hard_failure),
        "termination_cause": str(termination_cause),
        "no_viable_primitive": bool(termination_cause == "no_viable_primitive"),
        "x_y_terminal": bool(terminal_useful and "terminal" in str(termination_cause)),
        "governor_rejection_count": int(len(rejected)),
        "prediction_actual_agreement_rate": float(np.mean(prediction_agreements)) if prediction_agreements else 0.0,
        "belief_update_count_before": int(belief_before.update_count),
        "belief_update_count_after": int(belief_after.update_count),
        "observed_lift_evidence_m_s": float(observed_lift),
        **metrics,
        "claim_status": "simulation_only_paired_full_loop_validation_episode",
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


def _paired_episode_schedule(config: FullLoopValidationConfig) -> list[dict[str, object]]:
    rows = []
    pair_count = int(config.episodes_per_policy)
    for replicate_id in range(int(config.replicate_count)):
        for paired_index in range(pair_count):
            layer, mode = ENVIRONMENT_CASES[(paired_index + int(config.seed) + replicate_id) % len(ENVIRONMENT_CASES)]
            launch_seed = int(config.seed) * 1000 + replicate_id * 100000 + paired_index * 37 + 11
            environment_seed = int(config.seed) * 2000 + replicate_id * 100000 + paired_index * 41 + 17
            rows.append(
                {
                    "paired_episode_index": int(paired_index),
                    "replicate_id": int(replicate_id),
                    "W_layer": layer,
                    "environment_mode": mode,
                    "launch_state_seed": int(launch_seed),
                    "environment_seed": int(environment_seed),
                    "common_random_key": f"v49_pair_r{replicate_id:02d}_e{paired_index:05d}",
                    "paired_schedule_version": config.paired_schedule_version,
                    "claim_status": "simulation_only_common_random_paired_episode",
                }
            )
    return rows


def _expand_policy_schedule(
    config: FullLoopValidationConfig,
    paired_schedule: list[dict[str, object]],
    policy_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    index = 0
    for paired in paired_schedule:
        for policy in policy_rows:
            local_index = int(paired["paired_episode_index"])
            rows.append(
                {
                    **paired,
                    "episode_id": f"v49_r{int(config.run_id):03d}_{policy['policy_id']}_p{local_index:05d}_rep{int(paired['replicate_id']):02d}",
                    "episode_index": index,
                    "policy": policy,
                    "policy_id": str(policy["policy_id"]),
                    "memory_lambda": float(policy["lambda_value"]),
                    "uses_memory_features": bool(policy["uses_memory_features"]),
                    "updates_belief": bool(policy["updates_belief"]),
                }
            )
            index += 1
    return rows


def _policy_rows(static_prior_status: str = "ready") -> list[dict[str, object]]:
    rows = []
    for policy in POLICIES:
        row = {
            **policy,
            "static_map_status": static_prior_status if policy["policy_id"] == "static_map_baseline" else "not_applicable",
            "claim_status": "simulation_only_memory_policy",
        }
        rows.append(row)
    return rows


def _initial_policy_beliefs(
    policy_rows: list[dict[str, object]],
    static_prior_result: dict[str, object],
) -> dict[tuple[str, int], LiftBeliefGrid]:
    beliefs: dict[tuple[str, int], LiftBeliefGrid] = {}
    replicate_count = int(static_prior_result.get("replicate_count", 1))
    static_prior = static_prior_result.get("belief")
    for policy in policy_rows:
        for replicate_id in range(replicate_count):
            lambda_value = float(policy["lambda_value"])
            if policy["policy_id"] == "static_map_baseline" and isinstance(static_prior, LiftBeliefGrid):
                beliefs[(str(policy["policy_id"]), replicate_id)] = replace(static_prior, lambda_value=lambda_value, update_count=0)
            else:
                beliefs[(str(policy["policy_id"]), replicate_id)] = initial_belief(lambda_value=lambda_value if lambda_value in BELIEF_LAMBDA_VALUES else 0.0)
    return beliefs


def _build_static_map_prior(config: FullLoopValidationConfig, run_root: Path) -> dict[str, object]:
    base = initial_belief(lambda_value=0.8)
    values = np.zeros((len(base.x_edges_m) - 1, len(base.y_edges_m) - 1), dtype=float)
    sample_rows = []
    for ix in range(values.shape[0]):
        for iy in range(values.shape[1]):
            x_w = 0.5 * (base.x_edges_m[ix] + base.x_edges_m[ix + 1])
            y_w = 0.5 * (base.y_edges_m[iy] + base.y_edges_m[iy + 1])
            lift_values = []
            for case_index, (layer, mode) in enumerate(STATIC_PRIOR_ENVIRONMENT_CASES):
                state = _prior_grid_state(x_w=x_w, y_w=y_w)
                seed = int(config.seed) * 3000 + ix * 101 + iy * 17 + case_index
                try:
                    context_payload = _context_payload(
                        state=state,
                        layer=layer,
                        mode=mode,
                        seed=seed,
                        start_state_family="static_map_prior_grid",
                        episode_id="static_prior",
                        primitive_step_index=case_index,
                    )
                    lift = float(context_payload["row"]["w_wing_mean_m_s"])
                    if np.isfinite(lift):
                        lift_values.append(lift)
                        sample_rows.append(
                            {
                                "ix": ix,
                                "iy": iy,
                                "x_w_m": float(x_w),
                                "y_w_m": float(y_w),
                                "W_layer": layer,
                                "environment_mode": mode,
                                "seed": int(seed),
                                "w_wing_mean_m_s": lift,
                            }
                        )
                except Exception as exc:
                    sample_rows.append(
                        {
                            "ix": ix,
                            "iy": iy,
                            "x_w_m": float(x_w),
                            "y_w_m": float(y_w),
                            "W_layer": layer,
                            "environment_mode": mode,
                            "seed": int(seed),
                            "w_wing_mean_m_s": 0.0,
                            "blocked_reason": f"{type(exc).__name__}:{exc}",
                        }
                    )
            values[ix, iy] = float(np.mean(lift_values)) if lift_values else 0.0
    status = "ready"
    blocked_reason = ""
    if not np.all(np.isfinite(values)) or float(np.max(np.abs(values))) <= 1e-12:
        status = "blocked_static_map_unavailable_no_prior"
        blocked_reason = "static_map_prior_nonfinite_or_zero"
    belief = LiftBeliefGrid(
        x_edges_m=base.x_edges_m,
        y_edges_m=base.y_edges_m,
        values=tuple(tuple(float(value) for value in row) for row in values),
        lambda_value=0.8,
        update_count=0,
    )
    payload = {
        "static_map_prior_version": STATIC_MAP_PRIOR_VERSION,
        "status": status,
        "blocked_reason": blocked_reason,
        "source_modes": [f"{layer}:{mode}" for layer, mode in STATIC_PRIOR_ENVIRONMENT_CASES],
        "static_map_prior_mode": config.static_map_prior_mode,
        "x_edges_m": list(base.x_edges_m),
        "y_edges_m": list(base.y_edges_m),
        "values": [[float(value) for value in row] for row in values],
        "values_checksum": hashlib.sha256(np.round(values, decimals=12).tobytes()).hexdigest(),
        "nonzero": bool(float(np.max(np.abs(values))) > 1e-12),
        "claim_status": "simulation_only_static_map_prior",
    }
    if status == "ready":
        _write_json(run_root / "manifests" / "static_map_prior.json", payload)
    else:
        _write_json(run_root / "manifests" / "blocked_static_map_note.json", payload)
    _write_csv(run_root / "metrics" / "static_map_prior_samples.csv", pd.DataFrame(sample_rows))
    return {
        "status": status,
        "blocked_reason": blocked_reason,
        "belief": belief if status == "ready" else None,
        "replicate_count": int(config.replicate_count),
        "checksum": payload["values_checksum"],
    }


def _prior_grid_state(*, x_w: float, y_w: float) -> np.ndarray:
    state = np.zeros(STATE_SIZE, dtype=float)
    state[STATE_INDEX["x_w"]] = float(x_w)
    state[STATE_INDEX["y_w"]] = float(y_w)
    state[STATE_INDEX["z_w"]] = 1.7
    state[STATE_INDEX["u"]] = 4.5
    state[STATE_INDEX["v"]] = 0.0
    state[STATE_INDEX["w"]] = 0.0
    return as_state_vector(state)


def _belief_log_row(
    *,
    episode_id: str,
    paired_episode_index: int,
    primitive_step_index: int,
    policy: dict[str, object],
    layer: str,
    mode: str,
    phase: str,
    belief: LiftBeliefGrid,
    observation: LiftObservation | None,
    update_status: str,
) -> dict[str, object]:
    snapshot = belief_snapshot_row(belief, label=f"{phase}_p{primitive_step_index:02d}")
    return {
        "belief_snapshot_id": f"{episode_id}_{phase}_p{primitive_step_index:02d}",
        "episode_id": episode_id,
        "paired_episode_index": int(paired_episode_index),
        "primitive_step_index": int(primitive_step_index),
        "phase": phase,
        "policy_id": str(policy["policy_id"]),
        "lambda_value": float(policy["lambda_value"]),
        "uses_memory_features": bool(policy["uses_memory_features"]),
        "updates_belief": bool(policy["updates_belief"]),
        "W_layer": layer,
        "environment_mode": mode,
        "belief_update_count": int(belief.update_count),
        "belief_before_json": json.dumps(snapshot, sort_keys=True, separators=(",", ":")) if phase in {"before_episode", "before_primitive"} else "",
        "observed_lift_evidence_json": "{}" if observation is None else json.dumps(asdict(observation), sort_keys=True, separators=(",", ":")),
        "belief_after_json": json.dumps(snapshot, sort_keys=True, separators=(",", ":")) if phase in {"after_episode", "after_primitive"} else "",
        "memory_update_status": update_status,
        "belief_values": snapshot["values"],
        "claim_status": "simulation_only_episodic_lift_belief",
    }


def _prediction_alignment_row(
    episode_id: str,
    paired_episode_index: int,
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
        "paired_episode_index": int(paired_episode_index),
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


def _write_schedule_outputs(
    run_root: Path,
    paired_schedule: list[dict[str, object]],
    episode_schedule: list[dict[str, object]],
) -> None:
    _write_csv(run_root / "metrics" / "paired_episode_schedule.csv", pd.DataFrame(paired_schedule))
    _write_csv(run_root / "metrics" / "episode_schedule.csv", pd.DataFrame(_schedule_rows(episode_schedule)))


def _schedule_rows(schedule: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for row in schedule:
        rows.append(
            {
                "episode_id": row["episode_id"],
                "episode_index": row["episode_index"],
                "paired_episode_index": row["paired_episode_index"],
                "replicate_id": row["replicate_id"],
                "common_random_key": row["common_random_key"],
                "launch_state_seed": row["launch_state_seed"],
                "environment_seed": row["environment_seed"],
                "policy_id": row["policy_id"],
                "memory_lambda": row["memory_lambda"],
                "uses_memory_features": row["uses_memory_features"],
                "updates_belief": row["updates_belief"],
                "W_layer": row["W_layer"],
                "environment_mode": row["environment_mode"],
                "paired_schedule_version": row["paired_schedule_version"],
                "claim_status": "simulation_only_paired_full_loop_schedule",
            }
        )
    return rows


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
            max_belief_update_count=("belief_update_count_after", "max"),
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


def _belief_evolution_summary(rows: list[dict[str, object]], policy_rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame()
    after = frame[frame["phase"].astype(str) == "after_episode"].copy()
    summary = (
        after.groupby("policy_id", dropna=False)
        .agg(
            snapshot_count=("belief_snapshot_id", "count"),
            min_belief_update_count=("belief_update_count", "min"),
            max_belief_update_count=("belief_update_count", "max"),
            final_belief_update_count=("belief_update_count", "max"),
        )
        .reset_index()
    )
    policy_frame = pd.DataFrame(policy_rows)[["policy_id", "uses_memory_features", "updates_belief", "static_map_status"]]
    summary = summary.merge(policy_frame, on="policy_id", how="left")
    summary["belief_persistence_status"] = summary.apply(_belief_persistence_status, axis=1)
    return summary


def _belief_persistence_status(row: pd.Series) -> str:
    if bool(row.get("updates_belief", False)):
        return "persistent_updates_observed" if int(row.get("max_belief_update_count", 0)) > 1 else "blocked_no_persistent_update"
    if bool(row.get("uses_memory_features", False)):
        return "static_or_read_only_belief_no_episode_update"
    return "memory_not_queried_or_updated"


def _paired_policy_comparison(
    episode_rows: list[dict[str, object]],
    governor_rows: list[dict[str, object]],
    prediction_rows: list[dict[str, object]],
    static_prior_status: str,
) -> pd.DataFrame:
    frame = pd.DataFrame(episode_rows)
    if frame.empty:
        return pd.DataFrame()
    gov_counts = _episode_governor_counts(governor_rows)
    pred_counts = _episode_prediction_counts(prediction_rows)
    frame = frame.merge(gov_counts, on="episode_id", how="left").merge(pred_counts, on="episode_id", how="left")
    frame["governor_rejection_count"] = frame["governor_rejection_count_y"].fillna(frame["governor_rejection_count_x"]).fillna(0).astype(int)
    frame["prediction_actual_agreement_rate"] = frame["prediction_actual_agreement_rate_y"].fillna(frame["prediction_actual_agreement_rate_x"]).fillna(0.0)
    metric_columns = (
        "episode_terminal_useful",
        "hard_failure",
        "episode_duration_s",
        "primitive_count",
        "energy_residual_m",
        "lift_dwell_time_s",
        "governor_rejection_count",
        "no_viable_primitive",
        "prediction_actual_agreement_rate",
    )
    rows = []
    for treatment, baseline in PAIRED_COMPARISONS:
        if treatment == "static_map_baseline" and static_prior_status != "ready":
            continue
        left = frame[frame["policy_id"] == treatment]
        right = frame[frame["policy_id"] == baseline]
        merged = left.merge(
            right,
            on=["paired_episode_index", "replicate_id", "W_layer", "environment_mode", "common_random_key", "launch_state_seed", "environment_seed"],
            suffixes=("_treatment", "_baseline"),
        )
        if merged.empty:
            continue
        for _, item in merged.iterrows():
            row = {
                "comparison_id": f"{treatment}_vs_{baseline}",
                "treatment_policy_id": treatment,
                "baseline_policy_id": baseline,
                "paired_episode_index": int(item["paired_episode_index"]),
                "replicate_id": int(item["replicate_id"]),
                "W_layer": item["W_layer"],
                "environment_mode": item["environment_mode"],
                "common_random_key": item["common_random_key"],
            }
            for metric in metric_columns:
                treatment_value = _numeric(item[f"{metric}_treatment"])
                baseline_value = _numeric(item[f"{metric}_baseline"])
                row[f"{metric}_treatment"] = treatment_value
                row[f"{metric}_baseline"] = baseline_value
                row[f"{metric}_paired_difference"] = treatment_value - baseline_value
            row["memory_effect_label"] = _memory_effect_label(pd.DataFrame([row]))
            row["claim_status"] = "simulation_only_paired_memory_comparison"
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    labels = {
        comparison_id: _memory_effect_label(group)
        for comparison_id, group in out.groupby("comparison_id", dropna=False)
    }
    out["memory_effect_label"] = out["comparison_id"].map(labels)
    return out


def _episode_governor_counts(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["episode_id", "governor_rejection_count"])
    rejected = frame[frame["rejection_reason"].astype(str) != ""]
    if rejected.empty:
        return pd.DataFrame(columns=["episode_id", "governor_rejection_count"])
    return rejected.groupby("episode_id", dropna=False).size().reset_index(name="governor_rejection_count")


def _episode_prediction_counts(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["episode_id", "prediction_actual_agreement_rate"])
    frame = frame.assign(
        prediction_actual_agreement_rate=(
            frame["prediction_actual_agree_continuation"].astype(bool).astype(float)
            + frame["prediction_actual_agree_terminal"].astype(bool).astype(float)
            + frame["prediction_actual_agree_hard_failure"].astype(bool).astype(float)
        )
        / 3.0
    )
    return frame.groupby("episode_id", dropna=False)["prediction_actual_agreement_rate"].mean().reset_index()


def _memory_effect_label(group: pd.DataFrame) -> str:
    if group.empty:
        return "memory_benefit_not_supported"
    terminal_delta = float(group["episode_terminal_useful_paired_difference"].mean())
    hard_delta = float(group["hard_failure_paired_difference"].mean())
    no_viable_delta = float(group["no_viable_primitive_paired_difference"].mean())
    if terminal_delta > 0.02 and hard_delta <= 0.0 and no_viable_delta <= 0.0:
        return "memory_benefit_supported"
    if abs(terminal_delta) <= 0.02 and abs(hard_delta) <= 0.02:
        return "mixed_memory_effect"
    if terminal_delta > 0.0 or hard_delta < 0.0:
        return "mixed_memory_effect"
    return "memory_benefit_not_supported"


def _overall_memory_effect_label(paired_comparison: pd.DataFrame) -> str:
    if paired_comparison.empty:
        return "memory_benefit_not_supported"
    labels = set(paired_comparison["memory_effect_label"].astype(str))
    if labels == {"memory_benefit_supported"}:
        return "memory_benefit_supported"
    if "memory_benefit_supported" in labels or "mixed_memory_effect" in labels:
        return "mixed_memory_effect"
    return "memory_benefit_not_supported"


def _policy_summary(
    episode_rows: list[dict[str, object]],
    governor_rows: list[dict[str, object]],
    prediction_rows: list[dict[str, object]],
) -> pd.DataFrame:
    frame = pd.DataFrame(episode_rows)
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
            governor_rejection_count=("governor_rejection_count", "sum"),
            prediction_actual_agreement_rate=("prediction_actual_agreement_rate", "mean"),
        )
        .reset_index()
    )
    del governor_rows, prediction_rows
    return summary


def _write_policy_set(run_root: Path, policy_rows: list[dict[str, object]]) -> None:
    _write_json(
        run_root / "manifests" / "policy_set.json",
        {
            "policy_set_version": "v49_paired_memory_policy_set_v1",
            "policies": policy_rows,
            "claim_status": "simulation_only_memory_ablation_policy_set",
        },
    )
    _write_csv(run_root / "metrics" / "policy_set.csv", pd.DataFrame(policy_rows))


def _write_blocked_outputs(run_root: Path, config: FullLoopValidationConfig, blocked_reason: str, blockers: list[object]) -> None:
    _write_run_manifests(run_root, config, "blocked", {}, 0, 0, [blocked_reason, *[str(item) for item in blockers if item]], {"status": "not_evaluated"})
    _write_csv(run_root / "metrics" / "episode_summary.csv", pd.DataFrame())
    _write_file_size_audit(run_root)
    _write_reports(run_root=run_root, status="blocked", episode_rows=[], primitive_rows=[], paired_comparison=pd.DataFrame())


def _write_run_manifests(
    run_root: Path,
    config: FullLoopValidationConfig,
    status: str,
    library: dict[str, object],
    episode_count: int,
    primitive_execution_count: int,
    blockers: list[str],
    static_prior_result: dict[str, object],
) -> None:
    manifest = {
        "manifest_version": FULL_LOOP_VERSION,
        "project_title_version": PROJECT_TITLE_VERSION,
        "status": status,
        "run_id": int(config.run_id),
        "run_root": run_root.as_posix(),
        "episodes_per_policy": int(config.episodes_per_policy),
        "paired_episode_count": int(config.episodes_per_policy),
        "replicate_count": int(config.replicate_count),
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
        "source_audit_version": config.source_audit_version,
        "paired_schedule_version": config.paired_schedule_version,
        "static_map_prior_mode": config.static_map_prior_mode,
        "static_map_prior_status": str(static_prior_result.get("status", "")),
        "static_map_prior_checksum": str(static_prior_result.get("checksum", "")),
        "source_w01_root": config.source_w01_root.as_posix(),
        "source_w2_root": config.source_w2_root.as_posix(),
        "source_w3_root": config.source_w3_root.as_posix(),
        "post_w3_root": config.post_w3_root.as_posix(),
        "outcome_root": config.outcome_root.as_posix(),
        "compact_library_path": config.compact_library_path.as_posix(),
        "compact_representative_count": int(library.get("representative_count", 0)) if library else 0,
        "environment_cases": [f"{layer}:{mode}" for layer, mode in ENVIRONMENT_CASES],
        "policy_ids": [str(policy["policy_id"]) for policy in POLICIES],
        "paired_common_random_policy": "same_launch_state_seed_environment_seed_and_common_random_key_for_every_policy",
        "controller_mutation_allowed": False,
        "retuning_allowed": False,
        "claim_status": "simulation_only_paired_full_loop_validation",
        "blocked_claims": list(BLOCKED_CLAIMS),
        "blockers": blockers,
    }
    _write_json(run_root / "manifests" / "full_loop_validation_manifest.json", manifest)
    _write_json(
        run_root / "manifests" / "claim_boundary.json",
        {
            "allowed_claim": "simulation_only_paired_full_loop_validation_for_frozen_post_W3_compact_library",
            "blocked_claims": list(BLOCKED_CLAIMS),
            "controller_mutation_allowed": False,
            "hardware_or_real_flight_claim_allowed": False,
            "memory_improvement_claim_requires_paired_comparison_support": True,
        },
    )


def _write_reports(
    *,
    run_root: Path,
    status: str,
    episode_rows: list[dict[str, object]],
    primitive_rows: list[dict[str, object]],
    paired_comparison: pd.DataFrame,
) -> None:
    episode_count = len(episode_rows)
    primitive_count = len(primitive_rows)
    terminal_rate = 0.0 if not episode_rows else float(np.mean([bool(row["episode_terminal_useful"]) for row in episode_rows]))
    hard_rate = 0.0 if not episode_rows else float(np.mean([bool(row["hard_failure"]) for row in episode_rows]))
    memory_label = _overall_memory_effect_label(paired_comparison)
    report = [
        "# v4.9 Paired Full-Loop Validation Report",
        "",
        f"- Status: `{status}`",
        f"- Episodes: `{episode_count}`",
        f"- Primitive executions: `{primitive_count}`",
        f"- Terminal-useful rate: `{terminal_rate:.6f}`",
        f"- Hard-failure rate: `{hard_rate:.6f}`",
        f"- Paired memory effect label: `{memory_label}`",
        "- Policies are compared on common-random paired launch/environment episodes.",
        "- Episodic memory policies persist belief across launches.",
        "- X/y terminal-useful evidence is terminal episode evidence, not continuation success.",
        "- No controller retuning or identity mutation is performed.",
        "- Claim boundary: simulation-only paired full-loop validation.",
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
        f"- Paired schedule present: `{filesystem_path(run_root / 'metrics' / 'paired_episode_schedule.csv').is_file()}`",
        f"- Belief evolution present: `{filesystem_path(run_root / 'metrics' / 'belief_evolution_summary.csv').is_file()}`",
        f"- Paired comparison present: `{filesystem_path(run_root / 'metrics' / 'paired_policy_comparison.csv').is_file()}`",
        f"- File-size audit below 100 MB: `{_file_size_gate(run_root)}`",
        f"- Memory effect label: `{memory_label}`",
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


def _numeric(value: object, default: float = 0.0) -> float:
    try:
        if isinstance(value, (bool, np.bool_)):
            return float(bool(value))
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(filesystem_path(path).read_text(encoding="ascii"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    filesystem_path(path).parent.mkdir(parents=True, exist_ok=True)
    filesystem_path(path).write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="ascii")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    filesystem_path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(filesystem_path(path), index=False)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v4.9 paired full-loop validation with episodic memory.")
    parser.add_argument("--run-id", type=int, default=3)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--episodes-per-policy", type=int, default=100)
    parser.add_argument("--max-primitives-per-episode", type=int, default=4)
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--workers", default=1)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--storage-format", default="auto")
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--repair-incomplete", action="store_true", default=False)
    parser.add_argument("--dry-run-schedule", action="store_true", default=False)
    parser.add_argument("--replicate-count", type=int, default=1)
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
            replicate_count=args.replicate_count,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") != "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
