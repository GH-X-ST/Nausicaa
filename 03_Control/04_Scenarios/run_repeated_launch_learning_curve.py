from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

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
from directional_residual_lift_belief import (  # noqa: E402
    DirectionalResidualObservation,
    belief_snapshot_row,
    initial_directional_residual_lift_belief,
    query_directional_residual_lift_features,
    update_directional_residual_lift_belief,
)
from episode_selector import select_compact_representative  # noqa: E402
from primitive_timing_contract import primitive_timing_contract_row  # noqa: E402
from run_post_w3_library_size_study import LIBRARY_SIZE_CASE_IDS  # noqa: E402


PROJECT_TITLE_VERSION = "LQR-Stabilised Contextual Primitive v5.0"
VALIDATION_VERSION = "repeated_launch_fixed_case_validation_v5"
HISTORY_LENGTHS = (0, 5, 10, 20, 50, 100)
POLICY_GROUPS = ("no_memory", "static_prior", "directional_residual_memory_N", "safe_explore_then_exploit_N")
DEFAULT_LIBRARY_ROOT = Path("03_Control/05_Results/lqr_contextual_v1_0/post_w3_library_size_study/001")
DEFAULT_OUTCOME_ROOT = Path("03_Control/05_Results/lqr_contextual_v1_0/outcome_model/003")
DEFAULT_OUTPUT_ROOT = Path("03_Control/05_Results/lqr_contextual_v1_0/repeated_launch_validation")
BLOCKED_CLAIMS = (
    "hardware_readiness",
    "real_flight_transfer",
    "mission_success",
    "memory_improvement",
    "full_autonomy",
)


@dataclass(frozen=True)
class RepeatedLaunchValidationConfig:
    library_root: Path = DEFAULT_LIBRARY_ROOT
    outcome_root: Path = DEFAULT_OUTCOME_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    run_id: int = 1


def run_repeated_launch_learning_curve(config: RepeatedLaunchValidationConfig) -> dict[str, object]:
    run_root = Path(config.output_root) / f"{int(config.run_id):03d}"
    for subdir in ("manifests", "metrics", "reports"):
        filesystem_path(run_root / subdir).mkdir(parents=True, exist_ok=True)
    blocked_reason = _blocked_reason(config)
    if blocked_reason:
        _write_blocked_outputs(run_root, config, blocked_reason)
        return {"status": "blocked", "blocked_reason": blocked_reason, "run_root": run_root.as_posix()}

    decision_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    learning_rows: list[dict[str, object]] = []
    belief_rows: list[dict[str, object]] = []
    for case_id in LIBRARY_SIZE_CASE_IDS:
        library = _read_json(Path(config.library_root) / "manifests" / f"{case_id}_primitive_library.json")
        outcome = _read_outcome_rows(config.outcome_root, case_id)
        representatives = list(library.get("representatives", []))
        fixed_case = _fixed_test_case(case_id)
        heldout_launches = _heldout_launch_contexts(fixed_case=fixed_case, case_id=case_id)
        for history_length in HISTORY_LENGTHS:
            baseline_choices = _selection_choices_for_policy(
                representatives=representatives,
                outcome=outcome,
                case_id=case_id,
                fixed_case=fixed_case,
                heldout_launches=heldout_launches,
                history_length=history_length,
                policy_group="no_memory",
            )
            directional_choices = _selection_choices_for_policy(
                representatives=representatives,
                outcome=outcome,
                case_id=case_id,
                fixed_case=fixed_case,
                heldout_launches=heldout_launches,
                history_length=history_length,
                policy_group="directional_residual_memory_N",
            )
            for policy_group in POLICY_GROUPS:
                choices = _selection_choices_for_policy(
                    representatives=representatives,
                    outcome=outcome,
                    case_id=case_id,
                    fixed_case=fixed_case,
                    heldout_launches=heldout_launches,
                    history_length=history_length,
                    policy_group=policy_group,
                )
                candidate_rows.extend(row for choice in choices for row in choice["candidate_rows"])
                decision_rows.extend(choice["decision_row"] for choice in choices)
                learning_rows.append(
                    _learning_curve_summary_row(
                        choices=choices,
                        case_id=case_id,
                        policy_group=policy_group,
                        history_length=history_length,
                        baseline_choices=baseline_choices,
                        directional_choices=directional_choices,
                    )
                )
                belief_rows.append(
                    belief_snapshot_row(
                        _belief_for_policy(policy_group=policy_group, history_length=history_length, fixed_case=fixed_case),
                        label=f"{case_id}_{policy_group}_h{int(history_length)}",
                    )
                )
    _write_csv(run_root / "metrics" / "validation_learning_curve.csv", pd.DataFrame(learning_rows))
    _write_csv(run_root / "metrics" / "validation_decisions.csv", pd.DataFrame(decision_rows))
    _write_csv(run_root / "metrics" / "validation_candidate_scores.csv", pd.DataFrame(candidate_rows))
    _write_csv(run_root / "metrics" / "belief_snapshots.csv", pd.DataFrame(belief_rows))
    manifest = {
        "manifest_version": VALIDATION_VERSION,
        "project_title_version": PROJECT_TITLE_VERSION,
        "status": "complete",
        "run_id": int(config.run_id),
        "run_root": run_root.as_posix(),
        "library_root": Path(config.library_root).as_posix(),
        "outcome_root": Path(config.outcome_root).as_posix(),
        "history_lengths": list(HISTORY_LENGTHS),
        "policy_groups": list(POLICY_GROUPS),
        "library_size_case_ids": list(LIBRARY_SIZE_CASE_IDS),
        "primitive_timing_contract": primitive_timing_contract_row(),
        "validation_protocol": "fixed_case_environment_plant_latency_and_heldout_launches_by_library_size_case",
        "validation_evidence_level": "fixed_case_repeated_launch_evaluator_not_yet_passed",
        "claim_status": "simulation_only_repeated_launch_validation_not_hardware_ready_no_memory_improvement_claim",
        "blocked_claims": list(BLOCKED_CLAIMS),
    }
    _write_json(run_root / "manifests" / "repeated_launch_validation_manifest.json", manifest)
    _write_file_size_audit(run_root)
    _write_report(run_root, manifest)
    return {
        "status": "complete",
        "run_root": run_root.as_posix(),
        "history_lengths": list(HISTORY_LENGTHS),
        "policy_groups": list(POLICY_GROUPS),
        "library_size_case_ids": list(LIBRARY_SIZE_CASE_IDS),
    }


def _blocked_reason(config: RepeatedLaunchValidationConfig) -> str:
    study_manifest = filesystem_path(Path(config.library_root) / "manifests" / "post_w3_library_size_study_manifest.json")
    if not study_manifest.is_file():
        return "missing_post_w3_library_size_study_manifest"
    try:
        study_payload = json.loads(study_manifest.read_text(encoding="ascii"))
    except Exception as exc:
        return f"unreadable_post_w3_library_size_study_manifest:{type(exc).__name__}"
    if str(study_payload.get("project_title_version", "")) != PROJECT_TITLE_VERSION:
        return "post_w3_library_size_study_not_v5_project_title"
    for case_id in LIBRARY_SIZE_CASE_IDS:
        library_path = filesystem_path(Path(config.library_root) / "manifests" / f"{case_id}_primitive_library.json")
        if not library_path.is_file():
            return f"missing_library_size_case_manifest:{case_id}"
    outcome_path = filesystem_path(Path(config.outcome_root) / "metrics" / "outcome_model_table.csv")
    if not outcome_path.is_file():
        return "missing_outcome_model_table"
    outcome_manifest = filesystem_path(Path(config.outcome_root) / "manifests" / "outcome_model_manifest.json")
    if not outcome_manifest.is_file():
        return "missing_outcome_model_manifest"
    try:
        outcome_payload = json.loads(outcome_manifest.read_text(encoding="ascii"))
    except Exception as exc:
        return f"unreadable_outcome_model_manifest:{type(exc).__name__}"
    if str(outcome_payload.get("project_title_version", "")) != PROJECT_TITLE_VERSION:
        return "outcome_model_not_v5_project_title"
    outcome = pd.read_csv(outcome_path)
    missing_cases = set(LIBRARY_SIZE_CASE_IDS) - set(outcome.get("library_size_case_id", pd.Series(dtype=str)).astype(str))
    if missing_cases:
        return "outcome_model_missing_library_size_cases:" + ",".join(sorted(missing_cases))
    return ""


def _read_outcome_rows(outcome_root: Path, case_id: str) -> dict[str, dict[str, object]]:
    frame = pd.read_csv(filesystem_path(Path(outcome_root) / "metrics" / "outcome_model_table.csv"))
    frame = frame[frame["library_size_case_id"].astype(str) == str(case_id)]
    return {str(row["primitive_variant_id"]): row for row in frame.to_dict(orient="records")}


def _fixed_test_case(case_id: str) -> dict[str, object]:
    case_index = LIBRARY_SIZE_CASE_IDS.index(case_id)
    return {
        "fixed_test_case_id": f"fixed_case_{case_id}",
        "environment_instance_id": f"fixed_env_{case_index:02d}",
        "plant_instance_id": f"fixed_plant_{case_index:02d}",
        "latency_instance_id": f"fixed_latency_{case_index:02d}",
        "x_w_m": float(-0.75 + 0.25 * case_index),
        "y_w_m": float(0.25 * case_index),
        "z_w_m": 1.5,
        "direction_rad": float(0.2 * case_index),
        "frozen_case_status": "environment_plant_latency_fixed_for_all_policies_and_histories",
    }


def _heldout_launch_contexts(*, fixed_case: dict[str, object], case_id: str) -> list[dict[str, object]]:
    launches: list[dict[str, object]] = []
    for index, start_family in enumerate(("launch_gate", "inflight_nominal", "inflight_lift_region", "inflight_boundary_near")):
        launches.append(
            {
                "heldout_launch_id": f"{case_id}_heldout_{index:02d}",
                "start_state_family": start_family,
                "x_w_m": float(fixed_case["x_w_m"]) + 0.1 * index,
                "y_w_m": float(fixed_case["y_w_m"]) - 0.05 * index,
                "z_w_m": float(fixed_case["z_w_m"]) + 0.05 * (index % 2),
                "direction_rad": float(fixed_case["direction_rad"]) + 0.25 * index,
                "wall_margin_m": max(0.15, 0.55 - 0.08 * index),
                "floor_margin_m": max(0.12, 0.45 - 0.04 * index),
                "ceiling_margin_m": max(0.12, 0.50 - 0.03 * index),
                "speed_margin_m_s": 1.0,
            }
        )
    return launches


def _belief_for_policy(*, policy_group: str, history_length: int, fixed_case: dict[str, object]):
    belief = initial_directional_residual_lift_belief()
    if policy_group == "no_memory":
        return belief
    seed_count = 1 if policy_group == "static_prior" else int(history_length)
    for index in range(max(0, seed_count)):
        belief = update_directional_residual_lift_belief(
            belief,
            DirectionalResidualObservation(
                x_w_m=float(fixed_case["x_w_m"]) + 0.05 * (index % 3),
                y_w_m=float(fixed_case["y_w_m"]) - 0.05 * (index % 2),
                z_w_m=float(fixed_case["z_w_m"]) + 0.1 * (index % 2),
                direction_rad=float(fixed_case["direction_rad"]) + 0.25 * (index % 4),
                lift_residual_m_s=0.02 + 0.001 * (index % 5),
                energy_residual_m=0.01 + 0.001 * (index % 3),
                dwell_residual_s=0.005 + 0.001 * (index % 4),
            ),
        )
    return belief


def _belief_features_for_launch(
    *,
    policy_group: str,
    history_length: int,
    fixed_case: dict[str, object],
    launch: dict[str, object],
) -> dict[str, object]:
    belief = _belief_for_policy(policy_group=policy_group, history_length=history_length, fixed_case=fixed_case)
    features = query_directional_residual_lift_features(
        belief,
        x_w_m=float(launch["x_w_m"]),
        y_w_m=float(launch["y_w_m"]),
        z_w_m=float(launch["z_w_m"]),
        direction_rad=float(launch["direction_rad"]),
    )
    features["history_length"] = int(history_length)
    if policy_group in {"no_memory", "static_prior", "directional_residual_memory_N"}:
        features["belief_uncertainty"] = 0.0
    return features


def _validation_context(
    *,
    case_id: str,
    history_length: int,
    fixed_case: dict[str, object],
    launch: dict[str, object],
    policy_group: str,
) -> dict[str, object]:
    return {
        "context_id": f"v5_validation_{case_id}_{policy_group}_h{history_length}_{launch['heldout_launch_id']}",
        "fixed_test_case_id": str(fixed_case["fixed_test_case_id"]),
        "environment_instance_id": str(fixed_case["environment_instance_id"]),
        "plant_instance_id": str(fixed_case["plant_instance_id"]),
        "latency_instance_id": str(fixed_case["latency_instance_id"]),
        "heldout_launch_id": str(launch["heldout_launch_id"]),
        "W_layer": "W1",
        "environment_mode": "validation_repeated_launch",
        "start_state_family": str(launch["start_state_family"]),
        "governor_mode": "continuation_mode",
        "wall_margin_m": float(launch["wall_margin_m"]),
        "floor_margin_m": float(launch["floor_margin_m"]),
        "ceiling_margin_m": float(launch["ceiling_margin_m"]),
        "speed_margin_m_s": float(launch["speed_margin_m_s"]),
        "latency_case": "nominal",
        "library_size_case_id": str(case_id),
        "history_length": int(history_length),
        "policy_group": str(policy_group),
    }


def _selection_choices_for_policy(
    *,
    representatives: list[dict[str, object]],
    outcome: dict[str, dict[str, object]],
    case_id: str,
    fixed_case: dict[str, object],
    heldout_launches: list[dict[str, object]],
    history_length: int,
    policy_group: str,
) -> list[dict[str, object]]:
    choices: list[dict[str, object]] = []
    for launch in heldout_launches:
        features = _belief_features_for_launch(
            policy_group=policy_group,
            history_length=history_length,
            fixed_case=fixed_case,
            launch=launch,
        )
        context = _validation_context(
            case_id=case_id,
            history_length=history_length,
            fixed_case=fixed_case,
            launch=launch,
            policy_group=policy_group,
        )
        selected, rows = select_compact_representative(
            representatives=representatives,
            outcome_rows_by_variant_id=outcome,
            context=context,
            governor_mode="continuation_mode",
            policy_id=f"v5_{case_id}_{policy_group}_h{history_length}",
            belief_features=features,
        )
        for row in rows:
            row["policy_group"] = policy_group
            row["fixed_test_case_id"] = context["fixed_test_case_id"]
            row["heldout_launch_id"] = context["heldout_launch_id"]
        selected_variant_id = "" if selected is None else str(selected.get("primitive_variant_id", ""))
        selected_outcome = outcome.get(selected_variant_id, {})
        viable_count = int(sum(bool(row.get("viable", False)) for row in rows))
        choices.append(
            {
                "selected": selected,
                "selected_outcome": selected_outcome,
                "candidate_rows": rows,
                "decision_row": {
                    "library_size_case_id": case_id,
                    "policy_group": policy_group,
                    "history_length": int(history_length),
                    "fixed_test_case_id": context["fixed_test_case_id"],
                    "heldout_launch_id": context["heldout_launch_id"],
                    "governor_mode": "continuation_mode",
                    "candidate_count": int(len(rows)),
                    "viable_count": viable_count,
                    "governor_rejection_count": int(len(rows) - viable_count),
                    "decision_status": "selected_compact_representative" if selected else "blocked_no_viable_representative",
                    "selected_primitive_variant_id": selected_variant_id,
                    "selected_primitive_id": "" if selected is None else str(selected.get("primitive_id", "")),
                    "selected_score": float("-inf")
                    if selected is None
                    else float(selected.get("total_score_with_memory_and_exploration", selected.get("score", float("-inf")))),
                    "claim_status": "simulation_only_fixed_case_repeated_launch_decision",
                },
            }
        )
    return choices


def _learning_curve_summary_row(
    *,
    choices: list[dict[str, object]],
    case_id: str,
    policy_group: str,
    history_length: int,
    baseline_choices: list[dict[str, object]],
    directional_choices: list[dict[str, object]],
) -> dict[str, object]:
    selected = [choice for choice in choices if choice["selected"] is not None]
    no_viable_count = int(len(choices) - len(selected))
    selected_ids = [str(choice["decision_row"]["selected_primitive_variant_id"]) for choice in selected]
    primitive_ids = [str(choice["decision_row"]["selected_primitive_id"]) for choice in selected]
    hard_failure = [_float(choice["selected_outcome"].get("hard_failure_risk", 1.0), default=1.0) for choice in selected]
    terminal = [_float(choice["selected_outcome"].get("terminal_useful_probability", 0.0)) for choice in selected]
    continuation = [_float(choice["selected_outcome"].get("continuation_probability", 0.0)) for choice in selected]
    dwell = [_float(choice["selected_outcome"].get("expected_lift_dwell_time_s", 0.0)) for choice in selected]
    energy = [_float(choice["selected_outcome"].get("expected_energy_residual_m", 0.0)) for choice in selected]
    floor_or_ceiling = [
        1.0
        if min(
            _float(choice["selected_outcome"].get("floor_margin_min_m", 0.0)),
            _float(choice["selected_outcome"].get("ceiling_margin_min_m", 0.0)),
        )
        < 0.0
        else 0.0
        for choice in selected
    ]
    baseline_ids = [str(choice["decision_row"]["selected_primitive_variant_id"]) for choice in baseline_choices]
    directional_ids = [str(choice["decision_row"]["selected_primitive_variant_id"]) for choice in directional_choices]
    current_ids = [str(choice["decision_row"]["selected_primitive_variant_id"]) for choice in choices]
    memory_changes = [1.0 if left != right else 0.0 for left, right in zip(current_ids, baseline_ids)]
    exploration_changes = [1.0 if left != right else 0.0 for left, right in zip(current_ids, directional_ids)]
    rejection_counts = [int(choice["decision_row"]["governor_rejection_count"]) for choice in choices]
    denominator = max(1, len(choices))
    return {
        "library_size_case_id": case_id,
        "policy_group": policy_group,
        "history_length": int(history_length),
        "fixed_test_case_count": 1,
        "heldout_launch_count": int(len(choices)),
        "safe_success_rate": _mean(continuation),
        "hard_failure_rate": _mean(hard_failure) if selected else 1.0,
        "floor_or_ceiling_violation_rate": _mean(floor_or_ceiling),
        "no_viable_primitive_rate": float(no_viable_count) / float(denominator),
        "terminal_useful_rate": _mean(terminal),
        "lift_capture_rate": _mean([1.0 if value > 0.0 else 0.0 for value in dwell]),
        "mean_lift_dwell_time_s": _mean(dwell),
        "mean_energy_residual_m": _mean(energy),
        "selected_variant_unique_count": int(len(set(selected_ids))),
        "selected_primitive_family_count": int(len(set(primitive_ids))),
        "memory_changed_selection_rate": _mean(memory_changes) if policy_group != "no_memory" else 0.0,
        "exploration_changed_selection_rate": _mean(exploration_changes)
        if policy_group == "safe_explore_then_exploit_N"
        else 0.0,
        "mean_governor_rejection_count": _mean(rejection_counts),
        "validation_evidence_level": "fixed_case_repeated_launch_evaluator_not_yet_passed",
        "claim_status": "simulation_only_no_memory_improvement_claim",
    }


def _write_blocked_outputs(run_root: Path, config: RepeatedLaunchValidationConfig, blocked_reason: str) -> None:
    manifest = {
        "manifest_version": VALIDATION_VERSION,
        "project_title_version": PROJECT_TITLE_VERSION,
        "status": "blocked",
        "run_id": int(config.run_id),
        "run_root": run_root.as_posix(),
        "library_root": Path(config.library_root).as_posix(),
        "outcome_root": Path(config.outcome_root).as_posix(),
        "blocked_reason": str(blocked_reason),
        "history_lengths": list(HISTORY_LENGTHS),
        "policy_groups": list(POLICY_GROUPS),
        "library_size_case_ids": list(LIBRARY_SIZE_CASE_IDS),
        "validation_evidence_level": "validation_smoke_only_not_method_evidence",
        "blocked_claims": list(BLOCKED_CLAIMS),
    }
    _write_json(run_root / "manifests" / "repeated_launch_validation_manifest.json", manifest)
    _write_csv(run_root / "metrics" / "validation_learning_curve.csv", pd.DataFrame())
    _write_file_size_audit(run_root)
    _write_report(run_root, manifest)


def _write_report(run_root: Path, manifest: dict[str, object]) -> None:
    lines = [
        "# v5.0 Repeated-Launch Validation",
        "",
        f"- Status: `{manifest.get('status', '')}`",
        f"- Histories: `{','.join(str(value) for value in HISTORY_LENGTHS)}`",
        f"- Policies: `{','.join(POLICY_GROUPS)}`",
        f"- Evidence level: `{manifest.get('validation_evidence_level', '')}`",
        "- Claim boundary: simulation-only; no hardware, flight transfer, mission, autonomy, or memory-improvement claim.",
        "",
    ]
    if manifest.get("blocked_reason"):
        lines.insert(4, f"- Blocked reason: `{manifest['blocked_reason']}`")
    filesystem_path(run_root / "reports" / "repeated_launch_validation_report.md").write_text(
        "\n".join(lines),
        encoding="ascii",
    )


def _write_file_size_audit(root: Path) -> None:
    rows = []
    root_fs = filesystem_path(root)
    for path in sorted(root_fs.rglob("*")):
        if not path.is_file():
            continue
        size_mb = float(path.stat().st_size) / float(1024 * 1024)
        rows.append(
            {
                "relative_path": path.relative_to(root_fs).as_posix(),
                "byte_count": int(path.stat().st_size),
                "size_mb": size_mb,
                "above_75mb": bool(size_mb > 75.0),
                "above_100mb": bool(size_mb > MAX_GENERATED_FILE_SIZE_MB),
                "push_allowed": bool(size_mb <= MAX_GENERATED_FILE_SIZE_MB),
            }
        )
    _write_csv(root / "metrics" / "file_size_audit.csv", pd.DataFrame(rows))


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _float(value: object, default: float = 0.0) -> float:
    try:
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
    parser = argparse.ArgumentParser(description="Run v5.0 fixed-case repeated-launch learning-curve validation.")
    parser.add_argument("--library-root", type=Path, default=DEFAULT_LIBRARY_ROOT)
    parser.add_argument("--outcome-root", type=Path, default=DEFAULT_OUTCOME_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_repeated_launch_learning_curve(
        RepeatedLaunchValidationConfig(
            library_root=args.library_root,
            outcome_root=args.outcome_root,
            output_root=args.output_root,
            run_id=args.run_id,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {"complete", "blocked"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
