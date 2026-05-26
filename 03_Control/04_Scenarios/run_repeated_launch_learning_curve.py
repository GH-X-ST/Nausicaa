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
    initial_directional_residual_lift_belief,
    query_directional_residual_lift_features,
    update_directional_residual_lift_belief,
)
from episode_selector import select_compact_representative  # noqa: E402
from primitive_timing_contract import primitive_timing_contract_row  # noqa: E402
from run_post_w3_library_size_study import LIBRARY_SIZE_CASE_IDS  # noqa: E402


PROJECT_TITLE_VERSION = "LQR-Stabilised Contextual Primitive v4.11"
VALIDATION_VERSION = "repeated_launch_learning_curve_validation_v411"
HISTORY_LENGTHS = (0, 5, 10, 20, 50, 100)
DEFAULT_LIBRARY_ROOT = Path("03_Control/05_Results/lqr_contextual_v1_0/post_w3_library_size_study/001")
DEFAULT_OUTCOME_ROOT = Path("03_Control/05_Results/lqr_contextual_v1_0/outcome_model/002")
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
    for case_id in LIBRARY_SIZE_CASE_IDS:
        library = _read_json(Path(config.library_root) / "manifests" / f"{case_id}_primitive_library.json")
        outcome = _read_outcome_rows(config.outcome_root, case_id)
        representatives = list(library.get("representatives", []))
        for history_length in HISTORY_LENGTHS:
            belief = _belief_after_history(history_length)
            features = query_directional_residual_lift_features(
                belief,
                x_m=0.0,
                y_m=0.0,
                z_m=1.5,
                direction_rad=0.0,
            )
            context = _validation_context(case_id=case_id, history_length=history_length)
            selected, rows = select_compact_representative(
                representatives=representatives,
                outcome_rows_by_variant_id=outcome,
                context=context,
                governor_mode="continuation_mode",
                policy_id=f"v411_{case_id}_h{history_length}",
                belief_features=features,
            )
            candidate_rows.extend(rows)
            decision_rows.append(
                {
                    "library_size_case_id": case_id,
                    "history_length": int(history_length),
                    "governor_mode": "continuation_mode",
                    "candidate_count": int(len(rows)),
                    "viable_count": int(sum(bool(row.get("viable", False)) for row in rows)),
                    "decision_status": "selected_compact_representative" if selected else "blocked_no_viable_representative",
                    "selected_primitive_variant_id": "" if selected is None else str(selected.get("primitive_variant_id", "")),
                    "selected_score": float("-inf")
                    if selected is None
                    else float(selected.get("total_score_with_memory_and_exploration", selected.get("score", float("-inf")))),
                    "claim_status": "simulation_only_repeated_launch_validation_row",
                }
            )
    _write_csv(run_root / "metrics" / "validation_learning_curve.csv", pd.DataFrame(decision_rows))
    _write_csv(run_root / "metrics" / "validation_candidate_scores.csv", pd.DataFrame(candidate_rows))
    manifest = {
        "manifest_version": VALIDATION_VERSION,
        "project_title_version": PROJECT_TITLE_VERSION,
        "status": "complete",
        "run_id": int(config.run_id),
        "run_root": run_root.as_posix(),
        "library_root": Path(config.library_root).as_posix(),
        "outcome_root": Path(config.outcome_root).as_posix(),
        "history_lengths": list(HISTORY_LENGTHS),
        "library_size_case_ids": list(LIBRARY_SIZE_CASE_IDS),
        "primitive_timing_contract": primitive_timing_contract_row(),
        "claim_status": "simulation_only_repeated_launch_validation_not_hardware_ready",
        "blocked_claims": list(BLOCKED_CLAIMS),
    }
    _write_json(run_root / "manifests" / "repeated_launch_validation_manifest.json", manifest)
    _write_file_size_audit(run_root)
    _write_report(run_root, manifest)
    return {
        "status": "complete",
        "run_root": run_root.as_posix(),
        "history_lengths": list(HISTORY_LENGTHS),
        "library_size_case_ids": list(LIBRARY_SIZE_CASE_IDS),
    }


def _blocked_reason(config: RepeatedLaunchValidationConfig) -> str:
    for case_id in LIBRARY_SIZE_CASE_IDS:
        library_path = filesystem_path(Path(config.library_root) / "manifests" / f"{case_id}_primitive_library.json")
        if not library_path.is_file():
            return f"missing_library_size_case_manifest:{case_id}"
    outcome_path = filesystem_path(Path(config.outcome_root) / "metrics" / "outcome_model_table.csv")
    if not outcome_path.is_file():
        return "missing_outcome_model_table"
    outcome = pd.read_csv(outcome_path)
    missing_cases = set(LIBRARY_SIZE_CASE_IDS) - set(outcome.get("library_size_case_id", pd.Series(dtype=str)).astype(str))
    if missing_cases:
        return "outcome_model_missing_library_size_cases:" + ",".join(sorted(missing_cases))
    return ""


def _read_outcome_rows(outcome_root: Path, case_id: str) -> dict[str, dict[str, object]]:
    frame = pd.read_csv(filesystem_path(Path(outcome_root) / "metrics" / "outcome_model_table.csv"))
    frame = frame[frame["library_size_case_id"].astype(str) == str(case_id)]
    return {str(row["primitive_variant_id"]): row for row in frame.to_dict(orient="records")}


def _belief_after_history(history_length: int):
    belief = initial_directional_residual_lift_belief()
    for index in range(int(history_length)):
        belief = update_directional_residual_lift_belief(
            belief,
            DirectionalResidualObservation(
                x_m=0.0,
                y_m=0.0,
                z_m=1.5,
                direction_rad=0.0,
                lift_residual_m_s=0.02,
                energy_residual_m=0.01,
                dwell_residual_s=0.005,
            ),
        )
    return belief


def _validation_context(*, case_id: str, history_length: int) -> dict[str, object]:
    return {
        "context_id": f"v411_validation_{case_id}_h{history_length}",
        "W_layer": "W1",
        "environment_mode": "validation_repeated_launch",
        "start_state_family": "launch_gate",
        "governor_mode": "continuation_mode",
        "wall_margin_m": 0.5,
        "floor_margin_m": 0.5,
        "ceiling_margin_m": 0.5,
        "speed_margin_m_s": 1.0,
        "latency_case": "nominal",
        "library_size_case_id": str(case_id),
        "history_length": int(history_length),
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
        "library_size_case_ids": list(LIBRARY_SIZE_CASE_IDS),
        "blocked_claims": list(BLOCKED_CLAIMS),
    }
    _write_json(run_root / "manifests" / "repeated_launch_validation_manifest.json", manifest)
    _write_csv(run_root / "metrics" / "validation_learning_curve.csv", pd.DataFrame())
    _write_file_size_audit(run_root)
    _write_report(run_root, manifest)


def _write_report(run_root: Path, manifest: dict[str, object]) -> None:
    lines = [
        "# v4.11 Repeated-Launch Validation",
        "",
        f"- Status: `{manifest.get('status', '')}`",
        f"- Histories: `{','.join(str(value) for value in HISTORY_LENGTHS)}`",
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


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(filesystem_path(path).read_text(encoding="ascii"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    filesystem_path(path).parent.mkdir(parents=True, exist_ok=True)
    filesystem_path(path).write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="ascii")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    filesystem_path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(filesystem_path(path), index=False)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v4.11 repeated-launch learning-curve validation.")
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

