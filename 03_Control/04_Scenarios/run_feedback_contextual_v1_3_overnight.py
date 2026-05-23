from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

CONTROL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CONTROL_ROOT.parents[0]
for rel in ("02_Inner_Loop", "03_Primitives", "04_Scenarios"):
    path = CONTROL_ROOT / rel
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dense_archive_table_io import filesystem_path  # noqa: E402
from env_instance import environment_instance_for_mode, environment_metadata_from_instance  # noqa: E402
from env_surrogate import READY_STATUS, resolve_surrogate_binding  # noqa: E402
from evidence_stage_utils import (  # noqa: E402
    StageEvidenceStatus,
    runtime_projection_s,
    stage_status,
    write_evidence_status_manifest,
)
from run_ctx_archive import ContextArchiveConfig, run_contextual_archive_preflight  # noqa: E402
from run_primitive_selector_report import (  # noqa: E402
    SelectorReportConfig,
    run_primitive_selector_report,
)
from run_w2_replay import W2ReplayConfig, run_w2_replay  # noqa: E402
from run_w3_generalisation import W3GeneralisationConfig, run_w3_generalisation  # noqa: E402


@dataclass(frozen=True)
class OvernightEvidenceConfig:
    run_id: int
    output_root: Path
    r6_target_rows: int = 80_000
    r6_fallback_rows: int = 40_000
    r8_target_rows: int = 15_000
    r8_fallback_rows: int = 2_000
    r9_target_rows: int = 30_000
    r9_fallback_rows: int = 5_000
    candidate_chunk_size: int = 1000
    workers: int = 8
    max_workers: int = 8
    storage_format: str = "auto"
    compression_level: int = 1
    run_preflight_checks: bool = True
    run_r9: bool = True
    r6_time_budget_s: float = 8.0 * 3600.0
    r8_time_budget_s: float = 4.0 * 3600.0
    r9_time_budget_s: float = 6.0 * 3600.0


def parse_args(argv: list[str] | None = None) -> OvernightEvidenceConfig:
    parser = argparse.ArgumentParser(description="Run the staged v1.3 local evidence pipeline.")
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("03_Control/05_Results/feedback_contextual_v1_3"))
    parser.add_argument("--r6-target-rows", type=int, default=80_000)
    parser.add_argument("--r6-fallback-rows", type=int, default=40_000)
    parser.add_argument("--r8-target-rows", type=int, default=15_000)
    parser.add_argument("--r8-fallback-rows", type=int, default=2_000)
    parser.add_argument("--r9-target-rows", type=int, default=30_000)
    parser.add_argument("--r9-fallback-rows", type=int, default=5_000)
    parser.add_argument("--candidate-chunk-size", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--storage-format", default="auto")
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument("--skip-preflight-checks", action="store_true")
    parser.add_argument("--skip-r9", action="store_true")
    args = parser.parse_args(argv)
    return OvernightEvidenceConfig(
        run_id=int(args.run_id),
        output_root=Path(args.output_root),
        r6_target_rows=int(args.r6_target_rows),
        r6_fallback_rows=int(args.r6_fallback_rows),
        r8_target_rows=int(args.r8_target_rows),
        r8_fallback_rows=int(args.r8_fallback_rows),
        r9_target_rows=int(args.r9_target_rows),
        r9_fallback_rows=int(args.r9_fallback_rows),
        candidate_chunk_size=int(args.candidate_chunk_size),
        workers=int(args.workers),
        max_workers=int(args.max_workers),
        storage_format=str(args.storage_format),
        compression_level=int(args.compression_level),
        run_preflight_checks=not bool(args.skip_preflight_checks),
        run_r9=not bool(args.skip_r9),
    )


def run_feedback_contextual_v1_3_overnight(config: OvernightEvidenceConfig) -> dict[str, object]:
    root = Path(config.output_root)
    filesystem_path(root).mkdir(parents=True, exist_ok=True)
    statuses: list[StageEvidenceStatus] = []
    preflight = _run_preflight(config)
    first_chunk = _run_first_chunk_projection(config)
    r6_rows, r6_fallback_reason, r6_projection = _select_row_target(
        target_rows=config.r6_target_rows,
        fallback_rows=config.r6_fallback_rows,
        candidate_chunk_size=config.candidate_chunk_size,
        first_chunk_seconds=float(first_chunk["elapsed_s"]),
        actual_worker_count=int(first_chunk["actual_worker_count"]),
        budget_s=float(config.r6_time_budget_s),
        fallback_label="fallback_80k_to_40k_runtime_or_partition_limit",
    )

    r6_status = _run_r6_stage(config, r6_rows, r6_fallback_reason, r6_projection)
    statuses.append(r6_status)
    if r6_status.status in {"complete", "fallback"}:
        r7_status = _run_r7_stage(config, r6_status)
    else:
        r7_status = stage_status(stage="R7", status="deferred", fallback_reason="R6_not_available")
    statuses.append(r7_status)

    if r7_status.status in {"complete", "fallback", "partial"} and r6_status.table_manifest_path:
        r8_rows, r8_fallback_reason, r8_projection = _select_row_target(
            target_rows=config.r8_target_rows,
            fallback_rows=config.r8_fallback_rows,
            candidate_chunk_size=max(1, min(config.candidate_chunk_size, config.r8_fallback_rows)),
            first_chunk_seconds=max(float(first_chunk["elapsed_s"]) * 0.75, 0.001),
            actual_worker_count=int(first_chunk["actual_worker_count"]),
            budget_s=float(config.r8_time_budget_s),
            fallback_label="fallback_15k_to_2k_runtime_or_blocked_ratio_limit",
        )
        r8_status = _run_r8_stage(config, r6_status, r8_rows, r8_fallback_reason, r8_projection)
    else:
        r8_status = stage_status(stage="R8", status="deferred", fallback_reason="R7_not_available")
    statuses.append(r8_status)

    if config.run_r9 and r8_status.status in {"complete", "fallback"} and r8_status.table_manifest_path:
        r9_rows, r9_fallback_reason, r9_projection = _select_row_target(
            target_rows=config.r9_target_rows,
            fallback_rows=config.r9_fallback_rows,
            candidate_chunk_size=max(1, min(config.candidate_chunk_size, config.r9_fallback_rows)),
            first_chunk_seconds=max(float(first_chunk["elapsed_s"]), 0.001),
            actual_worker_count=int(first_chunk["actual_worker_count"]),
            budget_s=float(config.r9_time_budget_s),
            fallback_label="fallback_30k_to_5k_runtime_or_blocked_approximate_limit",
        )
        r9_status = _run_r9_stage(config, r8_status, r9_rows, r9_fallback_reason, r9_projection)
    else:
        r9_status = stage_status(stage="R9", status="deferred", fallback_reason="R8_not_available_or_R9_skipped")
    statuses.append(r9_status)

    status_manifest = root / "evidence_status" / "feedback_contextual_primitive_v1_3_status.json"
    write_evidence_status_manifest(
        status_manifest,
        statuses=statuses,
        run_label=f"feedback_contextual_primitive_v1_3_run_{config.run_id:03d}",
    )
    return {
        "output_root": root,
        "preflight": preflight,
        "first_chunk": first_chunk,
        "statuses": statuses,
        "status_manifest": status_manifest,
    }


def _run_preflight(config: OvernightEvidenceConfig) -> dict[str, object]:
    surrogate = {
        "R6": _surrogate_status(("W0", "W1")),
        "R8": _surrogate_status(("W2",)),
        "R9": _surrogate_status(("W3",)),
    }
    commands: list[dict[str, object]] = []
    if config.run_preflight_checks:
        commands.append(_run_command([sys.executable, "-m", "pytest", "-q", "03_Control/tests"]))
        commands.append(_run_command(["git", "diff", "--check"]))
    return {
        "surrogate_status": surrogate,
        "commands": commands,
        "result_root_clean_or_local_output": True,
    }


def _run_first_chunk_projection(config: OvernightEvidenceConfig) -> dict[str, object]:
    chunk_rows = max(1, min(int(config.candidate_chunk_size), 1000))
    started = time.perf_counter()
    result = run_contextual_archive_preflight(
        ContextArchiveConfig(
            run_id=int(config.run_id) * 10 + 1,
            rows=chunk_rows,
            seed=int(config.run_id) * 10 + 1,
            w_layers=("W0", "W1"),
            env_modes=("dry_air", "gaussian_single", "gaussian_four", "fan_shift", "power_scale"),
            candidate_chunk_size=chunk_rows,
            workers=config.workers,
            max_workers=config.max_workers,
            storage_format=config.storage_format,
            compression_level=config.compression_level,
            resume=True,
            repair_incomplete=True,
            dry_run_schedule=False,
            stop_after_chunks=1,
            continue_on_chunk_failure=False,
            output_root=Path(config.output_root) / "proj" / "r6",
            rollout_backend="model_backed_feedback",
        )
    )
    elapsed = max(time.perf_counter() - started, 0.001)
    run_manifest_path = Path(result["run_manifest"])
    payload = json.loads(filesystem_path(run_manifest_path).read_text(encoding="ascii"))
    return {
        "run_root": Path(result["run_root"]).as_posix(),
        "elapsed_s": elapsed,
        "row_count": int(chunk_rows),
        "actual_worker_count": int(payload.get("actual_worker_count", payload.get("selected_worker_count", 1))),
        "partition_count": int(result["partition_count"]),
    }


def _select_row_target(
    *,
    target_rows: int,
    fallback_rows: int,
    candidate_chunk_size: int,
    first_chunk_seconds: float,
    actual_worker_count: int,
    budget_s: float,
    fallback_label: str,
) -> tuple[int, str, float]:
    projection = runtime_projection_s(
        total_rows=int(target_rows),
        first_chunk_rows=int(candidate_chunk_size),
        first_chunk_seconds=float(first_chunk_seconds),
        actual_worker_count=int(actual_worker_count),
    )
    if projection <= float(budget_s):
        return int(target_rows), "", projection
    fallback_projection = runtime_projection_s(
        total_rows=int(fallback_rows),
        first_chunk_rows=int(candidate_chunk_size),
        first_chunk_seconds=float(first_chunk_seconds),
        actual_worker_count=int(actual_worker_count),
    )
    return int(fallback_rows), fallback_label, fallback_projection


def _run_r6_stage(
    config: OvernightEvidenceConfig,
    row_count: int,
    fallback_reason: str,
    projection_s: float,
) -> StageEvidenceStatus:
    try:
        result = run_contextual_archive_preflight(
            ContextArchiveConfig(
                run_id=int(config.run_id) * 10 + 6,
                rows=int(row_count),
                seed=int(config.run_id) * 10 + 6,
                w_layers=("W0", "W1"),
                env_modes=("dry_air", "gaussian_single", "gaussian_four", "fan_shift", "power_scale"),
                candidate_chunk_size=int(config.candidate_chunk_size),
                workers=config.workers,
                max_workers=config.max_workers,
                storage_format=config.storage_format,
                compression_level=config.compression_level,
                resume=True,
                repair_incomplete=True,
                dry_run_schedule=False,
                stop_after_chunks=None,
                continue_on_chunk_failure=False,
                output_root=Path(config.output_root) / "r6",
                rollout_backend="model_backed_feedback",
            )
        )
        run_root = Path(result["run_root"])
        ratio = _ratio_from_csv(run_root / "metrics" / "blocked_or_approximate_ratio_summary.csv", "blocked_ratio")
        status = "fallback" if fallback_reason else "complete"
        if ratio > 0.50:
            status = "blocked"
        elif ratio > 0.25:
            status = "fallback"
        return stage_status(
            stage="R6",
            status=status,
            run_root=run_root,
            row_count=int(row_count),
            table_manifest_path=Path(result["table_manifest"]),
            fallback_reason=fallback_reason,
            blocked_ratio=ratio,
            file_size_status=_file_size_status(run_root / "metrics" / "file_size_audit.csv"),
            coverage_status="written",
            surrogate_status="W0_W1_ready",
            runtime_projection_s=projection_s,
            claim_status="simulation_only_R6_W0_W1_contextual_archive",
        )
    except Exception as exc:
        return stage_status(
            stage="R6",
            status="blocked",
            fallback_reason=f"R6_exception:{type(exc).__name__}:{exc}",
            surrogate_status="W0_W1_stage_blocked",
            runtime_projection_s=projection_s,
        )


def _run_r7_stage(config: OvernightEvidenceConfig, r6: StageEvidenceStatus) -> StageEvidenceStatus:
    try:
        result = run_primitive_selector_report(
            SelectorReportConfig(
                run_id=int(config.run_id) * 10 + 7,
                archive_table=Path(r6.table_manifest_path),
                output_root=Path(config.output_root) / "r7",
                governor_modes=("continuation", "terminal_episode"),
                max_rows=0,
            )
        )
        manifest = json.loads(filesystem_path(Path(result["manifest"])).read_text(encoding="ascii"))
        return stage_status(
            stage="R7",
            status="complete" if manifest.get("R7_selector_report_complete") else "partial",
            run_root=Path(result["run_root"]),
            row_count=int(manifest.get("training_row_count", 0)),
            table_manifest_path=r6.table_manifest_path,
            blocked_ratio=float(manifest.get("blocked_ratio", 0.0)),
            file_size_status=str(manifest.get("file_size_status", "unknown")),
            coverage_status="written",
            surrogate_status="not_applicable_R7",
            claim_status=str(manifest.get("claim_status", "simulation_only_selector_report")),
        )
    except Exception as exc:
        return stage_status(stage="R7", status="blocked", fallback_reason=f"R7_exception:{type(exc).__name__}:{exc}")


def _run_r8_stage(
    config: OvernightEvidenceConfig,
    r6: StageEvidenceStatus,
    row_count: int,
    fallback_reason: str,
    projection_s: float,
) -> StageEvidenceStatus:
    try:
        result = run_w2_replay(
            W2ReplayConfig(
                run_id=int(config.run_id) * 10 + 8,
                output_root=Path(config.output_root) / "r8",
                source_archive=Path(r6.table_manifest_path),
                target_rows=int(row_count),
                fallback_rows=min(int(row_count), int(config.r8_fallback_rows)),
                storage_format=config.storage_format,
                compression_level=config.compression_level,
            )
        )
        manifest = json.loads(filesystem_path(Path(result["manifest"])).read_text(encoding="ascii"))
        return stage_status(
            stage="R8",
            status=str(manifest.get("replay_status", "partial")),
            run_root=Path(result["run_root"]),
            row_count=int(manifest.get("replayed_row_count", 0)),
            table_manifest_path=Path(result.get("table_manifest", "")) if result.get("table_manifest") else "",
            fallback_reason=fallback_reason,
            blocked_ratio=float(manifest.get("blocked_ratio", 1.0)),
            file_size_status=str(manifest.get("file_size_status", "unknown")),
            coverage_status="written",
            surrogate_status="W2_ready_or_stage_local_blocked",
            runtime_projection_s=projection_s,
            claim_status=str(manifest.get("claim_status", "simulation_only_w2_replay")),
        )
    except Exception as exc:
        return stage_status(stage="R8", status="blocked", fallback_reason=f"R8_exception:{type(exc).__name__}:{exc}")


def _run_r9_stage(
    config: OvernightEvidenceConfig,
    r8: StageEvidenceStatus,
    row_count: int,
    fallback_reason: str,
    projection_s: float,
) -> StageEvidenceStatus:
    try:
        result = run_w3_generalisation(
            W3GeneralisationConfig(
                run_id=int(config.run_id) * 10 + 9,
                output_root=Path(config.output_root) / "r9",
                source_replay=Path(r8.table_manifest_path),
                target_rows=int(row_count),
                fallback_rows=min(int(row_count), int(config.r9_fallback_rows)),
                execute_replay=True,
                storage_format=config.storage_format,
                compression_level=config.compression_level,
            )
        )
        manifest = json.loads(filesystem_path(Path(result["manifest"])).read_text(encoding="ascii"))
        return stage_status(
            stage="R9",
            status=str(manifest.get("generalisation_status", "partial")),
            run_root=Path(result["run_root"]),
            row_count=int(manifest.get("case_count", 0)),
            table_manifest_path=Path(result.get("table_manifest", "")) if result.get("table_manifest") else "",
            fallback_reason=fallback_reason,
            blocked_ratio=float(manifest.get("blocked_ratio", 1.0)),
            approximate_ratio=float(manifest.get("approximate_ratio", 0.0)),
            file_size_status=str(manifest.get("file_size_status", "unknown")),
            coverage_status="written",
            surrogate_status="W3_ready_or_stage_local_blocked",
            runtime_projection_s=projection_s,
            claim_status=str(manifest.get("claim_status", "simulation_only_w3_generalisation")),
        )
    except Exception as exc:
        return stage_status(stage="R9", status="blocked", fallback_reason=f"R9_exception:{type(exc).__name__}:{exc}")


def _surrogate_status(layers: tuple[str, ...]) -> dict[str, str]:
    statuses = {}
    for layer in layers:
        mode = {
            "W0": "dry_air",
            "W1": "gaussian_single",
            "W2": "gaussian_single",
            "W3": "w3_randomised",
        }[layer]
        instance = environment_instance_for_mode(layer, mode, 1)
        binding = resolve_surrogate_binding(layer, environment_metadata_from_instance(instance), randomisation_seed=1)
        statuses[layer] = binding.surrogate_binding_status if binding.surrogate_binding_status == READY_STATUS else binding.blocked_reason
    return statuses


def _run_command(command: list[str]) -> dict[str, object]:
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return {
        "command": " ".join(command),
        "returncode": int(completed.returncode),
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def _ratio_from_csv(path: Path, column: str) -> float:
    if not filesystem_path(path).is_file():
        return 1.0
    frame = pd.read_csv(filesystem_path(path))
    if frame.empty or column not in frame.columns:
        return 1.0
    return float(frame[column].iloc[0])


def _file_size_status(path: Path) -> str:
    if not filesystem_path(path).is_file():
        return "not_checked"
    frame = pd.read_csv(filesystem_path(path))
    if "above_100mb" in frame.columns and frame["above_100mb"].astype(bool).any():
        return "fail_oversized_file"
    return "pass"


def main(argv: list[str] | None = None) -> int:
    run_feedback_contextual_v1_3_overnight(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
