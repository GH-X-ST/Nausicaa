from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

CONTROL_ROOT = Path(__file__).resolve().parents[1]
for rel in ("02_Inner_Loop", "03_Primitives", "04_Scenarios"):
    path = CONTROL_ROOT / rel
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dense_archive_runtime import dense_run_manifest_fields, worker_count_decision  # noqa: E402
from dense_archive_table_io import (  # noqa: E402
    TableManifest,
    TablePartition,
    file_sha256,
    filesystem_path,
    read_table_partition,
    resolve_storage_format,
    write_table_manifest,
    write_table_partition,
)
from env_ctx import build_environment_context  # noqa: E402
from env_instance import environment_instance_for_mode, environment_metadata_from_instance  # noqa: E402
from env_surrogate import READY_STATUS, resolve_surrogate_binding, wind_field_for_binding  # noqa: E402
from evidence_stage_utils import write_coverage_summary, write_file_size_audit  # noqa: E402
from controller_registry import (  # noqa: E402
    SELECTED_CONTROLLER_STATUS,
    SMOKE_SELECTED_CONTROLLER_STATUS,
    controller_registry_row,
    write_selected_controller_registry,
)
from evidence_status import registry_claim_status_for  # noqa: E402
from lqr_controller import LQR_SYNTHESIS_SOLVED, synthesize_lqr_controller, synthesis_audit_row  # noqa: E402
from lqr_tuning import (  # noqa: E402
    FALLBACK_CANDIDATES_PER_PRIMITIVE,
    FALLBACK_PAIRED_TESTS_PER_CANDIDATE,
    HARD_GATE_LABELS,
    PREFERRED_CANDIDATES_PER_PRIMITIVE,
    PREFERRED_PAIRED_TESTS_PER_CANDIDATE,
    SOFT_OBJECTIVE_TERMS,
    candidate_weight_specs,
    lqr_tuning_schedule,
    tuning_candidate_row,
    tuning_candidates_for_primitive,
)
from prim_cat import active_primitive_catalogue  # noqa: E402
from prim_roll import RolloutConfig, blocked_rollout_evidence, rollout_with_context_row, simulate_primitive_rollout  # noqa: E402
from state_sampling import archive_state_sample_for_row, archive_state_sample_row  # noqa: E402


@dataclass(frozen=True)
class LQRTuningSweepConfig:
    run_id: int
    output_root: Path
    rows: int = 500
    seed: int = 1
    candidate_count: int = 16
    paired_tests_per_candidate: int = 50
    candidate_chunk_size: int = 125
    workers: str | int = "8"
    max_workers: int = 8
    storage_format: str = "auto"
    compression_level: int = 1
    resume: bool = True
    repair_incomplete: bool = False
    dry_run_schedule: bool = False
    stop_after_chunks: int | None = None
    continue_on_chunk_failure: bool = False


def parse_args(argv: list[str] | None = None) -> LQRTuningSweepConfig:
    parser = argparse.ArgumentParser(description="Run W0/W1 grouped Q/R LQR tuning sweep smoke.")
    parser.add_argument("--run-id", type=int, default=100)
    parser.add_argument("--output-root", type=Path, default=Path("03_Control/05_Results/lqr_contextual_v1_0/r6"))
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--paired-tests-per-candidate", type=int, default=50)
    parser.add_argument("--candidate-chunk-size", "--chunk-size", dest="candidate_chunk_size", type=int, default=125)
    parser.add_argument("--workers", default="8")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--storage-format", default="auto")
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--repair-incomplete", action="store_true")
    parser.add_argument("--dry-run-schedule", action="store_true")
    parser.add_argument("--stop-after-chunks", type=int, default=None)
    parser.add_argument("--continue-on-chunk-failure", action="store_true")
    args = parser.parse_args(argv)
    return LQRTuningSweepConfig(
        run_id=int(args.run_id),
        output_root=Path(args.output_root),
        rows=int(args.rows),
        seed=int(args.seed),
        candidate_count=int(args.candidate_count),
        paired_tests_per_candidate=int(args.paired_tests_per_candidate),
        candidate_chunk_size=int(args.candidate_chunk_size),
        workers=args.workers,
        max_workers=int(args.max_workers),
        storage_format=str(args.storage_format),
        compression_level=int(args.compression_level),
        resume=bool(args.resume),
        repair_incomplete=bool(args.repair_incomplete),
        dry_run_schedule=bool(args.dry_run_schedule),
        stop_after_chunks=args.stop_after_chunks,
        continue_on_chunk_failure=bool(args.continue_on_chunk_failure),
    )


def run_lqr_tuning_sweep(config: LQRTuningSweepConfig) -> dict[str, object]:
    run_root = Path(config.output_root) / f"tune_{config.run_id:03d}"
    for rel in ("manifests", "metrics", "reports", "tables", "chunk_manifests"):
        filesystem_path(run_root / rel).mkdir(parents=True, exist_ok=True)
    storage_format = resolve_storage_format(config.storage_format)
    worker_decision = worker_count_decision(config.workers, max_workers=config.max_workers)
    schedule = lqr_tuning_schedule(
        candidate_count=int(config.candidate_count),
        paired_tests_per_candidate=int(config.paired_tests_per_candidate),
        fallback_mode=(
            int(config.candidate_count) <= FALLBACK_CANDIDATES_PER_PRIMITIVE
            or int(config.paired_tests_per_candidate) <= FALLBACK_PAIRED_TESTS_PER_CANDIDATE
        ),
    )
    row_count = _effective_row_count(config)
    chunk_specs = _tuning_chunk_specs(row_count, int(config.candidate_chunk_size))
    manifest = _base_manifest(
        config=config,
        schedule=schedule,
        chunk_specs=chunk_specs,
        worker_decision=worker_decision,
    )
    filesystem_path(run_root / "manifests" / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="ascii",
    )
    if config.dry_run_schedule:
        pd.DataFrame(_scheduled_chunk_rows(chunk_specs)).to_csv(
            filesystem_path(run_root / "metrics" / "chunk_summary.csv"),
            index=False,
        )
        pd.DataFrame([manifest]).to_csv(filesystem_path(run_root / "metrics" / "runtime_summary.csv"), index=False)
        return {"run_root": run_root, "run_manifest": run_root / "manifests" / "run_manifest.json"}

    candidate_records, synthesis_rows, candidate_rows = _candidate_artifacts(config)
    execution = _execute_tuning_chunks(
        config=config,
        run_root=run_root,
        chunk_specs=chunk_specs,
        storage_format=storage_format,
        candidate_records=candidate_records,
        selected_worker_count=worker_decision.selected_worker_count,
    )
    partitions = execution["partitions"]
    validated_frame = _validated_tuning_frame(run_root, partitions)
    registry_status = _registry_status_for_tuning_run(
        config,
        schedule,
        validated_frame,
        failures=execution["failures"],
    )
    registry_claim_status = registry_claim_status_for(registry_status)
    evidence_reason = (
        f"eligible_tuning_registry_{registry_status}"
        if registry_status in {"complete", "accepted_fallback"}
        else ("blocked_tuning_registry" if registry_status == "blocked" else "debug_smoke_incomplete")
    )
    partitions = _rewrite_partitions_with_registry_status(
        config=config,
        run_root=run_root,
        partitions=partitions,
        chunk_specs=chunk_specs,
        storage_format=storage_format,
        registry_status=registry_status,
        registry_claim_status=registry_claim_status,
        evidence_reason=evidence_reason,
    )
    validated_frame = _validated_tuning_frame(run_root, partitions)
    chunk_records = _chunk_records_for_summary(
        run_root=run_root,
        chunk_specs=chunk_specs,
        partitions=partitions,
        skipped=execution["skipped"],
        failed=execution["failures"],
        corrupt=execution["corrupt"],
    )
    pd.DataFrame(synthesis_rows).to_csv(
        filesystem_path(run_root / "metrics" / "lqr_synthesis_audit.csv"),
        index=False,
    )
    pd.DataFrame(candidate_rows).to_csv(
        filesystem_path(run_root / "metrics" / "qr_candidate_rankings.csv"),
        index=False,
    )
    pd.DataFrame(chunk_records).to_csv(
        filesystem_path(run_root / "metrics" / "chunk_summary.csv"),
        index=False,
    )
    manifest.update(
        {
            **execution["worker_execution"],
            "registry_status": registry_status,
            "registry_claim_status": registry_claim_status,
            "archive_evidence_status": registry_status,
            "evidence_eligibility_reason": evidence_reason,
            "selected_controller_registry": (run_root / "metrics" / "selected_lqr_controllers.csv").as_posix(),
            "completed_chunk_count": int(sum(1 for row in chunk_records if row["status"] == "complete")),
            "skipped_chunk_count": int(sum(1 for row in chunk_records if row["status"] == "skipped")),
            "failed_chunk_count": int(sum(1 for row in chunk_records if row["status"] == "failed")),
            "corrupt_chunk_count": int(sum(1 for row in chunk_records if row["status"] == "corrupt")),
        }
    )
    filesystem_path(run_root / "manifests" / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="ascii",
    )
    registry_rows = _selected_controller_registry_rows(
        candidate_records,
        validated_frame,
        registry_status=registry_status,
        registry_claim_status=registry_claim_status,
    )
    write_selected_controller_registry(
        rows=registry_rows,
        csv_path=run_root / "metrics" / "selected_lqr_controllers.csv",
        json_path=run_root / "manifests" / "selected_lqr_controllers.json",
    )
    write_table_manifest(
        run_root / "manifests" / "table_manifest.json",
        TableManifest(
            run_id=int(config.run_id),
            root=run_root.as_posix(),
            storage_format=storage_format,
            tables=tuple(partitions),
        ),
    )
    write_coverage_summary(
        run_root / "metrics" / "coverage_summary.csv",
        validated_frame,
        columns=(
            "primitive_id",
            "controller_id",
            "W_layer",
            "latency_case",
            "start_state_family",
            "environment_id",
            "boundary_use_class",
            "continuation_valid",
            "episode_terminal_useful",
            "controller_selection_status",
            "candidate_weight_label",
            "registry_status",
            "registry_claim_status",
            "hard_gate_status",
        ),
    )
    _write_objective_summary(run_root / "metrics" / "objective_term_summary.csv", validated_frame)
    write_file_size_audit(run_root, run_root / "metrics" / "file_size_audit.csv")
    _write_runtime_summary(
        run_root / "metrics" / "runtime_summary.csv",
        manifest,
        row_count=len(validated_frame),
        status=registry_status,
    )
    filesystem_path(run_root / "reports" / "claim_boundary_report.md").write_text(
        "# LQR Tuning Claim Boundary\n\nSimulation-only W0/W1 Q/R tuning smoke. W2/W3 are replay-only.\n",
        encoding="ascii",
    )
    return {
        "run_root": run_root,
        "run_manifest": run_root / "manifests" / "run_manifest.json",
        "table_manifest": run_root / "manifests" / "table_manifest.json",
        "synthesis_audit": run_root / "metrics" / "lqr_synthesis_audit.csv",
        "candidate_rankings": run_root / "metrics" / "qr_candidate_rankings.csv",
        "selected_controller_registry": run_root / "metrics" / "selected_lqr_controllers.csv",
    }


def _base_manifest(
    *,
    config: LQRTuningSweepConfig,
    schedule,
    chunk_specs: list[dict[str, int]],
    worker_decision,
) -> dict[str, object]:
    return {
        **dense_run_manifest_fields(
            run_stage="R6_W0_W1_LQR_QR_tuning",
            environment_context="W0_W1_paired_start_keys",
            worker_decision=worker_decision,
        ),
        "run_id": int(config.run_id),
        "rows_requested": int(config.rows),
        "candidate_count": int(config.candidate_count),
        "paired_tests_per_candidate": int(config.paired_tests_per_candidate),
        "planned_rows": int(schedule.planned_rows),
        "scheduled_chunk_count": len(chunk_specs),
        "chunked_output": True,
        "tuning_rows_table": "lqr_tuning_rows",
        "hard_gates": list(HARD_GATE_LABELS),
        "soft_objective_terms": list(SOFT_OBJECTIVE_TERMS),
        "raw_K_tuning_allowed": False,
        "W0_W1_tune_controller_ids": True,
        "W2_W3_replay_only": True,
        "dry_run_schedule": bool(config.dry_run_schedule),
        "resume": bool(config.resume),
        "repair_incomplete": bool(config.repair_incomplete),
        "continue_on_chunk_failure": bool(config.continue_on_chunk_failure),
    }


def _candidate_artifacts(config: LQRTuningSweepConfig) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    primitives = active_primitive_catalogue()
    synthesis_rows = [synthesis_audit_row(primitive) for primitive in primitives]
    candidate_rows: list[dict[str, object]] = []
    candidate_records: list[dict[str, object]] = []
    for primitive in primitives:
        for candidate_index, weight_spec in enumerate(
            candidate_weight_specs(
                primitive_id=primitive.primitive_id,
                candidate_count=int(config.candidate_count),
            )
        ):
            controller = synthesize_lqr_controller(primitive, weight_spec=weight_spec)
            candidate_records.append(
                {
                    "primitive": primitive,
                    "controller": controller,
                    "candidate_index": int(candidate_index),
                    "candidate_weight_label": weight_spec.weight_label,
                }
            )
        candidate_rows.extend(
            tuning_candidate_row(candidate)
            for candidate in tuning_candidates_for_primitive(
                primitive,
                candidate_count=int(config.candidate_count),
            )
        )
    return candidate_records, synthesis_rows, candidate_rows


def _scheduled_chunk_rows(chunk_specs: list[dict[str, int]]) -> list[dict[str, object]]:
    return [
        {
            "chunk_index": int(chunk["chunk_index"]),
            "start_row": int(chunk["start_row"]),
            "row_count": int(chunk["row_count"]),
            "status": "scheduled",
        }
        for chunk in chunk_specs
    ]


def _execute_tuning_chunks(
    *,
    config: LQRTuningSweepConfig,
    run_root: Path,
    chunk_specs: list[dict[str, int]],
    storage_format: str,
    candidate_records: list[dict[str, object]],
    selected_worker_count: int,
) -> dict[str, object]:
    partitions: list[TablePartition] = []
    skipped: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    corrupt: list[dict[str, object]] = []
    pending: list[dict[str, int]] = []

    for chunk in chunk_specs:
        try:
            status = _tuning_chunk_status(config, run_root=run_root, chunk=chunk, storage_format=storage_format)
            if status == "complete" and config.resume:
                partition = _partition_from_tuning_manifest(config, run_root=run_root, chunk=chunk, storage_format=storage_format)
                partitions.append(partition)
                skipped.append({**chunk, "status": "skipped", "relative_path": partition.relative_path})
                continue
            if status == "corrupt":
                corrupt.append({**chunk, "status": "corrupt"})
                if config.repair_incomplete:
                    _remove_tuning_chunk_outputs(run_root, chunk=chunk, storage_format=storage_format)
                else:
                    raise ValueError(f"corrupt tuning chunk c{int(chunk['chunk_index']):05d}")
            pending.append(chunk)
        except Exception as exc:
            if config.repair_incomplete and "corrupt tuning chunk" not in str(exc):
                _remove_tuning_chunk_outputs(run_root, chunk=chunk, storage_format=storage_format)
                pending.append(chunk)
                continue
            failures.append(_tuning_failure_row(chunk, exc))
            if not config.continue_on_chunk_failure:
                raise

    worker_count = min(max(1, int(selected_worker_count)), max(1, len(pending) or 1))
    if worker_count > 1 and pending:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _run_tuning_chunk,
                    config=config,
                    run_root=run_root,
                    chunk=chunk,
                    storage_format=storage_format,
                    candidate_records=candidate_records,
                ): chunk
                for chunk in pending
            }
            for future in as_completed(futures):
                chunk = futures[future]
                try:
                    partitions.append(future.result())
                except Exception as exc:
                    failures.append(_tuning_failure_row(chunk, exc))
                    if not config.continue_on_chunk_failure:
                        raise
    else:
        for chunk in pending:
            try:
                partitions.append(
                    _run_tuning_chunk(
                        config=config,
                        run_root=run_root,
                        chunk=chunk,
                        storage_format=storage_format,
                        candidate_records=candidate_records,
                    )
                )
            except Exception as exc:
                failures.append(_tuning_failure_row(chunk, exc))
                if not config.continue_on_chunk_failure:
                    raise

    return {
        "partitions": sorted(partitions, key=lambda item: item.relative_path),
        "skipped": skipped,
        "failures": failures,
        "corrupt": corrupt,
        "worker_execution": {
            "chunk_execution_backend": "process_pool" if worker_count > 1 and pending else "single_worker",
            "actual_worker_count": int(worker_count),
            "parallel_chunk_count": int(len(pending)),
            "worker_enabled": bool(worker_count > 1 and pending),
        },
    }


def _run_tuning_chunk(
    *,
    config: LQRTuningSweepConfig,
    run_root: Path,
    chunk: dict[str, int],
    storage_format: str,
    candidate_records: list[dict[str, object]],
) -> TablePartition:
    started = time.perf_counter()
    rows = _smoke_rows(
        config,
        candidate_records,
        start_row=int(chunk["start_row"]),
        row_count=int(chunk["row_count"]),
    )
    frame = pd.DataFrame(rows)
    partition = write_table_partition(
        frame,
        _tuning_partition_path(run_root, chunk_index=int(chunk["chunk_index"]), storage_format=storage_format),
        storage_format=storage_format,
        compression_level=config.compression_level,
    )
    _write_tuning_chunk_manifest(
        config=config,
        run_root=run_root,
        chunk_index=int(chunk["chunk_index"]),
        chunk_count=len(_tuning_chunk_specs(_effective_row_count(config), int(config.candidate_chunk_size))),
        start_row=int(chunk["start_row"]),
        row_count=int(chunk["row_count"]),
        partition=partition,
        storage_format=storage_format,
        registry_status="pending_registry",
        columns=tuple(str(column) for column in frame.columns),
        elapsed_s=time.perf_counter() - started,
    )
    return partition


def _validated_tuning_frame(run_root: Path, partitions: list[TablePartition]) -> pd.DataFrame:
    frames = [
        read_table_partition(
            Path(run_root) / "tables" / partition.relative_path,
            storage_format=partition.storage_format,
        )
        for partition in partitions
    ]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _rewrite_partitions_with_registry_status(
    *,
    config: LQRTuningSweepConfig,
    run_root: Path,
    partitions: list[TablePartition],
    chunk_specs: list[dict[str, int]],
    storage_format: str,
    registry_status: str,
    registry_claim_status: str,
    evidence_reason: str,
) -> list[TablePartition]:
    rewritten: list[TablePartition] = []
    chunk_by_index = {int(chunk["chunk_index"]): chunk for chunk in chunk_specs}
    for partition in partitions:
        chunk_index = _chunk_index_from_partition(partition)
        chunk = chunk_by_index[chunk_index]
        if not _partition_needs_registry_rewrite(
            config,
            run_root=run_root,
            chunk=chunk,
            storage_format=storage_format,
            registry_status=registry_status,
        ):
            rewritten.append(partition)
            continue
        frame = read_table_partition(
            Path(run_root) / "tables" / partition.relative_path,
            storage_format=partition.storage_format,
        )
        if not frame.empty:
            frame["registry_status"] = registry_status
            frame["registry_claim_status"] = registry_claim_status
            frame["archive_evidence_status"] = registry_status
            frame["evidence_eligibility_reason"] = evidence_reason
        updated = write_table_partition(
            frame,
            _tuning_partition_path(run_root, chunk_index=chunk_index, storage_format=storage_format),
            storage_format=storage_format,
            compression_level=config.compression_level,
        )
        _write_tuning_chunk_manifest(
            config=config,
            run_root=run_root,
            chunk_index=chunk_index,
            chunk_count=len(chunk_specs),
            start_row=int(chunk["start_row"]),
            row_count=int(chunk["row_count"]),
            partition=updated,
            storage_format=storage_format,
            registry_status=registry_status,
            columns=tuple(str(column) for column in frame.columns),
            elapsed_s=0.0,
        )
        rewritten.append(updated)
    return sorted(rewritten, key=lambda item: item.relative_path)


def _partition_needs_registry_rewrite(
    config: LQRTuningSweepConfig,
    *,
    run_root: Path,
    chunk: dict[str, int],
    storage_format: str,
    registry_status: str,
) -> bool:
    try:
        manifest = _validate_tuning_chunk_manifest(config, run_root=run_root, chunk=chunk, storage_format=storage_format)
    except Exception:
        return True
    if str(manifest.get("registry_status", "")) != str(registry_status):
        return True
    columns = set(str(column) for column in manifest.get("columns", []))
    return not {"registry_status", "registry_claim_status", "archive_evidence_status", "evidence_eligibility_reason"}.issubset(columns)


def _chunk_records_for_summary(
    *,
    run_root: Path,
    chunk_specs: list[dict[str, int]],
    partitions: list[TablePartition],
    skipped: list[dict[str, object]],
    failed: list[dict[str, object]],
    corrupt: list[dict[str, object]],
) -> list[dict[str, object]]:
    by_partition = {_chunk_index_from_partition(partition): partition for partition in partitions}
    skipped_indices = {int(row["chunk_index"]) for row in skipped}
    failed_by_index = {int(row["chunk_index"]): row for row in failed}
    corrupt_indices = {int(row["chunk_index"]) for row in corrupt}
    rows: list[dict[str, object]] = []
    for chunk in chunk_specs:
        chunk_index = int(chunk["chunk_index"])
        partition = by_partition.get(chunk_index)
        status = "complete"
        if chunk_index in skipped_indices:
            status = "skipped"
        if chunk_index in failed_by_index:
            status = "failed"
        elif chunk_index in corrupt_indices and partition is None:
            status = "corrupt"
        rows.append(
            {
                "chunk_index": chunk_index,
                "start_row": int(chunk["start_row"]),
                "row_count": int(chunk["row_count"]),
                "status": status,
                "relative_path": "" if partition is None else partition.relative_path,
                "checksum_sha256": "" if partition is None else partition.checksum_sha256,
                "error_type": "" if chunk_index not in failed_by_index else failed_by_index[chunk_index]["error_type"],
                "error": "" if chunk_index not in failed_by_index else failed_by_index[chunk_index]["error"],
            }
        )
    return rows


def _chunk_index_from_partition(partition: TablePartition) -> int:
    name = Path(partition.relative_path).name
    return int(name.split(".")[0].lstrip("c"))


def _tuning_failure_row(chunk: dict[str, int], exc: Exception) -> dict[str, object]:
    return {
        "chunk_index": int(chunk["chunk_index"]),
        "start_row": int(chunk["start_row"]),
        "row_count": int(chunk["row_count"]),
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def _effective_row_count(config: LQRTuningSweepConfig) -> int:
    row_count = max(0, int(config.rows))
    if config.stop_after_chunks is None:
        return row_count
    chunk_limit = max(0, int(config.stop_after_chunks)) * max(1, int(config.candidate_chunk_size))
    return min(row_count, chunk_limit)


def _tuning_chunk_specs(row_count: int, chunk_size: int) -> list[dict[str, int]]:
    total = max(0, int(row_count))
    size = max(1, int(chunk_size))
    chunks: list[dict[str, int]] = []
    for chunk_index, start in enumerate(range(0, total, size)):
        chunks.append(
            {
                "chunk_index": int(chunk_index),
                "start_row": int(start),
                "row_count": int(min(size, total - start)),
            }
        )
    return chunks


def _tuning_partition_path(run_root: Path, *, chunk_index: int, storage_format: str) -> Path:
    return (
        Path(run_root)
        / "tables"
        / "lqr_tuning_rows"
        / f"c{int(chunk_index):05d}.{_extension(storage_format)}"
    )


def _tuning_chunk_manifest_path(run_root: Path, *, chunk_index: int) -> Path:
    return Path(run_root) / "chunk_manifests" / "lqr_tuning_rows" / f"c{int(chunk_index):05d}.json"


def _tuning_chunk_status(
    config: LQRTuningSweepConfig,
    *,
    run_root: Path,
    chunk: dict[str, int],
    storage_format: str,
) -> str:
    partition_path = _tuning_partition_path(run_root, chunk_index=int(chunk["chunk_index"]), storage_format=storage_format)
    manifest_path = _tuning_chunk_manifest_path(run_root, chunk_index=int(chunk["chunk_index"]))
    partition_exists = filesystem_path(partition_path).is_file()
    manifest_exists = filesystem_path(manifest_path).is_file()
    if not partition_exists and not manifest_exists:
        return "missing"
    if partition_exists != manifest_exists:
        return "corrupt"
    try:
        _validate_tuning_chunk_manifest(config, run_root=run_root, chunk=chunk, storage_format=storage_format)
    except Exception:
        return "corrupt"
    return "complete"


def _validate_tuning_chunk_manifest(
    config: LQRTuningSweepConfig,
    *,
    run_root: Path,
    chunk: dict[str, int],
    storage_format: str,
) -> dict[str, object]:
    partition_path = _tuning_partition_path(run_root, chunk_index=int(chunk["chunk_index"]), storage_format=storage_format)
    manifest_path = _tuning_chunk_manifest_path(run_root, chunk_index=int(chunk["chunk_index"]))
    if not filesystem_path(partition_path).is_file() or not filesystem_path(manifest_path).is_file():
        raise FileNotFoundError("missing tuning chunk partition or manifest")
    payload = json.loads(filesystem_path(manifest_path).read_text(encoding="ascii"))
    expected = {
        "status": "complete",
        "run_stage": "R6_W0_W1_LQR_QR_tuning",
        "run_id": int(config.run_id),
        "chunk_index": int(chunk["chunk_index"]),
        "start_row": int(chunk["start_row"]),
        "row_count": int(chunk["row_count"]),
        "candidate_count": int(config.candidate_count),
        "paired_tests_per_candidate": int(config.paired_tests_per_candidate),
        "storage_format": str(storage_format),
        "table_name": "lqr_tuning_rows",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"tuning chunk manifest mismatch for {key}")
    if str(payload.get("relative_path", "")) != partition_path.relative_to(Path(run_root) / "tables").as_posix():
        raise ValueError("tuning chunk manifest mismatch for relative_path")
    if int(payload["row_count"]) != len(read_table_partition(partition_path, storage_format=storage_format)):
        raise ValueError("tuning chunk row count mismatch")
    if str(payload["checksum_sha256"]) != file_sha256(partition_path):
        raise ValueError("tuning chunk checksum mismatch")
    return payload


def _remove_tuning_chunk_outputs(run_root: Path, *, chunk: dict[str, int], storage_format: str) -> None:
    partition_path = _tuning_partition_path(run_root, chunk_index=int(chunk["chunk_index"]), storage_format=storage_format)
    manifest_path = _tuning_chunk_manifest_path(run_root, chunk_index=int(chunk["chunk_index"]))
    for path in (
        partition_path,
        partition_path.with_name(f"{partition_path.name}.tmp"),
        manifest_path,
        manifest_path.with_name(f"{manifest_path.name}.tmp"),
    ):
        fs_path = filesystem_path(path)
        if fs_path.exists():
            fs_path.unlink()


def _partition_from_tuning_manifest(
    config: LQRTuningSweepConfig,
    *,
    run_root: Path,
    chunk: dict[str, int],
    storage_format: str,
) -> TablePartition:
    payload = _validate_tuning_chunk_manifest(config, run_root=run_root, chunk=chunk, storage_format=storage_format)
    partition_path = _tuning_partition_path(run_root, chunk_index=int(chunk["chunk_index"]), storage_format=storage_format)
    return TablePartition(
        table_name="lqr_tuning_rows",
        relative_path=str(payload["relative_path"]),
        storage_format=str(payload["storage_format"]),
        row_count=int(payload["row_count"]),
        byte_count=int(filesystem_path(partition_path).stat().st_size),
        columns=tuple(str(column) for column in payload.get("columns", [])),
        checksum_sha256=str(payload["checksum_sha256"]),
    )


def _write_tuning_chunk_manifest(
    *,
    config: LQRTuningSweepConfig,
    run_root: Path,
    chunk_index: int,
    chunk_count: int,
    start_row: int,
    row_count: int,
    partition,
    storage_format: str,
    registry_status: str,
    columns: tuple[str, ...],
    elapsed_s: float,
) -> None:
    path = _tuning_chunk_manifest_path(run_root, chunk_index=int(chunk_index))
    filesystem_path(path.parent).mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "complete",
        "run_stage": "R6_W0_W1_LQR_QR_tuning",
        "run_id": int(config.run_id),
        "chunk_index": int(chunk_index),
        "chunk_count": int(chunk_count),
        "start_row": int(start_row),
        "row_count": int(row_count),
        "candidate_count": int(config.candidate_count),
        "paired_tests_per_candidate": int(config.paired_tests_per_candidate),
        "table_name": partition.table_name,
        "relative_path": partition.relative_path,
        "storage_format": str(storage_format),
        "columns": list(columns),
        "checksum_sha256": file_sha256(Path(run_root) / "tables" / partition.relative_path),
        "registry_status": str(registry_status),
        "simulation_s": float(elapsed_s),
        "total_s": float(elapsed_s),
        "dense_runtime_contract": "chunked_resumable_checksum_manifest",
    }
    filesystem_path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")


def _smoke_rows(
    config: LQRTuningSweepConfig,
    candidate_records: list[dict[str, object]],
    *,
    start_row: int = 0,
    row_count: int | None = None,
) -> list[dict[str, object]]:
    rows = []
    row_count = int(config.rows) if row_count is None else int(row_count)
    if not candidate_records:
        return rows
    for row_index in range(int(start_row), int(start_row) + int(row_count)):
        candidate_position = (row_index // 2) % len(candidate_records)
        pair_index = (row_index // 2) // len(candidate_records)
        candidate = candidate_records[candidate_position]
        primitive = candidate["primitive"]
        controller = candidate["controller"]
        w_layer = "W0" if row_index % 2 == 0 else "W1"
        env_mode = "dry_air" if w_layer == "W0" else "gaussian_single"
        if w_layer == "W1":
            env_mode = "gaussian_single" if (pair_index + candidate_position) % 2 == 0 else "gaussian_four"
        sample_row_index = int((pair_index * len(candidate_records) + candidate_position) * 2)
        sample = archive_state_sample_for_row(
            sample_row_index,
            seed=int(config.seed),
            W_layer=w_layer,
            environment_mode=env_mode,
        )
        instance = environment_instance_for_mode(w_layer, env_mode, int(config.seed) + row_index)
        metadata = environment_metadata_from_instance(instance)
        binding = resolve_surrogate_binding(w_layer, metadata, randomisation_seed=int(config.seed) + row_index)
        wind = wind_field_for_binding(binding)
        context = build_environment_context(
            sample.state_vector,
            wind_field=wind,
            metadata=metadata,
            latency_case="none" if w_layer == "W0" else "nominal",
            surrogate_binding=binding,
        )
        config_rollout = RolloutConfig(W_layer=w_layer, rollout_backend="model_backed_lqr", wind_mode=binding.wind_mode)
        if binding.surrogate_binding_status != READY_STATUS:
            evidence = blocked_rollout_evidence(
                rollout_id=f"tune_{config.run_id:03d}_{row_index:06d}",
                initial_state=sample.state_vector,
                context=context,
                primitive=primitive,
                config=config_rollout,
                failure_label="surrogate_binding_blocked",
                termination_cause="surrogate_binding_blocked",
                controller=controller,
                controller_selection_status="W0_W1_candidate_rollout",
                candidate_index=int(candidate["candidate_index"]),
                candidate_weight_label=str(candidate["candidate_weight_label"]),
            )
        else:
            evidence = simulate_primitive_rollout(
                rollout_id=f"tune_{config.run_id:03d}_{row_index:06d}",
                initial_state=sample.state_vector,
                context=context,
                primitive=primitive,
                config=config_rollout,
                wind_field=wind,
                controller=controller,
                controller_selection_status="W0_W1_candidate_rollout",
                candidate_index=int(candidate["candidate_index"]),
                candidate_weight_label=str(candidate["candidate_weight_label"]),
            )
        row = rollout_with_context_row(evidence, context)
        row.update(archive_state_sample_row(sample))
        row["paired_start_key"] = sample.paired_start_key
        row["candidate_index"] = int(candidate["candidate_index"])
        row["candidate_weight_label"] = str(candidate["candidate_weight_label"])
        row["hard_gate_status"] = _hard_gate_status(
            evidence,
            binding,
            minimum_speed_required_m_s=float(config_rollout.minimum_speed_m_s),
        )
        row["soft_objective_terms"] = json.dumps(SOFT_OBJECTIVE_TERMS, separators=(",", ":"))
        rows.append(row)
    return rows


def _selected_controller_registry_rows(
    candidate_records: list[dict[str, object]],
    frame: pd.DataFrame,
    *,
    registry_status: str,
    registry_claim_status: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for primitive_id in sorted({record["controller"].primitive_id for record in candidate_records}):
        records = [record for record in candidate_records if record["controller"].primitive_id == primitive_id]
        scores = []
        for record in records:
            controller = record["controller"]
            subset = frame[
                (frame.get("primitive_id", "") == primitive_id)
                & (frame.get("candidate_index", -1).astype(str) == str(record["candidate_index"]))
            ] if not frame.empty else pd.DataFrame()
            status = _candidate_registry_status(controller, subset)
            score = _candidate_soft_score(subset) if status == "selected_candidate" else float("-inf")
            scores.append((score, record, status))
        best_record = max(scores, key=lambda item: item[0])[1] if scores else None
        for score, record, status in scores:
            controller = record["controller"]
            if best_record is record and status == "selected_candidate":
                if registry_status in {"complete", "accepted_fallback"}:
                    selected_status = SELECTED_CONTROLLER_STATUS
                    reason = "best_passed_W0_W1_soft_score"
                elif registry_status == "smoke_incomplete":
                    selected_status = SMOKE_SELECTED_CONTROLLER_STATUS
                    reason = "best_passed_W0_W1_soft_score_debug_smoke_incomplete"
                else:
                    selected_status = "blocked"
                    reason = "registry_blocked_not_active"
            elif controller.lqr_synthesis_status != LQR_SYNTHESIS_SOLVED:
                selected_status = "blocked"
                reason = controller.lqr_blocked_reason or controller.lqr_synthesis_status
            else:
                selected_status = "rejected"
                reason = "lower_W0_W1_soft_score" if status == "selected_candidate" else status
            rows.append(
                controller_registry_row(
                    controller,
                    selected_controller_status=selected_status,
                    selected_controller_reason=reason,
                    candidate_index=int(record["candidate_index"]),
                    candidate_weight_label=str(record["candidate_weight_label"]),
                    registry_status=registry_status,
                    registry_claim_status=registry_claim_status,
                )
            )
    return rows


def _registry_status_for_tuning_run(
    config: LQRTuningSweepConfig,
    schedule,
    frame: pd.DataFrame,
    *,
    failures: list[dict[str, object]] | None = None,
) -> str:
    if failures:
        return "smoke_incomplete" if frame is not None and not frame.empty else "blocked"
    if frame.empty:
        return "blocked"
    if len(frame) < int(schedule.planned_rows):
        return "smoke_incomplete"
    if not _has_complete_w0_w1_candidate_coverage(frame):
        return "smoke_incomplete"
    if not _has_selected_candidate_per_primitive(frame):
        return "blocked"
    if int(config.candidate_count) in PREFERRED_CANDIDATES_PER_PRIMITIVE and int(
        config.paired_tests_per_candidate
    ) in PREFERRED_PAIRED_TESTS_PER_CANDIDATE:
        return "complete"
    if (
        int(config.candidate_count) >= FALLBACK_CANDIDATES_PER_PRIMITIVE
        and int(config.paired_tests_per_candidate) >= FALLBACK_PAIRED_TESTS_PER_CANDIDATE
    ):
        return "accepted_fallback"
    return "smoke_incomplete"


def _candidate_registry_status(controller, subset: pd.DataFrame) -> str:
    if controller.lqr_synthesis_status != LQR_SYNTHESIS_SOLVED:
        return controller.lqr_synthesis_status
    if subset.empty:
        return "no_rollout_rows"
    if subset.get("hard_gate_status", pd.Series(dtype=str)).astype(str).eq("blocked").any():
        return "blocked_rollout_rows_present"
    return "selected_candidate"


def _has_complete_w0_w1_candidate_coverage(frame: pd.DataFrame) -> bool:
    required = {
        "primitive_id",
        "candidate_index",
        "W_layer",
        "paired_start_key",
        "hard_gate_status",
        "candidate_weight_label",
    }
    if frame.empty or not required.issubset(frame.columns):
        return False
    coverage = (
        frame.groupby(["primitive_id", "candidate_index", "W_layer"], dropna=False)
        .size()
        .reset_index(name="row_count")
    )
    expected_groups = set()
    for primitive_id in sorted(frame["primitive_id"].astype(str).unique()):
        for candidate_index in sorted(frame["candidate_index"].astype(str).unique()):
            expected_groups.add((primitive_id, candidate_index, "W0"))
            expected_groups.add((primitive_id, candidate_index, "W1"))
    actual_groups = {
        (str(row["primitive_id"]), str(row["candidate_index"]), str(row["W_layer"]))
        for _, row in coverage.iterrows()
    }
    if not expected_groups.issubset(actual_groups):
        return False
    paired = (
        frame.groupby(["primitive_id", "candidate_index", "paired_start_key"], dropna=False)["W_layer"]
        .agg(lambda values: set(str(value) for value in values))
        .reset_index(name="w_layers_present")
    )
    paired_ok = (
        paired.groupby(["primitive_id", "candidate_index"], dropna=False)["w_layers_present"]
        .agg(lambda values: any({"W0", "W1"}.issubset(set(item)) for item in values))
        .reset_index(name="has_paired_w0_w1")
    )
    if not bool(paired_ok["has_paired_w0_w1"].all()):
        return False
    metadata = frame[["candidate_weight_label", "lqr_Q_weights_json", "lqr_R_weights_json", "lqr_gain_checksum"]]
    for column in metadata.columns:
        values = metadata[column].astype(str).str.strip().str.lower()
        if values.isin(["", "nan", "none", "null"]).any():
            return False
    return True


def _has_selected_candidate_per_primitive(frame: pd.DataFrame) -> bool:
    if frame.empty or not {"primitive_id", "hard_gate_status"}.issubset(frame.columns):
        return False
    passed = frame[frame["hard_gate_status"].astype(str) == "passed"]
    primitives = set(frame["primitive_id"].astype(str))
    return primitives.issubset(set(passed["primitive_id"].astype(str)))


def _hard_gate_status(evidence, binding, *, minimum_speed_required_m_s: float) -> str:
    if evidence.controller_evidence_status not in {
        "candidate_executable_lqr",
        "executable_lqr",
        "registry_backed_executable",
    }:
        return "blocked"
    if getattr(binding, "surrogate_binding_status", "") != READY_STATUS:
        return "blocked"
    if evidence.trajectory_integrity_status not in {"finite_model_backed", "smoke_only_not_integrated"}:
        return "blocked"
    if evidence.boundary_use_class in {"hard_failure", "blocked"}:
        return "blocked"
    if str(evidence.failure_label) in {
        "nonfinite_initial_state",
        "nonfinite_trajectory",
        "z_boundary_exit",
        "impossible_initial_state",
        "corrupt_integration",
    }:
        return "blocked"
    if float(evidence.minimum_speed_m_s) < float(minimum_speed_required_m_s):
        return "blocked"
    if float(evidence.saturation_fraction) > 0.50:
        return "blocked"
    return "passed"


def _candidate_soft_score(subset: pd.DataFrame) -> float:
    if subset.empty:
        return float("-inf")
    terminal = subset.get("episode_terminal_useful", pd.Series([False] * len(subset))).astype(bool)
    return float(
        subset.get("energy_residual_m", pd.Series([0.0] * len(subset))).astype(float).mean()
        + 0.25 * subset.get("lift_dwell_time_s", pd.Series([0.0] * len(subset))).astype(float).mean()
        + 0.10 * subset.get("minimum_wall_margin_m", pd.Series([0.0] * len(subset))).astype(float).min()
        + 0.05 * terminal.mean()
        - 0.20 * subset.get("saturation_fraction", pd.Series([0.0] * len(subset))).astype(float).max()
    )


def _write_objective_summary(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        pd.DataFrame(columns=["primitive_id", "row_count"]).to_csv(filesystem_path(path), index=False)
        return
    summary = (
        frame.groupby(["primitive_id", "W_layer"], dropna=False)
        .agg(
            row_count=("primitive_id", "size"),
            mean_energy_residual_m=("energy_residual_m", "mean"),
            mean_lift_dwell_time_s=("lift_dwell_time_s", "mean"),
            min_wall_margin_m=("minimum_wall_margin_m", "min"),
            max_saturation_fraction=("saturation_fraction", "max"),
        )
        .reset_index()
    )
    summary.to_csv(filesystem_path(path), index=False)


def _write_runtime_summary(
    path: Path,
    manifest: dict[str, object],
    *,
    row_count: int,
    status: str,
) -> None:
    pd.DataFrame(
        [
            {
                "run_id": int(manifest["run_id"]),
                "stage": manifest["run_stage"],
                "row_count": int(row_count),
                "stage_status": str(status),
                "selected_worker_count": int(manifest.get("selected_worker_count", 0) or 0),
                "actual_worker_count": int(manifest.get("actual_worker_count", 0) or 0),
                "worker_enabled": bool(manifest.get("worker_enabled", False)),
                "chunk_execution_backend": str(manifest.get("chunk_execution_backend", "")),
                "scheduled_chunk_count": int(manifest.get("scheduled_chunk_count", 0) or 0),
                "completed_chunk_count": int(manifest.get("completed_chunk_count", 0) or 0),
                "skipped_chunk_count": int(manifest.get("skipped_chunk_count", 0) or 0),
                "failed_chunk_count": int(manifest.get("failed_chunk_count", 0) or 0),
                "registry_status": str(manifest.get("registry_status", "")),
                "registry_claim_status": str(manifest.get("registry_claim_status", "")),
                "archive_evidence_status": str(manifest.get("archive_evidence_status", "")),
                "evidence_eligibility_reason": str(manifest.get("evidence_eligibility_reason", "")),
            }
        ]
    ).to_csv(filesystem_path(path), index=False)


def _extension(storage_format: str) -> str:
    if storage_format == "parquet":
        return "parquet"
    if storage_format == "csv":
        return "csv"
    return "csv.gz"


def main(argv: list[str] | None = None) -> int:
    run_lqr_tuning_sweep(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
