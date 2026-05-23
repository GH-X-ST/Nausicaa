from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CONTROL_ROOT = Path(__file__).resolve().parents[1]
for rel in ("02_Inner_Loop", "03_Primitives", "04_Scenarios"):
    path = CONTROL_ROOT / rel
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from archive_table_reader import read_archive_table_with_info  # noqa: E402
from dense_archive_table_io import (  # noqa: E402
    TableManifest,
    TablePartition,
    file_sha256,
    filesystem_path,
    read_table_partition,
    table_extension,
    write_table_manifest,
    write_table_partition,
)
from env_ctx import build_environment_context  # noqa: E402
from env_instance import (  # noqa: E402
    environment_instance_for_mode,
    environment_instance_row,
    environment_metadata_from_instance,
)
from env_surrogate import (  # noqa: E402
    READY_STATUS,
    resolve_surrogate_binding,
    surrogate_binding_row,
    wind_field_for_binding,
)
from evidence_stage_utils import (  # noqa: E402
    write_blocked_approximate_ratio_summary,
    write_claim_boundary_report,
    write_coverage_summary,
    write_file_size_audit,
)
from implementation_instance import (  # noqa: E402
    implementation_instance_for_layer,
    implementation_instance_row,
)
from plant_instance import plant_instance_for_layer, plant_instance_row  # noqa: E402
from prim_cat import primitive_by_id  # noqa: E402
from prim_roll import (  # noqa: E402
    RolloutConfig,
    blocked_rollout_evidence,
    rollout_with_context_row,
    simulate_primitive_rollout,
)
from state_contract import STATE_NAMES  # noqa: E402


@dataclass(frozen=True)
class W2ReplayConfig:
    run_id: int
    output_root: Path
    source_archive: Path | None = None
    target_rows: int = 15_000
    fallback_rows: int = 2_000
    max_source_rows: int = 0
    representative_reuse_limit: int = 8
    latency_case: str = "nominal"
    storage_format: str = "csv_gz"
    compression_level: int = 1
    workers: int = 8
    max_workers: int = 8
    chunk_size: int = 1000
    resume: bool = True
    repair_incomplete: bool = False
    stop_after_chunks: int | None = None


def parse_args(argv: list[str] | None = None) -> W2ReplayConfig:
    parser = argparse.ArgumentParser(description="Run W2 model-backed replay from R6 rows.")
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, default=None)
    parser.add_argument("--target-rows", type=int, default=15_000)
    parser.add_argument("--fallback-rows", type=int, default=2_000)
    parser.add_argument("--max-source-rows", type=int, default=0)
    parser.add_argument("--representative-reuse-limit", type=int, default=8)
    parser.add_argument("--latency-case", choices=("nominal", "conservative"), default="nominal")
    parser.add_argument("--storage-format", default="csv_gz")
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--chunk-size", "--candidate-chunk-size", dest="chunk_size", type=int, default=1000)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--repair-incomplete", action="store_true")
    parser.add_argument("--stop-after-chunks", type=int, default=None)
    args = parser.parse_args(argv)
    return W2ReplayConfig(
        run_id=int(args.run_id),
        output_root=Path(args.output_root),
        source_archive=None if args.source_archive is None else Path(args.source_archive),
        target_rows=int(args.target_rows),
        fallback_rows=int(args.fallback_rows),
        max_source_rows=int(args.max_source_rows),
        representative_reuse_limit=int(args.representative_reuse_limit),
        latency_case=str(args.latency_case),
        storage_format=str(args.storage_format),
        compression_level=int(args.compression_level),
        workers=int(args.workers),
        max_workers=int(args.max_workers),
        chunk_size=int(args.chunk_size),
        resume=bool(args.resume),
        repair_incomplete=bool(args.repair_incomplete),
        stop_after_chunks=args.stop_after_chunks,
    )


def run_w2_replay_scaffold(config: W2ReplayConfig) -> dict[str, object]:
    """Backward-compatible wrapper for the replay-capable R8 entrypoint."""

    return run_w2_replay(config)


def run_w2_replay(config: W2ReplayConfig) -> dict[str, object]:
    run_root = Path(config.output_root) / f"w2_replay_{config.run_id:03d}"
    for rel in ("manifests", "reports", "metrics", "tables"):
        filesystem_path(run_root / rel).mkdir(parents=True, exist_ok=True)

    if config.source_archive is None:
        return _write_blocked_outputs(
            config=config,
            run_root=run_root,
            reason="blocked_until_R6_archive_exists",
        )

    max_rows = None if int(config.max_source_rows) <= 0 else int(config.max_source_rows)
    source_frame, source_info = read_archive_table_with_info(config.source_archive, max_rows=max_rows)
    selected = _select_representative_source_rows(
        source_frame,
        target_rows=int(config.target_rows),
        fallback_rows=int(config.fallback_rows),
        reuse_limit=int(config.representative_reuse_limit),
    )
    if selected.empty:
        return _write_blocked_outputs(
            config=config,
            run_root=run_root,
            reason="blocked_no_representative_W1_rows",
        )

    partitions, replay_frames, chunk_records = _write_replay_partitions(
        selected,
        config=config,
        run_root=run_root,
    )
    replay_frame = pd.concat(replay_frames, ignore_index=True) if replay_frames else pd.DataFrame()
    table_manifest = run_root / "manifests" / "table_manifest.json"
    write_table_manifest(
        table_manifest,
        TableManifest(
            run_id=int(config.run_id),
            root=run_root.as_posix(),
            storage_format=str(partitions[0].storage_format) if partitions else config.storage_format,
            tables=tuple(partitions),
        ),
    )
    _write_mapping_summary(run_root / "metrics" / "source_to_w2_mapping.csv", replay_frame)
    _write_survival_summary(run_root / "metrics" / "w1_to_w2_survival_summary.csv", replay_frame)
    _write_failure_summary(run_root / "metrics" / "w2_failure_label_summary.csv", replay_frame)
    write_coverage_summary(
        run_root / "metrics" / "coverage_summary.csv",
        replay_frame,
        columns=(
            "source_outcome_class",
            "source_start_state_family",
            "source_state_envelope_label",
            "primitive_id",
            "outcome_class",
            "boundary_use_class",
            "latency_case",
        ),
    )
    ratio_summary = write_blocked_approximate_ratio_summary(
        run_root / "metrics" / "blocked_or_approximate_ratio_summary.csv",
        replay_frame,
    )
    file_audit, file_status = write_file_size_audit(
        run_root,
        run_root / "metrics" / "file_size_audit.csv",
    )
    row_count = int(len(replay_frame))
    target_status = "complete" if row_count >= int(config.target_rows) else "fallback"
    if row_count < int(config.fallback_rows):
        target_status = "partial"
    if float(ratio_summary["blocked_ratio"]) > 0.60:
        stage_status = "blocked"
    elif float(ratio_summary["blocked_ratio"]) > 0.35:
        stage_status = "partial" if target_status == "fallback" else "fallback"
    else:
        stage_status = target_status
    actual_replay = bool(
        row_count > 0
        and set(replay_frame.get("replay_generation_path", [])) == {"simulate_primitive_rollout"}
        and not replay_frame.get("source_label_copied_as_evidence", pd.Series([True])).astype(bool).any()
    )
    r8_complete = bool(actual_replay and stage_status in {"complete", "fallback"} and file_status == "pass")
    manifest = {
        "run_id": int(config.run_id),
        "stage": "R8_W2_model_backed_replay",
        "source_archive": Path(config.source_archive).as_posix(),
        "archive_source_info": source_info.__dict__,
        "W_layer": "W2",
        "surrogate_family": "gp_corrected_annular_gaussian",
        "required_latency_mechanisms": ["state_feedback_delay", "command_delay", "actuator_lag"],
        "replay_status": stage_status,
        "R8_W2_replay_complete": r8_complete,
        "replayed_row_count": row_count,
        "partition_count": len(partitions),
        "chunk_size": int(config.chunk_size),
        "workers": int(config.workers),
        "max_workers": int(config.max_workers),
        "resume": bool(config.resume),
        "repair_incomplete": bool(config.repair_incomplete),
        "stop_after_chunks": config.stop_after_chunks,
        "target_rows": int(config.target_rows),
        "fallback_rows": int(config.fallback_rows),
        "blocked_ratio": float(ratio_summary["blocked_ratio"]),
        "file_size_status": file_status,
        "actual_model_backed_replay": actual_replay,
        "claim_status": "simulation_only_w2_replay_no_survival_claim",
        "blocked_claims": ["full_W2_survival", "hardware_readiness", "real_flight_transfer"],
    }
    manifest_path = run_root / "manifests" / "w2_replay_manifest.json"
    filesystem_path(manifest_path).write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    pd.DataFrame(chunk_records).to_csv(filesystem_path(run_root / "metrics" / "chunk_summary.csv"), index=False)
    _write_runtime_summary(run_root / "metrics" / "runtime_summary.csv", manifest)
    _write_outcome_summary(run_root / "metrics" / "outcome_summary.csv", replay_frame)
    filesystem_path(run_root / "reports" / "w2_replay_report.md").write_text(
        "# W2 Model-Backed Replay Report\n\nNo full W2 survival or hardware-readiness claim is made.\n",
        encoding="ascii",
    )
    write_claim_boundary_report(
        run_root / "reports" / "claim_boundary_report.md",
        stage="R8 W2 replay",
        status=stage_status,
        claim_status=str(manifest["claim_status"]),
        blocked_claims=tuple(str(item) for item in manifest["blocked_claims"]),
    )
    return {
        "run_root": run_root,
        "manifest": manifest_path,
        "replay_table": run_root / "tables" / partitions[0].relative_path if partitions else run_root / "tables" / "w2_replay_rows.csv",
        "table_manifest": table_manifest,
        "file_size_audit": run_root / "metrics" / "file_size_audit.csv",
    }


def _write_replay_partitions(
    selected: pd.DataFrame,
    *,
    config: W2ReplayConfig,
    run_root: Path,
) -> tuple[list[TablePartition], list[pd.DataFrame], list[dict[str, object]]]:
    partitions: list[TablePartition] = []
    frames: list[pd.DataFrame] = []
    chunk_records: list[dict[str, object]] = []
    chunk_size = max(1, int(config.chunk_size))
    chunk_count = int(np.ceil(len(selected) / chunk_size))
    if config.stop_after_chunks is not None:
        chunk_count = min(chunk_count, int(config.stop_after_chunks))
    for chunk_index in range(chunk_count):
        start = chunk_index * chunk_size
        stop = min(start + chunk_size, len(selected))
        chunk_source = selected.iloc[start:stop]
        partition_path = _chunk_partition_path(run_root, chunk_index, config.storage_format)
        manifest_path = _chunk_manifest_path(run_root, chunk_index)
        if config.resume and filesystem_path(partition_path).is_file() and filesystem_path(manifest_path).is_file():
            try:
                partition, frame = _read_existing_chunk(partition_path, manifest_path)
                partitions.append(partition)
                frames.append(frame)
                chunk_records.append(
                    {
                        "chunk_index": int(chunk_index),
                        "status": "resumed",
                        "row_count": int(len(frame)),
                        "partition_path": partition.relative_path,
                        "checksum_sha256": partition.checksum_sha256,
                    }
                )
                continue
            except Exception:
                if not config.repair_incomplete:
                    raise
        rows = [
            _w2_replay_row(row=row.to_dict(), row_index=start + offset, config=config)
            for offset, (_, row) in enumerate(chunk_source.iterrows())
        ]
        frame = pd.DataFrame(rows)
        partition = write_table_partition(
            frame,
            partition_path,
            storage_format=config.storage_format,
            compression_level=config.compression_level,
        )
        _write_chunk_manifest(
            manifest_path,
            chunk_index=chunk_index,
            chunk_count=chunk_count,
            partition=partition,
            chunk_size=chunk_size,
            status="complete",
        )
        partitions.append(partition)
        frames.append(frame)
        chunk_records.append(
            {
                "chunk_index": int(chunk_index),
                "status": "complete",
                "row_count": int(len(frame)),
                "partition_path": partition.relative_path,
                "checksum_sha256": partition.checksum_sha256,
            }
        )
    return partitions, frames, chunk_records


def _chunk_partition_path(run_root: Path, chunk_index: int, storage_format: str = "csv_gz") -> Path:
    return (
        Path(run_root)
        / "tables"
        / "w2_replay_rows"
        / f"chunk_index={int(chunk_index):05d}"
        / f"part-00000.{table_extension(storage_format)}"
    )


def _chunk_manifest_path(run_root: Path, chunk_index: int) -> Path:
    return Path(run_root) / "chunk_manifests" / "w2_replay_rows" / f"chunk-{int(chunk_index):05d}.json"


def _write_chunk_manifest(
    path: Path,
    *,
    chunk_index: int,
    chunk_count: int,
    partition: TablePartition,
    chunk_size: int,
    status: str,
) -> None:
    payload = {
        "status": str(status),
        "chunk_index": int(chunk_index),
        "chunk_count": int(chunk_count),
        "chunk_size": int(chunk_size),
        "row_count": int(partition.row_count),
        "storage_format": str(partition.storage_format),
        "relative_path": str(partition.relative_path),
        "byte_count": int(partition.byte_count),
        "checksum_sha256": str(partition.checksum_sha256),
    }
    filesystem_path(path).parent.mkdir(parents=True, exist_ok=True)
    filesystem_path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")


def _read_existing_chunk(partition_path: Path, manifest_path: Path) -> tuple[TablePartition, pd.DataFrame]:
    payload = json.loads(filesystem_path(manifest_path).read_text(encoding="ascii"))
    if payload.get("status") != "complete":
        raise ValueError("chunk manifest is not complete")
    checksum = file_sha256(partition_path)
    if checksum != str(payload["checksum_sha256"]):
        raise ValueError("chunk checksum mismatch")
    frame = read_table_partition(partition_path, storage_format=str(payload["storage_format"]))
    if len(frame) != int(payload["row_count"]):
        raise ValueError("chunk row count mismatch")
    partition = TablePartition(
        table_name="w2_replay_rows",
        relative_path=str(payload["relative_path"]),
        storage_format=str(payload["storage_format"]),
        row_count=int(payload["row_count"]),
        byte_count=int(payload["byte_count"]),
        columns=tuple(str(column) for column in frame.columns),
        checksum_sha256=checksum,
    )
    return partition, frame


def _select_representative_source_rows(
    frame: pd.DataFrame,
    *,
    target_rows: int,
    fallback_rows: int,
    reuse_limit: int,
) -> pd.DataFrame:
    if frame.empty or "outcome_class" not in frame.columns:
        return pd.DataFrame()
    source = frame.copy()
    if "W_layer" in source.columns:
        source = source[source["W_layer"].astype(str) == "W1"]
    if source.empty:
        return pd.DataFrame()
    requested = max(int(fallback_rows), min(int(target_rows), len(source) * max(1, int(reuse_limit))))
    strata = []
    for outcome in ("accepted", "weak", "boundary_terminal", "failed", "rejected"):
        subset = source[source["outcome_class"].astype(str) == outcome]
        if subset.empty:
            continue
        for start_kind in ("launch_gate", "inflight"):
            if start_kind == "launch_gate":
                start_subset = subset[subset.get("start_state_family", "").astype(str) == "launch_gate"]
            else:
                start_subset = subset[subset.get("start_state_family", "").astype(str) != "launch_gate"]
            if not start_subset.empty:
                strata.append(start_subset)
    if not strata:
        strata = [source]
    selected_rows = []
    reuse_counts: dict[str, int] = {}
    cursor = 0
    while len(selected_rows) < requested:
        subset = strata[cursor % len(strata)]
        row = subset.iloc[(cursor // len(strata)) % len(subset)]
        key = str(row.get("rollout_id", row.name))
        count = reuse_counts.get(key, 0)
        if count < max(1, int(reuse_limit)):
            reuse_counts[key] = count + 1
            selected = row.copy()
            selected["source_reuse_count"] = count + 1
            selected_rows.append(selected)
        cursor += 1
        if cursor > requested * max(4, len(strata) * max(1, int(reuse_limit))):
            break
    return pd.DataFrame(selected_rows)


def _w2_replay_row(
    *,
    row: dict[str, object],
    row_index: int,
    config: W2ReplayConfig,
) -> dict[str, object]:
    state = _state_from_source(row)
    mode = _w2_environment_mode(row)
    seed = int(config.run_id) * 100_000 + int(row_index)
    primitive = primitive_by_id(str(row.get("primitive_id", "glide")))
    instance = environment_instance_for_mode("W2", mode, seed)
    metadata = environment_metadata_from_instance(instance)
    binding = resolve_surrogate_binding("W2", metadata, randomisation_seed=seed)
    wind = wind_field_for_binding(binding)
    implementation = implementation_instance_for_layer("W2", seed, latency_case=config.latency_case)
    plant = plant_instance_for_layer("W2", seed)
    context = build_environment_context(
        state,
        wind_field=wind,
        metadata=metadata,
        latency_case=implementation.latency_case,
        actuator_case="nominal",
        surrogate_binding=binding,
    )
    rollout_config = RolloutConfig(
        W_layer="W2",
        rollout_backend="model_backed_feedback",
        wind_mode=binding.wind_mode,
    )
    rollout_id = f"w2_r{config.run_id:03d}_{row_index:06d}"
    if binding.surrogate_binding_status != READY_STATUS:
        evidence = blocked_rollout_evidence(
            rollout_id=rollout_id,
            episode_id=f"w2_episode_{row_index:06d}",
            initial_state=state,
            context=context,
            primitive=primitive,
            config=rollout_config,
            failure_label="W2_surrogate_binding_blocked",
        )
    else:
        evidence = simulate_primitive_rollout(
            rollout_id=rollout_id,
            episode_id=f"w2_episode_{row_index:06d}",
            initial_state=state,
            context=context,
            primitive=primitive,
            config=rollout_config,
            wind_field=wind,
            implementation_instance=implementation,
            plant_instance=plant,
        )
    result = rollout_with_context_row(evidence, context)
    result.update({f"surrogate_{key}": value for key, value in surrogate_binding_row(binding).items()})
    result.update({f"environment_instance_{key}": value for key, value in environment_instance_row(instance).items()})
    result.update({f"implementation_instance_{key}": value for key, value in implementation_instance_row(implementation).items()})
    result.update({f"plant_instance_{key}": value for key, value in plant_instance_row(plant).items()})
    result.update(_source_columns(row))
    result["replay_generation_path"] = "simulate_primitive_rollout"
    result["source_label_copied_as_evidence"] = False
    result["w2_outcome_class"] = result["outcome_class"]
    result["source_reuse_count"] = int(row.get("source_reuse_count", 1))
    for name in STATE_NAMES:
        result[f"entry_{name}"] = float(state[STATE_NAMES.index(name)])
    return result


def _state_from_source(row: dict[str, object]) -> np.ndarray:
    values = []
    for name in STATE_NAMES:
        values.append(float(row.get(f"initial_{name}", row.get(f"entry_{name}", 0.0))))
    return np.asarray(values, dtype=float)


def _w2_environment_mode(row: dict[str, object]) -> str:
    mode = str(row.get("environment_mode", row.get("source_environment_mode", "gaussian_single")))
    if mode in {"gaussian_four", "fan_shift", "power_scale"}:
        return mode
    return "gaussian_single"


def _source_columns(row: dict[str, object]) -> dict[str, object]:
    return {
        "source_rollout_id": row.get("rollout_id", ""),
        "source_outcome_class": row.get("outcome_class", ""),
        "source_start_state_family": row.get("start_state_family", ""),
        "source_state_envelope_label": row.get("state_envelope_label", ""),
        "source_previous_primitive_status": row.get("previous_primitive_status", ""),
        "source_W_layer": row.get("W_layer", ""),
        "source_environment_id": row.get("environment_id", ""),
        "source_environment_mode": row.get("environment_mode", ""),
        "source_boundary_use_class": row.get("boundary_use_class", ""),
    }


def _write_blocked_outputs(
    *,
    config: W2ReplayConfig,
    run_root: Path,
    reason: str,
) -> dict[str, object]:
    empty_path = run_root / "tables" / "w2_replay_rows.csv"
    pd.DataFrame().to_csv(filesystem_path(empty_path), index=False)
    manifest = {
        "run_id": int(config.run_id),
        "stage": "R8_W2_model_backed_replay",
        "replay_status": "blocked",
        "blocked_reason": str(reason),
        "R8_W2_replay_complete": False,
        "replayed_row_count": 0,
        "actual_model_backed_replay": False,
        "claim_status": "simulation_only_w2_blocked_no_survival_claim",
        "blocked_claims": ["full_W2_survival", "hardware_readiness", "real_flight_transfer"],
    }
    manifest_path = run_root / "manifests" / "w2_replay_manifest.json"
    filesystem_path(manifest_path).write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    write_claim_boundary_report(
        run_root / "reports" / "claim_boundary_report.md",
        stage="R8 W2 replay",
        status="blocked",
        claim_status=str(manifest["claim_status"]),
        blocked_claims=tuple(str(item) for item in manifest["blocked_claims"]),
    )
    write_file_size_audit(run_root, run_root / "metrics" / "file_size_audit.csv")
    return {"run_root": run_root, "manifest": manifest_path, "replay_table": empty_path}


def _write_mapping_summary(path: Path, frame: pd.DataFrame) -> None:
    columns = ["source_rollout_id", "rollout_id", "source_outcome_class", "w2_outcome_class", "source_reuse_count"]
    frame[[column for column in columns if column in frame.columns]].to_csv(filesystem_path(path), index=False)


def _write_survival_summary(path: Path, frame: pd.DataFrame) -> None:
    columns = ["source_outcome_class", "outcome_class"]
    if not set(columns).issubset(frame.columns):
        pd.DataFrame(columns=[*columns, "row_count"]).to_csv(filesystem_path(path), index=False)
        return
    summary = frame.groupby(columns, dropna=False).size().reset_index(name="row_count")
    summary.to_csv(filesystem_path(path), index=False)


def _write_failure_summary(path: Path, frame: pd.DataFrame) -> None:
    columns = ["failure_label", "termination_cause"]
    if not set(columns).issubset(frame.columns):
        pd.DataFrame(columns=[*columns, "row_count"]).to_csv(filesystem_path(path), index=False)
        return
    summary = frame.groupby(columns, dropna=False).size().reset_index(name="row_count")
    summary.to_csv(filesystem_path(path), index=False)


def _write_runtime_summary(path: Path, manifest: dict[str, object]) -> None:
    pd.DataFrame(
        [
            {
                "run_id": manifest["run_id"],
                "stage": manifest["stage"],
                "row_count": manifest["replayed_row_count"],
                "stage_status": manifest["replay_status"],
                "claim_status": manifest["claim_status"],
            }
        ]
    ).to_csv(filesystem_path(path), index=False)


def _write_outcome_summary(path: Path, frame: pd.DataFrame) -> None:
    columns = ["outcome_class", "continuation_status", "episode_terminal_status", "boundary_use_class"]
    if frame.empty:
        pd.DataFrame(columns=[*columns, "row_count"]).to_csv(filesystem_path(path), index=False)
        return
    summary = frame.groupby([column for column in columns if column in frame.columns], dropna=False).size().reset_index(name="row_count")
    summary.to_csv(filesystem_path(path), index=False)


def main(argv: list[str] | None = None) -> int:
    run_w2_replay(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
