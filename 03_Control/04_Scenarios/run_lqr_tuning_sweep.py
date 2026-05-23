from __future__ import annotations

import argparse
import json
import sys
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
    filesystem_path,
    resolve_storage_format,
    write_table_manifest,
    write_table_partition,
)
from env_ctx import build_environment_context  # noqa: E402
from env_instance import environment_instance_for_mode, environment_metadata_from_instance  # noqa: E402
from env_surrogate import READY_STATUS, resolve_surrogate_binding, wind_field_for_binding  # noqa: E402
from evidence_stage_utils import write_coverage_summary, write_file_size_audit  # noqa: E402
from controller_registry import controller_registry_row, write_selected_controller_registry  # noqa: E402
from lqr_controller import LQR_SYNTHESIS_SOLVED, synthesize_lqr_controller, synthesis_audit_row  # noqa: E402
from lqr_tuning import (  # noqa: E402
    FALLBACK_CANDIDATES_PER_PRIMITIVE,
    FALLBACK_PAIRED_TESTS_PER_CANDIDATE,
    SOFT_OBJECTIVE_TERMS,
    HARD_GATE_LABELS,
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
    for rel in ("manifests", "metrics", "reports", "tables"):
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
    manifest = {
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
        "hard_gates": list(HARD_GATE_LABELS),
        "soft_objective_terms": list(SOFT_OBJECTIVE_TERMS),
        "raw_K_tuning_allowed": False,
        "W0_W1_tune_controller_ids": True,
        "W2_W3_replay_only": True,
        "dry_run_schedule": bool(config.dry_run_schedule),
    }
    filesystem_path(run_root / "manifests" / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="ascii",
    )
    if config.dry_run_schedule:
        pd.DataFrame([manifest]).to_csv(filesystem_path(run_root / "metrics" / "runtime_summary.csv"), index=False)
        return {"run_root": run_root, "run_manifest": run_root / "manifests" / "run_manifest.json"}

    primitives = active_primitive_catalogue()
    synthesis_rows = [synthesis_audit_row(primitive) for primitive in primitives]
    candidate_rows = []
    candidate_records = []
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
    smoke_rows = _smoke_rows(config, candidate_records)

    partitions = []
    partitions.append(
        write_table_partition(
            pd.DataFrame(smoke_rows),
            run_root / "tables" / "lqr_tuning_smoke_rows" / f"c00000.{_extension(storage_format)}",
            storage_format=storage_format,
            compression_level=config.compression_level,
        )
    )
    pd.DataFrame(synthesis_rows).to_csv(
        filesystem_path(run_root / "metrics" / "lqr_synthesis_audit.csv"),
        index=False,
    )
    pd.DataFrame(candidate_rows).to_csv(
        filesystem_path(run_root / "metrics" / "qr_candidate_rankings.csv"),
        index=False,
    )
    registry_rows = _selected_controller_registry_rows(candidate_records, pd.DataFrame(smoke_rows))
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
    frame = pd.DataFrame(smoke_rows)
    write_coverage_summary(
        run_root / "metrics" / "coverage_summary.csv",
        frame,
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
        ),
    )
    _write_objective_summary(run_root / "metrics" / "objective_term_summary.csv", frame)
    write_file_size_audit(run_root, run_root / "metrics" / "file_size_audit.csv")
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


def _smoke_rows(
    config: LQRTuningSweepConfig,
    candidate_records: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    row_count = int(config.rows)
    chunk_limit = None
    if config.stop_after_chunks is not None:
        chunk_limit = int(config.stop_after_chunks) * int(config.candidate_chunk_size)
        row_count = min(row_count, chunk_limit)
    if not candidate_records:
        return rows
    for row_index in range(row_count):
        candidate_position = (row_index // 2) % len(candidate_records)
        pair_index = (row_index // 2) // len(candidate_records)
        candidate = candidate_records[candidate_position]
        primitive = candidate["primitive"]
        controller = candidate["controller"]
        w_layer = "W0" if row_index % 2 == 0 else "W1"
        env_mode = "dry_air" if w_layer == "W0" else "gaussian_single"
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
        row["hard_gate_status"] = "passed" if evidence.outcome_class != "blocked" else "blocked"
        row["soft_objective_terms"] = json.dumps(SOFT_OBJECTIVE_TERMS, separators=(",", ":"))
        rows.append(row)
    return rows


def _selected_controller_registry_rows(
    candidate_records: list[dict[str, object]],
    frame: pd.DataFrame,
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
                selected_status = "selected"
                reason = "best_passed_W0_W1_soft_score"
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
                )
            )
    return rows


def _candidate_registry_status(controller, subset: pd.DataFrame) -> str:
    if controller.lqr_synthesis_status != LQR_SYNTHESIS_SOLVED:
        return controller.lqr_synthesis_status
    if subset.empty:
        return "no_rollout_rows"
    if subset.get("hard_gate_status", pd.Series(dtype=str)).astype(str).eq("blocked").any():
        return "blocked_rollout_rows_present"
    return "selected_candidate"


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
