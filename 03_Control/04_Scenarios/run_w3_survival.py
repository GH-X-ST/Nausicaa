from __future__ import annotations

import argparse
import json
import math
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
from dense_archive_table_io import (  # noqa: E402
    TableManifest,
    filesystem_path,
    load_table_manifest,
    resolve_storage_format,
    table_extension,
    write_table_manifest,
    write_table_partition,
)
from env_ctx import build_environment_context, environment_context_row  # noqa: E402
from env_instance import environment_instance_for_mode, environment_instance_row, environment_metadata_from_instance  # noqa: E402
from env_surrogate import resolve_surrogate_binding, surrogate_binding_row, wind_field_for_binding  # noqa: E402
from frozen_w01_controller_bundle import FROZEN_CONTROLLER_READY, FrozenW01ControllerRecord, load_frozen_w01_controller_bundle  # noqa: E402
from implementation_instance import implementation_instance_for_layer, implementation_instance_row  # noqa: E402
from plant_instance import plant_instance_for_layer, plant_instance_row  # noqa: E402
from prim_cat import primitive_by_id  # noqa: E402
from prim_roll import RolloutConfig, blocked_rollout_evidence, rollout_evidence_row, simulate_primitive_rollout  # noqa: E402
from primitive_variant_registry import ENTRY_ROLE_REJECTION_LABEL, ENTRY_ROLE_REJECTION_STATUS, start_family_is_compatible, variant_row  # noqa: E402
from state_sampling import archive_state_sample_for_family, archive_state_sample_row, start_state_family_for_row  # noqa: E402


W3_SURVIVAL_VERSION = "w3_fixed_lqr_survival_replay_v1"
PROJECT_TITLE_VERSION = "LQR-Stabilised Contextual Primitive v4.5"
DEFAULT_OUTPUT_ROOT = Path("03_Control/05_Results/lqr_contextual_v1_0/w3_survival")
W3_TABLE_NAME = "w3_survival_rows"
W3_ENVIRONMENT_CASES = ("w3_randomised_single", "w3_randomised_four")
SURVIVAL_STATUS_VOCABULARY = ("blocked", "ready_for_fixed_lqr_replay", "complete", "not_run")
BLOCKED_CLAIMS = (
    "W3_robustness_complete",
    "post_W3_compact_library_ready",
    "governor_validation",
    "hardware_readiness",
    "real_flight_transfer",
    "mission_success",
)


@dataclass(frozen=True)
class W3SurvivalConfig:
    run_id: int
    input_root: Path
    output_root: Path = DEFAULT_OUTPUT_ROOT
    seed: int = 30
    paired_tests_per_variant: int = 5
    storage_format: str = "csv_gz"
    compression_level: int = 1
    dry_run_schedule: bool = False
    rollout_dt_s: float = 0.02
    latency_case: str = "nominal"


def run_w3_survival(config: W3SurvivalConfig) -> dict[str, object]:
    """Run a small fixed-LQR W3 replay from W2 survivors, or block cleanly."""

    storage_format = resolve_storage_format(config.storage_format)
    run_root = Path(config.output_root) / f"{int(config.run_id):03d}"
    for subdir in ("manifests", "metrics", "reports", "tables"):
        filesystem_path(run_root / subdir).mkdir(parents=True, exist_ok=True)

    blocked_reason = _input_blocked_reason(Path(config.input_root))
    if blocked_reason:
        _write_blocked_outputs(run_root=run_root, config=config, storage_format=storage_format, blocked_reason=blocked_reason)
        return _result_payload(run_root, "blocked")

    bundle_path = Path(config.input_root) / "manifests" / "frozen_w01_controller_bundle.json"
    if not filesystem_path(bundle_path).is_file():
        _write_blocked_outputs(
            run_root=run_root,
            config=config,
            storage_format=storage_format,
            blocked_reason="missing_frozen_w01_controller_bundle_from_W2_root",
        )
        return _result_payload(run_root, "blocked")
    bundle = load_frozen_w01_controller_bundle(bundle_path)
    records = _surviving_records(Path(config.input_root), bundle)
    if not records:
        _write_blocked_outputs(
            run_root=run_root,
            config=config,
            storage_format=storage_format,
            blocked_reason="missing_reconstructable_W2_surviving_frozen_records",
        )
        return _result_payload(run_root, "blocked")

    row_count = len(records) * len(W3_ENVIRONMENT_CASES) * int(config.paired_tests_per_variant)
    _write_run_manifest(
        run_root=run_root,
        config=config,
        storage_format=storage_format,
        status="ready_for_fixed_lqr_replay" if config.dry_run_schedule else "running",
        row_count=0 if config.dry_run_schedule else row_count,
        survivor_count=len(records),
        blocked_reason="",
    )
    if config.dry_run_schedule:
        _write_empty_table_manifest(run_root, config.run_id, storage_format)
        pd.DataFrame([{"status": "ready_for_fixed_lqr_replay", "planned_row_count": row_count}]).to_csv(
            filesystem_path(run_root / "metrics" / "w3_survival_summary.csv"),
            index=False,
        )
        _write_file_size_audit(run_root)
        _write_reports(run_root=run_root, status="ready_for_fixed_lqr_replay", row_count=0, blocked_reason="")
        return _result_payload(run_root, "ready_for_fixed_lqr_replay")

    rows = [_row_for_index(row_index=row_index, config=config, records=records) for row_index in range(row_count)]
    frame = pd.DataFrame(rows)
    partition_path = run_root / "tables" / W3_TABLE_NAME / f"c00000.{table_extension(storage_format)}"
    partition = write_table_partition(
        frame,
        partition_path,
        storage_format=storage_format,
        compression_level=int(config.compression_level),
    )
    write_table_manifest(
        run_root / "manifests" / "table_manifest.json",
        TableManifest(run_id=int(config.run_id), root=run_root.as_posix(), storage_format=storage_format, tables=(partition,)),
    )
    _write_randomisation_manifest(run_root=run_root, config=config, row_count=row_count)
    _write_metrics(run_root=run_root, frame=frame)
    _write_file_size_audit(run_root)
    _write_run_manifest(
        run_root=run_root,
        config=config,
        storage_format=storage_format,
        status="complete",
        row_count=row_count,
        survivor_count=len(records),
        blocked_reason="",
    )
    _write_reports(run_root=run_root, status="complete", row_count=row_count, blocked_reason="")
    return _result_payload(run_root, "complete")


def _input_blocked_reason(input_root: Path) -> str:
    w2_manifest = Path(input_root) / "manifests" / "w2_survival_manifest.json"
    survivor_registry = Path(input_root) / "manifests" / "w2_survivor_registry.json"
    if not filesystem_path(w2_manifest).is_file():
        return "missing_W2_survival_manifest"
    if not filesystem_path(survivor_registry).is_file():
        return "missing_W2_survivor_registry"
    source = json.loads(filesystem_path(w2_manifest).read_text(encoding="ascii"))
    survivors = json.loads(filesystem_path(survivor_registry).read_text(encoding="ascii"))
    if source.get("status") != "survived_variants_available":
        return "W2_survival_status_not_survived_variants_available"
    if survivors.get("status") != "survived_variants_available":
        return "W2_survivor_registry_not_available"
    if int(survivors.get("survivor_count", 0)) <= 0:
        return "missing_W2_surviving_variants"
    return ""


def _surviving_records(input_root: Path, bundle) -> tuple[FrozenW01ControllerRecord, ...]:
    survivor_registry = json.loads(
        filesystem_path(input_root / "manifests" / "w2_survivor_registry.json").read_text(encoding="ascii")
    )
    survivor_ids = {
        str(row.get("primitive_variant_id", ""))
        for row in survivor_registry.get("survivors", [])
    }
    records = []
    by_variant_id = bundle.records_by_variant_id
    for variant_id in sorted(survivor_ids):
        record = by_variant_id.get(variant_id)
        if record is not None and record.bundle_status == FROZEN_CONTROLLER_READY:
            records.append(record)
    return tuple(records)


def _row_for_index(
    *,
    row_index: int,
    config: W3SurvivalConfig,
    records: tuple[FrozenW01ControllerRecord, ...],
) -> dict[str, object]:
    record = records[int(row_index) % len(records)]
    grouped_index = int(row_index) // len(records)
    environment_mode = W3_ENVIRONMENT_CASES[grouped_index % len(W3_ENVIRONMENT_CASES)]
    paired_start_index = grouped_index // len(W3_ENVIRONMENT_CASES)
    start_family = start_state_family_for_row(paired_start_index)
    paired_start_key = f"w3_paired_{int(paired_start_index):07d}_{start_family}"
    sample = archive_state_sample_for_family(
        start_state_family=start_family,
        paired_start_key=paired_start_key,
        sample_index=int(paired_start_index),
        seed=int(config.seed),
        W_layer="W3",
        environment_mode=environment_mode,
    )
    environment = environment_instance_for_mode("W3", environment_mode, int(config.seed) + int(row_index))
    metadata = environment_metadata_from_instance(environment)
    binding = resolve_surrogate_binding("W3", metadata, randomisation_seed=int(config.seed) + int(row_index))
    wind_field = wind_field_for_binding(binding)
    context = build_environment_context(
        sample.state_vector,
        wind_field=wind_field,
        metadata=metadata,
        latency_case=str(config.latency_case),
        actuator_case="nominal",
        surrogate_binding=binding,
    )
    variant = record.variant
    primitive = primitive_by_id(variant.primitive_id)
    compatible = start_family_is_compatible(entry_role=variant.entry_role, start_state_family=start_family)
    rollout_config = RolloutConfig(W_layer="W3", dt_s=float(config.rollout_dt_s), rollout_backend="model_backed_lqr", wind_mode="panel")
    if not compatible or binding.surrogate_binding_status != "ready":
        failure_label = ENTRY_ROLE_REJECTION_LABEL if not compatible else "w3_surrogate_binding_blocked"
        termination = ENTRY_ROLE_REJECTION_STATUS if not compatible else str(binding.blocked_reason)
        evidence = blocked_rollout_evidence(
            rollout_id=f"w3r{int(config.run_id):03d}_{int(row_index):07d}",
            episode_id=f"w3_{int(row_index):07d}",
            initial_state=sample.state_vector,
            context=context,
            primitive=primitive,
            config=RolloutConfig(W_layer="W3", dt_s=float(config.rollout_dt_s), rollout_backend="blocked_lqr", wind_mode="panel"),
            failure_label=failure_label,
            controller=record.controller,
            controller_selection_status="W3_fixed_lqr_replay_frozen_w01_bundle",
            candidate_index=variant.candidate_index,
            candidate_weight_label=variant.candidate_weight_label,
            termination_cause=termination,
        )
        row = rollout_evidence_row(evidence)
        implementation = None
        plant = None
    else:
        implementation = implementation_instance_for_layer("W3", int(config.seed) + int(row_index), latency_case=str(config.latency_case))
        plant = plant_instance_for_layer("W3", int(config.seed) + int(row_index))
        evidence = simulate_primitive_rollout(
            rollout_id=f"w3r{int(config.run_id):03d}_{int(row_index):07d}",
            episode_id=f"w3_{int(row_index):07d}",
            initial_state=sample.state_vector,
            context=context,
            primitive=primitive,
            config=rollout_config,
            wind_field=wind_field,
            implementation_instance=implementation,
            plant_instance=plant,
            controller=record.controller,
            controller_selection_status="W3_fixed_lqr_replay_frozen_w01_bundle",
            candidate_index=variant.candidate_index,
            candidate_weight_label=variant.candidate_weight_label,
        )
        row = rollout_evidence_row(evidence)
    row.update(archive_state_sample_row(sample))
    row.update({f"variant_{key}": value for key, value in variant_row(variant).items()})
    row.update({f"context_{key}": value for key, value in environment_context_row(context).items()})
    row.update({f"surrogate_{key}": value for key, value in surrogate_binding_row(binding).items()})
    row.update({f"environment_{key}": value for key, value in environment_instance_row(environment).items()})
    if implementation is None or plant is None:
        row.update(_blocked_instance_prefix_rows("implementation"))
        row.update(_blocked_instance_prefix_rows("plant"))
    else:
        row.update({f"implementation_{key}": value for key, value in implementation_instance_row(implementation).items()})
        row.update({f"plant_{key}": value for key, value in plant_instance_row(plant).items()})
    row.update(
        {
            "runner_version": W3_SURVIVAL_VERSION,
            "project_title_version": PROJECT_TITLE_VERSION,
            "run_stage": "W3_fixed_LQR_randomised_survival_replay",
            "row_index": int(row_index),
            "source_w2_root": Path(config.input_root).as_posix(),
            "primitive_variant_id": variant.primitive_variant_id,
            "entry_role": variant.entry_role,
            "entry_role_compatible": bool(compatible),
            "environment_mode": environment_mode,
            "fixed_lqr_replay_only": True,
            "mutates_Q_R_K_reference_horizon_entry_set_or_entry_role": False,
            "w3_environment_contract": "randomised_single_and_four_annular_gp_survival_replay_only",
            "baseline_controller_active": False,
            "claim_boundary": "simulation_only_W3_fixed_LQR_randomised_replay_smoke_no_robustness_claim",
        }
    )
    return row


def _write_blocked_outputs(
    *,
    run_root: Path,
    config: W3SurvivalConfig,
    storage_format: str,
    blocked_reason: str,
) -> None:
    _write_empty_table_manifest(run_root, config.run_id, storage_format)
    pd.DataFrame([{"status": "blocked", "blocked_reason": blocked_reason}]).to_csv(
        filesystem_path(run_root / "metrics" / "w3_survival_summary.csv"),
        index=False,
    )
    _write_file_size_audit(run_root)
    _write_run_manifest(
        run_root=run_root,
        config=config,
        storage_format=storage_format,
        status="blocked",
        row_count=0,
        survivor_count=0,
        blocked_reason=blocked_reason,
    )
    _write_reports(run_root=run_root, status="blocked", row_count=0, blocked_reason=blocked_reason)


def _write_run_manifest(
    *,
    run_root: Path,
    config: W3SurvivalConfig,
    storage_format: str,
    status: str,
    row_count: int,
    survivor_count: int,
    blocked_reason: str,
) -> None:
    manifest = {
        "version": W3_SURVIVAL_VERSION,
        "project_title_version": PROJECT_TITLE_VERSION,
        "status": status,
        "run_id": int(config.run_id),
        "input_root": Path(config.input_root).as_posix(),
        "input_contract": "W2 survival root with w2_survival_manifest.json and w2_survivor_registry.json reporting survived_variants_available; raw W01 variants are not accepted",
        "row_count": int(row_count),
        "survivor_count": int(survivor_count),
        "storage_format": storage_format,
        "paired_tests_per_variant": int(config.paired_tests_per_variant),
        "fixed_lqr_replay_only": True,
        "mutates_Q_R_K_reference_horizon_entry_set_or_entry_role": False,
        "forbidden_mutations": "Q/R,K,reference,horizon,entry_set,entry_role",
        "status_vocabulary": list(SURVIVAL_STATUS_VOCABULARY),
        "redesign_policy": "new_ids_return_to_W01",
        "blocked_reason": blocked_reason,
        "blocked_claims": list(BLOCKED_CLAIMS),
    }
    _write_json(run_root / "manifests" / "w3_survival_manifest.json", manifest)


def _write_randomisation_manifest(*, run_root: Path, config: W3SurvivalConfig, row_count: int) -> None:
    _write_json(
        run_root / "manifests" / "w3_randomisation_manifest.json",
        {
            "version": "w3_randomisation_manifest_v1",
            "row_count": int(row_count),
            "environment_modes": list(W3_ENVIRONMENT_CASES),
            "seed": int(config.seed),
            "randomisation_source": "env_instance_W3_randomised_annular_gp_modes",
        },
    )


def _write_metrics(*, run_root: Path, frame: pd.DataFrame) -> None:
    rows = []
    for column in ("outcome_class", "failure_label", "boundary_use_class", "environment_mode", "timing_state_source"):
        if column in frame.columns:
            counts = frame[column].astype(str).value_counts(dropna=False)
            rows.extend(
                {"coverage_axis": column, "value": str(value), "row_count": int(count)}
                for value, count in counts.items()
            )
    pd.DataFrame(rows).to_csv(filesystem_path(run_root / "metrics" / "w3_survival_summary.csv"), index=False)


def _write_empty_table_manifest(run_root: Path, run_id: int, storage_format: str) -> None:
    write_table_manifest(
        run_root / "manifests" / "table_manifest.json",
        TableManifest(run_id=int(run_id), root=run_root.as_posix(), storage_format=storage_format, tables=()),
    )


def _write_file_size_audit(run_root: Path) -> None:
    rows = []
    root_fs = filesystem_path(run_root)
    for path in sorted(root_fs.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root_fs).as_posix()
        byte_count = int(path.stat().st_size)
        size_mb = float(byte_count) / float(1024 * 1024)
        rows.append(
            {
                "relative_path": rel,
                "byte_count": byte_count,
                "size_mb": size_mb,
                "above_100mb": bool(size_mb > MAX_GENERATED_FILE_SIZE_MB),
                "push_allowed": bool(size_mb <= MAX_GENERATED_FILE_SIZE_MB),
                "dense_table_partition": rel.startswith(f"tables/{W3_TABLE_NAME}/"),
            }
        )
    pd.DataFrame(rows).to_csv(filesystem_path(run_root / "metrics" / "file_size_audit.csv"), index=False)


def _write_reports(*, run_root: Path, status: str, row_count: int, blocked_reason: str) -> None:
    lines = [
        "# W3 Fixed-LQR Survival Replay",
        "",
        f"- Status: `{status}`",
        f"- Rows written: `{int(row_count)}`",
        f"- Blocked reason: `{blocked_reason}`",
        "- Fixed-LQR replay only: `True`",
        "- Q/R, K, reference, horizon, entry set, and entry role mutation: `False`",
        "- W3 robustness, compact-library readiness, hardware readiness, transfer, and mission success remain blocked.",
        "",
    ]
    filesystem_path(run_root / "reports" / "w3_survival_report.md").write_text("\n".join(lines), encoding="ascii")
    filesystem_path(run_root / "reports" / "l9_w3_move_on_check.md").write_text("\n".join(lines), encoding="ascii")


def _blocked_instance_prefix_rows(prefix: str) -> dict[str, object]:
    if prefix == "implementation":
        return {
            "implementation_implementation_instance_id": "",
            "implementation_W_layer": "",
            "implementation_latency_case": "",
            "implementation_implementation_adjustment_status": "not_applied_blocked_before_rollout",
            "implementation_implementation_adjustment_limitations": "blocked_before_rollout_simulation",
        }
    return {
        "plant_plant_instance_id": "",
        "plant_W_layer": "",
        "plant_plant_adjustment_status": "not_applied_blocked_before_rollout",
        "plant_plant_adjustment_limitations": "blocked_before_rollout_simulation",
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    filesystem_path(path).parent.mkdir(parents=True, exist_ok=True)
    filesystem_path(path).write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="ascii")


def _result_payload(run_root: Path, status: str) -> dict[str, object]:
    return {
        "status": status,
        "run_root": run_root.as_posix(),
        "manifest": (run_root / "manifests" / "w3_survival_manifest.json").as_posix(),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run W3 fixed-LQR survival replay from W2 survivors.")
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--paired-tests-per-variant", type=int, default=5)
    parser.add_argument("--storage-format", choices=("auto", "parquet", "csv_gz", "csv"), default="csv_gz")
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument("--dry-run-schedule", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_w3_survival(
        W3SurvivalConfig(
            run_id=args.run_id,
            input_root=args.input_root,
            output_root=args.output_root,
            seed=args.seed,
            paired_tests_per_variant=args.paired_tests_per_variant,
            storage_format=args.storage_format,
            compression_level=args.compression_level,
            dry_run_schedule=args.dry_run_schedule,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
