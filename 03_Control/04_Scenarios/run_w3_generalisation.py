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
    filesystem_path,
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
class W3GeneralisationConfig:
    run_id: int
    output_root: Path
    source_replay: Path | None = None
    target_rows: int = 30_000
    fallback_rows: int = 5_000
    max_source_rows: int = 0
    representative_reuse_limit: int = 6
    execute_replay: bool = True
    storage_format: str = "csv_gz"
    compression_level: int = 1


def parse_args(argv: list[str] | None = None) -> W3GeneralisationConfig:
    parser = argparse.ArgumentParser(description="Run W3 generalisation replay or blocked scaffold.")
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-replay", type=Path, default=None)
    parser.add_argument("--target-rows", type=int, default=30_000)
    parser.add_argument("--fallback-rows", type=int, default=5_000)
    parser.add_argument("--max-source-rows", type=int, default=0)
    parser.add_argument("--representative-reuse-limit", type=int, default=6)
    parser.add_argument("--execute-replay", action="store_true", default=True)
    parser.add_argument("--scaffold-only", dest="execute_replay", action="store_false")
    parser.add_argument("--storage-format", default="csv_gz")
    parser.add_argument("--compression-level", type=int, default=1)
    args = parser.parse_args(argv)
    return W3GeneralisationConfig(
        run_id=int(args.run_id),
        output_root=Path(args.output_root),
        source_replay=None if args.source_replay is None else Path(args.source_replay),
        target_rows=int(args.target_rows),
        fallback_rows=int(args.fallback_rows),
        max_source_rows=int(args.max_source_rows),
        representative_reuse_limit=int(args.representative_reuse_limit),
        execute_replay=bool(args.execute_replay),
        storage_format=str(args.storage_format),
        compression_level=int(args.compression_level),
    )


def run_w3_generalisation_scaffold(config: W3GeneralisationConfig) -> dict[str, object]:
    """Backward-compatible wrapper for the replay-capable R9 entrypoint."""

    return run_w3_generalisation(config)


def run_w3_generalisation(config: W3GeneralisationConfig) -> dict[str, object]:
    run_root = Path(config.output_root) / f"w3_generalisation_{config.run_id:03d}"
    for rel in ("manifests", "reports", "metrics", "tables"):
        filesystem_path(run_root / rel).mkdir(parents=True, exist_ok=True)
    if config.source_replay is None or not config.execute_replay:
        return _write_blocked_outputs(
            config=config,
            run_root=run_root,
            reason="deferred_or_missing_W2_source_replay",
        )
    max_rows = None if int(config.max_source_rows) <= 0 else int(config.max_source_rows)
    source_frame, source_info = read_archive_table_with_info(config.source_replay, max_rows=max_rows)
    selected = _select_w3_source_rows(
        source_frame,
        target_rows=int(config.target_rows),
        fallback_rows=int(config.fallback_rows),
        reuse_limit=int(config.representative_reuse_limit),
    )
    if selected.empty:
        return _write_blocked_outputs(
            config=config,
            run_root=run_root,
            reason="blocked_no_representative_W2_rows",
        )

    rows = [
        _w3_replay_row(
            row=row.to_dict(),
            row_index=index,
            config=config,
        )
        for index, (_, row) in enumerate(selected.iterrows())
    ]
    frame = pd.DataFrame(rows)
    partition = write_table_partition(
        frame,
        run_root / "tables" / "w3_generalisation_rows" / "part-00000.csv.gz",
        storage_format=config.storage_format,
        compression_level=config.compression_level,
    )
    table_manifest = run_root / "manifests" / "table_manifest.json"
    write_table_manifest(
        table_manifest,
        TableManifest(
            run_id=int(config.run_id),
            root=run_root.as_posix(),
            storage_format=str(partition.storage_format),
            tables=(partition,),
        ),
    )
    write_coverage_summary(
        run_root / "metrics" / "coverage_summary.csv",
        frame,
        columns=(
            "source_outcome_class",
            "source_boundary_use_class",
            "environment_instance_environment_mode",
            "implementation_instance_latency_case",
            "plant_instance_plant_adjustment_status",
            "outcome_class",
            "boundary_use_class",
        ),
    )
    ratio_summary = write_blocked_approximate_ratio_summary(
        run_root / "metrics" / "blocked_or_approximate_ratio_summary.csv",
        frame,
    )
    file_audit, file_status = write_file_size_audit(
        run_root,
        run_root / "metrics" / "file_size_audit.csv",
    )
    row_count = int(len(frame))
    target_status = "complete" if row_count >= int(config.target_rows) else "fallback"
    if row_count < int(config.fallback_rows):
        target_status = "partial"
    exact_blocking_ratio = float(ratio_summary["blocked_ratio"])
    if exact_blocking_ratio > 0.70:
        stage_status = "blocked"
    elif exact_blocking_ratio > 0.50:
        stage_status = "partial" if target_status == "fallback" else "fallback"
    else:
        stage_status = target_status
    actual_replay = bool(
        row_count > 0
        and set(frame.get("replay_generation_path", [])) == {"simulate_primitive_rollout"}
        and not frame.get("scaffold_case_table_only", pd.Series([True])).astype(bool).any()
    )
    r9_complete = bool(actual_replay and stage_status in {"complete", "fallback"} and file_status == "pass")
    manifest = {
        "run_id": int(config.run_id),
        "stage": "R9_W3_generalisation_replay",
        "source_replay": Path(config.source_replay).as_posix(),
        "archive_source_info": source_info.__dict__,
        "W_layer": "W3",
        "surrogate_family": "randomised_gp_corrected_annular_gaussian",
        "R9_W3_generalisation_complete": r9_complete,
        "generalisation_status": stage_status,
        "case_count": row_count,
        "target_rows": int(config.target_rows),
        "fallback_rows": int(config.fallback_rows),
        "blocked_ratio": float(ratio_summary["blocked_ratio"]),
        "approximate_ratio": float(ratio_summary["approximate_ratio"]),
        "file_size_status": file_status,
        "actual_model_backed_replay": actual_replay,
        "randomisation_scope": [
            "fan_position",
            "fan_power",
            "active_fan_subset",
            "amplitude",
            "width",
            "centre_shift",
            "residual_vertical_field_label",
            "uncertainty_scale",
            "mixed_primitive_start_state",
            "state_feedback_delay",
            "command_delay",
            "actuator_lag",
            "latency_jitter",
            "surface_effectiveness",
            "surface_neutral_bias",
            "surface_limit_scale",
            "mass_scale",
            "cg_offset",
            "inertia_scale",
            "surface_calibration_scale",
        ],
        "claim_status": "simulation_only_w3_generalisation_no_robustness_claim",
        "blocked_claims": ["W3_robustness", "environment_generalisation", "hardware_readiness"],
    }
    manifest_path = run_root / "manifests" / "w3_generalisation_manifest.json"
    filesystem_path(manifest_path).write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    _write_runtime_summary(run_root / "metrics" / "runtime_summary.csv", manifest)
    _write_outcome_summary(run_root / "metrics" / "outcome_summary.csv", frame)
    filesystem_path(run_root / "reports" / "w3_generalisation_report.md").write_text(
        "# W3 Generalisation Replay Report\n\nNo W3 robustness or environment-generalisation claim is made.\n",
        encoding="ascii",
    )
    write_claim_boundary_report(
        run_root / "reports" / "claim_boundary_report.md",
        stage="R9 W3 generalisation",
        status=stage_status,
        claim_status=str(manifest["claim_status"]),
        blocked_claims=tuple(str(item) for item in manifest["blocked_claims"]),
    )
    return {
        "run_root": run_root,
        "manifest": manifest_path,
        "case_table": run_root / "tables" / partition.relative_path,
        "table_manifest": table_manifest,
        "file_size_audit": run_root / "metrics" / "file_size_audit.csv",
    }


def _select_w3_source_rows(
    frame: pd.DataFrame,
    *,
    target_rows: int,
    fallback_rows: int,
    reuse_limit: int,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    source = frame.copy()
    requested = max(int(fallback_rows), min(int(target_rows), len(source) * max(1, int(reuse_limit))))
    strata = []
    for outcome in ("accepted", "weak", "boundary_terminal", "failed", "rejected"):
        subset = source[source.get("outcome_class", "").astype(str) == outcome]
        if not subset.empty:
            strata.append(subset)
    if not strata:
        strata = [source]
    rows = []
    reuse_counts: dict[str, int] = {}
    cursor = 0
    while len(rows) < requested:
        subset = strata[cursor % len(strata)]
        row = subset.iloc[(cursor // len(strata)) % len(subset)]
        key = str(row.get("rollout_id", row.get("source_rollout_id", row.name)))
        count = reuse_counts.get(key, 0)
        if count < max(1, int(reuse_limit)):
            reuse_counts[key] = count + 1
            selected = row.copy()
            selected["source_reuse_count"] = count + 1
            rows.append(selected)
        cursor += 1
        if cursor > requested * max(4, len(strata) * max(1, int(reuse_limit))):
            break
    return pd.DataFrame(rows)


def _w3_replay_row(
    *,
    row: dict[str, object],
    row_index: int,
    config: W3GeneralisationConfig,
) -> dict[str, object]:
    state = _state_from_source(row)
    seed = int(config.run_id) * 100_000 + int(row_index)
    primitive = primitive_by_id(str(row.get("primitive_id", "glide")))
    environment = environment_instance_for_mode("W3", "w3_randomised", seed)
    metadata = environment_metadata_from_instance(environment)
    binding = resolve_surrogate_binding("W3", metadata, randomisation_seed=seed)
    wind = wind_field_for_binding(binding)
    implementation = implementation_instance_for_layer("W3", seed, latency_case="nominal")
    plant = plant_instance_for_layer("W3", seed)
    context = build_environment_context(
        state,
        wind_field=wind,
        metadata=metadata,
        latency_case=implementation.latency_case,
        actuator_case="nominal",
        surrogate_binding=binding,
    )
    rollout_config = RolloutConfig(
        W_layer="W3",
        rollout_backend="model_backed_feedback",
        wind_mode=binding.wind_mode,
    )
    rollout_id = f"w3_r{config.run_id:03d}_{row_index:06d}"
    if binding.surrogate_binding_status != READY_STATUS:
        evidence = blocked_rollout_evidence(
            rollout_id=rollout_id,
            episode_id=f"w3_episode_{row_index:06d}",
            initial_state=state,
            context=context,
            primitive=primitive,
            config=rollout_config,
            failure_label="W3_surrogate_binding_blocked",
        )
    else:
        evidence = simulate_primitive_rollout(
            rollout_id=rollout_id,
            episode_id=f"w3_episode_{row_index:06d}",
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
    result.update({f"environment_instance_{key}": value for key, value in environment_instance_row(environment).items()})
    result.update({f"implementation_instance_{key}": value for key, value in implementation_instance_row(implementation).items()})
    result.update({f"plant_instance_{key}": value for key, value in plant_instance_row(plant).items()})
    result.update(
        {
            "source_rollout_id": row.get("rollout_id", row.get("source_rollout_id", "")),
            "source_outcome_class": row.get("outcome_class", row.get("w2_outcome_class", "")),
            "source_boundary_use_class": row.get("boundary_use_class", ""),
            "source_reuse_count": int(row.get("source_reuse_count", 1)),
            "replay_generation_path": "simulate_primitive_rollout",
            "scaffold_case_table_only": False,
            "nondecomposable_gp_grid_effect_status": "approximate",
            "approximate_limitation_label": "active_fan_subset_and_per_fan_power_not_exactly_decomposable",
        }
    )
    for name in STATE_NAMES:
        result[f"entry_{name}"] = float(state[STATE_NAMES.index(name)])
    return result


def _state_from_source(row: dict[str, object]) -> np.ndarray:
    return np.asarray(
        [float(row.get(f"initial_{name}", row.get(f"entry_{name}", 0.0))) for name in STATE_NAMES],
        dtype=float,
    )


def _write_blocked_outputs(
    *,
    config: W3GeneralisationConfig,
    run_root: Path,
    reason: str,
) -> dict[str, object]:
    empty_path = run_root / "tables" / "w3_generalisation_rows.csv"
    pd.DataFrame().to_csv(filesystem_path(empty_path), index=False)
    manifest = {
        "run_id": int(config.run_id),
        "stage": "R9_W3_generalisation_replay",
        "generalisation_status": "blocked",
        "blocked_reason": str(reason),
        "R9_W3_generalisation_complete": False,
        "case_count": 0,
        "actual_model_backed_replay": False,
        "claim_status": "simulation_only_w3_blocked_no_robustness_claim",
        "blocked_claims": ["W3_robustness", "environment_generalisation", "hardware_readiness"],
    }
    manifest_path = run_root / "manifests" / "w3_generalisation_manifest.json"
    filesystem_path(manifest_path).write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    write_claim_boundary_report(
        run_root / "reports" / "claim_boundary_report.md",
        stage="R9 W3 generalisation",
        status="blocked",
        claim_status=str(manifest["claim_status"]),
        blocked_claims=tuple(str(item) for item in manifest["blocked_claims"]),
    )
    write_file_size_audit(run_root, run_root / "metrics" / "file_size_audit.csv")
    return {"run_root": run_root, "manifest": manifest_path, "case_table": empty_path}


def _write_runtime_summary(path: Path, manifest: dict[str, object]) -> None:
    pd.DataFrame(
        [
            {
                "run_id": manifest["run_id"],
                "stage": manifest["stage"],
                "row_count": manifest["case_count"],
                "stage_status": manifest["generalisation_status"],
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
    run_w3_generalisation(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
