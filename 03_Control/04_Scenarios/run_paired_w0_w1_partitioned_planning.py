from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


SCENARIOS_DIR = Path(__file__).resolve().parent
CONTROL_DIR = SCENARIOS_DIR.parents[0]
REPO_ROOT = CONTROL_DIR.parents[0]
PRIMITIVES_DIR = CONTROL_DIR / "03_Primitives"
for path in (PRIMITIVES_DIR, SCENARIOS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dense_archive_chunking import partition_path  # noqa: E402
from dense_archive_runtime import (  # noqa: E402
    RUNTIME_CORE_VERSION,
    STORAGE_CONTRACT_VERSION,
    runtime_manifest_fields,
)
from dense_archive_schema import (  # noqa: E402
    BRANCH_DECISION_SCOPE,
    DenseArchivePlanConfig,
)
from dense_archive_table_io import (  # noqa: E402
    TableManifest,
    TablePartition,
    resolve_storage_format,
    write_table_manifest,
    write_table_partition,
)
from dense_start_state_sampling import (  # noqa: E402
    build_dry_run_candidate_inventory,
    build_start_state_manifest,
)


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Constants and Data Containers
# 2) Planning Table Construction
# 3) Output Writers
# 4) Public Runner and CLI
# =============================================================================


# =============================================================================
# 1) Constants and Data Containers
# =============================================================================
DEFAULT_RESULT_ROOT = CONTROL_DIR / "05_Results" / "10_dense_archive_planning"
PROTECTED_DEFAULT_RUN_IDS = frozenset({7, 8, 9, 10, 11})
PAIRED_ENVIRONMENT_MODES = (
    "W0_single_fan_branch",
    "W1_single_fan",
    "W0_four_fan_branch",
    "W1_four_fan",
)
W0_ENVIRONMENT_MODES = ("W0_single_fan_branch", "W0_four_fan_branch")
W1_ENVIRONMENT_MODES = ("W1_single_fan", "W1_four_fan")
BRANCH_IDS = ("single_fan_branch", "four_fan_branch")
SIMULATION_STAGE = "paired_w0_w1_proof"
NO_CLAIM_TEXT = (
    "Paired W0/W1 partitioned planning proof only; no full W1 production, "
    "W2/W3/W4/W5, mission, hardware, or sim-to-real claim is made."
)


@dataclass(frozen=True)
class PairedW0W1PartitionedPlanningConfig:
    run_id: int
    source_planning_run_id: int = 10
    result_root: Path | None = None
    w0_target_trials_per_branch: int = 250000
    w1_floor_trials_per_branch: int = 350000
    w1_target_trials_per_branch: int = 500000
    pilot_start_states_per_family_target_direction: int = 5000
    random_seed: int = 20260526
    storage_format: str = "auto"
    partition_rows: int = 2500
    include_w0: bool = True
    include_w1: bool = True
    overwrite: bool = False


@dataclass(frozen=True)
class PairedPlanningOutputs:
    root: Path
    manifest_json: Path
    table_manifest_json: Path
    report_md: Path
    branch_environment_counts_csv: Path
    schema_summary_csv: Path
    schema_json: Path
    preview_csv: Path

    def as_dict(self) -> dict[str, Path]:
        return {
            "root": self.root,
            "manifest_json": self.manifest_json,
            "table_manifest_json": self.table_manifest_json,
            "report_md": self.report_md,
            "branch_environment_counts_csv": self.branch_environment_counts_csv,
            "schema_summary_csv": self.schema_summary_csv,
            "schema_json": self.schema_json,
            "preview_csv": self.preview_csv,
        }


def _active_result_root(config: PairedW0W1PartitionedPlanningConfig) -> Path:
    return DEFAULT_RESULT_ROOT if config.result_root is None else Path(config.result_root)


def _outputs(config: PairedW0W1PartitionedPlanningConfig) -> PairedPlanningOutputs:
    root = _active_result_root(config) / f"{int(config.run_id):03d}"
    suffix = f"s{int(config.run_id):03d}"
    return PairedPlanningOutputs(
        root=root,
        manifest_json=root / "manifests" / f"paired_w0_w1_planning_manifest_{suffix}.json",
        table_manifest_json=root / "manifests" / f"table_manifest_{suffix}.json",
        report_md=root / "reports" / f"paired_w0_w1_planning_report_{suffix}.md",
        branch_environment_counts_csv=root
        / "metrics_summary"
        / f"paired_w0_w1_branch_environment_counts_{suffix}.csv",
        schema_summary_csv=root / "schema" / f"paired_w0_w1_schema_summary_{suffix}.csv",
        schema_json=root / "schema" / f"paired_w0_w1_schema_{suffix}.json",
        preview_csv=root / "sample_preview" / f"paired_w0_w1_preview_{suffix}.csv",
    )


def _path_text(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


# =============================================================================
# 2) Planning Table Construction
# =============================================================================
def _validate_config(config: PairedW0W1PartitionedPlanningConfig) -> None:
    if int(config.run_id) <= 0:
        raise ValueError("run_id must be positive.")
    if int(config.partition_rows) <= 0:
        raise ValueError("partition_rows must be positive.")
    for name, count in (
        ("w0_target_trials_per_branch", config.w0_target_trials_per_branch),
        ("w1_target_trials_per_branch", config.w1_target_trials_per_branch),
    ):
        if int(count) <= 0:
            raise ValueError(f"{name} must be positive.")
        if int(count) % int(config.partition_rows) != 0:
            raise ValueError(f"{name} must be divisible by partition_rows.")
    if int(config.w1_floor_trials_per_branch) > int(config.w1_target_trials_per_branch):
        raise ValueError("w1_floor_trials_per_branch must not exceed target.")
    resolve_storage_format(config.storage_format)


def _validate_output_guardrails(
    config: PairedW0W1PartitionedPlanningConfig,
    outputs: PairedPlanningOutputs,
) -> None:
    if config.result_root is None and int(config.run_id) in PROTECTED_DEFAULT_RUN_IDS:
        raise ValueError("refusing to write protected default dense-planning run.")
    if outputs.root.exists() and not bool(config.overwrite):
        raise ValueError(
            f"output directory already exists and overwrite=False: {outputs.root}"
        )


def _build_paired_tables(
    config: PairedW0W1PartitionedPlanningConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    plan_config = DenseArchivePlanConfig(
        run_id=int(config.run_id),
        random_seed=int(config.random_seed),
        pilot_start_states_per_family_target_direction=int(
            config.pilot_start_states_per_family_target_direction
        ),
    )
    start_states = build_start_state_manifest(plan_config)
    candidates = build_dry_run_candidate_inventory(plan_config, start_states)
    return _select_and_chunk_paired_tables(config, start_states, candidates)


def _select_and_chunk_paired_tables(
    config: PairedW0W1PartitionedPlanningConfig,
    start_states: pd.DataFrame,
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_frames: list[pd.DataFrame] = []
    start_frames: list[pd.DataFrame] = []
    start_by_sample = {
        str(row["sample_id"]): row for row in start_states.to_dict(orient="records")
    }
    for environment_mode in _active_environment_modes(config):
        target_count = _target_count_for_environment(config, environment_mode)
        environment = candidates[
            candidates["test_environment_mode"].astype(str).eq(environment_mode)
        ].copy()
        if len(environment) < int(target_count):
            raise RuntimeError(
                f"not enough candidates for {environment_mode}: "
                f"available={len(environment)}, required={int(target_count)}"
            )
        environment["_selection_key"] = [
            _candidate_sort_key(row) for row in environment.to_dict(orient="records")
        ]
        environment = (
            environment.sort_values("_selection_key", kind="mergesort")
            .head(int(target_count))
            .drop(columns=["_selection_key"])
            .reset_index(drop=True)
        )
        metadata = _chunk_metadata(
            row_count=len(environment),
            chunk_size=int(config.partition_rows),
        )
        environment = pd.concat([environment, metadata], axis=1)
        selected_frames.append(environment)

        rows: list[dict[str, object]] = []
        for candidate in environment.to_dict(orient="records"):
            sample_id = str(candidate["sample_id"])
            if sample_id not in start_by_sample:
                raise KeyError(f"candidate sample_id missing from start states: {sample_id}")
            row = dict(start_by_sample[sample_id])
            for column in (
                "candidate_id",
                "test_environment_mode",
                "paired_environment_mode",
                "latency_case_planned",
                "fan_config_id",
                "updraft_model_id",
            ):
                row[column] = candidate.get(column, "")
            rows.append(row)
        start_frame = pd.concat([pd.DataFrame(rows), metadata.copy()], axis=1)
        start_frames.append(start_frame)
    return (
        pd.concat(start_frames, ignore_index=True),
        pd.concat(selected_frames, ignore_index=True),
    )


def _active_environment_modes(config: PairedW0W1PartitionedPlanningConfig) -> tuple[str, ...]:
    modes: list[str] = []
    if config.include_w0:
        modes.extend(W0_ENVIRONMENT_MODES)
    if config.include_w1:
        modes.extend(W1_ENVIRONMENT_MODES)
    return tuple(mode for mode in PAIRED_ENVIRONMENT_MODES if mode in modes)


def _target_count_for_environment(
    config: PairedW0W1PartitionedPlanningConfig,
    environment_mode: str,
) -> int:
    if str(environment_mode).startswith("W0_"):
        return int(config.w0_target_trials_per_branch)
    return int(config.w1_target_trials_per_branch)


def _chunk_metadata(*, row_count: int, chunk_size: int) -> pd.DataFrame:
    indices = pd.Series(range(int(row_count)), dtype="int64")
    return pd.DataFrame(
        {
            "archive_chunk_index": indices // int(chunk_size),
            "archive_chunk_count": int(row_count) // int(chunk_size),
            "chunk_local_index": indices % int(chunk_size),
            "archive_chunk_size": int(chunk_size),
            "archive_branch_trial_index": indices,
        }
    )


def _candidate_sort_key(row: dict[str, object]) -> str:
    return "|".join(
        (
            str(row.get("layout_branch_id", "")),
            str(row.get("test_environment_mode", "")),
            str(row.get("family", "")),
            _target_text(row.get("target_heading_deg", "")),
            _direction_text(row.get("direction_sign", "")),
            str(row.get("start_class", "")),
            str(row.get("candidate_id", "")),
        )
    )


# =============================================================================
# 3) Output Writers
# =============================================================================
def _prepare_output_tree(
    config: PairedW0W1PartitionedPlanningConfig,
    outputs: PairedPlanningOutputs,
) -> None:
    _validate_output_guardrails(config, outputs)
    if outputs.root.exists() and config.overwrite:
        _clear_output_tree(outputs.root)
    for path in (
        outputs.manifest_json.parent,
        outputs.report_md.parent,
        outputs.branch_environment_counts_csv.parent,
        outputs.schema_summary_csv.parent,
        outputs.preview_csv.parent,
        outputs.root / "tables",
    ):
        path.mkdir(parents=True, exist_ok=True)


def _clear_output_tree(root: Path) -> None:
    root_resolved = root.resolve()
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        resolved = path.resolve()
        if resolved != root_resolved and root_resolved not in resolved.parents:
            raise RuntimeError(f"refusing to clear path outside output root: {path}")
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir() and path != root:
            try:
                path.rmdir()
            except OSError:
                pass


def _write_partitioned_table(
    frame: pd.DataFrame,
    *,
    root: Path,
    table_name: str,
    storage_format: str,
    compression_level: int,
) -> list[TablePartition]:
    partitions: list[TablePartition] = []
    keys = ["layout_branch_id", "test_environment_mode", "archive_chunk_index"]
    for key, part in frame.groupby(keys, sort=True):
        branch_id, environment_mode, chunk_index = key
        part = part.sort_values("chunk_local_index", kind="mergesort").copy()
        output_path = partition_path(
            root,
            table_name=table_name,
            layout_branch_id=str(branch_id),
            test_environment_mode=str(environment_mode),
            chunk_index=int(chunk_index),
            storage_format=storage_format,
        )
        partitions.append(
            write_table_partition(
                part,
                output_path,
                storage_format=storage_format,
                compression_level=int(compression_level),
            )
        )
    return partitions


def _branch_environment_counts(candidates: pd.DataFrame) -> pd.DataFrame:
    return (
        candidates.groupby(["layout_branch_id", "fan_layout", "test_environment_mode"], dropna=False)
        .size()
        .reset_index(name="candidate_rows")
        .sort_values(["layout_branch_id", "test_environment_mode"])
    )


def _schema_summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for table_name, frame in tables.items():
        for column in frame.columns:
            rows.append(
                {
                    "table_name": table_name,
                    "column_name": str(column),
                    "dtype": str(frame[column].dtype),
                }
            )
    return pd.DataFrame(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="ascii",
    )


def _write_report(path: Path, manifest: dict[str, object]) -> None:
    lines = [
        "# Paired W0/W1 Partitioned Planning Report",
        "",
        NO_CLAIM_TEXT,
        "",
        f"- Run id: `{manifest['run_id']}`",
        f"- Storage format: `{manifest['storage_format']}`",
        f"- Active environment modes: `{manifest['environment_modes']}`",
        f"- Branch decision scope: `{manifest['branch_decision_scope']}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="ascii")


def _manifest(
    *,
    config: PairedW0W1PartitionedPlanningConfig,
    outputs: PairedPlanningOutputs,
    table_manifest: TableManifest,
    start_states: pd.DataFrame,
    candidates: pd.DataFrame,
) -> dict[str, object]:
    payload = {
        "run_id": int(config.run_id),
        "source_planning_run_id": int(config.source_planning_run_id),
        "environment_modes": list(_active_environment_modes(config)),
        "w0_target_trials_per_branch": int(config.w0_target_trials_per_branch),
        "w1_floor_trials_per_branch": int(config.w1_floor_trials_per_branch),
        "w1_target_trials_per_branch": int(config.w1_target_trials_per_branch),
        "storage_format": str(table_manifest.storage_format),
        "partition_rows": int(config.partition_rows),
        "candidate_rows_total": int(len(candidates)),
        "start_state_rows_total": int(len(start_states)),
        "w1_selected_independently_of_w0_success": True,
        "branch_local_decisions_only": True,
        "branch_decision_scope": BRANCH_DECISION_SCOPE,
        "no_overclaiming_statement": NO_CLAIM_TEXT,
        "output_files": {
            "manifest": _path_text(outputs.manifest_json),
            "table_manifest": _path_text(outputs.table_manifest_json),
            "report": _path_text(outputs.report_md),
            "branch_environment_counts": _path_text(outputs.branch_environment_counts_csv),
            "schema_summary": _path_text(outputs.schema_summary_csv),
            "schema_json": _path_text(outputs.schema_json),
            "preview": _path_text(outputs.preview_csv),
        },
    }
    payload.update(
        runtime_manifest_fields(
            simulation_stage=SIMULATION_STAGE,
            environment_mode="multiple",
            branch_decision_scope=BRANCH_DECISION_SCOPE,
        )
    )
    return payload


# =============================================================================
# 4) Public Runner and CLI
# =============================================================================
def run_paired_w0_w1_partitioned_planning(
    *,
    run_id: int,
    source_planning_run_id: int = 10,
    result_root: Path | None = None,
    w0_target_trials_per_branch: int = 250000,
    w1_floor_trials_per_branch: int = 350000,
    w1_target_trials_per_branch: int = 500000,
    pilot_start_states_per_family_target_direction: int = 5000,
    random_seed: int = 20260526,
    storage_format: str = "auto",
    partition_rows: int = 2500,
    include_w0: bool = True,
    include_w1: bool = True,
    overwrite: bool = False,
) -> dict[str, Path]:
    config = PairedW0W1PartitionedPlanningConfig(
        run_id=int(run_id),
        source_planning_run_id=int(source_planning_run_id),
        result_root=result_root,
        w0_target_trials_per_branch=int(w0_target_trials_per_branch),
        w1_floor_trials_per_branch=int(w1_floor_trials_per_branch),
        w1_target_trials_per_branch=int(w1_target_trials_per_branch),
        pilot_start_states_per_family_target_direction=int(
            pilot_start_states_per_family_target_direction
        ),
        random_seed=int(random_seed),
        storage_format=str(storage_format),
        partition_rows=int(partition_rows),
        include_w0=bool(include_w0),
        include_w1=bool(include_w1),
        overwrite=bool(overwrite),
    )
    _validate_config(config)
    outputs = _outputs(config)
    _prepare_output_tree(config, outputs)
    effective_format = resolve_storage_format(config.storage_format)
    start_states, candidates = _build_paired_tables(config)

    partitions: list[TablePartition] = []
    partitions.extend(
        _write_partitioned_table(
            start_states,
            root=outputs.root,
            table_name="start_states",
            storage_format=effective_format,
            compression_level=1,
        )
    )
    partitions.extend(
        _write_partitioned_table(
            candidates,
            root=outputs.root,
            table_name="candidate_index",
            storage_format=effective_format,
            compression_level=1,
        )
    )
    table_manifest = TableManifest(
        run_id=int(config.run_id),
        root=_path_text(outputs.root),
        storage_format=effective_format,
        tables=tuple(partitions),
    )
    write_table_manifest(outputs.table_manifest_json, table_manifest)
    counts = _branch_environment_counts(candidates)
    schema = _schema_summary({"start_states": start_states, "candidate_index": candidates})
    counts.to_csv(outputs.branch_environment_counts_csv, index=False)
    schema.to_csv(outputs.schema_summary_csv, index=False)
    _write_json(
        outputs.schema_json,
        {
            "runtime_core_version": RUNTIME_CORE_VERSION,
            "storage_contract_version": STORAGE_CONTRACT_VERSION,
            "tables": {
                "start_states": list(start_states.columns),
                "candidate_index": list(candidates.columns),
            },
        },
    )
    candidates.head(1000).to_csv(outputs.preview_csv, index=False)
    manifest = _manifest(
        config=config,
        outputs=outputs,
        table_manifest=table_manifest,
        start_states=start_states,
        candidates=candidates,
    )
    _write_json(outputs.manifest_json, manifest)
    _write_report(outputs.report_md, manifest)
    return outputs.as_dict()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--source-planning-run-id", type=int, default=10)
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--w0-target-trials-per-branch", type=int, default=250000)
    parser.add_argument("--w1-floor-trials-per-branch", type=int, default=350000)
    parser.add_argument("--w1-target-trials-per-branch", type=int, default=500000)
    parser.add_argument("--pilot-start-states-per-family-target-direction", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=20260526)
    parser.add_argument("--storage-format", default="auto")
    parser.add_argument("--partition-rows", type=int, default=2500)
    parser.add_argument("--exclude-w0", action="store_true")
    parser.add_argument("--exclude-w1", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    paths = run_paired_w0_w1_partitioned_planning(
        run_id=args.run_id,
        source_planning_run_id=args.source_planning_run_id,
        result_root=args.result_root,
        w0_target_trials_per_branch=args.w0_target_trials_per_branch,
        w1_floor_trials_per_branch=args.w1_floor_trials_per_branch,
        w1_target_trials_per_branch=args.w1_target_trials_per_branch,
        pilot_start_states_per_family_target_direction=(
            args.pilot_start_states_per_family_target_direction
        ),
        random_seed=args.random_seed,
        storage_format=args.storage_format,
        partition_rows=args.partition_rows,
        include_w0=not args.exclude_w0,
        include_w1=not args.exclude_w1,
        overwrite=args.overwrite,
    )
    print(f"paired_w0_w1_partitioned_planning_outputs={_path_text(paths['root'])}")
    return 0


def _target_text(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "none"
    return f"{numeric:08.3f}"


def _direction_text(value: object) -> str:
    try:
        numeric = int(float(value))
    except (TypeError, ValueError):
        numeric = 0
    return f"{numeric:+d}"


if __name__ == "__main__":
    raise SystemExit(main())
