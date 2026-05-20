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

from dense_archive_schema import (  # noqa: E402
    BRANCH_DECISION_SCOPE,
    DenseArchivePlanConfig,
)
from dense_archive_table_io import (  # noqa: E402
    TableManifest,
    TablePartition,
    resolve_storage_format,
    table_extension,
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
# 1) Paths and Data Containers
# 2) Planning Table Construction
# 3) Output Writers
# 4) Public Runner and CLI
# =============================================================================


# =============================================================================
# 1) Paths and Data Containers
# =============================================================================
DEFAULT_RESULT_ROOT = CONTROL_DIR / "05_Results" / "10_dense_archive_planning"
PROTECTED_DEFAULT_RUN_IDS = frozenset({7, 8, 9, 10, 11})
W0_BRANCH_IDS = ("single_fan_branch", "four_fan_branch")
NO_CLAIM_TEXT = (
    "W0 partitioned planning only; no production W0 replay, W1/W2/W3/W4/W5 "
    "evidence, mission evaluation, hardware validation, or sim-to-real transfer "
    "is claimed."
)


@dataclass(frozen=True)
class W0PartitionedPlanningConfig:
    run_id: int = 12
    source_planning_run_id: int = 10
    result_root: Path | None = None
    target_trials_total: int = 500000
    target_trials_per_branch: int = 250000
    floor_trials_per_branch: int = 150000
    pilot_start_states_per_family_target_direction: int = 3677
    random_seed: int = 20260525
    storage_format: str = "auto"
    partition_rows: int = 25000
    overwrite: bool = False


@dataclass(frozen=True)
class W0PartitionedPlanningOutputs:
    root: Path
    manifest_json: Path
    table_manifest_json: Path
    report_md: Path
    branch_counts_csv: Path
    schema_summary_csv: Path
    schema_json: Path
    single_preview_csv: Path
    four_preview_csv: Path

    def as_dict(self) -> dict[str, Path]:
        return {
            "root": self.root,
            "manifest_json": self.manifest_json,
            "table_manifest_json": self.table_manifest_json,
            "report_md": self.report_md,
            "branch_counts_csv": self.branch_counts_csv,
            "schema_summary_csv": self.schema_summary_csv,
            "schema_json": self.schema_json,
            "single_preview_csv": self.single_preview_csv,
            "four_preview_csv": self.four_preview_csv,
        }


def _active_result_root(config: W0PartitionedPlanningConfig) -> Path:
    return DEFAULT_RESULT_ROOT if config.result_root is None else Path(config.result_root)


def _output_paths(config: W0PartitionedPlanningConfig) -> W0PartitionedPlanningOutputs:
    root = _active_result_root(config) / f"{int(config.run_id):03d}"
    suffix = f"s{int(config.run_id):03d}"
    return W0PartitionedPlanningOutputs(
        root=root,
        manifest_json=root
        / "manifests"
        / f"w0_partitioned_planning_manifest_{suffix}.json",
        table_manifest_json=root / "manifests" / f"table_manifest_{suffix}.json",
        report_md=root / "reports" / f"w0_partitioned_planning_report_{suffix}.md",
        branch_counts_csv=root
        / "metrics_summary"
        / f"w0_planning_branch_counts_{suffix}.csv",
        schema_summary_csv=root
        / "metrics_summary"
        / f"w0_planning_schema_summary_{suffix}.csv",
        schema_json=root / "schema" / f"w0_partitioned_planning_schema_{suffix}.json",
        single_preview_csv=root
        / "sample_preview"
        / f"w0_single_fan_start_preview_{suffix}.csv",
        four_preview_csv=root
        / "sample_preview"
        / f"w0_four_fan_start_preview_{suffix}.csv",
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
def _validate_config(config: W0PartitionedPlanningConfig) -> None:
    if int(config.run_id) <= 0:
        raise ValueError("run_id must be positive.")
    if int(config.source_planning_run_id) <= 0:
        raise ValueError("source_planning_run_id must be positive.")
    if int(config.target_trials_total) != 2 * int(config.target_trials_per_branch):
        raise ValueError("target_trials_total must equal two target_trials_per_branch.")
    if int(config.target_trials_per_branch) <= 0:
        raise ValueError("target_trials_per_branch must be positive.")
    if int(config.floor_trials_per_branch) <= 0:
        raise ValueError("floor_trials_per_branch must be positive.")
    if int(config.partition_rows) <= 0:
        raise ValueError("partition_rows must be positive.")
    resolve_storage_format(config.storage_format)


def _validate_output_guardrails(
    config: W0PartitionedPlanningConfig,
    outputs: W0PartitionedPlanningOutputs,
) -> None:
    if config.result_root is None and int(config.run_id) in PROTECTED_DEFAULT_RUN_IDS:
        raise ValueError("refusing to write protected default dense-planning run.")
    if outputs.root.exists() and not bool(config.overwrite):
        raise ValueError(
            f"output directory already exists and overwrite=False: {outputs.root}"
        )


def _build_w0_tables(
    config: W0PartitionedPlanningConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # The dense sampling helper still builds paired W0/W1 planning rows. This
    # W0 production input step filters those rows before any replay is run.
    plan_config = DenseArchivePlanConfig(
        run_id=int(config.run_id),
        random_seed=int(config.random_seed),
        pilot_start_states_per_family_target_direction=int(
            config.pilot_start_states_per_family_target_direction
        ),
    )
    start_states = build_start_state_manifest(plan_config)
    all_candidates = build_dry_run_candidate_inventory(plan_config, start_states)
    w0_candidates = all_candidates[
        all_candidates["test_environment_mode"].astype(str).str.startswith("W0_")
    ].copy()
    selected_candidates = _select_branch_quota(
        w0_candidates,
        per_branch=int(config.target_trials_per_branch),
    )
    sample_ids = set(selected_candidates["sample_id"].astype(str))
    selected_starts = start_states[
        start_states["sample_id"].astype(str).isin(sample_ids)
    ].copy()
    return selected_starts, selected_candidates


def _select_branch_quota(candidates: pd.DataFrame, *, per_branch: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for branch_id in W0_BRANCH_IDS:
        branch = candidates[
            candidates["layout_branch_id"].astype(str).eq(branch_id)
        ].copy()
        if len(branch) < int(per_branch):
            raise RuntimeError(
                "partitioned W0 planning generated too few candidates for "
                f"{branch_id}: available={len(branch)}, required={int(per_branch)}"
            )
        branch["_selection_key"] = [
            _candidate_sort_key(row) for row in branch.to_dict(orient="records")
        ]
        rows.append(
            branch.sort_values("_selection_key", kind="mergesort")
            .head(int(per_branch))
            .drop(columns=["_selection_key"])
        )
    return pd.concat(rows, ignore_index=True)


def _candidate_sort_key(row: dict[str, object]) -> str:
    target = _target_text(row.get("target_heading_deg", ""))
    direction = _direction_text(row.get("direction_sign", ""))
    return "|".join(
        (
            str(row.get("layout_branch_id", "")),
            str(row.get("family", "")),
            target,
            direction,
            str(row.get("start_class", "")),
            str(row.get("candidate_id", "")),
        )
    )


# =============================================================================
# 3) Output Writers
# =============================================================================
def _prepare_output_tree(
    config: W0PartitionedPlanningConfig,
    outputs: W0PartitionedPlanningOutputs,
) -> None:
    _validate_output_guardrails(config, outputs)
    if outputs.root.exists() and config.overwrite:
        _clear_output_tree(outputs.root)
    for path in (
        outputs.manifest_json.parent,
        outputs.report_md.parent,
        outputs.branch_counts_csv.parent,
        outputs.single_preview_csv.parent,
        outputs.schema_json.parent,
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
    partition_rows: int,
) -> list[TablePartition]:
    partitions: list[TablePartition] = []
    extension = table_extension(storage_format)
    for branch_id in W0_BRANCH_IDS:
        branch = frame[frame["layout_branch_id"].astype(str).eq(branch_id)].copy()
        branch = branch.reset_index(drop=True)
        branch_root = root / "tables" / table_name / f"layout_branch_id={branch_id}"
        for start in range(0, len(branch), int(partition_rows)):
            index = start // int(partition_rows)
            part = branch.iloc[start : start + int(partition_rows)].copy()
            path = branch_root / f"part-{index:05d}.{extension}"
            partitions.append(
                write_table_partition(part, path, storage_format=storage_format)
            )
    return partitions


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="ascii")


def _write_report(path: Path, manifest: dict[str, object]) -> None:
    lines = [
        "# W0 Partitioned Planning Report",
        "",
        NO_CLAIM_TEXT,
        "",
        f"- Run id: `{manifest['run_id']}`",
        f"- Source planning run id: `{manifest['source_planning_run_id']}`",
        f"- Target W0 trials total: `{manifest['target_trials_total']}`",
        f"- Target W0 trials per branch: `{manifest['target_trials_per_branch']}`",
        f"- Storage format: `{manifest['storage_format']}`",
        f"- Branch-local decisions only: `{str(manifest['branch_local_decisions_only']).lower()}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="ascii")


def _branch_counts(
    start_states: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for branch_id in W0_BRANCH_IDS:
        rows.append(
            {
                "layout_branch_id": branch_id,
                "start_state_rows": int(
                    start_states["layout_branch_id"].astype(str).eq(branch_id).sum()
                ),
                "candidate_rows": int(
                    candidates["layout_branch_id"].astype(str).eq(branch_id).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


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


def _manifest(
    config: W0PartitionedPlanningConfig,
    outputs: W0PartitionedPlanningOutputs,
    table_manifest: TableManifest,
    start_states: pd.DataFrame,
    candidates: pd.DataFrame,
) -> dict[str, object]:
    return {
        "run_id": int(config.run_id),
        "source_planning_run_id": int(config.source_planning_run_id),
        "target_trials_total": int(config.target_trials_total),
        "target_trials_per_branch": int(config.target_trials_per_branch),
        "floor_trials_per_branch": int(config.floor_trials_per_branch),
        "actual_candidate_rows_total": int(len(candidates)),
        "actual_candidate_rows_by_branch": {
            branch: int(candidates["layout_branch_id"].astype(str).eq(branch).sum())
            for branch in W0_BRANCH_IDS
        },
        "storage_format": str(table_manifest.storage_format),
        "partition_rows": int(config.partition_rows),
        "w0_partitioned_planning_performed": True,
        "w0_replay_performed": False,
        "w1_w2_w3_w4_w5_performed": False,
        "hardware_or_mission_claim": False,
        "sim_to_real_transfer_claim": False,
        "branch_local_decisions_only": True,
        "branch_decision_scope": BRANCH_DECISION_SCOPE,
        "no_overclaiming_statement": NO_CLAIM_TEXT,
        "output_files": {
            "manifest": _path_text(outputs.manifest_json),
            "table_manifest": _path_text(outputs.table_manifest_json),
            "report": _path_text(outputs.report_md),
            "branch_counts": _path_text(outputs.branch_counts_csv),
            "schema_summary": _path_text(outputs.schema_summary_csv),
            "schema_json": _path_text(outputs.schema_json),
            "single_fan_preview": _path_text(outputs.single_preview_csv),
            "four_fan_preview": _path_text(outputs.four_preview_csv),
        },
    }


# =============================================================================
# 4) Public Runner and CLI
# =============================================================================
def run_w0_partitioned_planning(
    *,
    run_id: int = 12,
    source_planning_run_id: int = 10,
    result_root: Path | None = None,
    target_trials_total: int = 500000,
    target_trials_per_branch: int = 250000,
    floor_trials_per_branch: int = 150000,
    pilot_start_states_per_family_target_direction: int = 3677,
    random_seed: int = 20260525,
    storage_format: str = "auto",
    partition_rows: int = 25000,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write partitioned W0 branch-local planning inputs without replay."""

    config = W0PartitionedPlanningConfig(
        run_id=int(run_id),
        source_planning_run_id=int(source_planning_run_id),
        result_root=result_root,
        target_trials_total=int(target_trials_total),
        target_trials_per_branch=int(target_trials_per_branch),
        floor_trials_per_branch=int(floor_trials_per_branch),
        pilot_start_states_per_family_target_direction=int(
            pilot_start_states_per_family_target_direction
        ),
        random_seed=int(random_seed),
        storage_format=str(storage_format),
        partition_rows=int(partition_rows),
        overwrite=bool(overwrite),
    )
    _validate_config(config)
    outputs = _output_paths(config)
    _prepare_output_tree(config, outputs)
    effective_format = resolve_storage_format(config.storage_format)

    start_states, candidates = _build_w0_tables(config)
    partitions = []
    partitions.extend(
        _write_partitioned_table(
            start_states,
            root=outputs.root,
            table_name="start_states",
            storage_format=effective_format,
            partition_rows=int(config.partition_rows),
        )
    )
    partitions.extend(
        _write_partitioned_table(
            candidates,
            root=outputs.root,
            table_name="candidate_index",
            storage_format=effective_format,
            partition_rows=int(config.partition_rows),
        )
    )
    table_manifest = TableManifest(
        run_id=int(config.run_id),
        root=_path_text(outputs.root),
        storage_format=effective_format,
        tables=tuple(partitions),
    )
    write_table_manifest(outputs.table_manifest_json, table_manifest)

    branch_counts = _branch_counts(start_states, candidates)
    schema_summary = _schema_summary(
        {"start_states": start_states, "candidate_index": candidates}
    )
    branch_counts.to_csv(outputs.branch_counts_csv, index=False)
    schema_summary.to_csv(outputs.schema_summary_csv, index=False)
    _write_json(
        outputs.schema_json,
        {
            "tables": {
                "start_states": list(start_states.columns),
                "candidate_index": list(candidates.columns),
            }
        },
    )
    start_states[
        start_states["layout_branch_id"].astype(str).eq("single_fan_branch")
    ].head(1000).to_csv(outputs.single_preview_csv, index=False)
    start_states[
        start_states["layout_branch_id"].astype(str).eq("four_fan_branch")
    ].head(1000).to_csv(outputs.four_preview_csv, index=False)

    manifest = _manifest(config, outputs, table_manifest, start_states, candidates)
    _write_json(outputs.manifest_json, manifest)
    _write_report(outputs.report_md, manifest)
    return outputs.as_dict()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int, default=12)
    parser.add_argument("--source-planning-run-id", type=int, default=10)
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--target-trials-total", type=int, default=500000)
    parser.add_argument("--target-trials-per-branch", type=int, default=250000)
    parser.add_argument("--floor-trials-per-branch", type=int, default=150000)
    parser.add_argument(
        "--pilot-start-states-per-family-target-direction",
        type=int,
        default=3677,
    )
    parser.add_argument("--random-seed", type=int, default=20260525)
    parser.add_argument("--storage-format", default="auto")
    parser.add_argument("--partition-rows", type=int, default=25000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    paths = run_w0_partitioned_planning(
        run_id=args.run_id,
        source_planning_run_id=args.source_planning_run_id,
        result_root=args.result_root,
        target_trials_total=args.target_trials_total,
        target_trials_per_branch=args.target_trials_per_branch,
        floor_trials_per_branch=args.floor_trials_per_branch,
        pilot_start_states_per_family_target_direction=(
            args.pilot_start_states_per_family_target_direction
        ),
        random_seed=args.random_seed,
        storage_format=args.storage_format,
        partition_rows=args.partition_rows,
        overwrite=args.overwrite,
    )
    print(f"w0_partitioned_planning_outputs={_path_text(paths['root'])}")
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
