from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd


SCENARIOS_DIR = Path(__file__).resolve().parent
CONTROL_DIR = SCENARIOS_DIR.parents[0]
REPO_ROOT = CONTROL_DIR.parents[0]
PRIMITIVES_DIR = CONTROL_DIR / "03_Primitives"
INNER_LOOP_DIR = CONTROL_DIR / "02_Inner_Loop"
for path in (INNER_LOOP_DIR, PRIMITIVES_DIR, SCENARIOS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dense_archive_chunking import (  # noqa: E402
    GenericChunkPaths,
    GenericChunkSpec,
    TIMING_FIELDS,
    chunk_status as generic_chunk_status,
    partition_path,
    remove_chunk_outputs as generic_remove_chunk_outputs,
    trial_outcome_paths,
)
from dense_archive_runtime import runtime_manifest_fields  # noqa: E402
from dense_archive_schema import BRANCH_DECISION_SCOPE  # noqa: E402
from dense_archive_table_io import (  # noqa: E402
    TablePartition,
    read_table_partition,
    resolve_storage_format,
    write_table_partition,
)
from dense_archive_trial_logging import DENSE_TRIAL_DESCRIPTOR_COLUMNS  # noqa: E402
from run_dense_archive_pilot_sweep import _run_pilot_replays  # noqa: E402
from run_paired_w0_w1_partitioned_planning import (  # noqa: E402
    PAIRED_ENVIRONMENT_MODES,
    SIMULATION_STAGE,
)


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Paths and Data Containers
# 2) Planning Reads and Validation
# 3) Chunk Execution and Manifest Writing
# 4) Public Runner and CLI
# =============================================================================


# =============================================================================
# 1) Paths and Data Containers
# =============================================================================
DEFAULT_RESULT_ROOT = CONTROL_DIR / "05_Results" / "12_paired_w0_w1_archive"
PLANNING_CAMPAIGN = "10_dense_archive_planning"
NO_CLAIM_TEXT = (
    "One paired W0/W1 archive chunk only; no full W1 production, W2/W3/W4/W5, "
    "mission, hardware, or sim-to-real claim is made."
)


@dataclass(frozen=True)
class PairedChunkConfig:
    run_id: int
    planning_run_id: int
    result_root: Path | None = None
    layout_branch_id: str = "single_fan_branch"
    test_environment_mode: str = "W1_single_fan"
    chunk_index: int = 0
    chunk_count: int = 1
    chunk_size: int = 2500
    latency_case: str = "nominal"
    dt_s: float = 0.02
    horizon_s: float = 0.60
    storage_format: str = "auto"
    compression_level: int = 1
    resume: bool = True
    overwrite_chunk: bool = False
    repair_incomplete: bool = False
    random_seed: int = 20260526
    simulation_stage: str = SIMULATION_STAGE


def active_result_root(config: PairedChunkConfig) -> Path:
    return DEFAULT_RESULT_ROOT if config.result_root is None else Path(config.result_root)


def archive_run_root(config: PairedChunkConfig) -> Path:
    return active_result_root(config) / f"{int(config.run_id):03d}"


def planning_run_root(config: PairedChunkConfig) -> Path:
    return active_result_root(config).parent / PLANNING_CAMPAIGN / f"{int(config.planning_run_id):03d}"


def generic_spec(config: PairedChunkConfig) -> GenericChunkSpec:
    return GenericChunkSpec(
        run_id=int(config.run_id),
        planning_run_id=int(config.planning_run_id),
        result_root=config.result_root,
        layout_branch_id=str(config.layout_branch_id),
        test_environment_mode=str(config.test_environment_mode),
        chunk_index=int(config.chunk_index),
        chunk_count=int(config.chunk_count),
        chunk_size=int(config.chunk_size),
        storage_format=str(config.storage_format),
        compression_level=int(config.compression_level),
        latency_case=str(config.latency_case),
        dt_s=float(config.dt_s),
        horizon_s=float(config.horizon_s),
        simulation_stage=str(config.simulation_stage),
    )


def output_paths(config: PairedChunkConfig) -> GenericChunkPaths:
    return trial_outcome_paths(generic_spec(config), run_root=archive_run_root(config))


def _path_text(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


# =============================================================================
# 2) Planning Reads and Validation
# =============================================================================
def validate_config(config: PairedChunkConfig) -> None:
    if int(config.run_id) <= 0 or int(config.planning_run_id) <= 0:
        raise ValueError("run_id and planning_run_id must be positive.")
    if str(config.test_environment_mode) not in PAIRED_ENVIRONMENT_MODES:
        raise ValueError("test_environment_mode is not a paired W0/W1 environment.")
    if int(config.chunk_index) < 0 or int(config.chunk_index) >= int(config.chunk_count):
        raise ValueError("chunk_index must be in [0, chunk_count).")
    if int(config.chunk_size) <= 0:
        raise ValueError("chunk_size must be positive.")
    if int(config.compression_level) < 0 or int(config.compression_level) > 9:
        raise ValueError("compression_level must be in [0, 9].")
    resolve_storage_format(config.storage_format)


def chunk_status(config: PairedChunkConfig) -> str:
    return generic_chunk_status(generic_spec(config), run_root=archive_run_root(config))


def remove_chunk_outputs(config: PairedChunkConfig) -> None:
    generic_remove_chunk_outputs(generic_spec(config), run_root=archive_run_root(config))


def _load_planning_chunk(
    config: PairedChunkConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    start = time.perf_counter()
    root = planning_run_root(config)
    candidate_path = partition_path(
        root,
        table_name="candidate_index",
        layout_branch_id=str(config.layout_branch_id),
        test_environment_mode=str(config.test_environment_mode),
        chunk_index=int(config.chunk_index),
        storage_format=str(config.storage_format),
    )
    start_path = partition_path(
        root,
        table_name="start_states",
        layout_branch_id=str(config.layout_branch_id),
        test_environment_mode=str(config.test_environment_mode),
        chunk_index=int(config.chunk_index),
        storage_format=str(config.storage_format),
    )
    if not candidate_path.exists():
        raise FileNotFoundError(f"missing paired candidate chunk: {candidate_path}")
    if not start_path.exists():
        raise FileNotFoundError(f"missing paired start-state chunk: {start_path}")
    candidates = read_table_partition(candidate_path, storage_format=config.storage_format)
    starts = read_table_partition(start_path, storage_format=config.storage_format)
    _validate_planning_chunk(starts, candidates, config)
    return starts, candidates, time.perf_counter() - start


def _validate_planning_chunk(
    starts: pd.DataFrame,
    candidates: pd.DataFrame,
    config: PairedChunkConfig,
) -> None:
    if len(candidates) != int(config.chunk_size):
        raise ValueError("candidate_index chunk row count mismatch.")
    if len(starts) != len(candidates):
        raise ValueError("start_states chunk row count mismatch.")
    if candidates["sample_id"].astype(str).to_list() != starts["sample_id"].astype(str).to_list():
        raise ValueError("candidate_index and start_states sample_id order mismatch.")
    for table_name, frame in (("candidate_index", candidates), ("start_states", starts)):
        for column in (
            "layout_branch_id",
            "test_environment_mode",
            "archive_chunk_index",
            "archive_chunk_count",
            "chunk_local_index",
            "archive_chunk_size",
        ):
            if column not in frame.columns:
                raise ValueError(f"{table_name} missing required column: {column}")
        if set(frame["layout_branch_id"].astype(str)) != {str(config.layout_branch_id)}:
            raise ValueError(f"{table_name} layout_branch_id mismatch.")
        if set(frame["test_environment_mode"].astype(str)) != {str(config.test_environment_mode)}:
            raise ValueError(f"{table_name} test_environment_mode mismatch.")
        if set(frame["archive_chunk_index"].astype(int)) != {int(config.chunk_index)}:
            raise ValueError(f"{table_name} archive_chunk_index mismatch.")
        if frame["chunk_local_index"].astype(int).to_list() != list(range(int(config.chunk_size))):
            raise ValueError(f"{table_name} chunk_local_index sequence mismatch.")


# =============================================================================
# 3) Chunk Execution and Manifest Writing
# =============================================================================
def _run_descriptors(
    starts: pd.DataFrame,
    candidates: pd.DataFrame,
    config: PairedChunkConfig,
) -> tuple[pd.DataFrame, float, float]:
    start = time.perf_counter()
    environment_offset = PAIRED_ENVIRONMENT_MODES.index(str(config.test_environment_mode))
    seed_offset = (
        environment_offset * int(config.chunk_count) * int(config.chunk_size)
        + int(config.chunk_index) * int(config.chunk_size)
    )
    replay_config = replace(config, random_seed=int(config.random_seed) + seed_offset)
    descriptors = _run_pilot_replays(
        starts,
        candidates.to_dict(orient="records"),
        replay_config,
    )
    return descriptors, time.perf_counter() - start, 0.0


def _write_chunk(
    descriptors: pd.DataFrame,
    config: PairedChunkConfig,
) -> tuple[TablePartition, float]:
    start = time.perf_counter()
    outputs = output_paths(config)
    partition = write_table_partition(
        descriptors,
        outputs.partition_path,
        storage_format=config.storage_format,
        compression_level=int(config.compression_level),
    )
    return partition, time.perf_counter() - start


def _write_manifest(
    *,
    config: PairedChunkConfig,
    partition: TablePartition,
    timing: dict[str, float],
    paired_environment_mode: str,
) -> None:
    outputs = output_paths(config)
    outputs.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    outputs.log_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "complete",
        "run_id": int(config.run_id),
        "planning_run_id": int(config.planning_run_id),
        "layout_branch_id": str(config.layout_branch_id),
        "test_environment_mode": str(config.test_environment_mode),
        "paired_environment_mode": str(paired_environment_mode),
        "paired_sample_key_scope": "branch_local_environment_pairs",
        "chunk_index": int(config.chunk_index),
        "chunk_count": int(config.chunk_count),
        "chunk_size": int(config.chunk_size),
        "storage_format": resolve_storage_format(config.storage_format),
        "compression_level": int(config.compression_level),
        "latency_case": str(config.latency_case),
        "dt_s": float(config.dt_s),
        "horizon_s": float(config.horizon_s),
        "row_count": int(partition.row_count),
        "partition_path": _path_text(outputs.partition_path),
        "checksum_sha256": str(partition.checksum_sha256),
        "branch_decision_scope": BRANCH_DECISION_SCOPE,
        "paired_w0_w1_chunk_replay_performed": True,
        "full_w1_production_claim": False,
        "w2_w3_w4_w5_performed": False,
        "hardware_or_mission_claim": False,
        "sim_to_real_transfer_claim": False,
        "no_overclaiming_statement": NO_CLAIM_TEXT,
        **runtime_manifest_fields(
            simulation_stage=str(config.simulation_stage),
            environment_mode=str(config.test_environment_mode),
            branch_decision_scope=BRANCH_DECISION_SCOPE,
        ),
        **{field: float(timing[field]) for field in TIMING_FIELDS},
    }
    tmp_path = outputs.manifest_json.with_name(f"{outputs.manifest_json.name}.tmp")
    tmp_path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="ascii")
    tmp_path.replace(outputs.manifest_json)
    outputs.log_path.write_text(
        f"status=complete row_count={int(partition.row_count)} total_s={float(timing['total_s']):.6f}\n",
        encoding="ascii",
    )


# =============================================================================
# 4) Public Runner and CLI
# =============================================================================
def run_paired_w0_w1_archive_chunk(
    *,
    run_id: int,
    planning_run_id: int,
    result_root: Path | None = None,
    layout_branch_id: str,
    test_environment_mode: str,
    chunk_index: int,
    chunk_count: int,
    chunk_size: int = 2500,
    latency_case: str = "nominal",
    dt_s: float = 0.02,
    horizon_s: float = 0.60,
    storage_format: str = "auto",
    compression_level: int = 1,
    resume: bool = True,
    overwrite_chunk: bool = False,
    repair_incomplete: bool = False,
    random_seed: int = 20260526,
) -> dict[str, Path]:
    config = PairedChunkConfig(
        run_id=int(run_id),
        planning_run_id=int(planning_run_id),
        result_root=result_root,
        layout_branch_id=str(layout_branch_id),
        test_environment_mode=str(test_environment_mode),
        chunk_index=int(chunk_index),
        chunk_count=int(chunk_count),
        chunk_size=int(chunk_size),
        latency_case=str(latency_case),
        dt_s=float(dt_s),
        horizon_s=float(horizon_s),
        storage_format=str(storage_format),
        compression_level=int(compression_level),
        resume=bool(resume),
        overwrite_chunk=bool(overwrite_chunk),
        repair_incomplete=bool(repair_incomplete),
        random_seed=int(random_seed),
    )
    validate_config(config)
    outputs = output_paths(config)
    if config.overwrite_chunk:
        remove_chunk_outputs(config)
    else:
        status = chunk_status(config)
        if status == "complete" and config.resume:
            return outputs.as_dict()
        if status == "corrupt" and config.repair_incomplete:
            remove_chunk_outputs(config)
        elif status == "corrupt":
            raise RuntimeError("paired chunk is incomplete/corrupt; use --repair-incomplete.")
        elif status == "complete":
            raise RuntimeError("paired chunk already complete; use --resume or --overwrite-chunk.")

    total_start = time.perf_counter()
    starts, candidates, planning_read_s = _load_planning_chunk(config)
    selection_s = 0.0
    paired_environment_mode = str(candidates["paired_environment_mode"].iloc[0])
    descriptors, simulation_s, descriptor_build_s = _run_descriptors(starts, candidates, config)
    if descriptors.empty:
        descriptors = pd.DataFrame(columns=DENSE_TRIAL_DESCRIPTOR_COLUMNS)
    partition, write_s = _write_chunk(descriptors, config)
    timing = {
        "planning_read_s": float(planning_read_s),
        "selection_s": float(selection_s),
        "simulation_s": float(simulation_s),
        "descriptor_build_s": float(descriptor_build_s),
        "write_s": float(write_s),
        "total_s": float(time.perf_counter() - total_start),
    }
    _write_manifest(
        config=config,
        partition=partition,
        timing=timing,
        paired_environment_mode=paired_environment_mode,
    )
    return outputs.as_dict()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--planning-run-id", type=int, required=True)
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--layout-branch-id", required=True)
    parser.add_argument("--test-environment-mode", required=True)
    parser.add_argument("--chunk-index", type=int, required=True)
    parser.add_argument("--chunk-count", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=2500)
    parser.add_argument("--latency-case", default="nominal")
    parser.add_argument("--dt-s", type=float, default=0.02)
    parser.add_argument("--horizon-s", type=float, default=0.60)
    parser.add_argument("--storage-format", default="auto")
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite-chunk", action="store_true")
    parser.add_argument("--repair-incomplete", action="store_true")
    parser.add_argument("--random-seed", type=int, default=20260526)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    paths = run_paired_w0_w1_archive_chunk(
        run_id=args.run_id,
        planning_run_id=args.planning_run_id,
        result_root=args.result_root,
        layout_branch_id=args.layout_branch_id,
        test_environment_mode=args.test_environment_mode,
        chunk_index=args.chunk_index,
        chunk_count=args.chunk_count,
        chunk_size=args.chunk_size,
        latency_case=args.latency_case,
        dt_s=args.dt_s,
        horizon_s=args.horizon_s,
        storage_format=args.storage_format,
        compression_level=args.compression_level,
        resume=args.resume,
        overwrite_chunk=args.overwrite_chunk,
        repair_incomplete=args.repair_incomplete,
        random_seed=args.random_seed,
    )
    print(f"paired_w0_w1_archive_chunk_outputs={_path_text(paths['root'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
