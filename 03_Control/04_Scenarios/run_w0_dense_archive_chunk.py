from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd


SCENARIOS_DIR = Path(__file__).resolve().parent
CONTROL_DIR = SCENARIOS_DIR.parents[0]
REPO_ROOT = CONTROL_DIR.parents[0]
PRIMITIVES_DIR = CONTROL_DIR / "03_Primitives"
INNER_LOOP_DIR = CONTROL_DIR / "02_Inner_Loop"
for path in (INNER_LOOP_DIR, PRIMITIVES_DIR, SCENARIOS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dense_archive_schema import BRANCH_DECISION_SCOPE  # noqa: E402
from dense_archive_table_io import (  # noqa: E402
    TablePartition,
    file_sha256,
    partition_row_count,
    read_table_partition,
    resolve_storage_format,
    table_extension,
    write_table_partition,
)
from dense_archive_trial_logging import DENSE_TRIAL_DESCRIPTOR_COLUMNS  # noqa: E402
from run_dense_archive_pilot_sweep import _run_pilot_replays  # noqa: E402


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Paths and Data Containers
# 2) Chunk Completion and Planning Reads
# 3) Chunk Execution and Manifest Writing
# 4) Public Runner and CLI
# =============================================================================


# =============================================================================
# 1) Paths and Data Containers
# =============================================================================
DEFAULT_RESULT_ROOT = CONTROL_DIR / "05_Results" / "11_w0_dense_archive"
PLANNING_CAMPAIGN = "10_dense_archive_planning"
W0_BRANCH_IDS = ("single_fan_branch", "four_fan_branch")
NO_CLAIM_TEXT = (
    "One W0 dense archive chunk only; no final W0 archive claim, W1/W2/W3/W4/W5 "
    "evidence, mission evaluation, hardware validation, or sim-to-real transfer "
    "is claimed at chunk level."
)
TIMING_FIELDS = (
    "planning_read_s",
    "selection_s",
    "simulation_s",
    "descriptor_build_s",
    "write_s",
    "total_s",
)


@dataclass(frozen=True)
class W0ChunkConfig:
    run_id: int = 13
    planning_run_id: int = 12
    result_root: Path | None = None
    layout_branch_id: str = "single_fan_branch"
    chunk_index: int = 0
    chunk_count: int = 100
    chunk_size: int = 2500
    latency_case: str = "nominal"
    dt_s: float = 0.02
    horizon_s: float = 0.60
    storage_format: str = "auto"
    compression_level: int = 1
    resume: bool = True
    overwrite_chunk: bool = False
    repair_incomplete: bool = False
    random_seed: int = 20260525


@dataclass(frozen=True)
class W0ChunkOutputs:
    root: Path
    partition_path: Path
    manifest_json: Path
    log_path: Path

    def as_dict(self) -> dict[str, Path]:
        return {
            "root": self.root,
            "partition_path": self.partition_path,
            "manifest_json": self.manifest_json,
            "log_path": self.log_path,
        }


def active_result_root(config: W0ChunkConfig) -> Path:
    return DEFAULT_RESULT_ROOT if config.result_root is None else Path(config.result_root)


def archive_run_root(config: W0ChunkConfig) -> Path:
    return active_result_root(config) / f"{int(config.run_id):03d}"


def planning_run_root(config: W0ChunkConfig) -> Path:
    return active_result_root(config).parent / PLANNING_CAMPAIGN / f"{int(config.planning_run_id):03d}"


def output_paths(config: W0ChunkConfig, storage_format: str | None = None) -> W0ChunkOutputs:
    effective_format = (
        resolve_storage_format(config.storage_format)
        if storage_format is None
        else resolve_storage_format(storage_format)
    )
    root = archive_run_root(config)
    branch_dir = f"layout_branch_id={config.layout_branch_id}"
    partition = (
        root
        / "tables"
        / "trial_outcomes"
        / branch_dir
        / f"chunk-{int(config.chunk_index):05d}.{table_extension(effective_format)}"
    )
    manifest = (
        root
        / "chunk_manifests"
        / branch_dir
        / f"chunk-{int(config.chunk_index):05d}.json"
    )
    log_path = (
        root
        / "chunk_logs"
        / branch_dir
        / f"chunk-{int(config.chunk_index):05d}.log"
    )
    return W0ChunkOutputs(root=root, partition_path=partition, manifest_json=manifest, log_path=log_path)


def _path_text(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


# =============================================================================
# 2) Chunk Completion and Planning Reads
# =============================================================================
def validate_config(config: W0ChunkConfig) -> None:
    if int(config.run_id) <= 0 or int(config.planning_run_id) <= 0:
        raise ValueError("run_id and planning_run_id must be positive.")
    if str(config.layout_branch_id) not in W0_BRANCH_IDS:
        raise ValueError("layout_branch_id must be single_fan_branch or four_fan_branch.")
    if int(config.chunk_index) < 0:
        raise ValueError("chunk_index must be nonnegative.")
    if int(config.chunk_count) <= 0:
        raise ValueError("chunk_count must be positive.")
    if int(config.chunk_index) >= int(config.chunk_count):
        raise ValueError("chunk_index must be less than chunk_count.")
    if int(config.chunk_size) <= 0:
        raise ValueError("chunk_size must be positive.")
    if not np.isfinite(float(config.dt_s)) or float(config.dt_s) <= 0.0:
        raise ValueError("dt_s must be finite and positive.")
    if not np.isfinite(float(config.horizon_s)) or float(config.horizon_s) <= 0.0:
        raise ValueError("horizon_s must be finite and positive.")
    if int(config.compression_level) < 0 or int(config.compression_level) > 9:
        raise ValueError("compression_level must be in [0, 9].")
    resolve_storage_format(config.storage_format)


def chunk_status(config: W0ChunkConfig) -> str:
    """Return complete, missing, or corrupt for one chunk output."""

    outputs = output_paths(config)
    if not outputs.manifest_json.exists() and not outputs.partition_path.exists():
        return "missing"
    if outputs.manifest_json.exists() != outputs.partition_path.exists():
        return "corrupt"
    try:
        return "complete" if chunk_is_complete(config) else "corrupt"
    except FileNotFoundError:
        return "missing"
    except (ValueError, KeyError, json.JSONDecodeError, OSError):
        return "corrupt"


def chunk_is_complete(config: W0ChunkConfig) -> bool:
    outputs = output_paths(config)
    if not outputs.manifest_json.exists() or not outputs.partition_path.exists():
        raise FileNotFoundError("missing chunk manifest or partition")
    manifest = json.loads(outputs.manifest_json.read_text(encoding="ascii"))
    expected = {
        "run_id": int(config.run_id),
        "planning_run_id": int(config.planning_run_id),
        "layout_branch_id": str(config.layout_branch_id),
        "chunk_index": int(config.chunk_index),
        "chunk_count": int(config.chunk_count),
        "chunk_size": int(config.chunk_size),
        "storage_format": resolve_storage_format(config.storage_format),
        "latency_case": str(config.latency_case),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"chunk manifest mismatch for {key}")
    for key, value in (("dt_s", config.dt_s), ("horizon_s", config.horizon_s)):
        if not math.isclose(float(manifest.get(key)), float(value), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"chunk manifest mismatch for {key}")
    if manifest.get("status") != "complete":
        raise ValueError("chunk manifest is not complete")
    if int(manifest["row_count"]) != partition_row_count(outputs.partition_path):
        raise ValueError("chunk row count mismatch")
    if str(manifest["checksum_sha256"]) != file_sha256(outputs.partition_path):
        raise ValueError("chunk checksum mismatch")
    for field in TIMING_FIELDS:
        value = float(manifest[field])
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"chunk timing field is not finite: {field}")
    return True


def remove_chunk_outputs(config: W0ChunkConfig) -> None:
    outputs = output_paths(config)
    for path in (
        outputs.partition_path,
        outputs.partition_path.with_name(f"{outputs.partition_path.name}.tmp"),
        outputs.manifest_json,
        outputs.manifest_json.with_name(f"{outputs.manifest_json.name}.tmp"),
        outputs.log_path,
        outputs.log_path.with_name(f"{outputs.log_path.name}.tmp"),
    ):
        if path.exists():
            path.unlink()


def _load_branch_planning(
    config: W0ChunkConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    start = time.perf_counter()
    root = planning_run_root(config)
    candidate_path = _planning_chunk_path(root, "candidate_index", config)
    start_path = _planning_chunk_path(root, "start_states", config)
    if not candidate_path.exists():
        raise FileNotFoundError(
            "missing candidate_index branch/chunk partition before replay: "
            f"{candidate_path}"
        )
    if not start_path.exists():
        raise FileNotFoundError(
            "missing start_states branch/chunk partition before replay: "
            f"{start_path}"
        )
    candidates = read_table_partition(
        candidate_path,
        storage_format=resolve_storage_format(config.storage_format),
    )
    starts = read_table_partition(
        start_path,
        storage_format=resolve_storage_format(config.storage_format),
    )
    _validate_planning_chunk(starts, candidates, config)
    return starts, candidates, time.perf_counter() - start


def _planning_chunk_path(
    root: Path,
    table_name: str,
    config: W0ChunkConfig,
) -> Path:
    return (
        root
        / "tables"
        / table_name
        / f"layout_branch_id={config.layout_branch_id}"
        / f"archive_chunk_index={int(config.chunk_index):05d}"
        / f"part-00000.{table_extension(config.storage_format)}"
    )


def _validate_planning_chunk(
    starts: pd.DataFrame,
    candidates: pd.DataFrame,
    config: W0ChunkConfig,
) -> None:
    if int(len(candidates)) != int(config.chunk_size):
        raise ValueError(
            "candidate_index chunk row count mismatch: "
            f"actual={len(candidates)}, expected={int(config.chunk_size)}"
        )
    if int(len(starts)) != int(len(candidates)):
        raise ValueError(
            "start_states chunk row count must match candidate_index row count."
        )
    candidate_ids = candidates["sample_id"].astype(str).to_list()
    start_ids = starts["sample_id"].astype(str).to_list()
    if candidate_ids != start_ids:
        raise ValueError("candidate_index and start_states sample_id order mismatch.")
    for name, frame in (("candidate_index", candidates), ("start_states", starts)):
        _validate_chunk_columns(name, frame, config)


def _validate_chunk_columns(
    table_name: str,
    frame: pd.DataFrame,
    config: W0ChunkConfig,
) -> None:
    required = {
        "archive_chunk_index",
        "archive_chunk_count",
        "chunk_local_index",
        "archive_chunk_size",
        "archive_branch_trial_index",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{table_name} missing archive chunk columns: {missing}")
    if set(frame["layout_branch_id"].astype(str)) != {str(config.layout_branch_id)}:
        raise ValueError(f"{table_name} layout_branch_id mismatch.")
    if set(frame["archive_chunk_index"].astype(int)) != {int(config.chunk_index)}:
        raise ValueError(f"{table_name} archive_chunk_index mismatch.")
    if set(frame["archive_chunk_count"].astype(int)) != {int(config.chunk_count)}:
        raise ValueError(f"{table_name} archive_chunk_count mismatch.")
    if set(frame["archive_chunk_size"].astype(int)) != {int(config.chunk_size)}:
        raise ValueError(f"{table_name} archive_chunk_size mismatch.")
    expected_local = list(range(int(config.chunk_size)))
    actual_local = frame["chunk_local_index"].astype(int).to_list()
    if actual_local != expected_local:
        raise ValueError(f"{table_name} chunk_local_index sequence mismatch.")


def _select_chunk_candidates(
    candidates: pd.DataFrame,
    config: W0ChunkConfig,
) -> tuple[list[dict[str, object]], float]:
    start = time.perf_counter()
    del config
    return candidates.to_dict(orient="records"), time.perf_counter() - start


def _select_start_rows(
    starts: pd.DataFrame,
    selected: list[dict[str, object]],
) -> pd.DataFrame:
    selected_ids = [str(row["sample_id"]) for row in selected]
    start_ids = starts["sample_id"].astype(str).to_list()
    if selected_ids != start_ids:
        raise ValueError("selected candidates and start rows have mismatched sample_id order.")
    return starts.copy()


# =============================================================================
# 3) Chunk Execution and Manifest Writing
# =============================================================================
def _run_descriptors(
    starts: pd.DataFrame,
    selected: list[dict[str, object]],
    config: W0ChunkConfig,
) -> tuple[pd.DataFrame, float, float]:
    start = time.perf_counter()
    branch_offset = W0_BRANCH_IDS.index(str(config.layout_branch_id)) * int(config.chunk_count) * int(config.chunk_size)
    seed_offset = branch_offset + int(config.chunk_index) * int(config.chunk_size)
    replay_config = replace(config, random_seed=int(config.random_seed) + seed_offset)
    descriptors = _run_pilot_replays(starts, selected, replay_config)
    simulation_s = time.perf_counter() - start
    return descriptors, simulation_s, 0.0


def _write_chunk(
    descriptors: pd.DataFrame,
    config: W0ChunkConfig,
) -> tuple[TablePartition, float]:
    start = time.perf_counter()
    outputs = output_paths(config)
    outputs.partition_path.parent.mkdir(parents=True, exist_ok=True)
    partition = write_table_partition(
        descriptors,
        outputs.partition_path,
        storage_format=config.storage_format,
        compression_level=int(config.compression_level),
    )
    return partition, time.perf_counter() - start


def _write_manifest(
    *,
    config: W0ChunkConfig,
    partition: TablePartition,
    timing: dict[str, float],
) -> None:
    outputs = output_paths(config)
    outputs.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    outputs.log_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "complete",
        "run_id": int(config.run_id),
        "planning_run_id": int(config.planning_run_id),
        "layout_branch_id": str(config.layout_branch_id),
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
        "w0_chunk_replay_performed": True,
        "final_w0_archive_claim": False,
        "w1_w2_w3_w4_w5_performed": False,
        "hardware_or_mission_claim": False,
        "sim_to_real_transfer_claim": False,
        "no_overclaiming_statement": NO_CLAIM_TEXT,
        **{field: float(timing[field]) for field in TIMING_FIELDS},
    }
    tmp_path = outputs.manifest_json.with_name(f"{outputs.manifest_json.name}.tmp")
    tmp_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n",
        encoding="ascii",
    )
    tmp_path.replace(outputs.manifest_json)
    outputs.log_path.write_text(
        f"status=complete row_count={int(partition.row_count)} total_s={float(timing['total_s']):.6f}\n",
        encoding="ascii",
    )


# =============================================================================
# 4) Public Runner and CLI
# =============================================================================
def run_w0_dense_archive_chunk(
    *,
    run_id: int = 13,
    planning_run_id: int = 12,
    result_root: Path | None = None,
    layout_branch_id: str = "single_fan_branch",
    chunk_index: int = 0,
    chunk_count: int = 100,
    chunk_size: int = 2500,
    latency_case: str = "nominal",
    dt_s: float = 0.02,
    horizon_s: float = 0.60,
    storage_format: str = "auto",
    compression_level: int = 1,
    resume: bool = True,
    overwrite_chunk: bool = False,
    repair_incomplete: bool = False,
    random_seed: int = 20260525,
) -> dict[str, Path]:
    """Replay one deterministic W0 branch/chunk and write one partition."""

    config = W0ChunkConfig(
        run_id=int(run_id),
        planning_run_id=int(planning_run_id),
        result_root=result_root,
        layout_branch_id=str(layout_branch_id),
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
    if bool(config.overwrite_chunk):
        remove_chunk_outputs(config)
    else:
        status = chunk_status(config)
        if bool(config.resume) and status == "complete":
            return outputs.as_dict()
        if status == "corrupt" and bool(config.repair_incomplete):
            remove_chunk_outputs(config)
        elif status == "corrupt":
            raise RuntimeError(
                "chunk output exists but is incomplete/corrupt; use "
                "--repair-incomplete to remove only this chunk's files, or "
                "--overwrite-chunk to rerun a complete chunk."
            )
        elif status == "complete":
            raise RuntimeError(
                "chunk output already complete; use --resume to skip it or "
                "--overwrite-chunk to rerun it."
            )

    total_start = time.perf_counter()
    starts, candidates, planning_read_s = _load_branch_planning(config)
    selected, selection_s = _select_chunk_candidates(candidates, config)
    selected_starts = _select_start_rows(starts, selected)
    descriptors, simulation_s, descriptor_build_s = _run_descriptors(
        selected_starts,
        selected,
        config,
    )
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
    _write_manifest(config=config, partition=partition, timing=timing)
    return outputs.as_dict()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int, default=13)
    parser.add_argument("--planning-run-id", type=int, default=12)
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--layout-branch-id", default="single_fan_branch")
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--chunk-count", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=2500)
    parser.add_argument("--latency-case", default="nominal")
    parser.add_argument("--dt-s", type=float, default=0.02)
    parser.add_argument("--horizon-s", type=float, default=0.60)
    parser.add_argument("--storage-format", default="auto")
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite-chunk", action="store_true")
    parser.add_argument("--repair-incomplete", action="store_true")
    parser.add_argument("--random-seed", type=int, default=20260525)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    paths = run_w0_dense_archive_chunk(
        run_id=args.run_id,
        planning_run_id=args.planning_run_id,
        result_root=args.result_root,
        layout_branch_id=args.layout_branch_id,
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
    print(f"w0_dense_archive_chunk_outputs={_path_text(paths['root'])}")
    return 0


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


def _target_text(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "none"
    if not np.isfinite(numeric):
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
