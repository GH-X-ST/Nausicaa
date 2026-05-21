from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import tracemalloc
from dataclasses import dataclass
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

from dense_archive_envelope_maps import build_envelope_map  # noqa: E402
from dense_archive_table_io import (  # noqa: E402
    list_table_partitions,
    read_table_partition,
    resolve_storage_format,
    table_extension,
    write_table_partition,
)
from run_dense_archive_pilot_sweep import _run_pilot_replays  # noqa: E402
from run_w0_dense_archive_chunked import (  # noqa: E402
    GPU_ACCELERATION_ASSESSMENT,
    PRODUCTION_COMMAND,
    worker_count_decision,
)


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Paths and Data Containers
# 2) Profile Sampling Helpers
# 3) Profile Writers
# 4) Public Runner and CLI
# =============================================================================


# =============================================================================
# 1) Paths and Data Containers
# =============================================================================
DEFAULT_RESULT_ROOT = CONTROL_DIR / "05_Results" / "11_w0_dense_archive"
PLANNING_CAMPAIGN = "10_dense_archive_planning"
W0_BRANCH_IDS = ("single_fan_branch", "four_fan_branch")


@dataclass(frozen=True)
class W0ProfileConfig:
    planning_run_id: int = 12
    result_root: Path | None = None
    profile_root: Path | None = None
    sample_trials: int = 2000
    storage_format: str = "auto"
    latency_case: str = "nominal"
    dt_s: float = 0.02
    horizon_s: float = 0.60
    workers: str | int = 1
    memory_safety_margin_gb: float = 8.0


@dataclass(frozen=True)
class W0ProfileOutputs:
    root: Path
    profile_json: Path
    profile_csv: Path

    def as_dict(self) -> dict[str, Path]:
        return {
            "root": self.root,
            "profile_json": self.profile_json,
            "profile_csv": self.profile_csv,
        }


def _active_result_root(config: W0ProfileConfig) -> Path:
    return DEFAULT_RESULT_ROOT if config.result_root is None else Path(config.result_root)


def _profile_root(config: W0ProfileConfig) -> Path:
    if config.profile_root is not None:
        return Path(config.profile_root)
    return (
        _active_result_root(config)
        / "profiles"
        / f"planning_s{int(config.planning_run_id):03d}"
    )


def _planning_root(config: W0ProfileConfig) -> Path:
    return _active_result_root(config).parent / PLANNING_CAMPAIGN / f"{int(config.planning_run_id):03d}"


def _outputs(config: W0ProfileConfig) -> W0ProfileOutputs:
    root = _profile_root(config)
    suffix = f"s{int(config.planning_run_id):03d}"
    return W0ProfileOutputs(
        root=root,
        profile_json=root / f"w0_profile_{suffix}.json",
        profile_csv=root / f"w0_profile_{suffix}.csv",
    )


# =============================================================================
# 2) Profile Sampling Helpers
# =============================================================================
def _load_profile_inputs(config: W0ProfileConfig) -> tuple[pd.DataFrame, list[dict[str, object]], float]:
    start = time.perf_counter()
    root = _planning_root(config)
    selected: list[pd.DataFrame] = []
    starts: list[pd.DataFrame] = []
    per_branch = max(1, int(config.sample_trials) // len(W0_BRANCH_IDS))
    candidate_paths = list_table_partitions(root, "candidate_index")
    start_paths = list_table_partitions(root, "start_states")
    for branch_id in W0_BRANCH_IDS:
        branch_candidates = pd.concat(
            [
                read_table_partition(path)
                for path in candidate_paths
                if f"layout_branch_id={branch_id}" in path.as_posix()
            ],
            ignore_index=True,
        )
        branch_selected = branch_candidates.head(per_branch).copy()
        selected.append(branch_selected)
        sample_ids = set(branch_selected["sample_id"].astype(str))
        branch_starts = pd.concat(
            [
                read_table_partition(path)
                for path in start_paths
                if f"layout_branch_id={branch_id}" in path.as_posix()
            ],
            ignore_index=True,
        )
        starts.append(branch_starts[branch_starts["sample_id"].astype(str).isin(sample_ids)].copy())
    return (
        pd.concat(starts, ignore_index=True),
        pd.concat(selected, ignore_index=True).to_dict(orient="records"),
        time.perf_counter() - start,
    )


def _profile_rows_per_second(row_count: int, simulation_s: float) -> float:
    if float(simulation_s) <= 0.0:
        return 0.0
    return float(row_count) / float(simulation_s)


def _scaled_rows_per_second(single_worker_rps: float) -> dict[str, float]:
    return {
        str(workers): float(single_worker_rps) * float(workers)
        for workers in (1, 4, 6, 8)
    }


def _runtime_estimates(rows_per_second: dict[str, float], total_trials: int = 500000) -> dict[str, float]:
    return {
        key: (
            float("inf")
            if float(value) <= 0.0
            else float(total_trials) / float(value)
        )
        for key, value in rows_per_second.items()
    }


# =============================================================================
# 3) Profile Writers
# =============================================================================
def _write_outputs(outputs: W0ProfileOutputs, payload: dict[str, object]) -> None:
    outputs.root.mkdir(parents=True, exist_ok=True)
    outputs.profile_json.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="ascii",
    )
    flat_rows = [
        {
            "field": key,
            "value": json.dumps(value, separators=(",", ":"))
            if isinstance(value, (dict, list))
            else value,
        }
        for key, value in payload.items()
    ]
    pd.DataFrame(flat_rows).to_csv(outputs.profile_csv, index=False)


def _measure_write_speed(
    descriptors: pd.DataFrame,
    outputs: W0ProfileOutputs,
    *,
    storage_format: str,
) -> tuple[float, int, int]:
    write_root = outputs.root / "_profile_write_tmp"
    write_path = write_root / f"write_speed_sample.{table_extension(storage_format)}"
    start = time.perf_counter()
    partition = write_table_partition(
        descriptors,
        write_path,
        storage_format=storage_format,
        compression_level=1,
    )
    write_s = time.perf_counter() - start
    byte_count = int(partition.byte_count)
    row_count = int(partition.row_count)
    if write_root.exists():
        shutil.rmtree(write_root)
    return write_s, byte_count, row_count


# =============================================================================
# 4) Public Runner and CLI
# =============================================================================
def profile_w0_dense_archive(
    *,
    planning_run_id: int = 12,
    result_root: Path | None = None,
    profile_root: Path | None = None,
    sample_trials: int = 2000,
    storage_format: str = "auto",
    latency_case: str = "nominal",
    dt_s: float = 0.02,
    horizon_s: float = 0.60,
    workers: str | int = 1,
    memory_safety_margin_gb: float = 8.0,
) -> dict[str, Path]:
    """Profile a small W0 sample outside the production run directory."""

    config = W0ProfileConfig(
        planning_run_id=int(planning_run_id),
        result_root=result_root,
        profile_root=profile_root,
        sample_trials=int(sample_trials),
        storage_format=str(storage_format),
        latency_case=str(latency_case),
        dt_s=float(dt_s),
        horizon_s=float(horizon_s),
        workers=workers,
        memory_safety_margin_gb=float(memory_safety_margin_gb),
    )
    effective_format = resolve_storage_format(config.storage_format)
    outputs = _outputs(config)
    tracemalloc.start()
    starts, selected, planning_read_s = _load_profile_inputs(config)
    selection_start = time.perf_counter()
    selected = selected[: int(config.sample_trials)]
    selection_s = time.perf_counter() - selection_start
    simulation_start = time.perf_counter()
    replay_config = _ReplayConfig(config)
    descriptors = _run_pilot_replays(starts, selected, replay_config)
    simulation_s = time.perf_counter() - simulation_start
    descriptor_build_s = 0.0
    write_s, write_byte_count, write_row_count = _measure_write_speed(
        descriptors,
        outputs,
        storage_format=effective_format,
    )
    aggregate_start = time.perf_counter()
    envelope = build_envelope_map(descriptors)
    aggregate_s = time.perf_counter() - aggregate_start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    rows_per_second_single = _profile_rows_per_second(len(descriptors), simulation_s)
    rps_by_worker = _scaled_rows_per_second(rows_per_second_single)
    runtime_by_worker = _runtime_estimates(rps_by_worker)
    estimated_worker_memory_gb = max(2.0, float(peak_bytes) / float(1024**3))
    decision = worker_count_decision(
        config.workers,
        memory_safety_margin_gb=float(config.memory_safety_margin_gb),
        estimated_worker_memory_gb=estimated_worker_memory_gb,
        max_workers=8,
    )
    payload = {
        "planning_run_id": int(config.planning_run_id),
        "sample_trials_requested": int(config.sample_trials),
        "sample_trials_profiled": int(len(descriptors)),
        "planning_read_s": float(planning_read_s),
        "selection_s": float(selection_s),
        "simulation_s": float(simulation_s),
        "descriptor_build_s": float(descriptor_build_s),
        "write_s": float(write_s),
        "aggregate_s": float(aggregate_s),
        "rows_per_second_single_worker": float(rows_per_second_single),
        "rows_per_second_by_worker_count": rps_by_worker,
        "estimated_500k_runtime_s_by_workers": runtime_by_worker,
        "recommended_worker_count": int(decision.selected_worker_count),
        "selected_worker_count": int(decision.selected_worker_count),
        "worker_fallback_reason": decision.fallback_reason,
        "os_cpu_count": decision.os_cpu_count,
        "memory_total_gb": decision.memory_total_gb,
        "memory_safety_margin_gb": decision.memory_safety_margin_gb,
        "estimated_worker_memory_gb": estimated_worker_memory_gb,
        "peak_memory_mb": float(peak_bytes) / float(1024**2),
        "storage_format": effective_format,
        "write_test_byte_count": int(write_byte_count),
        "write_test_row_count": int(write_row_count),
        "envelope_cell_count": int(len(envelope)),
        "gpu_acceleration_assessment": GPU_ACCELERATION_ASSESSMENT,
        "recommended_production_command": PRODUCTION_COMMAND,
    }
    _write_outputs(outputs, payload)
    return outputs.as_dict()


@dataclass(frozen=True)
class _ReplayConfig:
    source: W0ProfileConfig

    @property
    def latency_case(self) -> str:
        return str(self.source.latency_case)

    @property
    def dt_s(self) -> float:
        return float(self.source.dt_s)

    @property
    def horizon_s(self) -> float:
        return float(self.source.horizon_s)

    @property
    def random_seed(self) -> int:
        return 20260525


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planning-run-id", type=int, default=12)
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--profile-root", type=Path, default=None)
    parser.add_argument("--sample-trials", type=int, default=2000)
    parser.add_argument("--storage-format", default="auto")
    parser.add_argument("--latency-case", default="nominal")
    parser.add_argument("--dt-s", type=float, default=0.02)
    parser.add_argument("--horizon-s", type=float, default=0.60)
    parser.add_argument("--workers", default="1")
    parser.add_argument("--memory-safety-margin-gb", type=float, default=8.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    workers: str | int = int(args.workers) if str(args.workers).isdigit() else str(args.workers)
    paths = profile_w0_dense_archive(
        planning_run_id=args.planning_run_id,
        result_root=args.result_root,
        profile_root=args.profile_root,
        sample_trials=args.sample_trials,
        storage_format=args.storage_format,
        latency_case=args.latency_case,
        dt_s=args.dt_s,
        horizon_s=args.horizon_s,
        workers=workers,
        memory_safety_margin_gb=args.memory_safety_margin_gb,
    )
    print(f"w0_profile_outputs={paths['root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
