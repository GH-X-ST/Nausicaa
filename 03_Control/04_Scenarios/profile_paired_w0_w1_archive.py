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
from dense_archive_runtime import (  # noqa: E402
    GPU_ACCELERATION_ASSESSMENT,
    rows_per_second_by_worker_count,
    runtime_manifest_fields,
    runtime_estimates_by_worker_count,
    worker_count_decision,
)
from dense_archive_schema import BRANCH_DECISION_SCOPE  # noqa: E402
from dense_archive_table_io import (  # noqa: E402
    list_table_partitions,
    read_table_partition,
    resolve_storage_format,
    table_extension,
    write_table_partition,
)
from run_dense_archive_pilot_sweep import _run_pilot_replays  # noqa: E402
from run_paired_w0_w1_archive_chunked import RECOMMENDED_PAIRED_PROOF_COMMAND  # noqa: E402
from run_paired_w0_w1_partitioned_planning import (  # noqa: E402
    PAIRED_ENVIRONMENT_MODES,
    SIMULATION_STAGE,
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
DEFAULT_RESULT_ROOT = CONTROL_DIR / "05_Results" / "12_paired_w0_w1_archive"
PLANNING_CAMPAIGN = "10_dense_archive_planning"


@dataclass(frozen=True)
class PairedProfileConfig:
    planning_run_id: int = 13
    result_root: Path | None = None
    profile_root: Path | None = None
    sample_trials: int = 2000
    storage_format: str = "auto"
    latency_case: str = "nominal"
    dt_s: float = 0.02
    horizon_s: float = 0.60
    workers: str | int = "auto"
    memory_safety_margin_gb: float = 8.0


@dataclass(frozen=True)
class PairedProfileOutputs:
    root: Path
    profile_json: Path
    profile_csv: Path

    def as_dict(self) -> dict[str, Path]:
        return {
            "root": self.root,
            "profile_json": self.profile_json,
            "profile_csv": self.profile_csv,
        }


def _active_result_root(config: PairedProfileConfig) -> Path:
    return DEFAULT_RESULT_ROOT if config.result_root is None else Path(config.result_root)


def _profile_root(config: PairedProfileConfig) -> Path:
    if config.profile_root is not None:
        return Path(config.profile_root)
    return (
        _active_result_root(config)
        / "profiles"
        / f"paired_planning_s{int(config.planning_run_id):03d}"
    )


def _planning_root(config: PairedProfileConfig) -> Path:
    return _active_result_root(config).parent / PLANNING_CAMPAIGN / f"{int(config.planning_run_id):03d}"


def _outputs(config: PairedProfileConfig) -> PairedProfileOutputs:
    root = _profile_root(config)
    suffix = f"s{int(config.planning_run_id):03d}"
    return PairedProfileOutputs(
        root=root,
        profile_json=root / f"paired_w0_w1_profile_{suffix}.json",
        profile_csv=root / f"paired_w0_w1_profile_{suffix}.csv",
    )


# =============================================================================
# 2) Profile Sampling Helpers
# =============================================================================
def _load_profile_inputs(config: PairedProfileConfig) -> tuple[pd.DataFrame, list[dict[str, object]], float]:
    start = time.perf_counter()
    root = _planning_root(config)
    selected: list[pd.DataFrame] = []
    starts: list[pd.DataFrame] = []
    per_environment = max(1, int(config.sample_trials) // len(PAIRED_ENVIRONMENT_MODES))
    candidate_paths = list_table_partitions(root, "candidate_index")
    start_paths = list_table_partitions(root, "start_states")
    for environment_mode in PAIRED_ENVIRONMENT_MODES:
        branch_candidates = pd.concat(
            [
                read_table_partition(path)
                for path in candidate_paths
                if f"test_environment_mode={environment_mode}" in path.as_posix()
            ],
            ignore_index=True,
        )
        branch_selected = branch_candidates.head(per_environment).copy()
        selected.append(branch_selected)
        sample_ids = set(branch_selected["sample_id"].astype(str))
        branch_starts = pd.concat(
            [
                read_table_partition(path)
                for path in start_paths
                if f"test_environment_mode={environment_mode}" in path.as_posix()
            ],
            ignore_index=True,
        )
        starts.append(
            branch_starts[branch_starts["sample_id"].astype(str).isin(sample_ids)].copy()
        )
    return (
        pd.concat(starts, ignore_index=True),
        pd.concat(selected, ignore_index=True).to_dict(orient="records"),
        time.perf_counter() - start,
    )


def _measure_write_speed(
    descriptors: pd.DataFrame,
    outputs: PairedProfileOutputs,
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
# 3) Profile Writers
# =============================================================================
def _write_outputs(outputs: PairedProfileOutputs, payload: dict[str, object]) -> None:
    outputs.root.mkdir(parents=True, exist_ok=True)
    outputs.profile_json.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="ascii",
    )
    pd.DataFrame(
        [
            {
                "field": key,
                "value": json.dumps(value, separators=(",", ":"))
                if isinstance(value, (dict, list))
                else value,
            }
            for key, value in payload.items()
        ]
    ).to_csv(outputs.profile_csv, index=False)


# =============================================================================
# 4) Public Runner and CLI
# =============================================================================
def profile_paired_w0_w1_archive(
    *,
    planning_run_id: int = 13,
    result_root: Path | None = None,
    profile_root: Path | None = None,
    sample_trials: int = 2000,
    storage_format: str = "auto",
    latency_case: str = "nominal",
    dt_s: float = 0.02,
    horizon_s: float = 0.60,
    workers: str | int = "auto",
    memory_safety_margin_gb: float = 8.0,
) -> dict[str, Path]:
    config = PairedProfileConfig(
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
    selected = selected[: int(config.sample_trials)]
    selection_s = 0.0
    simulation_start = time.perf_counter()
    descriptors = _run_pilot_replays(starts, selected, _ReplayConfig(config))
    simulation_s = time.perf_counter() - simulation_start
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

    rows_per_second_single = (
        0.0 if simulation_s <= 0.0 else float(len(descriptors)) / float(simulation_s)
    )
    rps_by_worker = rows_per_second_by_worker_count(rows_per_second_single)
    runtime_by_worker = runtime_estimates_by_worker_count(
        rps_by_worker,
        total_trials=max(1, int(config.sample_trials)),
    )
    estimated_worker_memory_gb = max(2.0, float(peak_bytes) / float(1024**3))
    decision = worker_count_decision(
        config.workers,
        memory_safety_margin_gb=float(config.memory_safety_margin_gb),
        estimated_worker_memory_gb=estimated_worker_memory_gb,
        max_workers=8,
    )
    payload = {
        **runtime_manifest_fields(
            simulation_stage=SIMULATION_STAGE,
            environment_mode="multiple",
            branch_decision_scope=BRANCH_DECISION_SCOPE,
            worker_decision=decision,
            profiling_rows_per_second=rps_by_worker,
        ),
        "simulation_stage": SIMULATION_STAGE,
        "planning_run_id": int(config.planning_run_id),
        "sample_trials_requested": int(config.sample_trials),
        "sample_trials_profiled": int(len(descriptors)),
        "planning_read_s": float(planning_read_s),
        "selection_s": float(selection_s),
        "simulation_s": float(simulation_s),
        "descriptor_build_s": 0.0,
        "write_s": float(write_s),
        "aggregate_s": float(aggregate_s),
        "rows_per_second_single_worker": float(rows_per_second_single),
        "rows_per_second_by_worker_count": rps_by_worker,
        "estimated_runtime_s_by_workers": runtime_by_worker,
        "estimated_runtime_s_by_worker_count": runtime_by_worker,
        "selected_worker_count": int(decision.selected_worker_count),
        "os_cpu_count": decision.os_cpu_count,
        "memory_total_gb": decision.memory_total_gb,
        "memory_safety_margin_gb": decision.memory_safety_margin_gb,
        "estimated_worker_memory_gb": estimated_worker_memory_gb,
        "worker_fallback_reason": decision.fallback_reason,
        "peak_memory_mb": float(peak_bytes) / float(1024**2),
        "storage_format": effective_format,
        "write_test_byte_count": int(write_byte_count),
        "write_test_row_count": int(write_row_count),
        "envelope_cell_count": int(len(envelope)),
        "gpu_acceleration_assessment": GPU_ACCELERATION_ASSESSMENT,
        "recommended_paired_proof_command": RECOMMENDED_PAIRED_PROOF_COMMAND,
    }
    _write_outputs(outputs, payload)
    return outputs.as_dict()


@dataclass(frozen=True)
class _ReplayConfig:
    source: PairedProfileConfig

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
        return 20260526


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planning-run-id", type=int, default=13)
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--profile-root", type=Path, default=None)
    parser.add_argument("--sample-trials", type=int, default=2000)
    parser.add_argument("--storage-format", default="auto")
    parser.add_argument("--latency-case", default="nominal")
    parser.add_argument("--dt-s", type=float, default=0.02)
    parser.add_argument("--horizon-s", type=float, default=0.60)
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--memory-safety-margin-gb", type=float, default=8.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    workers: str | int = int(args.workers) if str(args.workers).isdigit() else str(args.workers)
    paths = profile_paired_w0_w1_archive(
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
    print(f"paired_w0_w1_profile_outputs={paths['root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
