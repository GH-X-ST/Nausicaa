from __future__ import annotations

import ctypes
import os
from dataclasses import asdict, dataclass


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Runtime contract constants
# 2) Worker count policy
# 3) Dense-run metadata helpers
# =============================================================================


# =============================================================================
# 1) Runtime Contract Constants
# =============================================================================
RUNTIME_CORE_VERSION = "contextual_runtime_v1"
STORAGE_CONTRACT_VERSION = "contextual_storage_contract_v1"
MAX_GENERATED_FILE_SIZE_MB = 100.0
PREFERRED_GENERATED_FILE_SIZE_MB = 75.0
DEFAULT_MEMORY_SAFETY_MARGIN_GB = 8.0
DEFAULT_ESTIMATED_WORKER_MEMORY_GB = 2.0
LOCAL_LAPTOP_WORKER_COUNT = 8
PROFILE_WORKER_COUNTS = (1, 4, 6, 8)
RUNTIME_STORAGE_CONTRACT = (
    "chunked, resumable, compressed, worker-enabled, checksum-manifested, "
    "and no generated file above 100 MB"
)


# =============================================================================
# 2) Worker Count Policy
# =============================================================================
@dataclass(frozen=True)
class WorkerCountDecision:
    requested: str | int
    selected_worker_count: int
    max_workers: int | None
    os_cpu_count: int | None
    memory_total_gb: float | None
    memory_safety_margin_gb: float
    estimated_worker_memory_gb: float | None
    fallback_reason: str

    def as_manifest_fields(self) -> dict[str, object]:
        return asdict(self)


def resolve_worker_count(
    requested: str | int,
    *,
    logical_cpu_count: int | None = None,
    memory_total_gb: float | None = None,
    memory_safety_margin_gb: float = DEFAULT_MEMORY_SAFETY_MARGIN_GB,
    estimated_worker_memory_gb: float | None = DEFAULT_ESTIMATED_WORKER_MEMORY_GB,
    max_workers: int | None = LOCAL_LAPTOP_WORKER_COUNT,
) -> int:
    return worker_count_decision(
        requested,
        logical_cpu_count=logical_cpu_count,
        memory_total_gb=memory_total_gb,
        memory_safety_margin_gb=memory_safety_margin_gb,
        estimated_worker_memory_gb=estimated_worker_memory_gb,
        max_workers=max_workers,
    ).selected_worker_count


def worker_count_decision(
    requested: str | int,
    *,
    logical_cpu_count: int | None = None,
    memory_total_gb: float | None = None,
    memory_safety_margin_gb: float = DEFAULT_MEMORY_SAFETY_MARGIN_GB,
    estimated_worker_memory_gb: float | None = DEFAULT_ESTIMATED_WORKER_MEMORY_GB,
    max_workers: int | None = LOCAL_LAPTOP_WORKER_COUNT,
) -> WorkerCountDecision:
    cpu_count = os.cpu_count() if logical_cpu_count is None else logical_cpu_count
    memory_gb = _memory_total_gb() if memory_total_gb is None else memory_total_gb
    fallback_reasons: list[str] = []

    if isinstance(requested, str) and requested.strip().lower() == "auto":
        if (cpu_count is None or int(cpu_count) >= 12) and (
            memory_gb is None or float(memory_gb) >= 31.0
        ):
            candidate = LOCAL_LAPTOP_WORKER_COUNT
        else:
            candidate = 6
            fallback_reasons.append("auto_reduced_by_machine_profile")
    else:
        try:
            candidate = int(requested)
        except (TypeError, ValueError) as exc:
            raise ValueError("workers must be 'auto' or a positive integer.") from exc
        if candidate < 1:
            raise ValueError("workers must be at least 1.")

    if max_workers is not None and candidate > int(max_workers):
        candidate = int(max_workers)
        fallback_reasons.append(f"capped_by_max_workers_{int(max_workers)}")

    if (
        estimated_worker_memory_gb is not None
        and memory_gb is not None
        and float(estimated_worker_memory_gb) > 0.0
    ):
        original_candidate = candidate
        while (
            candidate > 1
            and candidate * float(estimated_worker_memory_gb)
            + float(memory_safety_margin_gb)
            > float(memory_gb)
        ):
            candidate -= 1
        if candidate < original_candidate:
            fallback_reasons.append("reduced_by_memory_guardrail")

    return WorkerCountDecision(
        requested=requested,
        selected_worker_count=max(1, int(candidate)),
        max_workers=None if max_workers is None else int(max_workers),
        os_cpu_count=None if cpu_count is None else int(cpu_count),
        memory_total_gb=None if memory_gb is None else float(memory_gb),
        memory_safety_margin_gb=float(memory_safety_margin_gb),
        estimated_worker_memory_gb=(
            None
            if estimated_worker_memory_gb is None
            else float(estimated_worker_memory_gb)
        ),
        fallback_reason="none" if not fallback_reasons else ";".join(fallback_reasons),
    )


def _memory_total_gb() -> float | None:
    if os.name != "nt":
        return None

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return float(status.ullTotalPhys) / float(1024**3)


# =============================================================================
# 3) Dense-Run Metadata Helpers
# =============================================================================
def rows_per_second_by_worker_count(single_worker_rows_per_second: float) -> dict[str, float]:
    return {
        str(worker_count): float(single_worker_rows_per_second) * float(worker_count)
        for worker_count in PROFILE_WORKER_COUNTS
    }


def runtime_estimates_by_worker_count(
    rows_per_second: dict[str, float],
    *,
    total_rows: int,
) -> dict[str, float]:
    estimates: dict[str, float] = {}
    for key, value in rows_per_second.items():
        estimates[key] = (
            float("inf")
            if float(value) <= 0.0
            else float(total_rows) / float(value)
        )
    return estimates


def dense_run_manifest_fields(
    *,
    run_stage: str,
    environment_context: str,
    worker_decision: WorkerCountDecision | None = None,
    profiling_rows_per_second: dict[str, float] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "runtime_core_version": RUNTIME_CORE_VERSION,
        "storage_contract_version": STORAGE_CONTRACT_VERSION,
        "run_stage": str(run_stage),
        "environment_context": str(environment_context),
        "runtime_storage_contract": RUNTIME_STORAGE_CONTRACT,
        "preferred_generated_file_size_mb": PREFERRED_GENERATED_FILE_SIZE_MB,
        "max_generated_file_size_mb": MAX_GENERATED_FILE_SIZE_MB,
    }
    if worker_decision is not None:
        payload.update(
            {
                "selected_worker_count": int(worker_decision.selected_worker_count),
                "max_workers": worker_decision.max_workers,
                "os_cpu_count": worker_decision.os_cpu_count,
                "memory_total_gb": worker_decision.memory_total_gb,
                "memory_safety_margin_gb": worker_decision.memory_safety_margin_gb,
                "estimated_worker_memory_gb": worker_decision.estimated_worker_memory_gb,
                "worker_fallback_reason": worker_decision.fallback_reason,
            }
        )
    if profiling_rows_per_second is not None:
        payload["rows_per_second_by_worker_count"] = dict(profiling_rows_per_second)
    return payload


def runtime_manifest_fields(
    *,
    simulation_stage: str,
    environment_mode: str,
    worker_decision: WorkerCountDecision | None = None,
    profiling_rows_per_second: dict[str, float] | None = None,
) -> dict[str, object]:
    """Backward-compatible wrapper for retained foundation tests."""

    return dense_run_manifest_fields(
        run_stage=simulation_stage,
        environment_context=environment_mode,
        worker_decision=worker_decision,
        profiling_rows_per_second=profiling_rows_per_second,
    )
