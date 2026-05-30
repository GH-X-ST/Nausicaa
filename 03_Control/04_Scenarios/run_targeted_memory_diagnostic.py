from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


def _bootstrap_import_paths() -> None:
    control_root = Path(__file__).resolve().parents[1]
    for rel in ("03_Primitives", "04_Scenarios"):
        path = control_root / rel
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


_bootstrap_import_paths()

from run_changed_case_validation import DEFAULT_OUTPUT_ROOT  # noqa: E402
from run_repeated_launch_learning_curve import (  # noqa: E402
    DEFAULT_LIBRARY_ROOT,
    DEFAULT_OUTCOME_ROOT,
    HISTORY_LENGTH_SUM,
    LIBRARY_SIZE_CASE_IDS,
    POLICY_HISTORY_CONDITIONS,
    R11_L1_SINGLE_FAN_FIXED_NOMINAL_BLOCK_ID,
    R11_L2_FOUR_FAN_FIXED_NOMINAL_BLOCK_ID,
    ValidationBlockSpec,
    ValidationProtocol,
    ValidationRunConfig,
    run_repeated_launch_validation,
)


TARGETED_MEMORY_DIAGNOSTIC_VERSION = "targeted_memory_mechanism_challenge_v1"
TARGETED_POLICY_HISTORY_CONDITIONS = POLICY_HISTORY_CONDITIONS


@dataclass(frozen=True)
class TargetedMemoryDiagnosticConfig:
    library_root: Path = DEFAULT_LIBRARY_ROOT
    outcome_root: Path = DEFAULT_OUTCOME_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    source_w2_root: Path | None = None
    run_id: int = 1
    run_label: str = ""
    seed: int = 110
    target_block: str = "single_fan_fixed"
    outer_cases: int = 2
    max_episode_time_s: float = 4.0
    storage_format: str = "auto"
    compression_level: int = 1
    candidate_chunk_size: int = 800
    dry_run_schedule: bool = False
    workers: int = 1
    max_workers: int | None = None
    worker_backend: str = "process"
    history_log_mode: str = "plot_summary"
    history_debug_sample_stride: int = 10


def _target_block_spec(target_block: str, outer_cases: int) -> ValidationBlockSpec:
    if target_block == "single_fan_fixed":
        return ValidationBlockSpec(
            R11_L1_SINGLE_FAN_FIXED_NOMINAL_BLOCK_ID,
            "targeted single-fan fixed nominal memory challenge",
            "W2",
            "annular_gp_single",
            int(outer_cases),
            "targeted_single_fan_fixed_nominal_memory_challenge",
        )
    if target_block == "four_fan_fixed":
        return ValidationBlockSpec(
            R11_L2_FOUR_FAN_FIXED_NOMINAL_BLOCK_ID,
            "targeted four-fan fixed nominal memory challenge",
            "W2",
            "annular_gp_four",
            int(outer_cases),
            "targeted_four_fan_fixed_nominal_memory_challenge",
        )
    raise ValueError(f"unknown_target_block:{target_block}")


def _protocol(config: TargetedMemoryDiagnosticConfig) -> ValidationProtocol:
    block = _target_block_spec(config.target_block, config.outer_cases)
    expected_final = len(LIBRARY_SIZE_CASE_IDS) * len(TARGETED_POLICY_HISTORY_CONDITIONS) * int(config.outer_cases)
    expected_history = len(LIBRARY_SIZE_CASE_IDS) * int(config.outer_cases) * HISTORY_LENGTH_SUM
    return ValidationProtocol(
        stage_id="R10",
        manifest_name="targeted_memory_diagnostic_manifest.json",
        report_name="targeted_memory_diagnostic_report.md",
        manifest_version=TARGETED_MEMORY_DIAGNOSTIC_VERSION,
        validation_evidence_level="targeted_memory_mechanism_diagnostic_not_claim_evidence",
        outer_cases_per_condition=int(config.outer_cases),
        expected_final_heldout_launches=int(expected_final),
        expected_history_launches=int(expected_history),
        blocks=(block,),
        final_schedule_prefix=f"r10_targeted_{config.target_block}",
        policy_history_conditions=TARGETED_POLICY_HISTORY_CONDITIONS,
        reduced_diagnostic=True,
        requires_no_glider_latency_variation_audit=True,
        gate_profile="targeted_memory_mechanism_diagnostic_not_full_r10",
        max_hard_failure_rate=0.20,
        max_floor_or_ceiling_violation_rate=0.20,
        max_no_viable_rate=0.30,
        min_safe_success_rate=0.20,
        min_terminal_or_lift_capture_rate=0.0,
    )


def run_targeted_memory_diagnostic(config: TargetedMemoryDiagnosticConfig) -> dict[str, object]:
    return run_repeated_launch_validation(
        ValidationRunConfig(
            library_root=config.library_root,
            outcome_root=config.outcome_root,
            output_root=config.output_root,
            run_id=config.run_id,
            run_label=config.run_label,
            source_w2_root=config.source_w2_root,
            seed=config.seed,
            storage_format=config.storage_format,
            compression_level=config.compression_level,
            candidate_chunk_size=config.candidate_chunk_size,
            dry_run_schedule=config.dry_run_schedule,
            max_primitives_per_launch=0,
            max_episode_time_s=float(config.max_episode_time_s),
            smoke_outer_cases_per_block=0,
            workers=config.workers,
            max_workers=config.max_workers,
            worker_backend=config.worker_backend,
            governor_config=None,
            history_log_mode=config.history_log_mode,
            history_debug_sample_stride=config.history_debug_sample_stride,
        ),
        protocol=_protocol(config),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a targeted repeated-launch memory mechanism diagnostic.")
    parser.add_argument("--library-root", type=Path, default=DEFAULT_LIBRARY_ROOT)
    parser.add_argument("--outcome-root", type=Path, default=DEFAULT_OUTCOME_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-w2-root", type=Path, default=None)
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--run-label", default="")
    parser.add_argument("--seed", type=int, default=110)
    parser.add_argument("--target-block", choices=("single_fan_fixed", "four_fan_fixed"), default="single_fan_fixed")
    parser.add_argument("--outer-cases", type=int, default=2)
    parser.add_argument("--max-episode-time-s", type=float, default=4.0)
    parser.add_argument("--storage-format", default="auto", choices=("auto", "parquet", "csv_gz", "csv"))
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument("--candidate-chunk-size", type=int, default=800)
    parser.add_argument("--dry-run-schedule", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--worker-backend", choices=("thread", "process"), default="process")
    parser.add_argument("--history-log-mode", choices=("auto", "plot_summary", "sampled_debug", "full_debug"), default="plot_summary")
    parser.add_argument("--history-debug-sample-stride", type=int, default=10)
    args = parser.parse_args(argv)

    result = run_targeted_memory_diagnostic(
        TargetedMemoryDiagnosticConfig(
            library_root=args.library_root,
            outcome_root=args.outcome_root,
            output_root=args.output_root,
            source_w2_root=args.source_w2_root,
            run_id=args.run_id,
            run_label=args.run_label,
            seed=args.seed,
            target_block=args.target_block,
            outer_cases=args.outer_cases,
            max_episode_time_s=args.max_episode_time_s,
            storage_format=args.storage_format,
            compression_level=args.compression_level,
            candidate_chunk_size=args.candidate_chunk_size,
            dry_run_schedule=args.dry_run_schedule,
            workers=args.workers,
            max_workers=args.max_workers,
            worker_backend=args.worker_backend,
            history_log_mode=args.history_log_mode,
            history_debug_sample_stride=args.history_debug_sample_stride,
        )
    )
    print(result)
    return 0 if result.get("status") in {"complete", "dry_run_schedule", "smoke_run"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
