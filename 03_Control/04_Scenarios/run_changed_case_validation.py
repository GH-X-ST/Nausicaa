from __future__ import annotations

import argparse
import json
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

from run_repeated_launch_learning_curve import (  # noqa: E402
    DEFAULT_LIBRARY_ROOT,
    DEFAULT_OUTCOME_ROOT,
    DEFAULT_VALIDATION_MAX_EPISODE_TIME_S,
    HISTORY_LENGTH_SUM,
    LIBRARY_SIZE_CASE_IDS,
    POLICY_HISTORY_CONDITIONS,
    ValidationBlockSpec,
    ValidationProtocol,
    ValidationRunConfig,
    run_repeated_launch_validation,
)
from viability_governor import GovernorConfig, governor_config_from_row  # noqa: E402


R10_VALIDATION_VERSION = "environment_changed_case_repeated_launch_validation_v7"
R11_VALIDATION_VERSION = "heldout_environment_changed_case_repeated_launch_validation_v1"
R10_OUTER_CASES_PER_CONDITION = 120
R10_REDUCED_OUTER_CASES_PER_CONDITION = 50
R10_EXPECTED_FINAL_HELDOUT_LAUNCHES = len(LIBRARY_SIZE_CASE_IDS) * len(POLICY_HISTORY_CONDITIONS) * R10_OUTER_CASES_PER_CONDITION
R10_EXPECTED_HISTORY_LAUNCHES = len(LIBRARY_SIZE_CASE_IDS) * R10_OUTER_CASES_PER_CONDITION * (
    HISTORY_LENGTH_SUM + HISTORY_LENGTH_SUM
)
R10_REDUCED_EXPECTED_FINAL_HELDOUT_LAUNCHES = (
    len(LIBRARY_SIZE_CASE_IDS) * len(POLICY_HISTORY_CONDITIONS) * R10_REDUCED_OUTER_CASES_PER_CONDITION
)
R10_REDUCED_EXPECTED_HISTORY_LAUNCHES = len(LIBRARY_SIZE_CASE_IDS) * R10_REDUCED_OUTER_CASES_PER_CONDITION * (
    HISTORY_LENGTH_SUM + HISTORY_LENGTH_SUM
)
DEFAULT_OUTPUT_ROOT = Path("03_Control/05_Results/lqr_contextual_v1_0/changed_case_validation")
DEFAULT_R11_OUTPUT_ROOT = Path("03_Control/05_Results/lqr_contextual_v1_0/heldout_changed_case_validation")

R10_BLOCKS: tuple[ValidationBlockSpec, ...] = (
    ValidationBlockSpec(
        "nominal_single_fan_perturbations",
        "nominal single-fan perturbations",
        "W3",
        "w3_randomised_single",
        20,
        "nominal_single_fan_perturbations",
    ),
    ValidationBlockSpec(
        "nominal_four_fan_perturbations",
        "nominal four-fan perturbations",
        "W3",
        "w3_randomised_four",
        20,
        "nominal_four_fan_perturbations",
    ),
    ValidationBlockSpec(
        "shifted_single_fan_positions",
        "shifted single-fan positions",
        "W3",
        "w3_randomised_single",
        20,
        "shifted_single_fan_positions",
    ),
    ValidationBlockSpec(
        "shifted_four_fan_positions",
        "shifted four-fan positions",
        "W3",
        "w3_randomised_four",
        20,
        "shifted_four_fan_positions",
    ),
    ValidationBlockSpec(
        "active_fan_number_variation",
        "active-fan-number variation",
        "W3",
        "w3_randomised_four",
        20,
        "active_fan_number_variation",
    ),
    ValidationBlockSpec(
        "arena_wide_fan_position_generalisation",
        "arena-wide fan-position generalisation",
        "W3",
        "w3_randomised_four",
        20,
        "arena_wide_fan_position_generalisation",
    ),
)

R10_REDUCED_BLOCK_COUNTS = {
    "nominal_single_fan_perturbations": 8,
    "nominal_four_fan_perturbations": 8,
    "shifted_single_fan_positions": 8,
    "shifted_four_fan_positions": 8,
    "active_fan_number_variation": 9,
    "arena_wide_fan_position_generalisation": 9,
}

R10_REDUCED_BLOCKS: tuple[ValidationBlockSpec, ...] = tuple(
    ValidationBlockSpec(
        block.block_id,
        block.human_label,
        block.W_layer,
        block.environment_mode,
        R10_REDUCED_BLOCK_COUNTS[block.block_id],
        block.environment_change_family,
    )
    for block in R10_BLOCKS
)

R10_PROTOCOL = ValidationProtocol(
    stage_id="R10",
    manifest_name="environment_changed_case_manifest.json",
    report_name="environment_changed_case_report.md",
    manifest_version=R10_VALIDATION_VERSION,
    validation_evidence_level="changed_case_viability_governor_learning_rollout_validation_not_final_claim_gate",
    outer_cases_per_condition=R10_OUTER_CASES_PER_CONDITION,
    expected_final_heldout_launches=R10_EXPECTED_FINAL_HELDOUT_LAUNCHES,
    expected_history_launches=R10_EXPECTED_HISTORY_LAUNCHES,
    blocks=R10_BLOCKS,
    final_schedule_prefix="r10_changed",
    reduced_diagnostic=False,
    requires_no_glider_latency_variation_audit=True,
    gate_profile="relaxed_changed_case_viability_governor_learning_not_final_validation",
    max_hard_failure_rate=0.20,
    max_no_viable_rate=0.30,
    min_safe_success_rate=0.20,
    min_terminal_or_lift_capture_rate=0.30,
)

R11_PROTOCOL = ValidationProtocol(
    stage_id="R11",
    manifest_name="heldout_environment_validation_manifest.json",
    report_name="heldout_environment_validation_report.md",
    manifest_version=R11_VALIDATION_VERSION,
    validation_evidence_level="strict_heldout_environment_only_changed_case_repeated_launch_rollout_validation",
    outer_cases_per_condition=R10_OUTER_CASES_PER_CONDITION,
    expected_final_heldout_launches=R10_EXPECTED_FINAL_HELDOUT_LAUNCHES,
    expected_history_launches=R10_EXPECTED_HISTORY_LAUNCHES,
    blocks=R10_BLOCKS,
    final_schedule_prefix="r11_heldout",
    reduced_diagnostic=False,
    requires_no_glider_latency_variation_audit=True,
    gate_profile="strict_final_heldout_validation",
    max_hard_failure_rate=0.01,
    max_no_viable_rate=0.02,
    min_safe_success_rate=0.99,
    min_full_safe_success_rate=0.99,
    min_terminal_or_lift_capture_rate=0.90,
)

R10_REDUCED_PROTOCOL = ValidationProtocol(
    stage_id="R10",
    manifest_name="environment_changed_case_manifest.json",
    report_name="environment_changed_case_report.md",
    manifest_version=R10_VALIDATION_VERSION,
    validation_evidence_level="reduced_diagnostic_not_target_R10",
    outer_cases_per_condition=R10_REDUCED_OUTER_CASES_PER_CONDITION,
    expected_final_heldout_launches=R10_REDUCED_EXPECTED_FINAL_HELDOUT_LAUNCHES,
    expected_history_launches=R10_REDUCED_EXPECTED_HISTORY_LAUNCHES,
    blocks=R10_REDUCED_BLOCKS,
    final_schedule_prefix="r10_reduced_diagnostic",
    reduced_diagnostic=True,
    requires_no_glider_latency_variation_audit=True,
)


@dataclass(frozen=True)
class ChangedCaseValidationConfig:
    library_root: Path = DEFAULT_LIBRARY_ROOT
    outcome_root: Path = DEFAULT_OUTCOME_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    run_id: int = 1
    source_w2_root: Path | None = None
    seed: int = 110
    storage_format: str = "auto"
    compression_level: int = 1
    candidate_chunk_size: int = 800
    dry_run_schedule: bool = False
    max_primitives_per_launch: int = 0
    max_episode_time_s: float = DEFAULT_VALIDATION_MAX_EPISODE_TIME_S
    smoke_outer_cases_per_block: int = 0
    r10_mode: str = "full"
    workers: int = 1
    max_workers: int | None = None
    worker_backend: str = "thread"
    governor_config: GovernorConfig | None = None
    governor_config_path: Path | None = None


@dataclass(frozen=True)
class HeldoutChangedCaseValidationConfig(ChangedCaseValidationConfig):
    output_root: Path = DEFAULT_R11_OUTPUT_ROOT


def run_changed_case_validation(config: ChangedCaseValidationConfig) -> dict[str, object]:
    """Run full R10 changed-case validation, or an explicitly reduced diagnostic schedule."""

    protocol = R10_REDUCED_PROTOCOL if str(config.r10_mode) == "reduced_diagnostic_50" else R10_PROTOCOL
    governor_config = _resolve_governor_config(config)
    return run_repeated_launch_validation(
        ValidationRunConfig(
            library_root=config.library_root,
            outcome_root=config.outcome_root,
            output_root=config.output_root,
            run_id=config.run_id,
            source_w2_root=config.source_w2_root,
            seed=config.seed,
            storage_format=config.storage_format,
            compression_level=config.compression_level,
            candidate_chunk_size=config.candidate_chunk_size,
            dry_run_schedule=config.dry_run_schedule,
            max_primitives_per_launch=config.max_primitives_per_launch,
            max_episode_time_s=config.max_episode_time_s,
            smoke_outer_cases_per_block=config.smoke_outer_cases_per_block,
            workers=config.workers,
            max_workers=config.max_workers,
            worker_backend=config.worker_backend,
            governor_config=governor_config,
        ),
        protocol=protocol,
    )


def run_heldout_changed_case_validation(config: HeldoutChangedCaseValidationConfig) -> dict[str, object]:
    """Run strict R11 held-out changed-case validation after R10 governor tuning."""

    governor_config = _resolve_governor_config(config)
    return run_repeated_launch_validation(
        ValidationRunConfig(
            library_root=config.library_root,
            outcome_root=config.outcome_root,
            output_root=config.output_root,
            run_id=config.run_id,
            source_w2_root=config.source_w2_root,
            seed=config.seed,
            storage_format=config.storage_format,
            compression_level=config.compression_level,
            candidate_chunk_size=config.candidate_chunk_size,
            dry_run_schedule=config.dry_run_schedule,
            max_primitives_per_launch=config.max_primitives_per_launch,
            max_episode_time_s=config.max_episode_time_s,
            smoke_outer_cases_per_block=config.smoke_outer_cases_per_block,
            workers=config.workers,
            max_workers=config.max_workers,
            worker_backend=config.worker_backend,
            governor_config=governor_config,
        ),
        protocol=R11_PROTOCOL,
    )


def _resolve_governor_config(config: ChangedCaseValidationConfig) -> GovernorConfig | None:
    if config.governor_config is not None:
        return config.governor_config
    if config.governor_config_path is None:
        return None
    payload = json.loads(Path(config.governor_config_path).read_text(encoding="ascii"))
    row = payload.get("governor_config", payload)
    if not isinstance(row, dict):
        raise ValueError("governor_config_path_missing_governor_config_object")
    return governor_config_from_row(row)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-root", type=Path, default=DEFAULT_LIBRARY_ROOT)
    parser.add_argument("--outcome-root", type=Path, default=DEFAULT_OUTCOME_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-w2-root", type=Path, default=None)
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=110)
    parser.add_argument("--storage-format", default="auto", choices=("auto", "parquet", "csv_gz", "csv"))
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument("--candidate-chunk-size", type=int, default=800)
    parser.add_argument(
        "--max-primitives-per-launch",
        type=int,
        default=0,
        help="Optional diagnostic primitive-count cap. Use 0 to disable the cap for full validation.",
    )
    parser.add_argument("--max-episode-time-s", type=float, default=DEFAULT_VALIDATION_MAX_EPISODE_TIME_S)
    parser.add_argument("--smoke-outer-cases-per-block", type=int, default=0)
    parser.add_argument("--dry-run-schedule", action="store_true")
    parser.add_argument("--r10-mode", default="full", choices=("full", "reduced_diagnostic_50"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--worker-backend", choices=("thread", "process"), default="thread")
    parser.add_argument("--governor-config-path", type=Path, default=None)
    args = parser.parse_args(argv)

    result = run_changed_case_validation(
        ChangedCaseValidationConfig(
            library_root=args.library_root,
            outcome_root=args.outcome_root,
            output_root=args.output_root,
            source_w2_root=args.source_w2_root,
            run_id=args.run_id,
            seed=args.seed,
            storage_format=args.storage_format,
            compression_level=args.compression_level,
            candidate_chunk_size=args.candidate_chunk_size,
            dry_run_schedule=args.dry_run_schedule,
            max_primitives_per_launch=args.max_primitives_per_launch,
            max_episode_time_s=args.max_episode_time_s,
            smoke_outer_cases_per_block=args.smoke_outer_cases_per_block,
            r10_mode=args.r10_mode,
            workers=args.workers,
            max_workers=args.max_workers,
            worker_backend=args.worker_backend,
            governor_config_path=args.governor_config_path,
        )
    )
    print(result)
    return 0 if result.get("status") in {"complete", "dry_run_schedule", "smoke_run"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
