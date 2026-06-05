"""Run a small R11 readiness validation before real-flight deployment.

This is an engineering smoke check, not the final thesis/journal R11. It keeps
the same R11 ladder structure, library tiers, policy/history pairing, frozen
R10 governor handoff, actuator/latency model, and file-management path as the
full validation, but uses only a few paired local cases per ladder.

Default schedule:
    8 ladders x 2 local cases x 5 library tiers x 5 policies = 400 final launches
    8 ladders x 2 local cases x 5 library tiers x (3+10+30) = 3440 history launches
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


def _bootstrap_import_paths() -> None:
    control_root = Path(__file__).resolve().parents[1]
    for rel in ("03_Primitives", "04_Scenarios"):
        path = control_root / rel
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


_bootstrap_import_paths()

from run_changed_case_validation import (  # noqa: E402
    DEFAULT_R11_OUTPUT_ROOT,
    HISTORY_LENGTH_SUM,
    HeldoutChangedCaseValidationConfig,
    LIBRARY_SIZE_CASE_IDS,
    R11_POLICY_HISTORY_CONDITIONS,
    R11_FIDELITY_LADDER_BLOCK_IDS,
    run_heldout_changed_case_validation,
)


DEFAULT_LIBRARY_ROOT = Path("03_Control/05_Results/R8_library_size_study/E01")
DEFAULT_OUTCOME_ROOT = Path("03_Control/05_Results/R8_outcome/E01")
DEFAULT_SOURCE_W2_ROOT = Path("03_Control/05_Results/R5_dense/E01")
DEFAULT_GOVERNOR_CONFIG_PATH = Path(
    "03_Control/05_Results/R10_learn/E01/manifests/frozen_governor_config_for_r11.json"
)
DEFAULT_RUN_LABEL = "SM01"
DEFAULT_OUTER_CASES_PER_LADDER = 2
DEFAULT_SEED = 711
DEFAULT_WORKERS = 8


def _expected_counts(outer_cases_per_ladder: int) -> tuple[int, int]:
    outer_cases = len(R11_FIDELITY_LADDER_BLOCK_IDS) * int(outer_cases_per_ladder)
    final_count = len(LIBRARY_SIZE_CASE_IDS) * len(R11_POLICY_HISTORY_CONDITIONS) * outer_cases
    history_count = len(LIBRARY_SIZE_CASE_IDS) * outer_cases * HISTORY_LENGTH_SUM
    return final_count, history_count


def _write_readiness_manifest(
    *,
    result: dict[str, object],
    outer_cases_per_ladder: int,
    final_count: int,
    history_count: int,
    dry_run_schedule: bool,
    launch_rate_stress: str,
    launch_rate_stress_fraction: float,
) -> None:
    run_root_text = str(result.get("run_root", ""))
    if not run_root_text:
        return
    run_root = Path(run_root_text)
    manifest_dir = run_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest_type": "r11_readiness_smoke_claim_boundary",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": result.get("status", ""),
        "run_root": run_root.as_posix(),
        "outer_cases_per_ladder": int(outer_cases_per_ladder),
        "expected_final_launches": int(final_count),
        "expected_history_launches": int(history_count),
        "dry_run_schedule": bool(dry_run_schedule),
        "launch_rate_stress": str(launch_rate_stress),
        "launch_rate_stress_fraction": float(launch_rate_stress_fraction),
        "purpose": "engineering_readiness_check_after_r10_before_real_flight",
        "claim_status": "not_final_thesis_or_journal_validation_evidence",
        "full_r11_required_for": [
            "final_statistical_validation_claim",
            "final_memory_improvement_claim",
            "appendix_speed_ladder_tables",
        ],
        "preserved_contracts": [
            "all_8_r11_ladders",
            "all_5_library_tiers",
            "all_5_policy_history_final_conditions_including_open_loop_comparison",
            "paired_start_conditions",
            "paired_history_seed_sequences",
            "frozen_r10_governor_config",
            "same_actuator_latency_quantisation_runtime",
        ],
    }
    (manifest_dir / "r11_readiness_smoke_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="ascii",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-root", type=Path, default=DEFAULT_LIBRARY_ROOT)
    parser.add_argument("--outcome-root", type=Path, default=DEFAULT_OUTCOME_ROOT)
    parser.add_argument("--source-w2-root", type=Path, default=DEFAULT_SOURCE_W2_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_R11_OUTPUT_ROOT)
    parser.add_argument("--run-label", default=DEFAULT_RUN_LABEL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--outer-cases-per-ladder", type=int, default=DEFAULT_OUTER_CASES_PER_LADDER)
    parser.add_argument("--governor-config-path", type=Path, default=DEFAULT_GOVERNOR_CONFIG_PATH)
    parser.add_argument("--storage-format", default="auto", choices=("auto", "parquet", "csv_gz", "csv"))
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument("--candidate-chunk-size", type=int, default=50_000)
    parser.add_argument("--max-primitives-per-launch", type=int, default=0)
    parser.add_argument("--max-episode-time-s", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--worker-backend", choices=("thread", "process"), default="process")
    parser.add_argument("--history-log-mode", choices=("auto", "plot_summary", "sampled_debug", "full_debug"), default="auto")
    parser.add_argument("--history-debug-sample-stride", type=int, default=10)
    parser.add_argument("--launch-rate-stress", choices=("nominal", "high"), default="nominal")
    parser.add_argument("--launch-rate-stress-fraction", type=float, default=0.75)
    parser.add_argument("--dry-run-schedule", action="store_true")
    args = parser.parse_args(argv)

    if int(args.outer_cases_per_ladder) <= 0:
        raise ValueError("outer-cases-per-ladder must be positive")

    final_count, history_count = _expected_counts(int(args.outer_cases_per_ladder))
    print("[R11_SMOKE] engineering readiness check, not final claim evidence")
    print(f"[R11_SMOKE] run_label={args.run_label}")
    print(f"[R11_SMOKE] outer_cases_per_ladder={int(args.outer_cases_per_ladder)}")
    print(f"[R11_SMOKE] expected_final_launches={final_count}")
    print(f"[R11_SMOKE] expected_history_launches={history_count}")
    print(f"[R11_SMOKE] launch_rate_stress={args.launch_rate_stress}")
    if str(args.launch_rate_stress) == "high":
        os.environ["NAUSICAA_LAUNCH_GATE_RATE_STRESS"] = "high"
        os.environ["NAUSICAA_LAUNCH_GATE_RATE_STRESS_FRACTION"] = f"{float(args.launch_rate_stress_fraction):.12g}"
    else:
        os.environ.pop("NAUSICAA_LAUNCH_GATE_RATE_STRESS", None)
        os.environ.pop("NAUSICAA_LAUNCH_GATE_RATE_STRESS_FRACTION", None)

    result = run_heldout_changed_case_validation(
        HeldoutChangedCaseValidationConfig(
            library_root=args.library_root,
            outcome_root=args.outcome_root,
            output_root=args.output_root,
            source_w2_root=args.source_w2_root,
            run_label=args.run_label,
            seed=args.seed,
            storage_format=args.storage_format,
            compression_level=args.compression_level,
            candidate_chunk_size=args.candidate_chunk_size,
            dry_run_schedule=bool(args.dry_run_schedule),
            max_primitives_per_launch=args.max_primitives_per_launch,
            max_episode_time_s=args.max_episode_time_s,
            outer_cases_per_ladder=args.outer_cases_per_ladder,
            workers=args.workers,
            max_workers=args.max_workers,
            worker_backend=args.worker_backend,
            governor_config_path=args.governor_config_path,
            history_log_mode=args.history_log_mode,
            history_debug_sample_stride=args.history_debug_sample_stride,
        )
    )
    _write_readiness_manifest(
        result=result,
        outer_cases_per_ladder=int(args.outer_cases_per_ladder),
        final_count=final_count,
        history_count=history_count,
        dry_run_schedule=bool(args.dry_run_schedule),
        launch_rate_stress=str(args.launch_rate_stress),
        launch_rate_stress_fraction=float(args.launch_rate_stress_fraction),
    )
    print(result)
    return 0 if result.get("status") in {"complete", "dry_run_schedule", "smoke_run"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
