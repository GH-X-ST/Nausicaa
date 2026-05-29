from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap_import_paths() -> None:
    control_root = Path(__file__).resolve().parents[1]
    for rel in ("03_Primitives", "04_Scenarios"):
        path = control_root / rel
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


_bootstrap_import_paths()

from run_changed_case_validation import (  # noqa: E402
    DEFAULT_LIBRARY_ROOT,
    DEFAULT_OUTCOME_ROOT,
    DEFAULT_R11_OUTPUT_ROOT,
    DEFAULT_VALIDATION_MAX_EPISODE_TIME_S,
    HeldoutChangedCaseValidationConfig,
    run_heldout_changed_case_validation,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run strict R11 held-out changed-case repeated-launch validation.")
    parser.add_argument("--library-root", type=Path, default=DEFAULT_LIBRARY_ROOT)
    parser.add_argument("--outcome-root", type=Path, default=DEFAULT_OUTCOME_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_R11_OUTPUT_ROOT)
    parser.add_argument("--source-w2-root", type=Path, default=None)
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--storage-format", default="auto", choices=("auto", "parquet", "csv_gz", "csv"))
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument("--candidate-chunk-size", type=int, default=20_000)
    parser.add_argument(
        "--max-primitives-per-launch",
        type=int,
        default=0,
        help="Optional diagnostic primitive-count cap. Use 0 to disable the cap for full held-out validation.",
    )
    parser.add_argument("--max-episode-time-s", type=float, default=DEFAULT_VALIDATION_MAX_EPISODE_TIME_S)
    parser.add_argument("--smoke-outer-cases-per-block", type=int, default=0)
    parser.add_argument("--dry-run-schedule", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--worker-backend", choices=("thread", "process"), default="process")
    parser.add_argument("--governor-config-path", type=Path, default=None)
    parser.add_argument("--history-log-mode", choices=("auto", "plot_summary", "sampled_debug", "full_debug"), default="auto")
    parser.add_argument("--history-debug-sample-stride", type=int, default=10)
    args = parser.parse_args(argv)

    result = run_heldout_changed_case_validation(
        HeldoutChangedCaseValidationConfig(
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
            workers=args.workers,
            max_workers=args.max_workers,
            worker_backend=args.worker_backend,
            governor_config_path=args.governor_config_path,
            history_log_mode=args.history_log_mode,
            history_debug_sample_stride=args.history_debug_sample_stride,
        )
    )
    print(result)
    return 0 if result.get("status") in {"complete", "dry_run_schedule", "smoke_run"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
