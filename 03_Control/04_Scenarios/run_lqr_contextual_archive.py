from __future__ import annotations

import argparse
import sys
from pathlib import Path

CONTROL_ROOT = Path(__file__).resolve().parents[1]
for rel in ("02_Inner_Loop", "03_Primitives", "04_Scenarios"):
    path = CONTROL_ROOT / rel
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_ctx_archive import ContextArchiveConfig, run_contextual_archive_preflight  # noqa: E402


def parse_args(argv: list[str] | None = None) -> ContextArchiveConfig:
    parser = argparse.ArgumentParser(description="Run the active W0/W1 LQR contextual archive.")
    parser.add_argument("--run-id", type=int, default=100)
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--w-layers", default="W0,W1")
    parser.add_argument("--env-modes", default="dry_air,gaussian_single")
    parser.add_argument("--candidate-chunk-size", "--chunk-size", dest="candidate_chunk_size", type=int, default=125)
    parser.add_argument("--workers", default="8")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--storage-format", default="auto")
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--repair-incomplete", action="store_true")
    parser.add_argument("--dry-run-schedule", action="store_true")
    parser.add_argument("--stop-after-chunks", type=int, default=None)
    parser.add_argument("--continue-on-chunk-failure", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("03_Control/05_Results/lqr_contextual_v1_0/r6"))
    parser.add_argument("--rollout-backend", choices=("model_backed_lqr", "smoke_only"), default="model_backed_lqr")
    parser.add_argument("--selected-controller-registry", type=Path, default=None)
    args = parser.parse_args(argv)
    return ContextArchiveConfig(
        run_id=int(args.run_id),
        rows=int(args.rows),
        seed=int(args.seed),
        w_layers=_split_csv(args.w_layers),
        env_modes=_split_csv(args.env_modes),
        candidate_chunk_size=int(args.candidate_chunk_size),
        workers=args.workers,
        max_workers=int(args.max_workers),
        storage_format=str(args.storage_format),
        compression_level=int(args.compression_level),
        resume=bool(args.resume),
        repair_incomplete=bool(args.repair_incomplete),
        dry_run_schedule=bool(args.dry_run_schedule),
        stop_after_chunks=args.stop_after_chunks,
        continue_on_chunk_failure=bool(args.continue_on_chunk_failure),
        output_root=Path(args.output_root),
        rollout_backend=str(args.rollout_backend),
        selected_controller_registry=None
        if args.selected_controller_registry is None
        else Path(args.selected_controller_registry),
    )


def _split_csv(text: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in str(text).split(",") if item.strip())
    if not values:
        raise ValueError("comma-separated option must contain at least one value.")
    return values


def main(argv: list[str] | None = None) -> int:
    run_contextual_archive_preflight(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
