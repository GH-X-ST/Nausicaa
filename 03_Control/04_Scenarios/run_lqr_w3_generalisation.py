from __future__ import annotations

import argparse
import sys
from pathlib import Path

CONTROL_ROOT = Path(__file__).resolve().parents[1]
for rel in ("02_Inner_Loop", "03_Primitives", "04_Scenarios"):
    path = CONTROL_ROOT / rel
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_w3_generalisation import W3GeneralisationConfig, run_w3_generalisation  # noqa: E402


def parse_args(argv: list[str] | None = None) -> W3GeneralisationConfig:
    parser = argparse.ArgumentParser(description="Run W3 LQR replay-only generalisation evaluation.")
    parser.add_argument("--run-id", type=int, default=300)
    parser.add_argument("--output-root", type=Path, default=Path("03_Control/05_Results/lqr_contextual_v1_0/r9"))
    parser.add_argument("--source-replay", type=Path, default=None)
    parser.add_argument("--target-rows", type=int, default=30_000)
    parser.add_argument("--fallback-rows", type=int, default=5_000)
    parser.add_argument("--max-source-rows", type=int, default=0)
    parser.add_argument("--storage-format", default="auto")
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--chunk-size", "--candidate-chunk-size", dest="chunk_size", type=int, default=1000)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--repair-incomplete", action="store_true")
    parser.add_argument("--dry-run-schedule", action="store_true")
    parser.add_argument("--stop-after-chunks", type=int, default=None)
    parser.add_argument("--continue-on-chunk-failure", action="store_true")
    args = parser.parse_args(argv)
    return W3GeneralisationConfig(
        run_id=int(args.run_id),
        output_root=Path(args.output_root),
        source_replay=None if args.dry_run_schedule else args.source_replay,
        target_rows=int(args.target_rows),
        fallback_rows=int(args.fallback_rows),
        max_source_rows=int(args.max_source_rows),
        execute_replay=not bool(args.dry_run_schedule),
        storage_format=str(args.storage_format),
        compression_level=int(args.compression_level),
        workers=int(args.workers),
        max_workers=int(args.max_workers),
        chunk_size=int(args.chunk_size),
        resume=bool(args.resume),
        repair_incomplete=bool(args.repair_incomplete),
        stop_after_chunks=args.stop_after_chunks,
    )


def main(argv: list[str] | None = None) -> int:
    run_w3_generalisation(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

