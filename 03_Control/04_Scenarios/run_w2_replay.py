from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

CONTROL_ROOT = Path(__file__).resolve().parents[1]
for rel in ("02_Inner_Loop", "03_Primitives", "04_Scenarios"):
    path = CONTROL_ROOT / rel
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dense_archive_table_io import filesystem_path  # noqa: E402


@dataclass(frozen=True)
class W2ReplayConfig:
    run_id: int
    output_root: Path
    source_archive: Path | None = None


def parse_args(argv: list[str] | None = None) -> W2ReplayConfig:
    parser = argparse.ArgumentParser(description="Write a temp W2 replay scaffold manifest.")
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, default=None)
    args = parser.parse_args(argv)
    return W2ReplayConfig(
        run_id=int(args.run_id),
        output_root=Path(args.output_root),
        source_archive=None if args.source_archive is None else Path(args.source_archive),
    )


def run_w2_replay_scaffold(config: W2ReplayConfig) -> dict[str, object]:
    run_root = Path(config.output_root) / f"w2_replay_{config.run_id:03d}"
    for rel in ("manifests", "reports", "metrics"):
        filesystem_path(run_root / rel).mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": int(config.run_id),
        "stage": "R8_W2_replay_scaffold",
        "source_archive": "" if config.source_archive is None else Path(config.source_archive).as_posix(),
        "W_layer": "W2",
        "surrogate_family": "gp_corrected_annular_gaussian",
        "required_latency_mechanisms": ["state_feedback_delay", "command_delay", "actuator_lag"],
        "replay_status": "blocked_until_approved_R6_archive_exists",
        "claim_status": "simulation_only_w2_scaffold_no_survival_claim",
        "blocked_claims": ["W2_survival", "hardware_readiness", "real_flight_transfer"],
    }
    manifest_path = run_root / "manifests" / "w2_replay_manifest.json"
    filesystem_path(manifest_path).write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    pd.DataFrame([{"path": "manifests/w2_replay_manifest.json", "under_100mb": True}]).to_csv(
        filesystem_path(run_root / "metrics" / "file_size_audit.csv"),
        index=False,
    )
    filesystem_path(run_root / "reports" / "w2_replay_report.md").write_text(
        "# W2 Replay Scaffold\n\nNo W2 survival or hardware-readiness claim is made.\n",
        encoding="ascii",
    )
    return {"run_root": run_root, "manifest": manifest_path}


def main(argv: list[str] | None = None) -> int:
    run_w2_replay_scaffold(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
