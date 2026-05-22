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
class W3GeneralisationConfig:
    run_id: int
    output_root: Path
    source_replay: Path | None = None


def parse_args(argv: list[str] | None = None) -> W3GeneralisationConfig:
    parser = argparse.ArgumentParser(description="Write a temp W3 generalisation scaffold manifest.")
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-replay", type=Path, default=None)
    args = parser.parse_args(argv)
    return W3GeneralisationConfig(
        run_id=int(args.run_id),
        output_root=Path(args.output_root),
        source_replay=None if args.source_replay is None else Path(args.source_replay),
    )


def run_w3_generalisation_scaffold(config: W3GeneralisationConfig) -> dict[str, object]:
    run_root = Path(config.output_root) / f"w3_generalisation_{config.run_id:03d}"
    for rel in ("manifests", "reports", "metrics"):
        filesystem_path(run_root / rel).mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": int(config.run_id),
        "stage": "R9_W3_generalisation_scaffold",
        "source_replay": "" if config.source_replay is None else Path(config.source_replay).as_posix(),
        "W_layer": "W3",
        "surrogate_family": "randomised_gp_corrected_annular_gaussian",
        "randomisation_scope": [
            "fan_position",
            "fan_power",
            "active_fan_subset",
            "amplitude",
            "width",
            "centre_shift",
            "uncertainty_scale",
        ],
        "generalisation_status": "blocked_until_W2_supported_cases_exist",
        "claim_status": "simulation_only_w3_scaffold_no_robustness_claim",
        "blocked_claims": ["W3_robustness", "environment_generalisation", "hardware_readiness"],
    }
    manifest_path = run_root / "manifests" / "w3_generalisation_manifest.json"
    filesystem_path(manifest_path).write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    pd.DataFrame([{"path": "manifests/w3_generalisation_manifest.json", "under_100mb": True}]).to_csv(
        filesystem_path(run_root / "metrics" / "file_size_audit.csv"),
        index=False,
    )
    filesystem_path(run_root / "reports" / "w3_generalisation_report.md").write_text(
        "# W3 Generalisation Scaffold\n\nNo W3 robustness or environment-generalisation claim is made.\n",
        encoding="ascii",
    )
    return {"run_root": run_root, "manifest": manifest_path}


def main(argv: list[str] | None = None) -> int:
    run_w3_generalisation_scaffold(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
