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

from archive_table_reader import read_archive_table  # noqa: E402
from dense_archive_table_io import filesystem_path  # noqa: E402
from env_instance import environment_instance_for_mode, environment_instance_row  # noqa: E402
from implementation_instance import (  # noqa: E402
    implementation_instance_for_layer,
    implementation_instance_row,
)
from plant_instance import plant_instance_for_layer, plant_instance_row  # noqa: E402
from state_contract import STATE_NAMES  # noqa: E402


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
    for rel in ("manifests", "reports", "metrics", "tables"):
        filesystem_path(run_root / rel).mkdir(parents=True, exist_ok=True)
    cases = _w3_case_rows(config)
    environment = environment_instance_for_mode("W3", "w3_randomised", config.run_id)
    implementation = implementation_instance_for_layer("W3", config.run_id, latency_case="nominal")
    plant = plant_instance_for_layer("W3", config.run_id)
    manifest = {
        "run_id": int(config.run_id),
        "stage": "R9_W3_generalisation_scaffold",
        "source_replay": "" if config.source_replay is None else Path(config.source_replay).as_posix(),
        "W_layer": "W3",
        "surrogate_family": "randomised_gp_corrected_annular_gaussian",
        "R9_W3_generalisation_complete": False,
        "randomisation_scope": [
            "fan_position",
            "fan_power",
            "active_fan_subset",
            "amplitude",
            "width",
            "centre_shift",
            "residual_vertical_field_label",
            "uncertainty_scale",
            "mixed_primitive_start_state",
            "state_feedback_delay",
            "command_delay",
            "actuator_lag",
            "latency_jitter",
            "surface_effectiveness",
            "surface_neutral_bias",
            "surface_limit_scale",
            "mass_scale",
            "cg_offset",
            "inertia_scale",
            "surface_calibration_scale",
        ],
        "environment_adjustment_status": "approximate_for_gp_grid_active_mask_and_per_fan_power",
        "environment_adjustment_limitations": (
            "GP-grid fields are not exactly decomposable by active fan subset or per-fan power; "
            "amplitude width centre and uncertainty randomisation remain explicit"
        ),
        "implementation_adjustment_status": implementation.implementation_adjustment_status,
        "plant_adjustment_status": plant.plant_adjustment_status,
        "generalisation_status": (
            "partial_smoke_from_source_no_robustness_claim"
            if cases
            else "blocked_until_W2_supported_cases_exist"
        ),
        "case_count": len(cases),
        "claim_status": "simulation_only_w3_scaffold_no_robustness_claim",
        "blocked_claims": ["W3_robustness", "environment_generalisation", "hardware_readiness"],
    }
    case_path = run_root / "tables" / "w3_generalisation_cases.csv"
    for case in cases:
        case.update({f"environment_instance_{key}": value for key, value in environment_instance_row(environment).items()})
        case.update({f"implementation_instance_{key}": value for key, value in implementation_instance_row(implementation).items()})
        case.update({f"plant_instance_{key}": value for key, value in plant_instance_row(plant).items()})
    pd.DataFrame(cases).to_csv(filesystem_path(case_path), index=False)
    manifest_path = run_root / "manifests" / "w3_generalisation_manifest.json"
    filesystem_path(manifest_path).write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    filesystem_path(run_root / "reports" / "w3_generalisation_report.md").write_text(
        "# W3 Generalisation Scaffold\n\nNo W3 robustness or environment-generalisation claim is made.\n",
        encoding="ascii",
    )
    _write_file_size_audit(run_root)
    return {"run_root": run_root, "manifest": manifest_path, "case_table": case_path}


def _w3_case_rows(config: W3GeneralisationConfig) -> list[dict[str, object]]:
    if config.source_replay is None:
        return []
    frame = read_archive_table(Path(config.source_replay), max_rows=16)
    rows = []
    for _, row in frame.head(4).iterrows():
        source = row.to_dict()
        case = {
            "source_rollout_id": source.get("rollout_id", source.get("source_rollout_id", "")),
            "start_state_family": source.get("start_state_family", ""),
            "state_envelope_label": source.get("state_envelope_label", ""),
            "previous_primitive_status": source.get("previous_primitive_status", ""),
            "primitive_id": source.get("primitive_id", ""),
            "W_layer": "W3",
            "w3_case_status": "smoke_case_not_robustness_evidence",
            "claim_status": "simulation_only_w3_case_no_robustness_claim",
        }
        case.update(
            {
                f"entry_{name}": float(source.get(f"initial_{name}", source.get(f"entry_{name}", 0.0)))
                for name in STATE_NAMES
            }
        )
        rows.append(case)
    return rows


def _write_file_size_audit(run_root: Path) -> None:
    rows = []
    for path in sorted(run_root.rglob("*")):
        if path.is_file():
            size = path.stat().st_size
            rows.append(
                {
                    "path": path.relative_to(run_root).as_posix(),
                    "byte_count": int(size),
                    "under_100mb": bool(size <= 100 * 1024 * 1024),
                }
            )
    pd.DataFrame(rows).to_csv(filesystem_path(run_root / "metrics" / "file_size_audit.csv"), index=False)


def main(argv: list[str] | None = None) -> int:
    run_w3_generalisation_scaffold(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
