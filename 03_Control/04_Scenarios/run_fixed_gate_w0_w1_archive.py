from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


SCENARIOS_DIR = Path(__file__).resolve().parent
CONTROL_DIR = SCENARIOS_DIR.parents[0]
PRIMITIVES_DIR = CONTROL_DIR / "03_Primitives"
for path in (PRIMITIVES_DIR, SCENARIOS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dense_archive_table_io import TableManifest, filesystem_path, resolve_storage_format, write_table_manifest, write_table_partition
from fixed_gate_code_path_map import active_code_path_text
from fixed_gate_sampling import (
    FixedGateSamplingConfig,
    build_fixed_gate_w0_w1_candidate_rows,
    sample_fixed_gate_states,
)
from primitive_roles import active_mission_primitive_families


CAMPAIGN = "11_fixed_gate_repeated_launch"
RESULT_ROOT = CONTROL_DIR / "05_Results" / CAMPAIGN


def run_fixed_gate_w0_w1_archive_scaffold(
    *,
    run_id: int,
    sample_count_per_branch: int = 20,
    result_root: Path | None = None,
    storage_format: str = "auto",
    overwrite: bool = False,
) -> dict[str, Path]:
    root = (RESULT_ROOT if result_root is None else Path(result_root)) / f"{int(run_id):03d}" / "w0_w1_fixed_gate_archive"
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise RuntimeError(f"result tree already exists: {root}")
    for name in ("tables", "manifests", "reports", "metrics"):
        (root / name).mkdir(parents=True, exist_ok=True)
    samples = []
    for branch_index, fan_branch in enumerate(("single_fan_branch", "four_fan_branch")):
        samples.append(
            sample_fixed_gate_states(
                FixedGateSamplingConfig(
                    total_count=int(sample_count_per_branch),
                    random_seed=20260521 + int(branch_index),
                ),
                fan_branch=fan_branch,
                W_layer="W1",
            )
        )
    sample_frame = pd.concat(samples, ignore_index=True)
    candidates = build_fixed_gate_w0_w1_candidate_rows(
        sample_frame,
        primitive_families=active_mission_primitive_families(),
    )
    fmt = resolve_storage_format(storage_format)
    partitions = (
        write_table_partition(sample_frame, root / "tables" / "fixed_gate_samples" / f"part-00000.{_extension(fmt)}", storage_format=fmt),
        write_table_partition(candidates, root / "tables" / "candidate_index" / f"part-00000.{_extension(fmt)}", storage_format=fmt),
    )
    manifest_path = root / "manifests" / "fixed_gate_w0_w1_archive_manifest.json"
    table_manifest_path = root / "manifests" / "table_manifest.json"
    write_table_manifest(
        table_manifest_path,
        TableManifest(run_id=int(run_id), root=root.as_posix(), storage_format=fmt, tables=partitions),
    )
    branch_counts = candidates.groupby(["fan_branch", "W_layer"]).size().reset_index(name="row_count")
    branch_counts.to_csv(filesystem_path(root / "metrics" / "w0_w1_branch_layer_counts.csv"), index=False)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": int(run_id),
        "campaign": CAMPAIGN,
        "pass_name": "fixed_gate_w0_w1_archive_scaffold",
        "active_mission_path": active_code_path_text(),
        "sample_row_count": int(len(sample_frame)),
        "candidate_row_count": int(len(candidates)),
        "W1_independent_of_W0_success": True,
        "branch_local_separation": True,
        "rollout_execution_performed": False,
        "claim_status": "simulation_only",
        "claim_boundary": (
            "Fixed-gate W0/W1 archive planning scaffold only; no real-flight transfer, "
            "mission success, same-flight recapture, perching, all-arena validity, or "
            "hardware-ready agile claim is made."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    (root / "reports" / "fixed_gate_w0_w1_archive_report.md").write_text(
        "\n".join(
            [
                "# Fixed-Gate W0/W1 Archive Scaffold",
                "",
                f"Active mission path: `{manifest['active_mission_path']}`",
                "",
                f"- Sample rows: `{manifest['sample_row_count']}`",
                f"- Candidate rows: `{manifest['candidate_row_count']}`",
                "- W1 remains independent of W0 success.",
                "- Branch-local separation is preserved.",
                "",
                "No real-flight transfer, mission success, same-flight recapture, perching, all-arena validity, or hardware-ready agile claim is made.",
                "",
            ]
        ),
        encoding="ascii",
    )
    return {
        "root": root,
        "manifest_json": manifest_path,
        "table_manifest_json": table_manifest_path,
        "branch_counts_csv": root / "metrics" / "w0_w1_branch_layer_counts.csv",
    }


def _extension(storage_format: str) -> str:
    if storage_format == "parquet":
        return "parquet"
    if storage_format == "csv":
        return "csv"
    return "csv.gz"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--sample-count-per-branch", type=int, default=20)
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--storage-format", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_fixed_gate_w0_w1_archive_scaffold(
        run_id=args.run_id,
        sample_count_per_branch=args.sample_count_per_branch,
        result_root=args.result_root,
        storage_format=args.storage_format,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
