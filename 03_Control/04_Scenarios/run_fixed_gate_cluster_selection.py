from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


SCENARIOS_DIR = Path(__file__).resolve().parent
CONTROL_DIR = SCENARIOS_DIR.parents[0]
REPO_ROOT = CONTROL_DIR.parents[0]
PRIMITIVES_DIR = CONTROL_DIR / "03_Primitives"
for path in (PRIMITIVES_DIR, SCENARIOS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fixed_gate_code_path_map import active_code_path_text, code_path_map_frame
from primitive_envelope_clustering import (
    build_primitive_envelope_clusters,
    write_cluster_feature_scaling,
)


CAMPAIGN = "11_fixed_gate_repeated_launch"
PASS_NAME = "fixed_gate_primitive_envelope_cluster_selection"
RESULT_ROOT = CONTROL_DIR / "05_Results" / CAMPAIGN


def run_fixed_gate_cluster_selection(
    *,
    input_csv: Path,
    run_id: int,
    result_root: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    root = (RESULT_ROOT if result_root is None else Path(result_root)) / f"{int(run_id):03d}"
    paths = _prepare_tree(root, overwrite=overwrite)
    primitive_rows = pd.read_csv(input_csv)
    outputs = build_primitive_envelope_clusters(primitive_rows)

    cluster_input = outputs["cluster_input_table"]
    assignments = outputs["cluster_assignments"]
    medoids = outputs["cluster_medoids"]
    summary = outputs["cluster_summary"]
    package = outputs["governor_candidate_package"]
    scaling = outputs["cluster_feature_scaling"]
    mission_medoids = outputs["mission_medoids"]
    partial_feedback_medoids = outputs["partial_feedback_medoids"]
    diagnostic_medoids = outputs["diagnostic_medoids"]
    rejected_or_blocked_medoids = outputs["rejected_or_blocked_medoids"]

    output_paths = {
        "cluster_input_table_csv": paths["metrics"] / "cluster_input_table.csv",
        "cluster_feature_scaling_json": paths["metrics"] / "cluster_feature_scaling.json",
        "cluster_assignments_csv": paths["metrics"] / "cluster_assignments.csv",
        "cluster_medoids_csv": paths["metrics"] / "cluster_medoids.csv",
        "mission_medoids_csv": paths["metrics"] / "mission_medoids.csv",
        "partial_feedback_medoids_csv": paths["metrics"] / "partial_feedback_medoids.csv",
        "diagnostic_medoids_csv": paths["metrics"] / "diagnostic_medoids.csv",
        "rejected_or_blocked_medoids_csv": paths["metrics"] / "rejected_or_blocked_medoids.csv",
        "medoid_source_rows_csv": paths["metrics"] / "medoid_source_rows.csv",
        "cluster_summary_csv": paths["metrics"] / "cluster_summary.csv",
        "governor_candidate_package_csv": paths["metrics"] / "governor_candidate_package.csv",
        "code_path_map_csv": paths["metrics"] / "active_deprecated_code_path_map.csv",
        "manifest_json": paths["manifests"] / "fixed_gate_cluster_selection_manifest.json",
        "cluster_report_md": paths["reports"] / "cluster_report.md",
    }
    cluster_input.to_csv(output_paths["cluster_input_table_csv"], index=False)
    write_cluster_feature_scaling(output_paths["cluster_feature_scaling_json"], scaling)
    assignments.to_csv(output_paths["cluster_assignments_csv"], index=False)
    medoids.to_csv(output_paths["cluster_medoids_csv"], index=False)
    mission_medoids.to_csv(output_paths["mission_medoids_csv"], index=False)
    partial_feedback_medoids.to_csv(output_paths["partial_feedback_medoids_csv"], index=False)
    diagnostic_medoids.to_csv(output_paths["diagnostic_medoids_csv"], index=False)
    rejected_or_blocked_medoids.to_csv(output_paths["rejected_or_blocked_medoids_csv"], index=False)
    _medoid_source_rows(assignments, medoids).to_csv(output_paths["medoid_source_rows_csv"], index=False)
    summary.to_csv(output_paths["cluster_summary_csv"], index=False)
    package.to_csv(output_paths["governor_candidate_package_csv"], index=False)
    code_path_map_frame().to_csv(output_paths["code_path_map_csv"], index=False)
    manifest = _manifest(run_id, input_csv, outputs, output_paths)
    output_paths["manifest_json"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    output_paths["cluster_report_md"].write_text(_report(manifest), encoding="ascii")
    return output_paths


def _prepare_tree(root: Path, *, overwrite: bool) -> dict[str, Path]:
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise RuntimeError(f"result tree already exists: {root}")
    paths = {
        "root": root,
        "metrics": root / "metrics",
        "manifests": root / "manifests",
        "reports": root / "reports",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _manifest(
    run_id: int,
    input_csv: Path,
    outputs: dict[str, object],
    output_paths: dict[str, Path],
) -> dict[str, object]:
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": int(run_id),
        "campaign": CAMPAIGN,
        "pass_name": PASS_NAME,
        "active_mission_path": active_code_path_text(),
        "input_csv": str(Path(input_csv)),
        "primitive_row_clustering_first": True,
        "whole_episode_clustering_performed": False,
        "whole_episode_clustering_status": "deferred_until_repeated_launch_policy_exists",
        "cluster_input_row_count": int(len(outputs["cluster_input_table"])),
        "cluster_assignment_row_count": int(len(outputs["cluster_assignments"])),
        "cluster_medoid_count": int(len(outputs["cluster_medoids"])),
        "governor_candidate_package_row_count": int(len(outputs["governor_candidate_package"])),
        "mission_medoid_count": int(len(outputs["mission_medoids"])),
        "partial_feedback_medoid_count": int(len(outputs["partial_feedback_medoids"])),
        "diagnostic_medoid_count": int(len(outputs["diagnostic_medoids"])),
        "rejected_or_blocked_medoid_count": int(len(outputs["rejected_or_blocked_medoids"])),
        "governor_candidate_package_source_policy": "mission_or_approved_partial_feedback_medoids_only",
        "diagnostic_rows_can_enter_candidate_package": False,
        "claim_status": "simulation_only",
        "claim_boundary": (
            "Primitive-envelope medoid selection only; clustering does not replace "
            "the governor and does not claim real-flight transfer, mission success, "
            "same-flight recapture, perching, all-arena validity, or hardware-ready agile turns."
        ),
        "output_files": {name: str(path) for name, path in output_paths.items()},
    }


def _report(manifest: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Fixed-Gate Primitive-Envelope Cluster Report",
            "",
            f"Active mission path: `{manifest['active_mission_path']}`",
            "",
            "Primitive rollout rows were clustered before any repeated-launch episode clustering.",
            "Whole-episode clustering is deferred until repeated-launch policy logs exist.",
            "",
            f"- Input rows: `{manifest['cluster_input_row_count']}`",
            f"- Medoids: `{manifest['cluster_medoid_count']}`",
            f"- Mission medoids: `{manifest['mission_medoid_count']}`",
            f"- Partial-feedback medoids: `{manifest['partial_feedback_medoid_count']}`",
            f"- Diagnostic medoids: `{manifest['diagnostic_medoid_count']}`",
            f"- Governor candidate rows: `{manifest['governor_candidate_package_row_count']}`",
            "- Open-loop and command-template medoids are retained as diagnostics only and cannot enter the governor candidate package.",
            "",
            "No real-flight transfer, mission success, same-flight recapture, perching, all-arena validity, or hardware-ready agile claim is made.",
            "",
        ]
    )


def _medoid_source_rows(assignments: pd.DataFrame, medoids: pd.DataFrame) -> pd.DataFrame:
    if assignments.empty or medoids.empty or "sample_id" not in assignments.columns:
        return pd.DataFrame(columns=assignments.columns)
    medoid_ids = set(medoids["medoid_sample_id"].astype(str))
    source = assignments[assignments["sample_id"].astype(str).isin(medoid_ids)].copy()
    source["is_medoid"] = True
    return source.reset_index(drop=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_fixed_gate_cluster_selection(
        input_csv=args.input_csv,
        run_id=args.run_id,
        result_root=args.result_root,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
