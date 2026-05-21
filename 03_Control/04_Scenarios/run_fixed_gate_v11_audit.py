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

from episode_schema import assert_claim_safe_text  # noqa: E402
from fixed_gate_code_path_map import active_code_path_text, code_path_map_frame  # noqa: E402


CAMPAIGN = "11_fixed_gate_repeated_launch"
RESULT_ROOT = CONTROL_DIR / "05_Results" / CAMPAIGN


def run_fixed_gate_v11_audit(
    *,
    run_id: int,
    result_root: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    root = (RESULT_ROOT if result_root is None else Path(result_root)) / f"{int(run_id):03d}" / "001_scaffold_audit"
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise RuntimeError(f"result tree already exists: {root}")
    for name in ("metrics", "manifests", "reports"):
        (root / name).mkdir(parents=True, exist_ok=True)

    code_map = code_path_map_frame()
    readiness = _module_readiness_table()
    paths = {
        "root": root,
        "code_path_map_csv": root / "metrics" / "active_deprecated_code_path_map.csv",
        "module_readiness_csv": root / "metrics" / "module_readiness_table.csv",
        "manifest_json": root / "manifests" / "fixed_gate_v11_audit_manifest.json",
        "report_md": root / "reports" / "fixed_gate_v11_audit_report.md",
    }
    code_map.to_csv(paths["code_path_map_csv"], index=False)
    readiness.to_csv(paths["module_readiness_csv"], index=False)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": int(run_id),
        "campaign": CAMPAIGN,
        "pass_name": "fixed_gate_v11_audit",
        "active_mission_path": active_code_path_text(),
        "readiness_counts": readiness["readiness_status"].value_counts().to_dict(),
        "claim_status": "simulation_only",
        "claim_boundary": (
            "Audit and readiness classification only; no real-flight transfer, mission success, "
            "same-flight recapture, perching, all-arena validity, or hardware-ready agile claim is made."
        ),
        "output_files": {key: str(path) for key, path in paths.items()},
    }
    paths["manifest_json"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    report = _report(manifest, readiness)
    assert_claim_safe_text(report)
    paths["report_md"].write_text(report, encoding="ascii")
    return paths


def _module_readiness_table() -> pd.DataFrame:
    rows = [
        ("fixed_gate_contract.py", "ready_for_evidence", "launch gate admission and margins"),
        ("fixed_gate_sampling.py", "ready_for_evidence", "fixed-gate/local/reachable source labels"),
        ("fixed_gate_primitive_rollout.py", "needs_adapter", "diagnostic open-loop path active; delayed feedback path blocked"),
        ("run_fixed_gate_w0_w1_archive.py", "needs_adapter", "writes evidence rows but archive gate blocks without mission candidates"),
        ("run_reachable_state_extraction.py", "ready_for_evidence", "accepts mission-candidate fixed-gate source rows only"),
        ("primitive_envelope_clustering.py", "ready_for_evidence", "clusters primitive rows before episode clustering"),
        ("run_w2_focused_replay.py", "needs_adapter", "focused replay path present; mission evidence depends on upstream rows"),
        ("run_w3_domain_randomised_replay.py", "needs_adapter", "domain-randomised focused replay path present"),
        ("run_fixed_gate_repeated_launch_policy_eval.py", "scaffold_only", "policy logs require clustering-derived governor package"),
        ("run_real_flight_episode_ingest.py", "ready_for_evidence", "schema-ready real log ingest; transfer not claimed"),
        ("run_matched_real_replay.py", "ready_for_evidence", "matched comparison table with not_tested/partial/negative labels"),
        ("all-arena/all-yaw/agile expansion paths", "diagnostic_only", "retained as boundary evidence only"),
    ]
    return pd.DataFrame(rows, columns=["module_or_path", "readiness_status", "readiness_note"])


def _report(manifest: dict[str, object], readiness: pd.DataFrame) -> str:
    counts = readiness["readiness_status"].value_counts().to_dict()
    return "\n".join(
        [
            "# Fixed-Gate v11 Audit",
            "",
            f"Active mission path: `{manifest['active_mission_path']}`",
            "",
            f"- Readiness counts: `{counts}`",
            "- Open-loop or command-template rows are diagnostic unless a delayed-state feedback primitive path is present.",
            "- Broad all-arena, all-yaw, same-flight recapture, and high-angle agile paths remain diagnostic or boundary-only.",
            "",
            "No real-flight transfer, mission success, same-flight recapture, perching, all-arena validity, or hardware-ready agile claim is made.",
            "",
        ]
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_fixed_gate_v11_audit(
        run_id=args.run_id,
        result_root=args.result_root,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
