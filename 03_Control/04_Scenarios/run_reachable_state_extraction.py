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

from fixed_gate_code_path_map import active_code_path_text  # noqa: E402
from fixed_gate_sampling import (  # noqa: E402
    FixedGateSamplingConfig,
    build_reachable_downstream_states,
)


CAMPAIGN = "11_fixed_gate_repeated_launch"
RESULT_ROOT = CONTROL_DIR / "05_Results" / CAMPAIGN


def run_reachable_state_extraction(
    *,
    rollout_csv: Path,
    run_id: int,
    max_rows: int = 200,
    seed: int = 20260521,
    allow_partial_feedback_source: bool = True,
    allow_diagnostic_source: bool = False,
    result_root: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    root = (RESULT_ROOT if result_root is None else Path(result_root)) / f"{int(run_id):03d}" / "reachable_state_extraction"
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise RuntimeError(f"result tree already exists: {root}")
    for name in ("metrics", "manifests", "reports"):
        (root / name).mkdir(parents=True, exist_ok=True)

    rollouts = pd.read_csv(rollout_csv)
    eligible = _eligible_source_rows(
        rollouts,
        allow_partial_feedback_source=allow_partial_feedback_source,
        allow_diagnostic_source=allow_diagnostic_source,
    )
    frames = []
    for branch, branch_rows in eligible.groupby("fan_branch", sort=True):
        frames.append(
            build_reachable_downstream_states(
                branch_rows,
                max_rows=min(int(max_rows), int(len(branch_rows))),
                config=FixedGateSamplingConfig(total_count=int(max_rows), random_seed=int(seed), sampling_round="reachable_extraction"),
                fan_branch=str(branch),
                W_layer="W1",
            )
        )
    reachable = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    provenance = _provenance_audit(eligible, reachable)
    paths = {
        "root": root,
        "reachable_states_csv": root / "metrics" / "reachable_downstream_states.csv",
        "provenance_audit_csv": root / "metrics" / "reachable_downstream_provenance_audit.csv",
        "manifest_json": root / "manifests" / "reachable_state_extraction_manifest.json",
        "report_md": root / "reports" / "reachable_state_extraction_report.md",
    }
    reachable.to_csv(paths["reachable_states_csv"], index=False)
    provenance.to_csv(paths["provenance_audit_csv"], index=False)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": int(run_id),
        "campaign": CAMPAIGN,
        "pass_name": "reachable_state_extraction",
        "active_mission_path": active_code_path_text(),
        "rollout_csv": str(rollout_csv),
        "source_row_count": int(len(rollouts)),
        "eligible_source_row_count": int(len(eligible)),
        "reachable_downstream_row_count": int(len(reachable)),
        "readiness_status": "ready" if len(reachable) > 0 else "blocked_no_accepted_launch_gate_mission_or_partial_feedback_rows",
        "source_policy": _source_policy(
            allow_partial_feedback_source=allow_partial_feedback_source,
            allow_diagnostic_source=allow_diagnostic_source,
        ),
        "diagnostic_sources_allowed": bool(allow_diagnostic_source),
        "partial_feedback_sources_allowed": bool(allow_partial_feedback_source),
        "claim_status": "simulation_only",
        "claim_boundary": (
            "Reachable states are simulation-derived starts only; no real-flight transfer, mission success, "
            "same-flight recapture, perching, all-arena validity, or hardware-ready agile claim is made."
        ),
        "output_files": {key: str(path) for key, path in paths.items()},
    }
    paths["manifest_json"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    paths["report_md"].write_text(_report(manifest), encoding="ascii")
    return paths


def _eligible_source_rows(
    rollouts: pd.DataFrame,
    *,
    allow_partial_feedback_source: bool,
    allow_diagnostic_source: bool,
) -> pd.DataFrame:
    required = {"accepted", "entry_source", "evidence_role", "fan_branch"}
    missing = sorted(required.difference(rollouts.columns))
    if missing:
        raise ValueError(f"rollout_csv missing required reachable-source columns: {missing}")
    roles = {"mission_candidate"}
    if bool(allow_partial_feedback_source):
        roles.add("partial_feedback")
    if bool(allow_diagnostic_source):
        roles.update({"ablation_diagnostic", "boundary_diagnostic"})
    eligible = rollouts[
        rollouts["accepted"].astype(bool)
        & rollouts["evidence_role"].astype(str).isin(roles)
        & rollouts["entry_source"].astype(str).isin({"launch_gate_main", "launch_gate_tolerance_shell"})
    ].copy()
    if "trial_descriptor_id" not in eligible.columns:
        eligible["trial_descriptor_id"] = eligible.get("primitive_id", eligible.index.astype(str))
    return eligible


def _source_policy(*, allow_partial_feedback_source: bool, allow_diagnostic_source: bool) -> str:
    if allow_diagnostic_source:
        return "accepted_launch_gate_rows_including_explicit_diagnostic_sources"
    if allow_partial_feedback_source:
        return "accepted_mission_or_partial_feedback_launch_gate_rows_only"
    return "accepted_mission_candidate_launch_gate_rows_only"


def _provenance_audit(eligible: pd.DataFrame, reachable: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "eligible_source_row_count": int(len(eligible)),
                "reachable_downstream_row_count": int(len(reachable)),
                "all_sources_accepted": bool(eligible["accepted"].astype(bool).all()) if not eligible.empty else False,
                "all_sources_mission_candidate": bool(eligible["evidence_role"].astype(str).eq("mission_candidate").all()) if not eligible.empty else False,
                "all_sources_mission_or_partial_feedback": bool(eligible["evidence_role"].astype(str).isin({"mission_candidate", "partial_feedback"}).all()) if not eligible.empty else False,
                "arbitrary_arena_sources_used": False,
            }
        ]
    )


def _report(manifest: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Reachable Downstream State Extraction",
            "",
            f"Active mission path: `{manifest['active_mission_path']}`",
            "",
            f"- Eligible accepted launch-gate mission rows: `{manifest['eligible_source_row_count']}`",
            f"- Reachable downstream rows: `{manifest['reachable_downstream_row_count']}`",
            f"- Readiness status: `{manifest['readiness_status']}`",
            "",
            "No arbitrary centre/end-arena starts are used.",
            "No real-flight transfer, mission success, same-flight recapture, perching, all-arena validity, or hardware-ready agile claim is made.",
            "",
        ]
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-csv", type=Path, required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--max-rows", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--mission-only", action="store_true")
    parser.add_argument("--allow-diagnostic-source", action="store_true")
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_reachable_state_extraction(
        rollout_csv=args.rollout_csv,
        run_id=args.run_id,
        max_rows=args.max_rows,
        seed=args.seed,
        allow_partial_feedback_source=not args.mission_only,
        allow_diagnostic_source=args.allow_diagnostic_source,
        result_root=args.result_root,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
