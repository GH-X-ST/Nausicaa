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
from fixed_gate_primitive_rollout import (  # noqa: E402
    FixedGatePrimitiveRolloutConfig,
    build_rollout_outcome_summary,
    run_fixed_gate_primitive_rollouts,
)
from fixed_gate_sampling import select_focused_replay_cases  # noqa: E402
from fixed_gate_table_sources import read_fixed_gate_table_source  # noqa: E402


CAMPAIGN = "11_fixed_gate_repeated_launch"
RESULT_ROOT = CONTROL_DIR / "05_Results" / CAMPAIGN


def run_w3_domain_randomised_replay(
    *,
    source_csv: Path,
    run_id: int,
    max_cases: int = 200,
    latency_case: str = "conservative",
    seed: int = 20260521,
    allow_diagnostic_source: bool = False,
    result_root: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    root = (RESULT_ROOT if result_root is None else Path(result_root)) / f"{int(run_id):03d}" / "w3_domain_randomised_replay"
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise RuntimeError(f"result tree already exists: {root}")
    for name in ("metrics", "manifests", "reports"):
        (root / name).mkdir(parents=True, exist_ok=True)
    source_rows = read_fixed_gate_table_source(source_csv, table_name="primitive_rollout_rows")
    selected = select_focused_replay_cases(
        source_rows,
        target_W_layer="W3",
        max_cases=max_cases,
        allow_diagnostic_source=allow_diagnostic_source,
    )
    selected["domain_randomisation_requested"] = True
    selected["domain_randomisation_status"] = "blocked_until_feedback_replay_path_available"
    replay_rows = run_fixed_gate_primitive_rollouts(
        selected,
        FixedGatePrimitiveRolloutConfig(
            latency_case=latency_case,
            random_seed=int(seed),
            controller_mode="open_loop_rollout" if allow_diagnostic_source else "feedback_stabilised_primitive",
            feedback_mode="open_loop" if allow_diagnostic_source else "instant_state_feedback",
            allow_open_loop_diagnostic=bool(allow_diagnostic_source),
        ),
    )
    summary = build_rollout_outcome_summary(replay_rows)
    paths = {
        "root": root,
        "selected_cases_csv": root / "metrics" / "w3_selected_source_cases.csv",
        "replay_rows_csv": root / "metrics" / "w3_domain_randomised_replay_rows.csv",
        "summary_csv": root / "metrics" / "w3_pass_fail_summary.csv",
        "manifest_json": root / "manifests" / "w3_domain_randomised_replay_manifest.json",
        "report_md": root / "reports" / "w3_domain_randomised_replay_report.md",
    }
    selected.to_csv(paths["selected_cases_csv"], index=False)
    replay_rows.to_csv(paths["replay_rows_csv"], index=False)
    summary.to_csv(paths["summary_csv"], index=False)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": int(run_id),
        "campaign": CAMPAIGN,
        "pass_name": "w3_domain_randomised_replay",
        "active_mission_path": active_code_path_text(),
        "source_csv": str(source_csv),
        "target_W_layer": "W3",
        "source_row_count": int(len(source_rows)),
        "selected_row_count": int(len(selected)),
        "replay_row_count": int(len(replay_rows)),
        "latency_case": str(latency_case),
        "dense_all_state_sweep": False,
        "source_policy": "selected_W1_or_W2_medoid_cases_only",
        "diagnostic_sources_allowed": bool(allow_diagnostic_source),
        "domain_randomisation_requested": True,
        "domain_randomisation_status": "blocked_until_feedback_replay_path_available",
        "mission_candidate_row_count": int(replay_rows["evidence_role"].astype(str).eq("mission_candidate").sum()) if not replay_rows.empty else 0,
        "partial_feedback_row_count": int(replay_rows["evidence_role"].astype(str).eq("partial_feedback").sum()) if not replay_rows.empty else 0,
        "blocked_partial_row_count": int(replay_rows["evidence_role"].astype(str).eq("blocked_partial").sum()) if not replay_rows.empty else 0,
        "readiness_status": "ready" if _mission_or_partial_count(replay_rows) > 0 else _blocked_status(selected, replay_rows, allow_diagnostic_source),
        "claim_status": "simulation_only",
        "claim_boundary": (
            "W3 replay consumes selected W1/W2/medoid rows only; no real-flight transfer, mission success, "
            "same-flight recapture, perching, all-arena validity, or hardware-ready agile claim is made."
        ),
        "output_files": {key: str(path) for key, path in paths.items()},
    }
    paths["manifest_json"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    paths["report_md"].write_text(_report(manifest), encoding="ascii")
    return paths


def _report(manifest: dict[str, object]) -> str:
    return "\n".join(
        [
            "# W3 Domain-Randomised Focused Replay",
            "",
            f"Active mission path: `{manifest['active_mission_path']}`",
            "",
            f"- Source rows: `{manifest['source_row_count']}`",
            f"- Selected rows: `{manifest['selected_row_count']}`",
            f"- Replay rows: `{manifest['replay_row_count']}`",
            f"- Dense all-state sweep: `{manifest['dense_all_state_sweep']}`",
            f"- Domain-randomisation status: `{manifest['domain_randomisation_status']}`",
            "",
            "Open-loop replay rows remain diagnostic unless mission or approved partial-feedback evidence is present.",
            "No real-flight transfer, mission success, same-flight recapture, perching, all-arena validity, or hardware-ready agile claim is made.",
            "",
        ]
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--max-cases", type=int, default=200)
    parser.add_argument("--latency-case", choices=("none", "actuator_lag_only", "nominal", "conservative"), default="conservative")
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--allow-diagnostic-source", action="store_true")
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_w3_domain_randomised_replay(
        source_csv=args.source_csv,
        run_id=args.run_id,
        max_cases=args.max_cases,
        latency_case=args.latency_case,
        seed=args.seed,
        allow_diagnostic_source=args.allow_diagnostic_source,
        result_root=args.result_root,
        overwrite=args.overwrite,
    )
    return 0


def _mission_or_partial_count(frame: pd.DataFrame) -> int:
    if frame.empty or "evidence_role" not in frame.columns:
        return 0
    roles = frame["evidence_role"].astype(str).isin({"mission_candidate", "partial_feedback"})
    accepted = frame["accepted"].astype(bool) if "accepted" in frame.columns else True
    return int((roles & accepted).sum())


def _blocked_status(selected: pd.DataFrame, replay_rows: pd.DataFrame, allow_diagnostic_source: bool) -> str:
    if selected.empty:
        return "blocked_no_selected_W1_W2_or_medoid_rows"
    if allow_diagnostic_source:
        return "diagnostic_only_replay_not_mission_evidence"
    if not replay_rows.empty and replay_rows["evidence_role"].astype(str).eq("blocked_partial").any():
        return "blocked_feedback_replay_unavailable_for_selected_rows"
    return "blocked_no_mission_or_partial_feedback_replay_rows"


if __name__ == "__main__":
    raise SystemExit(main())
