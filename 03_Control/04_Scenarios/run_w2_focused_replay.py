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


CAMPAIGN = "11_fixed_gate_repeated_launch"
RESULT_ROOT = CONTROL_DIR / "05_Results" / CAMPAIGN


def run_w2_focused_replay(
    *,
    source_csv: Path,
    run_id: int,
    max_cases: int = 200,
    latency_case: str = "nominal",
    seed: int = 20260521,
    result_root: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    return _run_focused_replay(
        source_csv=source_csv,
        run_id=run_id,
        target_W_layer="W2",
        max_cases=max_cases,
        latency_case=latency_case,
        seed=seed,
        result_root=result_root,
        overwrite=overwrite,
        pass_name="w2_focused_replay",
        output_name="w2_focused_replay_rows.csv",
    )


def _run_focused_replay(
    *,
    source_csv: Path,
    run_id: int,
    target_W_layer: str,
    max_cases: int,
    latency_case: str,
    seed: int,
    result_root: Path | None,
    overwrite: bool,
    pass_name: str,
    output_name: str,
) -> dict[str, Path]:
    root = (RESULT_ROOT if result_root is None else Path(result_root)) / f"{int(run_id):03d}" / pass_name
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise RuntimeError(f"result tree already exists: {root}")
    for name in ("metrics", "manifests", "reports"):
        (root / name).mkdir(parents=True, exist_ok=True)
    source_rows = pd.read_csv(source_csv)
    selected = select_focused_replay_cases(source_rows, target_W_layer=target_W_layer, max_cases=max_cases)
    replay_rows = run_fixed_gate_primitive_rollouts(
        selected,
        FixedGatePrimitiveRolloutConfig(
            latency_case=latency_case,
            random_seed=int(seed),
            controller_mode="open_loop_rollout",
            feedback_mode="open_loop",
            allow_open_loop_diagnostic=True,
        ),
    )
    summary = build_rollout_outcome_summary(replay_rows)
    paths = {
        "root": root,
        "selected_cases_csv": root / "metrics" / f"{target_W_layer.lower()}_selected_source_cases.csv",
        "replay_rows_csv": root / "metrics" / output_name,
        "summary_csv": root / "metrics" / f"{target_W_layer.lower()}_pass_fail_summary.csv",
        "manifest_json": root / "manifests" / f"{pass_name}_manifest.json",
        "report_md": root / "reports" / f"{pass_name}_report.md",
    }
    selected.to_csv(paths["selected_cases_csv"], index=False)
    replay_rows.to_csv(paths["replay_rows_csv"], index=False)
    summary.to_csv(paths["summary_csv"], index=False)
    manifest = _manifest(
        run_id=run_id,
        source_csv=source_csv,
        target_W_layer=target_W_layer,
        source_rows=source_rows,
        selected=selected,
        replay_rows=replay_rows,
        latency_case=latency_case,
        pass_name=pass_name,
        paths=paths,
    )
    paths["manifest_json"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    paths["report_md"].write_text(_report(manifest), encoding="ascii")
    return paths


def _manifest(
    *,
    run_id: int,
    source_csv: Path,
    target_W_layer: str,
    source_rows: pd.DataFrame,
    selected: pd.DataFrame,
    replay_rows: pd.DataFrame,
    latency_case: str,
    pass_name: str,
    paths: dict[str, Path],
) -> dict[str, object]:
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": int(run_id),
        "campaign": CAMPAIGN,
        "pass_name": pass_name,
        "active_mission_path": active_code_path_text(),
        "source_csv": str(source_csv),
        "target_W_layer": str(target_W_layer),
        "source_row_count": int(len(source_rows)),
        "selected_row_count": int(len(selected)),
        "replay_row_count": int(len(replay_rows)),
        "latency_case": str(latency_case),
        "dense_all_state_sweep": False,
        "source_policy": "selected_W1_or_medoid_cases_only",
        "mission_candidate_row_count": int(replay_rows["evidence_role"].astype(str).eq("mission_candidate").sum()) if not replay_rows.empty else 0,
        "readiness_status": "ready" if not replay_rows.empty else "blocked_no_selected_W1_or_medoid_rows",
        "claim_status": "simulation_only",
        "claim_boundary": (
            f"{target_W_layer} focused replay consumes selected W1/medoid rows only; no real-flight transfer, "
            "mission success, same-flight recapture, perching, all-arena validity, or hardware-ready agile claim is made."
        ),
        "output_files": {key: str(path) for key, path in paths.items()},
    }


def _report(manifest: dict[str, object]) -> str:
    return "\n".join(
        [
            f"# {manifest['target_W_layer']} Focused Replay",
            "",
            f"Active mission path: `{manifest['active_mission_path']}`",
            "",
            f"- Source rows: `{manifest['source_row_count']}`",
            f"- Selected rows: `{manifest['selected_row_count']}`",
            f"- Replay rows: `{manifest['replay_row_count']}`",
            f"- Dense all-state sweep: `{manifest['dense_all_state_sweep']}`",
            "",
            "Open-loop replay rows remain diagnostic unless delayed-state feedback evidence is present.",
            "No real-flight transfer, mission success, same-flight recapture, perching, all-arena validity, or hardware-ready agile claim is made.",
            "",
        ]
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--max-cases", type=int, default=200)
    parser.add_argument("--latency-case", choices=("none", "actuator_lag_only", "nominal", "conservative"), default="nominal")
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_w2_focused_replay(
        source_csv=args.source_csv,
        run_id=args.run_id,
        max_cases=args.max_cases,
        latency_case=args.latency_case,
        seed=args.seed,
        result_root=args.result_root,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
