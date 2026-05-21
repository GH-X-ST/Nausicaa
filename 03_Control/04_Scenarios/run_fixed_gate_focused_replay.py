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

from fixed_gate_code_path_map import active_code_path_text
from fixed_gate_sampling import select_focused_replay_cases


CAMPAIGN = "11_fixed_gate_repeated_launch"
RESULT_ROOT = CONTROL_DIR / "05_Results" / CAMPAIGN


def run_fixed_gate_focused_replay_selection(
    *,
    source_csv: Path,
    target_W_layer: str,
    run_id: int,
    max_cases: int = 200,
    result_root: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    root = (RESULT_ROOT if result_root is None else Path(result_root)) / f"{int(run_id):03d}" / f"{target_W_layer.lower()}_focused_replay"
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise RuntimeError(f"result tree already exists: {root}")
    for name in ("metrics", "manifests", "reports"):
        (root / name).mkdir(parents=True, exist_ok=True)
    source_rows = pd.read_csv(source_csv)
    selected = select_focused_replay_cases(source_rows, target_W_layer=target_W_layer, max_cases=max_cases)
    selected_csv = root / "metrics" / f"{target_W_layer.lower()}_focused_replay_cases.csv"
    manifest_json = root / "manifests" / f"{target_W_layer.lower()}_focused_replay_manifest.json"
    report_md = root / "reports" / f"{target_W_layer.lower()}_focused_replay_report.md"
    selected.to_csv(selected_csv, index=False)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": int(run_id),
        "campaign": CAMPAIGN,
        "pass_name": f"{target_W_layer}_focused_replay_selection",
        "active_mission_path": active_code_path_text(),
        "source_csv": str(source_csv),
        "target_W_layer": str(target_W_layer),
        "source_row_count": int(len(source_rows)),
        "selected_row_count": int(len(selected)),
        "dense_all_state_sweep": False,
        "source_policy": "selected_W1_or_medoid_cases_only",
        "claim_status": "simulation_only",
        "claim_boundary": (
            "Focused replay selection only; W2/W3 consume selected W1 or medoid cases "
            "and do not claim real-flight transfer, mission success, same-flight recapture, "
            "perching, all-arena validity, or hardware-ready agile turns."
        ),
    }
    manifest_json.write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    report_md.write_text(
        "\n".join(
            [
                f"# {target_W_layer} Focused Replay Selection",
                "",
                f"Active mission path: `{manifest['active_mission_path']}`",
                "",
                f"- Source rows: `{manifest['source_row_count']}`",
                f"- Selected rows: `{manifest['selected_row_count']}`",
                "- Dense all-state sweep: `False`",
                "",
                "No real-flight transfer, mission success, same-flight recapture, perching, all-arena validity, or hardware-ready agile claim is made.",
                "",
            ]
        ),
        encoding="ascii",
    )
    return {
        "root": root,
        "selected_csv": selected_csv,
        "manifest_json": manifest_json,
        "report_md": report_md,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--target-W-layer", choices=("W2", "W3"), required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--max-cases", type=int, default=200)
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_fixed_gate_focused_replay_selection(
        source_csv=args.source_csv,
        target_W_layer=args.target_W_layer,
        run_id=args.run_id,
        max_cases=args.max_cases,
        result_root=args.result_root,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
