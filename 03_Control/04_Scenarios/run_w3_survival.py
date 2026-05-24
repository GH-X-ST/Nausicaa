from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


def _bootstrap_import_paths() -> None:
    control_root = Path(__file__).resolve().parents[1]
    for rel in ("03_Primitives", "04_Scenarios"):
        path = control_root / rel
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


_bootstrap_import_paths()

from dense_archive_table_io import filesystem_path  # noqa: E402


W3_SURVIVAL_VERSION = "w3_fixed_lqr_survival_replay_scaffold_v1"
DEFAULT_OUTPUT_ROOT = Path("03_Control/05_Results/lqr_contextual_v1_0/w3_survival")
BLOCKED_CLAIMS = (
    "W3_robustness_complete",
    "post_W3_compact_library_ready",
    "governor_validation",
    "hardware_readiness",
    "real_flight_transfer",
    "mission_success",
)


@dataclass(frozen=True)
class W3SurvivalConfig:
    run_id: int
    input_root: Path
    output_root: Path = DEFAULT_OUTPUT_ROOT


def run_w3_survival(config: W3SurvivalConfig) -> dict[str, object]:
    """Create a W3 fixed-LQR replay scaffold from W2 survivors."""

    run_root = Path(config.output_root) / f"{int(config.run_id):03d}"
    for subdir in ("manifests", "metrics", "reports"):
        filesystem_path(run_root / subdir).mkdir(parents=True, exist_ok=True)
    w2_manifest = Path(config.input_root) / "manifests" / "w2_survival_manifest.json"
    blocked_reason = ""
    if not filesystem_path(w2_manifest).is_file():
        blocked_reason = "missing_W2_survival_manifest"
    else:
        source = json.loads(filesystem_path(w2_manifest).read_text(encoding="ascii"))
        if source.get("status") != "survived_variants_available":
            blocked_reason = "missing_W2_surviving_variants"
    status = "blocked" if blocked_reason else "ready_for_fixed_lqr_w3_replay"
    manifest = {
        "version": W3_SURVIVAL_VERSION,
        "status": status,
        "run_id": int(config.run_id),
        "input_root": Path(config.input_root).as_posix(),
        "fixed_lqr_replay_only": True,
        "mutates_Q_R_K_reference_horizon_entry_set_or_entry_role": False,
        "redesign_policy": "new_ids_return_to_W01",
        "blocked_reason": blocked_reason,
        "blocked_claims": list(BLOCKED_CLAIMS),
    }
    _write_json(run_root / "manifests" / "w3_survival_manifest.json", manifest)
    pd.DataFrame([{"status": status, "blocked_reason": blocked_reason}]).to_csv(
        filesystem_path(run_root / "metrics" / "w3_survival_summary.csv"),
        index=False,
    )
    _write_report(run_root / "reports" / "w3_survival_report.md", "W3 Survival", manifest)
    return {"status": status, "run_root": run_root.as_posix(), "manifest": (run_root / "manifests" / "w3_survival_manifest.json").as_posix()}


def _write_json(path: Path, payload: dict[str, object]) -> None:
    filesystem_path(path).parent.mkdir(parents=True, exist_ok=True)
    filesystem_path(path).write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="ascii")


def _write_report(path: Path, title: str, manifest: dict[str, object]) -> None:
    lines = [
        f"# {title}",
        "",
        f"- Status: `{manifest['status']}`",
        "- Fixed-LQR replay only: `True`",
        "- Q/R, K, reference, horizon, entry set, and entry role mutation: `False`",
        "- Any redesign receives new IDs and returns to W01.",
        "",
    ]
    filesystem_path(path).write_text("\n".join(lines), encoding="ascii")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare W3 fixed-LQR survival replay.")
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_w3_survival(W3SurvivalConfig(run_id=args.run_id, input_root=args.input_root, output_root=args.output_root))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
