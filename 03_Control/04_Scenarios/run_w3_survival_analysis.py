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

from dense_archive_runtime import MAX_GENERATED_FILE_SIZE_MB  # noqa: E402
from dense_archive_table_io import (  # noqa: E402
    file_sha256,
    filesystem_path,
    load_table_manifest,
    read_table_partition,
)
from primitive_timing_contract import primitive_timing_contract_row  # noqa: E402


PROJECT_TITLE_VERSION = "LQR-Stabilised Contextual Primitive v4.11"
W3_ANALYSIS_VERSION = "w3_variant_survival_analysis_v411"
DEFAULT_W3_ROOT = Path("03_Control/05_Results/lqr_contextual_v1_0/w3_survival/013")
W3_ENVIRONMENT_MODES = ("w3_randomised_single", "w3_randomised_four")
STATUS_VOCABULARY = ("survived", "downgraded", "eliminated", "blocked", "not_run")
BLOCKED_CLAIMS = (
    "W3_robustness_proof",
    "hardware_readiness",
    "real_flight_transfer",
    "mission_success",
    "full_loop_validation_success",
    "memory_governor_performance_improvement",
    "formal_LQR_tree_funnel_region_of_attraction",
)


@dataclass(frozen=True)
class W3SurvivalAnalysisConfig:
    input_root: Path = DEFAULT_W3_ROOT
    survived_hard_failure_rate_max: float = 0.55
    downgraded_hard_failure_rate_max: float = 0.75


def run_w3_survival_analysis(config: W3SurvivalAnalysisConfig) -> dict[str, object]:
    """Build variant-level W3 survival labels from fixed-replay row evidence."""

    input_root = Path(config.input_root)
    blocked_reason = _input_blocked_reason(input_root)
    if blocked_reason:
        return {
            "status": "blocked",
            "input_root": input_root.as_posix(),
            "blocked_reason": blocked_reason,
        }

    for subdir in ("manifests", "metrics", "reports"):
        filesystem_path(input_root / subdir).mkdir(parents=True, exist_ok=True)

    frame = _read_w3_rows(input_root)
    if frame.empty:
        return {
            "status": "blocked",
            "input_root": input_root.as_posix(),
            "blocked_reason": "empty_W3_table",
        }

    variant_summary = _variant_summary(frame, config=config)
    environment_summary = _environment_summary(frame)
    boundary_summary = _count_summary(frame, ["boundary_use_class"])
    failure_summary = _count_summary(frame, ["failure_label", "outcome_class"])
    registry = _survivor_registry(
        input_root=input_root,
        variant_summary=variant_summary,
        config=config,
    )

    _write_csv(input_root / "metrics" / "w3_variant_survival_summary.csv", variant_summary)
    _write_csv(input_root / "metrics" / "w3_environment_variant_summary.csv", environment_summary)
    _write_csv(input_root / "metrics" / "w3_boundary_use_summary.csv", boundary_summary)
    _write_csv(input_root / "metrics" / "w3_failure_summary.csv", failure_summary)
    _write_json(input_root / "manifests" / "w3_survivor_registry.json", registry)
    _write_reports(input_root=input_root, registry=registry)
    _write_file_size_audit(input_root)
    return {
        "status": str(registry["status"]),
        "input_root": input_root.as_posix(),
        "survivor_registry": (input_root / "manifests" / "w3_survivor_registry.json").as_posix(),
        "survivor_count": int(registry["survivor_count"]),
        "downgraded_count": int(registry["downgraded_count"]),
        "eliminated_count": int(registry["eliminated_count"]),
    }


def _input_blocked_reason(input_root: Path) -> str:
    manifest_path = filesystem_path(input_root / "manifests" / "w3_survival_manifest.json")
    table_manifest_path = filesystem_path(input_root / "manifests" / "table_manifest.json")
    if not manifest_path.is_file():
        return "missing_w3_survival_manifest"
    if not table_manifest_path.is_file():
        return "missing_w3_table_manifest"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    except Exception as exc:
        return f"unreadable_w3_survival_manifest:{type(exc).__name__}"
    if str(manifest.get("status", "")) != "complete":
        return "w3_survival_root_not_complete"
    if "w3_survival" not in input_root.as_posix():
        return "input_root_is_not_w3_survival_root"
    return ""


def _read_w3_rows(input_root: Path) -> pd.DataFrame:
    manifest = load_table_manifest(input_root / "manifests" / "table_manifest.json")
    frames = []
    for partition in manifest.tables:
        frame = read_table_partition(
            input_root / "tables" / partition.relative_path,
            storage_format=partition.storage_format,
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _variant_summary(frame: pd.DataFrame, *, config: W3SurvivalAnalysisConfig) -> pd.DataFrame:
    rows = []
    for variant_id, group in frame.groupby("primitive_variant_id", sort=True):
        first = group.iloc[0]
        compatible = group[_bool_series(group.get("entry_role_compatible", False))]
        incompatible_count = int(len(group) - len(compatible))
        if compatible.empty:
            row = _base_variant_row(first, variant_id)
            row.update(
                {
                    "w3_variant_status": "not_run",
                    "eligible_for_post_w3_library_size_study": False,
                    "compatible_row_count": 0,
                    "incompatible_row_count": incompatible_count,
                    "continuation_valid_count": 0,
                    "episode_terminal_useful_count": 0,
                    "hard_failure_count": 0,
                    "blocked_count": int(_blocked_count(group)),
                    "positive_continuation_count": 0,
                    "hard_failure_rate": 0.0,
                    "continuation_valid_rate": 0.0,
                    "episode_terminal_useful_rate": 0.0,
                    "w3_environment_modes_seen": "",
                    "status_reason": "no_entry_role_compatible_W3_rows",
                }
            )
            rows.append(row)
            continue

        positive = _positive_series(compatible)
        terminal = _terminal_useful_series(compatible)
        hard = _hard_failure_series(compatible)
        blocked = _blocked_series(compatible)
        modes = tuple(sorted(str(value) for value in compatible["environment_mode"].dropna().unique()))
        mode_positive_frame = pd.DataFrame(
            {
                "environment_mode": compatible["environment_mode"].astype(str).to_numpy(),
                "_positive": positive.to_numpy(dtype=bool),
            }
        )
        mode_positive = mode_positive_frame.groupby("environment_mode")["_positive"].sum()
        both_modes_positive = all(int(mode_positive.get(mode, 0)) > 0 for mode in W3_ENVIRONMENT_MODES)
        hard_rate = float(int(hard.sum()) / max(1, len(compatible)))
        useful_count = int(positive.sum()) + int(terminal.sum())
        status, reason = _status_for_variant(
            both_modes_positive=both_modes_positive,
            useful_count=useful_count,
            compatible_row_count=len(compatible),
            hard_failure_rate=hard_rate,
            config=config,
        )
        row = _base_variant_row(first, variant_id)
        row.update(
            {
                "w3_variant_status": status,
                "eligible_for_post_w3_library_size_study": status == "survived",
                "compatible_row_count": int(len(compatible)),
                "incompatible_row_count": incompatible_count,
                "continuation_valid_count": int(_continuation_series(compatible).sum()),
                "episode_terminal_useful_count": int(terminal.sum()),
                "hard_failure_count": int(hard.sum()),
                "blocked_count": int(blocked.sum()),
                "positive_continuation_count": int(positive.sum()),
                "accepted_count": int((compatible["outcome_class"].astype(str) == "accepted").sum()),
                "continuation_valid_rate": float(_continuation_series(compatible).sum() / max(1, len(compatible))),
                "episode_terminal_useful_rate": float(terminal.sum() / max(1, len(compatible))),
                "hard_failure_rate": hard_rate,
                "minimum_wall_margin_min_m": _safe_min(compatible, "minimum_wall_margin_m"),
                "floor_margin_min_m": _safe_min(compatible, "floor_margin_m"),
                "ceiling_margin_min_m": _safe_min(compatible, "ceiling_margin_m"),
                "energy_residual_mean_m": _safe_mean(compatible, "energy_residual_m"),
                "lift_dwell_mean_s": _safe_mean(compatible, "lift_dwell_time_s"),
                "saturation_fraction_mean": _safe_mean(compatible, "saturation_fraction"),
                "w3_environment_modes_seen": ";".join(modes),
                "w3_environment_mode_count": int(len(modes)),
                "both_w3_modes_positive": bool(both_modes_positive),
                "status_reason": reason,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["primitive_id", "candidate_index", "primitive_variant_id"]).reset_index(drop=True)


def _base_variant_row(first: pd.Series, variant_id: str) -> dict[str, object]:
    _ = primitive_timing_contract_row()
    return {
        "primitive_variant_id": str(variant_id),
        "primitive_id": str(first.get("primitive_id", first.get("variant_primitive_id", ""))),
        "entry_role": str(first.get("entry_role", first.get("variant_entry_role", ""))),
        "controller_id": str(first.get("controller_id", first.get("variant_controller_id", ""))),
        "candidate_index": int(float(first.get("candidate_index", first.get("variant_candidate_index", 0)))),
        "candidate_weight_label": str(first.get("candidate_weight_label", first.get("variant_candidate_weight_label", ""))),
        "K_gain_checksum": str(first.get("variant_K_gain_checksum", first.get("lqr_gain_checksum", ""))),
        "augmented_A_checksum": str(first.get("variant_augmented_A_checksum", first.get("augmented_A_checksum", ""))),
        "augmented_B_checksum": str(first.get("variant_augmented_B_checksum", first.get("augmented_B_checksum", ""))),
        "augmented_gain_checksum": str(first.get("variant_augmented_gain_checksum", first.get("augmented_gain_checksum", ""))),
        "Q_weight_json": str(first.get("variant_Q_weight_json", first.get("lqr_Q_weights_json", ""))),
        "R_weight_json": str(first.get("variant_R_weight_json", first.get("lqr_R_weights_json", ""))),
        "reference_state_vector": str(first.get("variant_reference_state_vector", "")),
        "reference_command_vector": str(first.get("variant_reference_command_vector", "")),
        "finite_horizon_s": float(first.get("variant_finite_horizon_s", 0.0)),
        "controller_input_slots_per_primitive": int(
            float(
                first.get(
                    "variant_controller_input_slots_per_primitive",
                    first.get("controller_input_slots_per_primitive", 0),
                )
            )
        ),
        "controller_input_update_period_s": float(
            first.get(
                "variant_controller_input_update_period_s",
                first.get("controller_input_update_period_s", 0.0),
            )
        ),
        "primitive_timing_contract_version": str(
            first.get(
                "variant_primitive_timing_contract_version",
                first.get("primitive_timing_contract_version", "legacy_not_recorded"),
            )
        ),
        "timing_augmentation_type": str(first.get("variant_timing_augmentation_type", first.get("timing_augmentation_type", ""))),
    }


def _status_for_variant(
    *,
    both_modes_positive: bool,
    useful_count: int,
    compatible_row_count: int,
    hard_failure_rate: float,
    config: W3SurvivalAnalysisConfig,
) -> tuple[str, str]:
    if compatible_row_count <= 0:
        return "not_run", "no_compatible_rows"
    if both_modes_positive and hard_failure_rate <= float(config.survived_hard_failure_rate_max):
        return "survived", "both_W3_modes_positive_and_hard_failure_rate_within_survival_limit"
    if useful_count > 0 and hard_failure_rate <= float(config.downgraded_hard_failure_rate_max):
        return "downgraded", "useful_but_partial_or_terminal_W3_evidence"
    return "eliminated", "no_useful_evidence_or_hard_failures_dominate"


def _environment_summary(frame: pd.DataFrame) -> pd.DataFrame:
    compatible = frame[_bool_series(frame.get("entry_role_compatible", False))].copy()
    if compatible.empty:
        return pd.DataFrame()
    compatible["_positive_continuation"] = _positive_series(compatible)
    compatible["_terminal_useful"] = _terminal_useful_series(compatible)
    compatible["_hard_failure"] = _hard_failure_series(compatible)
    return (
        compatible.groupby(["primitive_variant_id", "primitive_id", "environment_mode"], dropna=False)
        .agg(
            compatible_row_count=("primitive_variant_id", "size"),
            positive_continuation_count=("_positive_continuation", "sum"),
            episode_terminal_useful_count=("_terminal_useful", "sum"),
            hard_failure_count=("_hard_failure", "sum"),
            minimum_wall_margin_min_m=("minimum_wall_margin_m", "min"),
            energy_residual_mean_m=("energy_residual_m", "mean"),
            lift_dwell_mean_s=("lift_dwell_time_s", "mean"),
        )
        .reset_index()
    )


def _count_summary(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        if column not in frame.columns:
            continue
        for value, count in frame[column].fillna("").astype(str).value_counts().sort_index().items():
            rows.append({"summary_axis": column, "value": value, "row_count": int(count)})
    return pd.DataFrame(rows)


def _survivor_registry(
    *,
    input_root: Path,
    variant_summary: pd.DataFrame,
    config: W3SurvivalAnalysisConfig,
) -> dict[str, object]:
    survivors = variant_summary[variant_summary["w3_variant_status"] == "survived"]
    downgraded = variant_summary[variant_summary["w3_variant_status"] == "downgraded"]
    eliminated = variant_summary[variant_summary["w3_variant_status"] == "eliminated"]
    blocked = variant_summary[variant_summary["w3_variant_status"] == "blocked"]
    source_manifest = json.loads(
        filesystem_path(input_root / "manifests" / "w3_survival_manifest.json").read_text(encoding="ascii")
    )
    source_w2_root = Path(str(source_manifest.get("input_root", "")))
    source_w01_root = _source_w01_from_w2_root(source_w2_root)
    status_rule = {
        "compatible_rows_only": True,
        "positive_continuation": "continuation_valid_true_or_outcome_class_accepted",
        "terminal_useful": "episode_terminal_useful_true_or_boundary_use_class_episode_terminal_useful",
        "hard_failure": "outcome_class_failed_or_boundary_use_class_hard_failure",
        "survived": "both_W3_modes_positive_and_hard_failure_rate_less_or_equal_survived_limit",
        "downgraded": "useful_terminal_or_partial_continuation_and_hard_failure_rate_less_or_equal_downgraded_limit",
        "survived_hard_failure_rate_max": float(config.survived_hard_failure_rate_max),
        "downgraded_hard_failure_rate_max": float(config.downgraded_hard_failure_rate_max),
    }
    return {
        "registry_version": W3_ANALYSIS_VERSION,
        "project_title_version": PROJECT_TITLE_VERSION,
        "source_w3_root": input_root.as_posix(),
        "source_w2_root": source_w2_root.as_posix(),
        "source_w01_root": source_w01_root.as_posix(),
        "source_table_manifest_sha256": file_sha256(input_root / "manifests" / "table_manifest.json"),
        "status": "w3_survivors_available" if len(survivors) else "blocked_no_w3_survivors",
        "survivor_count": int(len(survivors)),
        "downgraded_count": int(len(downgraded)),
        "eliminated_count": int(len(eliminated)),
        "blocked_count": int(len(blocked)),
        "not_run_count": int((variant_summary["w3_variant_status"] == "not_run").sum()),
        "status_vocabulary": list(STATUS_VOCABULARY),
        "status_rule": status_rule,
        "claim_status": "simulation_only_W3_variant_survival_analysis",
        "blocked_claims": list(BLOCKED_CLAIMS),
        "survivors": _records_for_registry(survivors),
        "downgraded": _records_for_registry(downgraded),
        "eliminated": _records_for_registry(eliminated),
    }


def _records_for_registry(frame: pd.DataFrame) -> list[dict[str, object]]:
    wanted = [
        "primitive_variant_id",
        "primitive_id",
        "entry_role",
        "controller_id",
        "candidate_index",
        "candidate_weight_label",
        "w3_variant_status",
        "eligible_for_post_w3_library_size_study",
        "continuation_valid_count",
        "episode_terminal_useful_count",
        "hard_failure_count",
        "blocked_count",
        "compatible_row_count",
        "incompatible_row_count",
        "w3_environment_modes_seen",
        "K_gain_checksum",
        "augmented_A_checksum",
        "augmented_B_checksum",
        "augmented_gain_checksum",
        "Q_weight_json",
        "R_weight_json",
        "reference_state_vector",
        "reference_command_vector",
        "finite_horizon_s",
        "controller_input_slots_per_primitive",
        "controller_input_update_period_s",
        "primitive_timing_contract_version",
        "timing_augmentation_type",
        "hard_failure_rate",
        "continuation_valid_rate",
        "episode_terminal_useful_rate",
        "minimum_wall_margin_min_m",
        "energy_residual_mean_m",
        "lift_dwell_mean_s",
    ]
    records = []
    for row in frame.to_dict("records"):
        records.append({key: _json_scalar(row.get(key, "")) for key in wanted})
    return records


def _source_w01_from_w2_root(source_w2_root: Path) -> Path:
    path = filesystem_path(source_w2_root / "manifests" / "w2_survival_manifest.json")
    if not path.is_file():
        return Path("")
    try:
        payload = json.loads(path.read_text(encoding="ascii"))
    except Exception:
        return Path("")
    return Path(str(payload.get("source_w01_root", "")))


def _write_reports(*, input_root: Path, registry: dict[str, object]) -> None:
    lines = [
        "# W3 Variant Survival Report",
        "",
        f"- Project title version: `{PROJECT_TITLE_VERSION}`",
        f"- Source W3 root: `{registry['source_w3_root']}`",
        f"- Status: `{registry['status']}`",
        f"- Survivors: `{registry['survivor_count']}`",
        f"- Downgraded: `{registry['downgraded_count']}`",
        f"- Eliminated: `{registry['eliminated_count']}`",
        "- Entry-role-incompatible rows are preserved as rejection/block evidence and excluded from survival scoring.",
        "- Continuation-valid evidence and terminal-useful evidence remain separate.",
        "- Claim boundary: simulation-only W3 variant analysis; no hardware, real-flight, mission, or formal ROA claim.",
        "",
    ]
    report = "\n".join(lines)
    filesystem_path(input_root / "reports" / "w3_variant_survival_report.md").write_text(report, encoding="ascii")
    move_on = [
        "# L9 W3 Move-On Check",
        "",
        f"- W3 survivor registry exists: `{True}`",
        f"- W3 survivors available: `{int(registry['survivor_count']) > 0}`",
        f"- Post-W3 clustering allowed: `{int(registry['survivor_count']) > 0}`",
        "- No Q/R, K, reference, horizon, entry role, controller ID, or primitive-variant ID mutation occurred.",
        "- Blocked claims remain hardware readiness, transfer, mission success, and formal LQR-tree/funnel/ROA guarantees.",
        "",
    ]
    filesystem_path(input_root / "reports" / "l9_w3_move_on_check.md").write_text("\n".join(move_on), encoding="ascii")


def _write_file_size_audit(root: Path) -> None:
    rows = []
    root_fs = filesystem_path(root)
    for path in sorted(root_fs.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root_fs).as_posix()
        byte_count = int(path.stat().st_size)
        size_mb = float(byte_count) / float(1024 * 1024)
        rows.append(
            {
                "relative_path": rel,
                "byte_count": byte_count,
                "size_mb": size_mb,
                "above_75mb": bool(size_mb > 75.0),
                "above_100mb": bool(size_mb > MAX_GENERATED_FILE_SIZE_MB),
                "push_allowed": bool(size_mb <= MAX_GENERATED_FILE_SIZE_MB),
                "dense_table_partition": rel.startswith("tables/"),
            }
        )
    _write_csv(root / "metrics" / "file_size_audit.csv", pd.DataFrame(rows))


def _bool_series(values) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.fillna(False).astype(str).str.lower().isin(("true", "1", "yes"))
    return pd.Series([bool(values)])


def _continuation_series(frame: pd.DataFrame) -> pd.Series:
    return _bool_series(frame.get("continuation_valid", False))


def _positive_series(frame: pd.DataFrame) -> pd.Series:
    return _continuation_series(frame) | (frame["outcome_class"].astype(str) == "accepted")


def _terminal_useful_series(frame: pd.DataFrame) -> pd.Series:
    return _bool_series(frame.get("episode_terminal_useful", False)) | (
        frame["boundary_use_class"].astype(str) == "episode_terminal_useful"
    )


def _hard_failure_series(frame: pd.DataFrame) -> pd.Series:
    return (frame["outcome_class"].astype(str) == "failed") | (
        frame["boundary_use_class"].astype(str) == "hard_failure"
    )


def _blocked_series(frame: pd.DataFrame) -> pd.Series:
    return frame["outcome_class"].astype(str).isin(("blocked", "rejected"))


def _blocked_count(frame: pd.DataFrame) -> int:
    return int(_blocked_series(frame).sum()) if "outcome_class" in frame.columns else 0


def _safe_min(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns or frame.empty:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").min())


def _safe_mean(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns or frame.empty:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").mean())


def _json_scalar(value):
    if pd.isna(value):
        return ""
    if hasattr(value, "item"):
        return value.item()
    return value


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    filesystem_path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(filesystem_path(path), index=False)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    filesystem_path(path).parent.mkdir(parents=True, exist_ok=True)
    filesystem_path(path).write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="ascii")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build W3 variant-level survival registry.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_W3_ROOT)
    parser.add_argument("--survived-hard-failure-rate-max", type=float, default=0.55)
    parser.add_argument("--downgraded-hard-failure-rate-max", type=float, default=0.75)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_w3_survival_analysis(
        W3SurvivalAnalysisConfig(
            input_root=args.input_root,
            survived_hard_failure_rate_max=args.survived_hard_failure_rate_max,
            downgraded_hard_failure_rate_max=args.downgraded_hard_failure_rate_max,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") != "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
