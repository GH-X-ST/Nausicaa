from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd


def _bootstrap_import_paths() -> None:
    control_root = Path(__file__).resolve().parents[1]
    for rel in ("03_Primitives", "04_Scenarios"):
        path = control_root / rel
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


_bootstrap_import_paths()

from dense_archive_runtime import MAX_GENERATED_FILE_SIZE_MB  # noqa: E402
from dense_archive_table_io import file_sha256, filesystem_path  # noqa: E402
from episode_selector import select_compact_representative  # noqa: E402
from run_full_loop_validation import FullLoopValidationConfig, run_full_loop_validation  # noqa: E402
from run_v410_source_audit import (  # noqa: E402
    BLOCKED_CLAIMS,
    DEFAULT_FULL_LOOP_ROOT,
    DEFAULT_GOVERNOR_SMOKE_ROOT,
    DEFAULT_OUTCOME_ROOT,
    DEFAULT_POST_W3_ROOT,
    DEFAULT_W01_ROOT,
    DEFAULT_W2_ROOT,
    DEFAULT_W3_ROOT,
    V410SourceAuditConfig,
    run_v410_source_audit,
)
from viability_governor import DEFAULT_GOVERNOR_CONFIG, GovernorConfig, governor_config_to_row  # noqa: E402


PROJECT_TITLE_VERSION = "LQR-Stabilised Contextual Primitive v4.10"
CALIBRATION_VERSION = "v410_outer_loop_governor_calibration_v1"
DEFAULT_OUTPUT_ROOT = Path("03_Control/05_Results/lqr_contextual_v1_0/governor_calibration")
DEFAULT_COMPACT_LIBRARY = Path("03_Control/05_Results/lqr_contextual_v1_0/post_w3_cluster/001/manifests/final_compact_primitive_library.json")
CALIBRATION_SUBRUN_DIR = "cal"
MAX_REPO_RELATIVE_PATH_CHARS = 140


@dataclass(frozen=True)
class GovernorCalibrationConfig:
    run_id: int = 1
    output_root: Path = DEFAULT_OUTPUT_ROOT
    ranking_config_count: int = 50
    top_config_count: int = 5
    calibration_episodes_per_policy: int = 100
    context_sample_count: int = 360
    seed: int = 410
    dry_run_schedule: bool = False
    source_w01_root: Path = DEFAULT_W01_ROOT
    source_w2_root: Path = DEFAULT_W2_ROOT
    source_w3_root: Path = DEFAULT_W3_ROOT
    post_w3_root: Path = DEFAULT_POST_W3_ROOT
    outcome_root: Path = DEFAULT_OUTCOME_ROOT
    governor_smoke_root: Path = DEFAULT_GOVERNOR_SMOKE_ROOT
    source_full_loop_root: Path = DEFAULT_FULL_LOOP_ROOT
    compact_library_path: Path = DEFAULT_COMPACT_LIBRARY


def run_governor_calibration(config: GovernorCalibrationConfig) -> dict[str, object]:
    """Calibrate outer-loop governor weights without mutating frozen controllers."""

    run_root = Path(config.output_root) / f"{int(config.run_id):03d}"
    for subdir in ("manifests", "metrics", "reports", CALIBRATION_SUBRUN_DIR):
        filesystem_path(run_root / subdir).mkdir(parents=True, exist_ok=True)

    source_result = run_v410_source_audit(
        V410SourceAuditConfig(
            output_root=run_root,
            w01_root=config.source_w01_root,
            w2_root=config.source_w2_root,
            w3_root=config.source_w3_root,
            post_w3_root=config.post_w3_root,
            outcome_root=config.outcome_root,
            governor_smoke_root=config.governor_smoke_root,
            full_loop_root=config.source_full_loop_root,
        )
    )
    if source_result["status"] != "source_audit_pass":
        _write_blocked(run_root, config, "source_audit_failed", source_result.get("blockers", []))
        return {"status": "blocked", "blocked_reason": "source_audit_failed", "run_root": run_root.as_posix()}

    representatives = list(_read_json(config.compact_library_path).get("representatives", []))
    outcome_rows = pd.read_csv(filesystem_path(config.outcome_root / "metrics" / "outcome_model_table.csv")).to_dict(orient="records")
    outcomes_by_variant = {str(row.get("primitive_variant_id", "")): row for row in outcome_rows}
    context_rows = _load_context_samples(config.source_full_loop_root, int(config.context_sample_count))
    config_grid = generate_governor_config_grid(int(config.ranking_config_count))
    _write_csv(run_root / "metrics" / "governor_config_grid.csv", pd.DataFrame(governor_config_to_row(item) for item in config_grid))

    ranking = _rank_configs(config_grid, representatives, outcomes_by_variant, context_rows)
    top_ids = list(ranking.sort_values("ranking_score", ascending=True)["governor_config_id"].head(int(config.top_config_count)))
    top_configs = [item for item in config_grid if item.config_id in set(top_ids)]
    _write_csv(run_root / "metrics" / "calibration_config_summary.csv", ranking)
    _write_sensitivity_outputs(run_root, ranking)

    if config.dry_run_schedule:
        _write_manifest(run_root, config, "dry_run_schedule", top_configs, selected_config=None, calibration_rows=[])
        _write_file_size_audit(run_root)
        _write_report(run_root, status="dry_run_schedule", selected_config=None, memory_label="not_run")
        return {"status": "dry_run_schedule", "run_root": run_root.as_posix(), "top_config_count": len(top_configs)}

    calibration_rows: list[dict[str, object]] = []
    calibration_schedule_rows: list[dict[str, object]] = []
    for index, governor_config in enumerate(top_configs, start=1):
        subrun_root = _calibration_subrun_output_root(run_root, index)
        result = run_full_loop_validation(
            FullLoopValidationConfig(
                run_id=index,
                output_root=subrun_root,
                episodes_per_policy=int(config.calibration_episodes_per_policy),
                seed=int(config.seed) + index * 101,
                source_audit_version="v410",
                source_full_loop_root=config.source_full_loop_root,
                governor_config=governor_config,
                resume=True,
                repair_incomplete=True,
            )
        )
        subrun_run_root = _calibration_subrun_run_root(run_root, index)
        summary = _summarise_full_loop_run(
            subrun_run_root,
            governor_config,
            result,
        )
        summary["calibration_subrun_dir"] = _calibration_subrun_dir_name(index)
        calibration_rows.append(summary)
        schedule_path = filesystem_path(subrun_run_root / "metrics" / "episode_schedule.csv")
        if schedule_path.is_file():
            schedule = pd.read_csv(schedule_path)
            schedule["governor_config_id"] = governor_config.config_id
            schedule["calibration_subrun_dir"] = _calibration_subrun_dir_name(index)
            calibration_schedule_rows.extend(schedule.to_dict(orient="records"))

    calibration_frame = pd.DataFrame(calibration_rows)
    _write_csv(run_root / "metrics" / "calibration_policy_summary.csv", calibration_frame)
    _write_csv(run_root / "metrics" / "calibration_episode_schedule.csv", pd.DataFrame(calibration_schedule_rows))
    selected_config = _select_frozen_config(top_configs, calibration_frame)
    selected_row = calibration_frame[calibration_frame["governor_config_id"] == selected_config.config_id].head(1)
    selection_payload = {
        "manifest_version": "v410_frozen_governor_config_v1",
        "project_title_version": PROJECT_TITLE_VERSION,
        "status": "selected",
        "selection_policy": "minimise_hard_failure_then_no_viable_then_maximise_terminal_useful",
        "governor_config": governor_config_to_row(selected_config),
        "selection_metrics": {} if selected_row.empty else selected_row.iloc[0].to_dict(),
        "source_calibration_root": run_root.as_posix(),
        "controller_mutation_allowed": False,
        "retuning_allowed": False,
        "claim_status": "simulation_only_frozen_outer_loop_governor_config",
        "blocked_claims": list(BLOCKED_CLAIMS),
    }
    _write_json(run_root / "manifests" / "frozen_governor_config.json", selection_payload)
    _write_csv(run_root / "metrics" / "governor_config_selection.csv", pd.DataFrame([selection_payload["selection_metrics"]]))
    _write_manifest(run_root, config, "complete", top_configs, selected_config=selected_config, calibration_rows=calibration_rows)
    _write_file_size_audit(run_root)
    _write_report(run_root, status="complete", selected_config=selected_config, memory_label=str(selected_row.get("memory_effect_label", pd.Series(["not_run"])).iloc[0]))
    return {
        "status": "complete",
        "run_root": run_root.as_posix(),
        "selected_governor_config_id": selected_config.config_id,
        "top_config_count": len(top_configs),
    }


def _calibration_subrun_dir_name(index: int) -> str:
    return f"c{int(index) - 1:03d}"


def _calibration_subrun_output_root(run_root: Path, index: int) -> Path:
    return run_root / CALIBRATION_SUBRUN_DIR / _calibration_subrun_dir_name(index)


def _calibration_subrun_run_root(run_root: Path, index: int) -> Path:
    return _calibration_subrun_output_root(run_root, index) / f"{int(index):03d}"


def generate_governor_config_grid(limit: int = 50) -> list[GovernorConfig]:
    configs: list[GovernorConfig] = []
    base = DEFAULT_GOVERNOR_CONFIG
    belief_weights = (0.05, 0.10, 0.20, 0.35, 0.55)
    hard_limits = (0.75, 0.72, 0.70, 0.68, 0.65)
    terminal_biases = (0.0, 0.05)
    for belief_weight in belief_weights:
        for hard_limit in hard_limits:
            for terminal_bias in terminal_biases:
                index = len(configs)
                configs.append(
                    replace(
                        base,
                        config_id=f"v410_cfg_{index:03d}_b{belief_weight:g}_h{hard_limit:g}_tb{terminal_bias:g}",
                        maximum_hard_failure_risk=hard_limit,
                        belief_weight=belief_weight,
                        terminal_mode_bias=terminal_bias,
                        continuation_mode_bias=-0.02 if terminal_bias > 0.0 else 0.0,
                        exploration_bonus_weight=0.02 * index / max(1, int(limit)),
                    )
                )
                if len(configs) >= int(limit):
                    return configs
    return configs[: int(limit)]


def _rank_configs(
    configs: list[GovernorConfig],
    representatives: list[dict[str, object]],
    outcomes_by_variant: dict[str, dict[str, object]],
    context_rows: list[dict[str, object]],
) -> pd.DataFrame:
    rows = []
    for governor_config in configs:
        selected_rows = []
        candidate_rows = []
        no_viable_count = 0
        for context in context_rows:
            belief_features = {
                "belief_local_lift_m_s": float(context.get("belief_local_lift_m_s", 0.0)),
                "belief_mean_lift_m_s": float(context.get("belief_mean_lift_m_s", context.get("belief_local_lift_m_s", 0.0))),
                "belief_max_lift_m_s": float(context.get("belief_max_lift_m_s", context.get("belief_local_lift_m_s", 0.0))),
            }
            selected, candidates = select_compact_representative(
                representatives=representatives,
                outcome_rows_by_variant_id=outcomes_by_variant,
                context=context,
                governor_mode=str(context.get("governor_mode", "continuation_mode")),
                policy_id=str(context.get("policy_id", "")),
                belief_features=belief_features,
                governor_config=governor_config,
            )
            candidate_rows.extend(candidates)
            if selected is None:
                no_viable_count += 1
            else:
                selected_rows.append(selected)
        selected_frame = pd.DataFrame(selected_rows)
        candidate_frame = pd.DataFrame(candidate_rows)
        mean_hard = float(selected_frame["hard_failure_risk"].mean()) if not selected_frame.empty else 1.0
        mean_terminal = float(selected_frame["terminal_useful_probability"].mean()) if not selected_frame.empty else 0.0
        mean_continuation = float(selected_frame["continuation_probability"].mean()) if not selected_frame.empty else 0.0
        rows.append(
            {
                "governor_config_id": governor_config.config_id,
                "context_count": len(context_rows),
                "no_viable_count": int(no_viable_count),
                "mean_selected_hard_failure_risk": mean_hard,
                "mean_selected_terminal_useful_probability": mean_terminal,
                "mean_selected_continuation_probability": mean_continuation,
                "mean_viable_count": float(candidate_frame.groupby("context_id")["viable"].sum().mean()) if not candidate_frame.empty else 0.0,
                "memory_changed_rank_count": int((candidate_frame.get("rank_change_due_to_memory", pd.Series(dtype=float)).astype(float) != 0.0).sum()) if not candidate_frame.empty else 0,
                "mean_abs_memory_score_component": float(candidate_frame.get("memory_score_component", pd.Series(dtype=float)).astype(float).abs().mean()) if not candidate_frame.empty else 0.0,
                "selected_primitive_unique_count": int(selected_frame["primitive_variant_id"].nunique()) if not selected_frame.empty else 0,
                "ranking_score": float(no_viable_count * 10.0 + mean_hard * 100.0 - mean_terminal * 20.0 - mean_continuation * 5.0),
                **{f"config_{key}": value for key, value in asdict(governor_config).items() if key != "config_id"},
            }
        )
    return pd.DataFrame(rows)


def _load_context_samples(full_loop_root: Path, limit: int) -> list[dict[str, object]]:
    path = filesystem_path(full_loop_root / "metrics" / "governor_rejection_log.csv")
    frame = pd.read_csv(path)
    for column in ("belief_mean_lift_m_s", "belief_max_lift_m_s"):
        if column not in frame.columns:
            frame[column] = frame["belief_local_lift_m_s"]
    columns = [
        "policy_id",
        "context_id",
        "W_layer",
        "environment_mode",
        "start_state_family",
        "governor_mode",
        "wall_margin_m",
        "floor_margin_m",
        "ceiling_margin_m",
        "speed_margin_m_s",
        "belief_local_lift_m_s",
        "belief_mean_lift_m_s",
        "belief_max_lift_m_s",
    ]
    return frame[columns].drop_duplicates().head(int(limit)).to_dict(orient="records")


def _summarise_full_loop_run(run_root: Path, governor_config: GovernorConfig, result: dict[str, object]) -> dict[str, object]:
    summary_path = filesystem_path(run_root / "metrics" / "memory_ablation_summary.csv")
    summary = pd.read_csv(summary_path) if summary_path.is_file() else pd.DataFrame()
    if summary.empty:
        return {"governor_config_id": governor_config.config_id, "status": result.get("status", "blocked")}
    return {
        "governor_config_id": governor_config.config_id,
        "status": result.get("status", ""),
        "run_root": run_root.as_posix(),
        "episode_count": int(summary["episode_count"].sum()),
        "terminal_useful_rate": float(summary["terminal_useful_rate"].mean()),
        "hard_failure_rate": float(summary["hard_failure_rate"].mean()),
        "mean_primitive_count": float(summary["mean_primitive_count"].mean()),
        "no_viable_primitive_count": int(summary["no_viable_primitive_count"].sum()),
        "memory_effect_label": str(result.get("memory_effect_label", "")),
        **{f"config_{key}": value for key, value in asdict(governor_config).items() if key != "config_id"},
    }


def _select_frozen_config(configs: list[GovernorConfig], summary: pd.DataFrame) -> GovernorConfig:
    if summary.empty or "governor_config_id" not in summary.columns:
        return configs[0]
    ordered = summary.sort_values(
        ["hard_failure_rate", "no_viable_primitive_count", "terminal_useful_rate", "governor_config_id"],
        ascending=[True, True, False, True],
    )
    selected_id = str(ordered.iloc[0]["governor_config_id"])
    return next((item for item in configs if item.config_id == selected_id), configs[0])


def _write_sensitivity_outputs(run_root: Path, ranking: pd.DataFrame) -> None:
    _write_csv(
        run_root / "metrics" / "governor_score_component_summary.csv",
        ranking[["governor_config_id", "mean_abs_memory_score_component", "mean_selected_hard_failure_risk", "mean_selected_terminal_useful_probability"]],
    )
    _write_csv(
        run_root / "metrics" / "memory_influence_summary.csv",
        ranking[["governor_config_id", "memory_changed_rank_count", "mean_abs_memory_score_component", "selected_primitive_unique_count"]],
    )
    _write_csv(
        run_root / "metrics" / "selector_rank_change_summary.csv",
        ranking[["governor_config_id", "memory_changed_rank_count", "no_viable_count", "mean_viable_count"]],
    )


def _write_manifest(
    run_root: Path,
    config: GovernorCalibrationConfig,
    status: str,
    top_configs: list[GovernorConfig],
    *,
    selected_config: GovernorConfig | None,
    calibration_rows: list[dict[str, object]],
) -> None:
    manifest = {
        "manifest_version": CALIBRATION_VERSION,
        "project_title_version": PROJECT_TITLE_VERSION,
        "status": status,
        "run_id": int(config.run_id),
        "run_root": run_root.as_posix(),
        "ranking_config_count": int(config.ranking_config_count),
        "top_config_count": int(config.top_config_count),
        "calibration_episodes_per_policy": int(config.calibration_episodes_per_policy),
        "context_sample_count": int(config.context_sample_count),
        "source_full_loop_root": config.source_full_loop_root.as_posix(),
        "top_governor_config_ids": [item.config_id for item in top_configs],
        "calibration_subrun_layout": f"{CALIBRATION_SUBRUN_DIR}/cNNN/<run_id>",
        "calibration_subrun_map": [
            {
                "governor_config_id": item.config_id,
                "calibration_subrun_dir": _calibration_subrun_dir_name(index),
            }
            for index, item in enumerate(top_configs, start=1)
        ],
        "selected_governor_config_id": "" if selected_config is None else selected_config.config_id,
        "calibration_run_count": len(calibration_rows),
        "controller_mutation_allowed": False,
        "retuning_allowed": False,
        "claim_status": "simulation_only_outer_loop_governor_calibration",
        "blocked_claims": list(BLOCKED_CLAIMS),
    }
    _write_json(run_root / "manifests" / "governor_calibration_manifest.json", manifest)


def _write_report(run_root: Path, *, status: str, selected_config: GovernorConfig | None, memory_label: str) -> None:
    lines = [
        "# v4.10 Governor Calibration Report",
        "",
        f"- Status: `{status}`",
        f"- Selected config: `{'' if selected_config is None else selected_config.config_id}`",
        f"- Calibration memory label: `{memory_label}`",
        "- Calibration tunes only outer-loop governor and selector weights.",
        "- W01/W2/W3/post-W3 compact representatives remain frozen.",
        "- No Q/R, K, reference, horizon, entry role, controller ID, or primitive variant ID mutation is allowed.",
        "",
    ]
    filesystem_path(run_root / "reports" / "governor_calibration_report.md").write_text("\n".join(lines), encoding="ascii")
    filesystem_path(run_root / "reports" / "governor_sensitivity_report.md").write_text("\n".join(lines), encoding="ascii")


def _write_blocked(run_root: Path, config: GovernorCalibrationConfig, reason: str, blockers: list[object]) -> None:
    _write_manifest(run_root, config, "blocked", [], selected_config=None, calibration_rows=[])
    _write_json(run_root / "manifests" / "blocked_governor_calibration.json", {"blocked_reason": reason, "blockers": [str(item) for item in blockers]})
    _write_file_size_audit(run_root)


def _write_file_size_audit(root: Path) -> None:
    rows = []
    root_fs = filesystem_path(root)
    repo_root = Path.cwd().resolve()
    for path in sorted(root_fs.rglob("*")):
        if not path.is_file() or path.name == "file_size_audit.csv":
            continue
        rel = path.relative_to(root_fs).as_posix()
        try:
            repo_rel = _normal_audit_path(path).resolve().relative_to(repo_root).as_posix()
        except ValueError:
            repo_rel = ""
        byte_count = int(path.stat().st_size)
        size_mb = float(byte_count) / float(1024 * 1024)
        repo_rel_len = len(repo_rel)
        path_within_limit = bool((not repo_rel) or repo_rel_len <= MAX_REPO_RELATIVE_PATH_CHARS)
        rows.append(
            {
                "relative_path": rel,
                "relative_path_length": len(rel),
                "repo_relative_path": repo_rel,
                "repo_relative_path_length": repo_rel_len,
                "byte_count": byte_count,
                "size_mb": size_mb,
                "above_75mb": bool(size_mb > 75.0),
                "above_100mb": bool(size_mb > MAX_GENERATED_FILE_SIZE_MB),
                "path_within_140_chars": path_within_limit,
                "push_allowed": bool(size_mb <= MAX_GENERATED_FILE_SIZE_MB and path_within_limit),
                "dense_table_partition": rel.startswith("tables/"),
                "sha256": file_sha256(path),
            }
        )
    _write_csv(root / "metrics" / "file_size_audit.csv", pd.DataFrame(rows))


def _normal_audit_path(path: Path) -> Path:
    text = str(path)
    if text.startswith("\\\\?\\UNC\\"):
        return Path("\\\\" + text[len("\\\\?\\UNC\\") :])
    if text.startswith("\\\\?\\"):
        return Path(text[len("\\\\?\\") :])
    return Path(text)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(filesystem_path(path).read_text(encoding="ascii"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    filesystem_path(path).parent.mkdir(parents=True, exist_ok=True)
    filesystem_path(path).write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="ascii")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    filesystem_path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(filesystem_path(path), index=False)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v4.10 outer-loop governor calibration.")
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--ranking-config-count", type=int, default=50)
    parser.add_argument("--top-config-count", type=int, default=5)
    parser.add_argument("--calibration-episodes-per-policy", type=int, default=100)
    parser.add_argument("--context-sample-count", type=int, default=360)
    parser.add_argument("--seed", type=int, default=410)
    parser.add_argument("--dry-run-schedule", action="store_true", default=False)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_governor_calibration(
        GovernorCalibrationConfig(
            run_id=args.run_id,
            output_root=args.output_root,
            ranking_config_count=args.ranking_config_count,
            top_config_count=args.top_config_count,
            calibration_episodes_per_policy=args.calibration_episodes_per_policy,
            context_sample_count=args.context_sample_count,
            seed=args.seed,
            dry_run_schedule=args.dry_run_schedule,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") != "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
