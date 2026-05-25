from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
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


PROJECT_TITLE_VERSION = "LQR-Stabilised Contextual Primitive v4.9"
FIGURE_RUN_VERSION = "v49_paired_full_loop_validation_figures_v1"
DEFAULT_INPUT_ROOT = Path("03_Control/05_Results/lqr_contextual_v1_0/full_loop_validation/004")
DEFAULT_OUTPUT_ROOT = Path("03_Control/05_Results/lqr_contextual_v1_0/figures/v49_validation")
FIGURE_NAMES = (
    "policy_terminal_hard_failure_bar.png",
    "memory_lambda_comparison.png",
    "belief_evolution_example.png",
    "prediction_alignment_summary.png",
    "governor_rejection_summary.png",
    "termination_summary.png",
)


@dataclass(frozen=True)
class V49ValidationFigureConfig:
    input_root: Path = DEFAULT_INPUT_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    run_id: int = 1


def run_v49_validation_figures(config: V49ValidationFigureConfig) -> dict[str, object]:
    """Generate compact v4.9 validation figures and Chapter 7 tables."""

    input_root = Path(config.input_root)
    run_root = Path(config.output_root) / f"{int(config.run_id):03d}"
    for subdir in ("figures", "tables", "metrics", "manifests", "reports"):
        filesystem_path(run_root / subdir).mkdir(parents=True, exist_ok=True)

    manifest = _read_json(input_root / "manifests" / "full_loop_validation_manifest.json")
    if str(manifest.get("project_title_version", "")) != PROJECT_TITLE_VERSION:
        _write_blocked(run_root, input_root, "input_root_not_v49_full_loop")
        return {"status": "blocked", "blocked_reason": "input_root_not_v49_full_loop", "run_root": run_root.as_posix()}
    if str(manifest.get("status", "")) != "complete":
        _write_blocked(run_root, input_root, "input_root_not_complete")
        return {"status": "blocked", "blocked_reason": "input_root_not_complete", "run_root": run_root.as_posix()}

    policy_summary = _read_csv(input_root / "metrics" / "policy_summary.csv")
    memory_summary = _read_csv(input_root / "metrics" / "memory_ablation_summary.csv")
    belief_summary = _read_csv(input_root / "metrics" / "belief_evolution_summary.csv")
    prediction_summary = _read_csv(input_root / "metrics" / "prediction_alignment_summary.csv")
    rejection_summary = _read_csv(input_root / "metrics" / "governor_rejection_summary.csv")
    termination_summary = _read_csv(input_root / "metrics" / "termination_summary.csv")
    paired = _read_csv(input_root / "metrics" / "paired_policy_comparison.csv")

    generated = [
        _plot_policy_terminal_hard_failure(run_root, memory_summary),
        _plot_memory_lambda_comparison(run_root, paired),
        _plot_belief_evolution(run_root, belief_summary),
        _plot_prediction_alignment(run_root, prediction_summary),
        _plot_governor_rejections(run_root, rejection_summary),
        _plot_termination_summary(run_root, termination_summary),
    ]
    table_paths = _write_chapter_tables(run_root, input_root, policy_summary, paired)
    figure_manifest = {
        "manifest_version": FIGURE_RUN_VERSION,
        "project_title_version": PROJECT_TITLE_VERSION,
        "status": "complete",
        "input_root": input_root.as_posix(),
        "run_root": run_root.as_posix(),
        "figure_count": len(generated),
        "table_count": len(table_paths),
        "figures": [path.name for path in generated],
        "tables": [path.name for path in table_paths],
        "claim_status": "simulation_only_thesis_ready_validation_figures",
        "blocked_claims": list(manifest.get("blocked_claims", [])),
    }
    _write_json(run_root / "manifests" / "v49_validation_figures_manifest.json", figure_manifest)
    _write_file_size_audit(run_root)
    _write_report(run_root, figure_manifest)
    return {"status": "complete", "run_root": run_root.as_posix(), "figure_count": len(generated), "table_count": len(table_paths)}


def _plot_policy_terminal_hard_failure(run_root: Path, frame: pd.DataFrame) -> Path:
    path = run_root / "figures" / "policy_terminal_hard_failure_bar.png"
    if frame.empty:
        return _empty_plot(path, "Policy terminal and hard-failure rates")
    summary = frame.groupby("policy_id", dropna=False).agg(
        terminal_useful_rate=("terminal_useful_rate", "mean"),
        hard_failure_rate=("hard_failure_rate", "mean"),
    )
    ax = summary.plot(kind="bar", figsize=(10, 4), color=["#3b7ddd", "#c44e52"], width=0.8)
    ax.set_ylabel("Rate")
    ax.set_xlabel("Policy")
    ax.set_title("Terminal-useful and hard-failure rates")
    ax.legend(["Terminal useful", "Hard failure"], loc="best")
    plt.xticks(rotation=30, ha="right")
    return _save_current(path)


def _plot_memory_lambda_comparison(run_root: Path, frame: pd.DataFrame) -> Path:
    path = run_root / "figures" / "memory_lambda_comparison.png"
    if frame.empty:
        return _empty_plot(path, "Paired memory policy differences")
    summary = frame.groupby("comparison_id", dropna=False).agg(
        terminal_delta=("episode_terminal_useful_paired_difference", "mean"),
        hard_failure_delta=("hard_failure_paired_difference", "mean"),
        no_viable_delta=("no_viable_primitive_paired_difference", "mean"),
    )
    ax = summary.plot(kind="bar", figsize=(11, 4), color=["#3b7ddd", "#c44e52", "#8172b3"], width=0.8)
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_ylabel("Treatment - baseline")
    ax.set_xlabel("Paired comparison")
    ax.set_title("Paired memory-policy deltas")
    plt.xticks(rotation=30, ha="right")
    return _save_current(path)


def _plot_belief_evolution(run_root: Path, frame: pd.DataFrame) -> Path:
    path = run_root / "figures" / "belief_evolution_example.png"
    if frame.empty:
        return _empty_plot(path, "Belief update counts by policy")
    ordered = frame.sort_values("policy_id")
    ax = ordered.plot(
        x="policy_id",
        y="final_belief_update_count",
        kind="bar",
        figsize=(10, 4),
        color="#55a868",
        legend=False,
    )
    ax.set_ylabel("Final belief update count")
    ax.set_xlabel("Policy")
    ax.set_title("Persistent belief updates")
    plt.xticks(rotation=30, ha="right")
    return _save_current(path)


def _plot_prediction_alignment(run_root: Path, frame: pd.DataFrame) -> Path:
    path = run_root / "figures" / "prediction_alignment_summary.png"
    if frame.empty:
        return _empty_plot(path, "Prediction alignment")
    summary = frame.groupby("policy_id", dropna=False).agg(
        continuation=("continuation_agreement_rate", "mean"),
        terminal=("terminal_agreement_rate", "mean"),
        hard_failure=("hard_failure_agreement_rate", "mean"),
    )
    ax = summary.plot(kind="bar", figsize=(10, 4), color=["#4c72b0", "#55a868", "#c44e52"], width=0.8)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Agreement rate")
    ax.set_xlabel("Policy")
    ax.set_title("Outcome-model prediction alignment")
    plt.xticks(rotation=30, ha="right")
    return _save_current(path)


def _plot_governor_rejections(run_root: Path, frame: pd.DataFrame) -> Path:
    path = run_root / "figures" / "governor_rejection_summary.png"
    if frame.empty:
        return _empty_plot(path, "Governor rejections")
    summary = (
        frame.groupby("rejection_reason", dropna=False)["governor_rejection_count"]
        .sum()
        .sort_values(ascending=False)
        .head(12)
    )
    ax = summary.plot(kind="bar", figsize=(10, 4), color="#8172b3")
    ax.set_ylabel("Count")
    ax.set_xlabel("Rejection reason")
    ax.set_title("Governor rejection reasons")
    plt.xticks(rotation=30, ha="right")
    return _save_current(path)


def _plot_termination_summary(run_root: Path, frame: pd.DataFrame) -> Path:
    path = run_root / "figures" / "termination_summary.png"
    if frame.empty:
        return _empty_plot(path, "Episode terminations")
    summary = frame.groupby("termination_cause", dropna=False)["episode_count"].sum().sort_values(ascending=False)
    ax = summary.plot(kind="bar", figsize=(10, 4), color="#dd8452")
    ax.set_ylabel("Episode count")
    ax.set_xlabel("Termination cause")
    ax.set_title("Full-loop termination causes")
    plt.xticks(rotation=30, ha="right")
    return _save_current(path)


def _write_chapter_tables(run_root: Path, input_root: Path, policy_summary: pd.DataFrame, paired: pd.DataFrame) -> list[Path]:
    table_root = run_root / "tables"
    source_summary = _read_csv(input_root / "metrics" / "source_audit_summary.csv")
    source_table = _minimal_frame(
        source_summary,
        ["stage", "status", "row_count", "survivor_count", "representative_count", "source_role"],
    )
    compact_manifest = _read_json(input_root / "manifests" / "full_loop_validation_manifest.json")
    compact_table = pd.DataFrame(
        [
            {
                "compact_library_path": compact_manifest.get("compact_library_path", ""),
                "compact_representative_count": compact_manifest.get("compact_representative_count", 0),
                "controller_mutation_allowed": compact_manifest.get("controller_mutation_allowed", True),
                "retuning_allowed": compact_manifest.get("retuning_allowed", True),
            }
        ]
    )
    policy_table = _minimal_frame(
        policy_summary,
        [
            "policy_id",
            "W_layer",
            "environment_mode",
            "episode_count",
            "terminal_useful_rate",
            "hard_failure_rate",
            "no_viable_primitive_count",
            "prediction_actual_agreement_rate",
        ],
    )
    claim_table = pd.DataFrame(
        [
            {
                "allowed_claim": "simulation_only_paired_full_loop_validation_for_frozen_post_W3_compact_library",
                "memory_effect_label": _overall_label(paired),
                "hardware_readiness_claimed": False,
                "real_flight_transfer_claimed": False,
                "mission_success_claimed": False,
                "memory_improvement_claim_requires_paired_support": True,
            }
        ]
    )
    outputs = [
        _write_csv(table_root / "chapter7_w01_w2_w3_evidence_table.csv", source_table),
        _write_csv(table_root / "chapter7_post_w3_compact_library_table.csv", compact_table),
        _write_csv(table_root / "chapter7_full_loop_policy_table.csv", policy_table),
        _write_csv(table_root / "chapter7_claim_boundary_table.csv", claim_table),
    ]
    return outputs


def _minimal_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=columns)
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = ""
    return out[columns]


def _overall_label(paired: pd.DataFrame) -> str:
    if paired.empty or "memory_effect_label" not in paired.columns:
        return "memory_benefit_not_supported"
    labels = set(paired["memory_effect_label"].astype(str))
    if labels == {"memory_benefit_supported"}:
        return "memory_benefit_supported"
    if "memory_benefit_supported" in labels or "mixed_memory_effect" in labels:
        return "mixed_memory_effect"
    return "memory_benefit_not_supported"


def _empty_plot(path: Path, title: str) -> Path:
    plt.figure(figsize=(7, 3))
    plt.title(title)
    plt.text(0.5, 0.5, "No rows available", ha="center", va="center")
    plt.axis("off")
    return _save_current(path)


def _save_current(path: Path) -> Path:
    filesystem_path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(filesystem_path(path), dpi=150)
    plt.close()
    return filesystem_path(path)


def _write_blocked(run_root: Path, input_root: Path, reason: str) -> None:
    manifest = {
        "manifest_version": FIGURE_RUN_VERSION,
        "project_title_version": PROJECT_TITLE_VERSION,
        "status": "blocked",
        "blocked_reason": reason,
        "input_root": input_root.as_posix(),
        "claim_status": "simulation_only_figure_generation_blocked",
    }
    _write_json(run_root / "manifests" / "v49_validation_figures_manifest.json", manifest)
    _write_file_size_audit(run_root)
    _write_report(run_root, manifest)


def _write_report(run_root: Path, manifest: dict[str, object]) -> None:
    lines = [
        "# v4.9 Validation Figures",
        "",
        f"- Status: `{manifest.get('status', '')}`",
        f"- Input root: `{manifest.get('input_root', '')}`",
        f"- Figure count: `{manifest.get('figure_count', 0)}`",
        f"- Table count: `{manifest.get('table_count', 0)}`",
        "- Figures and tables are thesis-facing summaries of simulation-only evidence.",
        "- No hardware, real-flight, mission-success, or formal ROA claim is made.",
        "",
    ]
    filesystem_path(run_root / "reports" / "v49_validation_figures_report.md").write_text("\n".join(lines), encoding="ascii")


def _write_file_size_audit(root: Path) -> None:
    rows = []
    root_fs = filesystem_path(root)
    for path in sorted(root_fs.rglob("*")):
        if not path.is_file() or path.name == "file_size_audit.csv":
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
                "dense_table_partition": False,
                "sha256": file_sha256(path),
            }
        )
    _write_csv(root / "metrics" / "file_size_audit.csv", pd.DataFrame(rows))


def _read_json(path: Path) -> dict[str, object]:
    fs_path = filesystem_path(path)
    if not fs_path.is_file():
        return {}
    return json.loads(fs_path.read_text(encoding="ascii"))


def _read_csv(path: Path) -> pd.DataFrame:
    fs_path = filesystem_path(path)
    if not fs_path.is_file():
        return pd.DataFrame()
    return pd.read_csv(fs_path)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    filesystem_path(path).parent.mkdir(parents=True, exist_ok=True)
    filesystem_path(path).write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="ascii")


def _write_csv(path: Path, frame: pd.DataFrame) -> Path:
    filesystem_path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(filesystem_path(path), index=False)
    return filesystem_path(path)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate v4.9 paired full-loop validation figures and Chapter 7 tables.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_v49_validation_figures(
        V49ValidationFigureConfig(input_root=args.input_root, output_root=args.output_root, run_id=args.run_id)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") != "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
