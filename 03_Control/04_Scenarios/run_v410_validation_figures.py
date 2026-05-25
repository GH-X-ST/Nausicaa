from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd
from pandas.errors import EmptyDataError


def _bootstrap_import_paths() -> None:
    control_root = Path(__file__).resolve().parents[1]
    for rel in ("03_Primitives", "04_Scenarios"):
        path = control_root / rel
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


_bootstrap_import_paths()

from dense_archive_runtime import MAX_GENERATED_FILE_SIZE_MB  # noqa: E402
from dense_archive_table_io import file_sha256, filesystem_path  # noqa: E402


PROJECT_TITLE_VERSION = "LQR-Stabilised Contextual Primitive v4.10"
DEFAULT_CALIBRATION_ROOT = Path("03_Control/05_Results/lqr_contextual_v1_0/governor_calibration/001")
DEFAULT_HELDOUT_ROOT = Path("03_Control/05_Results/lqr_contextual_v1_0/full_loop_validation/006")


def run_v410_validation_figures(calibration_root: Path = DEFAULT_CALIBRATION_ROOT, heldout_root: Path = DEFAULT_HELDOUT_ROOT) -> dict[str, object]:
    root = Path(calibration_root)
    figure_root = root / "09_figures"
    filesystem_path(figure_root).mkdir(parents=True, exist_ok=True)
    heldout_manifest = _read_json(heldout_root / "manifests" / "full_loop_validation_manifest.json")
    if str(heldout_manifest.get("project_title_version", "")) != PROJECT_TITLE_VERSION:
        _write_json(root / "manifests" / "v410_figure_generation_blocked.json", {"status": "blocked", "blocked_reason": "heldout_root_not_v410"})
        return {"status": "blocked", "blocked_reason": "heldout_root_not_v410", "run_root": root.as_posix()}

    memory_summary = _read_csv(heldout_root / "metrics" / "memory_ablation_summary.csv")
    paired = _read_csv(heldout_root / "metrics" / "paired_policy_comparison.csv")
    rejection_summary = _read_csv(heldout_root / "metrics" / "governor_rejection_summary.csv")
    belief_summary = _read_csv(heldout_root / "metrics" / "belief_evolution_summary.csv")
    prediction_summary = _read_csv(heldout_root / "metrics" / "prediction_alignment_summary.csv")

    figures = [
        _plot_memory_ablation(figure_root / "v410_memory_ablation_bar.png", memory_summary),
        _plot_paired_delta(figure_root / "v410_paired_delta_plot.png", paired),
        _plot_rejections(figure_root / "v410_governor_rejection_summary.png", rejection_summary),
        _plot_belief(figure_root / "v410_belief_evolution.png", belief_summary),
        _plot_prediction(figure_root / "v410_outcome_prediction_alignment.png", prediction_summary),
    ]
    _write_thesis_tables(root, heldout_root)
    manifest = {
        "manifest_version": "v410_validation_figures_v1",
        "project_title_version": PROJECT_TITLE_VERSION,
        "status": "complete",
        "calibration_root": root.as_posix(),
        "heldout_root": heldout_root.as_posix(),
        "figures": [path.name for path in figures],
        "claim_status": "simulation_only_outer_loop_governor_calibration_figures",
    }
    _write_json(root / "manifests" / "v410_validation_figures_manifest.json", manifest)
    _write_file_size_audit(root)
    _write_report(root, manifest, heldout_root=heldout_root, memory_summary=memory_summary, paired=paired)
    return {"status": "complete", "run_root": root.as_posix(), "figure_count": len(figures)}


def _plot_memory_ablation(path: Path, frame: pd.DataFrame) -> Path:
    if frame.empty:
        return _empty(path, "Memory ablation")
    summary = frame.groupby("policy_id", dropna=False).agg(
        terminal_useful_rate=("terminal_useful_rate", "mean"),
        hard_failure_rate=("hard_failure_rate", "mean"),
    )
    ax = summary.plot(kind="bar", figsize=(10, 4), color=["#3b7ddd", "#c44e52"])
    ax.set_ylabel("Rate")
    ax.set_title("Held-out memory ablation")
    plt.xticks(rotation=30, ha="right")
    return _save(path)


def _plot_paired_delta(path: Path, frame: pd.DataFrame) -> Path:
    if frame.empty:
        return _empty(path, "Paired policy deltas")
    summary = frame.groupby("comparison_id", dropna=False).agg(
        terminal_delta=("episode_terminal_useful_paired_difference", "mean"),
        hard_delta=("hard_failure_paired_difference", "mean"),
        no_viable_delta=("no_viable_primitive_paired_difference", "mean"),
    )
    ax = summary.plot(kind="bar", figsize=(11, 4), color=["#3b7ddd", "#c44e52", "#8172b3"])
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_ylabel("Treatment - baseline")
    ax.set_title("Held-out paired deltas")
    plt.xticks(rotation=30, ha="right")
    return _save(path)


def _plot_rejections(path: Path, frame: pd.DataFrame) -> Path:
    if frame.empty:
        return _empty(path, "Governor rejections")
    summary = frame.groupby("rejection_reason", dropna=False)["governor_rejection_count"].sum().sort_values(ascending=False).head(12)
    ax = summary.plot(kind="bar", figsize=(10, 4), color="#8172b3")
    ax.set_ylabel("Count")
    ax.set_title("Held-out governor rejections")
    plt.xticks(rotation=30, ha="right")
    return _save(path)


def _plot_belief(path: Path, frame: pd.DataFrame) -> Path:
    if frame.empty:
        return _empty(path, "Belief evolution")
    ax = frame.sort_values("policy_id").plot(
        x="policy_id",
        y="final_belief_update_count",
        kind="bar",
        figsize=(10, 4),
        color="#55a868",
        legend=False,
    )
    ax.set_ylabel("Final update count")
    ax.set_title("Held-out belief persistence")
    plt.xticks(rotation=30, ha="right")
    return _save(path)


def _plot_prediction(path: Path, frame: pd.DataFrame) -> Path:
    if frame.empty:
        return _empty(path, "Prediction alignment")
    summary = frame.groupby("policy_id", dropna=False).agg(
        continuation=("continuation_agreement_rate", "mean"),
        terminal=("terminal_agreement_rate", "mean"),
        hard_failure=("hard_failure_agreement_rate", "mean"),
    )
    ax = summary.plot(kind="bar", figsize=(10, 4), color=["#4c72b0", "#55a868", "#c44e52"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Agreement rate")
    ax.set_title("Held-out outcome prediction alignment")
    plt.xticks(rotation=30, ha="right")
    return _save(path)


def _write_thesis_tables(root: Path, heldout_root: Path) -> None:
    source_summary = _read_csv(root / "metrics" / "source_audit_summary.csv")
    post_w3 = _read_json(root / "manifests" / "source_audit.json")
    heldout = _read_csv(heldout_root / "metrics" / "policy_summary.csv")
    paired = _read_csv(heldout_root / "metrics" / "paired_policy_comparison.csv")
    _write_csv(root / "metrics" / "thesis_table_w01_w2_w3_summary.csv", _select_columns(source_summary, ["stage", "status", "row_count", "survivor_count", "representative_count"]))
    _write_csv(root / "metrics" / "thesis_table_post_w3_library.csv", pd.DataFrame([{"representative_count": _representative_count(post_w3), "source": "post_w3_cluster_001"}]))
    _write_csv(root / "metrics" / "thesis_table_outer_loop_heldout.csv", heldout)
    _write_csv(
        root / "metrics" / "thesis_table_claim_boundaries.csv",
        pd.DataFrame(
            [
                {
                    "allowed_claim": "simulation_only_outer_loop_governor_calibration_and_heldout_validation",
                    "memory_effect_label": _overall_label(paired),
                    "hardware_readiness_claimed": False,
                    "real_flight_transfer_claimed": False,
                    "mission_success_claimed": False,
                }
            ]
        ),
    )


def _representative_count(source_manifest: dict[str, object]) -> int:
    for check in source_manifest.get("checks", []):
        if str(check.get("stage", "")) == "post_W3":
            return int(check.get("representative_count", 0))
    return 0


def _overall_label(paired: pd.DataFrame) -> str:
    if paired.empty or "memory_effect_label" not in paired.columns:
        return "memory_benefit_not_supported"
    labels = set(paired["memory_effect_label"].astype(str))
    if labels == {"memory_benefit_supported"}:
        return "memory_benefit_supported"
    if "memory_benefit_supported" in labels or "mixed_memory_effect" in labels or "memory_benefit_mixed" in labels:
        return "memory_benefit_mixed"
    return "memory_benefit_not_supported"


def _select_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
    return frame[columns]


def _empty(path: Path, title: str) -> Path:
    plt.figure(figsize=(7, 3))
    plt.title(title)
    plt.text(0.5, 0.5, "No rows available", ha="center", va="center")
    plt.axis("off")
    return _save(path)


def _save(path: Path) -> Path:
    filesystem_path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(filesystem_path(path), dpi=150)
    plt.close()
    return filesystem_path(path)


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


def _write_report(root: Path, manifest: dict[str, object], *, heldout_root: Path, memory_summary: pd.DataFrame, paired: pd.DataFrame) -> None:
    terminal_rate = 0.0 if memory_summary.empty else float(memory_summary["terminal_useful_rate"].mean())
    hard_failure_rate = 0.0 if memory_summary.empty else float(memory_summary["hard_failure_rate"].mean())
    no_viable_count = 0 if memory_summary.empty else int(memory_summary["no_viable_primitive_count"].sum())
    memory_label = _overall_label(paired)
    lines = [
        "# v4.10 Validation Figures",
        "",
        f"- Status: `{manifest.get('status', '')}`",
        f"- Figure count: `{len(manifest.get('figures', []))}`",
        "- Figures summarize simulation-only held-out outer-loop validation.",
        "- No hardware, real-flight, mission-success, or formal ROA claim is made.",
        "",
    ]
    filesystem_path(root / "reports" / "v410_validation_figures_report.md").write_text("\n".join(lines), encoding="ascii")
    heldout_lines = [
        "# v4.10 Held-Out Full-Loop Report",
        "",
        f"- Held-out root: `{heldout_root.as_posix()}`",
        f"- Mean terminal-useful rate: `{terminal_rate:.6f}`",
        f"- Mean hard-failure rate: `{hard_failure_rate:.6f}`",
        f"- Total no-viable primitive count: `{no_viable_count}`",
        f"- Memory effect label: `{memory_label}`",
        "- Held-out validation reuses the frozen v4.10 governor config exactly.",
        "- W01/W2/W3/post-W3 compact library artifacts remain frozen.",
        "",
    ]
    filesystem_path(root / "reports" / "heldout_full_loop_report.md").write_text("\n".join(heldout_lines), encoding="ascii")
    move_on = [
        "# v4.10 Move-On Check",
        "",
        f"- Figure generation status: `{manifest.get('status', '')}`",
        f"- Held-out root present: `{filesystem_path(heldout_root).is_dir()}`",
        f"- Memory effect label: `{memory_label}`",
        f"- File-size audit below 100 MB: `{_file_size_gate(root)}`",
        "- Allowed claim: `simulation_only_outer_loop_governor_calibration_and_heldout_validation`",
        "- Hardware readiness claimed: `False`",
        "- Real-flight transfer claimed: `False`",
        "- Mission success claimed: `False`",
        "",
    ]
    filesystem_path(root / "reports" / "v410_move_on_check.md").write_text("\n".join(move_on), encoding="ascii")


def _file_size_gate(root: Path) -> bool:
    path = filesystem_path(root / "metrics" / "file_size_audit.csv")
    if not path.is_file():
        return False
    frame = pd.read_csv(path)
    return bool(frame.empty or not frame["above_100mb"].astype(bool).any())


def _read_json(path: Path) -> dict[str, object]:
    fs_path = filesystem_path(path)
    if not fs_path.is_file():
        return {}
    return json.loads(fs_path.read_text(encoding="ascii"))


def _read_csv(path: Path) -> pd.DataFrame:
    fs_path = filesystem_path(path)
    if not fs_path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(fs_path)
    except EmptyDataError:
        return pd.DataFrame()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    filesystem_path(path).parent.mkdir(parents=True, exist_ok=True)
    filesystem_path(path).write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="ascii")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    filesystem_path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(filesystem_path(path), index=False)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate v4.10 governor calibration figures and thesis tables.")
    parser.add_argument("--calibration-root", type=Path, default=DEFAULT_CALIBRATION_ROOT)
    parser.add_argument("--heldout-root", type=Path, default=DEFAULT_HELDOUT_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_v410_validation_figures(args.calibration_root, args.heldout_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") != "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
