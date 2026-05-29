from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
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


PROJECT_TITLE_VERSION = "LQR-Stabilised Contextual Primitive v4.7"
POST_W3_VERSION = "post_w3_compact_representative_library_v1"
DEFAULT_INPUT_ROOT = Path("03_Control/05_Results/R7_survival/A01")
DEFAULT_OUTPUT_ROOT = Path("03_Control/05_Results/R8_library_size_study")
BLOCKED_CLAIMS = (
    "hardware_readiness",
    "real_flight_transfer",
    "mission_success",
    "full_loop_validation_success",
    "memory_governor_performance_improvement",
    "compact_library_real_flight_approval",
    "formal_LQR_tree_funnel_region_of_attraction",
)
FEATURE_COLUMNS = (
    "continuation_valid_rate",
    "episode_terminal_useful_rate",
    "hard_failure_rate",
    "minimum_wall_margin_min_m",
    "floor_margin_min_m",
    "ceiling_margin_min_m",
    "energy_residual_mean_m",
    "lift_dwell_mean_s",
    "saturation_fraction_mean",
    "w3_environment_mode_count",
)


@dataclass(frozen=True)
class PostW3ClusterConfig:
    input_root: Path = DEFAULT_INPUT_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    run_id: int = 1
    max_representatives_per_primitive: int = 4
    distance_stop_threshold: float = 0.15
    allow_retired_diagnostic: bool = False


def run_post_w3_cluster_merge(
    *,
    input_root: Path = DEFAULT_INPUT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_id: int = 1,
    max_representatives_per_primitive: int = 4,
    distance_stop_threshold: float = 0.15,
    allow_retired_diagnostic: bool = False,
) -> dict[str, object]:
    """Retired v4.7 diagnostic wrapper for the old single compact-library flow."""

    config = PostW3ClusterConfig(
        input_root=Path(input_root),
        output_root=Path(output_root),
        run_id=int(run_id),
        max_representatives_per_primitive=int(max_representatives_per_primitive),
        distance_stop_threshold=float(distance_stop_threshold),
        allow_retired_diagnostic=bool(allow_retired_diagnostic),
    )
    run_root = config.output_root / f"{config.run_id:03d}"
    for subdir in ("manifests", "metrics", "reports"):
        filesystem_path(run_root / subdir).mkdir(parents=True, exist_ok=True)
    if not config.allow_retired_diagnostic:
        blocked_reason = "retired_diagnostic_requires_explicit_allow_retired_diagnostic"
        _write_blocked_outputs(run_root=run_root, config=config, blocked_reason=blocked_reason)
        return _result_payload(run_root, "blocked", blocked_reason)

    blocked_reason = _input_blocked_reason(config.input_root)
    if blocked_reason:
        _write_blocked_outputs(run_root=run_root, config=config, blocked_reason=blocked_reason)
        return _result_payload(run_root, "blocked", blocked_reason)

    registry = _read_json(config.input_root / "manifests" / "w3_survivor_registry.json")
    variant_summary = pd.read_csv(filesystem_path(config.input_root / "metrics" / "w3_variant_survival_summary.csv"))
    survived = variant_summary[variant_summary["w3_variant_status"].astype(str) == "survived"].copy()
    if survived.empty:
        _write_blocked_outputs(run_root=run_root, config=config, blocked_reason="w3_registry_has_no_surviving_variants")
        return _result_payload(run_root, "blocked", "w3_registry_has_no_surviving_variants")

    representative_rows, cluster_rows = _select_representatives(
        survived,
        max_per_group=config.max_representatives_per_primitive,
        distance_stop_threshold=config.distance_stop_threshold,
        source_roots={
            "source_w01_root": str(registry.get("source_w01_root", "")),
            "source_w2_root": str(registry.get("source_w2_root", "")),
            "source_w3_root": str(registry.get("source_w3_root", config.input_root.as_posix())),
        },
    )
    representative_frame = pd.DataFrame(representative_rows)
    cluster_frame = pd.DataFrame(cluster_rows)
    failure_summary = _failure_cluster_summary(variant_summary)
    boundary_summary = _boundary_terminal_summary(variant_summary)
    library = _compact_library_payload(
        config=config,
        registry=registry,
        representatives=representative_rows,
    )
    manifest = _cluster_manifest(
        config=config,
        run_root=run_root,
        registry=registry,
        representative_count=len(representative_rows),
        group_count=int(cluster_frame["cluster_group_id"].nunique()) if not cluster_frame.empty else 0,
    )

    _write_json(run_root / "manifests" / "post_w3_cluster_manifest.json", manifest)
    _write_json(run_root / "manifests" / "final_compact_primitive_library.json", library)
    _write_csv(run_root / "metrics" / "post_w3_cluster_summary.csv", cluster_frame)
    _write_csv(run_root / "metrics" / "post_w3_representative_library.csv", representative_frame)
    _write_csv(run_root / "metrics" / "post_w3_failure_cluster_summary.csv", failure_summary)
    _write_csv(run_root / "metrics" / "post_w3_boundary_terminal_summary.csv", boundary_summary)
    _write_file_size_audit(run_root)
    _write_reports(run_root=run_root, manifest=manifest, library=library)
    return {
        "status": "complete",
        "run_root": run_root.as_posix(),
        "representative_count": int(len(representative_rows)),
        "compact_library": (run_root / "manifests" / "final_compact_primitive_library.json").as_posix(),
        "l10_move_on_check": (run_root / "reports" / "l10_move_on_check.md").as_posix(),
    }


def _input_blocked_reason(input_root: Path) -> str:
    path = input_root.as_posix()
    if "w3_survival" not in path and "R7_survival" not in path:
        return "input_root_is_not_W3_survival_root"
    registry_path = filesystem_path(input_root / "manifests" / "w3_survivor_registry.json")
    summary_path = filesystem_path(input_root / "metrics" / "w3_variant_survival_summary.csv")
    if not registry_path.is_file():
        return "missing_w3_survivor_registry"
    if not summary_path.is_file():
        return "missing_w3_variant_survival_summary"
    try:
        registry = json.loads(registry_path.read_text(encoding="ascii"))
    except Exception as exc:
        return f"unreadable_w3_survivor_registry:{type(exc).__name__}"
    if str(registry.get("status", "")) != "w3_survivors_available":
        return "w3_survivor_registry_not_available"
    if int(registry.get("survivor_count", 0)) <= 0:
        return "w3_survivor_registry_has_zero_survivors"
    return ""


def _select_representatives(
    survived: pd.DataFrame,
    *,
    max_per_group: int,
    distance_stop_threshold: float,
    source_roots: dict[str, str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    representative_rows: list[dict[str, object]] = []
    cluster_rows: list[dict[str, object]] = []
    for (primitive_id, entry_role), group in survived.groupby(["primitive_id", "entry_role"], sort=True):
        group = group.copy().reset_index(drop=True)
        features = _normalised_features(group)
        scores = _representative_scores(group)
        selected_indices = _greedy_representative_indices(
            features,
            scores,
            max_count=max(1, int(max_per_group)),
            distance_stop_threshold=float(distance_stop_threshold),
        )
        cluster_group_id = f"postw3_{primitive_id}_{entry_role}"
        for local_rank, index in enumerate(selected_indices):
            row = group.iloc[int(index)].to_dict()
            cluster_id = f"{cluster_group_id}_r{local_rank:02d}"
            representative = _representative_row(
                row,
                cluster_id=cluster_id,
                rank=local_rank,
                score=float(scores[int(index)]),
                source_roots=source_roots,
            )
            representative_rows.append(representative)
        cluster_rows.append(
            {
                "cluster_group_id": cluster_group_id,
                "primitive_id": primitive_id,
                "entry_role": entry_role,
                "surviving_variant_count": int(len(group)),
                "representative_count": int(len(selected_indices)),
                "max_representatives_per_primitive": int(max_per_group),
                "distance_stop_threshold": float(distance_stop_threshold),
                "mean_continuation_valid_rate": float(pd.to_numeric(group["continuation_valid_rate"], errors="coerce").mean()),
                "mean_episode_terminal_useful_rate": float(pd.to_numeric(group["episode_terminal_useful_rate"], errors="coerce").mean()),
                "mean_hard_failure_rate": float(pd.to_numeric(group["hard_failure_rate"], errors="coerce").mean()),
            }
        )
    return representative_rows, cluster_rows


def _normalised_features(group: pd.DataFrame) -> np.ndarray:
    values = group.reindex(columns=FEATURE_COLUMNS, fill_value=0.0).apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if values.size == 0:
        return np.zeros((len(group), 1), dtype=float)
    mins = values.min(axis=0)
    spans = np.maximum(values.max(axis=0) - mins, 1e-9)
    return (values - mins) / spans


def _representative_scores(group: pd.DataFrame) -> np.ndarray:
    continuation = pd.to_numeric(group["continuation_valid_rate"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    terminal = pd.to_numeric(group["episode_terminal_useful_rate"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    hard = pd.to_numeric(group["hard_failure_rate"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    energy = pd.to_numeric(group["energy_residual_mean_m"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    dwell = pd.to_numeric(group["lift_dwell_mean_s"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    margin = pd.to_numeric(group["minimum_wall_margin_min_m"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return continuation + 0.35 * terminal - 0.75 * hard + 0.05 * energy + 0.03 * dwell + 0.02 * margin


def _greedy_representative_indices(
    features: np.ndarray,
    scores: np.ndarray,
    *,
    max_count: int,
    distance_stop_threshold: float,
) -> list[int]:
    if len(scores) == 0:
        return []
    selected = [int(np.argmax(scores))]
    while len(selected) < min(int(max_count), len(scores)):
        distances = _distance_to_selected(features, selected)
        candidate_index = int(np.argmax(distances))
        if float(distances[candidate_index]) < float(distance_stop_threshold):
            break
        selected.append(candidate_index)
    return selected


def _distance_to_selected(features: np.ndarray, selected_indices: list[int]) -> np.ndarray:
    selected = features[selected_indices, :]
    distances = np.linalg.norm(features[:, None, :] - selected[None, :, :], axis=2)
    return distances.min(axis=1)


def _representative_row(
    row: dict[str, object],
    *,
    cluster_id: str,
    rank: int,
    score: float,
    source_roots: dict[str, str],
) -> dict[str, object]:
    primitive_variant_id = str(row.get("primitive_variant_id", ""))
    library_id = _compact_library_id(primitive_variant_id, cluster_id)
    return {
        "compact_library_id": library_id,
        "primitive_variant_id": primitive_variant_id,
        "primitive_id": str(row.get("primitive_id", "")),
        "entry_role": str(row.get("entry_role", "")),
        "controller_id": str(row.get("controller_id", "")),
        "reference_state_vector": str(row.get("reference_state_vector", "")),
        "reference_command_vector": str(row.get("reference_command_vector", "")),
        "finite_horizon_s": float(row.get("finite_horizon_s", 0.0)),
        "Q_weight_json": str(row.get("Q_weight_json", "")),
        "R_weight_json": str(row.get("R_weight_json", "")),
        "K_gain_checksum": str(row.get("K_gain_checksum", "")),
        "augmented_A_checksum": str(row.get("augmented_A_checksum", "")),
        "augmented_B_checksum": str(row.get("augmented_B_checksum", "")),
        "augmented_gain_checksum": str(row.get("augmented_gain_checksum", "")),
        "source_w01_root": source_roots.get("source_w01_root", ""),
        "source_w2_root": source_roots.get("source_w2_root", ""),
        "source_w3_root": source_roots.get("source_w3_root", ""),
        "cluster_id": cluster_id,
        "representative_rank": int(rank),
        "representative_score": float(score),
        "representative_reason": "highest_score_then_diverse_W3_survivor_representative",
        "continuation_valid_count": int(row.get("continuation_valid_count", 0)),
        "continuation_valid_rate": float(row.get("continuation_valid_rate", 0.0)),
        "episode_terminal_useful_count": int(row.get("episode_terminal_useful_count", 0)),
        "episode_terminal_useful_rate": float(row.get("episode_terminal_useful_rate", 0.0)),
        "hard_failure_count": int(row.get("hard_failure_count", 0)),
        "hard_failure_rate": float(row.get("hard_failure_rate", 0.0)),
        "expected_energy_residual_m": float(row.get("energy_residual_mean_m", 0.0)),
        "expected_updraft_gain_proxy_m": float(
            row.get("updraft_gain_proxy_mean_m", max(float(row.get("energy_residual_mean_m", 0.0)), 0.0))
        ),
        "expected_positive_specific_energy_gain_m": float(
            row.get("positive_specific_energy_gain_mean_m", max(float(row.get("energy_residual_mean_m", 0.0)), 0.0))
        ),
        "expected_lift_dwell_time_s": float(row.get("lift_dwell_mean_s", 0.0)),
        "minimum_wall_margin_min_m": float(row.get("minimum_wall_margin_min_m", 0.0)),
        "floor_margin_min_m": float(row.get("floor_margin_min_m", 0.0)),
        "ceiling_margin_min_m": float(row.get("ceiling_margin_min_m", 0.0)),
        "saturation_fraction_mean": float(row.get("saturation_fraction_mean", 0.0)),
        "known_failure_boundaries": str(row.get("status_reason", "")),
        "w3_environment_modes_seen": str(row.get("w3_environment_modes_seen", "")),
        "w3_variant_status": str(row.get("w3_variant_status", "")),
        "claim_status": "simulation_only_post_w3_representative",
        "mutation_status": "references_existing_frozen_variant_no_Q_R_K_reference_horizon_entry_role_ID_mutation",
    }


def _failure_cluster_summary(variant_summary: pd.DataFrame) -> pd.DataFrame:
    return (
        variant_summary.groupby(["primitive_id", "entry_role", "w3_variant_status"], dropna=False)
        .agg(
            variant_count=("primitive_variant_id", "size"),
            hard_failure_count=("hard_failure_count", "sum"),
            blocked_count=("blocked_count", "sum"),
            mean_hard_failure_rate=("hard_failure_rate", "mean"),
        )
        .reset_index()
    )


def _boundary_terminal_summary(variant_summary: pd.DataFrame) -> pd.DataFrame:
    return (
        variant_summary.groupby(["primitive_id", "entry_role", "w3_variant_status"], dropna=False)
        .agg(
            variant_count=("primitive_variant_id", "size"),
            continuation_valid_count=("continuation_valid_count", "sum"),
            episode_terminal_useful_count=("episode_terminal_useful_count", "sum"),
            mean_continuation_valid_rate=("continuation_valid_rate", "mean"),
            mean_episode_terminal_useful_rate=("episode_terminal_useful_rate", "mean"),
        )
        .reset_index()
    )


def _compact_library_payload(
    *,
    config: PostW3ClusterConfig,
    registry: dict[str, object],
    representatives: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "library_version": POST_W3_VERSION,
        "project_title_version": PROJECT_TITLE_VERSION,
        "run_id": int(config.run_id),
        "source_w3_root": str(registry.get("source_w3_root", config.input_root.as_posix())),
        "source_w2_root": str(registry.get("source_w2_root", "")),
        "source_w01_root": str(registry.get("source_w01_root", "")),
        "source_w3_survivor_registry_sha256": file_sha256(config.input_root / "manifests" / "w3_survivor_registry.json"),
        "representative_count": int(len(representatives)),
        "claim_status": "simulation_only_post_w3_compact_representative_library",
        "no_controller_mutation": True,
        "continuation_and_terminal_evidence_separate": True,
        "entry_role_regime_separation_policy": "retired_diagnostic_groups_by_primitive_id_and_entry_role_no_cross_role_merge",
        "blocked_claims": list(BLOCKED_CLAIMS),
        "representatives": representatives,
    }


def _cluster_manifest(
    *,
    config: PostW3ClusterConfig,
    run_root: Path,
    registry: dict[str, object],
    representative_count: int,
    group_count: int,
) -> dict[str, object]:
    return {
        "manifest_version": POST_W3_VERSION,
        "project_title_version": PROJECT_TITLE_VERSION,
        "status": "complete" if representative_count > 0 else "blocked",
        "run_id": int(config.run_id),
        "run_root": run_root.as_posix(),
        "source_w3_root": config.input_root.as_posix(),
        "source_w3_registry_status": str(registry.get("status", "")),
        "source_w3_survivor_count": int(registry.get("survivor_count", 0)),
        "representative_count": int(representative_count),
        "cluster_group_count": int(group_count),
        "max_representatives_per_primitive": int(config.max_representatives_per_primitive),
        "distance_stop_threshold": float(config.distance_stop_threshold),
        "representative_selection_policy": "highest_score_then_farthest_feature_distance_with_stop_threshold",
        "entry_role_regime_separation_policy": "retired_diagnostic_groups_by_primitive_id_and_entry_role_no_cross_role_merge",
        "no_controller_mutation": True,
        "claim_status": "simulation_only_post_w3_compression",
        "blocked_claims": list(BLOCKED_CLAIMS),
    }


def _write_blocked_outputs(*, run_root: Path, config: PostW3ClusterConfig, blocked_reason: str) -> None:
    manifest = {
        "manifest_version": POST_W3_VERSION,
        "project_title_version": PROJECT_TITLE_VERSION,
        "status": "blocked",
        "run_id": int(config.run_id),
        "run_root": run_root.as_posix(),
        "source_w3_root": config.input_root.as_posix(),
        "blocked_reason": blocked_reason,
        "no_controller_mutation": True,
        "blocked_claims": list(BLOCKED_CLAIMS),
    }
    _write_json(run_root / "manifests" / "post_w3_cluster_manifest.json", manifest)
    _write_json(
        run_root / "manifests" / "final_compact_primitive_library.json",
        {
            "library_version": POST_W3_VERSION,
            "status": "blocked",
            "representative_count": 0,
            "blocked_reason": blocked_reason,
            "representatives": [],
        },
    )
    _write_csv(run_root / "metrics" / "post_w3_cluster_summary.csv", pd.DataFrame())
    _write_csv(run_root / "metrics" / "post_w3_representative_library.csv", pd.DataFrame())
    _write_file_size_audit(run_root)
    _write_reports(run_root=run_root, manifest=manifest, library={"representative_count": 0, "representatives": []})


def _write_reports(*, run_root: Path, manifest: dict[str, object], library: dict[str, object]) -> None:
    representative_count = int(library.get("representative_count", 0))
    lines = [
        "# Post-W3 Compact Library Report",
        "",
        f"- Status: `{manifest.get('status', '')}`",
        f"- Representatives: `{representative_count}`",
        f"- Source W3 root: `{manifest.get('source_w3_root', '')}`",
        "- Compact library references real frozen W3-surviving variants only.",
        "- Q/R, K, references, horizons, entry roles, controller IDs, and primitive-variant IDs are not mutated.",
        "- Continuation-valid evidence remains separate from terminal-useful evidence.",
        "- Claim boundary: simulation-only compact representative library.",
        "",
    ]
    report = "\n".join(lines)
    filesystem_path(run_root / "reports" / "post_w3_cluster_report.md").write_text(report, encoding="ascii")
    filesystem_path(run_root / "reports" / "final_compact_library_report.md").write_text(report, encoding="ascii")
    blockers = _l10_blockers(run_root=run_root, library=library)
    l10 = [
        "# L10 Move-On Check",
        "",
        f"- Representatives available: `{representative_count > 0}`",
        f"- Compact library references W3-surviving variants only: `{not blockers}`",
        f"- File-size audit below 100 MB: `{not _file_size_blocked(run_root)}`",
        f"- Governor smoke allowed: `{not blockers and not _file_size_blocked(run_root)}`",
        "",
        "## Blockers",
        "",
        *[f"- `{item}`" for item in (blockers or ["none"])],
        "",
    ]
    filesystem_path(run_root / "reports" / "l10_move_on_check.md").write_text("\n".join(l10), encoding="ascii")


def _l10_blockers(*, run_root: Path, library: dict[str, object]) -> list[str]:
    blockers = []
    representatives = list(library.get("representatives", []))
    if not representatives:
        blockers.append("missing_post_w3_representatives")
    for row in representatives:
        if row.get("w3_variant_status") != "survived":
            blockers.append("compact_library_contains_non_surviving_variant")
            break
        if row.get("mutation_status") != "references_existing_frozen_variant_no_Q_R_K_reference_horizon_entry_role_ID_mutation":
            blockers.append("compact_library_mutation_status_invalid")
            break
    if _file_size_blocked(run_root):
        blockers.append("file_size_audit_over_100mb")
    return blockers


def _file_size_blocked(run_root: Path) -> bool:
    audit = filesystem_path(run_root / "metrics" / "file_size_audit.csv")
    if not audit.is_file():
        return True
    frame = pd.read_csv(audit)
    return bool((frame.get("above_100mb", pd.Series(dtype=bool)).astype(str).str.lower() == "true").any())


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
                "dense_table_partition": False,
            }
        )
    _write_csv(root / "metrics" / "file_size_audit.csv", pd.DataFrame(rows))


def _compact_library_id(primitive_variant_id: str, cluster_id: str) -> str:
    digest = hashlib.sha256(f"{primitive_variant_id}:{cluster_id}".encode("utf-8")).hexdigest()[:12]
    return f"postw3lib_{digest}"


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(filesystem_path(path).read_text(encoding="ascii"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    filesystem_path(path).parent.mkdir(parents=True, exist_ok=True)
    filesystem_path(path).write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="ascii")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    filesystem_path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(filesystem_path(path), index=False)


def _result_payload(run_root: Path, status: str, blocked_reason: str) -> dict[str, object]:
    return {
        "status": status,
        "run_root": run_root.as_posix(),
        "blocked_reason": blocked_reason,
        "compact_library": (run_root / "manifests" / "final_compact_primitive_library.json").as_posix(),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build post-W3 compact primitive library.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--max-representatives-per-primitive", type=int, default=4)
    parser.add_argument("--distance-stop-threshold", type=float, default=0.15)
    parser.add_argument("--allow-retired-diagnostic", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_post_w3_cluster_merge(
        input_root=args.input_root,
        output_root=args.output_root,
        run_id=args.run_id,
        max_representatives_per_primitive=args.max_representatives_per_primitive,
        distance_stop_threshold=args.distance_stop_threshold,
        allow_retired_diagnostic=args.allow_retired_diagnostic,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") != "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
