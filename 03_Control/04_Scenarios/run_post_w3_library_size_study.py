from __future__ import annotations

import argparse
import hashlib
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
from dense_archive_table_io import file_sha256, filesystem_path  # noqa: E402
from primitive_timing_contract import primitive_timing_contract_row, primitive_timing_contract_status  # noqa: E402


PROJECT_TITLE_VERSION = "LQR-Stabilised Contextual Primitive v5.0"
POST_W3_LIBRARY_STUDY_VERSION = "post_w3_library_size_study_v411"
DEFAULT_W3_DISCOVERY_ROOT = Path("03_Control/05_Results/lqr_contextual_v1_0/w3_survival")
DEFAULT_INPUT_ROOT: Path | None = None
DEFAULT_OUTPUT_ROOT = Path("03_Control/05_Results/lqr_contextual_v1_0/post_w3_library_size_study")
LIBRARY_SIZE_CASES: tuple[dict[str, object], ...] = (
    {
        "library_size_case_id": "heavy_cluster",
        "library_size_human_label": "heavy clustering and merging",
        "max_representatives_per_group": 1,
        "selection_policy": "top_score_per_primitive_entry_role",
    },
    {
        "library_size_case_id": "balanced_cluster",
        "library_size_human_label": "balanced clustering and merging",
        "max_representatives_per_group": 3,
        "selection_policy": "top_score_then_diverse_per_primitive_entry_role",
    },
    {
        "library_size_case_id": "light_cluster",
        "library_size_human_label": "light clustering and merging",
        "max_representatives_per_group": 6,
        "selection_policy": "broad_top_score_then_diverse_per_primitive_entry_role",
    },
    {
        "library_size_case_id": "no_cluster_no_merge",
        "library_size_human_label": "no-clustering/no-merging",
        "max_representatives_per_group": 1_000_000,
        "selection_policy": "all_w3_survivors_no_clustering_no_merging",
    },
)
LIBRARY_SIZE_CASE_IDS = tuple(str(case["library_size_case_id"]) for case in LIBRARY_SIZE_CASES)
BLOCKED_CLAIMS = (
    "hardware_readiness",
    "real_flight_transfer",
    "mission_success",
    "memory_improvement",
    "full_loop_validation_success",
)


@dataclass(frozen=True)
class PostW3LibrarySizeStudyConfig:
    input_root: Path | None = DEFAULT_INPUT_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    run_id: int = 1


def run_post_w3_library_size_study(config: PostW3LibrarySizeStudyConfig) -> dict[str, object]:
    """Build the four v5.0 post-W3 library-size cases from W3 survivors."""

    config = PostW3LibrarySizeStudyConfig(
        input_root=_resolve_w3_input_root(config.input_root),
        output_root=config.output_root,
        run_id=config.run_id,
    )
    run_root = Path(config.output_root) / f"{int(config.run_id):03d}"
    for subdir in ("manifests", "metrics", "reports"):
        filesystem_path(run_root / subdir).mkdir(parents=True, exist_ok=True)
    blocked_reason = _input_blocked_reason(config.input_root)
    if blocked_reason:
        _write_blocked_outputs(run_root, config, blocked_reason)
        return {"status": "blocked", "blocked_reason": blocked_reason, "run_root": run_root.as_posix()}

    registry = _read_json(config.input_root / "manifests" / "w3_survivor_registry.json")
    variant_summary = pd.read_csv(filesystem_path(config.input_root / "metrics" / "w3_variant_survival_summary.csv"))
    survived = variant_summary[variant_summary["w3_variant_status"].astype(str) == "survived"].copy()
    blocked_reason = _survived_frame_blocked_reason(survived)
    if blocked_reason:
        _write_blocked_outputs(run_root, config, blocked_reason)
        return {"status": "blocked", "blocked_reason": blocked_reason, "run_root": run_root.as_posix()}

    all_representatives: list[dict[str, object]] = []
    case_manifest_rows: list[dict[str, object]] = []
    for case in LIBRARY_SIZE_CASES:
        case_id = str(case["library_size_case_id"])
        representatives = _representatives_for_case(
            survived,
            case=case,
            source_roots={
                "source_w01_root": str(registry.get("source_w01_root", "")),
                "source_w2_root": str(registry.get("source_w2_root", "")),
                "source_w3_root": str(registry.get("source_w3_root", config.input_root.as_posix())),
            },
        )
        all_representatives.extend(representatives)
        library_payload = _library_payload(config=config, registry=registry, case=case, representatives=representatives)
        _write_json(run_root / "manifests" / f"{case_id}_primitive_library.json", library_payload)
        _write_csv(run_root / "metrics" / f"{case_id}_representative_library.csv", pd.DataFrame(representatives))
        case_manifest_rows.append(
            {
                "library_size_case_id": case_id,
                "library_size_human_label": str(case["library_size_human_label"]),
                "representative_count": int(len(representatives)),
                "selection_policy": str(case["selection_policy"]),
                "library_manifest": f"manifests/{case_id}_primitive_library.json",
                "library_table": f"metrics/{case_id}_representative_library.csv",
            }
        )
    summary = pd.DataFrame(case_manifest_rows)
    _write_csv(run_root / "metrics" / "library_size_case_summary.csv", summary)
    _write_csv(run_root / "metrics" / "post_w3_representative_library_all_cases.csv", pd.DataFrame(all_representatives))
    manifest = _study_manifest(config=config, run_root=run_root, registry=registry, case_rows=case_manifest_rows)
    _write_json(run_root / "manifests" / "post_w3_library_size_study_manifest.json", manifest)
    _write_file_size_audit(run_root)
    _write_report(run_root, manifest)
    return {
        "status": "complete",
        "run_root": run_root.as_posix(),
        "library_size_case_ids": list(LIBRARY_SIZE_CASE_IDS),
        "representative_count": int(len(all_representatives)),
        "manifest": (run_root / "manifests" / "post_w3_library_size_study_manifest.json").as_posix(),
    }


def library_size_case_by_id(case_id: str) -> dict[str, object]:
    for case in LIBRARY_SIZE_CASES:
        if str(case["library_size_case_id"]) == str(case_id):
            return dict(case)
    raise KeyError(f"unknown library_size_case_id: {case_id}")


def discover_latest_w3_root_for_post_w3(discovery_root: Path = DEFAULT_W3_DISCOVERY_ROOT) -> Path | None:
    root = filesystem_path(discovery_root)
    if not root.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        try:
            numeric_id = int(path.name)
        except ValueError:
            continue
        candidate = Path(path)
        if _input_blocked_reason(candidate):
            continue
        try:
            survived = pd.read_csv(filesystem_path(candidate / "metrics" / "w3_variant_survival_summary.csv"))
            survived = survived[survived["w3_variant_status"].astype(str) == "survived"].copy()
        except Exception:
            continue
        if _survived_frame_blocked_reason(survived):
            continue
        candidates.append((numeric_id, candidate))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[-1][1]


def _resolve_w3_input_root(input_root: Path | None) -> Path:
    if input_root is not None:
        return Path(input_root)
    discovered = discover_latest_w3_root_for_post_w3()
    if discovered is not None:
        return discovered
    return DEFAULT_W3_DISCOVERY_ROOT / "__missing_eligible_w3_root__"


def _input_blocked_reason(input_root: Path) -> str:
    root = Path(input_root)
    if "w3_survival" not in root.as_posix():
        return "input_root_is_not_W3_survival_root"
    registry_path = filesystem_path(root / "manifests" / "w3_survivor_registry.json")
    summary_path = filesystem_path(root / "metrics" / "w3_variant_survival_summary.csv")
    source_manifest_path = filesystem_path(root / "manifests" / "w3_survival_manifest.json")
    if not registry_path.is_file():
        return "missing_w3_survivor_registry"
    if not summary_path.is_file():
        return "missing_w3_variant_survival_summary"
    if not source_manifest_path.is_file():
        return "missing_w3_survival_manifest"
    try:
        registry = json.loads(registry_path.read_text(encoding="ascii"))
        source_manifest = json.loads(source_manifest_path.read_text(encoding="ascii"))
    except Exception as exc:
        return f"unreadable_w3_survival_metadata:{type(exc).__name__}"
    if str(registry.get("project_title_version", "")) != PROJECT_TITLE_VERSION:
        return "w3_survivor_registry_not_v5_project_title"
    if str(source_manifest.get("project_title_version", "")) != PROJECT_TITLE_VERSION:
        return "w3_survival_manifest_not_v5_project_title"
    if bool(source_manifest.get("test_fixture_not_method_evidence", False)):
        return "w3_survival_fixture_not_method_evidence"
    if str(source_manifest.get("method_evidence_level", "")) not in {"w3_dense_survival_pass", "complete"}:
        return "w3_survival_not_dense_method_evidence"
    if str(registry.get("status", "")) != "w3_survivors_available":
        return "w3_survivor_registry_not_available"
    if int(registry.get("survivor_count", 0)) <= 0:
        return "w3_survivor_registry_has_zero_survivors"
    return ""


def _survived_frame_blocked_reason(survived: pd.DataFrame) -> str:
    if survived.empty:
        return "w3_registry_has_no_surviving_variants"
    required = (
        "finite_horizon_s",
        "controller_input_slots_per_primitive",
        "controller_input_update_period_s",
        "primitive_timing_contract_version",
    )
    missing = [name for name in required if name not in survived.columns]
    if missing:
        return "w3_survivor_summary_missing_" + "_".join(missing)
    for row in survived.to_dict(orient="records"):
        status, reason = primitive_timing_contract_status(
            finite_horizon_s=row.get("finite_horizon_s", 0.0),
            controller_input_slots_per_primitive=row.get("controller_input_slots_per_primitive", 5),
            controller_input_update_period_s=row.get("controller_input_update_period_s", 0.020),
            primitive_timing_contract_version=row.get("primitive_timing_contract_version", "legacy_not_recorded"),
        )
        if status != "compliant":
            return f"w3_survivor_timing_contract_noncompliant:{reason}"
    return ""


def _representatives_for_case(
    survived: pd.DataFrame,
    *,
    case: dict[str, object],
    source_roots: dict[str, str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    max_per_group = int(case["max_representatives_per_group"])
    for (primitive_id, entry_role), group in survived.groupby(["primitive_id", "entry_role"], sort=True):
        scored = group.copy()
        scored["_representative_score"] = _representative_score(scored)
        selected = scored.sort_values(
            by=["_representative_score", "primitive_variant_id"],
            ascending=[False, True],
        ).head(max_per_group)
        for rank, (_, row) in enumerate(selected.iterrows()):
            rows.append(_representative_row(row.to_dict(), case=case, source_roots=source_roots, rank=rank))
    return rows


def _representative_score(group: pd.DataFrame) -> pd.Series:
    continuation = pd.to_numeric(group.get("continuation_valid_rate", 0.0), errors="coerce").fillna(0.0)
    terminal = pd.to_numeric(group.get("episode_terminal_useful_rate", 0.0), errors="coerce").fillna(0.0)
    hard = pd.to_numeric(group.get("hard_failure_rate", 1.0), errors="coerce").fillna(1.0)
    energy = pd.to_numeric(group.get("energy_residual_mean_m", 0.0), errors="coerce").fillna(0.0)
    dwell = pd.to_numeric(group.get("lift_dwell_mean_s", 0.0), errors="coerce").fillna(0.0)
    return continuation + 0.35 * terminal - 0.75 * hard + 0.05 * energy + 0.03 * dwell


def _representative_row(
    row: dict[str, object],
    *,
    case: dict[str, object],
    source_roots: dict[str, str],
    rank: int,
) -> dict[str, object]:
    case_id = str(case["library_size_case_id"])
    variant_id = str(row.get("primitive_variant_id", ""))
    cluster_id = f"{case_id}_{row.get('primitive_id', '')}_{row.get('entry_role', '')}_r{int(rank):03d}"
    timing = primitive_timing_contract_row()
    return {
        "compact_library_id": _compact_library_id(case_id, variant_id, cluster_id),
        "library_size_case_id": case_id,
        "library_size_human_label": str(case["library_size_human_label"]),
        "selection_policy": str(case["selection_policy"]),
        "primitive_variant_id": variant_id,
        "primitive_id": str(row.get("primitive_id", "")),
        "entry_role": str(row.get("entry_role", "")),
        "controller_id": str(row.get("controller_id", "")),
        "reference_state_vector": str(row.get("reference_state_vector", "")),
        "reference_command_vector": str(row.get("reference_command_vector", "")),
        "finite_horizon_s": float(row.get("finite_horizon_s", timing["finite_horizon_s"])),
        "controller_input_slots_per_primitive": int(
            float(row.get("controller_input_slots_per_primitive", timing["controller_input_slots_per_primitive"]))
        ),
        "controller_input_update_period_s": float(
            row.get("controller_input_update_period_s", timing["controller_input_update_period_s"])
        ),
        "primitive_timing_contract_version": str(
            row.get("primitive_timing_contract_version", timing["primitive_timing_contract_version"])
        ),
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
        "representative_score": float(row.get("_representative_score", 0.0)),
        "continuation_valid_count": int(float(row.get("continuation_valid_count", 0))),
        "continuation_valid_rate": float(row.get("continuation_valid_rate", 0.0)),
        "episode_terminal_useful_count": int(float(row.get("episode_terminal_useful_count", 0))),
        "episode_terminal_useful_rate": float(row.get("episode_terminal_useful_rate", 0.0)),
        "hard_failure_count": int(float(row.get("hard_failure_count", 0))),
        "hard_failure_rate": float(row.get("hard_failure_rate", 0.0)),
        "expected_energy_residual_m": float(row.get("energy_residual_mean_m", 0.0)),
        "expected_lift_dwell_time_s": float(row.get("lift_dwell_mean_s", 0.0)),
        "minimum_wall_margin_min_m": float(row.get("minimum_wall_margin_min_m", 0.0)),
        "floor_margin_min_m": float(row.get("floor_margin_min_m", 0.0)),
        "ceiling_margin_min_m": float(row.get("ceiling_margin_min_m", 0.0)),
        "saturation_fraction_mean": float(row.get("saturation_fraction_mean", 0.0)),
        "known_failure_boundaries": str(row.get("status_reason", "")),
        "w3_environment_modes_seen": str(row.get("w3_environment_modes_seen", "")),
        "w3_variant_status": str(row.get("w3_variant_status", "")),
        "claim_status": "simulation_only_post_w3_library_size_case_representative",
        "mutation_status": "references_existing_frozen_variant_no_Q_R_K_reference_horizon_entry_role_ID_mutation",
    }


def _library_payload(
    *,
    config: PostW3LibrarySizeStudyConfig,
    registry: dict[str, object],
    case: dict[str, object],
    representatives: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "library_version": POST_W3_LIBRARY_STUDY_VERSION,
        "project_title_version": PROJECT_TITLE_VERSION,
        "run_id": int(config.run_id),
        "library_size_case_id": str(case["library_size_case_id"]),
        "library_size_human_label": str(case["library_size_human_label"]),
        "source_w3_root": str(registry.get("source_w3_root", config.input_root.as_posix())),
        "source_w2_root": str(registry.get("source_w2_root", "")),
        "source_w01_root": str(registry.get("source_w01_root", "")),
        "source_w3_survivor_registry_sha256": file_sha256(config.input_root / "manifests" / "w3_survivor_registry.json"),
        "representative_count": int(len(representatives)),
        "claim_status": "simulation_only_post_w3_library_size_case",
        "no_controller_mutation": True,
        "continuation_and_terminal_evidence_separate": True,
        "primitive_timing_contract": primitive_timing_contract_row(),
        "blocked_claims": list(BLOCKED_CLAIMS),
        "representatives": representatives,
    }


def _study_manifest(
    *,
    config: PostW3LibrarySizeStudyConfig,
    run_root: Path,
    registry: dict[str, object],
    case_rows: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "manifest_version": POST_W3_LIBRARY_STUDY_VERSION,
        "project_title_version": PROJECT_TITLE_VERSION,
        "status": "complete",
        "run_id": int(config.run_id),
        "run_root": run_root.as_posix(),
        "source_w3_root": config.input_root.as_posix(),
        "source_w3_registry_status": str(registry.get("status", "")),
        "source_w3_survivor_count": int(registry.get("survivor_count", 0)),
        "library_size_case_ids": list(LIBRARY_SIZE_CASE_IDS),
        "library_size_cases": case_rows,
        "primitive_timing_contract": primitive_timing_contract_row(),
        "claim_status": "simulation_only_post_w3_library_size_study",
        "blocked_claims": list(BLOCKED_CLAIMS),
    }


def _write_blocked_outputs(run_root: Path, config: PostW3LibrarySizeStudyConfig, blocked_reason: str) -> None:
    manifest = {
        "manifest_version": POST_W3_LIBRARY_STUDY_VERSION,
        "project_title_version": PROJECT_TITLE_VERSION,
        "status": "blocked",
        "run_id": int(config.run_id),
        "run_root": run_root.as_posix(),
        "source_w3_root": config.input_root.as_posix(),
        "blocked_reason": blocked_reason,
        "library_size_case_ids": list(LIBRARY_SIZE_CASE_IDS),
        "blocked_claims": list(BLOCKED_CLAIMS),
    }
    _write_json(run_root / "manifests" / "post_w3_library_size_study_manifest.json", manifest)
    _write_csv(run_root / "metrics" / "library_size_case_summary.csv", pd.DataFrame())
    _write_file_size_audit(run_root)
    _write_report(run_root, manifest)


def _write_report(run_root: Path, manifest: dict[str, object]) -> None:
    lines = [
        "# v5.0 Post-W3 Library-Size Study",
        "",
        f"- Status: `{manifest.get('status', '')}`",
        f"- Library-size cases: `{','.join(LIBRARY_SIZE_CASE_IDS)}`",
        "- Human label retained for no_cluster_no_merge: `no-clustering/no-merging`",
        "- Claim boundary: simulation-only; no hardware-readiness, transfer, mission, or memory-improvement claim.",
        "",
    ]
    if manifest.get("blocked_reason"):
        lines.insert(4, f"- Blocked reason: `{manifest['blocked_reason']}`")
    filesystem_path(run_root / "reports" / "post_w3_library_size_study_report.md").write_text(
        "\n".join(lines),
        encoding="ascii",
    )


def _write_file_size_audit(root: Path) -> None:
    rows = []
    root_fs = filesystem_path(root)
    for path in sorted(root_fs.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root_fs).as_posix()
        size_mb = float(path.stat().st_size) / float(1024 * 1024)
        rows.append(
            {
                "relative_path": rel,
                "byte_count": int(path.stat().st_size),
                "size_mb": size_mb,
                "above_75mb": bool(size_mb > 75.0),
                "above_100mb": bool(size_mb > MAX_GENERATED_FILE_SIZE_MB),
                "push_allowed": bool(size_mb <= MAX_GENERATED_FILE_SIZE_MB),
            }
        )
    _write_csv(root / "metrics" / "file_size_audit.csv", pd.DataFrame(rows))


def _compact_library_id(case_id: str, variant_id: str, cluster_id: str) -> str:
    digest = hashlib.sha256(f"{case_id}|{variant_id}|{cluster_id}".encode("ascii")).hexdigest()[:12]
    return f"v411lib_{case_id}_{digest}"


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(filesystem_path(path).read_text(encoding="ascii"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    filesystem_path(path).parent.mkdir(parents=True, exist_ok=True)
    filesystem_path(path).write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="ascii")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    filesystem_path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(filesystem_path(path), index=False)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v5.0 four-case post-W3 library-size study.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_post_w3_library_size_study(
        PostW3LibrarySizeStudyConfig(
            input_root=args.input_root,
            output_root=args.output_root,
            run_id=args.run_id,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {"complete", "blocked"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
