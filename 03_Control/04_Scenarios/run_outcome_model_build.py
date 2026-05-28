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
from dense_archive_table_io import filesystem_path  # noqa: E402
from context_conditioned_outcome import CONTEXT_CONDITIONED_OUTCOME_MODEL_VERSION  # noqa: E402
from primitive_timing_contract import primitive_timing_contract_row  # noqa: E402
from run_post_w3_library_size_study import LIBRARY_SIZE_CASE_IDS, POST_W3_LIBRARY_STUDY_VERSION  # noqa: E402


PROJECT_TITLE_VERSION = "LQR-Stabilised Contextual Primitive v5.20"
OUTCOME_MODEL_VERSION = "v53_context_conditioned_library_size_outcome_model_v2"
DEFAULT_COMPACT_LIBRARY = Path(
    "03_Control/05_Results/lqr_contextual_v1_0/post_w3_library_size_study/001/manifests/post_w3_library_size_study_manifest.json"
)
DEFAULT_OUTPUT_ROOT = Path("03_Control/05_Results/lqr_contextual_v1_0/outcome_model")
BLOCKED_CLAIMS = (
    "hardware_readiness",
    "real_flight_transfer",
    "mission_success",
    "memory_improves_performance_without_full_loop_evidence",
    "formal_LQR_tree_funnel_region_of_attraction",
    "post_hoc_controller_retuning",
)


@dataclass(frozen=True)
class OutcomeModelBuildConfig:
    compact_library_path: Path = DEFAULT_COMPACT_LIBRARY
    output_root: Path = DEFAULT_OUTPUT_ROOT
    run_id: int = 3
    library_size_case_id: str = "balanced_cluster"


def run_outcome_model_build(config: OutcomeModelBuildConfig) -> dict[str, object]:
    """Build the v5.20 interpretable W3-derived outcome model table."""

    run_root = Path(config.output_root) / f"{int(config.run_id):03d}"
    for subdir in ("manifests", "metrics", "reports"):
        filesystem_path(run_root / subdir).mkdir(parents=True, exist_ok=True)
    blocked_reason = _blocked_reason(config.compact_library_path)
    if blocked_reason:
        _write_blocked_outputs(run_root, config, blocked_reason)
        return {"status": "blocked", "blocked_reason": blocked_reason, "run_root": run_root.as_posix()}

    libraries = _load_libraries(config.compact_library_path, default_case_id=config.library_size_case_id)
    representatives = [row for library in libraries for row in list(library.get("representatives", []))]
    case_ids = sorted({str(library.get("library_size_case_id", config.library_size_case_id)) for library in libraries})
    rows = [
        row
        for library in libraries
        for row in build_outcome_model_rows(
            list(library.get("representatives", [])),
            library_size_case_id=str(library.get("library_size_case_id", config.library_size_case_id)),
        )
    ]
    frame = pd.DataFrame(rows)
    _write_csv(run_root / "metrics" / "outcome_model_table.csv", frame)
    _write_csv(run_root / "metrics" / "outcome_model_summary.csv", frame)
    manifest = {
        "manifest_version": OUTCOME_MODEL_VERSION,
        "project_title_version": PROJECT_TITLE_VERSION,
        "status": "complete",
        "run_id": int(config.run_id),
        "run_root": run_root.as_posix(),
        "compact_library_path": config.compact_library_path.as_posix(),
        "source_compact_library_version": ",".join(sorted({str(library.get("library_version", "")) for library in libraries})),
        "library_size_case_ids": case_ids,
        "library_size_case_id": ",".join(case_ids),
        "library_size_human_label": ",".join(
            sorted({str(library.get("library_size_human_label", "")) for library in libraries if library.get("library_size_human_label")})
        ),
        "representative_count": int(len(representatives)),
        "outcome_row_count": int(len(rows)),
        "prediction_source": "W3_summary_interpretable",
        "runtime_context_conditioning_required": True,
        "runtime_context_conditioning_version": CONTEXT_CONDITIONED_OUTCOME_MODEL_VERSION,
        "runtime_context_conditioning_policy": "robust_downgrade_only_never_optimistic",
        "continuation_and_terminal_evidence_separate": True,
        "controller_mutation_allowed": False,
        "claim_status": "simulation_only_outcome_model_for_full_loop_validation",
        "blocked_claims": list(BLOCKED_CLAIMS),
    }
    _write_json(run_root / "manifests" / "outcome_model_manifest.json", manifest)
    _write_file_size_audit(run_root)
    _write_report(run_root, manifest)
    return {
        "status": "complete",
        "run_root": run_root.as_posix(),
        "outcome_row_count": int(len(rows)),
        "outcome_model_table": (run_root / "metrics" / "outcome_model_table.csv").as_posix(),
    }


def build_outcome_model_rows(
    representatives: list[dict[str, object]],
    *,
    library_size_case_id: str = "balanced_cluster",
) -> list[dict[str, object]]:
    """Return one prediction row per compact representative."""

    rows = []
    timing = primitive_timing_contract_row()
    for representative in representatives:
        wall_margin = _float(representative.get("minimum_wall_margin_min_m", 0.0))
        floor_margin = _float(representative.get("floor_margin_min_m", 0.0))
        ceiling_margin = _float(representative.get("ceiling_margin_min_m", 0.0))
        rows.append(
            {
                "compact_library_id": str(representative.get("compact_library_id", "")),
                "library_size_case_id": str(representative.get("library_size_case_id", library_size_case_id)),
                "library_size_human_label": str(representative.get("library_size_human_label", "")),
                "primitive_variant_id": str(representative.get("primitive_variant_id", "")),
                "transition_object_id": str(representative.get("transition_object_id", "")),
                "primitive_id": str(representative.get("primitive_id", "")),
                "entry_role": str(representative.get("entry_role", "")),
                "transition_entry_class": str(representative.get("transition_entry_class", "")),
                "controller_id": str(representative.get("controller_id", "")),
                "finite_horizon_s": float(representative.get("finite_horizon_s", timing["finite_horizon_s"])),
                "controller_input_slots_per_primitive": int(
                    float(
                        representative.get(
                            "controller_input_slots_per_primitive",
                            timing["controller_input_slots_per_primitive"],
                        )
                    )
                ),
                "controller_input_update_period_s": float(
                    representative.get(
                        "controller_input_update_period_s",
                        timing["controller_input_update_period_s"],
                    )
                ),
                "primitive_timing_contract_version": str(
                    representative.get(
                        "primitive_timing_contract_version",
                        timing["primitive_timing_contract_version"],
                    )
                ),
                "primitive_timing_contract_status": "compliant",
                "transition_success_probability": _clamp_probability(
                    representative.get(
                        "transition_success_probability",
                        representative.get("transition_chain_compatible_rate", 0.0),
                    )
                ),
                "transition_chain_compatible_rate": _clamp_probability(
                    representative.get("transition_chain_compatible_rate", representative.get("transition_success_probability", 0.0))
                ),
                "transition_exit_classes_seen": str(representative.get("transition_exit_classes_seen", "")),
                "transition_pairs_seen": str(representative.get("transition_pairs_seen", "")),
                "continuation_probability": _clamp_probability(
                    representative.get(
                        "transition_success_probability",
                        representative.get("transition_chain_compatible_rate", representative.get("continuation_valid_rate", 0.0)),
                    )
                ),
                "terminal_useful_probability": _clamp_probability(representative.get("episode_terminal_useful_rate", 0.0)),
                "hard_failure_risk": _clamp_probability(representative.get("hard_failure_rate", 1.0)),
                "expected_energy_residual_m": _float(representative.get("expected_energy_residual_m", 0.0)),
                "expected_updraft_gain_proxy_m": _float(
                    representative.get(
                        "expected_updraft_gain_proxy_m",
                        max(_float(representative.get("expected_positive_specific_energy_gain_m", 0.0)), 0.0),
                    )
                ),
                "expected_lift_dwell_time_s": _float(representative.get("expected_lift_dwell_time_s", 0.0)),
                "minimum_wall_margin_min_m": wall_margin,
                "floor_margin_min_m": floor_margin,
                "ceiling_margin_min_m": ceiling_margin,
                "saturation_fraction_mean": _float(representative.get("saturation_fraction_mean", 0.0)),
                "known_failure_boundaries": str(representative.get("known_failure_boundaries", "")),
                "environment_coverage": str(representative.get("w3_environment_modes_seen", "")),
                "runtime_context_conditioning_required": True,
                "runtime_context_conditioning_version": CONTEXT_CONDITIONED_OUTCOME_MODEL_VERSION,
                "runtime_context_conditioning_policy": "robust_downgrade_only_never_optimistic",
                "context_conditioning_keys": (
                    "library_size_case_id;compact_library_id;start_state_family;environment_class;"
                    "local_w_wing;local_uncertainty;margins;environment_block;active_fan_count"
                ),
                "sample_count": int(
                    _float(representative.get("continuation_valid_count", 0.0))
                    + _float(representative.get("episode_terminal_useful_count", 0.0))
                    + _float(representative.get("hard_failure_count", 0.0))
                ),
                "margin_class": _margin_class(wall_margin, floor_margin, ceiling_margin),
                "prediction_source": "W3_summary_interpretable",
                "claim_status": "simulation_only_outcome_model",
            }
        )
    return rows


def _blocked_reason(compact_library_path: Path) -> str:
    path = filesystem_path(compact_library_path)
    if not path.is_file():
        return "missing_final_compact_primitive_library"
    try:
        library = json.loads(path.read_text(encoding="ascii"))
    except Exception as exc:
        return f"unreadable_final_compact_primitive_library:{type(exc).__name__}"
    if str(library.get("manifest_version", "")) == POST_W3_LIBRARY_STUDY_VERSION:
        if str(library.get("project_title_version", "")) != PROJECT_TITLE_VERSION:
            return "post_w3_library_size_study_not_v5_project_title"
        case_ids = {str(case.get("library_size_case_id", "")) for case in library.get("library_size_cases", [])}
        if case_ids != set(LIBRARY_SIZE_CASE_IDS):
            return "post_w3_library_size_study_missing_active_five_case_set"
        for case in library.get("library_size_cases", []):
            case_path = path.parent / Path(str(case.get("library_manifest", ""))).name
            if not case_path.is_file():
                return f"missing_library_size_case_file:{case.get('library_size_case_id', '')}"
        return ""
    if str(library.get("library_version", "")) not in {
        POST_W3_LIBRARY_STUDY_VERSION,
        "post_w3_compact_representative_library_v1",
    }:
        return "unsupported_compact_library_version"
    if str(library.get("library_version", "")) == "post_w3_compact_representative_library_v1":
        return "retired_single_compact_library_not_active_five_case_study"
    if not library.get("library_size_case_id"):
        return "missing_library_size_case_id"
    if int(library.get("representative_count", 0)) <= 0:
        return "compact_library_has_no_representatives"
    if not bool(library.get("no_controller_mutation", False)):
        return "compact_library_no_controller_mutation_gate_failed"
    if not bool(library.get("continuation_and_terminal_evidence_separate", False)):
        return "compact_library_terminal_continuation_separation_missing"
    for representative in library.get("representatives", []):
        if str(representative.get("w3_variant_status", "")) != "survived":
            return "compact_library_contains_non_survived_executable_representative"
        for key in (
            "controller_id",
            "primitive_variant_id",
            "K_gain_checksum",
            "augmented_gain_checksum",
            "source_w2_root",
            "library_size_case_id",
        ):
            if not representative.get(key):
                return f"compact_representative_missing_{key}"
    return ""


def _load_libraries(compact_library_path: Path, *, default_case_id: str) -> list[dict[str, object]]:
    path = filesystem_path(compact_library_path)
    payload = json.loads(path.read_text(encoding="ascii"))
    if str(payload.get("manifest_version", "")) == POST_W3_LIBRARY_STUDY_VERSION:
        libraries = []
        for case in payload.get("library_size_cases", []):
            case_path = path.parent / Path(str(case.get("library_manifest", ""))).name
            libraries.append(_read_json(case_path))
        return libraries
    payload.setdefault("library_size_case_id", default_case_id)
    return [payload]


def _margin_class(wall_margin: float, floor_margin: float, ceiling_margin: float) -> str:
    if min(float(wall_margin), float(floor_margin), float(ceiling_margin)) < 0.0:
        return "unsafe_negative_margin"
    if min(float(wall_margin), float(floor_margin), float(ceiling_margin)) < 0.10:
        return "tight_margin"
    return "margin_available"


def _clamp_probability(value: object) -> float:
    return float(max(0.0, min(1.0, _float(value, 0.0))))


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _write_blocked_outputs(run_root: Path, config: OutcomeModelBuildConfig, blocked_reason: str) -> None:
    _write_json(
        run_root / "manifests" / "outcome_model_manifest.json",
        {
            "manifest_version": OUTCOME_MODEL_VERSION,
            "project_title_version": PROJECT_TITLE_VERSION,
            "status": "blocked",
            "run_id": int(config.run_id),
            "blocked_reason": blocked_reason,
            "claim_status": "blocked_before_full_loop_validation",
            "blocked_claims": list(BLOCKED_CLAIMS),
        },
    )
    _write_csv(run_root / "metrics" / "outcome_model_table.csv", pd.DataFrame())
    _write_file_size_audit(run_root)


def _write_report(run_root: Path, manifest: dict[str, object]) -> None:
    report = [
        "# v5.20 Outcome Model Report",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Representatives: `{manifest['representative_count']}`",
        "- Prediction source: `W3_summary_interpretable`",
        "- Continuation and terminal-useful probabilities remain separate.",
        "- No Q/R, K, reference, horizon, entry-role, controller-ID, or variant-ID mutation is performed.",
        "- Claim boundary: simulation-only full-loop prediction input.",
        "",
    ]
    filesystem_path(run_root / "reports" / "outcome_model_report.md").write_text("\n".join(report), encoding="ascii")


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


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(filesystem_path(path).read_text(encoding="ascii"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    filesystem_path(path).parent.mkdir(parents=True, exist_ok=True)
    filesystem_path(path).write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="ascii")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    filesystem_path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(filesystem_path(path), index=False)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v5.20 library-size-case W3-derived outcome model.")
    parser.add_argument("--compact-library", dest="compact_library_path", type=Path, default=DEFAULT_COMPACT_LIBRARY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", type=int, default=2)
    parser.add_argument("--library-size-case-id", default="balanced_cluster")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_outcome_model_build(
        OutcomeModelBuildConfig(
            compact_library_path=args.compact_library_path,
            output_root=args.output_root,
            run_id=args.run_id,
            library_size_case_id=args.library_size_case_id,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") != "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
