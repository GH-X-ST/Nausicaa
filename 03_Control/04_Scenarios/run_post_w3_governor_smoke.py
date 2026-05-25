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
from env_ctx import build_environment_context, context_feature_vector  # noqa: E402
from env_instance import environment_instance_for_mode, environment_metadata_from_instance  # noqa: E402
from env_surrogate import resolve_surrogate_binding, wind_field_for_binding  # noqa: E402
from primitive_variant_registry import start_family_is_compatible  # noqa: E402
from state_sampling import archive_state_sample_for_family  # noqa: E402


PROJECT_TITLE_VERSION = "LQR-Stabilised Contextual Primitive v4.7"
GOVERNOR_SMOKE_VERSION = "post_w3_outcome_governor_smoke_v1"
DEFAULT_COMPACT_LIBRARY = Path(
    "03_Control/05_Results/lqr_contextual_v1_0/post_w3_cluster/001/manifests/final_compact_primitive_library.json"
)
DEFAULT_OUTCOME_ROOT = Path("03_Control/05_Results/lqr_contextual_v1_0/outcome_model")
DEFAULT_GOVERNOR_ROOT = Path("03_Control/05_Results/lqr_contextual_v1_0/governor_smoke")
BLOCKED_CLAIMS = (
    "full_loop_validation_success",
    "hardware_readiness",
    "real_flight_transfer",
    "mission_success",
    "memory_governor_performance_improvement",
    "compact_library_real_flight_approval",
)


@dataclass(frozen=True)
class PostW3GovernorSmokeConfig:
    compact_library_path: Path = DEFAULT_COMPACT_LIBRARY
    outcome_output_root: Path = DEFAULT_OUTCOME_ROOT
    governor_output_root: Path = DEFAULT_GOVERNOR_ROOT
    run_id: int = 1
    seed: int = 47


def run_post_w3_governor_smoke(config: PostW3GovernorSmokeConfig) -> dict[str, object]:
    """Run simulation-only outcome-model and governor smoke over a compact library."""

    blocked_reason = _input_blocked_reason(config.compact_library_path)
    outcome_root = config.outcome_output_root / f"{config.run_id:03d}"
    governor_root = config.governor_output_root / f"{config.run_id:03d}"
    for root in (outcome_root, governor_root):
        for subdir in ("manifests", "metrics", "reports"):
            filesystem_path(root / subdir).mkdir(parents=True, exist_ok=True)
    if blocked_reason:
        _write_blocked_outputs(
            outcome_root=outcome_root,
            governor_root=governor_root,
            config=config,
            blocked_reason=blocked_reason,
        )
        return {
            "status": "blocked",
            "blocked_reason": blocked_reason,
            "outcome_root": outcome_root.as_posix(),
            "governor_root": governor_root.as_posix(),
        }

    library = _read_json(config.compact_library_path)
    representatives = list(library.get("representatives", []))
    outcome_rows = [_outcome_row(row) for row in representatives]
    context_rows = _deterministic_context_rows(config.seed)
    candidate_rows, selection_rows = _governor_rows(representatives, context_rows)
    rejection_summary = _rejection_summary(candidate_rows)

    _write_outcome_outputs(
        root=outcome_root,
        config=config,
        library=library,
        outcome_rows=outcome_rows,
    )
    _write_governor_outputs(
        root=governor_root,
        config=config,
        library=library,
        candidate_rows=candidate_rows,
        selection_rows=selection_rows,
        rejection_summary=rejection_summary,
    )
    return {
        "status": "complete",
        "outcome_root": outcome_root.as_posix(),
        "governor_root": governor_root.as_posix(),
        "candidate_rows": int(len(candidate_rows)),
        "selection_rows": int(len(selection_rows)),
    }


def _input_blocked_reason(library_path: Path) -> str:
    path = filesystem_path(library_path)
    if not path.is_file():
        return "missing_final_compact_primitive_library"
    try:
        library = json.loads(path.read_text(encoding="ascii"))
    except Exception as exc:
        return f"unreadable_final_compact_primitive_library:{type(exc).__name__}"
    if int(library.get("representative_count", 0)) <= 0:
        return "compact_library_has_no_representatives"
    if not bool(library.get("no_controller_mutation", False)):
        return "compact_library_no_mutation_gate_failed"
    l10_path = library_path.parent.parent / "reports" / "l10_move_on_check.md"
    if filesystem_path(l10_path).is_file():
        text = filesystem_path(l10_path).read_text(encoding="ascii", errors="ignore")
        if "Governor smoke allowed: `False`" in text:
            return "l10_move_on_check_blocks_governor_smoke"
    return ""


def _outcome_row(row: dict[str, object]) -> dict[str, object]:
    wall_margin = float(row.get("minimum_wall_margin_min_m", 0.0))
    if wall_margin < 0.0:
        margin_class = "unsafe_negative_margin"
    elif wall_margin < 0.10:
        margin_class = "tight_margin"
    else:
        margin_class = "margin_available"
    return {
        "compact_library_id": str(row.get("compact_library_id", "")),
        "primitive_variant_id": str(row.get("primitive_variant_id", "")),
        "primitive_id": str(row.get("primitive_id", "")),
        "entry_role": str(row.get("entry_role", "")),
        "controller_id": str(row.get("controller_id", "")),
        "continuation_probability": float(row.get("continuation_valid_rate", 0.0)),
        "terminal_useful_probability": float(row.get("episode_terminal_useful_rate", 0.0)),
        "hard_failure_risk": float(row.get("hard_failure_rate", 0.0)),
        "expected_energy_residual_m": float(row.get("expected_energy_residual_m", 0.0)),
        "expected_lift_dwell_time_s": float(row.get("expected_lift_dwell_time_s", 0.0)),
        "minimum_wall_margin_class": margin_class,
        "known_failure_boundaries": str(row.get("known_failure_boundaries", "")),
        "claim_status": "simulation_only_W3_derived_outcome_smoke",
    }


def _deterministic_context_rows(seed: int) -> list[dict[str, object]]:
    specs = [
        ("W0", "dry_air", "launch_gate", "ctx_w0_launch"),
        ("W1", "gaussian_single", "inflight_nominal", "ctx_w1_nominal"),
        ("W2", "annular_gp_single", "inflight_lift_region", "ctx_w2_lift"),
        ("W3", "w3_randomised_four", "inflight_boundary_near", "ctx_w3_boundary"),
    ]
    rows = []
    for index, (layer, mode, family, label) in enumerate(specs):
        sample = archive_state_sample_for_family(
            start_state_family=family,
            paired_start_key=f"postw3gov_{index:03d}_{family}",
            sample_index=index,
            seed=int(seed),
            W_layer=layer,
            environment_mode=mode,
        )
        instance = environment_instance_for_mode(layer, mode, int(seed) + index)
        metadata = environment_metadata_from_instance(instance)
        binding = resolve_surrogate_binding(layer, metadata, randomisation_seed=int(seed) + index)
        wind = wind_field_for_binding(binding)
        context = build_environment_context(
            sample.state_vector,
            wind_field=wind,
            metadata=metadata,
            latency_case="none" if layer == "W0" else "nominal",
            actuator_case="nominal",
            surrogate_binding=binding,
        )
        rows.append(
            {
                "context_id": label,
                "W_layer": layer,
                "environment_mode": mode,
                "start_state_family": family,
                "context": context,
                "context_feature_vector": json.dumps([float(value) for value in context_feature_vector(context)]),
                "wall_margin_m": float(context.wall_margin_m),
                "floor_margin_m": float(context.floor_margin_m),
                "ceiling_margin_m": float(context.ceiling_margin_m),
                "speed_margin_m_s": float(context.speed_margin_m_s),
                "attitude_margin_rad": float(context.attitude_margin_rad),
            }
        )
    return rows


def _governor_rows(
    representatives: list[dict[str, object]],
    context_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    candidate_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    for context_row in context_rows:
        for governor_mode in ("continuation_mode", "terminal_episode_mode"):
            mode_candidates = []
            for representative in representatives:
                decision = _candidate_decision(representative, context_row, governor_mode)
                candidate_rows.append(decision)
                mode_candidates.append(decision)
            viable = [row for row in mode_candidates if row["viable"]]
            selected = max(viable, key=lambda item: float(item["score"])) if viable else None
            selection_rows.append(
                {
                    "context_id": context_row["context_id"],
                    "W_layer": context_row["W_layer"],
                    "environment_mode": context_row["environment_mode"],
                    "start_state_family": context_row["start_state_family"],
                    "governor_mode": governor_mode,
                    "candidate_count": int(len(mode_candidates)),
                    "viable_count": int(len(viable)),
                    "decision_status": "selected_compact_representative" if selected else "blocked_no_viable_representative",
                    "selected_compact_library_id": "" if selected is None else selected["compact_library_id"],
                    "selected_primitive_variant_id": "" if selected is None else selected["primitive_variant_id"],
                    "selected_primitive_id": "" if selected is None else selected["primitive_id"],
                    "claim_status": "simulation_only_governor_smoke_no_performance_claim",
                }
            )
    return candidate_rows, selection_rows


def _candidate_decision(
    representative: dict[str, object],
    context_row: dict[str, object],
    governor_mode: str,
) -> dict[str, object]:
    rejection_reason = _rejection_reason(representative, context_row, governor_mode)
    continuation_probability = float(representative.get("continuation_valid_rate", 0.0))
    terminal_probability = float(representative.get("episode_terminal_useful_rate", 0.0))
    hard_risk = float(representative.get("hard_failure_rate", 1.0))
    if governor_mode == "terminal_episode_mode":
        score = terminal_probability + 0.10 * continuation_probability - 0.40 * hard_risk
    else:
        score = continuation_probability - 0.35 * terminal_probability - 0.45 * hard_risk
    return {
        "context_id": context_row["context_id"],
        "W_layer": context_row["W_layer"],
        "environment_mode": context_row["environment_mode"],
        "start_state_family": context_row["start_state_family"],
        "governor_mode": governor_mode,
        "compact_library_id": str(representative.get("compact_library_id", "")),
        "primitive_variant_id": str(representative.get("primitive_variant_id", "")),
        "primitive_id": str(representative.get("primitive_id", "")),
        "entry_role": str(representative.get("entry_role", "")),
        "controller_id": str(representative.get("controller_id", "")),
        "viable": rejection_reason == "",
        "rejection_reason": rejection_reason,
        "score": float(score if rejection_reason == "" else float("-inf")),
        "continuation_probability": continuation_probability,
        "terminal_useful_probability": terminal_probability,
        "hard_failure_risk": hard_risk,
        "wall_margin_m": float(context_row["wall_margin_m"]),
        "floor_margin_m": float(context_row["floor_margin_m"]),
        "ceiling_margin_m": float(context_row["ceiling_margin_m"]),
        "speed_margin_m_s": float(context_row["speed_margin_m_s"]),
        "claim_status": "simulation_only_governor_candidate_smoke",
    }


def _rejection_reason(
    representative: dict[str, object],
    context_row: dict[str, object],
    governor_mode: str,
) -> str:
    if not start_family_is_compatible(
        entry_role=str(representative.get("entry_role", "")),
        start_state_family=str(context_row["start_state_family"]),
    ):
        return "entry_role_incompatible_start_family"
    if float(context_row["wall_margin_m"]) < 0.05:
        return "context_wall_margin_low"
    if float(context_row["floor_margin_m"]) < 0.0 or float(context_row["ceiling_margin_m"]) < 0.0:
        return "context_vertical_safety_violation"
    if float(context_row["speed_margin_m_s"]) < 0.0:
        return "context_speed_margin_low"
    if not representative.get("augmented_gain_checksum", ""):
        return "timing_payload_checksum_missing"
    hard_risk = float(representative.get("hard_failure_rate", 1.0))
    continuation_probability = float(representative.get("continuation_valid_rate", 0.0))
    terminal_probability = float(representative.get("episode_terminal_useful_rate", 0.0))
    if hard_risk > 0.75:
        return "known_hard_failure_boundary_high"
    if governor_mode == "continuation_mode" and continuation_probability <= 0.0:
        return "continuation_probability_zero"
    if governor_mode == "terminal_episode_mode" and max(terminal_probability, continuation_probability) <= 0.0:
        return "terminal_and_continuation_probability_zero"
    return ""


def _rejection_summary(candidate_rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(candidate_rows)
    if frame.empty:
        return pd.DataFrame()
    rejected = frame[frame["rejection_reason"].astype(str) != ""]
    if rejected.empty:
        return pd.DataFrame(columns=["governor_mode", "context_id", "rejection_reason", "row_count"])
    return (
        rejected.groupby(["governor_mode", "context_id", "rejection_reason"], dropna=False)
        .size()
        .reset_index(name="row_count")
    )


def _write_outcome_outputs(
    *,
    root: Path,
    config: PostW3GovernorSmokeConfig,
    library: dict[str, object],
    outcome_rows: list[dict[str, object]],
) -> None:
    _write_csv(root / "metrics" / "outcome_model_summary.csv", pd.DataFrame(outcome_rows))
    manifest = {
        "manifest_version": GOVERNOR_SMOKE_VERSION,
        "project_title_version": PROJECT_TITLE_VERSION,
        "status": "complete",
        "run_id": int(config.run_id),
        "compact_library_path": config.compact_library_path.as_posix(),
        "source_scope": "W3_derived_simulation_evidence_only",
        "representative_count": int(len(outcome_rows)),
        "source_compact_library_id": str(library.get("library_version", "")),
        "claim_status": "simulation_only_outcome_model_smoke",
        "blocked_claims": list(BLOCKED_CLAIMS),
    }
    _write_json(root / "manifests" / "outcome_model_manifest.json", manifest)
    _write_file_size_audit(root)
    report = [
        "# Outcome Model Smoke Report",
        "",
        f"- Representatives: `{len(outcome_rows)}`",
        "- Model backend: `interpretable_W3_summary_table`",
        "- Continuation and terminal-useful probabilities are separate.",
        "- Claim boundary: simulation-only smoke; no full-loop, hardware, real-flight, or mission claim.",
        "",
    ]
    filesystem_path(root / "reports" / "outcome_model_report.md").write_text("\n".join(report), encoding="ascii")


def _write_governor_outputs(
    *,
    root: Path,
    config: PostW3GovernorSmokeConfig,
    library: dict[str, object],
    candidate_rows: list[dict[str, object]],
    selection_rows: list[dict[str, object]],
    rejection_summary: pd.DataFrame,
) -> None:
    _write_csv(root / "metrics" / "governor_candidate_log.csv", pd.DataFrame(candidate_rows))
    _write_csv(root / "metrics" / "governor_selection_summary.csv", pd.DataFrame(selection_rows))
    _write_csv(root / "metrics" / "governor_rejection_summary.csv", rejection_summary)
    manifest = {
        "manifest_version": GOVERNOR_SMOKE_VERSION,
        "project_title_version": PROJECT_TITLE_VERSION,
        "status": "complete",
        "run_id": int(config.run_id),
        "compact_library_path": config.compact_library_path.as_posix(),
        "source_compact_library_id": str(library.get("library_version", "")),
        "candidate_row_count": int(len(candidate_rows)),
        "selection_row_count": int(len(selection_rows)),
        "governor_modes": ["continuation_mode", "terminal_episode_mode"],
        "claim_status": "simulation_only_governor_smoke_no_performance_claim",
        "blocked_claims": list(BLOCKED_CLAIMS),
    }
    _write_json(root / "manifests" / "governor_smoke_manifest.json", manifest)
    _write_file_size_audit(root)
    report = [
        "# Governor Smoke Report",
        "",
        f"- Candidate rows: `{len(candidate_rows)}`",
        f"- Selection rows: `{len(selection_rows)}`",
        "- Rejection reasons are logged for unsafe or incompatible representatives.",
        "- Terminal-useful evidence is retained as terminal episode evidence, not continuation success.",
        "- Claim boundary: no full-loop validation, hardware readiness, transfer, mission, or improvement claim.",
        "",
    ]
    filesystem_path(root / "reports" / "governor_smoke_report.md").write_text("\n".join(report), encoding="ascii")


def _write_blocked_outputs(
    *,
    outcome_root: Path,
    governor_root: Path,
    config: PostW3GovernorSmokeConfig,
    blocked_reason: str,
) -> None:
    _write_json(
        outcome_root / "manifests" / "outcome_model_manifest.json",
        {
            "manifest_version": GOVERNOR_SMOKE_VERSION,
            "status": "blocked",
            "run_id": int(config.run_id),
            "blocked_reason": blocked_reason,
            "blocked_claims": list(BLOCKED_CLAIMS),
        },
    )
    _write_json(
        governor_root / "manifests" / "governor_smoke_manifest.json",
        {
            "manifest_version": GOVERNOR_SMOKE_VERSION,
            "status": "blocked",
            "run_id": int(config.run_id),
            "blocked_reason": blocked_reason,
            "blocked_claims": list(BLOCKED_CLAIMS),
        },
    )
    _write_csv(outcome_root / "metrics" / "outcome_model_summary.csv", pd.DataFrame())
    _write_csv(governor_root / "metrics" / "governor_rejection_summary.csv", pd.DataFrame())
    _write_csv(governor_root / "metrics" / "governor_selection_summary.csv", pd.DataFrame())
    _write_file_size_audit(outcome_root)
    _write_file_size_audit(governor_root)


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
    parser = argparse.ArgumentParser(description="Run post-W3 outcome-model and governor smoke.")
    parser.add_argument("--compact-library", dest="compact_library_path", type=Path, default=DEFAULT_COMPACT_LIBRARY)
    parser.add_argument("--outcome-output-root", type=Path, default=DEFAULT_OUTCOME_ROOT)
    parser.add_argument("--governor-output-root", type=Path, default=DEFAULT_GOVERNOR_ROOT)
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=47)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_post_w3_governor_smoke(
        PostW3GovernorSmokeConfig(
            compact_library_path=args.compact_library_path,
            outcome_output_root=args.outcome_output_root,
            governor_output_root=args.governor_output_root,
            run_id=args.run_id,
            seed=args.seed,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") != "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
