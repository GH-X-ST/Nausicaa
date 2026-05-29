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
from dense_archive_table_io import file_sha256, filesystem_path, load_table_manifest, read_table_partition  # noqa: E402
from lqr_linearisation import lqr_speed_bin_id  # noqa: E402
from prim_cat import ACTIVE_PRIMITIVE_IDS  # noqa: E402
from primitive_timing_contract import primitive_timing_contract_row, primitive_timing_contract_status  # noqa: E402
from transition_labels import transition_contract_row  # noqa: E402


PROJECT_TITLE_VERSION = "LQR-Stabilised Contextual Primitive v5.20"
POST_W3_LIBRARY_STUDY_VERSION = "post_w3_library_size_study_v54_uncertainty_block_coverage_medoid"
DEFAULT_W3_DISCOVERY_ROOT = Path("03_Control/05_Results/R7_survival")
DEFAULT_INPUT_ROOT: Path | None = None
DEFAULT_OUTPUT_ROOT = Path("03_Control/05_Results/R8_library_size_study")
LIBRARY_SIZE_CASES: tuple[dict[str, object], ...] = (
    {
        "library_size_case_id": "heavy_cluster",
        "library_size_human_label": "heavy clustering and merging",
        "max_representatives_per_group": 1,
        "selection_policy": "coverage_medoid_best_worst_case_per_primitive_transition_entry",
    },
    {
        "library_size_case_id": "balanced_cluster",
        "library_size_human_label": "balanced clustering and merging",
        "max_representatives_per_group": 3,
        "selection_policy": "coverage_medoid_greedy_marginal_top_3_per_primitive_transition_entry",
    },
    {
        "library_size_case_id": "light_cluster",
        "library_size_human_label": "light clustering and merging",
        "max_representatives_per_group": 6,
        "selection_policy": "coverage_medoid_greedy_marginal_top_6_per_primitive_transition_entry",
    },
    {
        "library_size_case_id": "super_light_cluster",
        "library_size_human_label": "super-light clustering and merging",
        "max_representatives_per_group": 12,
        "selection_policy": "coverage_medoid_greedy_marginal_top_12_per_primitive_transition_entry",
    },
    {
        "library_size_case_id": "no_cluster_no_merge",
        "library_size_human_label": "no-clustering/no-merging",
        "max_representatives_per_group": 1_000_000,
        "selection_policy": "all_w3_eligible_transition_objects_no_clustering_no_merging",
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
    run_label: str = ""


def run_post_w3_library_size_study(config: PostW3LibrarySizeStudyConfig) -> dict[str, object]:
    """Build the five v5.20 post-W3 library-size cases from W3 survivors."""

    config = PostW3LibrarySizeStudyConfig(
        input_root=_resolve_w3_input_root(config.input_root),
        output_root=config.output_root,
        run_id=config.run_id,
        run_label=config.run_label,
    )
    run_root = Path(config.output_root) / _run_folder_name(config.run_id, config.run_label)
    for subdir in ("manifests", "metrics", "reports"):
        filesystem_path(run_root / subdir).mkdir(parents=True, exist_ok=True)
    blocked_reason = _input_blocked_reason(config.input_root)
    if blocked_reason:
        _write_blocked_outputs(run_root, config, blocked_reason)
        return {"status": "blocked", "blocked_reason": blocked_reason, "run_root": run_root.as_posix()}

    registry = _read_json(config.input_root / "manifests" / "w3_survivor_registry.json")
    variant_summary = pd.read_csv(filesystem_path(config.input_root / "metrics" / "w3_variant_survival_summary.csv"))
    survived = _eligible_w3_transition_objects(variant_summary)
    blocked_reason = _survived_frame_blocked_reason(survived)
    if blocked_reason:
        _write_blocked_outputs(run_root, config, blocked_reason)
        return {"status": "blocked", "blocked_reason": blocked_reason, "run_root": run_root.as_posix()}
    robustness_profiles = _robustness_profile_frame(config.input_root)
    if not robustness_profiles.empty:
        merge_columns = ["primitive_variant_id"]
        if "transition_entry_class" in survived.columns and "transition_entry_class" in robustness_profiles.columns:
            merge_columns.append("transition_entry_class")
        survived = survived.merge(robustness_profiles, on=merge_columns, how="left")

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
    _write_csv(run_root / "metrics" / "post_w3_robustness_profile.csv", robustness_profiles)
    _write_csv(run_root / "metrics" / "library_size_case_summary.csv", summary)
    _write_csv(run_root / "metrics" / "post_w3_representative_library_all_cases.csv", pd.DataFrame(all_representatives))
    _write_csv(run_root / "metrics" / "coverage_medoid_selection_audit.csv", _coverage_medoid_selection_audit(all_representatives))
    speed_bin_audit, speed_bin_blockers = _speed_bin_coverage_audit(survived, all_representatives)
    _write_csv(run_root / "metrics" / "speed_bin_coverage_audit.csv", speed_bin_audit)
    availability, availability_blockers = _launch_gate_candidate_availability(all_representatives)
    _write_csv(run_root / "metrics" / "launch_gate_candidate_availability.csv", availability)
    _write_csv(run_root / "metrics" / "launch_gate_entry_role_audit.csv", _launch_gate_entry_role_audit(all_representatives))
    if speed_bin_blockers:
        blocked_reason = "speed_bin_coverage_preservation_failed:" + ";".join(speed_bin_blockers)
        _write_blocked_outputs(run_root, config, blocked_reason)
        return {"status": "blocked", "blocked_reason": blocked_reason, "run_root": run_root.as_posix()}
    if availability_blockers:
        blocked_reason = "launch_gate_candidate_availability_failed:" + ";".join(availability_blockers)
        _write_blocked_outputs(run_root, config, blocked_reason)
        return {"status": "blocked", "blocked_reason": blocked_reason, "run_root": run_root.as_posix()}
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
            survived = _eligible_w3_transition_objects(
                pd.read_csv(filesystem_path(candidate / "metrics" / "w3_variant_survival_summary.csv"))
            )
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


def _eligible_w3_transition_objects(variant_summary: pd.DataFrame) -> pd.DataFrame:
    """Return W3 objects allowed into R8 while preserving their status labels."""

    if variant_summary.empty:
        return variant_summary.copy()
    if "eligible_for_post_w3_library_size_study" in variant_summary.columns:
        eligible = variant_summary["eligible_for_post_w3_library_size_study"].astype(str).str.lower().isin({"true", "1", "yes"})
        return variant_summary[eligible].copy()
    return variant_summary[variant_summary["w3_variant_status"].astype(str) == "survived"].copy()


def _input_blocked_reason(input_root: Path) -> str:
    root = Path(input_root)
    if "w3_survival" not in root.as_posix() and "R7_survival" not in root.as_posix():
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
    if "transition_entry_class" not in survived.columns:
        return "w3_survivor_summary_missing_transition_entry_class"
    if "primitive_id" not in survived.columns:
        return "w3_survivor_summary_missing_primitive_id"
    if "transition_chain_compatible_rate" not in survived.columns:
        return "w3_survivor_summary_missing_transition_chain_compatible_rate"
    if "transition_success_probability" not in survived.columns:
        return "w3_survivor_summary_missing_transition_success_probability"
    transition_rate = pd.to_numeric(survived["transition_chain_compatible_rate"], errors="coerce").fillna(0.0)
    if bool((transition_rate <= 0.0).any()):
        return "w3_survivor_summary_contains_non_transition_compatible_survivor"
    launch = survived[survived["transition_entry_class"].astype(str) == "launch_gate"]
    if launch.empty:
        return "w3_registry_has_no_launch_gate_entry_survivors"
    missing_launch = sorted(set(ACTIVE_PRIMITIVE_IDS) - set(launch["primitive_id"].astype(str)))
    if missing_launch:
        return "w3_registry_missing_launch_gate_entry_survivors:" + ",".join(missing_launch)
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


def _robustness_profile_frame(input_root: Path) -> pd.DataFrame:
    rows = _read_w3_evidence_rows(input_root)
    if rows.empty or "primitive_variant_id" not in rows.columns:
        return pd.DataFrame()
    compatible = rows[_bool_series(rows.get("entry_role_compatible", pd.Series(False, index=rows.index)))].copy()
    if compatible.empty:
        return pd.DataFrame()
    compatible = _ensure_local_speed_bin_columns(compatible)
    compatible["_positive"] = _transition_positive_series(compatible)
    compatible["_terminal_useful"] = _terminal_series(compatible)
    compatible["_hard_failure"] = _transition_hard_failure_series(compatible)
    labels = _profile_labels(compatible)
    profile_rows: list[dict[str, object]] = []
    group_columns = ["primitive_variant_id"]
    if "transition_entry_class" in compatible.columns:
        group_columns.append("transition_entry_class")
    for group_key, group in compatible.groupby(group_columns, sort=True, dropna=False):
        if isinstance(group_key, tuple):
            variant_id = str(group_key[0])
            transition_entry_class = str(group_key[1])
        else:
            variant_id = str(group_key)
            transition_entry_class = str(group.iloc[0].get("transition_entry_class", ""))
        coverage_rates = [_coverage_rate_for_label(group, label) for label in labels]
        terminal_rates = [_terminal_rate_for_label(group, label) for label in labels]
        hard_rates = [_hard_rate_for_label(group, label) for label in labels]
        profile_rows.append(
            {
                "primitive_variant_id": str(variant_id),
                "transition_object_id": f"{variant_id}__entry_{transition_entry_class or 'unknown'}",
                "transition_entry_class": transition_entry_class,
                "robustness_profile_version": "coverage_behavior_qr_medoid_profile_v1",
                "robustness_profile_row_count": int(len(group)),
                "robustness_profile_axis_count": int(len(labels)),
                "robustness_coverage_labels_json": json.dumps(labels, separators=(",", ":")),
                "robustness_coverage_rates_json": json.dumps(coverage_rates, separators=(",", ":")),
                "robustness_terminal_rates_json": json.dumps(terminal_rates, separators=(",", ":")),
                "robustness_hard_failure_rates_json": json.dumps(hard_rates, separators=(",", ":")),
                "robustness_worst_case_coverage": float(min(coverage_rates) if coverage_rates else 0.0),
                "robustness_mean_coverage": float(sum(coverage_rates) / max(1, len(coverage_rates))),
                "robustness_max_hard_failure_rate": float(max(hard_rates) if hard_rates else 0.0),
                "robustness_environment_modes_seen": _unique_join(group, "environment_mode"),
                "robustness_start_families_seen": _unique_join(group, "start_state_family"),
                "robustness_active_fan_counts_seen": _unique_join(group, "scheduled_active_fan_count"),
                "robustness_evidence_blocks_seen": _unique_join(group, "r7_evidence_block_id"),
                "robustness_uncertainty_tiers_seen": _unique_join(group, "r7_uncertainty_tier"),
                "robustness_active_fan_policies_seen": _unique_join(group, "r7_active_fan_count_policy"),
                "robustness_fan_position_policies_seen": _unique_join(group, "r7_fan_position_policy"),
                "robustness_speed_bins_seen": _unique_join(group, "local_lqr_speed_bin_id"),
                "updraft_gain_proxy_mean_m": _mean_or_zero(group, "updraft_specific_energy_gain_proxy_m"),
                "positive_specific_energy_gain_mean_m": _mean_positive_energy_gain(group),
                "rollout_duration_mean_s": _mean_or_zero(group, "rollout_duration_s"),
            }
        )
    return pd.DataFrame(profile_rows)


def _read_w3_evidence_rows(input_root: Path) -> pd.DataFrame:
    try:
        manifest = load_table_manifest(Path(input_root) / "manifests" / "table_manifest.json")
    except Exception:
        return pd.DataFrame()
    frames = []
    for partition in manifest.tables:
        try:
            frames.append(
                read_table_partition(
                    Path(input_root) / "tables" / partition.relative_path,
                    storage_format=partition.storage_format,
                )
            )
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _ensure_local_speed_bin_columns(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["local_lqr_speed_bin_id"] = [
        _speed_bin_from_row(row)
        for row in work.to_dict(orient="records")
    ]
    if "local_lqr_reference_speed_m_s" not in work.columns:
        work["local_lqr_reference_speed_m_s"] = [
            _speed_from_row(row)
            for row in work.to_dict(orient="records")
        ]
    return work


def _speed_bin_from_row(row: dict[str, object] | pd.Series) -> str:
    for key in ("local_lqr_speed_bin_id", "variant_local_lqr_speed_bin_id"):
        value = str(row.get(key, "")).strip()
        if value and value.lower() != "nan":
            return value
    speed = _speed_from_row(row)
    if speed > 0.0:
        return lqr_speed_bin_id(speed)
    return ""


def _speed_from_row(row: dict[str, object] | pd.Series) -> float:
    for key in ("local_lqr_reference_speed_m_s", "variant_local_lqr_reference_speed_m_s"):
        value = row.get(key, "")
        text = str(value).strip()
        if text and text.lower() != "nan":
            return _float(value, default=0.0)
    return 0.0


def _profile_labels(frame: pd.DataFrame) -> list[str]:
    labels: list[str] = []
    for column, prefix in (
        ("r7_evidence_block_id", "r7_block"),
        ("r7_uncertainty_tier", "tier"),
        ("r7_active_fan_count_policy", "fan_policy"),
        ("r7_fan_position_policy", "fan_position_policy"),
        ("environment_mode", "env"),
        ("start_state_family", "start"),
        ("scheduled_active_fan_count", "active_fan_count"),
        ("local_lqr_speed_bin_id", "speed_bin"),
        ("transition_pair", "transition"),
        ("transition_exit_class", "exit"),
    ):
        if column not in frame.columns:
            continue
        values = sorted(str(value) for value in frame[column].fillna("unknown").astype(str).unique())
        labels.extend(f"{prefix}:{value}" for value in values)
    return labels or ["all:compatible"]


def _coverage_rate_for_label(group: pd.DataFrame, label: str) -> float:
    subset = _profile_subset(group, label)
    if subset.empty:
        return 0.0
    return float(subset["_positive"].mean())


def _terminal_rate_for_label(group: pd.DataFrame, label: str) -> float:
    subset = _profile_subset(group, label)
    if subset.empty:
        return 0.0
    return float(subset["_terminal_useful"].mean())


def _hard_rate_for_label(group: pd.DataFrame, label: str) -> float:
    subset = _profile_subset(group, label)
    if subset.empty:
        return 0.0
    return float(subset["_hard_failure"].mean())


def _profile_subset(group: pd.DataFrame, label: str) -> pd.DataFrame:
    if label == "all:compatible":
        return group
    try:
        prefix, value = label.split(":", 1)
    except ValueError:
        return pd.DataFrame()
    column = {
        "env": "environment_mode",
        "r7_block": "r7_evidence_block_id",
        "tier": "r7_uncertainty_tier",
        "fan_policy": "r7_active_fan_count_policy",
        "fan_position_policy": "r7_fan_position_policy",
        "start": "start_state_family",
        "active_fan_count": "scheduled_active_fan_count",
        "speed_bin": "local_lqr_speed_bin_id",
        "transition": "transition_pair",
        "exit": "transition_exit_class",
    }.get(prefix)
    if column is None or column not in group.columns:
        return pd.DataFrame()
    return group[group[column].fillna("unknown").astype(str) == value]


def _positive_series(frame: pd.DataFrame) -> pd.Series:
    outcome = frame.get("outcome_class", pd.Series("", index=frame.index)).fillna("").astype(str)
    continuation = _bool_series(frame.get("continuation_valid", pd.Series(False, index=frame.index)))
    return continuation | outcome.eq("accepted")


def _transition_positive_series(frame: pd.DataFrame) -> pd.Series:
    if "transition_chain_compatible" in frame.columns:
        return _bool_series(frame["transition_chain_compatible"])
    return _positive_series(frame)


def _terminal_series(frame: pd.DataFrame) -> pd.Series:
    boundary = frame.get("boundary_use_class", pd.Series("", index=frame.index)).fillna("").astype(str)
    terminal = _bool_series(frame.get("episode_terminal_useful", pd.Series(False, index=frame.index)))
    return terminal | boundary.eq("episode_terminal_useful")


def _hard_failure_series(frame: pd.DataFrame) -> pd.Series:
    outcome = frame.get("outcome_class", pd.Series("", index=frame.index)).fillna("").astype(str)
    boundary = frame.get("boundary_use_class", pd.Series("", index=frame.index)).fillna("").astype(str)
    return outcome.eq("failed") | boundary.eq("hard_failure")


def _transition_hard_failure_series(frame: pd.DataFrame) -> pd.Series:
    if "transition_exit_class" in frame.columns:
        return frame["transition_exit_class"].fillna("").astype(str).eq("hard_failure") | _hard_failure_series(frame)
    return _hard_failure_series(frame)


def _bool_series(values: object) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})
    return pd.Series(bool(values))


def _unique_join(frame: pd.DataFrame, column: str) -> str:
    if column not in frame.columns:
        return ""
    values = sorted(str(value) for value in frame[column].dropna().astype(str).unique())
    return ";".join(values)


def _mean_or_zero(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if len(values) else 0.0


def _mean_positive_energy_gain(frame: pd.DataFrame) -> float:
    if "positive_specific_energy_gain_m" in frame.columns:
        return _mean_or_zero(frame, "positive_specific_energy_gain_m")
    if "energy_residual_m" in frame.columns:
        values = pd.to_numeric(frame["energy_residual_m"], errors="coerce").fillna(0.0).clip(lower=0.0)
        return float(values.mean()) if len(values) else 0.0
    return 0.0


def _representatives_for_case(
    survived: pd.DataFrame,
    *,
    case: dict[str, object],
    source_roots: dict[str, str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    max_per_group = int(case["max_representatives_per_group"])
    for (primitive_id, entry_class), group in survived.groupby(["primitive_id", "transition_entry_class"], sort=True):
        del primitive_id, entry_class
        selected = _coverage_medoid_selection(group.copy(), max_representatives=max_per_group, case_id=str(case["library_size_case_id"]))
        for rank, (_, row) in enumerate(selected.iterrows()):
            rows.append(_representative_row(row.to_dict(), case=case, source_roots=source_roots, rank=rank))
    return rows


def _coverage_medoid_selection(group: pd.DataFrame, *, max_representatives: int, case_id: str) -> pd.DataFrame:
    """Select real survived variants by robust coverage, design centrality, and marginal coverage."""

    if group.empty:
        return group
    scored = _hard_safety_filtered_group(group).copy().reset_index(drop=True)
    scored["_representative_score"] = _representative_score(scored)
    coverage = _coverage_matrix(scored)
    feature_matrix = _normalised_feature_matrix(scored, coverage)
    distance = _pairwise_distance(feature_matrix)
    centrality = _centrality_score(distance)
    worst_case = coverage.min(axis=1) if coverage.size else _numeric_metric_series(scored, "continuation_valid_rate", default=0.0).to_numpy(dtype=float)
    breadth = coverage.mean(axis=1) if coverage.size else _numeric_metric_series(scored, "continuation_valid_rate", default=0.0).to_numpy(dtype=float)
    safety = 1.0 - _numeric_metric_series(scored, "hard_failure_rate", default=1.0).clip(0.0, 1.0).to_numpy(dtype=float)
    perf = _normalise_array(scored["_representative_score"].to_numpy(dtype=float))
    medoid_quality = 0.45 * worst_case + 0.25 * breadth + 0.15 * safety + 0.10 * centrality + 0.05 * perf
    source_speed_bins = _speed_bin_set(scored)

    scored["_coverage_worst_case"] = worst_case
    scored["_coverage_breadth"] = breadth
    scored["_medoid_design_centrality"] = centrality
    scored["_medoid_quality_score"] = medoid_quality
    scored["_selection_algorithm"] = "coverage_aware_behavior_qr_medoid_greedy_marginal"
    group_center = feature_matrix.mean(axis=0) if feature_matrix.size else np.zeros(1)
    scored["_medoid_distance_to_group_center"] = np.linalg.norm(feature_matrix - group_center.reshape(1, -1), axis=1) if feature_matrix.size else 0.0

    max_count = min(int(max_representatives), len(scored))
    if max_count >= len(scored):
        selected_indices = list(range(len(scored)))
        selection_reasons = ["no_cluster_no_merge_keep_all_survivors" if str(case_id) == "no_cluster_no_merge" else "all_survivors_fit_case_limit"] * len(selected_indices)
        marginal_gains = [float(value) for value in breadth]
        redundancy = [0.0 for _ in selected_indices]
    else:
        sort_key = _variant_sort_key(scored)
        speed_bin_target_count = min(max_count, len(source_speed_bins)) if source_speed_bins else 0
        if str(case_id) == "heavy_cluster":
            first = _best_index(
                worst_case,
                tie_breakers=[
                    breadth,
                    safety,
                    centrality,
                    perf,
                    -sort_key,
                ],
            )
        else:
            first = _best_index(
                medoid_quality,
                tie_breakers=[
                    worst_case,
                    breadth,
                    safety,
                    centrality,
                    -sort_key,
                ],
            )
        selected_indices = [first]
        selection_reasons = ["best_worst_case_coverage_medoid"]
        marginal_gains = [float(breadth[first])]
        redundancy = [0.0]
        covered = coverage[first].copy()
        remaining = set(range(len(scored))) - {first}
        while remaining and len(selected_indices) < max_count:
            best_candidate = None
            best_tuple = None
            selected_speed_bins = _speed_bins_for_indices(scored, selected_indices)
            missing_speed_bins = source_speed_bins - selected_speed_bins
            speed_bin_fill_required = bool(missing_speed_bins) and len(selected_speed_bins) < speed_bin_target_count
            candidate_pool = [
                index for index in sorted(remaining)
                if not speed_bin_fill_required or _row_speed_bin(scored.iloc[index]) in missing_speed_bins
            ]
            if not candidate_pool:
                candidate_pool = sorted(remaining)
                speed_bin_fill_required = False
            for index in candidate_pool:
                marginal = _marginal_coverage_gain(covered, coverage[index])
                diversity = _normalised_min_distance(index, selected_indices, distance)
                speed_bin_gain = 1.0 if _row_speed_bin(scored.iloc[index]) in missing_speed_bins else 0.0
                total = (
                    0.40 * marginal
                    + 0.25 * speed_bin_gain
                    + 0.20 * medoid_quality[index]
                    + 0.10 * diversity
                    + 0.05 * centrality[index]
                )
                candidate_tuple = (
                    float(total),
                    float(speed_bin_gain),
                    float(marginal),
                    float(worst_case[index]),
                    float(diversity),
                    -float(sort_key[index]),
                )
                if best_tuple is None or candidate_tuple > best_tuple:
                    best_tuple = candidate_tuple
                    best_candidate = index
            if best_candidate is None:
                break
            selected_indices.append(best_candidate)
            selection_reasons.append(
                "greedy_speed_bin_marginal_coverage_medoid"
                if speed_bin_fill_required
                else "greedy_marginal_coverage_medoid"
            )
            gain = _marginal_coverage_gain(covered, coverage[best_candidate])
            marginal_gains.append(float(gain))
            redundancy.append(float(1.0 - _normalised_min_distance(best_candidate, selected_indices[:-1], distance)))
            covered = np.maximum(covered, coverage[best_candidate])
            remaining.remove(best_candidate)

    selected = scored.iloc[selected_indices].copy()
    selected["_coverage_marginal_gain"] = marginal_gains
    selected["_medoid_redundancy_penalty"] = redundancy
    selected["_medoid_selection_reason"] = selection_reasons
    selected["_coverage_vector_json"] = [
        json.dumps([float(value) for value in coverage[index].tolist()], separators=(",", ":"))
        for index in selected_indices
    ]
    selected["_coverage_feature_labels_json"] = json.dumps(_coverage_feature_labels(scored), separators=(",", ":"))
    selected["_selection_rank_order"] = list(range(len(selected_indices)))
    return selected


def _hard_safety_filtered_group(group: pd.DataFrame) -> pd.DataFrame:
    hard = _numeric_metric_series(group, "hard_failure_rate", default=1.0).clip(0.0, 1.0)
    filtered = group[hard < 0.75].copy()
    return filtered if not filtered.empty else group


def _representative_score(group: pd.DataFrame) -> pd.Series:
    transition = _numeric_metric_series(group, "transition_success_probability", "transition_chain_compatible_rate", default=0.0)
    continuation = _numeric_metric_series(group, "continuation_valid_rate", default=0.0)
    terminal = _numeric_metric_series(group, "episode_terminal_useful_rate", default=0.0)
    hard = _numeric_metric_series(group, "hard_failure_rate", default=1.0)
    updraft_gain = _numeric_metric_series(
        group,
        "updraft_gain_proxy_mean_m",
        "positive_specific_energy_gain_mean_m",
        default=0.0,
    )
    dwell = _numeric_metric_series(group, "lift_dwell_mean_s", default=0.0)
    return 1.50 * transition + 0.25 * continuation + 0.25 * terminal - 0.75 * hard + 0.05 * updraft_gain + 0.03 * dwell


def _coverage_matrix(group: pd.DataFrame) -> np.ndarray:
    labels = _coverage_feature_labels(group)
    if not labels:
        return _default_coverage_matrix(group)
    rows = []
    for _, row in group.iterrows():
        label_rates = _coverage_label_rate_map(row)
        rows.append([float(np.clip(label_rates.get(label, 0.0), 0.0, 1.0)) for label in labels])
    return np.asarray(rows, dtype=float)


def _coverage_feature_labels(group: pd.DataFrame) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for value in group.get("robustness_coverage_labels_json", pd.Series(dtype=str)).fillna("").astype(str):
        payload = _parse_json(value, default=[])
        if not isinstance(payload, list):
            continue
        for item in payload:
            label = str(item)
            if label and label not in seen:
                seen.add(label)
                labels.append(label)
    return labels


def _coverage_label_rate_map(row: pd.Series) -> dict[str, float]:
    labels = _parse_json(row.get("robustness_coverage_labels_json", "[]"), default=[])
    rates = _parse_json(row.get("robustness_coverage_rates_json", "[]"), default=[])
    if not isinstance(labels, list) or not isinstance(rates, list):
        return {}
    out: dict[str, float] = {}
    for index, label in enumerate(labels):
        if index >= len(rates):
            break
        out[str(label)] = _float(rates[index], default=0.0)
    return out


def _default_coverage_matrix(group: pd.DataFrame) -> np.ndarray:
    continuation = _numeric_metric_series(group, "continuation_valid_rate", default=0.0).clip(0.0, 1.0).to_numpy(dtype=float)
    terminal = _numeric_metric_series(group, "episode_terminal_useful_rate", default=0.0).clip(0.0, 1.0).to_numpy(dtype=float)
    hard = _numeric_metric_series(group, "hard_failure_rate", default=1.0).clip(0.0, 1.0).to_numpy(dtype=float)
    return np.vstack([continuation, terminal, 1.0 - hard]).T


def _normalised_feature_matrix(group: pd.DataFrame, coverage: np.ndarray) -> np.ndarray:
    feature_rows = []
    for _, row in group.iterrows():
        features = {}
        features.update(_controller_design_features(row))
        features.update(_behavior_features(row))
        feature_rows.append(features)
    keys = sorted({key for features in feature_rows for key in features})
    if not keys and coverage.size == 0:
        return np.zeros((len(group), 1), dtype=float)
    matrix = np.asarray([[float(features.get(key, 0.0)) for key in keys] for features in feature_rows], dtype=float)
    if coverage.size:
        matrix = np.hstack([matrix, coverage]) if matrix.size else coverage
    return _normalise_matrix(matrix)


def _controller_design_features(row: pd.Series) -> dict[str, float]:
    features: dict[str, float] = {}
    for prefix, field in (("q", "Q_weight_json"), ("r", "R_weight_json")):
        for key, value in _numeric_json_items(row.get(field, "")).items():
            features[f"{prefix}_{key}"] = float(np.log10(max(abs(float(value)), 1e-12)))
    for prefix, field in (("ref_state", "reference_state_vector"), ("ref_cmd", "reference_command_vector")):
        values = _numeric_json_vector(row.get(field, ""))
        for index, value in enumerate(values[:24]):
            features[f"{prefix}_{index:02d}"] = float(value)
    return features


def _behavior_features(row: pd.Series) -> dict[str, float]:
    fields = (
        "continuation_valid_rate",
        "episode_terminal_useful_rate",
        "hard_failure_rate",
        "minimum_wall_margin_min_m",
        "floor_margin_min_m",
        "ceiling_margin_min_m",
        "expected_updraft_gain_proxy_m",
        "updraft_gain_proxy_mean_m",
        "positive_specific_energy_gain_mean_m",
        "lift_dwell_mean_s",
        "saturation_fraction_mean",
    )
    return {field: _float(row.get(field, 0.0)) for field in fields}


def _pairwise_distance(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros((0, 0), dtype=float)
    diff = matrix[:, None, :] - matrix[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _centrality_score(distance: np.ndarray) -> np.ndarray:
    if distance.size == 0:
        return np.ones(1, dtype=float)
    if len(distance) == 1:
        return np.ones(1, dtype=float)
    mean_distance = distance.sum(axis=1) / max(1, len(distance) - 1)
    return 1.0 / (1.0 + mean_distance)


def _best_index(values: np.ndarray, *, tie_breakers: list[np.ndarray]) -> int:
    best_index = 0
    best_tuple = None
    for index, value in enumerate(values):
        candidate = (float(value),) + tuple(float(item[index]) for item in tie_breakers)
        if best_tuple is None or candidate > best_tuple:
            best_tuple = candidate
            best_index = index
    return int(best_index)


def _marginal_coverage_gain(covered: np.ndarray, candidate: np.ndarray) -> float:
    if covered.size == 0 or candidate.size == 0:
        return 0.0
    return float(np.maximum(candidate - covered, 0.0).mean())


def _normalised_min_distance(index: int, selected_indices: list[int], distance: np.ndarray) -> float:
    if not selected_indices or distance.size == 0:
        return 1.0
    max_distance = float(np.max(distance)) or 1.0
    return float(np.min(distance[index, selected_indices]) / max_distance)


def _speed_bin_set(frame: pd.DataFrame) -> set[str]:
    return {
        _row_speed_bin(row)
        for _, row in frame.iterrows()
        if _row_speed_bin(row)
    }


def _speed_bins_for_indices(frame: pd.DataFrame, indices: list[int]) -> set[str]:
    return {
        _row_speed_bin(frame.iloc[index])
        for index in indices
        if _row_speed_bin(frame.iloc[index])
    }


def _row_speed_bin(row: pd.Series) -> str:
    return _speed_bin_from_row(row)


def _variant_sort_key(frame: pd.DataFrame) -> np.ndarray:
    keys = []
    for value in frame.get("primitive_variant_id", pd.Series(range(len(frame)))).fillna("").astype(str):
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
        keys.append(int(digest, 16) / float(16**12))
    return np.asarray(keys, dtype=float)


def _normalise_matrix(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    finite = np.where(np.isfinite(matrix), matrix, 0.0)
    median = np.median(finite, axis=0)
    spread = np.percentile(finite, 75, axis=0) - np.percentile(finite, 25, axis=0)
    spread = np.where(spread > 1e-12, spread, 1.0)
    return (finite - median.reshape(1, -1)) / spread.reshape(1, -1)


def _normalise_array(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    finite = np.where(np.isfinite(values), values, 0.0)
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if hi - lo <= 1e-12:
        return np.zeros_like(finite)
    return (finite - lo) / (hi - lo)


def _numeric_json_items(value: object) -> dict[str, float]:
    payload = _parse_json(value, default={})
    items: dict[str, float] = {}

    def visit(prefix: str, obj: object) -> None:
        if isinstance(obj, dict):
            for key in sorted(obj):
                visit(f"{prefix}.{key}" if prefix else str(key), obj[key])
            return
        if isinstance(obj, (list, tuple)):
            for index, item in enumerate(obj):
                visit(f"{prefix}.{index:02d}" if prefix else f"{index:02d}", item)
            return
        try:
            items[prefix or "value"] = float(obj)
        except (TypeError, ValueError):
            return

    visit("", payload)
    return items


def _numeric_json_vector(value: object) -> list[float]:
    payload = _parse_json(value, default=[])
    if isinstance(payload, dict):
        return [float(item) for _, item in sorted(_numeric_json_items(value).items())]
    if isinstance(payload, (list, tuple)):
        values = []
        for item in payload:
            try:
                values.append(float(item))
            except (TypeError, ValueError):
                values.append(0.0)
        return values
    return []


def _parse_json(value: object, *, default: object) -> object:
    if isinstance(value, (dict, list, tuple)):
        return value
    text = str(value).strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def _float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def _numeric_metric_series(group: pd.DataFrame, *columns: str, default: float) -> pd.Series:
    for column in columns:
        if column in group.columns:
            return pd.to_numeric(group[column], errors="coerce").fillna(float(default))
    return pd.Series(float(default), index=group.index, dtype=float)


def _representative_row(
    row: dict[str, object],
    *,
    case: dict[str, object],
    source_roots: dict[str, str],
    rank: int,
) -> dict[str, object]:
    case_id = str(case["library_size_case_id"])
    variant_id = str(row.get("primitive_variant_id", ""))
    transition_entry_class = str(row.get("transition_entry_class", ""))
    transition_object_id = str(row.get("transition_object_id", f"{variant_id}__entry_{transition_entry_class or 'unknown'}"))
    cluster_id = f"{case_id}_{row.get('primitive_id', '')}_{transition_entry_class or 'unknown_entry'}_r{int(rank):03d}"
    timing = primitive_timing_contract_row()
    return {
        "compact_library_id": _compact_library_id(case_id, variant_id, cluster_id),
        "library_size_case_id": case_id,
        "library_size_human_label": str(case["library_size_human_label"]),
        "selection_policy": str(case["selection_policy"]),
        "primitive_variant_id": variant_id,
        "transition_object_id": transition_object_id,
        "primitive_id": str(row.get("primitive_id", "")),
        "entry_role": str(row.get("entry_role", "")),
        "transition_entry_class": transition_entry_class,
        "controller_id": str(row.get("controller_id", "")),
        "local_lqr_speed_bin_id": _speed_bin_from_row(row),
        "local_lqr_reference_speed_m_s": _speed_from_row(row),
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
        "representative_score_energy_term_source": "updraft_gain_proxy_mean_m_not_net_energy_residual",
        "selection_algorithm": str(row.get("_selection_algorithm", "coverage_aware_behavior_qr_medoid_greedy_marginal")),
        "medoid_selection_reason": str(row.get("_medoid_selection_reason", "")),
        "coverage_marginal_gain": float(row.get("_coverage_marginal_gain", 0.0)),
        "coverage_worst_case": float(row.get("_coverage_worst_case", row.get("robustness_worst_case_coverage", 0.0))),
        "coverage_breadth": float(row.get("_coverage_breadth", row.get("robustness_mean_coverage", 0.0))),
        "medoid_quality_score": float(row.get("_medoid_quality_score", 0.0)),
        "medoid_design_centrality": float(row.get("_medoid_design_centrality", 0.0)),
        "medoid_distance_to_group_center": float(row.get("_medoid_distance_to_group_center", 0.0)),
        "medoid_redundancy_penalty": float(row.get("_medoid_redundancy_penalty", 0.0)),
        "selection_rank_order": int(float(row.get("_selection_rank_order", rank))),
        "robustness_profile_version": str(row.get("robustness_profile_version", "")),
        "robustness_coverage_labels_json": str(row.get("_coverage_feature_labels_json", row.get("robustness_coverage_labels_json", "[]"))),
        "robustness_coverage_rates_json": str(row.get("_coverage_vector_json", row.get("robustness_coverage_rates_json", "[]"))),
        "robustness_terminal_rates_json": str(row.get("robustness_terminal_rates_json", "[]")),
        "robustness_hard_failure_rates_json": str(row.get("robustness_hard_failure_rates_json", "[]")),
        "robustness_environment_modes_seen": str(row.get("robustness_environment_modes_seen", "")),
        "robustness_start_families_seen": str(row.get("robustness_start_families_seen", "")),
        "robustness_active_fan_counts_seen": str(row.get("robustness_active_fan_counts_seen", "")),
        "robustness_evidence_blocks_seen": str(row.get("robustness_evidence_blocks_seen", "")),
        "robustness_uncertainty_tiers_seen": str(row.get("robustness_uncertainty_tiers_seen", "")),
        "robustness_active_fan_policies_seen": str(row.get("robustness_active_fan_policies_seen", "")),
        "robustness_fan_position_policies_seen": str(row.get("robustness_fan_position_policies_seen", "")),
        "robustness_speed_bins_seen": str(row.get("robustness_speed_bins_seen", "")),
        "continuation_valid_count": int(float(row.get("continuation_valid_count", 0))),
        "continuation_valid_rate": float(row.get("continuation_valid_rate", 0.0)),
        "transition_chain_compatible_count": int(float(row.get("transition_chain_compatible_count", 0))),
        "transition_chain_compatible_rate": float(row.get("transition_chain_compatible_rate", 0.0)),
        "transition_success_probability": float(row.get("transition_success_probability", row.get("transition_chain_compatible_rate", 0.0))),
        "transition_exit_classes_seen": str(row.get("transition_exit_classes_seen", "")),
        "transition_pairs_seen": str(row.get("transition_pairs_seen", "")),
        "episode_terminal_useful_count": int(float(row.get("episode_terminal_useful_count", 0))),
        "episode_terminal_useful_rate": float(row.get("episode_terminal_useful_rate", 0.0)),
        "hard_failure_count": int(float(row.get("hard_failure_count", 0))),
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
        "claim_status": "simulation_only_post_w3_library_size_case_representative",
        "mutation_status": "references_existing_frozen_transition_object_no_Q_R_K_reference_horizon_ID_mutation",
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
        "run_label": str(config.run_label).strip(),
        "library_size_case_id": str(case["library_size_case_id"]),
        "library_size_human_label": str(case["library_size_human_label"]),
        "source_w3_root": str(registry.get("source_w3_root", config.input_root.as_posix())),
        "source_w2_root": str(registry.get("source_w2_root", "")),
        "source_w01_root": str(registry.get("source_w01_root", "")),
        "source_w3_survivor_registry_sha256": file_sha256(config.input_root / "manifests" / "w3_survivor_registry.json"),
        "representative_count": int(len(representatives)),
        "selection_policy": str(case["selection_policy"]),
        "selection_algorithm": "coverage_aware_behavior_qr_medoid_greedy_marginal",
        "hard_safety_filter_policy": "prefer hard_failure_rate_below_0p75_within_primitive_transition_entry_group",
        "coverage_objective": "smallest_existing_transition_compatible_variant set covering useful entry/exit transitions, local speed bins, and R7 uncertainty blocks with low hard-failure risk",
        "speed_bin_coverage_policy": "preserve distinct W3-eligible local LQR speed bins within each primitive_id + transition_entry_class group up to the library-size case budget",
        "uncertainty_coverage_policy": "preserve R7 evidence block, uncertainty tier, active-fan policy, and fan-position policy coverage through robustness labels before medoid compression",
        "transition_contract": transition_contract_row(),
        "claim_status": "simulation_only_post_w3_library_size_case",
        "no_controller_mutation": True,
        "continuation_and_terminal_evidence_separate": True,
        "entry_role_regime_separation_policy": "representatives_grouped_by_primitive_id_and_transition_entry_class_no_cross_entry_merge",
        "primitive_timing_contract": primitive_timing_contract_row(),
        "blocked_claims": list(BLOCKED_CLAIMS),
        "representatives": representatives,
    }


def _launch_gate_candidate_availability(representatives: list[dict[str, object]]) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.DataFrame(representatives)
    rows: list[dict[str, object]] = []
    blockers: list[str] = []
    active_ids = set(ACTIVE_PRIMITIVE_IDS)
    for case_id in LIBRARY_SIZE_CASE_IDS:
        case = frame[frame.get("library_size_case_id", pd.Series(dtype=str)).astype(str) == str(case_id)] if not frame.empty else pd.DataFrame()
        launch_entry = case[case.get("transition_entry_class", pd.Series(dtype=str)).astype(str) == "launch_gate"] if not case.empty else pd.DataFrame()
        launch_entry_ids = set(launch_entry.get("primitive_id", pd.Series(dtype=str)).astype(str)) if not launch_entry.empty else set()
        launch_entry_count = int(len(launch_entry))
        launch_entry_family_count = int(len(launch_entry_ids.intersection(active_ids)))
        missing_launch_entry_ids = sorted(active_ids - launch_entry_ids)
        rows.append(
            {
                "stage_id": "R8",
                "library_size_case_id": case_id,
                "launch_gate_entry_primitive_family_count": launch_entry_family_count,
                "launch_gate_candidate_rows": launch_entry_count,
                "required_launch_gate_entry_primitive_family_count": int(len(active_ids)),
                "missing_launch_gate_entry_primitive_ids": ",".join(missing_launch_entry_ids),
                "first_decision_audit_mode": "post_w3_library_availability",
            }
        )
        if launch_entry_family_count < len(active_ids):
            blockers.append(f"{case_id}:launch_gate_entry_representative_missing:{','.join(missing_launch_entry_ids)}")
    return pd.DataFrame(rows), blockers


def _speed_bin_coverage_audit(survived: pd.DataFrame, representatives: list[dict[str, object]]) -> tuple[pd.DataFrame, list[str]]:
    source = _ensure_local_speed_bin_columns(survived)
    reps = _ensure_local_speed_bin_columns(pd.DataFrame(representatives)) if representatives else pd.DataFrame()
    rows: list[dict[str, object]] = []
    blockers: list[str] = []
    if source.empty:
        return pd.DataFrame(), ["source_w3_survivor_frame_empty_for_speed_bin_coverage"]
    for case in LIBRARY_SIZE_CASES:
        case_id = str(case["library_size_case_id"])
        max_per_group = int(case["max_representatives_per_group"])
        case_reps = (
            reps[reps.get("library_size_case_id", pd.Series(dtype=str)).astype(str) == case_id]
            if not reps.empty
            else pd.DataFrame()
        )
        for (primitive_id, entry_class), group in source.groupby(["primitive_id", "transition_entry_class"], sort=True, dropna=False):
            source_speed_bins = sorted(_speed_bin_set(group))
            selected_group = (
                case_reps[
                    (case_reps.get("primitive_id", pd.Series(dtype=str)).astype(str) == str(primitive_id))
                    & (case_reps.get("transition_entry_class", pd.Series(dtype=str)).astype(str) == str(entry_class))
                ]
                if not case_reps.empty
                else pd.DataFrame()
            )
            selected_speed_bins = sorted(_speed_bin_set(selected_group)) if not selected_group.empty else []
            target_count = min(max_per_group, len(source_speed_bins))
            status = "passed"
            if not source_speed_bins:
                status = "blocked_source_speed_bin_missing"
            elif len(selected_speed_bins) < target_count:
                status = "blocked_speed_bin_coverage_collapsed"
            rows.append(
                {
                    "stage_id": "R8",
                    "library_size_case_id": case_id,
                    "primitive_id": str(primitive_id),
                    "transition_entry_class": str(entry_class),
                    "source_speed_bin_count": int(len(source_speed_bins)),
                    "selected_speed_bin_count": int(len(selected_speed_bins)),
                    "target_speed_bin_count_for_case_budget": int(target_count),
                    "source_speed_bins": ";".join(source_speed_bins),
                    "selected_speed_bins": ";".join(selected_speed_bins),
                    "max_representatives_per_group": int(max_per_group),
                    "speed_bin_coverage_status": status,
                    "speed_bin_coverage_policy": (
                        "preserve distinct W3-eligible local LQR speed bins within each "
                        "primitive_id + transition_entry_class group up to the library-size case budget"
                    ),
                }
            )
            if status != "passed":
                blockers.append(f"{case_id}:{primitive_id}:{entry_class}:{status}")
    return pd.DataFrame(rows), blockers


def _coverage_medoid_selection_audit(representatives: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(representatives)
    if frame.empty:
        return pd.DataFrame()
    wanted = [
        "library_size_case_id",
        "primitive_id",
        "entry_role",
        "transition_entry_class",
        "transition_object_id",
        "primitive_variant_id",
        "selection_algorithm",
        "medoid_selection_reason",
        "selection_rank_order",
        "coverage_marginal_gain",
        "coverage_worst_case",
        "coverage_breadth",
        "medoid_quality_score",
        "medoid_design_centrality",
        "medoid_distance_to_group_center",
        "medoid_redundancy_penalty",
        "robustness_environment_modes_seen",
        "robustness_start_families_seen",
        "robustness_active_fan_counts_seen",
        "robustness_evidence_blocks_seen",
        "robustness_uncertainty_tiers_seen",
        "robustness_active_fan_policies_seen",
        "robustness_fan_position_policies_seen",
        "robustness_speed_bins_seen",
        "local_lqr_speed_bin_id",
        "local_lqr_reference_speed_m_s",
    ]
    available = [column for column in wanted if column in frame.columns]
    return frame[available].copy()


def _launch_gate_entry_role_audit(representatives: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(representatives)
    if frame.empty:
        return pd.DataFrame()
    audit = (
        frame.groupby(["library_size_case_id", "primitive_id", "transition_entry_class"], dropna=False)
        .size()
        .reset_index(name="representative_count")
    )
    audit.insert(0, "stage_id", "R8")
    audit["active_primitive_family"] = audit["primitive_id"].astype(str).isin(set(ACTIVE_PRIMITIVE_IDS))
    return audit


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
        "run_label": str(config.run_label).strip(),
        "run_root": run_root.as_posix(),
        "source_w3_root": config.input_root.as_posix(),
        "source_w3_registry_status": str(registry.get("status", "")),
        "source_w3_survivor_count": int(registry.get("survivor_count", 0)),
        "library_size_case_ids": list(LIBRARY_SIZE_CASE_IDS),
        "library_size_cases": case_rows,
        "selection_algorithm": "coverage_aware_behavior_qr_medoid_greedy_marginal",
        "selection_policy_summary": "hard safety filter, per-case coverage table including speed-bin coverage, behavior/Q_R medoid selection, greedy marginal coverage fill",
        "hard_safety_filter_policy": "prefer hard_failure_rate_below_0p75_within_primitive_transition_entry_group",
        "speed_bin_coverage_policy": "R8 compression must preserve local LQR speed-bin diversity within each primitive_id + transition_entry_class group up to the case representative budget",
        "primitive_timing_contract": primitive_timing_contract_row(),
        "entry_role_regime_separation_policy": "representatives_grouped_by_primitive_id_and_transition_entry_class_no_cross_entry_merge",
        "claim_status": "simulation_only_post_w3_library_size_study",
        "blocked_claims": list(BLOCKED_CLAIMS),
    }


def _write_blocked_outputs(run_root: Path, config: PostW3LibrarySizeStudyConfig, blocked_reason: str) -> None:
    manifest = {
        "manifest_version": POST_W3_LIBRARY_STUDY_VERSION,
        "project_title_version": PROJECT_TITLE_VERSION,
        "status": "blocked",
        "run_id": int(config.run_id),
        "run_label": str(config.run_label).strip(),
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
        "# v5.20 Post-W3 Library-Size Study",
        "",
        f"- Status: `{manifest.get('status', '')}`",
        f"- Library-size cases: `{','.join(LIBRARY_SIZE_CASE_IDS)}`",
        "- Human label retained for no_cluster_no_merge: `no-clustering/no-merging`",
        "- Selection: `coverage-aware behavior/Q-R medoid selection with greedy marginal coverage fill`",
        "- Speed-bin coverage: R8 preserves distinct W3-eligible local LQR speed bins within each primitive/entry-class group up to the case budget and writes `speed_bin_coverage_audit.csv`.",
        "- Hard safety filter: within each primitive/entry-role group, prefer survivors with `hard_failure_rate < 0.75`; if all candidates exceed that threshold the group is retained for explicit downstream blocking/audit.",
        "- Medoids are existing W3-eligible variants; no Q/R, K, reference, horizon, entry-role, controller-ID, or primitive-variant-ID mutation is performed.",
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
    return f"v53lib_{case_id}_{digest}"


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(filesystem_path(path).read_text(encoding="ascii"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    filesystem_path(path).parent.mkdir(parents=True, exist_ok=True)
    filesystem_path(path).write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="ascii")


def _run_folder_name(run_id: int, run_label: str = "") -> str:
    label = str(run_label).strip()
    return label if label else f"{int(run_id):03d}"


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    filesystem_path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(filesystem_path(path), index=False)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v5.20 five-case post-W3 library-size study.")
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
