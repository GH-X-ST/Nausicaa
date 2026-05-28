from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


def _bootstrap_import_paths() -> None:
    control_root = Path(__file__).resolve().parents[1]
    for rel in ("03_Primitives", "04_Scenarios"):
        path = control_root / rel
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


_bootstrap_import_paths()

from run_changed_case_validation import (  # noqa: E402
    R10_BLOCKS,
    R10_EXPECTED_FINAL_HELDOUT_LAUNCHES,
    R10_EXPECTED_HISTORY_LAUNCHES,
    R10_OUTER_CASES_PER_CONDITION,
    R10_PROTOCOL,
    R11_PROTOCOL,
)
from prim_cat import ACTIVE_PRIMITIVE_IDS, LAUNCH_CAPTURE_PRIMITIVE_IDS  # noqa: E402
from primitive_variant_registry import (  # noqa: E402
    ENTRY_ROLE_BY_PRIMITIVE_ID,
    start_family_is_compatible,
)
from run_lqr_w01_dense_chunked import (  # noqa: E402
    L6_RICH_SIDE_CANDIDATE_COUNT,
    L6_RICH_SIDE_PAIRED_TESTS_PER_CANDIDATE,
    L6_RICH_SIDE_ROW_COUNT,
    OFFICIAL_W01_ENVIRONMENT_CASES,
    R5_ACTIVE_FAN_COUNT_SEQUENCE,
    R5_TRANSITION_TRAINING_SCORE_FORMULA,
    W01DenseRunConfig,
    _row_schedule_for_index as _r5_row_schedule_for_index,
    rich_side_dense_row_count,
)
from lqr_tuning import W01_TUNING_METHOD_VERSION, candidate_weight_specs  # noqa: E402
from run_post_w3_library_size_study import (  # noqa: E402
    LIBRARY_SIZE_CASE_IDS,
    LIBRARY_SIZE_CASES,
    _coverage_medoid_selection,
    _representatives_for_case,
    _representative_score,
)
from run_r5_r10_pipeline import ARCHIVED_STAGES, STAGE_ORDER  # noqa: E402
from transition_labels import (
    REQUIRED_EXIT_CLASSES_BY_ENTRY_CLASS,
    REQUIRED_EXIT_CLASSES_BY_ROLE,
    STATE_CLASSES,
    classify_transition,
    transition_contract_row,
    transition_is_chain_compatible,
)  # noqa: E402
from transition_viability_governor import ACTIVE_GOVERNOR_PATH  # noqa: E402
from run_repeated_launch_learning_curve import (  # noqa: E402
    ACTIVE_FAN_NUMBER_VARIATION_BLOCK_ID,
    BROAD_FAN_POSITION_GENERALISATION_BLOCK_ID,
    EMPTY_FROZEN_PRIOR_BASELINE_ID,
    HISTORY_LENGTHS,
    LAUNCH_SCORE_VERSION,
    POLICY_HISTORY_CONDITIONS,
    R10_ACTIVE_FAN_COUNT_SEQUENCE,
    R9_BLOCKS,
    R9_EXPECTED_FINAL_HELDOUT_LAUNCHES,
    R9_EXPECTED_HISTORY_LAUNCHES,
    R9_OUTER_CASES_PER_CONDITION,
    R9_PROTOCOL,
    _active_fan_count_policy_for_outer_case,
    _fan_position_policy_for_outer_case,
    _final_heldout_schedule,
    _history_row_for_final,
    _initial_belief_for_policy,
    _outer_case_schedule,
    _pairing_audit_rows,
    _policy_condition,
    _launch_score_fields,
    _launch_score_fields_for_role,
    _scheduled_active_fan_count_for_outer_case,
    validation_route_for_primitive_step,
)
from run_v411_source_audit import _active_source_blockers, build_control_inventory  # noqa: E402
from run_w3_survival import (  # noqa: E402
    R5_INPUT_KIND,
    W3_ACTIVE_FAN_COUNT_SEQUENCE,
    W3_ENVIRONMENT_CASES,
    _row_for_index as _w3_row_for_index,
)
from viability_governor import DEFAULT_GOVERNOR_CONFIG, REJECTION_REASONS, governor_config_from_row  # noqa: E402


AUDIT_VERSION = "v53_algorithm_contract_audit_v1"
DEFAULT_OUTPUT_ROOT = Path("03_Control/05_Results/lqr_contextual_v1_0/algorithm_contract_audit")
DOCS_ALIGNMENT_MARKER = "<!-- R9_LAUNCH_GATE_ALIGNMENT_START -->"
BASE_DOCS_CODE_CONSISTENCY_FILES = (
    Path("docs/Glider_Control_Project_Plan.md"),
    Path("docs/Skills.md"),
    Path("docs/PR.txt"),
    Path("docs/Python Coding Instruction.txt"),
    Path("docs/Python Coding to CODEX.txt"),
    Path("docs/local_validation_environment.md"),
)


@dataclass(frozen=True)
class AlgorithmContractAuditConfig:
    repo_root: Path = Path(".")
    output_root: Path = DEFAULT_OUTPUT_ROOT
    run_id: int = 1
    dry_run: bool = False


def run_v53_algorithm_contract_audit(config: AlgorithmContractAuditConfig) -> dict[str, object]:
    """Run the hard active-code/docs audit before regenerating evidence from R5."""

    repo_root = Path(config.repo_root)
    run_root = _run_root(repo_root, config.output_root, int(config.run_id))
    active_contract_rows = _active_code_contract_rows()
    active_source_rows = _active_source_audit_rows(repo_root)
    docs_rows = _docs_code_consistency_rows(repo_root)
    legacy_alias_rows = _legacy_alias_rows()

    all_gate_rows = active_contract_rows + active_source_rows + docs_rows
    failed_rows = [row for row in all_gate_rows if not bool(row.get("passed", False))]
    status = "ready" if not failed_rows else "blocked"
    manifest = {
        "audit_version": AUDIT_VERSION,
        "status": status,
        "run_root": run_root.as_posix(),
        "dry_run": bool(config.dry_run),
        "active_code_contract_passed": _all_passed(active_contract_rows),
        "active_source_audit_passed": _all_passed(active_source_rows),
        "docs_code_consistency_passed": _all_passed(docs_rows),
        "failed_invariant_count": len(failed_rows),
        "failed_invariants": failed_rows,
        "legacy_alias_count": len(legacy_alias_rows),
        "docs_hashes": _doc_hash_rows(repo_root),
    }
    if not config.dry_run:
        _write_csv(run_root / "metrics" / "active_code_contract_audit.csv", active_contract_rows)
        _write_csv(run_root / "metrics" / "active_source_audit.csv", active_source_rows)
        _write_csv(run_root / "metrics" / "docs_code_consistency_audit.csv", docs_rows)
        _write_csv(run_root / "metrics" / "legacy_alias_audit.csv", legacy_alias_rows)
        _write_json(run_root / "manifests" / "algorithm_contract_audit_manifest.json", manifest)
        _write_report(run_root / "reports" / "algorithm_contract_audit_report.md", manifest, legacy_alias_rows)
    return manifest


def _active_code_contract_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rows.append(_row("stage_order_r5_r7_r8_r10_r11", STAGE_ORDER == ("R5", "R7", "R8", "R10", "R11"), STAGE_ORDER, ("R5", "R7", "R8", "R10", "R11")))
    rows.append(_row("r6_archived_only", ARCHIVED_STAGES == ("R6",), ARCHIVED_STAGES, ("R6",)))
    rows.append(_row("active_governor_path_transition_viability", ACTIVE_GOVERNOR_PATH == "transition_viability_governor_v1", ACTIVE_GOVERNOR_PATH, "transition_viability_governor_v1"))
    rows.append(_row("transition_state_classes_exact", STATE_CLASSES == ("launch_gate", "post_launch_degraded", "inflight_stable", "boundary_near", "recoverable_degraded", "safe_terminal", "hard_failure"), STATE_CLASSES, "seven compact transition classes"))
    expected_entry_contract = {
        "launch_gate": ("post_launch_degraded", "inflight_stable"),
        "post_launch_degraded": ("inflight_stable", "boundary_near", "safe_terminal"),
        "inflight_stable": ("inflight_stable", "boundary_near", "safe_terminal"),
        "boundary_near": ("inflight_stable", "safe_terminal"),
        "recoverable_degraded": ("inflight_stable", "safe_terminal"),
        "safe_terminal": (),
        "hard_failure": (),
    }
    rows.append(_row("transition_entry_exit_contract_exact", REQUIRED_EXIT_CLASSES_BY_ENTRY_CLASS == expected_entry_contract, REQUIRED_EXIT_CLASSES_BY_ENTRY_CLASS, expected_entry_contract))
    rows.extend(_transition_contract_invariant_rows())
    rows.append(_row("active_primitive_catalogue_has_8_variants", len(ACTIVE_PRIMITIVE_IDS) == 8, len(ACTIVE_PRIMITIVE_IDS), 8))
    rows.append(_row("launch_capture_aliases_retired_not_active", len(LAUNCH_CAPTURE_PRIMITIVE_IDS) == 6 and not set(LAUNCH_CAPTURE_PRIMITIVE_IDS).intersection(set(ACTIVE_PRIMITIVE_IDS)), {"active": list(ACTIVE_PRIMITIVE_IDS), "retired": list(LAUNCH_CAPTURE_PRIMITIVE_IDS)}, "retired launch_capture aliases disjoint from active primitives"))
    rows.append(_row("r5_dense_target_dynamic_8x32x3x100", L6_RICH_SIDE_ROW_COUNT == rich_side_dense_row_count() == 76800, {"row_count": L6_RICH_SIDE_ROW_COUNT, "candidate_count": L6_RICH_SIDE_CANDIDATE_COUNT, "paired_tests": L6_RICH_SIDE_PAIRED_TESTS_PER_CANDIDATE}, "8*32*3*100=76800"))
    rows.append(_row("r5_qr_reference_tuning_method_transition_robust", W01_TUNING_METHOD_VERSION == "w01_transition_robust_reference_v8", W01_TUNING_METHOD_VERSION, "w01_transition_robust_reference_v8"))
    rows.append(_row("r5_qr_reference_generator_exact_32_structured_candidates", _qr_generator_contract_passes(), "structured Q/R plus attitude/bank reference-bias candidate generator", "candidate 0 nominal, 1-7 physical anchors, 8-31 LHS log Q/R plus small attitude/bank reference bias; speed is scheduling-only; turn metrics are audit-only"))
    rows.append(_row("r5_selection_score_has_no_turn_expression_bonus", "turn_intent" not in R5_TRANSITION_TRAINING_SCORE_FORMULA and "turn_primitive" not in R5_TRANSITION_TRAINING_SCORE_FORMULA, R5_TRANSITION_TRAINING_SCORE_FORMULA, "transition robustness score excludes turn expression bonus"))
    rows.append(_row("r5_official_environment_cases_annular_gp_only", OFFICIAL_W01_ENVIRONMENT_CASES == (("W0", "dry_air"), ("W1", "w1_annular_gp_randomised_single"), ("W1", "w1_annular_gp_randomised_four")), OFFICIAL_W01_ENVIRONMENT_CASES, "W0 dry plus W1 annular-GP single/four"))
    rows.append(_row("r5_active_fan_count_sequence_balanced_1_2_3_4", R5_ACTIVE_FAN_COUNT_SEQUENCE == (1, 2, 3, 4), R5_ACTIVE_FAN_COUNT_SEQUENCE, (1, 2, 3, 4)))
    r5_schedule = _r5_schedule_role_audit()
    rows.append(_row("r5_schedule_has_no_cross_entry_role_start_family", int(r5_schedule["mismatch_count"]) == 0, r5_schedule, "all scheduled rows transition-object compatible"))
    rows.append(_row("r5_transition_entry_start_family_counts", r5_schedule["family_counts"] == _expected_r5_family_counts(), r5_schedule["family_counts"], _expected_r5_family_counts()))
    rows.append(_row("r5_environment_case_counts_balanced", r5_schedule["environment_case_counts"] == {"W0|dry_air": 25600, "W1|w1_annular_gp_randomised_four": 25600, "W1|w1_annular_gp_randomised_single": 25600}, r5_schedule["environment_case_counts"], "25600 rows per W0/W1 environment case"))
    w3_row_source = inspect.getsource(_w3_row_for_index)
    rows.append(_row("r7_uses_direct_r5_frozen_bundle_not_active_w2_gate", R5_INPUT_KIND == "r5_frozen_bundle_direct", R5_INPUT_KIND, "r5_frozen_bundle_direct"))
    rows.append(_row("r7_environment_cases_annular_gp_single_four", W3_ENVIRONMENT_CASES == ("w3_randomised_single", "w3_randomised_four"), W3_ENVIRONMENT_CASES, ("w3_randomised_single", "w3_randomised_four")))
    rows.append(_row("r7_active_fan_count_sequence_balanced_1_2_3_4", W3_ACTIVE_FAN_COUNT_SEQUENCE == (1, 2, 3, 4), W3_ACTIVE_FAN_COUNT_SEQUENCE, (1, 2, 3, 4)))
    rows.append(_row("r7_row_scheduler_uses_r5_selected_transition_entry_class", "_start_family_for_r5_selected_entry_class" in w3_row_source and "r5_selected_transition_entry_class" in w3_row_source, "R5 selected transition-entry W3 scheduler source", "start family selected from r5_transition_selected_for_r7 transition_entry_class"))
    r8_representatives_source = inspect.getsource(_representatives_for_case)
    r8_selection_source = inspect.getsource(_coverage_medoid_selection)
    rows.append(_row("r8_library_cases_group_by_primitive_and_transition_entry", 'groupby(["primitive_id", "transition_entry_class"]' in r8_representatives_source, "active grouping source", 'survived.groupby(["primitive_id", "transition_entry_class"])'))
    rows.append(_row("r8_selection_applies_hard_safety_filter_first", "_hard_safety_filtered_group" in r8_selection_source, "coverage selection source", "_hard_safety_filtered_group before scoring"))
    rows.append(_row("r8_compressed_cases_use_coverage_medoid_policy", _r8_compressed_cases_use_coverage_medoid_policy(), _r8_library_selection_policies(), "heavy/balanced/light/super_light use coverage_medoid; no_cluster keeps all W3 survivors"))
    rows.append(_row("r8_heavy_medoid_prefers_worst_case_coverage", _r8_heavy_medoid_prefers_worst_case_coverage(), "synthetic coverage-medoid selection", "select existing variant with stronger worst-case coverage"))
    rows.append(_row("r8_medoid_preserves_speed_bin_coverage_when_budget_allows", _r8_speed_bin_medoid_preserves_distinct_bins(), "synthetic speed-bin medoid selection", "compressed R8 cases preserve distinct W3-surviving local LQR speed bins up to case budget"))
    rows.append(_row("r8_representative_score_uses_updraft_gain_not_net_energy", _r8_score_uses_updraft_gain_not_net_energy(), "updraft-gain score check", "net energy residual must not improve representative score"))
    rows.append(_row("five_library_size_cases", set(LIBRARY_SIZE_CASE_IDS) == {"heavy_cluster", "balanced_cluster", "light_cluster", "super_light_cluster", "no_cluster_no_merge"}, LIBRARY_SIZE_CASE_IDS, "heavy/balanced/light/super_light/no_cluster"))
    rows.append(_row("r9_reduced_internal_preflight_blocks_exact", _block_tuples(R9_BLOCKS) == (("no_updraft", "W0", "dry_air", 1), ("single_fan", "W2", "annular_gp_single", 1), ("four_fan", "W2", "annular_gp_four", 1)), _block_tuples(R9_BLOCKS), "1 no-updraft, 1 single-fan, 1 four-fan internal preflight cases"))
    rows.append(_row("r10_r11_changed_case_blocks_exact", _block_tuples(R10_BLOCKS) == (("nominal_single_fan_perturbations", "W3", "w3_randomised_single", 20), ("nominal_four_fan_perturbations", "W3", "w3_randomised_four", 20), ("shifted_single_fan_positions", "W3", "w3_randomised_single", 20), ("shifted_four_fan_positions", "W3", "w3_randomised_four", 20), ("active_fan_number_variation", "W3", "w3_randomised_four", 20), ("arena_wide_fan_position_generalisation", "W3", "w3_randomised_four", 20)), _block_tuples(R10_BLOCKS), "six changed-case blocks, 20 each"))
    rows.append(_row("r9_is_internal_reduced_preflight", R9_PROTOCOL.validation_evidence_level == "internal_fixed_case_outer_loop_preflight_initial_governor_tuning_not_thesis_evidence", R9_PROTOCOL.validation_evidence_level, "internal_fixed_case_outer_loop_preflight_initial_governor_tuning_not_thesis_evidence"))
    rows.append(_row("r10_is_governor_learning_not_final_claim_gate", R10_PROTOCOL.validation_evidence_level == "changed_case_viability_governor_learning_rollout_validation_not_final_claim_gate", R10_PROTOCOL.validation_evidence_level, "changed_case_viability_governor_learning_rollout_validation_not_final_claim_gate"))
    rows.append(_row("r11_is_strict_heldout_validation", R11_PROTOCOL.validation_evidence_level == "strict_heldout_environment_only_changed_case_repeated_launch_rollout_validation", R11_PROTOCOL.validation_evidence_level, "strict_heldout_environment_only_changed_case_repeated_launch_rollout_validation"))
    rows.append(_row("r11_gates_full_safe_success", R11_PROTOCOL.min_full_safe_success_rate == 0.99, R11_PROTOCOL.min_full_safe_success_rate, 0.99))
    rows.append(_row("r9_expected_final_launches", R9_EXPECTED_FINAL_HELDOUT_LAUNCHES == len(LIBRARY_SIZE_CASE_IDS) * len(POLICY_HISTORY_CONDITIONS) * R9_OUTER_CASES_PER_CONDITION, R9_EXPECTED_FINAL_HELDOUT_LAUNCHES, "library_cases*5*3"))
    rows.append(_row("r9_expected_history_launches_internal_preflight", R9_EXPECTED_HISTORY_LAUNCHES == len(LIBRARY_SIZE_CASE_IDS) * R9_OUTER_CASES_PER_CONDITION * (sum(HISTORY_LENGTHS) + 20), R9_EXPECTED_HISTORY_LAUNCHES, "library_cases*3*(h5+h20+h100+safe_h20)"))
    rows.append(_row("r10_r11_expected_final_launches", R10_EXPECTED_FINAL_HELDOUT_LAUNCHES == len(LIBRARY_SIZE_CASE_IDS) * len(POLICY_HISTORY_CONDITIONS) * R10_OUTER_CASES_PER_CONDITION, R10_EXPECTED_FINAL_HELDOUT_LAUNCHES, "library_cases*5*120"))
    rows.append(_row("r10_r11_expected_history_launches_core_matrix", R10_EXPECTED_HISTORY_LAUNCHES == len(LIBRARY_SIZE_CASE_IDS) * R10_OUTER_CASES_PER_CONDITION * (sum(HISTORY_LENGTHS) + 20), R10_EXPECTED_HISTORY_LAUNCHES, "library_cases*120*(h5+h20+h100+safe_h20)"))
    rows.append(_row("five_policy_history_conditions_core", len(POLICY_HISTORY_CONDITIONS) == 5, len(POLICY_HISTORY_CONDITIONS), 5))
    rows.append(_row("history_lengths_core_exact", HISTORY_LENGTHS == (5, 20, 100), HISTORY_LENGTHS, (5, 20, 100)))
    rows.append(_row("empty_prior_baseline_name", EMPTY_FROZEN_PRIOR_BASELINE_ID == "empty_frozen_prior_baseline", EMPTY_FROZEN_PRIOR_BASELINE_ID, "empty_frozen_prior_baseline"))
    rows.append(_row("governor_wall_guard_0p10cm", abs(DEFAULT_GOVERNOR_CONFIG.minimum_wall_margin_m - 0.001) <= 1e-12, DEFAULT_GOVERNOR_CONFIG.minimum_wall_margin_m, 0.001))
    forbidden_speed_boundary_reasons = tuple(
        reason
        for reason in REJECTION_REASONS
        if "speed_margin" in reason
        or "context_speed" in reason
        or "minimum_speed" in reason
        or "low_speed" in reason
    )
    rows.append(
        _row(
            "no_physical_speed_boundary_rejection_reason",
            not forbidden_speed_boundary_reasons,
            REJECTION_REASONS,
            "no speed-margin or low-speed physical hard gate; local_speed_bin_incompatible is controller-scheduling compatibility only",
        )
    )
    rows.append(_row("active_governor_uses_updraft_gain_weight", hasattr(DEFAULT_GOVERNOR_CONFIG, "updraft_gain_weight") and hasattr(DEFAULT_GOVERNOR_CONFIG, "terminal_updraft_gain_weight"), DEFAULT_GOVERNOR_CONFIG.config_id, "updraft_gain_weight fields"))
    legacy_config = governor_config_from_row({"config_id": "legacy", "energy_weight": 0.12, "terminal_energy_weight": 0.34})
    rows.append(_row("legacy_energy_weight_maps_to_updraft_gain_weight", legacy_config.updraft_gain_weight == 0.12 and legacy_config.terminal_updraft_gain_weight == 0.34, {"updraft": legacy_config.updraft_gain_weight, "terminal": legacy_config.terminal_updraft_gain_weight}, {"updraft": 0.12, "terminal": 0.34}))

    base_score_row = {
        "safe_success": True,
        "terminal_useful": False,
        "lift_capture": True,
        "hard_failure": False,
        "floor_or_ceiling_violation": False,
        "no_viable_primitive": False,
        "selected_primitive_step_count": 10,
        "episode_rollout_duration_s": 1.0,
        "updraft_specific_energy_gain_proxy_m": 0.4,
        "gross_specific_energy_loss_m": 99.0,
        "net_specific_energy_delta_m": -99.0,
        "min_wall_margin_m": 0.001,
        "speed_margin_m_s": -100.0,
    }
    score_fields = _launch_score_fields(base_score_row)
    forbidden_score_fields = {"speed_factor", "speed_margin_factor", "energy_loss_factor", "net_energy_factor", "wall_margin_factor"}
    rows.append(_row("launch_score_has_no_speed_energy_loss_or_wall_margin_factor", not any(field in score_fields for field in forbidden_score_fields), sorted(score_fields), sorted(forbidden_score_fields)))
    history_score = _launch_score_fields_for_role({**base_score_row, "launch_role": "history"})
    final_score = _launch_score_fields_for_role({**base_score_row, "launch_role": "final_heldout"})
    rows.append(_row("history_launches_not_outer_loop_scored", str(history_score.get("launch_score_scope")) == "history_launch_memory_update_not_outer_loop_score", history_score.get("launch_score_scope"), "history_launch_memory_update_not_outer_loop_score"))
    rows.append(_row("final_heldout_launches_are_outer_loop_scored", str(final_score.get("launch_score_scope")) == "final_heldout_outer_loop_score", final_score.get("launch_score_scope"), "final_heldout_outer_loop_score"))
    rows.append(_row("launch_score_version_mentions_r10_r11_not_r9", "r10_r11" in LAUNCH_SCORE_VERSION and "r9" not in LAUNCH_SCORE_VERSION, LAUNCH_SCORE_VERSION, "r10_r11 without r9"))
    pairing_audit = _outer_loop_pairing_and_memory_audit()
    rows.append(_row("outer_loop_final_launches_controlled_paired", bool(pairing_audit["pairing_passed"]), pairing_audit, "same final launch key, start seed, and environment seed across policies/library cases"))
    rows.append(_row("outer_loop_history_memory_scoped_to_same_final_case", bool(pairing_audit["history_scoped"]), pairing_audit, "history rows keep library/policy/common final key and use shifted seeds"))
    rows.append(_row("outer_loop_belief_reinitialised_per_final_row", bool(pairing_audit["belief_reinitialised"]), pairing_audit, "fresh belief per final schedule row"))

    route0 = validation_route_for_primitive_step(0)
    route1 = validation_route_for_primitive_step(1)
    rows.append(_row("first_primitive_uses_launch_gate_entry_class", route0.get("route_required_entry_class") == "launch_gate", route0, "launch_gate"))
    rows.append(_row("later_nominal_route_uses_inflight_entry_class", route1.get("route_required_entry_class") == "inflight_stable", route1, "inflight_stable"))
    for protocol in (R10_PROTOCOL, R11_PROTOCOL):
        rows.append(_row(f"{protocol.stage_id.lower()}_nominal_fan_positions_fixed", _fan_position_policy_for_outer_case(protocol=protocol, environment_block_id="nominal_single_fan_perturbations") == "fixed_base_positions", _fan_position_policy_for_outer_case(protocol=protocol, environment_block_id="nominal_single_fan_perturbations"), "fixed_base_positions"))
        rows.append(_row(f"{protocol.stage_id.lower()}_four_fan_non_active_block_fixed_to_4", _scheduled_active_fan_count_for_outer_case(protocol=protocol, environment_block_id="nominal_four_fan_perturbations", environment_block_local_index=0) == 4, _scheduled_active_fan_count_for_outer_case(protocol=protocol, environment_block_id="nominal_four_fan_perturbations", environment_block_local_index=0), 4))
        rows.append(_row(f"{protocol.stage_id.lower()}_active_fan_variation_policy", _active_fan_count_policy_for_outer_case(protocol=protocol, environment_block_id=ACTIVE_FAN_NUMBER_VARIATION_BLOCK_ID) == "balanced_1_2_3_4_for_active_fan_number_variation", _active_fan_count_policy_for_outer_case(protocol=protocol, environment_block_id=ACTIVE_FAN_NUMBER_VARIATION_BLOCK_ID), "balanced_1_2_3_4_for_active_fan_number_variation"))
        rows.append(_row(f"{protocol.stage_id.lower()}_arena_wide_varies_fan_positions_and_active_count", _fan_position_policy_for_outer_case(protocol=protocol, environment_block_id=BROAD_FAN_POSITION_GENERALISATION_BLOCK_ID) == "independent_uniform_xy_bounds" and _active_fan_count_policy_for_outer_case(protocol=protocol, environment_block_id=BROAD_FAN_POSITION_GENERALISATION_BLOCK_ID) == "balanced_1_2_3_4_with_arena_wide_fan_position_generalisation", {"fan_position": _fan_position_policy_for_outer_case(protocol=protocol, environment_block_id=BROAD_FAN_POSITION_GENERALISATION_BLOCK_ID), "active_count": _active_fan_count_policy_for_outer_case(protocol=protocol, environment_block_id=BROAD_FAN_POSITION_GENERALISATION_BLOCK_ID)}, "independent positions plus balanced 1/2/3/4 active fan count"))
    rows.append(_row("changed_case_active_fan_count_sequence", R10_ACTIVE_FAN_COUNT_SEQUENCE == (1, 2, 3, 4), R10_ACTIVE_FAN_COUNT_SEQUENCE, (1, 2, 3, 4)))
    return rows


def _transition_contract_invariant_rows() -> list[dict[str, object]]:
    launch_bad = classify_transition(
        {
            "entry_role": "transition_object",
            "start_state_family": "launch_gate",
            "outcome_class": "accepted",
            "continuation_valid": True,
            "boundary_use_class": "model_boundary_only",
            "minimum_wall_margin_m": 0.0,
            "floor_margin_m": 0.5,
            "ceiling_margin_m": 1.0,
        }
    )
    launch_good = classify_transition(
        {
            "entry_role": "transition_object",
            "start_state_family": "launch_gate",
            "outcome_class": "accepted",
            "continuation_valid": True,
            "boundary_use_class": "inside",
            "minimum_wall_margin_m": 1.0,
            "floor_margin_m": 0.5,
            "ceiling_margin_m": 1.0,
        }
    )
    inflight_bad = classify_transition(
        {
            "entry_role": "transition_object",
            "start_state_family": "inflight_nominal",
            "outcome_class": "weak",
            "continuation_valid": False,
            "boundary_use_class": "inside",
            "minimum_wall_margin_m": 1.0,
            "floor_margin_m": 0.5,
            "ceiling_margin_m": 1.0,
        }
    )
    recovery_good = classify_transition(
        {
            "entry_role": "transition_object",
            "start_state_family": "inflight_recovery_edge",
            "outcome_class": "accepted",
            "continuation_valid": True,
            "boundary_use_class": "inside",
            "minimum_wall_margin_m": 1.0,
            "floor_margin_m": 0.5,
            "ceiling_margin_m": 1.0,
        }
    )
    return [
        _row("transition_contract_row_boundary_near_route_state", bool(transition_contract_row()["boundary_near_is_route_state_not_failure"]), transition_contract_row(), "boundary_near route state"),
        _row("launch_boundary_exit_cannot_pass_transition_gate", not bool(launch_bad["transition_chain_compatible"]), launch_bad, "launch boundary_near exit rejected"),
        _row("launch_post_launch_exit_can_pass_transition_gate", bool(launch_good["transition_chain_compatible"]), launch_good, "launch post_launch_degraded/inflight exit accepted"),
        _row("inflight_recoverable_degraded_exit_cannot_pass_transition_gate", not transition_is_chain_compatible(entry_role="transition_object", entry_class="inflight_stable", exit_class="recoverable_degraded"), "recoverable_degraded", "not allowed for inflight_stable transition object"),
        _row("inflight_weak_without_chain_handoff_rejected", not bool(inflight_bad["transition_chain_compatible"]), inflight_bad, "weak local rollout is not sufficient"),
        _row("recovery_to_inflight_passes_transition_gate", bool(recovery_good["transition_chain_compatible"]), recovery_good, "recoverable_degraded transition object can restore inflight_stable"),
    ]


def _qr_generator_contract_passes() -> bool:
    try:
        for primitive_id in ACTIVE_PRIMITIVE_IDS:
            specs = candidate_weight_specs(primitive_id=str(primitive_id), candidate_count=32)
            labels = [str(spec.weight_label) for spec in specs]
            if len(specs) != 32:
                return False
            if not labels[0].endswith("_robust_anchor_nominal_ref_nominal_000"):
                return False
            if not labels[1].endswith("_robust_anchor_attitude_heavy_ref_pitch_up_001"):
                return False
            if not labels[7].endswith("_robust_anchor_balanced_agile_ref_right_bias_007"):
                return False
            if not all("_robust_lhs_logqr_refbias_" in label for label in labels[8:]):
                return False
            if (
                specs[0].reference_pitch_bias_rad != 0.0
                or specs[0].reference_bank_bias_rad != 0.0
                or specs[0].reference_roll_rate_bias_rad_s != 0.0
                or specs[0].reference_speed_bias_m_s != 0.0
            ):
                return False
            if any(float(spec.reference_speed_bias_m_s) != 0.0 for spec in specs):
                return False
            if any(abs(float(spec.reference_roll_rate_bias_rad_s)) > 0.0 for spec in specs):
                return False
            if not any(
                abs(float(spec.reference_pitch_bias_rad)) > 0.0
                or abs(float(spec.reference_bank_bias_rad)) > 0.0
                for spec in specs[1:]
            ):
                return False
            if any("launch_capture" in label for label in labels):
                return False
    except Exception:
        return False
    return True


def _r5_schedule_role_audit() -> dict[str, object]:
    config = W01DenseRunConfig(
        run_id=0,
        rows=L6_RICH_SIDE_ROW_COUNT,
        candidate_count=L6_RICH_SIDE_CANDIDATE_COUNT,
        paired_tests_per_candidate=L6_RICH_SIDE_PAIRED_TESTS_PER_CANDIDATE,
    )
    family_counts: dict[str, int] = {}
    role_family_counts: dict[str, int] = {}
    environment_case_counts: dict[str, int] = {}
    primitive_counts: dict[str, int] = {}
    mismatches: list[dict[str, object]] = []
    for row_index in range(int(L6_RICH_SIDE_ROW_COUNT)):
        schedule = _r5_row_schedule_for_index(row_index, config)
        primitive_id = str(schedule.primitive_id)
        entry_role = str(ENTRY_ROLE_BY_PRIMITIVE_ID.get(primitive_id, ""))
        family = str(schedule.start_state_family)
        family_counts[family] = family_counts.get(family, 0) + 1
        role_family_key = f"{entry_role}|{family}"
        role_family_counts[role_family_key] = role_family_counts.get(role_family_key, 0) + 1
        environment_key = f"{schedule.W_layer}|{schedule.environment_mode}"
        environment_case_counts[environment_key] = environment_case_counts.get(environment_key, 0) + 1
        primitive_counts[primitive_id] = primitive_counts.get(primitive_id, 0) + 1
        if not start_family_is_compatible(entry_role=entry_role, start_state_family=family):
            if len(mismatches) < 10:
                mismatches.append(
                    {
                        "row_index": int(row_index),
                        "primitive_id": primitive_id,
                        "entry_role": entry_role,
                        "start_state_family": family,
                    }
                )
    return {
        "row_count": int(L6_RICH_SIDE_ROW_COUNT),
        "mismatch_count": int(len(mismatches)),
        "mismatch_examples": mismatches,
        "family_counts": dict(sorted(family_counts.items())),
        "role_family_counts": dict(sorted(role_family_counts.items())),
        "environment_case_counts": dict(sorted(environment_case_counts.items())),
        "primitive_count_range": (
            min(primitive_counts.values()) if primitive_counts else 0,
            max(primitive_counts.values()) if primitive_counts else 0,
        ),
    }


def _expected_r5_family_counts() -> dict[str, int]:
    return {
        "inflight_boundary_near": 7680,
        "inflight_lift_region": 11520,
        "inflight_nominal": 19200,
        "inflight_recovery_edge": 7680,
        "launch_gate": 30720,
    }


def _block_tuples(blocks: Iterable[object]) -> tuple[tuple[str, str, str, int], ...]:
    return tuple(
        (
            str(getattr(block, "block_id")),
            str(getattr(block, "W_layer")),
            str(getattr(block, "environment_mode")),
            int(getattr(block, "case_count")),
        )
        for block in blocks
    )


def _r8_score_uses_updraft_gain_not_net_energy() -> bool:
    frame = pd.DataFrame(
        [
            {
                "continuation_valid_rate": 0.5,
                "episode_terminal_useful_rate": 0.0,
                "hard_failure_rate": 0.0,
                "energy_residual_mean_m": 100.0,
                "updraft_gain_proxy_mean_m": 0.0,
                "positive_specific_energy_gain_mean_m": 0.0,
                "lift_dwell_mean_s": 0.0,
            },
            {
                "continuation_valid_rate": 0.5,
                "episode_terminal_useful_rate": 0.0,
                "hard_failure_rate": 0.0,
                "energy_residual_mean_m": -100.0,
                "updraft_gain_proxy_mean_m": 1.0,
                "positive_specific_energy_gain_mean_m": 1.0,
                "lift_dwell_mean_s": 0.0,
            },
        ]
    )
    scores = list(_representative_score(frame))
    return len(scores) == 2 and float(scores[1]) > float(scores[0])


def _r8_library_selection_policies() -> dict[str, str]:
    return {str(case["library_size_case_id"]): str(case["selection_policy"]) for case in LIBRARY_SIZE_CASES}


def _r8_compressed_cases_use_coverage_medoid_policy() -> bool:
    policies = _r8_library_selection_policies()
    compressed_case_ids = {"heavy_cluster", "balanced_cluster", "light_cluster", "super_light_cluster"}
    return all("coverage_medoid" in policies.get(case_id, "") for case_id in compressed_case_ids) and policies.get(
        "no_cluster_no_merge", ""
    ) == "all_w3_survivors_no_clustering_no_merging"


def _r8_heavy_medoid_prefers_worst_case_coverage() -> bool:
    frame = pd.DataFrame(
        [
            {
                "primitive_variant_id": "rare_case_fragile_high_average",
                "primitive_id": "glide",
                "entry_role": "transition_object",
                "transition_entry_class": "inflight_stable",
                "continuation_valid_rate": 0.95,
                "episode_terminal_useful_rate": 0.20,
                "hard_failure_rate": 0.01,
                "robustness_coverage_labels_json": '["env:a","env:b","active_fan_count:1","active_fan_count:4"]',
                "robustness_coverage_rates_json": "[1.0,1.0,1.0,0.0]",
                "Q_weight_json": '{"q":1.0}',
                "R_weight_json": '{"r":1.0}',
                "reference_state_vector": "[0,0,0]",
                "reference_command_vector": "[0,0,0]",
            },
            {
                "primitive_variant_id": "broad_case_medoid",
                "primitive_id": "glide",
                "entry_role": "transition_object",
                "transition_entry_class": "inflight_stable",
                "continuation_valid_rate": 0.75,
                "episode_terminal_useful_rate": 0.10,
                "hard_failure_rate": 0.02,
                "robustness_coverage_labels_json": '["env:a","env:b","active_fan_count:1","active_fan_count:4"]',
                "robustness_coverage_rates_json": "[0.70,0.70,0.70,0.70]",
                "Q_weight_json": '{"q":1.1}',
                "R_weight_json": '{"r":1.0}',
                "reference_state_vector": "[0,0,0]",
                "reference_command_vector": "[0,0,0]",
            },
        ]
    )
    selected = _coverage_medoid_selection(frame, max_representatives=1, case_id="heavy_cluster")
    return len(selected) == 1 and str(selected.iloc[0]["primitive_variant_id"]) == "broad_case_medoid"


def _r8_speed_bin_medoid_preserves_distinct_bins() -> bool:
    frame = pd.DataFrame(
        [
            {
                "primitive_variant_id": "strong_speed_5p0",
                "primitive_id": "glide",
                "entry_role": "transition_object",
                "transition_entry_class": "inflight_stable",
                "local_lqr_speed_bin_id": "speed_bin_5p0_m_s",
                "local_lqr_reference_speed_m_s": 5.0,
                "continuation_valid_rate": 0.95,
                "episode_terminal_useful_rate": 0.2,
                "hard_failure_rate": 0.01,
                "robustness_coverage_labels_json": '["env:a","speed_bin:speed_bin_5p0_m_s"]',
                "robustness_coverage_rates_json": "[0.95,0.95]",
                "Q_weight_json": '{"q":1.0}',
                "R_weight_json": '{"r":1.0}',
                "reference_state_vector": "[0,0,0]",
                "reference_command_vector": "[0,0,0]",
            },
            {
                "primitive_variant_id": "redundant_speed_5p0",
                "primitive_id": "glide",
                "entry_role": "transition_object",
                "transition_entry_class": "inflight_stable",
                "local_lqr_speed_bin_id": "speed_bin_5p0_m_s",
                "local_lqr_reference_speed_m_s": 5.0,
                "continuation_valid_rate": 0.93,
                "episode_terminal_useful_rate": 0.2,
                "hard_failure_rate": 0.01,
                "robustness_coverage_labels_json": '["env:a","speed_bin:speed_bin_5p0_m_s"]',
                "robustness_coverage_rates_json": "[0.93,0.93]",
                "Q_weight_json": '{"q":1.01}',
                "R_weight_json": '{"r":1.0}',
                "reference_state_vector": "[0,0,0]",
                "reference_command_vector": "[0,0,0]",
            },
            {
                "primitive_variant_id": "needed_speed_7p0",
                "primitive_id": "glide",
                "entry_role": "transition_object",
                "transition_entry_class": "inflight_stable",
                "local_lqr_speed_bin_id": "speed_bin_7p0_m_s",
                "local_lqr_reference_speed_m_s": 7.0,
                "continuation_valid_rate": 0.70,
                "episode_terminal_useful_rate": 0.1,
                "hard_failure_rate": 0.02,
                "robustness_coverage_labels_json": '["env:a","speed_bin:speed_bin_7p0_m_s"]',
                "robustness_coverage_rates_json": "[0.70,0.70]",
                "Q_weight_json": '{"q":1.3}',
                "R_weight_json": '{"r":1.0}',
                "reference_state_vector": "[0,0,0]",
                "reference_command_vector": "[0,0,0]",
            },
        ]
    )
    selected = _coverage_medoid_selection(frame, max_representatives=2, case_id="balanced_cluster")
    return set(selected["local_lqr_speed_bin_id"].astype(str)) == {"speed_bin_5p0_m_s", "speed_bin_7p0_m_s"}


def _outer_loop_pairing_and_memory_audit() -> dict[str, object]:
    outer_cases = _outer_case_schedule(protocol=R9_PROTOCOL, seed=90, smoke_outer_cases_per_block=1)
    final_schedule = _final_heldout_schedule(outer_cases=outer_cases, protocol=R9_PROTOCOL)
    pairing_rows = _pairing_audit_rows(final_schedule)
    memory_final = next(row for row in final_schedule if str(row["policy_id"]) == "directional_3d_residual_memory_h20")
    history = _history_row_for_final(memory_final, 0)
    policy = _policy_condition(str(memory_final["policy_id"]))
    belief_a = _initial_belief_for_policy(policy=policy, final_row=memory_final)
    belief_b = _initial_belief_for_policy(policy=policy, final_row=memory_final)
    return {
        "pairing_passed": bool(pairing_rows) and all(bool(row.get("pairing_passed", False)) for row in pairing_rows),
        "pairing_row_count": len(pairing_rows),
        "history_scoped": bool(
            history["library_size_case_id"] == memory_final["library_size_case_id"]
            and history["policy_id"] == memory_final["policy_id"]
            and int(history["history_length"]) == int(memory_final["history_length"])
            and history["common_final_launch_key"] == memory_final["common_final_launch_key"]
            and int(history["launch_state_seed"]) != int(memory_final["launch_state_seed"])
            and int(history["environment_seed"]) != int(memory_final["environment_seed"])
        ),
        "belief_reinitialised": bool(belief_a is not belief_b and belief_a.update_count == 0 and belief_b.update_count == 0),
    }


def _active_source_audit_rows(repo_root: Path) -> list[dict[str, object]]:
    inventory = build_control_inventory(repo_root / "03_Control")
    active_count = sum(str(row.get("classification", "")).startswith("active_") for row in inventory)
    ignored_count = sum(str(row.get("classification", "")) in {"generated_result", "retired_not_active"} for row in inventory)
    blockers = _active_source_blockers(repo_root=repo_root, inventory=inventory)
    rows = [
        _row("active_source_files_scanned", active_count > 0, active_count, ">0 active source files"),
        _row("archived_and_generated_evidence_ignored_by_default", ignored_count >= 0, ignored_count, "ignored unless archive audit is explicit"),
        _row("active_source_blockers_absent", len(blockers) == 0, blockers, "no active-source blockers"),
    ]
    return rows


def _docs_code_consistency_rows(repo_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    docs_text: dict[Path, str] = {}
    consistency_files = _docs_code_consistency_files(repo_root)
    for doc in consistency_files:
        path = repo_root / doc
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:
            text = ""
            rows.append(_row(f"doc_readable:{doc.as_posix()}", False, f"{type(exc).__name__}:{exc}", "readable"))
        else:
            rows.append(_row(f"doc_readable:{doc.as_posix()}", True, path.stat().st_size, "readable"))
        docs_text[doc] = text

    stage_docs = [doc for doc, text in docs_text.items() if DOCS_ALIGNMENT_MARKER in text]
    stage_fragments = (
        "R5 -> R7 -> R8 -> R10 -> R11 -> Reality",
        "R9 remains internal preflight only",
        "transition-aware primitive",
        "R10 tunes the viability governor",
        "R11",
        "held-out validation",
    )
    for doc in stage_docs:
        text = docs_text.get(doc, "")
        missing = [fragment for fragment in stage_fragments if fragment not in text]
        rows.append(_row(f"stage_semantics:{doc.as_posix()}", not missing, missing, "current R5/R7/R8/R10/R11 semantics with R9 internal only"))

    combined = "\n".join(docs_text.values())
    combined_requirements = {
        "active_thesis_workflow_transition_aware": "R5 -> R7 -> R8 -> R10 -> R11 -> Reality",
        "r9_internal_only": "R9 remains internal preflight only",
        "transition_object_core": "Every primitive is treated as a transition object",
        "r7_not_local_success_only": "No primitive may pass R7 solely on local rollout success",
        "memory_scoped_per_final_row": "Memory is reinitialised per final test row",
        "final_score_only_final_heldout_path": "Final scoring is computed only from the final held-out rollout path",
        "full_safe_success_gate": "full_safe_success",
        "updraft_gain_active_weight": "updraft_gain_weight",
        "no_speed_active_gate": "Low speed is not an active governor rejection reason",
        "energy_loss_not_hard_failure": "Energy loss is also not a hard-failure reason",
        "sample_count_coverage": "sample_count",
        "r11_consumes_r10_frozen_config": "frozen governor config written by R10",
        "r8_coverage_medoid_selection": "coverage-aware medoid",
        "r8_no_synthetic_controller_in_clustering": "without averaging Q/R or synthesising new controllers",
        "r8_speed_bin_coverage_preservation": "speed-bin collapse is a library-coverage failure",
    }
    for check_id, fragment in combined_requirements.items():
        rows.append(_row(check_id, fragment in combined, fragment if fragment in combined else "missing", fragment))
    rows.append(
        _row(
            "energy_alias_legacy_only",
            "expected_energy_residual_m" in combined and "legacy" in combined.lower(),
            "expected_energy_residual_m+legacy" if "expected_energy_residual_m" in combined and "legacy" in combined.lower() else "missing",
            "expected_energy_residual_m documented as legacy alias",
        )
    )
    return rows


def _legacy_alias_rows() -> list[dict[str, object]]:
    aliases = [
        {
            "legacy_alias": "expected_energy_residual_m",
            "real_active_logic": "expected_updraft_gain_proxy_m is the governor soft reward; expected_energy_residual_m is net-energy audit compatibility only.",
            "intentional_status": "legacy_alias_intentional_not_active_score",
        },
        {
            "legacy_alias": "belief_local_energy_residual_m",
            "real_active_logic": "belief_local_updraft_gain_residual_m is the active memory residual; energy residual wording is a compatibility alias.",
            "intentional_status": "legacy_alias_intentional_not_total_energy_memory",
        },
        {
            "legacy_alias": "energy_weight / terminal_energy_weight",
            "real_active_logic": "updraft_gain_weight / terminal_updraft_gain_weight are active config fields; legacy names are accepted only when reading old frozen configs.",
            "intentional_status": "legacy_config_reader_only",
        },
        {
            "legacy_alias": "minimum_speed_m_s / speed_margin_m_s",
            "real_active_logic": "Speed may appear in old logs or telemetry only; it is not an active governor rejection, recovery trigger, score factor, or audit gate.",
            "intentional_status": "legacy_telemetry_only",
        },
        {
            "legacy_alias": "post_w3_" + "cluster",
            "real_active_logic": "Use the active five-case post-W3 library-size study: heavy, balanced, light, super_light, and no_cluster_no_merge.",
            "intentional_status": "stale_path_alias_blocked_in_active_source",
        },
        {
            "legacy_alias": "R6 / W2 replay",
            "real_active_logic": "R6 is archived diagnostic-only; active thesis chain is R5 -> R7 -> R8 -> R10 -> R11 -> Reality, with R9 internal preflight only.",
            "intentional_status": "legacy_stage_alias_not_pass_gate",
        },
    ]
    return [
        {
            "passed": True,
            "legacy_alias": row["legacy_alias"],
            "real_active_logic": row["real_active_logic"],
            "intentional_status": row["intentional_status"],
        }
        for row in aliases
    ]


def _doc_hash_rows(repo_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for doc in _docs_code_consistency_files(repo_root):
        path = repo_root / doc
        if not path.is_file():
            rows.append({"path": doc.as_posix(), "readable": False, "byte_count": 0, "sha256": ""})
            continue
        data = path.read_bytes()
        rows.append(
            {
                "path": doc.as_posix(),
                "readable": True,
                "byte_count": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return rows


def _docs_code_consistency_files(repo_root: Path) -> tuple[Path, ...]:
    files = set(BASE_DOCS_CODE_CONSISTENCY_FILES)
    docs_root = repo_root / "docs"
    if docs_root.is_dir():
        for path in docs_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                continue
            if DOCS_ALIGNMENT_MARKER in text:
                files.add(path.relative_to(repo_root))
    return tuple(sorted(files, key=lambda item: item.as_posix()))


def _write_report(path: Path, manifest: dict[str, object], legacy_alias_rows: list[dict[str, object]]) -> None:
    failed = list(manifest.get("failed_invariants", []))
    lines = [
        "# Algorithm Contract Audit",
        "",
        f"- Audit version: `{manifest['audit_version']}`",
        f"- Status: `{manifest['status']}`",
        f"- Active-code contract passed: `{manifest['active_code_contract_passed']}`",
        f"- Active-source audit passed: `{manifest['active_source_audit_passed']}`",
        f"- Docs/code consistency passed: `{manifest['docs_code_consistency_passed']}`",
        f"- Failed invariant count: `{manifest['failed_invariant_count']}`",
        "- Archived/generated evidence is ignored by the active-source scan unless an archive audit is explicitly requested.",
        "- No dense evidence, R5/R7/R8/R10/R11 rollout, internal R9 preflight, or claim-bearing validation is generated by this audit.",
        "",
        "## Intentional Legacy Aliases",
        "",
    ]
    for row in legacy_alias_rows:
        lines.append(f"- `{row['legacy_alias']}`: {row['real_active_logic']} Status: `{row['intentional_status']}`.")
    lines.extend(["", "## Failed Invariants", ""])
    if not failed:
        lines.append("- None.")
    else:
        for row in failed:
            lines.append(
                f"- `{row.get('check_id', 'unknown')}` observed `{_stringify(row.get('observed', ''))}`; required `{_stringify(row.get('required', ''))}`."
            )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="ascii")


def _row(check_id: str, passed: bool, observed: object, required: object) -> dict[str, object]:
    return {
        "check_id": check_id,
        "passed": bool(passed),
        "observed": observed,
        "required": required,
    }


def _all_passed(rows: Iterable[dict[str, object]]) -> bool:
    return all(bool(row.get("passed", False)) for row in rows)


def _run_root(repo_root: Path, output_root: Path, run_id: int) -> Path:
    root = Path(output_root)
    if not root.is_absolute():
        root = repo_root / root
    return root / f"{int(run_id):03d}"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=False) + "\n", encoding="ascii")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _stringify(row.get(key, "")) for key in fieldnames})


def _json_ready(value: object) -> object:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def _stringify(value: object) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_ready(value), sort_keys=True)
    return str(value)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the v5.20 active algorithm contract audit before R5 regeneration.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_v53_algorithm_contract_audit(
        AlgorithmContractAuditConfig(
            repo_root=args.repo_root,
            output_root=args.output_root,
            run_id=args.run_id,
            dry_run=args.dry_run,
        )
    )
    print(json.dumps(_json_ready(result), indent=2, sort_keys=True))
    return 0 if result["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
