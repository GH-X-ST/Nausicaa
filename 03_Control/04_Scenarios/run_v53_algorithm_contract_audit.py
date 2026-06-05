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
    R11_BLOCKS,
    R11_EXPECTED_FINAL_HELDOUT_LAUNCHES,
    R11_EXPECTED_HISTORY_LAUNCHES,
    R11_OUTER_CASES_PER_CONDITION,
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
    R5_EVIDENCE_BLOCKS,
    R5_EVIDENCE_BLOCK_IDS,
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
    representative_budget_for_entry_class,
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
    GOVERNOR_CALIBRATION_SEARCH_POLICY,
    HISTORY_LENGTHS,
    HISTORY_LENGTH_SUM,
    LAUNCH_SCORE_VERSION,
    NO_UPDRAFT_CHANGED_CASE_BLOCK_ID,
    ONLINE_MEMORY_SCOPE,
    OUTER_LOOP_GOVERNOR_LEARNING_STRATEGY_VERSION,
    POLICY_HISTORY_CONDITIONS,
    R11_POLICY_HISTORY_CONDITIONS,
    R10_L7_FULL_DOMAIN_RANDOMISATION_BLOCK_ID,
    R10_GLOBAL_CALIBRATION_SCOPE,
    R9_POLICY_HISTORY_CONDITIONS,
    R11_GOVERNOR_HANDOFF_SCOPE,
    R10_ACTIVE_FAN_COUNT_SEQUENCE,
    R10_ARENA_WIDE_FAN_POSITION_SAFETY_RADIUS_M,
    R11_L0_DRY_AIR_FIXED_BLOCK_ID,
    R11_L1_SINGLE_FAN_FIXED_NOMINAL_BLOCK_ID,
    R11_L2_FOUR_FAN_FIXED_NOMINAL_BLOCK_ID,
    R11_L3_FAN_PARAMETER_UNCERTAINTY_BLOCK_ID,
    R11_L4_LOCAL_FAN_POSITION_UNCERTAINTY_BLOCK_ID,
    R11_L5_ACTIVE_FAN_COUNT_UNCERTAINTY_BLOCK_ID,
    R11_L6_ENVIRONMENT_ONLY_FULL_UNCERTAINTY_BLOCK_ID,
    R11_L7_FULL_DOMAIN_RANDOMISATION_BLOCK_ID,
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
    _no_variation_audit_rows,
    _outer_case_schedule,
    _pairing_audit_rows,
    _policy_condition,
    _launch_score_fields,
    _launch_score_fields_for_role,
    _scheduled_active_fan_count_for_outer_case,
    _uses_full_w3_randomisation_block,
    validation_route_for_primitive_step,
)
from run_v411_source_audit import _active_source_blockers, build_control_inventory  # noqa: E402
from run_w3_survival import (  # noqa: E402
    R5_INPUT_KIND,
    R7_EVIDENCE_BLOCK_IDS,
    W3_ACTIVE_FAN_COUNT_SEQUENCE,
    W3_ENVIRONMENT_CASES,
    _row_for_index as _w3_row_for_index,
)
from viability_governor import DEFAULT_GOVERNOR_CONFIG, REJECTION_REASONS, governor_config_from_row  # noqa: E402


AUDIT_VERSION = "v53_algorithm_contract_audit_v1"
DEFAULT_OUTPUT_ROOT = Path("03_Control/05_Results/F_algorithm_contract_audit")
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
    rows.append(_row("active_governor_path_mission_aligned_transition_viability", ACTIVE_GOVERNOR_PATH == "mission_aligned_transition_viability_governor_v1", ACTIVE_GOVERNOR_PATH, "mission_aligned_transition_viability_governor_v1"))
    rows.append(_row("transition_state_classes_exact", STATE_CLASSES == ("launch_gate", "post_launch_degraded", "inflight_stable", "boundary_near", "recoverable_degraded", "safe_terminal", "hard_failure"), STATE_CLASSES, "seven compact transition classes"))
    expected_entry_contract = {
        "launch_gate": ("post_launch_degraded", "inflight_stable"),
        "post_launch_degraded": ("inflight_stable", "boundary_near", "safe_terminal"),
        "inflight_stable": ("inflight_stable", "boundary_near", "safe_terminal"),
        "boundary_near": ("inflight_stable", "safe_terminal"),
        "recoverable_degraded": ("inflight_stable", "recoverable_degraded", "safe_terminal"),
        "safe_terminal": (),
        "hard_failure": (),
    }
    rows.append(_row("transition_entry_exit_contract_exact", REQUIRED_EXIT_CLASSES_BY_ENTRY_CLASS == expected_entry_contract, REQUIRED_EXIT_CLASSES_BY_ENTRY_CLASS, expected_entry_contract))
    rows.extend(_transition_contract_invariant_rows())
    rows.append(_row("active_primitive_catalogue_has_8_variants", len(ACTIVE_PRIMITIVE_IDS) == 8, len(ACTIVE_PRIMITIVE_IDS), 8))
    rows.append(_row("launch_capture_aliases_retired_not_active", len(LAUNCH_CAPTURE_PRIMITIVE_IDS) == 6 and not set(LAUNCH_CAPTURE_PRIMITIVE_IDS).intersection(set(ACTIVE_PRIMITIVE_IDS)), {"active": list(ACTIVE_PRIMITIVE_IDS), "retired": list(LAUNCH_CAPTURE_PRIMITIVE_IDS)}, "retired launch_capture aliases disjoint from active primitives"))
    rows.append(_row("r5_dense_target_dynamic_8x32x8x50", L6_RICH_SIDE_ROW_COUNT == rich_side_dense_row_count() == 102400, {"row_count": L6_RICH_SIDE_ROW_COUNT, "candidate_count": L6_RICH_SIDE_CANDIDATE_COUNT, "paired_tests": L6_RICH_SIDE_PAIRED_TESTS_PER_CANDIDATE, "evidence_block_count": len(R5_EVIDENCE_BLOCKS)}, "8*32*8*50=102400"))
    rows.append(_row("r5_qr_reference_tuning_method_transition_robust", W01_TUNING_METHOD_VERSION == "w01_transition_robust_reference_v8", W01_TUNING_METHOD_VERSION, "w01_transition_robust_reference_v8"))
    rows.append(_row("r5_qr_reference_generator_exact_32_structured_candidates", _qr_generator_contract_passes(), "structured Q/R plus attitude/bank reference-bias candidate generator", "candidate 0 nominal, 1-7 physical anchors, 8-31 LHS log Q/R plus small attitude/bank reference bias; speed is scheduling-only; turn metrics are audit-only"))
    rows.append(_row("r5_selection_score_has_no_turn_expression_bonus", "turn_intent" not in R5_TRANSITION_TRAINING_SCORE_FORMULA and "turn_primitive" not in R5_TRANSITION_TRAINING_SCORE_FORMULA, R5_TRANSITION_TRAINING_SCORE_FORMULA, "transition robustness score excludes turn expression bonus"))
    rows.append(_row("r5_official_environment_cases_annular_gp_only", OFFICIAL_W01_ENVIRONMENT_CASES == (("W0", "dry_air"), ("W1", "w1_annular_gp_randomised_single"), ("W1", "w1_annular_gp_randomised_four")), OFFICIAL_W01_ENVIRONMENT_CASES, "W0 dry plus W1 annular-GP single/four"))
    rows.append(_row("r5_evidence_blocks_anchor_plus_uncertainty_family", len(R5_EVIDENCE_BLOCK_IDS) == 8 and {"r5_anchor_dry_air", "r5_anchor_single_fan_fixed", "r5_anchor_four_fan_fixed", "r5_random_arena_wide"}.issubset(set(R5_EVIDENCE_BLOCK_IDS)), R5_EVIDENCE_BLOCK_IDS, "8 R5 blocks with dry/single/four anchors plus randomized uncertainty-family blocks"))
    rows.append(_row("r5_active_fan_count_sequence_balanced_0_1_2_3_4", R5_ACTIVE_FAN_COUNT_SEQUENCE == (0, 1, 2, 3, 4), R5_ACTIVE_FAN_COUNT_SEQUENCE, (0, 1, 2, 3, 4)))
    r5_schedule = _r5_schedule_role_audit()
    rows.append(_row("r5_schedule_has_no_cross_entry_role_start_family", int(r5_schedule["mismatch_count"]) == 0, r5_schedule, "all scheduled rows transition-object compatible"))
    rows.append(_row("r5_transition_entry_start_family_counts", r5_schedule["family_counts"] == _expected_r5_family_counts(), r5_schedule["family_counts"], _expected_r5_family_counts()))
    rows.append(_row("r5_environment_case_counts_reflect_block_ladder", r5_schedule["environment_case_counts"] == {"W0|dry_air": 12800, "W1|w1_annular_gp_randomised_four": 64000, "W1|w1_annular_gp_randomised_single": 25600}, r5_schedule["environment_case_counts"], "R5 block ladder: 1 dry block, 2 single-fan blocks, 5 four-fan blocks"))
    rows.append(_row("r5_evidence_block_counts_balanced", r5_schedule["evidence_block_counts"] == {block_id: 12800 for block_id in R5_EVIDENCE_BLOCK_IDS}, r5_schedule["evidence_block_counts"], "12800 rows per R5 evidence block"))
    w3_row_source = inspect.getsource(_w3_row_for_index)
    rows.append(_row("r7_uses_direct_r5_frozen_bundle_not_active_w2_gate", R5_INPUT_KIND == "r5_frozen_bundle_direct", R5_INPUT_KIND, "r5_frozen_bundle_direct"))
    rows.append(_row("r7_environment_cases_dry_air_annular_gp_single_four", W3_ENVIRONMENT_CASES == ("dry_air", "w3_randomised_single", "w3_randomised_four"), W3_ENVIRONMENT_CASES, ("dry_air", "w3_randomised_single", "w3_randomised_four")))
    rows.append(_row("r7_evidence_blocks_anchor_plus_uncertainty_family", len(R7_EVIDENCE_BLOCK_IDS) == 8 and {"r7_anchor_dry_air", "r7_anchor_single_fan_fixed", "r7_anchor_four_fan_fixed", "r7_random_arena_wide"}.issubset(set(R7_EVIDENCE_BLOCK_IDS)), R7_EVIDENCE_BLOCK_IDS, "8 R7 blocks with dry/single/four anchors plus held-out uncertainty-family blocks"))
    rows.append(_row("r7_active_fan_count_sequence_balanced_0_1_2_3_4", W3_ACTIVE_FAN_COUNT_SEQUENCE == (0, 1, 2, 3, 4), W3_ACTIVE_FAN_COUNT_SEQUENCE, (0, 1, 2, 3, 4)))
    rows.append(_row("r7_row_scheduler_uses_r5_selected_transition_entry_class", "_start_family_for_r5_selected_entry_class" in w3_row_source and "r5_selected_transition_entry_class" in w3_row_source, "R5 selected transition-entry W3 scheduler source", "start family selected from r5_transition_selected_for_r7 transition_entry_class"))
    r8_representatives_source = inspect.getsource(_representatives_for_case)
    r8_selection_source = inspect.getsource(_coverage_medoid_selection)
    rows.append(_row("r8_library_cases_group_by_primitive_and_transition_entry", 'groupby(["primitive_id", "transition_entry_class"]' in r8_representatives_source, "active grouping source", 'W3-eligible frame grouped by ["primitive_id", "transition_entry_class"]'))
    rows.append(_row("r8_selection_applies_hard_safety_filter_first", "_hard_safety_filtered_group" in r8_selection_source, "coverage selection source", "_hard_safety_filtered_group before scoring"))
    rows.append(_row("r8_compressed_cases_use_coverage_medoid_policy", _r8_compressed_cases_use_coverage_medoid_policy(), _r8_library_selection_policies(), "heavy/balanced/light/super_light use coverage_medoid; no_cluster keeps all W3-eligible transition objects"))
    rows.append(_row("r8_launch_gate_budget_override_active", _r8_launch_gate_budget_override_active(), _r8_effective_budget_table(), "launch_gate keeps wider step-0 medoid budgets while non-launch budgets stay compact"))
    rows.append(_row("r8_heavy_medoid_prefers_worst_case_coverage", _r8_heavy_medoid_prefers_worst_case_coverage(), "synthetic coverage-medoid selection", "select existing variant with stronger worst-case coverage"))
    rows.append(_row("r8_medoid_preserves_speed_bin_coverage_when_budget_allows", _r8_speed_bin_medoid_preserves_distinct_bins(), "synthetic speed-bin medoid selection", "compressed R8 cases preserve distinct W3-eligible local LQR speed bins up to case budget"))
    rows.append(_row("r8_representative_score_uses_updraft_gain_not_net_energy", _r8_score_uses_updraft_gain_not_net_energy(), "updraft-gain score check", "net energy residual must not improve representative score"))
    rows.append(_row("five_library_size_cases", set(LIBRARY_SIZE_CASE_IDS) == {"heavy_cluster", "balanced_cluster", "light_cluster", "super_light_cluster", "no_cluster_no_merge"}, LIBRARY_SIZE_CASE_IDS, "heavy/balanced/light/super_light/no_cluster"))
    rows.append(_row("r9_reduced_internal_preflight_blocks_exact", _block_tuples(R9_BLOCKS) == (("no_updraft", "W0", "dry_air", 1), ("single_fan", "W2", "annular_gp_single", 1), ("four_fan", "W2", "annular_gp_four", 1)), _block_tuples(R9_BLOCKS), "1 no-updraft, 1 single-fan, 1 four-fan internal preflight cases"))
    rows.append(_row("r10_single_l7_training_block_exact", _block_tuples(R10_BLOCKS) == ((R10_L7_FULL_DOMAIN_RANDOMISATION_BLOCK_ID, "W3", "w3_randomised_four", 50),), _block_tuples(R10_BLOCKS), "one hard L7 full-domain randomisation training block with 50 outer cases"))
    expected_r11_blocks = (
        (R11_L0_DRY_AIR_FIXED_BLOCK_ID, "W0", "dry_air", 50),
        (R11_L1_SINGLE_FAN_FIXED_NOMINAL_BLOCK_ID, "W2", "annular_gp_single", 50),
        (R11_L2_FOUR_FAN_FIXED_NOMINAL_BLOCK_ID, "W2", "annular_gp_four", 50),
        (R11_L3_FAN_PARAMETER_UNCERTAINTY_BLOCK_ID, "W3", "w3_randomised_four", 50),
        (R11_L4_LOCAL_FAN_POSITION_UNCERTAINTY_BLOCK_ID, "W3", "w3_randomised_four", 50),
        (R11_L5_ACTIVE_FAN_COUNT_UNCERTAINTY_BLOCK_ID, "W3", "w3_randomised_four", 50),
        (R11_L6_ENVIRONMENT_ONLY_FULL_UNCERTAINTY_BLOCK_ID, "W3", "w3_randomised_four", 50),
        (R11_L7_FULL_DOMAIN_RANDOMISATION_BLOCK_ID, "W3", "w3_randomised_four", 50),
    )
    rows.append(_row("r11_fidelity_ladder_blocks_exact", _block_tuples(R11_BLOCKS) == expected_r11_blocks, _block_tuples(R11_BLOCKS), "L0 dry, L1 single fixed, L2 four fixed, L3 fan params, L4 local position, L5 active count, L6 environment-only, L7 full-domain"))
    rows.append(_row("r10_r11_arena_wide_fan_positions_nonoverlap_radius_0p5m", R10_ARENA_WIDE_FAN_POSITION_SAFETY_RADIUS_M == 0.5, R10_ARENA_WIDE_FAN_POSITION_SAFETY_RADIUS_M, "0.5 m radius; centers must be at least 1.0 m apart"))
    rows.append(_row("r9_is_internal_reduced_preflight", R9_PROTOCOL.validation_evidence_level == "internal_fixed_case_outer_loop_preflight_initial_governor_tuning_not_thesis_evidence", R9_PROTOCOL.validation_evidence_level, "internal_fixed_case_outer_loop_preflight_initial_governor_tuning_not_thesis_evidence"))
    rows.append(_row("r10_is_governor_learning_not_final_claim_gate", R10_PROTOCOL.validation_evidence_level == "changed_case_viability_governor_learning_rollout_validation_not_final_claim_gate", R10_PROTOCOL.validation_evidence_level, "changed_case_viability_governor_learning_rollout_validation_not_final_claim_gate"))
    rows.append(_row("r10_relaxed_failure_gate_allows_bounded_floor_ceiling_violations", R10_PROTOCOL.max_hard_failure_rate == 0.20 and R10_PROTOCOL.max_floor_or_ceiling_violation_rate == 0.20, {"hard_failure": R10_PROTOCOL.max_hard_failure_rate, "floor_or_ceiling": R10_PROTOCOL.max_floor_or_ceiling_violation_rate}, "R10 tuning gate allows bounded failures under full-domain randomisation"))
    rows.append(_row("r11_is_strict_heldout_validation", R11_PROTOCOL.validation_evidence_level == "strict_heldout_environment_only_changed_case_repeated_launch_rollout_validation", R11_PROTOCOL.validation_evidence_level, "strict_heldout_environment_only_changed_case_repeated_launch_rollout_validation"))
    rows.append(_row("r11_strict_floor_ceiling_gate_zero", R11_PROTOCOL.max_floor_or_ceiling_violation_rate == 0.0, R11_PROTOCOL.max_floor_or_ceiling_violation_rate, 0.0))
    rows.append(_row("r11_gates_full_safe_success", R11_PROTOCOL.min_full_safe_success_rate == 0.99, R11_PROTOCOL.min_full_safe_success_rate, 0.99))
    rows.append(_row("r9_quick_preflight_policy_history_conditions", R9_POLICY_HISTORY_CONDITIONS == ("no_memory_baseline", "spatial_flow_belief_memory_h3", "spatial_flow_belief_memory_h10", "spatial_flow_belief_memory_h30"), R9_POLICY_HISTORY_CONDITIONS, "no_memory_baseline + h3/h10/h30"))
    rows.append(_row("r9_expected_final_launches_quick_preflight", R9_EXPECTED_FINAL_HELDOUT_LAUNCHES == len(LIBRARY_SIZE_CASE_IDS) * len(R9_POLICY_HISTORY_CONDITIONS) * R9_OUTER_CASES_PER_CONDITION == 60, R9_EXPECTED_FINAL_HELDOUT_LAUNCHES, "library_cases*4*3=60"))
    rows.append(_row("r9_expected_history_launches_quick_preflight", R9_EXPECTED_HISTORY_LAUNCHES == len(LIBRARY_SIZE_CASE_IDS) * R9_OUTER_CASES_PER_CONDITION * HISTORY_LENGTH_SUM == 645, R9_EXPECTED_HISTORY_LAUNCHES, "library_cases*3*(h3+h10+h30)=645"))
    rows.append(_row("r10_expected_final_launches_l7_training", R10_EXPECTED_FINAL_HELDOUT_LAUNCHES == len(LIBRARY_SIZE_CASE_IDS) * len(POLICY_HISTORY_CONDITIONS) * R10_OUTER_CASES_PER_CONDITION, R10_EXPECTED_FINAL_HELDOUT_LAUNCHES, "library_cases*4*50"))
    rows.append(_row("r10_expected_history_launches_l7_training", R10_EXPECTED_HISTORY_LAUNCHES == len(LIBRARY_SIZE_CASE_IDS) * R10_OUTER_CASES_PER_CONDITION * HISTORY_LENGTH_SUM, R10_EXPECTED_HISTORY_LAUNCHES, "library_cases*50*(h3+h10+h30)"))
    rows.append(_row("r11_expected_final_launches_fidelity_ladder", R11_EXPECTED_FINAL_HELDOUT_LAUNCHES == len(LIBRARY_SIZE_CASE_IDS) * len(R11_POLICY_HISTORY_CONDITIONS) * R11_OUTER_CASES_PER_CONDITION, R11_EXPECTED_FINAL_HELDOUT_LAUNCHES, "library_cases*5*400"))
    rows.append(_row("r11_expected_history_launches_fidelity_ladder", R11_EXPECTED_HISTORY_LAUNCHES == len(LIBRARY_SIZE_CASE_IDS) * R11_OUTER_CASES_PER_CONDITION * HISTORY_LENGTH_SUM, R11_EXPECTED_HISTORY_LAUNCHES, "library_cases*400*(h3+h10+h30)"))
    rows.append(_row("four_policy_history_conditions_core", len(POLICY_HISTORY_CONDITIONS) == 4, len(POLICY_HISTORY_CONDITIONS), 4))
    rows.append(_row("history_lengths_core_exact", HISTORY_LENGTHS == (3, 10, 30), HISTORY_LENGTHS, (3, 10, 30)))
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
    rows.append(_row("active_governor_uses_front_wall_mission_terms", hasattr(DEFAULT_GOVERNOR_CONFIG, "mission_front_progress_weight") and hasattr(DEFAULT_GOVERNOR_CONFIG, "mission_terminal_energy_weight"), DEFAULT_GOVERNOR_CONFIG.config_id, "front-wall progress and terminal-energy proxy fields"))
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
    rows.append(_row("outer_loop_governor_learning_strategy_two_level", OUTER_LOOP_GOVERNOR_LEARNING_STRATEGY_VERSION == "case_local_online_memory_plus_r10_global_deterministic_calibration_v1", OUTER_LOOP_GOVERNOR_LEARNING_STRATEGY_VERSION, "case-local online memory plus R10 aggregate deterministic calibration"))
    rows.append(_row("online_memory_scope_case_local_reset", ONLINE_MEMORY_SCOPE == "case_local_reset_per_final_schedule_row", ONLINE_MEMORY_SCOPE, "case_local_reset_per_final_schedule_row"))
    rows.append(_row("r10_global_calibration_scope_all_cases", R10_GLOBAL_CALIBRATION_SCOPE == "aggregate_all_r10_final_heldout_rows_and_selector_opportunity_diagnostics", R10_GLOBAL_CALIBRATION_SCOPE, "aggregate all R10 final held-out rows plus selector opportunity diagnostics"))
    rows.append(_row("r11_single_frozen_handoff_scope", R11_GOVERNOR_HANDOFF_SCOPE == "single_frozen_r10_governor_config_used_for_r11_validation", R11_GOVERNOR_HANDOFF_SCOPE, "single frozen R10 config"))
    rows.append(_row("governor_calibration_no_profile_ladder_or_black_box", GOVERNOR_CALIBRATION_SEARCH_POLICY == "deterministic_bounded_rule_update_no_profile_ladder_no_black_box_search", GOVERNOR_CALIBRATION_SEARCH_POLICY, "deterministic bounded update; no profile ladder or black-box search"))

    route0 = validation_route_for_primitive_step(0)
    route1 = validation_route_for_primitive_step(1)
    rows.append(_row("first_primitive_uses_launch_gate_entry_class", route0.get("route_required_entry_class") == "launch_gate", route0, "launch_gate"))
    rows.append(_row("later_nominal_route_uses_inflight_entry_class", route1.get("route_required_entry_class") == "inflight_stable", route1, "inflight_stable"))
    rows.append(_row("r10_l7_arena_wide_varies_fan_positions_and_active_count", _fan_position_policy_for_outer_case(protocol=R10_PROTOCOL, environment_block_id=R10_L7_FULL_DOMAIN_RANDOMISATION_BLOCK_ID) == "independent_uniform_xy_bounds" and _active_fan_count_policy_for_outer_case(protocol=R10_PROTOCOL, environment_block_id=R10_L7_FULL_DOMAIN_RANDOMISATION_BLOCK_ID) == "balanced_0_1_2_3_4_with_arena_wide_fan_position_generalisation", {"fan_position": _fan_position_policy_for_outer_case(protocol=R10_PROTOCOL, environment_block_id=R10_L7_FULL_DOMAIN_RANDOMISATION_BLOCK_ID), "active_count": _active_fan_count_policy_for_outer_case(protocol=R10_PROTOCOL, environment_block_id=R10_L7_FULL_DOMAIN_RANDOMISATION_BLOCK_ID)}, "R10 single hard L7 training distribution"))
    rows.append(_row("r10_l7_uses_full_w3_randomisation", _uses_full_w3_randomisation_block(protocol=R10_PROTOCOL, environment_block_id=R10_L7_FULL_DOMAIN_RANDOMISATION_BLOCK_ID), "r10_l7_full_w3_randomisation", True))
    rows.append(_row("r11_l0_no_updraft_zero_fan_block", _scheduled_active_fan_count_for_outer_case(protocol=R11_PROTOCOL, environment_block_id=R11_L0_DRY_AIR_FIXED_BLOCK_ID, environment_block_local_index=0) == 0, _scheduled_active_fan_count_for_outer_case(protocol=R11_PROTOCOL, environment_block_id=R11_L0_DRY_AIR_FIXED_BLOCK_ID, environment_block_local_index=0), 0))
    rows.append(_row("r11_l0_no_updraft_no_fan_positions", _fan_position_policy_for_outer_case(protocol=R11_PROTOCOL, environment_block_id=R11_L0_DRY_AIR_FIXED_BLOCK_ID) == "no_fan_positions", _fan_position_policy_for_outer_case(protocol=R11_PROTOCOL, environment_block_id=R11_L0_DRY_AIR_FIXED_BLOCK_ID), "no_fan_positions"))
    rows.append(_row("r11_l1_l2_nominal_fan_positions_fixed", _fan_position_policy_for_outer_case(protocol=R11_PROTOCOL, environment_block_id=R11_L1_SINGLE_FAN_FIXED_NOMINAL_BLOCK_ID) == "fixed_base_positions" and _fan_position_policy_for_outer_case(protocol=R11_PROTOCOL, environment_block_id=R11_L2_FOUR_FAN_FIXED_NOMINAL_BLOCK_ID) == "fixed_base_positions", {"l1": _fan_position_policy_for_outer_case(protocol=R11_PROTOCOL, environment_block_id=R11_L1_SINGLE_FAN_FIXED_NOMINAL_BLOCK_ID), "l2": _fan_position_policy_for_outer_case(protocol=R11_PROTOCOL, environment_block_id=R11_L2_FOUR_FAN_FIXED_NOMINAL_BLOCK_ID)}, "fixed_base_positions"))
    rows.append(_row("r11_l2_l3_l4_fixed_to_4_active_fans", all(_scheduled_active_fan_count_for_outer_case(protocol=R11_PROTOCOL, environment_block_id=block_id, environment_block_local_index=0) == 4 for block_id in (R11_L2_FOUR_FAN_FIXED_NOMINAL_BLOCK_ID, R11_L3_FAN_PARAMETER_UNCERTAINTY_BLOCK_ID, R11_L4_LOCAL_FAN_POSITION_UNCERTAINTY_BLOCK_ID)), "L2/L3/L4 active counts", 4))
    rows.append(_row("r11_l5_active_fan_variation_policy", _active_fan_count_policy_for_outer_case(protocol=R11_PROTOCOL, environment_block_id=R11_L5_ACTIVE_FAN_COUNT_UNCERTAINTY_BLOCK_ID) == "balanced_0_1_2_3_4_for_active_fan_number_variation", _active_fan_count_policy_for_outer_case(protocol=R11_PROTOCOL, environment_block_id=R11_L5_ACTIVE_FAN_COUNT_UNCERTAINTY_BLOCK_ID), "balanced_0_1_2_3_4_for_active_fan_number_variation"))
    rows.append(_row("r11_l7_arena_wide_varies_fan_positions_and_active_count", _fan_position_policy_for_outer_case(protocol=R11_PROTOCOL, environment_block_id=R11_L7_FULL_DOMAIN_RANDOMISATION_BLOCK_ID) == "independent_uniform_xy_bounds" and _active_fan_count_policy_for_outer_case(protocol=R11_PROTOCOL, environment_block_id=R11_L7_FULL_DOMAIN_RANDOMISATION_BLOCK_ID) == "balanced_0_1_2_3_4_with_arena_wide_fan_position_generalisation", {"fan_position": _fan_position_policy_for_outer_case(protocol=R11_PROTOCOL, environment_block_id=R11_L7_FULL_DOMAIN_RANDOMISATION_BLOCK_ID), "active_count": _active_fan_count_policy_for_outer_case(protocol=R11_PROTOCOL, environment_block_id=R11_L7_FULL_DOMAIN_RANDOMISATION_BLOCK_ID)}, "R11 L7 independent positions plus balanced 0/1/2/3/4 active fan count"))
    rows.append(_row("r11_l7_uses_full_w3_randomisation", _uses_full_w3_randomisation_block(protocol=R11_PROTOCOL, environment_block_id=R11_L7_FULL_DOMAIN_RANDOMISATION_BLOCK_ID), "r11_l7_full_w3_randomisation", True))
    for protocol, full_domain_block_id in ((R10_PROTOCOL, R10_L7_FULL_DOMAIN_RANDOMISATION_BLOCK_ID), (R11_PROTOCOL, R11_L7_FULL_DOMAIN_RANDOMISATION_BLOCK_ID)):
        no_variation_audit = pd.DataFrame(_no_variation_audit_rows(_final_heldout_schedule(outer_cases=_outer_case_schedule(protocol=protocol, seed=91), protocol=protocol)))
        case7 = no_variation_audit[no_variation_audit["environment_block_id"].astype(str).eq(full_domain_block_id)]
        rows.append(_row(f"{protocol.stage_id.lower()}_full_domain_fixed_within_outer_case", not case7.empty and bool(case7["glider_model_fixed"].all()) and bool(case7["fan_layout_fixed_within_outer_case"].all()) and bool(case7["active_fan_count_fixed_within_outer_case"].all()) and bool(case7["variation_audit_passed"].all()), case7.to_dict(orient="records")[:1], "full-domain block samples W3 perturbations across outer cases but fixes layout/count/plant within each outer case"))
        outer = next(row for row in _outer_case_schedule(protocol=protocol, seed=91) if row["environment_block_id"] == full_domain_block_id)
        history = _history_row_for_final({**outer, "episode_id": "audit", "launch_role": "final_heldout"}, 0)
        rows.append(_row(f"{protocol.stage_id.lower()}_history_varies_only_launch_and_updraft_parameters", int(history["environment_seed"]) != int(outer["environment_seed"]) and int(history["environment_layout_seed"]) == int(outer["environment_layout_seed"]) and int(history["environment_active_fan_seed"]) == int(outer["environment_active_fan_seed"]) and history["scheduled_active_fan_count"] == outer["scheduled_active_fan_count"] and int(history["plant_implementation_seed"]) == int(outer["plant_implementation_seed"]), {"outer": outer, "history": history}, "history shifts launch and mild updraft parameter seed while fan layout/count and plant/implementation stay fixed per outer case"))
    rows.append(_row("changed_case_active_fan_count_sequence", R10_ACTIVE_FAN_COUNT_SEQUENCE == (0, 1, 2, 3, 4), R10_ACTIVE_FAN_COUNT_SEQUENCE, (0, 1, 2, 3, 4)))
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
    recovery_initial = [3.0, 2.0, 1.0, 0.80, 0.45, 0.0, 4.0, 0.0, 0.0, 0.90, 0.0, 0.0, 0.0, 0.0, 0.0]
    recovery_exit_progress = [3.0, 2.0, 1.0, 0.74, 0.42, 0.0, 4.0, 0.0, 0.0, 0.80, 0.0, 0.0, 0.0, 0.0, 0.0]
    recovery_exit_no_progress = [3.0, 2.0, 1.0, 0.82, 0.46, 0.0, 4.0, 0.0, 0.0, 0.92, 0.0, 0.0, 0.0, 0.0, 0.0]
    recovery_exit_boundary = [5.30, 2.0, 1.0, 0.74, 0.42, 0.0, 4.0, 0.0, 0.0, 0.80, 0.0, 0.0, 0.0, 0.0, 0.0]
    recovery_progress = classify_transition(
        {
            "entry_role": "transition_object",
            "start_state_family": "inflight_recovery_edge",
            "outcome_class": "accepted",
            "continuation_valid": True,
            "boundary_use_class": "inside",
            "minimum_wall_margin_m": 1.0,
            "floor_margin_m": 0.6,
            "ceiling_margin_m": 1.0,
            "initial_state_vector": recovery_initial,
            "exit_state_vector": recovery_exit_progress,
        }
    )
    recovery_no_progress = classify_transition(
        {
            "entry_role": "transition_object",
            "start_state_family": "inflight_recovery_edge",
            "outcome_class": "accepted",
            "continuation_valid": True,
            "boundary_use_class": "inside",
            "minimum_wall_margin_m": 1.0,
            "floor_margin_m": 0.6,
            "ceiling_margin_m": 1.0,
            "initial_state_vector": recovery_initial,
            "exit_state_vector": recovery_exit_no_progress,
        }
    )
    recovery_boundary = classify_transition(
        {
            "entry_role": "transition_object",
            "start_state_family": "inflight_recovery_edge",
            "outcome_class": "accepted",
            "continuation_valid": True,
            "boundary_use_class": "inside",
            "minimum_wall_margin_m": 1.0,
            "floor_margin_m": 0.6,
            "ceiling_margin_m": 1.0,
            "initial_state_vector": recovery_initial,
            "exit_state_vector": recovery_exit_boundary,
        }
    )
    return [
        _row("transition_contract_row_boundary_near_route_state", bool(transition_contract_row()["boundary_near_is_route_state_not_failure"]), transition_contract_row(), "boundary_near route state"),
        _row("launch_boundary_exit_cannot_pass_transition_gate", not bool(launch_bad["transition_chain_compatible"]), launch_bad, "launch boundary_near exit rejected"),
        _row("launch_post_launch_exit_can_pass_transition_gate", bool(launch_good["transition_chain_compatible"]), launch_good, "launch post_launch_degraded/inflight exit accepted"),
        _row("inflight_recoverable_degraded_exit_cannot_pass_transition_gate", not transition_is_chain_compatible(entry_role="transition_object", entry_class="inflight_stable", exit_class="recoverable_degraded"), "recoverable_degraded", "not allowed for inflight_stable transition object"),
        _row("inflight_weak_without_chain_handoff_rejected", not bool(inflight_bad["transition_chain_compatible"]), inflight_bad, "weak local rollout is not sufficient"),
        _row("recovery_to_inflight_passes_transition_gate", bool(recovery_good["transition_chain_compatible"]), recovery_good, "recoverable_degraded transition object can restore inflight_stable"),
        _row("recovery_self_transition_requires_measurable_progress", bool(recovery_progress["transition_chain_compatible"]) and bool(recovery_progress["recovery_progress_valid"]), recovery_progress, "recoverable_degraded can remain recoverable only with measured progress"),
        _row("recovery_self_transition_without_progress_rejected", not bool(recovery_no_progress["transition_chain_compatible"]), recovery_no_progress, "recoverable_degraded self transition without progress rejected"),
        _row("recovery_to_boundary_near_is_route_not_full_pass", recovery_boundary["exit_class"] == "boundary_near" and not bool(recovery_boundary["transition_chain_compatible"]), recovery_boundary, "boundary_near from recoverable_degraded remains a route/weak condition"),
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
    evidence_block_counts: dict[str, int] = {}
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
        evidence_block_counts[schedule.evidence_block_id] = evidence_block_counts.get(schedule.evidence_block_id, 0) + 1
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
        "evidence_block_counts": dict(sorted(evidence_block_counts.items())),
        "primitive_count_range": (
            min(primitive_counts.values()) if primitive_counts else 0,
            max(primitive_counts.values()) if primitive_counts else 0,
        ),
    }


def _expected_r5_family_counts() -> dict[str, int]:
    return {
        "inflight_boundary_near": 10240,
        "inflight_lift_region": 15360,
        "inflight_nominal": 25600,
        "inflight_recovery_edge": 10240,
        "launch_gate": 40960,
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
    ) == "all_w3_eligible_transition_objects_no_clustering_no_merging"


def _r8_effective_budget_table() -> dict[str, dict[str, int]]:
    return {
        str(case["library_size_case_id"]): {
            "non_launch": representative_budget_for_entry_class(case, "inflight_stable"),
            "launch_gate": representative_budget_for_entry_class(case, "launch_gate"),
        }
        for case in LIBRARY_SIZE_CASES
    }


def _r8_launch_gate_budget_override_active() -> bool:
    budgets = _r8_effective_budget_table()
    return budgets == {
        "heavy_cluster": {"non_launch": 1, "launch_gate": 2},
        "balanced_cluster": {"non_launch": 3, "launch_gate": 5},
        "light_cluster": {"non_launch": 6, "launch_gate": 8},
        "super_light_cluster": {"non_launch": 12, "launch_gate": 12},
        "no_cluster_no_merge": {"non_launch": 1_000_000, "launch_gate": 1_000_000},
    }


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
    memory_final = next(row for row in final_schedule if str(row["policy_id"]) == "spatial_flow_belief_memory_h10")
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
        "two_level_governor_learning_strategy": "The learning strategy is two-level",
        "case_local_online_memory_scope": "case-local and reset per final test row",
        "r10_global_deterministic_calibration": "R10 performs deterministic global calibration",
        "r10_no_profile_ladder_or_black_box": "no profile ladder, Bayesian optimisation, neural tuning, or black-box search",
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
        "recovery_self_transition_progress_gate": "recoverable_degraded -> recoverable_degraded",
        "recovery_boundary_route_not_full_pass": "recoverable_degraded -> boundary_near",
        "r9_quick_preflight_60_final": "60 final held-out launches",
        "r9_quick_preflight_645_history": "645 history launches",
        "r10_single_hard_training_distribution": "R10 is governor/spatial-memory tuning on one hard training distribution",
        "r10_tunes_memory_shield_exploration_handoff": "R10 may tune memory sensitivity, cost-benefit memory weight/cap/cost terms, shield margins, exploration thresholds",
        "r11_eight_block_fidelity_ladder": "R11 is held-out validation on an eight-block fidelity ladder",
        "plant_implementation_fixed_per_outer_case": "plant/implementation are fixed across history/final launches",
        "outer_loop_realtime_scheduler_profile": "preferred 20 ms controller-slot budget and a hard 0.100 s primitive-boundary budget",
        "memory_opportunity_audit": "memory_opportunity_summary.csv",
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
            "real_active_logic": "expected_updraft_gain_proxy_m remains the representative soft reward; expected_energy_residual_m is used for specific-energy spatial-memory comparison and compatibility audit output.",
            "intentional_status": "not_representative_score_active_for_memory_residual_audit",
        },
        {
            "legacy_alias": "belief_local_energy_residual_m",
            "real_active_logic": "belief_local_specific_energy_residual_m and belief_candidate_path_memory_utility_m are the active total-specific-energy-dominant memory fields; updraft residual remains auxiliary.",
            "intentional_status": "active_specific_energy_memory_with_compatibility_alias",
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
