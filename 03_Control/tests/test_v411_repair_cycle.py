from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from directional_residual_lift_belief import (
    DirectionalResidualObservation,
    directional_residual_observation_from_rows,
    initial_directional_residual_lift_belief,
    query_directional_residual_lift_features,
    update_directional_residual_lift_belief,
)
from episode_selector import select_compact_representative
from primitive_timing_contract import (
    CONTROLLER_INPUT_SLOTS_PER_PRIMITIVE,
    CONTROLLER_INPUT_UPDATE_PERIOD_S,
    PRIMITIVE_FINITE_HORIZON_S,
    PRIMITIVE_TIMING_CONTRACT_VERSION,
    primitive_step_count,
)
from primitive_variant_registry import _variant_id
from run_post_w3_cluster_merge import run_post_w3_cluster_merge
from run_post_w3_library_size_study import LIBRARY_SIZE_CASE_IDS
from run_changed_case_validation import ChangedCaseValidationConfig
from run_repeated_launch_learning_curve import HISTORY_LENGTHS, RepeatedLaunchValidationConfig, _read_outcome_rows
from run_repeated_launch_learning_curve import (
    BOUNDARY_RECOVERY_START_FAMILY,
    FIRST_PRIMITIVE_START_FAMILY,
    LAUNCH_SEQUENCE_POLICY_ID,
    POST_LAUNCH_START_FAMILY,
    TERMINAL_SAFE_EXIT_START_FAMILY,
    _episode_specific_energy_summary,
    _launch_score_fields,
    _paired_safe_explore_delta_rows,
    _paired_score_delta_rows,
    _paired_score_delta_summary,
    validation_route_for_primitive_step,
)
from run_v411_source_audit import (
    SourceAuditConfig,
    build_control_inventory,
    classify_control_file,
    discover_superseded_result_roots,
    inspect_required_docs,
    run_v411_source_audit,
    write_diagnostic_not_passed_archive,
)
from state_contract import STATE_INDEX, STATE_SIZE


def test_v411_docs_gate_inventory_and_audit_are_ready() -> None:
    docs = inspect_required_docs(Path("docs"))
    assert docs
    assert all(row["readable"] for row in docs)
    assert {row["doc_name"] for row in docs} >= {
        "Glider_Control_Project_Plan.md",
        "Python Coding to CODEX.txt",
        "PR.txt",
    }

    assert classify_control_file("03_Primitives/prim_cat.py") == "active_repair_cycle"
    assert classify_control_file("04_Scenarios/run_v410_source_audit.py") == "retired_not_active"
    inventory = build_control_inventory(Path("03_Control"))
    assert any(row["classification"] == "active_repair_cycle" for row in inventory)

    result = run_v411_source_audit(SourceAuditConfig(dry_run=True))
    assert result["status"] == "ready"
    assert result["blockers"] == []


def test_v411_timing_contract_exact_five_slots_and_variant_id_sensitivity() -> None:
    assert PRIMITIVE_FINITE_HORIZON_S == 0.100
    assert CONTROLLER_INPUT_UPDATE_PERIOD_S == 0.020
    assert CONTROLLER_INPUT_SLOTS_PER_PRIMITIVE == 5
    assert primitive_step_count() == 5

    common = {
        "primitive_id": "glide",
        "entry_role": "inflight_only",
        "reference_state_vector": "[0,0,0]",
        "reference_command_vector": "[0,0,0]",
        "controller_id": "ctrl",
        "linearisation_id": "lin",
        "q_json": "{}",
        "r_json": "{}",
        "gain_checksum": "k",
        "timing_augmentation_type": "aug",
        "timing_design_version": "design",
        "sample_time_s": CONTROLLER_INPUT_UPDATE_PERIOD_S,
        "latency_case": "nominal",
        "command_delay_steps": 1,
        "predictor_horizon_steps": 1,
        "augmented_a_checksum": "a",
        "augmented_b_checksum": "b",
        "augmented_q_json": "{}",
        "augmented_r_json": "{}",
        "augmented_gain_checksum": "g",
        "exit_check_version": "exit",
    }
    v411_id = _variant_id(
        **common,
        finite_horizon_s=0.100,
        controller_input_slots_per_primitive=5,
        controller_input_update_period_s=0.020,
        primitive_timing_contract_version=PRIMITIVE_TIMING_CONTRACT_VERSION,
    )
    old_id = _variant_id(
        **common,
        finite_horizon_s=0.800,
        controller_input_slots_per_primitive=40,
        controller_input_update_period_s=0.020,
        primitive_timing_contract_version="legacy",
    )
    assert v411_id != old_id


def test_v411_dynamic_archive_marks_old_roots_diagnostic_not_passed(tmp_path: Path) -> None:
    old_root = tmp_path / "results" / "w01_dense" / "001"
    (old_root / "manifests").mkdir(parents=True)
    (old_root / "metrics").mkdir()
    (old_root / "manifests" / "run_manifest.json").write_text(
        json.dumps({"project_title_version": "LQR-Stabilised Contextual Primitive v4.10"}, indent=2),
        encoding="ascii",
    )
    (old_root / "metrics" / "primitive_variant_registry.csv").write_text(
        "primitive_variant_id,finite_horizon_s\nold,0.8\n",
        encoding="ascii",
    )

    roots = discover_superseded_result_roots(tmp_path / "results")
    assert len(roots) == 1
    assert roots[0]["diagnostic_archive_status"] == "diagnostic_not_passed"
    manifest = write_diagnostic_not_passed_archive(
        archive_root=tmp_path / "archive" / "diagnostic_not_passed_v410" / "001",
        docs=[],
        inventory=[],
        superseded_roots=roots,
    )
    archive_path = Path(manifest["manifest_path"])
    assert archive_path.is_file()
    assert "diagnostic_not_passed" in archive_path.read_text(encoding="ascii")


def test_v411_directional_memory_and_safe_exploration_after_filtering() -> None:
    belief = initial_directional_residual_lift_belief()
    belief = update_directional_residual_lift_belief(
        belief,
        DirectionalResidualObservation(
            x_w_m=0.0,
            y_w_m=0.0,
            z_w_m=1.5,
            direction_rad=0.0,
            lift_residual_m_s=0.2,
            energy_residual_m=0.1,
            dwell_residual_s=0.05,
        ),
    )
    features = query_directional_residual_lift_features(
        belief,
        x_w_m=0.0,
        y_w_m=0.0,
        z_w_m=1.5,
        direction_rad=0.0,
    )
    assert features["belief_local_lift_residual_m_s"] > 0.0
    assert features["belief_direction_bin"] == 0

    timing_payload = {
        "finite_horizon_s": 0.1,
        "controller_input_slots_per_primitive": 5,
        "controller_input_update_period_s": 0.02,
        "primitive_timing_contract_version": PRIMITIVE_TIMING_CONTRACT_VERSION,
    }
    representatives = [
        {
            "compact_library_id": "good",
            "primitive_variant_id": "good",
            "primitive_id": "launch_capture_glide_stabilise",
            "entry_role": "launch_capable",
            "controller_id": "ctrl",
            "K_gain_checksum": "k",
            "augmented_A_checksum": "a",
            "augmented_B_checksum": "b",
            "augmented_gain_checksum": "g",
            "library_size_case_id": "balanced_cluster",
            **timing_payload,
        },
        {
            "compact_library_id": "bad",
            "primitive_variant_id": "bad",
            "primitive_id": "lift_entry",
            "entry_role": "inflight_only",
            "controller_id": "ctrl",
            "K_gain_checksum": "k",
            "augmented_A_checksum": "a",
            "augmented_B_checksum": "b",
            "augmented_gain_checksum": "g",
            "library_size_case_id": "balanced_cluster",
            **timing_payload,
        },
    ]
    outcome_rows = {
        "good": {"continuation_probability": 0.8, "hard_failure_risk": 0.1},
        "bad": {"continuation_probability": 0.9, "hard_failure_risk": 0.1},
    }
    selected, rows = select_compact_representative(
        representatives=representatives,
        outcome_rows_by_variant_id=outcome_rows,
        context={
            "start_state_family": "launch_gate",
            "wall_margin_m": 0.5,
            "floor_margin_m": 0.5,
            "ceiling_margin_m": 0.5,
            "speed_margin_m_s": 1.0,
            "latency_case": "nominal",
            "history_length": 5,
            "library_size_case_id": "balanced_cluster",
        },
        governor_mode="continuation_mode",
        belief_features=features,
    )
    assert selected is not None
    viable = {row["primitive_variant_id"]: row for row in rows}
    assert viable["good"]["safe_exploration_status"] == "applied_after_viability_filter"
    assert viable["good"]["exploration_score_component"] > 0.0
    assert viable["bad"]["safe_exploration_status"] == "not_applied_rejected_before_exploration"
    assert viable["bad"]["exploration_score_component"] == 0.0
    try:
        directional_residual_observation_from_rows(
            expected_row={},
            observed_row={"initial_x_n": 1.0, "initial_y_e": 2.0, "initial_z_w": 1.5},
            direction_rad=0.0,
        )
    except ValueError as exc:
        assert "directional_residual_memory_missing_canonical_coordinate" in str(exc)
    else:
        raise AssertionError("old non-canonical coordinate fields must not silently pass")


def test_r9_r10_launch_sequence_routes_launch_inflight_and_state_recovery_selection() -> None:
    timing_payload = {
        "finite_horizon_s": 0.1,
        "controller_input_slots_per_primitive": 5,
        "controller_input_update_period_s": 0.02,
        "primitive_timing_contract_version": PRIMITIVE_TIMING_CONTRACT_VERSION,
    }
    representatives = [
        {
            "compact_library_id": "launch",
            "primitive_variant_id": "launch",
            "primitive_id": "launch_capture_glide_stabilise",
            "entry_role": "launch_capable",
            "controller_id": "ctrl_launch",
            "K_gain_checksum": "k",
            "augmented_A_checksum": "a",
            "augmented_B_checksum": "b",
            "augmented_gain_checksum": "g",
            "library_size_case_id": "balanced_cluster",
            **timing_payload,
        },
        {
            "compact_library_id": "glide",
            "primitive_variant_id": "glide",
            "primitive_id": "glide",
            "entry_role": "inflight_only",
            "controller_id": "ctrl_glide",
            "K_gain_checksum": "k",
            "augmented_A_checksum": "a",
            "augmented_B_checksum": "b",
            "augmented_gain_checksum": "g",
            "library_size_case_id": "balanced_cluster",
            **timing_payload,
        },
        {
            "compact_library_id": "recovery",
            "primitive_variant_id": "recovery",
            "primitive_id": "recovery",
            "entry_role": "terminal_or_recovery",
            "controller_id": "ctrl_recovery",
            "K_gain_checksum": "k",
            "augmented_A_checksum": "a",
            "augmented_B_checksum": "b",
            "augmented_gain_checksum": "g",
            "library_size_case_id": "balanced_cluster",
            **timing_payload,
        },
    ]
    outcome_rows = {
        "launch": {"continuation_probability": 0.7, "hard_failure_risk": 0.1},
        "glide": {"continuation_probability": 0.9, "hard_failure_risk": 0.1},
        "recovery": {"continuation_probability": 0.2, "terminal_useful_probability": 0.9, "hard_failure_risk": 0.1},
    }
    base_context = {
        "context_id": "ctx",
        "W_layer": "W2",
        "environment_mode": "annular_gp_single",
        "wall_margin_m": 0.5,
        "floor_margin_m": 0.5,
        "ceiling_margin_m": 0.5,
        "speed_margin_m_s": 1.0,
        "latency_case": "nominal",
        "history_length": 0,
        "library_size_case_id": "balanced_cluster",
        "launch_sequence_policy": LAUNCH_SEQUENCE_POLICY_ID,
    }

    nominal_state = np.zeros(STATE_SIZE, dtype=float)
    nominal_state[STATE_INDEX["x_w"]] = 3.0
    nominal_state[STATE_INDEX["y_w"]] = 2.0
    nominal_state[STATE_INDEX["z_w"]] = 1.5
    nominal_state[STATE_INDEX["u"]] = 5.0
    boundary_state = nominal_state.copy()
    boundary_state[STATE_INDEX["x_w"]] = 6.45
    recovery_edge_state = nominal_state.copy()
    recovery_edge_state[STATE_INDEX["u"]] = 3.6

    first_route = validation_route_for_primitive_step(0, state=nominal_state)
    second_route = validation_route_for_primitive_step(1, state=nominal_state)
    boundary_route = validation_route_for_primitive_step(1, state=boundary_state)
    recovery_edge_route = validation_route_for_primitive_step(1, state=recovery_edge_state)

    first_context = {**base_context, **first_route}
    second_context = {**base_context, **second_route}
    boundary_context = {**base_context, **boundary_route, "wall_margin_m": 0.15}
    recovery_context = {**base_context, **recovery_edge_route}
    first_selected, first_rows = select_compact_representative(
        representatives=representatives,
        outcome_rows_by_variant_id=outcome_rows,
        context=first_context,
        governor_mode="continuation_mode",
    )
    second_selected, second_rows = select_compact_representative(
        representatives=representatives,
        outcome_rows_by_variant_id=outcome_rows,
        context=second_context,
        governor_mode="continuation_mode",
    )
    boundary_selected, boundary_rows = select_compact_representative(
        representatives=representatives,
        outcome_rows_by_variant_id=outcome_rows,
        context=boundary_context,
        governor_mode="terminal_episode_mode",
    )
    recovery_selected, recovery_rows = select_compact_representative(
        representatives=representatives,
        outcome_rows_by_variant_id=outcome_rows,
        context=recovery_context,
        governor_mode="terminal_episode_mode",
    )

    assert FIRST_PRIMITIVE_START_FAMILY == "launch_gate"
    assert POST_LAUNCH_START_FAMILY == "inflight_nominal"
    assert BOUNDARY_RECOVERY_START_FAMILY == "inflight_boundary_near"
    assert TERMINAL_SAFE_EXIT_START_FAMILY == "inflight_recovery_edge"
    assert first_route["route_required_entry_role"] == "launch_capable"
    assert second_route["route_required_entry_role"] == "inflight_only"
    assert boundary_route["route_required_entry_role"] == "terminal_or_recovery"
    assert recovery_edge_route["route_required_entry_role"] == "terminal_or_recovery"
    assert first_selected["entry_role"] == "launch_capable"
    assert second_selected["entry_role"] == "inflight_only"
    assert boundary_selected["entry_role"] == "terminal_or_recovery"
    assert recovery_selected["entry_role"] == "terminal_or_recovery"
    assert {row["entry_role"]: row["viable"] for row in first_rows} == {
        "launch_capable": True,
        "inflight_only": False,
        "terminal_or_recovery": False,
    }
    assert {row["entry_role"]: row["viable"] for row in second_rows} == {
        "launch_capable": False,
        "inflight_only": True,
        "terminal_or_recovery": False,
    }
    assert {row["entry_role"]: row["viable"] for row in boundary_rows} == {
        "launch_capable": False,
        "inflight_only": False,
        "terminal_or_recovery": True,
    }
    assert {row["entry_role"]: row["viable"] for row in recovery_rows} == {
        "launch_capable": False,
        "inflight_only": False,
        "terminal_or_recovery": True,
    }


def test_r9_r10_launch_score_uses_specific_energy_loss_and_paired_deltas() -> None:
    def state(*, z: float, u: float) -> np.ndarray:
        x = np.zeros(STATE_SIZE, dtype=float)
        x[STATE_INDEX["z_w"]] = z
        x[STATE_INDEX["u"]] = u
        return x

    primitive_rows = [
        {
            "initial_state_vector": json.dumps(state(z=1.0, u=4.0).tolist()),
            "exit_state_vector": json.dumps(state(z=1.4, u=5.0).tolist()),
        },
        {
            "initial_state_vector": json.dumps(state(z=1.4, u=5.0).tolist()),
            "exit_state_vector": json.dumps(state(z=1.3, u=4.5).tolist()),
        },
    ]
    energy = _episode_specific_energy_summary(primitive_rows)
    assert energy["net_specific_energy_delta_m"] > 0.0
    assert energy["gross_specific_energy_gain_m"] > 0.0
    assert energy["gross_specific_energy_loss_m"] > 0.0

    safe_row = {
        "safe_success": True,
        "terminal_useful": False,
        "lift_capture": True,
        "hard_failure": False,
        "floor_or_ceiling_violation": False,
        "no_viable_primitive": False,
        "selected_primitive_step_count": 15,
        "min_wall_margin_m": 0.08,
        **energy,
    }
    hard_failure = {**safe_row, "hard_failure": True}
    no_viable_launch = {**safe_row, "safe_success": False, "lift_capture": False, "no_viable_primitive": True, "selected_primitive_step_count": 0}

    assert _launch_score_fields(safe_row)["launch_score"] > 0.0
    assert _launch_score_fields(hard_failure)["launch_score"] == -100.0
    assert _launch_score_fields(no_viable_launch)["launch_score"] == -70.0

    final = pd.DataFrame(
        [
            {
                "library_size_case_id": "balanced_cluster",
                "common_final_launch_key": "paired_001",
                "outer_case_index": 1,
                "policy_id": "no_memory_baseline",
                "history_length": 0,
                "launch_score": 10.0,
                "safe_success": True,
                "hard_failure": False,
                "floor_or_ceiling_violation": False,
                "no_viable_primitive": False,
                "net_specific_energy_delta_m": 0.1,
                "gross_specific_energy_loss_m": 0.2,
                "episode_flight_time_s": 1.0,
                "memory_changed_selection": False,
                "exploration_changed_selection": False,
            },
            {
                "library_size_case_id": "balanced_cluster",
                "common_final_launch_key": "paired_001",
                "outer_case_index": 1,
                "policy_id": "directional_3d_residual_memory_h20",
                "history_length": 20,
                "launch_score": 25.0,
                "safe_success": True,
                "hard_failure": False,
                "floor_or_ceiling_violation": False,
                "no_viable_primitive": False,
                "net_specific_energy_delta_m": 0.4,
                "gross_specific_energy_loss_m": 0.1,
                "episode_flight_time_s": 1.5,
                "memory_changed_selection": True,
                "exploration_changed_selection": False,
            },
            {
                "library_size_case_id": "balanced_cluster",
                "common_final_launch_key": "paired_001",
                "outer_case_index": 1,
                "policy_id": "safe_explore_then_exploit_h20",
                "history_length": 20,
                "launch_score": 30.0,
                "safe_success": True,
                "hard_failure": False,
                "floor_or_ceiling_violation": False,
                "no_viable_primitive": False,
                "net_specific_energy_delta_m": 0.5,
                "gross_specific_energy_loss_m": 0.1,
                "episode_flight_time_s": 1.5,
                "memory_changed_selection": True,
                "exploration_changed_selection": True,
            },
        ]
    )
    memory_delta = _paired_score_delta_rows(final, baseline_policy_id="no_memory_baseline")
    explore_delta = _paired_safe_explore_delta_rows(final)
    summary = _paired_score_delta_summary(pd.concat([memory_delta, explore_delta], ignore_index=True))

    assert set(memory_delta["comparison_type"]) == {"memory_vs_no_memory"}
    assert float(memory_delta.loc[memory_delta["policy_id"] == "directional_3d_residual_memory_h20", "paired_delta_launch_score"].iloc[0]) == 15.0
    assert set(explore_delta["comparison_type"]) == {"safe_explore_vs_matching_memory"}
    assert float(explore_delta["paired_delta_launch_score"].iloc[0]) == 5.0
    assert not summary.empty


def test_v411_case_ids_histories_and_retired_gate(tmp_path: Path) -> None:
    assert LIBRARY_SIZE_CASE_IDS == (
        "heavy_cluster",
        "balanced_cluster",
        "light_cluster",
        "super_light_cluster",
        "no_cluster_no_merge",
    )
    assert HISTORY_LENGTHS == (0, 5, 10, 20, 50, 100)
    result = run_post_w3_cluster_merge(
        input_root=tmp_path / "missing_w3",
        output_root=tmp_path / "post_w3_cluster",
        run_id=1,
    )
    assert result["status"] == "blocked"
    assert result["blocked_reason"] == "retired_diagnostic_requires_explicit_allow_retired_diagnostic"


def test_r9_repair_uses_compact_outcome_keys_and_full_multi_step_default(tmp_path: Path) -> None:
    outcome_root = tmp_path / "outcome"
    metrics = outcome_root / "metrics"
    metrics.mkdir(parents=True)
    (metrics / "outcome_model_table.csv").write_text(
        "\n".join(
            [
                "library_size_case_id,compact_library_id,primitive_variant_id,continuation_probability,hard_failure_risk",
                "heavy_cluster,heavy_rep,shared_variant,0.2,0.1",
                "balanced_cluster,balanced_rep,shared_variant,0.9,0.1",
            ]
        )
        + "\n",
        encoding="ascii",
    )

    rows = _read_outcome_rows(outcome_root)

    assert rows["heavy_rep"]["continuation_probability"] == 0.2
    assert rows["balanced_rep"]["continuation_probability"] == 0.9
    assert rows["heavy_cluster|shared_variant|heavy_rep"]["continuation_probability"] == 0.2
    assert RepeatedLaunchValidationConfig().max_primitives_per_launch == 4
    assert ChangedCaseValidationConfig().max_primitives_per_launch == 4
