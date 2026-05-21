from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from episode_logging import write_episode_log
from episode_schema import (
    assert_claim_safe_text,
    unsupported_claim_errors,
)
from fixed_gate_code_path_map import active_code_path_text, deprecated_default_paths
from fixed_gate_contract import (
    inside_fixed_launch_gate,
    inside_launch_tolerance_shell,
    launch_gate_admission_status,
    state_to_launch_gate_record,
)
from fixed_gate_sampling import (
    FixedGateSamplingConfig,
    build_fixed_gate_w0_w1_candidate_rows,
    build_reachable_downstream_states,
    sample_fixed_gate_states,
    select_focused_replay_cases,
    validate_reachable_source_rollouts,
    validate_w1_independent_of_w0,
)
from primitive_envelope_clustering import build_primitive_envelope_clusters
from repeated_launch_episode import RepeatedLaunchEpisodeConfig, run_repeated_launch_episode
from run_real_flight_episode_ingest import find_episode_start_row


def test_fixed_gate_admission_shells_and_degree_fields() -> None:
    state = _gate_state()

    assert inside_fixed_launch_gate(state)
    assert launch_gate_admission_status(state) == "admitted_main_gate"

    shell_state = state.copy()
    shell_state[0] = 1.5
    assert not inside_fixed_launch_gate(shell_state)
    assert inside_launch_tolerance_shell(shell_state, "launch_gate_tolerance_shell")
    assert launch_gate_admission_status(shell_state) == "admitted_tolerance_shell"

    bad_yaw = state.copy()
    bad_yaw[5] = np.deg2rad(80.0)
    assert launch_gate_admission_status(bad_yaw) == "rejected_yaw"

    bad_attitude = state.copy()
    bad_attitude[3] = np.deg2rad(50.0)
    assert launch_gate_admission_status(bad_attitude) == "rejected_roll_pitch"
    assert launch_gate_admission_status(np.full(14, np.nan)) == "invalid_state"

    record = state_to_launch_gate_record(state)
    degree_fields = [name for name in record if name.endswith("_deg")]
    assert {"phi_deg", "theta_deg", "psi_deg"}.issubset(degree_fields)
    assert not any("deg" in name and not name.endswith("_deg") for name in record)


def test_sampling_records_entry_sources_and_reachable_provenance() -> None:
    accepted = _accepted_rollout_rows()
    config = FixedGateSamplingConfig(total_count=10, random_seed=42)

    samples = sample_fixed_gate_states(
        config,
        fan_branch="single_fan_branch",
        W_layer="W1",
        reachable_source_rows=accepted,
    )

    assert len(samples) == 10
    assert set(samples["entry_source"]).issubset(
        {
            "launch_gate_main",
            "reachable_downstream",
            "local_robustness_shell",
            "diagnostic_broad_only",
        }
    )
    assert int(samples["entry_source"].eq("diagnostic_broad_only").sum()) <= 1
    reachable = samples[samples["entry_source"].eq("reachable_downstream")]
    assert set(reachable["reachable_provenance_id"]) == {"rollout_a", "rollout_b"}

    bad = accepted.copy()
    bad.loc[0, "entry_source"] = "diagnostic_broad_only"
    with pytest.raises(ValueError, match="fixed gate/shell"):
        validate_reachable_source_rollouts(bad)


def test_w0_w1_candidate_rows_keep_w1_independent_of_w0() -> None:
    samples = sample_fixed_gate_states(
        FixedGateSamplingConfig(total_count=4, random_seed=7),
        fan_branch="four_fan_branch",
        W_layer="W1",
    )
    candidates = build_fixed_gate_w0_w1_candidate_rows(samples, primitive_families=("glide", "recovery"))

    validate_w1_independent_of_w0(candidates)
    assert set(candidates["W_layer"]) == {"W0", "W1"}
    assert candidates["w1_scheduled_independent_of_w0_success"].all()

    missing_w1 = candidates[~((candidates["W_layer"] == "W1") & (candidates["primitive_family"] == "glide"))]
    with pytest.raises(ValueError, match="W1 must not be filtered"):
        validate_w1_independent_of_w0(missing_w1)


def test_w2_w3_focused_replay_consumes_selected_w1_or_medoids() -> None:
    source = pd.DataFrame(
        [
            _source_case("a", "W1", "launch_gate_main", True, "thesis"),
            _source_case("b", "W1", "diagnostic_broad_only", True, "diagnostic"),
            _source_case("c", "W2", "reachable_downstream", True, "hardware"),
            _source_case("d", "W0", "launch_gate_main", True, "thesis"),
        ]
    )

    w2 = select_focused_replay_cases(source, target_W_layer="W2", max_cases=10)
    assert set(w2["W_layer"]) == {"W2"}
    assert set(w2["source_W_layer"]) == {"W1"}
    assert "diagnostic_broad_only" not in set(w2["entry_source"])

    w3 = select_focused_replay_cases(source, target_W_layer="W3", max_cases=10)
    assert set(w3["source_W_layer"]).issubset({"W1", "W2"})
    assert "W0" not in set(w3["source_W_layer"])


def test_primitive_row_clustering_selects_actual_source_medoids() -> None:
    rows = pd.DataFrame(
        [
            _rollout_row("s1", "single_fan_branch", "W1", "glide", "accepted", 0.1),
            _rollout_row("s2", "single_fan_branch", "W1", "glide", "accepted", 0.2),
            _rollout_row("s3", "single_fan_branch", "W1", "glide", "failed", -0.5),
            _rollout_row("s4", "four_fan_branch", "W2", "recovery", "accepted", 0.3),
        ]
    )

    outputs = build_primitive_envelope_clusters(rows)
    medoids = outputs["cluster_medoids"]

    assert set(medoids["medoid_sample_id"]).issubset(set(rows["sample_id"]))
    assert {"fan_branch", "W_layer", "primitive_family", "latency_case", "entry_source", "outcome_class"}.issubset(
        outputs["cluster_summary"].columns
    )
    assert outputs["governor_candidate_package"]["governor_package_status"].eq(
        "candidate_summary_only_governor_still_required"
    ).all()


def test_repeated_launch_episode_and_logger_write_schema_bundle(tmp_path: Path) -> None:
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "primitive_id": "p1",
                "primitive_family": "glide",
                "entry_source": "launch_gate_main",
                "recommended_use": "thesis",
                "energy_residual_m": 0.01,
                "minimum_margin_m": 0.5,
                "dwell_time_s": 0.5,
                "duration_s": 0.5,
                "lift_confidence": 0.9,
            }
        ]
    )
    result = run_repeated_launch_episode(
        _gate_state(),
        candidates,
        RepeatedLaunchEpisodeConfig(episode_id="episode_test"),
    )
    outputs = write_episode_log(
        tmp_path / "episode",
        episode_summary=result["episode_summary"],
        primitive_steps=result["primitive_steps"],
        governor_queries=result["governor_queries"],
        belief_updates=result["belief_updates"],
    )

    assert outputs.episode_summary_csv.exists()
    assert outputs.episode_manifest_json.exists()
    with pytest.raises(RuntimeError, match="non-empty"):
        write_episode_log(
            tmp_path / "episode",
            episode_summary=result["episode_summary"],
            primitive_steps=result["primitive_steps"],
            governor_queries=result["governor_queries"],
            belief_updates=result["belief_updates"],
        )


def test_real_flight_ingest_start_trigger_handles_dropouts() -> None:
    rows = []
    for index in range(7):
        state = _gate_state()
        rows.append(
            {
                "time_s": 0.01 * index,
                "vicon_valid": index != 1,
                "controller_ready": True,
                "x_w_m": state[0],
                "y_w_m": state[1],
                "z_w_m": state[2],
                "phi_rad": state[3],
                "theta_rad": state[4],
                "psi_rad": state[5],
                "u_m_s": state[6],
                "v_m_s": state[7],
                "w_m_s": state[8],
            }
        )
    frame = pd.DataFrame(rows)
    assert find_episode_start_row(frame) == 2

    frame["vicon_valid"] = False
    assert find_episode_start_row(frame) is None


def test_code_path_map_and_claim_scan_are_explicit() -> None:
    assert "fixed launch gate" in active_code_path_text()
    assert any("all-arena" in item for item in deprecated_default_paths())

    assert unsupported_claim_errors("This proves mission success.") == ["mission success"]
    assert_claim_safe_text(
        "This is a claim boundary: no mission success or real-flight transfer is claimed."
    )


def _gate_state() -> np.ndarray:
    state = np.zeros(15, dtype=float)
    state[0] = 1.3
    state[1] = 2.0
    state[2] = 1.7
    state[6] = 5.5
    return state


def _accepted_rollout_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _accepted_rollout("rollout_a", 2.0),
            _accepted_rollout("rollout_b", 2.2),
        ]
    )


def _accepted_rollout(rollout_id: str, x_terminal: float) -> dict[str, object]:
    return {
        "trial_descriptor_id": rollout_id,
        "entry_source": "launch_gate_main",
        "accepted": True,
        "x_terminal_w_m": x_terminal,
        "y_terminal_w_m": 2.1,
        "z_terminal_w_m": 1.65,
        "phi_terminal_rad": 0.0,
        "theta_terminal_rad": 0.0,
        "psi_terminal_rad": 0.0,
        "u_terminal_m_s": 5.2,
        "v_terminal_m_s": 0.0,
        "w_terminal_m_s": 0.0,
        "p_terminal_rad_s": 0.0,
        "q_terminal_rad_s": 0.0,
        "r_terminal_rad_s": 0.0,
    }


def _source_case(
    sample_id: str,
    W_layer: str,
    entry_source: str,
    is_medoid: bool,
    recommended_use: str,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "fan_branch": "single_fan_branch",
        "W_layer": W_layer,
        "primitive_family": "glide",
        "entry_source": entry_source,
        "is_medoid": is_medoid,
        "recommended_use": recommended_use,
    }


def _rollout_row(
    sample_id: str,
    fan_branch: str,
    W_layer: str,
    primitive_family: str,
    outcome_class: str,
    energy: float,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "paired_sample_key": f"pair_{sample_id}",
        "primitive_id": f"primitive_{sample_id}",
        "fan_branch": fan_branch,
        "W_layer": W_layer,
        "primitive_family": primitive_family,
        "latency_case": "nominal_latency",
        "entry_source": "launch_gate_main",
        "outcome_class": outcome_class,
        "controller_mode": "feedback_stabilised_primitive",
        "feedback_mode": "instant_state_feedback",
        "claim_status": "simulation_only",
        "evidence_role": "partial_feedback",
        "accepted": outcome_class == "accepted",
        "x0_w_m": 1.3,
        "y0_w_m": 2.0,
        "z0_w_m": 1.7,
        "phi0_rad": 0.0,
        "theta0_rad": 0.0,
        "psi0_rad": 0.0,
        "speed0_m_s": 5.5,
        "minimum_margin_m": 0.4,
        "w_wing_mean_m_s": 0.1,
        "delta_w_lr_m_s": 0.0,
        "spanwise_gradient_m_s_m": 0.0,
        "dwell_time_s": 0.5,
        "energy_residual_m": energy,
        "exit_speed_m_s": 5.4,
        "control_saturation": 0.0,
        "failure_label": "none" if outcome_class == "accepted" else "low_margin",
    }
