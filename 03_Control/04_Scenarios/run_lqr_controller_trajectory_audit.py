from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _bootstrap_import_paths() -> None:
    control_root = Path(__file__).resolve().parents[1]
    for rel in ("02_Inner_Loop", "03_Primitives", "04_Scenarios"):
        path = control_root / rel
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


_bootstrap_import_paths()

from arena_contract import TRUE_SAFE_BOUNDS, position_margin_m  # noqa: E402
from command_contract import normalised_command_to_surface_rad, surface_rad_to_normalised_command  # noqa: E402
from dense_archive_table_io import filesystem_path  # noqa: E402
from env_ctx import build_environment_context  # noqa: E402
from env_instance import environment_instance_for_mode, environment_metadata_from_instance  # noqa: E402
from env_surrogate import resolve_surrogate_binding, wind_field_for_binding  # noqa: E402
from implementation_instance import (  # noqa: E402
    adjusted_actuator_tau_s,
    apply_aileron_asymmetry_to_aircraft,
    apply_surface_implementation,
    implementation_instance_for_layer,
)
from latency import (  # noqa: E402
    actuator_tau_for_case,
    delayed_state_sample,
    latency_adjusted_command_sample,
    latency_mechanism_flags_from_case,
)
from lqr_controller import initialised_timing_state_for_controller, synthesize_lqr_controller  # noqa: E402
from lqr_linearisation import (  # noqa: E402
    local_speed_from_state_vector,
    lqr_speed_bin_id,
    nearest_lqr_operating_speed_m_s,
)
from lqr_tuning import candidate_weight_specs  # noqa: E402
from plant_instance import apply_plant_instance_to_aircraft, plant_instance_for_layer  # noqa: E402
from prim_cat import ACTIVE_PRIMITIVE_IDS, primitive_by_id  # noqa: E402
from prim_ctrl import PrimitiveControlContext, primitive_lqr_command  # noqa: E402
from prim_roll import (  # noqa: E402
    RolloutConfig,
    _aircraft_model,
    _latency_for_implementation,
    _rk4_step,
    _timing_state_from_command_history,
    _trajectory_w_wing_mean_m_s,
    _wind_mode_for_rollout,
    rollout_evidence_row,
    simulate_primitive_rollout,
)
from primitive_timing_contract import CONTROLLER_INPUT_UPDATE_PERIOD_S, PRIMITIVE_FINITE_HORIZON_S  # noqa: E402
from run_lqr_w01_dense_chunked import _r5_implementation_plant_layer, _r5_randomisation_config  # noqa: E402
from state_contract import STATE_INDEX, STATE_NAMES  # noqa: E402
from state_sampling import archive_state_sample_for_family  # noqa: E402


PROJECT_TITLE_VERSION = "LQR-Stabilised Contextual Primitive v5.20"
AUDIT_VERSION = "lqr_controller_trajectory_audit_v2_continuous_command_history"
DEFAULT_OUTPUT_ROOT = Path("03_Control/05_Results/lqr_contextual_v1_0/lqr_controller_trajectory_audit")
DEFAULT_DURATION_S = 0.80
DEFAULT_CANDIDATE_INDICES = (0, 1, 3, 5, 6, 7)


@dataclass(frozen=True)
class AuditCase:
    case_id: str
    primitive_id: str
    start_state_family: str
    W_layer: str
    environment_mode: str
    seed_offset: int


@dataclass(frozen=True)
class TrajectoryAuditConfig:
    output_root: Path = DEFAULT_OUTPUT_ROOT
    run_id: int = 1
    seed: int = 520
    duration_s: float = DEFAULT_DURATION_S
    candidate_indices: tuple[int, ...] = DEFAULT_CANDIDATE_INDICES
    same_start_comparison: bool = False
    same_start_family: str = "inflight_nominal"
    same_W_layer: str = "W0"
    same_environment_mode: str = "dry_air"
    same_seed_offset: int = 0
    same_start_key: str = "same_start_controller_behavior_audit"
    center_start_override: bool = False
    center_start_x_w_m: float = 2.0
    center_start_y_w_m: float = 2.2
    center_start_z_w_m: float = 2.2
    center_start_u_m_s: float | None = None
    primitive_ids: tuple[str, ...] | None = None


def run_lqr_controller_trajectory_audit(config: TrajectoryAuditConfig) -> dict[str, object]:
    run_root = filesystem_path(Path(config.output_root) / f"{int(config.run_id):03d}")
    (run_root / "metrics").mkdir(parents=True, exist_ok=True)
    (run_root / "plots").mkdir(parents=True, exist_ok=True)
    (run_root / "reports").mkdir(parents=True, exist_ok=True)
    (run_root / "manifests").mkdir(parents=True, exist_ok=True)

    cases = _cases_for_config(config)
    candidate_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    official_rows: list[dict[str, object]] = []

    for case in cases:
        traces = []
        for candidate_index in config.candidate_indices:
            trace = _run_case_trace(
                case=case,
                candidate_index=int(candidate_index),
                config=config,
            )
            traces.append(trace)
            candidate_rows.append(trace["summary"])
        selected = max(traces, key=lambda item: float(item["summary"]["selection_score"]))
        selected_rows.append({**selected["summary"], "selected_for_plot": True})
        trace_rows.extend(selected["trajectory_rows"])
        official_rows.append(_official_0p1_row(case=case, selected=selected, config=config))

    candidate_frame = pd.DataFrame(candidate_rows)
    selected_frame = pd.DataFrame(selected_rows)
    trace_frame = pd.DataFrame(trace_rows)
    official_frame = pd.DataFrame(official_rows)

    candidate_frame.to_csv(run_root / "metrics" / "candidate_trace_summary.csv", index=False)
    selected_frame.to_csv(run_root / "metrics" / "selected_trace_summary.csv", index=False)
    trace_frame.to_csv(run_root / "metrics" / "selected_trajectory_trace.csv", index=False)
    official_frame.to_csv(run_root / "metrics" / "official_0p1_rollout_summary.csv", index=False)

    _plot_3d(trace_frame, run_root / "plots" / "selected_trajectories_3d.png", duration_s=float(config.duration_s))
    _plot_altitude(trace_frame, run_root / "plots" / "selected_altitude_time.png")
    _plot_attitude(trace_frame, run_root / "plots" / "selected_bank_pitch_time.png")
    _plot_command_roll_yaw(trace_frame, run_root / "plots" / "selected_command_roll_yaw_time.png")

    manifest = {
        "audit_version": AUDIT_VERSION,
        "project_title_version": PROJECT_TITLE_VERSION,
        "status": "complete",
        "run_root": run_root.as_posix(),
        "duration_s": float(config.duration_s),
        "controller_update_period_s": float(CONTROLLER_INPUT_UPDATE_PERIOD_S),
        "primitive_reschedule_period_s": float(PRIMITIVE_FINITE_HORIZON_S),
        "active_primitive_ids": list(ACTIVE_PRIMITIVE_IDS),
        "candidate_indices_tested": list(config.candidate_indices),
        "audit_case_mode": "same_start_comparison" if config.same_start_comparison else "regime_representative_cases",
        "same_start_family": str(config.same_start_family) if config.same_start_comparison else "",
        "same_W_layer": str(config.same_W_layer) if config.same_start_comparison else "",
        "same_environment_mode": str(config.same_environment_mode) if config.same_start_comparison else "",
        "same_seed_offset": int(config.same_seed_offset) if config.same_start_comparison else -1,
        "same_start_key": str(config.same_start_key) if config.same_start_comparison else "",
        "primitive_ids_tested": list(config.primitive_ids or ACTIVE_PRIMITIVE_IDS),
        "center_start_override": bool(config.center_start_override),
        "center_start_x_w_m": float(config.center_start_x_w_m) if config.center_start_override else "",
        "center_start_y_w_m": float(config.center_start_y_w_m) if config.center_start_override else "",
        "center_start_z_w_m": float(config.center_start_z_w_m) if config.center_start_override else "",
        "center_start_u_m_s": float(config.center_start_u_m_s) if config.center_start_override and config.center_start_u_m_s is not None else "",
        "case_count": len(cases),
        "selected_full_duration_count": int(selected_frame["completed_full_duration"].astype(bool).sum()),
        "selected_floor_failure_count": int(selected_frame["floor_violation"].astype(bool).sum()),
        "selected_wall_failure_count": int(selected_frame["wall_violation"].astype(bool).sum()),
        "selected_max_altitude_loss_m": float((-selected_frame["altitude_delta_m"]).max()),
        "plot_files": [
            "plots/selected_trajectories_3d.png",
            "plots/selected_altitude_time.png",
            "plots/selected_bank_pitch_time.png",
            "plots/selected_command_roll_yaw_time.png",
        ],
        "claim_status": "controller_sanity_audit_only_not_dense_evidence",
    }
    (run_root / "manifests" / "trajectory_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    _write_report(run_root, manifest, selected_frame)
    return manifest


def _cases_for_config(config: TrajectoryAuditConfig) -> tuple[AuditCase, ...]:
    if config.same_start_comparison:
        return _same_start_audit_cases(config)
    return _audit_cases()


def _audit_cases() -> tuple[AuditCase, ...]:
    return (
        AuditCase("glide_launch_dry", "glide", "launch_gate", "W0", "dry_air", 0),
        AuditCase("lift_entry_single", "lift_entry", "inflight_lift_region", "W1", "w1_annular_gp_randomised_single", 1),
        AuditCase("lift_dwell_four", "lift_dwell_arc", "inflight_lift_region", "W1", "w1_annular_gp_randomised_four", 2),
        AuditCase("mild_left_nominal", "mild_turn_left", "inflight_nominal", "W0", "dry_air", 3),
        AuditCase("mild_right_nominal", "mild_turn_right", "inflight_nominal", "W0", "dry_air", 4),
        AuditCase("energy_bank_four", "energy_retaining_bank", "inflight_nominal", "W1", "w1_annular_gp_randomised_four", 5),
        AuditCase("recovery_edge_dry", "recovery", "inflight_recovery_edge", "W0", "dry_air", 6),
        AuditCase("safe_exit_boundary_single", "safe_exit_or_recovery_handoff", "inflight_boundary_near", "W1", "w1_annular_gp_randomised_single", 7),
    )


def _same_start_audit_cases(config: TrajectoryAuditConfig) -> tuple[AuditCase, ...]:
    primitive_ids = tuple(config.primitive_ids or ACTIVE_PRIMITIVE_IDS)
    return tuple(
        AuditCase(
            f"same_start_{primitive_id}",
            primitive_id,
            str(config.same_start_family),
            str(config.same_W_layer),
            str(config.same_environment_mode),
            int(config.same_seed_offset),
        )
        for primitive_id in primitive_ids
    )


def _run_case_trace(
    *,
    case: AuditCase,
    candidate_index: int,
    config: TrajectoryAuditConfig,
) -> dict[str, object]:
    primitive = primitive_by_id(case.primitive_id)
    weight_spec = candidate_weight_specs(
        primitive_id=case.primitive_id,
        candidate_count=max(max(config.candidate_indices) + 1, 8),
        tuning_stage="W01_audit",
    )[int(candidate_index)]
    paired_start_key = str(config.same_start_key) if config.same_start_comparison else f"audit_{case.case_id}"
    sample = archive_state_sample_for_family(
        start_state_family=case.start_state_family,
        paired_start_key=paired_start_key,
        sample_index=int(case.seed_offset),
        seed=int(config.seed),
        W_layer=case.W_layer,
        environment_mode=case.environment_mode,
    )
    env_seed = int(config.seed) + int(case.seed_offset)
    environment = environment_instance_for_mode(
        case.W_layer,
        case.environment_mode,
        env_seed,
        randomisation_config=_r5_randomisation_config(
            environment_mode=case.environment_mode,
            paired_start_index=int(case.seed_offset),
        ),
    )
    metadata = environment_metadata_from_instance(environment)
    binding = resolve_surrogate_binding(case.W_layer, metadata, randomisation_seed=env_seed)
    wind_field = wind_field_for_binding(binding)
    implementation_plant_layer = _r5_implementation_plant_layer(
        W_layer=case.W_layer,
        environment_mode=case.environment_mode,
    )
    implementation = implementation_instance_for_layer(
        implementation_plant_layer,
        env_seed,
        latency_case="nominal",
    )
    plant = plant_instance_for_layer(implementation_plant_layer, env_seed)

    x = _audit_initial_state_vector(sample.state_vector, config=config)
    base_aircraft = _aircraft_model()
    aircraft = apply_plant_instance_to_aircraft(base_aircraft, plant)
    aircraft = apply_aileron_asymmetry_to_aircraft(aircraft, implementation)
    latency = _latency_for_implementation("nominal", implementation)
    mechanism_flags = latency_mechanism_flags_from_case(
        "nominal",
        state_feedback_delay_applied=True,
    )
    base_tau_s = actuator_tau_for_case(latency) if mechanism_flags["actuator_lag_applied"] else (1.0, 1.0, 1.0)
    tau_s = adjusted_actuator_tau_s(base_tau_s, implementation)
    config_rollout = RolloutConfig(
        W_layer=case.W_layer,
        dt_s=float(CONTROLLER_INPUT_UPDATE_PERIOD_S),
        rollout_backend="model_backed_lqr",
        wind_mode="panel",
    )
    wind_mode = _wind_mode_for_rollout(
        context=build_environment_context(
            x,
            wind_field=wind_field,
            metadata=metadata,
            latency_case="nominal",
            actuator_case="nominal",
            surrogate_binding=binding,
        ),
        config=config_rollout,
        wind_field=wind_field,
    )

    trace_rows: list[dict[str, object]] = []
    times_s = [0.0]
    states = [x.copy()]
    command_times_s: list[float] = []
    command_norm_history: list[np.ndarray] = []
    controller = None
    context = None
    saturation_count = 0
    max_abs_command_norm = 0.0
    max_abs_surface_rad = 0.0
    min_floor_margin_m = float("inf")
    min_wall_margin_m = float("inf")
    min_ceiling_margin_m = float("inf")
    min_speed_m_s = float(np.linalg.norm(x[6:9]))
    trajectory_integrated_updraft_gain_m = 0.0
    termination_cause = "completed_full_duration"

    total_steps = int(round(float(config.duration_s) / float(CONTROLLER_INPUT_UPDATE_PERIOD_S)))
    primitive_steps = int(round(float(PRIMITIVE_FINITE_HORIZON_S) / float(CONTROLLER_INPUT_UPDATE_PERIOD_S)))
    segment_index = -1
    speed_bin = nearest_lqr_operating_speed_m_s(local_speed_from_state_vector(x))
    for step_index in range(total_steps):
        if step_index % primitive_steps == 0:
            segment_index += 1
            context = build_environment_context(
                x,
                wind_field=wind_field,
                metadata=metadata,
                latency_case="nominal",
                actuator_case="nominal",
                surrogate_binding=binding,
            )
            speed_bin = nearest_lqr_operating_speed_m_s(local_speed_from_state_vector(x))
            controller = synthesize_lqr_controller(
                primitive,
                weight_spec=weight_spec,
                local_reference_speed_m_s=float(speed_bin),
            )
            if not command_times_s:
                initial_command = primitive_lqr_command(
                    primitive,
                    PrimitiveControlContext(
                        state_vector=x,
                        environment_context=context,
                        time_in_primitive_s=0.0,
                        timing_state=initialised_timing_state_for_controller(controller, x),
                    ),
                    controller,
                )
                reference_command_norm = surface_rad_to_normalised_command(
                    np.asarray(controller.reference_command_vector, dtype=float)
                )
                command_delay_s = float(latency.command_onset_delay_s + latency.command_transport_delay_s)
                if mechanism_flags["command_delay_applied"]:
                    command_times_s = [float(times_s[-1]) - (command_delay_s + 1e-9)]
                    command_norm_history = [reference_command_norm.copy()]
                else:
                    command_times_s = [float(times_s[-1])]
                    command_norm_history = [np.asarray(initial_command.command_norm, dtype=float)]

        assert context is not None
        assert controller is not None
        time_s = float(step_index) * float(CONTROLLER_INPUT_UPDATE_PERIOD_S)
        local_time_s = float(step_index % primitive_steps) * float(CONTROLLER_INPUT_UPDATE_PERIOD_S)
        if mechanism_flags["state_feedback_delay_applied"]:
            x_control = delayed_state_sample(
                np.asarray(times_s, dtype=float),
                np.asarray(states, dtype=float),
                time_s - float(latency.state_feedback_delay_s),
            )
        else:
            x_control = x
        control_command = primitive_lqr_command(
            primitive,
            PrimitiveControlContext(
                state_vector=x_control,
                environment_context=context,
                time_in_primitive_s=local_time_s,
                timing_state=_timing_state_from_command_history(
                    controller=controller,
                    state_vector=x_control,
                    command_times_s=command_times_s,
                    command_norm_history=command_norm_history,
                    time_s=time_s,
                    latency=latency,
                    dt_s=float(CONTROLLER_INPUT_UPDATE_PERIOD_S),
                ),
            ),
            controller,
        )
        desired_command_norm = np.asarray(control_command.command_norm, dtype=float)
        if time_s > command_times_s[-1]:
            command_times_s.append(time_s)
            command_norm_history.append(desired_command_norm.copy())
        else:
            command_norm_history[-1] = desired_command_norm.copy()
        applied_norm = (
            latency_adjusted_command_sample(
                np.asarray(command_times_s, dtype=float),
                np.asarray(command_norm_history, dtype=float),
                time_s,
                latency,
            )
            if mechanism_flags["command_delay_applied"]
            else desired_command_norm
        )
        command_rad = apply_surface_implementation(
            normalised_command_to_surface_rad(applied_norm),
            implementation,
        )
        raw_command_rad = np.asarray(control_command.raw_command_rad, dtype=float)
        saturation_count += int(control_command.saturation_applied)
        max_abs_command_norm = max(max_abs_command_norm, float(np.max(np.abs(applied_norm))))
        max_abs_surface_rad = max(max_abs_surface_rad, float(np.max(np.abs(command_rad))))
        if not mechanism_flags["actuator_lag_applied"]:
            x[12:15] = command_rad
        x = _rk4_step(
            x=x,
            command=command_rad,
            aircraft=aircraft,
            wind_field=wind_field,
            wind_mode=wind_mode,
            actuator_tau_s=tau_s,
            dt_s=float(CONTROLLER_INPUT_UPDATE_PERIOD_S),
        )
        if not np.all(np.isfinite(x)):
            termination_cause = "nonfinite_trajectory"
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            break
        margins = position_margin_m(x[:3], TRUE_SAFE_BOUNDS)
        min_wall_margin_m = min(min_wall_margin_m, float(margins["min_wall_margin_m"]))
        min_floor_margin_m = min(min_floor_margin_m, float(margins["floor_margin_m"]))
        min_ceiling_margin_m = min(min_ceiling_margin_m, float(margins["ceiling_margin_m"]))
        min_speed_m_s = min(min_speed_m_s, float(np.linalg.norm(x[6:9])))
        w_wing_step_m_s, _ = _trajectory_w_wing_mean_m_s(state=x, wind_field=wind_field)
        trajectory_integrated_updraft_gain_m += max(float(w_wing_step_m_s), 0.0) * float(CONTROLLER_INPUT_UPDATE_PERIOD_S)
        times_s.append(time_s + float(CONTROLLER_INPUT_UPDATE_PERIOD_S))
        states.append(x.copy())
        trace_rows.append(
            _trace_row(
                case=case,
                candidate_index=candidate_index,
                candidate_label=weight_spec.weight_label,
                segment_index=segment_index,
                speed_bin=speed_bin,
                time_s=time_s + float(CONTROLLER_INPUT_UPDATE_PERIOD_S),
                state=x,
                command_norm=applied_norm,
                command_rad=command_rad,
                raw_command_rad=raw_command_rad,
                saturation_applied=bool(control_command.saturation_applied),
            )
        )
        if min_floor_margin_m < 0.0:
            termination_cause = "floor_margin_stop"
            break
        if min_ceiling_margin_m < 0.0:
            termination_cause = "ceiling_margin_stop"
            break
        if min_wall_margin_m < 0.0:
            termination_cause = "wall_boundary_exit_retained"
            break

    initial_state = np.asarray(states[0], dtype=float)
    final_state = x.copy()
    duration_s = float(times_s[-1]) if times_s else 0.0
    completed_full_duration = duration_s >= float(config.duration_s) - 1e-12 and termination_cause == "completed_full_duration"
    altitude_delta_m = float(final_state[STATE_INDEX["z_w"]] - initial_state[STATE_INDEX["z_w"]])
    speed_initial = float(np.linalg.norm(initial_state[6:9]))
    speed_final = float(np.linalg.norm(final_state[6:9]))
    yaw_delta_deg = float(np.rad2deg(final_state[STATE_INDEX["psi"]] - initial_state[STATE_INDEX["psi"]]))
    roll_delta_deg = float(np.rad2deg(final_state[STATE_INDEX["phi"]] - initial_state[STATE_INDEX["phi"]]))
    pitch_delta_deg = float(np.rad2deg(final_state[STATE_INDEX["theta"]] - initial_state[STATE_INDEX["theta"]]))
    dx_w_m = float(final_state[STATE_INDEX["x_w"]] - initial_state[STATE_INDEX["x_w"]])
    dy_w_m = float(final_state[STATE_INDEX["y_w"]] - initial_state[STATE_INDEX["y_w"]])
    heading0 = float(initial_state[STATE_INDEX["psi"]])
    lateral_delta_m = float(-np.sin(heading0) * dx_w_m + np.cos(heading0) * dy_w_m)
    expected_lateral_sign = _expected_lateral_turn_sign(case.primitive_id)
    expected_roll_sign = _expected_roll_turn_sign(case.primitive_id)
    saturation_fraction = float(saturation_count / max(1, len(trace_rows)))
    summary = {
        "case_id": case.case_id,
        "primitive_id": case.primitive_id,
        "start_state_family": case.start_state_family,
        "W_layer": case.W_layer,
        "environment_mode": case.environment_mode,
        "candidate_index": int(candidate_index),
        "candidate_weight_label": weight_spec.weight_label,
        "duration_s": duration_s,
        "completed_full_duration": bool(completed_full_duration),
        "termination_cause": termination_cause,
        "floor_violation": bool(min_floor_margin_m < 0.0),
        "wall_violation": bool(min_wall_margin_m < 0.0),
        "altitude_delta_m": altitude_delta_m,
        "speed_initial_m_s": speed_initial,
        "speed_final_m_s": speed_final,
        "minimum_speed_m_s": float(min_speed_m_s),
        "minimum_floor_margin_m": float(min_floor_margin_m),
        "minimum_wall_margin_m": float(min_wall_margin_m),
        "minimum_ceiling_margin_m": float(min_ceiling_margin_m),
        "yaw_delta_deg": yaw_delta_deg,
        "roll_delta_deg": roll_delta_deg,
        "pitch_delta_deg": pitch_delta_deg,
        "initial_roll_rate_rad_s": float(initial_state[STATE_INDEX["p"]]),
        "final_roll_rate_rad_s": float(final_state[STATE_INDEX["p"]]),
        "roll_rate_delta_rad_s": float(final_state[STATE_INDEX["p"]] - initial_state[STATE_INDEX["p"]]),
        "lateral_delta_m": lateral_delta_m,
        "signed_lateral_delta_m": float(expected_lateral_sign * lateral_delta_m) if expected_lateral_sign else 0.0,
        "signed_roll_rate_delta_rad_s": float(expected_roll_sign * (final_state[STATE_INDEX["p"]] - initial_state[STATE_INDEX["p"]])) if expected_roll_sign else 0.0,
        "turn_expected_lateral_sign": float(expected_lateral_sign),
        "turn_expected_roll_sign": float(expected_roll_sign),
        "trajectory_integrated_updraft_gain_m": float(trajectory_integrated_updraft_gain_m),
        "saturation_fraction": saturation_fraction,
        "max_abs_command_norm": float(max_abs_command_norm),
        "max_abs_surface_rad": float(max_abs_surface_rad),
    }
    summary["selection_score"] = _selection_score(summary)
    return {"summary": summary, "trajectory_rows": trace_rows}


def _trace_row(
    *,
    case: AuditCase,
    candidate_index: int,
    candidate_label: str,
    segment_index: int,
    speed_bin: float,
    time_s: float,
    state: np.ndarray,
    command_norm: np.ndarray,
    command_rad: np.ndarray,
    raw_command_rad: np.ndarray,
    saturation_applied: bool,
) -> dict[str, object]:
    row = {
        "case_id": case.case_id,
        "primitive_id": case.primitive_id,
        "start_state_family": case.start_state_family,
        "environment_mode": case.environment_mode,
        "candidate_index": int(candidate_index),
        "candidate_weight_label": str(candidate_label),
        "segment_index": int(segment_index),
        "local_lqr_reference_speed_m_s": float(speed_bin),
        "local_lqr_speed_bin_id": lqr_speed_bin_id(speed_bin),
        "time_s": float(time_s),
        "speed_m_s": float(np.linalg.norm(state[6:9])),
        "command_aileron_norm": float(command_norm[0]),
        "command_elevator_norm": float(command_norm[1]),
        "command_rudder_norm": float(command_norm[2]),
        "applied_aileron_rad": float(command_rad[0]),
        "applied_elevator_rad": float(command_rad[1]),
        "applied_rudder_rad": float(command_rad[2]),
        "raw_aileron_rad": float(raw_command_rad[0]),
        "raw_elevator_rad": float(raw_command_rad[1]),
        "raw_rudder_rad": float(raw_command_rad[2]),
        "saturation_applied": bool(saturation_applied),
    }
    row.update({name: float(state[index]) for index, name in enumerate(STATE_NAMES)})
    return row


def _expected_lateral_turn_sign(primitive_id: str) -> float:
    if str(primitive_id) == "mild_turn_left":
        return -1.0
    if str(primitive_id) == "mild_turn_right":
        return 1.0
    return 0.0


def _expected_roll_turn_sign(primitive_id: str) -> float:
    if str(primitive_id) == "mild_turn_left":
        return -1.0
    if str(primitive_id) == "mild_turn_right":
        return 1.0
    return 0.0


def _audit_initial_state_vector(state_vector: object, *, config: TrajectoryAuditConfig) -> np.ndarray:
    x = np.asarray(state_vector, dtype=float).copy()
    if not config.center_start_override:
        return x
    x[STATE_INDEX["x_w"]] = float(config.center_start_x_w_m)
    x[STATE_INDEX["y_w"]] = float(config.center_start_y_w_m)
    x[STATE_INDEX["z_w"]] = float(config.center_start_z_w_m)
    if config.center_start_u_m_s is not None:
        x[STATE_INDEX["u"]] = float(config.center_start_u_m_s)
        x[STATE_INDEX["v"]] = 0.0
        x[STATE_INDEX["w"]] = 0.0
    return x


def _selection_score(summary: dict[str, object]) -> float:
    duration = float(summary["duration_s"])
    floor_margin = float(summary["minimum_floor_margin_m"])
    wall_margin = float(summary["minimum_wall_margin_m"])
    altitude_delta = float(summary["altitude_delta_m"])
    saturation = float(summary["saturation_fraction"])
    terminated_penalty = 0.0 if bool(summary["completed_full_duration"]) else 2.5
    floor_penalty = 5.0 if bool(summary["floor_violation"]) else 0.0
    wall_penalty = 2.0 if bool(summary["wall_violation"]) else 0.0
    return (
        2.0 * duration
        + 0.8 * min(floor_margin, 1.0)
        + 0.3 * min(wall_margin, 1.0)
        + 0.4 * altitude_delta
        - 0.8 * saturation
        - terminated_penalty
        - floor_penalty
        - wall_penalty
    )


def _official_0p1_row(
    *,
    case: AuditCase,
    selected: dict[str, object],
    config: TrajectoryAuditConfig,
) -> dict[str, object]:
    primitive = primitive_by_id(case.primitive_id)
    candidate_index = int(selected["summary"]["candidate_index"])
    weight_spec = candidate_weight_specs(
        primitive_id=case.primitive_id,
        candidate_count=max(max(config.candidate_indices) + 1, 8),
        tuning_stage="W01_audit",
    )[candidate_index]
    paired_start_key = str(config.same_start_key) if config.same_start_comparison else f"audit_{case.case_id}"
    sample = archive_state_sample_for_family(
        start_state_family=case.start_state_family,
        paired_start_key=paired_start_key,
        sample_index=int(case.seed_offset),
        seed=int(config.seed),
        W_layer=case.W_layer,
        environment_mode=case.environment_mode,
    )
    env_seed = int(config.seed) + int(case.seed_offset)
    environment = environment_instance_for_mode(
        case.W_layer,
        case.environment_mode,
        env_seed,
        randomisation_config=_r5_randomisation_config(
            environment_mode=case.environment_mode,
            paired_start_index=int(case.seed_offset),
        ),
    )
    metadata = environment_metadata_from_instance(environment)
    binding = resolve_surrogate_binding(case.W_layer, metadata, randomisation_seed=env_seed)
    wind_field = wind_field_for_binding(binding)
    initial_state = _audit_initial_state_vector(sample.state_vector, config=config)
    context = build_environment_context(
        initial_state,
        wind_field=wind_field,
        metadata=metadata,
        latency_case="nominal",
        actuator_case="nominal",
        surrogate_binding=binding,
    )
    implementation_plant_layer = _r5_implementation_plant_layer(
        W_layer=case.W_layer,
        environment_mode=case.environment_mode,
    )
    controller = synthesize_lqr_controller(
        primitive,
        weight_spec=weight_spec,
        local_reference_speed_m_s=nearest_lqr_operating_speed_m_s(
            local_speed_from_state_vector(initial_state)
        ),
    )
    evidence = simulate_primitive_rollout(
        rollout_id=f"audit_official_0p1_{case.case_id}",
        episode_id=f"audit_official_0p1_{case.case_id}",
        initial_state=initial_state,
        context=context,
        primitive=primitive,
        config=RolloutConfig(
            W_layer=case.W_layer,
            dt_s=float(CONTROLLER_INPUT_UPDATE_PERIOD_S),
            rollout_backend="model_backed_lqr",
            wind_mode="panel",
        ),
        wind_field=wind_field,
        implementation_instance=implementation_instance_for_layer(
            implementation_plant_layer,
            env_seed,
            latency_case="nominal",
        ),
        plant_instance=plant_instance_for_layer(implementation_plant_layer, env_seed),
        controller=controller,
        controller_selection_status="trajectory_audit_selected_candidate",
        candidate_index=candidate_index,
        candidate_weight_label=weight_spec.weight_label,
    )
    row = rollout_evidence_row(evidence)
    return {
        "case_id": case.case_id,
        "candidate_index": candidate_index,
        "candidate_weight_label": weight_spec.weight_label,
        "outcome_class": row.get("outcome_class", ""),
        "termination_cause": row.get("termination_cause", ""),
        "failure_label": row.get("failure_label", ""),
        "rollout_duration_s": row.get("rollout_duration_s", 0.0),
        "energy_residual_m": row.get("energy_residual_m", 0.0),
        "minimum_floor_margin_m": row.get("floor_margin_m", 0.0),
        "minimum_wall_margin_m": row.get("minimum_wall_margin_m", 0.0),
        "saturation_fraction": row.get("saturation_fraction", 0.0),
        "max_abs_command_norm": row.get("max_abs_command_norm", 0.0),
    }


def _plot_3d(frame: pd.DataFrame, path: Path, *, duration_s: float) -> None:
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    for case_id, group in frame.groupby("case_id", sort=False):
        label = f"{group['primitive_id'].iloc[0]} | {case_id}"
        ax.plot(group["x_w"], group["y_w"], group["z_w"], linewidth=1.8, label=label)
        ax.scatter(group["x_w"].iloc[0], group["y_w"].iloc[0], group["z_w"].iloc[0], s=18)
    ax.set_xlabel("x_w [m]")
    ax.set_ylabel("y_w [m]")
    ax.set_zlabel("z_w [m]")
    ax.set_title(f"Selected {float(duration_s):.1f} s LQR trajectory audit")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_altitude(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for case_id, group in frame.groupby("case_id", sort=False):
        label = f"{group['primitive_id'].iloc[0]} | {case_id}"
        ax.plot(group["time_s"], group["z_w"], linewidth=1.8, label=label)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("z_w [m]")
    ax.set_title("Altitude during selected LQR trajectory audit")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_attitude(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for case_id, group in frame.groupby("case_id", sort=False):
        label = f"{group['primitive_id'].iloc[0]} | {case_id}"
        axes[0].plot(group["time_s"], np.rad2deg(group["phi"]), linewidth=1.5, label=label)
        axes[1].plot(group["time_s"], np.rad2deg(group["theta"]), linewidth=1.5, label=label)
    axes[0].set_ylabel("bank phi [deg]")
    axes[1].set_ylabel("pitch theta [deg]")
    axes[1].set_xlabel("time [s]")
    axes[0].set_title("Bank and pitch during selected LQR trajectory audit")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_command_roll_yaw(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    for case_id, group in frame.groupby("case_id", sort=False):
        label = f"{group['primitive_id'].iloc[0]} | {case_id}"
        axes[0].plot(group["time_s"], group["command_aileron_norm"], linewidth=1.4, label=label)
        axes[1].plot(group["time_s"], np.rad2deg(group["p"]), linewidth=1.4, label=label)
        axes[2].plot(group["time_s"], np.rad2deg(group["phi"]), linewidth=1.4, label=label)
        axes[3].plot(group["time_s"], np.rad2deg(group["psi"] - group["psi"].iloc[0]), linewidth=1.4, label=label)
    axes[0].set_ylabel("aileron cmd [-]")
    axes[1].set_ylabel("roll rate p [deg/s]")
    axes[2].set_ylabel("bank phi [deg]")
    axes[3].set_ylabel("heading delta [deg]")
    axes[3].set_xlabel("time [s]")
    axes[0].set_title("Command, roll-rate, bank, and heading audit")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_report(run_root: Path, manifest: dict[str, object], selected_frame: pd.DataFrame) -> None:
    lines = [
        "# LQR Controller Trajectory Audit",
        "",
        f"- Project title version: `{PROJECT_TITLE_VERSION}`",
        f"- Audit version: `{AUDIT_VERSION}`",
        f"- Status: `{manifest['status']}`",
        f"- Audit case mode: `{manifest.get('audit_case_mode', 'regime_representative_cases')}`",
        f"- Duration per selected trace: `{manifest['duration_s']}` s",
        f"- Controller update period: `{manifest['controller_update_period_s']}` s",
        f"- Primitive reschedule period: `{manifest['primitive_reschedule_period_s']}` s",
        f"- Full-duration selected traces: `{manifest['selected_full_duration_count']}` / `{manifest['case_count']}`",
        f"- Floor failures: `{manifest['selected_floor_failure_count']}`",
        f"- Wall failures: `{manifest['selected_wall_failure_count']}`",
        f"- Maximum selected altitude loss: `{manifest['selected_max_altitude_loss_m']:.3f}` m",
        "",
        "Selected cases:",
    ]
    if str(manifest.get("audit_case_mode", "")) == "same_start_comparison":
        lines.insert(
            9,
            (
                f"- Shared start: `{manifest.get('same_start_family', '')}` "
                f"`{manifest.get('same_W_layer', '')}` `{manifest.get('same_environment_mode', '')}` "
                f"seed offset `{manifest.get('same_seed_offset', '')}`."
            ),
        )
    if bool(manifest.get("center_start_override", False)):
        lines.insert(
            10,
            (
                f"- Manual centred-state override: x/y/z = "
                f"`{manifest.get('center_start_x_w_m', '')}`, "
                f"`{manifest.get('center_start_y_w_m', '')}`, "
                f"`{manifest.get('center_start_z_w_m', '')}` m; "
                f"u = `{manifest.get('center_start_u_m_s', '')}` m/s."
            ),
        )
    for row in selected_frame.to_dict(orient="records"):
        lines.append(
            "- "
            f"`{row['case_id']}` `{row['primitive_id']}` candidate `{row['candidate_index']}` "
            f"duration `{float(row['duration_s']):.3f}` s, "
            f"dz `{float(row['altitude_delta_m']):+.3f}` m, "
            f"speed `{float(row['speed_initial_m_s']):.2f}->{float(row['speed_final_m_s']):.2f}` m/s, "
            f"yaw `{float(row['yaw_delta_deg']):+.2f}` deg, "
            f"sat `{float(row['saturation_fraction']):.2f}`, "
            f"termination `{row['termination_cause']}`."
        )
    lines.extend(
        [
            "",
            "Claim boundary: this is a lightweight controller sanity audit only. It is not R5 dense evidence, not R7 validation, and not a mission/autonomy/real-flight claim.",
        ]
    )
    (run_root / "reports" / "trajectory_audit_report.md").write_text("\n".join(lines) + "\n", encoding="ascii")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight 0.8 s LQR controller trajectory audit.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=520)
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--candidate-indices", type=str, default=",".join(str(item) for item in DEFAULT_CANDIDATE_INDICES))
    parser.add_argument("--same-start-comparison", action="store_true")
    parser.add_argument("--same-start-family", type=str, default="inflight_nominal")
    parser.add_argument("--same-W-layer", type=str, default="W0")
    parser.add_argument("--same-environment-mode", type=str, default="dry_air")
    parser.add_argument("--same-seed-offset", type=int, default=0)
    parser.add_argument("--same-start-key", type=str, default="same_start_controller_behavior_audit")
    parser.add_argument("--center-start-override", action="store_true")
    parser.add_argument("--center-start-x-w-m", type=float, default=2.0)
    parser.add_argument("--center-start-y-w-m", type=float, default=2.2)
    parser.add_argument("--center-start-z-w-m", type=float, default=2.2)
    parser.add_argument("--center-start-u-m-s", type=float, default=None)
    parser.add_argument("--primitive-ids", type=str, default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = TrajectoryAuditConfig(
        output_root=args.output_root,
        run_id=int(args.run_id),
        seed=int(args.seed),
        duration_s=float(args.duration_s),
        candidate_indices=tuple(int(item.strip()) for item in str(args.candidate_indices).split(",") if item.strip()),
        same_start_comparison=bool(args.same_start_comparison),
        same_start_family=str(args.same_start_family),
        same_W_layer=str(args.same_W_layer),
        same_environment_mode=str(args.same_environment_mode),
        same_seed_offset=int(args.same_seed_offset),
        same_start_key=str(args.same_start_key),
        center_start_override=bool(args.center_start_override),
        center_start_x_w_m=float(args.center_start_x_w_m),
        center_start_y_w_m=float(args.center_start_y_w_m),
        center_start_z_w_m=float(args.center_start_z_w_m),
        center_start_u_m_s=args.center_start_u_m_s,
        primitive_ids=tuple(item.strip() for item in str(args.primitive_ids).split(",") if item.strip()) or None,
    )
    manifest = run_lqr_controller_trajectory_audit(config)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest.get("status") == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
