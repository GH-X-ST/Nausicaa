"""Compare an R11 balanced-cluster control run against true neutral rollouts.

This is a diagnostic-only sim2real sanity check.  The official R11
``no_memory_baseline`` policy is still a closed-loop governor/LQR policy, so it
is not the same as no controller intervention.  This script reruns a zero
aggregate-command plant rollout from the same R11 final launch states and
summarises whether the controller is doing more than a neutral glider would do.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _bootstrap_import_paths() -> None:
    control_root = Path(__file__).resolve().parents[1]
    for rel in ("02_Inner_Loop", "03_Primitives", "04_Scenarios", "01_Plotting"):
        path = control_root / rel
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


_bootstrap_import_paths()

from arena_contract import TRUE_SAFE_BOUNDS, position_margin_m  # noqa: E402
from command_contract import normalised_command_to_surface_rad  # noqa: E402
from flight_dynamics import adapt_glider  # noqa: E402
from glider import build_nausicaa_glider  # noqa: E402
from implementation_instance import (  # noqa: E402
    adjusted_actuator_tau_s,
    apply_aileron_asymmetry_to_aircraft,
    apply_surface_implementation,
    implementation_instance_for_layer,
)
from latency import actuator_tau_for_case, latency_mechanism_flags_from_case  # noqa: E402
from plant_instance import apply_plant_instance_to_aircraft, plant_instance_for_layer  # noqa: E402
from prim_roll import _latency_for_implementation, _rk4_step  # noqa: E402
from run_changed_case_validation import R11_PROTOCOL  # noqa: E402
from run_r11_balanced_ladder_case_figures import (  # noqa: E402
    DEFAULT_LIBRARY_SIZE_CASE,
    DEFAULT_POLICY_IDS,
    DEFAULT_R11_ROOT,
    DEFAULT_UPDRAFT_NX,
    DEFAULT_UPDRAFT_NY,
    DEFAULT_UPDRAFT_NZ,
    FINAL_POLICY_COLORS,
    FINAL_POLICY_LABELS,
    FINAL_POLICY_LINESTYLES,
    FINAL_POLICY_MARKER_SIZES,
    FINAL_POLICY_ORDER,
    PRIMITIVE_MARKER_SIZE_FINAL,
    R11BalancedLadderFigureConfig,
    _add_legend,
    _draw_environment_fan_outlets,
    _draw_primitive_markers,
    _draw_updraft_context_from_case,
    _episode_points,
    _new_baseline_axis,
    _policy_safe_success,
    _read_selected_primitive_rows,
    _select_same_start_index,
    _selected_outer_cases,
    _short_ladder_id,
    _save_figure,
)
from run_repeated_launch_learning_curve import (  # noqa: E402
    R11_FIDELITY_LADDER_BLOCK_IDS,
    _environment_randomisation_config_for_context,
    _scheduled_active_fan_count_for_context,
    _specific_energy_m,
    _uses_full_w3_randomisation_block,
    _wall_exit_face,
)
from env_instance import environment_instance_for_mode, environment_metadata_from_instance  # noqa: E402
from env_surrogate import resolve_surrogate_binding, wind_field_for_binding  # noqa: E402
from state_contract import STATE_INDEX, as_state_vector  # noqa: E402


DIAGNOSTIC_VERSION = "r11_balanced_true_neutral_intervention_diagnostic_v1"
DEFAULT_OUTPUT_ROOT = Path("03_Control/A_figures/R11_E01_balanced_neutral_baseline")
NEUTRAL_POLICY_ID = "true_neutral_open_loop"
NEUTRAL_LABEL = "true neutral"
NEUTRAL_COLOR = "#111111"


@dataclass(frozen=True)
class NeutralDiagnosticConfig:
    r11_root: Path = DEFAULT_R11_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    library_size_case_id: str = DEFAULT_LIBRARY_SIZE_CASE
    paired_start_index: int | None = None
    policy_ids: tuple[str, ...] = DEFAULT_POLICY_IDS
    updraft_nx: int = DEFAULT_UPDRAFT_NX
    updraft_ny: int = DEFAULT_UPDRAFT_NY
    updraft_nz: int = DEFAULT_UPDRAFT_NZ
    dt_s: float = 0.02
    max_episode_time_s: float = 20.0
    plot_stride: int = 5
    summary_case_limit: int = 0


def run_r11_balanced_neutral_baseline_figures(config: NeutralDiagnosticConfig) -> dict[str, object]:
    r11_root = Path(config.r11_root)
    output_root = Path(config.output_root)
    run_label_slug = _run_label_slug_from_root(r11_root)
    for subdir in ("figures", "metrics", "manifests", "reports"):
        (output_root / subdir).mkdir(parents=True, exist_ok=True)

    outer_schedule = pd.read_csv(r11_root / "metrics" / "outer_case_schedule.csv")
    final_score = pd.read_csv(r11_root / "metrics" / "final_launch_score.csv")
    paired_index = (
        int(config.paired_start_index)
        if config.paired_start_index is not None
        else _select_same_start_index(
            final_score=final_score,
            outer_schedule=outer_schedule,
            config=_figure_config(config),
        )
    )
    selected_cases = _selected_outer_cases(outer_schedule, paired_index=paired_index)
    primitive_log = _read_selected_primitive_rows(
        r11_root=r11_root,
        outer_case_indices=tuple(int(v) for v in selected_cases["outer_case_index"]),
        config=_figure_config(config),
    )
    first_states = _read_first_primitive_initial_states(r11_root=r11_root, config=config)
    neutral_case_rows = _run_neutral_summary_cases(
        outer_schedule=outer_schedule,
        first_states=first_states,
        config=config,
    )
    neutral_case_frame = pd.DataFrame(neutral_case_rows)
    neutral_case_frame.to_csv(output_root / "metrics" / "neutral_rollout_by_case.csv", index=False)

    comparison_frame = _controller_comparison(
        final_score=final_score,
        neutral_case_frame=neutral_case_frame,
        config=config,
    )
    comparison_frame.to_csv(output_root / "metrics" / "neutral_vs_controller_by_policy_ladder.csv", index=False)

    figure_rows: list[dict[str, object]] = []
    for block_id in R11_FIDELITY_LADDER_BLOCK_IDS:
        rows = selected_cases[selected_cases["environment_block_id"].astype(str) == str(block_id)]
        if rows.empty:
            continue
        case_row = rows.iloc[0].to_dict()
        outer_case_index = int(case_row["outer_case_index"])
        neutral_rows = neutral_case_frame[
            neutral_case_frame["outer_case_index"].astype(int) == outer_case_index
        ]
        if neutral_rows.empty:
            continue
        neutral_path = _path_from_json(str(neutral_rows.iloc[0]["path_points_json"]))
        figure_path = (
            output_root
            / "figures"
            / f"r11_{run_label_slug}_bal_neutral_s{paired_index:02d}_{_short_ladder_id(block_id)}.png"
        )
        metadata = _plot_neutral_ladder_case(
            primitive_log=primitive_log,
            final_score=final_score,
            neutral_row=neutral_rows.iloc[0].to_dict(),
            neutral_path=neutral_path,
            case_row=case_row,
            output_path=figure_path,
            config=config,
        )
        figure_rows.append(
            {
                "diagnostic_version": DIAGNOSTIC_VERSION,
                "library_size_case_id": str(config.library_size_case_id),
                "paired_start_condition_index": int(paired_index),
                "outer_case_index": int(outer_case_index),
                "environment_block_id": str(block_id),
                "figure_path": figure_path.as_posix(),
                **metadata,
            }
        )

    pd.DataFrame(figure_rows).to_csv(output_root / "metrics" / "neutral_figure_summary.csv", index=False)
    manifest = {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "status": "complete",
        "run_label": run_label_slug.upper(),
        "r11_root": r11_root.as_posix(),
        "output_root": output_root.as_posix(),
        "library_size_case_id": str(config.library_size_case_id),
        "paired_start_condition_index": int(paired_index),
        "policy_ids": list(config.policy_ids),
        "neutral_policy_id": NEUTRAL_POLICY_ID,
        "r11_open_loop_layer_scope": (
            "one true-neutral rollout per unique R11 final outer case; library tiers "
            "and memory policies share the same final launch state inside an outer case, "
            "so duplicate neutral reruns are intentionally not repeated"
        ),
        "neutral_rollout": (
            "zero aggregate command, same plant/wind/implementation seed, same "
            "actuator lag and surface implementation perturbation"
        ),
        "summary_case_count": int(len(neutral_case_rows)),
        "figure_count": int(len(figure_rows)),
    }
    (output_root / "manifests" / "r11_balanced_neutral_baseline_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="ascii",
    )
    _write_report(
        output_root=output_root,
        manifest=manifest,
        comparison_frame=comparison_frame,
        figure_rows=figure_rows,
    )
    return {
        "status": "complete",
        "output_root": output_root.as_posix(),
        "paired_start_condition_index": int(paired_index),
        "neutral_case_count": int(len(neutral_case_rows)),
        "figure_count": int(len(figure_rows)),
    }


def _figure_config(config: NeutralDiagnosticConfig) -> R11BalancedLadderFigureConfig:
    return R11BalancedLadderFigureConfig(
        r11_root=config.r11_root,
        output_root=config.output_root,
        library_size_case_id=config.library_size_case_id,
        paired_start_index=config.paired_start_index,
        policy_ids=config.policy_ids,
        updraft_nx=config.updraft_nx,
        updraft_ny=config.updraft_ny,
        updraft_nz=config.updraft_nz,
    )


def _read_first_primitive_initial_states(*, r11_root: Path, config: NeutralDiagnosticConfig) -> pd.DataFrame:
    columns = [
        "library_size_case_id",
        "policy_id",
        "launch_role",
        "primitive_step_index",
        "outer_case_index",
        "environment_block_id",
        "episode_id",
        "initial_state_vector",
    ]
    frames: list[pd.DataFrame] = []
    for path in sorted((r11_root / "tables" / "primitive_execution_log").glob("*.csv.gz")):
        frame = pd.read_csv(path, usecols=columns)
        mask = (
            frame["library_size_case_id"].astype(str).eq(str(config.library_size_case_id))
            & frame["policy_id"].astype(str).eq("no_memory_baseline")
            & frame["launch_role"].astype(str).eq("final_heldout")
            & frame["primitive_step_index"].astype(int).eq(0)
        )
        if mask.any():
            frames.append(frame.loc[mask].copy())
    if not frames:
        raise ValueError("no_balanced_no_memory_first_primitive_rows_found")
    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(["outer_case_index", "environment_block_id"], keep="first")
    return result


def _run_neutral_summary_cases(
    *,
    outer_schedule: pd.DataFrame,
    first_states: pd.DataFrame,
    config: NeutralDiagnosticConfig,
) -> list[dict[str, object]]:
    frame = first_states.merge(
        outer_schedule,
        on=["outer_case_index", "environment_block_id"],
        how="left",
        suffixes=("", "_scheduled"),
    )
    frame = frame.sort_values(["outer_case_index"]).reset_index(drop=True)
    if int(config.summary_case_limit) > 0:
        frame = frame.iloc[: int(config.summary_case_limit)].copy()
    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        state = _state_from_json(row["initial_state_vector"])
        if state is None:
            rows.append(
                {
                    "outer_case_index": int(row["outer_case_index"]),
                    "environment_block_id": str(row["environment_block_id"]),
                    "neutral_status": "missing_initial_state",
                }
            )
            continue
        rows.append(_simulate_neutral_case(state=state, case_row=row.to_dict(), config=config))
    return rows


def _simulate_neutral_case(
    *,
    state: np.ndarray,
    case_row: dict[str, object],
    config: NeutralDiagnosticConfig,
) -> dict[str, object]:
    wind_field, env_meta = _wind_and_environment_metadata(case_row)
    implementation, plant, actuator_tau_s, wind_mode = _implementation_and_plant(case_row, wind_field)
    base_aircraft = adapt_glider(build_nausicaa_glider())
    aircraft = apply_plant_instance_to_aircraft(base_aircraft, plant)
    aircraft = apply_aileron_asymmetry_to_aircraft(aircraft, implementation)
    command = apply_surface_implementation(
        normalised_command_to_surface_rad(np.zeros(3, dtype=float)),
        implementation,
    )
    mechanism_flags = latency_mechanism_flags_from_case(
        str(implementation.latency_case),
        state_feedback_delay_applied=str(implementation.latency_case) in {"nominal", "conservative"},
    )
    x = as_state_vector(state).copy()
    if not bool(mechanism_flags["actuator_lag_applied"]):
        x[12:15] = command
    start_energy = _specific_energy_m(x)
    min_wall_margin = float("inf")
    min_floor_margin = float("inf")
    min_ceiling_margin = float("inf")
    max_abs_surface = float(np.max(np.abs(command)))
    termination = "time_limit_no_boundary_exit"
    trajectory_status = "finite_neutral_rollout"
    path_points: list[list[float]] = [list(map(float, x[:3]))]
    step_count = int(max(1, math.ceil(float(config.max_episode_time_s) / float(config.dt_s))))
    stop_index = step_count
    for step_index in range(step_count):
        margins = position_margin_m(x[:3], TRUE_SAFE_BOUNDS)
        min_wall_margin = min(min_wall_margin, float(margins["min_wall_margin_m"]))
        min_floor_margin = min(min_floor_margin, float(margins["floor_margin_m"]))
        min_ceiling_margin = min(min_ceiling_margin, float(margins["ceiling_margin_m"]))
        if margins["floor_margin_m"] < 0.0:
            termination = "floor_margin_stop"
            stop_index = step_index
            break
        if margins["ceiling_margin_m"] < 0.0:
            termination = "ceiling_margin_stop"
            stop_index = step_index
            break
        if margins["min_wall_margin_m"] < 0.0:
            termination = "wall_boundary_exit_retained"
            stop_index = step_index
            break
        x = _rk4_step(
            x=x,
            command=command,
            aircraft=aircraft,
            wind_field=wind_field,
            wind_mode=wind_mode,
            actuator_tau_s=actuator_tau_s,
            dt_s=float(config.dt_s),
        )
        max_abs_surface = max(max_abs_surface, float(np.max(np.abs(x[12:15]))))
        if not np.all(np.isfinite(x)):
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            termination = "nonfinite_trajectory"
            trajectory_status = "nonfinite_neutral_rollout"
            stop_index = step_index + 1
            break
        if step_index % max(1, int(config.plot_stride)) == 0:
            path_points.append(list(map(float, x[:3])))
    else:
        margins = position_margin_m(x[:3], TRUE_SAFE_BOUNDS)
        min_wall_margin = min(min_wall_margin, float(margins["min_wall_margin_m"]))
        min_floor_margin = min(min_floor_margin, float(margins["floor_margin_m"]))
        min_ceiling_margin = min(min_ceiling_margin, float(margins["ceiling_margin_m"]))
    if path_points[-1] != list(map(float, x[:3])):
        path_points.append(list(map(float, x[:3])))

    wall_exit = bool(termination == "wall_boundary_exit_retained")
    wall_face = _wall_exit_face(x[:3], wall_exit=wall_exit)
    front_wall_terminal = bool(
        wall_exit
        and wall_face == "front_wall_x_max"
        and min_floor_margin >= 0.0
        and min_ceiling_margin >= 0.0
        and _inside_terminal_yz(x[:3])
    )
    wrong_wall_exit = bool(wall_exit and not front_wall_terminal)
    floor_or_ceiling = bool(min_floor_margin < 0.0 or min_ceiling_margin < 0.0)
    hard_failure = bool(trajectory_status != "finite_neutral_rollout")
    safe_geometric = bool(not hard_failure and not floor_or_ceiling)
    end_energy = _specific_energy_m(x)
    row_for_score = {
        "selected_primitive_step_count": 0,
        "episode_rollout_duration_s": float(stop_index) * float(config.dt_s),
        "hard_failure": hard_failure,
        "floor_or_ceiling_violation": floor_or_ceiling,
        "no_viable_primitive": False,
        "expected_low_energy_dry_air_sink": False,
        "mission_success": front_wall_terminal,
        "front_wall_terminal_success": front_wall_terminal,
        "wrong_wall_exit": wrong_wall_exit,
        "safe_success": safe_geometric,
        "lift_capture": False,
        "terminal_wall_face": wall_face,
        "mission_outcome_label": (
            "front_wall_terminal_success"
            if front_wall_terminal
            else "wrong_wall_exit"
            if wrong_wall_exit
            else "floor_or_ceiling_violation"
            if floor_or_ceiling
            else "no_front_wall_terminal"
        ),
        "updraft_specific_energy_gain_proxy_m": 0.0,
        "lift_dwell_time_s": 0.0,
        "terminal_specific_energy_m": end_energy,
    }
    score = _neutral_score(row_for_score)
    return {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "outer_case_index": int(case_row["outer_case_index"]),
        "environment_block_id": str(case_row["environment_block_id"]),
        "paired_start_condition_index": int(float(case_row.get("paired_start_condition_index", -1))),
        "neutral_status": "complete",
        "neutral_policy_id": NEUTRAL_POLICY_ID,
        "initial_launch_speed_m_s": float(np.linalg.norm(state[6:9])),
        "termination_cause": termination,
        "trajectory_status": trajectory_status,
        "mission_success": bool(front_wall_terminal),
        "front_wall_terminal_success": bool(front_wall_terminal),
        "wrong_wall_exit": bool(wrong_wall_exit),
        "floor_or_ceiling_violation": bool(floor_or_ceiling),
        "hard_failure": bool(hard_failure),
        "safe_geometric": bool(safe_geometric),
        "terminal_wall_face": wall_face,
        "episode_flight_time_s": float(stop_index) * float(config.dt_s),
        "start_specific_energy_m": float(start_energy),
        "terminal_specific_energy_m": float(end_energy),
        "net_specific_energy_delta_m": float(end_energy - start_energy),
        "final_exit_x_w_m": float(x[STATE_INDEX["x_w"]]),
        "final_exit_y_w_m": float(x[STATE_INDEX["y_w"]]),
        "final_exit_z_w_m": float(x[STATE_INDEX["z_w"]]),
        "min_wall_margin_m": float(min_wall_margin),
        "floor_margin_m": float(min_floor_margin),
        "ceiling_margin_m": float(min_ceiling_margin),
        "max_abs_surface_rad": float(max_abs_surface),
        "neutral_diagnostic_score": float(score),
        "wind_mode": wind_mode,
        **env_meta,
        "path_points_json": json.dumps(path_points, separators=(",", ":")),
    }


def _wind_and_environment_metadata(case_row: dict[str, object]) -> tuple[object | None, dict[str, object]]:
    scheduled_active_fan_count = _scheduled_active_fan_count_for_context(
        protocol=R11_PROTOCOL,
        scheduled=case_row,
    )
    randomisation_config = _environment_randomisation_config_for_context(
        protocol=R11_PROTOCOL,
        scheduled=case_row,
        scheduled_active_fan_count=scheduled_active_fan_count,
    )
    w_layer = str(case_row.get("W_layer", "W0"))
    mode = str(case_row.get("environment_mode", "dry_air"))
    seed = int(float(case_row.get("environment_seed", 0)))
    instance = environment_instance_for_mode(
        w_layer,
        mode,
        seed,
        randomisation_config=randomisation_config,
    )
    metadata = environment_metadata_from_instance(instance)
    binding = resolve_surrogate_binding(
        w_layer,
        metadata,
        repo_root=Path(".").resolve(),
        randomisation_seed=seed,
    )
    wind = wind_field_for_binding(binding, repo_root=Path(".").resolve())
    return wind, {
        "environment_instance_id": instance.environment_id,
        "W_layer": w_layer,
        "environment_mode": mode,
        "fan_count": int(binding.fan_count),
        "active_fan_count": int(sum(bool(value) for value in binding.active_fan_mask)),
    }


def _implementation_and_plant(case_row: dict[str, object], wind_field: object | None):
    env_layer = str(case_row.get("W_layer", "W0"))
    seed = int(float(case_row.get("plant_implementation_seed", case_row.get("environment_seed", 0))))
    full_w3 = _uses_full_w3_randomisation_block(
        protocol=R11_PROTOCOL,
        environment_block_id=str(case_row.get("environment_block_id", "")),
    )
    plant_layer = env_layer if full_w3 else (
        "W2" if R11_PROTOCOL.requires_no_glider_latency_variation_audit else env_layer
    )
    implementation_layer = env_layer if full_w3 else (
        "W2" if R11_PROTOCOL.requires_no_glider_latency_variation_audit else env_layer
    )
    implementation = implementation_instance_for_layer(
        implementation_layer,
        seed,
        latency_case="nominal",
    )
    plant = plant_instance_for_layer(plant_layer, seed)
    latency = _latency_for_implementation(str(implementation.latency_case), implementation)
    mechanism_flags = latency_mechanism_flags_from_case(
        str(implementation.latency_case),
        state_feedback_delay_applied=str(implementation.latency_case) in {"nominal", "conservative"},
    )
    base_tau_s = (
        actuator_tau_for_case(latency)
        if bool(mechanism_flags["actuator_lag_applied"])
        else (1.0, 1.0, 1.0)
    )
    tau_s = adjusted_actuator_tau_s(base_tau_s, implementation)
    return implementation, plant, tau_s, "panel" if wind_field is not None else "none"


def _neutral_score(row: dict[str, object]) -> float:
    hard_failure = bool(row["hard_failure"])
    floor_or_ceiling = bool(row["floor_or_ceiling_violation"])
    wrong_wall_exit = bool(row["wrong_wall_exit"])
    mission_success = bool(row["mission_success"])
    if hard_failure or floor_or_ceiling:
        return -80.0
    if wrong_wall_exit:
        return -50.0
    if mission_success:
        reserve = max(float(row["terminal_specific_energy_m"]) - float(TRUE_SAFE_BOUNDS.z_w_m[0]), 0.0)
        return 100.0 + min(10.0 * reserve, 20.0)
    return 0.0


def _controller_comparison(
    *,
    final_score: pd.DataFrame,
    neutral_case_frame: pd.DataFrame,
    config: NeutralDiagnosticConfig,
) -> pd.DataFrame:
    neutral = neutral_case_frame[neutral_case_frame["neutral_status"].astype(str).eq("complete")].copy()
    controlled = final_score[
        final_score["library_size_case_id"].astype(str).eq(str(config.library_size_case_id))
        & final_score["policy_id"].astype(str).isin(config.policy_ids)
    ].copy()
    rows: list[dict[str, object]] = []
    for (block_id, policy_id), group in controlled.groupby(["environment_block_id", "policy_id"], dropna=False):
        merged = group.merge(
            neutral[
                [
                    "outer_case_index",
                    "mission_success",
                    "safe_geometric",
                    "wrong_wall_exit",
                    "floor_or_ceiling_violation",
                    "neutral_diagnostic_score",
                    "terminal_specific_energy_m",
                ]
            ].rename(
                columns={
                    "mission_success": "neutral_mission_success",
                    "safe_geometric": "neutral_safe_geometric",
                    "wrong_wall_exit": "neutral_wrong_wall_exit",
                    "floor_or_ceiling_violation": "neutral_floor_or_ceiling_violation",
                    "terminal_specific_energy_m": "neutral_terminal_specific_energy_m",
                }
            ),
            on="outer_case_index",
            how="inner",
        )
        if merged.empty:
            continue
        rows.append(
            {
                "diagnostic_version": DIAGNOSTIC_VERSION,
                "environment_block_id": str(block_id),
                "library_size_case_id": str(config.library_size_case_id),
                "policy_id": str(policy_id),
                "paired_case_count": int(len(merged)),
                "neutral_mission_success_rate": float(_mean_bool(merged["neutral_mission_success"])),
                "controlled_mission_success_rate": float(_mean_bool(merged["mission_success"])),
                "mission_success_delta_control_minus_neutral": float(
                    _mean_bool(merged["mission_success"]) - _mean_bool(merged["neutral_mission_success"])
                ),
                "neutral_safe_geometric_rate": float(_mean_bool(merged["neutral_safe_geometric"])),
                "controlled_safe_success_rate": float(_mean_bool(merged["safe_success"])),
                "safe_success_delta_control_minus_neutral": float(
                    _mean_bool(merged["safe_success"]) - _mean_bool(merged["neutral_safe_geometric"])
                ),
                "neutral_wrong_wall_exit_rate": float(_mean_bool(merged["neutral_wrong_wall_exit"])),
                "controlled_wrong_wall_exit_rate": float(_mean_bool(merged["wrong_wall_exit"])),
                "neutral_floor_or_ceiling_rate": float(_mean_bool(merged["neutral_floor_or_ceiling_violation"])),
                "controlled_floor_or_ceiling_rate": float(_mean_bool(merged["floor_or_ceiling_violation"])),
                "neutral_mean_score": float(pd.to_numeric(merged["neutral_diagnostic_score"], errors="coerce").mean()),
                "controlled_mean_launch_score": float(pd.to_numeric(merged["launch_score"], errors="coerce").mean()),
                "score_delta_control_minus_neutral": float(
                    pd.to_numeric(merged["launch_score"], errors="coerce").mean()
                    - pd.to_numeric(merged["neutral_diagnostic_score"], errors="coerce").mean()
                ),
                "neutral_mean_terminal_specific_energy_m": float(
                    pd.to_numeric(merged["neutral_terminal_specific_energy_m"], errors="coerce").mean()
                ),
                "controlled_mean_terminal_specific_energy_m": float(
                    pd.to_numeric(merged["terminal_specific_energy_m"], errors="coerce").mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["environment_block_id", "policy_id"]).reset_index(drop=True)


def _plot_neutral_ladder_case(
    *,
    primitive_log: pd.DataFrame,
    final_score: pd.DataFrame,
    neutral_row: dict[str, object],
    neutral_path: np.ndarray,
    case_row: dict[str, object],
    output_path: Path,
    config: NeutralDiagnosticConfig,
) -> dict[str, object]:
    outer_case_index = int(case_row["outer_case_index"])
    block_id = str(case_row["environment_block_id"])
    subset = primitive_log[
        (primitive_log["outer_case_index"].astype(int) == outer_case_index)
        & (primitive_log["environment_block_id"].astype(str) == block_id)
    ].copy()
    summary_subset = final_score[
        (final_score["library_size_case_id"].astype(str) == str(config.library_size_case_id))
        & (final_score["outer_case_index"].astype(int) == outer_case_index)
        & (final_score["environment_block_id"].astype(str) == block_id)
        & (final_score["policy_id"].astype(str).isin(config.policy_ids))
    ].copy()

    fig, ax = _new_baseline_axis()
    env_meta = _draw_updraft_context_from_case(ax=ax, case_row=case_row, config=_figure_config(config))
    plotted_count = 0
    if neutral_path.shape[0] >= 2:
        ax.plot(
            neutral_path[:, 0],
            neutral_path[:, 1],
            neutral_path[:, 2],
            color=NEUTRAL_COLOR,
            linestyle=":",
            linewidth=1.75,
            alpha=0.95,
            label=NEUTRAL_LABEL,
            zorder=32,
        )
        marker = "o" if bool(neutral_row.get("mission_success", False)) else "x"
        ax.scatter(
            [neutral_path[-1, 0]],
            [neutral_path[-1, 1]],
            [neutral_path[-1, 2]],
            color=NEUTRAL_COLOR,
            marker=marker,
            s=48,
            depthshade=False,
            zorder=34,
        )
    for policy_id in FINAL_POLICY_ORDER:
        if policy_id not in config.policy_ids:
            continue
        rows = subset[subset["policy_id"].astype(str) == str(policy_id)]
        if rows.empty:
            continue
        label = FINAL_POLICY_LABELS.get(policy_id, policy_id)
        color = FINAL_POLICY_COLORS.get(policy_id, "#333333")
        for episode_id in sorted(rows["episode_id"].astype(str).unique()):
            episode_rows = rows[rows["episode_id"].astype(str) == episode_id]
            points = _episode_points(episode_rows)
            if points.shape[0] < 2:
                continue
            ax.plot(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                color=color,
                linestyle=FINAL_POLICY_LINESTYLES.get(policy_id, "-"),
                linewidth=1.15,
                alpha=0.82,
                label=label,
                zorder=25,
            )
            _draw_primitive_markers(
                ax,
                episode_rows,
                color=color,
                marker_size=PRIMITIVE_MARKER_SIZE_FINAL,
                alpha=0.68,
                zorder=27,
                include_labels=(plotted_count == 0),
            )
            marker = "o" if _policy_safe_success(summary_subset, policy_id) else "x"
            marker_size = FINAL_POLICY_MARKER_SIZES.get(policy_id, 28)
            ax.scatter(
                [points[-1, 0]],
                [points[-1, 1]],
                [points[-1, 2]],
                color=color if marker == "x" else "none",
                edgecolors=color,
                linewidths=1.35,
                marker=marker,
                s=marker_size,
                depthshade=False,
                zorder=26,
            )
            plotted_count += 1
    label = str(case_row.get("environment_block_label", case_row.get("environment_block_id", "")))
    speed = float(pd.to_numeric(summary_subset.get("initial_launch_speed_m_s", pd.Series([np.nan])), errors="coerce").mean())
    ax.set_title(
        f"{label} | start {int(case_row.get('paired_start_condition_index', -1))} | "
        f"v0={speed:.2f} m/s | neutral={_bool01(neutral_row.get('mission_success', False))}",
        fontsize=10,
        pad=12,
    )
    _add_legend(ax)
    _save_figure(fig, output_path)
    return {
        "final_policy_count": int(plotted_count),
        "neutral_mission_success": bool(neutral_row.get("mission_success", False)),
        "neutral_terminal_wall_face": str(neutral_row.get("terminal_wall_face", "")),
        **env_meta,
    }


def _inside_terminal_yz(position: np.ndarray) -> bool:
    y_w = float(position[1])
    z_w = float(position[2])
    y_min, y_max = TRUE_SAFE_BOUNDS.y_w_m
    z_min, z_max = TRUE_SAFE_BOUNDS.z_w_m
    return bool(float(y_min) <= y_w <= float(y_max) and float(z_min) <= z_w <= float(z_max))


def _state_from_json(value: object) -> np.ndarray | None:
    try:
        return as_state_vector(np.asarray(json.loads(str(value)), dtype=float))
    except Exception:
        return None


def _path_from_json(value: str) -> np.ndarray:
    try:
        arr = np.asarray(json.loads(value), dtype=float)
    except Exception:
        return np.empty((0, 3), dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return np.empty((0, 3), dtype=float)
    return arr


def _mean_bool(values: pd.Series) -> float:
    if len(values) == 0:
        return float("nan")
    return float(pd.Series(values).map(_truthy).mean())


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "1.0", "true", "yes", "y"}


def _bool01(value: object) -> str:
    return "1" if _truthy(value) else "0"


def _run_label_slug_from_root(r11_root: Path) -> str:
    label = str(Path(r11_root).name).strip().lower()
    safe = "".join(ch if ch.isalnum() else "_" for ch in label).strip("_")
    return safe or "run"


def _write_report(
    *,
    output_root: Path,
    manifest: dict[str, object],
    comparison_frame: pd.DataFrame,
    figure_rows: list[dict[str, object]],
) -> None:
    run_label = str(manifest.get("run_label", "R11"))
    lines = [
        f"# R11 {run_label} Balanced-Cluster Neutral-Intervention Diagnostic",
        "",
        f"- Diagnostic version: `{DIAGNOSTIC_VERSION}`",
        f"- R11 root: `{manifest['r11_root']}`",
        f"- Library tier: `{manifest['library_size_case_id']}`",
        f"- True neutral cases: `{manifest['summary_case_count']}`",
        f"- Figure count: `{manifest['figure_count']}`",
        "",
        "`no_memory_baseline` is still closed-loop governor/LQR control. The neutral baseline here is a separate zero-command plant rollout from the same R11 final launch states.",
        "",
        "The open-loop layer is run once per unique R11 final outer case. This is the correct comparison unit because all library tiers and memory policies reuse the same final launch state within that outer case.",
        "",
        "| Ladder | Policy | n | Neutral mission | Controlled mission | Mission delta | Neutral score | Controlled score | Score delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in comparison_frame.iterrows():
        lines.append(
            "| {ladder} | {policy} | {n:d} | {nm:.3f} | {cm:.3f} | {dm:+.3f} | {ns:.2f} | {cs:.2f} | {ds:+.2f} |".format(
                ladder=row["environment_block_id"],
                policy=row["policy_id"],
                n=int(row["paired_case_count"]),
                nm=float(row["neutral_mission_success_rate"]),
                cm=float(row["controlled_mission_success_rate"]),
                dm=float(row["mission_success_delta_control_minus_neutral"]),
                ns=float(row["neutral_mean_score"]),
                cs=float(row["controlled_mean_launch_score"]),
                ds=float(row["score_delta_control_minus_neutral"]),
            )
        )
    lines.extend(
        [
            "",
            "Figures:",
            "",
        ]
    )
    for row in figure_rows:
        lines.append(
            f"- `{row['environment_block_id']}`: `{row['figure_path']}`"
        )
    (output_root / "reports" / "r11_balanced_neutral_baseline_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="ascii",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r11-root", type=Path, default=DEFAULT_R11_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--library-size-case-id", default=DEFAULT_LIBRARY_SIZE_CASE)
    parser.add_argument("--paired-start-index", type=int, default=None)
    parser.add_argument("--summary-case-limit", type=int, default=0)
    parser.add_argument("--max-episode-time-s", type=float, default=20.0)
    parser.add_argument("--dt-s", type=float, default=0.02)
    args = parser.parse_args(argv)
    result = run_r11_balanced_neutral_baseline_figures(
        NeutralDiagnosticConfig(
            r11_root=args.r11_root,
            output_root=args.output_root,
            library_size_case_id=args.library_size_case_id,
            paired_start_index=args.paired_start_index,
            summary_case_limit=args.summary_case_limit,
            max_episode_time_s=args.max_episode_time_s,
            dt_s=args.dt_s,
        )
    )
    print(result)
    return 0 if result.get("status") == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
