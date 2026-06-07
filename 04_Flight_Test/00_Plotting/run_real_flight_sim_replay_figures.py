"""Plot real flight beside two simulation replays.

For each valid real-flight throw this script starts from the approved launch
gate state, keeps the configured 0.04 s handoff window as a measured held-out
splice, then generates two simulation replies:

1. the simulator re-selects primitives with the frozen governor;
2. the simulator reuses the primitive variants selected during the real throw.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
FLIGHT_ROOT = ROOT / "04_Flight_Test"
RUNTIME_ROOT = FLIGHT_ROOT / "01_Runtime"
CONTROLLER_ROOT = FLIGHT_ROOT / "02_Controller"
CONTROL_ROOT = ROOT / "03_Control"
INNER_LOOP_ROOT = CONTROL_ROOT / "02_Inner_Loop"
CONTROL_PRIMITIVES_ROOT = CONTROL_ROOT / "03_Primitives"
CONTROL_PLOTTING_ROOT = CONTROL_ROOT / "01_Plotting"
CONTROL_SCENARIOS_ROOT = CONTROL_ROOT / "04_Scenarios"

for path in (
    RUNTIME_ROOT,
    CONTROLLER_ROOT,
    INNER_LOOP_ROOT,
    CONTROL_PLOTTING_ROOT,
    CONTROL_PRIMITIVES_ROOT,
    CONTROL_SCENARIOS_ROOT,
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from command_contract import normalised_command_to_surface_rad  # noqa: E402
from env_ctx import EnvironmentMetadata  # noqa: E402
from env_surrogate import resolve_surrogate_binding, wind_field_for_binding  # noqa: E402
from exit_gate import evaluate_exit_gate  # noqa: E402
from flight_config import FlightRuntimeConfig  # noqa: E402
from flight_dynamics import adapt_glider, state_derivative  # noqa: E402
from frozen_flight_controller import FrozenFlightController  # noqa: E402
from glider import build_nausicaa_glider  # noqa: E402
from state_contract import STATE_INDEX, STATE_NAMES, STATE_SIZE, as_state_vector  # noqa: E402

from run_r9_preflight_3d_figures import (  # noqa: E402
    R9PreflightFigureConfig,
    _build_alpha_cmap,
    _draw_center_slices,
    _draw_updraft_isosurfaces,
    _sample_updraft_volume,
)
from run_thesis_3d_baseline_figure import (  # noqa: E402
    AXIS_EDGE_LW,
    FAN_OUTLET_ALPHA,
    FAN_OUTLET_DASH,
    FAN_OUTLET_DIAMETER,
    FAN_OUTLET_EDGE_LW,
    FAN_OUTLET_PLOT_Z_M,
    TICK_LABEL_FONTSIZE,
    VIEW_AZIM,
    VIEW_ELEV,
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
    Z_MAX,
    Z_MIN,
    _draw_arena_box,
    _draw_floor_grid,
)


FIGURE_RUN_VERSION = "real_flight_sim_replay_measured_fan_updraft_v2"
DEFAULT_RESULT_ROOT = FLIGHT_ROOT / "05_Results"
DEFAULT_OUTPUT_ROOT = FLIGHT_ROOT / "A_figures" / "real_flight_sim_replay"
DEFAULT_LIBRARY_TIER = "balanced_cluster"
DEFAULT_REPLAY_DT_S = 0.005
FIRST_WINDOW_STATE_AUDIT_S = 0.04

TRACE_ORDER = ("real", "sim_self_governor", "sim_real_decisions")
TRACE_LABELS = {
    "real": "reality",
    "sim_self_governor": "sim own governor",
    "sim_real_decisions": "sim real decisions",
}
TRACE_COLORS = {
    "real": "#111111",
    "sim_self_governor": "#1f77b4",
    "sim_real_decisions": "#d97706",
}
TRACE_STYLES = {
    "real": "-",
    "sim_self_governor": "--",
    "sim_real_decisions": "-.",
}
TRACE_LINEWIDTHS = {
    "real": 2.3,
    "sim_self_governor": 1.9,
    "sim_real_decisions": 1.9,
}
TIME_SERIES_SPECS = (
    ("x_w", "x (m)", 1.0),
    ("y_w", "y (m)", 1.0),
    ("z_w", "z (m)", 1.0),
    ("phi", "roll phi (deg)", 180.0 / math.pi),
    ("theta", "pitch theta (deg)", 180.0 / math.pi),
    ("psi", "yaw psi (deg)", 180.0 / math.pi),
)
ANGLE_STATE_NAMES = {"phi", "theta", "psi"}
RELATIVE_POSITION_STATE_NAMES = {"x_w", "y_w", "z_w"}
FIRST_WINDOW_STATE_ERROR_FIELDS = [
    "figure_run_version",
    "case_id",
    "session_id",
    "throw_id",
    "model",
    "model_label",
    "window_s",
    "state_component",
    "residual_policy",
    "unit",
    "sample_count",
    "mae",
    "rmse",
    "max_abs",
    "final_residual",
]
EXECUTION_TIMING_FIELDS = [
    "figure_run_version",
    "case_id",
    "session_id",
    "throw_id",
    "valid_throw",
    "termination_reason",
    "launch_speed_m_s",
    "launch_handoff_duration_s",
    "first_active_command_elapsed_s",
    "first_active_command_lag_after_handoff_s",
    "controller_decision_count",
    "selected_decision_count",
    "continuation_decision_count",
    "max_decision_time_s_summary",
    "max_decision_time_s_logged",
    "p95_decision_time_s_logged",
    "mean_decision_time_s_logged",
    "max_continuation_decision_time_s",
    "p95_continuation_decision_time_s",
    "mean_continuation_decision_time_s",
    "continuation_prepared_before_boundary_count",
    "continuation_prepared_before_boundary_rate",
    "max_continuation_commit_lag_s",
    "p95_continuation_commit_lag_s",
    "mean_continuation_commit_lag_s",
    "continuation_commit_lag_gt_20ms_count",
    "continuation_commit_lag_gt_50ms_count",
    "continuation_late_decision_count_summary",
    "slot_command_update_count",
    "packet_count",
    "active_metric_buffered_row_count",
    "active_metric_buffer_flush_count",
    "scheduler_decision_source_counts_json",
    "throw_root",
]


@dataclass(frozen=True)
class ThrowCase:
    case_id: str
    session_id: str
    throw_id: str
    throw_root: Path
    summary: dict[str, Any]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class ReplayResult:
    trace: list[dict[str, Any]]
    status: str
    termination_reason: str
    selected_variant_ids: list[str]


@dataclass(frozen=True)
class ReplayEnvironment:
    wind_model: object | None
    wind_mode: str
    fan_positions_m: tuple[tuple[float, float], ...]
    metadata: dict[str, Any]


class CommandDelayBuffer:
    def __init__(self, *, delay_s: float, neutral_command_rad: np.ndarray) -> None:
        self.delay_s = max(0.0, float(delay_s))
        self.neutral_command_rad = np.asarray(neutral_command_rad, dtype=float).reshape(3).copy()
        self._times_s: list[float] = [-self.delay_s - 1.0e-9, 0.0]
        self._commands_rad: list[np.ndarray] = [self.neutral_command_rad.copy(), self.neutral_command_rad.copy()]

    def record(self, t_s: float, command_rad: np.ndarray) -> None:
        command = np.asarray(command_rad, dtype=float).reshape(3).copy()
        timestamp = float(t_s)
        if timestamp <= self._times_s[-1] + 1.0e-12:
            self._commands_rad[-1] = command
        else:
            self._times_s.append(timestamp)
            self._commands_rad.append(command)

    def command_at(self, t_s: float) -> np.ndarray:
        query_t_s = float(t_s) - self.delay_s
        index = int(np.searchsorted(np.asarray(self._times_s, dtype=float), query_t_s, side="right") - 1)
        index = int(np.clip(index, 0, len(self._commands_rad) - 1))
        return self._commands_rad[index].copy()


def main() -> None:
    args = _build_arg_parser().parse_args()
    output_root = Path(args.output_root)
    for subdir in ("figures", "metrics", "manifests", "reports"):
        (output_root / subdir).mkdir(parents=True, exist_ok=True)

    throws = discover_throw_cases(
        result_root=Path(args.result_root),
        throw_roots=[Path(value) for value in args.throw_root],
        case_id=str(args.case_id).strip(),
        session_id=str(args.session_id).strip(),
        include_invalid=bool(args.include_invalid),
    )
    if args.max_throws is not None:
        throws = throws[: max(0, int(args.max_throws))]
    if not throws:
        raise RuntimeError("No real-flight throws with state_samples.csv were found.")

    summary_rows: list[dict[str, Any]] = []
    state_error_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    environment_rows: list[dict[str, Any]] = []
    manifest_figures: list[str] = []
    for throw in throws:
        traces, replay_rows, replay_environment = build_throw_replays(
            throw,
            replay_dt_s=float(args.replay_dt_s),
            library_tier=str(args.library_tier),
            real_decision_timing=str(args.real_decision_timing),
        )
        figure_path = (
            output_root
            / "figures"
            / safe_name(throw.case_id)
            / safe_name(throw.session_id)
            / f"{safe_name(throw.throw_id)}_reality_vs_sim_replay.png"
        )
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        updraft_plot_meta = plot_throw_replay(
            throw,
            traces,
            output_path=figure_path,
            replay_environment=replay_environment,
        )
        environment_rows.append(
            {
                "figure_run_version": FIGURE_RUN_VERSION,
                "case_id": throw.case_id,
                "session_id": throw.session_id,
                "throw_id": throw.throw_id,
                **replay_environment.metadata,
                **updraft_plot_meta,
            }
        )
        trace_path = (
            output_root
            / "metrics"
            / safe_name(throw.case_id)
            / safe_name(throw.session_id)
            / f"{safe_name(throw.throw_id)}_replay_traces.csv"
        )
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        write_replay_trace_csv(trace_path, traces)
        state_error_rows.extend(
            first_window_state_error_rows(
                throw=throw,
                traces=traces,
                window_s=FIRST_WINDOW_STATE_AUDIT_S,
            )
        )
        timing_rows.append(execution_timing_row(throw))
        summary_rows.extend(
            summary_row(
                throw=throw,
                model_key=model_key,
                replay=replay_rows[model_key],
                figure_path=figure_path,
                trace_path=trace_path,
            )
            for model_key in TRACE_ORDER
        )
        manifest_figures.append(figure_path.as_posix())

    summary_path = output_root / "metrics" / "real_flight_sim_replay_summary.csv"
    write_csv(summary_path, summary_rows)
    state_error_path = output_root / "metrics" / "first_0p04_state_replay_error_summary.csv"
    write_csv(state_error_path, state_error_rows, fieldnames=FIRST_WINDOW_STATE_ERROR_FIELDS)
    timing_path = output_root / "metrics" / "execution_timing_audit.csv"
    write_csv(timing_path, timing_rows, fieldnames=EXECUTION_TIMING_FIELDS)
    environment_path = output_root / "metrics" / "replay_environment_summary.csv"
    write_csv(environment_path, environment_rows)
    manifest = {
        "figure_run_version": FIGURE_RUN_VERSION,
        "status": "complete",
        "result_root": Path(args.result_root).as_posix(),
        "output_root": output_root.as_posix(),
        "library_tier": str(args.library_tier),
        "replay_dt_s": float(args.replay_dt_s),
        "real_decision_timing": str(args.real_decision_timing),
        "replay_environment_summary_path": environment_path.as_posix(),
        "replay_environment_policy": (
            "Each throw builds a W2 annular-GP wind field from measured fan_positions.csv xy positions, "
            "using nominal fan power/width and active masks; dry-air is used only when no visible fan "
            "position is available."
        ),
        "launch_state_policy": (
            "approved launch-plane state from prelaunch_state_samples.csv when present; "
            "fallback to first active state sample; the 0.04 s launch handoff is copied from measured "
            "state samples and simulation dynamics start from the measured post-handoff state"
        ),
        "simulation_reply_policy": {
            "sim_self_governor": (
                "measured-held-out 0.04 s handoff splice, then the frozen governor re-selects each primitive "
                "from the simulated state using the runtime boundary-state predictor; "
                "plant commands pass through the runtime surface_command_delay_s buffer before actuator lag; "
                "plant dynamics use the measured-fan W2 annular-GP replay wind field"
            ),
            "sim_real_decisions": (
                "measured-held-out 0.04 s handoff splice, then the simulator applies the real throw's selected "
                "primitive_variant_id sequence and recomputes 50 Hz LQR slot commands from the simulated state; "
                "plant commands pass through the runtime surface_command_delay_s buffer before actuator lag; "
                "plant dynamics use the measured-fan W2 annular-GP replay wind field"
            ),
        },
        "figures": manifest_figures,
        "first_window_state_audit": {
            "window_s": FIRST_WINDOW_STATE_AUDIT_S,
            "summary_path": state_error_path.as_posix(),
            "policy": (
                "SysID-style actual-minus-simulation residuals on the measured real-flight sample times; "
                "x/y use relative displacement from the common launch state, z uses altitude loss, "
                "Euler angles use wrapped degree residuals, and remaining states use direct residuals"
            ),
        },
        "execution_timing_audit": {
            "summary_path": timing_path.as_posix(),
            "policy": (
                "decision_time_s is pure frozen-governor compute time; scheduler_commit_lag_s is the "
                "runtime loop commit offset relative to each 0.10 s primitive boundary"
            ),
        },
    }
    (output_root / "manifests" / "real_flight_sim_replay_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="ascii",
    )
    write_report(
        output_root / "reports" / "real_flight_sim_replay_report.md",
        summary_rows,
        manifest,
        state_error_rows,
        timing_rows,
    )
    print(output_root.as_posix())


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate real-flight versus two simulation-replay comparison figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--result-root", default=DEFAULT_RESULT_ROOT.as_posix())
    parser.add_argument("--throw-root", action="append", default=[], help="Direct throw root to plot.")
    parser.add_argument("--case-id", default="", help="Optional case id filter, e.g. E0.1.")
    parser.add_argument("--session-id", default="", help="Optional session folder filter.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT.as_posix())
    parser.add_argument("--library-tier", default=DEFAULT_LIBRARY_TIER)
    parser.add_argument("--replay-dt-s", type=float, default=DEFAULT_REPLAY_DT_S)
    parser.add_argument("--max-throws", type=int, default=None)
    parser.add_argument("--include-invalid", action="store_true")
    parser.add_argument(
        "--real-decision-timing",
        choices=("logged", "boundary"),
        default="logged",
        help="Use logged real commit times, or ideal 0.10 s primitive boundaries, for sim_real_decisions.",
    )
    return parser


def discover_throw_cases(
    *,
    result_root: Path,
    throw_roots: list[Path],
    case_id: str,
    session_id: str,
    include_invalid: bool,
) -> list[ThrowCase]:
    roots = list(throw_roots)
    if not roots:
        search_root = Path(result_root)
        if case_id:
            search_root = search_root / case_id
        if session_id:
            search_root = search_root / session_id
        roots = sorted(path.parents[1] for path in search_root.rglob("metrics/state_samples.csv"))

    cases: list[ThrowCase] = []
    seen: set[Path] = set()
    for root in roots:
        throw_root = Path(root)
        try:
            resolved = throw_root.resolve()
        except OSError:
            resolved = throw_root
        if resolved in seen:
            continue
        seen.add(resolved)
        if not (throw_root / "metrics" / "state_samples.csv").exists():
            continue
        summary = read_json(throw_root / "manifests" / "real_flight_runtime_summary.json")
        manifest = read_json(throw_root / "manifests" / "real_flight_runtime_manifest.json")
        if not include_invalid and summary.get("valid_throw") is False:
            continue
        found_case_id = str(summary.get("experiment_case_id") or infer_case_id(throw_root, result_root))
        found_session_id = infer_session_id(throw_root, result_root)
        if case_id and found_case_id != case_id:
            continue
        if session_id and found_session_id != session_id:
            continue
        cases.append(
            ThrowCase(
                case_id=found_case_id,
                session_id=found_session_id,
                throw_id=str(summary.get("run_label") or throw_root.name),
                throw_root=throw_root,
                summary=summary,
                manifest=manifest,
            )
        )
    return sorted(cases, key=lambda item: (item.case_id, item.session_id, item.throw_id))


def build_throw_replays(
    throw: ThrowCase,
    *,
    replay_dt_s: float,
    library_tier: str,
    real_decision_timing: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, ReplayResult], ReplayEnvironment]:
    launch_state, launch_host_t_s = launch_state_for_throw(throw.throw_root)
    real_trace = measured_trace(throw.throw_root, launch_state=launch_state, launch_host_t_s=launch_host_t_s)
    duration_s = max(
        [to_float(row.get("t_s"), 0.0) for row in real_trace]
        + [float(throw.summary.get("max_duration_s", 0.0) or 0.0)]
    )
    if duration_s <= 0.0:
        duration_s = max(0.0, to_float(throw.summary.get("first_active_command_elapsed_s"), 0.0))
    config = runtime_config_from_manifest(throw.manifest, library_tier=library_tier)
    replay_environment = replay_environment_for_throw(throw)
    handoff_state, handoff_trace = measured_handoff_state_and_trace(
        real_trace,
        handoff_s=float(config.launch_handoff_duration_s),
    )
    sim_self = simulate_self_governor_reply(
        launch_state=launch_state,
        handoff_state=handoff_state,
        handoff_trace=handoff_trace,
        config=config,
        duration_s=duration_s,
        replay_dt_s=replay_dt_s,
        replay_environment=replay_environment,
    )
    sim_real = simulate_real_decision_reply(
        handoff_state,
        throw.throw_root,
        handoff_trace=handoff_trace,
        config=config,
        duration_s=duration_s,
        replay_dt_s=replay_dt_s,
        timing=real_decision_timing,
        replay_environment=replay_environment,
    )
    real_replay = ReplayResult(
        trace=real_trace,
        status="measured",
        termination_reason=str(throw.summary.get("termination_reason", "")),
        selected_variant_ids=selected_variant_ids_from_decisions(throw.throw_root),
    )
    return (
        {
            "real": real_replay.trace,
            "sim_self_governor": sim_self.trace,
            "sim_real_decisions": sim_real.trace,
        },
        {
            "real": real_replay,
            "sim_self_governor": sim_self,
            "sim_real_decisions": sim_real,
        },
        replay_environment,
    )


def replay_environment_for_throw(throw: ThrowCase) -> ReplayEnvironment:
    fan_positions = fan_positions_from_log(throw.throw_root)
    if not fan_positions:
        return ReplayEnvironment(
            wind_model=None,
            wind_mode="none",
            fan_positions_m=(),
            metadata={
                "updraft_context_status": "no_visible_fan_position",
                "W_layer": "W0",
                "environment_mode": "dry_air",
                "updraft_model_id": "dry_air_zero_wind",
                "updraft_max_m_s": 0.0,
            },
        )
    active_mask = tuple(True for _ in fan_positions)
    fan_count = len(fan_positions)
    model_id = "four_annular_gp_grid" if fan_count >= 4 else "single_annular_gp_grid"
    environment_mode = "annular_gp_four" if fan_count >= 4 else "annular_gp_single"
    metadata = EnvironmentMetadata(
        environment_id=f"real_flight_{throw.case_id}_{throw.session_id}_{throw.throw_id}_measured_fan",
        environment_instance_id=f"real_flight_{throw.case_id}_{throw.session_id}_{throw.throw_id}_measured_fan",
        fan_count=int(fan_count),
        fan_positions_m=fan_positions,
        fan_power_scales=tuple(1.0 for _ in fan_positions),
        active_fan_mask=active_mask,
        updraft_model_id=model_id,
        updraft_amplitude_scale=1.0,
        updraft_width_scale=1.0,
        updraft_centre_shift_m=(0.0, 0.0),
        residual_field_id="real_flight_measured_fan_position_nominal_strength",
        local_uncertainty_scale=1.0,
        randomisation_seed=None,
        model_source="real_flight_fan_positions_csv",
        W_layer="W2",
        wind_mode="panel",
        environment_mode=environment_mode,
        claim_status="real_flight_replay_measured_fan_position_uses_nominal_w2_annular_gp",
    )
    binding = resolve_surrogate_binding("W2", metadata, repo_root=ROOT)
    wind_model = wind_field_for_binding(binding, repo_root=ROOT)
    wind_mode = "panel" if wind_model is not None else "none"
    return ReplayEnvironment(
        wind_model=wind_model,
        wind_mode=wind_mode,
        fan_positions_m=tuple((float(x), float(y)) for x, y in fan_positions),
        metadata={
            "updraft_context_status": (
                "measured_fan_w2_annular_gp"
                if wind_model is not None
                else f"no_ready_wind_field:{binding.blocked_reason}"
            ),
            "W_layer": binding.W_layer,
            "environment_mode": binding.environment_mode,
            "updraft_model_id": binding.updraft_model_id,
            "surrogate_family": binding.surrogate_family,
            "surrogate_role": binding.surrogate_role,
            "surrogate_binding_status": binding.surrogate_binding_status,
            "surrogate_blocked_reason": binding.blocked_reason,
            "fan_count": int(binding.fan_count),
            "active_fan_count": int(sum(bool(value) for value in binding.active_fan_mask)),
            "fan_positions_m": json.dumps(binding.fan_positions_m),
            "fan_power_scales": json.dumps(binding.fan_power_scales),
            "active_fan_mask": ";".join("1" if value else "0" for value in binding.active_fan_mask),
            "updraft_amplitude_scale": float(binding.updraft_amplitude_scale),
            "updraft_width_scale": float(binding.updraft_width_scale),
            "local_uncertainty_scale": float(binding.local_uncertainty_scale),
            "wind_mode": wind_mode,
        },
    )


def runtime_config_from_manifest(manifest: dict[str, Any], *, library_tier: str) -> FlightRuntimeConfig:
    config = dict(manifest.get("config", {})) if isinstance(manifest.get("config", {}), dict) else {}
    selected_tier = str(library_tier or config.get("library_tier") or DEFAULT_LIBRARY_TIER)
    return FlightRuntimeConfig(
        run_label=str(config.get("run_label") or "sim_replay"),
        library_tier=selected_tier,
        controller_mode="closed_loop",
        experiment_case_id=str(config.get("experiment_case_id", "")),
        experiment_case_name=str(config.get("experiment_case_name", "")),
        experiment_memory_enabled=truthy(config.get("experiment_memory_enabled", False)),
        experiment_layout_id=str(config.get("experiment_layout_id", "")),
        throw_index=int(to_float(config.get("throw_index"), 0.0)),
        attempt_index=int(to_float(config.get("attempt_index"), 0.0)),
        governor_period_s=to_float(config.get("governor_period_s"), 0.1),
        serial_period_s=to_float(config.get("serial_period_s"), 0.02),
        launch_handoff_duration_s=to_float(config.get("launch_handoff_duration_s"), 0.04),
        max_duration_s=to_float(config.get("max_duration_s"), 20.0),
        surface_command_delay_s=to_float(config.get("surface_command_delay_s"), 0.0),
        actuator_tau_s=tuple_from_config(config.get("actuator_tau_s"), (0.06, 0.06, 0.06)),
        output_root=Path(str(config.get("output_root") or DEFAULT_RESULT_ROOT)),
        library_manifest_root=Path(
            str(
                config.get("library_manifest_root")
                or (FLIGHT_ROOT / "03_Frozen_Inputs" / "R8_library_size_study" / "E03" / "manifests")
            )
        ),
        outcome_table_path=Path(
            str(
                config.get("outcome_table_path")
                or (FLIGHT_ROOT / "03_Frozen_Inputs" / "R8_outcome" / "E03" / "metrics" / "outcome_model_table.csv")
            )
        ),
        controller_bundle_path=Path(
            str(
                config.get("controller_bundle_path")
                or (
                    FLIGHT_ROOT
                    / "03_Frozen_Inputs"
                    / "R5_dense"
                    / "E03"
                    / "manifests"
                    / "frozen_w01_controller_bundle.json"
                )
            )
        ),
        governor_config_path=Path(
            str(
                config.get("governor_config_path")
                or (
                    FLIGHT_ROOT
                    / "03_Frozen_Inputs"
                    / "R10_learn"
                    / "E03"
                    / "manifests"
                    / "frozen_governor_config_for_r11.json"
                )
            )
        ),
    )


def launch_state_for_throw(throw_root: Path) -> tuple[np.ndarray, float]:
    prelaunch_rows = read_csv(throw_root / "metrics" / "prelaunch_state_samples.csv")
    approved_rows = [
        row
        for row in prelaunch_rows
        if truthy(row.get("approved"))
        or truthy(row.get("interpolated_plane_approved"))
        or str(row.get("reason", "")).startswith("approved")
    ]
    if approved_rows:
        row = approved_rows[-1]
        return state_from_row(row), to_float(row.get("t_host_s"), 0.0)
    state_rows = read_csv(throw_root / "metrics" / "state_samples.csv")
    if not state_rows:
        raise RuntimeError(f"No state rows found for {throw_root}")
    return state_from_row(state_rows[0]), to_float(state_rows[0].get("t_host_s"), 0.0)


def measured_trace(
    throw_root: Path,
    *,
    launch_state: np.ndarray,
    launch_host_t_s: float,
) -> list[dict[str, Any]]:
    trace = [trace_row_from_state(0.0, launch_state, model_key="real", source="approved_launch_gate")]
    for row in read_csv(throw_root / "metrics" / "state_samples.csv"):
        state = state_from_row(row)
        t_s = max(0.0, to_float(row.get("t_host_s"), launch_host_t_s) - float(launch_host_t_s))
        out = trace_row_from_state(t_s, state, model_key="real", source="state_samples")
        out["exit_gate_reason"] = str(row.get("exit_gate_reason", ""))
        trace.append(out)
    return sorted(trace, key=lambda item: float(item["t_s"]))


def measured_handoff_state_and_trace(
    real_trace: list[dict[str, Any]],
    *,
    handoff_s: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if not real_trace:
        raise RuntimeError("Cannot build handoff splice from an empty real trace.")
    handoff = max(0.0, float(handoff_s))
    prefix = [dict(row) for row in real_trace if to_float(row.get("t_s")) <= handoff + 1e-12]
    handoff_state = interpolate_trace_state(real_trace, handoff)
    if handoff_state is None:
        handoff_state = state_dict_from_trace_row(prefix[-1] if prefix else real_trace[0])
    if not prefix or abs(to_float(prefix[-1].get("t_s")) - handoff) > 1e-12:
        prefix.append(
            {
                "model": "real",
                "source": "measured_handoff_interpolation",
                "t_s": handoff,
                **handoff_state,
                "exit_gate_inside": True,
                "exit_gate_reason": "inside_operational_region",
            }
        )
    state = np.asarray([handoff_state[name] for name in STATE_NAMES], dtype=float).reshape(STATE_SIZE)
    return as_state_vector(state), sorted(prefix, key=lambda item: float(item["t_s"]))


def relabel_trace(trace: list[dict[str, Any]], *, model_key: str, source: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in trace:
        copied = dict(row)
        copied["model"] = model_key
        copied["source"] = source
        out.append(copied)
    return out


def simulate_self_governor_reply(
    *,
    launch_state: np.ndarray,
    handoff_state: np.ndarray,
    handoff_trace: list[dict[str, Any]],
    config: FlightRuntimeConfig,
    duration_s: float,
    replay_dt_s: float,
    replay_environment: ReplayEnvironment,
) -> ReplayResult:
    controller = FrozenFlightController(config)
    aircraft = adapt_glider(build_nausicaa_glider())
    neutral = normalised_command_to_surface_rad(np.zeros(3, dtype=float))
    controller.prepare_launch_handoff_decision(as_state_vector(launch_state), primitive_step_index=0)
    x = as_state_vector(handoff_state)
    handoff_s = float(config.launch_handoff_duration_s)
    t_s = handoff_s
    governor_period_s = float(config.governor_period_s)
    serial_period_s = float(config.serial_period_s)
    current_command = neutral.copy()
    command_buffer = CommandDelayBuffer(
        delay_s=float(config.surface_command_delay_s),
        neutral_command_rad=neutral,
    )
    next_serial_s = handoff_s + serial_period_s
    next_governor_s = handoff_s + governor_period_s
    primitive_step_index = 0
    selected_variant_ids: list[str] = []
    trace = relabel_trace(
        handoff_trace,
        model_key="sim_self_governor",
        source="measured_0p04s_handoff_splice",
    )
    status = "ok"
    termination_reason = ""

    decision = controller.commit_prepared_launch_handoff_decision(x)
    current_command = command_from_decision(decision)
    command_buffer.record(t_s, current_command)
    if decision.selected:
        selected_variant_ids.append(decision.primitive_variant_id)
    primitive_step_index = 1
    prepare_next_self_decision(
        controller=controller,
        state=x,
        primitive_step_index=primitive_step_index,
        loop_elapsed_s=t_s,
        target_boundary_s=next_governor_s,
    )

    while t_s < float(duration_s) - 1e-12:
        if t_s >= next_governor_s - 1e-12:
            decision = controller.commit_prepared_continuation_decision(
                x,
                primitive_step_index=primitive_step_index,
            )
            current_command = command_from_decision(decision)
            command_buffer.record(t_s, current_command)
            if decision.selected:
                selected_variant_ids.append(decision.primitive_variant_id)
            primitive_step_index += 1
            next_governor_s += governor_period_s
            next_serial_s = t_s + serial_period_s
            prepare_next_self_decision(
                controller=controller,
                state=x,
                primitive_step_index=primitive_step_index,
                loop_elapsed_s=t_s,
                target_boundary_s=next_governor_s,
            )

        if t_s >= next_serial_s - 1e-12:
            current_command = active_payload_command(controller, x, fallback=current_command)
            command_buffer.record(t_s, current_command)
            next_serial_s += serial_period_s

        next_event_s = min(value for value in (next_governor_s, next_serial_s) if value > t_s + 1e-12)
        dt_s = min(float(replay_dt_s), float(duration_s) - t_s, next_event_s - t_s)
        x = rk4_step(
            x,
            command_buffer.command_at(t_s),
            aircraft=aircraft,
            actuator_tau_s=config.actuator_tau_s,
            dt_s=dt_s,
            wind_model=replay_environment.wind_model,
            wind_mode=replay_environment.wind_mode,
        )
        t_s += dt_s
        trace.append(trace_row_from_state(t_s, x, model_key="sim_self_governor", source="simulation"))
        exit_status = evaluate_exit_gate(x)
        if not exit_status.inside:
            termination_reason = str(exit_status.reason)
            break
        if not np.all(np.isfinite(x)):
            status = "nonfinite_state"
            break
    return ReplayResult(trace=trace, status=status, termination_reason=termination_reason, selected_variant_ids=selected_variant_ids)


def prepare_next_self_decision(
    *,
    controller: FrozenFlightController,
    state: np.ndarray,
    primitive_step_index: int,
    loop_elapsed_s: float,
    target_boundary_s: float,
) -> None:
    prediction_dt_s = max(0.0, float(target_boundary_s) - float(loop_elapsed_s))
    predicted = predict_boundary_state(state, prediction_dt_s)
    controller.prepare_continuation_decision(
        predicted,
        primitive_step_index=int(primitive_step_index),
        target_boundary_s=float(target_boundary_s),
        prepare_started_elapsed_s=float(loop_elapsed_s),
        prediction_dt_s=float(prediction_dt_s),
    )


def simulate_real_decision_reply(
    handoff_state: np.ndarray,
    throw_root: Path,
    *,
    handoff_trace: list[dict[str, Any]],
    config: FlightRuntimeConfig,
    duration_s: float,
    replay_dt_s: float,
    timing: str,
    replay_environment: ReplayEnvironment,
) -> ReplayResult:
    controller = FrozenFlightController(config)
    aircraft = adapt_glider(build_nausicaa_glider())
    neutral = normalised_command_to_surface_rad(np.zeros(3, dtype=float))
    decisions = real_decision_schedule(
        throw_root,
        launch_handoff_s=float(config.launch_handoff_duration_s),
        governor_period_s=float(config.governor_period_s),
        timing=timing,
    )
    x = as_state_vector(handoff_state)
    t_s = float(config.launch_handoff_duration_s)
    current_command = neutral.copy()
    command_buffer = CommandDelayBuffer(
        delay_s=float(config.surface_command_delay_s),
        neutral_command_rad=neutral,
    )
    next_serial_s = t_s + float(config.serial_period_s)
    schedule_index = 0
    active_variant_ids: list[str] = []
    trace = relabel_trace(
        handoff_trace,
        model_key="sim_real_decisions",
        source="measured_0p04s_handoff_splice",
    )
    status = "ok"
    termination_reason = ""

    while t_s < float(duration_s) - 1e-12:
        while schedule_index < len(decisions) and t_s >= float(decisions[schedule_index]["start_s"]) - 1e-12:
            variant_id = str(decisions[schedule_index].get("primitive_variant_id", ""))
            payload = controller.controllers.get(variant_id) if variant_id else None
            if payload is None:
                controller._active_payload = None
                controller._active_variant_id = ""
                current_command = neutral.copy()
                command_buffer.record(t_s, current_command)
            else:
                controller._active_payload = payload
                controller._active_variant_id = variant_id
                current_command = active_payload_command(controller, x, fallback=neutral)
                command_buffer.record(t_s, current_command)
                active_variant_ids.append(variant_id)
            next_serial_s = t_s + float(config.serial_period_s)
            schedule_index += 1

        if controller._active_payload is not None and t_s >= next_serial_s - 1e-12:
            current_command = active_payload_command(controller, x, fallback=current_command)
            command_buffer.record(t_s, current_command)
            next_serial_s += float(config.serial_period_s)

        next_decision_s = (
            float(decisions[schedule_index]["start_s"])
            if schedule_index < len(decisions)
            else float("inf")
        )
        next_event_s = min(value for value in (next_decision_s, next_serial_s) if value > t_s + 1e-12)
        dt_s = min(float(replay_dt_s), float(duration_s) - t_s, next_event_s - t_s)
        x = rk4_step(
            x,
            command_buffer.command_at(t_s),
            aircraft=aircraft,
            actuator_tau_s=config.actuator_tau_s,
            dt_s=dt_s,
            wind_model=replay_environment.wind_model,
            wind_mode=replay_environment.wind_mode,
        )
        t_s += dt_s
        trace.append(trace_row_from_state(t_s, x, model_key="sim_real_decisions", source="simulation"))
        exit_status = evaluate_exit_gate(x)
        if not exit_status.inside:
            termination_reason = str(exit_status.reason)
            break
        if not np.all(np.isfinite(x)):
            status = "nonfinite_state"
            break
    return ReplayResult(trace=trace, status=status, termination_reason=termination_reason, selected_variant_ids=active_variant_ids)


def real_decision_schedule(
    throw_root: Path,
    *,
    launch_handoff_s: float,
    governor_period_s: float,
    timing: str,
) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for row in read_csv(throw_root / "metrics" / "controller_decisions.csv"):
        step = int(to_float(row.get("executed_primitive_step_index"), 0.0))
        if not truthy(row.get("selected")):
            variant_id = ""
        else:
            variant_id = str(row.get("primitive_variant_id", ""))
        boundary_s = to_float(row.get("scheduler_target_boundary_s"), launch_handoff_s + step * governor_period_s)
        logged_s = to_float(row.get("decision_elapsed_s"), boundary_s)
        if step == 0:
            start_s = max(float(launch_handoff_s), boundary_s)
        elif timing == "logged":
            start_s = max(float(launch_handoff_s), logged_s)
        else:
            start_s = max(float(launch_handoff_s), boundary_s)
        decisions.append(
            {
                "start_s": float(start_s),
                "primitive_step_index": step,
                "primitive_variant_id": variant_id,
            }
        )
    return sorted(decisions, key=lambda item: (float(item["start_s"]), int(item["primitive_step_index"])))


def selected_variant_ids_from_decisions(throw_root: Path) -> list[str]:
    return [
        str(row.get("primitive_variant_id", ""))
        for row in read_csv(throw_root / "metrics" / "controller_decisions.csv")
        if truthy(row.get("selected")) and str(row.get("primitive_variant_id", ""))
    ]


def command_from_decision(decision: object) -> np.ndarray:
    try:
        return np.asarray(getattr(decision, "command_rad"), dtype=float).reshape(3)
    except Exception:
        return normalised_command_to_surface_rad(np.zeros(3, dtype=float))


def active_payload_command(
    controller: FrozenFlightController,
    state: np.ndarray,
    *,
    fallback: np.ndarray,
) -> np.ndarray:
    payload = getattr(controller, "_active_payload", None)
    if payload is None:
        return np.asarray(fallback, dtype=float).reshape(3)
    try:
        _, command_rad = controller._command_for_payload(payload, as_state_vector(state))
    except Exception:
        return np.asarray(fallback, dtype=float).reshape(3)
    return np.asarray(command_rad, dtype=float).reshape(3)


def rk4_step(
    x: np.ndarray,
    command_rad: np.ndarray,
    *,
    aircraft: object,
    actuator_tau_s: tuple[float, float, float],
    dt_s: float,
    wind_model: object | None,
    wind_mode: str,
) -> np.ndarray:
    dt = float(dt_s)
    command = np.asarray(command_rad, dtype=float).reshape(3)
    wind_mode_value = str(wind_mode or "none") if wind_model is not None else "none"
    k1 = state_derivative(
        x,
        command,
        aircraft,
        wind_model=wind_model,
        actuator_tau_s=actuator_tau_s,
        wind_mode=wind_mode_value,
    )
    k2 = state_derivative(
        x + 0.5 * dt * k1,
        command,
        aircraft,
        wind_model=wind_model,
        actuator_tau_s=actuator_tau_s,
        wind_mode=wind_mode_value,
    )
    k3 = state_derivative(
        x + 0.5 * dt * k2,
        command,
        aircraft,
        wind_model=wind_model,
        actuator_tau_s=actuator_tau_s,
        wind_mode=wind_mode_value,
    )
    k4 = state_derivative(
        x + dt * k3,
        command,
        aircraft,
        wind_model=wind_model,
        actuator_tau_s=actuator_tau_s,
        wind_mode=wind_mode_value,
    )
    return as_state_vector(x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))


def predict_boundary_state(state_vector: np.ndarray, dt_s: float) -> np.ndarray:
    """Match the runtime's short-horizon governor-selection predictor."""

    state = as_state_vector(state_vector)
    dt = max(0.0, min(float(dt_s), 0.25))
    if dt <= 0.0:
        return state
    phi = float(state[STATE_INDEX["phi"]])
    theta = float(state[STATE_INDEX["theta"]])
    psi = float(state[STATE_INDEX["psi"]])
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    c_theta, s_theta = np.cos(theta), np.sin(theta)
    c_psi, s_psi = np.cos(psi), np.sin(psi)
    c_wb = np.asarray(
        [
            [
                c_theta * c_psi,
                s_phi * s_theta * c_psi - c_phi * s_psi,
                c_phi * s_theta * c_psi + s_phi * s_psi,
            ],
            [
                c_theta * s_psi,
                s_phi * s_theta * s_psi + c_phi * c_psi,
                c_phi * s_theta * s_psi - s_phi * c_psi,
            ],
            [-s_theta, s_phi * c_theta, c_phi * c_theta],
        ],
        dtype=float,
    )
    body_velocity = state[[STATE_INDEX["u"], STATE_INDEX["v"], STATE_INDEX["w"]]]
    world_internal_velocity = c_wb @ body_velocity
    state[STATE_INDEX["x_w"]] += float(world_internal_velocity[0]) * dt
    state[STATE_INDEX["y_w"]] += float(world_internal_velocity[1]) * dt
    state[STATE_INDEX["z_w"]] -= float(world_internal_velocity[2]) * dt
    state[STATE_INDEX["phi"]] += float(state[STATE_INDEX["p"]]) * dt
    state[STATE_INDEX["theta"]] += float(state[STATE_INDEX["q"]]) * dt
    state[STATE_INDEX["psi"]] = wrap_to_pi(float(state[STATE_INDEX["psi"]]) + float(state[STATE_INDEX["r"]]) * dt)
    return state


def plot_throw_replay(
    throw: ThrowCase,
    traces: dict[str, list[dict[str, Any]]],
    *,
    output_path: Path,
    replay_environment: ReplayEnvironment,
) -> dict[str, Any]:
    fig = plt.figure(figsize=(15.4, 8.8))
    fig.patch.set_facecolor("white")
    grid = fig.add_gridspec(
        3,
        3,
        left=0.045,
        right=0.985,
        bottom=0.145,
        top=0.91,
        wspace=0.28,
        hspace=0.34,
        width_ratios=(1.45, 1.0, 1.0),
    )
    ax3d = fig.add_subplot(grid[:, 0], projection="3d")
    axes = [
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[1, 2]),
        fig.add_subplot(grid[2, 1]),
        fig.add_subplot(grid[2, 2]),
    ]
    configure_r11_style_axis(ax3d)
    updraft_meta = draw_replay_updraft_context(ax3d, replay_environment=replay_environment)

    for model_key in TRACE_ORDER:
        trace = traces.get(model_key, [])
        if not trace:
            continue
        x = trace_values(trace, "x_w")
        y = trace_values(trace, "y_w")
        z = trace_values(trace, "z_w")
        ax3d.plot(
            x,
            y,
            z,
            color=TRACE_COLORS[model_key],
            linestyle=TRACE_STYLES[model_key],
            linewidth=TRACE_LINEWIDTHS[model_key],
            label=TRACE_LABELS[model_key],
            zorder=20 if model_key == "real" else 18,
        )
        ax3d.scatter(x[:1], y[:1], z[:1], color=TRACE_COLORS[model_key], s=22, depthshade=False, zorder=22)
        ax3d.scatter(x[-1:], y[-1:], z[-1:], color=TRACE_COLORS[model_key], marker="x", s=42, depthshade=False, zorder=23)

    for axis, (field, ylabel, scale) in zip(axes, TIME_SERIES_SPECS, strict=True):
        for model_key in TRACE_ORDER:
            trace = traces.get(model_key, [])
            if not trace:
                continue
            axis.plot(
                trace_values(trace, "t_s"),
                trace_values(trace, field) * float(scale),
                color=TRACE_COLORS[model_key],
                linestyle=TRACE_STYLES[model_key],
                linewidth=2.0 if model_key == "real" else 1.6,
                label=TRACE_LABELS[model_key],
            )
        axis.set_xlabel("time (s)")
        axis.set_ylabel(ylabel)
        axis.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.8)

    ax3d.legend(loc="upper left", frameon=False, fontsize=9.0)
    axes[0].legend(loc="best", frameon=False, fontsize=8.5)
    fig.suptitle(
        f"Real flight and simulation replies: {throw.case_id} / {throw.session_id} / {throw.throw_id}",
        fontsize=13,
    )
    fig.text(
        0.046,
        0.028,
        launch_condition_text(throw, traces),
        ha="left",
        va="bottom",
        fontsize=8.2,
        family="monospace",
        color="#202020",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#bdbdbd",
            "linewidth": 0.7,
            "alpha": 0.94,
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, facecolor="white")
    plt.close(fig)
    return updraft_meta


def draw_replay_updraft_context(ax, *, replay_environment: ReplayEnvironment) -> dict[str, Any]:
    fan_positions = replay_environment.fan_positions_m
    wind = replay_environment.wind_model
    if wind is None:
        draw_fan_positions(ax, fan_positions)
        return {
            "updraft_plot_status": "dry_air_or_no_wind_field",
            "updraft_max_m_s": 0.0,
            "updraft_iso_surface_count": 0,
        }
    try:
        config = R9PreflightFigureConfig()
        x_vec, y_vec, z_vec, w_grid = _sample_updraft_volume(wind, config)
        w_max = float(np.nanmax(w_grid)) if w_grid.size else 0.0
        if w_max <= 1e-9:
            draw_fan_positions(ax, fan_positions)
            return {
                "updraft_plot_status": "zero_updraft_volume",
                "updraft_max_m_s": 0.0,
                "updraft_grid_nx": int(config.updraft_nx),
                "updraft_grid_ny": int(config.updraft_ny),
                "updraft_grid_nz": int(config.updraft_nz),
                "updraft_iso_surface_count": 0,
            }
        cmap_alpha = _build_alpha_cmap()
        norm = matplotlib.colors.Normalize(vmin=0.0, vmax=8.0, clip=True)
        _draw_center_slices(
            ax,
            x_vec=x_vec,
            y_vec=y_vec,
            z_vec=z_vec,
            w_grid=w_grid,
            fan_positions=fan_positions,
            cmap_alpha=cmap_alpha,
            norm=norm,
        )
        iso_count = _draw_updraft_isosurfaces(
            ax,
            x_vec=x_vec,
            y_vec=y_vec,
            z_vec=z_vec,
            w_grid=w_grid,
            cmap_alpha=cmap_alpha,
            norm=norm,
        )
        draw_fan_positions(ax, fan_positions)
        return {
            "updraft_plot_status": "w2_annular_gp_3d_slices_and_isosurfaces",
            "updraft_max_m_s": w_max,
            "updraft_grid_nx": int(config.updraft_nx),
            "updraft_grid_ny": int(config.updraft_ny),
            "updraft_grid_nz": int(config.updraft_nz),
            "updraft_iso_surface_count": int(iso_count),
        }
    except Exception as exc:
        draw_fan_positions(ax, fan_positions)
        return {
            "updraft_plot_status": f"blocked:{type(exc).__name__}",
            "updraft_max_m_s": 0.0,
            "updraft_iso_surface_count": 0,
        }


def launch_condition_text(throw: ThrowCase, traces: dict[str, list[dict[str, Any]]]) -> str:
    real_trace = traces.get("real", [])
    launch = real_trace[0] if real_trace else {}
    speed = to_float(throw.summary.get("launch_speed_m_s"))
    if not math.isfinite(speed):
        speed = math.sqrt(
            sum(to_float(launch.get(name), 0.0) ** 2 for name in ("u", "v", "w"))
        )
    handoff_ms = 1000.0 * to_float(throw.summary.get("launch_handoff_duration_s"), FIRST_WINDOW_STATE_AUDIT_S)
    return (
        "Launch condition: "
        f"x/y/z={fmt_launch_value(launch.get('x_w'))}/"
        f"{fmt_launch_value(launch.get('y_w'))}/"
        f"{fmt_launch_value(launch.get('z_w'))} m, "
        f"V={fmt_launch_value(speed)} m/s, handoff={fmt_launch_value(handoff_ms, digits=0)} ms\n"
        "attitude deg: "
        f"phi/theta/psi={fmt_launch_value(rad_to_deg(launch.get('phi')), digits=1)}/"
        f"{fmt_launch_value(rad_to_deg(launch.get('theta')), digits=1)}/"
        f"{fmt_launch_value(rad_to_deg(launch.get('psi')), digits=1)}; "
        "body u/v/w="
        f"{fmt_launch_value(launch.get('u'))}/"
        f"{fmt_launch_value(launch.get('v'))}/"
        f"{fmt_launch_value(launch.get('w'))} m/s; "
        "p/q/r="
        f"{fmt_launch_value(launch.get('p'))}/"
        f"{fmt_launch_value(launch.get('q'))}/"
        f"{fmt_launch_value(launch.get('r'))} rad/s"
    )


def rad_to_deg(value: object) -> float:
    return to_float(value) * 180.0 / math.pi


def fmt_launch_value(value: object, *, digits: int = 2) -> str:
    numeric = to_float(value)
    if not math.isfinite(numeric):
        return "n/a"
    return f"{numeric:.{int(digits)}f}"


def configure_r11_style_axis(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(True)
    _draw_arena_box(ax)
    _draw_floor_grid(ax)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_zlim(Z_MIN, Z_MAX)
    ax.set_xticks(np.arange(X_MIN, X_MAX + 1e-9, 1.4))
    ax.set_yticks(np.arange(Y_MIN, Y_MAX + 1e-9, 1.2))
    ax.set_zticks(np.arange(Z_MIN, Z_MAX + 1e-9, 0.7))
    ax.set_xlabel("$x$ (m)", labelpad=14)
    ax.set_ylabel("$y$ (m)", labelpad=9)
    ax.set_zlabel("$z$ (m)", labelpad=5, rotation=90)
    ax.zaxis.set_rotate_label(False)
    ax.tick_params(axis="x", which="major", labelsize=TICK_LABEL_FONTSIZE)
    ax.tick_params(axis="y", which="major", labelsize=TICK_LABEL_FONTSIZE)
    ax.tick_params(axis="z", which="major", labelsize=TICK_LABEL_FONTSIZE)
    for label in ax.get_xticklabels():
        label.set_rotation(-20)
    for label in ax.get_yticklabels():
        label.set_rotation(20)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        axis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 1.0)
        axis._axinfo["grid"]["linewidth"] = 0.4
    try:
        ax.set_box_aspect((X_MAX - X_MIN, Y_MAX - Y_MIN, Z_MAX - Z_MIN))
    except AttributeError:
        pass
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    ax.xaxis.line.set_linewidth(AXIS_EDGE_LW)
    ax.yaxis.line.set_linewidth(AXIS_EDGE_LW)
    ax.zaxis.line.set_linewidth(AXIS_EDGE_LW)


def draw_fan_positions(ax, fan_positions: tuple[tuple[float, float], ...]) -> None:
    if not fan_positions:
        return
    theta = np.linspace(0.0, 2.0 * np.pi, 160)
    for index, (fx, fy) in enumerate(fan_positions):
        ax.plot(
            float(fx) + 0.5 * FAN_OUTLET_DIAMETER * np.cos(theta),
            float(fy) + 0.5 * FAN_OUTLET_DIAMETER * np.sin(theta),
            FAN_OUTLET_PLOT_Z_M * np.ones_like(theta),
            color=(0.0, 0.0, 0.0, FAN_OUTLET_ALPHA),
            linewidth=FAN_OUTLET_EDGE_LW,
            linestyle=FAN_OUTLET_DASH,
            label="Fan outlet" if index == 0 else None,
            zorder=7,
        )


def fan_positions_from_log(throw_root: Path) -> tuple[tuple[float, float], ...]:
    rows = read_csv(throw_root / "metrics" / "fan_positions.csv")
    positions: list[tuple[float, float]] = []
    seen: set[str] = set()
    for row in reversed(rows):
        if not truthy(row.get("visible")):
            continue
        subject = str(row.get("fan_subject", ""))
        if subject in seen:
            continue
        x = to_float(row.get("x_w"))
        y = to_float(row.get("y_w"))
        if math.isfinite(x) and math.isfinite(y):
            positions.append((x, y))
            seen.add(subject)
    return tuple(reversed(positions))


def trace_row_from_state(t_s: float, state: np.ndarray, *, model_key: str, source: str) -> dict[str, Any]:
    x = as_state_vector(state)
    row: dict[str, Any] = {
        "model": model_key,
        "source": source,
        "t_s": float(t_s),
    }
    for name, value in zip(STATE_NAMES, x, strict=True):
        row[name] = float(value)
    exit_status = evaluate_exit_gate(x)
    row["exit_gate_inside"] = bool(exit_status.inside)
    row["exit_gate_reason"] = str(exit_status.reason)
    return row


def state_from_row(row: dict[str, Any]) -> np.ndarray:
    values = [to_float(row.get(name)) for name in STATE_NAMES]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("state row contains nonfinite canonical state values")
    return as_state_vector(np.asarray(values, dtype=float).reshape(STATE_SIZE))


def trace_values(trace: list[dict[str, Any]], field: str) -> np.ndarray:
    return np.asarray([to_float(row.get(field)) for row in trace], dtype=float)


def summary_row(
    *,
    throw: ThrowCase,
    model_key: str,
    replay: ReplayResult,
    figure_path: Path,
    trace_path: Path,
) -> dict[str, Any]:
    trace = replay.trace
    first = trace[0] if trace else {}
    last = trace[-1] if trace else {}
    return {
        "figure_run_version": FIGURE_RUN_VERSION,
        "case_id": throw.case_id,
        "session_id": throw.session_id,
        "throw_id": throw.throw_id,
        "model": model_key,
        "model_label": TRACE_LABELS.get(model_key, model_key),
        "status": replay.status,
        "termination_reason": replay.termination_reason or str(last.get("exit_gate_reason", "")),
        "sample_count": len(trace),
        "duration_s": to_float(last.get("t_s"), 0.0),
        "x0_m": to_float(first.get("x_w")),
        "y0_m": to_float(first.get("y_w")),
        "z0_m": to_float(first.get("z_w")),
        "xf_m": to_float(last.get("x_w")),
        "yf_m": to_float(last.get("y_w")),
        "zf_m": to_float(last.get("z_w")),
        "final_dx_m": to_float(last.get("x_w")) - to_float(first.get("x_w")) if first and last else float("nan"),
        "final_dy_m": to_float(last.get("y_w")) - to_float(first.get("y_w")) if first and last else float("nan"),
        "altitude_loss_m": to_float(first.get("z_w")) - to_float(last.get("z_w")) if first and last else float("nan"),
        "selected_primitive_count": len(replay.selected_variant_ids),
        "selected_primitive_variant_ids": "|".join(replay.selected_variant_ids),
        "throw_root": throw.throw_root.as_posix(),
        "figure_path": figure_path.as_posix(),
        "trace_path": trace_path.as_posix(),
    }


def first_window_state_error_rows(
    *,
    throw: ThrowCase,
    traces: dict[str, list[dict[str, Any]]],
    window_s: float,
) -> list[dict[str, Any]]:
    real_trace = traces.get("real", [])
    if not real_trace:
        return []
    real_base = state_dict_from_trace_row(real_trace[0])
    rows: list[dict[str, Any]] = []
    for model_key in ("sim_self_governor", "sim_real_decisions"):
        sim_trace = traces.get(model_key, [])
        if not sim_trace:
            continue
        sim_base = state_dict_from_trace_row(sim_trace[0])
        residuals_by_state: dict[str, list[float]] = {name: [] for name in STATE_NAMES}
        for real_row in real_trace:
            t_s = to_float(real_row.get("t_s"))
            if not math.isfinite(t_s) or t_s < -1e-12 or t_s > float(window_s) + 1e-12:
                continue
            sim_state = interpolate_trace_state(sim_trace, t_s)
            if sim_state is None:
                continue
            real_state = state_dict_from_trace_row(real_row)
            for name in STATE_NAMES:
                residuals_by_state[name].append(
                    sysid_style_state_residual(
                        name,
                        actual=real_state,
                        simulated=sim_state,
                        actual_base=real_base,
                        simulated_base=sim_base,
                    )
                )
        for name, values in residuals_by_state.items():
            finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
            if finite.size == 0:
                continue
            rows.append(
                {
                    "figure_run_version": FIGURE_RUN_VERSION,
                    "case_id": throw.case_id,
                    "session_id": throw.session_id,
                    "throw_id": throw.throw_id,
                    "model": model_key,
                    "model_label": TRACE_LABELS.get(model_key, model_key),
                    "window_s": float(window_s),
                    "state_component": name,
                    "residual_policy": state_residual_policy(name),
                    "unit": state_residual_unit(name),
                    "sample_count": int(finite.size),
                    "mae": float(np.mean(np.abs(finite))),
                    "rmse": float(math.sqrt(np.mean(finite * finite))),
                    "max_abs": float(np.max(np.abs(finite))),
                    "final_residual": float(finite[-1]),
                }
            )
    return rows


def state_dict_from_trace_row(row: dict[str, Any]) -> dict[str, float]:
    return {name: to_float(row.get(name)) for name in STATE_NAMES}


def interpolate_trace_state(trace: list[dict[str, Any]], t_s: float) -> dict[str, float] | None:
    if not trace:
        return None
    times = np.asarray([to_float(row.get("t_s")) for row in trace], dtype=float)
    if not np.all(np.isfinite(times)):
        return None
    query = float(t_s)
    if query < float(times[0]) - 1e-12 or query > float(times[-1]) + 1e-12:
        return None
    out: dict[str, float] = {}
    for name in STATE_NAMES:
        values = np.asarray([to_float(row.get(name)) for row in trace], dtype=float)
        if not np.all(np.isfinite(values)):
            return None
        out[name] = float(np.interp(query, times, values))
    return out


def sysid_style_state_residual(
    name: str,
    *,
    actual: dict[str, float],
    simulated: dict[str, float],
    actual_base: dict[str, float],
    simulated_base: dict[str, float],
) -> float:
    if name == "x_w":
        return (actual[name] - actual_base[name]) - (simulated[name] - simulated_base[name])
    if name == "y_w":
        return (actual[name] - actual_base[name]) - (simulated[name] - simulated_base[name])
    if name == "z_w":
        actual_altitude_loss = actual_base[name] - actual[name]
        simulated_altitude_loss = simulated_base[name] - simulated[name]
        return actual_altitude_loss - simulated_altitude_loss
    if name in ANGLE_STATE_NAMES:
        return math.degrees(wrap_to_pi(actual[name] - simulated[name]))
    return actual[name] - simulated[name]


def state_residual_policy(name: str) -> str:
    if name in {"x_w", "y_w"}:
        return "relative_displacement_actual_minus_sim"
    if name == "z_w":
        return "altitude_loss_actual_minus_sim"
    if name in ANGLE_STATE_NAMES:
        return "wrapped_angle_actual_minus_sim"
    return "direct_state_actual_minus_sim"


def state_residual_unit(name: str) -> str:
    if name in RELATIVE_POSITION_STATE_NAMES:
        return "m"
    if name in ANGLE_STATE_NAMES:
        return "deg"
    if name in {"p", "q", "r"}:
        return "rad_s"
    if name in {"delta_a", "delta_e", "delta_r"}:
        return "rad"
    return "m_s"


def execution_timing_row(throw: ThrowCase) -> dict[str, Any]:
    decisions = read_csv(throw.throw_root / "metrics" / "controller_decisions.csv")
    selected = [row for row in decisions if truthy(row.get("selected"))]
    continuation = [
        row
        for row in decisions
        if int(to_float(row.get("executed_primitive_step_index"), -1.0)) > 0
    ]
    decision_times = finite_values([to_float(row.get("decision_time_s")) for row in decisions])
    continuation_decision_times = finite_values([to_float(row.get("decision_time_s")) for row in continuation])
    continuation_commit_lags = finite_values([to_float(row.get("scheduler_commit_lag_s")) for row in continuation])
    prepared_before_count = sum(truthy(row.get("scheduler_prepared_before_primitive_boundary")) for row in continuation)
    source_counts: dict[str, int] = {}
    for row in decisions:
        source = str(row.get("scheduler_decision_source", ""))
        source_counts[source] = source_counts.get(source, 0) + 1

    handoff_s = to_float(throw.summary.get("launch_handoff_duration_s"))
    first_active_s = to_float(throw.summary.get("first_active_command_elapsed_s"))
    first_active_lag_s = (
        first_active_s - handoff_s if math.isfinite(first_active_s) and math.isfinite(handoff_s) else float("nan")
    )
    return {
        "figure_run_version": FIGURE_RUN_VERSION,
        "case_id": throw.case_id,
        "session_id": throw.session_id,
        "throw_id": throw.throw_id,
        "valid_throw": truthy(throw.summary.get("valid_throw")),
        "termination_reason": str(throw.summary.get("termination_reason", "")),
        "launch_speed_m_s": to_float(throw.summary.get("launch_speed_m_s")),
        "launch_handoff_duration_s": handoff_s,
        "first_active_command_elapsed_s": first_active_s,
        "first_active_command_lag_after_handoff_s": first_active_lag_s,
        "controller_decision_count": int(to_float(throw.summary.get("controller_decision_count"), len(decisions))),
        "selected_decision_count": len(selected),
        "continuation_decision_count": len(continuation),
        "max_decision_time_s_summary": to_float(throw.summary.get("max_decision_time_s")),
        "max_decision_time_s_logged": max_or_nan(decision_times),
        "p95_decision_time_s_logged": percentile_or_nan(decision_times, 95.0),
        "mean_decision_time_s_logged": mean_or_nan(decision_times),
        "max_continuation_decision_time_s": max_or_nan(continuation_decision_times),
        "p95_continuation_decision_time_s": percentile_or_nan(continuation_decision_times, 95.0),
        "mean_continuation_decision_time_s": mean_or_nan(continuation_decision_times),
        "continuation_prepared_before_boundary_count": prepared_before_count,
        "continuation_prepared_before_boundary_rate": (
            prepared_before_count / len(continuation) if continuation else float("nan")
        ),
        "max_continuation_commit_lag_s": max_or_nan(continuation_commit_lags),
        "p95_continuation_commit_lag_s": percentile_or_nan(continuation_commit_lags, 95.0),
        "mean_continuation_commit_lag_s": mean_or_nan(continuation_commit_lags),
        "continuation_commit_lag_gt_20ms_count": sum(value > 0.020 for value in continuation_commit_lags),
        "continuation_commit_lag_gt_50ms_count": sum(value > 0.050 for value in continuation_commit_lags),
        "continuation_late_decision_count_summary": int(
            to_float(throw.summary.get("continuation_late_decision_count"), 0.0)
        ),
        "slot_command_update_count": int(to_float(throw.summary.get("slot_command_update_count"), 0.0)),
        "packet_count": int(to_float(throw.summary.get("packet_count"), 0.0)),
        "active_metric_buffered_row_count": int(to_float(throw.summary.get("active_metric_buffered_row_count"), 0.0)),
        "active_metric_buffer_flush_count": int(to_float(throw.summary.get("active_metric_buffer_flush_count"), 0.0)),
        "scheduler_decision_source_counts_json": json.dumps(source_counts, sort_keys=True, separators=(",", ":")),
        "throw_root": throw.throw_root.as_posix(),
    }


def finite_values(values: list[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def max_or_nan(values: list[float]) -> float:
    return max(values) if values else float("nan")


def mean_or_nan(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float))) if values else float("nan")


def percentile_or_nan(values: list[float], percentile: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float), float(percentile)))


def write_replay_trace_csv(path: Path, traces: dict[str, list[dict[str, Any]]]) -> None:
    rows = [row for model_key in TRACE_ORDER for row in traces.get(model_key, [])]
    fields = [
        "model",
        "source",
        "t_s",
        *STATE_NAMES,
        "exit_gate_inside",
        "exit_gate_reason",
    ]
    write_csv(path, rows, fieldnames=fields)


def write_report(
    path: Path,
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    state_error_rows: list[dict[str, Any]],
    timing_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Real-Flight Simulation Replay Figures",
        "",
        f"- Figure run version: `{manifest['figure_run_version']}`",
        f"- Output root: `{manifest['output_root']}`",
        f"- Library tier: `{manifest['library_tier']}`",
        f"- Replay dt (s): `{float(manifest['replay_dt_s']):.4f}`",
        f"- Real-decision timing: `{manifest['real_decision_timing']}`",
        f"- Replay environment: `{manifest['replay_environment_summary_path']}`",
        f"- Replay environment policy: {manifest['replay_environment_policy']}",
        f"- First-window state audit: `{manifest['first_window_state_audit']['summary_path']}`",
        f"- Execution timing audit: `{manifest['execution_timing_audit']['summary_path']}`",
        "",
        "| case | session | throw | model | status | termination | duration s | final x m | final y m | final z m |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {case_id} | {session_id} | {throw_id} | {model_label} | {status} | {termination_reason} | "
            "{duration_s:.3f} | {xf_m:.3f} | {yf_m:.3f} | {zf_m:.3f} |".format(**row)
        )
    if state_error_rows:
        lines.extend(
            [
                "",
                f"## First {float(manifest['first_window_state_audit']['window_s']):.2f} s State Residual Audit",
                "",
                "| case | throw | model | largest position MAE m | largest attitude MAE deg | largest velocity/rate MAE | largest surface MAE rad |",
                "|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in compact_first_window_error_rows(state_error_rows):
            lines.append(
                "| {case_id} | {throw_id} | {model_label} | {largest_position_mae_m:.4f} | "
                "{largest_attitude_mae_deg:.4f} | {largest_velocity_or_rate_mae:.4f} | "
                "{largest_surface_mae_rad:.4f} |".format(**row)
            )
    if timing_rows:
        lines.extend(
            [
                "",
                "## Execution Timing Audit",
                "",
                "| case | throw | launch speed m/s | first active lag ms | max decision ms | p95 decision ms | max commit lag ms | >20 ms lags | >50 ms lags | runtime late decisions |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in timing_rows:
            lines.append(
                "| {case_id} | {throw_id} | {launch_speed_m_s:.3f} | {first_active_lag_ms:.2f} | "
                "{max_decision_ms:.2f} | {p95_decision_ms:.2f} | {max_commit_lag_ms:.2f} | "
                "{gt20} | {gt50} | {late_count} |".format(
                    case_id=row["case_id"],
                    throw_id=row["throw_id"],
                    launch_speed_m_s=to_float(row.get("launch_speed_m_s")),
                    first_active_lag_ms=1000.0 * to_float(row.get("first_active_command_lag_after_handoff_s")),
                    max_decision_ms=1000.0 * to_float(row.get("max_decision_time_s_logged")),
                    p95_decision_ms=1000.0 * to_float(row.get("p95_decision_time_s_logged")),
                    max_commit_lag_ms=1000.0 * to_float(row.get("max_continuation_commit_lag_s")),
                    gt20=int(to_float(row.get("continuation_commit_lag_gt_20ms_count"), 0.0)),
                    gt50=int(to_float(row.get("continuation_commit_lag_gt_50ms_count"), 0.0)),
                    late_count=int(to_float(row.get("continuation_late_decision_count_summary"), 0.0)),
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def compact_first_window_error_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("case_id", "")), str(row.get("throw_id", "")), str(row.get("model", "")))
        grouped.setdefault(key, []).append(row)
    compact: list[dict[str, Any]] = []
    for (case_id, throw_id, model), group in sorted(grouped.items()):
        by_state = {str(row.get("state_component", "")): row for row in group}
        compact.append(
            {
                "case_id": case_id,
                "throw_id": throw_id,
                "model": model,
                "model_label": TRACE_LABELS.get(model, model),
                "largest_position_mae_m": max_mae(by_state, ("x_w", "y_w", "z_w")),
                "largest_attitude_mae_deg": max_mae(by_state, ("phi", "theta", "psi")),
                "largest_velocity_or_rate_mae": max_mae(by_state, ("u", "v", "w", "p", "q", "r")),
                "largest_surface_mae_rad": max_mae(by_state, ("delta_a", "delta_e", "delta_r")),
            }
        )
    return compact


def max_mae(rows_by_state: dict[str, dict[str, Any]], names: tuple[str, ...]) -> float:
    values = [to_float(rows_by_state.get(name, {}).get("mae")) for name in names]
    finite = [value for value in values if math.isfinite(value)]
    return max(finite) if finite else float("nan")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not Path(path).exists():
        return []
    with Path(path).open("r", newline="", encoding="ascii") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames or sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    if not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text(encoding="ascii"))


def tuple_from_config(value: object, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        parsed = tuple(to_float(item) for item in value)
        if all(math.isfinite(item) and item > 0.0 for item in parsed):
            return parsed  # type: ignore[return-value]
    return default


def infer_case_id(throw_root: Path, result_root: Path) -> str:
    try:
        rel = throw_root.resolve().relative_to(result_root.resolve())
        return rel.parts[0] if rel.parts else throw_root.parent.parent.name
    except ValueError:
        return throw_root.parent.parent.name


def infer_session_id(throw_root: Path, result_root: Path) -> str:
    try:
        rel = throw_root.resolve().relative_to(result_root.resolve())
        return rel.parts[1] if len(rel.parts) >= 2 else throw_root.parent.name
    except ValueError:
        return throw_root.parent.name


def safe_name(value: str) -> str:
    out = "".join(char if char.isalnum() or char in "._-" else "_" for char in str(value).strip())
    return out or "unnamed"


def to_float(value: object, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "ok"}


def wrap_to_pi(value: float) -> float:
    return float((float(value) + np.pi) % (2.0 * np.pi) - np.pi)


if __name__ == "__main__":
    main()
