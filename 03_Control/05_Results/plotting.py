from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

try:
    from skimage.measure import marching_cubes
except ImportError:  # pragma: no cover - optional thermal-style plume surfaces
    marching_cubes = None

from plot_style import (
    ACTUAL_END_COLOR,
    ACTUAL_START_COLOR,
    CBAR_LABEL,
    CONTROL_REFERENCE_COLOR,
    DESIRED_COMMAND_ALPHA,
    DESIRED_COMMAND_LINEWIDTH,
    FAN_OUTLET_ALPHA,
    FAN_OUTLET_DASH,
    FAN_OUTLET_DIAMETER_M,
    FAN_OUTLET_EDGE_LW,
    FAN_OUTLET_Z_M,
    PLOT_VMAX,
    PLOT_VMIN,
    SIMULATION_FLOOR_COLOR,
    SIMULATION_FLOOR_LINESTYLE,
    SIMULATION_FLOOR_LINEWIDTH,
    TRAJECTORY_SPECS,
    TRUE_SAFETY_VOLUME_COLOR,
    TRUE_SAFETY_VOLUME_LINESTYLE,
    TRUE_SAFETY_VOLUME_LINEWIDTH,
    PlotStyle,
    build_alpha_cmap,
    framed_legend,
    save_figure,
    style_3d_axis,
    style_command_axis,
    style_geometry_axis,
    style_time_axis,
    thermal_cmap,
)


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Import path setup and plotting dataclasses
# 2) Scenario execution and reference rollout helpers
# 3) Log loading and geometry helpers
# 4) Composite figure builder
# 5) Public figure-generation API
# =============================================================================

# =============================================================================
# 1) Import Path Setup and Plotting Dataclasses
# =============================================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
for rel in (
    "03_Control/02_Inner_Loop",
    "03_Control/03_Primitives",
    "03_Control/04_Scenarios",
):
    path = REPO_ROOT / rel
    if str(path) not in sys.path:
        # Plot scripts run from the results folder without requiring package installation.
        sys.path.insert(0, str(path))

from arena import ArenaConfig, safe_bounds, tracker_bounds  # noqa: E402
from flight_dynamics import adapt_glider  # noqa: E402
from glider import build_nausicaa_glider  # noqa: E402
from latency import (  # noqa: E402
    AGGREGATE_LIMITS,
    CommandToSurfaceConfig,
    CommandToSurfaceLayer,
    LatencyEnvelope,
)
from linearisation import INPUT_NAMES, STATE_NAMES, linearise_trim  # noqa: E402
from primitive import build_primitive_context  # noqa: E402
from rollout import RolloutConfig, RolloutResult, simulate_primitive, write_log  # noqa: E402
from run_one import _materialise_scenario_primitives, run_scenario  # noqa: E402
from scenarios import ScenarioDefinition, build_scenario  # noqa: E402
from updraft_models import FOUR_FAN_CENTERS_XY, SINGLE_FAN_CENTER_XY  # noqa: E402


SCENARIO_LABELS = {
    "s0_no_wind": "basic no-wind glide",
    "s1_latency_nominal_no_wind": "nominal-latency bank reversal",
    "s1_latency_robust_upper_no_wind": "upper-latency bank reversal",
    "s6_single_gaussian_var": "single-fan Gaussian updraft glide",
    "s6_four_gaussian_var": "four-fan Gaussian updraft glide",
    "s7_single_annular_gp": "single-fan annular-GP updraft glide",
    "s7_four_annular_gp": "four-fan annular-GP updraft glide",
    "s11_governor_rejection": "governor rejection at low altitude",
    "s4_full_nominal_glide_no_wind": "full-duration nominal glide, no wind",
    "s4_full_bank_reversal_left_no_wind": "full-duration left bank reversal, no wind",
    "s4_full_bank_reversal_right_no_wind": "full-duration right bank reversal, no wind",
    "s4_full_recovery_no_wind": "full-duration recovery, no wind",
    "s4_launch_nominal_glide_no_wind": "nominal hand-launch glide, no wind",
    "s4_latency_low_nominal_glide": "low-latency nominal glide",
    "s4_latency_nominal_nominal_glide": "nominal-latency nominal glide",
    "s4_latency_high_nominal_glide": "high-latency nominal glide",
    "s4_latency_low_bank_reversal_left": "low-latency left bank reversal",
    "s4_latency_nominal_bank_reversal_left": "nominal-latency left bank reversal",
    "s4_latency_high_bank_reversal_left": "high-latency left bank reversal",
    "s4_latency_low_recovery": "low-latency recovery",
    "s4_latency_nominal_recovery": "nominal-latency recovery",
    "s4_latency_high_recovery": "high-latency recovery",
    "s4_latency_robust_upper_bank_reversal_left": "upper-latency left bank reversal",
    "s4_gaussian_single_panel": "single-fan Gaussian updraft, panel wind",
    "s4_gaussian_single_cg": "single-fan Gaussian updraft, CG wind",
    "s4_gaussian_four_panel": "four-fan Gaussian updraft, panel wind",
    "s4_annular_single_panel": "single-fan annular-GP updraft, panel wind",
    "s4_annular_single_cg": "single-fan annular-GP updraft, CG wind",
    "s4_annular_four_panel": "four-fan annular-GP updraft, panel wind",
    "s4_gaussian_single_panel_randomised": "randomised single-fan Gaussian updraft",
    "s4_governor_selection": "governor selects recovery from high bank",
    "s9_agile_reversal_left_no_wind": "agile TVLQR reversal, no wind",
}

SCENARIO_SLUGS = {
    "s0_no_wind": "01_basic_no_wind_glide",
    "s1_latency_nominal_no_wind": "01_nominal_latency_bank_reversal",
    "s1_latency_robust_upper_no_wind": "02_upper_latency_bank_reversal",
    "s6_single_gaussian_var": "01_single_fan_gaussian_glide",
    "s6_four_gaussian_var": "02_four_fan_gaussian_glide",
    "s7_single_annular_gp": "01_single_fan_annular_gp_glide",
    "s7_four_annular_gp": "02_four_fan_annular_gp_glide",
    "s11_governor_rejection": "01_low_altitude_rejection",
    "s4_full_nominal_glide_no_wind": "01_baseline_glide_no_wind",
    "s4_full_bank_reversal_left_no_wind": "02_mild_bank_reversal_left_probe",
    "s4_full_bank_reversal_right_no_wind": "03_mild_bank_reversal_right_probe",
    "s4_full_recovery_no_wind": "04_recovery_no_wind",
    "s9_agile_reversal_left_no_wind": "05_agile_tvlqr_reversal_left",
    "s4_launch_nominal_glide_no_wind": "01_launch_nominal_glide",
    "s4_latency_low_nominal_glide": "10_low_latency_nominal_glide",
    "s4_latency_nominal_nominal_glide": "11_nominal_latency_nominal_glide",
    "s4_latency_high_nominal_glide": "12_high_latency_nominal_glide",
    "s4_latency_low_bank_reversal_left": "20_low_latency_bank_reversal_left",
    "s4_latency_nominal_bank_reversal_left": "21_nominal_latency_bank_reversal_left",
    "s4_latency_high_bank_reversal_left": "22_high_latency_bank_reversal_left",
    "s4_latency_robust_upper_bank_reversal_left": "23_upper_latency_bank_reversal_left",
    "s4_latency_low_recovery": "30_low_latency_recovery",
    "s4_latency_nominal_recovery": "31_nominal_latency_recovery",
    "s4_latency_high_recovery": "32_high_latency_recovery",
    "s4_gaussian_single_panel": "40_single_gaussian_panel_wind",
    "s4_gaussian_single_cg": "41_single_gaussian_cg_wind",
    "s4_gaussian_four_panel": "42_four_gaussian_panel_wind",
    "s4_annular_single_panel": "50_single_annular_panel_wind",
    "s4_annular_single_cg": "51_single_annular_cg_wind",
    "s4_annular_four_panel": "52_four_annular_panel_wind",
    "s4_gaussian_single_panel_randomised": "60_single_gaussian_randomised",
    "s4_governor_selection": "02_recovery_selection_high_bank",
}

SCENARIO_GROUPS = {
    "s0_no_wind": "00_smoke",
    "s1_latency_nominal_no_wind": "01_latency_interface",
    "s1_latency_robust_upper_no_wind": "01_latency_interface",
    "s4_full_nominal_glide_no_wind": "03_primitives",
    "s4_full_bank_reversal_left_no_wind": "03_primitives",
    "s4_full_bank_reversal_right_no_wind": "03_primitives",
    "s4_full_recovery_no_wind": "03_primitives",
    "s9_agile_reversal_left_no_wind": "03_primitives",
    "s4_launch_nominal_glide_no_wind": "04_scenario_matrix",
    "s4_latency_low_nominal_glide": "04_scenario_matrix",
    "s4_latency_nominal_nominal_glide": "04_scenario_matrix",
    "s4_latency_high_nominal_glide": "04_scenario_matrix",
    "s4_latency_low_bank_reversal_left": "04_scenario_matrix",
    "s4_latency_nominal_bank_reversal_left": "04_scenario_matrix",
    "s4_latency_high_bank_reversal_left": "04_scenario_matrix",
    "s4_latency_low_recovery": "04_scenario_matrix",
    "s4_latency_nominal_recovery": "04_scenario_matrix",
    "s4_latency_high_recovery": "04_scenario_matrix",
    "s4_latency_robust_upper_bank_reversal_left": "04_scenario_matrix",
    "s4_gaussian_single_panel": "04_scenario_matrix",
    "s4_gaussian_single_cg": "04_scenario_matrix",
    "s4_gaussian_four_panel": "04_scenario_matrix",
    "s4_annular_single_panel": "04_scenario_matrix",
    "s4_annular_single_cg": "04_scenario_matrix",
    "s4_annular_four_panel": "04_scenario_matrix",
    "s4_gaussian_single_panel_randomised": "04_scenario_matrix",
    "s6_single_gaussian_var": "06_updraft_models",
    "s6_four_gaussian_var": "06_updraft_models",
    "s7_single_annular_gp": "07_annular_gp_models",
    "s7_four_annular_gp": "07_annular_gp_models",
    "s11_governor_rejection": "11_governor",
    "s4_governor_selection": "11_governor",
}

FIGURE_STEMS = {
    "A": "A_trajectory_command_actuator",
    "B": "B_flight_rates",
    "C": "C_flight_state_alpha_beta",
    "D": "D_envelope_variables",
    "E": "E_2d_trajectory_geometry",
}
FIGURE_ORDER = ("A", "B", "C", "D", "E")
ATTITUDE_PLOT_SPECS = (
    ("theta", r"Pitch $\theta$ (deg)", False),
    ("phi", r"Roll $\phi$ (deg)", False),
    ("psi", r"Yaw $\psi$ (deg)", True),
)


@dataclass(frozen=True)
class TrajectorySeries:
    key: str
    label: str
    times_s: np.ndarray
    states: np.ndarray
    desired_commands_rad: np.ndarray
    target_commands_rad: np.ndarray
    log_columns: dict[str, np.ndarray]


@dataclass(frozen=True)
class PlotScenarioData:
    scenario: ScenarioDefinition
    actual_row: dict[str, object]
    actual: TrajectorySeries
    references: tuple[TrajectorySeries, ...]
    output_root: Path | None
    seed: int


@dataclass(frozen=True)
class RuntimeObjects:
    aircraft: object
    context: object
    scenario: ScenarioDefinition
    arena: ArenaConfig


@dataclass(frozen=True)
class ScenarioOutputPaths:
    root_dir: Path
    actual_log: Path
    actual_metrics: Path
    manifest: Path


# =============================================================================
# 2) Scenario Execution and Reference Rollout Helpers
# =============================================================================
def friendly_scenario_name(scenario_id: str) -> str:
    return SCENARIO_LABELS.get(scenario_id, scenario_id.replace("_", " "))


def friendly_scenario_slug(scenario_id: str) -> str:
    if scenario_id in SCENARIO_SLUGS:
        return SCENARIO_SLUGS[scenario_id]
    text = friendly_scenario_name(scenario_id).lower()
    chars = [ch if ch.isalnum() else "_" for ch in text]
    slug = "_".join(part for part in "".join(chars).split("_") if part)
    return slug or "scenario"


def scenario_result_group(scenario_id: str) -> str:
    return SCENARIO_GROUPS.get(scenario_id, "99_misc")


def scenario_output_paths(
    scenario_id: str,
    seed: int,
    output_root: str | Path | None,
) -> ScenarioOutputPaths:
    if output_root is None:
        base = REPO_ROOT / "03_Control" / "05_Results"
    else:
        base = Path(output_root)
    root = (
        base
        / scenario_result_group(scenario_id)
        / friendly_scenario_slug(scenario_id)
        / f"{int(seed):03d}"
    )
    return ScenarioOutputPaths(
        root_dir=root,
        actual_log=root / "actual_rollout.csv",
        actual_metrics=root / "actual_metrics.csv",
        manifest=root / "manifest.json",
    )


def _runtime_objects(scenario_id: str, seed: int) -> RuntimeObjects:
    aircraft = adapt_glider(build_nausicaa_glider())
    linear_model = linearise_trim(aircraft=aircraft)
    context = build_primitive_context(
        x_trim=linear_model.x_trim,
        u_trim=linear_model.u_trim,
        min_entry_altitude_m=0.75,
    )
    scenario = build_scenario(scenario_id, linear_model.x_trim, REPO_ROOT, seed=seed)
    scenario = _materialise_scenario_primitives(
        scenario=scenario,
        context=context,
        aircraft=aircraft,
    )
    return RuntimeObjects(
        aircraft=aircraft,
        context=context,
        scenario=scenario,
        arena=ArenaConfig(),
    )


def _actual_log_path(
    scenario_id: str,
    seed: int,
    output_root: str | Path | None,
) -> Path:
    return scenario_output_paths(scenario_id, seed, output_root).actual_log


def _select_executed_primitive(
    scenario: ScenarioDefinition,
    selected_name: str,
) -> object:
    candidates = scenario.candidate_primitives or (scenario.primitive,)
    for primitive in candidates:
        if primitive.name == selected_name:
            return primitive
    if scenario.primitive.name == selected_name:
        return scenario.primitive
    raise ValueError(f"Executed primitive '{selected_name}' is not in the scenario.")


def _ideal_command_layer() -> CommandToSurfaceLayer:
    config = CommandToSurfaceConfig(
        mode="nominal",
        quantise=False,
        use_onset_delay=False,
        use_state_feedback_delay=False,
    )
    # Ideal references remove command-path distortion but keep a stable actuator time constant.
    return CommandToSurfaceLayer(config=config, envelope=LatencyEnvelope())


def _reference_series(
    key: str,
    label: str,
    runtime: RuntimeObjects,
    primitive: object,
    scenario_id: str,
    seed: int,
    paths: ScenarioOutputPaths,
    use_environment: bool,
) -> TrajectorySeries:
    wind_model = None if use_environment else runtime.scenario.wind_model
    wind_name = "none" if use_environment else runtime.scenario.wind_model_name
    wind_label = "none" if use_environment else runtime.scenario.wind_param_label
    log_path = paths.root_dir / f"{key}_rollout.csv"
    result = simulate_primitive(
        scenario_id=f"{scenario_id}_{key}",
        seed=seed,
        primitive=primitive,
        x0=runtime.scenario.x0,
        context=runtime.context,
        aircraft=runtime.aircraft,
        wind_model=wind_model,
        wind_model_name=wind_name,
        wind_mode=runtime.scenario.wind_mode,
        command_layer=_ideal_command_layer(),
        log_path=log_path,
        repo_root=REPO_ROOT,
        rollout_config=RolloutConfig(),
        arena_config=runtime.arena,
        wind_param_label=wind_label,
        selected_primitive_name=primitive.name,
    )
    write_log(result, log_path)
    metrics = dict(result.metrics)
    metrics["log_path"] = f"{key}_rollout.csv"
    if "log_path_relative" in metrics:
        metrics["log_path_relative"] = metrics["log_path"]
    _write_single_row(paths.root_dir / f"{key}_metrics.csv", metrics)
    return _series_from_result(result, key=key, label=label)


# =============================================================================
# 3) Log Loading and Geometry Helpers
# =============================================================================
def _series_from_result(
    result: RolloutResult,
    key: str,
    label: str,
) -> TrajectorySeries:
    return TrajectorySeries(
        key=key,
        label=label,
        times_s=np.asarray(result.times_s, dtype=float),
        states=np.asarray(result.states, dtype=float),
        desired_commands_rad=np.asarray(result.desired_commands, dtype=float),
        target_commands_rad=np.asarray(result.target_commands, dtype=float),
        log_columns=_numeric_log_columns(result.log_rows),
    )


def _load_actual_log(path: Path) -> TrajectorySeries:
    if not path.exists():
        raise FileNotFoundError(f"Scenario log was not generated: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Scenario log is empty: {path}")

    # Logged fields use the canonical state and command names from linearisation.py.
    times = np.asarray([float(row["t_s"]) for row in rows], dtype=float)
    states = np.asarray(
        [[float(row[name]) for name in STATE_NAMES] for row in rows],
        dtype=float,
    )
    desired = np.asarray(
        [[float(row[f"desired_{name}_rad"]) for name in INPUT_NAMES] for row in rows],
        dtype=float,
    )
    target = np.asarray(
        [[float(row[f"target_{name}_rad"]) for name in INPUT_NAMES] for row in rows],
        dtype=float,
    )
    return TrajectorySeries(
        key="actual",
        label=TRAJECTORY_SPECS["actual"]["label"],
        times_s=times,
        states=states,
        desired_commands_rad=desired,
        target_commands_rad=target,
        log_columns=_numeric_log_columns(rows),
    )


def _numeric_log_columns(
    rows: tuple[dict[str, object], ...] | list[dict[str, object]],
) -> dict[str, np.ndarray]:
    if not rows:
        return {}
    columns: dict[str, np.ndarray] = {}
    for key in rows[0]:
        values: list[float] = []
        for row in rows:
            try:
                values.append(float(row[key]))
            except (TypeError, ValueError):
                values = []
                break
        if values:
            columns[key] = np.asarray(values, dtype=float)
    return columns


def _write_single_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _move_if_present(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        source.replace(destination)
    except FileNotFoundError:
        # OneDrive can briefly report files that have already been moved by a prior cleanup.
        return


def _remove_empty_dir(path: Path) -> None:
    try:
        path.rmdir()
    except OSError:
        pass


def _normalise_actual_outputs(
    paths: ScenarioOutputPaths,
    runner_root: Path,
    scenario_id: str,
    seed: int,
    row: dict[str, object],
) -> dict[str, object]:
    raw_metrics = runner_root / "metrics" / f"{scenario_id}_seed{int(seed)}.csv"
    raw_log = runner_root / "logs" / f"{scenario_id}_seed{int(seed)}.csv"
    raw_candidates = (
        runner_root / "metrics" / f"{scenario_id}_seed{int(seed)}_governor_candidates.csv"
    )
    raw_rejections = (
        runner_root / "metrics" / f"{scenario_id}_seed{int(seed)}_governor_rejections.csv"
    )

    # Generated simulator outputs are renamed into user-facing files for post analysis.
    _move_if_present(raw_log, paths.actual_log)
    _move_if_present(raw_candidates, paths.root_dir / "governor_candidates.csv")
    _move_if_present(raw_rejections, paths.root_dir / "governor_rejections.csv")
    _move_if_present(raw_metrics, paths.actual_metrics)
    _remove_empty_dir(runner_root / "logs")
    _remove_empty_dir(runner_root / "metrics")
    _remove_empty_dir(runner_root)

    candidate_path = paths.root_dir / "governor_candidates.csv"
    rejection_path = paths.root_dir / "governor_rejections.csv"
    friendly_row = dict(row)
    friendly_row["case_name"] = friendly_scenario_name(scenario_id)
    friendly_row["case_group"] = scenario_result_group(scenario_id)
    friendly_row["case_id"] = friendly_scenario_slug(scenario_id)
    friendly_row["case_folder"] = paths.root_dir.name
    friendly_row["case_path"] = (
        f"{scenario_result_group(scenario_id)}/"
        f"{friendly_scenario_slug(scenario_id)}/"
        f"{int(seed):03d}"
    )
    friendly_row["scenario_name"] = friendly_scenario_name(scenario_id)
    # Stored metrics reference files within the seed folder, not temporary runner paths.
    friendly_row["actual_rollout_file"] = "actual_rollout.csv"
    friendly_row["actual_metrics_file"] = "actual_metrics.csv"
    friendly_row["log_path"] = "actual_rollout.csv" if paths.actual_log.exists() else ""
    friendly_row["log_path_relative"] = friendly_row["log_path"]
    friendly_row["candidate_table_path"] = (
        "governor_candidates.csv" if candidate_path.exists() else ""
    )
    if "governor_rejection_log_path" in friendly_row:
        friendly_row["governor_rejection_log_path"] = (
            "governor_rejections.csv" if rejection_path.exists() else ""
        )
    _write_single_row(paths.actual_metrics, friendly_row)
    return friendly_row


def _draw_box(
    ax: mpl.axes.Axes,
    bounds: dict[str, tuple[float, float]],
    color: tuple[float, float, float, float],
    linewidth: float,
    linestyle: str,
    label: str | None,
) -> None:
    x0, x1 = bounds["x_w"]
    y0, y1 = bounds["y_w"]
    z0, z1 = bounds["z_w"]
    corners = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=float,
    )
    edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    for idx, (a, b) in enumerate(edges):
        ax.plot(
            [corners[a, 0], corners[b, 0]],
            [corners[a, 1], corners[b, 1]],
            [corners[a, 2], corners[b, 2]],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            label=label if idx == 0 else None,
        )


def _plot_arena(ax: mpl.axes.Axes, arena: ArenaConfig) -> None:
    _draw_box(
        ax,
        tracker_bounds(arena),
        color=(0.0, 0.0, 0.0, 0.28),
        linewidth=0.65,
        linestyle=":",
        label="Tracker limit",
    )
    # The true safety volume, not the tracker limit, is the primitive acceptance gate.
    _draw_box(
        ax,
        safe_bounds(arena),
        color=TRUE_SAFETY_VOLUME_COLOR,
        linewidth=TRUE_SAFETY_VOLUME_LINEWIDTH,
        linestyle=TRUE_SAFETY_VOLUME_LINESTYLE,
        label="True safety volume",
    )


def _fan_outlet_centres(wind_model: object) -> tuple[tuple[float, float], ...]:
    base_model = getattr(wind_model, "base", wind_model)
    centres = getattr(base_model, "fan_centers_xy", None)
    if centres:
        return tuple((float(x), float(y)) for x, y in centres)
    name = str(getattr(base_model, "name", ""))
    if "four" in name:
        return tuple((float(x), float(y)) for x, y in FOUR_FAN_CENTERS_XY)
    return ((float(SINGLE_FAN_CENTER_XY[0]), float(SINGLE_FAN_CENTER_XY[1])),)


def _sample_updraft_volume(
    wind_model: object,
    arena: ArenaConfig,
    style: PlotStyle,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bounds = safe_bounds(arena)
    x = np.linspace(bounds["x_w"][0], bounds["x_w"][1], style.updraft_grid_x)
    y = np.linspace(bounds["y_w"][0], bounds["y_w"][1], style.updraft_grid_y)
    z = np.linspace(bounds["z_w"][0], bounds["z_w"][1], style.updraft_grid_z)
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing="ij")
    points = np.column_stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()])
    w_vertical = np.asarray(wind_model(points), dtype=float)[:, 2].reshape(x_grid.shape)
    return x, y, z, np.maximum(w_vertical, 0.0)


def _plot_updraft_center_slices(
    ax: mpl.axes.Axes,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    w_grid: np.ndarray,
    centres: tuple[tuple[float, float], ...],
    cmap: mpl.colors.Colormap,
    norm: mpl.colors.Normalize,
) -> None:
    centre_xy = np.mean(np.asarray(centres, dtype=float), axis=0)
    ix_center = int(np.argmin(np.abs(x - centre_xy[0])))
    iy_center = int(np.argmin(np.abs(y - centre_xy[1])))
    visibility_floor = _updraft_visibility_floor(w_grid)

    y_mesh, z_mesh_y = np.meshgrid(y, z, indexing="ij")
    x_mesh_y = np.full_like(y_mesh, x[ix_center])
    w_slice_x = w_grid[ix_center, :, :]
    ax.plot_surface(
        x_mesh_y,
        y_mesh,
        z_mesh_y,
        facecolors=_updraft_facecolors(w_slice_x, cmap, norm, visibility_floor),
        rstride=1,
        cstride=1,
        linewidth=0.0,
        antialiased=False,
        shade=False,
        zorder=1,
    )

    x_mesh, z_mesh_x = np.meshgrid(x, z, indexing="ij")
    y_mesh_x = np.full_like(x_mesh, y[iy_center])
    w_slice_y = w_grid[:, iy_center, :]
    ax.plot_surface(
        x_mesh,
        y_mesh_x,
        z_mesh_x,
        facecolors=_updraft_facecolors(w_slice_y, cmap, norm, visibility_floor),
        rstride=1,
        cstride=1,
        linewidth=0.0,
        antialiased=False,
        shade=False,
        zorder=1,
    )


def _updraft_visibility_floor(w_grid: np.ndarray) -> float:
    w_max = float(np.nanmax(w_grid))
    if not np.isfinite(w_max) or w_max <= 0.0:
        return float("inf")
    return max(0.04 * w_max, 0.05)


def _updraft_facecolors(
    values: np.ndarray,
    cmap: mpl.colors.Colormap,
    norm: mpl.colors.Normalize,
    visibility_floor: float,
) -> np.ndarray:
    facecolors = cmap(norm(np.clip(values, PLOT_VMIN, PLOT_VMAX)))
    facecolors[..., 3] = np.where(values >= visibility_floor, facecolors[..., 3], 0.0)
    return facecolors


def _plot_updraft_outlet_slice(
    ax: mpl.axes.Axes,
    wind_model: object,
    arena: ArenaConfig,
    cmap: mpl.colors.Colormap,
    norm: mpl.colors.Normalize,
    visibility_floor: float,
    style: PlotStyle,
) -> None:
    bounds = safe_bounds(arena)
    x = np.linspace(bounds["x_w"][0], bounds["x_w"][1], style.updraft_grid_x)
    y = np.linspace(bounds["y_w"][0], bounds["y_w"][1], style.updraft_grid_y)
    z_plane = float(np.clip(FAN_OUTLET_Z_M, bounds["z_w"][0], bounds["z_w"][1]))
    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")
    points = np.column_stack(
        [
            x_grid.ravel(),
            y_grid.ravel(),
            np.full(x_grid.size, z_plane),
        ]
    )
    w_plane = np.asarray(wind_model(points), dtype=float)[:, 2].reshape(x_grid.shape)
    ax.plot_surface(
        x_grid,
        y_grid,
        np.full_like(x_grid, z_plane),
        facecolors=_updraft_facecolors(w_plane, cmap, norm, visibility_floor),
        rstride=1,
        cstride=1,
        linewidth=0.0,
        antialiased=False,
        shade=False,
        zorder=4,
    )


def _plot_updraft_isosurfaces(
    ax: mpl.axes.Axes,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    w_grid: np.ndarray,
    style: PlotStyle,
    cmap: mpl.colors.Colormap,
    norm: mpl.colors.Normalize,
) -> None:
    if marching_cubes is None:
        return
    w_max = float(np.nanmax(w_grid))
    if not np.isfinite(w_max) or w_max <= 1e-9:
        return
    spacing = (
        float(x[1] - x[0]),
        float(y[1] - y[0]),
        float(z[1] - z[0]),
    )
    for frac in style.updraft_iso_fracs:
        level = float(frac * w_max)
        if level <= float(np.nanmin(w_grid)) or level >= w_max:
            continue
        verts, faces, _, _ = marching_cubes(w_grid, level=level, spacing=spacing)
        verts[:, 0] += float(x[0])
        verts[:, 1] += float(y[0])
        verts[:, 2] += float(z[0])
        mesh = Poly3DCollection(verts[faces], linewidths=0.0, zorder=2)
        mesh.set_facecolor(cmap(norm(np.clip(level, PLOT_VMIN, PLOT_VMAX))))
        mesh.set_edgecolor("none")
        ax.add_collection3d(mesh)


def _plot_fan_outlets(
    ax: mpl.axes.Axes,
    centres: tuple[tuple[float, float], ...],
) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 160)
    radius = 0.5 * FAN_OUTLET_DIAMETER_M
    for idx, (cx, cy) in enumerate(centres):
        ax.plot(
            float(cx) + radius * np.cos(theta),
            float(cy) + radius * np.sin(theta),
            np.full_like(theta, FAN_OUTLET_Z_M),
            color=(0.0, 0.0, 0.0, FAN_OUTLET_ALPHA),
            linewidth=FAN_OUTLET_EDGE_LW,
            linestyle=FAN_OUTLET_DASH,
            label="Fan outlet" if idx == 0 else None,
            zorder=8,
        )


def _plot_updraft_field(
    fig: mpl.figure.Figure,
    ax: mpl.axes.Axes,
    cbar_ax: mpl.axes.Axes,
    wind_model: object,
    arena: ArenaConfig,
    style: PlotStyle,
) -> None:
    x, y, z, w_grid = _sample_updraft_volume(wind_model, arena, style)
    centres = _fan_outlet_centres(wind_model)
    cmap = build_alpha_cmap(style)
    norm = mpl.colors.Normalize(vmin=PLOT_VMIN, vmax=PLOT_VMAX)

    # Thermal-style centre slices and sparse isosurfaces keep plume structure visible
    # without hiding the trajectory or making batch generation unnecessarily slow.
    visibility_floor = _updraft_visibility_floor(w_grid)
    _plot_updraft_outlet_slice(ax, wind_model, arena, cmap, norm, visibility_floor, style)
    _plot_updraft_center_slices(ax, x, y, z, w_grid, centres, cmap, norm)
    _plot_updraft_isosurfaces(ax, x, y, z, w_grid, style, cmap, norm)
    _plot_fan_outlets(ax, centres)

    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=thermal_cmap())
    mappable.set_array([])
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    cbar.set_label(CBAR_LABEL, fontsize=style.colorbar_label_size)
    cbar.set_ticks(np.arange(PLOT_VMIN, PLOT_VMAX + 1e-9, 1.0))
    cbar.formatter = mpl.ticker.FormatStrFormatter("%.2f")
    cbar.update_ticks()
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_label_position("left")
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(width=0.6, length=2, labelsize=style.colorbar_tick_size)
    cbar.outline.set_linewidth(0.8)
    cbar.outline.set_edgecolor("black")


def _actual_progress_cmap() -> mpl.colors.LinearSegmentedColormap:
    return mpl.colors.LinearSegmentedColormap.from_list(
        "actual_time_progress",
        [ACTUAL_START_COLOR, ACTUAL_END_COLOR],
    )


def _progress_values(times_s: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=float)
    times = np.asarray(times_s, dtype=float)
    if times.size < 2 or not np.isfinite(times).all():
        return np.linspace(0.0, 1.0, count)
    lower = float(times[0])
    upper = float(times[-1])
    if upper <= lower:
        return np.linspace(0.0, 1.0, count)
    segment_times = 0.5 * (times[:-1] + times[1:])
    progress = np.clip((segment_times - lower) / (upper - lower), 0.0, 1.0)
    if count == 1:
        return np.array([0.0], dtype=float)
    span = float(progress[-1] - progress[0])
    if span <= 0.0:
        return np.linspace(0.0, 1.0, count)
    return (progress - progress[0]) / span


def _actual_gradient_colors(times_s: np.ndarray, count: int) -> np.ndarray:
    return _actual_progress_cmap()(_progress_values(times_s, count))


def _plot_actual_gradient_line_2d(
    ax: mpl.axes.Axes,
    times_s: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    linewidth: float,
    label: str | None,
    zorder: int,
) -> LineCollection | None:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.size < 2 or y_arr.size < 2:
        return None
    points = np.column_stack([x_arr, y_arr])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    collection = LineCollection(
        segments,
        colors=_actual_gradient_colors(times_s, segments.shape[0]),
        linewidths=linewidth,
        linestyles="-",
        label=label,
        zorder=zorder,
    )
    ax.add_collection(collection)
    ax.update_datalim(points)
    ax.autoscale_view()
    return collection


def _plot_actual_gradient_line_3d(
    ax: mpl.axes.Axes,
    times_s: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    linewidth: float,
    label: str | None,
    zorder: int,
) -> Line3DCollection | None:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    z_arr = np.asarray(z, dtype=float)
    if x_arr.size < 2 or y_arr.size < 2 or z_arr.size < 2:
        return None
    points = np.column_stack([x_arr, y_arr, z_arr])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    collection = Line3DCollection(
        segments,
        colors=_actual_gradient_colors(times_s, segments.shape[0]),
        linewidths=linewidth,
        linestyles="-",
        label=label,
        zorder=zorder,
    )
    ax.add_collection3d(collection)
    return collection


def _legend_handles_labels(
    axes: tuple[mpl.axes.Axes, ...] | list[mpl.axes.Axes],
) -> tuple[list[object], list[str]]:
    unique_handles: list[object] = []
    unique_labels: list[str] = []
    seen: set[str] = set()
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if not label or label.startswith("_") or label in seen:
                continue
            unique_handles.append(handle)
            unique_labels.append(label)
            seen.add(label)
    return unique_handles, unique_labels


def _framed_figure_legend(
    fig: mpl.figure.Figure,
    axes: tuple[mpl.axes.Axes, ...] | list[mpl.axes.Axes],
    style: PlotStyle,
    loc: str,
    bbox_to_anchor: tuple[float, float],
    ncol: int,
) -> mpl.legend.Legend | None:
    handles, labels = _legend_handles_labels(axes)
    if not handles:
        return None
    legend = fig.legend(
        handles,
        labels,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        ncol=ncol,
        frameon=True,
        framealpha=1.0,
        edgecolor="black",
        fontsize=style.legend_size,
        handlelength=1.7,
        handletextpad=0.45,
        columnspacing=0.9,
        borderpad=0.45,
        labelspacing=0.25,
    )
    legend.get_frame().set_linewidth(0.8)
    return legend


def _plot_series_line(
    ax: mpl.axes.Axes,
    series: TrajectorySeries,
    x: np.ndarray,
    y: np.ndarray,
    label: str | None = None,
    zorder: int = 10,
) -> None:
    spec = TRAJECTORY_SPECS[series.key]
    if series.key == "actual":
        _plot_actual_gradient_line_2d(
            ax,
            series.times_s,
            x,
            y,
            linewidth=float(spec["linewidth"]),
            label=label,
            zorder=zorder,
        )
        return
    ax.plot(
        x,
        y,
        color=spec["color"],
        linestyle=spec["linestyle"],
        linewidth=spec["linewidth"],
        alpha=float(spec.get("alpha", 1.0)),
        label=label,
        zorder=zorder,
    )


def _all_series(data: PlotScenarioData) -> tuple[TrajectorySeries, ...]:
    return (data.actual, *data.references)


def _state_column(series: TrajectorySeries, name: str) -> np.ndarray:
    return series.states[:, STATE_NAMES.index(name)]


def _log_column(series: TrajectorySeries, name: str) -> np.ndarray | None:
    values = series.log_columns.get(name)
    if values is None or values.shape[0] != series.times_s.shape[0]:
        return None
    return values


def _alpha_beta_rad(series: TrajectorySeries) -> tuple[np.ndarray, np.ndarray]:
    alpha = _log_column(series, "alpha_rad")
    beta = _log_column(series, "beta_rad")
    if alpha is not None and beta is not None:
        return alpha, beta
    velocities = series.states[:, 6:9]
    speed = np.linalg.norm(velocities, axis=1)
    alpha_calc = np.arctan2(velocities[:, 2], np.maximum(velocities[:, 0], 1e-12))
    beta_calc = np.arcsin(
        np.clip(velocities[:, 1] / np.maximum(speed, 1e-12), -1.0, 1.0)
    )
    return alpha_calc, beta_calc


def _safety_margin_series(series: TrajectorySeries) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wall = _log_column(series, "min_wall_distance_m")
    floor = _log_column(series, "floor_margin_m")
    ceiling = _log_column(series, "ceiling_margin_m")
    if wall is not None and floor is not None and ceiling is not None:
        return wall, floor, ceiling
    bounds = safe_bounds(ArenaConfig())
    x_w = _state_column(series, "x_w")
    y_w = _state_column(series, "y_w")
    z_w = _state_column(series, "z_w")
    x_margin = np.minimum(x_w - bounds["x_w"][0], bounds["x_w"][1] - x_w)
    y_margin = np.minimum(y_w - bounds["y_w"][0], bounds["y_w"][1] - y_w)
    return (
        np.minimum(x_margin, y_margin),
        z_w - bounds["z_w"][0],
        bounds["z_w"][1] - z_w,
    )


def _add_rectangle_projection(
    ax: mpl.axes.Axes,
    bounds: dict[str, tuple[float, float]],
    x_key: str,
    y_key: str,
    edgecolor: str | tuple[float, float, float, float],
    linewidth: float,
    linestyle: str,
    label: str,
    zorder: int,
) -> None:
    x0, x1 = bounds[x_key]
    y0, y1 = bounds[y_key]
    rect = Rectangle(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        fill=False,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label,
        zorder=zorder,
    )
    ax.add_patch(rect)


def _plot_start_end_projection(
    ax: mpl.axes.Axes,
    actual: TrajectorySeries,
    x_index: int,
    y_index: int,
) -> None:
    start = actual.states[0]
    end = actual.states[-1]
    ax.scatter(
        [start[x_index]],
        [start[y_index]],
        color=ACTUAL_START_COLOR,
        s=18,
        marker="o",
        label="Start",
        zorder=20,
    )
    ax.scatter(
        [end[x_index]],
        [end[y_index]],
        color=ACTUAL_END_COLOR,
        s=24,
        marker="x",
        label="End",
        zorder=20,
    )


def _plot_fan_outlets_2d(
    ax: mpl.axes.Axes,
    centres: tuple[tuple[float, float], ...],
) -> None:
    radius = 0.5 * FAN_OUTLET_DIAMETER_M
    for idx, (cx, cy) in enumerate(centres):
        outlet = Circle(
            (float(cx), float(cy)),
            radius=radius,
            fill=False,
            edgecolor=(0.0, 0.0, 0.0, FAN_OUTLET_ALPHA),
            linewidth=FAN_OUTLET_EDGE_LW,
            linestyle=FAN_OUTLET_DASH,
            label="Fan outlet" if idx == 0 else None,
            zorder=8,
        )
        ax.add_patch(outlet)


# =============================================================================
# 4) Figure Builders
# =============================================================================
def _plot_trajectory(ax: mpl.axes.Axes, series: TrajectorySeries) -> None:
    spec = TRAJECTORY_SPECS[series.key]
    if series.key == "actual":
        _plot_actual_gradient_line_3d(
            ax,
            series.times_s,
            series.states[:, 0],
            series.states[:, 1],
            series.states[:, 2],
            linewidth=float(spec["linewidth"]),
            label=series.label,
            zorder=10,
        )
        return
    ax.plot(
        series.states[:, 0],
        series.states[:, 1],
        series.states[:, 2],
        color=spec["color"],
        linestyle=spec["linestyle"],
        linewidth=spec["linewidth"],
        alpha=float(spec.get("alpha", 1.0)),
        label=series.label,
        zorder=10,
    )


def _plot_start_end(ax: mpl.axes.Axes, actual: TrajectorySeries) -> None:
    start = actual.states[0, :3]
    end = actual.states[-1, :3]
    ax.scatter(
        [start[0]],
        [start[1]],
        [start[2]],
        color=ACTUAL_START_COLOR,
        s=26,
        marker="o",
        label="Start",
        zorder=20,
    )
    ax.scatter(
        [end[0]],
        [end[1]],
        [end[2]],
        color=ACTUAL_END_COLOR,
        s=32,
        marker="x",
        label="End",
        zorder=20,
    )


def _plot_command_axes(
    axes: tuple[mpl.axes.Axes, mpl.axes.Axes, mpl.axes.Axes],
    actual: TrajectorySeries,
    style: PlotStyle,
) -> None:
    labels = (
        r"$\delta_a$ (deg)",
        r"$\delta_e$ (deg)",
        r"$\delta_r$ (deg)",
    )
    limit_keys = ("delta_a", "delta_e", "delta_r")
    desired_deg = np.rad2deg(actual.desired_commands_rad)
    target_deg = np.rad2deg(actual.target_commands_rad)
    for idx, ax in enumerate(axes):
        ax.plot(
            actual.times_s,
            desired_deg[:, idx],
            color=CONTROL_REFERENCE_COLOR,
            linewidth=DESIRED_COMMAND_LINEWIDTH,
            linestyle="--",
            alpha=DESIRED_COMMAND_ALPHA,
            label="Desired",
        )
        _plot_actual_gradient_line_2d(
            ax,
            actual.times_s,
            actual.times_s,
            target_deg[:, idx],
            linewidth=1.2,
            label="Command path",
            zorder=10,
        )
        limit = AGGREGATE_LIMITS[limit_keys[idx]]
        ax.axhline(
            limit.positive_deg,
            color="black",
            linestyle=":",
            linewidth=0.8,
            label="surface limit",
        )
        ax.axhline(
            limit.negative_deg,
            color="black",
            linestyle=":",
            linewidth=0.8,
        )
        style_command_axis(ax, labels[idx], style)
        if idx < len(axes) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(r"$t$ (s)", fontsize=style.command_label_size)
        if idx == 0:
            framed_legend(
                ax,
                style,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.03),
                ncol=3,
            )


def build_figure_a_trajectory_command_actuator(
    data: PlotScenarioData,
    style: PlotStyle | None = None,
) -> mpl.figure.Figure:
    style = style or PlotStyle()
    fig = plt.figure(figsize=style.figsize, dpi=style.dpi)
    fig.patch.set_facecolor("white")
    grid = fig.add_gridspec(
        3,
        4,
        width_ratios=style.width_ratios,
        height_ratios=style.height_ratios,
        wspace=style.wspace,
        hspace=style.hspace,
    )
    ax3d = fig.add_subplot(grid[:, 0], projection="3d")
    cbar_ax = fig.add_subplot(grid[:, 1])
    _shrink_colorbar_axis(cbar_ax, style)
    command_axes = (
        fig.add_subplot(grid[0, 3]),
        fig.add_subplot(grid[1, 3]),
        fig.add_subplot(grid[2, 3]),
    )

    _plot_arena(ax3d, ArenaConfig())
    if data.scenario.wind_model is not None:
        _plot_updraft_field(fig, ax3d, cbar_ax, data.scenario.wind_model, ArenaConfig(), style)
    else:
        cbar_ax.set_visible(False)
    _plot_trajectory(ax3d, data.actual)
    for reference in data.references:
        _plot_trajectory(ax3d, reference)
    _plot_start_end(ax3d, data.actual)

    room = tracker_bounds(ArenaConfig())
    style_3d_axis(
        ax3d,
        xlim=room["x_w"],
        ylim=room["y_w"],
        zlim=room["z_w"],
        style=style,
    )
    framed_legend(
        ax3d,
        style,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
    )
    _plot_command_axes(command_axes, data.actual, style)
    return fig


def build_composite_figure(
    data: PlotScenarioData,
    style: PlotStyle | None = None,
) -> mpl.figure.Figure:
    return build_figure_a_trajectory_command_actuator(data, style=style)


def build_figure_b_flight_rates(
    data: PlotScenarioData,
    style: PlotStyle | None = None,
) -> mpl.figure.Figure:
    style = style or PlotStyle()
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(7.2, 4.6),
        dpi=style.dpi,
        sharex="col",
    )
    fig.patch.set_facecolor("white")
    velocity_names = ("u", "v", "w")
    velocity_labels = (
        r"$u_b$ (m $\!$s$^{-1}$)",
        r"$v_b$ (m $\!$s$^{-1}$)",
        r"$w_b$ (m $\!$s$^{-1}$)",
    )
    rate_names = ("p", "q", "r")
    rate_labels = (
        r"$p$ (deg $\!$s$^{-1}$)",
        r"$q$ (deg $\!$s$^{-1}$)",
        r"$r$ (deg $\!$s$^{-1}$)",
    )
    for row_idx, (name, label) in enumerate(zip(velocity_names, velocity_labels)):
        ax = axes[row_idx, 0]
        for series in _all_series(data):
            _plot_series_line(
                ax,
                series,
                series.times_s,
                _state_column(series, name),
                label=series.label if row_idx == 0 else None,
            )
        style_time_axis(ax, label, style, show_xlabel=row_idx == 2)
    for row_idx, (name, label) in enumerate(zip(rate_names, rate_labels)):
        ax = axes[row_idx, 1]
        for series in _all_series(data):
            _plot_series_line(
                ax,
                series,
                series.times_s,
                np.rad2deg(_state_column(series, name)),
                label=series.label if row_idx == 0 else None,
            )
        style_time_axis(ax, label, style, show_xlabel=row_idx == 2)
    _framed_figure_legend(
        fig,
        [axes[0, 0], axes[0, 1]],
        style,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    return fig


def build_figure_c_flight_state_alpha_beta(
    data: PlotScenarioData,
    style: PlotStyle | None = None,
) -> mpl.figure.Figure:
    style = style or PlotStyle()
    fig = plt.figure(figsize=(7.2, 4.2), dpi=style.dpi)
    fig.patch.set_facecolor("white")
    grid = fig.add_gridspec(6, 2, wspace=0.28, hspace=0.26)
    axes_left = (
        fig.add_subplot(grid[0:2, 0]),
        fig.add_subplot(grid[2:4, 0]),
        fig.add_subplot(grid[4:6, 0]),
    )
    axes_right = (
        fig.add_subplot(grid[0:3, 1]),
        fig.add_subplot(grid[3:6, 1]),
    )

    for idx, (name, label, unwrap) in enumerate(ATTITUDE_PLOT_SPECS):
        ax = axes_left[idx]
        for series in _all_series(data):
            values = _state_column(series, name)
            if unwrap:
                values = np.unwrap(values)
            _plot_series_line(
                ax,
                series,
                series.times_s,
                np.rad2deg(values),
                label=series.label if idx == 0 else None,
            )
        style_time_axis(ax, label, style, show_xlabel=idx == 2)

    for idx, (angle_idx, label) in enumerate(
        ((0, r"$\alpha$ (deg)"), (1, r"$\beta$ (deg)"))
    ):
        ax = axes_right[idx]
        for series in _all_series(data):
            alpha, beta = _alpha_beta_rad(series)
            values = alpha if angle_idx == 0 else beta
            _plot_series_line(
                ax,
                series,
                series.times_s,
                np.rad2deg(values),
                label=series.label if idx == 0 else None,
            )
        if angle_idx == 0:
            limit_deg = float(np.rad2deg(RolloutConfig().max_abs_alpha_rad))
            ax.axhline(
                limit_deg,
                color="black",
                linestyle=":",
                linewidth=0.8,
                label=r"$\alpha$ limit",
            )
            ax.axhline(-limit_deg, color="black", linestyle=":", linewidth=0.8)
        style_time_axis(ax, label, style, show_xlabel=idx == 1)
    _framed_figure_legend(
        fig,
        [axes_left[0], axes_right[0]],
        style,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
    )
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.12, top=0.88)
    return fig


def build_figure_d_envelope_variables(
    data: PlotScenarioData,
    style: PlotStyle | None = None,
) -> mpl.figure.Figure:
    style = style or PlotStyle()
    fig = plt.figure(figsize=(7.2, 3.6), dpi=style.dpi)
    fig.patch.set_facecolor("white")
    grid = fig.add_gridspec(3, 2, width_ratios=(1.18, 1.0), wspace=0.28, hspace=0.18)
    height_ax = fig.add_subplot(grid[:, 0])
    margin_axes = (
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[2, 1]),
    )
    bounds = safe_bounds(ArenaConfig())
    for series in _all_series(data):
        _plot_series_line(
            height_ax,
            series,
            series.times_s,
            _state_column(series, "z_w"),
            label=series.label,
        )
    height_ax.axhline(bounds["z_w"][0], color="black", linestyle=":", linewidth=0.8)
    height_ax.axhline(bounds["z_w"][1], color="black", linestyle=":", linewidth=0.8)
    height_ax.axhline(
        RolloutConfig().min_altitude_m,
        color=SIMULATION_FLOOR_COLOR,
        linestyle=SIMULATION_FLOOR_LINESTYLE,
        linewidth=SIMULATION_FLOOR_LINEWIDTH,
        label="simulation floor",
    )
    style_time_axis(height_ax, r"$z_w$ (m)", style, show_xlabel=True)

    margin_labels = (
        r"$d_{\mathrm{wall,min}}$ (m)",
        r"$d_{\mathrm{floor}}$ (m)",
        r"$d_{\mathrm{ceiling}}$ (m)",
    )
    for axis_idx, ax in enumerate(margin_axes):
        for series in _all_series(data):
            margins = _safety_margin_series(series)
            _plot_series_line(
                ax,
                series,
                series.times_s,
                margins[axis_idx],
                label=series.label if axis_idx == 0 else None,
            )
        ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8)
        style_time_axis(ax, margin_labels[axis_idx], style, show_xlabel=axis_idx == 2)
    _framed_figure_legend(
        fig,
        [height_ax, *margin_axes],
        style,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
    )
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.14, top=0.86)
    return fig


def build_figure_e_2d_trajectory_geometry(
    data: PlotScenarioData,
    style: PlotStyle | None = None,
) -> mpl.figure.Figure:
    style = style or PlotStyle()
    fig, (xy_ax, zx_ax) = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.2),
        dpi=style.dpi,
    )
    fig.patch.set_facecolor("white")
    tracker = tracker_bounds(ArenaConfig())
    safe = safe_bounds(ArenaConfig())

    for series in _all_series(data):
        _plot_series_line(
            xy_ax,
            series,
            _state_column(series, "x_w"),
            _state_column(series, "y_w"),
            label=series.label,
            zorder=10,
        )
        _plot_series_line(
            zx_ax,
            series,
            _state_column(series, "x_w"),
            _state_column(series, "z_w"),
            label=series.label,
            zorder=10,
        )
    _add_rectangle_projection(
        xy_ax,
        tracker,
        "x_w",
        "y_w",
        edgecolor=(0.0, 0.0, 0.0, 0.28),
        linewidth=0.65,
        linestyle=":",
        label="Tracker limit",
        zorder=4,
    )
    _add_rectangle_projection(
        xy_ax,
        safe,
        "x_w",
        "y_w",
        edgecolor=TRUE_SAFETY_VOLUME_COLOR,
        linewidth=TRUE_SAFETY_VOLUME_LINEWIDTH,
        linestyle=TRUE_SAFETY_VOLUME_LINESTYLE,
        label="True safety volume",
        zorder=5,
    )
    _add_rectangle_projection(
        zx_ax,
        tracker,
        "x_w",
        "z_w",
        edgecolor=(0.0, 0.0, 0.0, 0.28),
        linewidth=0.65,
        linestyle=":",
        label="Tracker limit",
        zorder=4,
    )
    _add_rectangle_projection(
        zx_ax,
        safe,
        "x_w",
        "z_w",
        edgecolor=TRUE_SAFETY_VOLUME_COLOR,
        linewidth=TRUE_SAFETY_VOLUME_LINEWIDTH,
        linestyle=TRUE_SAFETY_VOLUME_LINESTYLE,
        label="True safety volume",
        zorder=5,
    )
    if data.scenario.wind_model is not None:
        _plot_fan_outlets_2d(xy_ax, _fan_outlet_centres(data.scenario.wind_model))
    _plot_start_end_projection(
        xy_ax,
        data.actual,
        STATE_NAMES.index("x_w"),
        STATE_NAMES.index("y_w"),
    )
    _plot_start_end_projection(
        zx_ax,
        data.actual,
        STATE_NAMES.index("x_w"),
        STATE_NAMES.index("z_w"),
    )

    xy_ax.set_xlim(*tracker["x_w"])
    xy_ax.set_ylim(*tracker["y_w"])
    zx_ax.set_xlim(*tracker["x_w"])
    zx_ax.set_ylim(*tracker["z_w"])
    style_geometry_axis(xy_ax, r"$x_w$ (m)", r"$y_w$ (m)", style, equal_aspect=True)
    style_geometry_axis(zx_ax, r"$x_w$ (m)", r"$z_w$ (m)", style, equal_aspect=False)
    _framed_figure_legend(
        fig,
        [xy_ax, zx_ax],
        style,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=4,
    )
    fig.tight_layout(rect=(0.0, 0.16, 1.0, 1.0))
    return fig


def _shrink_colorbar_axis(ax: mpl.axes.Axes, style: PlotStyle) -> None:
    pos = ax.get_position()
    height = pos.height * float(style.colorbar_vertical_shrink)
    bottom = pos.y0 + 0.5 * (pos.height - height)
    ax.set_position([pos.x0, bottom, pos.width, height])


# =============================================================================
# 5) Public Figure-Generation API
# =============================================================================
def load_scenario_plot_data(
    scenario_id: str,
    seed: int,
    output_root: str | Path | None = None,
    include_control_reference: bool = True,
    include_environment_reference: bool = True,
) -> PlotScenarioData:
    runtime = _runtime_objects(scenario_id, seed)
    paths = scenario_output_paths(scenario_id, seed, output_root)
    runner_root = paths.root_dir / "_run"
    # The existing runner expects metrics/log subfolders before governor CSV writes.
    (runner_root / "metrics").mkdir(parents=True, exist_ok=True)
    (runner_root / "logs").mkdir(parents=True, exist_ok=True)
    row = run_scenario(scenario_id, seed=seed, output_root=runner_root)
    row = _normalise_actual_outputs(paths, runner_root, scenario_id, seed, row)
    log_path = _actual_log_path(scenario_id, seed, output_root)
    if not log_path.exists():
        _write_status_manifest(
            scenario_id=scenario_id,
            seed=seed,
            row=row,
            output_paths=paths,
            status="no executed trajectory; plot skipped",
        )
        raise FileNotFoundError(f"Scenario log was not generated: {log_path}")
    actual = _load_actual_log(log_path)
    selected_name = str(row.get("selected_primitive") or row.get("primitive_name") or "")
    if not selected_name:
        raise ValueError("Scenario did not execute a primitive that can be plotted.")
    primitive = _select_executed_primitive(runtime.scenario, selected_name)

    references: list[TrajectorySeries] = []
    if include_control_reference:
        references.append(
            _reference_series(
                key="control_reference",
                label=TRAJECTORY_SPECS["control_reference"]["label"],
                runtime=runtime,
                primitive=primitive,
                scenario_id=scenario_id,
                seed=seed,
                paths=paths,
                use_environment=False,
            )
        )
    if include_environment_reference:
        references.append(
            _reference_series(
                key="environment_reference",
                label=TRAJECTORY_SPECS["environment_reference"]["label"],
                runtime=runtime,
                primitive=primitive,
                scenario_id=scenario_id,
                seed=seed,
                paths=paths,
                use_environment=True,
            )
        )
    return PlotScenarioData(
        scenario=runtime.scenario,
        actual_row=row,
        actual=actual,
        references=tuple(references),
        output_root=None if output_root is None else Path(output_root),
        seed=int(seed),
    )


def generate_scenario_figure(
    scenario_id: str,
    seed: int,
    output_root: str | Path | None = None,
    include_control_reference: bool = True,
    include_environment_reference: bool = True,
    save_png: bool = True,
    save_pdf: bool = False,
) -> dict[str, Path]:
    """Run or load a scenario result and save the five canonical figures."""
    data = load_scenario_plot_data(
        scenario_id=scenario_id,
        seed=seed,
        output_root=output_root,
        include_control_reference=include_control_reference,
        include_environment_reference=include_environment_reference,
    )
    style = PlotStyle()
    output_paths = scenario_output_paths(scenario_id, seed, output_root)
    builders = {
        "A": build_figure_a_trajectory_command_actuator,
        "B": build_figure_b_flight_rates,
        "C": build_figure_c_flight_state_alpha_beta,
        "D": build_figure_d_envelope_variables,
        "E": build_figure_e_2d_trajectory_geometry,
    }
    plot_paths: dict[str, Path] = {}
    if save_png or save_pdf:
        for label in FIGURE_ORDER:
            fig = builders[label](data, style=style)
            try:
                stem = FIGURE_STEMS[label]
                if save_png:
                    png_path = output_paths.root_dir / f"{stem}.png"
                    save_figure(fig, png_path, style)
                    plot_paths[f"{label}_png"] = png_path
                if save_pdf:
                    pdf_path = output_paths.root_dir / f"{stem}.pdf"
                    save_figure(fig, pdf_path, style)
                    plot_paths[f"{label}_pdf"] = pdf_path
            finally:
                plt.close(fig)
    _write_manifest(
        scenario_id=scenario_id,
        seed=seed,
        plot_paths=plot_paths,
        data=data,
        output_paths=output_paths,
    )
    return plot_paths


def _relative_to_result(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _manifest_figure_paths(
    plot_paths: dict[str, Path],
    root: Path,
) -> dict[str, dict[str, str]]:
    figures: dict[str, dict[str, str]] = {}
    for key, path in plot_paths.items():
        if "_" not in key:
            continue
        label, extension = key.split("_", 1)
        figures.setdefault(extension, {})[label] = _relative_to_result(path, root)
    return {
        extension: {
            label: values[label]
            for label in FIGURE_ORDER
            if label in values
        }
        for extension, values in sorted(figures.items())
    }


def _write_manifest(
    scenario_id: str,
    seed: int,
    plot_paths: dict[str, Path],
    data: PlotScenarioData,
    output_paths: ScenarioOutputPaths,
) -> None:
    manifest = {
        "scenario_id": scenario_id,
        "scenario_name": friendly_scenario_name(scenario_id),
        "seed": int(seed),
        "status": "plot generated",
        "selected_primitive": data.actual_row.get("selected_primitive"),
        "folder": output_paths.root_dir.name,
        "result_group": scenario_result_group(scenario_id),
        "case_id": friendly_scenario_slug(scenario_id),
        "case_path": (
            f"{scenario_result_group(scenario_id)}/"
            f"{friendly_scenario_slug(scenario_id)}/"
            f"{int(seed):03d}"
        ),
        "analysis_data": _digital_file_list(output_paths),
        "figures": _manifest_figure_paths(plot_paths, output_paths.root_dir),
    }
    output_paths.manifest.parent.mkdir(parents=True, exist_ok=True)
    output_paths.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _digital_file_list(output_paths: ScenarioOutputPaths) -> list[str]:
    csv_files = sorted(path for path in output_paths.root_dir.glob("*.csv") if path.is_file())
    return [_relative_to_result(path, output_paths.root_dir) for path in csv_files]


def _write_status_manifest(
    scenario_id: str,
    seed: int,
    row: dict[str, object],
    output_paths: ScenarioOutputPaths,
    status: str,
) -> None:
    manifest = {
        "scenario_id": scenario_id,
        "scenario_name": friendly_scenario_name(scenario_id),
        "seed": int(seed),
        "status": status,
        "selected_primitive": row.get("selected_primitive"),
        "folder": output_paths.root_dir.name,
        "result_group": scenario_result_group(scenario_id),
        "case_id": friendly_scenario_slug(scenario_id),
        "case_path": (
            f"{scenario_result_group(scenario_id)}/"
            f"{friendly_scenario_slug(scenario_id)}/"
            f"{int(seed):03d}"
        ),
        "analysis_data": _digital_file_list(output_paths),
        "figures": {},
    }
    output_paths.manifest.parent.mkdir(parents=True, exist_ok=True)
    output_paths.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--no-control-reference", action="store_true")
    parser.add_argument("--no-environment-reference", action="store_true")
    parser.add_argument("--no-png", action="store_true")
    parser.add_argument("--save-pdf", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        paths = generate_scenario_figure(
            scenario_id=args.scenario,
            seed=args.seed,
            output_root=None if args.output_root is None else Path(args.output_root),
            include_control_reference=not args.no_control_reference,
            include_environment_reference=not args.no_environment_reference,
            save_png=not args.no_png,
            save_pdf=args.save_pdf,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"plot skipped: {exc}")
        raise SystemExit(1) from exc
    for kind, path in paths.items():
        print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
