from __future__ import annotations

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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    from skimage.measure import marching_cubes
except ImportError:  # pragma: no cover - optional thermal-style plume surfaces
    marching_cubes = None

from plot_style import (
    CBAR_LABEL,
    FAN_OUTLET_ALPHA,
    FAN_OUTLET_DASH,
    FAN_OUTLET_DIAMETER_M,
    FAN_OUTLET_EDGE_LW,
    FAN_OUTLET_Z_M,
    PLOT_VMAX,
    PLOT_VMIN,
    TRAJECTORY_SPECS,
    PlotStyle,
    build_alpha_cmap,
    framed_legend,
    save_figure,
    style_3d_axis,
    style_command_axis,
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
from latency import CommandToSurfaceConfig, CommandToSurfaceLayer, LatencyEnvelope  # noqa: E402
from linearisation import INPUT_NAMES, STATE_NAMES, linearise_trim  # noqa: E402
from primitive import build_primitive_context  # noqa: E402
from rollout import RolloutConfig, RolloutResult, simulate_primitive, write_log  # noqa: E402
from run_one import run_scenario  # noqa: E402
from scenarios import ScenarioDefinition, build_scenario  # noqa: E402
from updraft_models import FOUR_FAN_CENTERS_XY, SINGLE_FAN_CENTER_XY  # noqa: E402


RESULTS_ROOT_DIR = "flight_case_results"
DIGITAL_DIR = "analysis_data"
PLOTS_DIR = "figures"

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
}

SCENARIO_SLUGS = {
    "s0_no_wind": "baseline_no_wind_glide",
    "s1_latency_nominal_no_wind": "latency_nominal_bank_reversal_no_wind",
    "s1_latency_robust_upper_no_wind": "latency_upper_bank_reversal_no_wind",
    "s6_single_gaussian_var": "updraft_single_fan_gaussian_glide",
    "s6_four_gaussian_var": "updraft_four_fan_gaussian_glide",
    "s7_single_annular_gp": "updraft_single_fan_annular_gp_glide",
    "s7_four_annular_gp": "updraft_four_fan_annular_gp_glide",
    "s11_governor_rejection": "governor_low_altitude_rejection",
    "s4_full_nominal_glide_no_wind": "primitive_nominal_glide_no_wind",
    "s4_full_bank_reversal_left_no_wind": "primitive_bank_reversal_left_no_wind",
    "s4_full_bank_reversal_right_no_wind": "primitive_bank_reversal_right_no_wind",
    "s4_full_recovery_no_wind": "primitive_recovery_no_wind",
    "s4_latency_low_nominal_glide": "latency_low_nominal_glide_no_wind",
    "s4_latency_nominal_nominal_glide": "latency_nominal_glide_no_wind",
    "s4_latency_high_nominal_glide": "latency_high_nominal_glide_no_wind",
    "s4_latency_low_bank_reversal_left": "latency_low_bank_reversal_left",
    "s4_latency_nominal_bank_reversal_left": "latency_nominal_bank_reversal_left",
    "s4_latency_high_bank_reversal_left": "latency_high_bank_reversal_left",
    "s4_latency_low_recovery": "latency_low_recovery",
    "s4_latency_nominal_recovery": "latency_nominal_recovery",
    "s4_latency_high_recovery": "latency_high_recovery",
    "s4_latency_robust_upper_bank_reversal_left": "latency_upper_bank_reversal_left",
    "s4_gaussian_single_panel": "updraft_single_fan_gaussian_panel_wind",
    "s4_gaussian_single_cg": "updraft_single_fan_gaussian_centre_wind",
    "s4_gaussian_four_panel": "updraft_four_fan_gaussian_panel_wind",
    "s4_annular_single_panel": "updraft_single_fan_annular_gp_panel_wind",
    "s4_annular_single_cg": "updraft_single_fan_annular_gp_centre_wind",
    "s4_annular_four_panel": "updraft_four_fan_annular_gp_panel_wind",
    "s4_gaussian_single_panel_randomised": "updraft_single_fan_gaussian_randomised",
    "s4_governor_selection": "governor_recovery_selection_high_bank",
}


@dataclass(frozen=True)
class TrajectorySeries:
    key: str
    label: str
    times_s: np.ndarray
    states: np.ndarray
    desired_commands_rad: np.ndarray
    target_commands_rad: np.ndarray


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
    digital_dir: Path
    plots_dir: Path
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


def scenario_output_paths(
    scenario_id: str,
    seed: int,
    output_root: str | Path | None,
) -> ScenarioOutputPaths:
    if output_root is None:
        base = REPO_ROOT / "03_Control" / "05_Results"
    else:
        base = Path(output_root)
    root = base / RESULTS_ROOT_DIR / f"{friendly_scenario_slug(scenario_id)}_seed_{int(seed):03d}"
    digital_dir = root / DIGITAL_DIR
    plots_dir = root / PLOTS_DIR
    return ScenarioOutputPaths(
        root_dir=root,
        digital_dir=digital_dir,
        plots_dir=plots_dir,
        actual_log=digital_dir / "actual_rollout.csv",
        actual_metrics=digital_dir / "actual_metrics.csv",
        manifest=digital_dir / "manifest.json",
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
    log_path = paths.digital_dir / f"{key}_rollout.csv"
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
    _write_single_row(paths.digital_dir / f"{key}_metrics.csv", result.metrics)
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
    )


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
    source.replace(destination)


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
    _move_if_present(raw_candidates, paths.digital_dir / "governor_candidates.csv")
    _move_if_present(raw_rejections, paths.digital_dir / "governor_rejections.csv")
    _move_if_present(raw_metrics, paths.actual_metrics)
    _remove_empty_dir(runner_root / "logs")
    _remove_empty_dir(runner_root / "metrics")
    _remove_empty_dir(runner_root)

    candidate_path = paths.digital_dir / "governor_candidates.csv"
    rejection_path = paths.digital_dir / "governor_rejections.csv"
    friendly_row = dict(row)
    friendly_row["case_name"] = friendly_scenario_name(scenario_id)
    friendly_row["case_folder"] = paths.root_dir.name
    friendly_row["scenario_name"] = friendly_scenario_name(scenario_id)
    # Stored metrics reference files within analysis_data, not temporary runner paths.
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
        color=(0.0, 0.0, 0.0, 0.45),
        linewidth=0.9,
        linestyle="--",
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


# =============================================================================
# 4) Composite Figure Builder
# =============================================================================
def _plot_trajectory(ax: mpl.axes.Axes, series: TrajectorySeries) -> None:
    spec = TRAJECTORY_SPECS[series.key]
    ax.plot(
        series.states[:, 0],
        series.states[:, 1],
        series.states[:, 2],
        color=spec["color"],
        linestyle=spec["linestyle"],
        linewidth=spec["linewidth"],
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
        color="#2CA02C",
        s=26,
        marker="o",
        label="Start",
        zorder=20,
    )
    ax.scatter(
        [end[0]],
        [end[1]],
        [end[2]],
        color="#111111",
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
    desired_deg = np.rad2deg(actual.desired_commands_rad)
    target_deg = np.rad2deg(actual.target_commands_rad)
    for idx, ax in enumerate(axes):
        ax.plot(
            actual.times_s,
            desired_deg[:, idx],
            color="#1F77B4",
            linewidth=1.2,
            linestyle="--",
            label="Desired",
        )
        ax.plot(
            actual.times_s,
            target_deg[:, idx],
            color="#111111",
            linewidth=1.2,
            linestyle="-",
            label="Command path",
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
                loc="lower left",
                bbox_to_anchor=(0.0, 1.03),
                ncol=2,
            )


def build_composite_figure(
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
    """Run or load a scenario result and save the composite figure."""
    data = load_scenario_plot_data(
        scenario_id=scenario_id,
        seed=seed,
        output_root=output_root,
        include_control_reference=include_control_reference,
        include_environment_reference=include_environment_reference,
    )
    style = PlotStyle()
    fig = build_composite_figure(data, style=style)
    output_paths = scenario_output_paths(scenario_id, seed, output_root)
    stem = "flight_trajectory_and_control_commands"
    plot_paths: dict[str, Path] = {}
    if save_png:
        png_path = output_paths.plots_dir / f"{stem}.png"
        save_figure(fig, png_path, style)
        plot_paths["png"] = png_path
    if save_pdf:
        pdf_path = output_paths.plots_dir / f"{stem}.pdf"
        save_figure(fig, pdf_path, style)
        plot_paths["pdf"] = pdf_path
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
        "analysis_data": _digital_file_list(output_paths),
        "figures": {
            key: _relative_to_result(path, output_paths.root_dir)
            for key, path in plot_paths.items()
        },
    }
    output_paths.manifest.parent.mkdir(parents=True, exist_ok=True)
    output_paths.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _digital_file_list(output_paths: ScenarioOutputPaths) -> list[str]:
    digital_files = sorted(
        path
        for path in output_paths.digital_dir.glob("*.csv")
        if path.is_file()
    )
    return [_relative_to_result(path, output_paths.root_dir) for path in digital_files]


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
        "analysis_data": _digital_file_list(output_paths),
        "figures": {},
    }
    output_paths.manifest.parent.mkdir(parents=True, exist_ok=True)
    output_paths.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
