"""Generate R9 fixed-case 3D history and paired-final trajectory figures."""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

try:  # noqa: E402
    from skimage.measure import marching_cubes  # type: ignore
except ImportError:  # pragma: no cover - optional plotting dependency
    marching_cubes = None

from run_thesis_3d_baseline_figure import (  # noqa: E402
    AXIS_EDGE_LW,
    AXIS_LABEL_FONTSIZE,
    DPI,
    FAN_OUTLET_ALPHA,
    FAN_OUTLET_DASH,
    FAN_OUTLET_DIAMETER,
    FAN_OUTLET_EDGE_LW,
    FAN_OUTLET_PLOT_Z_M,
    TICK_LABEL_FONTSIZE,
    TRACKER_LIMIT_BOUNDS,
    TRUE_SAFE_BOUNDS,
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


def _bootstrap_import_paths() -> None:
    control_root = Path(__file__).resolve().parents[1]
    for rel in ("03_Primitives", "04_Scenarios"):
        path = control_root / rel
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


_bootstrap_import_paths()

from env_instance import environment_instance_for_mode, environment_metadata_from_instance  # noqa: E402
from env_surrogate import resolve_surrogate_binding, wind_field_for_binding  # noqa: E402


FIGURE_RUN_VERSION = "r9_preflight_3d_paths_v2"
DEFAULT_INPUT_ROOT = Path("03_Control/05_Results/R9_test/A05")
DEFAULT_OUTPUT_ROOT = Path("03_Control/A_figures")
DEFAULT_RUN_LABEL = "01_r9_a05_preflight_paths"
DEFAULT_LIBRARY_SIZE_CASE = "balanced_cluster"
DEFAULT_HISTORY_POLICY = "spatial_flow_belief_memory_h10"
H30_HISTORY_POLICY = "spatial_flow_belief_memory_h30"
ENVIRONMENT_BLOCKS = ("no_updraft", "single_fan", "four_fan")
R9_FIGSIZE_3D = (10.8, 5.625)
R9_AXES_POSITION = [0.20, -0.08, 0.98, 1.14]
R9_LEGEND_FONTSIZE = 12.5
R9_LEGEND_ANCHOR = (0.67, 1.05)
FINAL_POLICY_ORDER = (
    "no_memory_baseline",
    "spatial_flow_belief_memory_h3",
    "spatial_flow_belief_memory_h10",
    "spatial_flow_belief_memory_h30",
    "directional_3d_residual_memory_h3",
    "directional_3d_residual_memory_h10",
    "directional_3d_residual_memory_h30",
)
FINAL_POLICY_LABELS = {
    "no_memory_baseline": "no memory",
    "spatial_flow_belief_memory_h3": "h3 memory",
    "spatial_flow_belief_memory_h10": "h10 memory",
    "spatial_flow_belief_memory_h30": "h30 memory",
    "directional_3d_residual_memory_h3": "h3 memory",
    "directional_3d_residual_memory_h10": "h10 memory",
    "directional_3d_residual_memory_h30": "h30 memory",
}
FINAL_POLICY_COLORS = {
    "no_memory_baseline": "#24476f",
    "spatial_flow_belief_memory_h3": "#2e7d32",
    "spatial_flow_belief_memory_h10": "#d97706",
    "spatial_flow_belief_memory_h30": "#7e57c2",
    "directional_3d_residual_memory_h3": "#2e7d32",
    "directional_3d_residual_memory_h10": "#d97706",
    "directional_3d_residual_memory_h30": "#7e57c2",
}
FINAL_POLICY_LINESTYLES = {
    "no_memory_baseline": "-",
    "spatial_flow_belief_memory_h3": ":",
    "spatial_flow_belief_memory_h10": "--",
    "spatial_flow_belief_memory_h30": "-.",
    "directional_3d_residual_memory_h3": ":",
    "directional_3d_residual_memory_h10": "--",
    "directional_3d_residual_memory_h30": "-.",
}
FINAL_POLICY_MARKER_SIZES = {
    "no_memory_baseline": 46,
    "spatial_flow_belief_memory_h3": 34,
    "spatial_flow_belief_memory_h10": 24,
    "spatial_flow_belief_memory_h30": 15,
    "directional_3d_residual_memory_h3": 34,
    "directional_3d_residual_memory_h10": 24,
    "directional_3d_residual_memory_h30": 15,
}
HISTORY_COLOR = "#5f6f82"
FINAL_HISTORY_COLOR = "#111111"
HISTORY_LINE_WIDTH = 0.45
HISTORY_FINAL_LINE_WIDTH = 1.25
FINAL_POLICY_LINE_WIDTH = 1.08
PRIMITIVE_MARKER_EDGE_WIDTH = 0.20
PRIMITIVE_MARKER_SIZE_HISTORY = 2.6
PRIMITIVE_MARKER_SIZE_FINAL = 5.6
PRIMITIVE_MARKER = "o"
PRIMITIVE_MARKER_LABEL = "primitive endpoint"
DEFAULT_UPDRAFT_NX = 56
DEFAULT_UPDRAFT_NY = 36
DEFAULT_UPDRAFT_NZ = 30
UPDRAFT_CBAR_X = 0.91
UPDRAFT_CBAR_Y = 0.16
UPDRAFT_CBAR_W = 0.018
UPDRAFT_CBAR_H = 0.70
UPDRAFT_CBAR_LABEL = r"$w$ (m $\!$s$^{-1}$)"
UPDRAFT_CBAR_VMIN = 0.0
UPDRAFT_CBAR_VMAX = 8.0
UPDRAFT_CBAR_TICK_STEP = 1.0
UPDRAFT_CBAR_TICK_FONTSIZE = 9
UPDRAFT_CBAR_LABEL_FONTSIZE = 10
UPDRAFT_ALPHA_EXP_RATE = 3.0
UPDRAFT_ALPHA_SCALE = 0.62
UPDRAFT_CONTOURF_MIN_FRAC = 0.10
UPDRAFT_CONTOURF_LEVEL_COUNT = 9
UPDRAFT_CONTOURF_MAX_Z_SLICES = 18
UPDRAFT_ISO_FRACTIONS = (0.25, 0.45, 0.65, 0.85)
UPDRAFT_ISO_METHOD = "marching_cubes" if marching_cubes is not None else "contourf_slice_fallback"
UPDRAFT_SLICE_EDGE_COLOR = (0.0, 0.0, 0.0, 0.20)
UPDRAFT_SLICE_EDGE_LW = 0.2
UPDRAFT_MESH_LINEWIDTH = 0.05


@dataclass(frozen=True)
class R9PreflightFigureConfig:
    input_root: Path = DEFAULT_INPUT_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    run_label: str = DEFAULT_RUN_LABEL
    library_size_case_id: str = DEFAULT_LIBRARY_SIZE_CASE
    history_policy_id: str = DEFAULT_HISTORY_POLICY
    environment_block_ids: tuple[str, ...] = ENVIRONMENT_BLOCKS
    outer_case_indices: tuple[int, ...] = ()
    updraft_nx: int = DEFAULT_UPDRAFT_NX
    updraft_ny: int = DEFAULT_UPDRAFT_NY
    updraft_nz: int = DEFAULT_UPDRAFT_NZ


def run_r9_preflight_3d_figures(config: R9PreflightFigureConfig) -> dict[str, object]:
    input_root = Path(config.input_root)
    run_root = Path(config.output_root) / str(config.run_label)
    for subdir in ("figures", "metrics", "manifests", "reports"):
        (run_root / subdir).mkdir(parents=True, exist_ok=True)

    primitive_log = _read_result_table(input_root, "primitive_execution_log")
    episode_summary = _read_result_table(input_root, "episode_summary")
    outer_schedule = _read_csv(input_root / "metrics" / "outer_case_schedule.csv")

    figure_rows: list[dict[str, object]] = []
    history_policy_ids = _ordered_unique((str(config.history_policy_id), H30_HISTORY_POLICY))
    figure_prefix = _figure_file_prefix(input_root)
    for environment_block_id in config.environment_block_ids:
        case_indices = _resolve_outer_case_indices(
            outer_schedule=outer_schedule,
            environment_block_id=environment_block_id,
            requested_indices=config.outer_case_indices,
        )
        include_case_suffix = bool(config.outer_case_indices) or len(case_indices) > 1
        for outer_case_index in case_indices:
            block_file_id = _figure_block_file_component(environment_block_id)
            case_suffix = f"_case{int(outer_case_index):04d}" if include_case_suffix else ""
            history_path = (
                run_root
                / "figures"
                / f"{figure_prefix}_{block_file_id}{case_suffix}_history_paths_3d.png"
            )
            history_h30_path = (
                run_root
                / "figures"
                / f"{figure_prefix}_{block_file_id}{case_suffix}_history_h30_paths_3d.png"
            )
            final_path = (
                run_root
                / "figures"
                / f"{figure_prefix}_{block_file_id}{case_suffix}_final_paired_paths_3d.png"
            )
            history_meta = _plot_history_paths(
                primitive_log=primitive_log,
                episode_summary=episode_summary,
                outer_schedule=outer_schedule,
                output_path=history_path,
                environment_block_id=environment_block_id,
                outer_case_index=outer_case_index,
                config=config,
                history_policy_id=history_policy_ids[0],
            )
            history_h30_meta = None
            if H30_HISTORY_POLICY in history_policy_ids[1:]:
                history_h30_meta = _plot_history_paths(
                    primitive_log=primitive_log,
                    episode_summary=episode_summary,
                    outer_schedule=outer_schedule,
                    output_path=history_h30_path,
                    environment_block_id=environment_block_id,
                    outer_case_index=outer_case_index,
                    config=config,
                    history_policy_id=H30_HISTORY_POLICY,
                )
            elif history_policy_ids[0] == H30_HISTORY_POLICY:
                history_h30_meta = history_meta
            final_meta = _plot_final_paired_paths(
                primitive_log=primitive_log,
                episode_summary=episode_summary,
                outer_schedule=outer_schedule,
                output_path=final_path,
                environment_block_id=environment_block_id,
                outer_case_index=outer_case_index,
                config=config,
            )
            figure_rows.append(history_meta)
            if history_h30_meta is not None and history_h30_meta is not history_meta:
                figure_rows.append(history_h30_meta)
            figure_rows.append(final_meta)

    figure_frame = pd.DataFrame(figure_rows)
    figure_frame.to_csv(run_root / "metrics" / "r9_preflight_figure_summary.csv", index=False)
    manifest = {
        "figure_run_version": FIGURE_RUN_VERSION,
        "status": "complete",
        "input_root": input_root.as_posix(),
        "run_root": run_root.as_posix(),
        "figure_file_prefix": figure_prefix,
        "library_size_case_id": str(config.library_size_case_id),
        "history_policy_id": str(config.history_policy_id),
        "history_policy_ids": history_policy_ids,
        "environment_blocks": list(config.environment_block_ids),
        "outer_case_indices": [int(v) for v in config.outer_case_indices],
        "final_policy_order": list(FINAL_POLICY_ORDER),
        "figsize_width_in": float(R9_FIGSIZE_3D[0]),
        "figsize_height_in": float(R9_FIGSIZE_3D[1]),
        "updraft_grid_nx": int(config.updraft_nx),
        "updraft_grid_ny": int(config.updraft_ny),
        "updraft_grid_nz": int(config.updraft_nz),
        "updraft_iso_surface_method": UPDRAFT_ISO_METHOD,
        "updraft_style_reference": "01_Thermal/four_fan_gp_3D.py centre slices plus isosurfaces",
        "plot_frame_boundary_name": TRUE_SAFE_BOUNDS.name,
        "plot_frame_x_w_m": list(TRUE_SAFE_BOUNDS.x_w_m),
        "plot_frame_y_w_m": list(TRUE_SAFE_BOUNDS.y_w_m),
        "plot_frame_z_w_m": list(TRUE_SAFE_BOUNDS.z_w_m),
        "tracker_limit_boundary_name": TRACKER_LIMIT_BOUNDS.name,
        "tracker_limit_x_w_m": list(TRACKER_LIMIT_BOUNDS.x_w_m),
        "tracker_limit_y_w_m": list(TRACKER_LIMIT_BOUNDS.y_w_m),
        "tracker_limit_z_w_m": list(TRACKER_LIMIT_BOUNDS.z_w_m),
        "true_safe_boundary_name": TRUE_SAFE_BOUNDS.name,
        "true_safe_x_w_m": list(TRUE_SAFE_BOUNDS.x_w_m),
        "true_safe_y_w_m": list(TRUE_SAFE_BOUNDS.y_w_m),
        "true_safe_z_w_m": list(TRUE_SAFE_BOUNDS.z_w_m),
        "primitive_marker": PRIMITIVE_MARKER,
        "figure_count": int(len(figure_rows)),
        "figures": [str(row["figure_path"]) for row in figure_rows],
        "claim_status": "r9_preflight_visualisation_only_no_memory_improvement_claim",
    }
    (run_root / "manifests" / "r9_preflight_3d_figures_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    _write_report(run_root, manifest, figure_frame)
    return manifest


def _ordered_unique(values: tuple[str, ...]) -> list[str]:
    ordered: list[str] = []
    for value in values:
        if value not in ordered:
            ordered.append(value)
    return ordered


def _history_policy_short_label(policy_id: str) -> str:
    text = str(policy_id)
    if text.endswith("_h30"):
        return "h30"
    if text.endswith("_h10"):
        return "h10"
    if text.endswith("_h3"):
        return "h3"
    return text


def _figure_file_prefix(input_root: Path) -> str:
    run_name = Path(input_root).name.strip().lower()
    if run_name.startswith("a") and run_name[1:].isdigit():
        return f"r9_{run_name}"
    safe_name = "".join(ch if ch.isalnum() else "_" for ch in run_name).strip("_")
    if safe_name.startswith("c") and len(safe_name) >= 3 and safe_name[1:3].isdigit():
        digits = []
        for ch in safe_name[1:]:
            if not ch.isdigit():
                break
            digits.append(ch)
        return f"r10_c{''.join(digits)}"
    return safe_name or "r9_preflight"


def _figure_block_file_component(environment_block_id: str) -> str:
    text = str(environment_block_id).strip().lower()
    aliases = {
        "targeted_memory_opportunity_arena_wide_four_fan": "targeted_four_fan",
    }
    if text in aliases:
        return aliases[text]
    safe = "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")
    return safe[:48] or "environment"


def _resolve_outer_case_indices(
    *,
    outer_schedule: pd.DataFrame,
    environment_block_id: str,
    requested_indices: tuple[int, ...],
) -> tuple[int, ...]:
    if requested_indices:
        return tuple(int(v) for v in requested_indices)
    if "outer_case_index" not in outer_schedule.columns:
        return (0,)
    schedule = outer_schedule[
        outer_schedule["environment_block_id"].astype(str) == str(environment_block_id)
    ]
    if schedule.empty:
        return (0,)
    values = sorted(int(float(v)) for v in schedule["outer_case_index"].dropna().unique())
    return tuple(values) or (0,)


def _plot_history_paths(
    *,
    primitive_log: pd.DataFrame,
    episode_summary: pd.DataFrame,
    outer_schedule: pd.DataFrame,
    output_path: Path,
    environment_block_id: str,
    outer_case_index: int,
    config: R9PreflightFigureConfig,
    history_policy_id: str,
) -> dict[str, object]:
    subset = primitive_log[
        (primitive_log["library_size_case_id"].astype(str) == str(config.library_size_case_id))
        & (primitive_log["environment_block_id"].astype(str) == str(environment_block_id))
        & (primitive_log["policy_id"].astype(str) == str(history_policy_id))
    ].copy()
    subset = _filter_outer_case(subset, outer_case_index)
    history = subset[subset["launch_role"].astype(str) == "history"]
    final = subset[subset["launch_role"].astype(str) == "final_heldout"]

    fig, ax = _new_baseline_axis()
    updraft_meta = _draw_updraft_context(
        ax=ax,
        outer_schedule=outer_schedule,
        environment_block_id=environment_block_id,
        outer_case_index=outer_case_index,
        config=config,
    )
    history_episode_ids = sorted(history["episode_id"].astype(str).unique())
    for episode_id in history_episode_ids:
        episode_rows = history[history["episode_id"].astype(str) == episode_id]
        points = _episode_points(episode_rows)
        if points.shape[0] < 2:
            continue
        ax.plot(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color=HISTORY_COLOR,
            alpha=0.34,
            linewidth=HISTORY_LINE_WIDTH,
            zorder=15,
        )
        _draw_primitive_markers(
            ax,
            episode_rows,
            color=HISTORY_COLOR,
            marker_size=PRIMITIVE_MARKER_SIZE_HISTORY,
            alpha=0.24,
            zorder=17,
            include_labels=False,
        )
        ax.scatter(
            [points[-1, 0]],
            [points[-1, 1]],
            [points[-1, 2]],
            color=HISTORY_COLOR,
            alpha=0.46,
            s=8,
            depthshade=False,
            zorder=16,
        )

    final_label_used = False
    for episode_id in sorted(final["episode_id"].astype(str).unique()):
        episode_rows = final[final["episode_id"].astype(str) == episode_id]
        final_points = _episode_points(episode_rows)
        if final_points.shape[0] < 2:
            continue
        ax.plot(
            final_points[:, 0],
            final_points[:, 1],
            final_points[:, 2],
            color=FINAL_HISTORY_COLOR,
            linewidth=HISTORY_FINAL_LINE_WIDTH,
            label=(
                f"final {_history_policy_short_label(history_policy_id)}"
                if not final_label_used
                else None
            ),
            zorder=25,
        )
        _draw_primitive_markers(
            ax,
            episode_rows,
            color=FINAL_HISTORY_COLOR,
            marker_size=PRIMITIVE_MARKER_SIZE_FINAL,
            alpha=0.74,
            zorder=27,
            include_labels=not final_label_used,
        )
        ax.scatter(
            [final_points[-1, 0]],
            [final_points[-1, 1]],
            [final_points[-1, 2]],
            color=FINAL_HISTORY_COLOR,
            s=22,
            depthshade=False,
            zorder=26,
        )
        final_label_used = True

    _add_legend(ax)
    _save_figure(fig, output_path)
    return {
        "figure_type": "history_paths",
        "environment_block_id": str(environment_block_id),
        "outer_case_index": int(outer_case_index),
        "figure_path": output_path.as_posix(),
        "history_episode_count": int(len(history_episode_ids)),
        "history_policy_id": str(history_policy_id),
        "final_policy_id": str(history_policy_id),
        **updraft_meta,
    }


def _plot_final_paired_paths(
    *,
    primitive_log: pd.DataFrame,
    episode_summary: pd.DataFrame,
    outer_schedule: pd.DataFrame,
    output_path: Path,
    environment_block_id: str,
    outer_case_index: int,
    config: R9PreflightFigureConfig,
) -> dict[str, object]:
    subset = primitive_log[
        (primitive_log["library_size_case_id"].astype(str) == str(config.library_size_case_id))
        & (primitive_log["environment_block_id"].astype(str) == str(environment_block_id))
        & (primitive_log["launch_role"].astype(str) == "final_heldout")
    ].copy()
    subset = _filter_outer_case(subset, outer_case_index)
    summary = episode_summary[
        (episode_summary["library_size_case_id"].astype(str) == str(config.library_size_case_id))
        & (episode_summary["environment_block_id"].astype(str) == str(environment_block_id))
        & (episode_summary["launch_role"].astype(str) == "final_heldout")
    ].copy()
    summary = _filter_outer_case(summary, outer_case_index)

    fig, ax = _new_baseline_axis()
    updraft_meta = _draw_updraft_context(
        ax=ax,
        outer_schedule=outer_schedule,
        environment_block_id=environment_block_id,
        outer_case_index=outer_case_index,
        config=config,
    )
    plotted_count = 0
    for policy_id in FINAL_POLICY_ORDER:
        rows = subset[subset["policy_id"].astype(str) == str(policy_id)]
        episode_ids = sorted(rows["episode_id"].astype(str).unique())
        label = FINAL_POLICY_LABELS.get(policy_id, policy_id)
        color = FINAL_POLICY_COLORS.get(policy_id, "#333333")
        label_used = False
        for episode_id in episode_ids:
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
                linewidth=FINAL_POLICY_LINE_WIDTH,
                alpha=0.84,
                label=label if not label_used else None,
                zorder=25,
            )
            _draw_primitive_markers(
                ax,
                episode_rows,
                color=color,
                marker_size=PRIMITIVE_MARKER_SIZE_FINAL,
                alpha=0.72,
                zorder=27,
                include_labels=not label_used,
            )
            marker = "o" if _policy_safe_success(summary, policy_id) else "x"
            marker_size = FINAL_POLICY_MARKER_SIZES.get(policy_id, 28)
            if marker == "o":
                ax.scatter(
                    [points[-1, 0]],
                    [points[-1, 1]],
                    [points[-1, 2]],
                    facecolors="none",
                    edgecolors=color,
                    linewidths=1.35,
                    marker=marker,
                    s=marker_size,
                    depthshade=False,
                    zorder=26,
                )
            else:
                ax.scatter(
                    [points[-1, 0]],
                    [points[-1, 1]],
                    [points[-1, 2]],
                    color=color,
                    marker=marker,
                    s=marker_size,
                    depthshade=False,
                    zorder=26,
                )
            plotted_count += 1
            label_used = True

    _add_legend(ax)
    _save_figure(fig, output_path)
    return {
        "figure_type": "final_paired_paths",
        "environment_block_id": str(environment_block_id),
        "outer_case_index": int(outer_case_index),
        "figure_path": output_path.as_posix(),
        "final_policy_count": int(plotted_count),
        **updraft_meta,
    }


def _new_baseline_axis():
    fig = plt.figure(figsize=R9_FIGSIZE_3D)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(True)
    _draw_arena_box(ax)
    _draw_floor_grid(ax)
    _configure_axis(ax)
    return fig, ax


def _draw_primitive_markers(
    ax,
    rows: pd.DataFrame,
    *,
    color: str,
    marker_size: float,
    alpha: float,
    zorder: int,
    include_labels: bool,
) -> None:
    if rows.empty:
        return
    ordered = rows.sort_values("primitive_step_index")
    label_used = False
    for _, row in ordered.iterrows():
        exit_state = _parse_state_vector(row.get("exit_state_vector", ""))
        if len(exit_state) < 3:
            continue
        label = None
        if include_labels and not label_used:
            label = PRIMITIVE_MARKER_LABEL
            label_used = True
        common = {
            "marker": PRIMITIVE_MARKER,
            "s": marker_size,
            "alpha": alpha,
            "linewidths": PRIMITIVE_MARKER_EDGE_WIDTH,
            "depthshade": False,
            "label": label,
            "zorder": zorder,
        }
        ax.scatter(
            [float(exit_state[0])],
            [float(exit_state[1])],
            [float(exit_state[2])],
            facecolors=color,
            edgecolors=color,
            **common,
        )


def _configure_axis(ax) -> None:
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_zlim(Z_MIN, Z_MAX)
    ax.set_xticks(np.arange(X_MIN, X_MAX + 1e-9, 1.4))
    ax.set_yticks(np.arange(Y_MIN, Y_MAX + 1e-9, 1.2))
    ax.set_zticks(np.arange(Z_MIN, Z_MAX + 1e-9, 0.7))
    ax.set_xlabel("$x$ (m)", labelpad=17)
    ax.set_ylabel("$y$ (m)", labelpad=10)
    ax.set_zlabel("$z$ (m)", labelpad=5, rotation=90)
    ax.zaxis.set_rotate_label(False)
    ax.xaxis.label.set_size(AXIS_LABEL_FONTSIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_FONTSIZE)
    ax.zaxis.label.set_size(AXIS_LABEL_FONTSIZE)
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
    ax.set_position(R9_AXES_POSITION)
    ax.set_anchor("W")


def _draw_updraft_context(
    *,
    ax,
    outer_schedule: pd.DataFrame,
    environment_block_id: str,
    outer_case_index: int,
    config: R9PreflightFigureConfig,
) -> dict[str, object]:
    schedule = outer_schedule[
        outer_schedule["environment_block_id"].astype(str) == str(environment_block_id)
    ]
    if "outer_case_index" in schedule.columns:
        schedule = schedule[
            schedule["outer_case_index"].map(lambda value: int(float(value)) == int(outer_case_index))
        ]
    if schedule.empty:
        return {"updraft_context_status": "missing_outer_schedule", "updraft_max_m_s": 0.0}
    row = schedule.iloc[0]
    w_layer = str(row.get("W_layer", "W0"))
    environment_mode = str(row.get("environment_mode", "dry_air"))
    environment_seed = int(float(row.get("environment_seed", 0)))
    if w_layer.upper() == "W0":
        _draw_environment_fan_outlets(ax, ())
        return {"updraft_context_status": "dry_air_no_updraft", "updraft_max_m_s": 0.0}
    try:
        instance = environment_instance_for_mode(
            w_layer,
            environment_mode,
            environment_seed,
        )
        metadata = environment_metadata_from_instance(instance)
        binding = resolve_surrogate_binding(
            w_layer,
            metadata,
            repo_root=Path(".").resolve(),
            randomisation_seed=environment_seed,
        )
        wind = wind_field_for_binding(binding, repo_root=Path(".").resolve())
        _draw_environment_fan_outlets(ax, binding.fan_positions_m)
        if wind is None:
            return {"updraft_context_status": "no_ready_wind_field", "updraft_max_m_s": 0.0}

        x_vec, y_vec, z_vec, w_grid = _sample_updraft_volume(wind, config)
        w_max = float(np.nanmax(w_grid)) if w_grid.size else 0.0
        if w_max <= 1e-9:
            return {"updraft_context_status": "zero_updraft_volume", "updraft_max_m_s": 0.0}

        cmap_alpha = _build_alpha_cmap()
        norm = mpl.colors.Normalize(vmin=UPDRAFT_CBAR_VMIN, vmax=UPDRAFT_CBAR_VMAX, clip=True)
        _draw_center_slices(
            ax,
            x_vec=x_vec,
            y_vec=y_vec,
            z_vec=z_vec,
            w_grid=w_grid,
            fan_positions=binding.fan_positions_m,
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
        _add_updraft_colorbar(ax.figure, cmap_alpha, norm)
        return {
            "updraft_context_status": "low_resolution_3d_slices_and_isosurfaces",
            "updraft_max_m_s": w_max,
            "updraft_grid_nx": int(config.updraft_nx),
            "updraft_grid_ny": int(config.updraft_ny),
            "updraft_grid_nz": int(config.updraft_nz),
            "updraft_iso_surface_count": int(iso_count),
            "updraft_iso_surface_method": UPDRAFT_ISO_METHOD,
            "updraft_style_reference": "01_Thermal/four_fan_gp_3D.py",
        }
    except Exception as exc:
        return {
            "updraft_context_status": f"blocked:{type(exc).__name__}",
            "updraft_max_m_s": 0.0,
        }


def _build_alpha_cmap() -> mcolors.ListedColormap:
    try:
        import cmocean  # type: ignore

        base_cmap = cmocean.cm.thermal
    except ImportError:
        base_cmap = mpl.colormaps["inferno"]
    colors = base_cmap(np.linspace(0.0, 1.0, 256))
    t_norm = np.linspace(0.0, 1.0, colors.shape[0])
    exp_scale = np.exp(UPDRAFT_ALPHA_EXP_RATE * t_norm)
    exp_full = np.exp(UPDRAFT_ALPHA_EXP_RATE)
    alpha = (exp_scale - 1.0) / (exp_full - 1.0)
    alpha[0] = 0.0
    alpha[-1] = 1.0
    colors[:, 3] = alpha
    colors[:, 3] *= UPDRAFT_ALPHA_SCALE
    return mcolors.ListedColormap(colors)


def _sample_updraft_volume(
    wind,
    config: R9PreflightFigureConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_vec = np.linspace(X_MIN, X_MAX, max(8, int(config.updraft_nx)))
    y_vec = np.linspace(Y_MIN, Y_MAX, max(7, int(config.updraft_ny)))
    z_vec = np.linspace(Z_MIN, Z_MAX, max(6, int(config.updraft_nz)))
    x_grid, y_grid, z_grid = np.meshgrid(x_vec, y_vec, z_vec, indexing="ij")
    points = np.column_stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()])
    wind_values = np.asarray(wind(points), dtype=float)
    if wind_values.ndim != 2 or wind_values.shape[1] < 3:
        raise ValueError("wind field must return an N x 3 array")
    w_grid = wind_values[:, 2].reshape(x_grid.shape)
    w_grid = np.maximum(np.nan_to_num(w_grid, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    return x_vec, y_vec, z_vec, w_grid


def _draw_center_slices(
    ax,
    *,
    x_vec: np.ndarray,
    y_vec: np.ndarray,
    z_vec: np.ndarray,
    w_grid: np.ndarray,
    fan_positions: tuple[tuple[float, float], ...],
    cmap_alpha: mcolors.ListedColormap,
    norm: mpl.colors.Normalize,
) -> None:
    ix_center, iy_center = _slice_indices_for_fan_centre(x_vec, y_vec, fan_positions)

    y_mesh_x, z_mesh_x = np.meshgrid(y_vec, z_vec, indexing="ij")
    x_mesh_x = np.full_like(y_mesh_x, x_vec[ix_center])
    w_slice_x = w_grid[ix_center, :, :]
    ax.plot_surface(
        x_mesh_x,
        y_mesh_x,
        z_mesh_x,
        facecolors=cmap_alpha(norm(w_slice_x)),
        rstride=1,
        cstride=1,
        linewidth=0.0,
        antialiased=False,
        shade=False,
        zorder=5,
    )

    x_mesh_y, z_mesh_y = np.meshgrid(x_vec, z_vec, indexing="ij")
    y_mesh_y = np.full_like(x_mesh_y, y_vec[iy_center])
    w_slice_y = w_grid[:, iy_center, :]
    ax.plot_surface(
        x_mesh_y,
        y_mesh_y,
        z_mesh_y,
        facecolors=cmap_alpha(norm(w_slice_y)),
        rstride=1,
        cstride=1,
        linewidth=0.0,
        antialiased=False,
        shade=False,
        zorder=5,
    )

    ax.plot(
        np.full_like(y_vec, x_vec[ix_center]),
        y_vec,
        np.full_like(y_vec, Z_MIN),
        color=UPDRAFT_SLICE_EDGE_COLOR,
        linewidth=UPDRAFT_SLICE_EDGE_LW,
        zorder=6,
    )
    ax.plot(
        x_vec,
        np.full_like(x_vec, y_vec[iy_center]),
        np.full_like(x_vec, Z_MIN),
        color=UPDRAFT_SLICE_EDGE_COLOR,
        linewidth=UPDRAFT_SLICE_EDGE_LW,
        zorder=6,
    )

def _slice_indices_for_fan_centre(
    x_vec: np.ndarray,
    y_vec: np.ndarray,
    fan_positions: tuple[tuple[float, float], ...],
) -> tuple[int, int]:
    positions = tuple(fan_positions or ())
    if not positions:
        return int(len(x_vec) // 2), int(len(y_vec) // 2)
    centre = np.mean(np.asarray(positions, dtype=float), axis=0)
    ix_center = int(np.argmin(np.abs(x_vec - float(centre[0]))))
    iy_center = int(np.argmin(np.abs(y_vec - float(centre[1]))))
    return ix_center, iy_center


def _draw_updraft_isosurfaces(
    ax,
    *,
    x_vec: np.ndarray,
    y_vec: np.ndarray,
    z_vec: np.ndarray,
    w_grid: np.ndarray,
    cmap_alpha: mcolors.ListedColormap,
    norm: mpl.colors.Normalize,
) -> int:
    if marching_cubes is None:
        return _draw_contourf_slice_surfaces(
            ax,
            x_vec=x_vec,
            y_vec=y_vec,
            z_vec=z_vec,
            w_grid=w_grid,
            cmap_alpha=cmap_alpha,
            norm=norm,
        )
    w_max = float(np.nanmax(w_grid)) if w_grid.size else 0.0
    if w_max <= 1e-9:
        return 0
    spacing = (
        float(x_vec[1] - x_vec[0]) if x_vec.size > 1 else 1.0,
        float(y_vec[1] - y_vec[0]) if y_vec.size > 1 else 1.0,
        float(z_vec[1] - z_vec[0]) if z_vec.size > 1 else 1.0,
    )
    iso_count = 0
    for fraction in UPDRAFT_ISO_FRACTIONS:
        level = float(fraction * w_max)
        if level <= float(np.nanmin(w_grid)) or level >= w_max:
            continue
        try:
            verts, faces, _, _ = marching_cubes(w_grid, level=level, spacing=spacing)
        except (RuntimeError, ValueError):
            continue
        verts[:, 0] += float(x_vec[0])
        verts[:, 1] += float(y_vec[0])
        verts[:, 2] += float(z_vec[0])
        mesh = Poly3DCollection(verts[faces], linewidths=UPDRAFT_MESH_LINEWIDTH, zorder=10)
        mesh.set_facecolor(cmap_alpha(norm(level)))
        mesh.set_edgecolor("none")
        ax.add_collection3d(mesh)
        iso_count += 1
    return iso_count


def _draw_contourf_slice_surfaces(
    ax,
    *,
    x_vec: np.ndarray,
    y_vec: np.ndarray,
    z_vec: np.ndarray,
    w_grid: np.ndarray,
    cmap_alpha: mcolors.ListedColormap,
    norm: mpl.colors.Normalize,
) -> int:
    if min(x_vec.size, y_vec.size, z_vec.size) < 3:
        return 0
    w_max = float(np.nanmax(w_grid)) if w_grid.size else 0.0
    if w_max <= 1e-9:
        return 0

    x_mesh, y_mesh = np.meshgrid(x_vec, y_vec, indexing="ij")
    z_indices = np.linspace(
        0,
        z_vec.size - 1,
        min(UPDRAFT_CONTOURF_MAX_Z_SLICES, z_vec.size),
    ).astype(int)
    levels = np.linspace(
        float(UPDRAFT_CONTOURF_MIN_FRAC * w_max),
        float(w_max),
        int(UPDRAFT_CONTOURF_LEVEL_COUNT),
    )
    plotted_slice_count = 0
    for k in z_indices:
        layer = w_grid[:, :, int(k)]
        if float(np.nanmax(layer)) <= float(levels[0]):
            continue
        ax.contourf(
            x_mesh,
            y_mesh,
            layer,
            levels=levels,
            zdir="z",
            offset=float(z_vec[int(k)]),
            cmap=cmap_alpha,
            norm=norm,
            antialiased=True,
            zorder=9,
        )
        plotted_slice_count += 1
    return plotted_slice_count


def _add_updraft_colorbar(
    fig,
    cmap_alpha: mcolors.ListedColormap,
    norm: mpl.colors.Normalize,
) -> None:
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_alpha)
    mappable.set_array([])
    cax = fig.add_axes([UPDRAFT_CBAR_X, UPDRAFT_CBAR_Y, UPDRAFT_CBAR_W, UPDRAFT_CBAR_H])
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(UPDRAFT_CBAR_LABEL, fontsize=UPDRAFT_CBAR_LABEL_FONTSIZE)
    cbar.set_ticks(np.arange(UPDRAFT_CBAR_VMIN, UPDRAFT_CBAR_VMAX + 1e-9, UPDRAFT_CBAR_TICK_STEP))
    cbar.ax.tick_params(width=0.6, length=2, labelsize=UPDRAFT_CBAR_TICK_FONTSIZE)
    cbar.outline.set_linewidth(AXIS_EDGE_LW)
    cbar.outline.set_edgecolor("k")
    cbar.ax.set_frame_on(True)


def _draw_environment_fan_outlets(ax, fan_positions: tuple[tuple[float, float], ...]) -> None:
    positions = tuple((float(x), float(y)) for x, y in tuple(fan_positions or ()))
    if not positions:
        return
    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    for idx, (fx, fy) in enumerate(positions):
        circle_x = fx + 0.5 * FAN_OUTLET_DIAMETER * np.cos(theta)
        circle_y = fy + 0.5 * FAN_OUTLET_DIAMETER * np.sin(theta)
        circle_z = FAN_OUTLET_PLOT_Z_M * np.ones_like(theta)
        ax.plot(
            circle_x,
            circle_y,
            circle_z,
            color=(0.0, 0.0, 0.0, FAN_OUTLET_ALPHA),
            linewidth=FAN_OUTLET_EDGE_LW,
            linestyle=FAN_OUTLET_DASH,
            label="Fan outlet" if idx == 0 else None,
            zorder=7,
        )


def _episode_points(rows: pd.DataFrame) -> np.ndarray:
    if rows.empty:
        return np.empty((0, 3), dtype=float)
    ordered = rows.sort_values("primitive_step_index")
    points: list[list[float]] = []
    first = ordered.iloc[0]
    points.append(
        [
            float(first.get("initial_x_w", np.nan)),
            float(first.get("initial_y_w", np.nan)),
            float(first.get("initial_z_w", np.nan)),
        ]
    )
    for _, row in ordered.iterrows():
        exit_state = _parse_state_vector(row.get("exit_state_vector", ""))
        if len(exit_state) >= 3:
            points.append([float(exit_state[0]), float(exit_state[1]), float(exit_state[2])])
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return np.empty((0, 3), dtype=float)
    return arr[np.all(np.isfinite(arr), axis=1)]


def _parse_state_vector(value: object) -> list[float]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(v) for v in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, (list, tuple)):
        return []
    out: list[float] = []
    for item in parsed:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            return []
    return out


def _policy_safe_success(summary: pd.DataFrame, policy_id: str) -> bool:
    rows = summary[summary["policy_id"].astype(str) == str(policy_id)]
    if rows.empty:
        return False
    row = rows.iloc[0]
    if "mission_success" in rows.columns:
        return bool(row.get("mission_success", False))
    return bool(row.get("safe_success", False))


def _add_legend(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    unique_handles = []
    unique_labels = []
    seen: set[str] = set()
    for handle, label in zip(handles, labels, strict=False):
        if not label or label.startswith("_") or label in seen:
            continue
        unique_handles.append(handle)
        unique_labels.append(label)
        seen.add(label)
    if not unique_handles:
        return
    leg = ax.legend(
        unique_handles,
        unique_labels,
        loc="upper left",
        bbox_to_anchor=R9_LEGEND_ANCHOR,
        bbox_transform=ax.figure.transFigure,
        frameon=True,
        framealpha=1.0,
        edgecolor="black",
        fontsize=R9_LEGEND_FONTSIZE,
        handlelength=1.2,
        borderpad=0.35,
        labelspacing=0.16,
        markerscale=0.72,
    )
    if leg is not None:
        leg.get_frame().set_linewidth(AXIS_EDGE_LW)


def _save_figure(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.stem}.tmp.{os.getpid()}{output_path.suffix}")
    try:
        fig.savefig(tmp_path, dpi=DPI, facecolor="white", bbox_inches="tight", pad_inches=0.35)
        _replace_output_file(tmp_path, output_path)
    finally:
        plt.close(fig)


def _replace_output_file(tmp_path: Path, output_path: Path) -> None:
    try:
        tmp_path.replace(output_path)
        return
    except PermissionError as first_error:
        pass

    try:
        if output_path.exists():
            output_path.chmod(0o666)
            output_path.unlink()
        tmp_path.replace(output_path)
    except PermissionError as second_error:
        raise PermissionError(
            "Could not overwrite the figure PNG. Close any image preview/editor "
            f"and let OneDrive finish syncing, then rerun. Target: {output_path}; "
            f"temporary file kept at: {tmp_path}"
        ) from second_error
    except OSError as second_error:
        raise OSError(
            "Could not replace the figure PNG after writing the temporary file. "
            f"Target: {output_path}; temporary file kept at: {tmp_path}"
        ) from second_error


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _read_result_table(input_root: Path, table_name: str) -> pd.DataFrame:
    table_dir = input_root / "tables" / table_name
    if table_dir.exists():
        files = sorted(table_dir.glob("*.csv")) + sorted(table_dir.glob("*.csv.gz"))
        if files:
            frames = [pd.read_csv(path) for path in files]
            return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    metric_path = input_root / "metrics" / f"{table_name}.csv"
    if metric_path.exists():
        return pd.read_csv(metric_path)
    legacy_path = input_root / "tables" / table_name / "c00000.csv.gz"
    if legacy_path.exists():
        return pd.read_csv(legacy_path)
    raise FileNotFoundError(table_dir)


def _filter_outer_case(frame: pd.DataFrame, outer_case_index: int) -> pd.DataFrame:
    if frame.empty or "outer_case_index" not in frame.columns:
        return frame.copy()
    mask = frame["outer_case_index"].map(lambda value: int(float(value)) == int(outer_case_index))
    return frame[mask].copy()


def _write_report(run_root: Path, manifest: dict[str, object], figure_frame: pd.DataFrame) -> None:
    lines = [
        "# R9 Preflight 3D Figures",
        "",
        f"This run visualises fixed-case R9 behaviour from `{manifest['input_root']}` using the shared four-fan 3D plotting baseline.",
        "The updraft context uses low-resolution 3D centre slices and transparent isosurfaces, "
        "following the `01_Thermal/four_fan_gp_3D.py` visual style.",
        "",
        f"- Input root: `{manifest['input_root']}`",
        f"- Figure count: {manifest['figure_count']}",
        f"- Library case: `{manifest['library_size_case_id']}`",
        f"- History policy: `{manifest['history_policy_id']}`",
        f"- Plot frame: `{manifest['plot_frame_boundary_name']}` "
        f"x={manifest['plot_frame_x_w_m']} m, y={manifest['plot_frame_y_w_m']} m, "
        f"z={manifest['plot_frame_z_w_m']} m",
        f"- Updraft grid: {manifest['updraft_grid_nx']} x "
        f"{manifest['updraft_grid_ny']} x {manifest['updraft_grid_nz']}",
        f"- Updraft surface method: `{UPDRAFT_ISO_METHOD}`",
        f"- Primitive markers: `{manifest['primitive_marker']}` endpoint circles",
        "- Claim status: R9 preflight visualisation only; no memory-improvement claim.",
        "",
        "## Figures",
        "",
    ]
    for _, row in figure_frame.iterrows():
        case_text = ""
        if "outer_case_index" in row and pd.notna(row["outer_case_index"]):
            case_text = f" / case {int(float(row['outer_case_index']))}"
        lines.append(
            f"- `{row['figure_path']}`: {row['figure_type']} / "
            f"{row['environment_block_id']}{case_text}"
        )
    lines.append("")
    (run_root / "reports" / "r9_preflight_3d_figures_report.md").write_text(
        "\n".join(lines),
        encoding="ascii",
    )


def _parse_args() -> R9PreflightFigureConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-label", default=DEFAULT_RUN_LABEL)
    parser.add_argument("--library-size-case-id", default=DEFAULT_LIBRARY_SIZE_CASE)
    parser.add_argument("--history-policy-id", default=DEFAULT_HISTORY_POLICY)
    parser.add_argument(
        "--environment-block-ids",
        default=",".join(ENVIRONMENT_BLOCKS),
        help="Comma-separated environment_block_id list to plot.",
    )
    parser.add_argument(
        "--outer-case-indices",
        default="",
        help="Optional comma-separated outer_case_index list. Defaults to all cases in each block.",
    )
    parser.add_argument("--updraft-nx", type=int, default=DEFAULT_UPDRAFT_NX)
    parser.add_argument("--updraft-ny", type=int, default=DEFAULT_UPDRAFT_NY)
    parser.add_argument("--updraft-nz", type=int, default=DEFAULT_UPDRAFT_NZ)
    args = parser.parse_args()
    return R9PreflightFigureConfig(
        input_root=args.input_root,
        output_root=args.output_root,
        run_label=str(args.run_label),
        library_size_case_id=str(args.library_size_case_id),
        history_policy_id=str(args.history_policy_id),
        environment_block_ids=_parse_environment_block_ids(str(args.environment_block_ids)),
        outer_case_indices=_parse_outer_case_indices(str(args.outer_case_indices)),
        updraft_nx=int(args.updraft_nx),
        updraft_ny=int(args.updraft_ny),
        updraft_nz=int(args.updraft_nz),
    )


def _parse_environment_block_ids(value: str) -> tuple[str, ...]:
    block_ids = tuple(part.strip() for part in str(value).split(",") if part.strip())
    return block_ids or ENVIRONMENT_BLOCKS


def _parse_outer_case_indices(value: str) -> tuple[int, ...]:
    indices: list[int] = []
    for part in str(value).split(","):
        text = part.strip()
        if not text:
            continue
        indices.append(int(text))
    return tuple(indices)


if __name__ == "__main__":
    result = run_r9_preflight_3d_figures(_parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
