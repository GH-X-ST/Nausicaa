"""Composite E03 h30-memory case plot.

This figure follows the real-flight replay template: a 3D trajectory panel on
the left and compact state panels on the right.  The selected case is the R11
E03.2 L4 local fan-position uncertainty row where the h30 memory policy changes the
selected primitive sequence while staying close to the no-memory score.
"""

from __future__ import annotations

import ast
import json
import math
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
from matplotlib.ticker import FormatStrFormatter, FuncFormatter  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

try:  # noqa: E402
    from skimage.measure import marching_cubes  # type: ignore
except ImportError:  # pragma: no cover - optional plotting dependency
    marching_cubes = None


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "03_Control").is_dir() and (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(f"Could not resolve repository root from {start}")


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = _find_repo_root(SCRIPT_PATH.parent)
CONTROL_ROOT = REPO_ROOT / "03_Control"
for rel in ("03_Primitives", "04_Scenarios"):
    path = CONTROL_ROOT / rel
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from arena_contract import TRUE_SAFE_BOUNDS  # noqa: E402
from env_instance import environment_instance_for_mode, environment_metadata_from_instance  # noqa: E402
from env_surrogate import resolve_surrogate_binding, wind_field_for_binding  # noqa: E402
from run_changed_case_validation import R11_PROTOCOL  # noqa: E402
from run_repeated_launch_learning_curve import (  # noqa: E402
    _environment_randomisation_config_for_context,
    _scheduled_active_fan_count_for_context,
)


AXIS_EDGE_LW = 0.80
TICK_LABEL_FONTSIZE = 11
X_MIN, X_MAX = TRUE_SAFE_BOUNDS.x_w_m
Y_MIN, Y_MAX = TRUE_SAFE_BOUNDS.y_w_m
Z_MIN, Z_MAX = TRUE_SAFE_BOUNDS.z_w_m

FINAL_POLICY_COLORS = {
    "no_memory_baseline": "#24476f",
    "spatial_flow_belief_memory_h3": "#2e7d32",
    "spatial_flow_belief_memory_h10": "#d97706",
    "spatial_flow_belief_memory_h30": "#7e57c2",
    "directional_3d_residual_memory_h3": "#2e7d32",
    "directional_3d_residual_memory_h10": "#d97706",
    "directional_3d_residual_memory_h30": "#7e57c2",
}
PRIMITIVE_MARKER_EDGE_WIDTH = 0.20
PRIMITIVE_MARKER = "o"
PRIMITIVE_MARKER_LABEL = "primitive endpoint"

FAN_OUTLET_DIAMETER = 0.8
FAN_VERTICAL_OFFSET_M = 0.330
FAN_OUTLET_PLOT_Z_M = max(FAN_VERTICAL_OFFSET_M, Z_MIN)
FAN_OUTLET_EDGE_LW = 1.1
FAN_OUTLET_ALPHA = 0.60
FAN_OUTLET_DASH = (0, (2, 2))

ARENA_EDGE_COLOR = (0.12, 0.12, 0.12, 0.88)
ARENA_EDGE_LW = 0.85
ARENA_GRID_COLOR = (0.75, 0.75, 0.75, 0.35)
ARENA_GRID_LW = 0.45

UPDRAFT_ALPHA_EXP_RATE = 3.0
UPDRAFT_ALPHA_SCALE = 0.62
UPDRAFT_CONTOURF_MIN_FRAC = 0.10
UPDRAFT_CONTOURF_LEVEL_COUNT = 9
UPDRAFT_CONTOURF_MAX_Z_SLICES = 18
UPDRAFT_ISO_FRACTIONS = (0.25, 0.45, 0.65, 0.85)
UPDRAFT_SLICE_EDGE_COLOR = (0.0, 0.0, 0.0, 0.20)
UPDRAFT_SLICE_EDGE_LW = 0.2
UPDRAFT_MESH_LINEWIDTH = 0.05


R11_ROOT = CONTROL_ROOT / "05_Results" / "R11_validation" / "E03.2"
OUTPUT_DIR = CONTROL_ROOT / "A_figures"
OUTPUT_PATH = OUTPUT_DIR / "r11_e03_2_l4_case0082_h30_history_memory_no_memory_openloop_composite.png"

LIBRARY_SIZE_CASE_ID = "light_cluster"
ENVIRONMENT_BLOCK_ID = "r11_l4_local_fan_position_uncertainty"
OUTER_CASE_INDEX = 82
PRIMITIVE_PARTITIONS = ("c00003.csv.gz", "c00004.csv.gz")

POLICY_H30 = "spatial_flow_belief_memory_h30"
POLICY_NO_MEMORY = "no_memory_baseline"
POLICY_OPEN_LOOP = "open_loop_zero_command_baseline"
FINAL_POLICIES = (POLICY_H30, POLICY_NO_MEMORY, POLICY_OPEN_LOOP)

POLICY_LABELS = {
    POLICY_H30: "h30 memory",
    POLICY_NO_MEMORY: "no memory",
    POLICY_OPEN_LOOP: "open-loop baseline",
}
POLICY_STYLES = {
    POLICY_H30: "-",
    POLICY_NO_MEMORY: "--",
    POLICY_OPEN_LOOP: ":",
}
POLICY_WIDTHS = {
    POLICY_H30: 2.1,
    POLICY_NO_MEMORY: 1.65,
    POLICY_OPEN_LOOP: 1.85,
}
POLICY_COLORS = {
    POLICY_H30: FINAL_POLICY_COLORS[POLICY_H30],
    POLICY_NO_MEMORY: FINAL_POLICY_COLORS[POLICY_NO_MEMORY],
    POLICY_OPEN_LOOP: "#111111",
}

HISTORY_COLOR = "#C2B9D4"
HISTORY_ALPHA = 0.30
HISTORY_LINEWIDTH = 0.52
PRIMITIVE_MARKER_SIZE = 10.0
DPI = 600

# Figure, 3D-panel, and 2D-panel layout knobs.  These are intentionally
# top-level so thesis sizing and bounds can be tuned without changing plotting
# logic.
FIGURE_SIZE_IN = (15.0, 4.2)
FIGURE_SAVE_PAD_IN = 0.24
FIGURE_GRID = {
    "nrows": 1,
    "ncols": 2,
    "width_ratios": (0.30, 0.70),
}
FIGURE_GRID_MARGINS = {
    "left": 0.035,
    "right": 0.995,
    "top": 0.965,
    "bottom": 0.125,
    "wspace": 0.035,
    "hspace": 0.0,
}

PANEL_3D_LAYOUT = {
    "row": 0,
    "col": 0,
}
PANEL_3D_BOUNDS = {
    "x": TRUE_SAFE_BOUNDS.x_w_m,
    "y": TRUE_SAFE_BOUNDS.y_w_m,
    "z": TRUE_SAFE_BOUNDS.z_w_m,
}
PANEL_3D_TICKS = {
    "x": np.arange(X_MIN, X_MAX + 1e-9, 1.4),
    "y": np.arange(Y_MIN, Y_MAX + 1e-9, 1.2),
    "z": np.arange(Z_MIN, Z_MAX + 1e-9, 0.7),
}
PANEL_3D_VIEW = {
    "elev": 50,
    "azim": -150,
}
PANEL_3D_LABELPADS = {
    "x": 5,
    "y": 3,
    "z": 1,
}
PANEL_3D_AXIS_LABELSIZE = 11.0
PANEL_3D_TICK_LABELSIZE = TICK_LABEL_FONTSIZE + 1

PANEL_2D_LAYOUT = {
    "outer_row": 0,
    "outer_col": 1,
    "row_count": 3,
    "column_count": 4,
    "width_ratios": (1.0, 1.0, 1.0, 1.0),
    "wspace": 0.32,
    "hspace": 0.12,
}

# 2D state-panel format follows 01_Thermal/four_fan_annular_gaussian_bemt_heat_map_main.py.
STATE_AXIS_EDGE_LW = 0.80
STATE_GRID_COLOR = (0.85, 0.85, 0.85, 1.0)
STATE_GRID_LINEWIDTH = 0.4
STATE_TICK_WIDTH = 0.6
STATE_TICK_LENGTH = 1.8
STATE_TICK_LABELSIZE = 11.0
STATE_AXIS_LABELSIZE = 11.0
STATE_YLABEL_PAD = -1.0
STATE_COLUMN_TITLE_SIZE = 12.0
STATE_LEGEND_FONTSIZE = 9.0
STATE_XTICKS_S = (0.0, 0.5, 1.0, 1.3)
STATE_XLIM_S = (0.0, 1.30)
STATE_YLIMS: dict[str, tuple[float, float]] = {
    "u_b": (5.10, 5.50),
    "v_b": (-0.60, 0.90),
    "w_b": (-0.80, 0.50),
    "phi": (-40.0, 15.0),
    "theta": (-25.0, 3.0),
    "psi": (4.0, 32.0),
    "p_b": (-3.25, 0.75),
    "q_b": (-1.60, 1.20),
    "r_b": (-0.50, 1.05),
    "delta_a": (-7.0, 6.5),
    "delta_e": (-7.0, 9.0),
    "delta_r": (-17.0, 13.0),
}
STATE_XTICK_LABELS = ("0", "0.5", "1", "1.3")
STATE_VELOCITY_YTICK_FORMAT = "%.1f"
STATE_COMPACT_YTICK_COLUMNS = (1, 2, 3)
STATE_HIDE_YTICK_LABEL_COLUMNS: tuple[int, ...] = ()


@dataclass(frozen=True)
class UpdraftConfig:
    updraft_nx: int = 44
    updraft_ny: int = 28
    updraft_nz: int = 24


def main() -> int:
    primitive_rows = _read_primitive_rows()
    final_score = _read_final_score()
    case_row = _read_case_schedule_row()

    _apply_heatmap_2d_rcparams()
    fig = plt.figure(figsize=FIGURE_SIZE_IN)
    gs = fig.add_gridspec(
        FIGURE_GRID["nrows"],
        FIGURE_GRID["ncols"],
        width_ratios=FIGURE_GRID["width_ratios"],
        **FIGURE_GRID_MARGINS,
    )
    ax3d = fig.add_subplot(
        gs[PANEL_3D_LAYOUT["row"], PANEL_3D_LAYOUT["col"]],
        projection="3d",
    )
    gs2d = gs[PANEL_2D_LAYOUT["outer_row"], PANEL_2D_LAYOUT["outer_col"]].subgridspec(
        PANEL_2D_LAYOUT["row_count"],
        PANEL_2D_LAYOUT["column_count"],
        width_ratios=PANEL_2D_LAYOUT["width_ratios"],
        wspace=PANEL_2D_LAYOUT["wspace"],
        hspace=PANEL_2D_LAYOUT["hspace"],
    )
    mini_axes = [
        [
            fig.add_subplot(gs2d[r, c])
            for c in range(PANEL_2D_LAYOUT["column_count"])
        ]
        for r in range(PANEL_2D_LAYOUT["row_count"])
    ]

    env_meta = _draw_trajectory_panel(ax3d, primitive_rows, final_score, case_row)
    _draw_state_panels(mini_axes, primitive_rows)

    score_h30 = _score(final_score, POLICY_H30)
    score_no = _score(final_score, POLICY_NO_MEMORY)
    score_open = _score(final_score, POLICY_OPEN_LOOP)
    history_count = _history_episode_count(primitive_rows)
    delta = score_h30 - score_no

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTPUT_PATH,
        dpi=DPI,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=FIGURE_SAVE_PAD_IN,
    )
    plt.close(fig)

    metadata_path = OUTPUT_PATH.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "figure": OUTPUT_PATH.name,
                "r11_root": R11_ROOT.as_posix(),
                "library_size_case_id": LIBRARY_SIZE_CASE_ID,
                "environment_block_id": ENVIRONMENT_BLOCK_ID,
                "outer_case_index": OUTER_CASE_INDEX,
                "primitive_partitions": list(PRIMITIVE_PARTITIONS),
                "history_policy_id": POLICY_H30,
                "history_episode_count": history_count,
                "final_policies": list(FINAL_POLICIES),
                "launch_score_h30": score_h30,
                "launch_score_no_memory": score_no,
                "launch_score_h30_minus_no_memory": delta,
                "launch_score_open_loop": score_open,
                "compact_panel_layout": "3_component_rows_by_4_state_group_columns",
                "figure_size_in": list(FIGURE_SIZE_IN),
                "figure_save_pad_in": FIGURE_SAVE_PAD_IN,
                "figure_grid": {
                    key: list(value) if isinstance(value, tuple) else value
                    for key, value in FIGURE_GRID.items()
                },
                "figure_grid_margins": FIGURE_GRID_MARGINS,
                "panel_3d_layout": {
                    "row": PANEL_3D_LAYOUT["row"],
                    "col": PANEL_3D_LAYOUT["col"],
                },
                "panel_3d_bounds": {key: list(value) for key, value in PANEL_3D_BOUNDS.items()},
                "panel_3d_ticks": {key: [float(v) for v in value] for key, value in PANEL_3D_TICKS.items()},
                "panel_3d_view": PANEL_3D_VIEW,
                "panel_3d_labelpads": PANEL_3D_LABELPADS,
                "panel_3d_axis_labelsize": PANEL_3D_AXIS_LABELSIZE,
                "panel_3d_tick_labelsize": PANEL_3D_TICK_LABELSIZE,
                "panel_2d_layout": PANEL_2D_LAYOUT,
                "state_axis_style_reference": "01_Thermal/four_fan_annular_gaussian_bemt_heat_map_main.py",
                "state_axis_xlim_s": list(STATE_XLIM_S) if STATE_XLIM_S is not None else "auto",
                "state_axis_ylims": {key: list(value) for key, value in STATE_YLIMS.items()},
                "environment": env_meta,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )
    print(OUTPUT_PATH.as_posix())
    return 0


def _read_primitive_rows() -> pd.DataFrame:
    columns = [
        "library_size_case_id",
        "policy_id",
        "history_length",
        "adaptation_launch_index",
        "outer_case_index",
        "environment_block_id",
        "episode_id",
        "launch_role",
        "primitive_step_index",
        "initial_state_vector",
        "exit_state_vector",
        "initial_x_w",
        "initial_y_w",
        "initial_z_w",
        "initial_phi",
        "initial_theta",
        "initial_psi",
        "initial_u",
        "initial_v",
        "initial_w",
        "initial_p",
        "initial_q",
        "initial_r",
        "initial_delta_a",
        "initial_delta_e",
        "initial_delta_r",
        "rollout_absolute_start_time_s",
        "rollout_absolute_end_time_s",
        "termination_cause",
    ]
    frames: list[pd.DataFrame] = []
    table_root = R11_ROOT / "tables" / "primitive_execution_log"
    for name in PRIMITIVE_PARTITIONS:
        path = table_root / name
        frame = pd.read_csv(path, usecols=columns)
        mask = (
            frame["library_size_case_id"].astype(str).eq(LIBRARY_SIZE_CASE_ID)
            & frame["outer_case_index"].astype(int).eq(OUTER_CASE_INDEX)
            & frame["environment_block_id"].astype(str).eq(ENVIRONMENT_BLOCK_ID)
            & frame["policy_id"].astype(str).isin(FINAL_POLICIES)
        )
        frames.append(frame.loc[mask].copy())
    out = pd.concat(frames, ignore_index=True)
    if out.empty:
        raise RuntimeError("No primitive rows found for selected case.")
    return out


def _apply_heatmap_2d_rcparams() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": STATE_AXIS_LABELSIZE,
            "axes.titlesize": STATE_COLUMN_TITLE_SIZE,
            "xtick.labelsize": STATE_TICK_LABELSIZE,
            "ytick.labelsize": STATE_TICK_LABELSIZE,
            "legend.fontsize": STATE_LEGEND_FONTSIZE,
            "axes.edgecolor": "k",
            "axes.linewidth": STATE_AXIS_EDGE_LW,
            "patch.edgecolor": "k",
        }
    )


def _read_final_score() -> pd.DataFrame:
    frame = pd.read_csv(R11_ROOT / "metrics" / "final_launch_score.csv")
    if "launch_role" in frame.columns:
        frame = frame[frame["launch_role"].astype(str).eq("final_heldout")].copy()
    mask = (
        frame["library_size_case_id"].astype(str).eq(LIBRARY_SIZE_CASE_ID)
        & frame["outer_case_index"].astype(int).eq(OUTER_CASE_INDEX)
        & frame["environment_block_id"].astype(str).eq(ENVIRONMENT_BLOCK_ID)
        & frame["policy_id"].astype(str).isin(FINAL_POLICIES)
    )
    out = frame.loc[mask].copy()
    if len(out) < len(FINAL_POLICIES):
        raise RuntimeError("Final-score rows are incomplete for selected policies.")
    return out


def _read_case_schedule_row() -> dict[str, object]:
    schedule = pd.read_csv(R11_ROOT / "metrics" / "outer_case_schedule.csv")
    rows = schedule[
        schedule["outer_case_index"].astype(int).eq(OUTER_CASE_INDEX)
        & schedule["environment_block_id"].astype(str).eq(ENVIRONMENT_BLOCK_ID)
    ]
    if rows.empty:
        raise RuntimeError("Selected outer-case schedule row was not found.")
    return rows.iloc[0].to_dict()


def _draw_arena_box(ax) -> None:
    corners = {
        "000": (X_MIN, Y_MIN, Z_MIN),
        "100": (X_MAX, Y_MIN, Z_MIN),
        "010": (X_MIN, Y_MAX, Z_MIN),
        "110": (X_MAX, Y_MAX, Z_MIN),
        "001": (X_MIN, Y_MIN, Z_MAX),
        "101": (X_MAX, Y_MIN, Z_MAX),
        "011": (X_MIN, Y_MAX, Z_MAX),
        "111": (X_MAX, Y_MAX, Z_MAX),
    }
    edge_keys = (
        ("000", "100"),
        ("100", "110"),
        ("110", "010"),
        ("010", "000"),
        ("001", "101"),
        ("101", "111"),
        ("111", "011"),
        ("011", "001"),
        ("000", "001"),
        ("100", "101"),
        ("110", "111"),
        ("010", "011"),
    )
    for start_key, end_key in edge_keys:
        start = corners[start_key]
        end = corners[end_key]
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color=ARENA_EDGE_COLOR,
            linewidth=ARENA_EDGE_LW,
            zorder=0,
        )


def _draw_floor_grid(ax) -> None:
    x_ticks = np.arange(X_MIN, X_MAX + 1e-9, 1.4)
    y_ticks = np.arange(Y_MIN, Y_MAX + 1e-9, 1.2)
    for x_value in x_ticks:
        ax.plot(
            [x_value, x_value],
            [Y_MIN, Y_MAX],
            [Z_MIN, Z_MIN],
            color=ARENA_GRID_COLOR,
            linewidth=ARENA_GRID_LW,
            zorder=0,
        )
    for y_value in y_ticks:
        ax.plot(
            [X_MIN, X_MAX],
            [y_value, y_value],
            [Z_MIN, Z_MIN],
            color=ARENA_GRID_COLOR,
            linewidth=ARENA_GRID_LW,
            zorder=0,
        )


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
    config: UpdraftConfig,
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


def _draw_trajectory_panel(
    ax,
    primitive_rows: pd.DataFrame,
    final_score: pd.DataFrame,
    case_row: dict[str, object],
) -> dict[str, object]:
    ax.set_facecolor("white")
    ax.grid(True)
    _draw_arena_box(ax)
    _draw_floor_grid(ax)
    env_meta = _draw_environment_context(ax, case_row)
    _configure_3d_axis(ax)

    history_rows = primitive_rows[
        primitive_rows["policy_id"].astype(str).eq(POLICY_H30)
        & primitive_rows["launch_role"].astype(str).eq("history")
    ].copy()
    label_used = False
    for _, episode_rows in history_rows.groupby("episode_id", sort=True):
        points = _episode_points(episode_rows)
        if points.shape[0] < 2:
            continue
        ax.plot(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color=HISTORY_COLOR,
            linewidth=HISTORY_LINEWIDTH,
            alpha=HISTORY_ALPHA,
            label="h30 launch history" if not label_used else None,
            zorder=12,
        )
        label_used = True

    final_rows = primitive_rows[primitive_rows["launch_role"].astype(str).eq("final_heldout")].copy()
    for policy_id in FINAL_POLICIES:
        rows = final_rows[final_rows["policy_id"].astype(str).eq(policy_id)].copy()
        if rows.empty:
            continue
        points = _episode_points(rows)
        if points.shape[0] < 2:
            continue
        color = POLICY_COLORS[policy_id]
        ax.plot(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color=color,
            linestyle=POLICY_STYLES[policy_id],
            linewidth=POLICY_WIDTHS[policy_id],
            alpha=0.96,
            label=POLICY_LABELS[policy_id],
            zorder=26,
        )
        _draw_primitive_markers(
            ax,
            rows,
            color=color,
            marker_size=PRIMITIVE_MARKER_SIZE,
            alpha=0.62,
            zorder=28,
            include_labels=False,
        )
        marker = "o" if _success(final_score, policy_id) else "x"
        if marker == "o":
            ax.scatter(
                [points[-1, 0]],
                [points[-1, 1]],
                [points[-1, 2]],
                facecolors="none",
                edgecolors=color,
                linewidths=1.45,
                marker=marker,
                s=46,
                depthshade=False,
                zorder=29,
            )
        else:
            ax.scatter(
                [points[-1, 0]],
                [points[-1, 1]],
                [points[-1, 2]],
                color=color,
                marker=marker,
                s=46,
                depthshade=False,
                zorder=29,
            )

    handles, labels = ax.get_legend_handles_labels()
    unique: dict[str, object] = {}
    for handle, label in zip(handles, labels, strict=False):
        if label and label not in unique:
            unique[label] = handle
    legend = ax.legend(
        unique.values(),
        unique.keys(),
        loc="upper left",
        bbox_to_anchor=(0.03, 1.10),
        frameon=True,
        framealpha=1.0,
        edgecolor="black",
        fontsize=11,
        handlelength=1.8,
        borderpad=0.35,
        labelspacing=0.18,
    )
    if legend is not None:
        legend.get_frame().set_linewidth(AXIS_EDGE_LW)
    return env_meta


def _draw_environment_context(ax, case_row: dict[str, object]) -> dict[str, object]:
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
    environment_mode = str(case_row.get("environment_mode", "dry_air"))
    environment_seed = int(float(case_row.get("environment_seed", 0)))
    instance = environment_instance_for_mode(
        w_layer,
        environment_mode,
        environment_seed,
        randomisation_config=randomisation_config,
    )
    metadata = environment_metadata_from_instance(instance)
    binding = resolve_surrogate_binding(
        w_layer,
        metadata,
        repo_root=REPO_ROOT,
        randomisation_seed=environment_seed,
    )
    active_fan_positions_m = tuple(
        position
        for position, active in zip(binding.fan_positions_m, binding.active_fan_mask, strict=False)
        if bool(active)
    )
    _draw_environment_fan_outlets(ax, active_fan_positions_m)
    active_count = int(sum(bool(value) for value in binding.active_fan_mask))
    meta = {
        "environment_instance_id": instance.environment_id,
        "environment_mode": environment_mode,
        "W_layer": w_layer,
        "fan_count": int(binding.fan_count),
        "active_fan_count": int(active_count),
        "updraft_width_scale": float(binding.updraft_width_scale),
        "updraft_amplitude_scale": float(binding.updraft_amplitude_scale),
        "local_uncertainty_scale": float(binding.local_uncertainty_scale),
        "fan_positions_m": [[float(x), float(y)] for x, y in binding.fan_positions_m],
        "plotted_fan_positions_m": [[float(x), float(y)] for x, y in active_fan_positions_m],
        "fan_outlet_plot_policy": "active_fans_only_hide_no_updraft_outlets",
        "active_fan_mask": [bool(value) for value in binding.active_fan_mask],
    }
    if w_layer.upper() == "W0":
        return {**meta, "updraft_context_status": "dry_air_no_updraft", "updraft_max_m_s": 0.0}

    wind = wind_field_for_binding(binding, repo_root=REPO_ROOT)
    if wind is None:
        return {**meta, "updraft_context_status": "no_ready_wind_field", "updraft_max_m_s": 0.0}
    try:
        config = UpdraftConfig()
        x_vec, y_vec, z_vec, w_grid = _sample_updraft_volume(wind, config)
        w_max = float(np.nanmax(w_grid)) if w_grid.size else 0.0
        if w_max <= 1e-9:
            return {**meta, "updraft_context_status": "zero_updraft_volume", "updraft_max_m_s": 0.0}
        cmap_alpha = _build_alpha_cmap()
        norm = mpl.colors.Normalize(vmin=0.0, vmax=8.0, clip=True)
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
        return {
            **meta,
            "updraft_context_status": "seeded_full_domain_slices_and_isosurfaces",
            "updraft_max_m_s": w_max,
            "updraft_iso_surface_count": int(iso_count),
            "updraft_grid": [config.updraft_nx, config.updraft_ny, config.updraft_nz],
        }
    except Exception as exc:
        return {
            **meta,
            "updraft_context_status": f"blocked:{type(exc).__name__}",
            "updraft_max_m_s": 0.0,
        }


def _configure_3d_axis(ax) -> None:
    ax.set_xlim(*PANEL_3D_BOUNDS["x"])
    ax.set_ylim(*PANEL_3D_BOUNDS["y"])
    ax.set_zlim(*PANEL_3D_BOUNDS["z"])
    ax.set_xticks(PANEL_3D_TICKS["x"])
    ax.set_yticks(PANEL_3D_TICKS["y"])
    ax.set_zticks(PANEL_3D_TICKS["z"])
    ax.set_xlabel("$x$ (m)", labelpad=PANEL_3D_LABELPADS["x"], fontsize=PANEL_3D_AXIS_LABELSIZE)
    ax.set_ylabel("$y$ (m)", labelpad=PANEL_3D_LABELPADS["y"], fontsize=PANEL_3D_AXIS_LABELSIZE)
    ax.set_zlabel("$z$ (m)", labelpad=PANEL_3D_LABELPADS["z"], rotation=90, fontsize=PANEL_3D_AXIS_LABELSIZE)
    ax.zaxis.set_rotate_label(False)
    ax.tick_params(axis="x", which="major", labelsize=PANEL_3D_TICK_LABELSIZE)
    ax.tick_params(axis="y", which="major", labelsize=PANEL_3D_TICK_LABELSIZE)
    ax.tick_params(axis="z", which="major", labelsize=PANEL_3D_TICK_LABELSIZE)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        axis._axinfo["grid"]["color"] = (0.86, 0.86, 0.86, 1.0)
        axis._axinfo["grid"]["linewidth"] = 0.38
    try:
        ax.set_box_aspect(
            (
                PANEL_3D_BOUNDS["x"][1] - PANEL_3D_BOUNDS["x"][0],
                PANEL_3D_BOUNDS["y"][1] - PANEL_3D_BOUNDS["y"][0],
                PANEL_3D_BOUNDS["z"][1] - PANEL_3D_BOUNDS["z"][0],
            )
        )
    except AttributeError:
        pass
    ax.view_init(elev=PANEL_3D_VIEW["elev"], azim=PANEL_3D_VIEW["azim"])


def _draw_state_panels(axes: list[list[plt.Axes]], primitive_rows: pd.DataFrame) -> None:
    final_rows = primitive_rows[primitive_rows["launch_role"].astype(str).eq("final_heldout")].copy()
    traces = {
        policy_id: _boundary_trace(final_rows[final_rows["policy_id"].astype(str).eq(policy_id)])
        for policy_id in FINAL_POLICIES
    }
    panel_columns = [
        (
            "Body velocity",
            (("$u_b$", 6, 1.0), ("$v_b$", 7, 1.0), ("$w_b$", 8, 1.0)),
            r"m $\!$s$^{-1}$",
        ),
        (
            "Body attitude",
            (
                ("$\\phi$", 3, 180.0 / math.pi),
                ("$\\theta$", 4, 180.0 / math.pi),
                ("$\\psi$", 5, 180.0 / math.pi),
            ),
            "deg",
        ),
        (
            "Body angular rate",
            (("$p_b$", 9, 1.0), ("$q_b$", 10, 1.0), ("$r_b$", 11, 1.0)),
            r"rad $\!$s$^{-1}$",
        ),
        (
            "Controller deflection",
            (
                ("$\\delta_a$", 12, 180.0 / math.pi),
                ("$\\delta_e$", 13, 180.0 / math.pi),
                ("$\\delta_r$", 14, 180.0 / math.pi),
            ),
            "deg",
        ),
    ]

    for col_index, (column_label, specs, units) in enumerate(panel_columns):
        for row_index, (state_label, state_index, scale) in enumerate(specs):
            ax = axes[row_index][col_index]
            for policy_id in FINAL_POLICIES:
                trace = traces[policy_id]
                if trace["states"].size == 0:
                    continue
                ax.plot(
                    trace["time"],
                    trace["states"][:, state_index] * scale,
                    color=POLICY_COLORS[policy_id],
                    linestyle=POLICY_STYLES[policy_id],
                    linewidth=1.18 if policy_id != POLICY_H30 else 1.45,
                    alpha=0.95,
                    label=POLICY_LABELS[policy_id],
                )
            if row_index == 0:
                ax.set_title(column_label, fontsize=STATE_COLUMN_TITLE_SIZE, pad=13)
            ax.set_axisbelow(True)
            ax.grid(True, color=STATE_GRID_COLOR, linewidth=STATE_GRID_LINEWIDTH)
            if col_index in STATE_COMPACT_YTICK_COLUMNS:
                ax.yaxis.set_major_formatter(FuncFormatter(_compact_y_tick_label))
            else:
                ax.yaxis.set_major_formatter(FormatStrFormatter(STATE_VELOCITY_YTICK_FORMAT))
            ax.tick_params(
                axis="both",
                which="major",
                labelsize=STATE_TICK_LABELSIZE,
                width=STATE_TICK_WIDTH,
                length=STATE_TICK_LENGTH,
            )
            for spine in ax.spines.values():
                spine.set_edgecolor("k")
                spine.set_linewidth(STATE_AXIS_EDGE_LW)
            ax.set_ylabel(
                f"{state_label} ({units})",
                fontsize=STATE_AXIS_LABELSIZE,
                labelpad=STATE_YLABEL_PAD,
            )
            if col_index in STATE_HIDE_YTICK_LABEL_COLUMNS:
                ax.set_yticklabels([])
            _apply_custom_axis_limits(ax, state_label)
            _apply_time_ticks(ax)
            if row_index == len(specs) - 1:
                ax.set_xlabel("time (s)", fontsize=STATE_AXIS_LABELSIZE)
            else:
                ax.set_xticklabels([])


def _compact_y_tick_label(value: float, _position: int) -> str:
    label = f"{value:.1f}".rstrip("0").rstrip(".")
    return "0" if label == "-0" else label


def _apply_time_ticks(ax: plt.Axes) -> None:
    if STATE_XLIM_S is not None:
        xmin, xmax = STATE_XLIM_S
    else:
        xmin, xmax = ax.get_xlim()
        xmin = max(0.0, float(xmin))
        xmax = float(xmax)
    ticks = [tick for tick in STATE_XTICKS_S if xmin - 1e-9 <= float(tick) <= xmax + 1e-9]
    labels = [
        label
        for tick, label in zip(STATE_XTICKS_S, STATE_XTICK_LABELS, strict=False)
        if xmin - 1e-9 <= float(tick) <= xmax + 1e-9
    ]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(xmin, xmax)


def _apply_custom_axis_limits(ax: plt.Axes, state_label: str) -> None:
    key = (
        state_label.replace("$", "")
        .replace("\\", "")
        .replace("{", "")
        .replace("}", "")
        .strip()
    )
    if key in STATE_YLIMS:
        ax.set_ylim(*STATE_YLIMS[key])


def _boundary_trace(rows: pd.DataFrame) -> dict[str, np.ndarray]:
    if rows.empty:
        return {"time": np.empty(0), "states": np.empty((0, 15))}
    ordered = rows.sort_values("primitive_step_index").copy()
    times: list[float] = []
    states: list[list[float]] = []

    first = ordered.iloc[0]
    start_time = _to_float(first.get("rollout_absolute_start_time_s"), 0.0)
    first_state = _state_from_row(first, initial=True)
    if len(first_state) >= 15:
        times.append(start_time)
        states.append(first_state[:15])

    fallback_time = start_time
    for _, row in ordered.iterrows():
        end_time = _to_float(row.get("rollout_absolute_end_time_s"), float("nan"))
        if not math.isfinite(end_time):
            fallback_time += _to_float(row.get("rollout_duration_s"), 0.10)
            end_time = fallback_time
        exit_state = _parse_vector(row.get("exit_state_vector"))
        if len(exit_state) >= 15:
            times.append(float(end_time))
            states.append(exit_state[:15])

    if not states:
        return {"time": np.empty(0), "states": np.empty((0, 15))}
    order = np.argsort(np.asarray(times, dtype=float))
    return {
        "time": np.asarray(times, dtype=float)[order],
        "states": np.asarray(states, dtype=float)[order],
    }


def _state_from_row(row: pd.Series, *, initial: bool) -> list[float]:
    if initial:
        state = _parse_vector(row.get("initial_state_vector"))
        if len(state) >= 15:
            return state
        return [
            _to_float(row.get("initial_x_w"), np.nan),
            _to_float(row.get("initial_y_w"), np.nan),
            _to_float(row.get("initial_z_w"), np.nan),
            _to_float(row.get("initial_phi"), np.nan),
            _to_float(row.get("initial_theta"), np.nan),
            _to_float(row.get("initial_psi"), np.nan),
            _to_float(row.get("initial_u"), np.nan),
            _to_float(row.get("initial_v"), np.nan),
            _to_float(row.get("initial_w"), np.nan),
            _to_float(row.get("initial_p"), np.nan),
            _to_float(row.get("initial_q"), np.nan),
            _to_float(row.get("initial_r"), np.nan),
            _to_float(row.get("initial_delta_a"), np.nan),
            _to_float(row.get("initial_delta_e"), np.nan),
            _to_float(row.get("initial_delta_r"), np.nan),
        ]
    return _parse_vector(row.get("exit_state_vector"))


def _parse_vector(value: object) -> list[float]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(v) for v in value]
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, (list, tuple)):
        return []
    try:
        return [float(item) for item in parsed]
    except (TypeError, ValueError):
        return []


def _history_episode_count(primitive_rows: pd.DataFrame) -> int:
    rows = primitive_rows[
        primitive_rows["policy_id"].astype(str).eq(POLICY_H30)
        & primitive_rows["launch_role"].astype(str).eq("history")
    ]
    return int(rows["episode_id"].astype(str).nunique())


def _score(final_score: pd.DataFrame, policy_id: str) -> float:
    rows = final_score[final_score["policy_id"].astype(str).eq(policy_id)]
    if rows.empty:
        return float("nan")
    return float(rows.iloc[0]["launch_score"])


def _success(final_score: pd.DataFrame, policy_id: str) -> bool:
    rows = final_score[final_score["policy_id"].astype(str).eq(policy_id)]
    if rows.empty:
        return False
    value = rows.iloc[0].get("mission_success", rows.iloc[0].get("safe_success", False))
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "1.0", "true", "yes"}


def _to_float(value: object, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(out):
        return float(default)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
