"""Generate the empty four-fan 3D arena baseline for thesis figures."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FormatStrFormatter  # noqa: E402


def _bootstrap_import_paths() -> None:
    control_root = Path(__file__).resolve().parents[1]
    scenarios = control_root / "04_Scenarios"
    if str(scenarios) not in sys.path:
        sys.path.insert(0, str(scenarios))


_bootstrap_import_paths()

from arena_contract import TRACKER_LIMIT_BOUNDS, TRUE_SAFE_BOUNDS  # noqa: E402


FIGURE_RUN_VERSION = "thesis_3d_baseline_four_fan_annular_gp_style_v1"
DEFAULT_OUTPUT_ROOT = Path("03_Control/A_figures")
DEFAULT_RUN_LABEL = "00_baseline_3d"

# Style copied from the 01_Thermal four-fan annular-GP 3D figure family.
FIGSIZE_3D = (10.8, 4.8)
DPI = 600
BASELINE_AXES_POSITION = [0.06, -0.08, 0.98, 1.14]
BASELINE_LEGEND_ANCHOR = (0.66, 0.84)
AXIS_EDGE_LW = 0.80
AXIS_LABEL_FONTSIZE = 12
TICK_LABEL_FONTSIZE = 11
LEGEND_FONTSIZE = 9.0
VIEW_ELEV = 20
VIEW_AZIM = -120

TRACKER_X_MIN, TRACKER_X_MAX = TRACKER_LIMIT_BOUNDS.x_w_m
TRACKER_Y_MIN, TRACKER_Y_MAX = TRACKER_LIMIT_BOUNDS.y_w_m
TRACKER_Z_MIN, TRACKER_Z_MAX = TRACKER_LIMIT_BOUNDS.z_w_m
X_MIN, X_MAX = TRUE_SAFE_BOUNDS.x_w_m
Y_MIN, Y_MAX = TRUE_SAFE_BOUNDS.y_w_m
Z_MIN, Z_MAX = TRUE_SAFE_BOUNDS.z_w_m
TRUE_SAFE_X_MIN, TRUE_SAFE_X_MAX = TRUE_SAFE_BOUNDS.x_w_m
TRUE_SAFE_Y_MIN, TRUE_SAFE_Y_MAX = TRUE_SAFE_BOUNDS.y_w_m
TRUE_SAFE_Z_MIN, TRUE_SAFE_Z_MAX = TRUE_SAFE_BOUNDS.z_w_m

FAN_OUTLET_POINTS = (
    (3.0, 3.6),
    (5.4, 3.6),
    (5.4, 1.2),
    (3.0, 1.2),
)
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
TRUE_SAFE_EDGE_COLOR = (0.10, 0.10, 0.10, 0.62)
TRUE_SAFE_EDGE_LW = 0.70
TRUE_SAFE_EDGE_DASH = (0, (4, 2))


@dataclass(frozen=True)
class BaselineFigureConfig:
    output_root: Path = DEFAULT_OUTPUT_ROOT
    run_label: str = DEFAULT_RUN_LABEL


def run_thesis_3d_baseline_figure(config: BaselineFigureConfig) -> dict[str, object]:
    run_root = Path(config.output_root) / str(config.run_label)
    for subdir in ("figures", "metrics", "manifests", "reports"):
        (run_root / subdir).mkdir(parents=True, exist_ok=True)

    figure_path = run_root / "figures" / "four_fan_empty_3d_baseline.png"
    _plot_empty_four_fan_baseline(figure_path)

    standard = _figure_standard_row(figure_path)
    pd.DataFrame([standard]).to_csv(run_root / "metrics" / "figure_standard.csv", index=False)

    manifest = {
        "figure_run_version": FIGURE_RUN_VERSION,
        "status": "complete",
        "run_root": run_root.as_posix(),
        "figure_path": figure_path.as_posix(),
        "source_style_reference": "01_Thermal/four_fan_annular_gp_3D.py via four_fan_gp_3D.py",
        "figure_role": "empty_3d_arena_baseline_for_thesis_plotting",
        "claim_status": "plotting_standard_only_no_control_or_memory_claim",
        **standard,
    }
    (run_root / "manifests" / "figure_baseline_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    _write_report(run_root, manifest)
    return manifest


def _plot_empty_four_fan_baseline(output_path: Path) -> None:
    fig = plt.figure(figsize=FIGSIZE_3D)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(True)

    _draw_arena_box(ax)
    _draw_floor_grid(ax)
    _draw_fan_outlets(ax)

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

    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        axis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 1.0)
        axis._axinfo["grid"]["linewidth"] = 0.4

    leg = ax.legend(
        loc="upper left",
        bbox_to_anchor=BASELINE_LEGEND_ANCHOR,
        bbox_transform=fig.transFigure,
        frameon=True,
        framealpha=1.0,
        edgecolor="black",
        fontsize=LEGEND_FONTSIZE,
        handlelength=1.2,
        borderpad=0.35,
        labelspacing=0.16,
    )
    if leg is not None:
        leg.get_frame().set_linewidth(AXIS_EDGE_LW)

    try:
        ax.set_box_aspect((X_MAX - X_MIN, Y_MAX - Y_MIN, Z_MAX - Z_MIN))
    except AttributeError:
        pass
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    ax.set_position(BASELINE_AXES_POSITION)
    ax.set_anchor("W")

    fig.savefig(output_path, dpi=DPI, facecolor="white", bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


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


def _draw_true_safe_box(ax) -> None:
    corners = {
        "000": (TRUE_SAFE_X_MIN, TRUE_SAFE_Y_MIN, TRUE_SAFE_Z_MIN),
        "100": (TRUE_SAFE_X_MAX, TRUE_SAFE_Y_MIN, TRUE_SAFE_Z_MIN),
        "010": (TRUE_SAFE_X_MIN, TRUE_SAFE_Y_MAX, TRUE_SAFE_Z_MIN),
        "110": (TRUE_SAFE_X_MAX, TRUE_SAFE_Y_MAX, TRUE_SAFE_Z_MIN),
        "001": (TRUE_SAFE_X_MIN, TRUE_SAFE_Y_MIN, TRUE_SAFE_Z_MAX),
        "101": (TRUE_SAFE_X_MAX, TRUE_SAFE_Y_MIN, TRUE_SAFE_Z_MAX),
        "011": (TRUE_SAFE_X_MIN, TRUE_SAFE_Y_MAX, TRUE_SAFE_Z_MAX),
        "111": (TRUE_SAFE_X_MAX, TRUE_SAFE_Y_MAX, TRUE_SAFE_Z_MAX),
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
            color=TRUE_SAFE_EDGE_COLOR,
            linewidth=TRUE_SAFE_EDGE_LW,
            linestyle=TRUE_SAFE_EDGE_DASH,
            zorder=1,
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


def _draw_fan_outlets(ax) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    for idx, (fx, fy) in enumerate(FAN_OUTLET_POINTS):
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
            zorder=5,
        )


def _figure_standard_row(figure_path: Path) -> dict[str, object]:
    return {
        "figure_run_version": FIGURE_RUN_VERSION,
        "figure_file": figure_path.as_posix(),
        "figsize_width_in": float(FIGSIZE_3D[0]),
        "figsize_height_in": float(FIGSIZE_3D[1]),
        "dpi": int(DPI),
        "x_min_m": float(X_MIN),
        "x_max_m": float(X_MAX),
        "y_min_m": float(Y_MIN),
        "y_max_m": float(Y_MAX),
        "z_min_m": float(Z_MIN),
        "z_max_m": float(Z_MAX),
        "outer_boundary_name": TRUE_SAFE_BOUNDS.name,
        "tracker_limit_boundary_name": TRACKER_LIMIT_BOUNDS.name,
        "true_safe_boundary_name": TRUE_SAFE_BOUNDS.name,
        "true_safe_x_min_m": float(TRUE_SAFE_X_MIN),
        "true_safe_x_max_m": float(TRUE_SAFE_X_MAX),
        "true_safe_y_min_m": float(TRUE_SAFE_Y_MIN),
        "true_safe_y_max_m": float(TRUE_SAFE_Y_MAX),
        "true_safe_z_min_m": float(TRUE_SAFE_Z_MIN),
        "true_safe_z_max_m": float(TRUE_SAFE_Z_MAX),
        "view_elev_deg": float(VIEW_ELEV),
        "view_azim_deg": float(VIEW_AZIM),
        "fan_vertical_offset_m": float(FAN_VERTICAL_OFFSET_M),
        "fan_outlet_plot_z_m": float(FAN_OUTLET_PLOT_Z_M),
        "fan_outlet_diameter_m": float(FAN_OUTLET_DIAMETER),
        "fan_outlet_count": int(len(FAN_OUTLET_POINTS)),
        "tracker_limit_x_min_m": float(TRACKER_X_MIN),
        "tracker_limit_x_max_m": float(TRACKER_X_MAX),
        "tracker_limit_y_min_m": float(TRACKER_Y_MIN),
        "tracker_limit_y_max_m": float(TRACKER_Y_MAX),
        "tracker_limit_z_min_m": float(TRACKER_Z_MIN),
        "tracker_limit_z_max_m": float(TRACKER_Z_MAX),
    }


def _write_report(run_root: Path, manifest: dict[str, object]) -> None:
    lines = [
        "# Thesis 3D Baseline Figure",
        "",
        "This run defines the empty 3D arena baseline for later thesis figures.",
        "It uses the four-fan annular-GP visual convention from `01_Thermal` "
        "but plots no thermal field and no trajectory.",
        "",
        f"- Figure: `{manifest['figure_path']}`",
        f"- Plot frame: `{TRUE_SAFE_BOUNDS.name}` x=[{X_MIN:.1f}, {X_MAX:.1f}] m, "
        f"y=[{Y_MIN:.1f}, {Y_MAX:.1f}] m, z=[{Z_MIN:.1f}, {Z_MAX:.1f}] m",
        f"- Tracker limit provenance: `{TRACKER_LIMIT_BOUNDS.name}` "
        f"x=[{TRACKER_X_MIN:.1f}, {TRACKER_X_MAX:.1f}] m, "
        f"y=[{TRACKER_Y_MIN:.1f}, {TRACKER_Y_MAX:.1f}] m, "
        f"z=[{TRACKER_Z_MIN:.1f}, {TRACKER_Z_MAX:.1f}] m",
        f"- Fan outlet rings are drawn at z={FAN_OUTLET_PLOT_Z_M:.2f} m for visibility; "
        f"physical outlet height is z={FAN_VERTICAL_OFFSET_M:.2f} m.",
        f"- View: elevation {VIEW_ELEV:.1f} deg, azimuth {VIEW_AZIM:.1f} deg",
        "- Claim status: plotting standard only; no controller, memory, or validation claim.",
        "",
    ]
    (run_root / "reports" / "figure_baseline_report.md").write_text(
        "\n".join(lines),
        encoding="ascii",
    )


def _parse_args() -> BaselineFigureConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-label", default=DEFAULT_RUN_LABEL)
    args = parser.parse_args()
    return BaselineFigureConfig(output_root=args.output_root, run_label=str(args.run_label))


if __name__ == "__main__":
    result = run_thesis_3d_baseline_figure(_parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
