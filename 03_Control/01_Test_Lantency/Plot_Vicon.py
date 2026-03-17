"""
Plot 3D Vicon trajectories from an exported See_Vicon workbook.

For each workbook:
    1. Read all subject sheets and the optional metadata sheet.
    2. Build a 3D trajectory plot in configured room bounds, if available.
    3. Build a second 3D trajectory plot with auto-tight bounds.
"""

###### Initialization

### Imports
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d.art3d import Line3DCollection

try:
    import cmocean
except ImportError:
    cmocean = None


### User settings
MAKE_PLOTS = True

WORKBOOK_XLSX = Path("B_See_Vicon/20260317_174456_Test.xlsx")
OUT_DIR = Path("B_See_Vicon")

SAVE_PLOTS = True
SHOW_PLOTS = False

ROOM_BOUNDS_SUFFIX = "_RoomBounds.png"
AUTO_BOUNDS_SUFFIX = "_AutoBounds.png"

# Fixed axis size with tight, dynamic export around labels and legend.
FIGURE_SCALE = 2.0
AXES_WIDTH_INCHES = 4.25 * FIGURE_SCALE
AXES_HEIGHT_INCHES = 2.75 * FIGURE_SCALE
FIG_MARGIN_LEFT_INCHES = 0.60 * FIGURE_SCALE
FIG_MARGIN_RIGHT_INCHES = 1.40
FIG_MARGIN_BOTTOM_INCHES = 0.32 * FIGURE_SCALE
FIG_MARGIN_TOP_INCHES = 0.06 * FIGURE_SCALE
EXPORT_PAD_INCHES = 0.02

# Typography
FONT_NAME = "DejaVu Sans"
AXIS_LABEL_FONTSIZE = 10
TICK_LABEL_FONTSIZE = 9
LEGEND_FONTSIZE = 8.5

# Legend layout
LEGEND_LOC = "upper right"
LEGEND_BBOX_TO_ANCHOR = (0.75, 0.72)
LEGEND_HANDLE_LENGTH = 1.5
LEGEND_BORDERPAD = 0.5
LEGEND_LABEL_SPACING = 0.2

# Axes and export styling
AXIS_EDGE_LW = 0.80
TRAJECTORY_LINE_LW = 1.50
BOUNDS_LINE_LW = 0.80
GRID_COLOR = (0.85, 0.85, 0.85, 1.0)
BOUNDS_COLOR = (0.0, 0.0, 0.0, 1.0)
VIEW_ELEVATION_DEG = 30.0
VIEW_AZIMUTH_DEG = -120.0
EXPORT_DPI = 600

# Auto-bound padding
AUTO_PADDING_FRAC = 0.05
AUTO_PADDING_MIN_M = 0.05

# Workbook layout
METADATA_SHEET = "Metadata"
ROOM_BOUNDS_KEY = "Room Bounds (m)"
REQUIRED_SUBJECT_COLUMNS = ("x_m", "y_m", "z_m", "is_occluded")

# Trajectory colors
SINGLE_LINE_THERMAL_POS = 0.66
MULTI_LINE_HEX_COLORS = (
    "#44035b",
    "#404185",
    "#31688e",
    "#1f918d",
    "#38b775",
    "#90d543",
    "#f8e630",
)


### Helpers
FLOAT_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
THERMAL_ANCHOR_COLORS = np.array(
    [
        [0.084521, 0.200528, 0.466051],
        [0.300558, 0.217076, 0.622225],
        [0.458841, 0.288918, 0.572709],
        [0.610871, 0.348396, 0.535604],
        [0.769168, 0.397026, 0.470814],
        [0.915881, 0.468902, 0.348558],
        [0.983448, 0.607335, 0.244967],
        [0.975436, 0.783249, 0.258553],
    ],
    dtype=float,
)
MULTI_LINE_RGB_COLORS = np.array(
    [mpl.colors.to_rgb(hex_color) for hex_color in MULTI_LINE_HEX_COLORS],
    dtype=float,
)


def configure_matplotlib_style() -> None:
    """
    Apply figure-wide typography settings.
    """
    mpl.rcParams["font.family"] = FONT_NAME
    mpl.rcParams["mathtext.fontset"] = "dejavusans"


def create_figure_with_fixed_axes_size(
    figure_title: str,
) -> Tuple[plt.Figure, mpl.axes.Axes]:
    """
    Create a figure with a fixed-size 3D axis box.
    """
    figure_width = (
        FIG_MARGIN_LEFT_INCHES + AXES_WIDTH_INCHES + FIG_MARGIN_RIGHT_INCHES
    )
    figure_height = (
        FIG_MARGIN_BOTTOM_INCHES + AXES_HEIGHT_INCHES + FIG_MARGIN_TOP_INCHES
    )

    fig = plt.figure(figsize=(figure_width, figure_height))
    fig.patch.set_facecolor("white")

    try:
        fig.canvas.manager.set_window_title(figure_title)
    except Exception:
        pass

    axes_position = [
        FIG_MARGIN_LEFT_INCHES / figure_width,
        FIG_MARGIN_BOTTOM_INCHES / figure_height,
        AXES_WIDTH_INCHES / figure_width,
        AXES_HEIGHT_INCHES / figure_height,
    ]
    ax = fig.add_axes(axes_position, projection="3d")
    ax.set_facecolor("white")

    try:
        ax.set_proj_type("persp")
    except Exception:
        pass

    return fig, ax


def load_workbook_contents(
    xlsx_path: Path,
) -> Tuple[List[Tuple[str, pd.DataFrame]], Optional[pd.DataFrame]]:
    """
    Load subject sheets and the optional metadata sheet.
    """
    if not xlsx_path.is_file():
        raise FileNotFoundError(f"Workbook not found: '{xlsx_path}'.")

    workbook = pd.ExcelFile(xlsx_path)
    subject_sheet_names = [name for name in workbook.sheet_names if name != METADATA_SHEET]
    if len(subject_sheet_names) == 0:
        raise ValueError(f"No subject sheets were found in workbook '{xlsx_path}'.")

    subjects = []
    for sheet_name in subject_sheet_names:
        samples = pd.read_excel(workbook, sheet_name=sheet_name)
        subjects.append((sheet_name, samples))

    metadata_df = None
    if METADATA_SHEET in workbook.sheet_names:
        metadata_df = pd.read_excel(workbook, sheet_name=METADATA_SHEET, header=None)

    return subjects, metadata_df


def parse_room_bounds_from_metadata(metadata_df: Optional[pd.DataFrame]) -> Optional[np.ndarray]:
    """
    Parse [xmin xmax; ymin ymax; zmin zmax] from the metadata sheet.
    """
    if metadata_df is None or metadata_df.empty or metadata_df.shape[1] < 2:
        return None

    keys = metadata_df.iloc[:, 0].astype(str).str.strip()
    matches = np.flatnonzero(keys.to_numpy() == ROOM_BOUNDS_KEY)
    if matches.size == 0:
        return None

    value = metadata_df.iloc[matches[0], 1]
    if pd.isna(value):
        return None

    numeric_values = [float(match) for match in FLOAT_PATTERN.findall(str(value))]
    if len(numeric_values) != 6:
        return None

    return np.asarray(numeric_values, dtype=float).reshape(3, 2)


def parse_occluded_value(value: object) -> bool:
    """
    Convert exported occlusion values to a conservative boolean flag.
    """
    if pd.isna(value):
        return True

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)

    text = str(value).strip().lower()
    if text in {"true", "1", "1.0", "yes", "y"}:
        return True
    if text in {"false", "0", "0.0", "no", "n", ""}:
        return False

    return True


def get_valid_points(samples: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract valid [x, y, z] points from a subject sheet.
    """
    missing = [name for name in REQUIRED_SUBJECT_COLUMNS if name not in samples.columns]
    if missing:
        raise ValueError(
            "Subject sheet is missing required columns: "
            + ", ".join(REQUIRED_SUBJECT_COLUMNS)
        )

    points_df = samples.loc[:, ["x_m", "y_m", "z_m"]].apply(pd.to_numeric, errors="coerce")
    points = points_df.to_numpy(dtype=float)

    occluded_mask = np.array(
        [parse_occluded_value(value) for value in samples["is_occluded"]],
        dtype=bool,
    )
    valid_mask = (~occluded_mask) & np.all(np.isfinite(points), axis=1)

    return points, valid_mask


def auto_bounds_from_subjects(
    subjects: Sequence[Tuple[str, pd.DataFrame]],
    fallback_bounds: Optional[np.ndarray],
) -> np.ndarray:
    """
    Build auto-tight 3D bounds from valid subject samples.
    """
    valid_blocks = []
    for _, samples in subjects:
        points, valid_mask = get_valid_points(samples)
        if np.any(valid_mask):
            valid_blocks.append(points[valid_mask, :])

    if len(valid_blocks) == 0:
        if fallback_bounds is not None:
            return fallback_bounds
        return np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]], dtype=float)

    all_points = np.vstack(valid_blocks)
    minimum_point = np.min(all_points, axis=0)
    maximum_point = np.max(all_points, axis=0)
    span = maximum_point - minimum_point
    padding = np.maximum(AUTO_PADDING_FRAC * span, AUTO_PADDING_MIN_M)

    return np.stack((minimum_point - padding, maximum_point + padding), axis=1)


def get_thermal_colormap() -> mpl.colors.Colormap:
    """
    Build the thermal colormap used for the single-trajectory time gradient.
    """
    if cmocean is not None:
        return cmocean.cm.thermal

    return mpl.colors.LinearSegmentedColormap.from_list(
        "thermal_fallback",
        THERMAL_ANCHOR_COLORS,
    )


def get_multi_line_colors(color_count: int) -> np.ndarray:
    """
    Build fixed colors for multi-object plots.
    """
    if color_count <= 0:
        return np.zeros((0, 3), dtype=float)

    palette_indices = np.arange(color_count) % MULTI_LINE_RGB_COLORS.shape[0]
    return MULTI_LINE_RGB_COLORS[palette_indices, :]


def get_valid_time_values(samples: pd.DataFrame, valid_mask: np.ndarray) -> np.ndarray:
    """
    Extract time values aligned with the valid trajectory points.
    """
    if "time_s" in samples.columns:
        time_values = pd.to_numeric(samples["time_s"], errors="coerce").to_numpy(dtype=float)
        time_values = time_values[valid_mask]
        if time_values.size > 0 and np.all(np.isfinite(time_values)):
            if np.ptp(time_values) > 0.0:
                return time_values

    return np.arange(np.count_nonzero(valid_mask), dtype=float)


def plot_single_trajectory_with_time_gradient(
    ax: mpl.axes.Axes,
    valid_points: np.ndarray,
    time_values: np.ndarray,
) -> Line2D:
    """
    Plot one trajectory with segment colors varying over time.
    """
    thermal_cmap = get_thermal_colormap()

    if valid_points.shape[0] < 2:
        thermal_color = thermal_cmap(np.array([SINGLE_LINE_THERMAL_POS], dtype=float))[0, :3]
        ax.plot(
            valid_points[:, 0],
            valid_points[:, 1],
            valid_points[:, 2],
            linestyle="-",
            linewidth=TRAJECTORY_LINE_LW,
            color=thermal_color,
            zorder=10,
        )
        return Line2D([0], [0], color=thermal_color, linewidth=TRAJECTORY_LINE_LW)

    segment_values = 0.5 * (time_values[:-1] + time_values[1:])
    value_min = float(np.min(segment_values))
    value_max = float(np.max(segment_values))
    if value_max <= value_min:
        segment_values = np.linspace(0.0, 1.0, valid_points.shape[0] - 1)
        value_min = 0.0
        value_max = 1.0

    segments = np.stack((valid_points[:-1, :], valid_points[1:, :]), axis=1)
    collection = Line3DCollection(
        segments,
        cmap=thermal_cmap,
        norm=mpl.colors.Normalize(vmin=value_min, vmax=value_max),
        linewidth=TRAJECTORY_LINE_LW,
        zorder=10,
    )
    collection.set_array(segment_values)
    ax.add_collection3d(collection)

    proxy_color = thermal_cmap(np.array([SINGLE_LINE_THERMAL_POS], dtype=float))[0, :3]
    return Line2D([0], [0], color=proxy_color, linewidth=TRAJECTORY_LINE_LW)


def plot_bounds_box(ax: mpl.axes.Axes, axes_bounds: np.ndarray) -> None:
    """
    Plot a dashed wireframe box for the supplied [min, max] axis bounds.
    """
    x_values = axes_bounds[0, :]
    y_values = axes_bounds[1, :]
    z_values = axes_bounds[2, :]

    vertices = np.array(
        [
            [x_values[0], y_values[0], z_values[0]],
            [x_values[1], y_values[0], z_values[0]],
            [x_values[1], y_values[1], z_values[0]],
            [x_values[0], y_values[1], z_values[0]],
            [x_values[0], y_values[0], z_values[1]],
            [x_values[1], y_values[0], z_values[1]],
            [x_values[1], y_values[1], z_values[1]],
            [x_values[0], y_values[1], z_values[1]],
        ],
        dtype=float,
    )

    edge_pairs = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ],
        dtype=int,
    )

    for start_idx, end_idx in edge_pairs:
        edge_vertices = vertices[[start_idx, end_idx], :]
        ax.plot(
            edge_vertices[:, 0],
            edge_vertices[:, 1],
            edge_vertices[:, 2],
            linestyle="--",
            linewidth=BOUNDS_LINE_LW,
            color=BOUNDS_COLOR,
            zorder=0,
        )


def style_3d_axes(ax: mpl.axes.Axes, axes_bounds: np.ndarray) -> None:
    """
    Apply the shared 3D axis styling.
    """
    ax.set_xlim(*axes_bounds[0, :])
    ax.set_ylim(*axes_bounds[1, :])
    ax.set_zlim(*axes_bounds[2, :])
    ax.set_box_aspect(np.maximum(axes_bounds[:, 1] - axes_bounds[:, 0], 1e-9))

    ax.set_xlabel(r"$x$ (m)", labelpad=10, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(r"$y$ (m)", labelpad=8, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_zlabel(r"$z$ (m)", labelpad=5, fontsize=AXIS_LABEL_FONTSIZE, rotation=90)
    ax.zaxis.set_rotate_label(False)

    ax.tick_params(axis="x", which="major", labelsize=TICK_LABEL_FONTSIZE)
    ax.tick_params(axis="y", which="major", labelsize=TICK_LABEL_FONTSIZE)
    ax.tick_params(axis="z", which="major", labelsize=TICK_LABEL_FONTSIZE)

    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    ax.grid(True)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        axis._axinfo["grid"]["color"] = GRID_COLOR
        axis._axinfo["grid"]["linewidth"] = 0.4
        axis.line.set_color("black")
        axis.line.set_linewidth(AXIS_EDGE_LW)

    ax.view_init(elev=VIEW_ELEVATION_DEG, azim=VIEW_AZIMUTH_DEG)


def create_trajectory_plot(
    subjects: Sequence[Tuple[str, pd.DataFrame]],
    axes_bounds: np.ndarray,
    figure_title: str,
    figure_path: Path,
    save_plots: bool,
) -> plt.Figure:
    """
    Build a single 3D trajectory figure.
    """
    fig, ax = create_figure_with_fixed_axes_size(figure_title=figure_title)

    plotted_subjects = []
    for subject_name, samples in subjects:
        points, valid_mask = get_valid_points(samples)
        if not np.any(valid_mask):
            continue

        time_values = get_valid_time_values(samples=samples, valid_mask=valid_mask)
        plotted_subjects.append((subject_name, points[valid_mask, :], time_values))

    has_trajectory = len(plotted_subjects) > 0
    legend_handles = []
    legend_labels = []

    if len(plotted_subjects) == 1:
        subject_name, valid_points, time_values = plotted_subjects[0]
        legend_handles.append(
            plot_single_trajectory_with_time_gradient(
                ax=ax,
                valid_points=valid_points,
                time_values=time_values,
            )
        )
        legend_labels.append(subject_name)
    else:
        line_colors = get_multi_line_colors(len(plotted_subjects))
        for subject_idx, (subject_name, valid_points, _) in enumerate(plotted_subjects):
            ax.plot(
                valid_points[:, 0],
                valid_points[:, 1],
                valid_points[:, 2],
                linestyle="-",
                linewidth=TRAJECTORY_LINE_LW,
                color=line_colors[subject_idx, :],
                zorder=10,
            )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=line_colors[subject_idx, :],
                    linewidth=TRAJECTORY_LINE_LW,
                )
            )
            legend_labels.append(subject_name)

    plot_bounds_box(ax=ax, axes_bounds=axes_bounds)
    style_3d_axes(ax=ax, axes_bounds=axes_bounds)

    legend = None
    if has_trajectory:
        legend = fig.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc=LEGEND_LOC,
            bbox_to_anchor=LEGEND_BBOX_TO_ANCHOR,
            bbox_transform=fig.transFigure,
            frameon=True,
            framealpha=1.0,
            edgecolor="black",
            fontsize=LEGEND_FONTSIZE,
            handlelength=LEGEND_HANDLE_LENGTH,
            borderpad=LEGEND_BORDERPAD,
            labelspacing=LEGEND_LABEL_SPACING,
        )
        legend.get_frame().set_linewidth(AXIS_EDGE_LW)

    if save_plots:
        extra_artists = [ax.xaxis.label, ax.yaxis.label, ax.zaxis.label]
        if legend is not None:
            extra_artists.append(legend)

        fig.savefig(
            figure_path,
            dpi=EXPORT_DPI,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=EXPORT_PAD_INCHES,
            bbox_extra_artists=tuple(extra_artists),
        )

    return fig


def plot_vicon_workbook(
    xlsx_path: Path,
    output_dir: Optional[Path] = None,
    save_plots: bool = True,
    show_plots: bool = False,
) -> Dict[str, str]:
    """
    Post-process a See_Vicon workbook and save trajectory plots.
    """
    if output_dir is None:
        output_dir = xlsx_path.parent
    if len(str(output_dir)) == 0:
        output_dir = Path(".")

    output_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib_style()

    subjects, metadata_df = load_workbook_contents(xlsx_path=xlsx_path)
    room_bounds = parse_room_bounds_from_metadata(metadata_df=metadata_df)
    auto_bounds = auto_bounds_from_subjects(subjects=subjects, fallback_bounds=room_bounds)

    plot_info = {
        "xlsx_file": str(xlsx_path),
        "room_bounds_figure_path": "",
        "auto_bounds_figure_path": "",
    }

    open_figures = []
    file_stem = xlsx_path.stem

    if room_bounds is not None:
        room_bounds_path = output_dir / f"{file_stem}{ROOM_BOUNDS_SUFFIX}"
        figure = create_trajectory_plot(
            subjects=subjects,
            axes_bounds=room_bounds,
            figure_title="Trajectory in Configured Vicon Bounds",
            figure_path=room_bounds_path,
            save_plots=save_plots,
        )
        plot_info["room_bounds_figure_path"] = str(room_bounds_path)
        open_figures.append(figure)

    auto_bounds_path = output_dir / f"{file_stem}{AUTO_BOUNDS_SUFFIX}"
    figure = create_trajectory_plot(
        subjects=subjects,
        axes_bounds=auto_bounds,
        figure_title="Trajectory with Auto-Tight Bounds",
        figure_path=auto_bounds_path,
        save_plots=save_plots,
    )
    plot_info["auto_bounds_figure_path"] = str(auto_bounds_path)
    open_figures.append(figure)

    if show_plots:
        plt.show()

    for figure in open_figures:
        plt.close(figure)

    return plot_info


### Main
def main() -> None:
    if not MAKE_PLOTS:
        print("MAKE_PLOTS is False; nothing to do.")
        return

    plot_info = plot_vicon_workbook(
        xlsx_path=WORKBOOK_XLSX,
        output_dir=OUT_DIR,
        save_plots=SAVE_PLOTS,
        show_plots=SHOW_PLOTS,
    )

    if len(plot_info["room_bounds_figure_path"]) > 0:
        print(f"Saved room-bounds plot to: {Path(plot_info['room_bounds_figure_path']).resolve()}")
    print(f"Saved auto-bounds plot to: {Path(plot_info['auto_bounds_figure_path']).resolve()}")


if __name__ == "__main__":
    main()
