from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Constants and CSV Helpers
# 2) Trace Geometry
# 3) SVG Rendering
# 4) Figure Builders
# 5) CLI Entry Point
# =============================================================================

# =============================================================================
# 1) Constants and CSV Helpers
# =============================================================================
SURFACES = ("Aileron_L", "Aileron_R", "Rudder", "Elevator")
SURFACE_LABELS = {
    "Aileron_L": "Aileron L",
    "Aileron_R": "Aileron R",
    "Rudder": "Rudder",
    "Elevator": "Elevator",
}
SURFACE_COLORS = {
    "Aileron_L": "#21318C",
    "Aileron_R": "#2254A3",
    "Rudder": "#1E80B8",
    "Elevator": "#30A5C2",
}
RECEIVER_CHANNEL_BY_SURFACE = {
    "Aileron_L": "RX_CH1",
    "Aileron_R": "RX_CH2",
    "Rudder": "RX_CH3",
    "Elevator": "RX_CH4",
}

DEFAULT_DIRECT_LOGGER = Path("C_Arduino_Test") / "Seed_5_Controller_ArduinoLogger"
DEFAULT_TRANSMITTER_LOGGER = (
    Path("D_Transmitter_Test") / "Seed_5_Nano_Transmitter_TransmitterLogger"
)

EXPORT_DPI = 600
FIGURE_WIDTH_IN = 2.7
FIGURE_HEIGHT_IN = 2.05
FONT_FAMILY = "DejaVu Sans, Arial, sans-serif"
PX_PER_POINT = EXPORT_DPI / 72.0
FONT_SIZE_PX = 10.0 * PX_PER_POINT
AXES_LABEL_SIZE_PX = 9.0 * PX_PER_POINT
AXES_TITLE_SIZE_PX = 10.0 * PX_PER_POINT
TICK_LABEL_SIZE_PX = 9.0 * PX_PER_POINT
LEGEND_FONT_SIZE_PX = 8.5 * PX_PER_POINT
AXIS_EDGE_LW_PX = 0.80 * PX_PER_POINT
MARKER_LINE_LW_PX = 0.95 * PX_PER_POINT
GRID_LINEWIDTH_PX = 0.40 * PX_PER_POINT
TICK_WIDTH_PX = 0.60 * PX_PER_POINT
TICK_LENGTH_PX = 2.0 * PX_PER_POINT
SVG_WIDTH = int(FIGURE_WIDTH_IN * EXPORT_DPI)
SVG_HEIGHT = int(FIGURE_HEIGHT_IN * EXPORT_DPI)
PLOT_LEFT = 760
PLOT_RIGHT = 160
PLOT_TOP = 250
PLOT_BOTTOM = 250
TRACE_HEIGHT = 70
TRACE_SPACING = 145
AXIS_COLOR = "#000000"
GRID_COLOR = "#cecece"
TEXT_COLOR = "#1d2b36"
MUTED_TEXT_COLOR = "#555555"
REFERENCE_COLOR = "#4A4A4A"
REFERENCE_PULSE_COLOR = "#8B7866"
TRAINER_PPM_COLOR = "#2CC6A5"
REFERENCE_PULSE_CHANNEL = "REF_D2"
REFERENCE_PULSE_LABEL = "50 Hz Ref"
DIRECT_WINDOW_BEFORE_S = 0.002
DIRECT_WINDOW_AFTER_S = 0.035
PPM_WINDOW_BEFORE_S = 0.002
PPM_WINDOW_AFTER_S = 0.035

# Edit this block to change the Arduino direct-PWM figure only.
DIRECT_PWM_FIGURE_STYLE = {
    # Overall SVG/PNG canvas and plot placement.
    "figure_width": SVG_WIDTH,
    "minimum_height": SVG_HEIGHT,
    "plot_left": PLOT_LEFT,
    "plot_right": PLOT_RIGHT,
    "plot_top": PLOT_TOP,
    "plot_bottom": PLOT_BOTTOM,
    # Row spacing and digital high/low amplitude in screen pixels.
    "trace_height": TRACE_HEIGHT,
    "trace_spacing": TRACE_SPACING,
    # Axis and grid appearance.
    "axis_color": AXIS_COLOR,
    "axis_width": AXIS_EDGE_LW_PX,
    "grid_color": GRID_COLOR,
    "grid_width": GRID_LINEWIDTH_PX,
    "tick_width": TICK_WIDTH_PX,
    "tick_length": TICK_LENGTH_PX,
    "tick_label_size": TICK_LABEL_SIZE_PX,
    "tick_label_color": MUTED_TEXT_COLOR,
    "axis_label_size": AXES_LABEL_SIZE_PX,
    "axis_label_color": MUTED_TEXT_COLOR,
    "y_axis_label": "Logic state",
    "x_axis_label": "Time (ms)",
    "y_axis_label_x": 80,
    "x_tick_label_offset": 92,
    "x_axis_label_offset": 190,
    # Figure text.
    "show_title": False,
    "title_size": AXES_TITLE_SIZE_PX,
    "title_color": TEXT_COLOR,
    "subtitle_size": FONT_SIZE_PX,
    "subtitle_color": MUTED_TEXT_COLOR,
    "row_label_size": AXES_LABEL_SIZE_PX,
    "row_label_color": TEXT_COLOR,
    "logic_label_size": TICK_LABEL_SIZE_PX,
    "logic_label_color": MUTED_TEXT_COLOR,
    "show_logic_level_labels": False,
    # Logic-level guides are faint horizontal grid lines behind each trace row.
    "baseline_color": GRID_COLOR,
    "baseline_width": GRID_LINEWIDTH_PX,
    "trace_width": AXIS_EDGE_LW_PX,
    # Marker labels are staggered automatically if they are close together.
    "marker_label_size": LEGEND_FONT_SIZE_PX,
    "marker_label_min_spacing": 175,
    "marker_label_lane_height": 95,
    "marker_label_top_offset": 95,
    "marker_line_top_extension": 70,
    "marker_dash": "22 14",
    # Per-row trace styling. end_ms truncates the right end of that row,
    # measured relative to the scheduled marker at t = 0 ms.
    "trace_styles": {
        "Aileron_L": {"color": SURFACE_COLORS["Aileron_L"], "width": AXIS_EDGE_LW_PX, "end_ms": 35.0},
        "Aileron_R": {"color": SURFACE_COLORS["Aileron_R"], "width": AXIS_EDGE_LW_PX, "end_ms": 35.0},
        "Rudder": {"color": SURFACE_COLORS["Rudder"], "width": AXIS_EDGE_LW_PX, "end_ms": 35.0},
        "Elevator": {"color": SURFACE_COLORS["Elevator"], "width": AXIS_EDGE_LW_PX, "end_ms": 35.0},
    },
    # Vertical timing-marker line styles.
    "marker_styles": {
        "scheduled": {"color": "#000000", "width": MARKER_LINE_LW_PX},
        "dispatch": {"color": "#EFA143", "width": MARKER_LINE_LW_PX},
        "board apply": {"color": "#692F7C", "width": MARKER_LINE_LW_PX},
        "PWM update": {"color": "#D96558", "width": MARKER_LINE_LW_PX},
    },
}

DEFAULT_FIGURE_STYLE = {
    "figure_width": SVG_WIDTH,
    "minimum_height": SVG_HEIGHT,
    "plot_left": PLOT_LEFT,
    "plot_right": PLOT_RIGHT,
    "plot_top": PLOT_TOP,
    "plot_bottom": PLOT_BOTTOM,
    "trace_height": TRACE_HEIGHT,
    "trace_spacing": TRACE_SPACING,
    "axis_color": AXIS_COLOR,
    "axis_width": AXIS_EDGE_LW_PX,
    "grid_color": GRID_COLOR,
    "grid_width": GRID_LINEWIDTH_PX,
    "tick_width": TICK_WIDTH_PX,
    "tick_length": TICK_LENGTH_PX,
    "tick_label_size": TICK_LABEL_SIZE_PX,
    "tick_label_color": MUTED_TEXT_COLOR,
    "axis_label_size": AXES_LABEL_SIZE_PX,
    "axis_label_color": MUTED_TEXT_COLOR,
    "y_axis_label": "Logic state",
    "x_axis_label": "Time (ms)",
    "y_axis_label_x": 80,
    "x_tick_label_offset": 92,
    "x_axis_label_offset": 190,
    "show_title": False,
    "title_size": AXES_TITLE_SIZE_PX,
    "title_color": TEXT_COLOR,
    "subtitle_size": FONT_SIZE_PX,
    "subtitle_color": MUTED_TEXT_COLOR,
    "row_label_size": AXES_LABEL_SIZE_PX,
    "row_label_color": TEXT_COLOR,
    "logic_label_size": TICK_LABEL_SIZE_PX,
    "logic_label_color": MUTED_TEXT_COLOR,
    "show_logic_level_labels": False,
    "baseline_color": GRID_COLOR,
    "baseline_width": GRID_LINEWIDTH_PX,
    "trace_width": AXIS_EDGE_LW_PX,
    "marker_label_size": LEGEND_FONT_SIZE_PX,
    "marker_label_min_spacing": 175,
    "marker_label_lane_height": 95,
    "marker_label_top_offset": 95,
    "marker_line_top_extension": 70,
    "marker_dash": "22 14",
    "trace_styles": {},
    "marker_styles": {},
}


@dataclass(frozen=True)
class PulseRow:
    surface_name: str
    time_s: float
    pulse_us: float
    sample_rate_hz: float | None = None


@dataclass(frozen=True)
class Marker:
    time_s: float
    label: str
    color: str
    line_width: float = MARKER_LINE_LW_PX


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def count_csv_rows(path: Path) -> int:
    return len(read_csv_rows(path))


def as_float(row: dict[str, str], column: str, default: float = float("nan")) -> float:
    value = row.get(column, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def is_truthy(row: dict[str, str], column: str) -> bool:
    value = str(row.get(column, "")).strip().lower()
    if value in {"1", "true", "yes"}:
        return True
    if value in {"0", "false", "no", ""}:
        return False
    try:
        return float(value) != 0.0
    except ValueError:
        return False


def read_pulse_capture(path: Path) -> list[PulseRow]:
    rows = []
    for row in read_csv_rows(path):
        rows.append(
            PulseRow(
                surface_name=str(row.get("surface_name", "")).strip(),
                time_s=as_float(row, "time_s"),
                pulse_us=as_float(row, "pulse_us"),
                sample_rate_hz=as_float(row, "sample_rate_hz"),
            )
        )
    return [row for row in rows if row.surface_name and row.time_s == row.time_s]


def median(values: Iterable[float]) -> float:
    finite = sorted(value for value in values if value == value)
    if not finite:
        return float("nan")
    midpoint = len(finite) // 2
    if len(finite) % 2:
        return finite[midpoint]
    return 0.5 * (finite[midpoint - 1] + finite[midpoint])


def mean(values: Iterable[float]) -> float:
    finite = [value for value in values if value == value]
    if not finite:
        return float("nan")
    return sum(finite) / len(finite)


def first_numeric(values: Iterable[float]) -> float:
    for value in values:
        if value == value:
            return value
    return float("nan")


# =============================================================================
# 2) Trace Geometry
# =============================================================================
def pulse_segments(
    pulses: list[PulseRow],
    surface_name: str,
    window_start_s: float,
    window_stop_s: float,
) -> list[tuple[float, float]]:
    segments = []
    for pulse in pulses:
        if pulse.surface_name != surface_name:
            continue
        if pulse.pulse_us != pulse.pulse_us:
            continue
        start = pulse.time_s
        stop = pulse.time_s + pulse.pulse_us / 1e6
        if stop < window_start_s or start > window_stop_s:
            continue
        segments.append((max(start, window_start_s), min(stop, window_stop_s)))
    return sorted(segments)


def segments_to_steps(
    segments: list[tuple[float, float]],
    window_start_s: float,
    window_stop_s: float,
) -> list[tuple[float, int]]:
    points: list[tuple[float, int]] = [(window_start_s, 0)]
    state = 0
    for start, stop in segments:
        if start > points[-1][0]:
            points.append((start, state))
        if state == 0:
            points.append((start, 1))
            state = 1
        if stop > start:
            points.append((stop, state))
            points.append((stop, 0))
            state = 0
    if points[-1][0] < window_stop_s:
        points.append((window_stop_s, state))
    return points


def logic_state_steps(
    rows: list[dict[str, str]],
    channel_name: str,
    window_start_s: float,
    window_stop_s: float,
) -> list[tuple[float, int]]:
    initial_state = 0
    transitions: list[tuple[float, int]] = []
    for row in rows:
        time_s = as_float(row, "time_s")
        state = int(round(as_float(row, channel_name, 0.0)))
        if time_s != time_s:
            continue
        if time_s <= window_start_s:
            initial_state = state
        elif time_s <= window_stop_s:
            transitions.append((time_s, state))
        elif time_s > window_stop_s:
            break

    points: list[tuple[float, int]] = [(window_start_s, initial_state)]
    state = initial_state
    for time_s, next_state in transitions:
        if next_state == state:
            continue
        if time_s > points[-1][0]:
            points.append((time_s, state))
        points.append((time_s, next_state))
        state = next_state
    if points[-1][0] < window_stop_s:
        points.append((window_stop_s, state))
    return points


def first_pulse_start_s(
    pulses: list[PulseRow],
    surface_name: str,
    search_start_s: float,
    search_stop_s: float,
) -> float:
    candidates = [
        pulse.time_s
        for pulse in pulses
        if pulse.surface_name == surface_name
        and pulse.time_s == pulse.time_s
        and search_start_s <= pulse.time_s <= search_stop_s
    ]
    if not candidates:
        return float("nan")
    return min(candidates)


def first_rising_edge_s(
    rows: list[dict[str, str]],
    channel_name: str,
    search_start_s: float,
    search_stop_s: float,
) -> float:
    state = 0
    for row in rows:
        time_s = as_float(row, "time_s")
        if time_s != time_s:
            continue
        next_state = int(round(as_float(row, channel_name, 0.0)))
        if time_s < search_start_s:
            state = next_state
            continue
        if time_s > search_stop_s:
            break
        if state == 0 and next_state == 1:
            return time_s
        state = next_state
    return float("nan")


# =============================================================================
# 3) SVG Rendering
# =============================================================================
def xml_escape(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


class SvgFigure:
    def __init__(
        self,
        title: str,
        subtitle: str,
        window_start_s: float,
        window_stop_s: float,
        anchor_s: float,
        trace_count: int,
        style: dict | None = None,
    ) -> None:
        self.title = title
        self.subtitle = subtitle
        self.window_start_s = window_start_s
        self.window_stop_s = window_stop_s
        self.anchor_s = anchor_s
        self.trace_count = trace_count
        self.style = DEFAULT_FIGURE_STYLE.copy()
        if style:
            self.style.update(style)

        self.figure_width = int(self.style["figure_width"])
        self.minimum_height = int(self.style["minimum_height"])
        self.plot_left = float(self.style["plot_left"])
        self.plot_right = float(self.style["plot_right"])
        self.plot_top = float(self.style["plot_top"])
        self.plot_bottom = float(self.style["plot_bottom"])
        self.trace_height = float(self.style["trace_height"])
        self.trace_spacing = float(self.style["trace_spacing"])
        self.plot_width = self.figure_width - self.plot_left - self.plot_right
        self.plot_height = max(360, (trace_count - 1) * self.trace_spacing + self.trace_height + 60)
        self.height = max(self.minimum_height, int(self.plot_top + self.plot_height + self.plot_bottom))
        self.elements: list[str] = []

    def x(self, time_s: float) -> float:
        span = self.window_stop_s - self.window_start_s
        return self.plot_left + (time_s - self.window_start_s) / span * self.plot_width

    def y(self, trace_index: int, state: int) -> float:
        baseline = self.plot_top + 38 + trace_index * self.trace_spacing
        return baseline - int(state) * self.trace_height

    def text(
        self,
        x: float,
        y: float,
        content: str,
        *,
        size: int = 18,
        color: str = TEXT_COLOR,
        anchor: str = "start",
        weight: str = "400",
    ) -> None:
        self.elements.append(
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
            f'font-family="{FONT_FAMILY}" font-size="{size}" '
            f'font-weight="{weight}" fill="{color}">{xml_escape(content)}</text>'
        )

    def rotated_text(
        self,
        x: float,
        y: float,
        content: str,
        *,
        angle: float = -90.0,
        size: int = 18,
        color: str = TEXT_COLOR,
        anchor: str = "middle",
        weight: str = "400",
    ) -> None:
        self.elements.append(
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
            f'transform="rotate({angle:.1f} {x:.1f} {y:.1f})" '
            f'font-family="{FONT_FAMILY}" font-size="{size}" '
            f'font-weight="{weight}" fill="{color}">{xml_escape(content)}</text>'
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        color: str = AXIS_COLOR,
        width: float = 1.0,
        dash: str = "",
    ) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.elements.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{width:.2f}"{dash_attr}/>'
        )

    def polyline(self, points: list[tuple[float, float]], *, color: str, width: float = 3.0) -> None:
        point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        self.elements.append(
            f'<polyline points="{point_text}" fill="none" stroke="{color}" '
            f'stroke-width="{width:.2f}" stroke-linecap="butt" stroke-linejoin="miter"/>'
        )

    def draw_axes(self, tick_ms: float = 10.0) -> None:
        if bool(self.style["show_title"]):
            title_size = int(self.style["title_size"])
            subtitle_size = int(self.style["subtitle_size"])
            title_y = title_size + 26
            subtitle_y = title_y + subtitle_size + 22
            self.text(
                self.plot_left,
                title_y,
                self.title,
                size=title_size,
                color=str(self.style["title_color"]),
                weight="650",
            )
            self.text(
                self.plot_left,
                subtitle_y,
                self.subtitle,
                size=subtitle_size,
                color=str(self.style["subtitle_color"]),
            )

        y_axis_top = self.plot_top - 18
        y_axis_bottom = self.plot_top + self.plot_height - 32
        axis_color = str(self.style["axis_color"])
        axis_width = float(self.style["axis_width"])
        self.line(self.plot_left, y_axis_bottom, self.plot_left + self.plot_width, y_axis_bottom, color=axis_color, width=axis_width)
        self.line(self.plot_left, y_axis_top, self.plot_left, y_axis_bottom, color=axis_color, width=axis_width)
        self.rotated_text(
            float(self.style["y_axis_label_x"]),
            (y_axis_top + y_axis_bottom) / 2,
            str(self.style["y_axis_label"]),
            size=int(self.style["axis_label_size"]),
            color=str(self.style["axis_label_color"]),
        )

        start_ms = (self.window_start_s - self.anchor_s) * 1e3
        stop_ms = (self.window_stop_s - self.anchor_s) * 1e3
        drawn_ticks: set[float] = set()

        def draw_x_tick(tick: float, *, show_label: bool = True) -> None:
            time_s = self.anchor_s + tick / 1e3
            if self.window_start_s <= time_s <= self.window_stop_s:
                x = self.x(time_s)
                if abs(x - self.plot_left) > 0.5:
                    self.line(x, y_axis_top, x, y_axis_bottom, color=str(self.style["grid_color"]), width=float(self.style["grid_width"]))
                self.line(
                    x,
                    y_axis_bottom,
                    x,
                    y_axis_bottom + float(self.style["tick_length"]),
                    color=axis_color,
                    width=float(self.style["tick_width"]),
                )
                if show_label:
                    label = f"{tick:.0f}"
                    self.text(
                        x,
                        y_axis_bottom + float(self.style["x_tick_label_offset"]),
                        label,
                        size=int(self.style["tick_label_size"]),
                        color=str(self.style["tick_label_color"]),
                        anchor="middle",
                    )
                drawn_ticks.add(round(tick, 6))

        first_tick = int(start_ms // tick_ms) * tick_ms
        def is_major_tick(tick: float) -> bool:
            return abs(tick / tick_ms - round(tick / tick_ms)) < 1e-6

        if round(start_ms, 6) not in drawn_ticks:
            draw_x_tick(start_ms, show_label=is_major_tick(start_ms))
        tick = first_tick
        while tick <= stop_ms + 0.01:
            draw_x_tick(tick, show_label=is_major_tick(tick))
            tick += tick_ms
        if round(stop_ms, 6) not in drawn_ticks:
            draw_x_tick(stop_ms, show_label=is_major_tick(stop_ms))

        self.text(
            self.plot_left + self.plot_width / 2,
            y_axis_bottom + float(self.style["x_axis_label_offset"]),
            str(self.style["x_axis_label"]),
            size=int(self.style["axis_label_size"]),
            color=str(self.style["axis_label_color"]),
            anchor="middle",
        )

    def draw_trace(
        self,
        trace_index: int,
        label: str,
        step_points: list[tuple[float, int]],
        color: str,
        width: float | None = None,
    ) -> None:
        baseline_y = self.y(trace_index, 0)
        high_y = self.y(trace_index, 1)
        for guide_y in (high_y, baseline_y):
            self.line(
                self.plot_left,
                guide_y,
                self.plot_left + self.plot_width,
                guide_y,
                color=str(self.style["grid_color"]),
                width=float(self.style["grid_width"]),
            )
        self.text(
            self.plot_left - 70,
            baseline_y - 8,
            label,
            size=int(self.style["row_label_size"]),
            color=str(self.style["row_label_color"]),
            anchor="end",
            weight="400",
        )
        if bool(self.style["show_logic_level_labels"]):
            self.text(self.plot_left - 22, high_y + 5, "1", size=int(self.style["logic_label_size"]), color=str(self.style["logic_label_color"]), anchor="end")
            self.text(self.plot_left - 22, baseline_y + 5, "0", size=int(self.style["logic_label_size"]), color=str(self.style["logic_label_color"]), anchor="end")
        def trace_x(time_s: float) -> float:
            x = self.x(time_s)
            if abs(x - self.plot_left) <= 0.5:
                return x + float(self.style["axis_width"]) + 1.0
            return x

        svg_points = [(trace_x(time_s), self.y(trace_index, state)) for time_s, state in step_points]
        self.polyline(
            svg_points,
            color=color,
            width=float(width if width is not None else self.style["trace_width"]),
        )

    def draw_markers(self, markers: list[Marker]) -> None:
        y_axis_top = self.plot_top - 16
        y_top = y_axis_top - float(self.style["marker_line_top_extension"])
        y_bottom = self.plot_top + self.plot_height - 32
        visible_markers = [
            marker
            for marker in markers
            if self.window_start_s <= marker.time_s <= self.window_stop_s
        ]
        label_lanes: list[float] = []
        for marker in sorted(visible_markers, key=lambda item: item.time_s):
            x = self.x(marker.time_s)
            lane_index = 0
            while (
                lane_index < len(label_lanes)
                and x - label_lanes[lane_index] < float(self.style["marker_label_min_spacing"])
            ):
                lane_index += 1
            if lane_index == len(label_lanes):
                label_lanes.append(x)
            else:
                label_lanes[lane_index] = x

            label_y = y_axis_top - float(self.style["marker_label_top_offset"]) - lane_index * float(self.style["marker_label_lane_height"])
            if not (self.window_start_s <= marker.time_s <= self.window_stop_s):
                continue
            self.line(
                x,
                y_top,
                x,
                y_bottom,
                color=marker.color,
                width=marker.line_width,
                dash=str(self.style["marker_dash"]),
            )
            self.text(x + 7, label_y, marker.label, size=int(self.style["marker_label_size"]), color=marker.color, anchor="start", weight="400")

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.figure_width}" height="{self.height}" viewBox="0 0 {self.figure_width} {self.height}">',
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            *self.elements,
            "</svg>",
        ]
        path.write_text("\n".join(content) + "\n", encoding="utf-8")


# =============================================================================
# 4) Figure Builders
# =============================================================================
def group_rows_by_sequence(rows: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        sequence = int(round(as_float(row, "command_sequence", -1)))
        if sequence < 0:
            continue
        grouped.setdefault(sequence, []).append(row)
    return grouped


def direct_sequence_summary(event_rows: list[dict[str, str]]) -> dict[str, float]:
    scheduled_s = first_numeric(as_float(row, "scheduled_time_s") for row in event_rows)
    dispatch_s = median(as_float(row, "command_dispatch_s") for row in event_rows)
    apply_s = median(as_float(row, "board_apply_s") for row in event_rows)
    output_s = median(as_float(row, "output_time_s") for row in event_rows)
    return {
        "scheduled_s": scheduled_s,
        "dispatch_offset_s": dispatch_s - scheduled_s,
        "apply_offset_s": apply_s - scheduled_s,
        "output_offset_s": output_s - scheduled_s,
    }


def ppm_sequence_summary(event_rows: list[dict[str, str]]) -> dict[str, float]:
    trainer_s = median(as_float(row, "trainer_transition_s") for row in event_rows)
    receiver_s = median(as_float(row, "receiver_transition_s") for row in event_rows)
    commit_s = median(as_float(row, "commit_capture_time_s") for row in event_rows)
    return {
        "trainer_s": trainer_s,
        "commit_offset_s": commit_s - trainer_s,
        "receiver_offset_s": receiver_s - trainer_s,
    }


def squared_offset_distance(summary: dict[str, float], mean_offsets: dict[str, float]) -> float:
    distance = 0.0
    for key, target_value in mean_offsets.items():
        value = summary.get(key, float("nan"))
        if value != value or target_value != target_value:
            continue
        distance += (value - target_value) ** 2
    return distance


def select_direct_event(
    events_path: Path,
    explicit_sequence: int | None,
) -> tuple[int, list[dict[str, str]], dict[str, float], int]:
    rows = read_csv_rows(events_path)
    valid_rows = []
    for row in rows:
        if explicit_sequence is not None:
            if int(round(as_float(row, "command_sequence", -1))) == explicit_sequence:
                valid_rows.append(row)
            continue
        if not is_truthy(row, "matched_output_transition"):
            continue
        if "is_valid_e2e" in row and not is_truthy(row, "is_valid_e2e"):
            continue
        valid_rows.append(row)

    grouped = group_rows_by_sequence(valid_rows)
    summaries = {
        sequence: direct_sequence_summary(sequence_rows)
        for sequence, sequence_rows in grouped.items()
    }
    summaries = {
        sequence: summary
        for sequence, summary in summaries.items()
        if all(value == value for value in summary.values())
    }
    if not summaries:
        raise ValueError(f"No suitable direct-PWM event found in {events_path}")

    mean_offsets = {
        "dispatch_offset_s": mean(summary["dispatch_offset_s"] for summary in summaries.values()),
        "apply_offset_s": mean(summary["apply_offset_s"] for summary in summaries.values()),
        "output_offset_s": mean(summary["output_offset_s"] for summary in summaries.values()),
    }
    if explicit_sequence is not None:
        selected_sequence = explicit_sequence
    else:
        selected_sequence = min(
            summaries,
            key=lambda sequence: squared_offset_distance(summaries[sequence], mean_offsets),
        )
    selected_rows = grouped[selected_sequence]
    return selected_sequence, selected_rows, mean_offsets, len(summaries)


def select_ppm_event(
    events_path: Path,
    explicit_sequence: int | None,
) -> tuple[int, list[dict[str, str]], dict[str, float], int]:
    rows = read_csv_rows(events_path)
    valid_rows = []
    for row in rows:
        if explicit_sequence is not None:
            if int(round(as_float(row, "command_sequence", -1))) == explicit_sequence:
                valid_rows.append(row)
            continue
        if not is_truthy(row, "trainer_transition_found"):
            continue
        if not is_truthy(row, "receiver_transition_found"):
            continue
        valid_rows.append(row)

    grouped = group_rows_by_sequence(valid_rows)
    summaries = {
        sequence: ppm_sequence_summary(sequence_rows)
        for sequence, sequence_rows in grouped.items()
    }
    summaries = {
        sequence: summary
        for sequence, summary in summaries.items()
        if all(value == value for value in summary.values())
    }
    if not summaries:
        raise ValueError(f"No suitable PPM-to-receiver event found in {events_path}")

    mean_offsets = {
        "commit_offset_s": mean(summary["commit_offset_s"] for summary in summaries.values()),
        "receiver_offset_s": mean(summary["receiver_offset_s"] for summary in summaries.values()),
    }
    if explicit_sequence is not None:
        selected_sequence = explicit_sequence
    else:
        selected_sequence = min(
            summaries,
            key=lambda sequence: squared_offset_distance(summaries[sequence], mean_offsets),
        )
    selected_rows = grouped[selected_sequence]
    return selected_sequence, selected_rows, mean_offsets, len(summaries)


def direct_marker(label: str, time_s: float) -> Marker:
    marker_style = DIRECT_PWM_FIGURE_STYLE["marker_styles"].get(label, {})
    return Marker(
        time_s,
        label,
        str(marker_style.get("color", "#405261")),
        float(marker_style.get("width", 1.4)),
    )


def direct_trace_style(surface_name: str) -> dict:
    return DIRECT_PWM_FIGURE_STYLE["trace_styles"].get(surface_name, {})


def direct_trace_stop_s(surface_name: str, anchor_s: float, figure_stop_s: float) -> float:
    trace_style = direct_trace_style(surface_name)
    end_ms = float(trace_style.get("end_ms", (figure_stop_s - anchor_s) * 1e3))
    return min(figure_stop_s, anchor_s + end_ms / 1e3)


def build_direct_pwm_figure(
    logger_folder: Path,
    out_path: Path,
    command_sequence: int | None = None,
) -> Path:
    output_capture_path = logger_folder / "output_capture.csv"
    events_path = logger_folder / "e2e_output_events.csv"
    pulses = read_pulse_capture(output_capture_path)
    selected_sequence, event_rows, _mean_offsets, sequence_count = select_direct_event(events_path, command_sequence)

    selected_summary = direct_sequence_summary(event_rows)
    event_anchor_s = selected_summary["scheduled_s"]
    search_start_s = event_anchor_s
    search_stop_s = event_anchor_s + DIRECT_WINDOW_AFTER_S
    anchor_s = first_pulse_start_s(pulses, "Aileron_L", search_start_s, search_stop_s)
    if anchor_s != anchor_s:
        anchor_s = event_anchor_s
    window_start_s = anchor_s - DIRECT_WINDOW_BEFORE_S
    window_stop_s = anchor_s + DIRECT_WINDOW_AFTER_S

    sample_rate = median(pulse.sample_rate_hz or float("nan") for pulse in pulses)
    subtitle = (
        f"{logger_folder.name}; mean timing from {sequence_count} commands; trace command {selected_sequence}; "
        f"decoded analyser PWM pulses at {sample_rate / 1e6:.1f} MHz"
    )

    figure = SvgFigure(
        "Representative direct PWM generation",
        subtitle,
        window_start_s,
        window_stop_s,
        anchor_s,
        trace_count=len(SURFACES),
        style=DIRECT_PWM_FIGURE_STYLE,
    )
    figure.draw_axes(tick_ms=10.0)

    for trace_index, surface_name in enumerate(SURFACES):
        trace_style = direct_trace_style(surface_name)
        row_stop_s = direct_trace_stop_s(surface_name, anchor_s, window_stop_s)
        segments = pulse_segments(pulses, surface_name, window_start_s, row_stop_s)
        steps = segments_to_steps(segments, window_start_s, row_stop_s)
        figure.draw_trace(
            trace_index,
            SURFACE_LABELS[surface_name],
            steps,
            str(trace_style.get("color", SURFACE_COLORS[surface_name])),
            width=float(trace_style.get("width", 3.2)),
        )

    figure.write(out_path)
    return out_path


def build_ppm_receiver_figure(
    logger_folder: Path,
    out_path: Path,
    command_sequence: int | None = None,
) -> Path:
    event_path = logger_folder / "post_transition_e2e_events.csv"
    logic_state_path = logger_folder / f"{logger_folder.name.replace('_TransmitterLogger', '')}_sigrok_logic_state.csv"
    if not logic_state_path.is_file():
        matches = sorted(logger_folder.glob("*_sigrok_logic_state.csv"))
        if not matches:
            raise FileNotFoundError(f"No logic-state CSV found in {logger_folder}")
        logic_state_path = matches[0]

    selected_sequence, event_rows, _mean_offsets, sequence_count = select_ppm_event(event_path, command_sequence)
    logic_rows = sorted(read_csv_rows(logic_state_path), key=lambda row: as_float(row, "time_s"))
    selected_summary = ppm_sequence_summary(event_rows)
    event_anchor_s = selected_summary["trainer_s"]
    search_start_s = event_anchor_s
    search_stop_s = event_anchor_s + PPM_WINDOW_AFTER_S
    anchor_s = first_rising_edge_s(
        logic_rows,
        REFERENCE_PULSE_CHANNEL,
        search_start_s,
        search_stop_s,
    )
    if anchor_s != anchor_s:
        anchor_s = event_anchor_s
    window_start_s = anchor_s - PPM_WINDOW_BEFORE_S
    window_stop_s = anchor_s + PPM_WINDOW_AFTER_S

    subtitle = (
        f"{logger_folder.name}; mean timing from {sequence_count} commands; trace command {selected_sequence}; "
        "compact sigrok logic-state export at 4.0 MHz"
    )
    figure = SvgFigure(
        "Representative PPM-to-receiver PWM regeneration",
        subtitle,
        window_start_s,
        window_stop_s,
        anchor_s,
        trace_count=2 + len(SURFACES),
    )
    figure.draw_axes(tick_ms=10.0)

    reference_steps = logic_state_steps(logic_rows, REFERENCE_PULSE_CHANNEL, window_start_s, window_stop_s)
    figure.draw_trace(0, REFERENCE_PULSE_LABEL, reference_steps, REFERENCE_PULSE_COLOR)
    trainer_steps = logic_state_steps(logic_rows, "TRAINER_PPM_D3", window_start_s, window_stop_s)
    figure.draw_trace(1, "Trainer PPM", trainer_steps, TRAINER_PPM_COLOR)
    for trace_index, surface_name in enumerate(SURFACES, start=2):
        channel_name = RECEIVER_CHANNEL_BY_SURFACE[surface_name]
        steps = logic_state_steps(logic_rows, channel_name, window_start_s, window_stop_s)
        figure.draw_trace(
            trace_index,
            f"RX {SURFACE_LABELS[surface_name]}",
            steps,
            SURFACE_COLORS[surface_name],
        )

    figure.write(out_path)
    return out_path


# =============================================================================
# 5) CLI Entry Point
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot two representative logic-analyser figures from the final measured "
            "direct-PWM and PPM-to-receiver datasets."
        )
    )
    parser.add_argument(
        "--direct-logger",
        type=Path,
        default=DEFAULT_DIRECT_LOGGER,
        help="Arduino direct-PWM *_ArduinoLogger folder.",
    )
    parser.add_argument(
        "--transmitter-logger",
        type=Path,
        default=DEFAULT_TRANSMITTER_LOGGER,
        help="Transmitter *_TransmitterLogger folder for PPM-to-receiver plotting.",
    )
    parser.add_argument(
        "--direct-sequence",
        type=int,
        default=None,
        help="Optional command_sequence to use for the direct-PWM figure.",
    )
    parser.add_argument(
        "--ppm-sequence",
        type=int,
        default=None,
        help="Optional command_sequence to use for the PPM-to-receiver figure.",
    )
    parser.add_argument(
        "--direct-out",
        type=Path,
        default=None,
        help="Optional direct-PWM PNG output path.",
    )
    parser.add_argument(
        "--ppm-out",
        type=Path,
        default=None,
        help="Optional PPM-to-receiver PNG output path.",
    )
    return parser.parse_args()


def default_direct_out(logger_folder: Path) -> Path:
    return logger_folder.parent / "A_figures" / f"{logger_folder.name.replace('_ArduinoLogger', '')}_direct_pwm_logic_trace.png"


def default_ppm_out(logger_folder: Path) -> Path:
    return logger_folder.parent / "A_figures" / f"{logger_folder.name.replace('_TransmitterLogger', '')}_ppm_to_receiver_logic_trace.png"


def find_browser_executable() -> str:
    candidates = [
        Path("C:/Program Files/Google/Chrome/Application/chrome.exe"),
        Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    for command_name in ("chrome", "msedge"):
        resolved = shutil.which(command_name)
        if resolved:
            return resolved
    raise FileNotFoundError("Chrome or Edge is required to render SVG figures to PNG.")


def render_svg_to_png(svg_path: Path, png_path: Path, width: int, height: int) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    browser = find_browser_executable()
    command = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--disable-software-rasterizer",
        "--no-sandbox",
        f"--screenshot={png_path.resolve()}",
        f"--window-size={int(width)},{int(height)}",
        svg_path.resolve().as_uri(),
    ]
    subprocess.run(command, check=True)
    file_size = png_path.stat().st_size
    if file_size > 100 * 1024 * 1024:
        raise ValueError(f"Generated plot exceeds 100 MB file-size rule: {png_path}")


def svg_canvas_size(svg_path: Path) -> tuple[int, int]:
    text = svg_path.read_text(encoding="utf-8")
    width_match = re.search(r'<svg[^>]*\swidth="([0-9]+)"', text)
    height_match = re.search(r'<svg[^>]*\sheight="([0-9]+)"', text)
    if not width_match or not height_match:
        raise ValueError(f"Could not read SVG canvas size from {svg_path}")
    return int(width_match.group(1)), int(height_match.group(1))


def source_note_path(png_path: Path) -> Path:
    return png_path.with_suffix(".source.txt")


def write_source_note(
    png_path: Path,
    *,
    caption: str,
    input_files: list[Path],
    filtering_rules: list[str],
    sample_count: int,
    latency_case: str,
    run_id: str,
    policy_name: str,
) -> Path:
    note_path = source_note_path(png_path)
    note_lines = [
        f"caption: {caption}",
        "input files:",
        *[f"  - {path.resolve()}" for path in input_files],
        "filtering rules:",
        *[f"  - {rule}" for rule in filtering_rules],
        f"sample count: {sample_count}",
        "environment_id: not_applicable",
        f"latency case: {latency_case}",
        f"policy name: {policy_name}",
        f"run ID: {run_id}",
        f"file size: {png_path.stat().st_size} bytes",
    ]
    note_path.write_text("\n".join(note_lines) + "\n", encoding="utf-8")
    return note_path


def png_output_path(path: Path) -> Path:
    return path if path.suffix.lower() == ".png" else path.with_suffix(".png")


def temporary_svg_path(png_path: Path) -> Path:
    return png_path.with_name(f"{png_path.stem}__tmp.svg")


def main() -> None:
    args = parse_args()
    direct_logger = args.direct_logger.resolve()
    transmitter_logger = args.transmitter_logger.resolve()
    direct_out = png_output_path(args.direct_out or default_direct_out(direct_logger))
    ppm_out = png_output_path(args.ppm_out or default_ppm_out(transmitter_logger))
    direct_svg = temporary_svg_path(direct_out)
    ppm_svg = temporary_svg_path(ppm_out)

    build_direct_pwm_figure(
        direct_logger,
        direct_svg,
        command_sequence=args.direct_sequence,
    )
    build_ppm_receiver_figure(
        transmitter_logger,
        ppm_svg,
        command_sequence=args.ppm_sequence,
    )
    direct_width, direct_height = svg_canvas_size(direct_svg)
    ppm_width, ppm_height = svg_canvas_size(ppm_svg)
    render_svg_to_png(direct_svg, direct_out, direct_width, direct_height)
    render_svg_to_png(ppm_svg, ppm_out, ppm_width, ppm_height)
    direct_svg.unlink(missing_ok=True)
    ppm_svg.unlink(missing_ok=True)
    write_source_note(
        direct_out,
        caption=(
            "Measured direct-PWM logic-analyser trace showing the regenerated servo PWM "
            "waveforms for the representative transition, with time zero aligned to the "
            "first measured Aileron L pulse."
        ),
        input_files=[
            direct_logger / "output_capture.csv",
            direct_logger / "e2e_output_events.csv",
        ],
        filtering_rules=[
            "The visible waveform is the measured command whose timing vector is closest to mean offsets over all valid matched output transitions.",
            "With --direct-sequence, the requested command is used intentionally.",
            "Timing-event markers are intentionally omitted; timing statistics are reported separately in tables.",
            "Plots decoded measured PWM pulse starts and widths from -2 ms to 35 ms around the first measured Aileron L pulse.",
            "Each row is truncated by DIRECT_PWM_FIGURE_STYLE['trace_styles'][surface]['end_ms'].",
        ],
        sample_count=count_csv_rows(direct_logger / "output_capture.csv"),
        latency_case="direct PWM generation",
        run_id=direct_logger.name,
        policy_name="measured logic-analyser redraw",
    )
    write_source_note(
        ppm_out,
        caption=(
            "Measured PPM-to-receiver regeneration trace showing the 50 Hz D2 reference, "
            "trainer PPM pulses, and receiver PWM responses for the representative transition, "
            "with time zero aligned to the first measured 50 Hz D2 reference pulse."
        ),
        input_files=[
            transmitter_logger / "post_transition_e2e_events.csv",
            transmitter_logger
            / f"{transmitter_logger.name.replace('_TransmitterLogger', '')}_sigrok_logic_state.csv",
        ],
        filtering_rules=[
            "The visible waveform is the measured command whose timing vector is closest to mean offsets over all events with trainer and receiver transitions.",
            "With --ppm-sequence, the requested command is used intentionally.",
            "Timing-event markers are intentionally omitted; timing statistics are reported separately in tables.",
            "Includes the measured 50 Hz D2 reference pulse train as a timing reference row.",
            "Plots compact measured sigrok logic-state transitions from -2 ms to 35 ms around the first measured 50 Hz D2 reference pulse.",
        ],
        sample_count=count_csv_rows(
            transmitter_logger
            / f"{transmitter_logger.name.replace('_TransmitterLogger', '')}_sigrok_logic_state.csv"
        ),
        latency_case="PPM-to-receiver PWM regeneration",
        run_id=transmitter_logger.name,
        policy_name="measured logic-analyser redraw",
    )

    print("Logic-analyser trace figures generated")
    print(f"  Direct PWM:       {direct_out.resolve()}")
    print(f"  PPM to receiver:  {ppm_out.resolve()}")


if __name__ == "__main__":
    main()
