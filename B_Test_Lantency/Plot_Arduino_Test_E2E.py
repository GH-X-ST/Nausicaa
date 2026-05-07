from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

DEFAULT_SEED: int | None = 5
DEFAULT_EVENT_PREFIX = "e2e_output"
WORKBOOK_ROOT = Path("C_Arduino_Test")
MS_PER_SECOND = 1e3

SURFACE_STYLES = [
    {"surface_name": "Aileron_L", "label": "Aileron L", "color": "#2a9d8F", "marker": "o", "marker_size": 3.0},
    {"surface_name": "Aileron_R", "label": "Aileron R", "color": "#e9c46a", "marker": "s", "marker_size": 2.8},
    {"surface_name": "Rudder", "label": "Rudder", "color": "#f4a261", "marker": "^", "marker_size": 3.2},
    {"surface_name": "Elevator", "label": "Elevator", "color": "#e76f51", "marker": "D", "marker_size": 2.6},
]

LATENCY_PLOT_METRICS = [
    ("Scheduled to dispatch", "host_scheduling_delay_s", "#264653"),
    ("Dispatch to Arduino RX", "dispatch_to_rx_latency_s", "#2a9d8F"),
    ("Arduino RX to servo output", "rx_to_output_latency_s", "#f4a261"),
    ("Scheduled to servo output", "scheduled_to_output_latency_s", "#e76f51"),
]

SUMMARY_METRICS = [
    ("Scheduled to dispatch", "host_scheduling_delay_s"),
    ("Dispatch to Arduino RX", "dispatch_to_rx_latency_s"),
    ("Arduino RX to servo output", "rx_to_output_latency_s"),
    ("Scheduled to servo output", "scheduled_to_output_latency_s"),
]

SUMMARY_STATS = [
    ("Median_s", "Median", "#264653"),
    ("P95_s", "P95", "#2a9d8F"),
    ("P99_s", "P99", "#f4a261"),
    ("Max_s", "Max", "#e76f51"),
]


def _latest_logger_folder(root: Path) -> Path:
    candidates = [path for path in root.glob("*_ArduinoLogger") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No '*_ArduinoLogger' folder found under: {root}")
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _logger_folder_from_seed(root: Path, seed: int) -> Path:
    folder = root / f"Seed_{int(seed)}_Controller_ArduinoLogger"
    if not folder.is_dir():
        raise FileNotFoundError(f"Seed logger folder not found: {folder}")
    return folder


def _resolve_logger_folder(logger_folder: str | None, seed: int | None) -> Path:
    if logger_folder:
        path = Path(logger_folder)
        if not path.is_absolute():
            path = WORKBOOK_ROOT / path
        if not path.is_dir():
            raise FileNotFoundError(f"Logger folder not found: {path}")
        return path
    chosen_seed = DEFAULT_SEED if seed is None else seed
    if chosen_seed is not None:
        return _logger_folder_from_seed(WORKBOOK_ROOT, chosen_seed)
    raise ValueError("Explicitly provide --seed <N> or --logger-folder <PATH> when plotting Arduino E2E runs.")


def _read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Missing file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def _to_numeric(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")


def _latency_stats(series: pd.Series) -> Dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "SampleCount": 0,
            "Mean_s": np.nan,
            "Std_s": np.nan,
            "Median_s": np.nan,
            "P95_s": np.nan,
            "P99_s": np.nan,
            "Max_s": np.nan,
        }
    return {
        "SampleCount": int(values.size),
        "Mean_s": float(np.mean(values)),
        "Std_s": float(np.std(values, ddof=0)) if values.size >= 2 else np.nan,
        "Median_s": float(np.percentile(values, 50)),
        "P95_s": float(np.percentile(values, 95)),
        "P99_s": float(np.percentile(values, 99)),
        "Max_s": float(np.max(values)),
    }


def _run_label_from_logger(logger_folder: Path) -> str:
    name = logger_folder.name
    if name.endswith("_ArduinoLogger"):
        return name[: -len("_ArduinoLogger")]
    return name


def _build_surface_wide_sheet(
    events: pd.DataFrame,
    surfaces: List[str],
    time_source_column: str,
    per_surface_columns: Dict[str, str],
) -> pd.DataFrame:
    sample_base = (
        events.groupby("sample_index", as_index=False)
        .agg(time_s=(time_source_column, "median"))
        .sort_values("sample_index", kind="stable")
    )
    out = sample_base.copy()
    for surface in surfaces:
        surface_events = events[events["surface_name"] == surface].copy()
        keep = ["sample_index", "command_sequence", "scheduled_time_s", "command_dispatch_s"] + list(per_surface_columns.values())
        keep = [column for column in keep if column in surface_events.columns]
        surface_events = surface_events[keep].drop_duplicates(subset=["sample_index"], keep="first")
        rename = {}
        if "command_sequence" in surface_events.columns:
            rename["command_sequence"] = f"{surface}_command_sequence"
        if "scheduled_time_s" in surface_events.columns:
            rename["scheduled_time_s"] = f"{surface}_scheduled_time_s"
        if "command_dispatch_s" in surface_events.columns:
            rename["command_dispatch_s"] = f"{surface}_command_dispatch_s"
        for suffix, column in per_surface_columns.items():
            if column in surface_events.columns:
                rename[column] = f"{surface}_{suffix}"
        surface_events = surface_events.rename(columns=rename)
        out = out.merge(surface_events, on="sample_index", how="left")
    return out.drop(columns=["sample_index"])


def _aggregate_latency_series(events: pd.DataFrame, surfaces: List[str], latency_column: str) -> tuple[np.ndarray, np.ndarray]:
    if events.empty or latency_column not in events.columns:
        return np.array([], dtype=float), np.array([], dtype=float)
    rows = []
    grouped = events.groupby("sample_index", sort=True)
    for sample_index, group in grouped:
        time_s = float(np.nanmedian(pd.to_numeric(group["scheduled_time_s"], errors="coerce")))
        values = []
        for surface in surfaces:
            sub = pd.to_numeric(group.loc[group["surface_name"] == surface, latency_column], errors="coerce").to_numpy(dtype=float)
            sub = sub[np.isfinite(sub)]
            if sub.size:
                values.append(float(np.nanmedian(sub)))
        if values:
            rows.append((sample_index, time_s, float(np.nanmedian(values))))
    if not rows:
        return np.array([], dtype=float), np.array([], dtype=float)
    arr = np.asarray(rows, dtype=float)
    return arr[:, 1], arr[:, 2]


def _pulse_to_equivalent_deg(pulse_us: pd.Series) -> pd.Series:
    pulse = pd.to_numeric(pulse_us, errors="coerce")
    return 180.0 * (pulse - 1500.0) / 1000.0


def _build_output_response(output_capture: pd.DataFrame) -> List[dict]:
    responses = []
    if output_capture.empty:
        return responses
    for style in SURFACE_STYLES:
        surface = style["surface_name"]
        sub = output_capture[output_capture["surface_name"] == surface].copy()
        if sub.empty:
            continue
        responses.append(
            {
                **style,
                "time": pd.to_numeric(sub["time_s"], errors="coerce").to_numpy(dtype=float),
                "value": _pulse_to_equivalent_deg(sub["pulse_us"]).to_numpy(dtype=float),
            }
        )
    return responses


def _active_time_window(input_signal: pd.DataFrame, events: pd.DataFrame) -> tuple[float, float] | None:
    scheduled = pd.to_numeric(events.get("scheduled_time_s"), errors="coerce")
    scheduled = scheduled[np.isfinite(scheduled)]
    if scheduled.empty:
        scheduled = pd.to_numeric(input_signal.get("scheduled_time_s"), errors="coerce")
        scheduled = scheduled[np.isfinite(scheduled)]
    if scheduled.empty:
        return None
    return float(scheduled.min()), float(scheduled.max())


def _clip_time_frame(frame: pd.DataFrame, time_column: str, window: tuple[float, float] | None) -> pd.DataFrame:
    if frame.empty or window is None or time_column not in frame.columns:
        return frame
    time_values = pd.to_numeric(frame[time_column], errors="coerce")
    mask = np.isfinite(time_values) & (time_values >= window[0]) & (time_values <= window[1])
    return frame.loc[mask].copy()


def _build_command_traces(input_signal: pd.DataFrame, surfaces: List[str], profile_type: str) -> List[dict]:
    traces: List[dict] = []
    if profile_type != "latency_vector_step_train":
        return traces

    time_values = pd.to_numeric(input_signal.get("time_s"), errors="coerce").to_numpy(dtype=float)
    for style in SURFACE_STYLES:
        surface_name = style["surface_name"]
        if surfaces and surface_name not in surfaces:
            continue
        column_name = f"{surface_name}_desired_deg"
        if column_name not in input_signal.columns:
            continue
        command_values = pd.to_numeric(input_signal.get(column_name), errors="coerce").to_numpy(dtype=float)
        if not np.any(np.isfinite(command_values)):
            continue
        traces.append({**style, "time": time_values, "value": command_values})
    return traces


def _build_summary_from_events(events: pd.DataFrame, surfaces: List[str]) -> pd.DataFrame:
    rows = []
    active_events = events[events["surface_name"].astype(str).isin(surfaces)].copy() if surfaces else events.copy()
    for metric_label, column_name in SUMMARY_METRICS:
        stats = _latency_stats(active_events.get(column_name, pd.Series(dtype=float)))
        row = {"Metric": metric_label, "SampleCount": int(stats["SampleCount"])}
        for stat_suffix, stat_label, _ in SUMMARY_STATS:
            row[stat_label] = float(stats[stat_suffix]) if np.isfinite(stats[stat_suffix]) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _build_integrity_table(integrity_summary: pd.DataFrame) -> pd.DataFrame:
    if integrity_summary.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "Surface": integrity_summary["SurfaceName"],
        "Transitions": pd.to_numeric(integrity_summary["TransitionCommandCount"], errors="coerce").fillna(0).astype(int),
        "Matched": pd.to_numeric(integrity_summary["MatchedOutputTransitionCount"], errors="coerce").fillna(0).astype(int),
        "Valid": pd.to_numeric(integrity_summary["ValidE2ECount"], errors="coerce").fillna(0).astype(int),
        "Unmatched": pd.to_numeric(integrity_summary["UnmatchedOutputTransitionCount"], errors="coerce").fillna(0).astype(int),
        "Unmatched %": 100.0 * pd.to_numeric(integrity_summary["UnmatchedOutputTransitionFraction"], errors="coerce").fillna(0.0),
    })
    return out


def _infer_profile_type(input_signal: pd.DataFrame, profile_events: pd.DataFrame) -> str:
    if "AxisLabel" in profile_events.columns:
        return "latency_vector_step_train"
    base_command = pd.to_numeric(input_signal.get("base_command_deg"), errors="coerce")
    if base_command.isna().all():
        return "latency_vector_step_train"
    return "latency_step_train"


def _build_chain_text(events: pd.DataFrame) -> str:
    chain_steps = ["Scheduled", "Dispatch"]
    if "board_rx_s" in events.columns:
        chain_steps.append("Arduino RX")
    if "board_apply_s" in events.columns:
        chain_steps.append("Arduino apply")
    if "output_time_s" in events.columns:
        chain_steps.append("Servo output")
    return "Chain: " + " -> ".join(chain_steps)


def _build_critical_settings(
    run_label: str,
    logger_folder: Path,
    event_prefix: str,
    events: pd.DataFrame,
    input_signal: pd.DataFrame,
    profile_events: pd.DataFrame,
) -> pd.DataFrame:
    clock_mode = ""
    if "anchor_source" in events.columns and not events.empty:
        anchor_values = events["anchor_source"].astype(str)
        if (anchor_values == "reference").any():
            clock_mode = "shared_clock"
        elif (anchor_values == "apply").any():
            clock_mode = "apply_anchored"
    profile_type = _infer_profile_type(input_signal, profile_events)
    settings = [
        ("Run", "RunLabel", run_label),
        ("Run", "Status", "completed"),
        ("Run", "OutputFolder", str(logger_folder.parent.resolve())),
        ("Run", "LoggerFolder", str(logger_folder.resolve())),
        ("Run", "EventPrefix", event_prefix),
        ("ArduinoTransport", "ResolvedMode", "nano_logger_udp"),
        ("ArduinoTransport", "OperatingMode", "controller"),
        ("ArduinoTransport", "CommandEncoding", "binary_vector"),
        ("Command", "Mode", "all"),
        ("Profile", "Type", profile_type),
        ("LogicAnalyzer", "AnalysisBasis", "servo_output_pwm_logic_analyzer"),
        ("Matching", "Mode", clock_mode or "apply_anchored"),
        ("Analysis", "LatencyChain", _build_chain_text(events)),
        ("Analysis", "LatencySummarySource", "arduino_e2e_python"),
    ]
    return pd.DataFrame(settings, columns=["Category", "Setting", "Value"])


def _plot_time_series(
    ax: plt.Axes,
    input_signal: pd.DataFrame,
    command_traces: List[dict],
    responses: List[dict],
    profile_events: pd.DataFrame,
) -> None:
    ax.set_facecolor("white")
    ax.grid(True, color=(0.85, 0.85, 0.85, 1.0), linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if not profile_events.empty:
        for _, row in profile_events.iterrows():
            label = str(row.get("EventLabel", ""))
            if label not in {"positive_step", "negative_step"}:
                continue
            start = pd.to_numeric(row.get("StartTime_s"), errors="coerce")
            stop = pd.to_numeric(row.get("StopTime_s"), errors="coerce")
            if np.isfinite(start) and np.isfinite(stop):
                color = "#2a9d8F" if label == "positive_step" else "#e76f51"
                ax.axvspan(float(start), float(stop), color=color, alpha=0.08, linewidth=0.0)

    host_t = pd.to_numeric(input_signal.get("time_s"), errors="coerce").to_numpy(dtype=float)
    host_y = pd.to_numeric(input_signal.get("base_command_deg"), errors="coerce").to_numpy(dtype=float)
    if host_t.size > 0 and host_y.size > 0 and host_t.size == host_y.size:
        mask = np.isfinite(host_t) & np.isfinite(host_y)
        ax.plot(host_t[mask], host_y[mask], color="#264653", linewidth=1.0, label="Host command")

    for command_trace in command_traces:
        mask = np.isfinite(command_trace["time"]) & np.isfinite(command_trace["value"])
        ax.plot(
            command_trace["time"][mask],
            command_trace["value"][mask],
            color=mcolors.to_rgba(command_trace["color"], 0.80),
            linewidth=1.0,
            linestyle="--",
            label=f"{command_trace['label']} cmd",
        )

    for response in responses:
        mask = np.isfinite(response["time"]) & np.isfinite(response["value"])
        ax.plot(
            response["time"][mask],
            response["value"][mask],
            color=mcolors.to_rgba(response["color"], 0.72),
            linewidth=0.9,
            marker=response["marker"],
            markersize=response["marker_size"],
            markerfacecolor=mcolors.to_rgba(response["color"], 0.45),
            markeredgewidth=0.0,
            label=response["label"],
        )

    ax.set_title("Command and analyser-observed servo output", fontsize=12)
    ax.set_xlabel(r"Time, $t$ (s)", fontsize=11)
    ax.set_ylabel("Angle (deg)", fontsize=11)
    ax.tick_params(axis="x", labelrotation=-25, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, fontsize=9)


def _plot_latency(ax: plt.Axes, events: pd.DataFrame, surfaces: List[str]) -> None:
    ax.set_facecolor("white")
    ax.grid(True, color=(0.85, 0.85, 0.85, 1.0), linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for label, column, color in LATENCY_PLOT_METRICS:
        t, y = _aggregate_latency_series(events, surfaces, column)
        if t.size:
            ax.plot(t, MS_PER_SECOND * y, color=color, linewidth=1.1, label=label)
    ax.set_title("Latency decomposition", fontsize=12)
    ax.set_xlabel(r"Time, $t$ (s)", fontsize=11)
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.tick_params(axis="x", labelrotation=-25, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, fontsize=9)


def _plot_summary(ax: plt.Axes, latency_summary: pd.DataFrame) -> None:
    ax.set_facecolor("white")
    ax.grid(True, axis="y", color=(0.85, 0.85, 0.85, 1.0), linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if latency_summary.empty:
        ax.text(0.5, 0.5, "Latency summary unavailable", ha="center", va="center", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    x = np.arange(len(latency_summary))
    width = 0.18
    for idx, (_, stat_label, color) in enumerate(SUMMARY_STATS):
        vals = MS_PER_SECOND * pd.to_numeric(latency_summary[stat_label], errors="coerce").to_numpy(dtype=float)
        ax.bar(x + (idx - 1.5) * width, vals, width=width, color=color, label=stat_label, alpha=0.9)
    tick_labels = [f"{metric}\n(n={int(count)})" for metric, count in zip(latency_summary["Metric"], latency_summary["SampleCount"])]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=-20, ha="left", fontsize=9)
    ax.set_title("Latency summary statistics", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.tick_params(axis="y", labelsize=10)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, fontsize=9)


def _plot_integrity(ax: plt.Axes, integrity_table: pd.DataFrame) -> None:
    ax.axis("off")
    ax.set_title("Transition integrity", fontsize=12)
    if integrity_table.empty:
        ax.text(0.5, 0.5, "Integrity summary unavailable", ha="center", va="center", fontsize=11)
        return
    display = integrity_table.copy()
    if "Unmatched %" in display.columns:
        display["Unmatched %"] = display["Unmatched %"].map(lambda v: f"{v:.2f}")
    table = ax.table(cellText=display.values, colLabels=display.columns, loc="center", cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    for (r, _), cell in table.get_celld().items():
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor("#e9f1ef")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#ffffff")


def _build_bundle(logger_folder: Path, event_prefix: str) -> dict:
    events = _read_csv(logger_folder / f"{event_prefix}_events.csv")
    input_signal = _read_csv(logger_folder / f"{event_prefix}_input_signal.csv")
    surface_summary = _read_csv(logger_folder / f"{event_prefix}_surface_summary.csv")
    overall_summary = _read_csv(logger_folder / f"{event_prefix}_overall_summary.csv", required=False)
    integrity_summary = _read_csv(logger_folder / f"{event_prefix}_integrity_summary.csv")
    profile_events = _read_csv(logger_folder / f"{event_prefix}_profile_events.csv", required=False)
    output_capture = _read_csv(logger_folder / "output_capture.csv", required=False)
    reference_capture = _read_csv(logger_folder / "reference_capture.csv", required=False)

    _to_numeric(events, [
        "sample_index", "command_sequence", "scheduled_time_s", "command_dispatch_s", "host_scheduling_delay_s",
        "board_rx_s", "board_apply_s", "dispatch_to_rx_latency_s", "dispatch_to_apply_latency_s",
        "scheduled_to_rx_latency_s", "scheduled_to_apply_latency_s", "rx_to_apply_latency_s", "reference_time_s",
        "anchor_time_s", "expected_pulse_us", "previous_expected_pulse_us", "output_time_s", "output_pulse_us",
        "apply_to_output_latency_s", "dispatch_to_output_latency_s", "scheduled_to_output_latency_s",
    ])
    _to_numeric(input_signal, ["time_s", "scheduled_time_s", "command_write_start_s", "command_write_stop_s", "base_command_deg"])
    _to_numeric(output_capture, ["time_s", "pulse_us", "sample_index", "sample_count", "sample_rate_hz"])
    _to_numeric(reference_capture, ["time_s", "sample_index", "sample_rate_hz"])

    active_window = _active_time_window(input_signal, events)
    input_signal = _clip_time_frame(input_signal, "scheduled_time_s", active_window)
    output_capture = _clip_time_frame(output_capture, "time_s", active_window)
    reference_capture = _clip_time_frame(reference_capture, "time_s", active_window)
    profile_events = _clip_time_frame(profile_events, "StartTime_s", active_window)

    surfaces = [style["surface_name"] for style in SURFACE_STYLES if style["surface_name"] in events["surface_name"].astype(str).unique()]
    if not surfaces:
        surfaces = [style["surface_name"] for style in SURFACE_STYLES]

    if {"rx_to_apply_latency_s", "apply_to_output_latency_s"}.issubset(events.columns):
        events["rx_to_output_latency_s"] = pd.to_numeric(events["rx_to_apply_latency_s"], errors="coerce") + pd.to_numeric(events["apply_to_output_latency_s"], errors="coerce")
    else:
        events["rx_to_output_latency_s"] = np.nan

    latency_summary = _build_summary_from_events(events, surfaces)
    integrity_table = _build_integrity_table(integrity_summary)
    run_label = _run_label_from_logger(logger_folder)
    profile_type = _infer_profile_type(input_signal, profile_events)
    command_traces = _build_command_traces(input_signal, surfaces, profile_type)
    critical_settings = _build_critical_settings(
        run_label,
        logger_folder,
        event_prefix,
        events,
        input_signal,
        profile_events,
    )

    sheets = {
        "CriticalSettings": critical_settings,
        "InputSignal": input_signal,
        "OutputCapture": output_capture,
        "ReferenceCapture": reference_capture,
        "HostSchedulingDelay": _build_surface_wide_sheet(events, surfaces, "scheduled_time_s", {"host_scheduling_delay_s": "host_scheduling_delay_s"}),
        "ComputerToArduinoRxLatency": _build_surface_wide_sheet(events, surfaces, "scheduled_time_s", {"computer_to_arduino_rx_latency_s": "dispatch_to_rx_latency_s"}),
        "ScheduledToArduinoRxLatency": _build_surface_wide_sheet(events, surfaces, "scheduled_time_s", {"scheduled_to_arduino_rx_latency_s": "scheduled_to_rx_latency_s"}),
        "ComputerToArduinoApplyLatency": _build_surface_wide_sheet(events, surfaces, "scheduled_time_s", {"computer_to_arduino_apply_latency_s": "dispatch_to_apply_latency_s"}),
        "ArduinoReceiveToApplyLatency": _build_surface_wide_sheet(events, surfaces, "scheduled_time_s", {"arduino_receive_to_apply_latency_s": "rx_to_apply_latency_s"}),
        "ScheduledToApplyLatency": _build_surface_wide_sheet(events, surfaces, "scheduled_time_s", {"scheduled_to_apply_latency_s": "scheduled_to_apply_latency_s"}),
        "ApplyToOutputLatency": _build_surface_wide_sheet(events, surfaces, "scheduled_time_s", {"apply_to_output_latency_s": "apply_to_output_latency_s"}),
        "DispatchToOutputLatency": _build_surface_wide_sheet(events, surfaces, "scheduled_time_s", {"dispatch_to_output_latency_s": "dispatch_to_output_latency_s"}),
        "ScheduledToOutputLatency": _build_surface_wide_sheet(events, surfaces, "scheduled_time_s", {"scheduled_to_output_latency_s": "scheduled_to_output_latency_s"}),
        "LatencySummary": surface_summary,
        "OverallSummary": overall_summary,
        "IntegritySummary": integrity_summary,
        "ProfileEvents": profile_events,
    }

    return {
        "run_label": run_label,
        "surfaces": surfaces,
        "events": events,
        "input_signal": input_signal,
        "profile_type": profile_type,
        "chain_text": _build_chain_text(events),
        "command_traces": command_traces,
        "output_capture": output_capture,
        "profile_events": profile_events,
        "latency_summary": latency_summary,
        "integrity_table": integrity_table,
        "sheets": sheets,
    }


def _write_workbook(bundle: dict, workbook_path: Path) -> None:
    workbook_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        for sheet_name, frame in bundle["sheets"].items():
            frame.to_excel(writer, sheet_name=sheet_name[:31], index=False)


def _plot_figure(bundle: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13.5, 8.4), dpi=300)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.07, right=0.80, top=0.80, bottom=0.08, hspace=0.42, wspace=0.62)
    grid = fig.add_gridspec(2, 2)
    ax_time = fig.add_subplot(grid[0, 0])
    ax_latency = fig.add_subplot(grid[0, 1])
    ax_summary = fig.add_subplot(grid[1, 0])
    ax_integrity = fig.add_subplot(grid[1, 1])

    _plot_time_series(
        ax_time,
        bundle["input_signal"],
        bundle["command_traces"],
        _build_output_response(bundle["output_capture"]),
        bundle["profile_events"],
    )
    _plot_latency(ax_latency, bundle["events"], bundle["surfaces"])
    _plot_summary(ax_summary, bundle["latency_summary"])
    _plot_integrity(ax_integrity, bundle["integrity_table"])

    fig.suptitle(bundle["run_label"], fontsize=13, y=0.975)
    fig.text(0.5, 0.935, "Controller-origin latency with analyser-observed servo output", ha="center", va="top", fontsize=9.5, color="#4f5d75")
    fig.text(0.5, 0.910, bundle["chain_text"], ha="center", va="top", fontsize=9.0, color="#6c757d")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build workbook and figure for Arduino logic-analyser E2E results.")
    parser.add_argument("--logger-folder", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--event-prefix", type=str, default=DEFAULT_EVENT_PREFIX)
    args = parser.parse_args()

    logger_folder = _resolve_logger_folder(args.logger_folder, args.seed)
    bundle = _build_bundle(logger_folder, args.event_prefix)
    run_label = _run_label_from_logger(logger_folder)
    workbook_path = logger_folder.parent / f"{run_label}_{args.event_prefix}.xlsx"
    figure_path = logger_folder.parent / "A_figures" / f"{run_label}_{args.event_prefix}.png"

    _write_workbook(bundle, workbook_path)
    _plot_figure(bundle, figure_path)

    selection_source = f"seed {args.seed}" if args.seed is not None else "logger folder"
    print(f"Source: {selection_source}")
    print(f"Logger folder: {logger_folder.resolve()}")
    print(f"Workbook: {workbook_path.resolve()}")
    print(f"Figure: {figure_path.resolve()}")


if __name__ == "__main__":
    main()
