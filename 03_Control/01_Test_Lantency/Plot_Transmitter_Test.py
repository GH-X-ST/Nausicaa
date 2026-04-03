from pathlib import Path
import time
import zipfile

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

from Plot_Arduino_Test import (
    AXIS_EDGE_LW,
    FIGURE_BOTTOM,
    FIGURE_HSPACE,
    FIGURE_LEFT,
    FIGURE_RIGHT,
    FIGURE_TOP,
    FIGURE_WSPACE,
    LEGEND_BBOX_TO_ANCHOR,
    LEGEND_BORDERPAD,
    LEGEND_FONT_SIZE,
    LEGEND_FRAME_LW,
    LEGEND_HANDLE_LENGTH,
    LEGEND_LABEL_SPACING,
    LEGEND_LOC,
    MS_PER_SECOND,
    SUMMARY_STATS,
    SURFACE_STYLES,
    add_legend,
    add_profile_event_spans,
    build_settings_lookup,
    configure_axes,
    finite_pair,
    find_latest_workbook,
    get_active_surfaces,
    get_setting,
    nanmedian_or_nan,
    normalize_text,
    read_sheet,
    resolve_workbook_path,
)


LATENCY_METRICS = [
    {
        "sheet": "HostSchedulingDelay",
        "suffix": "_host_scheduling_delay_s",
        "label": "Scheduled to dispatch",
        "summary_prefix": "HostSchedulingDelay",
        "color": "#264653",
    },
    {
        "sheet": "ComputerToArduinoRxLatency",
        "suffix": "_computer_to_arduino_rx_latency_s",
        "label": "Dispatch to Uno RX",
        "summary_prefix": "ComputerToArduinoRxLatency",
        "color": "#2a9d8F",
    },
    {
        "sheet": "ArduinoRxToPpmCommitLatency",
        "suffix": "_arduino_receive_to_ppm_commit_latency_s",
        "label": "Uno RX to PPM commit",
        "summary_prefix": "ArduinoReceiveToPpmCommitLatency",
        "color": "#e9c46a",
    },
    {
        "sheet": "PpmToReceiverLatency",
        "suffix": "_ppm_to_receiver_latency_s",
        "label": "PPM to receiver PWM",
        "summary_prefix": "PpmToReceiverLatency",
        "color": "#f4a261",
    },
    {
        "sheet": "ScheduledToReceiverLatency",
        "suffix": "_scheduled_to_receiver_latency_s",
        "label": "Scheduled to receiver PWM",
        "summary_prefix": "ScheduledToReceiverLatency",
        "color": "#e76f51",
    },
]

WORKBOOK_OPEN_RETRY_COUNT = 5
WORKBOOK_OPEN_RETRY_DELAY_SECONDS = 0.5


def numeric_series(frame: pd.DataFrame, column_name: str) -> np.ndarray:
    if column_name not in frame.columns:
        return np.array([], dtype=float)
    return pd.to_numeric(frame[column_name], errors="coerce").to_numpy(dtype=float)


def open_excel_file(excel_path: Path) -> pd.ExcelFile:
    last_error = None
    for attempt_index in range(WORKBOOK_OPEN_RETRY_COUNT):
        try:
            if not zipfile.is_zipfile(excel_path):
                raise zipfile.BadZipFile(
                    f"Workbook is not yet a valid ZIP container: {excel_path}"
                )
            return pd.ExcelFile(excel_path)
        except (PermissionError, zipfile.BadZipFile) as error:
            last_error = error
            if attempt_index >= WORKBOOK_OPEN_RETRY_COUNT - 1:
                break
            time.sleep(WORKBOOK_OPEN_RETRY_DELAY_SECONDS)

    raise last_error


def build_transmitter_workbook_metadata(
    excel_path: Path,
    critical_settings_df: pd.DataFrame,
    active_surfaces: list[str],
) -> dict:
    settings_lookup = build_settings_lookup(critical_settings_df)
    run_label = get_setting(settings_lookup, "Run", "RunLabel", excel_path.stem) or excel_path.stem
    command_mode = get_setting(settings_lookup, "Command", "Mode")
    profile_type = get_setting(settings_lookup, "Profile", "Type")
    board = get_setting(settings_lookup, "Run", "ArduinoBoard", "Uno")
    sample_rate_hz = get_setting(settings_lookup, "LogicAnalyzer", "SampleRateHz")
    trainer_frame_us = get_setting(settings_lookup, "TrainerPPM", "FrameLengthUs")

    metadata_lines = [
        "Controller-origin basis",
        f"Transport: MATLAB -> {board} serial -> trainer PPM -> receiver PWM",
    ]
    if command_mode:
        metadata_lines.append(f"Command mode: {command_mode}")
    if profile_type:
        metadata_lines.append(f"Profile: {profile_type}")
    if trainer_frame_us:
        metadata_lines.append(f"PPM frame: {trainer_frame_us} us")
    if sample_rate_hz:
        metadata_lines.append(f"Analyser: {sample_rate_hz} Hz")
    metadata_lines.append(f"Active surfaces: {', '.join(active_surfaces)}")

    return {
        "run_label": run_label,
        "metadata_line": " | ".join(metadata_lines),
        "note_line": "Controller-origin total is scheduled-to-receiver PWM latency; robust comparison should prioritize median and percentile bands rather than max.",
    }


def pulse_us_to_equivalent_deg(pulse_us: np.ndarray) -> np.ndarray:
    pulse_us = np.asarray(pulse_us, dtype=float)
    return 180.0 * (pulse_us - 1000.0) / 1000.0 - 90.0


def load_transmitter_time_series(
    input_df: pd.DataFrame,
    receiver_capture_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    host_time = numeric_series(input_df, "time_s")
    host_command = numeric_series(input_df, "base_command_deg")

    responses: list[dict] = []
    if receiver_capture_df.empty:
        return host_time, host_command, responses

    receiver_df = receiver_capture_df.copy()
    receiver_df["surface_name"] = receiver_df["surface_name"].astype(str)
    receiver_df["time_s"] = pd.to_numeric(receiver_df["time_s"], errors="coerce")
    receiver_df["pulse_us"] = pd.to_numeric(receiver_df["pulse_us"], errors="coerce")

    for style in SURFACE_STYLES:
        surface_rows = receiver_df[receiver_df["surface_name"] == style["surface_name"]].copy()
        if surface_rows.empty:
            continue

        response_time = surface_rows["time_s"].to_numpy(dtype=float)
        response_value = pulse_us_to_equivalent_deg(surface_rows["pulse_us"].to_numpy(dtype=float))
        if not np.any(np.isfinite(response_value)):
            continue

        responses.append({**style, "time": response_time, "value": response_value})

    return host_time, host_command, responses


def summarize_latency_metrics(latency_summary_df: pd.DataFrame, active_surfaces: list[str]) -> pd.DataFrame:
    if latency_summary_df.empty:
        return pd.DataFrame()

    summary_df = latency_summary_df.copy()
    summary_df["SurfaceName"] = summary_df["SurfaceName"].astype(str)
    active_summary = summary_df[summary_df["SurfaceName"].isin(active_surfaces)].copy()
    if active_summary.empty:
        active_summary = summary_df.copy()

    rows = []
    for metric in LATENCY_METRICS:
        row = {"Metric": metric["label"], "SampleCount": 0}
        sample_count_column = f"{metric['summary_prefix']}SampleCount"
        if sample_count_column in active_summary.columns:
            sample_count_values = pd.to_numeric(active_summary[sample_count_column], errors="coerce").fillna(0.0)
            row["SampleCount"] = int(np.nanmedian(sample_count_values.to_numpy(dtype=float)))
        for stat_suffix, stat_label, _ in SUMMARY_STATS:
            column_name = f"{metric['summary_prefix']}{stat_suffix}"
            if column_name in active_summary.columns:
                values = pd.to_numeric(active_summary[column_name], errors="coerce").to_numpy(dtype=float)
                row[stat_label] = nanmedian_or_nan(values)
            else:
                row[stat_label] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_latency_series(frame: pd.DataFrame, suffix: str, active_surfaces: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if frame.empty:
        return np.array([], dtype=float), np.array([], dtype=float)

    candidate_columns = [
        f"{surface_name}{suffix}" for surface_name in active_surfaces if f"{surface_name}{suffix}" in frame.columns
    ]
    if not candidate_columns:
        candidate_columns = [column for column in frame.columns if column.endswith(suffix)]
    if not candidate_columns:
        return np.array([], dtype=float), np.array([], dtype=float)

    time_s = numeric_series(frame, "time_s")
    values = frame[candidate_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    aggregate_values = np.full(values.shape[0], np.nan, dtype=float)
    for row_index, row_values in enumerate(values):
        aggregate_values[row_index] = nanmedian_or_nan(np.asarray(row_values, dtype=float))
    return finite_pair(time_s, aggregate_values)


def build_integrity_table(integrity_summary_df: pd.DataFrame, active_surfaces: list[str]) -> pd.DataFrame:
    if integrity_summary_df.empty:
        return pd.DataFrame(columns=["Surface", "Dispatch", "RX %", "Commit %", "Drop %", "Receiver %"])

    integrity_df = integrity_summary_df.copy()
    integrity_df["SurfaceName"] = integrity_df["SurfaceName"].astype(str)
    integrity_df = integrity_df[integrity_df["SurfaceName"].isin(active_surfaces)].copy()
    if integrity_df.empty:
        return pd.DataFrame(columns=["Surface", "Dispatch", "RX %", "Commit %", "Drop %", "Receiver %"])

    dispatch = pd.to_numeric(integrity_df["DispatchedCommandCount"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    matched_rx = pd.to_numeric(integrity_df["MatchedRxCount"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    matched_commit = pd.to_numeric(integrity_df["MatchedCommitCount"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    dropped_commit = pd.to_numeric(integrity_df["DroppedBeforeCommitCount"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    matched_receiver = pd.to_numeric(integrity_df["MatchedReceiverCount"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    safe_dispatch = np.maximum(dispatch, 1.0)
    return pd.DataFrame(
        {
            "Surface": integrity_df["SurfaceName"],
            "Dispatch": dispatch.astype(int),
            "RX %": 100.0 * matched_rx / safe_dispatch,
            "Commit %": 100.0 * matched_commit / safe_dispatch,
            "Drop %": 100.0 * dropped_commit / safe_dispatch,
            "Receiver %": 100.0 * matched_receiver / safe_dispatch,
        }
    )


def build_workbook_bundle(excel_path: Path) -> dict:
    excel_file = open_excel_file(excel_path)
    workbook_sheets = {
        "CriticalSettings": read_sheet(excel_file, "CriticalSettings"),
        "InputSignal": read_sheet(excel_file, "InputSignal", required=True),
        "HostSchedulingDelay": read_sheet(excel_file, "HostSchedulingDelay"),
        "ComputerToArduinoRxLatency": read_sheet(excel_file, "ComputerToArduinoRxLatency"),
        "ArduinoRxToPpmCommitLatency": read_sheet(excel_file, "ArduinoRxToPpmCommitLatency"),
        "PpmToReceiverLatency": read_sheet(excel_file, "PpmToReceiverLatency"),
        "ScheduledToReceiverLatency": read_sheet(excel_file, "ScheduledToReceiverLatency"),
        "LatencySummary": read_sheet(excel_file, "LatencySummary"),
        "IntegritySummary": read_sheet(excel_file, "IntegritySummary"),
        "ProfileEvents": read_sheet(excel_file, "ProfileEvents"),
        "ReceiverCapture": read_sheet(excel_file, "ReceiverCapture"),
    }

    active_surfaces = get_active_surfaces(workbook_sheets["LatencySummary"])
    metadata = build_transmitter_workbook_metadata(excel_path, workbook_sheets["CriticalSettings"], active_surfaces)
    host_time, host_command, responses = load_transmitter_time_series(
        workbook_sheets["InputSignal"], workbook_sheets["ReceiverCapture"]
    )
    latency_summary_df = summarize_latency_metrics(workbook_sheets["LatencySummary"], active_surfaces)
    integrity_table_df = build_integrity_table(workbook_sheets["IntegritySummary"], active_surfaces)

    return {
        "sheets": workbook_sheets,
        "active_surfaces": active_surfaces,
        "host_time": host_time,
        "host_command": host_command,
        "responses": responses,
        "latency_summary_df": latency_summary_df,
        "integrity_table_df": integrity_table_df,
        "metadata": metadata,
    }


def plot_time_series_panel(
    ax: plt.Axes,
    host_time: np.ndarray,
    host_command: np.ndarray,
    responses: list[dict],
    profile_events_df: pd.DataFrame,
):
    configure_axes(ax)
    add_profile_event_spans(ax, profile_events_df)

    host_x, host_y = finite_pair(host_time, host_command)
    ax.plot(
        host_x,
        host_y,
        color=mcolors.to_rgba("#264653", 0.75),
        linewidth=1.0,
        label="MATLAB command",
    )

    finite_blocks = [host_y[np.isfinite(host_y)]]
    for response in responses:
        response_x, response_y = finite_pair(response["time"], response["value"])
        finite_blocks.append(response_y[np.isfinite(response_y)])
        ax.plot(
            response_x,
            response_y,
            color=mcolors.to_rgba(response["color"], 0.70),
            linewidth=1.0,
            marker=response["marker"],
            markersize=response["marker_size"],
            markerfacecolor=mcolors.to_rgba(response["color"], response["marker_alpha"]),
            markeredgecolor=mcolors.to_rgba(response["color"], response["marker_alpha"]),
            markeredgewidth=0.0,
            markevery=1,
            label=f"{response['label']} PWM",
            zorder=8,
        )

    finite_blocks = [block for block in finite_blocks if block.size > 0]
    if finite_blocks:
        all_values = np.concatenate(finite_blocks)
        y_min = float(np.min(all_values))
        y_max = float(np.max(all_values))
        y_span = y_max - y_min
        y_pad = max(2.0, 0.05 * y_span if y_span > 0.0 else 2.0)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    finite_time_blocks = [host_x]
    finite_time_blocks.extend(finite_pair(response["time"], response["value"])[0] for response in responses)
    finite_time_blocks = [block for block in finite_time_blocks if block.size > 0]
    if finite_time_blocks:
        all_times = np.concatenate(finite_time_blocks)
        ax.set_xlim(float(np.min(all_times)), float(np.max(all_times)))

    ax.set_title("Command and receiver PWM response", fontsize=12)
    ax.set_xlabel(r"Time, $t$ (s)", fontsize=11)
    ax.set_ylabel("Equivalent angle (deg)", fontsize=11)
    ax.tick_params(axis="both", labelsize=10)
    ax.tick_params(axis="x", labelrotation=-25)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    add_legend(ax)


def plot_latency_panel(
    ax: plt.Axes,
    workbook_sheets: dict[str, pd.DataFrame],
    active_surfaces: list[str],
):
    configure_axes(ax)

    for metric in LATENCY_METRICS:
        series_time, series_value = aggregate_latency_series(
            workbook_sheets.get(metric["sheet"], pd.DataFrame()),
            metric["suffix"],
            active_surfaces,
        )
        if series_time.size == 0:
            continue
        ax.plot(
            series_time,
            MS_PER_SECOND * series_value,
            color=metric["color"],
            linewidth=1.1,
            label=metric["label"],
        )

    ax.set_title("Latency decomposition (controller-origin basis)", fontsize=12)
    ax.set_xlabel(r"Time, $t$ (s)", fontsize=11)
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.tick_params(axis="both", labelsize=10)
    ax.tick_params(axis="x", labelrotation=-25)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    add_legend(ax)


def plot_summary_panel(ax: plt.Axes, summary_df: pd.DataFrame):
    configure_axes(ax)
    if summary_df.empty:
        ax.text(0.5, 0.5, "LatencySummary not available", ha="center", va="center", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    x_positions = np.arange(len(summary_df))
    bar_width = 0.18

    for bar_index, (_, stat_label, color) in enumerate(SUMMARY_STATS):
        bar_values_ms = MS_PER_SECOND * pd.to_numeric(summary_df[stat_label], errors="coerce").to_numpy(dtype=float)
        ax.bar(
            x_positions + (bar_index - 1.5) * bar_width,
            bar_values_ms,
            width=bar_width,
            color=color,
            label=stat_label,
            alpha=0.90,
        )

    tick_labels = [
        f"{metric_label}\n(n={int(sample_count)})"
        for metric_label, sample_count in zip(summary_df["Metric"], summary_df["SampleCount"])
    ]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(tick_labels, rotation=-20, ha="left", fontsize=9)
    ax.set_title("Latency summary statistics", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.tick_params(axis="y", labelsize=10)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    legend = ax.legend(
        loc=LEGEND_LOC,
        bbox_to_anchor=LEGEND_BBOX_TO_ANCHOR,
        frameon=True,
        framealpha=1.0,
        edgecolor="black",
        fontsize=LEGEND_FONT_SIZE,
        handlelength=LEGEND_HANDLE_LENGTH,
        borderpad=LEGEND_BORDERPAD,
        labelspacing=LEGEND_LABEL_SPACING,
    )
    if legend is not None:
        legend.get_frame().set_linewidth(LEGEND_FRAME_LW)
        legend.set_in_layout(False)


def plot_integrity_panel(ax: plt.Axes, integrity_table_df: pd.DataFrame, metadata: dict):
    ax.set_facecolor("white")
    ax.axis("off")
    ax.set_title("Coverage summary", fontsize=12)

    if integrity_table_df.empty:
        ax.text(0.5, 0.5, "IntegritySummary not available", ha="center", va="center", fontsize=11)
        return

    display_df = integrity_table_df.copy()
    for column_name in ["RX %", "Commit %", "Drop %", "Receiver %"]:
        display_df[column_name] = display_df[column_name].map(lambda value: f"{value:.1f}")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)

    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_linewidth(0.5)
        if row_index == 0:
            cell.set_facecolor("#e9f1ef")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#ffffff")

    note_line = normalize_text(metadata.get("note_line", ""))
    if note_line:
        ax.text(
            0.0,
            1.10,
            note_line,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            color="#4f5d75",
        )


def plot_transmitter_summary_figure(excel_path: Path, out_path: Path, workbook_bundle: dict):
    workbook_sheets = workbook_bundle["sheets"]
    fig = plt.figure(figsize=(13.5, 8.4), dpi=300)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(
        left=FIGURE_LEFT,
        right=FIGURE_RIGHT,
        top=FIGURE_TOP,
        bottom=FIGURE_BOTTOM,
        hspace=FIGURE_HSPACE,
        wspace=FIGURE_WSPACE,
    )
    grid = fig.add_gridspec(2, 2)

    ax_time = fig.add_subplot(grid[0, 0])
    ax_latency = fig.add_subplot(grid[0, 1])
    ax_summary = fig.add_subplot(grid[1, 0])
    ax_integrity = fig.add_subplot(grid[1, 1])

    plot_time_series_panel(
        ax_time,
        workbook_bundle["host_time"],
        workbook_bundle["host_command"],
        workbook_bundle["responses"],
        workbook_sheets["ProfileEvents"],
    )
    plot_latency_panel(
        ax_latency,
        workbook_sheets,
        workbook_bundle["active_surfaces"],
    )
    plot_summary_panel(ax_summary, workbook_bundle["latency_summary_df"])
    plot_integrity_panel(ax_integrity, workbook_bundle["integrity_table_df"], workbook_bundle["metadata"])

    fig.suptitle(workbook_bundle["metadata"]["run_label"], fontsize=13, y=0.975)
    metadata_line = normalize_text(workbook_bundle["metadata"].get("metadata_line", ""))
    if metadata_line:
        fig.text(0.5, 0.935, metadata_line, ha="center", va="top", fontsize=9.5, color="#4f5d75")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    workbook_dir = Path("D_Transmitter_Test")
    excel_path = "Seed_5_Transmitter.xlsx"
    out_dir = workbook_dir / "A_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    workbook_path = resolve_workbook_path(excel_path, workbook_dir)
    workbook_bundle = build_workbook_bundle(workbook_path)
    out_path = out_dir / f"{workbook_path.stem}.png"

    plot_transmitter_summary_figure(workbook_path, out_path, workbook_bundle)
    print(f"Done. Saved plot to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
