from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter


LEGEND_FONT_SIZE = 9
AXIS_EDGE_LW = 0.80
LEGEND_FRAME_LW = AXIS_EDGE_LW
LEGEND_LOC = "lower right"
LEGEND_BBOX_TO_ANCHOR = (1.02, 0.02)
LEGEND_HANDLE_LENGTH = 1.5
LEGEND_BORDERPAD = 0.5
LEGEND_LABEL_SPACING = 0.2
MS_PER_SECOND = 1e3

SURFACE_STYLES = [
    {
        "surface_name": "Aileron_L",
        "label": "Aileron L",
        "color": "#2a9d8F",
        "marker": "o",
        "marker_size": 3.0,
        "marker_alpha": 0.50,
    },
    {
        "surface_name": "Aileron_R",
        "label": "Aileron R",
        "color": "#e9c46a",
        "marker": "s",
        "marker_size": 2.7,
        "marker_alpha": 0.50,
    },
    {
        "surface_name": "Rudder",
        "label": "Rudder",
        "color": "#f4a261",
        "marker": "^",
        "marker_size": 3.2,
        "marker_alpha": 0.50,
    },
    {
        "surface_name": "Elevator",
        "label": "Elevator",
        "color": "#e76f51",
        "marker": "D",
        "marker_size": 2.5,
        "marker_alpha": 0.50,
    },
]

LATENCY_METRICS = [
    {
        "sheet": "HostSchedulingDelay",
        "suffix": "_host_scheduling_delay_s",
        "label": "Host scheduling",
        "summary_prefix": "HostSchedulingDelay",
        "color": "#264653",
    },
    {
        "sheet": "ComputerToArduinoRxLatency",
        "suffix": "_computer_to_arduino_rx_latency_s",
        "label": "Host to Arduino RX",
        "summary_prefix": "ComputerToArduinoRxLatency",
        "color": "#2a9d8F",
    },
    {
        "sheet": "ComputerToArduinoApplyLatency",
        "suffix": "_computer_to_arduino_apply_latency_s",
        "label": "Host to Arduino APPLY",
        "summary_prefix": "ComputerToArduinoApplyLatency",
        "color": "#f4a261",
    },
    {
        "sheet": "ArduinoReceiveToApplyLatency",
        "suffix": "_arduino_receive_to_apply_latency_s",
        "label": "Arduino RX to APPLY",
        "summary_prefix": "ArduinoReceiveToApplyLatency",
        "color": "#7b6d8d",
    },
    {
        "sheet": "ScheduledToApplyLatency",
        "suffix": "_scheduled_to_apply_latency_s",
        "label": "Scheduled to APPLY",
        "summary_prefix": "ScheduledToApplyLatency",
        "color": "#e76f51",
    },
]

SUMMARY_STATS = [
    ("Median_s", "Median", "#264653"),
    ("P95_s", "P95", "#2a9d8F"),
    ("P99_s", "P99", "#f4a261"),
    ("Max_s", "Max", "#e76f51"),
]


def find_latest_workbook(workbook_dir: Path) -> Path:
    workbook_paths = sorted(
        path for path in workbook_dir.glob("*.xlsx") if not path.name.startswith("~$")
    )
    if not workbook_paths:
        raise FileNotFoundError(f"No .xlsx files were found in: {workbook_dir}")
    return workbook_paths[-1]


def resolve_workbook_path(excel_path, workbook_dir: Path) -> Path:
    if excel_path is None:
        return find_latest_workbook(workbook_dir)

    workbook_path = Path(excel_path)
    if not workbook_path.is_absolute():
        workbook_path = workbook_dir / workbook_path

    if not workbook_path.is_file():
        raise FileNotFoundError(f"Workbook not found: {workbook_path}")

    return workbook_path


def read_sheet(excel_file: pd.ExcelFile, sheet_name: str, required: bool = False) -> pd.DataFrame:
    if sheet_name in excel_file.sheet_names:
        return pd.read_excel(excel_file, sheet_name=sheet_name)
    if required:
        raise KeyError(f"Workbook is missing required sheet: {sheet_name}")
    return pd.DataFrame()


def numeric_series(frame: pd.DataFrame, column_name: str) -> np.ndarray:
    if column_name not in frame.columns:
        return np.array([], dtype=float)
    return pd.to_numeric(frame[column_name], errors="coerce").to_numpy(dtype=float)


def finite_pair(x: np.ndarray, y: np.ndarray):
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def get_active_surfaces(latency_summary_df: pd.DataFrame) -> list[str]:
    if latency_summary_df.empty:
        return [style["surface_name"] for style in SURFACE_STYLES]

    surface_names = latency_summary_df.get("SurfaceName", pd.Series(dtype=str)).astype(str)
    active_mask = latency_summary_df.get("IsActive", pd.Series(dtype=bool)).fillna(False)
    active_surfaces = surface_names[active_mask.astype(bool)].tolist()
    if active_surfaces:
        return active_surfaces
    return [style["surface_name"] for style in SURFACE_STYLES]


def load_time_series(input_df: pd.DataFrame, echo_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    host_time = numeric_series(input_df, "time_s")
    host_command = numeric_series(input_df, "base_command_deg")
    response_time = numeric_series(echo_df, "time_s")

    responses: list[dict] = []
    for style in SURFACE_STYLES:
        value_column = f"{style['surface_name']}_applied_equivalent_deg"
        if value_column not in echo_df.columns:
            continue

        response_value = numeric_series(echo_df, value_column)
        if not np.any(np.isfinite(response_value)):
            continue

        responses.append({**style, "time": response_time, "value": response_value})

    return host_time, host_command, responses


def collect_finite_values(host_command: np.ndarray, responses: list[dict]) -> np.ndarray:
    finite_blocks = [host_command[np.isfinite(host_command)]]
    for response in responses:
        finite_blocks.append(response["value"][np.isfinite(response["value"])])

    finite_blocks = [block for block in finite_blocks if block.size > 0]
    if not finite_blocks:
        return np.array([], dtype=float)

    return np.concatenate(finite_blocks)


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
    aggregate_values = np.nanmedian(values, axis=1)
    return finite_pair(time_s, aggregate_values)


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
        row = {"Metric": metric["label"]}
        for stat_suffix, stat_label, _ in SUMMARY_STATS:
            column_name = f"{metric['summary_prefix']}{stat_suffix}"
            if column_name in active_summary.columns:
                values = pd.to_numeric(active_summary[column_name], errors="coerce").to_numpy(dtype=float)
                row[stat_label] = np.nanmedian(values)
            else:
                row[stat_label] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def build_integrity_table(integrity_summary_df: pd.DataFrame, active_surfaces: list[str]) -> pd.DataFrame:
    if integrity_summary_df.empty:
        return pd.DataFrame(
            columns=["Surface", "Dispatch", "Rx miss %", "Apply miss %", "Dup %", "Unexpected %", "Non-mono"]
        )

    integrity_df = integrity_summary_df.copy()
    integrity_df["SurfaceName"] = integrity_df["SurfaceName"].astype(str)
    integrity_df = integrity_df[integrity_df["SurfaceName"].isin(active_surfaces)].copy()
    if integrity_df.empty:
        return pd.DataFrame(
            columns=["Surface", "Dispatch", "Rx miss %", "Apply miss %", "Dup %", "Unexpected %", "Non-mono"]
        )

    table_df = pd.DataFrame(
        {
            "Surface": integrity_df["SurfaceName"],
            "Dispatch": pd.to_numeric(integrity_df["DispatchedCommandCount"], errors="coerce").fillna(0).astype(int),
            "Rx miss %": 100.0
            * pd.to_numeric(integrity_df["UnmatchedRxCommandFraction"], errors="coerce").fillna(0.0),
            "Apply miss %": 100.0
            * pd.to_numeric(integrity_df["UnmatchedApplyCommandFraction"], errors="coerce").fillna(0.0),
            "Dup %": 100.0
            * pd.to_numeric(integrity_df["DuplicateTelemetryKeyFraction"], errors="coerce").fillna(0.0),
            "Unexpected %": 100.0
            * pd.to_numeric(integrity_df["UnexpectedTelemetryRowFraction"], errors="coerce").fillna(0.0),
            "Non-mono": pd.to_numeric(integrity_df["NonMonotonicSequenceCount"], errors="coerce").fillna(0).astype(int),
        }
    )
    return table_df


def build_workbook_bundle(excel_path: Path) -> dict:
    excel_file = pd.ExcelFile(excel_path)
    workbook_sheets = {
        "InputSignal": read_sheet(excel_file, "InputSignal", required=True),
        "ArduinoEcho": read_sheet(excel_file, "ArduinoEcho", required=True),
        "HostSchedulingDelay": read_sheet(excel_file, "HostSchedulingDelay"),
        "ComputerToArduinoRxLatency": read_sheet(excel_file, "ComputerToArduinoRxLatency"),
        "ComputerToArduinoApplyLatency": read_sheet(excel_file, "ComputerToArduinoApplyLatency"),
        "ArduinoReceiveToApplyLatency": read_sheet(excel_file, "ArduinoReceiveToApplyLatency"),
        "ScheduledToApplyLatency": read_sheet(excel_file, "ScheduledToApplyLatency"),
        "LatencySummary": read_sheet(excel_file, "LatencySummary"),
        "IntegritySummary": read_sheet(excel_file, "IntegritySummary"),
        "ProfileEvents": read_sheet(excel_file, "ProfileEvents"),
    }

    active_surfaces = get_active_surfaces(workbook_sheets["LatencySummary"])
    host_time, host_command, responses = load_time_series(
        workbook_sheets["InputSignal"], workbook_sheets["ArduinoEcho"]
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
    }


def add_profile_event_spans(ax: plt.Axes, profile_events_df: pd.DataFrame):
    if profile_events_df.empty:
        return

    label_to_style = {
        "positive_step": ("#2a9d8F", 0.10),
        "negative_step": ("#e76f51", 0.10),
    }

    for _, event_row in profile_events_df.iterrows():
        event_label = str(event_row.get("EventLabel", ""))
        if event_label not in label_to_style:
            continue

        start_time_s = pd.to_numeric(event_row.get("StartTime_s"), errors="coerce")
        stop_time_s = pd.to_numeric(event_row.get("StopTime_s"), errors="coerce")
        if not np.isfinite(start_time_s) or not np.isfinite(stop_time_s):
            continue

        color, alpha = label_to_style[event_label]
        ax.axvspan(float(start_time_s), float(stop_time_s), color=color, alpha=alpha, linewidth=0.0)


def configure_axes(ax: plt.Axes):
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(True, color=(0.85, 0.85, 0.85, 1.0), linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_legend(ax: plt.Axes):
    legend_kwargs = {
        "loc": LEGEND_LOC,
        "frameon": True,
        "framealpha": 1.0,
        "edgecolor": "black",
        "fontsize": LEGEND_FONT_SIZE,
        "handlelength": LEGEND_HANDLE_LENGTH,
        "borderpad": LEGEND_BORDERPAD,
        "labelspacing": LEGEND_LABEL_SPACING,
    }
    if LEGEND_BBOX_TO_ANCHOR is not None:
        legend_kwargs["bbox_to_anchor"] = LEGEND_BBOX_TO_ANCHOR
    legend = ax.legend(**legend_kwargs)
    if legend is not None:
        legend.get_frame().set_linewidth(LEGEND_FRAME_LW)


def plot_time_series_panel(ax: plt.Axes, host_time: np.ndarray, host_command: np.ndarray, responses: list[dict], profile_events_df: pd.DataFrame):
    configure_axes(ax)
    add_profile_event_spans(ax, profile_events_df)

    host_x, host_y = finite_pair(host_time, host_command)
    ax.plot(
        host_x,
        host_y,
        color=mcolors.to_rgba("#264653", 0.75),
        linewidth=1.0,
        label="Host command",
    )

    for response in responses:
        response_x, response_y = finite_pair(response["time"], response["value"])
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
            label=response["label"],
            zorder=8,
        )

    finite_values = collect_finite_values(host_command, responses)
    if finite_values.size > 0:
        y_min = float(np.min(finite_values))
        y_max = float(np.max(finite_values))
        y_span = y_max - y_min
        y_pad = max(2.0, 0.05 * y_span if y_span > 0.0 else 2.0)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    finite_time_blocks = [host_x]
    finite_time_blocks.extend(finite_pair(response["time"], response["value"])[0] for response in responses)
    finite_time_blocks = [block for block in finite_time_blocks if block.size > 0]
    if finite_time_blocks:
        all_times = np.concatenate(finite_time_blocks)
        ax.set_xlim(float(np.min(all_times)), float(np.max(all_times)))

    ax.set_title("Command and applied response", fontsize=12)
    ax.set_xlabel(r"Time, $t$ (s)", fontsize=11)
    ax.set_ylabel("Angle (deg)", fontsize=11)
    ax.tick_params(axis="both", labelsize=10)
    ax.tick_params(axis="x", labelrotation=-25)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    add_legend(ax)


def plot_latency_panel(ax: plt.Axes, workbook_sheets: dict[str, pd.DataFrame], active_surfaces: list[str]):
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

    ax.set_title("Latency decomposition", fontsize=12)
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

    ax.set_xticks(x_positions)
    ax.set_xticklabels(summary_df["Metric"], rotation=-20, ha="left", fontsize=10)
    ax.set_title("Latency summary statistics", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.tick_params(axis="y", labelsize=10)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.legend(loc="upper left", frameon=True, framealpha=1.0, edgecolor="black", fontsize=9)


def plot_integrity_panel(ax: plt.Axes, integrity_table_df: pd.DataFrame):
    ax.set_facecolor("white")
    ax.axis("off")
    ax.set_title("Integrity summary", fontsize=12)

    if integrity_table_df.empty:
        ax.text(0.5, 0.5, "IntegritySummary not available", ha="center", va="center", fontsize=11)
        return

    display_df = integrity_table_df.copy()
    for column_name in ["Rx miss %", "Apply miss %", "Dup %", "Unexpected %"]:
        display_df[column_name] = display_df[column_name].map(lambda value: f"{value:.2f}")

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


def plot_arduino_summary_figure(excel_path: Path, out_path: Path, workbook_bundle: dict):
    workbook_sheets = workbook_bundle["sheets"]
    fig = plt.figure(figsize=(11.5, 8.0), dpi=300, constrained_layout=True)
    fig.patch.set_facecolor("white")
    grid = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.20)

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
    plot_latency_panel(ax_latency, workbook_sheets, workbook_bundle["active_surfaces"])
    plot_summary_panel(ax_summary, workbook_bundle["latency_summary_df"])
    plot_integrity_panel(ax_integrity, workbook_bundle["integrity_table_df"])

    fig.suptitle(excel_path.stem, fontsize=13, y=0.995)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    workbook_dir = Path("C_Arduino_Test")
    excel_path = "Seed_5.xlsx"
    out_dir = workbook_dir / "A_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    workbook_path = resolve_workbook_path(excel_path, workbook_dir)
    workbook_bundle = build_workbook_bundle(workbook_path)
    out_path = out_dir / f"{workbook_path.stem}.png"

    plot_arduino_summary_figure(workbook_path, out_path, workbook_bundle)
    print(f"Done. Saved plot to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
