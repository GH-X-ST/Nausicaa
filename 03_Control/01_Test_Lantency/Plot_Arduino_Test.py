###### Initialization

### Imports
from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


### Legend styling
LEGEND_FONT_SIZE = 9
AXIS_EDGE_LW = 0.80
LEGEND_FRAME_LW = AXIS_EDGE_LW
LEGEND_LOC = "lower right"
LEGEND_BBOX_TO_ANCHOR = (1.05, 0.02)
LEGEND_HANDLE_LENGTH = 1.5
LEGEND_BORDERPAD = 0.5
LEGEND_LABEL_SPACING = 0.2


### Parsing utilities
def find_latest_workbook(workbook_dir: Path):
    workbook_paths = sorted(workbook_dir.glob("20260318_123632_Test.xlsx"))
    if len(workbook_paths) == 0:
        raise FileNotFoundError(f"No .xlsx files were found in: {workbook_dir}")
    return workbook_paths[-1]


def resolve_workbook_path(excel_path, workbook_dir: Path):
    if excel_path is None:
        return find_latest_workbook(workbook_dir)

    workbook_path = Path(excel_path)
    if not workbook_path.is_absolute():
        workbook_path = workbook_dir / workbook_path

    if not workbook_path.is_file():
        raise FileNotFoundError(f"Workbook not found: {workbook_path}")

    return workbook_path


def load_time_series(excel_path: Path):
    input_df = pd.read_excel(excel_path, sheet_name="InputSignal")
    echo_df = pd.read_excel(excel_path, sheet_name="ArduinoEcho")

    host_time = pd.to_numeric(input_df["time_s"], errors="coerce").to_numpy(dtype=float)
    host_command = pd.to_numeric(
        input_df["base_command_deg"], errors="coerce"
    ).to_numpy(dtype=float)

    response_styles = [
        {
            "surface_name": "Aileron_L",
            "label": "Aileron L",
            "color": "#2a9d8F",
            "marker": "o",
            "marker_size": 3,
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

    responses = []
    response_time = pd.to_numeric(echo_df["time_s"], errors="coerce").to_numpy(dtype=float)

    for style in response_styles:
        surface_name = style["surface_name"]
        value_column = f"{surface_name}_applied_equivalent_deg"
        if value_column not in echo_df.columns:
            continue

        response_value = pd.to_numeric(
            echo_df[value_column], errors="coerce"
        ).to_numpy(dtype=float)
        if not np.any(np.isfinite(response_value)):
            continue

        responses.append(
            {
                "label": style["label"],
                "color": style["color"],
                "marker": style["marker"],
                "marker_size": style["marker_size"],
                "marker_alpha": style["marker_alpha"],
                "time": response_time,
                "value": response_value,
            }
        )

    return host_time, host_command, responses


def finite_pair(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def collect_finite_values(host_command, responses):
    finite_blocks = [host_command[np.isfinite(host_command)]]

    for response in responses:
        finite_blocks.append(response["value"][np.isfinite(response["value"])])

    finite_blocks = [block for block in finite_blocks if block.size > 0]
    if len(finite_blocks) == 0:
        return np.array([], dtype=float)

    return np.concatenate(finite_blocks)


### Plotting
def plot_arduino_time_series(excel_path: Path, out_path: Path):
    host_time, host_command, responses = load_time_series(excel_path)

    fig, ax = plt.subplots(figsize=(5.67, 3.5), dpi=600)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(True, color=(0.85, 0.85, 0.85, 1.0), linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    mean_linewidth = 1.0
    line_alpha = 0.7
    axis_label_size = 12
    tick_label_size = 11

    host_x, host_y = finite_pair(host_time, host_command)
    ax.plot(
        host_x,
        host_y,
        color=mcolors.to_rgba("#264653", line_alpha),
        linewidth=mean_linewidth,
        label="Host",
    )
    for response in responses:
        response_x, response_y = finite_pair(response["time"], response["value"])
        ax.plot(
            response_x,
            response_y,
            color=mcolors.to_rgba(response["color"], line_alpha),
            linewidth=mean_linewidth,
            marker=response["marker"],
            markersize=response["marker_size"],
            markerfacecolor=mcolors.to_rgba(response["color"], response["marker_alpha"]),
            markeredgecolor=mcolors.to_rgba(response["color"], response["marker_alpha"]),
            markeredgewidth=0.0,
            markevery=1,
            label=response["label"],
            zorder=9,
        )

    finite_values = collect_finite_values(host_command, responses)
    if finite_values.size > 0:
        y_min = float(np.min(finite_values))
        y_max = float(np.max(finite_values))
        y_span = y_max - y_min
        y_pad = max(2.0, 0.05 * y_span if y_span > 0.0 else 2.0)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    finite_time_blocks = [host_x]
    finite_time_blocks.extend(
        finite_pair(response["time"], response["value"])[0] for response in responses
    )
    finite_time_blocks = [block for block in finite_time_blocks if block.size > 0]
    if len(finite_time_blocks) > 0:
        all_times = np.concatenate(finite_time_blocks)
        ax.set_xlim(float(np.min(all_times)), float(np.max(all_times)))

    ax.set_xlabel(r"Time, $t$ (s)", fontsize=axis_label_size)
    ax.set_ylabel("Command / response angle (deg)", fontsize=axis_label_size)
    ax.tick_params(axis="both", labelsize=tick_label_size)
    ax.tick_params(axis="x", labelrotation=-30)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
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
    leg = ax.legend(**legend_kwargs)
    if leg is not None:
        leg.get_frame().set_linewidth(LEGEND_FRAME_LW)

    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    workbook_dir = Path("C_Arduino_Test")
    excel_path = None  # <-- set to a workbook name or path to override the latest file
    out_dir = workbook_dir / "A_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    workbook_path = resolve_workbook_path(excel_path, workbook_dir)
    out_path = out_dir / f"{workbook_path.stem}.png"

    plot_arduino_time_series(workbook_path, out_path)
    print(f"Done. Saved plot to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
