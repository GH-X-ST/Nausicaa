import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import FormatStrFormatter


# -----------------------------
# Parsing utilities (your sheet layout)
# -----------------------------
def parse_points_multiheader(excel_path: str, sheet_name: str):
    """
    Parse TS-like sheets where 3 points are laid out across columns,
    and points 4-6 appear in a second section below with a repeated header row.
    Returns a list of points in order: P1..P6, each with x,y,w,mean,std,n.
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)

    # Identify header rows: first cell == 'x'
    header_rows = []
    for r in range(df.shape[0]):
        v = df.iat[r, 0]
        if isinstance(v, str) and v.strip().lower() == "x":
            header_rows.append(r)
    header_rows.sort()

    points = []

    for idx, hr in enumerate(header_rows):
        r_start = hr + 1
        r_end = header_rows[idx + 1] if idx + 1 < len(header_rows) else df.shape[0]

        seg = df.iloc[r_start:r_end, :].dropna(how="all")
        if seg.empty:
            continue

        header = df.iloc[hr].tolist()
        # block starts are where header cell is "x" across columns
        starts = [i for i, v in enumerate(header) if isinstance(v, str) and v.strip().lower() == "x"]

        for si in starts:
            if si + 3 >= df.shape[1]:
                continue

            x = seg.iloc[0, si]
            y = seg.iloc[0, si + 1]
            w_col = seg.iloc[:, si + 3]

            w_numeric = pd.to_numeric(w_col, errors="coerce").dropna()
            w = w_numeric.to_numpy(dtype=float)

            if w.size == 0:
                continue

            mean = float(np.mean(w))
            std = float(np.std(w, ddof=1)) if w.size > 1 else 0.0

            points.append(
                {
                    "x": float(x) if pd.notna(x) else None,
                    "y": float(y) if pd.notna(y) else None,
                    "w": w,
                    "mean": mean,
                    "std": std,
                    "n": int(w.size),
                }
            )

    return points


def discover_measurement_heights(sheet_names):
    """
    Finds all measurement heights h from sheets named like z020_TS, z035_TS, ...
    Returns sorted list of integer heights in cm-like format (e.g., 20,35,50,...).
    """
    heights = []
    for s in sheet_names:
        m = re.fullmatch(r"z(\d{3})_TS", s)
        if m:
            heights.append(int(m.group(1)))
    return sorted(set(heights))


def build_summary(excel_path: str):
    """
    Build a tidy summary table with mean/std for each point, measurement height,
    and outlet height case (TS only).
    """
    xls = pd.ExcelFile(excel_path)
    sheet_names = xls.sheet_names

    heights = discover_measurement_heights(sheet_names)

    cases = {
        "TS": {"sheet": lambda h: f"z{h:03d}_TS", "z_outlet": 960},
    }

    records = []
    for h in heights:
        z_meas = h / 100.0
        for case, meta in cases.items():
            sheet = meta["sheet"](h)
            if sheet not in sheet_names:
                continue

            pts = parse_points_multiheader(excel_path, sheet)

            # P1..P6 (or fewer if a sheet is incomplete)
            for i, p in enumerate(pts, start=1):
                records.append(
                    {
                        "point": f"P{i}",
                        "x_m": p["x"],
                        "y_m": p["y"],
                        "z_meas_m": z_meas,
                        "z_outlet_m": meta["z_outlet"],
                        "case": case,
                        "w_vals": p["w"],
                        "mean_w": p["mean"],
                        "std_w": p["std"],
                        "n": p["n"],
                    }
                )

    return pd.DataFrame.from_records(records)


# -----------------------------
# Plotting (2D mean +/- std)
# -----------------------------
def plot_point_2d(summary_df: pd.DataFrame, point_id: str, out_path: Path):
    d = summary_df[summary_df["point"] == point_id].copy()
    if d.empty:
        return

    dd = d.sort_values("z_meas_m")
    x = dd["z_meas_m"].to_numpy(dtype=float)
    mean = dd["mean_w"].to_numpy(dtype=float)
    std = dd["std_w"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(5.67, 3.5), dpi=600)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(True, color=(0.0, 0.0, 0.0, 0.1), linewidth=0.1)

    color = "#16058b"
    ax.plot(x, mean, color=color, linewidth=1.8, marker="o", markersize=3)
    ax.errorbar(
        x,
        mean,
        yerr=std,
        fmt="none",
        ecolor=color,
        elinewidth=1.0,
        capsize=2.0,
        alpha=0.9,
        zorder=5,
    )
    z_base = min(0.0, float(np.min(mean - std)))
    ax.fill_between(
        x,
        z_base,
        mean + std,
        color=color,
        alpha=0.18,
        edgecolor="none",
    )
    ax.fill_between(
        x,
        mean - std,
        mean + std,
        color=color,
        alpha=0.32,
        edgecolor="none",
    )
    ax.plot(x, mean + std, color=color, linewidth=1.0, alpha=0.32)
    ax.plot(x, mean - std, color=color, linewidth=1.0, alpha=0.32)

    y_range = float(np.max(mean + std) - np.min(mean - std))
    y_text_offset = 0.03 * y_range if y_range > 0 else 0.03
    for xi, mi in zip(x, mean):
        ax.text(
            xi,
            mi + y_text_offset,
            f"{mi:.2f}",
            color=color,
            fontsize=8,
            ha="center",
            va="bottom",
            clip_on=False,
            zorder=20,
            path_effects=[
                pe.withStroke(
                    linewidth=3,
                    foreground=(1.0, 1.0, 1.0, 0.6),
                )
            ],
        )

    # all raw points (small, semi-transparent)
    for row in dd.itertuples(index=False):
        w_vals = np.asarray(row.w_vals, dtype=float)
        ax.scatter(
            np.full(w_vals.shape, row.z_meas_m),
            w_vals,
            s=5,
            color=color,
            alpha=0.25,
        )

    ax.set_xlabel("Measurement height above fan outlet, z (m)")
    ax.set_ylabel("w (m/s)")
    ax.set_xticks(sorted(d["z_meas_m"].unique()))
    for label in ax.get_xticklabels():
        label.set_rotation(-20)
    ax.set_ylim(0, 7)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    fig.tight_layout()
    fig.savefig(out_path, dpi=600)
    plt.close(fig)


def main():
    excel_path = "S02.xlsx"  # <-- change if needed
    out_dir = Path("A_figures") / "Sampling_Points"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary(excel_path)

    # Force P1..P6 in order (skip if missing)
    for k in range(1, 7):
        pid = f"P{k}"
        plot_point_2d(summary, pid, out_dir / f"{pid}_four_fan_time_series.png")

    print(f"Done. Saved plots to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
