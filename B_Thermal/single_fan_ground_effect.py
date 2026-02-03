###### Initialization

### Imports
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    import cmocean
except ImportError:
    cmocean = None


### Parsing utilities (your sheet layout)
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
    and outlet height case (general/lower/higher).
    """
    xls = pd.ExcelFile(excel_path)
    sheet_names = xls.sheet_names

    heights = discover_measurement_heights(sheet_names)

    cases = {
        "general": {"sheet": lambda h: f"z{h:03d}_TS", "z_outlet": 0.33},
        "lower":   {"sheet": lambda h: f"z{h:03d}_Lower", "z_outlet": 0.245},
        "higher":  {"sheet": lambda h: f"z{h:03d}_Higher", "z_outlet": 0.425},
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


### Plotting
def add_mean_std_band(
    ax,
    x,
    y_const,
    mean,
    std,
    color,
    alpha_band=0.30,
    alpha_mean=0.10,
    mean_linewidth=1.0,
    errorbar_linewidth=0.75,
    errorbar_capsize=2.0,
):
    """
    Adds a filled polygon at y=y_const for the band [mean-std, mean+std].
    """
    upper = mean + std
    lower = mean - std

    verts_band = []
    # upper curve forward
    for xi, zi in zip(x, upper):
        verts_band.append((xi, y_const, zi))
    # lower curve backward to close
    for xi, zi in zip(x[::-1], lower[::-1]):
        verts_band.append((xi, y_const, zi))

    poly_band = Poly3DCollection(
        [verts_band], alpha=alpha_band, facecolor=color, edgecolor=color
    )
    ax.add_collection3d(poly_band)

    z_base = min(0.0, float(np.min(lower)))
    verts_mean = []
    for xi, zi in zip(x, upper):
        verts_mean.append((xi, y_const, zi))
    for xi, zi in zip(x[::-1], np.full_like(x, z_base)[::-1]):
        verts_mean.append((xi, y_const, zi))

    poly_mean = Poly3DCollection(
        [verts_mean], alpha=alpha_mean, facecolor=color, edgecolor="none"
    )
    ax.add_collection3d(poly_mean)

    # mean line
    ax.plot(
        x,
        np.full_like(x, y_const),
        mean,
        linewidth=mean_linewidth,
        color=color,
    )

    # error bars at each height: mean +/- std
    ax.errorbar(
        x,
        np.full_like(x, y_const),
        mean,
        zerr=std,
        fmt="none",
        ecolor=color,
        elinewidth=errorbar_linewidth,
        capsize=errorbar_capsize,
        capthick=errorbar_linewidth,
    )


def plot_point_3d(
    summary_df: pd.DataFrame,
    point_id: str,
    out_path: Path,
    mean_linewidth=1.0,
    errorbar_linewidth=0.75,
    errorbar_capsize=2.0,
):
    d = summary_df[summary_df["point"] == point_id].copy()
    if d.empty:
        return

    # Sort outlet heights so the stacked "slices" are ordered (draw lowermost first)
    outlet_levels = sorted(d["z_outlet_m"].unique(), reverse=True)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection="3d")

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(True)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        axis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 1.0)
        axis._axinfo["grid"]["linewidth"] = 0.4
    color_map = {
        0.245: "#eb7852",
        0.33: "#cc4a74",
        0.425: "#6200aa",
    }

    for y0 in outlet_levels:
        dd = d[d["z_outlet_m"] == y0].sort_values("z_meas_m")
        x = dd["z_meas_m"].to_numpy(dtype=float)
        mean = dd["mean_w"].to_numpy(dtype=float)
        std = dd["std_w"].to_numpy(dtype=float)

        color = color_map.get(y0, "#4c78a8")
        add_mean_std_band(
            ax,
            x=x,
            y_const=y0,
            mean=mean,
            std=std,
            color=color,
            mean_linewidth=mean_linewidth,
            errorbar_linewidth=errorbar_linewidth,
            errorbar_capsize=errorbar_capsize,
        )

        # all raw points
        for row in dd.itertuples(index=False):
            w_vals = np.asarray(row.w_vals, dtype=float)
            ax.scatter(
                np.full(w_vals.shape, row.z_meas_m),
                np.full(w_vals.shape, y0),
                w_vals,
                s=5,
                color=color,
                alpha=0.25,
            )

        # mean markers
        ax.scatter(
            x,
            np.full_like(x, y0),
            mean,
            s=15,
            color=color,
            alpha=1.0,
            edgecolors="none",
            zorder=9,
        )

    # Connect points at the same measurement height
    z_meas_vals = sorted(d["z_meas_m"].unique())
    if cmocean is not None:
        z_cmap = cmocean.cm.phase
    else:
        z_cmap = plt.get_cmap("plasma")
    z_min = min(z_meas_vals)
    z_max = max(z_meas_vals)
    if z_min == z_max:
        z_min -= 1e-6
        z_max += 1e-6
    for z_meas in sorted(d["z_meas_m"].unique()):
        t = (z_meas - z_min) / (z_max - z_min)
        line_color = z_cmap(t)
        dd_z = d[d["z_meas_m"] == z_meas].sort_values("z_outlet_m")
        if dd_z.empty:
            continue
        y_vals = dd_z["z_outlet_m"].to_numpy(dtype=float)
        z_vals = dd_z["mean_w"].to_numpy(dtype=float)
        std_vals = dd_z["std_w"].to_numpy(dtype=float)
        x_vals = np.full_like(y_vals, z_meas)
        ax.plot(
            x_vals,
            y_vals,
            z_vals,
            linestyle="--",
            color=line_color,
            linewidth=1,
            alpha=0.45,
            zorder=6,
        )
        for xi, yi, zi, si in zip(x_vals, y_vals, z_vals, std_vals):
            local_range = float(2.0 * si)
            z_text_offset = max(0.35, 0.75 * local_range)
            label_color = color_map.get(yi, "black")
            if yi == 0.245:
                label_zorder = 300
            elif yi == 0.33:
                label_zorder = 200
            elif yi == 0.425:
                label_zorder = 100
            else:
                label_zorder = 150
            ax.text(
                xi,
                yi,
                zi + z_text_offset,
                f"{zi:.2f}",
                color=label_color,
                fontsize=8,
                ha="center",
                zorder=label_zorder,
                clip_on=False,
                path_effects=[
                    pe.withStroke(
                        linewidth=2.5,
                        foreground=(1.0, 1.0, 1.0, 0.5),
                    )
                ],
            )

    # Labels / view
    ax.set_xlabel("Measurement height above fan outlet, z (m)", labelpad=17)
    ax.set_ylabel("Fan outlet height (m)", labelpad=5)
    ax.set_zlabel("w (m/s)", labelpad=5, rotation=90)
    ax.zaxis.set_rotate_label(False)
    ax.set_xticks(sorted(d["z_meas_m"].unique()))
    for label in ax.get_xticklabels():
        label.set_rotation(-20)
    ax.set_zlim(0, 8)
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    try:
        ax.set_box_aspect((3.0, 1.0, 1.5))
    except AttributeError:
        pass
    ax.set_yticks([0.245, 0.33, 0.425])

    ax.view_init(elev=20, azim=-120)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.08, top=0.98)

    extra_artists = (ax.xaxis.label, ax.yaxis.label, ax.zaxis.label)
    fig.savefig(
        out_path,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
        bbox_extra_artists=extra_artists,
    )
    plt.close(fig)


def main():
    excel_path = "S01.xlsx"  # <-- change if needed
    out_dir = Path("A_figures") / "Sampling_Points"
    out_dir.mkdir(parents=True, exist_ok=True)
    mean_linewidth = 1.0
    errorbar_linewidth = 0.75
    errorbar_capsize = 3.0

    summary = build_summary(excel_path)

    # Force P1..P6 in order (skip if missing)
    for k in range(1, 7):
        pid = f"P{k}"
        plot_point_3d(
            summary,
            pid,
            out_dir / f"{pid}_ground_effect_3D.png",
            mean_linewidth=mean_linewidth,
            errorbar_linewidth=errorbar_linewidth,
            errorbar_capsize=errorbar_capsize,
        )

    print(f"Done. Saved plots to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
