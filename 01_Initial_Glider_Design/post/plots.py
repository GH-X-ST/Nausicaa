from __future__ import annotations

import copy

import numpy as onp

from config import Config
from post.postprocess import SolvedModel


def make_all_plots(cfg: Config, solved: SolvedModel) -> None:
    """Generate all figures into cfg.paths.figures_dir."""
    if not cfg.plot.make_plots:
        return

    cfg.paths.figures_dir.mkdir(parents=True, exist_ok=True)

    _plot_three_view(cfg, solved)
    _plot_mass_budget(cfg, solved)
    _plot_thermal_with_rollin_trajectory(cfg, solved)
    _plot_rollrate_and_aileron_vs_time(cfg, solved)


def _plot_three_view(cfg: Config, solved: SolvedModel) -> None:
    """Three-view drawing of solved airplane."""
    import aerosandbox.tools.pretty_plots as p

    solved.airplane.draw_three_view(show=False)
    p.show_plot(
        tight_layout=False,
        savefig=str(cfg.paths.figures_dir / "three_view.png"),
    )


def _plot_mass_budget(cfg: Config, solved: SolvedModel) -> None:
    """Mass budget pie chart."""
    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    fig, _ax = plt.subplots(
        figsize=(12, 5),
        subplot_kw=dict(aspect="equal"),
        dpi=cfg.plot.dpi,
    )

    name_remaps = {k: k.replace("_", " ").title() for k in solved.mass_props.keys()}

    mass_props_to_plot = copy.deepcopy(solved.mass_props)
    if "ballast" in mass_props_to_plot and mass_props_to_plot["ballast"].mass < 1e-6:
        mass_props_to_plot.pop("ballast")

    p.pie(
        values=[mp.mass for mp in mass_props_to_plot.values()],
        names=[
            n if n not in name_remaps else name_remaps[n]
            for n in mass_props_to_plot.keys()
        ],
        center_text=(
            f"$\\bf{{Mass\\ Budget}}$\nTOGW: {solved.mass_props_togw.mass * 1e3:.2f} g"
        ),
        label_format=lambda name, value, percentage: (
            f"{name}, {value * 1e3:.2f} g, {percentage:.1f}%"
        ),
        startangle=110,
        arm_length=30,
        arm_radius=20,
        y_max_labels=1.1,
    )

    p.show_plot(savefig=str(cfg.paths.figures_dir / "mass_budget.png"))


def _plot_thermal_with_rollin_trajectory(cfg: Config, solved: SolvedModel) -> None:
    """
    Thermal field contour + target orbit radius + roll-in trajectory colored by roll rate.
    """
    import matplotlib as mpl
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Experiment volume bounds
    x_min, x_max = 0.0, 8.0
    y_min, y_max = 0.0, 5.0

    # Thermal parameters from solved
    q_v = float(solved.thermal["q_v"])
    r_th0 = float(solved.thermal["r_th0"])
    k_th = float(solved.thermal["k_th"])
    z0 = float(solved.thermal["z0"])
    x_center = float(solved.thermal["x_center"])
    y_center = float(solved.thermal["y_center"])
    fan_spacing = float(solved.thermal["fan_spacing"])

    # Solved mission/roll quantities
    r_target = float(solved.mission["r_target"])
    z_plot = float(solved.mission["z_th"]) if "z_th" in solved.mission else float(cfg.mission.z_th)

    # Colormap with white fade-in at low values
    base_cmap = plt.cm.YlOrRd
    colors = base_cmap(onp.linspace(0, 1, 256))
    n_fade = 15
    first_color = colors[n_fade].copy()
    for i in range(n_fade):
        t_ = i / (n_fade - 1)
        colors[i] = (1 - t_) * onp.array([1, 1, 1, 1]) + t_ * first_color
    cmap_white0 = mcolors.ListedColormap(colors)

    fan_centres_plot = [
        (x_center - fan_spacing / 2, y_center - fan_spacing / 2),
        (x_center + fan_spacing / 2, y_center - fan_spacing / 2),
        (x_center - fan_spacing / 2, y_center + fan_spacing / 2),
        (x_center + fan_spacing / 2, y_center + fan_spacing / 2),
    ]

    def vertical_velocity_field_single(
        q_v_local: float,
        r_th0_local: float,
        k_th_local: float,
        x_local,
        y_local,
        z_local: float,
        z0_local: float,
        x_center_local: float,
        y_center_local: float,
    ):
        r_local = onp.sqrt((x_local - x_center_local) ** 2 + (y_local - y_center_local) ** 2)
        r_th_local = r_th0_local + k_th_local * (z_local - z0_local)
        r_th_local = onp.maximum(r_th_local, 1e-6)

        w_th_local = q_v_local / (onp.pi * r_th_local**2)
        w_local = w_th_local * onp.exp(-(r_local / r_th_local) ** 2)
        w_local = onp.where(z_local < z0_local, 0.0, w_local)
        return w_local

    def vertical_velocity_field_multi(
        q_v_local: float,
        r_th0_local: float,
        k_th_local: float,
        x_local,
        y_local,
        z_local: float,
        z0_local: float,
        fan_centres_local,
    ):
        w_total_local = 0.0
        for x_c, y_c in fan_centres_local:
            w_total_local += vertical_velocity_field_single(
                q_v_local=q_v_local,
                r_th0_local=r_th0_local,
                k_th_local=k_th_local,
                x_local=x_local,
                y_local=y_local,
                z_local=z_local,
                z0_local=z0_local,
                x_center_local=x_c,
                y_center_local=y_c,
            )
        return w_total_local

    # Grid
    nx, ny = 120, 80
    xg = onp.linspace(x_min, x_max, nx)
    yg = onp.linspace(y_min, y_max, ny)
    xg2, yg2 = onp.meshgrid(xg, yg, indexing="xy")
    zg2 = z_plot * onp.ones_like(xg2)

    w_slice = vertical_velocity_field_multi(
        q_v_local=q_v,
        r_th0_local=r_th0,
        k_th_local=k_th,
        x_local=xg2,
        y_local=yg2,
        z_local=zg2,
        z0_local=z0,
        fan_centres_local=fan_centres_plot,
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=cfg.plot.dpi)

    levels = onp.linspace(w_slice.min(), w_slice.max(), 256)
    cf = ax.contourf(xg2, yg2, w_slice, levels=levels, cmap=cmap_white0, zorder=1)

    divider = make_axes_locatable(ax)
    cax_w = divider.append_axes("right", size="3.8%", pad=0.25)

    # Initial thermal radius circles
    theta_circle = onp.linspace(0, 2 * onp.pi, 200)
    for x_c, y_c in fan_centres_plot:
        ax.plot(
            x_c + r_th0 * onp.cos(theta_circle),
            y_c + r_th0 * onp.sin(theta_circle),
            color="k",
            linewidth=1.3,
            zorder=0,
        )

    # Target orbit radius
    ax.plot(
        x_center + r_target * onp.cos(theta_circle),
        y_center + r_target * onp.sin(theta_circle),
        color="k",
        linestyle="--",
        linewidth=1.3,
        zorder=900,
    )

    x_txt = x_center + r_target
    y_txt = y_center
    ax.text(
        x_txt + 0.04 * r_target,
        y_txt,
        rf"$R_\mathrm{{target}} = {r_target:.2f}\,\mathrm{{m}}$",
        fontsize=11.5,
        color="k",
        verticalalignment="center",
        zorder=950,
    )

    # Roll-in trajectory colored by roll rate
    xr = onp.asarray(solved.roll["x"], dtype=float)
    yr = onp.asarray(solved.roll["y"], dtype=float)
    p_roll_arr = onp.degrees(onp.asarray(solved.roll["p_roll"], dtype=float))

    if xr.size >= 2:
        points = onp.stack([xr, yr], axis=1).reshape(-1, 1, 2)
        segments = onp.concatenate([points[:-1], points[1:]], axis=1)

        p_abs = onp.abs(p_roll_arr)
        p_seg = 0.5 * (p_abs[:-1] + p_abs[1:])

        p_min = float(p_seg.min())
        p_max = float(p_seg.max())
        if p_max <= p_min:
            p_max = p_min + 1e-6

        norm_p = mpl.colors.Normalize(vmin=p_min, vmax=p_max)

        lc = LineCollection(
            segments,
            cmap="winter",
            norm=norm_p,
            array=p_seg,
            linewidth=1.5,
            zorder=1100,
        )
        ax.add_collection(lc)

        cmap_p = mpl.colormaps["winter"]
        c_start = cmap_p(norm_p(p_seg[0]))
        c_end = cmap_p(norm_p(p_seg[-1]))

        ax.scatter(
            [xr[0]],
            [yr[0]],
            s=35,
            marker="o",
            facecolor=c_start,
            edgecolor="k",
            linewidth=0.5,
            zorder=1200,
        )
        ax.scatter(
            [xr[-1]],
            [yr[-1]],
            s=55,
            marker="X",
            facecolor=c_end,
            edgecolor="k",
            linewidth=0.5,
            zorder=1200,
        )

        cax_p = divider.append_axes("left", size="3.8%", pad=0.55)
        cbar_p = fig.colorbar(lc, cax=cax_p)
        cax_p.yaxis.set_ticks_position("left")
        cax_p.yaxis.set_label_position("left")
        cbar_p.set_label(r"$p$ (deg/s)")

        cbar_w = fig.colorbar(cf, cax=cax_w)
        cbar_w.set_label(f"w (m/s) at z = {z_plot:.2f} m")
    else:
        ax.plot(xr, yr, color="k", linewidth=1.5, zorder=1100)
        cbar_w = fig.colorbar(cf, cax=cax_w)
        cbar_w.set_label(f"w (m/s) at z = {z_plot:.2f} m")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")

    ax.tick_params(axis="both", colors="k", which="both")
    ax.xaxis.label.set_color("k")
    ax.yaxis.label.set_color("k")
    for spine in ax.spines.values():
        spine.set_color("k")
        spine.set_linewidth(1.0)

    ax.set_axisbelow(False)
    ax.grid(True, linestyle=":", linewidth=0.5, color="k", alpha=0.6, zorder=1000)

    fig.tight_layout()
    fig.savefig(
        str(cfg.paths.figures_dir / "thermal_with_rollin_trajectory.png"),
        dpi=cfg.plot.dpi,
        bbox_inches="tight",
    )


def _plot_rollrate_and_aileron_vs_time(cfg: Config, solved: SolvedModel) -> None:
    """
    Roll rate and aileron deflection vs time.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    t_roll = float(solved.roll["t_roll"])

    x_roll = onp.asarray(solved.roll["x"], dtype=float)
    n = int(len(onp.atleast_1d(x_roll)))
    t_vec = onp.linspace(0.0, t_roll, n)

    p_ts = onp.degrees(onp.asarray(solved.roll["p_roll"], dtype=float))
    da_ts = onp.asarray(solved.roll["delta_a_roll_deg"], dtype=float)

    fig_ts, ax_p = plt.subplots(
        figsize=(7.2, 4.8),
        dpi=cfg.plot.dpi,
        facecolor="white",
    )
    ax_p.set_facecolor("white")

    c_p = "#0000CD"
    c_da = "#DC143C"

    (line_p,) = ax_p.plot(
        t_vec,
        p_ts,
        color=c_p,
        alpha=0.75,
        linewidth=1.3,
        label=r"$p$ (deg/s)",
    )
    ax_p.set_xlabel("t (s)")
    ax_p.set_ylabel("p (deg/s)")

    p_top = 100.0
    p_step = 20.0
    ax_p.set_xlim(left=0.0)
    ax_p.set_ylim(0.0, p_top)
    ax_p.yaxis.set_major_locator(mticker.MultipleLocator(p_step))

    ax_p.grid(True, which="major", linestyle=":", linewidth=0.5, color="k", alpha=0.35)
    ax_p.set_axisbelow(True)

    ax_p.tick_params(axis="y", colors=c_p, which="both")
    ax_p.yaxis.label.set_color(c_p)

    ax_p.tick_params(axis="x", colors="k", which="both")
    ax_p.xaxis.label.set_color("k")

    ax_p.spines["left"].set_visible(True)
    ax_p.spines["bottom"].set_visible(True)
    ax_p.spines["left"].set_color(c_p)
    ax_p.spines["bottom"].set_color("k")
    ax_p.spines["left"].set_linewidth(1.2)
    ax_p.spines["bottom"].set_linewidth(1.2)
    ax_p.spines["left"].set_zorder(10)
    ax_p.spines["bottom"].set_zorder(10)
    ax_p.spines["top"].set_color("k")
    ax_p.spines["top"].set_linewidth(1.0)

    ax_da = ax_p.twinx()
    ax_da.set_facecolor("none")
    ax_da.patch.set_alpha(0.0)

    (line_da,) = ax_da.plot(
        t_vec,
        da_ts,
        color=c_da,
        alpha=0.75,
        linewidth=1.3,
        label=r"$\delta_A$ (deg)",
    )
    ax_da.set_ylabel(r"$\delta_A$ (deg)")

    da_top = 25.0
    da_step = 5.0
    ax_da.set_ylim(0.0, da_top)
    ax_da.yaxis.set_major_locator(mticker.MultipleLocator(da_step))

    ax_da.grid(False)

    ax_da.tick_params(axis="y", colors=c_da, which="both")
    ax_da.yaxis.label.set_color(c_da)
    ax_da.spines["right"].set_color(c_da)
    ax_da.spines["right"].set_linewidth(1.2)
    ax_da.spines["right"].set_zorder(10)

    ax_da.spines["top"].set_color("k")
    ax_da.spines["top"].set_linewidth(1.0)
    ax_da.spines["left"].set_visible(False)
    ax_da.spines["bottom"].set_visible(False)

    leg = ax_p.legend(
        handles=[line_p, line_da],
        labels=[line_p.get_label(), line_da.get_label()],
        loc="best",
        frameon=True,
    )
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(1.0)
    leg.get_frame().set_edgecolor((0, 0, 0, 0.75))
    leg.get_frame().set_linewidth(0.75)

    fig_ts.tight_layout()
    fig_ts.savefig(
        str(cfg.paths.figures_dir / "rollrate_and_aileron_vs_time.png"),
        dpi=cfg.plot.dpi,
        bbox_inches="tight",
        facecolor=fig_ts.get_facecolor(),
    )