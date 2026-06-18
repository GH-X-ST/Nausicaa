"""Matched E4a real-flight and replay comparison for the thesis.

The figure reuses the thesis composite layout, but replaces the simulation
memory/no-memory/open-loop policies with real-flight traces:

* real closed-loop memory flight, E4a.2 throw_018;
* simulation replay of the same closed-loop decisions;
* matched real open-loop failure, E4a.0 throw_008;
* simulation replay of the matched open-loop flight.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from plot_r11_e03_case0082_composite import (  # noqa: E402
    AXIS_EDGE_LW,
    DPI,
    FIGURE_GRID,
    FIGURE_GRID_MARGINS,
    FIGURE_SAVE_PAD_IN,
    FIGURE_SIZE_IN,
    PANEL_2D_LAYOUT,
    PANEL_3D_AXIS_LABELSIZE,
    PANEL_3D_BOUNDS,
    PANEL_3D_LABELPADS,
    PANEL_3D_LAYOUT,
    PANEL_3D_TICK_LABELSIZE,
    PANEL_3D_TICKS,
    PANEL_3D_VIEW,
    REPO_ROOT,
    STATE_AXIS_EDGE_LW,
    STATE_AXIS_LABELSIZE,
    STATE_COLUMN_TITLE_SIZE,
    STATE_GRID_COLOR,
    STATE_GRID_LINEWIDTH,
    STATE_TICK_LABELSIZE,
    STATE_TICK_LENGTH,
    STATE_TICK_WIDTH,
    STATE_XLIM_S,
    STATE_XTICK_LABELS,
    STATE_XTICKS_S,
    STATE_YLABEL_PAD,
    _apply_heatmap_2d_rcparams,
    _build_alpha_cmap,
    _compact_y_tick_label,
    _draw_arena_box,
    _draw_center_slices,
    _draw_environment_fan_outlets,
    _draw_floor_grid,
    _draw_updraft_isosurfaces,
    _sample_updraft_volume,
)

REPLAY_PLOTTING_ROOT = REPO_ROOT / "04_Flight_Test" / "00_Plotting"
if str(REPLAY_PLOTTING_ROOT) not in sys.path:
    sys.path.insert(0, str(REPLAY_PLOTTING_ROOT))

from env_ctx import EnvironmentMetadata  # noqa: E402
from env_surrogate import resolve_surrogate_binding, wind_field_for_binding  # noqa: E402

@dataclass(frozen=True)
class R9PreflightFigureConfig:
    updraft_nx: int = 56
    updraft_ny: int = 36
    updraft_nz: int = 30


@dataclass(frozen=True)
class ThrowCase:
    case_id: str
    session_id: str
    throw_id: str
    throw_root: Path
    summary: dict[str, object]
    manifest: dict[str, object]


@dataclass(frozen=True)
class ReplayEnvironment:
    wind_model: object | None
    wind_mode: str
    fan_positions_m: tuple[tuple[float, float], ...]
    metadata: dict[str, object]


def replay_environment_for_throw(throw: ThrowCase) -> ReplayEnvironment:
    fan_positions = fan_positions_from_log(throw.throw_root)
    if not fan_positions:
        return ReplayEnvironment(
            wind_model=None,
            wind_mode="none",
            fan_positions_m=(),
            metadata={
                "updraft_context_status": "no_visible_fan_position",
                "W_layer": "W0",
                "environment_mode": "dry_air",
                "updraft_model_id": "dry_air_zero_wind",
                "updraft_max_m_s": 0.0,
            },
        )
    active_mask = tuple(True for _ in fan_positions)
    fan_count = len(fan_positions)
    metadata = EnvironmentMetadata(
        environment_id=f"real_flight_{throw.case_id}_{throw.session_id}_{throw.throw_id}_measured_fan",
        environment_instance_id=f"real_flight_{throw.case_id}_{throw.session_id}_{throw.throw_id}_measured_fan",
        fan_count=int(fan_count),
        fan_positions_m=fan_positions,
        fan_power_scales=tuple(1.0 for _ in fan_positions),
        active_fan_mask=active_mask,
        updraft_model_id="four_annular_gp_grid" if fan_count >= 4 else "single_annular_gp_grid",
        updraft_amplitude_scale=1.0,
        updraft_width_scale=1.0,
        updraft_centre_shift_m=(0.0, 0.0),
        residual_field_id="real_flight_measured_fan_position_nominal_strength",
        local_uncertainty_scale=1.0,
        randomisation_seed=None,
        model_source="real_flight_fan_positions_csv",
        W_layer="W2",
        wind_mode="panel",
        environment_mode="annular_gp_four" if fan_count >= 4 else "annular_gp_single",
        claim_status="real_flight_replay_measured_fan_position_uses_nominal_w2_annular_gp",
    )
    binding = resolve_surrogate_binding("W2", metadata, repo_root=REPO_ROOT)
    wind_model = wind_field_for_binding(binding, repo_root=REPO_ROOT)
    return ReplayEnvironment(
        wind_model=wind_model,
        wind_mode="panel" if wind_model is not None else "none",
        fan_positions_m=tuple((float(x), float(y)) for x, y in fan_positions),
        metadata={
            "updraft_context_status": "measured_fan_w2_annular_gp" if wind_model is not None else f"no_ready_wind_field:{binding.blocked_reason}",
            "W_layer": binding.W_layer,
            "environment_mode": binding.environment_mode,
            "updraft_model_id": binding.updraft_model_id,
            "surrogate_family": binding.surrogate_family,
            "surrogate_role": binding.surrogate_role,
            "surrogate_binding_status": binding.surrogate_binding_status,
            "surrogate_blocked_reason": binding.blocked_reason,
            "fan_count": int(binding.fan_count),
            "active_fan_count": int(sum(bool(value) for value in binding.active_fan_mask)),
            "fan_positions_m": json.dumps(binding.fan_positions_m),
            "fan_power_scales": json.dumps(binding.fan_power_scales),
            "active_fan_mask": ";".join("1" if value else "0" for value in binding.active_fan_mask),
            "updraft_amplitude_scale": float(binding.updraft_amplitude_scale),
            "updraft_width_scale": float(binding.updraft_width_scale),
            "local_uncertainty_scale": float(binding.local_uncertainty_scale),
            "wind_mode": "panel" if wind_model is not None else "none",
        },
    )


def fan_positions_from_log(throw_root: Path) -> tuple[tuple[float, float], ...]:
    fan_path = throw_root / "metrics" / "fan_positions.csv"
    if not fan_path.is_file():
        return ()
    frame = pd.read_csv(fan_path)
    positions: list[tuple[float, float]] = []
    seen: set[str] = set()
    for _, row in frame.iloc[::-1].iterrows():
        if "visible" in row and not _truthy(row.get("visible")):
            continue
        subject = str(row.get("fan_subject", ""))
        if subject in seen:
            continue
        x = _to_float(row.get("x_w"))
        y = _to_float(row.get("y_w"))
        if math.isfinite(x) and math.isfinite(y):
            positions.append((x, y))
            seen.add(subject)
    return tuple(reversed(positions))


def _truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "nan"}
    try:
        return bool(value)
    except Exception:
        return False


def _to_float(value: object, default: float = float("nan")) -> float:
    try:
        numeric = float(value)
    except Exception:
        return default
    return numeric if math.isfinite(numeric) else default


OUTPUT_DIR = REPO_ROOT / "04_Flight_Test" / "A_figures"

OUTPUT_PATH = OUTPUT_DIR / "e4a_matched_reality_replay_openloop_composite.png"
SOURCE_ROOT = OUTPUT_DIR / "e4a_matched_replay_source"
UPDRAFT_RENDER_MODE = "r11_composed_measured_fans"

CLOSED_LOOP_TRACE = (
    SOURCE_ROOT
    / "metrics"
    / "E4a.2"
    / "20260607_231704"
    / "throw_018_replay_traces.csv"
)
OPEN_LOOP_TRACE = (
    SOURCE_ROOT
    / "metrics"
    / "E4a.0"
    / "20260607_230250"
    / "throw_008_replay_traces.csv"
)
CLOSED_LOOP_THROW_ROOT = (
    REPO_ROOT
    / "04_Flight_Test"
    / "05_Results"
    / "E4a.2"
    / "20260607_231704"
    / "throw_018"
)
OPEN_LOOP_THROW_ROOT = (
    REPO_ROOT
    / "04_Flight_Test"
    / "05_Results"
    / "E4a.0"
    / "20260607_230250"
    / "throw_008"
)

REAL_CLOSED_COLOR = "#005EB8"
REPLAY_CLOSED_COLOR = "#00A3E0"
OPEN_LOOP_COLOR = "#FF4202"
OPEN_LOOP_ALPHA = 0.70
OPEN_LOOP_REPLAY_COLOR = "#111111"
OPEN_LOOP_REPLAY_ALPHA = 0.86


@dataclass(frozen=True)
class TraceSpec:
    key: str
    label: str
    frame: pd.DataFrame
    color: str
    linestyle: str
    linewidth: float
    alpha: float
    zorder: int


def _render_current_case() -> int:
    traces = _load_traces()
    _apply_heatmap_2d_rcparams()

    fig = plt.figure(figsize=FIGURE_SIZE_IN)
    gs = fig.add_gridspec(
        FIGURE_GRID["nrows"],
        FIGURE_GRID["ncols"],
        width_ratios=FIGURE_GRID["width_ratios"],
        **FIGURE_GRID_MARGINS,
    )
    ax3d = fig.add_subplot(
        gs[PANEL_3D_LAYOUT["row"], PANEL_3D_LAYOUT["col"]],
        projection="3d",
    )
    gs2d = gs[PANEL_2D_LAYOUT["outer_row"], PANEL_2D_LAYOUT["outer_col"]].subgridspec(
        PANEL_2D_LAYOUT["row_count"],
        PANEL_2D_LAYOUT["column_count"],
        width_ratios=PANEL_2D_LAYOUT["width_ratios"],
        wspace=PANEL_2D_LAYOUT["wspace"],
        hspace=PANEL_2D_LAYOUT["hspace"],
    )
    mini_axes = [
        [
            fig.add_subplot(gs2d[row, col])
            for col in range(PANEL_2D_LAYOUT["column_count"])
        ]
        for row in range(PANEL_2D_LAYOUT["row_count"])
    ]

    _draw_trajectory_panel(ax3d, traces)
    _draw_state_panels(mini_axes, traces)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTPUT_PATH,
        dpi=DPI,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=FIGURE_SAVE_PAD_IN,
    )
    plt.close(fig)

    metadata = {
        "figure": OUTPUT_PATH.name,
        "source_root": SOURCE_ROOT.as_posix(),
        "closed_loop_real_throw": CLOSED_LOOP_THROW_ROOT.as_posix(),
        "open_loop_real_throw": OPEN_LOOP_THROW_ROOT.as_posix(),
        "closed_loop_trace_csv": CLOSED_LOOP_TRACE.as_posix(),
        "open_loop_trace_csv": OPEN_LOOP_TRACE.as_posix(),
        "trace_policy": _trace_policy_metadata(),
        "open_loop_color": OPEN_LOOP_COLOR,
        "open_loop_replay_color": OPEN_LOOP_REPLAY_COLOR,
        "open_loop_real_alpha": OPEN_LOOP_ALPHA,
        "open_loop_replay_alpha": OPEN_LOOP_REPLAY_ALPHA,
        "updraft_render_mode": UPDRAFT_RENDER_MODE,
    }
    OUTPUT_PATH.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(OUTPUT_PATH.as_posix())
    return 0


def _load_traces() -> list[TraceSpec]:
    closed = pd.read_csv(CLOSED_LOOP_TRACE)
    open_loop = pd.read_csv(OPEN_LOOP_TRACE)
    specs = [
        TraceSpec(
            key="real_closed_loop",
            label="real closed-loop",
            frame=_trace(closed, "real"),
            color=REAL_CLOSED_COLOR,
            linestyle="-",
            linewidth=2.10,
            alpha=0.98,
            zorder=35,
        ),
        TraceSpec(
            key="closed_loop_replay",
            label="sim replay",
            frame=_trace(closed, "sim_real_decisions"),
            color=REPLAY_CLOSED_COLOR,
            linestyle="--",
            linewidth=1.80,
            alpha=0.95,
            zorder=28,
        ),
        TraceSpec(
            key="real_open_loop",
            label="matched open-loop",
            frame=_trace(open_loop, "real"),
            color=OPEN_LOOP_COLOR,
            linestyle="-",
            linewidth=1.95,
            alpha=OPEN_LOOP_ALPHA,
            zorder=24,
        ),
        TraceSpec(
            key="open_loop_replay",
            label="open-loop replay",
            frame=_trace(open_loop, "sim_real_decisions"),
            color=OPEN_LOOP_REPLAY_COLOR,
            linestyle=":",
            linewidth=1.60,
            alpha=OPEN_LOOP_REPLAY_ALPHA,
            zorder=20,
        ),
    ]
    return specs


def _trace_policy_metadata() -> dict[str, str]:
    closed_case, _, closed_throw = CLOSED_LOOP_THROW_ROOT.parts[-3:]
    open_case, _, open_throw = OPEN_LOOP_THROW_ROOT.parts[-3:]
    return {
        "real_closed_loop": f"{closed_case} {closed_throw} model=real",
        "closed_loop_replay": f"{closed_case} {closed_throw} model=sim_real_decisions",
        "real_open_loop": f"{open_case} {open_throw} model=real",
        "open_loop_replay": f"{open_case} {open_throw} model=sim_real_decisions",
    }


def _trace(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    out = frame[frame["model"].astype(str).eq(model)].copy()
    if out.empty:
        raise RuntimeError(f"No trace rows found for model={model!r}.")
    out = out.sort_values("t_s").drop_duplicates(subset=["t_s"], keep="last")
    return out.reset_index(drop=True)


def _draw_trajectory_panel(ax, traces: list[TraceSpec]) -> None:
    ax.set_facecolor("white")
    ax.grid(True)
    _draw_arena_box(ax)
    _draw_floor_grid(ax)
    _draw_fans(ax)
    _configure_3d_axis(ax)

    for spec in traces:
        frame = spec.frame
        ax.plot(
            frame["x_w"].to_numpy(float),
            frame["y_w"].to_numpy(float),
            frame["z_w"].to_numpy(float),
            color=spec.color,
            linestyle=spec.linestyle,
            linewidth=spec.linewidth,
            alpha=spec.alpha,
            label=spec.label,
            zorder=spec.zorder,
        )
        marker = "o" if _is_exit_face(spec) else "x"
        ax.scatter(
            [float(frame["x_w"].iloc[-1])],
            [float(frame["y_w"].iloc[-1])],
            [float(frame["z_w"].iloc[-1])],
            marker=marker,
            s=48,
            color=spec.color,
            alpha=min(1.0, spec.alpha + 0.10),
            depthshade=False,
            zorder=spec.zorder + 2,
        )

    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 1.10),
        frameon=True,
        framealpha=1.0,
        edgecolor="black",
        fontsize=10.5,
        handlelength=1.9,
        borderpad=0.35,
        labelspacing=0.18,
    )
    if legend is not None:
        legend.get_frame().set_linewidth(AXIS_EDGE_LW)


def _draw_fans(ax) -> None:
    case_id, session_id, throw_id = CLOSED_LOOP_THROW_ROOT.parts[-3:]
    throw = ThrowCase(
        case_id=case_id,
        session_id=session_id,
        throw_id=throw_id,
        throw_root=CLOSED_LOOP_THROW_ROOT,
        summary={},
        manifest={},
    )
    replay_environment = replay_environment_for_throw(throw)
    fan_positions = tuple(replay_environment.fan_positions_m or ())
    if not fan_positions:
        return
    if UPDRAFT_RENDER_MODE == "replay_environment":
        wind = replay_environment.wind_model
    else:
        wind = _centered_plot_wind(fan_positions)
    if wind is not None:
        _draw_updraft_volume(ax, wind, fan_positions)
    _draw_environment_fan_outlets(ax, fan_positions)


def _centered_plot_wind(fan_positions: tuple[tuple[float, float], ...]):
    """Build the same composed annular-GP field used by R11 full-domain plots."""

    try:
        fan_count = len(fan_positions)
        metadata = EnvironmentMetadata(
            environment_id="thesis_replay_visual_measured_fans",
            environment_instance_id="thesis_replay_visual_measured_fans",
            fan_count=int(fan_count),
            fan_positions_m=tuple((float(x), float(y)) for x, y in fan_positions),
            fan_power_scales=tuple(1.0 for _ in fan_positions),
            active_fan_mask=tuple(True for _ in fan_positions),
            updraft_model_id="four_annular_gp_grid" if fan_count >= 4 else "single_annular_gp_grid",
            updraft_amplitude_scale=1.0,
            updraft_width_scale=1.0,
            updraft_centre_shift_m=(0.0, 0.0),
            residual_field_id="thesis_replay_visual_measured_fan_positions",
            local_uncertainty_scale=1.0,
            randomisation_seed=0,
            model_source="measured_fan_positions_from_real_flight_replay",
            W_layer="W3",
            wind_mode="panel",
            environment_mode="annular_gp_four_full_domain_randomised"
            if fan_count >= 4
            else "annular_gp_single_full_domain_randomised",
            claim_status="plot_only_same_composed_annular_gp_path_as_r11_full_domain",
        )
        binding = resolve_surrogate_binding(
            "W3",
            metadata,
            repo_root=REPO_ROOT,
            randomisation_seed=0,
        )
        return wind_field_for_binding(binding, repo_root=REPO_ROOT)
    except Exception:
        return None


def _draw_updraft_volume(
    ax,
    wind,
    fan_positions: tuple[tuple[float, float], ...],
) -> None:
    config = R9PreflightFigureConfig()
    x_vec, y_vec, z_vec, w_grid = _sample_updraft_volume(wind, config)
    if not w_grid.size or float(np.nanmax(w_grid)) <= 1e-9:
        return
    cmap_alpha = _build_alpha_cmap()
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=8.0, clip=True)
    _draw_center_slices(
        ax,
        x_vec=x_vec,
        y_vec=y_vec,
        z_vec=z_vec,
        w_grid=w_grid,
        fan_positions=fan_positions,
        cmap_alpha=cmap_alpha,
        norm=norm,
    )
    _draw_updraft_isosurfaces(
        ax,
        x_vec=x_vec,
        y_vec=y_vec,
        z_vec=z_vec,
        w_grid=w_grid,
        cmap_alpha=cmap_alpha,
        norm=norm,
    )


def _configure_3d_axis(ax) -> None:
    ax.set_xlim(*PANEL_3D_BOUNDS["x"])
    ax.set_ylim(*PANEL_3D_BOUNDS["y"])
    ax.set_zlim(*PANEL_3D_BOUNDS["z"])
    ax.set_xticks(PANEL_3D_TICKS["x"])
    ax.set_yticks(PANEL_3D_TICKS["y"])
    ax.set_zticks(PANEL_3D_TICKS["z"])
    ax.set_xlabel("$x$ (m)", labelpad=PANEL_3D_LABELPADS["x"], fontsize=PANEL_3D_AXIS_LABELSIZE)
    ax.set_ylabel("$y$ (m)", labelpad=PANEL_3D_LABELPADS["y"], fontsize=PANEL_3D_AXIS_LABELSIZE)
    ax.set_zlabel("$z$ (m)", labelpad=PANEL_3D_LABELPADS["z"], rotation=90, fontsize=PANEL_3D_AXIS_LABELSIZE)
    ax.zaxis.set_rotate_label(False)
    ax.tick_params(axis="x", which="major", labelsize=PANEL_3D_TICK_LABELSIZE)
    ax.tick_params(axis="y", which="major", labelsize=PANEL_3D_TICK_LABELSIZE)
    ax.tick_params(axis="z", which="major", labelsize=PANEL_3D_TICK_LABELSIZE)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        axis._axinfo["grid"]["color"] = (0.86, 0.86, 0.86, 1.0)
        axis._axinfo["grid"]["linewidth"] = 0.38
    try:
        ax.set_box_aspect(
            (
                PANEL_3D_BOUNDS["x"][1] - PANEL_3D_BOUNDS["x"][0],
                PANEL_3D_BOUNDS["y"][1] - PANEL_3D_BOUNDS["y"][0],
                PANEL_3D_BOUNDS["z"][1] - PANEL_3D_BOUNDS["z"][0],
            )
        )
    except AttributeError:
        pass
    ax.view_init(elev=PANEL_3D_VIEW["elev"], azim=PANEL_3D_VIEW["azim"])


def _draw_state_panels(axes: list[list[plt.Axes]], traces: list[TraceSpec]) -> None:
    panel_columns = [
        (
            "Body velocity",
            (("$u_b$", "u", 1.0), ("$v_b$", "v", 1.0), ("$w_b$", "w", 1.0)),
            r"m $\!$s$^{-1}$",
        ),
        (
            "Body attitude",
            (("$\\phi$", "phi", 180.0 / math.pi), ("$\\theta$", "theta", 180.0 / math.pi), ("$\\psi$", "psi", 180.0 / math.pi)),
            "deg",
        ),
        (
            "Body angular rate",
            (("$p_b$", "p", 1.0), ("$q_b$", "q", 1.0), ("$r_b$", "r", 1.0)),
            r"rad $\!$s$^{-1}$",
        ),
        (
            "Controller deflection",
            (("$\\delta_a$", "delta_a", 180.0 / math.pi), ("$\\delta_e$", "delta_e", 180.0 / math.pi), ("$\\delta_r$", "delta_r", 180.0 / math.pi)),
            "deg",
        ),
    ]
    ylimits = _dynamic_ylimits(traces, panel_columns)
    for col_index, (column_label, specs, units) in enumerate(panel_columns):
        for row_index, (state_label, state_name, scale) in enumerate(specs):
            ax = axes[row_index][col_index]
            for spec in traces:
                frame = spec.frame
                ax.plot(
                    frame["t_s"].to_numpy(float),
                    frame[state_name].to_numpy(float) * scale,
                    color=spec.color,
                    linestyle=spec.linestyle,
                    linewidth=1.18 if "replay" in spec.key else 1.42,
                    alpha=spec.alpha,
                    label=spec.label,
                )
            if row_index == 0:
                ax.set_title(column_label, fontsize=STATE_COLUMN_TITLE_SIZE, pad=13)
            ax.set_axisbelow(True)
            ax.grid(True, color=STATE_GRID_COLOR, linewidth=STATE_GRID_LINEWIDTH)
            ax.yaxis.set_major_formatter(FuncFormatter(_compact_y_tick_label))
            ax.tick_params(
                axis="both",
                which="major",
                labelsize=STATE_TICK_LABELSIZE,
                width=STATE_TICK_WIDTH,
                length=STATE_TICK_LENGTH,
            )
            for spine in ax.spines.values():
                spine.set_edgecolor("k")
                spine.set_linewidth(STATE_AXIS_EDGE_LW)
            ax.set_ylabel(
                f"{state_label} ({units})",
                fontsize=STATE_AXIS_LABELSIZE,
                labelpad=STATE_YLABEL_PAD,
            )
            ax.set_ylim(*ylimits[state_name])
            _apply_time_ticks(ax)
            if row_index == len(specs) - 1:
                ax.set_xlabel("time (s)", fontsize=STATE_AXIS_LABELSIZE)
            else:
                ax.set_xticklabels([])


def _dynamic_ylimits(
    traces: list[TraceSpec],
    panel_columns: list[tuple[str, tuple[tuple[str, str, float], ...], str]],
) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for _, specs, _ in panel_columns:
        for _, state_name, scale in specs:
            values: list[np.ndarray] = []
            for spec in traces:
                if state_name in spec.frame:
                    values.append(spec.frame[state_name].to_numpy(float) * scale)
            merged = np.concatenate(values) if values else np.asarray([0.0])
            finite = merged[np.isfinite(merged)]
            if finite.size == 0:
                finite = np.asarray([0.0])
            lo = float(np.nanmin(finite))
            hi = float(np.nanmax(finite))
            span = max(hi - lo, 1.0)
            pad = 0.12 * span
            out[state_name] = (lo - pad, hi + pad)
    return out


def _apply_time_ticks(ax: plt.Axes) -> None:
    xmin, xmax = STATE_XLIM_S if STATE_XLIM_S is not None else ax.get_xlim()
    xmax = max(float(xmax), 1.25)
    ticks = [tick for tick in STATE_XTICKS_S if xmin - 1e-9 <= float(tick) <= xmax + 1e-9]
    labels = [
        label
        for tick, label in zip(STATE_XTICKS_S, STATE_XTICK_LABELS, strict=False)
        if xmin - 1e-9 <= float(tick) <= xmax + 1e-9
    ]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(float(xmin), xmax)


def _is_exit_face(spec: TraceSpec) -> bool:
    reasons = spec.frame.get("exit_gate_reason")
    if reasons is None or reasons.empty:
        return False
    last_reason = str(reasons.iloc[-1]).lower()
    return "front" in last_reason or float(spec.frame["x_w"].iloc[-1]) >= 6.45


def main() -> int:
    _render_case(
        output_name="e4a_matched_reality_replay_openloop_composite.png",
        source_root=OUTPUT_DIR / "e4a_matched_replay_source",
        closed_case="E4a.2",
        closed_session="20260607_231704",
        closed_throw="throw_018",
        open_case="E4a.0",
        open_session="20260607_230250",
        open_throw="throw_008",
        updraft_render_mode="r11_composed_measured_fans",
    )
    _render_case(
        output_name="e3_four_fan_reality_replay_openloop_composite.png",
        source_root=OUTPUT_DIR / "real_flight_sim_replay_E3_representative",
        closed_case="E3.2",
        closed_session="20260607_213312",
        closed_throw="throw_013",
        open_case="E3.0",
        open_session="20260607_202556",
        open_throw="throw_008",
        updraft_render_mode="replay_environment",
    )
    _render_case(
        output_name="e4b_matched_reality_replay_openloop_composite.png",
        source_root=OUTPUT_DIR / "e4b_matched_replay_source",
        closed_case="E4b.2",
        closed_session="20260608_013501",
        closed_throw="throw_026",
        open_case="E4b.0",
        open_session="20260608_003841",
        open_throw="throw_007",
        updraft_render_mode="r11_composed_measured_fans",
    )
    return 0


def _render_case(
    *,
    output_name: str,
    source_root: Path,
    closed_case: str,
    closed_session: str,
    closed_throw: str,
    open_case: str,
    open_session: str,
    open_throw: str,
    updraft_render_mode: str,
) -> None:
    global OUTPUT_PATH, SOURCE_ROOT, CLOSED_LOOP_TRACE, OPEN_LOOP_TRACE
    global CLOSED_LOOP_THROW_ROOT, OPEN_LOOP_THROW_ROOT, UPDRAFT_RENDER_MODE

    OUTPUT_PATH = OUTPUT_DIR / output_name
    SOURCE_ROOT = source_root
    CLOSED_LOOP_TRACE = source_root / "metrics" / closed_case / closed_session / f"{closed_throw}_replay_traces.csv"
    OPEN_LOOP_TRACE = source_root / "metrics" / open_case / open_session / f"{open_throw}_replay_traces.csv"
    CLOSED_LOOP_THROW_ROOT = REPO_ROOT / "04_Flight_Test" / "05_Results" / closed_case / closed_session / closed_throw
    OPEN_LOOP_THROW_ROOT = REPO_ROOT / "04_Flight_Test" / "05_Results" / open_case / open_session / open_throw
    UPDRAFT_RENDER_MODE = updraft_render_mode

    missing = [path for path in (CLOSED_LOOP_TRACE, OPEN_LOOP_TRACE) if not path.is_file()]
    if missing and OUTPUT_PATH.is_file():
        print(f"{OUTPUT_PATH.as_posix()} (kept existing; missing replay trace CSV source)")
        return
    if missing:
        missing_text = ", ".join(path.as_posix() for path in missing)
        raise FileNotFoundError(f"Missing replay trace CSV source for {output_name}: {missing_text}")
    _render_current_case()


if __name__ == "__main__":
    raise SystemExit(main())
