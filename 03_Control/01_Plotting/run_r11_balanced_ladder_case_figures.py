"""Generate same-start R11 balanced-cluster ladder trajectory figures."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from run_r9_preflight_3d_figures import (  # noqa: E402
    FINAL_POLICY_COLORS,
    FINAL_POLICY_LABELS,
    FINAL_POLICY_LINESTYLES,
    FINAL_POLICY_MARKER_SIZES,
    FINAL_POLICY_ORDER,
    PRIMITIVE_MARKER_SIZE_FINAL,
    _add_legend,
    _add_updraft_colorbar,
    _build_alpha_cmap,
    _draw_center_slices,
    _draw_environment_fan_outlets,
    _draw_primitive_markers,
    _draw_updraft_isosurfaces,
    _episode_points,
    _new_baseline_axis,
    _policy_safe_success,
    _sample_updraft_volume,
    _save_figure,
)


def _bootstrap_import_paths() -> None:
    control_root = Path(__file__).resolve().parents[1]
    for rel in ("03_Primitives", "04_Scenarios"):
        path = control_root / rel
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


_bootstrap_import_paths()

from env_instance import environment_instance_for_mode, environment_metadata_from_instance  # noqa: E402
from env_surrogate import resolve_surrogate_binding, wind_field_for_binding  # noqa: E402
from run_changed_case_validation import R11_PROTOCOL  # noqa: E402
from run_repeated_launch_learning_curve import (  # noqa: E402
    R11_FIDELITY_LADDER_BLOCK_IDS,
    _environment_randomisation_config_for_context,
    _scheduled_active_fan_count_for_context,
)


FIGURE_RUN_VERSION = "r11_balanced_same_start_ladder_case_paths_v1"
DEFAULT_R11_ROOT = Path("03_Control/05_Results/R11_validation/E01")
DEFAULT_OUTPUT_ROOT = Path("03_Control/A_figures/R11_E01_balanced_paths")
DEFAULT_NEUTRAL_ROLLOUT_PATH = Path(
    "03_Control/A_figures/R11_E01_balanced_neutral_baseline/metrics/neutral_rollout_by_case.csv"
)
DEFAULT_LIBRARY_SIZE_CASE = "balanced_cluster"
DEFAULT_POLICY_IDS = (
    "no_memory_baseline",
    "spatial_flow_belief_memory_h3",
    "spatial_flow_belief_memory_h10",
    "spatial_flow_belief_memory_h30",
)
DEFAULT_UPDRAFT_NX = 44
DEFAULT_UPDRAFT_NY = 28
DEFAULT_UPDRAFT_NZ = 24
NEUTRAL_LABEL = "open-loop neutral"
NEUTRAL_COLOR = "#111111"


@dataclass(frozen=True)
class R11BalancedLadderFigureConfig:
    r11_root: Path = DEFAULT_R11_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    neutral_rollout_path: Path | None = DEFAULT_NEUTRAL_ROLLOUT_PATH
    library_size_case_id: str = DEFAULT_LIBRARY_SIZE_CASE
    paired_start_index: int | None = None
    policy_ids: tuple[str, ...] = DEFAULT_POLICY_IDS
    updraft_nx: int = DEFAULT_UPDRAFT_NX
    updraft_ny: int = DEFAULT_UPDRAFT_NY
    updraft_nz: int = DEFAULT_UPDRAFT_NZ


def run_r11_balanced_ladder_case_figures(config: R11BalancedLadderFigureConfig) -> dict[str, object]:
    r11_root = Path(config.r11_root)
    output_root = Path(config.output_root)
    for subdir in ("figures", "metrics", "manifests", "reports"):
        (output_root / subdir).mkdir(parents=True, exist_ok=True)

    outer_schedule = pd.read_csv(r11_root / "metrics" / "outer_case_schedule.csv")
    final_score = pd.read_csv(r11_root / "metrics" / "final_launch_score.csv")
    paired_index = (
        int(config.paired_start_index)
        if config.paired_start_index is not None
        else _select_same_start_index(final_score=final_score, outer_schedule=outer_schedule, config=config)
    )
    selected_cases = _selected_outer_cases(outer_schedule, paired_index=paired_index)
    primitive_log = _read_selected_primitive_rows(
        r11_root=r11_root,
        outer_case_indices=tuple(int(v) for v in selected_cases["outer_case_index"]),
        config=config,
    )
    neutral_frame = _read_neutral_rollout_rows(
        neutral_rollout_path=config.neutral_rollout_path,
        outer_case_indices=tuple(int(v) for v in selected_cases["outer_case_index"]),
    )
    summary = final_score[
        (final_score["library_size_case_id"].astype(str) == str(config.library_size_case_id))
        & (final_score["outer_case_index"].isin(selected_cases["outer_case_index"]))
        & (final_score["policy_id"].astype(str).isin(config.policy_ids))
    ].copy()

    figure_rows: list[dict[str, object]] = []
    for block_id in R11_FIDELITY_LADDER_BLOCK_IDS:
        rows = selected_cases[selected_cases["environment_block_id"].astype(str) == str(block_id)]
        if rows.empty:
            continue
        case_row = rows.iloc[0].to_dict()
        outer_case_index = int(case_row["outer_case_index"])
        ladder_id = _short_ladder_id(block_id)
        figure_path = output_root / "figures" / f"r11_e01_bal_s{paired_index:02d}_{ladder_id}.png"
        metadata = _plot_ladder_case(
            primitive_log=primitive_log,
            neutral_frame=neutral_frame,
            summary=summary,
            case_row=case_row,
            output_path=figure_path,
            config=config,
        )
        figure_rows.append(
            {
                "figure_run_version": FIGURE_RUN_VERSION,
                "library_size_case_id": str(config.library_size_case_id),
                "paired_start_condition_index": int(paired_index),
                "outer_case_index": int(outer_case_index),
                "environment_block_id": str(block_id),
                "figure_path": figure_path.as_posix(),
                **metadata,
            }
        )

    figure_frame = pd.DataFrame(figure_rows)
    figure_frame.to_csv(output_root / "metrics" / "r11_balanced_ladder_case_figure_summary.csv", index=False)
    manifest = {
        "figure_run_version": FIGURE_RUN_VERSION,
        "status": "complete",
        "r11_root": r11_root.as_posix(),
        "output_root": output_root.as_posix(),
        "neutral_rollout_path": "" if config.neutral_rollout_path is None else Path(config.neutral_rollout_path).as_posix(),
        "library_size_case_id": str(config.library_size_case_id),
        "paired_start_condition_index": int(paired_index),
        "policy_ids": list(config.policy_ids),
        "updraft_grid": {
            "nx": int(config.updraft_nx),
            "ny": int(config.updraft_ny),
            "nz": int(config.updraft_nz),
        },
        "environment_source": (
            "outer_case_schedule seeds plus run_repeated_launch_learning_curve "
            "_environment_randomisation_config_for_context"
        ),
        "figures": [str(row["figure_path"]) for row in figure_rows],
    }
    (output_root / "manifests" / "r11_balanced_ladder_case_figures_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="ascii",
    )
    _write_report(output_root=output_root, figure_rows=figure_rows, manifest=manifest)
    return {
        "status": "complete",
        "output_root": output_root.as_posix(),
        "paired_start_condition_index": int(paired_index),
        "figure_count": int(len(figure_rows)),
    }


def _select_same_start_index(
    *,
    final_score: pd.DataFrame,
    outer_schedule: pd.DataFrame,
    config: R11BalancedLadderFigureConfig,
) -> int:
    frame = final_score[
        (final_score["library_size_case_id"].astype(str) == str(config.library_size_case_id))
        & (final_score["policy_id"].astype(str).isin(config.policy_ids))
    ].copy()
    frame = frame.merge(
        outer_schedule[["outer_case_index", "environment_block_id", "paired_start_condition_index"]],
        on=["outer_case_index", "environment_block_id"],
        how="left",
    )
    pivot = frame.pivot_table(
        index=["paired_start_condition_index", "environment_block_id"],
        columns="policy_id",
        values="launch_score",
        aggfunc="first",
    )
    for policy_id in config.policy_ids:
        if policy_id == "no_memory_baseline" or policy_id not in pivot.columns:
            continue
        pivot[f"{policy_id}_delta_abs"] = (
            pivot[policy_id].astype(float) - pivot["no_memory_baseline"].astype(float)
        ).abs()
    delta_columns = [col for col in pivot.columns if str(col).endswith("_delta_abs")]
    if delta_columns:
        delta = pivot[delta_columns].max(axis=1).reset_index(name="max_abs_policy_delta")
    else:
        delta = pivot.reset_index()[["paired_start_condition_index", "environment_block_id"]]
        delta["max_abs_policy_delta"] = 0.0
    delta_summary = delta.groupby("paired_start_condition_index", dropna=False).agg(
        sum_abs_delta=("max_abs_policy_delta", "sum"),
        max_abs_delta=("max_abs_policy_delta", "max"),
    )
    summary = frame.groupby("paired_start_condition_index", dropna=False).agg(
        row_count=("episode_id", "count"),
        ladder_count=("environment_block_id", "nunique"),
        mean_launch_speed_m_s=("initial_launch_speed_m_s", "mean"),
        mission_success_rate=("mission_success", "mean"),
        safe_success_rate=("safe_success", "mean"),
        mean_launch_score=("launch_score", "mean"),
        no_viable_rate=("no_viable_primitive", "mean"),
    )
    summary = summary.join(delta_summary, how="left").fillna({"sum_abs_delta": 0.0, "max_abs_delta": 0.0})
    summary = summary[summary["ladder_count"] == len(R11_FIDELITY_LADDER_BLOCK_IDS)]
    if summary.empty:
        raise ValueError("no_paired_start_index_covers_all_r11_ladders")
    summary = summary.sort_values(
        [
            "mission_success_rate",
            "safe_success_rate",
            "sum_abs_delta",
            "mean_launch_score",
            "mean_launch_speed_m_s",
        ],
        ascending=[False, False, False, False, False],
    )
    return int(summary.index[0])


def _selected_outer_cases(outer_schedule: pd.DataFrame, *, paired_index: int) -> pd.DataFrame:
    frame = outer_schedule[
        outer_schedule["paired_start_condition_index"].map(lambda value: int(float(value)) == int(paired_index))
    ].copy()
    order = {block: idx for idx, block in enumerate(R11_FIDELITY_LADDER_BLOCK_IDS)}
    frame["_block_order"] = frame["environment_block_id"].astype(str).map(order)
    frame = frame.sort_values(["_block_order", "outer_case_index"])
    frame = frame.drop(columns=["_block_order"])
    missing = set(R11_FIDELITY_LADDER_BLOCK_IDS) - set(frame["environment_block_id"].astype(str))
    if missing:
        raise ValueError("paired_start_index_missing_ladders:" + ",".join(sorted(missing)))
    return frame


def _read_selected_primitive_rows(
    *,
    r11_root: Path,
    outer_case_indices: tuple[int, ...],
    config: R11BalancedLadderFigureConfig,
) -> pd.DataFrame:
    columns = [
        "library_size_case_id",
        "policy_id",
        "history_length",
        "outer_case_index",
        "outer_case_type",
        "environment_block_id",
        "episode_id",
        "launch_role",
        "primitive_step_index",
        "primitive_id",
        "initial_x_w",
        "initial_y_w",
        "initial_z_w",
        "exit_state_vector",
        "transition_exit_class",
        "termination_cause",
    ]
    selected = set(int(v) for v in outer_case_indices)
    frames: list[pd.DataFrame] = []
    for path in sorted((r11_root / "tables" / "primitive_execution_log").glob("*.csv.gz")):
        frame = pd.read_csv(path, usecols=columns)
        mask = (
            frame["library_size_case_id"].astype(str).eq(str(config.library_size_case_id))
            & frame["launch_role"].astype(str).eq("final_heldout")
            & frame["policy_id"].astype(str).isin(config.policy_ids)
            & frame["outer_case_index"].astype(int).isin(selected)
        )
        if mask.any():
            frames.append(frame.loc[mask].copy())
    if not frames:
        raise ValueError("no_selected_primitive_execution_rows_found")
    return pd.concat(frames, ignore_index=True)


def _read_neutral_rollout_rows(
    *,
    neutral_rollout_path: Path | None,
    outer_case_indices: tuple[int, ...],
) -> pd.DataFrame:
    if neutral_rollout_path is None:
        return pd.DataFrame()
    path = Path(neutral_rollout_path)
    if not path.exists():
        return pd.DataFrame()
    selected = set(int(v) for v in outer_case_indices)
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    mask = frame["outer_case_index"].astype(int).isin(selected)
    if "neutral_status" in frame.columns:
        mask &= frame["neutral_status"].astype(str).eq("complete")
    return frame.loc[mask].copy()


def _plot_ladder_case(
    *,
    primitive_log: pd.DataFrame,
    neutral_frame: pd.DataFrame,
    summary: pd.DataFrame,
    case_row: dict[str, object],
    output_path: Path,
    config: R11BalancedLadderFigureConfig,
) -> dict[str, object]:
    outer_case_index = int(case_row["outer_case_index"])
    block_id = str(case_row["environment_block_id"])
    subset = primitive_log[
        (primitive_log["outer_case_index"].astype(int) == outer_case_index)
        & (primitive_log["environment_block_id"].astype(str) == block_id)
    ].copy()
    summary_subset = summary[
        (summary["outer_case_index"].astype(int) == outer_case_index)
        & (summary["environment_block_id"].astype(str) == block_id)
    ].copy()

    fig, ax = _new_baseline_axis()
    env_meta = _draw_updraft_context_from_case(ax=ax, case_row=case_row, config=config)
    neutral_rows = neutral_frame[
        (neutral_frame.get("outer_case_index", pd.Series(dtype=int)).astype(int) == outer_case_index)
        & (neutral_frame.get("environment_block_id", pd.Series(dtype=str)).astype(str) == block_id)
    ].copy()
    neutral_plotted = False
    neutral_success = False
    if not neutral_rows.empty:
        neutral_row = neutral_rows.iloc[0].to_dict()
        neutral_path = _path_from_json(str(neutral_row.get("path_points_json", "")))
        neutral_success = _truthy(neutral_row.get("mission_success", False))
        if neutral_path.shape[0] >= 2:
            ax.plot(
                neutral_path[:, 0],
                neutral_path[:, 1],
                neutral_path[:, 2],
                color=NEUTRAL_COLOR,
                linestyle=":",
                linewidth=1.75,
                alpha=0.92,
                label=NEUTRAL_LABEL,
                zorder=23,
            )
            if neutral_success:
                ax.scatter(
                    [neutral_path[-1, 0]],
                    [neutral_path[-1, 1]],
                    [neutral_path[-1, 2]],
                    facecolors="none",
                    edgecolors=NEUTRAL_COLOR,
                    linewidths=1.35,
                    marker="o",
                    s=46,
                    depthshade=False,
                    zorder=24,
                )
            else:
                ax.scatter(
                    [neutral_path[-1, 0]],
                    [neutral_path[-1, 1]],
                    [neutral_path[-1, 2]],
                    color=NEUTRAL_COLOR,
                    marker="x",
                    s=46,
                    depthshade=False,
                    zorder=24,
                )
            neutral_plotted = True
    plotted_count = 0
    for policy_id in FINAL_POLICY_ORDER:
        if policy_id not in config.policy_ids:
            continue
        rows = subset[subset["policy_id"].astype(str) == str(policy_id)]
        if rows.empty:
            continue
        label = FINAL_POLICY_LABELS.get(policy_id, policy_id)
        color = FINAL_POLICY_COLORS.get(policy_id, "#333333")
        for episode_id in sorted(rows["episode_id"].astype(str).unique()):
            episode_rows = rows[rows["episode_id"].astype(str) == episode_id]
            points = _episode_points(episode_rows)
            if points.shape[0] < 2:
                continue
            ax.plot(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                color=color,
                linestyle=FINAL_POLICY_LINESTYLES.get(policy_id, "-"),
                linewidth=1.18,
                alpha=0.86,
                label=label,
                zorder=25,
            )
            _draw_primitive_markers(
                ax,
                episode_rows,
                color=color,
                marker_size=PRIMITIVE_MARKER_SIZE_FINAL,
                alpha=0.72,
                zorder=27,
                include_labels=(plotted_count == 0),
            )
            marker = "o" if _policy_safe_success(summary_subset, policy_id) else "x"
            marker_size = FINAL_POLICY_MARKER_SIZES.get(policy_id, 28)
            if marker == "o":
                ax.scatter(
                    [points[-1, 0]],
                    [points[-1, 1]],
                    [points[-1, 2]],
                    facecolors="none",
                    edgecolors=color,
                    linewidths=1.35,
                    marker=marker,
                    s=marker_size,
                    depthshade=False,
                    zorder=26,
                )
            else:
                ax.scatter(
                    [points[-1, 0]],
                    [points[-1, 1]],
                    [points[-1, 2]],
                    color=color,
                    marker=marker,
                    s=marker_size,
                    depthshade=False,
                    zorder=26,
                )
            plotted_count += 1
    title = _case_title(
        case_row=case_row,
        env_meta=env_meta,
        summary=summary_subset,
        neutral_plotted=neutral_plotted,
        neutral_success=neutral_success,
    )
    ax.set_title(title, fontsize=10, pad=12)
    _add_legend(ax)
    _save_figure(fig, output_path)
    return {
        "final_policy_count": int(plotted_count),
        "neutral_open_loop_plotted": bool(neutral_plotted),
        "neutral_open_loop_mission_success": bool(neutral_success),
        **env_meta,
    }


def _draw_updraft_context_from_case(
    *,
    ax,
    case_row: dict[str, object],
    config: R11BalancedLadderFigureConfig,
) -> dict[str, object]:
    scheduled_active_fan_count = _scheduled_active_fan_count_for_context(
        protocol=R11_PROTOCOL,
        scheduled=case_row,
    )
    randomisation_config = _environment_randomisation_config_for_context(
        protocol=R11_PROTOCOL,
        scheduled=case_row,
        scheduled_active_fan_count=scheduled_active_fan_count,
    )
    w_layer = str(case_row.get("W_layer", "W0"))
    environment_mode = str(case_row.get("environment_mode", "dry_air"))
    environment_seed = int(float(case_row.get("environment_seed", 0)))
    instance = environment_instance_for_mode(
        w_layer,
        environment_mode,
        environment_seed,
        randomisation_config=randomisation_config,
    )
    metadata = environment_metadata_from_instance(instance)
    binding = resolve_surrogate_binding(
        w_layer,
        metadata,
        repo_root=Path(".").resolve(),
        randomisation_seed=environment_seed,
    )
    _draw_environment_fan_outlets(ax, binding.fan_positions_m)
    active_count = int(sum(bool(value) for value in binding.active_fan_mask))
    meta = {
        "environment_instance_id": instance.environment_id,
        "environment_block_id": str(case_row.get("environment_block_id", "")),
        "environment_mode": environment_mode,
        "W_layer": w_layer,
        "fan_count": int(binding.fan_count),
        "active_fan_count": int(active_count),
        "fan_positions_m": _jsonable_tuple(binding.fan_positions_m),
        "fan_power_scales": _jsonable_tuple(binding.fan_power_scales),
        "active_fan_mask": ";".join("1" if value else "0" for value in binding.active_fan_mask),
        "updraft_width_scale": float(binding.updraft_width_scale),
        "updraft_amplitude_scale": float(binding.updraft_amplitude_scale),
        "local_uncertainty_scale": float(binding.local_uncertainty_scale),
        "fan_position_policy": str(case_row.get("fan_position_policy", "")),
        "scheduled_active_fan_count": (
            "" if scheduled_active_fan_count is None else int(scheduled_active_fan_count)
        ),
        "environment_layout_seed": int(float(case_row.get("environment_layout_seed", environment_seed))),
        "environment_active_fan_seed": int(float(case_row.get("environment_active_fan_seed", environment_seed))),
        "environment_parameter_seed": int(float(case_row.get("environment_parameter_seed", environment_seed))),
    }
    if w_layer.upper() == "W0":
        return {**meta, "updraft_context_status": "dry_air_no_updraft", "updraft_max_m_s": 0.0}
    wind = wind_field_for_binding(binding, repo_root=Path(".").resolve())
    if wind is None:
        return {**meta, "updraft_context_status": "no_ready_wind_field", "updraft_max_m_s": 0.0}
    try:
        x_vec, y_vec, z_vec, w_grid = _sample_updraft_volume(wind, config)
        w_max = float(np.nanmax(w_grid)) if w_grid.size else 0.0
        if w_max <= 1e-9:
            return {**meta, "updraft_context_status": "zero_updraft_volume", "updraft_max_m_s": 0.0}
        cmap_alpha = _build_alpha_cmap()
        norm = mpl.colors.Normalize(vmin=0.0, vmax=8.0, clip=True)
        _draw_center_slices(
            ax,
            x_vec=x_vec,
            y_vec=y_vec,
            z_vec=z_vec,
            w_grid=w_grid,
            fan_positions=binding.fan_positions_m,
            cmap_alpha=cmap_alpha,
            norm=norm,
        )
        iso_count = _draw_updraft_isosurfaces(
            ax,
            x_vec=x_vec,
            y_vec=y_vec,
            z_vec=z_vec,
            w_grid=w_grid,
            cmap_alpha=cmap_alpha,
            norm=norm,
        )
        _add_updraft_colorbar(ax.figure, cmap_alpha, norm)
        return {
            **meta,
            "updraft_context_status": "actual_case_seeded_3d_slices_and_isosurfaces",
            "updraft_max_m_s": w_max,
            "updraft_grid_nx": int(config.updraft_nx),
            "updraft_grid_ny": int(config.updraft_ny),
            "updraft_grid_nz": int(config.updraft_nz),
            "updraft_iso_surface_count": int(iso_count),
        }
    except Exception as exc:
        return {
            **meta,
            "updraft_context_status": f"blocked:{type(exc).__name__}",
            "updraft_max_m_s": 0.0,
        }


def _case_title(
    *,
    case_row: dict[str, object],
    env_meta: dict[str, object],
    summary: pd.DataFrame,
    neutral_plotted: bool = False,
    neutral_success: bool = False,
) -> str:
    label = str(case_row.get("environment_block_label", case_row.get("environment_block_id", "")))
    speed = float(pd.to_numeric(summary.get("initial_launch_speed_m_s", pd.Series([np.nan])), errors="coerce").mean())
    mission_rate = float(pd.to_numeric(summary.get("mission_success", pd.Series([np.nan])), errors="coerce").mean())
    active = env_meta.get("active_fan_count", 0)
    total = env_meta.get("fan_count", 0)
    wmax = float(env_meta.get("updraft_max_m_s", 0.0))
    neutral_text = f" | open={_bool01(neutral_success)}" if neutral_plotted else ""
    return (
        f"{label} | start {int(case_row.get('paired_start_condition_index', -1))} | "
        f"v0={speed:.2f} m/s | fans {active}/{total} | wmax={wmax:.2f} m/s | mission {mission_rate:.2f}"
        f"{neutral_text}"
    )


def _path_from_json(value: str) -> np.ndarray:
    try:
        arr = np.asarray(json.loads(value), dtype=float)
    except Exception:
        return np.empty((0, 3), dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return np.empty((0, 3), dtype=float)
    return arr


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "y"}


def _bool01(value: object) -> str:
    return "1" if _truthy(value) else "0"


def _jsonable_tuple(value: object) -> object:
    if isinstance(value, tuple):
        return [_jsonable_tuple(item) for item in value]
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _safe_file_component(value: object) -> str:
    text = str(value).strip().lower()
    safe = "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")
    return safe[:60] or "case"


def _short_ladder_id(block_id: object) -> str:
    text = str(block_id)
    for token in text.split("_"):
        if token.startswith("l") and token[1:].isdigit():
            return token
    aliases = {
        "r11_l0_dry_air_fixed": "l0",
        "r11_l1_single_fan_fixed_nominal": "l1",
        "r11_l2_four_fan_fixed_nominal": "l2",
        "r11_l3_fan_parameter_uncertainty": "l3",
        "r11_l4_local_fan_position_uncertainty": "l4",
        "r11_l5_active_fan_count_uncertainty": "l5",
        "r11_l6_environment_only_full_uncertainty": "l6",
        "r11_l7_full_domain_randomisation_arena_wide": "l7",
    }
    return aliases.get(text, _safe_file_component(text)[:8] or "ladder")


def _write_report(
    *,
    output_root: Path,
    figure_rows: list[dict[str, object]],
    manifest: dict[str, object],
) -> None:
    lines = [
        "# R11 E01 Balanced-Cluster Same-Start Ladder Figures",
        "",
        f"- Figure run version: `{FIGURE_RUN_VERSION}`",
        f"- R11 root: `{manifest['r11_root']}`",
        f"- Library tier: `{manifest['library_size_case_id']}`",
        f"- Paired start condition index: `{manifest['paired_start_condition_index']}`",
        f"- Neutral open-loop source: `{manifest.get('neutral_rollout_path', '')}`",
        f"- Figure count: `{len(figure_rows)}`",
        "",
        "The fan layout and updraft parameters are reconstructed from each R11 outer-case row using the stored layout, active-count, and parameter seeds.",
        "",
        "| Ladder | Outer case | Active fans | Updraft max (m/s) | Open-loop plotted | Open-loop target | Figure |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in figure_rows:
        lines.append(
            "| {block} | {case} | {active}/{total} | {wmax:.3f} | {open_plot} | {open_success} | `{fig}` |".format(
                block=row.get("environment_block_id", ""),
                case=int(row.get("outer_case_index", -1)),
                active=int(row.get("active_fan_count", 0)),
                total=int(row.get("fan_count", 0)),
                wmax=float(row.get("updraft_max_m_s", 0.0)),
                open_plot=_bool01(row.get("neutral_open_loop_plotted", False)),
                open_success=_bool01(row.get("neutral_open_loop_mission_success", False)),
                fig=row.get("figure_path", ""),
            )
        )
    (output_root / "reports" / "r11_balanced_ladder_case_figures_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="ascii",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r11-root", type=Path, default=DEFAULT_R11_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--neutral-rollout", type=Path, default=DEFAULT_NEUTRAL_ROLLOUT_PATH)
    parser.add_argument("--library-size-case-id", default=DEFAULT_LIBRARY_SIZE_CASE)
    parser.add_argument("--paired-start-index", type=int, default=None)
    parser.add_argument("--updraft-nx", type=int, default=DEFAULT_UPDRAFT_NX)
    parser.add_argument("--updraft-ny", type=int, default=DEFAULT_UPDRAFT_NY)
    parser.add_argument("--updraft-nz", type=int, default=DEFAULT_UPDRAFT_NZ)
    args = parser.parse_args(argv)
    result = run_r11_balanced_ladder_case_figures(
        R11BalancedLadderFigureConfig(
            r11_root=args.r11_root,
            output_root=args.output_root,
            neutral_rollout_path=args.neutral_rollout,
            library_size_case_id=args.library_size_case_id,
            paired_start_index=args.paired_start_index,
            updraft_nx=args.updraft_nx,
            updraft_ny=args.updraft_ny,
            updraft_nz=args.updraft_nz,
        )
    )
    print(result)
    return 0 if result.get("status") == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
