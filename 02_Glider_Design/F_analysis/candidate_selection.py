from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "C_results"
FIGURES_DIR = PROJECT_ROOT / "B_figures"
WORKFLOW_XLSX = RESULTS_DIR / "nausicaa_workflow.xlsx"
RESULTS_XLSX = RESULTS_DIR / "nausicaa_results.xlsx"
WEIGHT_SWEEP_XLSX = RESULTS_DIR / "weight_sweep.xlsx"
TOP_RERUN_XLSX = RESULTS_DIR / "weight_sweep_top_rerun.xlsx"
FIGURE_PATH = FIGURES_DIR / "candidate_selection.png"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STAGE_ORDER = [
    "Weight sweep",
    "Selected sweep rows",
    "Top rerun starts",
    "Final rerun starts",
    "Final kept",
    "Robust rank",
]
STAGE_JITTER = {
    "Weight sweep": 0.08,
    "Selected sweep rows": 0.07,
    "Top rerun starts": 0.07,
    "Final rerun starts": 0.06,
    "Final kept": 0.05,
    "Robust rank": 0.04,
}
EDGE_STYLE_MAP = {
    "background_terminated": {
        "color": "#9e9e9e",
        "linewidth": 0.75,
        "alpha": 0.12,
        "zorder": 1,
    },
    "promoted": {
        "color": "#7f9fc5",
        "linewidth": 0.85,
        "alpha": 0.22,
        "zorder": 2,
    },
    "final_rerun": {
        "color": "#4e79a7",
        "linewidth": 1.05,
        "alpha": 0.36,
        "zorder": 3,
    },
    "kept": {
        "color": "#4e79a7",
        "linewidth": 1.15,
        "alpha": 0.52,
        "zorder": 4,
    },
    "selected": {
        "color": "#f28e2b",
        "linewidth": 2.45,
        "alpha": 0.98,
        "zorder": 7,
    },
}


@dataclass(frozen=True)
class CandidateSelectionData:
    weight_sweep_df: pd.DataFrame
    selected_sweep_rows_df: pd.DataFrame
    top_rerun_summary_df: pd.DataFrame
    top_rerun_all_starts_df: pd.DataFrame
    all_starts_df: pd.DataFrame
    candidates_df: pd.DataFrame
    robust_summary_df: pd.DataFrame


def _open_workbook(path: Path) -> pd.ExcelFile | None:
    if path.exists():
        return pd.ExcelFile(path)
    return None


def _read_sheet(
    sheet_name: str,
    workflow_book: pd.ExcelFile | None,
    results_book: pd.ExcelFile | None,
) -> pd.DataFrame | None:
    if workflow_book is not None and sheet_name in workflow_book.sheet_names:
        return pd.read_excel(workflow_book, sheet_name=sheet_name)
    if results_book is not None and sheet_name in results_book.sheet_names:
        return pd.read_excel(results_book, sheet_name=sheet_name)
    return None


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"1", "true", "yes"})


def _sort_all_starts(all_starts_df: pd.DataFrame) -> pd.DataFrame:
    return all_starts_df.sort_values(by=["start_index"], kind="mergesort").reset_index(
        drop=True
    )


def _sort_feasible(feasible_df: pd.DataFrame) -> pd.DataFrame:
    return feasible_df.sort_values(
        by=["objective", "start_index"],
        kind="mergesort",
    ).reset_index(drop=True)


def _sort_kept(kept_df: pd.DataFrame) -> pd.DataFrame:
    if "kept_rank" in kept_df.columns and kept_df["kept_rank"].notna().any():
        return kept_df.sort_values(
            by=["kept_rank", "start_index"],
            kind="mergesort",
        ).reset_index(drop=True)
    return kept_df.sort_values(
        by=["objective", "start_index"],
        kind="mergesort",
    ).reset_index(drop=True)


def _resolve_kept_candidate_map(
    kept_df: pd.DataFrame,
    ranked_df: pd.DataFrame,
) -> dict[int, int]:
    candidate_ids = set(ranked_df["candidate_id"].astype(int).tolist())
    start_ids = kept_df["start_index"].astype(int)
    start_overlap = int(start_ids.isin(candidate_ids).sum())

    if "kept_rank" in kept_df.columns and kept_df["kept_rank"].notna().any():
        kept_rank_ids = kept_df["kept_rank"].dropna().astype(int)
        kept_rank_overlap = int(kept_rank_ids.isin(candidate_ids).sum())
        if kept_rank_overlap >= start_overlap and kept_rank_overlap > 0:
            return {
                int(row["start_index"]): int(row["kept_rank"])
                for _, row in kept_df.iterrows()
                if pd.notna(row["kept_rank"])
                and int(row["kept_rank"]) in candidate_ids
            }

    return {
        int(row["start_index"]): int(row["start_index"])
        for _, row in kept_df.iterrows()
        if int(row["start_index"]) in candidate_ids
    }


def _normalise_boolean_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    normalized_df = df.copy()
    for column in columns:
        if column in normalized_df.columns:
            normalized_df[column] = _coerce_bool_series(normalized_df[column])
    return normalized_df


def load_candidate_selection_data() -> CandidateSelectionData:
    workflow_book = _open_workbook(WORKFLOW_XLSX)
    results_book = _open_workbook(RESULTS_XLSX)
    weight_sweep_book = _open_workbook(WEIGHT_SWEEP_XLSX)
    top_rerun_book = _open_workbook(TOP_RERUN_XLSX)

    if weight_sweep_book is None:
        raise FileNotFoundError(f"Required workbook not found: {WEIGHT_SWEEP_XLSX}")
    if top_rerun_book is None:
        raise FileNotFoundError(f"Required workbook not found: {TOP_RERUN_XLSX}")

    weight_sweep_df = _read_sheet("WeightSweep", weight_sweep_book, None)
    if weight_sweep_df is None:
        weight_sweep_df = _read_sheet("TradeStudy", weight_sweep_book, None)
    selected_sweep_rows_df = _read_sheet("SelectedSweepRows", top_rerun_book, None)
    top_rerun_summary_df = _read_sheet("TopRerunSummary", top_rerun_book, None)
    top_rerun_all_starts_df = _read_sheet("TopRerunAllStarts", top_rerun_book, None)
    all_starts_df = _read_sheet("AllStarts", workflow_book, results_book)
    candidates_df = _read_sheet("Candidates", workflow_book, results_book)
    robust_summary_df = _read_sheet("RobustSummary", workflow_book, results_book)

    if weight_sweep_df is None:
        raise FileNotFoundError(
            "Required sheet 'WeightSweep' (or fallback 'TradeStudy') was not found in "
            f"{WEIGHT_SWEEP_XLSX}."
        )
    if (
        selected_sweep_rows_df is None
        or top_rerun_summary_df is None
        or top_rerun_all_starts_df is None
    ):
        raise FileNotFoundError(
            "Required top-rerun sheets were not found in "
            f"{TOP_RERUN_XLSX}."
        )
    if all_starts_df is None or candidates_df is None or robust_summary_df is None:
        raise FileNotFoundError(
            "Required workflow sheets were not found. Expected "
            f"{WORKFLOW_XLSX} with 'AllStarts', 'Candidates', and 'RobustSummary'."
        )

    weight_sweep_df = _normalise_boolean_columns(
        weight_sweep_df,
        ["success", "is_best"],
    )
    selected_sweep_rows_df = _normalise_boolean_columns(
        selected_sweep_rows_df,
        ["success", "is_best"],
    )
    top_rerun_summary_df = _normalise_boolean_columns(
        top_rerun_summary_df,
        ["success", "is_best"],
    )
    top_rerun_all_starts_df = _normalise_boolean_columns(
        top_rerun_all_starts_df,
        ["success", "kept_after_dedup"],
    )
    all_starts_df = _normalise_boolean_columns(
        all_starts_df,
        ["success", "kept_after_dedup"],
    )
    robust_summary_df = _normalise_boolean_columns(
        robust_summary_df,
        ["is_selected"],
    )

    return CandidateSelectionData(
        weight_sweep_df=weight_sweep_df,
        selected_sweep_rows_df=selected_sweep_rows_df,
        top_rerun_summary_df=top_rerun_summary_df,
        top_rerun_all_starts_df=top_rerun_all_starts_df,
        all_starts_df=all_starts_df,
        candidates_df=candidates_df,
        robust_summary_df=robust_summary_df,
    )


def resolve_tail_metric_column(df: pd.DataFrame) -> str:
    if "nom_sink_tail_mean_k" in df.columns:
        return "nom_sink_tail_mean_k"
    if "nom_sink_cvar_20" in df.columns:
        return "nom_sink_cvar_20"
    raise KeyError("Neither 'nom_sink_tail_mean_k' nor 'nom_sink_cvar_20' is available.")


def build_rank_columns(df: pd.DataFrame, tail_metric_col: str) -> pd.DataFrame:
    ranked_df = df.sort_values(
        by=["nom_success_rate", tail_metric_col, "objective"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked_df["robust_rank"] = np.arange(1, len(ranked_df) + 1, dtype=int)

    if "is_selected" in ranked_df.columns and ranked_df["is_selected"].any():
        ranked_df["is_selected"] = _coerce_bool_series(ranked_df["is_selected"])
    else:
        ranked_df["is_selected"] = ranked_df["robust_rank"].eq(1)

    return ranked_df


def _stable_jitter(identifier: int, scale: float) -> float:
    value = np.sin(float(identifier) * 12.9898) * 43758.5453
    frac = value - np.floor(value)
    return scale * (frac - 0.5)


def _draw_curve(
    ax: plt.Axes,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: str,
    linewidth: float,
    alpha: float,
    zorder: int,
) -> None:
    t = np.linspace(0.0, 1.0, 40)
    c1x = x0 + 0.35 * (x1 - x0)
    c2x = x0 + 0.65 * (x1 - x0)
    c1y = y0
    c2y = y1
    x = (
        (1.0 - t) ** 3 * x0
        + 3.0 * (1.0 - t) ** 2 * t * c1x
        + 3.0 * (1.0 - t) * t**2 * c2x
        + t**3 * x1
    )
    y = (
        (1.0 - t) ** 3 * y0
        + 3.0 * (1.0 - t) ** 2 * t * c1y
        + 3.0 * (1.0 - t) * t**2 * c2y
        + t**3 * y1
    )
    ax.plot(
        x,
        y,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
    )


def infer_final_rerun_rank(
    final_all_starts_df: pd.DataFrame,
    top_rerun_all_starts_df: pd.DataFrame,
) -> int:
    compare_columns = [
        "start_index",
        "objective",
        "success",
        "kept_after_dedup",
        "kept_rank",
    ]
    required_columns = [
        column
        for column in compare_columns
        if column in final_all_starts_df.columns and column in top_rerun_all_starts_df.columns
    ]
    if len(required_columns) != len(compare_columns):
        raise ValueError(
            "Cannot infer final rerun rank because the required AllStarts signature "
            "columns are incomplete."
        )

    final_df = _sort_all_starts(final_all_starts_df[required_columns].copy())
    final_df["start_index"] = pd.to_numeric(
        final_df["start_index"],
        errors="coerce",
    ).fillna(-1).astype(int)
    final_df["objective"] = pd.to_numeric(final_df["objective"], errors="coerce")
    final_df["kept_rank"] = pd.to_numeric(final_df["kept_rank"], errors="coerce")

    matched_ranks: list[int] = []
    for rerun_rank, group_df in top_rerun_all_starts_df.groupby("rerun_rank", sort=False):
        group_compare_df = _sort_all_starts(group_df[required_columns].copy())
        group_compare_df["start_index"] = pd.to_numeric(
            group_compare_df["start_index"],
            errors="coerce",
        ).fillna(-1).astype(int)
        group_compare_df["objective"] = pd.to_numeric(
            group_compare_df["objective"],
            errors="coerce",
        )
        group_compare_df["kept_rank"] = pd.to_numeric(
            group_compare_df["kept_rank"],
            errors="coerce",
        )

        if len(group_compare_df) != len(final_df):
            continue
        if not np.array_equal(
            group_compare_df["start_index"].to_numpy(),
            final_df["start_index"].to_numpy(),
        ):
            continue
        if not np.array_equal(
            group_compare_df["success"].to_numpy(dtype=bool),
            final_df["success"].to_numpy(dtype=bool),
        ):
            continue
        if not np.array_equal(
            group_compare_df["kept_after_dedup"].to_numpy(dtype=bool),
            final_df["kept_after_dedup"].to_numpy(dtype=bool),
        ):
            continue
        if not np.allclose(
            group_compare_df["objective"].fillna(np.nan).to_numpy(dtype=float),
            final_df["objective"].fillna(np.nan).to_numpy(dtype=float),
            atol=1e-9,
            rtol=1e-9,
            equal_nan=True,
        ):
            continue
        if not np.allclose(
            group_compare_df["kept_rank"].fillna(-1.0).to_numpy(dtype=float),
            final_df["kept_rank"].fillna(-1.0).to_numpy(dtype=float),
            atol=1e-9,
            rtol=1e-9,
            equal_nan=True,
        ):
            continue
        matched_ranks.append(int(rerun_rank))

    if len(matched_ranks) != 1:
        raise ValueError(
            "Could not infer a unique final rerun rank from TopRerunAllStarts "
            f"and final AllStarts. Matches found: {matched_ranks}"
        )

    return matched_ranks[0]


def _resolve_final_kept_candidate_map(
    final_kept_df: pd.DataFrame,
    ranked_df: pd.DataFrame,
) -> dict[int, int]:
    candidate_ids = set(ranked_df["candidate_id"].astype(int).tolist())
    kept_rank_ids = (
        final_kept_df["kept_rank"].dropna().astype(int)
        if "kept_rank" in final_kept_df.columns
        else pd.Series(dtype=int)
    )
    start_ids = final_kept_df["start_index"].astype(int)
    kept_rank_overlap = int(kept_rank_ids.isin(candidate_ids).sum())
    start_overlap = int(start_ids.isin(candidate_ids).sum())

    if kept_rank_overlap >= start_overlap and kept_rank_overlap > 0:
        return {
            int(row["kept_rank"]): int(row["kept_rank"])
            for _, row in final_kept_df.iterrows()
            if pd.notna(row["kept_rank"])
            and int(row["kept_rank"]) in candidate_ids
        }

    return {
        int(row["kept_rank"]): int(row["start_index"])
        for _, row in final_kept_df.iterrows()
        if pd.notna(row["kept_rank"])
        and int(row["start_index"]) in candidate_ids
    }


def build_full_provenance(
    data: CandidateSelectionData,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, int]:
    retained_df = data.candidates_df.merge(
        data.robust_summary_df,
        on="candidate_id",
        how="left",
        suffixes=("", "_robust"),
    )
    tail_metric_col = resolve_tail_metric_column(retained_df)
    ranked_df = build_rank_columns(retained_df, tail_metric_col)
    ranked_df = ranked_df.sort_values(by="robust_rank", kind="mergesort").reset_index(
        drop=True
    )
    selected_candidate_id = int(
        ranked_df.loc[ranked_df["is_selected"], "candidate_id"].iloc[0]
    )

    weight_sweep_df = data.weight_sweep_df.sort_values(
        by=["run_index"],
        kind="mergesort",
    ).reset_index(drop=True)
    selected_sweep_rows_df = data.selected_sweep_rows_df.sort_values(
        by=["run_index"],
        kind="mergesort",
    ).reset_index(drop=True)
    top_rerun_summary_df = data.top_rerun_summary_df.sort_values(
        by=["rerun_rank"],
        kind="mergesort",
    ).reset_index(drop=True)
    top_rerun_all_starts_df = data.top_rerun_all_starts_df.sort_values(
        by=["rerun_rank", "start_index"],
        kind="mergesort",
    ).reset_index(drop=True)
    final_all_starts_df = _sort_all_starts(data.all_starts_df)
    final_rerun_rank = infer_final_rerun_rank(
        final_all_starts_df,
        top_rerun_all_starts_df,
    )
    final_rerun_source_run_index = int(
        top_rerun_summary_df.loc[
            top_rerun_summary_df["rerun_rank"] == final_rerun_rank,
            "source_sweep_run_index",
        ].iloc[0]
    )

    final_kept_df = _sort_kept(
        final_all_starts_df.loc[final_all_starts_df["kept_after_dedup"]].copy()
    )
    start_to_candidate_map = _resolve_kept_candidate_map(final_kept_df, ranked_df)
    kept_to_candidate_map = _resolve_final_kept_candidate_map(final_kept_df, ranked_df)
    selected_start_id = next(
        start_id
        for start_id, candidate_id in start_to_candidate_map.items()
        if candidate_id == selected_candidate_id
    )
    selected_kept_rank = next(
        kept_rank
        for kept_rank, candidate_id in kept_to_candidate_map.items()
        if candidate_id == selected_candidate_id
    )

    selected_sweep_run_indices = set(
        selected_sweep_rows_df["run_index"].astype(int).tolist()
    )

    node_rows: list[dict[str, object]] = []
    for _, row in weight_sweep_df.iterrows():
        run_index = int(row["run_index"])
        node_rows.append(
            {
                "stage": "Weight sweep",
                "node_key": str(run_index),
                "display_label": str(run_index),
                "order_value": float(run_index),
                "group_key": "weight_sweep",
                "status_class": (
                    "final_rerun"
                    if run_index == final_rerun_source_run_index
                    else "promoted"
                    if run_index in selected_sweep_run_indices
                    else "background_terminated"
                ),
            }
        )

    for _, row in selected_sweep_rows_df.iterrows():
        run_index = int(row["run_index"])
        node_rows.append(
            {
                "stage": "Selected sweep rows",
                "node_key": str(run_index),
                "display_label": str(run_index),
                "order_value": float(run_index),
                "group_key": "selected_sweep_rows",
                "status_class": (
                    "final_rerun"
                    if run_index == final_rerun_source_run_index
                    else "promoted"
                ),
            }
        )

    for _, row in top_rerun_all_starts_df.iterrows():
        rerun_rank = int(row["rerun_rank"])
        start_index = int(row["start_index"])
        node_rows.append(
            {
                "stage": "Top rerun starts",
                "node_key": f"{rerun_rank}:{start_index}",
                "display_label": f"{rerun_rank}:{start_index}",
                "order_value": float(1000 * rerun_rank + start_index),
                "group_key": str(rerun_rank),
                "status_class": (
                    "final_rerun"
                    if rerun_rank == final_rerun_rank
                    else "promoted"
                ),
            }
        )

    for _, row in final_all_starts_df.iterrows():
        start_index = int(row["start_index"])
        node_rows.append(
            {
                "stage": "Final rerun starts",
                "node_key": str(start_index),
                "display_label": str(start_index),
                "order_value": float(start_index),
                "group_key": "final_rerun",
                "status_class": (
                    "kept" if bool(row["kept_after_dedup"]) else "background_terminated"
                ),
            }
        )

    for _, row in final_kept_df.iterrows():
        kept_rank = int(row["kept_rank"])
        node_rows.append(
            {
                "stage": "Final kept",
                "node_key": str(kept_rank),
                "display_label": str(kept_rank),
                "order_value": float(kept_rank),
                "group_key": "final_kept",
                "status_class": (
                    "selected" if kept_rank == selected_kept_rank else "kept"
                ),
            }
        )

    for _, row in ranked_df.iterrows():
        candidate_id = int(row["candidate_id"])
        node_rows.append(
            {
                "stage": "Robust rank",
                "node_key": str(candidate_id),
                "display_label": f"#{int(row['robust_rank'])}",
                "order_value": float(row["robust_rank"]),
                "group_key": "robust_rank",
                "status_class": (
                    "selected" if candidate_id == selected_candidate_id else "kept"
                ),
                "nom_success_rate": float(row["nom_success_rate"]),
            }
        )

    edge_rows: list[dict[str, object]] = []
    for _, row in selected_sweep_rows_df.iterrows():
        run_index = int(row["run_index"])
        edge_rows.append(
            {
                "source_stage": "Weight sweep",
                "source_key": str(run_index),
                "target_stage": "Selected sweep rows",
                "target_key": str(run_index),
                "edge_class": "promoted",
                "trace_group": f"sweep:{run_index}",
                "is_selected_path": run_index == final_rerun_source_run_index,
            }
        )

    for _, row in top_rerun_all_starts_df.iterrows():
        rerun_rank = int(row["rerun_rank"])
        source_run_index = int(row["source_sweep_run_index"])
        start_index = int(row["start_index"])
        edge_rows.append(
            {
                "source_stage": "Selected sweep rows",
                "source_key": str(source_run_index),
                "target_stage": "Top rerun starts",
                "target_key": f"{rerun_rank}:{start_index}",
                "edge_class": "promoted",
                "trace_group": f"rerun:{rerun_rank}",
                "is_selected_path": (
                    rerun_rank == final_rerun_rank and start_index == selected_start_id
                ),
            }
        )

    final_top_rerun_starts_df = _sort_all_starts(
        top_rerun_all_starts_df.loc[
            top_rerun_all_starts_df["rerun_rank"] == final_rerun_rank
        ].copy()
    )
    for _, row in final_top_rerun_starts_df.iterrows():
        start_index = int(row["start_index"])
        edge_rows.append(
            {
                "source_stage": "Top rerun starts",
                "source_key": f"{final_rerun_rank}:{start_index}",
                "target_stage": "Final rerun starts",
                "target_key": str(start_index),
                "edge_class": "final_rerun",
                "trace_group": f"final_rerun:{start_index}",
                "is_selected_path": start_index == selected_start_id,
            }
        )

    for _, row in final_kept_df.iterrows():
        start_index = int(row["start_index"])
        kept_rank = int(row["kept_rank"])
        edge_rows.append(
            {
                "source_stage": "Final rerun starts",
                "source_key": str(start_index),
                "target_stage": "Final kept",
                "target_key": str(kept_rank),
                "edge_class": "kept",
                "trace_group": f"kept:{kept_rank}",
                "is_selected_path": start_index == selected_start_id,
            }
        )

    for kept_rank, candidate_id in kept_to_candidate_map.items():
        edge_rows.append(
            {
                "source_stage": "Final kept",
                "source_key": str(int(kept_rank)),
                "target_stage": "Robust rank",
                "target_key": str(int(candidate_id)),
                "edge_class": "kept",
                "trace_group": f"candidate:{candidate_id}",
                "is_selected_path": int(candidate_id) == selected_candidate_id,
            }
        )

    nodes_df = pd.DataFrame(node_rows)
    edges_df = pd.DataFrame(edge_rows)
    return nodes_df, edges_df, ranked_df, tail_metric_col, selected_candidate_id


def build_stage_positions(
    nodes_df: pd.DataFrame,
) -> tuple[dict[str, float], dict[tuple[str, str], float]]:
    stage_x_map = {stage: float(index) for index, stage in enumerate(STAGE_ORDER)}
    position_map: dict[tuple[str, str], float] = {}

    for stage in STAGE_ORDER:
        stage_df = nodes_df.loc[nodes_df["stage"] == stage].copy()
        stage_df = stage_df.sort_values(
            by=["order_value", "node_key"],
            kind="mergesort",
        ).reset_index(drop=True)
        stage_count = len(stage_df)
        jitter_scale = STAGE_JITTER.get(stage, 0.05)
        for index, row in stage_df.iterrows():
            base_y = float(stage_count - index - 1)
            identifier_str = f"{stage}|{row['node_key']}"
            identifier = sum(
                (char_index + 1) * ord(character)
                for char_index, character in enumerate(identifier_str)
            )
            position_map[(stage, str(row["node_key"]))] = base_y + _stable_jitter(
                identifier,
                jitter_scale,
            )

    return stage_x_map, position_map


def draw_provenance_panel(
    ax: plt.Axes,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    ranked_df: pd.DataFrame,
    stage_x_map: dict[str, float],
    position_map: dict[tuple[str, str], float],
    selected_candidate_id: int,
) -> None:
    max_stage_size = max(nodes_df.groupby("stage", sort=False).size().tolist() + [1])
    y_top = float(max_stage_size) + 0.7

    for stage in STAGE_ORDER:
        x_center = stage_x_map[stage]
        ax.axvspan(
            x_center - 0.10,
            x_center + 0.10,
            facecolor="#d9d9d9",
            alpha=0.08,
            zorder=0,
        )
        ax.axvline(
            x_center,
            color="#c7c7c7",
            linewidth=0.6,
            alpha=0.45,
            zorder=0,
        )

    edge_class_order = {
        "background_terminated": 0,
        "promoted": 1,
        "final_rerun": 2,
        "kept": 3,
        "selected": 4,
    }
    edges_to_draw = edges_df.copy()
    edges_to_draw["draw_class"] = np.where(
        edges_to_draw["is_selected_path"],
        "selected",
        edges_to_draw["edge_class"],
    )
    edges_to_draw["draw_order"] = edges_to_draw["draw_class"].map(edge_class_order)
    edges_to_draw = edges_to_draw.sort_values(
        by=["draw_order", "source_stage", "target_stage", "trace_group"],
        kind="mergesort",
    )

    for _, row in edges_to_draw.iterrows():
        draw_class = str(row["draw_class"])
        style = EDGE_STYLE_MAP[draw_class]
        source_key = (str(row["source_stage"]), str(row["source_key"]))
        target_key = (str(row["target_stage"]), str(row["target_key"]))
        _draw_curve(
            ax,
            stage_x_map[source_key[0]],
            position_map[source_key],
            stage_x_map[target_key[0]],
            position_map[target_key],
            color=str(style["color"]),
            linewidth=float(style["linewidth"]),
            alpha=float(style["alpha"]),
            zorder=int(style["zorder"]),
        )

    rank_success_map = {
        str(int(row["candidate_id"])): float(row["nom_success_rate"])
        for _, row in ranked_df.iterrows()
    }
    selected_trace_key = str(selected_candidate_id)

    for stage in STAGE_ORDER:
        stage_df = nodes_df.loc[nodes_df["stage"] == stage].copy()
        stage_df = stage_df.sort_values(
            by=["order_value", "node_key"],
            kind="mergesort",
        ).reset_index(drop=True)
        x_values = np.full(len(stage_df), stage_x_map[stage])
        y_values = np.asarray(
            [position_map[(stage, str(node_key))] for node_key in stage_df["node_key"]],
            dtype=float,
        )

        if stage == "Weight sweep":
            promoted_mask = stage_df["status_class"].isin(["promoted", "final_rerun"])
            background_mask = ~promoted_mask
            if background_mask.any():
                ax.scatter(
                    x_values[background_mask.to_numpy()],
                    y_values[background_mask.to_numpy()],
                    s=16,
                    facecolors="#bfbfbf",
                    edgecolors="none",
                    alpha=0.75,
                    zorder=4,
                )
            if promoted_mask.any():
                ax.scatter(
                    x_values[promoted_mask.to_numpy()],
                    y_values[promoted_mask.to_numpy()],
                    s=22,
                    facecolors="#7f9fc5",
                    edgecolors="white",
                    linewidths=0.4,
                    alpha=0.9,
                    zorder=5,
                )
        elif stage == "Selected sweep rows":
            final_mask = stage_df["status_class"].eq("final_rerun")
            other_mask = ~final_mask
            if other_mask.any():
                ax.scatter(
                    x_values[other_mask.to_numpy()],
                    y_values[other_mask.to_numpy()],
                    s=26,
                    facecolors="#7f9fc5",
                    edgecolors="white",
                    linewidths=0.5,
                    alpha=0.92,
                    zorder=5,
                )
            if final_mask.any():
                ax.scatter(
                    x_values[final_mask.to_numpy()],
                    y_values[final_mask.to_numpy()],
                    s=34,
                    facecolors="#4e79a7",
                    edgecolors="black",
                    linewidths=0.6,
                    alpha=0.96,
                    zorder=6,
                )
        elif stage == "Top rerun starts":
            final_mask = stage_df["status_class"].eq("final_rerun")
            other_mask = ~final_mask
            if other_mask.any():
                ax.scatter(
                    x_values[other_mask.to_numpy()],
                    y_values[other_mask.to_numpy()],
                    s=18,
                    facecolors="#c9d7e6",
                    edgecolors="white",
                    linewidths=0.4,
                    alpha=0.85,
                    zorder=4,
                )
            if final_mask.any():
                ax.scatter(
                    x_values[final_mask.to_numpy()],
                    y_values[final_mask.to_numpy()],
                    s=24,
                    facecolors="#4e79a7",
                    edgecolors="white",
                    linewidths=0.4,
                    alpha=0.92,
                    zorder=5,
                )
        elif stage == "Final rerun starts":
            kept_mask = stage_df["status_class"].eq("kept")
            background_mask = ~kept_mask
            if background_mask.any():
                ax.scatter(
                    x_values[background_mask.to_numpy()],
                    y_values[background_mask.to_numpy()],
                    s=22,
                    facecolors="white",
                    edgecolors="#9e9e9e",
                    linewidths=0.8,
                    alpha=0.9,
                    zorder=4,
                )
            if kept_mask.any():
                ax.scatter(
                    x_values[kept_mask.to_numpy()],
                    y_values[kept_mask.to_numpy()],
                    s=26,
                    facecolors="#4e79a7",
                    edgecolors="white",
                    linewidths=0.4,
                    alpha=0.94,
                    zorder=5,
                )
        elif stage == "Final kept":
            selected_mask = stage_df["status_class"].eq("selected")
            kept_mask = ~selected_mask
            if kept_mask.any():
                ax.scatter(
                    x_values[kept_mask.to_numpy()],
                    y_values[kept_mask.to_numpy()],
                    s=30,
                    facecolors="#4e79a7",
                    edgecolors="white",
                    linewidths=0.5,
                    alpha=0.96,
                    zorder=5,
                )
            if selected_mask.any():
                ax.scatter(
                    x_values[selected_mask.to_numpy()],
                    y_values[selected_mask.to_numpy()],
                    s=46,
                    facecolors="#f28e2b",
                    edgecolors="black",
                    linewidths=0.8,
                    alpha=0.98,
                    zorder=7,
                )
        else:
            color_values = np.asarray(
                [rank_success_map[str(node_key)] for node_key in stage_df["node_key"]],
                dtype=float,
            )
            ax.scatter(
                x_values,
                y_values,
                c=color_values,
                cmap="viridis",
                vmin=float(ranked_df["nom_success_rate"].min()),
                vmax=float(ranked_df["nom_success_rate"].max()),
                s=40,
                edgecolors="white",
                linewidths=0.5,
                alpha=0.96,
                zorder=5,
            )
            ax.scatter(
                [stage_x_map[stage]],
                [position_map[(stage, selected_trace_key)]],
                marker="*",
                s=180,
                c="#ffcc00",
                edgecolors="black",
                linewidths=1.0,
                zorder=8,
            )

    ax.annotate(
        "selected",
        xy=(stage_x_map["Robust rank"], position_map[("Robust rank", selected_trace_key)]),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold",
    )

    top_rank_df = ranked_df.nsmallest(min(3, len(ranked_df)), "robust_rank")
    for _, row in top_rank_df.iterrows():
        candidate_id = str(int(row["candidate_id"]))
        if candidate_id == selected_trace_key:
            continue
        ax.annotate(
            f"#{int(row['robust_rank'])}",
            xy=(stage_x_map["Robust rank"], position_map[("Robust rank", candidate_id)]),
            xytext=(6, -10),
            textcoords="offset points",
            fontsize=8,
        )

    for stage in STAGE_ORDER:
        stage_count = int((nodes_df["stage"] == stage).sum())
        ax.text(
            stage_x_map[stage],
            y_top,
            f"{stage}\n$n$ = {stage_count}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title("Selection trace")
    ax.set_xlim(-0.35, float(len(STAGE_ORDER) - 1) + 0.35)
    ax.set_ylim(-1.0, y_top + 1.2)
    ax.set_xticks([stage_x_map[stage] for stage in STAGE_ORDER], labels=STAGE_ORDER)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(False)


def draw_tradeoff_panel(
    ax: plt.Axes,
    ranked_df: pd.DataFrame,
    tail_metric_col: str,
) -> None:
    trade_scatter = ax.scatter(
        ranked_df["objective"],
        ranked_df[tail_metric_col],
        c=ranked_df["nom_success_rate"],
        cmap="viridis",
        s=np.full(len(ranked_df), 100.0),
        edgecolors="white",
        linewidths=0.8,
        alpha=0.95,
        zorder=2,
    )

    if (
        "mass_total_kg" in ranked_df.columns
        and ranked_df["mass_total_kg"].notna().any()
    ):
        mass = ranked_df["mass_total_kg"].to_numpy(dtype=float)
        span = np.nanmax(mass) - np.nanmin(mass)
        sizes = 70.0 + 110.0 * (mass - np.nanmin(mass)) / max(span, 1e-9)
        trade_scatter.set_sizes(sizes)

    selected_row = ranked_df.loc[ranked_df["is_selected"]].iloc[0]
    selected_x = float(selected_row["objective"])
    selected_y = float(selected_row[tail_metric_col])
    ax.scatter(
        [selected_x],
        [selected_y],
        marker="*",
        s=260,
        c="#ffcc00",
        edgecolors="black",
        linewidths=1.1,
        zorder=4,
    )
    ax.annotate(
        "selected",
        xy=(selected_x, selected_y),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
    )

    top_rank_df = ranked_df.nsmallest(min(3, len(ranked_df)), "robust_rank")
    for _, row in top_rank_df.iterrows():
        if bool(row["is_selected"]):
            continue
        ax.annotate(
            f"#{int(row['robust_rank'])}",
            xy=(row["objective"], row[tail_metric_col]),
            xytext=(6, -12),
            textcoords="offset points",
            fontsize=9,
        )

    colorbar = plt.gcf().colorbar(trade_scatter, ax=ax)
    colorbar.set_label("Nominal robust success rate")

    metric_label = (
        "Tail-risk sink mean (worst 20%) [m/s]"
        if tail_metric_col == "nom_sink_tail_mean_k"
        else "Sink CVaR20 [m/s]"
    )
    ax.set_title("Robust trade-off")
    ax.set_xlabel("Nominal objective")
    ax.set_ylabel(metric_label)
    ax.grid(True, alpha=0.2)

    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    ax.plot(
        [x_limits[0], selected_x],
        [selected_y, selected_y],
        color="#777777",
        linewidth=0.8,
        linestyle="--",
        alpha=0.35,
        zorder=1,
    )
    ax.plot(
        [selected_x, selected_x],
        [y_limits[0], selected_y],
        color="#777777",
        linewidth=0.8,
        linestyle="--",
        alpha=0.35,
        zorder=1,
    )
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)

    ax.text(
        0.03,
        0.97,
        "\n".join(
            [
                "ranking: success desc",
                "tail risk asc",
                "objective asc",
                f"retained candidates: {len(ranked_df)}",
            ]
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
    )


def make_candidate_selection_plot(data: CandidateSelectionData) -> Path:
    nodes_df, edges_df, ranked_df, tail_metric_col, selected_candidate_id = (
        build_full_provenance(data)
    )
    stage_x_map, position_map = build_stage_positions(nodes_df)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 5.8),
        gridspec_kw={"width_ratios": [1.0, 1.25]},
    )
    draw_provenance_panel(
        ax=axes[0],
        nodes_df=nodes_df,
        edges_df=edges_df,
        ranked_df=ranked_df,
        stage_x_map=stage_x_map,
        position_map=position_map,
        selected_candidate_id=selected_candidate_id,
    )
    draw_tradeoff_panel(
        ax=axes[1],
        ranked_df=ranked_df,
        tail_metric_col=tail_metric_col,
    )

    fig.savefig(FIGURE_PATH, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return FIGURE_PATH


def main() -> None:
    data = load_candidate_selection_data()
    figure_path = make_candidate_selection_plot(data)
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
