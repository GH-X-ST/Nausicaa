"""Shared workbook readers and robust-ranking helpers for analysis scripts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Workbook paths and robust-scenario order
# 2) Workbook readers
# 3) Ranking and label helpers
# =============================================================================

# =============================================================================
# 1) Workbook Paths and Robust-Scenario Order
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "C_results"
# `nausicaa_workflow.xlsx` is the current canonical export; the legacy workbook
# remains readable so older design runs can still be audited.
WORKFLOW_XLSX = RESULTS_DIR / "nausicaa_workflow.xlsx"
LEGACY_RESULTS_XLSX = RESULTS_DIR / "nausicaa_results.xlsx"

# Ordered from the harshest compound family to the mildest build-only family.
SCENARIO_TAG_ORDER: tuple[str, ...] = (
    "harsh_compound",
    "harsh_build",
    "gusty_only",
    "mild_compound",
    "mild_build",
)


# =============================================================================
# 2) Workbook Readers
# =============================================================================

def open_canonical_workbook() -> tuple[Path, pd.ExcelFile]:
    for path in (WORKFLOW_XLSX, LEGACY_RESULTS_XLSX):
        if path.exists():
            return path, pd.ExcelFile(path)
    raise FileNotFoundError(
        "No canonical workbook found. Expected "
        f"{WORKFLOW_XLSX} or {LEGACY_RESULTS_XLSX}."
    )


def read_sheet_optional(book: pd.ExcelFile, sheet_name: str) -> pd.DataFrame:
    if sheet_name in book.sheet_names:
        return pd.read_excel(book, sheet_name=sheet_name)
    return pd.DataFrame()


def read_sheet_required(book: pd.ExcelFile, sheet_name: str) -> pd.DataFrame:
    sheet_df = read_sheet_optional(book, sheet_name)
    if sheet_df.empty and sheet_name not in book.sheet_names:
        raise FileNotFoundError(
            f"Required sheet '{sheet_name}' was not found in {book.io}."
        )
    return sheet_df


# =============================================================================
# 3) Ranking and Label Helpers
# =============================================================================

def coerce_bool_series(series: pd.Series) -> pd.Series:
    # Workbook exports may store booleans as strings, numbers, or native bools.
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"1", "true", "yes"})


def resolve_tail_metric_name(df: pd.DataFrame) -> str:
    # Prefer the explicit worst-k tail metric; CVaR 20% is the legacy name.
    if "nom_sink_tail_mean_k" in df.columns:
        return "nom_sink_tail_mean_k"
    if "nom_sink_cvar_20" in df.columns:
        return "nom_sink_cvar_20"
    raise KeyError("Neither robust tail-risk metric is available.")


def resolve_selected_candidate_id(
    candidates_df: pd.DataFrame,
    robust_summary_df: pd.DataFrame,
) -> int:
    if not robust_summary_df.empty and "is_selected" in robust_summary_df.columns:
        # Manual/exported selection takes precedence over reconstructed ranking.
        selected_mask = coerce_bool_series(robust_summary_df["is_selected"])
        if selected_mask.any():
            return int(
                robust_summary_df.loc[selected_mask, "candidate_id"].iloc[0]
            )

    if not robust_summary_df.empty and "robust_rank" in robust_summary_df.columns:
        # Robust-rank columns are already computed by the optimization workflow.
        rank_series = pd.to_numeric(robust_summary_df["robust_rank"], errors="coerce")
        ranked_df = robust_summary_df.loc[rank_series.notna()].copy()
        if not ranked_df.empty:
            ranked_df["_robust_rank_sort"] = pd.to_numeric(
                ranked_df["robust_rank"],
                errors="coerce",
            )
            ranked_df = ranked_df.sort_values(
                by="_robust_rank_sort",
                ascending=True,
                kind="mergesort",
            )
            return int(ranked_df.iloc[0]["candidate_id"])

    if not candidates_df.empty and not robust_summary_df.empty:
        # Reconstruct the same ranking policy: success first, tail-risk sink second.
        tail_metric = resolve_tail_metric_name(robust_summary_df)
        merged_df = robust_summary_df.merge(
            candidates_df[["candidate_id", "objective"]],
            on="candidate_id",
            how="left",
        )
        ranked_df = merged_df.sort_values(
            by=["nom_success_rate", tail_metric, "objective"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        if not ranked_df.empty:
            return int(ranked_df.iloc[0]["candidate_id"])

    if candidates_df.empty:
        raise ValueError(
            "Candidates sheet is required to resolve the selected candidate."
        )
    ranked_df = candidates_df.sort_values("objective", kind="mergesort")
    return int(ranked_df.iloc[0]["candidate_id"])


def sort_scenario_tags(tags: list[str] | pd.Index | pd.Series) -> list[str]:
    unique_tags = list(dict.fromkeys(pd.Series(tags).dropna().astype(str)))
    # Unknown tags keep a deterministic alphabetical order after known families.
    order_map = {
        tag: index for index, tag in enumerate(SCENARIO_TAG_ORDER)
    }
    return sorted(
        unique_tags,
        key=lambda tag: (order_map.get(tag, len(SCENARIO_TAG_ORDER)), tag),
    )
