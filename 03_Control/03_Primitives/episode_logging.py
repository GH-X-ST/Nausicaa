from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from episode_schema import (
    assert_claim_safe_text,
    validate_episode_tables,
)


@dataclass(frozen=True)
class EpisodeLogOutputs:
    root: Path
    episode_summary_csv: Path
    episode_primitive_steps_csv: Path
    episode_governor_queries_csv: Path
    episode_belief_update_csv: Path
    episode_manifest_json: Path
    episode_report_md: Path

    def as_dict(self) -> dict[str, Path]:
        return {
            "root": self.root,
            "episode_summary_csv": self.episode_summary_csv,
            "episode_primitive_steps_csv": self.episode_primitive_steps_csv,
            "episode_governor_queries_csv": self.episode_governor_queries_csv,
            "episode_belief_update_csv": self.episode_belief_update_csv,
            "episode_manifest_json": self.episode_manifest_json,
            "episode_report_md": self.episode_report_md,
        }


def episode_log_paths(root: Path) -> EpisodeLogOutputs:
    output_root = Path(root)
    return EpisodeLogOutputs(
        root=output_root,
        episode_summary_csv=output_root / "episode_summary.csv",
        episode_primitive_steps_csv=output_root / "episode_primitive_steps.csv",
        episode_governor_queries_csv=output_root / "episode_governor_queries.csv",
        episode_belief_update_csv=output_root / "episode_belief_update.csv",
        episode_manifest_json=output_root / "episode_manifest.json",
        episode_report_md=output_root / "episode_report.md",
    )


def write_episode_log(
    root: Path,
    *,
    episode_summary: pd.DataFrame,
    primitive_steps: pd.DataFrame,
    governor_queries: pd.DataFrame,
    belief_updates: pd.DataFrame,
    manifest_extra: dict[str, object] | None = None,
    overwrite: bool = False,
) -> EpisodeLogOutputs:
    """Write one fixed-gate episode log bundle.

    The no-overwrite guard is intentionally at the bundle root. Episode logs are
    evidence, not scratch files, so accidental reuse of a directory is blocked.
    """

    outputs = episode_log_paths(root)
    if outputs.root.exists() and any(outputs.root.iterdir()) and not overwrite:
        raise RuntimeError(f"episode log root is non-empty: {outputs.root}")
    outputs.root.mkdir(parents=True, exist_ok=True)

    validations = validate_episode_tables(
        episode_summary,
        primitive_steps,
        governor_queries,
        belief_updates,
    )
    episode_summary.to_csv(outputs.episode_summary_csv, index=False)
    primitive_steps.to_csv(outputs.episode_primitive_steps_csv, index=False)
    governor_queries.to_csv(outputs.episode_governor_queries_csv, index=False)
    belief_updates.to_csv(outputs.episode_belief_update_csv, index=False)

    manifest = _manifest_payload(validations, manifest_extra or {})
    outputs.episode_manifest_json.write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n",
        encoding="ascii",
    )
    report_text = _report_text(episode_summary, manifest)
    assert_claim_safe_text(report_text)
    outputs.episode_report_md.write_text(report_text, encoding="ascii")
    return outputs


def _manifest_payload(validations: tuple[object, ...], extra: dict[str, object]) -> dict[str, object]:
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "schema_version": "fixed_gate_episode_schema_v1",
        "claim_boundary": (
            "fixed-gate episode log; no real-flight transfer, mission success, "
            "same-flight recapture, perching, all-arena validity, or hardware-ready "
            "agile claim is made without direct evidence"
        ),
        "table_row_counts": {
            item.table_name: int(item.row_count)
            for item in validations
        },
        "claim_status": "simulation_only",
    }
    payload.update(extra)
    return payload


def _report_text(episode_summary: pd.DataFrame, manifest: dict[str, object]) -> str:
    claim_counts = (
        episode_summary["claim_status"].value_counts(dropna=False).to_dict()
        if not episode_summary.empty and "claim_status" in episode_summary
        else {}
    )
    termination_counts = (
        episode_summary["termination_cause"].value_counts(dropna=False).to_dict()
        if not episode_summary.empty and "termination_cause" in episode_summary
        else {}
    )
    return "\n".join(
        [
            "# Fixed-Gate Episode Report",
            "",
            "This log records fixed-gate repeated-launch episode evidence.",
            "It does not claim real-flight transfer, mission success, same-flight recapture, perching, all-arena validity, or hardware-ready agile turns without direct evidence.",
            "",
            "## Summary",
            "",
            f"- Schema version: `{manifest['schema_version']}`",
            f"- Episode rows: `{manifest['table_row_counts'].get('episode_summary', 0)}`",
            f"- Claim-status counts: `{claim_counts}`",
            f"- Termination-cause counts: `{termination_counts}`",
            "",
        ]
    )
