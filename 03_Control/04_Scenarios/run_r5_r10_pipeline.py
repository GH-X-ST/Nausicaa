from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd


def _bootstrap_import_paths() -> None:
    control_root = Path(__file__).resolve().parents[1]
    for rel in ("03_Primitives", "04_Scenarios"):
        path = control_root / rel
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


_bootstrap_import_paths()

from dense_archive_runtime import MAX_GENERATED_FILE_SIZE_MB  # noqa: E402
from dense_archive_table_io import filesystem_path  # noqa: E402
from primitive_timing_contract import (  # noqa: E402
    CONTROLLER_INPUT_SLOTS_PER_PRIMITIVE,
    CONTROLLER_INPUT_UPDATE_PERIOD_S,
    PRIMITIVE_FINITE_HORIZON_S,
    PRIMITIVE_TIMING_CONTRACT_VERSION,
)
from run_changed_case_validation import (  # noqa: E402
    ChangedCaseValidationConfig,
    DEFAULT_R11_OUTPUT_ROOT,
    HeldoutChangedCaseValidationConfig,
    R10_EXPECTED_FINAL_HELDOUT_LAUNCHES,
    R10_EXPECTED_HISTORY_LAUNCHES,
    run_changed_case_validation,
    run_heldout_changed_case_validation,
)
from run_lqr_w01_dense_chunked import (  # noqa: E402
    BALANCED_SCHEDULE_MODE,
    DEFAULT_OUTPUT_ROOT as W01_OUTPUT_ROOT,
    L6_RICH_SIDE_CANDIDATE_COUNT,
    L6_RICH_SIDE_PAIRED_TESTS_PER_CANDIDATE,
    L6_RICH_SIDE_ROW_COUNT,
    R5_EVIDENCE_BLOCKS,
    R5_LAUNCH_AWARE_DENSE_PASSED_FOR_REVIEW,
    W01_DRY_SCHEDULE_ONLY,
    W01DenseRunConfig,
    run_lqr_w01_dense_chunked,
)
from run_outcome_model_build import DEFAULT_OUTPUT_ROOT as OUTCOME_OUTPUT_ROOT  # noqa: E402
from run_outcome_model_build import OutcomeModelBuildConfig, run_outcome_model_build  # noqa: E402
from run_post_w3_library_size_study import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT as POST_W3_OUTPUT_ROOT,
    LIBRARY_SIZE_CASE_IDS,
    PostW3LibrarySizeStudyConfig,
    run_post_w3_library_size_study,
)
from run_repeated_launch_learning_curve import (  # noqa: E402
    R9_POLICY_HISTORY_CONDITIONS,
    R9_EXPECTED_FINAL_HELDOUT_LAUNCHES,
    R9_EXPECTED_HISTORY_LAUNCHES,
    RepeatedLaunchValidationConfig,
    run_repeated_launch_learning_curve,
)
from run_v411_source_audit import SourceAuditConfig, run_v411_source_audit  # noqa: E402
from run_w3_survival import DEFAULT_OUTPUT_ROOT as W3_OUTPUT_ROOT  # noqa: E402
from run_w3_survival import W3SurvivalConfig, run_w3_survival  # noqa: E402
from run_w3_survival_analysis import W3SurvivalAnalysisConfig, run_w3_survival_analysis  # noqa: E402
from prim_cat import ACTIVE_PRIMITIVE_IDS  # noqa: E402


PROJECT_TITLE_VERSION = "LQR-Stabilised Contextual Primitive v5.20"
PIPELINE_VERSION = "v55_r5_r7_r11_pipeline_r6_archived_with_repeated_docs_guard"
DEFAULT_OUTPUT_ROOT = Path("03_Control/05_Results/A_Pipeline")
DEFAULT_REPEATED_OUTPUT_ROOT = Path("03_Control/05_Results/R9_test")
DEFAULT_CHANGED_OUTPUT_ROOT = Path("03_Control/05_Results/R10_learn")
DEFAULT_HELDOUT_CHANGED_OUTPUT_ROOT = DEFAULT_R11_OUTPUT_ROOT
CONTROLLING_DOCS = (
    Path("docs/Glider_Control_Project_Plan.md"),
    Path("docs/Daily_Schedule.txt"),
    Path("docs/Skills.md"),
    Path("docs/Python Coding Instruction.txt"),
    Path("docs/Python Coding to CODEX.txt"),
    Path("docs/MATLAB Coding.txt"),
    Path("docs/housekeeping_and_naming_rules.md"),
    Path("docs/PR.txt"),
)
STAGE_ORDER = ("R5", "R7", "R8", "R10", "R11")
INTERNAL_PREFLIGHT_STAGES = ("R9",)
ARCHIVED_STAGES = ("R6",)
BLOCKED_CLAIMS = (
    "hardware_readiness",
    "real_flight_transfer",
    "mission_success",
    "full_autonomy",
    "memory_improvement",
)


@dataclass(frozen=True)
class R5R10PipelineConfig:
    output_root: Path = DEFAULT_OUTPUT_ROOT
    run_id: int | None = None
    run_label: str = ""
    stage_run_label: str = ""
    start_stage: str = "R5"
    stop_after_stage: str = ""
    resume: bool = True
    repair_incomplete: bool = True
    dry_run_schedule: bool = False
    workers: int | str = 8
    max_workers: int | None = 8
    storage_format: str = "auto"
    compression_level: int = 1
    candidate_chunk_size: int = 800
    r10_mode: str = "full"
    history_log_mode: str = "auto"
    history_debug_sample_stride: int = 10
    allow_stage_smoke: bool = False
    continue_on_stage_failure: bool = False


@dataclass(frozen=True)
class DocsAlignmentResult:
    status: str
    checkpoint: str
    controlling_hashes: str
    docs_changed_during_run: bool
    changed_reasons: tuple[str, ...]


class DocsAlignmentGuard:
    def __init__(self, run_root: Path, baseline: dict[str, dict[str, object]] | None = None) -> None:
        self.run_root = run_root
        self.audit_path = run_root / "metrics" / "docs_alignment_audit.csv"
        self.baseline = baseline
        self.audit_rows: list[dict[str, object]] = []
        self.changed_during_run = False

    def snapshot(self, checkpoint: str, stage_id: str, decision_context: str) -> DocsAlignmentResult:
        timestamp = _utc_now()
        rows = [_doc_snapshot_row(path, checkpoint, stage_id, decision_context, timestamp) for path in CONTROLLING_DOCS]
        current = {str(row["path"]): row for row in rows}
        changed_reasons: list[str] = []
        if self.baseline is None:
            unreadable = [path for path, row in current.items() if not bool(row["readable"])]
            if unreadable:
                changed_reasons.extend(f"baseline_unreadable:{path}" for path in unreadable)
                status = "docs_changed_reaudit_required"
                self.changed_during_run = True
            else:
                self.baseline = _baseline_from_rows(rows)
                status = "baseline_recorded"
        else:
            changed_reasons = _compare_docs_snapshots(self.baseline, current)
            status = "aligned" if not changed_reasons else "docs_changed_reaudit_required"
            if changed_reasons:
                self.changed_during_run = True

        controlling_hashes = _controlling_hashes_json(current)
        for row in rows:
            row["docs_alignment_status"] = status
            row["docs_changed_during_run"] = bool(self.changed_during_run)
            row["changed_reasons"] = ";".join(changed_reasons)
            row["controlling_hashes"] = controlling_hashes
        self.audit_rows.extend(rows)
        _write_csv(self.audit_path, pd.DataFrame(self.audit_rows))
        return DocsAlignmentResult(
            status=status,
            checkpoint=checkpoint,
            controlling_hashes=controlling_hashes,
            docs_changed_during_run=bool(self.changed_during_run),
            changed_reasons=tuple(changed_reasons),
        )


_ACTIVE_DOCS_ALIGNMENT_GUARD: DocsAlignmentGuard | None = None


def read_docs_alignment_snapshot(checkpoint: str, stage_id: str, decision_context: str) -> DocsAlignmentResult:
    """Read, hash, audit, and compare all controlling documents for one checkpoint."""

    if _ACTIVE_DOCS_ALIGNMENT_GUARD is None:
        raise RuntimeError("docs alignment guard is not active")
    return _ACTIVE_DOCS_ALIGNMENT_GUARD.snapshot(checkpoint, stage_id, decision_context)


def run_r5_r10_pipeline(config: R5R10PipelineConfig) -> dict[str, object]:
    run_id = int(config.run_id) if config.run_id is not None else _next_run_id(config.output_root)
    run_root = Path(config.output_root) / _run_folder_name(run_id, config.run_label)
    conflicts = _official_run_label_conflicts(config=config, run_id=run_id, run_root=run_root)
    if conflicts and not config.resume:
        return {
            "status": "blocked",
            "run_root": run_root.as_posix(),
            "blocked_reason": "official_run_label_already_exists:" + ",".join(path.as_posix() for path in conflicts),
        }
    for subdir in ("manifests", "metrics", "reports"):
        filesystem_path(run_root / subdir).mkdir(parents=True, exist_ok=True)

    previous_state = _read_json_if_exists(run_root / "manifests" / "stage_state.json") if config.resume else {}
    baseline = previous_state.get("docs_alignment_baseline") if isinstance(previous_state.get("docs_alignment_baseline"), dict) else None
    guard = DocsAlignmentGuard(run_root, baseline=baseline)
    global _ACTIVE_DOCS_ALIGNMENT_GUARD
    _ACTIVE_DOCS_ALIGNMENT_GUARD = guard

    context: dict[str, object] = {
        "run_id": run_id,
        "run_label": str(config.run_label).strip(),
        "stage_run_label": str(config.stage_run_label).strip(),
        "run_root": run_root.as_posix(),
        "stages": previous_state.get("stages", {}) if isinstance(previous_state.get("stages"), dict) else {},
        "blocked_reason": "",
        "status": "running",
    }
    decision_rows: list[dict[str, object]] = _read_csv_rows(run_root / "metrics" / "decision_log.csv") if config.resume else []
    preflight_rows: list[dict[str, object]] = _read_csv_rows(run_root / "metrics" / "preflight_checks.csv") if config.resume else []
    post_rows: list[dict[str, object]] = _read_csv_rows(run_root / "metrics" / "post_stage_checks.csv") if config.resume else []
    repair_rows: list[dict[str, object]] = _read_csv_rows(run_root / "metrics" / "repair_log.csv") if config.resume else []
    stage_summary_rows: list[dict[str, object]] = _read_csv_rows(run_root / "metrics" / "stage_summary.csv") if config.resume else []

    try:
        alignment = read_docs_alignment_snapshot("pipeline_start", "PIPELINE", "baseline_or_resume_check")
        _write_pipeline_manifest(run_root, config, context, guard)
        if alignment.status == "docs_changed_reaudit_required":
            return _terminate_blocked(
                run_root=run_root,
                config=config,
                context=context,
                guard=guard,
                decision_rows=decision_rows,
                stage_summary_rows=stage_summary_rows,
                reason="docs_changed_reaudit_required:" + ";".join(alignment.changed_reasons),
                stage_id="PIPELINE",
                checkpoint_context="pipeline_start_docs_changed",
            )

        source_audit_result = run_v411_source_audit(SourceAuditConfig(dry_run=True))
        source_audit_passed = source_audit_result.get("status") == "ready" and not source_audit_result.get("blockers")
        preflight_rows.append(
            _check_row("PIPELINE", "source_audit_ready", source_audit_passed, source_audit_result.get("status", ""), "ready")
        )
        _write_csv(run_root / "metrics" / "preflight_checks.csv", pd.DataFrame(preflight_rows))
        if not source_audit_passed:
            return _terminate_blocked(
                run_root=run_root,
                config=config,
                context=context,
                guard=guard,
                decision_rows=decision_rows,
                stage_summary_rows=stage_summary_rows,
                reason="source_audit_not_ready:" + json.dumps(source_audit_result.get("blockers", []), sort_keys=True),
                stage_id="PIPELINE",
                checkpoint_context="source_audit_gate",
            )

        for stage_id in _selected_stages(config.start_stage, config.stop_after_stage):
            stage_result = _run_stage_with_gates(
                stage_id=stage_id,
                config=config,
                run_root=run_root,
                context=context,
                guard=guard,
                decision_rows=decision_rows,
                preflight_rows=preflight_rows,
                post_rows=post_rows,
                repair_rows=repair_rows,
                stage_summary_rows=stage_summary_rows,
            )
            if stage_result.get("status") == "blocked":
                return stage_result

        final_alignment = read_docs_alignment_snapshot("before_final_report", "PIPELINE", "final_report")
        if final_alignment.status == "docs_changed_reaudit_required":
            return _terminate_blocked(
                run_root=run_root,
                config=config,
                context=context,
                guard=guard,
                decision_rows=decision_rows,
                stage_summary_rows=stage_summary_rows,
                reason="docs_changed_reaudit_required:" + ";".join(final_alignment.changed_reasons),
                stage_id="PIPELINE",
                checkpoint_context="before_final_report_docs_changed",
            )
        context["status"] = "complete"
        _write_pipeline_state(run_root, context, guard)
        _write_pipeline_manifest(run_root, config, context, guard)
        _write_file_size_audit(run_root)
        _write_pipeline_report(run_root, context, status="complete", blocked_reason="")
        return {"status": "complete", "run_root": run_root.as_posix(), "stages": context["stages"]}
    finally:
        _ACTIVE_DOCS_ALIGNMENT_GUARD = None


def _run_stage_with_gates(
    *,
    stage_id: str,
    config: R5R10PipelineConfig,
    run_root: Path,
    context: dict[str, object],
    guard: DocsAlignmentGuard,
    decision_rows: list[dict[str, object]],
    preflight_rows: list[dict[str, object]],
    post_rows: list[dict[str, object]],
    repair_rows: list[dict[str, object]],
    stage_summary_rows: list[dict[str, object]],
) -> dict[str, object]:
    del guard
    preflight_alignment = read_docs_alignment_snapshot("before_stage_preflight", stage_id, "preflight")
    if preflight_alignment.status == "docs_changed_reaudit_required":
        return _terminate_blocked(
            run_root=run_root,
            config=config,
            context=context,
            guard=_ACTIVE_DOCS_ALIGNMENT_GUARD,
            decision_rows=decision_rows,
            stage_summary_rows=stage_summary_rows,
            reason="docs_changed_reaudit_required:" + ";".join(preflight_alignment.changed_reasons),
            stage_id=stage_id,
            checkpoint_context="before_stage_preflight_docs_changed",
        )

    checks = _stage_preflight(stage_id, context)
    preflight_rows.extend(checks)
    _write_csv(run_root / "metrics" / "preflight_checks.csv", pd.DataFrame(preflight_rows))
    if not all(bool(row["passed"]) for row in checks):
        reason = _blocked_reason_from_checks(checks)
        return _terminate_blocked(
            run_root=run_root,
            config=config,
            context=context,
            guard=_ACTIVE_DOCS_ALIGNMENT_GUARD,
            decision_rows=decision_rows,
            stage_summary_rows=stage_summary_rows,
            reason=reason,
            stage_id=stage_id,
            checkpoint_context="stage_preflight_failed",
        )

    if config.resume:
        resume_alignment = read_docs_alignment_snapshot("before_resuming_from_chunks", stage_id, "resume_or_chunk_repair")
        if resume_alignment.status == "docs_changed_reaudit_required":
            return _terminate_blocked(
                run_root=run_root,
                config=config,
                context=context,
                guard=_ACTIVE_DOCS_ALIGNMENT_GUARD,
                decision_rows=decision_rows,
                stage_summary_rows=stage_summary_rows,
                reason="docs_changed_reaudit_required:" + ";".join(resume_alignment.changed_reasons),
                stage_id=stage_id,
                checkpoint_context="before_resuming_docs_changed",
            )

    exec_alignment = read_docs_alignment_snapshot("before_stage_execution", stage_id, "execute")
    if exec_alignment.status == "docs_changed_reaudit_required":
        return _terminate_blocked(
            run_root=run_root,
            config=config,
            context=context,
            guard=_ACTIVE_DOCS_ALIGNMENT_GUARD,
            decision_rows=decision_rows,
            stage_summary_rows=stage_summary_rows,
            reason="docs_changed_reaudit_required:" + ";".join(exec_alignment.changed_reasons),
            stage_id=stage_id,
            checkpoint_context="before_stage_execution_docs_changed",
        )

    started = time.time()
    try:
        result = _execute_stage(stage_id, config, context)
    except Exception as exc:
        result = {"status": "blocked", "blocked_reason": f"stage_execution_exception:{type(exc).__name__}:{exc}"}
    if config.repair_incomplete:
        repair_alignment = read_docs_alignment_snapshot("after_auto_repair", stage_id, "repair_incomplete_enabled")
        repair_rows.append(
            {
                "stage_id": stage_id,
                "timestamp_utc": _utc_now(),
                "repair_mode": "repair_incomplete_enabled",
                "repair_status": "runner_internal_or_not_needed",
                "docs_alignment_status": repair_alignment.status,
                "controlling_hashes": repair_alignment.controlling_hashes,
                "docs_changed_during_run": repair_alignment.docs_changed_during_run,
                "docs_alignment_checkpoint": repair_alignment.checkpoint,
            }
        )
        _write_csv(run_root / "metrics" / "repair_log.csv", pd.DataFrame(repair_rows))
        if repair_alignment.status == "docs_changed_reaudit_required":
            return _terminate_blocked(
                run_root=run_root,
                config=config,
                context=context,
                guard=_ACTIVE_DOCS_ALIGNMENT_GUARD,
                decision_rows=decision_rows,
                stage_summary_rows=stage_summary_rows,
                reason="docs_changed_reaudit_required:" + ";".join(repair_alignment.changed_reasons),
                stage_id=stage_id,
                checkpoint_context="after_auto_repair_docs_changed",
            )

    post_checks = _stage_post_checks(stage_id, result, context)
    post_rows.extend(post_checks)
    _write_csv(run_root / "metrics" / "post_stage_checks.csv", pd.DataFrame(post_rows))
    passed = all(bool(row["passed"]) for row in post_checks)
    decision_context = "continue" if passed else "terminate_blocked"
    decision_checkpoint = "before_continue_decision" if passed else "before_terminate_blocked"
    decision_alignment = read_docs_alignment_snapshot(decision_checkpoint, stage_id, decision_context)
    if decision_alignment.status == "docs_changed_reaudit_required":
        passed = False
        result["blocked_reason"] = "docs_changed_reaudit_required:" + ";".join(decision_alignment.changed_reasons)
    elif not passed:
        result["blocked_reason"] = _blocked_reason_from_checks(post_checks)

    decision_action = "continue" if passed else "terminate_blocked"
    decision_rows.append(
        {
            "timestamp_utc": _utc_now(),
            "stage_id": stage_id,
            "decision": decision_action,
            "stage_status": result.get("status", ""),
            "passed": bool(passed),
            "blocked_reason": "" if passed else str(result.get("blocked_reason", "")),
            "run_root": str(result.get("run_root", "")),
            "duration_s": round(time.time() - started, 3),
            "docs_alignment_status": decision_alignment.status,
            "controlling_hashes": decision_alignment.controlling_hashes,
            "docs_changed_during_run": decision_alignment.docs_changed_during_run,
            "docs_alignment_checkpoint": decision_alignment.checkpoint,
        }
    )
    _write_csv(run_root / "metrics" / "decision_log.csv", pd.DataFrame(decision_rows))

    stage_summary_rows.append(
        {
            "stage_id": stage_id,
            "status": "passed" if passed else "blocked",
            "stage_run_root": str(result.get("run_root", "")),
            "blocked_reason": "" if passed else str(result.get("blocked_reason", "")),
            "duration_s": round(time.time() - started, 3),
            "docs_alignment_status": decision_alignment.status,
        }
    )
    _write_csv(run_root / "metrics" / "stage_summary.csv", pd.DataFrame(stage_summary_rows))

    if passed:
        stages = context.setdefault("stages", {})
        assert isinstance(stages, dict)
        stages[stage_id] = dict(result)
        _write_pipeline_state(run_root, context, _ACTIVE_DOCS_ALIGNMENT_GUARD)
        _write_pipeline_manifest(run_root, config, context, _ACTIVE_DOCS_ALIGNMENT_GUARD)
        return {"status": "passed", "stage_id": stage_id, "run_root": result.get("run_root", "")}

    if config.continue_on_stage_failure:
        return {"status": "blocked", "stage_id": stage_id, "blocked_reason": "continue_on_stage_failure_forbidden_by_protocol"}
    return _terminate_blocked(
        run_root=run_root,
        config=config,
        context=context,
        guard=_ACTIVE_DOCS_ALIGNMENT_GUARD,
        decision_rows=decision_rows,
        stage_summary_rows=stage_summary_rows,
        reason=str(result.get("blocked_reason", "")),
        stage_id=stage_id,
        checkpoint_context="post_stage_gate_failed",
    )


def _execute_stage(stage_id: str, config: R5R10PipelineConfig, context: dict[str, object]) -> dict[str, object]:
    if stage_id == "R5":
        stage_run_id = _stage_run_id(config, W01_OUTPUT_ROOT)
        return run_lqr_w01_dense_chunked(
            W01DenseRunConfig(
                run_id=stage_run_id,
                run_label=config.stage_run_label,
                rows=L6_RICH_SIDE_ROW_COUNT,
                candidate_count=L6_RICH_SIDE_CANDIDATE_COUNT,
                paired_tests_per_candidate=L6_RICH_SIDE_PAIRED_TESTS_PER_CANDIDATE,
                candidate_chunk_size=int(config.candidate_chunk_size),
                workers=config.workers,
                max_workers=config.max_workers,
                storage_format=config.storage_format,
                compression_level=int(config.compression_level),
                resume=bool(config.resume),
                repair_incomplete=bool(config.repair_incomplete),
                dry_run_schedule=bool(config.dry_run_schedule),
                schedule_mode=BALANCED_SCHEDULE_MODE,
            )
        )
    if stage_id == "R7":
        r5_root = Path(str(context["stages"]["R5"]["run_root"]))
        stage_run_id = _stage_run_id(config, W3_OUTPUT_ROOT)
        replay = run_w3_survival(
            W3SurvivalConfig(
                run_id=stage_run_id,
                run_label=config.stage_run_label,
                input_root=r5_root,
                paired_tests_per_variant=20,
                candidate_chunk_size=int(config.candidate_chunk_size),
                workers=config.workers,
                max_workers=config.max_workers,
                storage_format=config.storage_format,
                compression_level=int(config.compression_level),
                resume=bool(config.resume),
                repair_incomplete=bool(config.repair_incomplete),
                dry_run_schedule=bool(config.dry_run_schedule),
            )
        )
        if replay.get("status") == "complete":
            analysis = run_w3_survival_analysis(W3SurvivalAnalysisConfig(input_root=Path(str(replay["run_root"]))))
        else:
            analysis = {"status": "blocked", "blocked_reason": "W3_replay_not_complete"}
        replay["analysis_status"] = analysis.get("status", "")
        replay["survivor_count"] = analysis.get("survivor_count", 0)
        replay["survivor_registry"] = analysis.get("survivor_registry", "")
        return replay
    if stage_id == "R8":
        r7_root = Path(str(context["stages"]["R7"]["run_root"]))
        study_id = _stage_run_id(config, POST_W3_OUTPUT_ROOT)
        study = run_post_w3_library_size_study(
            PostW3LibrarySizeStudyConfig(
                input_root=r7_root,
                output_root=POST_W3_OUTPUT_ROOT,
                run_id=study_id,
                run_label=config.stage_run_label,
            )
        )
        if study.get("status") != "complete":
            return study
        outcome_id = _stage_run_id(config, OUTCOME_OUTPUT_ROOT)
        outcome = run_outcome_model_build(
            OutcomeModelBuildConfig(
                compact_library_path=Path(str(study["manifest"])),
                output_root=OUTCOME_OUTPUT_ROOT,
                run_id=outcome_id,
                run_label=config.stage_run_label,
                library_size_case_id="balanced_cluster",
            )
        )
        study["outcome_status"] = outcome.get("status", "")
        study["outcome_run_root"] = outcome.get("run_root", "")
        study["outcome_model_table"] = outcome.get("outcome_model_table", "")
        return study
    if stage_id == "R9":
        r8_root = Path(str(context["stages"]["R8"]["run_root"]))
        outcome_root = Path(str(context["stages"]["R8"]["outcome_run_root"]))
        stage_run_id = _stage_run_id(config, DEFAULT_REPEATED_OUTPUT_ROOT)
        return run_repeated_launch_learning_curve(
            RepeatedLaunchValidationConfig(
                library_root=r8_root,
                outcome_root=outcome_root,
                output_root=DEFAULT_REPEATED_OUTPUT_ROOT,
                run_id=stage_run_id,
                run_label=config.stage_run_label,
                storage_format=config.storage_format,
                compression_level=int(config.compression_level),
                candidate_chunk_size=_validation_chunk_size(config.candidate_chunk_size),
                dry_run_schedule=bool(config.dry_run_schedule),
                max_primitives_per_launch=0,
                workers=_validation_worker_count(config.workers),
                max_workers=config.max_workers,
                worker_backend="process",
                history_log_mode=config.history_log_mode,
                history_debug_sample_stride=config.history_debug_sample_stride,
            )
        )
    if stage_id == "R10":
        r8_root = Path(str(context["stages"]["R8"]["run_root"]))
        outcome_root = Path(str(context["stages"]["R8"]["outcome_run_root"]))
        governor_config_path = _r9_initial_governor_config_path(context)
        stage_run_id = _stage_run_id(config, DEFAULT_CHANGED_OUTPUT_ROOT)
        return run_changed_case_validation(
            ChangedCaseValidationConfig(
                library_root=r8_root,
                outcome_root=outcome_root,
                output_root=DEFAULT_CHANGED_OUTPUT_ROOT,
                run_id=stage_run_id,
                run_label=config.stage_run_label,
                storage_format=config.storage_format,
                compression_level=int(config.compression_level),
                candidate_chunk_size=_validation_chunk_size(config.candidate_chunk_size),
                dry_run_schedule=bool(config.dry_run_schedule),
                max_primitives_per_launch=0,
                r10_mode=config.r10_mode,
                workers=_validation_worker_count(config.workers),
                max_workers=config.max_workers,
                governor_config_path=governor_config_path,
                worker_backend="process",
                history_log_mode=config.history_log_mode,
                history_debug_sample_stride=config.history_debug_sample_stride,
            )
        )
    if stage_id == "R11":
        r8_root = Path(str(context["stages"]["R8"]["run_root"]))
        outcome_root = Path(str(context["stages"]["R8"]["outcome_run_root"]))
        governor_config_path = _r10_frozen_governor_config_path(context)
        stage_run_id = _stage_run_id(config, DEFAULT_HELDOUT_CHANGED_OUTPUT_ROOT)
        return run_heldout_changed_case_validation(
            HeldoutChangedCaseValidationConfig(
                library_root=r8_root,
                outcome_root=outcome_root,
                output_root=DEFAULT_HELDOUT_CHANGED_OUTPUT_ROOT,
                run_id=stage_run_id,
                run_label=config.stage_run_label,
                storage_format=config.storage_format,
                compression_level=int(config.compression_level),
                candidate_chunk_size=_validation_chunk_size(config.candidate_chunk_size),
                dry_run_schedule=bool(config.dry_run_schedule),
                max_primitives_per_launch=0,
                workers=_validation_worker_count(config.workers),
                max_workers=config.max_workers,
                governor_config_path=governor_config_path,
                worker_backend="process",
                history_log_mode=config.history_log_mode,
                history_debug_sample_stride=config.history_debug_sample_stride,
            )
        )
    raise KeyError(f"unknown stage_id: {stage_id}")


def _stage_preflight(stage_id: str, context: dict[str, object]) -> list[dict[str, object]]:
    rows = [
        _check_row(stage_id, "primitive_horizon_exact_0p100s", PRIMITIVE_FINITE_HORIZON_S == 0.100, PRIMITIVE_FINITE_HORIZON_S, 0.100),
        _check_row(stage_id, "controller_input_slots_exact_5", CONTROLLER_INPUT_SLOTS_PER_PRIMITIVE == 5, CONTROLLER_INPUT_SLOTS_PER_PRIMITIVE, 5),
        _check_row(stage_id, "controller_update_period_exact_0p020s", CONTROLLER_INPUT_UPDATE_PERIOD_S == 0.020, CONTROLLER_INPUT_UPDATE_PERIOD_S, 0.020),
        _check_row(stage_id, "primitive_timing_contract_v411", PRIMITIVE_TIMING_CONTRACT_VERSION == "v411_0p10s_5slot_20ms", PRIMITIVE_TIMING_CONTRACT_VERSION, "v411_0p10s_5slot_20ms"),
    ]
    stages = context.get("stages", {})
    if stage_id != "R5":
        previous = STAGE_ORDER[STAGE_ORDER.index(stage_id) - 1]
        rows.append(_check_row(stage_id, f"previous_stage_{previous}_passed", previous in stages, previous in stages, True))
    if stage_id == "R7" and "R5" in stages:
        rows.append(_check_row(stage_id, "R5_frozen_bundle_root_available", bool(stages["R5"].get("run_root")), stages["R5"].get("run_root", ""), "nonempty"))
    if stage_id in {"R9", "R10", "R11"} and "R8" in stages:
        rows.append(_check_row(stage_id, "R8_library_and_outcome_available", bool(stages["R8"].get("run_root")) and bool(stages["R8"].get("outcome_run_root")), stages["R8"], "library_root_and_outcome_root"))
    if stage_id == "R10":
        path = _r9_initial_governor_config_path(context)
        rows.append(_check_row(stage_id, "R9_initial_governor_config_for_R10_optional_internal_preflight_only", True, "" if path is None else path.as_posix(), "optional_not_active_gate"))
    if stage_id == "R11":
        path = _r10_frozen_governor_config_path(context)
        rows.append(_check_row(stage_id, "R10_frozen_governor_config_for_R11_available", path is not None, "" if path is None else path.as_posix(), "exists_after_R10_pass"))
    return rows


def _stage_post_checks(stage_id: str, result: dict[str, object], context: dict[str, object]) -> list[dict[str, object]]:
    if result.get("status") == "blocked":
        return [_check_row(stage_id, "stage_execution_not_blocked", False, result.get("blocked_reason", ""), "unblocked")]
    run_root = Path(str(result.get("run_root", "")))
    if stage_id == "R5":
        manifest = _read_json_if_exists(run_root / "manifests" / "run_manifest.json")
        if bool(manifest.get("dry_run_schedule", False)) or result.get("status") == "dry_run_schedule":
            start_counts = dict(manifest.get("per_start_family_row_counts", {}))
            required_start_families = {
                "launch_gate",
                "inflight_nominal",
                "inflight_lift_region",
                "inflight_boundary_near",
                "inflight_recovery_edge",
            }
            return [
                _check_row(stage_id, "stage_status_dry_run_schedule", result.get("status") == "dry_run_schedule", result.get("status", ""), "dry_run_schedule"),
                _check_row(stage_id, "method_evidence_level_dry_schedule_only", manifest.get("method_evidence_level") == W01_DRY_SCHEDULE_ONLY, manifest.get("method_evidence_level", ""), W01_DRY_SCHEDULE_ONLY),
                _check_row(stage_id, "dry_schedule_rows_requested_active_dense_target", int(manifest.get("rows_requested", 0)) == L6_RICH_SIDE_ROW_COUNT, manifest.get("rows_requested", 0), L6_RICH_SIDE_ROW_COUNT),
                _check_row(
                    stage_id,
                    "dry_schedule_dense_target_derived_from_active_catalogue",
                    int(manifest.get("rows_requested", 0))
                    == len(ACTIVE_PRIMITIVE_IDS)
                    * L6_RICH_SIDE_CANDIDATE_COUNT
                    * len(R5_EVIDENCE_BLOCKS)
                    * L6_RICH_SIDE_PAIRED_TESTS_PER_CANDIDATE,
                    manifest.get("rows_requested", 0),
                    f"{len(ACTIVE_PRIMITIVE_IDS)}*{L6_RICH_SIDE_CANDIDATE_COUNT}*{len(R5_EVIDENCE_BLOCKS)}*{L6_RICH_SIDE_PAIRED_TESTS_PER_CANDIDATE}",
                ),
                _check_row(stage_id, "dry_schedule_candidate_count_32", int(manifest.get("candidate_count", 0)) == 32, manifest.get("candidate_count", 0), 32),
                _check_row(stage_id, "dry_schedule_paired_tests_per_candidate_50", int(manifest.get("paired_tests_per_candidate", 0)) == L6_RICH_SIDE_PAIRED_TESTS_PER_CANDIDATE, manifest.get("paired_tests_per_candidate", 0), L6_RICH_SIDE_PAIRED_TESTS_PER_CANDIDATE),
                _check_row(stage_id, "dry_schedule_active_primitive_count_8", int(manifest.get("active_primitive_count", 0)) == 8, manifest.get("active_primitive_count", 0), 8),
                _check_row(stage_id, "dry_schedule_retired_launch_capture_not_active", not bool(manifest.get("retired_launch_capture_primitive_ids_active", True)), manifest.get("retired_launch_capture_primitive_ids_active", ""), False),
                _check_row(stage_id, "dry_schedule_all_start_state_regimes", set(start_counts) == required_start_families, sorted(start_counts), sorted(required_start_families)),
                _check_row(stage_id, "dry_schedule_r6_archived", bool(manifest.get("R6_W2_archived_diagnostic_only", False)), manifest.get("R6_W2_archived_diagnostic_only", False), True),
                _check_row(stage_id, "dry_schedule_w3_direct_source", manifest.get("R7_W3_direct_source") == "r5_transition_selected_for_r7_frozen_controller_bundle", manifest.get("R7_W3_direct_source", ""), "r5_transition_selected_for_r7_frozen_controller_bundle"),
                _file_size_check(stage_id, run_root),
            ]
        return [
            _check_row(stage_id, "stage_status_not_diagnostic", result.get("status") == "complete", result.get("status", ""), "complete"),
            _check_row(stage_id, "method_evidence_level_w01_dense", manifest.get("method_evidence_level") == "w01_dense_evidence_complete", manifest.get("method_evidence_level", ""), "w01_dense_evidence_complete"),
            _check_row(stage_id, "w01_dense_evidence_complete_true", bool(manifest.get("w01_dense_evidence_complete", False)), manifest.get("w01_dense_evidence_complete", False), True),
            _check_row(stage_id, "actual_row_count_active_dense_target", int(manifest.get("actual_row_count", 0)) == L6_RICH_SIDE_ROW_COUNT, manifest.get("actual_row_count", 0), L6_RICH_SIDE_ROW_COUNT),
            _check_row(stage_id, "candidate_count_32", int(manifest.get("candidate_count", 0)) == 32, manifest.get("candidate_count", 0), 32),
            _check_row(stage_id, "paired_tests_per_candidate_50", int(manifest.get("paired_tests_per_candidate", 0)) == L6_RICH_SIDE_PAIRED_TESTS_PER_CANDIDATE, manifest.get("paired_tests_per_candidate", 0), L6_RICH_SIDE_PAIRED_TESTS_PER_CANDIDATE),
            _check_row(stage_id, "active_primitive_count_8", int(manifest.get("active_primitive_count", 0)) == 8, manifest.get("active_primitive_count", 0), 8),
            _check_row(stage_id, "retired_launch_capture_not_active", not bool(manifest.get("retired_launch_capture_primitive_ids_active", True)), manifest.get("retired_launch_capture_primitive_ids_active", ""), False),
            _check_row(stage_id, "r5_launch_aware_decision_passed_for_review", manifest.get("r5_launch_aware_decision") == R5_LAUNCH_AWARE_DENSE_PASSED_FOR_REVIEW, manifest.get("r5_launch_aware_decision", ""), R5_LAUNCH_AWARE_DENSE_PASSED_FOR_REVIEW),
            _check_row(stage_id, "launch_gate_candidate_availability_audit", _r5_launch_gate_audit_passed(run_root), "audit", "passed"),
            _check_row(stage_id, "w3_replay_only", bool(manifest.get("W3_replay_only", False)), manifest.get("W3_replay_only", False), True),
            _check_row(stage_id, "w2_not_required_for_move_on", not bool(manifest.get("W2_required_for_move_on", True)), manifest.get("W2_required_for_move_on", True), False),
            _check_row(stage_id, "r6_archived_diagnostic_only", bool(manifest.get("R6_W2_archived_diagnostic_only", False)), manifest.get("R6_W2_archived_diagnostic_only", False), True),
            _check_row(stage_id, "no_clustering_before_w3", bool(manifest.get("no_clustering_before_W3", False)), manifest.get("no_clustering_before_W3", False), True),
            _check_row(stage_id, "selected_worker_count_8", int(manifest.get("selected_worker_count", 0)) == 8, manifest.get("selected_worker_count", 0), 8),
            _check_row(stage_id, "frozen_bundle_exists", (run_root / "manifests" / "frozen_w01_controller_bundle.json").is_file(), (run_root / "manifests" / "frozen_w01_controller_bundle.json").as_posix(), "exists"),
            _check_row(stage_id, "table_manifest_exists", (run_root / "manifests" / "table_manifest.json").is_file(), (run_root / "manifests" / "table_manifest.json").as_posix(), "exists"),
            _file_size_check(stage_id, run_root),
        ]
    if stage_id == "R7":
        manifest = _read_json_if_exists(run_root / "manifests" / "w3_survival_manifest.json")
        registry = _read_json_if_exists(run_root / "manifests" / "w3_survivor_registry.json")
        return [
            _check_row(stage_id, "w3_replay_complete", result.get("status") == "complete", result.get("status", ""), "complete"),
            _check_row(stage_id, "w3_analysis_survivors_available", result.get("analysis_status") == "w3_survivors_available", result.get("analysis_status", ""), "w3_survivors_available"),
            _check_row(stage_id, "method_evidence_level_w3_dense", manifest.get("method_evidence_level") == "w3_dense_survival_pass", manifest.get("method_evidence_level", ""), "w3_dense_survival_pass"),
            _check_row(stage_id, "fixed_lqr_replay_only", bool(manifest.get("fixed_lqr_replay_only", False)), manifest.get("fixed_lqr_replay_only", False), True),
            _check_row(stage_id, "source_input_kind_r5", manifest.get("source_input_kind") == "r5_frozen_bundle_direct", manifest.get("source_input_kind", ""), "r5_frozen_bundle_direct"),
            _check_row(stage_id, "no_w3_retuning", not bool(manifest.get("mutates_Q_R_K_reference_horizon_entry_set_or_entry_role", True)), manifest.get("mutates_Q_R_K_reference_horizon_entry_set_or_entry_role", True), False),
            _check_row(stage_id, "not_fixture_evidence", not bool(manifest.get("test_fixture_not_method_evidence", True)), manifest.get("test_fixture_not_method_evidence", True), False),
            _check_row(stage_id, "w3_survivor_count_positive", int(registry.get("survivor_count", 0)) > 0, registry.get("survivor_count", 0), ">0"),
            _check_row(stage_id, "w3_launch_gate_entry_survivor_count_positive", int(registry.get("launch_gate_entry_survivor_count", 0)) > 0, registry.get("launch_gate_entry_survivor_count", 0), ">0"),
            _check_row(stage_id, "w3_all_active_families_have_launch_gate_entry_survivors", set(registry.get("surviving_launch_gate_entry_primitive_ids", [])) == set(ACTIVE_PRIMITIVE_IDS), registry.get("surviving_launch_gate_entry_primitive_ids", []), list(ACTIVE_PRIMITIVE_IDS)),
            _file_size_check(stage_id, run_root),
        ]
    if stage_id == "R8":
        study_root = Path(str(result.get("run_root", "")))
        outcome_root = Path(str(result.get("outcome_run_root", "")))
        summary = _read_csv_if_exists(study_root / "metrics" / "library_size_case_summary.csv")
        outcome_manifest = _read_json_if_exists(outcome_root / "manifests" / "outcome_model_manifest.json")
        return [
            _check_row(stage_id, "post_w3_study_complete", result.get("status") == "complete", result.get("status", ""), "complete"),
            _check_row(stage_id, "outcome_model_complete", result.get("outcome_status") == "complete", result.get("outcome_status", ""), "complete"),
            _check_row(stage_id, "all_active_library_size_cases_present", set(summary.get("library_size_case_id", pd.Series(dtype=str)).astype(str)) == set(LIBRARY_SIZE_CASE_IDS), sorted(set(summary.get("library_size_case_id", pd.Series(dtype=str)).astype(str))), sorted(LIBRARY_SIZE_CASE_IDS)),
            _check_row(stage_id, "outcome_all_active_library_size_cases_present", set(str(value) for value in outcome_manifest.get("library_size_case_ids", [])) == set(LIBRARY_SIZE_CASE_IDS), outcome_manifest.get("library_size_case_ids", []), list(LIBRARY_SIZE_CASE_IDS)),
            _check_row(stage_id, "launch_gate_candidate_availability_audit", _r8_launch_gate_audit_passed(study_root), "audit", "passed"),
            _file_size_check(stage_id, study_root),
            _file_size_check(stage_id, outcome_root),
        ]
    if stage_id == "R9":
        manifest = _read_json_if_exists(run_root / "manifests" / "repeated_launch_fixed_case_manifest.json")
        rows = [
            _check_row(stage_id, "stage_status_complete", result.get("status") == "complete", result.get("status", ""), "complete"),
            _check_row(stage_id, "not_dry_run_or_diagnostic", not bool(manifest.get("dry_run_schedule", False)), manifest.get("dry_run_schedule", False), False),
            _check_row(stage_id, "pass_gate_true", bool(manifest.get("pass_gate", False)), manifest.get("pass_gate", False), True),
            _check_row(stage_id, "final_heldout_launch_count_exact", int(manifest.get("actual_final_heldout_launches", 0)) == R9_EXPECTED_FINAL_HELDOUT_LAUNCHES, manifest.get("actual_final_heldout_launches", 0), R9_EXPECTED_FINAL_HELDOUT_LAUNCHES),
            _check_row(stage_id, "history_launch_count_exact", int(manifest.get("actual_history_launches", 0)) == R9_EXPECTED_HISTORY_LAUNCHES, manifest.get("actual_history_launches", 0), R9_EXPECTED_HISTORY_LAUNCHES),
            _check_row(stage_id, "r9_internal_preflight_protocol", manifest.get("validation_protocol") == "internal_fixed_case_outer_loop_preflight_initial_governor_tuning_not_thesis_evidence", manifest.get("validation_protocol", ""), "internal_fixed_case_outer_loop_preflight_initial_governor_tuning_not_thesis_evidence"),
            _check_row(stage_id, "r9_initial_governor_config_for_r10_exists", (run_root / "manifests" / "initial_governor_config_for_r10.json").is_file(), (run_root / "manifests" / "initial_governor_config_for_r10.json").as_posix(), "exists"),
            _check_row(stage_id, "r9_initial_governor_config_selected_for_r10", _read_json_if_exists(run_root / "manifests" / "initial_governor_config_for_r10.json").get("status") == "selected_for_r10_initialisation", _read_json_if_exists(run_root / "manifests" / "initial_governor_config_for_r10.json").get("status", ""), "selected_for_r10_initialisation"),
            _check_row(stage_id, "all_active_library_size_cases", set(manifest.get("library_size_case_ids", [])) == set(LIBRARY_SIZE_CASE_IDS), manifest.get("library_size_case_ids", []), list(LIBRARY_SIZE_CASE_IDS)),
            _check_row(stage_id, "r9_quick_preflight_memory_ladder_conditions", set(manifest.get("policy_history_conditions", [])) == set(R9_POLICY_HISTORY_CONDITIONS), manifest.get("policy_history_condition_count", 0), len(R9_POLICY_HISTORY_CONDITIONS)),
            _check_row(stage_id, "first_decision_launch_gate_audits_present", _validation_launch_gate_audit_passed(run_root), "audit", "passed"),
            _file_size_check(stage_id, run_root),
        ]
        rows.extend(_validation_gate_checks(stage_id, run_root, prefix="r9_validation_gate"))
        return rows
    if stage_id == "R10":
        manifest = _read_json_if_exists(run_root / "manifests" / "environment_changed_case_manifest.json")
        rows = [
            _check_row(stage_id, "stage_status_complete", result.get("status") == "complete", result.get("status", ""), "complete"),
            _check_row(stage_id, "not_dry_run_or_reduced_diagnostic", not bool(manifest.get("dry_run_schedule", False)) and manifest.get("validation_protocol") != "reduced_diagnostic_not_target_R10", manifest.get("validation_protocol", ""), "changed_case_viability_governor_learning_rollout_validation_not_final_claim_gate"),
            _check_row(stage_id, "pass_gate_true", bool(manifest.get("pass_gate", False)), manifest.get("pass_gate", False), True),
            _check_row(stage_id, "r10_not_gated_by_r9_internal_preflight", True, manifest.get("governor_config_override_active", False), "R9 optional internal preflight only"),
            _check_row(stage_id, "final_heldout_launch_count_exact", int(manifest.get("actual_final_heldout_launches", 0)) == R10_EXPECTED_FINAL_HELDOUT_LAUNCHES, manifest.get("actual_final_heldout_launches", 0), R10_EXPECTED_FINAL_HELDOUT_LAUNCHES),
            _check_row(stage_id, "history_launch_count_exact", int(manifest.get("actual_history_launches", 0)) == R10_EXPECTED_HISTORY_LAUNCHES, manifest.get("actual_history_launches", 0), R10_EXPECTED_HISTORY_LAUNCHES),
            _check_row(stage_id, "r10_no_model_latency_variation_audit", _r10_variation_audit_passed(run_root), "audit", "passed"),
            _check_row(stage_id, "r10_frozen_governor_config_for_r11_exists", (run_root / "manifests" / "frozen_governor_config_for_r11.json").is_file(), (run_root / "manifests" / "frozen_governor_config_for_r11.json").as_posix(), "exists"),
            _check_row(stage_id, "r10_frozen_governor_config_selected_for_r11", _read_json_if_exists(run_root / "manifests" / "frozen_governor_config_for_r11.json").get("status") == "selected_for_r11", _read_json_if_exists(run_root / "manifests" / "frozen_governor_config_for_r11.json").get("status", ""), "selected_for_r11"),
            _check_row(stage_id, "first_decision_launch_gate_audits_present", _validation_launch_gate_audit_passed(run_root), "audit", "passed"),
            _file_size_check(stage_id, run_root),
        ]
        rows.extend(_validation_gate_checks(stage_id, run_root, prefix="r10_validation_gate"))
        return rows
    if stage_id == "R11":
        manifest = _read_json_if_exists(run_root / "manifests" / "heldout_environment_validation_manifest.json")
        rows = [
            _check_row(stage_id, "stage_status_complete", result.get("status") == "complete", result.get("status", ""), "complete"),
            _check_row(stage_id, "strict_heldout_protocol", manifest.get("validation_protocol") == "strict_heldout_environment_only_changed_case_repeated_launch_rollout_validation", manifest.get("validation_protocol", ""), "strict_heldout_environment_only_changed_case_repeated_launch_rollout_validation"),
            _check_row(stage_id, "pass_gate_true", bool(manifest.get("pass_gate", False)), manifest.get("pass_gate", False), True),
            _check_row(stage_id, "r11_consumed_r10_frozen_governor_config", bool(manifest.get("governor_config_override_active", False)), manifest.get("governor_config_override_active", False), True),
            _check_row(stage_id, "final_heldout_launch_count_exact", int(manifest.get("actual_final_heldout_launches", 0)) == R10_EXPECTED_FINAL_HELDOUT_LAUNCHES, manifest.get("actual_final_heldout_launches", 0), R10_EXPECTED_FINAL_HELDOUT_LAUNCHES),
            _check_row(stage_id, "history_launch_count_exact", int(manifest.get("actual_history_launches", 0)) == R10_EXPECTED_HISTORY_LAUNCHES, manifest.get("actual_history_launches", 0), R10_EXPECTED_HISTORY_LAUNCHES),
            _check_row(stage_id, "r11_no_model_latency_variation_audit", _r10_variation_audit_passed(run_root), "audit", "passed"),
            _check_row(stage_id, "first_decision_launch_gate_audits_present", _validation_launch_gate_audit_passed(run_root), "audit", "passed"),
            _file_size_check(stage_id, run_root),
        ]
        rows.extend(_validation_gate_checks(stage_id, run_root, prefix="r11_validation_gate"))
        return rows
    return [_check_row(stage_id, "unknown_stage", False, stage_id, "known_stage")]


def _terminate_blocked(
    *,
    run_root: Path,
    config: R5R10PipelineConfig,
    context: dict[str, object],
    guard: DocsAlignmentGuard | None,
    decision_rows: list[dict[str, object]],
    stage_summary_rows: list[dict[str, object]],
    reason: str,
    stage_id: str,
    checkpoint_context: str,
) -> dict[str, object]:
    if guard is None:
        raise RuntimeError("docs alignment guard missing during terminate_blocked")
    alignment = read_docs_alignment_snapshot("before_terminate_blocked", stage_id, checkpoint_context)
    context["status"] = "blocked"
    context["blocked_reason"] = reason
    decision_rows.append(
        {
            "timestamp_utc": _utc_now(),
            "stage_id": stage_id,
            "decision": "terminate_blocked",
            "stage_status": "blocked",
            "passed": False,
            "blocked_reason": reason,
            "run_root": "",
            "duration_s": 0.0,
            "docs_alignment_status": alignment.status,
            "controlling_hashes": alignment.controlling_hashes,
            "docs_changed_during_run": alignment.docs_changed_during_run,
            "docs_alignment_checkpoint": alignment.checkpoint,
        }
    )
    _write_csv(run_root / "metrics" / "decision_log.csv", pd.DataFrame(decision_rows))
    _write_csv(run_root / "metrics" / "stage_summary.csv", pd.DataFrame(stage_summary_rows))
    _write_pipeline_state(run_root, context, guard)
    _write_pipeline_manifest(run_root, config, context, guard)
    _write_file_size_audit(run_root)
    final_alignment = read_docs_alignment_snapshot("before_final_report", stage_id, "blocked_final_report")
    if final_alignment.status == "docs_changed_reaudit_required" and not reason.startswith("docs_changed_reaudit_required"):
        context["blocked_reason"] = "docs_changed_reaudit_required:" + ";".join(final_alignment.changed_reasons)
        _write_pipeline_state(run_root, context, guard)
        _write_pipeline_manifest(run_root, config, context, guard)
        reason = str(context["blocked_reason"])
    _write_pipeline_report(run_root, context, status="blocked", blocked_reason=reason)
    return {"status": "blocked", "blocked_reason": reason, "run_root": run_root.as_posix(), "stages": context.get("stages", {})}


def _write_pipeline_manifest(run_root: Path, config: R5R10PipelineConfig, context: dict[str, object], guard: DocsAlignmentGuard | None) -> None:
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "project_title_version": PROJECT_TITLE_VERSION,
        "status": context.get("status", "running"),
        "run_id": context.get("run_id"),
        "run_label": str(config.run_label).strip(),
        "stage_run_label": str(config.stage_run_label).strip(),
        "run_root": run_root.as_posix(),
        "config": _json_ready(asdict(config)),
        "stage_order": list(STAGE_ORDER),
        "stage_role_map": {
            "R5": "primitive_learning_dense_synthesis",
            "R6": "archived_diagnostic_only",
            "R7": "primitive_validation_frozen_w3_holdout",
            "R8": "post_w3_clustering_and_library_size_study",
            "R9": "internal_preflight_ablation_only_not_active_pipeline_gate",
            "R10": "viability_governor_changed_case_tuning",
            "R11": "viability_governor_strict_heldout_validation",
        },
        "full_run_policy": "R5_then_R7_R8_R10_R11_only_after_previous_stage_passes_R6_archived_diagnostic_only_R9_internal_preflight_only",
        "thesis_facing_workflow": "R5 -> R7 -> R8 -> R10 -> R11 -> Reality",
        "r8_thesis_role": "post_R7_library_compression_and_library_size_cross_study",
        "r9_thesis_role": "internal_preflight_excluded_from_thesis_workflow_narrative",
        "execution_scope": "R5_only" if (config.stop_after_stage or "").upper() == "R5" else "R5_R7_R8_R10_R11_chain",
        "archived_stages": list(ARCHIVED_STAGES),
        "r5_only_scope": bool((config.stop_after_stage or "").upper() == "R5"),
        "R7_R8_R10_R11_deliberately_not_run_when_stop_after_stage_R5": bool((config.stop_after_stage or "").upper() == "R5"),
        "r5_expected_dense_rows": int(L6_RICH_SIDE_ROW_COUNT),
        "active_primitive_count": int(len(ACTIVE_PRIMITIVE_IDS)),
        "retired_launch_capture_primitive_ids_active": False,
        "docs_alignment_baseline": {} if guard is None or guard.baseline is None else guard.baseline,
        "docs_changed_during_run": False if guard is None else bool(guard.changed_during_run),
        "blocked_reason": context.get("blocked_reason", ""),
        "blocked_claims": list(BLOCKED_CLAIMS),
    }
    _write_json(run_root / "manifests" / "pipeline_manifest.json", payload)


def _write_pipeline_state(run_root: Path, context: dict[str, object], guard: DocsAlignmentGuard | None) -> None:
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "status": context.get("status", "running"),
        "run_id": context.get("run_id"),
        "run_root": run_root.as_posix(),
        "stages": context.get("stages", {}),
        "blocked_reason": context.get("blocked_reason", ""),
        "docs_alignment_baseline": {} if guard is None or guard.baseline is None else guard.baseline,
        "docs_changed_during_run": False if guard is None else bool(guard.changed_during_run),
        "updated_utc": _utc_now(),
    }
    _write_json(run_root / "manifests" / "stage_state.json", payload)


def _write_pipeline_report(run_root: Path, context: dict[str, object], *, status: str, blocked_reason: str) -> None:
    stages = context.get("stages", {})
    r5_stage = stages.get("R5", {}) if isinstance(stages, dict) else {}
    r5_manifest = _read_json_if_exists(Path(str(r5_stage.get("run_root", ""))) / "manifests" / "run_manifest.json") if r5_stage else {}
    r5_decision = str(r5_manifest.get("r5_launch_aware_decision", "not_run"))
    r5_row_count = r5_manifest.get("actual_row_count", 0)
    r5_run_root = str(r5_stage.get("run_root", ""))
    docs_hashes = _latest_docs_hashes_for_report(run_root)
    lines = [
        "# v5.20 Transition-Aware Dense Pipeline Report",
        "",
        f"- Status: `{status}`",
        f"- Blocked reason: `{blocked_reason}`",
        f"- Project title version: `{PROJECT_TITLE_VERSION}`",
        f"- Docs hashes: `{docs_hashes}`",
        f"- R5 decision: `{r5_decision}`",
        f"- R5 run root: `{r5_run_root}`",
        f"- R5 rows written: `{r5_row_count}` / `{L6_RICH_SIDE_ROW_COUNT}`",
        f"- Worker/chunk/storage settings: workers `{r5_manifest.get('selected_worker_count', '')}`, chunk count `{r5_manifest.get('chunk_count', '')}`, chunk size `{r5_manifest.get('candidate_chunk_size', '')}`, storage `{r5_manifest.get('storage_format', '')}`.",
        f"- Active primitive IDs: `{json.dumps(list(ACTIVE_PRIMITIVE_IDS))}`",
        "- R6 is archived as diagnostic-only and is not an active gate in this pipeline.",
        "- Active thesis workflow: R5 -> R7 -> R8 -> R10 -> R11 -> Reality; R9 is internal preflight/ablation only.",
        "- R7-R8-R10-R11 deliberately not run when `--stop-after-stage R5` is selected.",
        "- Claim boundary: simulation evidence only; no hardware readiness, real-flight transfer, mission success, full autonomy, or memory-improvement claim is made here.",
        "",
        "Stages:",
    ]
    if isinstance(stages, dict):
        for stage_id in STAGE_ORDER:
            stage = stages.get(stage_id, {})
            lines.append(f"- `{stage_id}`: `{stage.get('status', 'not_run')}` root `{stage.get('run_root', '')}`")
    _write_text(run_root / "reports" / "pipeline_report.md", "\n".join(lines) + "\n")


def _latest_docs_hashes_for_report(run_root: Path) -> str:
    frame = _read_csv_if_exists(run_root / "metrics" / "docs_alignment_audit.csv")
    if frame.empty or "path" not in frame.columns or "sha256" not in frame.columns:
        return "{}"
    latest = frame.dropna(subset=["path"]).groupby("path", as_index=False).tail(1)
    hashes = {str(row["path"]): str(row.get("sha256", "")) for row in latest.to_dict(orient="records")}
    return json.dumps(hashes, sort_keys=True, separators=(",", ":"))


def _doc_snapshot_row(path: Path, checkpoint: str, stage_id: str, decision_context: str, timestamp: str) -> dict[str, object]:
    fs_path = filesystem_path(path)
    readable = fs_path.is_file()
    byte_count = 0
    sha256 = ""
    modified_time = ""
    error = ""
    if readable:
        try:
            stat = fs_path.stat()
            byte_count = int(stat.st_size)
            modified_time = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
            sha256 = _file_sha256(fs_path)
        except Exception as exc:
            readable = False
            error = f"{type(exc).__name__}:{exc}"
    return {
        "timestamp_utc": timestamp,
        "checkpoint": checkpoint,
        "stage_id": stage_id,
        "decision_context": decision_context,
        "path": path.as_posix(),
        "readable": bool(readable),
        "byte_count": int(byte_count),
        "sha256": sha256,
        "modified_time_utc": modified_time,
        "error": error,
    }


def _baseline_from_rows(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        str(row["path"]): {
            "path": str(row["path"]),
            "readable": bool(row["readable"]),
            "byte_count": int(row["byte_count"]),
            "sha256": str(row["sha256"]),
            "modified_time_utc": str(row["modified_time_utc"]),
        }
        for row in rows
    }


def _compare_docs_snapshots(baseline: dict[str, dict[str, object]], current: dict[str, dict[str, object]]) -> list[str]:
    reasons: list[str] = []
    expected_paths = {path.as_posix() for path in CONTROLLING_DOCS}
    if set(baseline) != expected_paths:
        reasons.append("baseline_controlling_doc_set_mismatch")
    if set(current) != expected_paths:
        reasons.append("current_controlling_doc_set_mismatch")
    for path in sorted(expected_paths):
        base = baseline.get(path)
        cur = current.get(path)
        if base is None:
            reasons.append(f"added_or_missing_baseline:{path}")
            continue
        if cur is None:
            reasons.append(f"removed_current:{path}")
            continue
        if not bool(cur.get("readable", False)):
            reasons.append(f"unreadable_current:{path}")
        if not bool(base.get("readable", False)):
            reasons.append(f"unreadable_baseline:{path}")
        if str(base.get("sha256", "")) != str(cur.get("sha256", "")):
            reasons.append(f"hash_changed:{path}")
        if int(base.get("byte_count", -1)) != int(cur.get("byte_count", -2)):
            reasons.append(f"byte_count_changed:{path}")
    return reasons


def _controlling_hashes_json(current: dict[str, dict[str, object]]) -> str:
    payload = {path: str(row.get("sha256", "")) for path, row in sorted(current.items())}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _selected_stages(start_stage: str, stop_after_stage: str) -> tuple[str, ...]:
    start = start_stage.upper()
    stop = stop_after_stage.upper() if stop_after_stage else "R11"
    if start not in STAGE_ORDER:
        raise ValueError(f"start_stage must be one of {STAGE_ORDER}")
    if stop not in STAGE_ORDER:
        raise ValueError(f"stop_after_stage must be one of {STAGE_ORDER}")
    start_index = STAGE_ORDER.index(start)
    stop_index = STAGE_ORDER.index(stop)
    if stop_index < start_index:
        raise ValueError("stop_after_stage cannot precede start_stage")
    return STAGE_ORDER[start_index : stop_index + 1]


def _run_folder_name(run_id: int, run_label: str = "") -> str:
    label = str(run_label).strip()
    return label if label else f"{int(run_id):03d}"


def _stage_run_id(config: R5R10PipelineConfig, output_root: Path) -> int:
    if str(config.stage_run_label).strip() and config.run_id is not None:
        return int(config.run_id)
    return _next_run_id(output_root)


def _official_run_label_conflicts(*, config: R5R10PipelineConfig, run_id: int, run_root: Path) -> list[Path]:
    labels_active = bool(str(config.run_label).strip() or str(config.stage_run_label).strip())
    if not labels_active:
        return []
    stage_label = str(config.stage_run_label).strip()
    selected = set(_selected_stages(config.start_stage, config.stop_after_stage))
    targets: list[Path] = [run_root]
    if stage_label:
        if "R5" in selected:
            targets.append(W01_OUTPUT_ROOT / stage_label)
        if "R7" in selected:
            targets.append(W3_OUTPUT_ROOT / stage_label)
        if "R8" in selected:
            targets.extend((POST_W3_OUTPUT_ROOT / stage_label, OUTCOME_OUTPUT_ROOT / stage_label))
        if "R9" in selected:
            targets.append(DEFAULT_REPEATED_OUTPUT_ROOT / stage_label)
        if "R10" in selected:
            targets.append(DEFAULT_CHANGED_OUTPUT_ROOT / stage_label)
        if "R11" in selected:
            targets.append(DEFAULT_HELDOUT_CHANGED_OUTPUT_ROOT / stage_label)
    return [Path(path) for path in targets if filesystem_path(path).exists()]


def _next_run_id(output_root: Path) -> int:
    root = filesystem_path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    ids: list[int] = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        try:
            ids.append(int(path.name))
        except ValueError:
            continue
    return max(ids, default=0) + 1


def _validation_worker_count(workers: int | str) -> int:
    if str(workers).strip().lower() == "auto":
        return 8
    return max(1, int(workers))


def _validation_chunk_size(candidate_chunk_size: int | str) -> int:
    return max(20_000, int(candidate_chunk_size))


def _r9_initial_governor_config_path(context: dict[str, object]) -> Path | None:
    stage = dict(dict(context.get("stages", {})).get("R9", {}))
    run_root = Path(str(stage.get("run_root", "")))
    path = run_root / "manifests" / "initial_governor_config_for_r10.json"
    return path if path.is_file() else None


def _r10_frozen_governor_config_path(context: dict[str, object]) -> Path | None:
    stage = dict(dict(context.get("stages", {})).get("R10", {}))
    run_root = Path(str(stage.get("run_root", "")))
    path = run_root / "manifests" / "frozen_governor_config_for_r11.json"
    return path if path.is_file() else None


def _check_row(stage_id: str, check_id: str, passed: bool, observed: object, required: object) -> dict[str, object]:
    return {
        "timestamp_utc": _utc_now(),
        "stage_id": stage_id,
        "check_id": check_id,
        "passed": bool(passed),
        "observed": _stringify(observed),
        "required": _stringify(required),
    }


def _file_size_check(stage_id: str, root: Path) -> dict[str, object]:
    max_size = 0.0
    path_count = 0
    if root and filesystem_path(root).is_dir():
        for path in filesystem_path(root).rglob("*"):
            if path.is_file():
                path_count += 1
                max_size = max(max_size, float(path.stat().st_size) / float(1024 * 1024))
    return _check_row(stage_id, "no_generated_file_above_100mb", max_size <= MAX_GENERATED_FILE_SIZE_MB, round(max_size, 3), f"<={MAX_GENERATED_FILE_SIZE_MB}")


def _r10_variation_audit_passed(run_root: Path) -> bool:
    frame = _read_csv_if_exists(run_root / "metrics" / "no_glider_latency_variation_audit.csv")
    if frame.empty or "variation_audit_passed" not in frame.columns:
        return False
    return bool(frame["variation_audit_passed"].astype(bool).all())


def _r5_launch_gate_audit_passed(run_root: Path) -> bool:
    frame = _read_csv_if_exists(run_root / "metrics" / "r5_launch_gate_entry_diagnosis.csv")
    if frame.empty:
        frame = _read_csv_if_exists(run_root / "metrics" / "r5_launch_capture_diagnosis.csv")
    if frame.empty:
        return False
    required = {
        "start_state_family",
        "transition_entry_class",
        "primitive_id",
        "entry_role",
        "total_rows",
        "entry_role_rejection_count",
        "accepted_count",
        "weak_count",
        "continuation_valid_count",
        "terminal_useful_count",
        "hard_failure_count",
        "blocked_count",
        "rejected_count",
        "r5_launch_entry_gate_passed",
    }
    if not required.issubset(set(frame.columns)):
        return False
    launch = frame[frame["start_state_family"].astype(str) == "launch_gate"].copy()
    if int(launch["primitive_id"].astype(str).nunique()) < len(ACTIVE_PRIMITIVE_IDS):
        return False
    expected_per_primitive = 32 * 3 * 40
    for primitive_id in ACTIVE_PRIMITIVE_IDS:
        rows = launch[launch["primitive_id"].astype(str) == str(primitive_id)]
        if rows.empty:
            return False
        total_rows = int(pd.to_numeric(rows["total_rows"], errors="coerce").fillna(0).sum())
        if total_rows != expected_per_primitive:
            return False
        if set(rows["transition_entry_class"].astype(str)) != {"launch_gate"}:
            return False
        if int(pd.to_numeric(rows["entry_role_rejection_count"], errors="coerce").fillna(0).sum()) != 0:
            return False
        accepted = int(pd.to_numeric(rows["accepted_count"], errors="coerce").fillna(0).sum())
        weak = int(pd.to_numeric(rows["weak_count"], errors="coerce").fillna(0).sum())
        continuation = int(pd.to_numeric(rows["continuation_valid_count"], errors="coerce").fillna(0).sum())
        terminal = int(pd.to_numeric(rows["terminal_useful_count"], errors="coerce").fillna(0).sum())
        hard_failure = int(pd.to_numeric(rows["hard_failure_count"], errors="coerce").fillna(0).sum())
        blocked = int(pd.to_numeric(rows["blocked_count"], errors="coerce").fillna(0).sum())
        rejected = int(pd.to_numeric(rows["rejected_count"], errors="coerce").fillna(0).sum())
        if accepted + weak + continuation + terminal <= 0:
            return False
        if hard_failure >= total_rows:
            return False
        if blocked + rejected >= total_rows:
            return False
        if not rows["r5_launch_entry_gate_passed"].astype(str).str.lower().isin({"true", "1"}).any():
            return False
    return True


def _r8_launch_gate_audit_passed(run_root: Path) -> bool:
    frame = _read_csv_if_exists(run_root / "metrics" / "launch_gate_candidate_availability.csv")
    if frame.empty:
        return False
    if set(frame.get("library_size_case_id", pd.Series(dtype=str)).astype(str)) != set(LIBRARY_SIZE_CASE_IDS):
        return False
    for row in frame.to_dict(orient="records"):
        if int(float(row.get("launch_gate_entry_primitive_family_count", 0))) < len(ACTIVE_PRIMITIVE_IDS):
            return False
        if _nonempty_csv_cell(row.get("missing_launch_gate_entry_primitive_ids", "")):
            return False
        if int(float(row.get("launch_gate_candidate_rows", 0))) <= 0:
            return False
    return True


def _validation_launch_gate_audit_passed(run_root: Path) -> bool:
    required = (
        "launch_gate_entry_role_audit.csv",
        "launch_gate_outcome_audit.csv",
        "launch_gate_candidate_availability.csv",
        "first_decision_candidate_summary.csv",
        "first_decision_governor_rejection_summary.csv",
    )
    for name in required:
        frame = _read_csv_if_exists(run_root / "metrics" / name)
        if frame.empty:
            return False
    availability = _read_csv_if_exists(run_root / "metrics" / "launch_gate_candidate_availability.csv")
    for row in availability.to_dict(orient="records"):
        if int(float(row.get("launch_gate_entry_primitive_family_count", 0))) < 1:
            return False
    return True


def _nonempty_csv_cell(value: object) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(str(value).strip())


def _validation_gate_checks(stage_id: str, run_root: Path, *, prefix: str) -> list[dict[str, object]]:
    frame = _read_csv_if_exists(run_root / "metrics" / "pass_fail_gate_summary.csv")
    if frame.empty:
        return [_check_row(stage_id, f"{prefix}_summary_present", False, "missing", "metrics/pass_fail_gate_summary.csv")]
    rows: list[dict[str, object]] = []
    for row in frame.to_dict(orient="records"):
        gate_id = str(row.get("gate_id", "unknown_gate"))
        passed = bool(row.get("passed", False))
        rows.append(
            _check_row(
                stage_id,
                f"{prefix}_{gate_id}",
                passed,
                row.get("observed", ""),
                row.get("required", ""),
            )
        )
    return rows


def _blocked_reason_from_checks(rows: list[dict[str, object]]) -> str:
    failed = [str(row["check_id"]) for row in rows if not bool(row.get("passed", False))]
    return "failed_gate:" + ",".join(failed)


def _read_json_if_exists(path: Path) -> dict[str, object]:
    fs_path = filesystem_path(path)
    if not fs_path.is_file():
        return {}
    try:
        return json.loads(fs_path.read_text(encoding="ascii"))
    except Exception:
        return {}


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    fs_path = filesystem_path(path)
    if not fs_path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(fs_path)
    except Exception:
        return pd.DataFrame()


def _read_csv_rows(path: Path) -> list[dict[str, object]]:
    frame = _read_csv_if_exists(path)
    return [] if frame.empty else frame.to_dict(orient="records")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    filesystem_path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(filesystem_path(path), index=False)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    filesystem_path(path).parent.mkdir(parents=True, exist_ok=True)
    filesystem_path(path).write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="ascii")


def _write_text(path: Path, text: str) -> None:
    filesystem_path(path).parent.mkdir(parents=True, exist_ok=True)
    filesystem_path(path).write_text(text, encoding="ascii")


def _write_file_size_audit(root: Path) -> None:
    rows = []
    root_fs = filesystem_path(root)
    if root_fs.is_dir():
        for path in sorted(root_fs.rglob("*")):
            if not path.is_file():
                continue
            size_mb = float(path.stat().st_size) / float(1024 * 1024)
            rows.append(
                {
                    "relative_path": path.relative_to(root_fs).as_posix(),
                    "byte_count": int(path.stat().st_size),
                    "size_mb": size_mb,
                    "above_75mb": bool(size_mb > 75.0),
                    "above_100mb": bool(size_mb > MAX_GENERATED_FILE_SIZE_MB),
                    "push_allowed": bool(size_mb <= MAX_GENERATED_FILE_SIZE_MB),
                }
            )
    _write_csv(root / "metrics" / "file_size_audit.csv", pd.DataFrame(rows))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value: object) -> object:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _stringify(value: object) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_ready(value), sort_keys=True)
    return str(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the full R5-R10 evidence pipeline with repeated docs alignment.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--run-label", default="")
    parser.add_argument("--stage-run-label", default="")
    parser.add_argument("--start-stage", default="R5", choices=STAGE_ORDER)
    parser.add_argument("--stop-after-stage", default="", choices=("", *STAGE_ORDER))
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--repair-incomplete", dest="repair_incomplete", action="store_true", default=True)
    parser.add_argument("--no-repair-incomplete", dest="repair_incomplete", action="store_false")
    parser.add_argument("--dry-run-schedule", action="store_true")
    parser.add_argument("--workers", default=8)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--storage-format", default="auto", choices=("auto", "parquet", "csv_gz", "csv"))
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument("--candidate-chunk-size", type=int, default=800)
    parser.add_argument("--r10-mode", default="full", choices=("full", "reduced_diagnostic_10", "reduced_diagnostic_50"))
    parser.add_argument("--history-log-mode", default="auto", choices=("auto", "plot_summary", "sampled_debug", "full_debug"))
    parser.add_argument("--history-debug-sample-stride", type=int, default=10)
    parser.add_argument("--allow-stage-smoke", type=_parse_bool, default=False)
    parser.add_argument("--continue-on-stage-failure", type=_parse_bool, default=False)
    args = parser.parse_args(argv)
    result = run_r5_r10_pipeline(
        R5R10PipelineConfig(
            output_root=args.output_root,
            run_id=args.run_id,
            run_label=args.run_label,
            stage_run_label=args.stage_run_label,
            start_stage=args.start_stage,
            stop_after_stage=args.stop_after_stage,
            resume=args.resume,
            repair_incomplete=args.repair_incomplete,
            dry_run_schedule=args.dry_run_schedule,
            workers=args.workers,
            max_workers=args.max_workers,
            storage_format=args.storage_format,
            compression_level=args.compression_level,
            candidate_chunk_size=args.candidate_chunk_size,
            r10_mode=args.r10_mode,
            history_log_mode=args.history_log_mode,
            history_debug_sample_stride=args.history_debug_sample_stride,
            allow_stage_smoke=args.allow_stage_smoke,
            continue_on_stage_failure=args.continue_on_stage_failure,
        )
    )
    print(result)
    return 0 if result.get("status") == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
