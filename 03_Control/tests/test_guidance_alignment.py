from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


def test_strict_surrogate_and_boundary_semantics_are_present_in_code() -> None:
    env_surrogate = _read("03_Control/04_Scenarios/env_surrogate.py")
    prim_roll = _read("03_Control/03_Primitives/prim_roll.py")
    prim_model = _read("03_Control/03_Primitives/prim_model.py")
    prim_select = _read("03_Control/03_Primitives/prim_select.py")
    run_ctx_archive = _read("03_Control/04_Scenarios/run_ctx_archive.py")

    for token in (
        "W0_requires_dry_air_zero_wind",
        "W1_requires_gaussian_plume_surrogate",
        "W2_requires_gp_corrected_annular_gaussian_surrogate",
        "W3_requires_randomised_gp_corrected_annular_gaussian_surrogate",
        "blocked_no_fallback",
    ):
        assert token in env_surrogate
    for token in (
        "lqr_rollout_candidate",
        "blocked_lqr_synthesis",
        "episode_terminal_useful",
        "continuation_valid",
        "not_continuation_valid",
        "terminal_useful",
    ):
        assert token in prim_roll
    assert "DEFAULT_TRAINING_EVIDENCE_ROLES = (\"lqr_rollout_candidate\",)" in prim_model
    assert "blocked_lqr_synthesis" in prim_model
    assert "continuation" in prim_select
    assert "terminal_episode" in prim_select
    assert "model_backed_lqr" in run_ctx_archive
    assert "process_pool" in run_ctx_archive


def test_online_feedback_and_selector_do_not_branch_on_environment_identity() -> None:
    text = (
        _read("03_Control/03_Primitives/prim_ctrl.py")
        + "\n"
        + _read("03_Control/03_Primitives/prim_select.py")
    ).lower()
    forbidden = (
        "fan_count",
        "w_layer",
        "surrogate_family",
        "environment_id",
        "fixed" + "_gate",
        "reach" + "able",
        "med" + "oid",
    )
    assert [token for token in forbidden if token in text] == []


def test_preserved_guidance_uses_current_method_language() -> None:
    guidance = "\n".join(
        _read(path)
        for path in (
            "docs/Daily_Schedule.txt",
            "docs/Python Coding Instruction.txt",
            "docs/Python Plotting Guidance.txt",
            "docs/MATLAB Coding.txt",
            "docs/housekeeping_and_naming_rules.md",
        )
    )
    for token in (
        "W0 is dry air only",
        "W1 is Gaussian plume only",
        "W2 is GP-corrected annular-Gaussian only",
        "W3 is randomised GP-corrected annular-Gaussian only",
        "W labels are validation layers",
        "environment instances",
        "boundary_terminal",
        "continuation-valid",
        "episode-terminal-useful",
        "04_context_archive/01_r6_lqr_w1_gaussian/001",
        "workers = 8",
        "compressed",
        "resume",
        "checksum",
        "100 MB",
    ):
        assert token in guidance
    assert "Fixed-gate and repeated-launch support" not in guidance


def test_r6_r8_alignment_report_records_statuses() -> None:
    report = _read("docs/model_audit/r6_r8_guidance_alignment_audit.md")
    for token in (
        "fixed",
        "intentionally_preserved_non_contract_support",
        "episode_terminal_useful",
        "lqr_rollout_candidate",
        "100 MB",
    ):
        assert token in report


def test_v3_2_evidence_status_enums_are_active_contract_fields() -> None:
    status = _read("03_Control/03_Primitives/evidence_status.py")
    archive_reader = _read("03_Control/04_Scenarios/archive_table_reader.py")
    run_ctx_archive = _read("03_Control/04_Scenarios/run_ctx_archive.py")
    run_lqr_archive = _read("03_Control/04_Scenarios/run_lqr_contextual_archive.py")
    for token in (
        "complete",
        "accepted_fallback",
        "smoke_incomplete",
        "blocked",
        "retired_not_active",
        "simulation_only_registry_complete",
        "simulation_only_registry_accepted_fallback",
        "simulation_only_smoke_incomplete",
        "simulation_only_blocked",
    ):
        assert token in status
    for token in (
        "registry_backed_row_count",
        "missing_controller_row_count",
        "missing_controller_ratio",
        "archive_evidence_status",
        "evidence_eligibility_reason",
        "blocked_missing_candidate_metadata",
        "blocked_retired_source",
    ):
        assert token in archive_reader
    assert "--selected-controller-registry" in run_ctx_archive
    assert 'default=Path("03_Control/05_Results/lqr_contextual_v1_0/r6")' in run_lqr_archive
    assert "lqr_contextual_v1_0/r6_lqr_" not in run_ctx_archive
