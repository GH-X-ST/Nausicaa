from __future__ import annotations

import csv
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTROL_ROOT = REPO_ROOT / "03_Control"
RESULT_ROOT = CONTROL_ROOT / "05_Results"
ACTIVE_CONTRACT_DOCS = (
    "docs/Glider_Control_Project_Plan.md",
    "docs/Skills.md",
    "docs/Python Coding Instruction.txt",
    "docs/housekeeping_and_naming_rules.md",
    "docs/Daily_Schedule.txt",
)
ALLOWLISTED_ACTIVE_PATHS = {
    ".gitignore",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-control.txt",
    "requirements-control-dev.txt",
    "requirements-design.txt",
    "README.md",
    "LICENSE",
    "docs/local_validation_environment.md",
    "docs/Glider_Control_Project_Plan.md",
    "docs/Skills.md",
    "docs/Python Coding Instruction.txt",
    "docs/housekeeping_and_naming_rules.md",
    "docs/Daily_Schedule.txt",
    "docs/abbr.md",
    "docs/reset_or_next_step_build_note.md",
    "docs/lqr_restart_archive_manifest.md",
    "docs/lqr_foundation_audit.md",
    "docs/model_audit/model_only_foundation_audit.md",
    "docs/model_audit/r6_r8_guidance_alignment_audit.md",
    "docs/code_audits/lqr_plan_alignment_fix_report.md",
    "docs/code_audits/lqr_current_code_audit_fix_v3_4_report.md",
    "docs/code_audits/r6_readiness_validation_closure_report.md",
    "docs/code_audits/r6_validation_start_report.md",
    "docs/code_audits/r6_lqr_tuning_dense_report.md",
    "docs/reset/model_only_reset_manifest.md",
    "docs/reset/kept_file_audit.csv",
    "docs/reset/kept_file_audit.md",
    "docs/reset/model_only_reset_report.md",
    "03_Control/02_Inner_Loop/A_model_parameters/mass_properties_estimate.py",
    "03_Control/02_Inner_Loop/glider.py",
    "03_Control/02_Inner_Loop/flight_dynamics.py",
    "03_Control/02_Inner_Loop/linearisation.py",
    "03_Control/02_Inner_Loop/trim_solver.py",
    "03_Control/03_Primitives/state_contract.py",
    "03_Control/03_Primitives/command_contract.py",
    "03_Control/03_Primitives/metric_contract.py",
    "03_Control/03_Primitives/latency.py",
    "03_Control/03_Primitives/implementation_instance.py",
    "03_Control/03_Primitives/plant_instance.py",
    "03_Control/03_Primitives/wing_wind_descriptors.py",
    "03_Control/03_Primitives/episodic_lift_belief.py",
    "03_Control/03_Primitives/lqr_linearisation.py",
    "03_Control/03_Primitives/lqr_controller.py",
    "03_Control/03_Primitives/lqr_tuning.py",
    "03_Control/03_Primitives/controller_registry.py",
    "03_Control/03_Primitives/evidence_status.py",
    "03_Control/03_Primitives/primitive_evidence_schema.py",
    "03_Control/03_Primitives/prim_cat.py",
    "03_Control/03_Primitives/prim_ctrl.py",
    "03_Control/03_Primitives/prim_features.py",
    "03_Control/03_Primitives/prim_roll.py",
    "03_Control/03_Primitives/prim_model.py",
    "03_Control/03_Primitives/prim_select.py",
    "03_Control/03_Primitives/dense_archive_runtime.py",
    "03_Control/03_Primitives/dense_archive_table_io.py",
    "03_Control/03_Primitives/dense_archive_chunking.py",
    "03_Control/04_Scenarios/arena.py",
    "03_Control/04_Scenarios/arena_contract.py",
    "03_Control/04_Scenarios/archive_table_reader.py",
    "03_Control/04_Scenarios/evidence_stage_utils.py",
    "03_Control/04_Scenarios/scenario_contract.py",
    "03_Control/04_Scenarios/updraft_models.py",
    "03_Control/04_Scenarios/env_ctx.py",
    "03_Control/04_Scenarios/env_instance.py",
    "03_Control/04_Scenarios/env_surrogate.py",
    "03_Control/04_Scenarios/run_active_contract_audit.py",
    "03_Control/04_Scenarios/run_ctx_archive.py",
    "03_Control/04_Scenarios/run_ctx_episode_smoke.py",
    "03_Control/04_Scenarios/run_lqr_contextual_archive.py",
    "03_Control/04_Scenarios/run_lqr_tuning_sweep.py",
    "03_Control/04_Scenarios/run_lqr_w2_replay.py",
    "03_Control/04_Scenarios/run_lqr_w3_generalisation.py",
    "03_Control/04_Scenarios/run_primitive_selector_report.py",
    "03_Control/04_Scenarios/run_w2_replay.py",
    "03_Control/04_Scenarios/run_w3_generalisation.py",
    "03_Control/04_Scenarios/state_sampling.py",
}
ALLOWLISTED_SUPPORT_DOCS = {
    "docs/MATLAB Coding.txt",
    "docs/Python Plotting Guidance.txt",
}
ALLOWLISTED_TESTS = {
    "03_Control/tests/conftest.py",
    "03_Control/tests/test_model_foundation_smoke.py",
    "03_Control/tests/test_arena_bounds.py",
    "03_Control/tests/test_control_contract_arena.py",
    "03_Control/tests/test_control_contract_scenario_paths.py",
    "03_Control/tests/test_control_contract_state_command.py",
    "03_Control/tests/test_control_signs.py",
    "03_Control/tests/test_dense_archive_chunking.py",
    "03_Control/tests/test_dense_archive_runtime.py",
    "03_Control/tests/test_dense_archive_table_io.py",
    "03_Control/tests/test_implementation_plant_instances.py",
    "03_Control/tests/test_latency_acceptance_labels.py",
    "03_Control/tests/test_latency_chain.py",
    "03_Control/tests/test_latency_step_response.py",
    "03_Control/tests/test_linearisation_finite_difference.py",
    "03_Control/tests/test_metric_contract.py",
    "03_Control/tests/test_panel_cg_uniform_wind.py",
    "03_Control/tests/test_repo_housekeeping.py",
    "03_Control/tests/test_servo_command_limit.py",
    "03_Control/tests/test_state_order.py",
    "03_Control/tests/test_surface_limits.py",
    "03_Control/tests/test_surface_quantisation.py",
    "03_Control/tests/test_trim_residual.py",
    "03_Control/tests/test_updraft_model_reproduction.py",
    "03_Control/tests/test_updraft_model_shapes.py",
    "03_Control/tests/test_updraft_randomisation.py",
    "03_Control/tests/test_wing_wind_descriptors.py",
    "03_Control/tests/test_env_ctx.py",
    "03_Control/tests/test_env_instance.py",
    "03_Control/tests/test_surrogate_ladder.py",
    "03_Control/tests/test_state_sampling.py",
    "03_Control/tests/test_prim_cat.py",
    "03_Control/tests/test_lqr_controller.py",
    "03_Control/tests/test_lqr_tuning.py",
    "03_Control/tests/test_prim_ctrl.py",
    "03_Control/tests/test_prim_features.py",
    "03_Control/tests/test_prim_roll.py",
    "03_Control/tests/test_prim_roll_model_backed.py",
    "03_Control/tests/test_prim_model.py",
    "03_Control/tests/test_prim_select.py",
    "03_Control/tests/test_ctx_archive_smoke.py",
    "03_Control/tests/test_ctx_episode_smoke.py",
    "03_Control/tests/test_episodic_lift_belief.py",
    "03_Control/tests/test_guidance_alignment.py",
    "03_Control/tests/test_report_scaffolds.py",
    "03_Control/tests/test_active_contract_audit.py",
    "03_Control/tests/test_lqr_contextual_archive_dry_run.py",
    "03_Control/tests/test_lqr_tuning_dry_run_contract.py",
}
ALLOWLISTED_DATA = {
    "02_Glider_Design/C_results/nausicaa_results.csv",
    "01_Thermal/S01.xlsx",
    "01_Thermal/S02.xlsx",
    "01_Thermal/B_results/single_var_params.xlsx",
    "01_Thermal/B_results/four_var_params.xlsx",
    "01_Thermal/B_results/Single_Fan_Annular_GP/single_annular_gp_grid_predictions.xlsx",
    "01_Thermal/B_results/Four_Fan_Annular_GP/four_annular_gp_grid_predictions.xlsx",
}
FORBIDDEN_ACTIVE_PATTERNS = tuple(
    "".join(parts)
    for parts in (
        ("fixed", "_gate"),
        ("paired", "_w0", "_w1"),
        ("reachable", "_state"),
        ("reachable", " chain"),
        ("branch", "-specific"),
        ("single", "_fan", "_branch"),
        ("four", "_fan", "_branch"),
        ("med", "oid_package"),
        ("governor", "_package"),
        ("primitive", "_library"),
        ("old ", "Codex"),
        ("old ", "CODEX"),
        ("03_Control", "/05_Results/", "0"),
        ("03_Control", "/05_Results/", "1"),
    )
)
PROJECT_PLAN_BOUNDARY_DOCS = {"docs/Glider_Control_Project_Plan.md"}
NEGATIVE_BOUNDARY_MARKERS = (
    "no longer",
    "do not",
    "must not",
    "should not",
    "is not",
    "not:",
    "avoid",
    "reject",
    "discarded implementation",
    "historical work",
)


def _repo_files() -> set[str]:
    skip_parts = {".git", ".pytest_cache", "__pycache__"}
    files: set[str] = set()
    for path in REPO_ROOT.rglob("*"):
        if any(part in skip_parts for part in path.parts):
            continue
        if path.is_file():
            files.add(path.relative_to(REPO_ROOT).as_posix())
    return files


def _allowed_local_evidence_root() -> str:
    value = os.environ.get("NAUSICAA_ALLOW_LOCAL_EVIDENCE_ROOT", "").strip()
    if not value:
        return ""
    path = Path(value)
    if path.is_absolute():
        try:
            path = path.resolve().relative_to(REPO_ROOT)
        except ValueError:
            return ""
    return path.as_posix().rstrip("/")


def _audit_rows() -> list[dict[str, str]]:
    path = REPO_ROOT / "docs" / "reset" / "kept_file_audit.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _forbidden_trace_offenders(rel_path: str, text: str) -> list[str]:
    lines = text.splitlines()
    offenders: list[str] = []
    for index, line in enumerate(lines):
        for pattern in FORBIDDEN_ACTIVE_PATTERNS:
            if pattern not in line:
                continue
            if _is_allowed_project_plan_boundary(rel_path, lines, index):
                continue
            offenders.append(f"{rel_path}: {pattern}")
    return offenders


def _is_allowed_project_plan_boundary(
    rel_path: str,
    lines: list[str],
    index: int,
) -> bool:
    if rel_path not in PROJECT_PLAN_BOUNDARY_DOCS:
        return False
    lower_context = " ".join(lines[max(0, index - 8) : index + 2]).lower()
    return any(marker in lower_context for marker in NEGATIVE_BOUNDARY_MARKERS)


def test_active_contract_documents_are_present_unignored_and_uncontaminated() -> None:
    for path in ACTIVE_CONTRACT_DOCS:
        assert (REPO_ROOT / path).is_file()
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", "--", path],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert ignored.returncode != 0
        text = (REPO_ROOT / path).read_text(encoding="utf-8")
        assert _forbidden_trace_offenders(path, text) == []


def test_kept_file_audit_has_required_active_contract_docs() -> None:
    rows = _audit_rows()
    by_path = {row["path"]: row for row in rows}
    for path in ACTIVE_CONTRACT_DOCS:
        assert by_path[path]["category"] == "active_contract_doc"
        assert by_path[path]["old_method_token_status"] in {
            "none",
            "boundary_prohibition_context_only",
        }
    assert not [
        row
        for row in rows
        if row["old_method_token_status"] == "must_remove_before_reset_passes"
    ]


def test_active_import_paths_do_not_include_results() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    conftest = (CONTROL_ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    assert "03_Control/05_Results" not in pyproject
    assert "05_Results" not in conftest


def test_r6_control_dependencies_do_not_require_design_side_aerosandbox() -> None:
    control = (REPO_ROOT / "requirements-control.txt").read_text(encoding="utf-8")
    control_dev = (REPO_ROOT / "requirements-control-dev.txt").read_text(encoding="utf-8")
    design = (REPO_ROOT / "requirements-design.txt").read_text(encoding="utf-8")
    root = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    dev = (REPO_ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "casadi" in control
    assert "openpyxl" in control
    assert "pytest" not in control
    assert "aerosandbox" not in control
    assert "-r requirements-control.txt" in control_dev
    assert "pytest>=7.0" in control_dev
    assert "aerosandbox" not in control_dev
    assert "aerosandbox>=4.2,<5" in design
    assert "-r requirements-control.txt" in root
    assert "-r requirements-design.txt" in root
    assert "-r requirements-control-dev.txt" in dev
    assert "-r requirements-design.txt" in dev
    runtime_dependencies = pyproject.split("[project.optional-dependencies]", 1)[0]
    assert "pytest" not in runtime_dependencies
    assert "[project.optional-dependencies]" in pyproject
    assert "control-dev = [" in pyproject
    assert "design = [" in pyproject


def test_active_control_imports_do_not_use_aerosandbox_and_do_require_casadi() -> None:
    active_roots = (
        CONTROL_ROOT / "02_Inner_Loop",
        CONTROL_ROOT / "03_Primitives",
        CONTROL_ROOT / "04_Scenarios",
    )
    aerosandbox_imports: list[str] = []
    casadi_imports: list[str] = []
    for root in active_roots:
        for path in root.rglob("*.py"):
            rel_path = path.relative_to(REPO_ROOT).as_posix()
            text = path.read_text(encoding="utf-8")
            if "import aerosandbox" in text or "from aerosandbox" in text:
                aerosandbox_imports.append(rel_path)
            if "import casadi" in text or "from casadi" in text:
                casadi_imports.append(rel_path)
    assert aerosandbox_imports == []
    assert sorted(casadi_imports) == [
        "03_Control/02_Inner_Loop/flight_dynamics.py",
        "03_Control/02_Inner_Loop/linearisation.py",
        "03_Control/02_Inner_Loop/trim_solver.py",
    ]


def test_r6_validation_docs_use_control_dev_dependency_route() -> None:
    active_environment_docs = (
        "docs/housekeeping_and_naming_rules.md",
        "docs/local_validation_environment.md",
    )
    for rel_path in active_environment_docs:
        text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        assert "requirements-control-dev.txt" in text
        assert "requirements-dev.txt" in text
        assert "same `.venv`" in text

    r6_docs = (
        "docs/code_audits/r6_readiness_validation_closure_report.md",
        "docs/code_audits/r6_validation_start_report.md",
    )
    for rel_path in r6_docs:
        text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        assert "requirements-control-dev.txt" in text
        if "pip install -r requirements-dev.txt" in text:
            assert "Whole-repository" in text or "whole-repository" in text
            assert "same `.venv`" in text
    report = (REPO_ROOT / "docs/code_audits/r6_validation_start_report.md").read_text(
        encoding="utf-8"
    )
    assert "aerosandbox` is not required for R6 `03_Control` validation" in report


def test_active_validation_environment_is_repo_local_venv() -> None:
    active_docs = (
        "docs/housekeeping_and_naming_rules.md",
        "docs/local_validation_environment.md",
        "docs/Skills.md",
        "docs/Python Coding Instruction.txt",
        "docs/code_audits/r6_validation_start_report.md",
    )
    for rel_path in active_docs:
        text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        normalized = " ".join(text.split())
        assert ".venv\\Scripts\\python.exe" in text
        assert "Paul_Li_FYP" not in text or "not the active validation environment" in normalized
        if rel_path != "docs/code_audits/r6_validation_start_report.md":
            assert "requirements-control-dev.txt" in text
    env_doc = (REPO_ROOT / "docs/local_validation_environment.md").read_text(encoding="utf-8")
    assert "requirements-dev.txt" in env_doc
    assert "same `.venv`" in env_doc
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".venv/" in gitignore
    assert ".codex_run_logs/" in gitignore


def test_results_root_contains_only_gitkeep() -> None:
    entries = sorted(path.relative_to(RESULT_ROOT).as_posix() for path in RESULT_ROOT.rglob("*"))
    allowed_root = _allowed_local_evidence_root()
    if not allowed_root:
        assert entries == [".gitkeep"]
        return
    allowed_prefix = Path(allowed_root).relative_to("03_Control/05_Results").as_posix()
    allowed_parents = {
        Path(allowed_prefix).parents[index].as_posix()
        for index in range(len(Path(allowed_prefix).parents))
        if Path(allowed_prefix).parents[index].as_posix() != "."
    }
    assert all(
        entry == ".gitkeep"
        or entry == allowed_prefix
        or entry in allowed_parents
        or entry.startswith(f"{allowed_prefix}/")
        for entry in entries
    )


def test_active_files_are_allowlisted() -> None:
    files = _repo_files()
    allowed = (
        ALLOWLISTED_ACTIVE_PATHS
        | ALLOWLISTED_SUPPORT_DOCS
        | ALLOWLISTED_TESTS
        | ALLOWLISTED_DATA
        | {"03_Control/05_Results/.gitkeep"}
    )
    active_prefixes = (
        "03_Control/02_Inner_Loop/",
        "03_Control/03_Primitives/",
        "03_Control/04_Scenarios/",
        "03_Control/tests/",
        "03_Control/05_Results/",
        "docs/",
    )
    unexpected = sorted(
        path
        for path in files
        if path.startswith(active_prefixes)
        and path not in allowed
        and not path.startswith("docs/reset/")
        and not (
            _allowed_local_evidence_root()
            and path.startswith(f"{_allowed_local_evidence_root()}/")
        )
    )
    assert unexpected == []


def test_retained_active_files_do_not_contain_forbidden_old_method_traces() -> None:
    paths = sorted(ALLOWLISTED_ACTIVE_PATHS | ALLOWLISTED_TESTS)
    offenders: list[str] = []
    for rel_path in paths:
        path = REPO_ROOT / rel_path
        if not path.exists() or rel_path.startswith("docs/reset/"):
            continue
        text = path.read_text(encoding="utf-8")
        offenders.extend(_forbidden_trace_offenders(rel_path, text))
    assert offenders == []


def test_dense_runtime_storage_contract_is_explicit() -> None:
    text = (CONTROL_ROOT / "03_Primitives" / "dense_archive_runtime.py").read_text(
        encoding="utf-8"
    )
    for token in (
        "chunked",
        "resumable",
        "compressed",
        "worker-enabled",
        "checksum-manifested",
        "MAX_GENERATED_FILE_SIZE_MB = 100.0",
    ):
        assert token in text


def test_no_tracked_nonapproved_file_above_100_mb() -> None:
    oversized = []
    for rel_path in _repo_files():
        path = REPO_ROOT / rel_path
        if path.is_file() and path.stat().st_size > 100 * 1024 * 1024:
            oversized.append(rel_path)
    assert oversized == []


def test_active_paths_do_not_group_or_name_by_raw_position_bins() -> None:
    paths = [
        "03_Control/03_Primitives/prim_model.py",
        "03_Control/03_Primitives/prim_select.py",
        "03_Control/04_Scenarios/run_ctx_archive.py",
        "03_Control/04_Scenarios/run_primitive_selector_report.py",
        "03_Control/04_Scenarios/run_w2_replay.py",
        "03_Control/04_Scenarios/run_w3_generalisation.py",
    ]
    forbidden = (
        "groupby([\"initial_x_w\"",
        "groupby(['initial_x_w'",
        "groupby([\"x_w\"",
        "groupby(['x_w'",
        "x_w_bin",
        "y_w_bin",
        "z_w_bin",
        "arena_coordinate_bin",
        "position_branch",
        "coordinate_branch",
        "primitive_id_from_position",
        "evidence_group_x_w",
    )
    offenders = []
    for rel_path in paths:
        text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        offenders.extend(
            f"{rel_path}: {token}" for token in forbidden if token in text
        )
    assert offenders == []
