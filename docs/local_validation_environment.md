# Local Validation Environment

Date: 2026-05-25

Status: active environment note. The workflow details are controlled by
`docs/Glider_Control_Project_Plan.md`, `docs/Daily_Schedule.txt`,
`docs/Skills.md`, `docs/Python Coding Instruction.txt`,
`docs/MATLAB Coding.txt`, and `docs/housekeeping_and_naming_rules.md`.

This repository uses one project-owned virtual environment for active Python
work:

```text
.\.venv\Scripts\python.exe
```

Do not use the old `Paul_Li_FYP` Conda environment for active validation or new
development work. Historical audit notes may mention it as a past fallback, but
it is not the active project environment.

The current `.venv` was created from the Miniforge base interpreter:

```text
C:\ProgramData\miniforge3\python.exe
```

Active control validation dependencies are installed from:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-control-dev.txt
```

Whole-repository development should still use the same `.venv`. If design-side
code is being run, install the aggregate development dependencies into the same
environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Do not create or switch to a second named environment for `02_Glider_Design` or
`03_Control`. The dependency files define which packages are needed; the active
interpreter stays the same.

Required active validation baseline:

```powershell
.\.venv\Scripts\python.exe -m py_compile 03_Control/02_Inner_Loop/*.py 03_Control/03_Primitives/*.py 03_Control/04_Scenarios/*.py
.\.venv\Scripts\python.exe -m pytest -q 03_Control/tests --basetemp .codex_run_logs\pytest_tmp -o cache_dir=.codex_run_logs\pytest_cache
.\.venv\Scripts\python.exe 03_Control/04_Scenarios/run_v411_source_audit.py --dry-run --no-write-archive
.\.venv\Scripts\python.exe 03_Control/04_Scenarios/run_v53_algorithm_contract_audit.py
git diff --check
```

`run_active_contract_audit.py` and W01/W2/W3-only audit names are retained only
as compatibility references. New instructions should name the active source
audit and the v5.20 algorithm contract audit directly. The current evidence
workflow is controlled by `docs/Glider_Control_Project_Plan.md`: R5 is robust
primitive synthesis, R6/W2 is archived diagnostic-only, R7 is frozen W3
validation, R8 is the five-case coverage-aware medoid post-W3 library-size
study, R9 is an internal reduced fixed-case preflight that writes an initial
governor handoff for R10, R10 is environment-only changed-case governor tuning,
and R11 is strict held-out changed-case validation.

Use the repo-local pytest temp/cache paths above so validation does not depend
on the Windows user temp directory. Local `.venv` and `.codex_run_logs`
contents are ignored and must not be committed. Generated evidence roots must
also follow the 100 MB file-size rule, path-length audit, and local-only result
handling in `docs/housekeeping_and_naming_rules.md`.

`aerosandbox` is not part of active `03_Control` validation. It remains isolated
in `requirements-design.txt` for glider-design-side work and is installed only
when whole-repository or design-side validation is needed. The active control
route requires `casadi`, `pytest`, and `openpyxl` through
`requirements-control-dev.txt`.
