# Local Validation Environment

Date: 2026-05-24

This repository uses one project-owned virtual environment for active work:

```text
.\.venv\Scripts\python.exe
```

Do not use the old `Paul_Li_FYP` Conda environment for active validation or new development work. `Paul_Li_FYP` is not the active validation environment. Historical audit notes may mention it as a past fallback, but it is not the active project environment.

The current `.venv` was created from the Miniforge base interpreter:

```text
C:\ProgramData\miniforge3\python.exe
```

R6 control validation dependencies are installed from:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-control-dev.txt
```

Whole-repository development should still use the same `.venv`. If design-side code is being run, install the aggregate development dependencies into the same environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Do not create or switch to a second named environment for `02_Glider_Design` or `03_Control`. The dependency files define which packages are needed; the active interpreter stays the same.

Required active validation commands:

```powershell
.\.venv\Scripts\python.exe -m compileall 03_Control
.\.venv\Scripts\python.exe -m pytest -q 03_Control/tests --basetemp .codex_run_logs\pytest_tmp -o cache_dir=.codex_run_logs\pytest_cache
.\.venv\Scripts\python.exe 03_Control/04_Scenarios/run_active_contract_audit.py
git diff --check
```

Use the repo-local pytest temp/cache paths above so validation does not depend on the Windows user temp directory. Local `.venv` and `.codex_run_logs` contents are ignored and must not be committed.

`aerosandbox` is not part of R6 `03_Control` validation. It remains isolated in `requirements-design.txt` for glider-design-side work and is installed only when whole-repository or design-side validation is needed. The active control route requires `casadi`, `pytest`, and `openpyxl` through `requirements-control-dev.txt`.
