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
$files = Get-ChildItem -Path 03_Control/02_Inner_Loop,03_Control/03_Primitives,03_Control/04_Scenarios -Filter *.py -File | ForEach-Object { $_.FullName }
.\.venv\Scripts\python.exe -m py_compile @files
.\.venv\Scripts\python.exe -m pytest -q 03_Control/tests --basetemp .codex_run_logs\pytest_tmp -o cache_dir=.codex_run_logs\pytest_cache
.\.venv\Scripts\python.exe 03_Control/04_Scenarios/run_v411_source_audit.py --dry-run --no-write-archive
.\.venv\Scripts\python.exe 03_Control/04_Scenarios/run_v53_algorithm_contract_audit.py
git diff --check
```

The default pytest command is now the fast regression tier. Slow
pipeline/archive integration tests are marked `slow` and skipped unless
explicitly requested. Run them only before dense evidence regeneration or when
touching archive/replay orchestration:

```powershell
.\.venv\Scripts\python.exe -m pytest -q 03_Control/tests --run-slow -m slow --basetemp .codex_run_logs\pytest_tmp -o cache_dir=.codex_run_logs\pytest_cache
```

`run_active_contract_audit.py` and W01/W2/W3-only audit names are retained only
as compatibility references. New instructions should name the active source
audit and the current transition-aware algorithm contract audit directly. The current evidence
workflow is controlled by `docs/Glider_Control_Project_Plan.md`: R5 is robust
primitive synthesis, R6/W2 is archived diagnostic-only, R7 is frozen W3
validation, R8 is the five-case coverage-aware medoid post-W3 library-size
study, R9 is internal quick fixed-case preflight/ablation only and is not
thesis-facing evidence, R10 is single-block full-domain arena-wide governor
tuning, and R11 is strict held-out eight-block fidelity-ladder validation. R9 defaults to all five
library-size cases, no-updraft/single-fan/four-fan fixed blocks, no-memory plus
h3/h10/h30 recency-weighted candidate-path residual memory, 60 final held-out launches, and 645 history launches.
The R9/R10/R11 governor uses the same candidate-path geometry for no-memory and
memory policies: forward progress to `x_w = 6.6 m`, front-wall terminal proxy,
progress-gated terminal total specific-energy proxy, wrong-boundary avoidance,
and then updraft/lift plus optional residual-memory correction after unchanged
safety and transition-entry filters.
R9/R10/R11 final launch scoring uses the current front-wall mission score:
front-wall terminal success at `x_w = 6.6 m` with y/z inside the true safe
bounds is the main success component, updraft-gain and lift-dwell terms remain
capped lift-usefulness evidence, terminal total specific-energy reserve is
rewarded only after front-wall success, and airborne time/net/gross energy drift
remain audit-only fields.
R9/R10/R11 also write a repeated-launch real-time scheduler profile: context
construction, memory query, and compact-library selection are measured against
a preferred 20 ms controller-slot budget and a hard 0.100 s primitive-boundary
budget, with next decisions prepared before primitive-boundary commit where
possible. This is an offline wall-clock audit, not a hardware real-time claim.

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
