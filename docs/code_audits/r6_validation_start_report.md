# R6 Validation Start Report

Date: 2026-05-23

## Result

- R6 W0/W1 tuning is not cleared to start in this shell.
- No project code was changed in this pass because the first blocker is the execution environment, not an observed code failure.
- `03_Control/05_Results` remains clean for active evidence; only `.gitkeep` is present.

## Interpreter and Dependencies

- Interpreter used: `C:\Users\GH-X-ST\.conda\envs\Paul_Li_FYP\python.exe`.
- Shell `python.exe` and `python3.exe` still resolve to WindowsApps shims and were not used for validation.
- `.venv` was not present in the repository, and `conda` was not discoverable on `PATH`.
- Dependency install command:
  - `C:\Users\GH-X-ST\.conda\envs\Paul_Li_FYP\python.exe -m pip install -r requirements-dev.txt` - blocked.
  - pip defaulted to user installation because the normal site-packages directory is not writable, then failed to reach PyPI with `WinError 10013`; `aerosandbox` could not be resolved.
- Dependency import check:
  - present: `numpy`, `scipy`, `pandas`, `matplotlib`.
  - missing: `pytest`, `casadi`, `aerosandbox`, `openpyxl`.

## Validation Commands

- `C:\Users\GH-X-ST\.conda\envs\Paul_Li_FYP\python.exe -m compileall 03_Control` - passed.
- `C:\Users\GH-X-ST\.conda\envs\Paul_Li_FYP\python.exe -m pytest -q 03_Control/tests` - blocked: `No module named pytest`.
- `C:\Users\GH-X-ST\.conda\envs\Paul_Li_FYP\python.exe 03_Control/04_Scenarios/run_active_contract_audit.py` - failed honestly with one environment finding: `ModuleNotFoundError: No module named 'casadi'`.
- `git diff --check` - passed.

## R6 Smoke Status

- Bounded smoke command was attempted with run id `902`, eight rows, one candidate, one paired test, serial workers, `csv_gz`, and `--resume`.
- Temporary output root requested: `C:\Users\GH-X-ST\AppData\Local\Temp\nausicaa_r6_validation_start_902`.
- The command stopped before output creation because `03_Control/02_Inner_Loop/linearisation.py` imports `casadi`, which is not installed.
- Resume and repair behavior were not exercised because no tuning chunk was created.
- No generated smoke output was kept or tracked.

## Move-On Status

- R6 is blocked by missing project dependencies in the available real interpreter.
- After installing dependencies into a real interpreter, rerun compileall, pytest, active-contract audit, `git diff --check`, and the bounded R6 smoke/resume/repair sequence before starting scheduled W0/W1 tuning evidence.

## Claim Boundary

No dense thesis archive, hardware-readiness, real-flight transfer, W2/W3 robustness, mission-success, PD/PID fallback, TVLQR, MPC, LQR-tree, reachable-chain, online fan-layout branch, or non-LQR controller claim is made by this pass.
