# R6 Validation Start and Dependency Hygiene Report

Date: 2026-05-24

## Result

- R6 W0/W1 tuning is not cleared to start in this shell.
- No LQR method code was changed in this pass because the remaining blocker is the execution environment, not an observed controller-code failure.
- `03_Control/05_Results` remains clean for active evidence; only `.gitkeep` is present.

## Interpreter and Dependencies

- Interpreter used: `C:\Users\GH-X-ST\.conda\envs\Paul_Li_FYP\python.exe`.
- Shell `python.exe` and `python3.exe` still resolve to WindowsApps shims and were not used for validation.
- `.venv` was not present in the repository, and `conda` was not discoverable on `PATH`.
- Dependency file used for R6 control validation: `requirements-control-dev.txt`.
- Dependency install command:
  - `C:\Users\GH-X-ST\.conda\envs\Paul_Li_FYP\python.exe -m pip install -r requirements-control-dev.txt` - blocked.
  - pip defaulted to user installation because normal site-packages is not writeable; installed control packages already present were reused, then network/socket permission failure `WinError 10013` prevented resolving `casadi`.
- Dependency import check for the active R6 control validation route:
  - present: `numpy`, `scipy`, `pandas`, `matplotlib`.
  - missing direct R6 blockers: `pytest`, `casadi`, `openpyxl`.
  - `aerosandbox` is not required for R6 `03_Control` validation; it is a design-side dependency for `02_Glider_Design` and is isolated in `requirements-design.txt`.
  - `casadi` is not installed in this interpreter; it is currently required by the active `03_Control` trim, linearisation, and flight-dynamics path.

## Validation Commands

- `C:\Users\GH-X-ST\.conda\envs\Paul_Li_FYP\python.exe -m compileall 03_Control` - passed.
- `C:\Users\GH-X-ST\.conda\envs\Paul_Li_FYP\python.exe -m pytest -q 03_Control/tests` - blocked: `No module named pytest`.
- `C:\Users\GH-X-ST\.conda\envs\Paul_Li_FYP\python.exe 03_Control/04_Scenarios/run_active_contract_audit.py` - failed honestly with one environment finding: `ModuleNotFoundError: No module named 'casadi'`.
- `git diff --check` - passed with line-ending warnings only.

## R6 Smoke Status

- Bounded smoke command remains blocked before output creation because `03_Control/02_Inner_Loop/linearisation.py` imports `casadi`, which is not installed.
- Previous temporary output root requested: `C:\Users\GH-X-ST\AppData\Local\Temp\nausicaa_r6_validation_start_902`.
- Resume and repair behavior were not exercised because no tuning chunk was created.
- No generated smoke output was kept or tracked.

## File-Size and Path Audit

- No R6 smoke table, chunk manifest, or registry output was generated in this pass.
- `03_Control/05_Results` was checked and still contains only `.gitkeep`.
- No generated file-size/path audit could be produced before the `casadi` import blocker.

## Move-On Status

- R6 is blocked by missing control dependencies in the available real interpreter, specifically `pytest`, `casadi`, and `openpyxl`.
- After installing `requirements-control-dev.txt` into a real interpreter, rerun compileall, pytest, active-contract audit, `git diff --check`, and the bounded R6 smoke/resume/repair sequence before starting scheduled W0/W1 tuning evidence.

## Claim Boundary

No dense thesis archive, hardware-readiness, real-flight transfer, W2/W3 robustness, mission-success, PD/PID fallback, TVLQR, MPC, LQR-tree, reachable-chain, online fan-layout branch, or non-LQR controller claim is made by this pass.
