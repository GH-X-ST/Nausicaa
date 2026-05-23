# LQR Plan Alignment Fix Report

Date: 2026-05-23

## Fixed mismatches

- Re-centred source priority on `docs/Glider_Control_Project_Plan.md`, with the uploaded 2026-05-23 guidance treated as implementation guidance under that active contract.
- Added canonical rollout evidence fields `continuation_valid`, `episode_terminal_useful`, and canonical `boundary_use_class` values.
- Removed `boundary_terminal` as an active `outcome_class`; finite x-y boundary exits are retained as terminal-use evidence when justified and are never continuation-valid.
- Made blocked LQR controllers non-executable and routed blocked controller/surrogate/state cases to blocked evidence rows.
- Added selected-controller registry plumbing so W0/W1 selects active controller IDs and W2/W3 only verify and replay those IDs.
- Updated selector, model, replay, and report paths to keep continuation-valid and episode-terminal-useful targets separate.
- Removed the selector-report regression metric that compared speed against continuation margin.

## Changed areas

- Primitive evidence schema, rollout, LQR command handling, controller registry, primitive model, and selector.
- W0/W1 tuning, contextual archive, W2 replay, W3 generalisation, selector report, environment randomisation, and state sampling provenance.
- Tests and housekeeping allowlists covering the updated schema and claim boundaries.

## Validation status

- `C:\Program Files\Blender Foundation\Blender 4.4\4.4\python\bin\python.exe -m compileall 03_Control` passed.
- `C:\Program Files\Blender Foundation\Blender 4.4\4.4\python\bin\python.exe -m pytest -q 03_Control/tests` was blocked because that interpreter does not have `pytest` installed. The normal Windows `python.exe` and `python3.exe` commands resolve to WindowsApps launcher stubs in this shell.
- No smoke evidence run was executed in this pass.

## Remaining approximate or blocked items

- W3 active fan subset and per-fan power are labelled exact only when directly represented; otherwise rows carry approximate component status.
- Rows without selected-controller registry identity are blocked or labelled nominal/unselected smoke rather than promoted as tuned evidence.

## Claim boundary

No dense thesis archive, hardware-readiness claim, real-flight transfer claim, W2/W3 robustness claim, environment-generalisation claim, mission-success claim, PD/PID fallback, TVLQR/MPC/LQR-tree method, reachable-chain construction, or online fan-layout branch was introduced.

## v3.2 Post-Fix Integration Addendum

Date: 2026-05-23

### Fixed mismatches

- Added enum-like registry/archive evidence status fields: `registry_status`, `registry_claim_status`, `archive_evidence_status`, and `evidence_eligibility_reason`.
- Made selected-controller records carry candidate labels, candidate indices, Q/R JSON, gain checksums, linearisation IDs, registry paths, and registry claim status.
- Tightened archive eligibility so rows, model labels, or selected controller IDs alone cannot make R6/W2/W3/selector/report outputs thesis-eligible.
- Preserved W0/W1 as the single source of active `controller_id` evidence; W2/W3 verify and replay registry-backed controllers without retuning or promotion.
- Marked missing-controller rows non-executable and blocked, and made retired `03_Control/99_Archive` sources `retired_not_active` by default.
- Replaced coarse W3 randomisation status with environment, implementation, and plant component labels using `exact`, `approximate`, `metadata_only`, or `blocked`.

### Changed areas

- Controller registry interfaces, rollout metadata, W0/W1 tuning status assignment, R6 archive command pass-through, archive table source metadata, selector report source status, W2 replay provenance, W3 replay provenance/randomisation labels, tests, and housekeeping allowlists.

### Validation status

- `C:\Program Files\Blender Foundation\Blender 4.4\4.4\python\bin\python.exe -m compileall 03_Control` passed during this pass.
- `python -m pytest -q 03_Control/tests` was attempted and blocked by the WindowsApps launcher (`specified logon session does not exist`).
- `C:\Users\GH-X-ST\.conda\envs\Paul_Li_FYP\python.exe -m pytest -q 03_Control/tests` was attempted and blocked because that environment lacks `pytest`; runtime smoke import also showed it lacks `casadi`.
- `git diff --check` passed, and active forbidden-method scans found no TVLQR/MPC/LQR-tree/reachable-chain/online fan-layout branch tokens in active code.
- No dense archive, thesis-scale run, or new evidence campaign was executed.

### Remaining approximate or blocked items

- Smoke and incomplete fallback outputs remain readable for debugging but are labelled `smoke_incomplete` or `blocked` and excluded from active thesis claims.
- Four-fan/per-fan power, aileron asymmetry, CG offset, and cross-inertia effects remain approximate, metadata-only, or blocked where the source implementation does not apply them exactly.

### Preserved claim boundary

No hardware-readiness, real-flight transfer, W3 robustness, mission-success, dense-archive, PD/PID fallback, TVLQR/MPC/LQR-tree, reachable-chain, online fan-layout branch, or claim beyond `simulation_only` was introduced by this integration pass.

## Current-Code Plan Audit Addendum

Date: 2026-05-23

### Fixed current-code deviations

- Tightened W3 replay source selection so W3 consumes only W2 rows with `controller_selection_status = W2_verified_registry_replay`; direct R6/W1 registry-backed rows now block instead of bypassing the W2 replay step.
- Updated contextual episode smoke rollout to pass an explicit nominal LQR controller and label it `nominal_unselected_smoke`; selector-blocked episodes now write a non-execution audit row with an empty selected primitive instead of executing a placeholder primitive.
- Split executable controller evidence status so W0/W1 candidate rows and nominal smoke rows are not labelled `registry_backed_executable`.
- Made W3 replay row evidence status follow the source table's `ArchiveTableSourceInfo` unless the source itself is evidence-eligible, matching the W2 rule and preventing row-level metadata from upgrading smoke or retired sources.
- Corrected the public LQR contextual archive default and deferred command output roots to the `lqr_contextual_v1_0/r6` stage folder, preserving the documented short-stage result layout.
- Changed direct model-backed rollouts with an explicit LQR controller but no registry provenance from `W0_W1_registry_selected` to `explicit_lqr_unverified`, so only registry-loaded rows carry active selected-controller evidence.
- Tightened W0/W1 tuning coverage so completion/fallback registry status requires paired W0 and W1 rows for the same primitive, candidate, and `paired_start_key`.
- Removed retired feedback-contextual runner stubs from the active scenario tree and renamed their dry-run tests to LQR-specific names, leaving retired evidence only under `03_Control/99_Archive`.

### Validation status

- `C:\Program Files\Blender Foundation\Blender 4.4\4.4\python\bin\python.exe -m compileall 03_Control` passed.
- `python -m pytest -q 03_Control/tests` was attempted and blocked by the WindowsApps launcher (`specified logon session does not exist`).
- `C:\Users\GH-X-ST\.conda\envs\Paul_Li_FYP\python.exe -m pytest -q 03_Control/tests` was attempted and blocked because that environment lacks `pytest`.
- `git diff --check` passed with line-ending warnings only.
- Active-code scans found no PD/PID fallback, TVLQR, MPC, LQR-tree, reachable-chain, or online fan-layout branch tokens in active code outside generated/retired roots.
- Active `boundary_terminal` references are tests, state-sample detail labels, or `xy_boundary_terminal` failure labels; active outcome-class tests assert it is not emitted as an `outcome_class`.

### Claim boundary

No dense archive, hardware-readiness, real-flight transfer, W2/W3 robustness, mission-success, PD/PID fallback, TVLQR/MPC/LQR-tree, reachable-chain, online fan-layout branch, or claim beyond `simulation_only` was introduced by this current-code audit fix.

## Active Contract Guard Addendum

Date: 2026-05-23

### Fixed current-code drift guards

- Added a dependency-free active contract audit entrypoint at `03_Control/04_Scenarios/run_active_contract_audit.py`.
- Added pytest coverage that requires the active contract audit to have no findings once a working pytest environment is available.
- The audit checks canonical evidence-status vocabulary, absence of active forbidden controller-method tokens, no active `boundary_terminal` outcome assignment, and W2/W3 replay-only provenance without retuning paths.
- Housekeeping allowlists now include the audit entrypoint and its test.

### Validation status

- `C:\Users\GH-X-ST\.conda\envs\Paul_Li_FYP\python.exe 03_Control\04_Scenarios\run_active_contract_audit.py` passed.
- `C:\Program Files\Blender Foundation\Blender 4.4\4.4\python\bin\python.exe -m compileall 03_Control` passed.
- `git diff --check` passed with line-ending warnings only.
- `python -m pytest -q 03_Control/tests` remains blocked by the WindowsApps launcher (`specified logon session does not exist`).
- `C:\Users\GH-X-ST\.conda\envs\Paul_Li_FYP\python.exe -m pytest -q 03_Control/tests` remains blocked because that environment lacks `pytest`.

### Claim boundary

No dense archive, hardware-readiness, real-flight transfer, W2/W3 robustness, mission-success, PD/PID fallback, TVLQR/MPC/LQR-tree, reachable-chain, online fan-layout branch, or claim beyond `simulation_only` was introduced by this guard pass.
