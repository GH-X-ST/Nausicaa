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
