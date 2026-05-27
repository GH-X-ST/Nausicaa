# Model-Only Contextual Primitive Reset Report

Status: historical reset report. This file is retained under `docs/reset/` as
deleted-history context only. It is not an active project plan, allowlist, or
stage contract. For current method control, use
`docs/Glider_Control_Project_Plan.md`.

## Status

Working-tree reset complete. Git metadata finalisation is blocked by workspace ACLs:

- Branch creation failed because `.git/refs/heads` is write-denied.
- `git rm`/staging failed because `.git/index.lock` cannot be created.
- The reset is therefore implemented in the working tree on `main`, with deleted files visible as unstaged deletions until Git metadata permissions are repaired.

## Kept Categories

- Glider, flight dynamics, trim, and linearisation foundation.
- State, command, metric, arena, scenario, latency, and local-flow contracts.
- Measured/fitted updraft loader and required updraft input workbooks.
- Contextual dense runtime and table storage utilities.
- Active environment-conditioned contract documents.
- Foundation and housekeeping tests.

## Removed Categories

- Old fixed-gate and paired archive implementation paths.
- Old reachable-chain and branch-specific package paths.
- Old primitive-library and agile-turn expansion paths.
- Old generated result evidence under `03_Control/05_Results`.

## Validation

- `python -m py_compile` on retained importable modules: pass.
- `python -m pytest -q 03_Control/tests`: `87 passed`.
- `python -m py_compile` on all retained Python files: pass.
- `git diff --check`: pass, with line-ending warnings only.
- Result-root audit: `03_Control/05_Results` contains only `.gitkeep`.
- File-size audit: no retained non-Git file exceeds 100 MB.
- Contamination audit: retained active docs, source, tests, and import paths pass the housekeeping gate; old-method history is confined to `docs/reset/`.
- Active project plan: `docs/Glider_Control_Project_Plan.md` was replaced with the latest supplied environment-conditioned plan; its old-method wording is allowed only as explicit negative/historical boundary language.
- Generated cache cleanup: `.pytest_cache` and all `__pycache__` folders were removed after validation.
- Out-of-scope latency/hardware folders: empty remnants of `B_Test_Lantency` and `C_Overall_Latency` were removed from the working tree.

## Claim Boundary

No controller result, primitive policy, transfer claim, hardware-readiness claim, or environment-generalisation claim is produced by this reset.

The only permitted repository-level claim is that the working tree has been reduced to a clean modelling and runtime foundation for environment-conditioned primitive development.
