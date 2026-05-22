# Kept File Audit

This audit is strict allowlist-based. Every active retained code file, test file, data loader, importable module, and active contract document is listed in `kept_file_audit.csv`.

## Audit Rules

- `active_contract_doc` applies only to the five active contract documents.
- `old_method_token_status=none` means no old-method contamination remains in the retained file.
- `old_method_token_status=boundary_prohibition_context_only` is used for the latest active project plan, where old-method wording appears only in explicit "do not use" or historical-boundary statements from the current plan.
- `old_method_token_status=allowed_historical_reset_context` is allowed only under `docs/reset/`, where old paths are described as deleted history.
- Any retained active source, active doc, test, or import path with `must_remove_before_reset_passes` blocks completion.

## Contamination Result

The allowlisted active files are intended to be free of:

```text
fixed-gate branch-specific controller logic
old W0/W1 archive runner requirements
reachable-chain-as-required-gate wording
branch-specific governor package wording
old medoid package plumbing
old generated archive/result paths
agile-turn final dependency wording
old Codex plan references
fan branch as online algorithm wording
```

Reset-history references are confined to `docs/reset/` as deleted-history context.

## Runtime/Storage Result

The retained runtime/storage utilities are kept only for the current dense-run contract:

```text
chunked
resumable
compressed
worker-enabled
checksum-manifested
no generated file above 100 MB
```

The housekeeping tests enforce that active docs are present and not ignored, active imports do not include result folders, retained files do not contain forbidden active traces, and `03_Control/05_Results` contains only `.gitkeep`.

## R6-R7 Addendum

The active allowlist now includes the strict surrogate resolver, model-backed rollout path, first auditable primitive outcome model, viability selector, worker-enabled contextual archive scaffold, and their focused tests. These additions keep W-layer and fan-layout information as environment/surrogate metadata rather than separate online controller logic.

R6 validation remains temp-only. Official local archive generation is deferred and must preserve chunked execution, compressed partition tables, worker execution, checksums, resumable manifests, and the 100 MB file limit.

The Git metadata directory is write-denied in this workspace, so branch creation, staging, and final tracked-state verification are recorded in the reset report rather than asserted by the retained test suite.

## R6-R8 Feedback Addendum

The active allowlist now also includes `prim_ctrl.py`, feedback-backed rollout evidence roles, continuation/terminal target separation in `prim_model.py`, explicit `continuation` and `terminal_episode` selector modes, and a temp-only episode smoke runner.

Preserved support documents are not deleted. MATLAB and plotting guidance are retained as non-contract support material after wording alignment. They now describe W labels as validation layers, fan cases as environment instances, and x/y boundary exits as retained terminal episode evidence.

The R6-R8 alignment audit records old-code mismatches as fixed or intentionally preserved non-contract support. No controller-performance, W2/W3 robustness, real-flight transfer, mission-success, hardware-readiness, or repeated-launch improvement claim is made.
