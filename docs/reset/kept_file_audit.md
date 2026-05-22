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

The Git metadata directory is write-denied in this workspace, so branch creation, staging, and final tracked-state verification are recorded in the reset report rather than asserted by the retained test suite.
