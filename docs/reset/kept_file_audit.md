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

## R6-R8 Pipeline Upgrade Addendum

The active allowlist now includes the Path A code-plus-smoke additions:

- deterministic environment instances and adjusted wind effects;
- deterministic launch and envelope state sampling with measured-log compatibility;
- primitive feature rows using state, context, primitive, latency, and uncertainty fields;
- episodic lift-belief smoke support with the required lambda values;
- selector, W2 replay, and W3 generalisation report scaffolds.

These files are kept as implementation scaffolds and test-covered interfaces, not as completed evidence runs. The audit boundary remains unchanged: official R6 archive, R7 selector-report, and R8 W2 replay completion are deferred until explicitly approved local runs are produced.

## Mixed-Start W3 Scaffold Addendum

The active allowlist now includes first-class mixed primitive-start sampling, archive table reading, implementation/actuator instances, plant instances, W2 mixed-start replay rows, and W3 environment/implementation/plant scaffold cases.

The audit guard is explicit: full raw canonical entry state is logged for replay and safety calculations, but raw arena position is not used as a primitive identifier, online branch, evidence group, result-family name, or primary coordinate-bin summary key. Unsupported W3 perturbations remain labelled `blocked_not_yet_applied` or approximate.
