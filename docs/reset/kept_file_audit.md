# Kept File Audit

Status: historical reset audit. This file is retained under `docs/reset/` as
deleted-history context only. It is not an active allowlist, project plan, or
stage contract. For current method control, use
`docs/Glider_Control_Project_Plan.md`.

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

## Feedback Contextual Primitive v1.3 Addendum

The active allowlist now includes a stage-isolated overnight evidence driver and shared stage evidence utilities. R6 archive generation is constrained to W0/W1 only; W2 is reserved for actual R8 model-backed replay, and W3 is reserved for R9 generalisation replay.

R8 and R9 completion cannot be assigned from copied source labels or scaffold case tables. Completion requires rows generated through the retained primitive rollout path, plus compressed partitions, table manifests, checksums, coverage summaries, blocked/approximate ratio summaries, file-size audits, and claim-boundary reports.

Stage status is independent: R6, R7, R8, and R9 are recorded as complete, fallback, partial, blocked, or deferred without invalidating earlier completed evidence when a later stage blocks. No controller-performance, mission-success, hardware-readiness, real-flight-transfer, full W2 survival, W3 robustness, or environment-generalisation claim is made by these utilities.

## Feedback Contextual Primitive v1.4 Addendum

The active allowlist now includes the v1.4 run-hardening driver and tests. The preferred entrypoint updates the evidence-status manifest after preflight and after every stage, supports dry-run schedules, stop-after-stage operation, resume, and repair-incomplete handling, and keeps the v1.3 driver available as the previous compatibility path.

The hardening scope is operational safety only. Preflight failures stop before first-chunk projection, R6/R8/R9 use stage-specific first chunks for runtime and partition-size projection, R7 records full-source training and bounded-evaluation metadata, and R8/R9 replay writers produce chunked compressed partitions with chunk manifests and checksums. No full overnight evidence run is launched by this reset or by the v1.4 tests.

## R6 Dependency Hygiene Addendum

The active allowlist now includes separate control and design dependency files. R6 control validation uses `requirements-control-dev.txt`, which includes the active `03_Control` runtime stack plus pytest; `aerosandbox` is isolated in `requirements-design.txt` and must not gate R6 control validation.

The root requirement files remain whole-repository aggregates only. This dependency split changes validation hygiene and audit wording only; it does not change the LQR-only, simulation-only method boundary or promote any dense, hardware, transfer, robustness, or mission-success claim.
