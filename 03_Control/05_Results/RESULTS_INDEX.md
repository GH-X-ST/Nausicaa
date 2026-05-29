# Results Index

This folder intentionally keeps the existing result paths stable. Do not rename
historical evidence folders just to improve readability: manifests, audit logs,
and reports may reference the current paths.

Active evidence currently lives under:

```text
03_Control/05_Results/lqr_contextual_v1_0/
```

## Thesis-Facing Stage Mapping

| Current folder | Meaning |
| --- | --- |
| `lqr_contextual_v1_0/r5_r11_pipeline/` | Thesis workflow orchestration logs for `R5 -> R7 -> R8 -> R10 -> R11`. |
| `lqr_contextual_v1_0/w01_dense/` | `R5`: transition-aware primitive learning / dense W0-W1 evidence. |
| `lqr_contextual_v1_0/w3_survival/` | `R7`: held-out transition validation under domain randomisation. |
| `lqr_contextual_v1_0/post_w3_library_size_study/` | `R8`: coverage-aware medoid library compression. |
| `lqr_contextual_v1_0/outcome_model/` | `R8`: outcome model built from compressed post-W3 libraries. |
| `lqr_contextual_v1_0/changed_case_validation/` | `R10`: viability-governor and residual-memory tuning. |
| `lqr_contextual_v1_0/heldout_changed_case_validation/` | `R11`: held-out validation, when present. |

## Internal, Diagnostic, Or Legacy Mapping

| Current folder | Meaning |
| --- | --- |
| `lqr_contextual_v1_0/w2_survival/` | Archived/internal W2-style survival evidence; not part of the active thesis workflow. |
| `lqr_contextual_v1_0/r5_r10_pipeline/` | Older orchestration roots kept for audit compatibility. |
| `lqr_contextual_v1_0/repeated_launch_validation/` | R9-style internal preflight / diagnostic repeated-launch validation. |
| `lqr_contextual_v1_0/repeated_launch_level_launch_diagnostic/` | Level-launch diagnostic checks. |
| `lqr_contextual_v1_0/repeated_launch_validation_command_check/` | Command/scheduler diagnostic checks for repeated-launch validation. |
| `lqr_contextual_v1_0/lqr_controller_trajectory_audit/` | Lightweight controller trajectory sanity plots and audit outputs. |
| `lqr_contextual_v1_0/algorithm_contract_audit/` | Contract-audit outputs for active algorithm invariants. |
| `lqr_contextual_v1_0/runtime_smoke/` | Runtime smoke-test outputs. |
| `lqr_contextual_v1_0/run_logs/` | Local run logs. |
| `lqr_contextual_v1_0/archive/` | Historical or rejected evidence retained for traceability. |

## Current Official Run Label

`A01` is the first official thesis-facing workflow label. It is a folder label,
not a replacement for numeric manifest `run_id` values.

Important A01 roots:

```text
lqr_contextual_v1_0/r5_r11_pipeline/A01
lqr_contextual_v1_0/w01_dense/A01
lqr_contextual_v1_0/w3_survival/A01
lqr_contextual_v1_0/post_w3_library_size_study/A01
lqr_contextual_v1_0/outcome_model/A01
lqr_contextual_v1_0/changed_case_validation/A01
```

In A01, R5, R7, and R8 passed. R10 completed but blocked before R11 because
the full-domain governor-tuning gates did not pass. Therefore A01 does not
support R11 validation, memory-improvement, hardware-readiness, real-flight
transfer, mission-success, or autonomy claims.

