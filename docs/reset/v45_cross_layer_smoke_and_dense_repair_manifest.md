# v4.5 Cross-Layer Smoke and Dense Repair Manifest

- Project line: `LQR-Stabilised Contextual Primitive v4.5`
- Branch: `main`
- HEAD inspected before cross-layer repair edits: `af6f5d6d8fe868053ff6c7ff9e6c3ff490cf813d`
- Timestamp: `2026-05-25 Europe/London`
- Controlling docs: `docs/Glider_Control_Project_Plan.md`, `docs/Daily_Schedule.txt`, `docs/Skills.md`, `docs/Python Coding Instruction.txt`, `docs/MATLAB Coding.txt`, `docs/housekeeping_and_naming_rules.md`

## Retained Diagnostic Roots

- W01 run `008`: `superseded_artifact_incomplete_not_w2_replayable`
- W2 run `010`: `blocked_diagnostic_artifact_incomplete`
- W01 run `012`: `artifact_complete_smoke_bundle_ready_but_start_family_incomplete`
- W2 run `012`: `artifact_complete_w2_smoke_blocked_no_survivors_due_sparse_launch_only_smoke`
- W3 run `011`: `clean_no_survivor_block`

These roots are retained as historical diagnostic artifacts. They are not deleted, and default W2/W3 input discovery must not select superseded, sparse-family, blocked, or no-survivor roots as move-on sources.

## Corrected Cross-Layer Contract

- W01 frozen-controller bundle emission remains mandatory through `manifests/frozen_w01_controller_bundle.json`.
- W01 cross-layer smoke requires at least 20 paired starts per candidate, or an equivalent stratified start-family schedule with the exact `8/5/3/2/2` family mix.
- W01 runs below that coverage are labelled `artifact_smoke_only_start_family_incomplete`.
- W2 uses only frozen W01 controller bundles, never physical-K-only fallback or mutable LQR synthesis.
- W2 separates smoke and dense labels: `w2_artifact_smoke_pass`, `w2_smoke_no_survivors`, `w2_dense_survival_pass`, and `w2_dense_no_survivors`.
- W3 fixed replay is chunked, resumable, repairable, and file-size audited; no dense W3 runner may build the full row table in memory.
- W3 method evidence accepts only real W2 survivor roots. Fixture plumbing evidence is labelled `test_fixture_not_method_evidence` and must not be written under method result roots.

## Next Clean Run IDs

- W01 `013`: dry cross-family smoke schedule.
- W01 `014`: executable cross-family smoke.
- W2 `013`: dry W2 smoke from W01 `014`.
- W2 `014`: executable W2 smoke from W01 `014`.
- W3 `012`: real-artifact W3 smoke only if W2 `014` has real survivors; otherwise clean no-survivor block.
- W01 `015`: corrected rich dense run after smoke gates pass.
- W2 `015`: corrected dense replay from W01 `015`.
- W3 `013`: corrected dense fixed replay only after real W2 survivors exist.

## Claim Boundary

This repair may claim only that the v4.5 artifact-complete W01-to-W2-to-W3 replay contract has been smoke-tested across all five start-state families and is ready for corrected dense W01/W2 replay if the gates pass. It does not claim W3 robustness, post-W3 compact-library readiness, governor validation, hardware readiness, real-flight transfer, mission success, or formal LQR-tree/funnel/region-of-attraction guarantees.
