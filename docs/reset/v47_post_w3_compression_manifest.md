# v4.7 Post-W3 Compression Manifest

- Project line: `LQR-Stabilised Contextual Primitive v4.7`
- Branch: `main`
- HEAD inspected before v4.7 edits: `aa6dd8cbad5d73f67b685e245fd1c69b49ab7a2b`
- Timestamp: `2026-05-25 Europe/London`
- Controlling docs: `docs/Glider_Control_Project_Plan.md`, `docs/Daily_Schedule.txt`, `docs/Skills.md`, `docs/Python Coding Instruction.txt`, `docs/MATLAB Coding.txt`, `docs/housekeeping_and_naming_rules.md`

## Source Dense Roots

| Stage | Source root | Rows | Chunks | Storage | Largest file | Size MB | File-size status |
| --- | --- | ---: | ---: | --- | --- | ---: | --- |
| W01 | `03_Control/05_Results/lqr_contextual_v1_0/w01_dense/015` | 76,800 | 96 | `csv_gz` | `manifests/frozen_w01_controller_bundle.json` | 4.515 | below 100 MB |
| W2 | `03_Control/05_Results/lqr_contextual_v1_0/w2_survival/015` | 51,200 | 64 | `csv_gz` | `manifests/frozen_w01_controller_bundle.json` | 4.515 | below 100 MB |
| W3 | `03_Control/05_Results/lqr_contextual_v1_0/w3_survival/013` | 51,200 | 64 | `csv_gz` | `tables/w3_survival_rows/c00011.csv.gz` | 1.559 | below 100 MB |

W01 `015` provides 256 ready frozen timing-aware controllers. W2 `015` reports `w2_dense_survival_pass` with 256 W2 survivors. W3 `013` is complete and contains row-level fixed-replay evidence; before this pass it did not yet contain a variant-level W3 survivor registry.

## Retained Non-Default Roots

Earlier roots including W01 `008`, W2 `010`, W01 `012`, W2 `012`, W3 `011`, W01/W2 smoke `013/014`, and W3 smoke `012` are retained as historical readiness, smoke, or diagnostic artifacts. They are not default post-W3 compression inputs.

## v4.7 Contract

- W3 analysis scores only entry-role-compatible rows and preserves incompatible rows as rejection/block evidence.
- Post-W3 compression uses only W3-surviving variants and selects real frozen representative variants without mutating Q/R, K, references, horizons, entry roles, `controller_id`, or `primitive_variant_id`.
- Outcome-model and governor smoke may run only after the L10 compact-library gate passes.

## Claim Boundary

Allowed after gates pass: simulation-only W01/W2/W3 dense evidence exists, a post-W3 compact representative primitive library exists, and the first outcome-model/governor smoke exists. Blocked claims remain W3 robustness proof, hardware readiness, real-flight transfer, mission success, full-loop validation success, memory/governor performance improvement, compact-library real-flight approval, and formal LQR-tree/funnel/region-of-attraction guarantees.
