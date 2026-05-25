# v4.5 Artifact-Completeness Repair Manifest

- Project line: `LQR-Stabilised Contextual Primitive v4.5`
- Branch: `main`
- HEAD inspected before repair edits: `f72bcad4`
- Timestamp: `2026-05-25 Europe/London`
- Controlling docs: `docs/Glider_Control_Project_Plan.md`, `docs/Daily_Schedule.txt`, `docs/Skills.md`, `docs/Python Coding Instruction.txt`, `docs/MATLAB Coding.txt`, `docs/housekeeping_and_naming_rules.md`

## Diagnostic Status

- W01 run `008` is labelled `superseded_artifact_incomplete_not_w2_replayable`.
- W2 run `010` is labelled `blocked_diagnostic_artifact_incomplete`.
- The W2 block is not treated as algorithm, controller, or survival-method failure evidence.
- Root cause: W01 `008` did not provide the full executable frozen timing-aware controller payload required by strict W2 replay, including `augmented_gain_matrix_json` and `predictor_A_reduced_json`; the v4.5 repair also requires `augmented_A_matrix_json` and `augmented_B_matrix_json`.

## Retained Historical Roots

- Retained dense W01 root: `03_Control/05_Results/lqr_contextual_v1_0/w01_dense/008`
- Retained dense W2 root: `03_Control/05_Results/lqr_contextual_v1_0/w2_survival/010`
- Dense partitions are retained as historical diagnostic artifacts and must not be selected by default W2/W3 inputs.
- Corrected W2 must consume a later W01 root that contains `manifests/frozen_w01_controller_bundle.json`.

## Compact Evidence Hashes

| Path | Bytes | SHA256 |
| --- | ---: | --- |
| `03_Control/05_Results/lqr_contextual_v1_0/w01_dense/008/manifests/run_manifest.json` | 4377 | `64cb7b45602c3438a2e4eaaa7c96cfbf656523692273e824e709dc357c9ebc92` |
| `03_Control/05_Results/lqr_contextual_v1_0/w01_dense/008/manifests/primitive_variant_registry.json` | 1380216 | `4e5b18f4ab4d58e86f541bf2cbb11f4e08c8b9a2027856faf86ff53ff5e9a942` |
| `03_Control/05_Results/lqr_contextual_v1_0/w01_dense/008/manifests/table_manifest.json` | 1158286 | `664c50a4da4cbdb00cf751f597a193b82cf6f8e081801b1bb8b1c2312caccdfd` |
| `03_Control/05_Results/lqr_contextual_v1_0/w2_survival/010/manifests/w2_survival_manifest.json` | 2405 | `fa2a2242e35e2d4a3ac0b747ec748f0fa6eee15ea398f012d51a69a9992ee5c8` |
| `03_Control/05_Results/lqr_contextual_v1_0/w2_survival/010/manifests/w2_survivor_registry.json` | 175 | `85084a8214077cad65eb4536ee8fa3fa872213aba35b28ffd0ae3b71a88cb2db` |
| `03_Control/05_Results/lqr_contextual_v1_0/w2_survival/010/manifests/table_manifest.json` | 714742 | `13e7027c1364ea2dcfeec3a3936ec256272906ac8237fdb3ef1d0dfbd6f96b70` |

## Corrected Artifact Contract

- W01 emits `manifests/frozen_w01_controller_bundle.json`, `metrics/frozen_w01_controller_bundle_summary.csv`, and `reports/frozen_controller_bundle_audit.md`.
- The frozen bundle is the only executable controller source for W2.
- W2 reconstruction restores exact stored controller payloads and verifies controller ID, primitive-variant ID, physical gain checksum, augmented A/B checksums, and augmented gain checksum.
- W2 must not call LQR synthesis, DARE/CARE solvers, Q/R candidate generation, or mutable controller-design helpers.
- W3 accepts only W2 roots whose survival manifest and survivor registry report `survived_variants_available`.

## Next Clean Run IDs

- W01 `011`: artifact-complete dry schedule.
- W01 `012`: artifact-complete executable smoke.
- W2 `011`: dry fixed-replay smoke from W01 `012`.
- W2 `012`: executable fixed-replay smoke from W01 `012`.
- W3 `011`/`012`: real-artifact smoke only if W2 survivors exist; otherwise blocked no-survivor report.
- W01 `013`: corrected rich-side W01 dense library.
- W2 `013`: corrected preferred dense W2 replay.
- W3 `013`: corrected preferred W3 replay only after W2 survivors exist.

## Claim Boundary

This repair may claim only that the active Python control-simulation pipeline has been repaired toward artifact-complete W01 frozen-controller generation and strict W2/W3 fixed-replay consumption. It does not claim W0/W1 dense completion, W2 survival success, W3 robustness, compact-library readiness, governor validation, hardware readiness, real-flight transfer, mission success, or formal LQR-tree/funnel/region-of-attraction guarantees.
