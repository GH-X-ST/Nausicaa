# W01/W2/W3 Alignment Cleanup Manifest

Date/time: 2026-05-24 17:42:17 +01:00

Current branch: `main`

Current HEAD: `d52a67f9f46420996c8c95047f6aecf595f57e1c`

## Source Documents Used

- `docs/Glider_Control_Project_Plan.md`
- `docs/Daily_Schedule.txt`
- `docs/Skills.md`
- `docs/Python Coding Instruction.txt`
- `docs/MATLAB Coding.txt`
- `docs/housekeeping_and_naming_rules.md`
- User-provided revised W01 dense sweep readiness plan from 2026-05-24.

## Active Mismatch Summary

The active control tree still exposes stale selected-controller and staged
R6/R6.1/R7/R8/R9 workflow assumptions. These conflict with the current
W01 -> W2 -> W3 contract because W0/W1 must generate and preserve a rich
primitive-controller variant library, not a small selected-controller registry
or pre-W2/W3 shortlist.

Active cleanup targets include:

- selected-controller registry outputs and imports;
- R6/R6.1 finalist-selection and accepted-fallback statuses;
- R7/R8/R9 active stage naming in runners/tests;
- pre-W2/W3 medoid, clustering, and hardware-shortlist outputs;
- W1 fan-shift and power-scale official dense modes;
- stale result roots under `03_Control/05_Results`;
- any active bounded-PD/contextual-feedback result copy outside
  `03_Control/99_Archive/retired_pd_contextual_v1_4`.

## Result Folders Deleted After This Manifest

| Path | Files | Size bytes | Reason |
|---|---:|---:|---|
| `03_Control/05_Results/lqr_contextual_v1_0/r6/tune_103/` | 38 | 9832060 | R6 selected-controller tuning result; retired from active W01 workflow. |
| `03_Control/05_Results/lqr_contextual_v1_0/r6/tune_111_r6_1/` | 274 | 13123709 | R6.1 finalist/selected-controller result; retired from active W01 workflow. |

The active result root should contain only `03_Control/05_Results/.gitkeep`
until approved W01 dry-run or tiny-smoke outputs are generated.

## Compact Evidence Hashes Recorded Before Deletion

### `tune_103`

| File | Bytes | SHA256 |
|---|---:|---|
| `manifests/run_manifest.json` | 2154 | `591bdeb94607d0b6913efc9bc243a5e12ea5a7eb7a0fe3fa3fcec711a11e9e32` |
| `manifests/selected_lqr_controllers.json` | 494203 | `480e789f77bb39a9636749ed059b98f0ef28c1ec26a86c436e27cc24720196a5` |
| `manifests/table_manifest.json` | 67637 | `016b74f3471af8eb4053de84861d1430e701f0330b5357c11e1be2aca1d47327` |
| `metrics/chunk_summary.csv` | 1623 | `6b93c1b6c4329053779e71151dd2256c54a173e1e418f551e62a90e90534a751` |
| `metrics/coverage_summary.csv` | 650337 | `0513c0dc321d0a990f74286d016c46533d9b8187e154d47febb397c7a82e11cf` |
| `metrics/file_size_audit.csv` | 5485 | `17df917b1f344e4f972397d5606506499d98f1d6c9d9af09c0356c84cc2889da` |
| `metrics/selected_lqr_controllers.csv` | 308001 | `9b868bb6f487672c0aa50209da013a7507e9dd61052f2197127130c4f632cd6e` |
| `reports/claim_boundary_report.md` | 95 | `3efb54648c3f1c7d7c00e746a29aa0aab1e4d3bcd4f5087273af7df826f42bc1` |

### `tune_111_r6_1`

| File | Bytes | SHA256 |
|---|---:|---|
| `manifests/run_manifest.json` | 2418 | `1052637581c8ad54046aada8eed4b4ad98f18e2c40f52b09e3158456084f4351` |
| `manifests/selected_lqr_controllers.json` | 1002442 | `175254535a85c461fc20b396c99d52f7d36b6b05f4cec265e31f0cd775b2f9ad` |
| `manifests/table_manifest.json` | 661997 | `6d944392e8ea0bb9388c850e65f60c32e14cb1f7443841258eebfd365531f217` |
| `metrics/chunk_summary.csv` | 18665 | `c67d9f9a676f76b7c0c70e71572000fb3234622667f1f696291d59c0e034fd0b` |
| `metrics/coverage_summary.csv` | 7770 | `142a0a854976dddc5af0d8f352168b9fd2b812ae8fce3aef255872ed940dabd3` |
| `metrics/file_size_audit.csv` | 40872 | `70dfec89e28b1a3b72f410b70ded88346678b57103232fc5db462ce9c34c4820` |
| `metrics/selected_lqr_controllers.csv` | 629371 | `57ef52719bcd9462f72cb6abfaaafb43a2297d7c0dc39cf8d8e8a03b846c9823` |
| `reports/claim_boundary_report.md` | 240 | `49d6d31f0035127cc78afd4d99d6436c0b562db54860887511b667dac225b75d` |
| `reports/r6_1_smart_lqr_tuning_report.md` | 1578 | `91451daed82d1ee80d8f0770f3077ecef3dbe7d8eb901314c9bc9d39a12b3147` |

## Files Rewritten Or Replaced

Planned active rewrites in this pass:

- `03_Control/03_Primitives/primitive_variant_registry.py` added.
- `03_Control/03_Primitives/lqr_tuning.py` cleaned to W01 candidate semantics.
- `03_Control/03_Primitives/controller_registry.py` removed from active use.
- `03_Control/04_Scenarios/state_sampling.py` cleaned to the required
  40/25/15/10/10 mixed primitive-start sampler.
- `03_Control/04_Scenarios/run_lqr_w01_dense_chunked.py` added as the next
  readiness target.
- W2, W3, and post-W3 runners added or replaced as fixed-LQR/blocked scaffolds.
- Active contract audit replaced by W01/W2/W3 contamination checks.
- Tests rewritten away from R6/R6.1 selected-finalist assumptions.

## Retained Foundations

The glider model, flight dynamics, trim solver, foundation linearisation,
state/command/metric contracts, latency helpers, wing-wind descriptors,
dense-runtime/table/chunk helpers, arena contracts, and updraft model loaders
are retained unless a direct bug is found during validation.

## Claim Boundary

This cleanup may claim only that the active control repository is being
cleaned and restructured for corrected W0/W1 rich primitive-controller dense
generation readiness.

It does not claim W0/W1 dense evidence completion, W2 survival, W3 robustness,
post-W3 compact-library readiness, governor validation, hardware readiness,
real-flight transfer, or mission success.

## Validation Status

Validation after implementation:

- PowerShell-expanded `py_compile` over `03_Control/02_Inner_Loop`,
  `03_Control/03_Primitives`, and `03_Control/04_Scenarios`: passed.
- `python -m pytest -q 03_Control/tests --basetemp .codex_run_logs\pytest_tmp -o cache_dir=.codex_run_logs\pytest_cache`:
  passed, 150 tests.
- `python 03_Control/04_Scenarios/run_w01_w2_w3_contract_audit.py`: passed.
- W01 dry-run readiness command for run `001`: passed.
- W01 tiny smoke readiness command for run `002`: passed.
- Active generated file-size audit: passed; largest generated file is below
  1 MB, well under the 100 MB hard limit.
- `git diff --check`: passed with line-ending warnings only.

Remaining boundary: the repo is ready only for the first W0/W1 dense preflight.
No W0/W1 dense evidence completion, W2 survival, W3 robustness, compact-library,
governor, hardware, transfer, or mission-success claim is made.

## LQR-Stabilised Contextual Primitive v4.0 Readiness Fix Addendum

The tracked W01 readiness roots `03_Control/05_Results/lqr_contextual_v1_0/w01_dense/001`
and `03_Control/05_Results/lqr_contextual_v1_0/w01_dense/002` are retained as
prior approved dry-run/tiny-smoke artifacts. They are superseded for future
readiness by the v4.0 W01 dense-readiness fix pass and must not be interpreted
as completed W0/W1 dense evidence.

This addendum keeps the same claim boundary: W01 evidence generation readiness
only. It does not claim W0/W1 dense evidence completion, W2 survival, W3
robustness, post-W3 compact-library readiness, governor validation, hardware
readiness, real-flight transfer, or mission success.

## LQR-Stabilised Contextual Primitive v4.3 Timing-Aware Addendum

Date/time: 2026-05-24 20:15:09 +01:00

Current branch: `main`

Current HEAD: `f54941b256571499771e7fdaa3c5213d206e786a`

This pass supersedes the v4.2 trim/local reduced-order active W01 controller
with `predictor_compensated_augmented_discrete_lqr_v1` as the active W01
controller method. The implementation uses a discrete-time augmented LQR with
actuator surface states, command-delay FIFO states, and predictor compensation
for nominal measured-state delay. It does not claim true full delayed-state
feedback augmentation.

Historical W01 roots `001` through `004` are retained as prior approved
readiness artifacts. New v4.3 readiness artifacts are:

- `03_Control/05_Results/lqr_contextual_v1_0/w01_dense/005`: dry-run
  schedule/manifest readiness only, no table partitions.
- `03_Control/05_Results/lqr_contextual_v1_0/w01_dense/006`: 2000-row
  timing-aware W01 preflight validation with partitioned dense rows.

Validation after this v4.3 pass:

- PowerShell-expanded `py_compile` over `03_Control/02_Inner_Loop`,
  `03_Control/03_Primitives`, and `03_Control/04_Scenarios`: passed.
- `python -m pytest -q 03_Control/tests --basetemp .codex_run_logs\pytest_tmp -o cache_dir=.codex_run_logs\pytest_cache`:
  passed, 154 tests.
- `python 03_Control/04_Scenarios/run_w01_w2_w3_contract_audit.py`: passed.
- W01 dry-run readiness command for run `005`: passed.
- W01 timing-aware preflight command for run `006`: passed, 2000 rows.
- Separate `.codex_run_logs` repair validation regenerated corrupted chunk
  `c00000` and preserved deterministic row count/checksum behavior; no active
  evidence root was deliberately corrupted.
- File-size audit for run `006`: passed; largest generated file is below 1 MB,
  well under the 100 MB hard limit.
- `git diff --check`: passed with line-ending warnings only.

The v4.3 claim boundary remains workflow and readiness only. It does not claim
W0/W1 dense evidence completion, W2 survival, W3 robustness,
post-W3 compact-library readiness, governor validation, hardware readiness,
real-flight transfer, or mission success.
