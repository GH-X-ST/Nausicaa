# Outer-Loop Spatial Flow-Belief Memory Governor v8.17

Date: 2026-06-08

## Scope

This note records one additional full dense R11 E03 repeat for the frozen
outer-loop spatial flow-belief memory governor. No controller code, primitive
catalogue, R10 handoff config, launch-gate contract, real-flight runtime
contract, scoring gate, or public API was changed.

The outputs are the `E03.9` R11 validation root, matched true-neutral/open-loop
diagnostics, one separate aggregate appendix table, and active-doc wording
updates from the `E03.1` through `E03.8` boundary to the `E03.1` through
`E03.9` boundary.

## Dense R11 Repeat

The full dense command ran in the foreground with a 3-hour watchdog
(`timeout_ms=10800000`). No reduced readiness smoke run was inserted before the
dense repeat.

| Run | Run ID | Seed | Wall Time | Status | Final | History | Strict Gate | Max File |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: |
| `E03.9` | `11` | `119` | `10071.7 s` | complete | `4000` | `34400` | false | `61.064 MB` |

The run produced a complete manifest, required launch counts, pass/fail
summary, file-size audit, and real-time scheduler audit. The strict gate stayed
false, so this root extends the held-out evidence boundary but does not create
a strict R11 pass claim.

## Neutral Diagnostics And Table

Matched true-neutral/open-loop diagnostics completed for:

- `03_Control/A_figures/R11_E03.9_balanced_neutral_baseline`

The diagnostic completed with `160` neutral cases and `8` figures.

The separate aggregate appendix table was generated without overwriting earlier
E03 aggregate tables:

- `03_Control/A_figures/R11_E03_appendix_tables/e03_r11_speed_tables_e03_1_to_e03_9.tex`
- `03_Control/A_figures/R11_E03_appendix_tables/e03_r11_speed_table_summary_e03_1_to_e03_9.csv`
- `03_Control/A_figures/R11_E03_appendix_tables/e03_controller_minus_open_loop_e03_1_to_e03_9.csv`

The summary includes all completed run labels (`E03.1` through `E03.9`), all
eight R11 ladder IDs, the R11 policy rows, and the matched
`true_neutral_open_loop` rows.

## PR-Style Message

```text
Outer-Loop Spatial Flow-Belief Memory Governor v8.17
```

```text
Python control-simulation pipeline

- Adds full dense R11 repeat `E03.9` using explicit E03 roots, run ID 11, seed 119, 20 outer cases per ladder, 4000 final held-out launches, and 34400 history launches.
- Extends the repeated full-R11 evidence boundary from `E03.1` through `E03.8` to `E03.1` through `E03.9` while preserving the frozen R10 handoff, primitive library, spatial flow-belief memory contract, launch-gate contract, and 0.100 s primitive-boundary timing audit.
- Regenerates matched true-neutral/open-loop diagnostics for `E03.9` and writes the separate aggregate appendix table at `03_Control/A_figures/R11_E03_appendix_tables/e03_r11_speed_tables_e03_1_to_e03_9.tex`.
- Verifies manifest completion, required launch counts, pass/fail summary, file-size audit, scheduler/timing evidence, neutral diagnostic completion, and aggregate table schema; the strict R11 gate remains false.
- No strict R11 pass, memory-improvement, hardware-readiness, real-flight transfer, mission-success, or autonomy claim is made by this repeat.
```

## Claim Boundary

The `E03.1` through `E03.9` repeat set is the current held-out validation
boundary for this frozen governor. It is accepted only as physically
explainable evidence where the failure modes remain visible in the pass/fail
summaries and neutral/open-loop comparisons.

This workflow does not claim strict R11 pass status, broad spatial-memory
improvement, hardware readiness, real-flight transfer, mission success, or
autonomy.

## Checks

- `git status --short`
- `python -m py_compile 03_Control/04_Scenarios/run_heldout_changed_case_validation.py 03_Control/04_Scenarios/run_changed_case_validation.py 03_Control/04_Scenarios/run_repeated_launch_learning_curve.py 03_Control/01_Plotting/run_r11_appendix_speed_tables.py 03_Control/01_Plotting/run_r11_balanced_neutral_baseline_figures.py`
- R11 dry schedule probe with explicit E03 roots: `4000` final held-out launches and `34400` history launches
- Manifest, pass/fail, file-size, and scheduler/timing audits for `E03.9`
- `python 03_Control/01_Plotting/run_r11_balanced_neutral_baseline_figures.py` for `E03.9`
- `python 03_Control/01_Plotting/run_r11_appendix_speed_tables.py` for the `E03.1` through `E03.9` aggregate
- Aggregate summary schema check: `960` rows, `source_validation_run_labels=E03.1;E03.2;E03.3;E03.4;E03.5;E03.6;E03.7;E03.8;E03.9`, `true_neutral_open_loop` present, and `8` R11 ladders present
- Final docs audit for active `.md` and `.txt` guidance
- `python -m pytest 03_Control/tests/test_v53_algorithm_contract.py::test_v53_stage_contract_is_r5_r7_r8_r10_r11_with_r6_archived_and_r9_internal 03_Control/tests/test_v53_algorithm_contract.py::test_v53_r10_and_r11_changed_case_randomisation_semantics_match 03_Control/tests/test_v53_algorithm_contract.py::test_v53_r11_full_safe_success_gate_catches_safe_exit_only_passes --basetemp .pytest_tmp_r11_repeat_e03_9 -p no:cacheprovider`
- `git diff --check`
