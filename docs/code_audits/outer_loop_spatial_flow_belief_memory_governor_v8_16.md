# Outer-Loop Spatial Flow-Belief Memory Governor v8.16

Date: 2026-06-08

## Scope

This note records the R11 E03.5+ repeat-evidence workflow for the frozen
outer-loop spatial flow-belief memory governor. The workflow used wall-clock
time, starting at `2026-06-08T04:26:14.6515136+01:00`, immediately before the
initial docs audit. CPU time was not used for the repeat-loop stopping rule.

No controller code, primitive catalogue, R10 handoff config, launch-gate
contract, real-flight runtime contract, scoring gate, or public API was changed
by this workflow. The outputs are additional R11 validation roots, matched
true-neutral/open-loop diagnostics, one separate aggregate appendix table, and
active-doc wording updates.

## Preflight

The initial docs audit covered `48` active and historical `.md`/`.txt` files
under `docs`, totalling `5875` lines. Active docs matched the current pre-run
code/manifests and identified `E03.1` through `E03.4` as the current R11
boundary. Historical/reset audit notes were preserved.

The worktree was checked before dense execution. Existing deletions under
`docs/project_report` were treated as user-owned changes and were not restored
or overwritten.

The R11 runner and table scripts passed syntax compilation. The dry schedule
probe used explicit E03 roots and reported the required full-R11 unit:

- final held-out launches: `4000`
- history launches: `34400`

The existing `E03.1` through `E03.4` roots were physically explainable but not
strict R11 gate passes. Each root had a complete manifest with `4000` final
held-out launches and `34400` history launches, plus pass/fail, file-size, and
real-time scheduler audits.

## Dense R11 Repeats

Each full dense command ran in the foreground with a 3-hour watchdog
(`timeout_ms=10800000`). No reduced readiness smoke run was inserted before the
dense repeats.

| Run | Run ID | Seed | Wall Time | Status | Final | History | Strict Gate | Max File |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: |
| `E03.5` | `7` | `115` | `9429.8 s` | complete | `4000` | `34400` | false | `61.124 MB` |
| `E03.6` | `8` | `116` | `9394.2 s` | complete | `4000` | `34400` | false | `61.028 MB` |
| `E03.7` | `9` | `117` | `9244.1 s` | complete | `4000` | `34400` | false | `61.085 MB` |
| `E03.8` | `10` | `118` | `9630.2 s` | complete | `4000` | `34400` | false | `61.116 MB` |

The repeat loop launched `E03.8` because elapsed real time after verifying
`E03.7` was about `7.940 h`, still below the `8 h` launch threshold. The loop
stopped after `E03.8` because elapsed real time was about `10.627 h`, so
`E03.9` was not started.

All completed repeats had complete manifests, the required launch counts,
file-size audits that allowed local evidence storage, and scheduler/timing
audits. The strict gate stayed false for every repeat, so these roots extend
the held-out evidence boundary but do not create a strict R11 pass claim.

## Neutral Diagnostics And Table

Matched true-neutral/open-loop diagnostics were generated for every new
aggregate input:

- `03_Control/A_figures/R11_E03.5_balanced_neutral_baseline`
- `03_Control/A_figures/R11_E03.6_balanced_neutral_baseline`
- `03_Control/A_figures/R11_E03.7_balanced_neutral_baseline`
- `03_Control/A_figures/R11_E03.8_balanced_neutral_baseline`

Each neutral diagnostic completed with `160` neutral cases and `8` figures.

The separate aggregate appendix table was generated without overwriting the
existing E03.1 or E03.1-to-E03.4 tables:

- `03_Control/A_figures/R11_E03_appendix_tables/e03_r11_speed_tables_e03_1_to_e03_8.tex`
- `03_Control/A_figures/R11_E03_appendix_tables/e03_r11_speed_table_summary_e03_1_to_e03_8.csv`
- `03_Control/A_figures/R11_E03_appendix_tables/e03_controller_minus_open_loop_e03_1_to_e03_8.csv`

The summary includes all completed run labels (`E03.1` through `E03.8`), all
eight R11 ladder IDs, the R11 policy rows, and the matched
`true_neutral_open_loop` rows.

## PR-Style Messages

```text
Outer-Loop Spatial Flow-Belief Memory Governor v8.13
```

```text
Python control-simulation pipeline

- Adds full dense R11 repeat `E03.5` using explicit E03 roots, run ID 7, seed 115, 20 outer cases per ladder, 4000 final held-out launches, and 34400 history launches.
- Keeps the frozen R10 handoff, primitive library, spatial flow-belief memory contract, launch-gate contract, and 0.100 s primitive-boundary timing audit unchanged.
- Verifies `E03.5` manifest completion, required launch counts, file-size audit, scheduler/timing evidence, and pass/fail summary; the strict R11 gate remains false.
- No strict R11 pass, memory-improvement, hardware-readiness, real-flight transfer, mission-success, or autonomy claim is made by this repeat.
```

```text
Outer-Loop Spatial Flow-Belief Memory Governor v8.14
```

```text
Python control-simulation pipeline

- Adds full dense R11 repeat `E03.6` using explicit E03 roots, run ID 8, seed 116, 20 outer cases per ladder, 4000 final held-out launches, and 34400 history launches.
- Extends the repeated full-R11 evidence set while preserving the frozen governor, active E03 calibration boundary, and foreground 3-hour watchdog execution rule.
- Verifies `E03.6` manifest completion, required launch counts, file-size audit, scheduler/timing evidence, and pass/fail summary; the strict R11 gate remains false.
- No strict R11 pass, memory-improvement, hardware-readiness, real-flight transfer, mission-success, or autonomy claim is made by this repeat.
```

```text
Outer-Loop Spatial Flow-Belief Memory Governor v8.15
```

```text
Python control-simulation pipeline

- Adds full dense R11 repeat `E03.7` using explicit E03 roots, run ID 9, seed 117, 20 outer cases per ladder, 4000 final held-out launches, and 34400 history launches.
- Keeps the same R11 eight-ladder schedule, no reduced smoke substitution, process-worker dense execution, and bounded evidence interpretation.
- Verifies `E03.7` manifest completion, required launch counts, file-size audit, scheduler/timing evidence, and pass/fail summary; the strict R11 gate remains false.
- No strict R11 pass, memory-improvement, hardware-readiness, real-flight transfer, mission-success, or autonomy claim is made by this repeat.
```

```text
Outer-Loop Spatial Flow-Belief Memory Governor v8.16
```

```text
Python control-simulation pipeline

- Adds full dense R11 repeat `E03.8` using explicit E03 roots, run ID 10, seed 118, 20 outer cases per ladder, 4000 final held-out launches, and 34400 history launches.
- Stops the repeat loop after `E03.8` because elapsed real time from the initial docs audit exceeded the 8-hour launch threshold, so `E03.9` was not consumed.
- Regenerates matched true-neutral/open-loop diagnostics for `E03.1` through `E03.8` and writes the separate aggregate appendix table at `03_Control/A_figures/R11_E03_appendix_tables/e03_r11_speed_tables_e03_1_to_e03_8.tex`.
- Verifies all completed repeats have complete manifests, required launch counts, file-size audits, scheduler/timing evidence, and bounded pass/fail summaries; the strict R11 gate remains false.
- No strict R11 pass, memory-improvement, hardware-readiness, real-flight transfer, mission-success, or autonomy claim is made by this repeat set.
```

## Claim Boundary

The `E03.1` through `E03.8` repeat set is the current held-out validation
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
- Manifest, pass/fail, file-size, and scheduler/timing audits for `E03.1` through `E03.8`
- `python 03_Control/01_Plotting/run_r11_balanced_neutral_baseline_figures.py` for `E03.5`, `E03.6`, `E03.7`, and `E03.8`
- `python 03_Control/01_Plotting/run_r11_appendix_speed_tables.py` for the `E03.1` through `E03.8` aggregate
- Aggregate summary schema check: `960` rows, `source_validation_run_labels=E03.1;E03.2;E03.3;E03.4;E03.5;E03.6;E03.7;E03.8`, `true_neutral_open_loop` present, and `8` R11 ladders present
- Final docs audit for active `.md` and `.txt` guidance
- `python -m pytest 03_Control/tests/test_v53_algorithm_contract.py::test_v53_stage_contract_is_r5_r7_r8_r10_r11_with_r6_archived_and_r9_internal 03_Control/tests/test_v53_algorithm_contract.py::test_v53_r10_and_r11_changed_case_randomisation_semantics_match 03_Control/tests/test_v53_algorithm_contract.py::test_v53_r11_full_safe_success_gate_catches_safe_exit_only_passes --basetemp .pytest_tmp_r11_repeat_e03_8 -p no:cacheprovider`
- `git diff --check`
