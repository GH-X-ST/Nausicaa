# Outer-Loop Spatial Flow-Belief Memory Governor v8.12

Date: 2026-06-07

## Scope

This note records the R11 E03 repeat-evidence workflow for the frozen
outer-loop spatial flow-belief memory governor. The workflow used wall-clock
time, starting at `2026-06-07T00:39:58.8217267+01:00`, immediately before the
initial docs audit. CPU time was not used for the repeat-loop stopping rule.

No controller code, primitive catalogue, R10 handoff config, launch-gate
contract, real-flight runtime contract, or scoring gate was changed by this
workflow. The only code change is table-generator summary metadata that records
which validation run labels were aggregated. The outputs are additional R11
validation roots, matched true-neutral/open-loop diagnostics, one separate
aggregate appendix table, and active-doc wording updates.

## Preflight

The worktree was checked before dense execution. The R11 runner and table
scripts passed syntax compilation:

- `03_Control/04_Scenarios/run_heldout_changed_case_validation.py`
- `03_Control/04_Scenarios/run_changed_case_validation.py`
- `03_Control/04_Scenarios/run_repeated_launch_learning_curve.py`
- `03_Control/01_Plotting/run_r11_appendix_speed_tables.py`
- `03_Control/01_Plotting/run_r11_balanced_neutral_baseline_figures.py`

The dry schedule probe used explicit E03 roots, not wrapper defaults, and
reported the required full-R11 unit:

- final held-out launches: `4000`
- history launches: `34400`

The existing `E03.1` reference root was physically explainable but not a strict
R11 pass. Its manifest was complete with `4000` final held-out launches and
`34400` history launches, and the file-size audit was push-compatible, but the
strict pass gate remained false because hard failures, floor/ceiling
violations, no-viable events, safe success, terminal-or-lift capture, and full
safe success did not meet the strict R11 thresholds.

## Dense R11 Repeats

Each full dense command ran in the foreground with a 3-hour watchdog
(`timeout_ms=10800000`). No reduced readiness smoke run was inserted before the
dense repeats.

| Run | Run ID | Seed | Wall Time | Status | Final | History | Strict Gate |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| `E03.2` | `4` | `112` | `8652.5 s` | complete | `4000` | `34400` | false |
| `E03.3` | `5` | `113` | `8456.1 s` | complete | `4000` | `34400` | false |
| `E03.4` | `6` | `114` | `8355.5 s` | complete | `4000` | `34400` | false |

The repeat loop launched `E03.4` because elapsed real time after `E03.3` was
about `4.789 h`, still below the `5.5 h` launch threshold. The loop stopped
after `E03.4` because elapsed real time was about `7.123 h`, so `E03.5` was not
started.

All completed repeats had complete manifests, the required launch counts,
file-size audits that allowed local evidence storage, and scheduler/timing
audits. The strict gate stayed false for every repeat, so these roots extend
the held-out evidence boundary but do not create a strict R11 pass claim.

## Neutral Diagnostics And Table

Matched true-neutral/open-loop diagnostics were regenerated for every aggregate
input:

- `03_Control/A_figures/R11_E03.1_balanced_neutral_baseline`
- `03_Control/A_figures/R11_E03.2_balanced_neutral_baseline`
- `03_Control/A_figures/R11_E03.3_balanced_neutral_baseline`
- `03_Control/A_figures/R11_E03.4_balanced_neutral_baseline`

Each neutral diagnostic completed with `160` neutral cases and `8` figures.

The separate aggregate appendix table was generated without overwriting the
existing E03.1 table:

- `03_Control/A_figures/R11_E03_appendix_tables/e03_r11_speed_tables_e03_1_to_e03_4.tex`
- `03_Control/A_figures/R11_E03_appendix_tables/e03_r11_speed_table_summary_e03_1_to_e03_4.csv`
- `03_Control/A_figures/R11_E03_appendix_tables/e03_controller_minus_open_loop_e03_1_to_e03_4.csv`

The summary includes all completed run labels (`E03.1` through `E03.4`), all
eight R11 ladder IDs, the R11 policy rows, and the matched
`true_neutral_open_loop` rows. The LaTeX table renders the neutral/open-loop
label as thesis-facing `Open`, while the summary CSV keeps the explicit
`true_neutral_open_loop` policy ID.

## Documentation Audit

All active `docs/**/*.md` and `docs/**/*.txt` guidance files were audited
before dense execution and updated again after the new results and aggregate
table were generated. Active guidance now states that the current held-out
validation boundary is the `E03.1` through `E03.4` repeat set and cites the new
aggregate table path.

Historical audit notes under `docs/code_audits` were preserved as historical
records unless this note created a new active audit record.

## PR-Style Messages

```text
Outer-Loop Spatial Flow-Belief Memory Governor v8.10
```

```text
Python control-simulation pipeline

- Adds full dense R11 repeat `E03.2` using explicit E03 roots, run ID 4, seed 112, 20 outer cases per ladder, 4000 final held-out launches, and 34400 history launches.
- Keeps the frozen R10 handoff, primitive library, spatial flow-belief memory contract, launch-gate contract, and 0.100 s primitive-boundary timing audit unchanged.
- Verifies `E03.2` manifest completion, required launch counts, file-size audit, scheduler/timing evidence, and pass/fail summary; the strict R11 gate remains false.
- No strict R11 pass, memory-improvement, hardware-readiness, real-flight transfer, mission-success, or autonomy claim is made by this repeat.
```

```text
Outer-Loop Spatial Flow-Belief Memory Governor v8.11
```

```text
Python control-simulation pipeline

- Adds full dense R11 repeat `E03.3` using explicit E03 roots, run ID 5, seed 113, 20 outer cases per ladder, 4000 final held-out launches, and 34400 history launches.
- Extends the repeated full-R11 evidence set while preserving the frozen governor, active E03 calibration boundary, and foreground 3-hour watchdog execution rule.
- Verifies `E03.3` manifest completion, required launch counts, file-size audit, scheduler/timing evidence, and pass/fail summary; the strict R11 gate remains false.
- No strict R11 pass, memory-improvement, hardware-readiness, real-flight transfer, mission-success, or autonomy claim is made by this repeat.
```

```text
Outer-Loop Spatial Flow-Belief Memory Governor v8.12
```

```text
Python control-simulation pipeline

- Adds full dense R11 repeat `E03.4` using explicit E03 roots, run ID 6, seed 114, 20 outer cases per ladder, 4000 final held-out launches, and 34400 history launches.
- Stops the repeat loop after `E03.4` because elapsed real time from the initial docs audit exceeded the 5.5-hour launch threshold, so `E03.5` was not consumed.
- Regenerates matched true-neutral/open-loop diagnostics for `E03.1` through `E03.4` and writes the separate aggregate appendix table at `03_Control/A_figures/R11_E03_appendix_tables/e03_r11_speed_tables_e03_1_to_e03_4.tex`.
- Verifies all completed repeats have complete manifests, required launch counts, file-size audits, scheduler/timing evidence, and bounded pass/fail summaries; the strict R11 gate remains false.
- No strict R11 pass, memory-improvement, hardware-readiness, real-flight transfer, mission-success, or autonomy claim is made by this repeat set.
```

## Claim Boundary

The `E03.1` through `E03.4` repeat set is the current held-out validation
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
- Manifest, pass/fail, file-size, and scheduler/timing audits for `E03.1`, `E03.2`, `E03.3`, and `E03.4`
- `python 03_Control/01_Plotting/run_r11_balanced_neutral_baseline_figures.py` for `E03.1`, `E03.2`, `E03.3`, and `E03.4`
- `python 03_Control/01_Plotting/run_r11_appendix_speed_tables.py` for the `E03.1` through `E03.4` aggregate
- Aggregate summary schema check: `960` rows, `source_validation_run_labels=E03.1;E03.2;E03.3;E03.4`, `true_neutral_open_loop` present, and `8` R11 ladders present
- Final docs audit for active `.md` and `.txt` guidance
- `python -m pytest 03_Control/tests/test_v53_algorithm_contract.py::test_v53_stage_contract_is_r5_r7_r8_r10_r11_with_r6_archived_and_r9_internal 03_Control/tests/test_v53_algorithm_contract.py::test_v53_r10_and_r11_changed_case_randomisation_semantics_match 03_Control/tests/test_v53_algorithm_contract.py::test_v53_r11_full_safe_success_gate_catches_safe_exit_only_passes --basetemp .pytest_tmp_r11_repeat -p no:cacheprovider`
- `git diff --check`
