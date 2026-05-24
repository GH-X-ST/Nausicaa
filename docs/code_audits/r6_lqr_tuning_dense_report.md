# R6 W0/W1 LQR Dense Tuning Report

Date: 2026-05-24

## Result

- Move-on decision: `R7 blocked`.
- Exact blocker: the dense R6 run completed, but the selected-controller registry contains `0` selected controllers. It contains rejected controller IDs for every active primitive, so there is no active selected LQR controller for R7 to consume.
- No method code was changed for this run.
- Dense outputs are local evidence under `03_Control/05_Results/lqr_contextual_v1_0/r6/tune_103`; raw partitions are not approved for commit by this report.

## Environment and Commands

- Interpreter: `.\.venv\Scripts\python.exe`.
- Output root: `03_Control/05_Results/lqr_contextual_v1_0/r6`.
- Run ID: `103`.
- Dry-run command:

```powershell
.\.venv\Scripts\python.exe 03_Control/04_Scenarios/run_lqr_tuning_sweep.py --run-id 103 --output-root 03_Control/05_Results/lqr_contextual_v1_0/r6 --rows 12800 --seed 103 --candidate-count 16 --paired-tests-per-candidate 50 --candidate-chunk-size 1000 --workers 8 --max-workers 8 --storage-format auto --compression-level 1 --resume --repair-incomplete --dry-run-schedule
```

- Dense command:

```powershell
.\.venv\Scripts\python.exe 03_Control/04_Scenarios/run_lqr_tuning_sweep.py --run-id 103 --output-root 03_Control/05_Results/lqr_contextual_v1_0/r6 --rows 12800 --seed 103 --candidate-count 16 --paired-tests-per-candidate 50 --candidate-chunk-size 1000 --workers 8 --max-workers 8 --storage-format auto --compression-level 1 --resume --repair-incomplete
```

- Resume verification reran the same dense command after completion.

## Validation

- `.\.venv\Scripts\python.exe -m compileall 03_Control` - passed.
- `.\.venv\Scripts\python.exe -m pytest -q 03_Control/tests --basetemp .codex_run_logs\pytest_tmp -o cache_dir=.codex_run_logs\pytest_cache` - passed: 166 tests.
- `.\.venv\Scripts\python.exe 03_Control/04_Scenarios/run_active_contract_audit.py` - passed before dense output was created.
- `git diff --check` - passed before dense output was created.

## Run Summary

- Requested rows: `12800`.
- Written rows: `12800`.
- Active primitives: `8`.
- Candidate count per primitive: `16`.
- Paired tests per candidate: `50`.
- W layers: `W0` and `W1`.
- Storage format: `csv_gz`.
- Chunk size: `1000`.
- Scheduled partitions: `13`.
- Written partitions: `13`.
- Initial dense execution: `13` complete chunks, process-pool execution with `8` workers.
- Resume verification: `13` chunks skipped, `0` partitions rewritten.
- Repair status: `--repair-incomplete` enabled; no corrupt chunks were found in the completed run.

## Evidence Status

- Registry status: `complete`.
- Registry claim status: `simulation_only_registry_complete`.
- Archive evidence status: `complete`.
- Evidence eligibility reason: `eligible_tuning_registry_complete`.
- Selected-controller JSON reports `selected_controller_count = 0` and `primitive_count = 8`.
- Registry row reconstruction from controller metadata succeeded for all `128` registry rows.
- Candidate metadata completeness check found no missing candidate labels, Q/R JSON, gain checksums, linearisation IDs, or controller IDs.

## Controller Status by Primitive

| Primitive | Selected | Rejected | Blocked |
|---|---:|---:|---:|
| `energy_retaining_bank` | 0 | 16 | 0 |
| `glide` | 0 | 16 | 0 |
| `lift_dwell_arc` | 0 | 16 | 0 |
| `lift_entry` | 0 | 16 | 0 |
| `mild_turn_left` | 0 | 16 | 0 |
| `mild_turn_right` | 0 | 16 | 0 |
| `recovery` | 0 | 16 | 0 |
| `safe_exit_or_recovery_handoff` | 0 | 16 | 0 |

## Outcome Summary

- Outcome rows: `1421` accepted, `2884` weak, `8495` failed.
- Boundary-use rows: `1698` continuation-valid, `2607` episode-terminal-useful, `8495` hard-failure.
- All `12800` rollout rows used executable LQR controllers.
- Main hard-failure labels were `xy_boundary_terminal`, `terminal_recovery_limited`, `floor_violation`, and `speed_low`.
- Rejection reason in the registry is `blocked_rollout_rows_present`, meaning no candidate passed the current all-start hard-gate selection rule.

## File-Size and Path Audit

- Largest generated file: `metrics/coverage_summary.csv`, `0.620210` MB.
- Largest table partition: `tables/lqr_tuning_rows/c00009.csv.gz`, `0.607274` MB.
- File-size audit: all files are below `75` MB and `100` MB.
- Path audit: all filename stems are under `64` characters, relative paths are under `140` characters, and `push_allowed` is true for audited files.

## Claim Boundary

This run is simulation-only R6 W0/W1 Q/R tuning evidence. It makes no dense thesis archive, R7 contextual archive, W2/W3 replay, hardware-readiness, real-flight transfer, robustness, mission-success, PD/PID fallback, TVLQR, MPC, LQR-tree, reachable-chain, online fan-layout branch, or non-LQR controller claim.

