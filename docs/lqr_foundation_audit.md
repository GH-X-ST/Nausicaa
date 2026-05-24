# LQR Foundation Audit

Date: 2026-05-23

Status: retained historical audit. Use `docs/Skills.md`,
`docs/Glider_Control_Project_Plan.md`,
`docs/housekeeping_and_naming_rules.md`, and
`docs/local_validation_environment.md` as the current source of truth when this
audit's old command spelling or run-folder labels differ from the updated
housekeeping rules.

## Active Contract

- Active controller family: time-invariant LQR only.
- Forbidden active methods: TVLQR, PD/PID, bounded-feedback fallback, LQR-tree/funnel branding.
- Active primitive labels are exactly: `glide`, `recovery`, `lift_entry`, `lift_dwell_arc`, `mild_turn_left`, `mild_turn_right`, `energy_retaining_bank`, `safe_exit_or_recovery_handoff`.
- Controller variants are represented by `controller_id`, not by new `primitive_id` labels.

## retired_not_active Audit

- `03_Control/05_Results/feedback_contextual_v1_4/` was moved to `03_Control/99_Archive/retired_pd_contextual_v1_4/results/`.
- Retired feedback-contextual runner stubs were removed from `03_Control/04_Scenarios`; only the archived result root remains under `03_Control/99_Archive`.
- retired_not_active old active functions and strings: `feedback_mode_for_primitive`, `primitive_command_norm`, `_feedback_command_norm`, `_primitive_command_template_norm`, `bounded_local_feedback`, `state_and_context_feedback`, `contextual_feedback_placeholder`, `model_backed_feedback`, `feedback_rollout_candidate`, and `diagnostic_model_rollout`.
- Active primitive control now routes through `primitive_lqr_command`, `lqr_mode_for_primitive`, and `lqr_controller_for_primitive_id`.
- Active rollout evidence now uses `model_backed_lqr`, `lqr_rollout_candidate`, or explicit `blocked_lqr_synthesis`. No old feedback substitution branch is present.

Valid `context`, `environment context`, and `environment-conditioned` terminology remains allowed where it describes environment features rather than the retired controller.

## LQR Synthesis Snapshot

- Reduced mask: `phi, theta, psi, u, v, w, p, q, r, delta_a, delta_e, delta_r`.
- Expansion: each controller stores a `3x15` gain with zero gains for `x_w`, `y_w`, and `z_w`.
- All eight nominal controllers solved reduced-order CARE on 2026-05-23.
- Full-state CARE status is recorded per primitive as `solved` or `unsuitable_use_reduced_order`.
- CARE residuals were all below `1e-8`.
- Sampled-data spectral radii at rollout `dt` were below `1.0`.
- Each controller row records checksum, controllability ranks, closed-loop eigenvalue summary, sampled-data status, command clipping summary, and latency/actuator-lag survival status.

## Smoke Evidence

Executed simulated W0/W1 smoke. The `r6/tune_100` folder name below is a
historical run identifier, not the preferred naming pattern for new evidence
roots:

```text
.\.venv\Scripts\python.exe 03_Control/04_Scenarios/run_lqr_tuning_sweep.py --rows 500 --seed 1 --candidate-count 16 --paired-tests-per-candidate 50 --candidate-chunk-size 125 --workers 1 --max-workers 1 --storage-format csv_gz --compression-level 1 --stop-after-chunks 4 --repair-incomplete
```

Smoke root: `03_Control/05_Results/lqr_contextual_v1_0/r6/tune_100/`

Observed smoke coverage:

- 500 total rows.
- W0/W1 split: 250 W0, 250 W1.
- Primitive row counts: `glide` 63, `recovery` 63, `lift_entry` 63, `lift_dwell_arc` 63, `mild_turn_left` 62, `mild_turn_right` 62, `energy_retaining_bank` 62, `safe_exit_or_recovery_handoff` 62.
- Start-state mixture: `launch_gate` 200, `inflight_nominal` 125, `inflight_lift_region` 75, `inflight_boundary_near` 50, `inflight_recovery_edge` 50.
- LQR synthesis status: 500 solved rows, 0 surrogate substitutions.
- File-size audit: all generated files under 100 MB.
- Manifest flags: `W0_W1_tune_controller_ids=true`, `W2_W3_replay_only=true`.

## Historical Go/No-Go Commands Passed

Run the same checks through the repo-local `.venv` for current validation:

```text
.\.venv\Scripts\python.exe -m compileall 03_Control
.\.venv\Scripts\python.exe -m pytest -q 03_Control/tests --basetemp .codex_run_logs\pytest_tmp -o cache_dir=.codex_run_logs\pytest_cache
.\.venv\Scripts\python.exe 03_Control/04_Scenarios/run_active_contract_audit.py
.\.venv\Scripts\python.exe 03_Control/04_Scenarios/run_lqr_tuning_sweep.py --dry-run-schedule --stop-after-chunks 1 --workers 1 --max-workers 1
.\.venv\Scripts\python.exe 03_Control/04_Scenarios/run_lqr_contextual_archive.py --dry-run-schedule --stop-after-chunks 1 --workers 1 --max-workers 1
.\.venv\Scripts\python.exe 03_Control/04_Scenarios/run_lqr_w2_replay.py --dry-run-schedule --stop-after-chunks 1 --workers 1 --max-workers 1
.\.venv\Scripts\python.exe 03_Control/04_Scenarios/run_lqr_w3_generalisation.py --dry-run-schedule --stop-after-chunks 1 --workers 1 --max-workers 1
```

Final active-code search found no retired controller/fallback tokens in `03_Control/02_Inner_Loop`, `03_Control/03_Primitives`, `03_Control/04_Scenarios`, or `03_Control/tests`, excluding `03_Control/99_Archive` and generated `03_Control/05_Results`.
