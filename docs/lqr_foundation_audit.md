# LQR Foundation Audit

Date: 2026-05-25

Status: retained historical audit. This file is still useful as a record of the
first LQR foundation cleanup, but it is not the current move-on plan. Use
`docs/Glider_Control_Project_Plan.md`, `docs/Daily_Schedule.txt`,
`docs/Skills.md`, `docs/Python Coding Instruction.txt`,
`docs/MATLAB Coding.txt`, `docs/housekeeping_and_naming_rules.md`, and
`docs/local_validation_environment.md` as the current source of truth when this
audit differs from the latest workflow.

## Current Contract Snapshot

- Active controller family: time-invariant LQR-stabilised
  primitive-controller variants.
- Current primitive horizon contract: `finite_horizon_s = 0.100`, with
  5 controller-input slots at a 20 ms controller update period.
- Active primitive labels are exactly: `glide`, `recovery`, `lift_entry`,
  `lift_dwell_arc`, `mild_turn_left`, `mild_turn_right`,
  `energy_retaining_bank`, and `safe_exit_or_recovery_handoff`.
- Controller identity is carried by stable `controller_id` and
  `primitive_variant_id` values. W2, W3, post-W3 processing, governor
  calibration, and repeated-launch validation must not mutate Q/R, K,
  references, horizons, entry roles, or IDs in place.
- Forbidden active methods: TVLQR, PD/PID fallback, bounded-feedback fallback,
  selected-controller registry, 12--24 W0/W1 shortlist, pre-W3 clustering, and
  LQR-tree/funnel/ROA claims unless a future formal proof exists.

## retired_not_active Audit

- `03_Control/05_Results/feedback_contextual_v1_4/` was moved to
  `03_Control/99_Archive/retired_pd_contextual_v1_4/results/`.
- Retired feedback-contextual runner stubs were removed from
  `03_Control/04_Scenarios`; only the archived result root remains under
  `03_Control/99_Archive`.
- retired_not_active old active functions and strings:
  `feedback_mode_for_primitive`, `primitive_command_norm`,
  `_feedback_command_norm`, `_primitive_command_template_norm`,
  `bounded_local_feedback`, `state_and_context_feedback`,
  `contextual_feedback_placeholder`, `model_backed_feedback`,
  `feedback_rollout_candidate`, and `diagnostic_model_rollout`.
- Active primitive control must route through LQR primitive-controller paths and
  explicitly labelled blocked synthesis paths, never through an old feedback
  substitution branch.

Valid `context`, `environment context`, and `environment-conditioned`
terminology remains allowed where it describes environment features rather than
the retired controller.

## Historical LQR Synthesis Snapshot

The details below describe the May 2026 foundation snapshot. They are retained
for traceability only and do not clear the current 0.10 s W0/W1 rerun gate.

- Reduced mask: `phi, theta, psi, u, v, w, p, q, r, delta_a, delta_e, delta_r`.
- Expansion: each controller stored a `3x15` gain with zero gains for `x_w`,
  `y_w`, and `z_w`.
- All eight nominal controllers solved reduced-order CARE on 2026-05-23.
- Full-state CARE status was recorded per primitive as `solved` or
  `unsuitable_use_reduced_order`.
- CARE residuals were all below `1e-8`.
- Sampled-data spectral radii at rollout `dt` were below `1.0`.
- Each controller row recorded checksum, controllability ranks, closed-loop
  eigenvalue summary, sampled-data status, command clipping summary, and
  latency/actuator-lag survival status.

## Historical Smoke Evidence

The old simulated W0/W1 smoke root below is no longer active move-on evidence.
The `r6/tune_100` folder name is a historical run identifier, not the preferred
naming pattern for new evidence roots:

```text
03_Control/05_Results/lqr_contextual_v1_0/r6/tune_100/
```

Observed historical smoke coverage:

- 500 total rows.
- W0/W1 split: 250 W0, 250 W1.
- Primitive row counts: `glide` 63, `recovery` 63, `lift_entry` 63,
  `lift_dwell_arc` 63, `mild_turn_left` 62, `mild_turn_right` 62,
  `energy_retaining_bank` 62, `safe_exit_or_recovery_handoff` 62.
- Start-state mixture: `launch_gate` 200, `inflight_nominal` 125,
  `inflight_lift_region` 75, `inflight_boundary_near` 50,
  `inflight_recovery_edge` 50.
- File-size audit: all generated files under 100 MB.

That smoke used old result naming and does not satisfy the current 0.10 s
primitive horizon, post-v4.10 archive, four-case post-W3 library-size, or
repeated-launch validation gates.

## Current Validation Baseline

Run current checks through the repo-local `.venv`:

```powershell
.\.venv\Scripts\python.exe -m py_compile 03_Control/02_Inner_Loop/*.py 03_Control/03_Primitives/*.py 03_Control/04_Scenarios/*.py
.\.venv\Scripts\python.exe -m pytest -q 03_Control/tests --basetemp .codex_run_logs\pytest_tmp -o cache_dir=.codex_run_logs\pytest_cache
.\.venv\Scripts\python.exe 03_Control/04_Scenarios/run_w01_w2_w3_contract_audit.py
git diff --check
```

Current source-audit, dense-run, post-W3, governor, and repeated-launch
commands are stage-specific and must follow the latest plan in
`docs/Daily_Schedule.txt`.
