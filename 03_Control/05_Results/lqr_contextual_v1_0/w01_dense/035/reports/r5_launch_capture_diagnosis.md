# R5 Launch-Capture Diagnosis

- Project title version: `LQR-Stabilised Contextual Primitive v5.3`
- Status: `dry_run_schedule`
- Rows written: `0`
- Expected dense rows: `134400`
- Expected launch-gate rows per launch-capture primitive: `9600`
- R5 launch-aware decision: `R5_LAUNCH_AWARE_DENSE_INCOMPLETE_RESUME_REQUIRED`
- Regime labels: `launch_capture_from_launch_gate`, `inflight_from_nominal_or_lift_region`, `recovery_or_safe_exit_from_boundary_or_recovery_edge`.
- In-flight and recovery/safe-exit rows are diagnostics only and cannot satisfy launch-capture gates.

Launch-capture launch-gate summary:

- `missing_or_empty_r5_launch_capture_diagnosis`

Blockers:

- `dry_run_schedule_only_no_rollout_evidence`

Later validation stages are deliberately not claimed by this R5-only evidence pass.
