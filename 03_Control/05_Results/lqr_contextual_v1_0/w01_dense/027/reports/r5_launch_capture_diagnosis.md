# R5 Launch-Capture Diagnosis

- Project title version: `LQR-Stabilised Contextual Primitive v5.3`
- Status: `complete`
- Rows written: `134400`
- Expected dense rows: `134400`
- Expected launch-gate rows per launch-capture primitive: `9600`
- R5 launch-aware decision: `R5_LAUNCH_AWARE_DENSE_PASSED_FOR_REVIEW`
- Regime labels: `launch_capture_from_launch_gate`, `inflight_from_nominal_or_lift_region`, `recovery_or_safe_exit_from_boundary_or_recovery_edge`.
- In-flight and recovery/safe-exit rows are diagnostics only and cannot satisfy launch-capture gates.

Launch-capture launch-gate summary:

- `launch_capture_glide_stabilise: rows=9600, accepted=3821, weak=1143, continuation_valid=4964, terminal_useful=0, hard_failure=4636, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `launch_capture_lift_seek: rows=9600, accepted=3816, weak=1144, continuation_valid=4960, terminal_useful=0, hard_failure=4640, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `launch_capture_energy_build: rows=9600, accepted=3825, weak=1131, continuation_valid=4956, terminal_useful=0, hard_failure=4644, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `launch_capture_shallow_left: rows=9600, accepted=3815, weak=1143, continuation_valid=4958, terminal_useful=0, hard_failure=4642, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `launch_capture_shallow_right: rows=9600, accepted=3816, weak=1138, continuation_valid=4954, terminal_useful=0, hard_failure=4646, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `launch_capture_safe_handoff: rows=9600, accepted=3813, weak=1147, continuation_valid=4960, terminal_useful=0, hard_failure=4640, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`

Blockers:

- `none`

Later validation stages are deliberately not claimed by this R5-only evidence pass.
