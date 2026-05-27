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

- `launch_capture_glide_stabilise: rows=9600, accepted=3812, weak=1136, continuation_valid=4948, terminal_useful=0, hard_failure=4652, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `launch_capture_lift_seek: rows=9600, accepted=3810, weak=1141, continuation_valid=4951, terminal_useful=0, hard_failure=4649, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `launch_capture_energy_build: rows=9600, accepted=3812, weak=1139, continuation_valid=4951, terminal_useful=0, hard_failure=4649, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `launch_capture_shallow_left: rows=9600, accepted=3813, weak=1136, continuation_valid=4949, terminal_useful=0, hard_failure=4651, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `launch_capture_shallow_right: rows=9600, accepted=3812, weak=1137, continuation_valid=4949, terminal_useful=0, hard_failure=4651, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`
- `launch_capture_safe_handoff: rows=9600, accepted=3812, weak=1141, continuation_valid=4953, terminal_useful=0, hard_failure=4647, rejected=0, blocked=0, entry_role_rejections=0, gate_passed=True`

Blockers:

- `none`

Later validation stages are deliberately not claimed by this R5-only evidence pass.
