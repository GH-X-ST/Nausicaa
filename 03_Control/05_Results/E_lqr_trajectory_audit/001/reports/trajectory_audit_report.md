# LQR Controller Trajectory Audit

- Project title version: `LQR-Stabilised Contextual Primitive v5.20`
- Audit version: `lqr_controller_trajectory_audit_v1`
- Status: `complete`
- Duration per selected trace: `0.8` s
- Controller update period: `0.02` s
- Primitive reschedule period: `0.1` s
- Full-duration selected traces: `1` / `8`
- Floor failures: `0`
- Wall failures: `7`
- Maximum selected altitude loss: `0.929` m

Selected cases:
- `glide_launch_dry` `glide` candidate `3` duration `0.720` s, dz `-0.166` m, speed `7.96->7.48` m/s, yaw `-10.91` deg, sat `0.22`, termination `wall_boundary_exit_retained`.
- `lift_entry_single` `lift_entry` candidate `7` duration `0.540` s, dz `+0.021` m, speed `5.16->5.02` m/s, yaw `+13.50` deg, sat `0.56`, termination `wall_boundary_exit_retained`.
- `lift_dwell_four` `lift_dwell_arc` candidate `3` duration `0.440` s, dz `+0.178` m, speed `6.36->5.75` m/s, yaw `+2.35` deg, sat `0.27`, termination `wall_boundary_exit_retained`.
- `mild_left_nominal` `mild_turn_left` candidate `0` duration `0.260` s, dz `-0.313` m, speed `6.18->6.44` m/s, yaw `+2.80` deg, sat `0.62`, termination `wall_boundary_exit_retained`.
- `mild_right_nominal` `mild_turn_right` candidate `0` duration `0.680` s, dz `+0.337` m, speed `6.58->5.50` m/s, yaw `+18.15` deg, sat `0.62`, termination `wall_boundary_exit_retained`.
- `energy_bank_four` `energy_retaining_bank` candidate `3` duration `0.340` s, dz `+0.277` m, speed `5.96->5.43` m/s, yaw `-15.47` deg, sat `0.29`, termination `wall_boundary_exit_retained`.
- `recovery_edge_dry` `recovery` candidate `3` duration `0.800` s, dz `-0.929` m, speed `3.41->3.12` m/s, yaw `-12.44` deg, sat `0.62`, termination `completed_full_duration`.
- `safe_exit_boundary_single` `safe_exit_or_recovery_handoff` candidate `3` duration `0.480` s, dz `+0.141` m, speed `5.22->4.55` m/s, yaw `-2.23` deg, sat `0.46`, termination `wall_boundary_exit_retained`.

Claim boundary: this is a lightweight controller sanity audit only. It is not R5 dense evidence, not R7 validation, and not a mission/autonomy/real-flight claim.
