# LQR Controller Trajectory Audit

- Project title version: `LQR-Stabilised Contextual Primitive v5.20`
- Audit version: `lqr_controller_trajectory_audit_v1`
- Status: `complete`
- Duration per selected trace: `0.4` s
- Controller update period: `0.02` s
- Primitive reschedule period: `0.1` s
- Full-duration selected traces: `6` / `8`
- Floor failures: `0`
- Wall failures: `2`
- Maximum selected altitude loss: `0.352` m

Selected cases:
- `glide_launch_dry` `glide` candidate `3` duration `0.400` s, dz `+0.059` m, speed `7.96->7.49` m/s, yaw `-1.76` deg, sat `0.25`, termination `completed_full_duration`.
- `lift_entry_single` `lift_entry` candidate `7` duration `0.400` s, dz `+0.137` m, speed `5.16->4.89` m/s, yaw `+7.97` deg, sat `0.45`, termination `completed_full_duration`.
- `lift_dwell_four` `lift_dwell_arc` candidate `3` duration `0.400` s, dz `+0.160` m, speed `6.36->5.81` m/s, yaw `+2.34` deg, sat `0.30`, termination `completed_full_duration`.
- `mild_left_nominal` `mild_turn_left` candidate `0` duration `0.260` s, dz `-0.313` m, speed `6.18->6.44` m/s, yaw `+2.80` deg, sat `0.62`, termination `wall_boundary_exit_retained`.
- `mild_right_nominal` `mild_turn_right` candidate `0` duration `0.400` s, dz `+0.281` m, speed `6.58->5.81` m/s, yaw `+9.57` deg, sat `0.70`, termination `completed_full_duration`.
- `energy_bank_four` `energy_retaining_bank` candidate `3` duration `0.340` s, dz `+0.277` m, speed `5.96->5.43` m/s, yaw `-15.47` deg, sat `0.29`, termination `wall_boundary_exit_retained`.
- `recovery_edge_dry` `recovery` candidate `0` duration `0.400` s, dz `-0.352` m, speed `3.41->3.69` m/s, yaw `-1.63` deg, sat `0.35`, termination `completed_full_duration`.
- `safe_exit_boundary_single` `safe_exit_or_recovery_handoff` candidate `3` duration `0.400` s, dz `+0.118` m, speed `5.22->4.67` m/s, yaw `-0.70` deg, sat `0.55`, termination `completed_full_duration`.

Claim boundary: this is a lightweight controller sanity audit only. It is not R5 dense evidence, not R7 validation, and not a mission/autonomy/real-flight claim.
