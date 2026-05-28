# LQR Controller Trajectory Audit

- Project title version: `LQR-Stabilised Contextual Primitive v5.20`
- Audit version: `lqr_controller_trajectory_audit_v2_continuous_command_history`
- Status: `complete`
- Audit case mode: `same_start_comparison`
- Duration per selected trace: `0.8` s
- Controller update period: `0.02` s
- Primitive reschedule period: `0.1` s
- Shared start: `inflight_nominal` `W0` `dry_air` seed offset `0`.
- Manual centred-state override: x/y/z = `2.0`, `2.2`, `2.2` m; u = `5.5` m/s.
- Full-duration selected traces: `8` / `8`
- Floor failures: `0`
- Wall failures: `0`
- Maximum selected altitude loss: `0.431` m

Selected cases:
- `same_start_glide` `glide` candidate `1` duration `0.800` s, dz `-0.390` m, speed `5.50->5.02` m/s, yaw `-7.40` deg, sat `0.65`, termination `completed_full_duration`.
- `same_start_recovery` `recovery` candidate `7` duration `0.800` s, dz `-0.268` m, speed `5.50->4.59` m/s, yaw `-8.18` deg, sat `0.70`, termination `completed_full_duration`.
- `same_start_lift_entry` `lift_entry` candidate `0` duration `0.800` s, dz `-0.307` m, speed `5.50->4.74` m/s, yaw `-8.70` deg, sat `0.65`, termination `completed_full_duration`.
- `same_start_lift_dwell_arc` `lift_dwell_arc` candidate `5` duration `0.800` s, dz `-0.167` m, speed `5.50->4.12` m/s, yaw `-1.95` deg, sat `0.70`, termination `completed_full_duration`.
- `same_start_mild_turn_left` `mild_turn_left` candidate `7` duration `0.800` s, dz `-0.431` m, speed `5.50->5.11` m/s, yaw `-8.88` deg, sat `0.80`, termination `completed_full_duration`.
- `same_start_mild_turn_right` `mild_turn_right` candidate `1` duration `0.800` s, dz `-0.235` m, speed `5.50->4.52` m/s, yaw `-1.99` deg, sat `0.75`, termination `completed_full_duration`.
- `same_start_energy_retaining_bank` `energy_retaining_bank` candidate `7` duration `0.800` s, dz `-0.279` m, speed `5.50->4.74` m/s, yaw `+1.59` deg, sat `0.68`, termination `completed_full_duration`.
- `same_start_safe_exit_or_recovery_handoff` `safe_exit_or_recovery_handoff` candidate `0` duration `0.800` s, dz `-0.299` m, speed `5.50->4.69` m/s, yaw `-9.33` deg, sat `0.72`, termination `completed_full_duration`.

Claim boundary: this is a lightweight controller sanity audit only. It is not R5 dense evidence, not R7 validation, and not a mission/autonomy/real-flight claim.
