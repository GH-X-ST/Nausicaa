# LQR Controller Trajectory Audit

- Project title version: `LQR-Stabilised Contextual Primitive v5.20`
- Audit version: `lqr_controller_trajectory_audit_v2_continuous_command_history`
- Status: `complete`
- Audit case mode: `same_start_comparison`
- Duration per selected trace: `0.6` s
- Controller update period: `0.02` s
- Primitive reschedule period: `0.1` s
- Shared start: `inflight_nominal` `W0` `dry_air` seed offset `0`.
- Manual centred-state override: x/y/z = `2.0`, `2.2`, `2.2` m; u = `` m/s.
- Full-duration selected traces: `8` / `8`
- Floor failures: `0`
- Wall failures: `0`
- Maximum selected altitude loss: `0.709` m

Selected cases:
- `same_start_glide` `glide` candidate `0` duration `0.600` s, dz `-0.692` m, speed `3.91->4.14` m/s, yaw `-6.19` deg, sat `0.60`, termination `completed_full_duration`.
- `same_start_recovery` `recovery` candidate `6` duration `0.600` s, dz `-0.695` m, speed `3.91->4.07` m/s, yaw `-6.17` deg, sat `0.73`, termination `completed_full_duration`.
- `same_start_lift_entry` `lift_entry` candidate `6` duration `0.600` s, dz `-0.694` m, speed `3.91->4.06` m/s, yaw `-5.69` deg, sat `0.73`, termination `completed_full_duration`.
- `same_start_lift_dwell_arc` `lift_dwell_arc` candidate `6` duration `0.600` s, dz `-0.660` m, speed `3.91->3.99` m/s, yaw `-4.33` deg, sat `0.63`, termination `completed_full_duration`.
- `same_start_mild_turn_left` `mild_turn_left` candidate `7` duration `0.600` s, dz `-0.709` m, speed `3.91->4.16` m/s, yaw `-5.83` deg, sat `0.67`, termination `completed_full_duration`.
- `same_start_mild_turn_right` `mild_turn_right` candidate `6` duration `0.600` s, dz `-0.662` m, speed `3.91->4.02` m/s, yaw `-4.73` deg, sat `0.63`, termination `completed_full_duration`.
- `same_start_energy_retaining_bank` `energy_retaining_bank` candidate `0` duration `0.600` s, dz `-0.664` m, speed `3.91->4.04` m/s, yaw `-4.68` deg, sat `0.60`, termination `completed_full_duration`.
- `same_start_safe_exit_or_recovery_handoff` `safe_exit_or_recovery_handoff` candidate `6` duration `0.600` s, dz `-0.694` m, speed `3.91->4.06` m/s, yaw `-5.99` deg, sat `0.80`, termination `completed_full_duration`.

Claim boundary: this is a lightweight controller sanity audit only. It is not R5 dense evidence, not R7 validation, and not a mission/autonomy/real-flight claim.
