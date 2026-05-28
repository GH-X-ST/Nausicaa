# LQR Controller Trajectory Audit

- Project title version: `LQR-Stabilised Contextual Primitive v5.20`
- Audit version: `lqr_controller_trajectory_audit_v2_continuous_command_history`
- Status: `complete`
- Audit case mode: `same_start_comparison`
- Duration per selected trace: `0.6` s
- Controller update period: `0.02` s
- Primitive reschedule period: `0.1` s
- Shared start: `inflight_nominal` `W0` `dry_air` seed offset `0`.
- Manual centred-state override: x/y/z = `2.0`, `2.2`, `2.2` m; u = `5.5` m/s.
- Full-duration selected traces: `4` / `4`
- Floor failures: `0`
- Wall failures: `0`
- Maximum selected altitude loss: `0.452` m

Selected cases:
- `same_start_glide` `glide` candidate `1` duration `0.600` s, dz `-0.421` m, speed `5.50->5.43` m/s, yaw `-4.17` deg, sat `0.63`, termination `completed_full_duration`.
- `same_start_mild_turn_left` `mild_turn_left` candidate `0` duration `0.600` s, dz `-0.452` m, speed `5.50->5.50` m/s, yaw `-5.73` deg, sat `0.80`, termination `completed_full_duration`.
- `same_start_mild_turn_right` `mild_turn_right` candidate `7` duration `0.600` s, dz `-0.419` m, speed `5.50->5.45` m/s, yaw `+1.49` deg, sat `0.67`, termination `completed_full_duration`.
- `same_start_energy_retaining_bank` `energy_retaining_bank` candidate `7` duration `0.600` s, dz `-0.361` m, speed `5.50->5.35` m/s, yaw `+1.59` deg, sat `0.63`, termination `completed_full_duration`.

Claim boundary: this is a lightweight controller sanity audit only. It is not R5 dense evidence, not R7 validation, and not a mission/autonomy/real-flight claim.
