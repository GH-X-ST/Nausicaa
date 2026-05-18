# Agile Turn Family Comparison Report

This W0/no-wind evidence pass compares reusable speed-retaining turn families.
Existing run-002 high-alpha pitch-brake/perch-like evidence is preserved as archived boundary evidence only; retired speed-collapse branches are not active reusable candidates in this comparison.

No OCP, TVLQR, governor, outer loop, updraft validation, real-flight, hardware,
or high-incidence validation claim is made from this pass.

## Command Path

- Requested command: `u_norm_requested`.
- Applied command: `u_norm_applied`, clipped to the normalised contract.
- Plant command: `delta_cmd_rad`.
- `rk4_step` and `state_derivative` receive physical radian commands only.

## Acceptance Gates

- Strict success heading gate: `0.8 * target_heading_deg`.
- Useful recoverable heading gate: `0.6 * target_heading_deg`, or `15 deg` for the `30 deg` target.
- Strict terminal/min speed: `5.0` / `4.0` m/s.
- Useful terminal/min speed: `4.5` / `3.8` m/s.

## Target Summary

| target_deg | selected_family | horizon_s | class | heading_deg | terminal_speed_m_s | reason |
| --- | --- | ---: | --- | ---: | ---: | --- |
| 15 | canyon_steep_bank | 0.80 | useful_recoverable_candidate | 21.165 | 6.553 | useful_recoverable_candidate |
| 30 | canyon_steep_bank | 0.76 | useful_recoverable_candidate | 18.732 | 6.480 | useful_recoverable_candidate |

## Family Status

| family | status | failure or retention cause | best_heading_deg | best_terminal_speed_m_s |
| --- | --- | --- | ---: | ---: |
| canyon_steep_bank | selected_for_next_stage | recoverable_at_30_under_current_gates | 21.165 | 6.553 |
| wingover_lite | retained_as_thesis_discussion_evidence | useful_at_15_only_not_ready_for_45_60_escalation | 15.350 | 5.752 |
| bank_yaw_energy_retaining | retained_as_thesis_discussion_evidence | useful_at_15_only_not_ready_for_45_60_escalation | 11.868 | 6.417 |

## Escalation

- Escalation allowed from 30 deg evidence: `True`.
- Escalation targets run: `[]`.
- Escalation reason: `not_requested_default_15_30_only`.
