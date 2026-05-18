# Agile Turn Precision Ladder Cleanup Report

This W0/no-wind pass enforces terminal-heading target bands for commandable agile-turn labels.
Archived high-alpha/perch-like boundary reference is preserved exactly as negative evidence; it is not an active reusable agile-turn family.

No OCP, TVLQR, governor, outer loop, updraft validation, real-flight, hardware,
or high-incidence validation claim is made from this pass.

## Target Summary

| target_deg | commandable | selected_family | selected_horizon_s | best_safe_partial | best_boundary | escalation_reason |
| --- | --- | --- | ---: | --- | --- | --- |
| 15 | True | canyon_steep_bank | 0.76 | canyon_steep_bank_t015_h036_a115_q110 | bank_yaw_energy_retaining_t015_h090_a115_q110 | 30deg_not_commandable_safe_partial_or_boundary_only |
| 30 | False |  |  | canyon_steep_bank_t030_h046_a115_q090 | canyon_steep_bank_t030_h100_a115_q100 | 30deg_not_commandable_safe_partial_or_boundary_only |

## Family Status

| family | status | best_class | best_terminal_heading_deg | limiter |
| --- | --- | --- | ---: | --- |
| canyon_steep_bank | selected_for_next_stage | commandable_target_candidate | 16.978 | target_band_commandable |
| wingover_lite | selected_for_next_stage | commandable_target_candidate | 13.806 | target_band_commandable |
| bank_yaw_energy_retaining | retained_as_thesis_discussion_evidence | accurate_boundary_evidence | 16.701 | safety_boundary_target_miss |

## Cleanup

- Archived boundary reference preserved: `True`.
- Old branch active: `False`.
- Fixed target ladder: `[15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 150.0, 180.0]`.
- No 20-degree bin: `True`.
- Command bridge: `u_norm_requested -> u_norm_applied -> delta_cmd_rad`.
