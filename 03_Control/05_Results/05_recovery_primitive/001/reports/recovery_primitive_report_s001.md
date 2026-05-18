# W0 Recovery Primitive Report

This is the first actual no-wind recovery primitive. It uses local
feedback in normalised command space and hands off to the existing
`glide_w0_nominal` entry contract as a proxy check.

It does not implement bank, agile reversal, OCP, TVLQR, governor,
outer-loop, Vicon, hardware, real-flight, or high-incidence validation.

## Batch Status

- Overall status: `pass`
- Required case success: `True`
- Optional failures: `[]`
- Diagnostic failures: `['recovery_w0_0p80_boundary']`
- Failure label: `success`
- Notes: `required_recovery_case_passed`

## Handoff Contract

- Terminal glide-entry proxy source: `build_glide_primitive_spec + evaluate_entry_set`
- The terminal proxy checks exact compatibility with the existing
  `glide_w0_nominal` entry set and does not relax that glide primitive.
- The terminal proxy is primitive-chaining evidence only, not proof of
  a governor, real-flight transfer, or recovery-to-glide hardware handoff.

## Command Path

- Trim command: physical radians from `solve_straight_trim()`.
- Trim bridge: `surface_rad_to_normalised_command`.
- Feedback correction: normalised command space only.
- Applied bridge: `u_norm_requested -> u_norm_applied -> delta_cmd_rad`.
- Plant input: `delta_cmd_rad`; raw normalised commands do not enter dynamics.

## Case Outcomes

### recovery_w0_moderate_attitude_rate

- Role: `required`
- Success: `True`
- Failure label: `success`
- Notes: `recovery_w0_local_feedback_handoff_to_glide_proxy`
- Duration: `0.64` s
- Terminal speed: `6.143048818339836` m/s
- Terminal x_w: `5.286061` m
- Terminal glide-entry x margin: `0.213939` m
- Terminal glide-entry proxy: `True`

### recovery_w0_low_speed_pitch_up

- Role: `optional`
- Success: `True`
- Failure label: `success`
- Notes: `recovery_w0_local_feedback_handoff_to_glide_proxy`
- Duration: `0.64` s
- Terminal speed: `5.679061252297` m/s
- Terminal x_w: `4.932764` m
- Terminal glide-entry x margin: `0.567236` m
- Terminal glide-entry proxy: `True`

### recovery_w0_sideslip_yaw_rate

- Role: `optional`
- Success: `True`
- Failure label: `success`
- Notes: `recovery_w0_local_feedback_handoff_to_glide_proxy`
- Duration: `0.64` s
- Terminal speed: `6.435190131835644` m/s
- Terminal x_w: `5.228109` m
- Terminal glide-entry x margin: `0.271891` m
- Terminal glide-entry proxy: `True`

### recovery_w0_high_rate_boundary

- Role: `diagnostic`
- Success: `True`
- Failure label: `success`
- Notes: `recovery_w0_local_feedback_handoff_to_glide_proxy`
- Duration: `0.64` s
- Terminal speed: `5.840029824546152` m/s
- Terminal x_w: `5.100636` m
- Terminal glide-entry x margin: `0.399364` m
- Terminal glide-entry proxy: `True`

### recovery_w0_0p80_boundary

- Role: `diagnostic`
- Success: `False`
- Failure label: `terminal_recovery_limited`
- Notes: `glide_entry_x_bound_limited`
- Duration: `0.8` s
- Terminal speed: `6.157376995924838` m/s
- Terminal x_w: `6.259237` m
- Terminal glide-entry x margin: `-0.759237` m
- Terminal glide-entry proxy: `False`

## Implementation Flags

- Actual recovery primitive implemented: `True`
- Actual glide primitive implemented: `True`
- Actual bank primitive implemented: `False`
- Actual agile reversal primitive implemented: `False`
- OCP implemented: `False`
- TVLQR implemented: `False`
- Governor implemented: `False`
- Outer loop implemented: `False`
- High-incidence validation claim: `False`
