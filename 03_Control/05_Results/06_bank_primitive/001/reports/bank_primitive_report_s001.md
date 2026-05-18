# W0 Bank Primitive Report

This is the first actual W0 no-wind bank primitive and a baseline for
later updraft-encounter use. It is not an updraft robustness result,
not an agile reversal, not OCP, not TVLQR, not governor, not real
flight, and not high-incidence validation.

- Bank/updraft-encounter role: `w0_lateral_repositioning_baseline_only`

## Batch Status

- Overall status: `pass`
- Required case success: `True`
- Optional failures: `[]`
- Diagnostic failures: `['bank_w0_0p80_handoff_boundary']`
- Failure label: `success`
- Notes: `required_bank_case_passed`

## Geometry And Handoff Contract

- Default initial position: `(1.3, 2.2, 1.8)` m
- Terminal glide-entry proxy source: `build_glide_primitive_spec + evaluate_entry_set`
- The diagnostic 0.80 s boundary is handoff-limited only when the
  trajectory remains finite and true-safe while failing only the existing
  glide-entry x bound.

## Command Path

- Trim command: physical radians from `solve_straight_trim()`.
- Trim bridge: `surface_rad_to_normalised_command`.
- Feedback correction: normalised command space only.
- Applied bridge: `u_norm_requested -> u_norm_applied -> delta_cmd_rad`.
- Plant input: `delta_cmd_rad`; raw normalised commands do not enter dynamics.

## Case Outcomes

### bank_w0_left_mild

- Role: `required`
- Direction sign: `-1`
- Success: `True`
- Failure label: `success`
- Notes: `bank_w0_mild_lateral_repositioning_feedback`
- Duration: `0.6` s
- Lateral displacement: `-0.059759` m
- Terminal x_w: `5.183286` m
- Terminal true-safe x margin: `1.416714` m
- Terminal glide-entry x margin: `0.316714` m
- Terminal glide-entry proxy: `True`

### bank_w0_right_mild

- Role: `required`
- Direction sign: `1`
- Success: `True`
- Failure label: `success`
- Notes: `bank_w0_mild_lateral_repositioning_feedback`
- Duration: `0.6` s
- Lateral displacement: `0.055402` m
- Terminal x_w: `5.183407` m
- Terminal true-safe x margin: `1.416593` m
- Terminal glide-entry x margin: `0.316593` m
- Terminal glide-entry proxy: `True`

### bank_w0_left_sideslip_entry

- Role: `optional`
- Direction sign: `-1`
- Success: `True`
- Failure label: `success`
- Notes: `bank_w0_mild_lateral_repositioning_feedback`
- Duration: `0.6` s
- Lateral displacement: `-0.357755` m
- Terminal x_w: `5.113890` m
- Terminal true-safe x margin: `1.486110` m
- Terminal glide-entry x margin: `0.386110` m
- Terminal glide-entry proxy: `True`

### bank_w0_right_sideslip_entry

- Role: `optional`
- Direction sign: `1`
- Success: `True`
- Failure label: `success`
- Notes: `bank_w0_mild_lateral_repositioning_feedback`
- Duration: `0.6` s
- Lateral displacement: `0.352427` m
- Terminal x_w: `5.114573` m
- Terminal true-safe x margin: `1.485427` m
- Terminal glide-entry x margin: `0.385427` m
- Terminal glide-entry proxy: `True`

### bank_w0_0p80_handoff_boundary

- Role: `diagnostic`
- Direction sign: `1`
- Success: `False`
- Failure label: `terminal_recovery_limited`
- Notes: `glide_entry_x_bound_limited`
- Duration: `0.8` s
- Lateral displacement: `0.129015` m
- Terminal x_w: `6.477693` m
- Terminal true-safe x margin: `0.122307` m
- Terminal glide-entry x margin: `-0.977693` m
- Terminal glide-entry proxy: `False`

## Implementation Flags

- Actual bank primitive implemented: `True`
- Actual recovery primitive implemented: `True`
- Actual glide primitive implemented: `True`
- Actual agile reversal primitive implemented: `False`
- Updraft validation claim: `False`
- W1/W2/W3 updraft validation claim: `False`
- OCP implemented: `False`
- TVLQR implemented: `False`
- Governor implemented: `False`
- Outer loop implemented: `False`
- High-incidence validation claim: `False`
