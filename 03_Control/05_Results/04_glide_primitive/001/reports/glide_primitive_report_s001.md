# W0 Glide Primitive Report

This is the first actual no-wind glide primitive. It uses the existing
primitive-interface checks, the existing RK4 plant step, and a local
trim-hold feedback law in normalised command space.

It is not bank, recovery, agile reversal, OCP, TVLQR, governor, outer-loop,
Vicon, hardware, real-flight, or high-incidence validation evidence.

## Status

- Overall status: `pass`
- Entry checks pass: `True`
- Exit checks pass: `True`
- Glide checks pass: `True`
- Rollout ran: `True`
- Primitive success: `True`
- Failure label: `success`
- Notes: `glide_w0_nominal_trim_hold_feedback`

## Command Path

- Trim command: physical radians from `solve_straight_trim()`.
- Trim bridge: `surface_rad_to_normalised_command`.
- Feedback correction: normalised command space only.
- Applied bridge: `u_norm_requested -> u_norm_applied -> delta_cmd_rad`.
- Plant input: `delta_cmd_rad`; raw normalised commands do not enter dynamics.

## Observed Metrics

- Duration: `0.5` s
- Terminal speed: `6.500000000026716` m/s
- Height change: `-0.29174554345663384` m
- Minimum wall margin: `0.8` m
- Minimum floor margin: `1.3082544565433663` m
- Maximum alpha: `4.166892635583639` deg
- Maximum beta: `9.965660702112121e-17` deg
- Maximum rate norm: `5.601079859443244e-11` rad/s
- Saturation fraction: `0.0`

## Terminal Proxy

`terminal_recoverable_proxy` is only a plausibility check for later
primitive expansion. It does not claim that a recovery primitive exists.

## Implementation Flags

- Actual glide primitive implemented: `True`
- Local feedback controller implemented: `True`
- Actual bank primitive implemented: `False`
- Actual recovery primitive implemented: `False`
- Actual agile reversal primitive implemented: `False`
- OCP implemented: `False`
- TVLQR implemented: `False`
- Governor implemented: `False`
- Outer loop implemented: `False`
- High-incidence validation claim: `False`
