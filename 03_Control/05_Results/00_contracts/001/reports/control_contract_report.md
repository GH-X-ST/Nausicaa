# Control Contract Audit Report

This task implements contracts only. It does not implement a controller,
rollout integrator, OCP, TVLQR, governor, outer loop, Vicon interface,
or high-incidence validation.

## State And Command Order

- State order: `x_w, y_w, z_w, phi, theta, psi, u, v, w, p, q, r, delta_a, delta_e, delta_r`
- Command order: `delta_a_cmd, delta_e_cmd, delta_r_cmd`
- Positive aileron: positive roll moment, right wing down
- Positive elevator: positive pitch moment, nose up
- Positive rudder: positive yaw moment, nose right

## Arena Bounds

- Tracker-limit bounds and true-safety bounds are separate contract objects.
- Primitive acceptance and later governor checks must use true safety bounds.

## Primitive Families

- Mandatory families: `glide, bank, recovery, agile_reversal`

## Metric And Scenario Contracts

- Metric schema is fixed for later primitive, OCP, TVLQR, governor, and outer-loop evidence.
- Scenario metadata records wind mode, latency case, timing, seed, and true-safety use.

## Status Flags

- High-incidence validation claim: `False`
- Controller implemented: `False`
- OCP implemented: `False`
- TVLQR implemented: `False`
- Governor implemented: `False`
- Outer loop implemented: `False`

## Next Step

Rebuild rollout and logging base, then primitive interface execution,
then agile OCP formulation.
