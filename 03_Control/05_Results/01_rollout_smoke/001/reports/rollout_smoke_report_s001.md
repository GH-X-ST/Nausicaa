# Rollout Smoke Report

This is an open-loop plant rollout smoke test only. It does not implement
a primitive, controller, OCP, TVLQR, governor, outer loop, Vicon interface,
hardware path, or high-incidence validation.

## Command Path

- Requested command: `u_norm_requested`.
- Applied command: `u_norm_applied`, clipped to `[-1, +1]`.
- Plant command: `delta_cmd_rad`, produced by `normalised_command_to_surface_rad`.
- `flight_dynamics.state_derivative` receives `delta_cmd_rad`, never raw normalised commands.

## Result

- Success flag: `False`
- Failure label: `not_run`
- Notes: `rollout_smoke_no_primitive`
- Rollout implemented: `True`
- Primitive implemented: `False`
- Controller implemented: `False`
- OCP implemented: `False`
- TVLQR implemented: `False`
- Governor implemented: `False`
- Outer loop implemented: `False`
- High-incidence validation claim: `False`
