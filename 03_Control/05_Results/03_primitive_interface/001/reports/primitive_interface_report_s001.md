# Primitive Interface Smoke Report

This is primitive-interface execution evidence only. It connects the
primitive metadata contract to the existing open-loop rollout/logging
base, but it is not a real glide primitive result.

It does not implement a controller, OCP, TVLQR, governor, outer loop,
Vicon interface, hardware path, or high-incidence validation.

## Status

- Overall status: `pass`
- Interface checks pass: `True`
- Entry checks pass: `True`
- Exit checks pass: `True`
- Rollout ran: `True`
- Final primitive success: `False`
- Final success: `False`
- Failure label: `not_run`
- Notes: `primitive_interface_smoke_no_controller`

## Command Path

- Requested normalised command: `u_norm_requested`.
- Applied normalised command: `u_norm_applied`, clipped by the rollout layer.
- Plant command: `delta_cmd_rad`, produced by `normalised_command_to_surface_rad`.
- `state_derivative` receives `delta_cmd_rad`, never raw normalised commands.

## Implementation Flags

- Primitive interface implemented: `True`
- Actual glide primitive implemented: `False`
- Controller implemented: `False`
- OCP implemented: `False`
- TVLQR implemented: `False`
- Governor implemented: `False`
- Outer loop implemented: `False`
- High-incidence validation claim: `False`

## Next Step

Implement the first actual primitive family, likely glide, using this
interface and keeping primitive success separate from rollout integrity.
