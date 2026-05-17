# Control Contract Audit Report

This task implements contracts only. It does not implement a controller,
rollout integrator, OCP, TVLQR, governor, outer loop, Vicon interface,
or high-incidence validation.

## State And Command Order

- State order: `x_w, y_w, z_w, phi, theta, psi, u, v, w, p, q, r, delta_a, delta_e, delta_r`
- Command order: `delta_a_cmd, delta_e_cmd, delta_r_cmd`
- Model-facing command input to `state_derivative`: `delta_cmd_rad`
- Positive aileron: positive roll moment, right wing down
- Positive elevator: positive pitch moment, nose up
- Positive rudder: positive yaw moment, nose right

## Command Bridge

- `u_norm` is the normalised aggregate command in `[-1, +1]`.
- `normalised_command_to_surface_rad` converts `u_norm` into calibrated
  physical aggregate surface targets `delta_cmd_rad` using `latency.py`.
- Future rollout and OCP code must pass `delta_cmd_rad` into
  `flight_dynamics.state_derivative`, never raw normalised commands.
- Surface states remain `delta_a`, `delta_e`, and `delta_r` in the
  canonical state vector.

## Arena Bounds

- Tracker-limit bounds and true-safety bounds are separate contract objects.
- Primitive acceptance and later governor checks must use true safety bounds.

## Primitive Families

- Mandatory families: `glide, bank, recovery, agile_reversal`

## Metric And Scenario Contracts

- Metric schema is fixed for later primitive, OCP, TVLQR, governor, and outer-loop evidence.
- `success` is final primitive-level success; finite-state, rollout,
  primitive, closed-loop replay, source-trajectory, and gain-construction
  success are recorded as separate Boolean evidence fields.
- Scenario metadata records wind mode, latency case, timing, seed, and true-safety use.

## Status Flags

- High-incidence validation claim: `False`
- Controller implemented: `False`
- OCP implemented: `False`
- TVLQR implemented: `False`
- Governor implemented: `False`
- Outer loop implemented: `False`

## Validation Commands

- `python 03_Control/04_Scenarios/run_control_contract_audit.py --overwrite`
- `python -m pytest -q tests/test_control_contract_state_command.py tests/test_control_contract_arena.py tests/test_control_contract_primitive_metric.py tests/test_control_contract_scenario_paths.py tests/test_control_contract_audit_runner.py`

## Next Step

Rebuild rollout and logging base, then primitive interface execution,
then agile OCP formulation.
