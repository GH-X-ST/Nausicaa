# Real-Flight FIFO Order Fix Report

Date: 2026-06-04

## Summary

The real-flight closed-loop runtime command-delay FIFO has been aligned with the
augmented LQR and simulation delay-line convention. The FIFO is now stored in
old-to-new order: index 0 is the delayed/applied command state and the newest
requested command is kept at the tail.

## Change

- Fixed `FrozenFlightController._push_command_fifo` so new 20 ms slot commands
  are appended and only the most recent `command_delay_steps` entries are kept.
- Added runtime contract coverage for direct FIFO order and for equivalence
  between explicitly preloaded FIFO history and live `_push_command_fifo` calls.
- Left simulation rollout and frozen controller bundles unchanged.

## Scope

This change affects closed-loop real-flight execution through
`04_Flight_Test/01_Runtime/frozen_flight_controller.py` only. It does not change
open-loop glider calibration collection, `run_glider_calibration_sequence.py`,
or previously collected calibration result folders.

## Remaining Audit Item

The runtime FIFO remains scoped by `controller_id`. Simulation rollouts can
preserve a global command history across primitive boundaries, so cross-primitive
FIFO continuity should be audited separately before making final closed-loop
real-flight readiness claims.
