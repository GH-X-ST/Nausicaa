# Real-Flight Deployment v4.5

## Scope

This note aligns the current documentation with the post-v4.4 real-flight
memory timing repair and the completed dry-air E1.0/E1.1 workflow evidence. It
is a runtime, documentation, and evidence-boundary record, not a new R10/R11
validation, aerodynamic SysID refit, mission-success claim, or
memory-improvement claim.

Supersession note: the E1 dry-air evidence summary in this v4.5 record is
superseded by `docs/code_audits/real_flight_deployment_v4_6.md`. The current
workflow record uses redo E1.0 session `20260607_124146` and completed E1.2
session `20260607_122640`.

## Current-Code Alignment

- Real-flight memory now honours the same selector contract as simulation: the
  selector can request a cheap `geometry_only` candidate-path pass for all
  transition-compatible candidates, then run the full spatial flow-belief query
  only for the selector shortlist.
- `real_flight_memory.py` builds one residual-belief lookup context per
  governor decision and reuses it for candidate evaluation. The full memory
  score still uses the bounded current-to-exit, reachable-cone, and
  short-horizon route-flow probes from the candidate exit.
- Open-loop and no-memory paths continue to bypass spatial-memory feature
  evaluation; their controller selection and logging contracts are unchanged.
- A step-0 memory selection that misses the fixed 0.040 s launch handoff is now
  logged as `first_launch_decision_missed_handoff_budget`. The launch-gate
  approval remains in the record, but the attempt is marked non-valid for
  controlled evidence, so it does not update memory or consume a target valid
  throw.
- The repeated-session fan evidence protocol from v4.4 remains active:
  `E2.2`/`E3.2` are independent `3 x 30` memory sessions, while `E4*.2` and
  `E5*.2` are independent `2 x 30` memory sessions.

## Superseded E1 Dry-Air Workflow

- The original v4.5 E1.0 open-loop neutral baseline has been replaced by redo
  E1.0 session `20260607_124146`; see v4.6 for the current dry-air workflow
  record.
- `E1.1` session `20260606_230007` is the dry-air closed-loop no-memory
  baseline: 30 valid throws, 16 rejected/timeout starts, 30/30 valid throws
  with active controller decisions, 10--12 controller decisions per valid throw,
  max decision time 0.00432 s, speed range 5.295--6.841 m/s, mean final
  observable specific energy 1.747 m, 28 front-wall exits, and 2 floor exits.
- The superseded v4.5 dry-air records used active calibration profile hash
  `c4fca6c930dd4a9c9836c53fd3bb796ac14973d08c32796a1ecf0d155edd2d2f`, profile
  id
  `nausicaa_real_flight_vicon_calibration_20260606_192524_position+nausicaa_real_flight_vicon_calibration_20260604_125825_attitude`,
  200 Hz Vicon tracking, and the 3 s pre-arm neutral hold.
- The E1.1 invalid starts are launch-gate quality rejections or timeouts; they
  do not update memory and do not count as valid controlled evidence throws.

## Evidence Boundary

The v4.5 timing repair is an implementation alignment with the existing
simulation selector design. It preserves the balanced-cluster E03 deployment
tier, the 0.040 s launch-handoff contract, the 0.100 s / five-slot / 20 ms
primitive contract, old-to-new FIFO command order, and posthoc logging
boundaries. The E1.0/E1.1 runs are dry-air workflow and posthoc-score audit
records. They support proceeding to richer real-flight workflow evidence, but
do not by themselves claim mission success, memory improvement, fan-flow
validation, hardware autonomy, or a new R10/R11 gate result.

## Checks

- Runtime `py_compile` for the touched real-flight memory/controller files.
- Focused flight-runtime contract tests covering missed handoff validity and
  memory candidate-feature context reuse: `5 passed, 46 deselected`.
- Synthetic high-cell memory timing probe: 159 memory cells, 30 launch
  decisions, max first-decision time 0.0239 s, 0 decisions above 0.040 s.
- Full runtime contract file was also run; it reported `50 passed` and one
  unrelated glider-calibration manifest-directory failure outside this change.
