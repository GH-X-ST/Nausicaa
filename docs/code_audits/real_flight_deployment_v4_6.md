# Real-Flight Deployment v4.6

## Scope

This note aligns the current documentation with the dry-air real-flight workflow
after v4.5: redo E1.0 open-loop neutral evidence and completed E1.2 dry-air
memory-null evidence. It is a real-flight workflow and documentation record, not
a new R10/R11 validation, aerodynamic SysID refit, fan-flow validation,
mission-success claim, hardware-autonomy claim, or memory-improvement claim.

## Current-Code Alignment

- The experiment registry still defines E1.0 as dry-air open-loop neutral,
  E1.1 as dry-air closed-loop no-memory, and E1.2 as dry-air closed-loop memory
  enabled. Launch-gate rejected or timed-out starts do not count as valid
  evidence throws and do not update memory.
- The top-of-file convenience default in `run_experiment_sequence.py` currently
  points at `E1.0`, matching the redo E1.0 session. The registry remains the
  authoritative workflow definition, and operators should select the next case
  explicitly before continuing into E2/E3 fan evidence.
- E1.2 uses the balanced-cluster E03 real-flight controller, the same 0.040 s
  launch-handoff contract, the 0.100 s / five-slot / 20 ms primitive contract,
  and the same geometry-first shortlisted spatial-memory selector contract
  documented in v4.5.

## Current E1 Dry-Air Workflow

- Redo `E1.0` session `20260607_124146` is the current dry-air open-loop neutral
  baseline: 10 valid throws, 4 launch-gate rejected starts, zero controller
  decisions, speed range 4.921--6.364 m/s, mean final observable specific energy
  1.516 m, 1 front-wall exit, and 9 floor exits.
- `E1.1` session `20260606_230007` remains the dry-air closed-loop no-memory
  baseline: 30 valid throws, 16 rejected/timeout starts, 30/30 valid throws with
  active controller decisions, 10--12 controller decisions per valid throw, max
  decision time 0.00432 s, speed range 5.295--6.841 m/s, mean final observable
  specific energy 1.747 m, 28 front-wall exits, and 2 floor exits.
- `E1.2` session `20260607_122640` is the dry-air memory null test: 30 valid
  throws, 10 launch-gate rejected starts, 30/30 valid throws with active
  controller decisions, 10--12 controller decisions per valid throw, max
  step-0 first-decision time 0.0386 s, max decision time 0.0570 s, zero
  decisions above 0.100 s, speed range 5.287--6.774 m/s, mean final observable
  specific energy 1.771 m, 21 front-wall exits, 9 floor exits, 297 final memory
  cells, and 339 memory updates over history buckets h0/h1-3/h4-10/h11-30 =
  1/3/7/19.
- All three current dry-air records use active calibration profile hash
  `c4fca6c930dd4a9c9836c53fd3bb796ac14973d08c32796a1ecf0d155edd2d2f`, profile
  id
  `nausicaa_real_flight_vicon_calibration_20260606_192524_position+nausicaa_real_flight_vicon_calibration_20260604_125825_attitude`,
  200 Hz Vicon tracking, and the 3 s pre-arm neutral hold.

## Evidence Boundary

E1.0/E1.1/E1.2 are dry-air workflow and posthoc-score audit records. E1.2 is a
negative-control memory test: in no-updraft dry air, spatial memory is expected
to remain bounded, avoid inventing a false useful-flow field, and not consume
invalid starts as memory evidence. The dry-air memory result can support a
safety/null-test argument, but the memory-improvement claim must come from fan
or repeated-layout cases where persistent spatial flow exists.

The E1.2 timing result supports real-flight workflow readiness for the current
memory selector implementation: the first launch decision stayed within the
0.040 s handoff window, every decision stayed within the 0.100 s primitive
boundary, and rejected starts remained launch-gate quality filters rather than
controller failures. It does not claim that dry-air memory should outperform the
no-memory baseline.

## Checks

- Current E1 manifests and session posthoc CSVs were inspected for E1.0 redo,
  E1.1 retained baseline, and E1.2 completed memory-null evidence.
- Controller-decision timing was audited from E1.1/E1.2
  `controller_decisions.csv` rows.
- Docs stale-wording audit was run across all `docs/**/*.txt` and
  `docs/**/*.md` files.
