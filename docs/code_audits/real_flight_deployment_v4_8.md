# Real-Flight Deployment v4.8

## Scope

This note aligns documentation with the post-v4.7 preflight fan-placement and
runtime fan-logging changes. It is a real-flight workflow and logging-boundary
update, not a new R10/R11 validation, aerodynamic SysID refit, fan-flow
validation, hardware-autonomy claim, mission-success claim, or
memory-improvement claim.

## Current-Code Alignment

- `04_Flight_Test/01_Runtime/run_vicon_frame_calibration.py` keeps the Vicon
  fan check as a preflight placement tool only. The `single_fan` and `four_fan`
  modes compare live fan translations, after the active arena transform, against
  the fixed simulation fan targets from `03_Control/04_Scenarios/updraft_models.py`.
- Fan placement now uses independent horizontal tolerance: the error is
  `max(abs(dx), abs(dy))`, the default tolerance is `0.05 m` per x/y axis, and
  z is displayed and logged only for operator awareness. Fan height is not part
  of the placement pass/fail decision.
- The legacy `--fan-position-error-axis` option is compatibility-only. Requests
  for `xyz` are ignored by design because the runtime check is an x/y placement
  check, not a height adjustment workflow.
- `04_Flight_Test/01_Runtime/run_real_flight.py` now reports
  `active_fan_logging_policy = single_prelaunch_snapshot_only`. A throw records
  fan positions once at the first valid prelaunch Vicon sample, before active
  flight starts.
- The runtime no longer polls or logs fan positions during launch handoff or
  post-exit. The `fan_positions.csv` schema, visible-count fields, and summary
  fields are preserved, so post-analysis structure is unchanged while live
  tracking/logging overhead is reduced.

## Documentation Alignment

All `docs/**/*.txt` and `docs/**/*.md` files were checked for the post-v4.7
runtime/calibration wording. The repeated bigmap text now states that Vicon fan
checks use independent `0.05 m` x/y tolerance with z display-only, and that
real-flight fan positions are logged as one prelaunch snapshot rather than
prelaunch/handoff/post-exit snapshots.

## Checks

- `python -m py_compile`
  `04_Flight_Test/01_Runtime/run_vicon_frame_calibration.py`
  `04_Flight_Test/01_Runtime/run_real_flight.py`
  `04_Flight_Test/04_Tests/test_flight_runtime_contract.py`
- Focused runtime smoke for buffered active metrics and single prelaunch fan
  snapshot.
- Full `04_Flight_Test/04_Tests/test_flight_runtime_contract.py`.
- `git diff --check`.

