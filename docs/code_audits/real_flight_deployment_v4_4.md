# Real-Flight Deployment v4.4

## Scope

This note aligns the current documentation with the post-v4.3 real-flight
runtime and replay-plotting changes. It is a documentation and diagnostic
alignment record, not a new R10/R11 evidence claim.

## Historical Code Alignment

- `04_Flight_Test/01_Runtime/run_real_flight.py` keeps the hybrid closed-loop
  scheduler from v4.3, but now buffers active metric rows with
  `buffer_active_rows_flush_after_active_record` so time-critical governor
  commit and 50 Hz packet emission occur before active metric flushing.
- At v4.4, active fan-position logging was intentionally limited to
  prelaunch/handoff/post-exit snapshots. This behavior is now historical:
  v4.8 supersedes it with one prelaunch fan snapshot per throw and no fan
  polling during launch handoff or post-exit.
- The active runtime wakes 2 ms ahead of scheduled active-loop events to reduce
  scheduler-lag risk while preserving the 0.040 s launch handoff and 0.100 s /
  five-slot primitive contract.
- `04_Flight_Test/00_Plotting/run_real_flight_sim_replay_figures.py` is the
  current real-flight replay plotting entry point. Outputs are written under
  `04_Flight_Test/A_figures/real_flight_sim_replay/`, separated from the older
  `03_Control/A_figures/real_flight_replay_comparison/` SysID/model-replay
  figures.
- Each valid real throw is compared with two simulation replies: one where the
  frozen governor re-selects primitives in simulation, and one where the
  simulator reuses the logged real `primitive_variant_id` sequence.
- Both simulation replies copy the measured first 0.040 s handoff state trace,
  start simulation from the measured post-handoff state, and apply the runtime
  `surface_command_delay_s` buffer before actuator lag.
- The replay output includes per-throw trace CSVs,
  `real_flight_sim_replay_summary.csv`,
  `first_0p04_state_replay_error_summary.csv`,
  `execution_timing_audit.csv`, a manifest, and a Markdown report. Figures now
  include a bottom-left launch-condition block with launch state, speed,
  attitude, body velocity/rates, and handoff duration.
- The real-flight memory case registry now encodes repeated independent
  sessions instead of one long monolithic memory run for the fan evidence cases:
  `E2.2` and `E3.2` are `3 x 30` valid throws, while `E4*.2` and `E5*.2` are
  `2 x 30` valid throws. `run_experiment_sequence.py` exposes
  `--repeat-sessions`; each repeat creates a separate session folder and starts
  a new controller instance with empty memory.

## Evidence Boundary

The replay figures are diagnostic sim2real evidence for E0.1 dry-air shakedown
throws. They can show timing health, first-window state alignment, model
residuals, and whether mismatch remains when replaying logged real decisions.
They do not refit the aerodynamic model, change R10/R11 gates, or by themselves
claim mission success, memory improvement, or full hardware autonomy. The
`30`-throw repeated-session memory protocol improves repeatability evidence but
must still be analysed as separate memory episodes before pooling.

## Checks

- `python -m py_compile 04_Flight_Test/00_Plotting/run_real_flight_sim_replay_figures.py`
- `python 04_Flight_Test/00_Plotting/run_real_flight_sim_replay_figures.py --case-id E0.1 --library-tier balanced_cluster`
- `git diff --check`
