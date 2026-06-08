# Real-Flight Deployment v4.10

## Scope

This note aligns documentation with the post-v4.9 real-flight state: the
completed fixed four-fan E3 workflow, the E3 measured-fan simulation-replay
figures, and the active random-layout case naming cleanup. It is a real-flight
workflow, plotting, replay-diagnostic, and documentation record, not a new
R10/R11 validation, aerodynamic SysID refit, fan-flow validation,
mission-success claim, hardware-autonomy claim, or broad memory-improvement
claim.

## Completed E3 Fixed Four-Fan Workflow

- `E3.0` session `20260607_202556` is the fixed four-fan open-loop neutral
  baseline: 10 valid throws, 4 rejected starts, four visible fan snapshots per
  throw, speed range 5.769--6.642 m/s, mean speed 6.123 m/s, mean final
  observable specific energy 1.119 m, 1 front-wall exit, and 9 floor exits.
- `E3.1` session `20260607_204604` is fixed four-fan closed-loop no-memory:
  30 recorded valid rows under pre-v4.11 validity logic, 20 rejected starts, 29
  usable controlled rows plus one archived launch-handoff abort artifact that
  current v4.11 runtime logic marks non-valid, speed range 5.345--6.803 m/s,
  mean speed 6.170 m/s, max decision time 0.01959 s, mean final observable
  specific energy 1.773 m, and 18 front-wall exits / 11 floor exits over usable
  controlled rows.
- `E3.2` was collected as two 30-valid-throw memory sessions,
  `20260607_213312` and `20260607_214908`: 60 valid rows total, 33 rejected
  starts, max decision times 0.08203 s and 0.07195 s, mean speed 6.232 m/s,
  mean final observable specific energy 1.600 m, and 34 front-wall exits / 26
  floor exits.
- All completed E3 sessions record calibration profile hash
  `24be0ecbcf807922bfad624ddae1d4031c3262c51592c9063b22bab5239c161f`.

## E3 Interpretation Boundary

- E3 fixed four-fan evidence is strongly speed-conditioned. Open-loop neutral
  is non-robust in this setup: 1/10 front-wall exits, with 9/10 floor exits.
- Closed-loop no-memory has 18/29 usable front-wall exits overall, 14/18 in the
  6.0--6.5 m/s transition band, and 3/3 above 6.5 m/s.
- Closed-loop memory has 34/60 front-wall exits overall, 18/32 in the
  6.0--6.5 m/s transition band, and 14/15 above 6.5 m/s.
- Memory remains meaningful as a bounded decision-quality/opportunity signal:
  the E3.2 selected-score audit is positive in the 6.0--6.5 m/s transition band
  and stronger above 6.5 m/s, but the evidence should not be phrased as memory
  rescuing very low-energy launches.
- The thesis-facing interpretation is therefore: closed-loop control is
  essential, launch speed sets the practical viability threshold, and spatial
  memory improves decision opportunity within the viable band.

## E3 Replay Outputs

- Representative replay figures are stored under
  `04_Flight_Test/A_figures/real_flight_sim_replay_E3_representative/`.
- The plotted throws are E3.0 `throw_001` and `throw_008`, E3.1 `throw_009` and
  `throw_023`, and E3.2 `throw_013` and `throw_016`.
- Each replay uses `balanced_cluster`, logged real-decision timing, and replay
  version `real_flight_sim_replay_measured_fan_updraft_v2`.
- `replay_environment_summary.csv` records W2 measured-fan annular-GP context
  with `four_annular_gp_grid`, `fan_count=4`, and `active_fan_count=4` for every
  plotted E3 throw.
- `first_0p04_state_replay_error_summary.csv` reports zero residual over the
  measured 0.040 s handoff splice for position, attitude, velocity/rate, and
  surface state across the plotted sample set.
- These figures remain replay/timing/model-mismatch diagnostics; they do not
  validate fan-flow strength, recompute a simulation mission score, or establish
  mission-success or full autonomy.

## Current-Code Alignment

- `04_Flight_Test/01_Runtime/experiment_cases.py` removes the old hard-shifted
  E4 diagnostic stage from the active real-flight registry.
- At the v4.10 checkpoint, the former random-layout E5 family had been promoted
  into the active E4 family, while the time-limited active registry kept only
  `E4a` and `E4b`; this schedule was later completed and closed by v4.12.
- Each random-layout family keeps the same suffix contract:
  `.0` open-loop neutral baseline, `.1` closed-loop no-memory, and `.2`
  closed-loop memory enabled.
- Random-layout E4 cases expect one to four visible fans and keep the active
  time-limited memory protocol: 30 valid throws per `.2` invocation, with a
  second independent session collected manually when time allows.
- Old `E5a`--`E5d` names are no longer active registry entries. Existing result
  folders, if any, remain historical and are not renamed by this workflow note.

## Then-Active Random-Layout Cases

- `E4a.0`, `E4a.1`, `E4a.2`: random layout 1.
- `E4b.0`, `E4b.1`, `E4b.2`: random layout 2.

## Documentation Alignment

- All `docs/**/*.txt` and `docs/**/*.md` files were checked after the v4.9 E2
  replay/evidence update.
- The repeated bigmap/current-workflow text now names the completed E3 fixed
  four-fan workflow, its speed-conditioned controller/memory interpretation, and
  the measured-fan W2 four-fan representative replay outputs.
- `docs/local_validation_environment.md` now carries the same E3 workflow and
  replay boundary in its local validation summary.
- The v4.10 case-name text recorded that the old hard-shifted E4 diagnostic
  stage was retired, that random-layout testing was narrowed to E4a/E4b, and
  that E4c/E4d were not active schedule entries. Current docs supersede this
  with the completed-through-E4b, post-analysis-only boundary.

## Checks

- E3 session summaries, posthoc session CSVs, and runtime summaries were
  inspected.
- E3 replay summary, replay environment summary, first-0.040 s state residual
  audit, and execution timing audit were inspected.
- `python -m py_compile 04_Flight_Test/00_Plotting/run_real_flight_sim_replay_figures.py`
- `git diff --check`
