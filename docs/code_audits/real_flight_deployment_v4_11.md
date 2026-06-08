# Real-Flight Deployment v4.11

## Scope

This note aligns documentation with the post-v4.10 real-flight state: the
completed E4a random-layout workflow, the E4a measured-fan simulation-replay
figures, the then-narrowed random-layout schedule, and the runtime validity
boundary for Vicon loss and launch-handoff aborts. It is a real-flight workflow,
plotting, replay-diagnostic, runtime-validity, and documentation record, not a
new R10/R11 validation, aerodynamic SysID refit, fan-flow validation,
mission-success claim, hardware-autonomy claim, or broad memory-improvement
claim.

## Runtime Validity Boundary

- Launch-handoff Vicon/safety/exit aborts are non-valid evidence attempts in
  current code. They keep the launch-gate context in the record, stream neutral,
  return to the sequencer, and do not consume a target valid throw or update
  memory.
- Sustained post-launch active Vicon tracking loss beyond
  `stale_vicon_timeout_s` now logs `active_vicon_tracking_lost:*`, streams
  neutral, returns to the sequencer, and marks the attempt non-valid.
- This boundary prevents pickup-contaminated states and pre-active Vicon aborts
  from being counted as physical flight outcomes. Earlier archived E2/E3/E4a
  rows that were recorded under pre-v4.11 validity logic remain historical data,
  but current summaries must identify those artifacts explicitly before using
  controlled-evidence counts.

## Completed E4a Random-Layout Workflow

- `E4a.0` session `20260607_230250` is the three-visible-fan open-loop neutral
  baseline: 10 valid throws, 7 rejected/invalid starts, `fan_count=3`, speed
  range 4.600--5.969 m/s, mean speed 5.382 m/s, mean final observable specific
  energy 1.627 m, 3 front-wall exits, 6 floor exits, and 1 y-max exit.
- `E4a.1` session `20260607_224440` is three-visible-fan closed-loop no-memory:
  30 valid throws, 32 rejected/invalid starts, `fan_count=3`, speed range
  4.749--6.184 m/s, mean speed 5.655 m/s, mean final observable specific energy
  1.905 m, 26 front-wall exits, and 4 floor exits.
- `E4a.2` session `20260607_231704` is three-visible-fan closed-loop memory:
  30 recorded valid rows in the archived session, but two rows are
  launch-handoff Vicon abort artifacts under old validity logic and should be
  excluded from controlled evidence, leaving 28 usable controlled throws with
  mean speed 5.360 m/s, mean final observable specific energy 1.771 m, 21
  front-wall exits, 5 floor exits, 2 y-max exits, and final memory state around
  394 updated cells.

## E4a Interpretation Boundary

- Open-loop is not reliable in this E4a three-fan random layout.
- Closed-loop no-memory is the strongest E4a physical result.
- Memory is active and changes score structure, but it does not improve this
  E4a run. The raw no-memory versus memory comparison is speed-confounded and
  should be discussed as a random-layout case where memory can underperform,
  not as a broad memory-improvement failure.
- E4a is useful thesis evidence because it separates controller robustness from
  the stronger memory claims: control still carries the flight through the
  random layout, while the memory policy has an interpretable case where the
  remembered spatial correction is not beneficial.

## E4a Replay Outputs

- Representative replay figures are stored under
  `04_Flight_Test/A_figures/real_flight_sim_replay_E4a_representative/`.
- The plotted throws are E4a.0 `throw_004` and `throw_006`, E4a.1 `throw_004`
  and `throw_010`, and E4a.2 `throw_024` and `throw_027`.
- Each replay uses `balanced_cluster`, logged real-decision timing, exact
  first-0.040 s state splice, and replay version
  `real_flight_sim_replay_measured_fan_updraft_v2`.
- `replay_environment_summary.csv` records W2 measured-fan context with
  `fan_count=3`, `active_fan_count=3`, nominal fan power/width, and measured
  x/y fan snapshot replay.
- `first_0p04_state_replay_error_summary.csv` reports zero residual over the
  measured 0.040 s handoff splice for state and surface fields across the
  plotted E4a sample set.
- These figures remain replay/timing/model-mismatch and workflow-validity
  diagnostics; they do not validate fan-flow strength, recompute a simulation
  mission score, or establish mission success, full autonomy, or broad memory
  improvement.

## Current-Code Alignment

- `04_Flight_Test/01_Runtime/run_real_flight.py` marks launch-handoff Vicon,
  safety, and exit aborts non-valid, and also marks sustained active Vicon
  tracking loss non-valid before returning to the sequencer.
- At the v4.11 checkpoint, `04_Flight_Test/01_Runtime/experiment_cases.py` kept
  only `E4a` and `E4b` as random-layout families after E3 fixed four-fan
  evidence; this was later completed and closed by v4.12.
- `E4c` and `E4d` are no longer active schedule entries. Old `E5a`--`E5d` names
  remain historical only.
- `04_Flight_Test/REAL_FLIGHT_EXPERIMENT_INSTRUCTIONS.txt` and the repeated
  bigmap docs described E4a/E4b as the then-remaining random-layout workflow and
  framed old E2/E3/E4a archived abort rows against v4.11 validity logic. Current
  docs supersede this with the completed-through-E4b, post-analysis-only
  boundary.

## Documentation Alignment

- All `docs/**/*.txt` and `docs/**/*.md` files were checked after the v4.10 E3
  update and the E4a workflow discussion.
- The repeated bigmap/current-workflow text now names the completed E4a
  random-layout workflow, its bounded memory interpretation, the measured-fan
  three-fan representative replay outputs, and the launch-handoff/active-Vicon
  non-valid evidence boundary.
- Historical audit files retain old run context only when explicitly framed as
  historical or pre-v4.11 validity semantics.

## Checks

- E4a session summaries, posthoc session CSVs, runtime summaries, and
  representative replay outputs were inspected.
- E4a replay environment summary, first-0.040 s state residual audit, and
  execution timing audit were inspected.
- `python -m py_compile 04_Flight_Test/01_Runtime/run_real_flight.py`
- `python -m py_compile 04_Flight_Test/01_Runtime/experiment_cases.py`
- `python -m py_compile 04_Flight_Test/04_Tests/test_flight_runtime_contract.py`
- Focused flight-runtime validity tests for launch-handoff aborts, active Vicon
  tracking loss, and E4c/E4d registry removal passed.
- `git diff --check`
