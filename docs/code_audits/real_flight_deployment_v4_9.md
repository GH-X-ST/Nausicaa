# Real-Flight Deployment v4.9

## Scope

This note aligns documentation with the post-v4.8 real-flight replay and
experiment-evidence state: the current redo E1.0 dry-air open-loop record, the
completed fixed single-fan E2 workflow, and the measured-fan updraft replay
plots. It is a documentation, plotting, replay-diagnostic, and workflow record,
not a new R10/R11 validation, aerodynamic SysID refit, fan-flow validation,
mission-success claim, hardware-autonomy claim, or broad memory-improvement
claim.

## Current-Code Alignment

- `04_Flight_Test/00_Plotting/run_real_flight_sim_replay_figures.py` now writes
  replay version `real_flight_sim_replay_measured_fan_updraft_v2`.
- The replay still compares each real throw against two simulation replies:
  simulator-own governor decisions and simulation replay using the logged real
  `primitive_variant_id` sequence when present.
- Open-loop neutral throws remain valid replay inputs. Their
  `sim_real_decisions` trace has an empty selected-primitive sequence and stays
  neutral after the measured 0.040 s handoff splice.
- E1/no-visible-fan samples use W0 zero-wind replay. E2 fixed single-fan samples
  build a W2 annular-GP single-fan replay environment from the measured
  `fan_positions.csv` x/y snapshot, nominal power/width, and the active mask.
- The replay copies the measured first 0.040 s handoff state trace exactly,
  starts simulation from the measured post-handoff state, applies runtime
  `surface_command_delay_s` before actuator lag, passes the replay wind field
  into the dynamics, and writes `replay_environment_summary.csv` beside the
  earlier replay summary, trace, timing, manifest, and report outputs.
- Fan-case figures draw R11-style 3D updraft slices and isosurfaces, while
  dry-air figures explicitly record zero wind. These figures remain
  replay/timing diagnostics and do not recompute mission score.

## Current E1 Dry-Air Workflow

- Redo `E1.0` session `20260607_190445` is the current dry-air open-loop neutral
  baseline: 10 valid throws, 6 rejected starts, zero controller decisions, speed
  range 5.016--6.653 m/s, mean speed 5.727 m/s, mean final observable specific
  energy 1.634 m, 3 front-wall exits, and 7 floor exits.
- `E1.1` session `20260606_230007` remains the dry-air closed-loop no-memory
  baseline: 30 valid throws, 16 rejected/timeout starts, 30/30 valid throws with
  active controller decisions, 10--12 controller decisions per valid throw, max
  decision time 0.00432 s, speed range 5.295--6.841 m/s, mean final observable
  specific energy 1.747 m, 28 front-wall exits, and 2 floor exits.
- `E1.2` session `20260607_122640` remains the dry-air memory null test: 30
  valid throws, 10 rejected starts, 30/30 valid throws with active controller
  decisions, max step-0 first-decision time 0.0386 s, max decision time 0.0570 s,
  speed range 5.287--6.774 m/s, mean final observable specific energy 1.771 m,
  21 front-wall exits, 9 floor exits, 297 final memory cells, and 339 memory
  updates over history buckets h0/h1-3/h4-10/h11-30 = 1/3/7/19.
- Dry-air memory remains a bounded null/safety test. It should not be described
  as a fan-updraft memory-improvement claim, and rejected starts do not update
  memory or consume valid evidence throws.

## Completed E2 Single-Fan Workflow

- `E2.0` session `20260607_163303` is the fixed single-fan open-loop neutral
  baseline: 10 valid throws, 8 rejected starts, one visible fan snapshot per
  throw, speed range 5.626--6.260 m/s, mean speed 5.977 m/s, mean final
  observable specific energy 1.506 m, 6 front-wall exits, and 4 floor exits.
- `E2.1` session `20260607_165533` is fixed single-fan closed-loop no-memory:
  30 valid throws, 10 rejected starts, controller decisions on every valid
  controlled throw, max decision time 0.03197 s, speed range 5.053--6.412 m/s,
  mean speed 5.936 m/s, mean final observable specific energy 1.635 m, 24
  front-wall exits, and 6 floor exits.
- `E2.2` was collected as two 30-valid-throw memory sessions,
  `20260607_173345` and `20260607_175359`: 60 valid rows total, 19 rejected
  starts, max decision times 0.08695 s and 0.07169 s, mean speed 5.758 m/s,
  mean final observable specific energy 1.625 m, 44 front-wall exits, 14 floor
  exits, and two blank terminal rows. One blank row is the logged non-controlled
  launch-handoff abort `launch_handoff_abort:vicon_invalid:vicon_subject_occluded`.
- Across non-abort E2.2 memory rows, accumulated selected score remains positive
  on average and accumulated memory score is positive. The h4-10 history bucket
  gives 13/14 front-wall exits, mean speed 5.876 m/s, and mean final observable
  specific energy 1.681 m.
- E2 supports single-fan transfer, timing, speed, and memory diagnostics. The
  raw E2.2-versus-E2.1 comparison is speed-confounded and should not be claimed
  as a broad standalone memory-improvement proof.

## Replay Outputs

- Current E1 replay samples are under
  `04_Flight_Test/A_figures/real_flight_sim_replay_E1_random_samples/`:
  E1.0 `20260607_190445` throws 005 and 010, E1.1 `20260606_230007` throws 005
  and 011, and E1.2 `20260607_122640` throws 002 and 029.
- Current E2 replay samples are under
  `04_Flight_Test/A_figures/real_flight_sim_replay_E2_random_samples/`: E2.0
  `20260607_163303` throws 001 and 002, E2.1 `20260607_165533` throws 016 and
  027, E2.2 `20260607_173345` throws 020 and 030, and E2.2 `20260607_175359`
  throws 001 and 007.
- The E1 replay environment summary records W0 dry-air / zero-wind context and
  `updraft_max_m_s = 0.0` for every sample.
- The E2 replay environment summary records W2 measured-fan annular-GP context,
  `single_annular_gp_grid`, 18 plotted updraft isosurfaces, and
  `updraft_max_m_s = 5.151357802910417` for the replay grid.
- The earlier dry-air-only interpretation that real flight generally exceeded
  simulation replay is retired for E2. With measured-fan updraft included, the
  replay is a model-mismatch and decision-consistency diagnostic rather than a
  direct real-versus-sim score claim.

## Documentation Alignment

All `docs/**/*.txt` and `docs/**/*.md` files were checked after the v4.8 fan
logging update. The repeated bigmap/current-workflow text now names the active
redo E1.0 session, the completed E2 workflow, and the measured-fan updraft replay
contract. Historical v4.5/v4.6/v4.7 notes are explicitly framed as superseded
where they keep older session IDs or earlier replay interpretations.

## Checks

- Current E1/E2 session summaries and posthoc session CSVs were inspected.
- E1 and E2 replay summaries plus replay environment summaries were inspected.
- `python -m py_compile 04_Flight_Test/00_Plotting/run_real_flight_sim_replay_figures.py`
- `git diff --check`
