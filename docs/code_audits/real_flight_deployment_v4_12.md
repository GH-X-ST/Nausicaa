# Real-Flight Deployment v4.12

## Scope

This note aligns documentation with the post-v4.11 real-flight state: the
completed E4b random-layout workflow, the E4b measured-fan simulation-replay
figures, the closed real-flight evidence-collection boundary, and the remaining
post-analysis-only project phase. It is a real-flight workflow, plotting,
replay-diagnostic, and documentation record, not a new R10/R11 validation,
aerodynamic SysID refit, fan-flow validation, hardware-autonomy claim, or broad
memory-improvement claim.

## Completed E4b Random-Layout Workflow

- `E4b.0` session `20260608_003841` is the four-visible-fan open-loop neutral
  baseline: 10 valid throws, 12 rejected/invalid starts, `fan_count=4`, speed
  range 5.101--6.350 m/s, mean speed 5.584 m/s, mean final observable specific
  energy 1.742 m, 2 front-wall exits, 7 floor exits, and 1 y-min exit.
- `E4b.1` is the four-visible-fan closed-loop no-memory case combined from two
  split sessions: `20260608_004535` contributes the first 20 valid throws and
  `20260608_010525` contributes the 10 valid top-up throws. The combined
  no-memory record has 30 valid throws, 32 rejected/invalid starts,
  `fan_count=4`, speed range 5.015--6.256 m/s, mean speed 5.701 m/s, mean final
  observable specific energy 1.830 m, 28 front-wall exits, 0 floor exits, 2
  y-min exits, and max decision time 0.0544 s.
- `E4b.2` session `20260608_013501` is the four-visible-fan closed-loop memory
  case: 30 valid throws, 17 rejected/invalid starts, `fan_count=4`, speed range
  4.900--6.188 m/s, mean speed 5.718 m/s, mean final observable specific energy
  1.959 m, median final observable specific energy 1.839 m, 27 front-wall exits,
  3 floor exits, and final memory state around 382 updated cells.

## E4b Interpretation Boundary

- Open-loop is not a reliable mission baseline in E4b: 2/10 front-wall exits,
  7/10 floor exits, and 1/10 y-min exit.
- Closed-loop no-memory is the strongest E4b reliability result: 28/30
  front-wall exits, no floor exits, and only 2 y-min exits across the combined
  split-session record.
- Memory is active and operational, but it is slightly worse than no-memory on
  front-wall/floor reliability in this four-fan random layout: 27/30 front-wall
  exits and 3/30 floor exits. Its mean final observable energy is higher because
  of one exceptional high-energy memory throw (`throw_020`, 6.149 m), while its
  typical/median energy remains below the no-memory median.
- The thesis-facing interpretation is therefore that closed-loop transfer is
  robust, while spatial flow-belief memory is context-dependent rather than
  automatically beneficial in every random fan layout.

## E4b Replay Outputs

- Representative replay figures are stored under
  `04_Flight_Test/A_figures/real_flight_sim_replay_E4b_representative/`.
- The plotted throws are E4b.0 `throw_004` and `throw_010`, E4b.1
  `throw_013` from `20260608_004535` and `throw_008` from `20260608_010525`,
  and E4b.2 `throw_001` and `throw_020`.
- Each replay uses `balanced_cluster`, logged real-decision timing, exact
  first-0.040 s state splice, and replay version
  `real_flight_sim_replay_measured_fan_updraft_v2`.
- `replay_environment_summary.csv` records W2 measured-fan context with
  `fan_count=4`, `active_fan_count=4`, `active_fan_mask=1;1;1;1`, nominal fan
  power/width, and `four_annular_gp_grid` replay for every plotted E4b throw.
- `first_0p04_state_replay_error_summary.csv` reports zero residual over the
  measured 0.040 s handoff splice for state and surface fields across the
  plotted E4b sample set.
- These figures remain replay/timing/model-mismatch and workflow-validity
  diagnostics; they do not validate fan-flow strength, recompute a simulation
  mission score, or establish mission success, full autonomy, or broad memory
  improvement.

## Project Boundary

- The real-flight evidence-collection workflow is complete through E4b.
- The project is now finished for required flight execution. Remaining work is
  post-analysis, figure/table curation, thesis writing, and retrospective
  audit/documentation.
- Any further real-flight collection should be opened as a new optional
  extension rather than treated as required to complete the current workflow.
- The old hard-shifted E4 diagnostic stage, `E4c`/`E4d`, and old `E5a`--`E5d`
  names remain historical/non-active only.

## Documentation Alignment

- All `docs/**/*.txt` and `docs/**/*.md` files were checked after the v4.11 E4a
  update and the completed E4b workflow discussion.
- The repeated bigmap/current-workflow text now names the completed E4b
  four-fan random-layout workflow, the split-session E4b.1 aggregation, the
  bounded memory interpretation, the measured-fan four-fan representative
  replay outputs, and the finished-flight/post-analysis-only project boundary.
- Historical audit files retain old run context only when explicitly framed as
  historical or pre-current-workflow semantics.

## Checks

- E4b session summaries, per-throw runtime summaries, posthoc session CSVs, and
  representative replay outputs were inspected.
- E4b replay environment summary, first-0.040 s state residual audit, execution
  timing audit, and one generated figure were inspected.
- Docs consistency searches were run for stale E4 active-schedule wording,
  E4b split-session reporting, and finished-project wording.
- `git diff --check -- docs`, `git diff --cached --check -- docs`, and a
  trailing-whitespace check on this new audit note passed.
