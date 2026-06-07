# Real-Flight Deployment v4.7

## Scope

This note aligns the current documentation with the post-v4.6 real-flight
simulation-replay and post-analysis audit. It records the E1 random-sample
replay figures, tracking-quality interpretation, and score/reporting boundary.
It is a plotting, replay-diagnostic, and documentation record, not a new
R10/R11 validation, aerodynamic SysID refit, fan-flow validation,
mission-success claim, hardware-autonomy claim, or memory-improvement claim.

## Current-Code Alignment

- `04_Flight_Test/00_Plotting/run_real_flight_sim_replay_figures.py` now plots
  valid open-loop neutral throws even when `metrics/controller_decisions.csv` is
  absent. Closed-loop throws still replay the logged real `primitive_variant_id`
  sequence. For open-loop neutral throws, the `sim_real_decisions` replay has an
  empty selected-primitive sequence and therefore remains neutral after the
  measured 0.040 s handoff splice.
- The replay script still writes to `04_Flight_Test/A_figures`, keeps the same
  two simulation replies, copies the measured first 0.040 s handoff trace, starts
  simulation from the measured post-handoff state, applies the runtime
  `surface_command_delay_s` buffer before actuator lag, and writes
  `real_flight_sim_replay_summary.csv`, per-throw replay traces,
  `first_0p04_state_replay_error_summary.csv`, `execution_timing_audit.csv`,
  a manifest, and a Markdown report.
- The generated E1 random-sample replay root is
  `04_Flight_Test/A_figures/real_flight_sim_replay_E1_random_samples/`.

## E1 Random-Sample Replay Set

The deterministic random sample uses two valid throws from each current E1
record:

- E1.0 redo `20260607_124146`: `throw_002`, `throw_010`.
- E1.1 no-memory `20260606_230007`: `throw_005`, `throw_011`.
- E1.2 memory `20260607_122640`: `throw_002`, `throw_029`.

All six generated figures include the reality trajectory, the simulator's own
frozen-governor reply, and the simulation replay of the real decision sequence.
The first-0.040 s state-splice audit reports zero maximum residual across the
sample set, so the replay comparison starts from the agreed SysID-style
measured-handoff alignment rather than a mismatched initial condition.

## Tracking And Post-Analysis Boundary

The valid active-flight samples remain usable for post-analysis. E1.0 redo uses
200 Hz Vicon samples throughout active flight, with average latency 16.4 ms, max
active estimator gap 35 ms, 3.67 percent low-confidence/spike-rejected samples,
and 1.98 percent body-rate-limited samples. E1.2 uses 200 Hz Vicon samples
throughout active flight, with average latency 16.4 ms, max active estimator gap
65 ms, 2.48 percent low-confidence/spike-rejected samples, and 0.98 percent
body-rate-limited samples. These are acceptable for trajectory, launch
condition, timing, memory/no-memory, and replay diagnostics, but they should not
be described as pristine raw motion-capture body-rate evidence.

Large prelaunch frame gaps are not active-flight tracking failures, because the
runtime intentionally keeps Vicon inactive during cooldown/pre-arm neutral-hold
periods. Keyword audit found only the expected pre-arm Vicon-inactive manifest
settings, not active-flight NoFrame, missing-subject, or lost-tracking events.

## Replay And Score Interpretation

The replay figures diagnose model mismatch, controller-decision consistency, and
timing. They do not currently recompute a simulation-side mission score for the
two replay trajectories. The real posthoc selected score remains the
runtime/report-only accumulated selected-decision score from
`posthoc_throw.csv`, `posthoc_session.csv`, and closed-loop
`controller_decisions.csv`; E1.0 has score zero because it has no selected
controller decisions and is not directly comparable as a selected-score policy.

For the current sessions, E1.1 has mean accumulated selected score
`-1.2406162083333327`, while E1.2 has mean accumulated selected score
`0.8705906510995837`. That supports the dry-air memory score-audit record, but
it is not a real-vs-simulation score comparison. The replay figures should be
used to state that open-loop dry-air exposes model mismatch and that closed-loop
control remains robust enough to operate through that mismatch, not that the
real flight generally scores higher than the simulation replay.

## Checks

- `python -m py_compile 04_Flight_Test/00_Plotting/run_real_flight_sim_replay_figures.py`
- `python 04_Flight_Test/00_Plotting/run_real_flight_sim_replay_figures.py`
  with six explicit E1 throw roots and `--library-tier balanced_cluster`.
- Generated six PNG figures, six replay trace CSVs, replay summary, first
  0.040 s state residual audit, execution timing audit, manifest, and report
  under `04_Flight_Test/A_figures/real_flight_sim_replay_E1_random_samples/`.
- The first 0.040 s replay residual audit has maximum `max_abs = 0` across 180
  state/model rows.
- Docs stale-wording audit was run across all `docs/**/*.txt` and
  `docs/**/*.md` files.
