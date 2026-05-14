# Tight-Turn Trajectory Optimisation Phase 1/2 Report

Seed: `1`
Output root: `03_Control/05_Results/03_primitives/09_tight_turn_ocp_phase2/001`

## Scope

Implemented Phase 1/2 only: OCP smoke, W0 30 deg hard solve, W0 30 deg soft-boundary diagnostic when required, and TVLQR replay only for an accepted hard 30 deg candidate.

## Prior Agile Boundary Evidence

- `C:/Users/GH-X-ST/OneDrive - Imperial College London/Year 4/Final Year Project/01 - Github/Nausicaa/docs/control/agile_problem_1_2_7_report.md`: not found in this checkout
- `C:/Users/GH-X-ST/OneDrive - Imperial College London/Year 4/Final Year Project/01 - Github/Nausicaa/docs/control/agile_feasibility_boundary.md`: not found in this checkout

## Best 30 Deg Result

- label: `accepted_low_alpha`
- success: `True`
- directed heading change deg: `28.293163149683984`
- actual wrapped heading change deg: `-28.293163149683977`
- dynamics defect max: `1.5777329664956596e-08`
- slack max: `0.0`
- failure reason: ``

## Replay

Replay rows produced: `6`
- closed-loop no-latency gate: `True`
- nominal-latency gate: `False`
- terminal-altitude recovery sensitivity gate: `False`
- promoted beyond Phase 2: `False`
- replay limitation: `angle of attack exceeded bound`

## Metrics Paths

- `03_Control/05_Results/03_primitives/09_tight_turn_ocp_phase2/001/metrics/turn_ocp_candidates_seed1.csv`
- `03_Control/05_Results/03_primitives/09_tight_turn_ocp_phase2/001/metrics/turn_ocp_best_by_target_seed1.csv`
- `03_Control/05_Results/03_primitives/09_tight_turn_ocp_phase2/001/metrics/turn_tvlqr_replay_seed1.csv` if replay ran

## Limitation

This report is simulation-only and does not include Phase 3/4 continuation, entry sweeps, W0-W3 stress, outer-loop simulation, or hardware/Vicon execution.
