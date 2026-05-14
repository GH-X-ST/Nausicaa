# Phase 2 TVLQR OCP30 Replay Debug Report

Seed: `1`
Output root: `03_Control/05_Results/03_primitives/10_tight_turn_phase2_tvlqr_debug/001`

## Scope

Phase 2 debug only: W0 30 deg OCP reproduction, TrajectoryPrimitive conversion, open-loop replay, closed-loop TVLQR replay, nominal-latency replay, and terminal recovery sensitivity.

## Prior Agile Boundary Evidence

- `C:/Users/GH-X-ST/OneDrive - Imperial College London/Year 4/Final Year Project/01 - Github/Nausicaa/docs/control/agile_problem_1_2_7_report.md`: not found in this checkout
- `C:/Users/GH-X-ST/OneDrive - Imperial College London/Year 4/Final Year Project/01 - Github/Nausicaa/docs/control/agile_feasibility_boundary.md`: not found in this checkout
- `C:/Users/GH-X-ST/OneDrive - Imperial College London/Year 4/Final Year Project/01 - Github/Nausicaa/docs/control/turn_trajectory_optimisation_report.md`: found

## Best 30 Deg Result

- label: `accepted_low_alpha`
- success: `True`
- directed heading change deg: `28.293163149683984`
- actual wrapped heading change deg: `-28.293163149683977`
- dynamics defect max: `1.5777329664956596e-08`
- slack max: `0.0`
- failure reason: ``

## Phase 2 Gate Summary

Replay rows produced: `6`
- hard 30 deg OCP reproduced: `True`
- open-loop no-latency gate: `True`
- closed-loop no-latency gate: `True`
- nominal-latency gate: `False`
- terminal-altitude recovery sensitivity gate: `False`
- phase 2 status: `boundary_only`
- active failure class: `latency_limited_high_alpha`
- all failure classes: `latency_limited_high_alpha;terminal_recovery_limited`
- limitation: `angle of attack exceeded bound`

## Metrics Paths

- `03_Control/05_Results/03_primitives/10_tight_turn_phase2_tvlqr_debug/001/metrics/turn_ocp_candidates_s001.csv`
- `03_Control/05_Results/03_primitives/10_tight_turn_phase2_tvlqr_debug/001/metrics/turn_ocp_best_by_target_s001.csv`
- `03_Control/05_Results/03_primitives/10_tight_turn_phase2_tvlqr_debug/001/metrics/turn_tvlqr_replay_s001.csv`

## Limitation

This report is simulation-only and does not include Phase 3/4 continuation, entry sweeps, W0-W3 stress, outer-loop simulation, or hardware/Vicon execution.
