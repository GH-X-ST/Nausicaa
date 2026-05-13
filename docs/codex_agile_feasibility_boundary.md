# Agile Confined-Reversal Feasibility Boundary

Date: 2026-05-13

## Boundary Conclusion

The 30 deg agile fixed-start gate fails after deterministic compact search. The correct outcome is a feasibility-boundary/no-go result, not a successful agile primitive claim.

Frozen gate:

- `abs(actual_heading_change_deg) >= 24`
- `success == True`
- `exit_recoverable == True`
- `min_wall_distance_m > 0`
- `feasibility_label == fixed_start_feasible`

Post-search 30 deg result:

- Candidate: `brake_roll_yaw_recovery / 030_a`
- Actual heading change: `-21.687866218439385 deg`
- Absolute heading change: `21.687866218439385 deg`
- Minimum wall distance: `0.25 m`
- Height change: `-2.1796363483969063 m`
- Terminal speed: `8.1426643951816 m/s`
- Maximum alpha: `7.708991443804745 deg`
- Saturation fraction: `0.38333333333333336`
- Exit recoverable: `False`
- Feasibility label: `fixed_start_unrecoverable`

Gate failure reasons:

- Heading change is below the required `24 deg` acceptance threshold.
- Exit state is not recoverable under the preserved recoverability rule.

## Search Boundary Evidence

Search path:

- `03_Control/05_Results/codex_agile_search/metrics/agile_template_search_best_by_target_seed1.csv`

Best searched candidate by target:

- `30 deg`: `brake_roll_yaw_recovery / 030_a`, `21.69 deg` absolute heading, `fixed_start_unrecoverable`.
- `60 deg`: `brake_roll_yaw_recovery / 060_a`, `33.07 deg` absolute heading, `fixed_start_unsafe_floor_or_ceiling`, altitude below floor.
- `90 deg`: `brake_roll_yaw_recovery / 090_a`, `19.65 deg` absolute heading, `physically_infeasible_candidate`, pitch bound exceeded.
- `120 deg`: `brake_roll_yaw_recovery / 120_a`, `19.65 deg` absolute heading, `physically_infeasible_candidate`, pitch bound exceeded.
- `180 deg`: `brake_roll_yaw_recovery / 180_a`, `19.65 deg` absolute heading, `physically_infeasible_candidate`, pitch bound exceeded.

Final feasibility path:

- `03_Control/05_Results/codex_problem_1_2_7/agile_fixed/metrics/s9_agile_feasibility_seed1.csv`

The final feasibility runner consumed the search manifest and selected:

- `30 deg`: `brake_roll_yaw_recovery / 030_a`
- `60 deg`: `brake_roll_yaw_recovery / 060_a`
- `90 deg`: `brake_roll_yaw_recovery / 090_a`
- `120 deg`: `brake_roll_yaw_recovery / 120_a`
- `180 deg`: `brake_roll_yaw_recovery / 180_a`

Targets above 30 deg are not accepted by the governor in final feasibility because the selected searched candidates are predicted unsafe or physically infeasible under the same preserved bounds.

## Next Action

Do not weaken the gate or relabel this as successful. The next valid action is to change the physical problem or modelling evidence, for example by changing the available manoeuvre volume, initial altitude, confirmed surface authority, or validated aerodynamic envelope, then rerun the same deterministic search and fixed gate.
