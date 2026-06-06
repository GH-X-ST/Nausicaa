# R11 E03 Extreme-Case Trajectory Figures

- Figure run version: `r11_e03_extreme_case_paths_v1`
- R11 root: `03_Control/05_Results/R11_validation/E03.1`
- Neutral open-loop source: `03_Control/A_figures/R11_E03_balanced_neutral_baseline/metrics/neutral_rollout_by_case.csv`
- Default open/closed library tier: `balanced_cluster`
- Memory selection scope: `all`
- Figure count: `3`

Each figure uses the same R11 E03 trajectory style as the balanced ladder figures: seeded fan/updraft context, true neutral open-loop path when available, and final held-out closed-loop policy paths.

| Category | Reason | Library tier | Ladder | Outer case | Selected policy | Score delta | Figure |
|---|---|---|---|---:|---|---:|---|
| open_closed_largest_score_gap | largest_abs_true_neutral_open_loop_vs_best_closed_loop_score_gap | balanced_cluster | r11_l4_local_fan_position_uncertainty | 93 | no_memory_baseline | 233.702 | `03_Control/A_figures/R11_E03_extreme_case_paths/figures/r11_e03_1_open_closed_largest_score_gap_balanced_cluster_l4_case0093.png` |
| memory_largest_score_impact | largest_positive_memory_delta_with_selection_change | no_cluster_no_merge | r11_l5_active_fan_count_uncertainty | 107 | spatial_flow_belief_memory_h3 | 60.000 | `03_Control/A_figures/R11_E03_extreme_case_paths/figures/r11_e03_1_memory_largest_score_impact_no_cluster_no_merge_l5_case0107.png` |
| open_fail_closed_success | largest_true_neutral_open_loop_failure_to_closed_loop_success_score_gain | balanced_cluster | r11_l4_local_fan_position_uncertainty | 93 | no_memory_baseline | 233.702 | `03_Control/A_figures/R11_E03_extreme_case_paths/figures/r11_e03_1_open_fail_closed_success_balanced_cluster_l4_case0093.png` |
