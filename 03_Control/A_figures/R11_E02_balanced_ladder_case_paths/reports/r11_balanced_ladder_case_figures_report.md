# R11 E02 Balanced-Cluster Same-Start Ladder Figures

- Figure run version: `r11_balanced_same_start_ladder_case_paths_v1`
- R11 root: `03_Control/05_Results/R11_validation/E02`
- Library tier: `balanced_cluster`
- Paired start condition index: `0`
- Neutral open-loop source: `03_Control/A_figures/R11_E02_balanced_neutral_baseline/metrics/neutral_rollout_by_case.csv`
- Figure count: `8`

The fan layout and updraft parameters are reconstructed from each R11 outer-case row using the stored layout, active-count, and parameter seeds.

| Ladder | Outer case | Active fans | Updraft max (m/s) | Open-loop plotted | Open-loop target | Figure |
|---|---:|---:|---:|---:|---:|---|
| r11_l0_dry_air_fixed | 0 | 0/0 | 0.000 | 1 | 1 | `03_Control/A_figures/R11_E02_balanced_ladder_case_paths/figures/r11_e02_bal_s00_l0.png` |
| r11_l1_single_fan_fixed_nominal | 2 | 1/1 | 5.211 | 1 | 1 | `03_Control/A_figures/R11_E02_balanced_ladder_case_paths/figures/r11_e02_bal_s00_l1.png` |
| r11_l2_four_fan_fixed_nominal | 4 | 4/4 | 6.200 | 1 | 1 | `03_Control/A_figures/R11_E02_balanced_ladder_case_paths/figures/r11_e02_bal_s00_l2.png` |
| r11_l3_fan_parameter_uncertainty | 6 | 4/4 | 5.507 | 1 | 1 | `03_Control/A_figures/R11_E02_balanced_ladder_case_paths/figures/r11_e02_bal_s00_l3.png` |
| r11_l4_local_fan_position_uncertainty | 8 | 4/4 | 5.580 | 1 | 1 | `03_Control/A_figures/R11_E02_balanced_ladder_case_paths/figures/r11_e02_bal_s00_l4.png` |
| r11_l5_active_fan_count_uncertainty | 10 | 0/4 | 0.000 | 1 | 1 | `03_Control/A_figures/R11_E02_balanced_ladder_case_paths/figures/r11_e02_bal_s00_l5.png` |
| r11_l6_environment_only_full_uncertainty | 12 | 0/4 | 0.000 | 1 | 1 | `03_Control/A_figures/R11_E02_balanced_ladder_case_paths/figures/r11_e02_bal_s00_l6.png` |
| r11_l7_full_domain_randomisation_arena_wide | 14 | 0/4 | 0.000 | 1 | 1 | `03_Control/A_figures/R11_E02_balanced_ladder_case_paths/figures/r11_e02_bal_s00_l7.png` |
