# R11 E01 Balanced-Cluster Same-Start Ladder Figures

- Figure run version: `r11_balanced_same_start_ladder_case_paths_v1`
- R11 root: `03_Control/05_Results/R11_validation/E01`
- Library tier: `balanced_cluster`
- Paired start condition index: `23`
- Neutral open-loop source: `03_Control/A_figures/R11_E01_balanced_neutral_baseline/metrics/neutral_rollout_by_case.csv`
- Figure count: `8`

The fan layout and updraft parameters are reconstructed from each R11 outer-case row using the stored layout, active-count, and parameter seeds.

| Ladder | Outer case | Active fans | Updraft max (m/s) | Open-loop plotted | Open-loop target | Figure |
|---|---:|---:|---:|---:|---:|---|
| r11_l0_dry_air_fixed | 23 | 0/0 | 0.000 | 1 | 1 | `03_Control/A_figures/R11_E01_balanced_paths/figures/r11_e01_bal_s23_l0.png` |
| r11_l1_single_fan_fixed_nominal | 73 | 1/1 | 5.211 | 1 | 1 | `03_Control/A_figures/R11_E01_balanced_paths/figures/r11_e01_bal_s23_l1.png` |
| r11_l2_four_fan_fixed_nominal | 123 | 4/4 | 6.200 | 1 | 0 | `03_Control/A_figures/R11_E01_balanced_paths/figures/r11_e01_bal_s23_l2.png` |
| r11_l3_fan_parameter_uncertainty | 173 | 4/4 | 5.558 | 1 | 1 | `03_Control/A_figures/R11_E01_balanced_paths/figures/r11_e01_bal_s23_l3.png` |
| r11_l4_local_fan_position_uncertainty | 223 | 4/4 | 5.504 | 1 | 0 | `03_Control/A_figures/R11_E01_balanced_paths/figures/r11_e01_bal_s23_l4.png` |
| r11_l5_active_fan_count_uncertainty | 273 | 3/4 | 5.171 | 1 | 1 | `03_Control/A_figures/R11_E01_balanced_paths/figures/r11_e01_bal_s23_l5.png` |
| r11_l6_environment_only_full_uncertainty | 323 | 3/4 | 5.930 | 1 | 1 | `03_Control/A_figures/R11_E01_balanced_paths/figures/r11_e01_bal_s23_l6.png` |
| r11_l7_full_domain_randomisation_arena_wide | 373 | 3/4 | 4.754 | 1 | 0 | `03_Control/A_figures/R11_E01_balanced_paths/figures/r11_e01_bal_s23_l7.png` |
