# R11 E01 Balanced-Cluster Neutral-Intervention Diagnostic

- Diagnostic version: `r11_balanced_true_neutral_intervention_diagnostic_v1`
- R11 root: `03_Control/05_Results/R11_validation/E01`
- Library tier: `balanced_cluster`
- True neutral cases: `400`
- Figure count: `8`

`no_memory_baseline` is still closed-loop governor/LQR control. The neutral baseline here is a separate zero-command plant rollout from the same R11 final launch states.

| Ladder | Policy | n | Neutral mission | Controlled mission | Mission delta | Neutral score | Controlled score | Score delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| r11_l0_dry_air_fixed | no_memory_baseline | 50 | 0.680 | 0.780 | +0.100 | 57.33 | 90.03 | +32.70 |
| r11_l0_dry_air_fixed | spatial_flow_belief_memory_h10 | 50 | 0.680 | 0.780 | +0.100 | 57.33 | 90.01 | +32.68 |
| r11_l0_dry_air_fixed | spatial_flow_belief_memory_h3 | 50 | 0.680 | 0.780 | +0.100 | 57.33 | 90.02 | +32.69 |
| r11_l0_dry_air_fixed | spatial_flow_belief_memory_h30 | 50 | 0.680 | 0.780 | +0.100 | 57.33 | 90.02 | +32.68 |
| r11_l1_single_fan_fixed_nominal | no_memory_baseline | 50 | 0.500 | 0.720 | +0.220 | 24.55 | 88.50 | +63.96 |
| r11_l1_single_fan_fixed_nominal | spatial_flow_belief_memory_h10 | 50 | 0.500 | 0.720 | +0.220 | 24.55 | 88.54 | +63.99 |
| r11_l1_single_fan_fixed_nominal | spatial_flow_belief_memory_h3 | 50 | 0.500 | 0.720 | +0.220 | 24.55 | 88.54 | +64.00 |
| r11_l1_single_fan_fixed_nominal | spatial_flow_belief_memory_h30 | 50 | 0.500 | 0.720 | +0.220 | 24.55 | 87.54 | +63.00 |
| r11_l2_four_fan_fixed_nominal | no_memory_baseline | 50 | 0.380 | 0.740 | +0.360 | 1.67 | 93.50 | +91.83 |
| r11_l2_four_fan_fixed_nominal | spatial_flow_belief_memory_h10 | 50 | 0.380 | 0.760 | +0.380 | 1.67 | 97.38 | +95.72 |
| r11_l2_four_fan_fixed_nominal | spatial_flow_belief_memory_h3 | 50 | 0.380 | 0.760 | +0.380 | 1.67 | 97.38 | +95.71 |
| r11_l2_four_fan_fixed_nominal | spatial_flow_belief_memory_h30 | 50 | 0.380 | 0.760 | +0.380 | 1.67 | 97.39 | +95.73 |
| r11_l3_fan_parameter_uncertainty | no_memory_baseline | 50 | 0.440 | 0.720 | +0.280 | 11.52 | 89.71 | +78.19 |
| r11_l3_fan_parameter_uncertainty | spatial_flow_belief_memory_h10 | 50 | 0.440 | 0.720 | +0.280 | 11.52 | 89.89 | +78.37 |
| r11_l3_fan_parameter_uncertainty | spatial_flow_belief_memory_h3 | 50 | 0.440 | 0.720 | +0.280 | 11.52 | 89.72 | +78.20 |
| r11_l3_fan_parameter_uncertainty | spatial_flow_belief_memory_h30 | 50 | 0.440 | 0.720 | +0.280 | 11.52 | 89.68 | +78.16 |
| r11_l4_local_fan_position_uncertainty | no_memory_baseline | 50 | 0.460 | 0.740 | +0.280 | 15.27 | 95.79 | +80.51 |
| r11_l4_local_fan_position_uncertainty | spatial_flow_belief_memory_h10 | 50 | 0.460 | 0.720 | +0.260 | 15.27 | 91.33 | +76.06 |
| r11_l4_local_fan_position_uncertainty | spatial_flow_belief_memory_h3 | 50 | 0.460 | 0.720 | +0.260 | 15.27 | 91.34 | +76.07 |
| r11_l4_local_fan_position_uncertainty | spatial_flow_belief_memory_h30 | 50 | 0.460 | 0.720 | +0.260 | 15.27 | 91.31 | +76.04 |
| r11_l5_active_fan_count_uncertainty | no_memory_baseline | 50 | 0.620 | 0.780 | +0.160 | 45.62 | 95.49 | +49.87 |
| r11_l5_active_fan_count_uncertainty | spatial_flow_belief_memory_h10 | 50 | 0.620 | 0.760 | +0.140 | 45.62 | 92.12 | +46.50 |
| r11_l5_active_fan_count_uncertainty | spatial_flow_belief_memory_h3 | 50 | 0.620 | 0.760 | +0.140 | 45.62 | 92.13 | +46.51 |
| r11_l5_active_fan_count_uncertainty | spatial_flow_belief_memory_h30 | 50 | 0.620 | 0.780 | +0.160 | 45.62 | 95.47 | +49.85 |
| r11_l6_environment_only_full_uncertainty | no_memory_baseline | 50 | 0.520 | 0.740 | +0.220 | 27.62 | 90.13 | +62.51 |
| r11_l6_environment_only_full_uncertainty | spatial_flow_belief_memory_h10 | 50 | 0.520 | 0.740 | +0.220 | 27.62 | 90.84 | +63.22 |
| r11_l6_environment_only_full_uncertainty | spatial_flow_belief_memory_h3 | 50 | 0.520 | 0.720 | +0.200 | 27.62 | 87.46 | +59.83 |
| r11_l6_environment_only_full_uncertainty | spatial_flow_belief_memory_h30 | 50 | 0.520 | 0.740 | +0.220 | 27.62 | 90.90 | +63.28 |
| r11_l7_full_domain_randomisation_arena_wide | no_memory_baseline | 50 | 0.560 | 0.740 | +0.180 | 32.22 | 87.30 | +55.07 |
| r11_l7_full_domain_randomisation_arena_wide | spatial_flow_belief_memory_h10 | 50 | 0.560 | 0.760 | +0.200 | 32.22 | 89.68 | +57.46 |
| r11_l7_full_domain_randomisation_arena_wide | spatial_flow_belief_memory_h3 | 50 | 0.560 | 0.760 | +0.200 | 32.22 | 89.68 | +57.46 |
| r11_l7_full_domain_randomisation_arena_wide | spatial_flow_belief_memory_h30 | 50 | 0.560 | 0.760 | +0.200 | 32.22 | 90.48 | +58.26 |

Figures:

- `r11_l0_dry_air_fixed`: `03_Control/A_figures/R11_E01_balanced_neutral_baseline/figures/r11_e01_bal_neutral_s23_l0.png`
- `r11_l1_single_fan_fixed_nominal`: `03_Control/A_figures/R11_E01_balanced_neutral_baseline/figures/r11_e01_bal_neutral_s23_l1.png`
- `r11_l2_four_fan_fixed_nominal`: `03_Control/A_figures/R11_E01_balanced_neutral_baseline/figures/r11_e01_bal_neutral_s23_l2.png`
- `r11_l3_fan_parameter_uncertainty`: `03_Control/A_figures/R11_E01_balanced_neutral_baseline/figures/r11_e01_bal_neutral_s23_l3.png`
- `r11_l4_local_fan_position_uncertainty`: `03_Control/A_figures/R11_E01_balanced_neutral_baseline/figures/r11_e01_bal_neutral_s23_l4.png`
- `r11_l5_active_fan_count_uncertainty`: `03_Control/A_figures/R11_E01_balanced_neutral_baseline/figures/r11_e01_bal_neutral_s23_l5.png`
- `r11_l6_environment_only_full_uncertainty`: `03_Control/A_figures/R11_E01_balanced_neutral_baseline/figures/r11_e01_bal_neutral_s23_l6.png`
- `r11_l7_full_domain_randomisation_arena_wide`: `03_Control/A_figures/R11_E01_balanced_neutral_baseline/figures/r11_e01_bal_neutral_s23_l7.png`
