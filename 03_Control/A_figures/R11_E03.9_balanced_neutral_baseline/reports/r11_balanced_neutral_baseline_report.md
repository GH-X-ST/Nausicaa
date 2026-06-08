# R11 E03_9 Balanced-Cluster Neutral-Intervention Diagnostic

- Diagnostic version: `r11_balanced_true_neutral_intervention_diagnostic_v1`
- R11 root: `03_Control/05_Results/R11_validation/E03.9`
- Library tier: `balanced_cluster`
- True neutral cases: `160`
- Figure count: `8`

`no_memory_baseline` is still closed-loop governor/LQR control. The neutral baseline here is a separate zero-command plant rollout from the same R11 final launch states.

The open-loop layer is run once per unique R11 final outer case. This is the correct comparison unit because all library tiers and memory policies reuse the same final launch state within that outer case.

| Ladder | Policy | n | Neutral mission | Controlled mission | Mission delta | Neutral score | Controlled score | Score delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| r11_l0_dry_air_fixed | no_memory_baseline | 20 | 0.550 | 0.550 | +0.000 | 29.23 | 40.37 | +11.14 |
| r11_l0_dry_air_fixed | spatial_flow_belief_memory_h10 | 20 | 0.550 | 0.550 | +0.000 | 29.23 | 40.37 | +11.14 |
| r11_l0_dry_air_fixed | spatial_flow_belief_memory_h3 | 20 | 0.550 | 0.550 | +0.000 | 29.23 | 40.37 | +11.14 |
| r11_l0_dry_air_fixed | spatial_flow_belief_memory_h30 | 20 | 0.550 | 0.550 | +0.000 | 29.23 | 40.37 | +11.14 |
| r11_l1_single_fan_fixed_nominal | no_memory_baseline | 20 | 0.700 | 0.800 | +0.100 | 58.76 | 99.22 | +40.46 |
| r11_l1_single_fan_fixed_nominal | spatial_flow_belief_memory_h10 | 20 | 0.700 | 0.800 | +0.100 | 58.76 | 99.22 | +40.46 |
| r11_l1_single_fan_fixed_nominal | spatial_flow_belief_memory_h3 | 20 | 0.700 | 0.800 | +0.100 | 58.76 | 99.22 | +40.46 |
| r11_l1_single_fan_fixed_nominal | spatial_flow_belief_memory_h30 | 20 | 0.700 | 0.800 | +0.100 | 58.76 | 99.22 | +40.46 |
| r11_l2_four_fan_fixed_nominal | no_memory_baseline | 20 | 0.850 | 0.950 | +0.100 | 89.83 | 135.20 | +45.37 |
| r11_l2_four_fan_fixed_nominal | spatial_flow_belief_memory_h10 | 20 | 0.850 | 0.950 | +0.100 | 89.83 | 135.20 | +45.37 |
| r11_l2_four_fan_fixed_nominal | spatial_flow_belief_memory_h3 | 20 | 0.850 | 0.950 | +0.100 | 89.83 | 135.20 | +45.37 |
| r11_l2_four_fan_fixed_nominal | spatial_flow_belief_memory_h30 | 20 | 0.850 | 0.950 | +0.100 | 89.83 | 135.20 | +45.37 |
| r11_l3_fan_parameter_uncertainty | no_memory_baseline | 20 | 0.750 | 0.900 | +0.150 | 68.60 | 121.05 | +52.44 |
| r11_l3_fan_parameter_uncertainty | spatial_flow_belief_memory_h10 | 20 | 0.750 | 0.900 | +0.150 | 68.60 | 121.04 | +52.44 |
| r11_l3_fan_parameter_uncertainty | spatial_flow_belief_memory_h3 | 20 | 0.750 | 0.900 | +0.150 | 68.60 | 121.04 | +52.44 |
| r11_l3_fan_parameter_uncertainty | spatial_flow_belief_memory_h30 | 20 | 0.750 | 0.900 | +0.150 | 68.60 | 121.04 | +52.44 |
| r11_l4_local_fan_position_uncertainty | no_memory_baseline | 20 | 0.700 | 0.800 | +0.100 | 59.56 | 102.72 | +43.16 |
| r11_l4_local_fan_position_uncertainty | spatial_flow_belief_memory_h10 | 20 | 0.700 | 0.850 | +0.150 | 59.56 | 113.71 | +54.15 |
| r11_l4_local_fan_position_uncertainty | spatial_flow_belief_memory_h3 | 20 | 0.700 | 0.850 | +0.150 | 59.56 | 113.71 | +54.15 |
| r11_l4_local_fan_position_uncertainty | spatial_flow_belief_memory_h30 | 20 | 0.700 | 0.800 | +0.100 | 59.56 | 102.71 | +43.15 |
| r11_l5_active_fan_count_uncertainty | no_memory_baseline | 20 | 0.650 | 0.700 | +0.050 | 50.08 | 77.30 | +27.22 |
| r11_l5_active_fan_count_uncertainty | spatial_flow_belief_memory_h10 | 20 | 0.650 | 0.700 | +0.050 | 50.08 | 77.30 | +27.22 |
| r11_l5_active_fan_count_uncertainty | spatial_flow_belief_memory_h3 | 20 | 0.650 | 0.700 | +0.050 | 50.08 | 77.30 | +27.22 |
| r11_l5_active_fan_count_uncertainty | spatial_flow_belief_memory_h30 | 20 | 0.650 | 0.700 | +0.050 | 50.08 | 77.31 | +27.23 |
| r11_l6_environment_only_full_uncertainty | no_memory_baseline | 20 | 0.650 | 0.700 | +0.050 | 50.13 | 80.15 | +30.02 |
| r11_l6_environment_only_full_uncertainty | spatial_flow_belief_memory_h10 | 20 | 0.650 | 0.750 | +0.100 | 50.13 | 88.08 | +37.95 |
| r11_l6_environment_only_full_uncertainty | spatial_flow_belief_memory_h3 | 20 | 0.650 | 0.750 | +0.100 | 50.13 | 88.08 | +37.95 |
| r11_l6_environment_only_full_uncertainty | spatial_flow_belief_memory_h30 | 20 | 0.650 | 0.700 | +0.050 | 50.13 | 80.15 | +30.02 |
| r11_l7_full_domain_randomisation_arena_wide | no_memory_baseline | 20 | 0.750 | 0.700 | -0.050 | 68.51 | 67.74 | -0.77 |
| r11_l7_full_domain_randomisation_arena_wide | spatial_flow_belief_memory_h10 | 20 | 0.750 | 0.700 | -0.050 | 68.51 | 67.73 | -0.78 |
| r11_l7_full_domain_randomisation_arena_wide | spatial_flow_belief_memory_h3 | 20 | 0.750 | 0.700 | -0.050 | 68.51 | 67.73 | -0.78 |
| r11_l7_full_domain_randomisation_arena_wide | spatial_flow_belief_memory_h30 | 20 | 0.750 | 0.700 | -0.050 | 68.51 | 67.73 | -0.78 |

Figures:

- `r11_l0_dry_air_fixed`: `03_Control/A_figures/R11_E03.9_balanced_neutral_baseline/figures/r11_e03_9_bal_neutral_s21_l0.png`
- `r11_l1_single_fan_fixed_nominal`: `03_Control/A_figures/R11_E03.9_balanced_neutral_baseline/figures/r11_e03_9_bal_neutral_s21_l1.png`
- `r11_l2_four_fan_fixed_nominal`: `03_Control/A_figures/R11_E03.9_balanced_neutral_baseline/figures/r11_e03_9_bal_neutral_s21_l2.png`
- `r11_l3_fan_parameter_uncertainty`: `03_Control/A_figures/R11_E03.9_balanced_neutral_baseline/figures/r11_e03_9_bal_neutral_s21_l3.png`
- `r11_l4_local_fan_position_uncertainty`: `03_Control/A_figures/R11_E03.9_balanced_neutral_baseline/figures/r11_e03_9_bal_neutral_s21_l4.png`
- `r11_l5_active_fan_count_uncertainty`: `03_Control/A_figures/R11_E03.9_balanced_neutral_baseline/figures/r11_e03_9_bal_neutral_s21_l5.png`
- `r11_l6_environment_only_full_uncertainty`: `03_Control/A_figures/R11_E03.9_balanced_neutral_baseline/figures/r11_e03_9_bal_neutral_s21_l6.png`
- `r11_l7_full_domain_randomisation_arena_wide`: `03_Control/A_figures/R11_E03.9_balanced_neutral_baseline/figures/r11_e03_9_bal_neutral_s21_l7.png`
