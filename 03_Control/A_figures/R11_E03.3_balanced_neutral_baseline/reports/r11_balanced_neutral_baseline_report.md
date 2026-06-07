# R11 E03_3 Balanced-Cluster Neutral-Intervention Diagnostic

- Diagnostic version: `r11_balanced_true_neutral_intervention_diagnostic_v1`
- R11 root: `03_Control/05_Results/R11_validation/E03.3`
- Library tier: `balanced_cluster`
- True neutral cases: `160`
- Figure count: `8`

`no_memory_baseline` is still closed-loop governor/LQR control. The neutral baseline here is a separate zero-command plant rollout from the same R11 final launch states.

The open-loop layer is run once per unique R11 final outer case. This is the correct comparison unit because all library tiers and memory policies reuse the same final launch state within that outer case.

| Ladder | Policy | n | Neutral mission | Controlled mission | Mission delta | Neutral score | Controlled score | Score delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| r11_l0_dry_air_fixed | no_memory_baseline | 20 | 0.400 | 0.550 | +0.150 | 0.92 | 41.48 | +40.56 |
| r11_l0_dry_air_fixed | spatial_flow_belief_memory_h10 | 20 | 0.400 | 0.550 | +0.150 | 0.92 | 41.48 | +40.56 |
| r11_l0_dry_air_fixed | spatial_flow_belief_memory_h3 | 20 | 0.400 | 0.550 | +0.150 | 0.92 | 41.48 | +40.56 |
| r11_l0_dry_air_fixed | spatial_flow_belief_memory_h30 | 20 | 0.400 | 0.550 | +0.150 | 0.92 | 41.48 | +40.56 |
| r11_l1_single_fan_fixed_nominal | no_memory_baseline | 20 | 0.650 | 0.750 | +0.100 | 51.95 | 85.71 | +33.75 |
| r11_l1_single_fan_fixed_nominal | spatial_flow_belief_memory_h10 | 20 | 0.650 | 0.750 | +0.100 | 51.95 | 85.71 | +33.75 |
| r11_l1_single_fan_fixed_nominal | spatial_flow_belief_memory_h3 | 20 | 0.650 | 0.750 | +0.100 | 51.95 | 85.71 | +33.75 |
| r11_l1_single_fan_fixed_nominal | spatial_flow_belief_memory_h30 | 20 | 0.650 | 0.750 | +0.100 | 51.95 | 85.71 | +33.75 |
| r11_l2_four_fan_fixed_nominal | no_memory_baseline | 20 | 0.700 | 0.800 | +0.100 | 65.53 | 108.29 | +42.76 |
| r11_l2_four_fan_fixed_nominal | spatial_flow_belief_memory_h10 | 20 | 0.700 | 0.800 | +0.100 | 65.53 | 108.32 | +42.79 |
| r11_l2_four_fan_fixed_nominal | spatial_flow_belief_memory_h3 | 20 | 0.700 | 0.800 | +0.100 | 65.53 | 108.29 | +42.76 |
| r11_l2_four_fan_fixed_nominal | spatial_flow_belief_memory_h30 | 20 | 0.700 | 0.800 | +0.100 | 65.53 | 108.31 | +42.78 |
| r11_l3_fan_parameter_uncertainty | no_memory_baseline | 20 | 0.650 | 0.750 | +0.100 | 54.04 | 93.27 | +39.23 |
| r11_l3_fan_parameter_uncertainty | spatial_flow_belief_memory_h10 | 20 | 0.650 | 0.750 | +0.100 | 54.04 | 93.27 | +39.23 |
| r11_l3_fan_parameter_uncertainty | spatial_flow_belief_memory_h3 | 20 | 0.650 | 0.750 | +0.100 | 54.04 | 93.27 | +39.23 |
| r11_l3_fan_parameter_uncertainty | spatial_flow_belief_memory_h30 | 20 | 0.650 | 0.750 | +0.100 | 54.04 | 93.27 | +39.23 |
| r11_l4_local_fan_position_uncertainty | no_memory_baseline | 20 | 0.600 | 0.750 | +0.150 | 44.54 | 93.15 | +48.62 |
| r11_l4_local_fan_position_uncertainty | spatial_flow_belief_memory_h10 | 20 | 0.600 | 0.750 | +0.150 | 44.54 | 93.16 | +48.62 |
| r11_l4_local_fan_position_uncertainty | spatial_flow_belief_memory_h3 | 20 | 0.600 | 0.750 | +0.150 | 44.54 | 93.16 | +48.62 |
| r11_l4_local_fan_position_uncertainty | spatial_flow_belief_memory_h30 | 20 | 0.600 | 0.750 | +0.150 | 44.54 | 93.16 | +48.62 |
| r11_l5_active_fan_count_uncertainty | no_memory_baseline | 20 | 0.650 | 0.650 | +0.000 | 51.50 | 69.95 | +18.44 |
| r11_l5_active_fan_count_uncertainty | spatial_flow_belief_memory_h10 | 20 | 0.650 | 0.650 | +0.000 | 51.50 | 69.95 | +18.44 |
| r11_l5_active_fan_count_uncertainty | spatial_flow_belief_memory_h3 | 20 | 0.650 | 0.650 | +0.000 | 51.50 | 69.95 | +18.44 |
| r11_l5_active_fan_count_uncertainty | spatial_flow_belief_memory_h30 | 20 | 0.650 | 0.650 | +0.000 | 51.50 | 69.94 | +18.44 |
| r11_l6_environment_only_full_uncertainty | no_memory_baseline | 20 | 0.550 | 0.650 | +0.100 | 31.70 | 63.24 | +31.54 |
| r11_l6_environment_only_full_uncertainty | spatial_flow_belief_memory_h10 | 20 | 0.550 | 0.650 | +0.100 | 31.70 | 63.24 | +31.54 |
| r11_l6_environment_only_full_uncertainty | spatial_flow_belief_memory_h3 | 20 | 0.550 | 0.650 | +0.100 | 31.70 | 63.24 | +31.54 |
| r11_l6_environment_only_full_uncertainty | spatial_flow_belief_memory_h30 | 20 | 0.550 | 0.650 | +0.100 | 31.70 | 63.22 | +31.51 |
| r11_l7_full_domain_randomisation_arena_wide | no_memory_baseline | 20 | 0.350 | 0.500 | +0.150 | -8.18 | 34.04 | +42.22 |
| r11_l7_full_domain_randomisation_arena_wide | spatial_flow_belief_memory_h10 | 20 | 0.350 | 0.500 | +0.150 | -8.18 | 34.04 | +42.22 |
| r11_l7_full_domain_randomisation_arena_wide | spatial_flow_belief_memory_h3 | 20 | 0.350 | 0.500 | +0.150 | -8.18 | 34.04 | +42.22 |
| r11_l7_full_domain_randomisation_arena_wide | spatial_flow_belief_memory_h30 | 20 | 0.350 | 0.500 | +0.150 | -8.18 | 34.04 | +42.22 |

Figures:

- `r11_l0_dry_air_fixed`: `03_Control/A_figures/R11_E03.3_balanced_neutral_baseline/figures/r11_e03_3_bal_neutral_s49_l0.png`
- `r11_l1_single_fan_fixed_nominal`: `03_Control/A_figures/R11_E03.3_balanced_neutral_baseline/figures/r11_e03_3_bal_neutral_s49_l1.png`
- `r11_l2_four_fan_fixed_nominal`: `03_Control/A_figures/R11_E03.3_balanced_neutral_baseline/figures/r11_e03_3_bal_neutral_s49_l2.png`
- `r11_l3_fan_parameter_uncertainty`: `03_Control/A_figures/R11_E03.3_balanced_neutral_baseline/figures/r11_e03_3_bal_neutral_s49_l3.png`
- `r11_l4_local_fan_position_uncertainty`: `03_Control/A_figures/R11_E03.3_balanced_neutral_baseline/figures/r11_e03_3_bal_neutral_s49_l4.png`
- `r11_l5_active_fan_count_uncertainty`: `03_Control/A_figures/R11_E03.3_balanced_neutral_baseline/figures/r11_e03_3_bal_neutral_s49_l5.png`
- `r11_l6_environment_only_full_uncertainty`: `03_Control/A_figures/R11_E03.3_balanced_neutral_baseline/figures/r11_e03_3_bal_neutral_s49_l6.png`
- `r11_l7_full_domain_randomisation_arena_wide`: `03_Control/A_figures/R11_E03.3_balanced_neutral_baseline/figures/r11_e03_3_bal_neutral_s49_l7.png`
