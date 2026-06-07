# R11 E03_4 Balanced-Cluster Neutral-Intervention Diagnostic

- Diagnostic version: `r11_balanced_true_neutral_intervention_diagnostic_v1`
- R11 root: `03_Control/05_Results/R11_validation/E03.4`
- Library tier: `balanced_cluster`
- True neutral cases: `160`
- Figure count: `8`

`no_memory_baseline` is still closed-loop governor/LQR control. The neutral baseline here is a separate zero-command plant rollout from the same R11 final launch states.

The open-loop layer is run once per unique R11 final outer case. This is the correct comparison unit because all library tiers and memory policies reuse the same final launch state within that outer case.

| Ladder | Policy | n | Neutral mission | Controlled mission | Mission delta | Neutral score | Controlled score | Score delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| r11_l0_dry_air_fixed | no_memory_baseline | 20 | 0.350 | 0.350 | +0.000 | -5.59 | -1.61 | +3.98 |
| r11_l0_dry_air_fixed | spatial_flow_belief_memory_h10 | 20 | 0.350 | 0.350 | +0.000 | -5.59 | -1.61 | +3.98 |
| r11_l0_dry_air_fixed | spatial_flow_belief_memory_h3 | 20 | 0.350 | 0.350 | +0.000 | -5.59 | -1.61 | +3.98 |
| r11_l0_dry_air_fixed | spatial_flow_belief_memory_h30 | 20 | 0.350 | 0.350 | +0.000 | -5.59 | -1.61 | +3.98 |
| r11_l1_single_fan_fixed_nominal | no_memory_baseline | 20 | 0.450 | 0.600 | +0.150 | 13.00 | 63.79 | +50.79 |
| r11_l1_single_fan_fixed_nominal | spatial_flow_belief_memory_h10 | 20 | 0.450 | 0.600 | +0.150 | 13.00 | 63.79 | +50.79 |
| r11_l1_single_fan_fixed_nominal | spatial_flow_belief_memory_h3 | 20 | 0.450 | 0.600 | +0.150 | 13.00 | 63.79 | +50.79 |
| r11_l1_single_fan_fixed_nominal | spatial_flow_belief_memory_h30 | 20 | 0.450 | 0.600 | +0.150 | 13.00 | 63.79 | +50.79 |
| r11_l2_four_fan_fixed_nominal | no_memory_baseline | 20 | 0.450 | 0.650 | +0.200 | 17.43 | 75.24 | +57.81 |
| r11_l2_four_fan_fixed_nominal | spatial_flow_belief_memory_h10 | 20 | 0.450 | 0.650 | +0.200 | 17.43 | 75.25 | +57.82 |
| r11_l2_four_fan_fixed_nominal | spatial_flow_belief_memory_h3 | 20 | 0.450 | 0.650 | +0.200 | 17.43 | 75.25 | +57.82 |
| r11_l2_four_fan_fixed_nominal | spatial_flow_belief_memory_h30 | 20 | 0.450 | 0.650 | +0.200 | 17.43 | 75.20 | +57.77 |
| r11_l3_fan_parameter_uncertainty | no_memory_baseline | 20 | 0.550 | 0.550 | +0.000 | 33.92 | 57.31 | +23.39 |
| r11_l3_fan_parameter_uncertainty | spatial_flow_belief_memory_h10 | 20 | 0.550 | 0.600 | +0.050 | 33.92 | 70.58 | +36.65 |
| r11_l3_fan_parameter_uncertainty | spatial_flow_belief_memory_h3 | 20 | 0.550 | 0.600 | +0.050 | 33.92 | 70.58 | +36.65 |
| r11_l3_fan_parameter_uncertainty | spatial_flow_belief_memory_h30 | 20 | 0.550 | 0.550 | +0.000 | 33.92 | 57.31 | +23.39 |
| r11_l4_local_fan_position_uncertainty | no_memory_baseline | 20 | 0.500 | 0.700 | +0.200 | 23.93 | 88.07 | +64.14 |
| r11_l4_local_fan_position_uncertainty | spatial_flow_belief_memory_h10 | 20 | 0.500 | 0.700 | +0.200 | 23.93 | 87.75 | +63.82 |
| r11_l4_local_fan_position_uncertainty | spatial_flow_belief_memory_h3 | 20 | 0.500 | 0.700 | +0.200 | 23.93 | 87.75 | +63.82 |
| r11_l4_local_fan_position_uncertainty | spatial_flow_belief_memory_h30 | 20 | 0.500 | 0.700 | +0.200 | 23.93 | 85.06 | +61.14 |
| r11_l5_active_fan_count_uncertainty | no_memory_baseline | 20 | 0.400 | 0.550 | +0.150 | 5.90 | 48.79 | +42.89 |
| r11_l5_active_fan_count_uncertainty | spatial_flow_belief_memory_h10 | 20 | 0.400 | 0.550 | +0.150 | 5.90 | 48.78 | +42.89 |
| r11_l5_active_fan_count_uncertainty | spatial_flow_belief_memory_h3 | 20 | 0.400 | 0.550 | +0.150 | 5.90 | 48.78 | +42.89 |
| r11_l5_active_fan_count_uncertainty | spatial_flow_belief_memory_h30 | 20 | 0.400 | 0.550 | +0.150 | 5.90 | 48.78 | +42.89 |
| r11_l6_environment_only_full_uncertainty | no_memory_baseline | 20 | 0.400 | 0.500 | +0.100 | 4.13 | 35.45 | +31.33 |
| r11_l6_environment_only_full_uncertainty | spatial_flow_belief_memory_h10 | 20 | 0.400 | 0.500 | +0.100 | 4.13 | 35.45 | +31.33 |
| r11_l6_environment_only_full_uncertainty | spatial_flow_belief_memory_h3 | 20 | 0.400 | 0.500 | +0.100 | 4.13 | 35.45 | +31.33 |
| r11_l6_environment_only_full_uncertainty | spatial_flow_belief_memory_h30 | 20 | 0.400 | 0.500 | +0.100 | 4.13 | 35.37 | +31.25 |
| r11_l7_full_domain_randomisation_arena_wide | no_memory_baseline | 20 | 0.450 | 0.500 | +0.050 | 14.36 | 43.79 | +29.43 |
| r11_l7_full_domain_randomisation_arena_wide | spatial_flow_belief_memory_h10 | 20 | 0.450 | 0.500 | +0.050 | 14.36 | 43.79 | +29.43 |
| r11_l7_full_domain_randomisation_arena_wide | spatial_flow_belief_memory_h3 | 20 | 0.450 | 0.500 | +0.050 | 14.36 | 43.79 | +29.43 |
| r11_l7_full_domain_randomisation_arena_wide | spatial_flow_belief_memory_h30 | 20 | 0.450 | 0.500 | +0.050 | 14.36 | 43.79 | +29.43 |

Figures:

- `r11_l0_dry_air_fixed`: `03_Control/A_figures/R11_E03.4_balanced_neutral_baseline/figures/r11_e03_4_bal_neutral_s18_l0.png`
- `r11_l1_single_fan_fixed_nominal`: `03_Control/A_figures/R11_E03.4_balanced_neutral_baseline/figures/r11_e03_4_bal_neutral_s18_l1.png`
- `r11_l2_four_fan_fixed_nominal`: `03_Control/A_figures/R11_E03.4_balanced_neutral_baseline/figures/r11_e03_4_bal_neutral_s18_l2.png`
- `r11_l3_fan_parameter_uncertainty`: `03_Control/A_figures/R11_E03.4_balanced_neutral_baseline/figures/r11_e03_4_bal_neutral_s18_l3.png`
- `r11_l4_local_fan_position_uncertainty`: `03_Control/A_figures/R11_E03.4_balanced_neutral_baseline/figures/r11_e03_4_bal_neutral_s18_l4.png`
- `r11_l5_active_fan_count_uncertainty`: `03_Control/A_figures/R11_E03.4_balanced_neutral_baseline/figures/r11_e03_4_bal_neutral_s18_l5.png`
- `r11_l6_environment_only_full_uncertainty`: `03_Control/A_figures/R11_E03.4_balanced_neutral_baseline/figures/r11_e03_4_bal_neutral_s18_l6.png`
- `r11_l7_full_domain_randomisation_arena_wide`: `03_Control/A_figures/R11_E03.4_balanced_neutral_baseline/figures/r11_e03_4_bal_neutral_s18_l7.png`
