# R11 E03_1 Balanced-Cluster Neutral-Intervention Diagnostic

- Diagnostic version: `r11_balanced_true_neutral_intervention_diagnostic_v1`
- R11 root: `03_Control/05_Results/R11_validation/E03.1`
- Library tier: `balanced_cluster`
- True neutral cases: `160`
- Figure count: `8`

`no_memory_baseline` is still closed-loop governor/LQR control. The neutral baseline here is a separate zero-command plant rollout from the same R11 final launch states.

The open-loop layer is run once per unique R11 final outer case. This is the correct comparison unit because all library tiers and memory policies reuse the same final launch state within that outer case.

| Ladder | Policy | n | Neutral mission | Controlled mission | Mission delta | Neutral score | Controlled score | Score delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| r11_l0_dry_air_fixed | no_memory_baseline | 20 | 0.400 | 0.400 | +0.000 | 2.71 | 17.41 | +14.70 |
| r11_l0_dry_air_fixed | spatial_flow_belief_memory_h10 | 20 | 0.400 | 0.400 | +0.000 | 2.71 | 17.41 | +14.70 |
| r11_l0_dry_air_fixed | spatial_flow_belief_memory_h3 | 20 | 0.400 | 0.400 | +0.000 | 2.71 | 17.41 | +14.70 |
| r11_l0_dry_air_fixed | spatial_flow_belief_memory_h30 | 20 | 0.400 | 0.400 | +0.000 | 2.71 | 17.41 | +14.70 |
| r11_l1_single_fan_fixed_nominal | no_memory_baseline | 20 | 0.400 | 0.550 | +0.150 | 4.35 | 44.12 | +39.78 |
| r11_l1_single_fan_fixed_nominal | spatial_flow_belief_memory_h10 | 20 | 0.400 | 0.550 | +0.150 | 4.35 | 44.12 | +39.78 |
| r11_l1_single_fan_fixed_nominal | spatial_flow_belief_memory_h3 | 20 | 0.400 | 0.550 | +0.150 | 4.35 | 44.12 | +39.78 |
| r11_l1_single_fan_fixed_nominal | spatial_flow_belief_memory_h30 | 20 | 0.400 | 0.550 | +0.150 | 4.35 | 44.12 | +39.78 |
| r11_l2_four_fan_fixed_nominal | no_memory_baseline | 20 | 0.700 | 0.850 | +0.150 | 63.37 | 114.77 | +51.40 |
| r11_l2_four_fan_fixed_nominal | spatial_flow_belief_memory_h10 | 20 | 0.700 | 0.850 | +0.150 | 63.37 | 114.77 | +51.40 |
| r11_l2_four_fan_fixed_nominal | spatial_flow_belief_memory_h3 | 20 | 0.700 | 0.850 | +0.150 | 63.37 | 114.77 | +51.40 |
| r11_l2_four_fan_fixed_nominal | spatial_flow_belief_memory_h30 | 20 | 0.700 | 0.850 | +0.150 | 63.37 | 114.77 | +51.40 |
| r11_l3_fan_parameter_uncertainty | no_memory_baseline | 20 | 0.700 | 0.750 | +0.050 | 62.78 | 94.92 | +32.14 |
| r11_l3_fan_parameter_uncertainty | spatial_flow_belief_memory_h10 | 20 | 0.700 | 0.750 | +0.050 | 62.78 | 94.91 | +32.13 |
| r11_l3_fan_parameter_uncertainty | spatial_flow_belief_memory_h3 | 20 | 0.700 | 0.750 | +0.050 | 62.78 | 94.92 | +32.14 |
| r11_l3_fan_parameter_uncertainty | spatial_flow_belief_memory_h30 | 20 | 0.700 | 0.750 | +0.050 | 62.78 | 94.91 | +32.13 |
| r11_l4_local_fan_position_uncertainty | no_memory_baseline | 20 | 0.650 | 0.700 | +0.050 | 52.79 | 84.40 | +31.62 |
| r11_l4_local_fan_position_uncertainty | spatial_flow_belief_memory_h10 | 20 | 0.650 | 0.700 | +0.050 | 52.79 | 84.40 | +31.62 |
| r11_l4_local_fan_position_uncertainty | spatial_flow_belief_memory_h3 | 20 | 0.650 | 0.700 | +0.050 | 52.79 | 84.40 | +31.62 |
| r11_l4_local_fan_position_uncertainty | spatial_flow_belief_memory_h30 | 20 | 0.650 | 0.700 | +0.050 | 52.79 | 84.40 | +31.62 |
| r11_l5_active_fan_count_uncertainty | no_memory_baseline | 20 | 0.650 | 0.650 | +0.000 | 51.46 | 67.31 | +15.85 |
| r11_l5_active_fan_count_uncertainty | spatial_flow_belief_memory_h10 | 20 | 0.650 | 0.650 | +0.000 | 51.46 | 67.31 | +15.85 |
| r11_l5_active_fan_count_uncertainty | spatial_flow_belief_memory_h3 | 20 | 0.650 | 0.650 | +0.000 | 51.46 | 67.31 | +15.85 |
| r11_l5_active_fan_count_uncertainty | spatial_flow_belief_memory_h30 | 20 | 0.650 | 0.650 | +0.000 | 51.46 | 67.31 | +15.85 |
| r11_l6_environment_only_full_uncertainty | no_memory_baseline | 20 | 0.600 | 0.550 | -0.050 | 42.55 | 54.42 | +11.87 |
| r11_l6_environment_only_full_uncertainty | spatial_flow_belief_memory_h10 | 20 | 0.600 | 0.550 | -0.050 | 42.55 | 54.42 | +11.87 |
| r11_l6_environment_only_full_uncertainty | spatial_flow_belief_memory_h3 | 20 | 0.600 | 0.550 | -0.050 | 42.55 | 54.42 | +11.87 |
| r11_l6_environment_only_full_uncertainty | spatial_flow_belief_memory_h30 | 20 | 0.600 | 0.550 | -0.050 | 42.55 | 54.42 | +11.87 |
| r11_l7_full_domain_randomisation_arena_wide | no_memory_baseline | 20 | 0.500 | 0.450 | -0.050 | 22.22 | 26.78 | +4.56 |
| r11_l7_full_domain_randomisation_arena_wide | spatial_flow_belief_memory_h10 | 20 | 0.500 | 0.450 | -0.050 | 22.22 | 26.78 | +4.56 |
| r11_l7_full_domain_randomisation_arena_wide | spatial_flow_belief_memory_h3 | 20 | 0.500 | 0.450 | -0.050 | 22.22 | 26.78 | +4.56 |
| r11_l7_full_domain_randomisation_arena_wide | spatial_flow_belief_memory_h30 | 20 | 0.500 | 0.450 | -0.050 | 22.22 | 26.78 | +4.56 |

Figures:

- `r11_l0_dry_air_fixed`: `03_Control/A_figures/R11_E03_balanced_neutral_baseline/figures/r11_e03_1_bal_neutral_s00_l0.png`
- `r11_l1_single_fan_fixed_nominal`: `03_Control/A_figures/R11_E03_balanced_neutral_baseline/figures/r11_e03_1_bal_neutral_s00_l1.png`
- `r11_l2_four_fan_fixed_nominal`: `03_Control/A_figures/R11_E03_balanced_neutral_baseline/figures/r11_e03_1_bal_neutral_s00_l2.png`
- `r11_l3_fan_parameter_uncertainty`: `03_Control/A_figures/R11_E03_balanced_neutral_baseline/figures/r11_e03_1_bal_neutral_s00_l3.png`
- `r11_l4_local_fan_position_uncertainty`: `03_Control/A_figures/R11_E03_balanced_neutral_baseline/figures/r11_e03_1_bal_neutral_s00_l4.png`
- `r11_l5_active_fan_count_uncertainty`: `03_Control/A_figures/R11_E03_balanced_neutral_baseline/figures/r11_e03_1_bal_neutral_s00_l5.png`
- `r11_l6_environment_only_full_uncertainty`: `03_Control/A_figures/R11_E03_balanced_neutral_baseline/figures/r11_e03_1_bal_neutral_s00_l6.png`
- `r11_l7_full_domain_randomisation_arena_wide`: `03_Control/A_figures/R11_E03_balanced_neutral_baseline/figures/r11_e03_1_bal_neutral_s00_l7.png`
