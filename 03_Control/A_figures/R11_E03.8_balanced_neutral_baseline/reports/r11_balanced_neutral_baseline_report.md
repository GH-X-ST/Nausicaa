# R11 E03_8 Balanced-Cluster Neutral-Intervention Diagnostic

- Diagnostic version: `r11_balanced_true_neutral_intervention_diagnostic_v1`
- R11 root: `03_Control/05_Results/R11_validation/E03.8`
- Library tier: `balanced_cluster`
- True neutral cases: `160`
- Figure count: `8`

`no_memory_baseline` is still closed-loop governor/LQR control. The neutral baseline here is a separate zero-command plant rollout from the same R11 final launch states.

The open-loop layer is run once per unique R11 final outer case. This is the correct comparison unit because all library tiers and memory policies reuse the same final launch state within that outer case.

| Ladder | Policy | n | Neutral mission | Controlled mission | Mission delta | Neutral score | Controlled score | Score delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| r11_l0_dry_air_fixed | no_memory_baseline | 20 | 0.500 | 0.500 | +0.000 | 20.64 | 32.84 | +12.20 |
| r11_l0_dry_air_fixed | spatial_flow_belief_memory_h10 | 20 | 0.500 | 0.500 | +0.000 | 20.64 | 32.84 | +12.20 |
| r11_l0_dry_air_fixed | spatial_flow_belief_memory_h3 | 20 | 0.500 | 0.500 | +0.000 | 20.64 | 32.84 | +12.20 |
| r11_l0_dry_air_fixed | spatial_flow_belief_memory_h30 | 20 | 0.500 | 0.500 | +0.000 | 20.64 | 32.84 | +12.20 |
| r11_l1_single_fan_fixed_nominal | no_memory_baseline | 20 | 0.600 | 0.650 | +0.050 | 40.38 | 64.27 | +23.89 |
| r11_l1_single_fan_fixed_nominal | spatial_flow_belief_memory_h10 | 20 | 0.600 | 0.650 | +0.050 | 40.38 | 64.26 | +23.88 |
| r11_l1_single_fan_fixed_nominal | spatial_flow_belief_memory_h3 | 20 | 0.600 | 0.650 | +0.050 | 40.38 | 64.26 | +23.88 |
| r11_l1_single_fan_fixed_nominal | spatial_flow_belief_memory_h30 | 20 | 0.600 | 0.650 | +0.050 | 40.38 | 64.26 | +23.88 |
| r11_l2_four_fan_fixed_nominal | no_memory_baseline | 20 | 0.800 | 0.900 | +0.100 | 80.00 | 126.33 | +46.32 |
| r11_l2_four_fan_fixed_nominal | spatial_flow_belief_memory_h10 | 20 | 0.800 | 0.900 | +0.100 | 80.00 | 126.33 | +46.32 |
| r11_l2_four_fan_fixed_nominal | spatial_flow_belief_memory_h3 | 20 | 0.800 | 0.900 | +0.100 | 80.00 | 126.31 | +46.31 |
| r11_l2_four_fan_fixed_nominal | spatial_flow_belief_memory_h30 | 20 | 0.800 | 0.900 | +0.100 | 80.00 | 126.32 | +46.32 |
| r11_l3_fan_parameter_uncertainty | no_memory_baseline | 20 | 0.550 | 0.650 | +0.100 | 32.54 | 75.36 | +42.83 |
| r11_l3_fan_parameter_uncertainty | spatial_flow_belief_memory_h10 | 20 | 0.550 | 0.650 | +0.100 | 32.54 | 75.37 | +42.83 |
| r11_l3_fan_parameter_uncertainty | spatial_flow_belief_memory_h3 | 20 | 0.550 | 0.650 | +0.100 | 32.54 | 75.37 | +42.83 |
| r11_l3_fan_parameter_uncertainty | spatial_flow_belief_memory_h30 | 20 | 0.550 | 0.650 | +0.100 | 32.54 | 75.37 | +42.83 |
| r11_l4_local_fan_position_uncertainty | no_memory_baseline | 20 | 0.550 | 0.750 | +0.200 | 30.56 | 90.20 | +59.64 |
| r11_l4_local_fan_position_uncertainty | spatial_flow_belief_memory_h10 | 20 | 0.550 | 0.750 | +0.200 | 30.56 | 90.20 | +59.64 |
| r11_l4_local_fan_position_uncertainty | spatial_flow_belief_memory_h3 | 20 | 0.550 | 0.750 | +0.200 | 30.56 | 90.20 | +59.64 |
| r11_l4_local_fan_position_uncertainty | spatial_flow_belief_memory_h30 | 20 | 0.550 | 0.750 | +0.200 | 30.56 | 90.20 | +59.64 |
| r11_l5_active_fan_count_uncertainty | no_memory_baseline | 20 | 0.550 | 0.550 | +0.000 | 31.18 | 44.13 | +12.96 |
| r11_l5_active_fan_count_uncertainty | spatial_flow_belief_memory_h10 | 20 | 0.550 | 0.550 | +0.000 | 31.18 | 44.13 | +12.95 |
| r11_l5_active_fan_count_uncertainty | spatial_flow_belief_memory_h3 | 20 | 0.550 | 0.550 | +0.000 | 31.18 | 44.13 | +12.95 |
| r11_l5_active_fan_count_uncertainty | spatial_flow_belief_memory_h30 | 20 | 0.550 | 0.550 | +0.000 | 31.18 | 44.13 | +12.95 |
| r11_l6_environment_only_full_uncertainty | no_memory_baseline | 20 | 0.500 | 0.550 | +0.050 | 20.83 | 51.22 | +30.38 |
| r11_l6_environment_only_full_uncertainty | spatial_flow_belief_memory_h10 | 20 | 0.500 | 0.550 | +0.050 | 20.83 | 51.22 | +30.38 |
| r11_l6_environment_only_full_uncertainty | spatial_flow_belief_memory_h3 | 20 | 0.500 | 0.550 | +0.050 | 20.83 | 51.22 | +30.38 |
| r11_l6_environment_only_full_uncertainty | spatial_flow_belief_memory_h30 | 20 | 0.500 | 0.550 | +0.050 | 20.83 | 51.22 | +30.38 |
| r11_l7_full_domain_randomisation_arena_wide | no_memory_baseline | 20 | 0.450 | 0.400 | -0.050 | 10.03 | 16.17 | +6.14 |
| r11_l7_full_domain_randomisation_arena_wide | spatial_flow_belief_memory_h10 | 20 | 0.450 | 0.400 | -0.050 | 10.03 | 16.17 | +6.14 |
| r11_l7_full_domain_randomisation_arena_wide | spatial_flow_belief_memory_h3 | 20 | 0.450 | 0.400 | -0.050 | 10.03 | 16.17 | +6.14 |
| r11_l7_full_domain_randomisation_arena_wide | spatial_flow_belief_memory_h30 | 20 | 0.450 | 0.400 | -0.050 | 10.03 | 16.17 | +6.14 |

Figures:

- `r11_l0_dry_air_fixed`: `03_Control/A_figures/R11_E03.8_balanced_neutral_baseline/figures/r11_e03_8_bal_neutral_s31_l0.png`
- `r11_l1_single_fan_fixed_nominal`: `03_Control/A_figures/R11_E03.8_balanced_neutral_baseline/figures/r11_e03_8_bal_neutral_s31_l1.png`
- `r11_l2_four_fan_fixed_nominal`: `03_Control/A_figures/R11_E03.8_balanced_neutral_baseline/figures/r11_e03_8_bal_neutral_s31_l2.png`
- `r11_l3_fan_parameter_uncertainty`: `03_Control/A_figures/R11_E03.8_balanced_neutral_baseline/figures/r11_e03_8_bal_neutral_s31_l3.png`
- `r11_l4_local_fan_position_uncertainty`: `03_Control/A_figures/R11_E03.8_balanced_neutral_baseline/figures/r11_e03_8_bal_neutral_s31_l4.png`
- `r11_l5_active_fan_count_uncertainty`: `03_Control/A_figures/R11_E03.8_balanced_neutral_baseline/figures/r11_e03_8_bal_neutral_s31_l5.png`
- `r11_l6_environment_only_full_uncertainty`: `03_Control/A_figures/R11_E03.8_balanced_neutral_baseline/figures/r11_e03_8_bal_neutral_s31_l6.png`
- `r11_l7_full_domain_randomisation_arena_wide`: `03_Control/A_figures/R11_E03.8_balanced_neutral_baseline/figures/r11_e03_8_bal_neutral_s31_l7.png`
