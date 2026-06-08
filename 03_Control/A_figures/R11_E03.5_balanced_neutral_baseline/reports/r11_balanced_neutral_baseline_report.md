# R11 E03_5 Balanced-Cluster Neutral-Intervention Diagnostic

- Diagnostic version: `r11_balanced_true_neutral_intervention_diagnostic_v1`
- R11 root: `03_Control/05_Results/R11_validation/E03.5`
- Library tier: `balanced_cluster`
- True neutral cases: `160`
- Figure count: `8`

`no_memory_baseline` is still closed-loop governor/LQR control. The neutral baseline here is a separate zero-command plant rollout from the same R11 final launch states.

The open-loop layer is run once per unique R11 final outer case. This is the correct comparison unit because all library tiers and memory policies reuse the same final launch state within that outer case.

| Ladder | Policy | n | Neutral mission | Controlled mission | Mission delta | Neutral score | Controlled score | Score delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| r11_l0_dry_air_fixed | no_memory_baseline | 20 | 0.500 | 0.600 | +0.100 | 23.95 | 61.08 | +37.13 |
| r11_l0_dry_air_fixed | spatial_flow_belief_memory_h10 | 20 | 0.500 | 0.600 | +0.100 | 23.95 | 61.08 | +37.13 |
| r11_l0_dry_air_fixed | spatial_flow_belief_memory_h3 | 20 | 0.500 | 0.600 | +0.100 | 23.95 | 61.08 | +37.13 |
| r11_l0_dry_air_fixed | spatial_flow_belief_memory_h30 | 20 | 0.500 | 0.600 | +0.100 | 23.95 | 61.08 | +37.13 |
| r11_l1_single_fan_fixed_nominal | no_memory_baseline | 20 | 0.500 | 0.700 | +0.200 | 25.72 | 84.39 | +58.67 |
| r11_l1_single_fan_fixed_nominal | spatial_flow_belief_memory_h10 | 20 | 0.500 | 0.700 | +0.200 | 25.72 | 84.39 | +58.67 |
| r11_l1_single_fan_fixed_nominal | spatial_flow_belief_memory_h3 | 20 | 0.500 | 0.700 | +0.200 | 25.72 | 84.39 | +58.67 |
| r11_l1_single_fan_fixed_nominal | spatial_flow_belief_memory_h30 | 20 | 0.500 | 0.700 | +0.200 | 25.72 | 84.39 | +58.67 |
| r11_l2_four_fan_fixed_nominal | no_memory_baseline | 20 | 0.650 | 0.850 | +0.200 | 55.98 | 112.49 | +56.50 |
| r11_l2_four_fan_fixed_nominal | spatial_flow_belief_memory_h10 | 20 | 0.650 | 0.850 | +0.200 | 55.98 | 112.49 | +56.50 |
| r11_l2_four_fan_fixed_nominal | spatial_flow_belief_memory_h3 | 20 | 0.650 | 0.850 | +0.200 | 55.98 | 112.49 | +56.50 |
| r11_l2_four_fan_fixed_nominal | spatial_flow_belief_memory_h30 | 20 | 0.650 | 0.850 | +0.200 | 55.98 | 112.49 | +56.51 |
| r11_l3_fan_parameter_uncertainty | no_memory_baseline | 20 | 0.500 | 0.700 | +0.200 | 26.65 | 86.46 | +59.81 |
| r11_l3_fan_parameter_uncertainty | spatial_flow_belief_memory_h10 | 20 | 0.500 | 0.700 | +0.200 | 26.65 | 86.44 | +59.79 |
| r11_l3_fan_parameter_uncertainty | spatial_flow_belief_memory_h3 | 20 | 0.500 | 0.700 | +0.200 | 26.65 | 86.47 | +59.81 |
| r11_l3_fan_parameter_uncertainty | spatial_flow_belief_memory_h30 | 20 | 0.500 | 0.700 | +0.200 | 26.65 | 86.46 | +59.81 |
| r11_l4_local_fan_position_uncertainty | no_memory_baseline | 20 | 0.550 | 0.650 | +0.100 | 33.59 | 73.22 | +39.64 |
| r11_l4_local_fan_position_uncertainty | spatial_flow_belief_memory_h10 | 20 | 0.550 | 0.650 | +0.100 | 33.59 | 73.22 | +39.64 |
| r11_l4_local_fan_position_uncertainty | spatial_flow_belief_memory_h3 | 20 | 0.550 | 0.650 | +0.100 | 33.59 | 73.22 | +39.64 |
| r11_l4_local_fan_position_uncertainty | spatial_flow_belief_memory_h30 | 20 | 0.550 | 0.650 | +0.100 | 33.59 | 73.22 | +39.64 |
| r11_l5_active_fan_count_uncertainty | no_memory_baseline | 20 | 0.400 | 0.600 | +0.200 | 7.27 | 64.11 | +56.84 |
| r11_l5_active_fan_count_uncertainty | spatial_flow_belief_memory_h10 | 20 | 0.400 | 0.600 | +0.200 | 7.27 | 64.11 | +56.84 |
| r11_l5_active_fan_count_uncertainty | spatial_flow_belief_memory_h3 | 20 | 0.400 | 0.600 | +0.200 | 7.27 | 64.11 | +56.84 |
| r11_l5_active_fan_count_uncertainty | spatial_flow_belief_memory_h30 | 20 | 0.400 | 0.600 | +0.200 | 7.27 | 64.11 | +56.84 |
| r11_l6_environment_only_full_uncertainty | no_memory_baseline | 20 | 0.500 | 0.600 | +0.100 | 23.63 | 64.58 | +40.95 |
| r11_l6_environment_only_full_uncertainty | spatial_flow_belief_memory_h10 | 20 | 0.500 | 0.600 | +0.100 | 23.63 | 64.58 | +40.95 |
| r11_l6_environment_only_full_uncertainty | spatial_flow_belief_memory_h3 | 20 | 0.500 | 0.600 | +0.100 | 23.63 | 64.58 | +40.95 |
| r11_l6_environment_only_full_uncertainty | spatial_flow_belief_memory_h30 | 20 | 0.500 | 0.600 | +0.100 | 23.63 | 64.58 | +40.95 |
| r11_l7_full_domain_randomisation_arena_wide | no_memory_baseline | 20 | 0.450 | 0.500 | +0.050 | 15.83 | 38.12 | +22.29 |
| r11_l7_full_domain_randomisation_arena_wide | spatial_flow_belief_memory_h10 | 20 | 0.450 | 0.500 | +0.050 | 15.83 | 38.12 | +22.29 |
| r11_l7_full_domain_randomisation_arena_wide | spatial_flow_belief_memory_h3 | 20 | 0.450 | 0.500 | +0.050 | 15.83 | 38.12 | +22.29 |
| r11_l7_full_domain_randomisation_arena_wide | spatial_flow_belief_memory_h30 | 20 | 0.450 | 0.500 | +0.050 | 15.83 | 38.12 | +22.29 |

Figures:

- `r11_l0_dry_air_fixed`: `03_Control/A_figures/R11_E03.5_balanced_neutral_baseline/figures/r11_e03_5_bal_neutral_s46_l0.png`
- `r11_l1_single_fan_fixed_nominal`: `03_Control/A_figures/R11_E03.5_balanced_neutral_baseline/figures/r11_e03_5_bal_neutral_s46_l1.png`
- `r11_l2_four_fan_fixed_nominal`: `03_Control/A_figures/R11_E03.5_balanced_neutral_baseline/figures/r11_e03_5_bal_neutral_s46_l2.png`
- `r11_l3_fan_parameter_uncertainty`: `03_Control/A_figures/R11_E03.5_balanced_neutral_baseline/figures/r11_e03_5_bal_neutral_s46_l3.png`
- `r11_l4_local_fan_position_uncertainty`: `03_Control/A_figures/R11_E03.5_balanced_neutral_baseline/figures/r11_e03_5_bal_neutral_s46_l4.png`
- `r11_l5_active_fan_count_uncertainty`: `03_Control/A_figures/R11_E03.5_balanced_neutral_baseline/figures/r11_e03_5_bal_neutral_s46_l5.png`
- `r11_l6_environment_only_full_uncertainty`: `03_Control/A_figures/R11_E03.5_balanced_neutral_baseline/figures/r11_e03_5_bal_neutral_s46_l6.png`
- `r11_l7_full_domain_randomisation_arena_wide`: `03_Control/A_figures/R11_E03.5_balanced_neutral_baseline/figures/r11_e03_5_bal_neutral_s46_l7.png`
