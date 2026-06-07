# R11 E03_2 Balanced-Cluster Neutral-Intervention Diagnostic

- Diagnostic version: `r11_balanced_true_neutral_intervention_diagnostic_v1`
- R11 root: `03_Control/05_Results/R11_validation/E03.2`
- Library tier: `balanced_cluster`
- True neutral cases: `160`
- Figure count: `8`

`no_memory_baseline` is still closed-loop governor/LQR control. The neutral baseline here is a separate zero-command plant rollout from the same R11 final launch states.

The open-loop layer is run once per unique R11 final outer case. This is the correct comparison unit because all library tiers and memory policies reuse the same final launch state within that outer case.

| Ladder | Policy | n | Neutral mission | Controlled mission | Mission delta | Neutral score | Controlled score | Score delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| r11_l0_dry_air_fixed | no_memory_baseline | 20 | 0.350 | 0.500 | +0.150 | -10.20 | 32.85 | +43.05 |
| r11_l0_dry_air_fixed | spatial_flow_belief_memory_h10 | 20 | 0.350 | 0.500 | +0.150 | -10.20 | 34.85 | +45.05 |
| r11_l0_dry_air_fixed | spatial_flow_belief_memory_h3 | 20 | 0.350 | 0.500 | +0.150 | -10.20 | 34.85 | +45.05 |
| r11_l0_dry_air_fixed | spatial_flow_belief_memory_h30 | 20 | 0.350 | 0.500 | +0.150 | -10.20 | 34.85 | +45.05 |
| r11_l1_single_fan_fixed_nominal | no_memory_baseline | 20 | 0.500 | 0.600 | +0.100 | 20.85 | 65.46 | +44.60 |
| r11_l1_single_fan_fixed_nominal | spatial_flow_belief_memory_h10 | 20 | 0.500 | 0.600 | +0.100 | 20.85 | 65.46 | +44.60 |
| r11_l1_single_fan_fixed_nominal | spatial_flow_belief_memory_h3 | 20 | 0.500 | 0.600 | +0.100 | 20.85 | 65.46 | +44.60 |
| r11_l1_single_fan_fixed_nominal | spatial_flow_belief_memory_h30 | 20 | 0.500 | 0.600 | +0.100 | 20.85 | 65.46 | +44.60 |
| r11_l2_four_fan_fixed_nominal | no_memory_baseline | 20 | 0.750 | 0.850 | +0.100 | 70.22 | 117.90 | +47.69 |
| r11_l2_four_fan_fixed_nominal | spatial_flow_belief_memory_h10 | 20 | 0.750 | 0.850 | +0.100 | 70.22 | 117.91 | +47.69 |
| r11_l2_four_fan_fixed_nominal | spatial_flow_belief_memory_h3 | 20 | 0.750 | 0.850 | +0.100 | 70.22 | 117.91 | +47.69 |
| r11_l2_four_fan_fixed_nominal | spatial_flow_belief_memory_h30 | 20 | 0.750 | 0.850 | +0.100 | 70.22 | 117.91 | +47.69 |
| r11_l3_fan_parameter_uncertainty | no_memory_baseline | 20 | 0.600 | 0.650 | +0.050 | 41.51 | 82.19 | +40.67 |
| r11_l3_fan_parameter_uncertainty | spatial_flow_belief_memory_h10 | 20 | 0.600 | 0.650 | +0.050 | 41.51 | 82.19 | +40.67 |
| r11_l3_fan_parameter_uncertainty | spatial_flow_belief_memory_h3 | 20 | 0.600 | 0.650 | +0.050 | 41.51 | 82.19 | +40.67 |
| r11_l3_fan_parameter_uncertainty | spatial_flow_belief_memory_h30 | 20 | 0.600 | 0.650 | +0.050 | 41.51 | 82.19 | +40.67 |
| r11_l4_local_fan_position_uncertainty | no_memory_baseline | 20 | 0.600 | 0.700 | +0.100 | 40.12 | 85.18 | +45.06 |
| r11_l4_local_fan_position_uncertainty | spatial_flow_belief_memory_h10 | 20 | 0.600 | 0.700 | +0.100 | 40.12 | 85.18 | +45.06 |
| r11_l4_local_fan_position_uncertainty | spatial_flow_belief_memory_h3 | 20 | 0.600 | 0.700 | +0.100 | 40.12 | 85.18 | +45.06 |
| r11_l4_local_fan_position_uncertainty | spatial_flow_belief_memory_h30 | 20 | 0.600 | 0.700 | +0.100 | 40.12 | 88.04 | +47.92 |
| r11_l5_active_fan_count_uncertainty | no_memory_baseline | 20 | 0.550 | 0.650 | +0.100 | 30.19 | 74.62 | +44.43 |
| r11_l5_active_fan_count_uncertainty | spatial_flow_belief_memory_h10 | 20 | 0.550 | 0.650 | +0.100 | 30.19 | 76.62 | +46.43 |
| r11_l5_active_fan_count_uncertainty | spatial_flow_belief_memory_h3 | 20 | 0.550 | 0.650 | +0.100 | 30.19 | 76.62 | +46.43 |
| r11_l5_active_fan_count_uncertainty | spatial_flow_belief_memory_h30 | 20 | 0.550 | 0.650 | +0.100 | 30.19 | 76.62 | +46.43 |
| r11_l6_environment_only_full_uncertainty | no_memory_baseline | 20 | 0.500 | 0.600 | +0.100 | 19.31 | 66.19 | +46.88 |
| r11_l6_environment_only_full_uncertainty | spatial_flow_belief_memory_h10 | 20 | 0.500 | 0.600 | +0.100 | 19.31 | 64.19 | +44.88 |
| r11_l6_environment_only_full_uncertainty | spatial_flow_belief_memory_h3 | 20 | 0.500 | 0.600 | +0.100 | 19.31 | 64.19 | +44.88 |
| r11_l6_environment_only_full_uncertainty | spatial_flow_belief_memory_h30 | 20 | 0.500 | 0.600 | +0.100 | 19.31 | 66.19 | +46.88 |
| r11_l7_full_domain_randomisation_arena_wide | no_memory_baseline | 20 | 0.500 | 0.550 | +0.050 | 20.00 | 50.22 | +30.23 |
| r11_l7_full_domain_randomisation_arena_wide | spatial_flow_belief_memory_h10 | 20 | 0.500 | 0.550 | +0.050 | 20.00 | 50.24 | +30.25 |
| r11_l7_full_domain_randomisation_arena_wide | spatial_flow_belief_memory_h3 | 20 | 0.500 | 0.550 | +0.050 | 20.00 | 50.22 | +30.23 |
| r11_l7_full_domain_randomisation_arena_wide | spatial_flow_belief_memory_h30 | 20 | 0.500 | 0.550 | +0.050 | 20.00 | 50.24 | +30.25 |

Figures:

- `r11_l0_dry_air_fixed`: `03_Control/A_figures/R11_E03.2_balanced_neutral_baseline/figures/r11_e03_2_bal_neutral_s18_l0.png`
- `r11_l1_single_fan_fixed_nominal`: `03_Control/A_figures/R11_E03.2_balanced_neutral_baseline/figures/r11_e03_2_bal_neutral_s18_l1.png`
- `r11_l2_four_fan_fixed_nominal`: `03_Control/A_figures/R11_E03.2_balanced_neutral_baseline/figures/r11_e03_2_bal_neutral_s18_l2.png`
- `r11_l3_fan_parameter_uncertainty`: `03_Control/A_figures/R11_E03.2_balanced_neutral_baseline/figures/r11_e03_2_bal_neutral_s18_l3.png`
- `r11_l4_local_fan_position_uncertainty`: `03_Control/A_figures/R11_E03.2_balanced_neutral_baseline/figures/r11_e03_2_bal_neutral_s18_l4.png`
- `r11_l5_active_fan_count_uncertainty`: `03_Control/A_figures/R11_E03.2_balanced_neutral_baseline/figures/r11_e03_2_bal_neutral_s18_l5.png`
- `r11_l6_environment_only_full_uncertainty`: `03_Control/A_figures/R11_E03.2_balanced_neutral_baseline/figures/r11_e03_2_bal_neutral_s18_l6.png`
- `r11_l7_full_domain_randomisation_arena_wide`: `03_Control/A_figures/R11_E03.2_balanced_neutral_baseline/figures/r11_e03_2_bal_neutral_s18_l7.png`
