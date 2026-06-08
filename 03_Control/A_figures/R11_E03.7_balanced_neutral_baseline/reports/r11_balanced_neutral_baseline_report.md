# R11 E03_7 Balanced-Cluster Neutral-Intervention Diagnostic

- Diagnostic version: `r11_balanced_true_neutral_intervention_diagnostic_v1`
- R11 root: `03_Control/05_Results/R11_validation/E03.7`
- Library tier: `balanced_cluster`
- True neutral cases: `160`
- Figure count: `8`

`no_memory_baseline` is still closed-loop governor/LQR control. The neutral baseline here is a separate zero-command plant rollout from the same R11 final launch states.

The open-loop layer is run once per unique R11 final outer case. This is the correct comparison unit because all library tiers and memory policies reuse the same final launch state within that outer case.

| Ladder | Policy | n | Neutral mission | Controlled mission | Mission delta | Neutral score | Controlled score | Score delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| r11_l0_dry_air_fixed | no_memory_baseline | 20 | 0.550 | 0.600 | +0.050 | 31.61 | 57.08 | +25.47 |
| r11_l0_dry_air_fixed | spatial_flow_belief_memory_h10 | 20 | 0.550 | 0.600 | +0.050 | 31.61 | 57.08 | +25.47 |
| r11_l0_dry_air_fixed | spatial_flow_belief_memory_h3 | 20 | 0.550 | 0.600 | +0.050 | 31.61 | 57.08 | +25.47 |
| r11_l0_dry_air_fixed | spatial_flow_belief_memory_h30 | 20 | 0.550 | 0.600 | +0.050 | 31.61 | 57.08 | +25.47 |
| r11_l1_single_fan_fixed_nominal | no_memory_baseline | 20 | 0.650 | 0.650 | +0.000 | 53.55 | 70.78 | +17.23 |
| r11_l1_single_fan_fixed_nominal | spatial_flow_belief_memory_h10 | 20 | 0.650 | 0.650 | +0.000 | 53.55 | 70.78 | +17.23 |
| r11_l1_single_fan_fixed_nominal | spatial_flow_belief_memory_h3 | 20 | 0.650 | 0.650 | +0.000 | 53.55 | 70.78 | +17.23 |
| r11_l1_single_fan_fixed_nominal | spatial_flow_belief_memory_h30 | 20 | 0.650 | 0.650 | +0.000 | 53.55 | 70.78 | +17.23 |
| r11_l2_four_fan_fixed_nominal | no_memory_baseline | 20 | 0.750 | 0.800 | +0.050 | 73.19 | 105.03 | +31.84 |
| r11_l2_four_fan_fixed_nominal | spatial_flow_belief_memory_h10 | 20 | 0.750 | 0.800 | +0.050 | 73.19 | 105.04 | +31.85 |
| r11_l2_four_fan_fixed_nominal | spatial_flow_belief_memory_h3 | 20 | 0.750 | 0.800 | +0.050 | 73.19 | 105.03 | +31.84 |
| r11_l2_four_fan_fixed_nominal | spatial_flow_belief_memory_h30 | 20 | 0.750 | 0.800 | +0.050 | 73.19 | 105.01 | +31.82 |
| r11_l3_fan_parameter_uncertainty | no_memory_baseline | 20 | 0.600 | 0.750 | +0.150 | 45.81 | 90.67 | +44.86 |
| r11_l3_fan_parameter_uncertainty | spatial_flow_belief_memory_h10 | 20 | 0.600 | 0.750 | +0.150 | 45.81 | 90.67 | +44.86 |
| r11_l3_fan_parameter_uncertainty | spatial_flow_belief_memory_h3 | 20 | 0.600 | 0.750 | +0.150 | 45.81 | 90.67 | +44.86 |
| r11_l3_fan_parameter_uncertainty | spatial_flow_belief_memory_h30 | 20 | 0.600 | 0.750 | +0.150 | 45.81 | 90.67 | +44.86 |
| r11_l4_local_fan_position_uncertainty | no_memory_baseline | 20 | 0.550 | 0.750 | +0.200 | 37.52 | 94.52 | +57.00 |
| r11_l4_local_fan_position_uncertainty | spatial_flow_belief_memory_h10 | 20 | 0.550 | 0.750 | +0.200 | 37.52 | 96.51 | +58.98 |
| r11_l4_local_fan_position_uncertainty | spatial_flow_belief_memory_h3 | 20 | 0.550 | 0.750 | +0.200 | 37.52 | 94.51 | +56.98 |
| r11_l4_local_fan_position_uncertainty | spatial_flow_belief_memory_h30 | 20 | 0.550 | 0.750 | +0.200 | 37.52 | 94.55 | +57.03 |
| r11_l5_active_fan_count_uncertainty | no_memory_baseline | 20 | 0.550 | 0.700 | +0.150 | 34.34 | 78.22 | +43.88 |
| r11_l5_active_fan_count_uncertainty | spatial_flow_belief_memory_h10 | 20 | 0.550 | 0.700 | +0.150 | 34.34 | 78.29 | +43.94 |
| r11_l5_active_fan_count_uncertainty | spatial_flow_belief_memory_h3 | 20 | 0.550 | 0.700 | +0.150 | 34.34 | 78.23 | +43.89 |
| r11_l5_active_fan_count_uncertainty | spatial_flow_belief_memory_h30 | 20 | 0.550 | 0.700 | +0.150 | 34.34 | 78.23 | +43.89 |
| r11_l6_environment_only_full_uncertainty | no_memory_baseline | 20 | 0.600 | 0.700 | +0.100 | 42.73 | 78.21 | +35.48 |
| r11_l6_environment_only_full_uncertainty | spatial_flow_belief_memory_h10 | 20 | 0.600 | 0.700 | +0.100 | 42.73 | 78.23 | +35.50 |
| r11_l6_environment_only_full_uncertainty | spatial_flow_belief_memory_h3 | 20 | 0.600 | 0.700 | +0.100 | 42.73 | 78.23 | +35.50 |
| r11_l6_environment_only_full_uncertainty | spatial_flow_belief_memory_h30 | 20 | 0.600 | 0.700 | +0.100 | 42.73 | 78.21 | +35.48 |
| r11_l7_full_domain_randomisation_arena_wide | no_memory_baseline | 20 | 0.550 | 0.650 | +0.100 | 33.09 | 63.75 | +30.66 |
| r11_l7_full_domain_randomisation_arena_wide | spatial_flow_belief_memory_h10 | 20 | 0.550 | 0.650 | +0.100 | 33.09 | 63.75 | +30.66 |
| r11_l7_full_domain_randomisation_arena_wide | spatial_flow_belief_memory_h3 | 20 | 0.550 | 0.650 | +0.100 | 33.09 | 63.75 | +30.66 |
| r11_l7_full_domain_randomisation_arena_wide | spatial_flow_belief_memory_h30 | 20 | 0.550 | 0.650 | +0.100 | 33.09 | 63.75 | +30.66 |

Figures:

- `r11_l0_dry_air_fixed`: `03_Control/A_figures/R11_E03.7_balanced_neutral_baseline/figures/r11_e03_7_bal_neutral_s23_l0.png`
- `r11_l1_single_fan_fixed_nominal`: `03_Control/A_figures/R11_E03.7_balanced_neutral_baseline/figures/r11_e03_7_bal_neutral_s23_l1.png`
- `r11_l2_four_fan_fixed_nominal`: `03_Control/A_figures/R11_E03.7_balanced_neutral_baseline/figures/r11_e03_7_bal_neutral_s23_l2.png`
- `r11_l3_fan_parameter_uncertainty`: `03_Control/A_figures/R11_E03.7_balanced_neutral_baseline/figures/r11_e03_7_bal_neutral_s23_l3.png`
- `r11_l4_local_fan_position_uncertainty`: `03_Control/A_figures/R11_E03.7_balanced_neutral_baseline/figures/r11_e03_7_bal_neutral_s23_l4.png`
- `r11_l5_active_fan_count_uncertainty`: `03_Control/A_figures/R11_E03.7_balanced_neutral_baseline/figures/r11_e03_7_bal_neutral_s23_l5.png`
- `r11_l6_environment_only_full_uncertainty`: `03_Control/A_figures/R11_E03.7_balanced_neutral_baseline/figures/r11_e03_7_bal_neutral_s23_l6.png`
- `r11_l7_full_domain_randomisation_arena_wide`: `03_Control/A_figures/R11_E03.7_balanced_neutral_baseline/figures/r11_e03_7_bal_neutral_s23_l7.png`
