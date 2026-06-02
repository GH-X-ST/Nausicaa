# R11 E01 Balanced-Cluster Neutral-Intervention Diagnostic

- Diagnostic version: `r11_balanced_true_neutral_intervention_diagnostic_v1`
- R11 root: `03_Control/05_Results/R11_validation/D01`
- Library tier: `balanced_cluster`
- True neutral cases: `400`
- Figure count: `8`

`no_memory_baseline` is still closed-loop governor/LQR control. The neutral baseline here is a separate zero-command plant rollout from the same R11 final launch states.

The open-loop layer is run once per unique R11 final outer case. This is the correct comparison unit because all library tiers and memory policies reuse the same final launch state within that outer case.

| Ladder | Policy | n | Neutral mission | Controlled mission | Mission delta | Neutral score | Controlled score | Score delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| r11_l0_dry_air_fixed | no_memory_baseline | 50 | 0.680 | 0.740 | +0.060 | 57.35 | 84.82 | +27.47 |
| r11_l0_dry_air_fixed | spatial_flow_belief_memory_h10 | 50 | 0.680 | 0.740 | +0.060 | 57.35 | 84.82 | +27.47 |
| r11_l0_dry_air_fixed | spatial_flow_belief_memory_h3 | 50 | 0.680 | 0.740 | +0.060 | 57.35 | 84.82 | +27.47 |
| r11_l0_dry_air_fixed | spatial_flow_belief_memory_h30 | 50 | 0.680 | 0.740 | +0.060 | 57.35 | 84.82 | +27.47 |
| r11_l1_single_fan_fixed_nominal | no_memory_baseline | 50 | 0.460 | 0.700 | +0.240 | 16.74 | 86.01 | +69.27 |
| r11_l1_single_fan_fixed_nominal | spatial_flow_belief_memory_h10 | 50 | 0.460 | 0.700 | +0.240 | 16.74 | 86.72 | +69.99 |
| r11_l1_single_fan_fixed_nominal | spatial_flow_belief_memory_h3 | 50 | 0.460 | 0.700 | +0.240 | 16.74 | 85.99 | +69.26 |
| r11_l1_single_fan_fixed_nominal | spatial_flow_belief_memory_h30 | 50 | 0.460 | 0.700 | +0.240 | 16.74 | 86.80 | +70.07 |
| r11_l2_four_fan_fixed_nominal | no_memory_baseline | 50 | 0.320 | 0.820 | +0.500 | -9.05 | 109.41 | +118.46 |
| r11_l2_four_fan_fixed_nominal | spatial_flow_belief_memory_h10 | 50 | 0.320 | 0.820 | +0.500 | -9.05 | 109.43 | +118.48 |
| r11_l2_four_fan_fixed_nominal | spatial_flow_belief_memory_h3 | 50 | 0.320 | 0.820 | +0.500 | -9.05 | 109.41 | +118.46 |
| r11_l2_four_fan_fixed_nominal | spatial_flow_belief_memory_h30 | 50 | 0.320 | 0.840 | +0.520 | -9.05 | 113.24 | +122.29 |
| r11_l3_fan_parameter_uncertainty | no_memory_baseline | 50 | 0.440 | 0.760 | +0.320 | 11.37 | 98.31 | +86.94 |
| r11_l3_fan_parameter_uncertainty | spatial_flow_belief_memory_h10 | 50 | 0.440 | 0.760 | +0.320 | 11.37 | 99.05 | +87.68 |
| r11_l3_fan_parameter_uncertainty | spatial_flow_belief_memory_h3 | 50 | 0.440 | 0.760 | +0.320 | 11.37 | 98.32 | +86.95 |
| r11_l3_fan_parameter_uncertainty | spatial_flow_belief_memory_h30 | 50 | 0.440 | 0.740 | +0.300 | 11.37 | 94.34 | +82.97 |
| r11_l4_local_fan_position_uncertainty | no_memory_baseline | 50 | 0.440 | 0.660 | +0.220 | 11.28 | 81.34 | +70.06 |
| r11_l4_local_fan_position_uncertainty | spatial_flow_belief_memory_h10 | 50 | 0.440 | 0.660 | +0.220 | 11.28 | 82.18 | +70.91 |
| r11_l4_local_fan_position_uncertainty | spatial_flow_belief_memory_h3 | 50 | 0.440 | 0.660 | +0.220 | 11.28 | 81.35 | +70.07 |
| r11_l4_local_fan_position_uncertainty | spatial_flow_belief_memory_h30 | 50 | 0.440 | 0.680 | +0.240 | 11.28 | 86.03 | +74.76 |
| r11_l5_active_fan_count_uncertainty | no_memory_baseline | 50 | 0.620 | 0.720 | +0.100 | 44.86 | 85.51 | +40.66 |
| r11_l5_active_fan_count_uncertainty | spatial_flow_belief_memory_h10 | 50 | 0.620 | 0.720 | +0.100 | 44.86 | 85.42 | +40.57 |
| r11_l5_active_fan_count_uncertainty | spatial_flow_belief_memory_h3 | 50 | 0.620 | 0.720 | +0.100 | 44.86 | 85.50 | +40.64 |
| r11_l5_active_fan_count_uncertainty | spatial_flow_belief_memory_h30 | 50 | 0.620 | 0.720 | +0.100 | 44.86 | 85.51 | +40.66 |
| r11_l6_environment_only_full_uncertainty | no_memory_baseline | 50 | 0.560 | 0.720 | +0.160 | 35.46 | 89.08 | +53.63 |
| r11_l6_environment_only_full_uncertainty | spatial_flow_belief_memory_h10 | 50 | 0.560 | 0.720 | +0.160 | 35.46 | 89.08 | +53.63 |
| r11_l6_environment_only_full_uncertainty | spatial_flow_belief_memory_h3 | 50 | 0.560 | 0.720 | +0.160 | 35.46 | 88.29 | +52.84 |
| r11_l6_environment_only_full_uncertainty | spatial_flow_belief_memory_h30 | 50 | 0.560 | 0.740 | +0.180 | 35.46 | 92.31 | +56.85 |
| r11_l7_full_domain_randomisation_arena_wide | no_memory_baseline | 50 | 0.600 | 0.660 | +0.060 | 40.71 | 74.28 | +33.57 |
| r11_l7_full_domain_randomisation_arena_wide | spatial_flow_belief_memory_h10 | 50 | 0.600 | 0.660 | +0.060 | 40.71 | 74.29 | +33.58 |
| r11_l7_full_domain_randomisation_arena_wide | spatial_flow_belief_memory_h3 | 50 | 0.600 | 0.660 | +0.060 | 40.71 | 74.28 | +33.57 |
| r11_l7_full_domain_randomisation_arena_wide | spatial_flow_belief_memory_h30 | 50 | 0.600 | 0.680 | +0.080 | 40.71 | 79.14 | +38.43 |

Figures:

- `r11_l0_dry_air_fixed`: `03_Control/A_figures/R11_D01_balanced_neutral_baseline/figures/r11_e01_bal_neutral_s01_l0.png`
- `r11_l1_single_fan_fixed_nominal`: `03_Control/A_figures/R11_D01_balanced_neutral_baseline/figures/r11_e01_bal_neutral_s01_l1.png`
- `r11_l2_four_fan_fixed_nominal`: `03_Control/A_figures/R11_D01_balanced_neutral_baseline/figures/r11_e01_bal_neutral_s01_l2.png`
- `r11_l3_fan_parameter_uncertainty`: `03_Control/A_figures/R11_D01_balanced_neutral_baseline/figures/r11_e01_bal_neutral_s01_l3.png`
- `r11_l4_local_fan_position_uncertainty`: `03_Control/A_figures/R11_D01_balanced_neutral_baseline/figures/r11_e01_bal_neutral_s01_l4.png`
- `r11_l5_active_fan_count_uncertainty`: `03_Control/A_figures/R11_D01_balanced_neutral_baseline/figures/r11_e01_bal_neutral_s01_l5.png`
- `r11_l6_environment_only_full_uncertainty`: `03_Control/A_figures/R11_D01_balanced_neutral_baseline/figures/r11_e01_bal_neutral_s01_l6.png`
- `r11_l7_full_domain_randomisation_arena_wide`: `03_Control/A_figures/R11_D01_balanced_neutral_baseline/figures/r11_e01_bal_neutral_s01_l7.png`
