# R11 E03_6 Balanced-Cluster Neutral-Intervention Diagnostic

- Diagnostic version: `r11_balanced_true_neutral_intervention_diagnostic_v1`
- R11 root: `03_Control/05_Results/R11_validation/E03.6`
- Library tier: `balanced_cluster`
- True neutral cases: `160`
- Figure count: `8`

`no_memory_baseline` is still closed-loop governor/LQR control. The neutral baseline here is a separate zero-command plant rollout from the same R11 final launch states.

The open-loop layer is run once per unique R11 final outer case. This is the correct comparison unit because all library tiers and memory policies reuse the same final launch state within that outer case.

| Ladder | Policy | n | Neutral mission | Controlled mission | Mission delta | Neutral score | Controlled score | Score delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| r11_l0_dry_air_fixed | no_memory_baseline | 20 | 0.300 | 0.350 | +0.050 | -12.72 | 9.92 | +22.63 |
| r11_l0_dry_air_fixed | spatial_flow_belief_memory_h10 | 20 | 0.300 | 0.350 | +0.050 | -12.72 | 9.92 | +22.63 |
| r11_l0_dry_air_fixed | spatial_flow_belief_memory_h3 | 20 | 0.300 | 0.350 | +0.050 | -12.72 | 9.92 | +22.63 |
| r11_l0_dry_air_fixed | spatial_flow_belief_memory_h30 | 20 | 0.300 | 0.350 | +0.050 | -12.72 | 9.92 | +22.63 |
| r11_l1_single_fan_fixed_nominal | no_memory_baseline | 20 | 0.400 | 0.450 | +0.050 | 8.61 | 35.98 | +27.37 |
| r11_l1_single_fan_fixed_nominal | spatial_flow_belief_memory_h10 | 20 | 0.400 | 0.450 | +0.050 | 8.61 | 35.98 | +27.37 |
| r11_l1_single_fan_fixed_nominal | spatial_flow_belief_memory_h3 | 20 | 0.400 | 0.450 | +0.050 | 8.61 | 35.98 | +27.37 |
| r11_l1_single_fan_fixed_nominal | spatial_flow_belief_memory_h30 | 20 | 0.400 | 0.450 | +0.050 | 8.61 | 35.98 | +27.37 |
| r11_l2_four_fan_fixed_nominal | no_memory_baseline | 20 | 0.500 | 0.650 | +0.150 | 29.65 | 77.34 | +47.69 |
| r11_l2_four_fan_fixed_nominal | spatial_flow_belief_memory_h10 | 20 | 0.500 | 0.650 | +0.150 | 29.65 | 77.34 | +47.69 |
| r11_l2_four_fan_fixed_nominal | spatial_flow_belief_memory_h3 | 20 | 0.500 | 0.650 | +0.150 | 29.65 | 77.34 | +47.69 |
| r11_l2_four_fan_fixed_nominal | spatial_flow_belief_memory_h30 | 20 | 0.500 | 0.650 | +0.150 | 29.65 | 77.34 | +47.69 |
| r11_l3_fan_parameter_uncertainty | no_memory_baseline | 20 | 0.350 | 0.550 | +0.200 | -1.30 | 55.28 | +56.59 |
| r11_l3_fan_parameter_uncertainty | spatial_flow_belief_memory_h10 | 20 | 0.350 | 0.550 | +0.200 | -1.30 | 55.27 | +56.57 |
| r11_l3_fan_parameter_uncertainty | spatial_flow_belief_memory_h3 | 20 | 0.350 | 0.550 | +0.200 | -1.30 | 55.27 | +56.57 |
| r11_l3_fan_parameter_uncertainty | spatial_flow_belief_memory_h30 | 20 | 0.350 | 0.550 | +0.200 | -1.30 | 55.27 | +56.57 |
| r11_l4_local_fan_position_uncertainty | no_memory_baseline | 20 | 0.350 | 0.550 | +0.200 | -1.04 | 49.16 | +50.21 |
| r11_l4_local_fan_position_uncertainty | spatial_flow_belief_memory_h10 | 20 | 0.350 | 0.550 | +0.200 | -1.04 | 49.14 | +50.19 |
| r11_l4_local_fan_position_uncertainty | spatial_flow_belief_memory_h3 | 20 | 0.350 | 0.550 | +0.200 | -1.04 | 49.14 | +50.19 |
| r11_l4_local_fan_position_uncertainty | spatial_flow_belief_memory_h30 | 20 | 0.350 | 0.550 | +0.200 | -1.04 | 49.14 | +50.19 |
| r11_l5_active_fan_count_uncertainty | no_memory_baseline | 20 | 0.300 | 0.400 | +0.100 | -11.10 | 26.62 | +37.72 |
| r11_l5_active_fan_count_uncertainty | spatial_flow_belief_memory_h10 | 20 | 0.300 | 0.400 | +0.100 | -11.10 | 26.62 | +37.72 |
| r11_l5_active_fan_count_uncertainty | spatial_flow_belief_memory_h3 | 20 | 0.300 | 0.400 | +0.100 | -11.10 | 26.62 | +37.72 |
| r11_l5_active_fan_count_uncertainty | spatial_flow_belief_memory_h30 | 20 | 0.300 | 0.400 | +0.100 | -11.10 | 26.62 | +37.72 |
| r11_l6_environment_only_full_uncertainty | no_memory_baseline | 20 | 0.250 | 0.450 | +0.200 | -21.10 | 37.74 | +58.84 |
| r11_l6_environment_only_full_uncertainty | spatial_flow_belief_memory_h10 | 20 | 0.250 | 0.450 | +0.200 | -21.10 | 37.75 | +58.85 |
| r11_l6_environment_only_full_uncertainty | spatial_flow_belief_memory_h3 | 20 | 0.250 | 0.450 | +0.200 | -21.10 | 37.75 | +58.85 |
| r11_l6_environment_only_full_uncertainty | spatial_flow_belief_memory_h30 | 20 | 0.250 | 0.450 | +0.200 | -21.10 | 37.75 | +58.85 |
| r11_l7_full_domain_randomisation_arena_wide | no_memory_baseline | 20 | 0.250 | 0.300 | +0.050 | -22.50 | 8.66 | +31.16 |
| r11_l7_full_domain_randomisation_arena_wide | spatial_flow_belief_memory_h10 | 20 | 0.250 | 0.300 | +0.050 | -22.50 | 8.66 | +31.16 |
| r11_l7_full_domain_randomisation_arena_wide | spatial_flow_belief_memory_h3 | 20 | 0.250 | 0.300 | +0.050 | -22.50 | 8.66 | +31.16 |
| r11_l7_full_domain_randomisation_arena_wide | spatial_flow_belief_memory_h30 | 20 | 0.250 | 0.300 | +0.050 | -22.50 | 8.66 | +31.16 |

Figures:

- `r11_l0_dry_air_fixed`: `03_Control/A_figures/R11_E03.6_balanced_neutral_baseline/figures/r11_e03_6_bal_neutral_s36_l0.png`
- `r11_l1_single_fan_fixed_nominal`: `03_Control/A_figures/R11_E03.6_balanced_neutral_baseline/figures/r11_e03_6_bal_neutral_s36_l1.png`
- `r11_l2_four_fan_fixed_nominal`: `03_Control/A_figures/R11_E03.6_balanced_neutral_baseline/figures/r11_e03_6_bal_neutral_s36_l2.png`
- `r11_l3_fan_parameter_uncertainty`: `03_Control/A_figures/R11_E03.6_balanced_neutral_baseline/figures/r11_e03_6_bal_neutral_s36_l3.png`
- `r11_l4_local_fan_position_uncertainty`: `03_Control/A_figures/R11_E03.6_balanced_neutral_baseline/figures/r11_e03_6_bal_neutral_s36_l4.png`
- `r11_l5_active_fan_count_uncertainty`: `03_Control/A_figures/R11_E03.6_balanced_neutral_baseline/figures/r11_e03_6_bal_neutral_s36_l5.png`
- `r11_l6_environment_only_full_uncertainty`: `03_Control/A_figures/R11_E03.6_balanced_neutral_baseline/figures/r11_e03_6_bal_neutral_s36_l6.png`
- `r11_l7_full_domain_randomisation_arena_wide`: `03_Control/A_figures/R11_E03.6_balanced_neutral_baseline/figures/r11_e03_6_bal_neutral_s36_l7.png`
