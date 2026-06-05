# R11 E02 Balanced-Cluster Neutral-Intervention Diagnostic

- Diagnostic version: `r11_balanced_true_neutral_intervention_diagnostic_v1`
- R11 root: `03_Control/05_Results/R11_validation/E02`
- Library tier: `balanced_cluster`
- True neutral cases: `16`
- Figure count: `8`

`no_memory_baseline` is still closed-loop governor/LQR control. The neutral baseline here is a separate zero-command plant rollout from the same R11 final launch states.

The open-loop layer is run once per unique R11 final outer case. This is the correct comparison unit because all library tiers and memory policies reuse the same final launch state within that outer case.

| Ladder | Policy | n | Neutral mission | Controlled mission | Mission delta | Neutral score | Controlled score | Score delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| r11_l0_dry_air_fixed | no_memory_baseline | 2 | 0.500 | 0.500 | +0.000 | 20.00 | 10.00 | -10.00 |
| r11_l0_dry_air_fixed | spatial_flow_belief_memory_h10 | 2 | 0.500 | 0.500 | +0.000 | 20.00 | 10.00 | -10.00 |
| r11_l0_dry_air_fixed | spatial_flow_belief_memory_h3 | 2 | 0.500 | 0.500 | +0.000 | 20.00 | 10.00 | -10.00 |
| r11_l0_dry_air_fixed | spatial_flow_belief_memory_h30 | 2 | 0.500 | 0.500 | +0.000 | 20.00 | 10.00 | -10.00 |
| r11_l1_single_fan_fixed_nominal | no_memory_baseline | 2 | 1.000 | 1.000 | +0.000 | 119.32 | 132.28 | +12.97 |
| r11_l1_single_fan_fixed_nominal | spatial_flow_belief_memory_h10 | 2 | 1.000 | 1.000 | +0.000 | 119.32 | 132.28 | +12.97 |
| r11_l1_single_fan_fixed_nominal | spatial_flow_belief_memory_h3 | 2 | 1.000 | 1.000 | +0.000 | 119.32 | 132.28 | +12.97 |
| r11_l1_single_fan_fixed_nominal | spatial_flow_belief_memory_h30 | 2 | 1.000 | 1.000 | +0.000 | 119.32 | 132.28 | +12.97 |
| r11_l2_four_fan_fixed_nominal | no_memory_baseline | 2 | 1.000 | 1.000 | +0.000 | 119.77 | 139.27 | +19.51 |
| r11_l2_four_fan_fixed_nominal | spatial_flow_belief_memory_h10 | 2 | 1.000 | 1.000 | +0.000 | 119.77 | 139.27 | +19.50 |
| r11_l2_four_fan_fixed_nominal | spatial_flow_belief_memory_h3 | 2 | 1.000 | 1.000 | +0.000 | 119.77 | 139.27 | +19.50 |
| r11_l2_four_fan_fixed_nominal | spatial_flow_belief_memory_h30 | 2 | 1.000 | 1.000 | +0.000 | 119.77 | 139.27 | +19.51 |
| r11_l3_fan_parameter_uncertainty | no_memory_baseline | 2 | 1.000 | 1.000 | +0.000 | 118.34 | 127.20 | +8.86 |
| r11_l3_fan_parameter_uncertainty | spatial_flow_belief_memory_h10 | 2 | 1.000 | 1.000 | +0.000 | 118.34 | 127.20 | +8.86 |
| r11_l3_fan_parameter_uncertainty | spatial_flow_belief_memory_h3 | 2 | 1.000 | 1.000 | +0.000 | 118.34 | 127.20 | +8.86 |
| r11_l3_fan_parameter_uncertainty | spatial_flow_belief_memory_h30 | 2 | 1.000 | 1.000 | +0.000 | 118.34 | 127.20 | +8.86 |
| r11_l4_local_fan_position_uncertainty | no_memory_baseline | 2 | 1.000 | 1.000 | +0.000 | 115.09 | 125.76 | +10.67 |
| r11_l4_local_fan_position_uncertainty | spatial_flow_belief_memory_h10 | 2 | 1.000 | 1.000 | +0.000 | 115.09 | 125.68 | +10.60 |
| r11_l4_local_fan_position_uncertainty | spatial_flow_belief_memory_h3 | 2 | 1.000 | 1.000 | +0.000 | 115.09 | 125.68 | +10.60 |
| r11_l4_local_fan_position_uncertainty | spatial_flow_belief_memory_h30 | 2 | 1.000 | 1.000 | +0.000 | 115.09 | 125.68 | +10.60 |
| r11_l5_active_fan_count_uncertainty | no_memory_baseline | 2 | 1.000 | 1.000 | +0.000 | 118.35 | 121.46 | +3.11 |
| r11_l5_active_fan_count_uncertainty | spatial_flow_belief_memory_h10 | 2 | 1.000 | 1.000 | +0.000 | 118.35 | 121.46 | +3.11 |
| r11_l5_active_fan_count_uncertainty | spatial_flow_belief_memory_h3 | 2 | 1.000 | 1.000 | +0.000 | 118.35 | 121.46 | +3.11 |
| r11_l5_active_fan_count_uncertainty | spatial_flow_belief_memory_h30 | 2 | 1.000 | 1.000 | +0.000 | 118.35 | 121.62 | +3.27 |
| r11_l6_environment_only_full_uncertainty | no_memory_baseline | 2 | 0.500 | 0.500 | +0.000 | 20.00 | 40.00 | +20.00 |
| r11_l6_environment_only_full_uncertainty | spatial_flow_belief_memory_h10 | 2 | 0.500 | 0.500 | +0.000 | 20.00 | 40.00 | +20.00 |
| r11_l6_environment_only_full_uncertainty | spatial_flow_belief_memory_h3 | 2 | 0.500 | 0.500 | +0.000 | 20.00 | 40.00 | +20.00 |
| r11_l6_environment_only_full_uncertainty | spatial_flow_belief_memory_h30 | 2 | 0.500 | 0.500 | +0.000 | 20.00 | 40.00 | +20.00 |
| r11_l7_full_domain_randomisation_arena_wide | no_memory_baseline | 2 | 0.500 | 0.500 | +0.000 | 20.00 | 10.00 | -10.00 |
| r11_l7_full_domain_randomisation_arena_wide | spatial_flow_belief_memory_h10 | 2 | 0.500 | 0.500 | +0.000 | 20.00 | 10.00 | -10.00 |
| r11_l7_full_domain_randomisation_arena_wide | spatial_flow_belief_memory_h3 | 2 | 0.500 | 0.500 | +0.000 | 20.00 | 10.00 | -10.00 |
| r11_l7_full_domain_randomisation_arena_wide | spatial_flow_belief_memory_h30 | 2 | 0.500 | 0.500 | +0.000 | 20.00 | 10.00 | -10.00 |

Figures:

- `r11_l0_dry_air_fixed`: `03_Control/A_figures/R11_E02_balanced_neutral_baseline/figures/r11_e02_bal_neutral_s00_l0.png`
- `r11_l1_single_fan_fixed_nominal`: `03_Control/A_figures/R11_E02_balanced_neutral_baseline/figures/r11_e02_bal_neutral_s00_l1.png`
- `r11_l2_four_fan_fixed_nominal`: `03_Control/A_figures/R11_E02_balanced_neutral_baseline/figures/r11_e02_bal_neutral_s00_l2.png`
- `r11_l3_fan_parameter_uncertainty`: `03_Control/A_figures/R11_E02_balanced_neutral_baseline/figures/r11_e02_bal_neutral_s00_l3.png`
- `r11_l4_local_fan_position_uncertainty`: `03_Control/A_figures/R11_E02_balanced_neutral_baseline/figures/r11_e02_bal_neutral_s00_l4.png`
- `r11_l5_active_fan_count_uncertainty`: `03_Control/A_figures/R11_E02_balanced_neutral_baseline/figures/r11_e02_bal_neutral_s00_l5.png`
- `r11_l6_environment_only_full_uncertainty`: `03_Control/A_figures/R11_E02_balanced_neutral_baseline/figures/r11_e02_bal_neutral_s00_l6.png`
- `r11_l7_full_domain_randomisation_arena_wide`: `03_Control/A_figures/R11_E02_balanced_neutral_baseline/figures/r11_e02_bal_neutral_s00_l7.png`
