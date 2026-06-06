# Control Surface Effectiveness v3.1 Current-Model Refit

Date: 2026-06-06

## Scope

This refit freezes the active 40 ms neutral dry-air replay-aligned model and
fits only scalar aerodynamic surface-effectiveness multipliers on the existing
`control_mix` columns. It does not fit command conversion, measured surface
angles, hardware packet mapping, lateral/cross-axis derivatives, or
alpha-regime surface schedules.

Result root:

`03_Control/05_Results/control_surface_effectiveness/control_surface_effectiveness_v3_1_current_model_surface_refit/`

The result uses compact GitHub-safe paths:

- `metrics/`
- `reports/report.md`
- `manifests/manifest.json`
- `figures/`

Largest generated file: `metrics/cand_replay.csv`, about 5.6 MB.
Longest generated path observed: 241 characters.

## Promotion Decision

The held-out surface-scale gate accepts all three scalar surface-effectiveness
updates:

- `DELTA_A_AERO_EFFECTIVENESS_SCALE = 0.65`
- `DELTA_E_AERO_EFFECTIVENESS_SCALE = 0.70`
- `DELTA_R_AERO_EFFECTIVENESS_SCALE = 0.45`

The active calibration ID is now:

`neutral_dry_air_replay_040_local_s5_yaw0p75_clr0p60_surface_scale_v3p1_a0p65_e0p70_r0p45`

## Held-Out Replay Check

Compared with the frozen active baseline on held-out pulse rows, the combined
accepted-scale candidate improves all-surface replay:

| metric | frozen baseline | combined accepted scales |
|---|---:|---:|
| primary antisym residual | 0.4495 | 0.3574 |
| dx MAE [m] | 0.2857 | 0.2774 |
| dy MAE [m] | 0.4387 | 0.4346 |
| altitude-loss MAE [m] | 0.1809 | 0.1630 |
| final roll MAE [deg] | 14.88 | 11.33 |
| final pitch MAE [deg] | 12.54 | 11.78 |
| final yaw MAE [deg] | 20.08 | 16.07 |

## Claim Boundary

This is a scalar surface-effectiveness update only. Lateral/cross-axis
derivatives, side-force terms, aileron adverse-yaw derivatives, rudder roll
coupling, and alpha-regime surface schedules remain diagnostic rows and are not
promoted by this refit.
