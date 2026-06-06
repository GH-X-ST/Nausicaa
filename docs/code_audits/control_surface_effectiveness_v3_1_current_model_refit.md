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
updates. These values are retained as historical v3.1 scalar replay evidence:

- `DELTA_A_AERO_EFFECTIVENESS_SCALE = 0.65`
- `DELTA_E_AERO_EFFECTIVENESS_SCALE = 0.70`
- `DELTA_R_AERO_EFFECTIVENESS_SCALE = 0.45`

The later active calibration ID is now:

`neutral_dry_air_residual_calibrated_replay_n30_joint_pareto_040_local_s5_yaw0p75_clr0p60_surface_schedule_v3p2_cons_nominal`

The active nominal model no longer bakes these scalar values into
`control_mix`. It preserves the neutral-glide residual calibration and promotes
only a conservative alpha-regime surface-effectiveness schedule:

| surface | normal | transition | post-stall |
|---|---:|---:|---:|
| aileron | 0.85 | 0.55 | 0.45 |
| elevator | 0.75 | 0.55 | 0.45 |
| rudder | 0.85 | 0.55 | 0.40 |

The schedule is blended with the same 14--18 deg smooth residual-blend
activation used by the aero model, so `control_mix` remains the geometry/sign
mapping and aerodynamic surface authority is applied inside the dynamics.

Reference support: NASA post-stall/gain-scheduled control evidence
(https://ntrs.nasa.gov/citations/20050207439), NASA high-AoA
aileron/rudder effectiveness evidence
(https://ntrs.nasa.gov/api/citations/19740018330/downloads/19740018330.pdf),
NASA-derived high-AoA control-effectiveness summary
(https://pressbooks.lib.vt.edu/configurationaerodynamics/chapter/high-angle-of-attack-aerodynamics/),
and small-UAV SysID data-quality caution
(https://dept.aem.umn.edu/~mettler/Courses/AEM%205333%20%28spring%202013%29/AEM5333%20CourseDropbox/Week%207%20Identification/Ultrastick%20Identification/2012_AIAA_JA_SYSID.pdf).

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

This is a conservative scheduled surface-effectiveness update only.
Lateral/cross-axis derivatives, side-force terms, aileron adverse-yaw
derivatives, rudder roll coupling, command conversion, measured surface-angle
conventions, hardware packet mapping, and servo signs remain unchanged. The
schedule is not claimed as exact identified control derivatives or full lateral
aerodynamic SysID.
