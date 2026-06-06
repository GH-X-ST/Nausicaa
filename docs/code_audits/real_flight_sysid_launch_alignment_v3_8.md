# Real-Flight SysID Launch Alignment v3.8

Date: 2026-06-06

## Scope

This note records the documentation and active-model status after
`Real-Flight SysID Launch Alignment v3.7`.

The active dry-air replay correction is now:

`neutral_dry_air_residual_calibrated_replay_n30_joint_pareto_040_local_s5_yaw0p75_clr0p60_surface_schedule_v3p2_cons_nominal`

This preserves the v3.7 promoted 40 ms neutral replay alignment and changes
only the control-surface aerodynamic effectiveness layer. The v3.1 scalar
surface replay gate remains historical evidence; the active nominal model now
uses a physics-informed, replay-regularized, conservative alpha-regime schedule.

## Control-Surface Effectiveness Update

The v3.1 current-model control-surface study writes to:

`03_Control/05_Results/control_surface_effectiveness/control_surface_effectiveness_v3_1_current_model_surface_refit/`

The output follows the file-management rule with compact GitHub-safe paths:

- `metrics/replay.csv`
- `metrics/cand_replay.csv`
- `metrics/replay_err.csv`
- `metrics/regime_err.csv`
- `metrics/scale_gate.csv`
- `reports/report.md`
- `manifests/manifest.json`

Largest generated file observed: `metrics/cand_replay.csv`, about 5.6 MB.
Longest generated path observed: 241 characters.

The held-out surface-scale gate accepted all three scalar surface-effectiveness
updates, now retained as historical v3.1 evidence:

- `DELTA_A_AERO_EFFECTIVENESS_SCALE = 0.65`
- `DELTA_E_AERO_EFFECTIVENESS_SCALE = 0.70`
- `DELTA_R_AERO_EFFECTIVENESS_SCALE = 0.45`

The combined accepted-scale held-out replay improves the frozen active baseline:

| metric | frozen baseline | combined accepted scales |
|---|---:|---:|
| primary antisym residual | 0.4495 | 0.3574 |
| dx MAE [m] | 0.2857 | 0.2774 |
| dy MAE [m] | 0.4387 | 0.4346 |
| altitude-loss MAE [m] | 0.1809 | 0.1630 |
| final roll MAE [deg] | 14.88 | 11.33 |
| final pitch MAE [deg] | 12.54 | 11.78 |
| final yaw MAE [deg] | 20.08 | 16.07 |

The active nominal control-surface model is now the conservative scheduled
surface-effectiveness table:

| surface | normal | transition | post-stall |
|---|---:|---:|---:|
| aileron | 0.85 | 0.55 | 0.45 |
| elevator | 0.75 | 0.55 | 0.45 |
| rudder | 0.85 | 0.55 | 0.40 |

The schedule is applied inside the dynamics, not by command conversion or
measured surface-angle correction. `control_mix` remains the geometry/sign
mapping, and each derivative evaluation multiplies it by the smoothly blended
surface-effectiveness scale from the current simulated absolute AoA. The blend
uses the same 14--18 deg residual-blend smoothstep and normalized
normal/transition/post-stall weights as the calibrated aero model.

Reference support:

- NASA post-stall modeling and gain-scheduled control supports scheduling
  control authority across low-to-high AoA transitions:
  https://ntrs.nasa.gov/citations/20050207439
- NASA high-AoA wind-tunnel/free-flight evidence shows aileron rolling moment
  decreasing near stall and rudder effectiveness dropping after wing stall:
  https://ntrs.nasa.gov/api/citations/19740018330/downloads/19740018330.pdf
- NASA-derived high-AoA teaching material summarizes AoA-driven loss of control
  effectiveness, especially yawing-moment generation:
  https://pressbooks.lib.vt.edu/configurationaerodynamics/chapter/high-angle-of-attack-aerodynamics/
- Small-UAV SysID literature cautions that low-cost/small-aircraft data quality
  and missing flow-angle sensing justify conservative regularization rather than
  claiming exact derivative identification:
  https://dept.aem.umn.edu/~mettler/Courses/AEM%205333%20%28spring%202013%29/AEM5333%20CourseDropbox/Week%207%20Identification/Ultrastick%20Identification/2012_AIAA_JA_SYSID.pdf

## Documentation Audit

The active workflow docs now describe:

- the v3.8 active calibration ID and surface scales;
- v3.1 evidence-gated surface-effectiveness promotion for aileron, elevator,
  and rudder;
- v3.2/v3.3 stage-wise and constrained alpha-regime surface-effectiveness
  evidence as the basis for the conservative scheduled surface model;
- `control_mix` as geometry/sign mapping, with surface effectiveness applied
  inside dynamics by smooth AoA-regime schedule;
- compact result filenames under `metrics/`, `reports/`, and `manifests/`;
- W3 plant perturbations as a global `0.75--1.25` multiplier plus per-axis
  `0.85--1.15` multipliers on the active scheduled surface authority, not old
  v3.1 scalar-scale perturbations.

Historical v3.6/v3.7 audit notes are left as historical records and retain the
older calibration IDs where they describe earlier evidence.

## Claim Boundary

This update is a conservative scheduled surface-effectiveness update only.
Lateral/cross-axis derivatives, side-force terms, aileron adverse-yaw
derivatives, rudder roll coupling, command conversion, measured surface-angle
conventions, and hardware packet mapping remain unchanged. The schedule is not
claimed as exact identified control derivatives or full lateral aerodynamic
SysID.

No R10/R11 validation, hardware-readiness, real-flight transfer, mission
success, full aerodynamic SysID, or autonomy claim is made by this update.

## Checks

- `python -m pytest 03_Control/tests/test_control_surface_effectiveness_study.py --basetemp .codex_pytest_tmp_cse -p no:cacheprovider`
- `python -m pytest 03_Control/tests/test_model_foundation_smoke.py 03_Control/tests/test_prim_model.py --basetemp .codex_pytest_tmp_model -p no:cacheprovider`
- `git diff --check`
