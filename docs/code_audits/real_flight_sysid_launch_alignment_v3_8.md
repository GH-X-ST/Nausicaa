# Real-Flight SysID Launch Alignment v3.8

Date: 2026-06-06

## Scope

This note records the documentation and active-model status after
`Real-Flight SysID Launch Alignment v3.7`.

The active dry-air replay correction is now:

`neutral_dry_air_replay_040_local_s5_yaw0p75_clr0p60_surface_scale_v3p1_a0p65_e0p70_r0p45`

This extends the v3.7 promoted 40 ms neutral replay alignment with a
current-model control-surface effectiveness refit. The refit freezes the active
neutral dry-air model and changes only scalar aerodynamic effectiveness on the
existing `control_mix` columns.

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

The held-out surface-scale gate accepts all three scalar surface-effectiveness
updates:

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

## Documentation Audit

The active workflow docs now describe:

- the v3.8 active calibration ID and surface scales;
- v3.1 evidence-gated surface-effectiveness promotion for aileron, elevator,
  and rudder;
- `control_mix`-column scaling as the only promoted surface-effectiveness
  mechanism;
- compact result filenames under `metrics/`, `reports/`, and `manifests/`;
- W3 plant/implementation perturbations around active scales
  `delta_a=0.65`, `delta_e=0.70`, and `delta_r=0.45`.

Historical v3.6/v3.7 audit notes are left as historical records and retain the
older calibration IDs where they describe earlier evidence.

## Claim Boundary

This update is a scalar surface-effectiveness replay-alignment update only.
Lateral/cross-axis derivatives, side-force terms, aileron adverse-yaw
derivatives, rudder roll coupling, command conversion, measured surface-angle
conventions, hardware packet mapping, and alpha-regime surface schedules remain
diagnostic and are not promoted by v3.8.

No R10/R11 validation, hardware-readiness, real-flight transfer, mission
success, full aerodynamic SysID, or autonomy claim is made by this update.

## Checks

- `python -m pytest 03_Control/tests/test_control_surface_effectiveness_study.py --basetemp .codex_pytest_tmp_cse -p no:cacheprovider`
- `python -m pytest 03_Control/tests/test_model_foundation_smoke.py 03_Control/tests/test_prim_model.py --basetemp .codex_pytest_tmp_model -p no:cacheprovider`
- `git diff --check`
