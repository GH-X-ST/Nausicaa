# Real-Flight Neutral SysID Rich Fit Diagnosis v1

Date: 2026-06-04

## Data And Commands

Neutral data source:

```text
04_Flight_Test/05_Results/cal/n30
```

The source contains 8 `neutral_30` sessions and 80 valid neutral throws. The
held-out split used 16 throws with seed `606`.

Primary rich staged command:

```powershell
& 'C:\ProgramData\miniforge3\python.exe' '03_Control/02_Inner_Loop/run_fit_neutral_dry_air_calibration.py' `
  --session-root '04_Flight_Test/05_Results/cal/n30' `
  --run-label 'n30_staged_bias_fit_rich_v1' `
  --heldout-count 16 `
  --heldout-seed 606 `
  --alignment-window-s 0.10 `
  --workers 8 `
  --coordinate-passes 3 `
  --fit-workflow staged `
  --fit-aero-moment-bias `
  --no-fit-neutral-trim
```

Additional diagnostics were run with opt-in surface trim, full Cm0/Cl0/Cn0
moment bias, and the coupled longitudinal grid. The final diagnostic run was:

```text
03_Control/05_Results/glider_model_calibration_prep/n30_staged_coupled_moment_bias_rich_v2
```

## Result

The rich neutral SysID result is not acceptable for promotion to the active
calibration constants.

The final coupled diagnostic kept all 80 replay cases in `ok` status, but the
fit quality was still outside the planned warning scales:

```text
best parameters:
  cd0_strip_scale              2.0
  drag_area_fuse_scale         4.2
  efficiency_strip_scale       0.2
  roll_moment_bias_coeff      -0.02
  pitch_moment_bias_coeff     -0.04
  yaw_moment_bias_coeff        0.0
  delta_a/e/r_trim_rad         0.0

train residuals:
  dx MAE                       0.4247 m
  dy MAE                       0.3735 m
  altitude-loss MAE            0.1764 m
  sink-rate MAE                0.1704 m/s
  roll MAE                    14.365 deg
  pitch MAE                   32.056 deg
  yaw MAE                     16.500 deg

held-out residuals:
  dx MAE                       0.5234 m
  dy MAE                       0.4503 m
  altitude-loss MAE            0.1826 m
  sink-rate MAE                0.1773 m/s
  roll MAE                    11.855 deg
  pitch MAE                   28.946 deg
  yaw MAE                     11.936 deg
```

The strongest failure is longitudinal attitude: measured throws rotate
nose-down by exit, while the simulated replay remains substantially more
nose-up. The coupled grid can reduce pitch error only by degrading position and
energy residuals, so the mismatch is not explained cleanly by a small,
physically interpretable Cm0 bias alone.

The trim diagnostic is also not acceptable. It needs large neutral surface
trims, including a rudder trim at the search bound, and it does not resolve the
longitudinal residual. That is a diagnostic signal, not evidence for promoting
surface offsets.

## Active Calibration Decision

Historical note: this section records the decision at the time of this audit,
before the later n30 compact residual-calibrated replay model was promoted.

At the time of this historical audit, the rich-run parameters were not promoted
and the active checked-in constants were still the older accepted
`neutral_dry_air_aligned_0p20_N07` values. Later code now uses the selected
compact n30 replay row with attached Cm, transition Cm, post-stall Cm/Cmq,
14--18 deg blend timing, and selected compact coupling terms from
`neutral_dry_air_residual_calibrated_replay_n30_joint_pareto_040_local_s5_yaw0p75_clr0p60_elevator_rudder_effectiveness_v1`.
This audit was replay-diagnosis evidence, not the current active dry-air
correction.

## Control-Effect Audit

No completed per-axis pulse ladder dataset was found under the calibration
result root during this audit. The runtime design is suitable for collecting
visible control-effect evidence:

- `pulse_ladder_elevator_30`, `pulse_ladder_aileron_30`, and
  `pulse_ladder_rudder_30` cover the three axes separately.
- Commands are `+/-0.2`, `+/-0.4`, `+/-0.6`, `+/-0.8`, and `+/-1.0`.
- Each command case targets 3 valid throws.
- Each command starts 0.15 s after launch approval.
- Nonzero commands are sustained until normal wall/floor exit through the long
  60 s command window.
- Logs include aggregate command, physical surface command, packet surface
  command, and estimated actuator state.

This design is good enough for sign, monotonicity, and visibility checks.
However, control-effectiveness fitting should wait until the neutral dry-air
mismatch is resolved or explicitly labelled as a preliminary diagnostic.

## Next Technical Checks

Before claiming a new neutral model:

1. Re-run or inspect the Vicon orientation check with a simple physical
   nose-up/nose-down test and save the profile summary.
2. Inspect pitch-axis modelling assumptions: CG location, inertia, wing/tail
   incidence, elevator neutral geometry, and pitch-moment sign convention.
3. Keep pre-fix and post-fix airframe throws separated by session/profile/build
   tag if the vertical tail, CG, or control surfaces are physically adjusted.
4. Collect the relevant per-axis pulse ladder only as control-response evidence
   until the neutral model passes the held-out promotion gate.

No hardware-readiness, mission-success, real-flight transfer, control-
effectiveness fit, or autonomy claim is made by this audit.
