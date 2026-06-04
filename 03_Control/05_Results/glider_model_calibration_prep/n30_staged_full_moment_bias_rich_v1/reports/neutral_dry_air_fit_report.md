# Neutral Dry-Air Lateral/Longitudinal Aerodynamic Bias Fit

This fit uses only neutral open-loop real throws and a first-motion alignment window.
Pulse/control-effectiveness throws are intentionally excluded.

## Rerun Recipe

- source session root: `04_Flight_Test\05_Results\cal\n30`
- alignment window: `0.100` s
- held-out count: `16`
- workers: `8`
- fit workflow: `staged`
- coordinate passes: `3`
- fit aerodynamic moment bias: `True`
- fit neutral trim: `False`

```powershell
python 03_Control/02_Inner_Loop/run_fit_neutral_dry_air_calibration.py --session-root 04_Flight_Test\05_Results\cal\n30 --run-label n30_staged_full_moment_bias_rich_v1 --heldout-count 16 --heldout-seed 606 --replay-dt-s 0.005 --alignment-window-s 0.1 --workers 8 --coordinate-passes 3 --fit-workflow staged --fit-aero-moment-bias --no-fit-neutral-trim
```

## Best Parameters

- cd0 strip scale: `2.5`
- fuselage drag-area scale: `0.2`
- strip efficiency scale: `1.36`
- roll moment bias coefficient: `-0.02`
- pitch moment bias coefficient: `-0.02`
- yaw moment bias coefficient: `0`
- aileron neutral trim: `0` rad
- elevator neutral trim: `0` rad
- rudder neutral trim: `0` rad

## Replay Fit Quality

- train count: `64`
- train dx MAE: `0.3574` m
- train dy MAE: `0.4164` m
- train altitude-loss MAE: `0.1765` m
- train sink-rate MAE: `0.1702` m/s
- train final roll MAE: `15.408` deg
- train final pitch MAE: `38.268` deg
- train final yaw MAE: `16.579` deg
- held-out count: `16`
- held-out dx MAE: `0.4537` m
- held-out dy MAE: `0.4719` m
- held-out altitude-loss MAE: `0.1740` m
- held-out sink-rate MAE: `0.1667` m/s
- held-out final roll MAE: `12.816` deg
- held-out final pitch MAE: `35.993` deg
- held-out final yaw MAE: `10.466` deg

## Interpretation

The staged workflow first fits the longitudinal drag/efficiency terms and Cm0-style pitch moment bias using forward, vertical, and pitch residuals. It then freezes those terms and fits Cl0/Cn0-style aerodynamic roll/yaw moment bias using lateral, roll, and yaw residuals.
The default moment correction is a Cm0/Cl0/Cn0-style aerodynamic bias, not a commanded surface offset. Surface trim is a separate opt-in diagnostic and should only be activated if physical surface-zero error is measured.
