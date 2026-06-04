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
- fit aerodynamic roll/yaw moment bias: `True`
- fit neutral trim: `True`

```powershell
python 03_Control/02_Inner_Loop/run_fit_neutral_dry_air_calibration.py --session-root 04_Flight_Test\05_Results\cal\n30 --run-label n30_staged_trim_diagnostic_rich_v1 --heldout-count 16 --heldout-seed 606 --replay-dt-s 0.005 --alignment-window-s 0.1 --workers 8 --coordinate-passes 3 --fit-workflow staged --fit-aero-moment-bias --fit-neutral-trim
```

## Best Parameters

- cd0 strip scale: `3`
- fuselage drag-area scale: `0.2`
- strip efficiency scale: `1.12`
- roll moment bias coefficient: `-0.02`
- yaw moment bias coefficient: `0`
- aileron neutral trim: `0.12` rad
- elevator neutral trim: `-0.105` rad
- rudder neutral trim: `0.18` rad

## Replay Fit Quality

- train count: `64`
- train dx MAE: `0.4195` m
- train dy MAE: `0.3927` m
- train altitude-loss MAE: `0.1639` m
- train sink-rate MAE: `0.1580` m/s
- train final roll MAE: `16.104` deg
- train final pitch MAE: `28.174` deg
- train final yaw MAE: `8.790` deg
- held-out count: `16`
- held-out dx MAE: `0.5137` m
- held-out dy MAE: `0.4851` m
- held-out altitude-loss MAE: `0.1884` m
- held-out sink-rate MAE: `0.1804` m/s
- held-out final roll MAE: `13.388` deg
- held-out final pitch MAE: `26.020` deg
- held-out final yaw MAE: `6.300` deg

## Interpretation

The staged workflow first fits the longitudinal drag/efficiency terms using forward, vertical, and pitch residuals. It then freezes those terms and fits Cl0/Cn0-style aerodynamic roll/yaw moment bias using lateral, roll, and yaw residuals.
The default lateral correction is a Cl0/Cn0-style aerodynamic moment bias, not a commanded surface offset. Surface trim is a separate opt-in diagnostic and should only be activated if physical surface-zero error is measured.
