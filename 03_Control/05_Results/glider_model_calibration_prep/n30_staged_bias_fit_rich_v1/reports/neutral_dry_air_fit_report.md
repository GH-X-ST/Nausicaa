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
- fit neutral trim: `False`

```powershell
python 03_Control/02_Inner_Loop/run_fit_neutral_dry_air_calibration.py --session-root 04_Flight_Test\05_Results\cal\n30 --run-label n30_staged_bias_fit_rich_v1 --heldout-count 16 --heldout-seed 606 --replay-dt-s 0.005 --alignment-window-s 0.1 --workers 8 --coordinate-passes 3 --fit-workflow staged --fit-aero-moment-bias --no-fit-neutral-trim
```

## Best Parameters

- cd0 strip scale: `2`
- fuselage drag-area scale: `0.2`
- strip efficiency scale: `1.36`
- roll moment bias coefficient: `-0.02`
- yaw moment bias coefficient: `0`
- aileron neutral trim: `0` rad
- elevator neutral trim: `0` rad
- rudder neutral trim: `0` rad

## Replay Fit Quality

- train count: `64`
- train dx MAE: `0.2695` m
- train dy MAE: `0.3931` m
- train altitude-loss MAE: `0.1682` m
- train sink-rate MAE: `0.1625` m/s
- train final roll MAE: `15.315` deg
- train final pitch MAE: `46.060` deg
- train final yaw MAE: `17.326` deg
- held-out count: `16`
- held-out dx MAE: `0.3342` m
- held-out dy MAE: `0.4602` m
- held-out altitude-loss MAE: `0.1837` m
- held-out sink-rate MAE: `0.1781` m/s
- held-out final roll MAE: `12.556` deg
- held-out final pitch MAE: `43.739` deg
- held-out final yaw MAE: `11.331` deg

## Interpretation

The staged workflow first fits the longitudinal drag/efficiency terms using forward, vertical, and pitch residuals. It then freezes those terms and fits Cl0/Cn0-style aerodynamic roll/yaw moment bias using lateral, roll, and yaw residuals.
The default lateral correction is a Cl0/Cn0-style aerodynamic moment bias, not a commanded surface offset. Surface trim is a separate opt-in diagnostic and should only be activated if physical surface-zero error is measured.
