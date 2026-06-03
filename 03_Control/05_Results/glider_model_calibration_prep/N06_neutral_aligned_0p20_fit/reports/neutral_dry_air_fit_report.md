# Neutral Dry-Air Aligned-Replay Fit

This fit uses only neutral open-loop real throws and a 0.20 s first-motion alignment window.
Pulse/control-effectiveness throws are intentionally excluded.

## Best Parameters

- cd0 strip scale: `2.5`
- fuselage drag-area scale: `1`
- strip efficiency scale: `0.31`
- aileron neutral trim: `-0.1` rad
- elevator neutral trim: `0.045` rad
- rudder neutral trim: `-0.16` rad

## Replay Fit Quality

- train count: `75`
- train dx MAE: `0.1987` m
- train dy MAE: `0.6587` m
- train altitude-loss MAE: `0.1142` m
- train sink-rate MAE: `0.0976` m/s
- held-out count: `5`
- held-out dx MAE: `0.1972` m
- held-out dy MAE: `0.4667` m
- held-out altitude-loss MAE: `0.1489` m
- held-out sink-rate MAE: `0.1241` m/s

## Interpretation

If held-out dy remains large after neutral trim fitting, lateral error should not be forced into pulse effectiveness. Inspect physical trim/asymmetry and then fit control derivatives separately.
