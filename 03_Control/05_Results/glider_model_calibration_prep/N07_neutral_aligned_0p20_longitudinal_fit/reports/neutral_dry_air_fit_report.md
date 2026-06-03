# Neutral Dry-Air Aligned-Replay Fit

This fit uses only neutral open-loop real throws and a 0.20 s first-motion alignment window.
Pulse/control-effectiveness throws are intentionally excluded.

## Best Parameters

- cd0 strip scale: `3`
- fuselage drag-area scale: `5`
- strip efficiency scale: `0.31`
- aileron neutral trim: `0` rad
- elevator neutral trim: `0` rad
- rudder neutral trim: `0` rad

## Replay Fit Quality

- train count: `75`
- train dx MAE: `0.2522` m
- train dy MAE: `0.7442` m
- train altitude-loss MAE: `0.1138` m
- train sink-rate MAE: `0.0966` m/s
- held-out count: `5`
- held-out dx MAE: `0.2334` m
- held-out dy MAE: `0.3742` m
- held-out altitude-loss MAE: `0.1562` m
- held-out sink-rate MAE: `0.1358` m/s

## Interpretation

If held-out dy remains large after neutral trim fitting, lateral error should not be forced into pulse effectiveness. Inspect physical trim/asymmetry and then fit control derivatives separately.
