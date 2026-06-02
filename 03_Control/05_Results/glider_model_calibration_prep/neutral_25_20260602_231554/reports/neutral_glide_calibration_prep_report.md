# Real Glider Calibration Prep Report

- source session: `04_Flight_Test/05_Results/glider_calibration/neutral_30/20260602_231554`
- output root: `C:/Users/GH-X-ST/OneDrive - Imperial College London/Year 4/Final Year Project/01 - Github/Nausicaa/03_Control/05_Results/glider_model_calibration_prep/neutral_25_20260602_231554`
- generated: `2026-06-02T23:28:24`
- valid throws: `25`
- invalid launch attempts: `21`
- termination counts: `{'exit_gate_front_wall': 3, 'exit_gate_floor': 22}`

## Neutral-Glide Calibration Targets

These values are evidence targets for a later grey-box fit. They do not by themselves update the simulator.

- mean sink rate: `0.8106169604` m/s
- mean x/altitude-loss glide ratio: `3.911328229`
- mean launch speed: `5.627771673` m/s
- mean lateral displacement: `0.6456383089` m
- mean rate-estimator confidence: `0.9390002202`
- mean spike-downweighted fraction: `0.02444395504`

## Recommended Calibration Order

1. Use the full neutral set to fit bare-airframe trim/polar consistency first: sink rate, glide ratio, and pitch tendency.
2. Use held-out neutral throws to check that the fitted model predicts terminal wall/floor behaviour, not only mean sink.
3. Use pulse-ladder throws only after the neutral fit is stable, fitting control effectiveness and damping separately.
4. Regenerate R5/R7/R8/R10/R11 only after the model update is fixed and documented.

## Files Written

- `metrics/neutral_throw_summary.csv`
- `metrics/neutral_aggregate_summary.csv`
- `metrics/invalid_attempt_summary.csv`
- `manifests/calibration_prep_manifest.json`

## Claims Not Made

- No aerodynamic parameter was changed.
- No controller/library evidence was regenerated.
- No zero-shot transfer claim is made from this prep report alone.
