# Real Glider Calibration Prep Report

- source session: `04_Flight_Test/05_Results/glider_calibration/neutral_30/20260602_225953`
- output root: `C:/Users/GH-X-ST/OneDrive - Imperial College London/Year 4/Final Year Project/01 - Github/Nausicaa/03_Control/05_Results/glider_model_calibration_prep/smoke_20260602_225953`
- generated: `2026-06-02T23:11:53`
- valid throws: `5`
- invalid launch attempts: `2`
- termination counts: `{'exit_gate_floor': 3, 'exit_gate_front_wall': 2}`

## Neutral-Glide Calibration Targets

These values are evidence targets for a later grey-box fit. They do not by themselves update the simulator.

- mean sink rate: `0.7089972278` m/s
- mean x/altitude-loss glide ratio: `4.596943501`
- mean launch speed: `6.231372817` m/s
- mean lateral displacement: `1.417728732` m
- mean rate-estimator confidence: `0.9413090011`
- mean spike-downweighted fraction: `0.01931333089`

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
