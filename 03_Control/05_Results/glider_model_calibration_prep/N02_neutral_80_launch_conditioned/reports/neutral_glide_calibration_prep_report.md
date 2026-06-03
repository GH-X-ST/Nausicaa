# Real Glider Calibration Prep Report

- source session count: `5`
- source sessions: `['04_Flight_Test/05_Results/cal/n30/20260602_225953', '04_Flight_Test/05_Results/cal/n30/20260602_231554', '04_Flight_Test/05_Results/cal/n30/20260603_134313', '04_Flight_Test/05_Results/cal/n30/20260603_143411', '04_Flight_Test/05_Results/cal/n30/20260603_144119']`
- output root: `C:/Users/GH-X-ST/OneDrive - Imperial College London/Year 4/Final Year Project/01 - Github/Nausicaa/03_Control/05_Results/glider_model_calibration_prep/N02_neutral_80_launch_conditioned`
- generated: `2026-06-03T15:24:41`
- valid throws: `80`
- invalid launch attempts: `43`
- termination counts: `{'exit_gate_floor': 73, 'exit_gate_front_wall': 7}`

## Neutral-Glide Calibration Targets

These values are evidence targets for a later grey-box fit. They do not by themselves update the simulator.
The empirical diagnostic fits `dx`, `dy`, `sink_rate`, and `duration` from launch-conditioned features; glide ratio is reported only as a derived residual from predicted distance and altitude loss.

- mean sink rate: `0.8495454038` m/s
- mean derived x/altitude-loss glide ratio: `3.964128281`
- mean launch speed: `5.318273152` m/s
- mean lateral displacement: `0.3887230348` m
- mean launch lateral offset: `0.1086578744` m
- mean launch height offset: `-0.01713613782` m
- mean rate-estimator confidence: `0.9459165238`
- mean spike-downweighted fraction: `0.009074391456`

## Empirical Held-Out Check

- empirical validation rows: `80`
- held-out rows: `5`
- held-out derived glide-ratio MAE: `0.1802132112`
- held-out dx MAE: `0.1386100658` m
- held-out sink-rate MAE: `0.02998220298` m/s

## Recommended Calibration Order

1. Use the full neutral set to fit bare-airframe trim/polar consistency first: distance, sink rate, duration, and pitch tendency.
2. Use held-out neutral throws to check that the fitted model predicts terminal wall/floor behaviour, not only mean sink.
3. Use pulse-ladder throws only after the neutral fit is stable, fitting control effectiveness and damping separately.
4. Regenerate R5/R7/R8/R10/R11 only after the model update is fixed and documented.
5. Inspect physical neutral trim before assigning residuals to aerodynamic coefficients: rudder/aileron zero, wing/tail asymmetry, CG yaw/roll bias, and elevator trim.

## Files Written

- `metrics/neutral_throw_summary.csv`
- `metrics/neutral_feature_target_table.csv`
- `metrics/session_termination_summary.csv`
- `metrics/neutral_aggregate_summary.csv`
- `metrics/empirical_fit_coefficients.csv`
- `metrics/empirical_heldout_validation.csv`
- `metrics/invalid_attempt_summary.csv`
- `manifests/calibration_prep_manifest.json`

## Claims Not Made

- No aerodynamic parameter was changed.
- No controller/library evidence was regenerated.
- No zero-shot transfer claim is made from this prep report alone.
