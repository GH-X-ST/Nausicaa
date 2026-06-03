# Real Glider Calibration Prep Report

- source session count: `5`
- source sessions: `['04_Flight_Test/05_Results/cal/n30/20260602_225953', '04_Flight_Test/05_Results/cal/n30/20260602_231554', '04_Flight_Test/05_Results/cal/n30/20260603_134313', '04_Flight_Test/05_Results/cal/n30/20260603_143411', '04_Flight_Test/05_Results/cal/n30/20260603_144119']`
- output root: `C:/Users/GH-X-ST/OneDrive - Imperial College London/Year 4/Final Year Project/01 - Github/Nausicaa/03_Control/05_Results/glider_model_calibration_prep/N04_neutral_80_calibrated_measured_launch_replay`
- generated: `2026-06-03T16:44:43`
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
- held-out derived glide-ratio MAE: `0.1568784761`
- held-out dx MAE: `0.1044116002` m
- held-out sink-rate MAE: `0.04049114259` m/s

## Measured-Launch Simulation Replay

Each valid throw is replayed in the current dry-air simulator from the measured launch-plane state and the logged command history. Residuals are actual minus simulation, so launch variability is not fitted away as model error.

- replay rows: `80`
- successful replay rows: `80`
- mean replay dx residual: `-0.5005828801` m
- mean replay dy residual: `-0.2655962938` m
- mean replay altitude-loss residual: `-0.1063891777` m
- mean replay sink-rate residual: `-0.08467562569` m/s
- replay dy residual MAE: `0.8182829647` m

## Current Dry-Air Model Calibration

- calibration active: `True`
- calibration id: `neutral_dry_air_measured_launch_replay_N04`
- source prep run: `03_Control/05_Results/glider_model_calibration_prep/N03_neutral_80_measured_launch_replay`
- source throw count: `80`
- held-out policy: `randomised_stratified_by_session_label`
- cd0 strip scale: `6.0`
- fuselage drag-area scale: `10.0`
- strip efficiency scale: `0.75`

## Recommended Calibration Order

1. Use measured-launch replay residuals as the primary fair SysID target; do not compare real throws to a nominal launch.
2. Fit bare-airframe trim/polar consistency first: distance, sink rate, duration, and pitch tendency from neutral throws.
3. Check whether lateral residual remains after measured launch conditioning before assigning it to rudder/aileron trim, wing asymmetry, or y-CG offset.
4. Use pulse-ladder throws only after the neutral fit is stable, fitting control effectiveness and damping separately.
5. Regenerate R5/R7/R8/R10/R11 only after the model update is fixed and documented.
6. Inspect physical neutral trim before assigning residuals to aerodynamic coefficients: rudder/aileron zero, wing/tail asymmetry, CG yaw/roll bias, and elevator trim.

## Files Written

- `metrics/neutral_throw_summary.csv`
- `metrics/neutral_feature_target_table.csv`
- `metrics/session_termination_summary.csv`
- `metrics/neutral_aggregate_summary.csv`
- `metrics/empirical_fit_coefficients.csv`
- `metrics/empirical_heldout_validation.csv`
- `metrics/measured_launch_replay_residuals.csv`
- `metrics/invalid_attempt_summary.csv`
- `manifests/calibration_prep_manifest.json`

## Claims Not Made

- No controller/library evidence was regenerated.
- The current aerodynamic correction is neutral dry-air only; pulse/control-effectiveness fitting is still separate.
- No zero-shot transfer claim is made from this prep report alone.
