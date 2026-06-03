# Real Glider Calibration Prep Report

- source session count: `6`
- source sessions: `['04_Flight_Test/05_Results/cal/n30/20260602_225953', '04_Flight_Test/05_Results/cal/n30/20260602_231554', '04_Flight_Test/05_Results/cal/n30/20260603_134313', '04_Flight_Test/05_Results/cal/p30/20260603_140315', '04_Flight_Test/05_Results/cal/n30/20260603_143411', '04_Flight_Test/05_Results/cal/n30/20260603_144119']`
- output root: `C:/Users/GH-X-ST/OneDrive - Imperial College London/Year 4/Final Year Project/01 - Github/Nausicaa/03_Control/05_Results/glider_model_calibration_prep/N08_active_N07_all_cases_aligned_0p20_replay`
- generated: `2026-06-03T17:51:17`
- valid throws: `170`
- invalid launch attempts: `58`
- termination counts: `{'exit_gate_floor': 163, 'exit_gate_front_wall': 7}`

## Calibration Data Targets

These values are evidence targets for later grey-box fitting. They do not by themselves update the simulator.
The empirical diagnostic fits `dx`, `dy`, `sink_rate`, and `duration` from launch-conditioned features; glide ratio is reported only as a derived residual from predicted distance and altitude loss.

- mean sink rate: `0.899296581` m/s
- mean derived x/altitude-loss glide ratio: `3.722673834`
- mean launch speed: `5.017893048` m/s
- mean lateral displacement: `0.290507313` m
- mean launch lateral offset: `0.09776404059` m
- mean launch height offset: `-0.06124689963` m
- mean rate-estimator confidence: `0.9480388568`
- mean spike-downweighted fraction: `0.004358095942`

## Empirical Held-Out Check

- empirical validation rows: `170`
- held-out rows: `10`
- held-out derived glide-ratio MAE: `0.2744446602`
- held-out dx MAE: `0.189660393` m
- held-out sink-rate MAE: `0.04159071603` m/s

## Measured-Launch Simulation Replay

Each valid throw is replayed in the current dry-air simulator from the measured launch-plane state and the logged command history. Residuals are actual minus simulation, so launch variability is not fitted away as model error.

- replay rows: `170`
- successful replay rows: `170`
- mean replay dx residual: `-0.596517515` m
- mean replay dy residual: `-0.2841170883` m
- mean replay altitude-loss residual: `-0.103497214` m
- mean replay sink-rate residual: `-0.09873253992` m/s
- replay dy residual MAE: `0.7103617976` m

## First-Motion-Aligned Simulation Replay

Each valid throw is also replayed from a state aligned to the first short segment of measured motion. This diagnostic uses the target pose after the alignment window, regresses world velocity over the window, estimates body rate from the SO(3) rotation change, then predicts only the remaining trajectory.

- alignment window: `0.2` s
- aligned replay rows: `170`
- successful aligned replay rows: `170`
- mean aligned dx residual: `-0.1700052175` m
- mean aligned dy residual: `-0.2614982484` m
- mean aligned altitude-loss residual: `-0.06473783076` m
- mean aligned sink-rate residual: `-0.06479844655` m/s
- aligned dy residual MAE: `0.5679395001` m

## Current Dry-Air Model Calibration

- calibration active: `True`
- calibration id: `neutral_dry_air_aligned_0p20_N07`
- source prep run: `03_Control/05_Results/glider_model_calibration_prep/N07_neutral_aligned_0p20_longitudinal_fit`
- source throw count: `80`
- held-out policy: `randomised_stratified_by_session_label`
- cd0 strip scale: `3.0`
- fuselage drag-area scale: `5.0`
- strip efficiency scale: `0.31`
- aileron neutral trim: `0.0` rad
- elevator neutral trim: `0.0` rad
- rudder neutral trim: `0.0` rad

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
- `metrics/aligned_motion_replay_residuals.csv`
- `metrics/invalid_attempt_summary.csv`
- `manifests/calibration_prep_manifest.json`

## Claims Not Made

- No controller/library evidence was regenerated.
- The current aerodynamic correction is neutral dry-air only; pulse/control-effectiveness fitting is still separate.
- No zero-shot transfer claim is made from this prep report alone.
