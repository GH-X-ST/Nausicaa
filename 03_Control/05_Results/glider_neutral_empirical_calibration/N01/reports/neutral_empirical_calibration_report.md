# Neutral Glide Empirical Calibration N01

- source: `04_Flight_Test/05_Results/glider_calibration/neutral_30`
- valid throws: `60`
- split: `55 train / 5 random held-out`, seed `603`
- holdout throws: `['20260602_231554/throw_008', '20260602_231554/throw_014', '20260602_231554/throw_015', '20260602_231554/throw_020', '20260603_134313/throw_003']`
- terminations: `{'exit_gate_floor': 53, 'exit_gate_front_wall': 7}`

## Aggregate Targets

- launch_speed_m_s: mean `5.415`, median `5.478`, range `[3.932, 6.825]`
- sink_rate_m_s: mean `0.8388`, median `0.8505`, range `[0.3762, 1.146]`
- glide_ratio_x_over_altloss: mean `4.068`, median `3.906`, range `[1.872, 10.47]`
- dx_m: mean `4.423`, median `4.488`, range `[2.579, 5.352]`
- dy_m: mean `0.5106`, median `0.4041`, range `[-0.6302, 2.053]`
- duration_s: mean `1.357`, median `1.371`, range `[1.01, 1.702]`

## Held-Out Validation

- sink_rate_m_s: holdout MAE `0.02509`, RMSE `0.03082`, bias `-0.009306`
- dx_m: holdout MAE `0.0726`, RMSE `0.08939`, bias `0.01552`
- dy_m: holdout MAE `0.2285`, RMSE `0.2542`, bias `0.09301`
- duration_s: holdout MAE `0.03431`, RMSE `0.04187`, bias `-0.03431`
- glide_ratio_x_over_altloss: holdout MAE `0.434`, RMSE `0.4699`, bias `0.2248`

## Interpretation

This is an empirical neutral-glide calibration target, not an aerodynamic coefficient mutation. Use it to set dry-air sink/glide/trim targets before changing the 6-DoF model, then re-run simulation evidence after coefficients are updated.

## Methodology Alignment

This calibration stage follows a simple grey-box system-identification workflow:

1. Separate the problem into physically interpretable stages. Neutral dry-air
   throws identify bare-airframe sink, glide ratio, duration, lateral drift, and
   launch-condition dependence. Pulse-ladder throws are reserved for control
   effectiveness after the neutral model is credible.
2. Use a small model first. The neutral empirical model uses launch speed,
   launch height, launch pitch, and launch body rates to predict sink rate,
   forward travel, lateral travel, duration, and glide ratio. It is deliberately
   not a high-order black-box model.
3. Validate on held-out flights. Five neutral throws are randomly held out and
   are not used for fitting. Their residuals are reported separately from the
   training residuals.
4. Update only a small set of physical parameters next. The intended next
   calibration targets are drag/polar consistency, trim/static-margin tendency,
   and low-order damping/effectiveness terms. The controller, primitive library,
   and memory governor should not be changed directly from this empirical fit.
5. Regenerate evidence after any model mutation. Any accepted aerodynamic or
   actuator-parameter update must be followed by a fresh R5/R7/R8/R10/R11
   evidence chain before closed-loop deployment claims are made.

This matches standard system-identification practice: design/collect informative
data, select the simplest adequate model structure, estimate parameters, and
validate on data not used for fitting. It also matches aircraft system
identification practice, where model structure, data quality, residuals, and
uncertainty are treated as part of the evidence rather than hidden tuning.

References used for methodology framing:

- Lennart Ljung, *System Identification: Theory for the User*, 2nd ed.,
  Prentice Hall, 1999.
- Vladislav Klein and Eugene A. Morelli, *Aircraft System Identification:
  Theory and Practice*, AIAA/Sunflyte.
- Eugene A. Morelli and Jared A. Grauer, "Advances in Aircraft System
  Identification at NASA Langley Research Center", NASA Technical Reports
  Server, 2023.

## Claims Not Made

- No aerodynamic coefficient has been changed by this report.
- No actuator effectiveness coefficient has been changed by this report.
- No primitive library, governor, memory logic, or R11 validation result is
  updated by this report.
- The empirical fit is a calibration target and sanity check, not the final
  flight model.
