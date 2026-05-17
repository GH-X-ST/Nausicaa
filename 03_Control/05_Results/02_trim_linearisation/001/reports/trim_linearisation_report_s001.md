# Trim And Linearisation Audit Report

This is an audit-only trim and linearisation evidence report. It does not
implement a controller, primitive, OCP, TVLQR, governor, outer loop,
Vicon interface, hardware path, or high-incidence validation.

## Status

- Overall status: `pass`
- Required trim case: `natural_glide_6p5_none`
- Model-facing command input: `delta_cmd_rad` in radians.
- Raw normalised commands do not enter `state_derivative`.

## Tolerances

- Dynamic residual norm tolerance: `1e-06`
- Finite-difference epsilon: `1e-06`
- Jacobian max absolute error tolerance: `0.0001`
- Actuator Jacobian absolute tolerance: `1e-09`
- Uniform wind max derivative difference tolerance: `1e-09`

## Speed Coverage

- `natural_glide_6p5_none`: role `required`, speed `6.5` m/s, status `converged`
- `natural_glide_5p5_none`: role `optional`, speed `5.5` m/s, status `converged`
- `natural_glide_7p5_none`: role `optional`, speed `7.5` m/s, status `converged`
- `natural_glide_4p5_none`: role `diagnostic`, speed `4.5` m/s, status `converged`
- `uniform_cg_updraft_6p5`: role `optional_wind`, speed `6.5` m/s, status `converged`

Optional 5.5 m/s and 7.5 m/s trim failures are recorded without failing
the core 6.5 m/s acceptance audit. The 4.5 m/s case is diagnostic only.

Steady trim is not used to represent the agile post-turn exit. Post-agile
speed variation will be handled later by trajectory-knot linearisation
and primitive-library expansion over entry and exit speed.

## Dynamic Residuals

- `natural_glide_6p5_none`: dynamic residual `4.68837e-09`, pass `True`
- `natural_glide_5p5_none`: dynamic residual `4.02224e-09`, pass `True`
- `natural_glide_7p5_none`: dynamic residual `4.73149e-09`, pass `True`
- `natural_glide_4p5_none`: dynamic residual `4.54296e-09`, pass `True`
- `uniform_cg_updraft_6p5`: dynamic residual `4.68973e-09`, pass `True`

## Finite-Difference Check

- `natural_glide_6p5_none`: A max `1.93451e-07`, B max `1.22427e-11`, pass `True`
- `natural_glide_5p5_none`: A max `1.26949e-07`, B max `1.66693e-11`, pass `True`
- `natural_glide_7p5_none`: A max `2.40489e-07`, B max `1.22427e-11`, pass `True`
- `natural_glide_4p5_none`: A max `6.54793e-08`, B max `4.45926e-10`, pass `True`

## Derivative Sign Checks

- `natural_glide_6p5_none` `m_delta_e` = `285.66`, expected `positive`, pass `True`
- `natural_glide_6p5_none` `l_delta_a` = `386.701`, expected `positive`, pass `True`
- `natural_glide_6p5_none` `n_delta_r` = `32.5335`, expected `positive`, pass `True`
- `natural_glide_5p5_none` `m_delta_e` = `194.321`, expected `positive`, pass `True`
- `natural_glide_5p5_none` `l_delta_a` = `268.83`, expected `positive`, pass `True`
- `natural_glide_5p5_none` `n_delta_r` = `23.1742`, expected `positive`, pass `True`
- `natural_glide_7p5_none` `m_delta_e` = `383.589`, expected `positive`, pass `True`
- `natural_glide_7p5_none` `l_delta_a` = `517.765`, expected `positive`, pass `True`
- `natural_glide_7p5_none` `n_delta_r` = `43.4137`, expected `positive`, pass `True`
- `natural_glide_4p5_none` `m_delta_e` = `53.9499`, expected `positive`, pass `True`
- `natural_glide_4p5_none` `l_delta_a` = `94.9154`, expected `positive`, pass `True`
- `natural_glide_4p5_none` `n_delta_r` = `15.2557`, expected `positive`, pass `True`

Reduced-model eigenvalues, controllability-style rank, and open-loop
stability are diagnostic only. They do not fail the audit unless the
reduced matrices have wrong shape, nonfinite values, or inconsistent indexing.

## Uniform Wind Consistency

- Wind vector: `(0.0, 0.0, 0.5)` m/s in public z-up world axes.
- Max absolute derivative difference: `0`
- Pass: `True`

## Future Cases

- `mild_bank_segment`: `not_available_yet`
- `recovery_segment`: `not_available_yet`
- `agile_reversal_knots`: `not_available_yet`
- `high_alpha_braking_segment`: `not_available_yet`

## Validation Commands

- `python -m py_compile 03_Control/04_Scenarios/trim_linearisation_audit.py 03_Control/04_Scenarios/run_trim_linearisation_audit.py`
- `python 03_Control/04_Scenarios/run_trim_linearisation_audit.py --run-id 1 --overwrite`
- `python -m pytest -q 03_Control/tests/test_trim_linearisation_audit.py`
- `python -m pytest -q 03_Control/tests`
