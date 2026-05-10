# Linearisation Audit

The command Jacobian reflects actuator lag. Aerodynamic control effectiveness appears in the full-state columns for actual surface deflections.

## Trim Summary

| Field | Value |
|---|---:|
| V_trim_m_s | 6.500000e+00 |
| alpha_rad | 5.329947e-02 |
| theta_rad | -4.808280e-02 |
| gamma_rad | -1.013823e-01 |
| sink_rate_m_s | 6.578564e-01 |
| delta_a_rad | 0.000000e+00 |
| delta_e_rad | -1.006085e-01 |
| delta_r_rad | 0.000000e+00 |

## Matrix Shapes

| Field | Value |
|---|---:|
| A shape | (15, 15) |
| B_cmd shape | (15, 3) |
| A_longitudinal shape | (5, 5) |
| B_longitudinal shape | (5, 1) |
| A_lateral shape | (6, 6) |
| B_lateral shape | (6, 2) |

## Dynamic Residual

Maximum residual excluding position rates: `2.098698e-09`

## Reduced Model Dimensions

| Model | A shape | B shape |
|---|---:|---:|
| Longitudinal | (5, 5) | (5, 1) |
| Lateral-directional | (6, 6) | (6, 2) |

## Key Derivatives

| Derivative | Value |
|---|---:|
| x_u | -2.257078e-01 |
| x_w | 1.507429e+00 |
| z_u | -1.817038e+00 |
| z_w | -2.253463e+01 |
| m_u | -5.120360e+00 |
| m_w | 9.597676e+01 |
| m_q | -4.390601e+01 |
| l_p | -4.395790e+01 |
| l_r | -1.225271e+00 |
| n_p | -3.251122e+00 |
| n_r | -2.733355e+00 |
| x_delta_e | 1.094691e+00 |
| z_delta_e | 1.860089e+01 |
| m_delta_e | 2.908672e+02 |
| y_delta_a | 6.924629e+00 |
| l_delta_a | 4.123815e+02 |
| n_delta_a | 2.681497e+01 |
| y_delta_r | -4.583089e+00 |
| l_delta_r | -1.103926e+01 |
| n_delta_r | 3.487058e+01 |
| delta_a_cmd | 1.666667e+01 |
| delta_e_cmd | 1.666667e+01 |
| delta_r_cmd | 1.666667e+01 |

## Sign Audit

| Check | Value | Status |
|---|---:|---|
| l_delta_a > 0 | 4.123815e+02 | pass |
| m_delta_e > 0 | 2.908672e+02 | pass |
| n_delta_r > 0 | 3.487058e+01 | pass |
| delta_a_cmd > 0 | 1.666667e+01 | pass |
| delta_e_cmd > 0 | 1.666667e+01 | pass |
| delta_r_cmd > 0 | 1.666667e+01 | pass |
