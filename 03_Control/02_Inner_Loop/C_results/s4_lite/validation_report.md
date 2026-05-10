# S4-Lite Validation Report

Measured residual-GP data were not available to this runner. Updraft stress cases marked as proxy use a deterministic analytic residual proxy.

## Linearisation Sign Snapshot

| Check | Value | Status |
|---|---:|---|
| l_delta_a > 0 | 4.123815e+02 | pass |
| m_delta_e > 0 | 2.908672e+02 | pass |
| n_delta_r > 0 | 3.487058e+01 | pass |

## Entry Conditions

| Scenario | Primitive | Accepted | Reasons |
|---|---|---|---|
| nominal_glide_zero_wind | nominal_glide | True |  |
| nominal_glide_crosswind | nominal_glide | True |  |
| bank_reversal_zero_wind | bank_reversal | True |  |
| bank_reversal_mild_updraft_proxy | bank_reversal | True |  |
| recovery_perturbed_strong_updraft_proxy | recovery | True |  |
| invalid_low_altitude_rejection | bank_reversal | False | altitude below entry floor: 0.100 m; governor: altitude below floor 0.100 m |

## Scenario Metrics

| Scenario | Primitive | Status | Duration s | Altitude loss m | Final speed m/s | Max bank deg | Max alpha deg | Log |
|---|---|---|---:|---:|---:|---:|---:|---|
| nominal_glide_zero_wind | nominal_glide | completed | 5 | 3.2639 | 6.47818 | 7.48831e-06 | 3.1102 | C:\Users\GH-X-ST\OneDrive - Imperial College London\Year 4\Final Year Project\01 - Github\Nausicaa\03_Control\02_Inner_Loop\C_results\s4_lite\logs\nominal_glide_zero_wind.csv |
| nominal_glide_crosswind | nominal_glide | completed | 5 | 2.96391 | 6.14138 | 0.3232 | 3.46584 | C:\Users\GH-X-ST\OneDrive - Imperial College London\Year 4\Final Year Project\01 - Github\Nausicaa\03_Control\02_Inner_Loop\C_results\s4_lite\logs\nominal_glide_crosswind.csv |
| bank_reversal_zero_wind | bank_reversal | completed | 7 | 5.62964 | 7.10252 | 16.5561 | 3.17345 | C:\Users\GH-X-ST\OneDrive - Imperial College London\Year 4\Final Year Project\01 - Github\Nausicaa\03_Control\02_Inner_Loop\C_results\s4_lite\logs\bank_reversal_zero_wind.csv |
| bank_reversal_mild_updraft_proxy | bank_reversal | completed | 7 | 5.61244 | 7.10367 | 16.5519 | 4.07929 | C:\Users\GH-X-ST\OneDrive - Imperial College London\Year 4\Final Year Project\01 - Github\Nausicaa\03_Control\02_Inner_Loop\C_results\s4_lite\logs\bank_reversal_mild_updraft_proxy.csv |
| recovery_perturbed_strong_updraft_proxy | recovery | completed | 5.5 | 3.5103 | 6.47085 | 18 | 4.28978 | C:\Users\GH-X-ST\OneDrive - Imperial College London\Year 4\Final Year Project\01 - Github\Nausicaa\03_Control\02_Inner_Loop\C_results\s4_lite\logs\recovery_perturbed_strong_updraft_proxy.csv |

## Implementation Wrappers

| Name | Limit deg | Deadband deg | Quantization deg | Extra delay s |
|---|---:|---:|---:|---:|
| deterministic_direct_servo_proxy | [25.0, 25.0, 25.0] | [0.15, 0.15, 0.15] | [0.25, 0.25, 0.25] | [0.0, 0.0, 0.0] |
| deterministic_surface_delay_proxy | [25.0, 25.0, 25.0] | [0.15, 0.15, 0.15] | [0.25, 0.25, 0.25] | [0.02, 0.0, 0.02] |
