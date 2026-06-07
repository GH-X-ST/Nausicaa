# Real-Flight Simulation Replay Figures

- Figure run version: `real_flight_sim_replay_v1`
- Output root: `04_Flight_Test/A_figures/real_flight_sim_replay_E1_random_samples`
- Library tier: `balanced_cluster`
- Replay dt (s): `0.0050`
- Real-decision timing: `logged`
- First-window state audit: `04_Flight_Test/A_figures/real_flight_sim_replay_E1_random_samples/metrics/first_0p04_state_replay_error_summary.csv`
- Execution timing audit: `04_Flight_Test/A_figures/real_flight_sim_replay_E1_random_samples/metrics/execution_timing_audit.csv`

| case | session | throw | model | status | termination | duration s | final x m | final y m | final z m |
|---|---|---|---|---|---|---:|---:|---:|---:|
| E1.0 | 20260607_124146 | throw_002 | reality | measured | exit_gate_floor | 1.206 | 6.110 | 1.524 | 0.398 |
| E1.0 | 20260607_124146 | throw_002 | sim own governor | ok | exit_gate_front_wall | 1.135 | 6.602 | 2.290 | 0.636 |
| E1.0 | 20260607_124146 | throw_002 | sim real decisions | ok | exit_gate_front_wall | 1.130 | 6.605 | 2.504 | 0.615 |
| E1.0 | 20260607_124146 | throw_010 | reality | measured | exit_gate_floor | 1.059 | 5.729 | 1.827 | 0.394 |
| E1.0 | 20260607_124146 | throw_010 | sim own governor | ok | inside_operational_region | 1.059 | 6.007 | 2.558 | 0.470 |
| E1.0 | 20260607_124146 | throw_010 | sim real decisions | ok | inside_operational_region | 1.059 | 5.891 | 2.711 | 0.424 |
| E1.1 | 20260606_230007 | throw_005 | reality | measured | exit_gate_front_wall | 1.084 | 6.607 | 1.836 | 0.881 |
| E1.1 | 20260606_230007 | throw_005 | sim own governor | ok | exit_gate_front_wall | 0.930 | 6.605 | 2.065 | 0.527 |
| E1.1 | 20260606_230007 | throw_005 | sim real decisions | ok | exit_gate_front_wall | 0.933 | 6.625 | 2.028 | 0.393 |
| E1.1 | 20260606_230007 | throw_011 | reality | measured | exit_gate_front_wall | 1.063 | 6.616 | 1.993 | 1.326 |
| E1.1 | 20260606_230007 | throw_011 | sim own governor | ok | exit_gate_floor | 0.765 | 5.966 | 2.028 | 0.394 |
| E1.1 | 20260606_230007 | throw_011 | sim real decisions | ok | exit_gate_floor | 0.773 | 6.016 | 2.033 | 0.397 |
| E1.2 | 20260607_122640 | throw_002 | reality | measured | exit_gate_floor | 1.164 | 5.896 | 1.967 | 0.394 |
| E1.2 | 20260607_122640 | throw_002 | sim own governor | ok | inside_operational_region | 1.164 | 6.089 | 1.991 | 0.464 |
| E1.2 | 20260607_122640 | throw_002 | sim real decisions | ok | inside_operational_region | 1.164 | 6.084 | 2.024 | 0.494 |
| E1.2 | 20260607_122640 | throw_029 | reality | measured | exit_gate_floor | 1.260 | 6.284 | 1.795 | 0.391 |
| E1.2 | 20260607_122640 | throw_029 | sim own governor | ok | inside_operational_region | 1.260 | 6.462 | 2.039 | 0.543 |
| E1.2 | 20260607_122640 | throw_029 | sim real decisions | ok | inside_operational_region | 1.260 | 6.470 | 2.064 | 0.546 |

## First 0.04 s State Residual Audit

| case | throw | model | largest position MAE m | largest attitude MAE deg | largest velocity/rate MAE | largest surface MAE rad |
|---|---|---|---:|---:|---:|---:|
| E1.0 | throw_002 | sim real decisions | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E1.0 | throw_002 | sim own governor | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E1.0 | throw_010 | sim real decisions | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E1.0 | throw_010 | sim own governor | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E1.1 | throw_005 | sim real decisions | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E1.1 | throw_005 | sim own governor | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E1.1 | throw_011 | sim real decisions | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E1.1 | throw_011 | sim own governor | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E1.2 | throw_002 | sim real decisions | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E1.2 | throw_002 | sim own governor | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E1.2 | throw_029 | sim real decisions | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E1.2 | throw_029 | sim own governor | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Execution Timing Audit

| case | throw | launch speed m/s | first active lag ms | max decision ms | p95 decision ms | max commit lag ms | >20 ms lags | >50 ms lags | runtime late decisions |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E1.0 | throw_002 | 5.395 | 0.87 | nan | nan | nan | 0 | 0 | 0 |
| E1.0 | throw_010 | 5.319 | 0.50 | nan | nan | nan | 0 | 0 | 0 |
| E1.1 | throw_005 | 6.330 | 3.23 | 1.85 | 1.85 | 30.69 | 1 | 0 | 0 |
| E1.1 | throw_011 | 6.801 | 2.49 | 1.89 | 1.72 | 2.84 | 0 | 0 | 0 |
| E1.2 | throw_002 | 5.431 | 12.37 | 57.04 | 50.30 | 36.98 | 2 | 0 | 0 |
| E1.2 | throw_029 | 5.698 | 31.58 | 32.32 | 31.53 | 19.42 | 0 | 0 | 0 |
