# Real-Flight Simulation Replay Figures

- Figure run version: `real_flight_sim_replay_measured_fan_updraft_v2`
- Output root: `04_Flight_Test/A_figures/real_flight_sim_replay_E2_random_samples`
- Library tier: `balanced_cluster`
- Replay dt (s): `0.0050`
- Real-decision timing: `logged`
- Replay environment: `04_Flight_Test/A_figures/real_flight_sim_replay_E2_random_samples/metrics/replay_environment_summary.csv`
- Replay environment policy: Each throw builds a W2 annular-GP wind field from measured fan_positions.csv xy positions, using nominal fan power/width and active masks; dry-air is used only when no visible fan position is available.
- First-window state audit: `04_Flight_Test/A_figures/real_flight_sim_replay_E2_random_samples/metrics/first_0p04_state_replay_error_summary.csv`
- Execution timing audit: `04_Flight_Test/A_figures/real_flight_sim_replay_E2_random_samples/metrics/execution_timing_audit.csv`

| case | session | throw | model | status | termination | duration s | final x m | final y m | final z m |
|---|---|---|---|---|---|---:|---:|---:|---:|
| E2.0 | 20260607_163303 | throw_001 | reality | measured | exit_gate_floor | 1.476 | 5.858 | 3.046 | 0.392 |
| E2.0 | 20260607_163303 | throw_001 | sim own governor | ok | exit_gate_front_wall | 1.305 | 6.603 | 1.748 | 0.739 |
| E2.0 | 20260607_163303 | throw_001 | sim real decisions | ok | inside_operational_region | 1.476 | 6.055 | 1.853 | 0.501 |
| E2.0 | 20260607_163303 | throw_002 | reality | measured | exit_gate_front_wall | 1.445 | 6.613 | 2.683 | 0.665 |
| E2.0 | 20260607_163303 | throw_002 | sim own governor | ok | exit_gate_front_wall | 1.045 | 6.609 | 1.773 | 1.079 |
| E2.0 | 20260607_163303 | throw_002 | sim real decisions | ok | exit_gate_front_wall | 1.080 | 6.600 | 1.823 | 1.236 |
| E2.1 | 20260607_165533 | throw_016 | reality | measured | exit_gate_front_wall | 1.284 | 6.603 | 1.775 | 0.705 |
| E2.1 | 20260607_165533 | throw_016 | sim own governor | ok | inside_operational_region | 1.284 | 5.998 | 2.179 | 1.266 |
| E2.1 | 20260607_165533 | throw_016 | sim real decisions | ok | inside_operational_region | 1.284 | 6.167 | 2.152 | 1.162 |
| E2.1 | 20260607_165533 | throw_027 | reality | measured | exit_gate_front_wall | 1.262 | 6.604 | 1.695 | 0.645 |
| E2.1 | 20260607_165533 | throw_027 | sim own governor | ok | exit_gate_front_wall | 1.025 | 6.608 | 2.000 | 0.994 |
| E2.1 | 20260607_165533 | throw_027 | sim real decisions | ok | exit_gate_front_wall | 1.030 | 6.621 | 1.941 | 0.767 |
| E2.2 | 20260607_173345 | throw_020 | reality | measured | exit_gate_front_wall | 1.192 | 6.630 | 2.280 | 0.824 |
| E2.2 | 20260607_173345 | throw_020 | sim own governor | ok | exit_gate_front_wall | 1.050 | 6.601 | 1.926 | 0.849 |
| E2.2 | 20260607_173345 | throw_020 | sim real decisions | ok | exit_gate_front_wall | 1.034 | 6.619 | 1.847 | 0.691 |
| E2.2 | 20260607_173345 | throw_030 | reality | measured | exit_gate_floor | 1.293 | 6.466 | 1.539 | 0.397 |
| E2.2 | 20260607_173345 | throw_030 | sim own governor | ok | inside_operational_region | 1.293 | 6.178 | 2.438 | 1.019 |
| E2.2 | 20260607_173345 | throw_030 | sim real decisions | ok | inside_operational_region | 1.293 | 6.103 | 2.473 | 1.100 |
| E2.2 | 20260607_175359 | throw_001 | reality | measured | exit_gate_floor | 1.322 | 6.588 | 2.251 | 0.394 |
| E2.2 | 20260607_175359 | throw_001 | sim own governor | ok | inside_operational_region | 1.322 | 6.552 | 1.825 | 0.554 |
| E2.2 | 20260607_175359 | throw_001 | sim real decisions | ok | inside_operational_region | 1.322 | 6.345 | 1.833 | 0.631 |
| E2.2 | 20260607_175359 | throw_007 | reality | measured | exit_gate_front_wall | 1.261 | 6.601 | 2.126 | 0.699 |
| E2.2 | 20260607_175359 | throw_007 | sim own governor | ok | exit_gate_front_wall | 1.090 | 6.610 | 2.063 | 1.011 |
| E2.2 | 20260607_175359 | throw_007 | sim real decisions | ok | exit_gate_front_wall | 1.076 | 6.612 | 1.994 | 0.874 |

## First 0.04 s State Residual Audit

| case | throw | model | largest position MAE m | largest attitude MAE deg | largest velocity/rate MAE | largest surface MAE rad |
|---|---|---|---:|---:|---:|---:|
| E2.0 | throw_001 | sim real decisions | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E2.0 | throw_001 | sim own governor | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E2.0 | throw_002 | sim real decisions | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E2.0 | throw_002 | sim own governor | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E2.1 | throw_016 | sim real decisions | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E2.1 | throw_016 | sim own governor | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E2.1 | throw_027 | sim real decisions | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E2.1 | throw_027 | sim own governor | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E2.2 | throw_001 | sim real decisions | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E2.2 | throw_001 | sim own governor | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E2.2 | throw_007 | sim real decisions | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E2.2 | throw_007 | sim own governor | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E2.2 | throw_020 | sim real decisions | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E2.2 | throw_020 | sim own governor | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E2.2 | throw_030 | sim real decisions | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E2.2 | throw_030 | sim own governor | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Execution Timing Audit

| case | throw | launch speed m/s | first active lag ms | max decision ms | p95 decision ms | max commit lag ms | >20 ms lags | >50 ms lags | runtime late decisions |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E2.0 | throw_001 | 6.077 | 0.78 | nan | nan | nan | 0 | 0 | 0 |
| E2.0 | throw_002 | 6.260 | 0.64 | nan | nan | nan | 0 | 0 | 0 |
| E2.1 | throw_016 | 5.947 | 3.15 | 1.81 | 1.76 | 50.27 | 2 | 1 | 0 |
| E2.1 | throw_027 | 6.020 | 1.34 | 2.20 | 2.05 | 18.83 | 0 | 0 | 0 |
| E2.2 | throw_020 | 5.910 | 5.91 | 37.53 | 35.22 | 18.57 | 0 | 0 | 0 |
| E2.2 | throw_030 | 5.765 | 87.96 | 86.95 | 64.16 | 32.42 | 1 | 0 | 0 |
| E2.2 | throw_001 | 5.737 | 35.74 | 4.17 | 3.76 | 83.72 | 1 | 1 | 0 |
| E2.2 | throw_007 | 5.824 | 18.18 | 23.46 | 19.88 | 9.24 | 0 | 0 | 0 |
