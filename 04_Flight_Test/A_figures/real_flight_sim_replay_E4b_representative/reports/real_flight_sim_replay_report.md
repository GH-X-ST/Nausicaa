# Real-Flight Simulation Replay Figures

- Figure run version: `real_flight_sim_replay_measured_fan_updraft_v2`
- Output root: `04_Flight_Test/A_figures/real_flight_sim_replay_E4b_representative`
- Library tier: `balanced_cluster`
- Replay dt (s): `0.0050`
- Real-decision timing: `logged`
- Replay environment: `04_Flight_Test/A_figures/real_flight_sim_replay_E4b_representative/metrics/replay_environment_summary.csv`
- Replay environment policy: Each throw builds a W2 annular-GP wind field from measured fan_positions.csv xy positions, using nominal fan power/width and active masks; dry-air is used only when no visible fan position is available.
- First-window state audit: `04_Flight_Test/A_figures/real_flight_sim_replay_E4b_representative/metrics/first_0p04_state_replay_error_summary.csv`
- Execution timing audit: `04_Flight_Test/A_figures/real_flight_sim_replay_E4b_representative/metrics/execution_timing_audit.csv`

| case | session | throw | model | status | termination | duration s | final x m | final y m | final z m |
|---|---|---|---|---|---|---:|---:|---:|---:|
| E4b.0 | 20260608_003841 | throw_004 | reality | measured | exit_gate_y_min | 1.539 | 5.858 | -0.006 | 0.935 |
| E4b.0 | 20260608_003841 | throw_004 | sim own governor | ok | exit_gate_front_wall | 1.225 | 6.608 | 2.462 | 1.012 |
| E4b.0 | 20260608_003841 | throw_004 | sim real decisions | ok | exit_gate_front_wall | 1.315 | 6.615 | 2.824 | 0.955 |
| E4b.0 | 20260608_003841 | throw_010 | reality | measured | exit_gate_front_wall | 1.519 | 6.604 | 1.427 | 1.677 |
| E4b.0 | 20260608_003841 | throw_010 | sim own governor | ok | exit_gate_front_wall | 1.110 | 6.606 | 2.731 | 1.657 |
| E4b.0 | 20260608_003841 | throw_010 | sim real decisions | ok | exit_gate_front_wall | 1.200 | 6.601 | 3.148 | 1.591 |
| E4b.1 | 20260608_004535 | throw_013 | reality | measured | exit_gate_front_wall | 1.131 | 6.605 | 2.480 | 1.668 |
| E4b.1 | 20260608_004535 | throw_013 | sim own governor | ok | exit_gate_front_wall | 1.095 | 6.612 | 2.908 | 1.446 |
| E4b.1 | 20260608_004535 | throw_013 | sim real decisions | ok | exit_gate_front_wall | 1.031 | 6.619 | 2.761 | 1.260 |
| E4b.1 | 20260608_010525 | throw_008 | reality | measured | exit_gate_front_wall | 1.353 | 6.610 | 2.913 | 1.522 |
| E4b.1 | 20260608_010525 | throw_008 | sim own governor | ok | exit_gate_front_wall | 1.045 | 6.619 | 2.384 | 1.114 |
| E4b.1 | 20260608_010525 | throw_008 | sim real decisions | ok | exit_gate_front_wall | 1.046 | 6.616 | 2.369 | 1.079 |
| E4b.2 | 20260608_013501 | throw_001 | reality | measured | exit_gate_floor | 1.856 | 6.156 | 1.535 | 0.361 |
| E4b.2 | 20260608_013501 | throw_001 | sim own governor | ok | exit_gate_front_wall | 1.385 | 6.619 | 1.929 | 0.432 |
| E4b.2 | 20260608_013501 | throw_001 | sim real decisions | ok | exit_gate_front_wall | 1.365 | 6.618 | 1.904 | 0.443 |
| E4b.2 | 20260608_013501 | throw_020 | reality | measured | exit_gate_front_wall | 1.393 | 6.613 | 0.668 | 0.972 |
| E4b.2 | 20260608_013501 | throw_020 | sim own governor | ok | exit_gate_front_wall | 1.035 | 6.605 | 2.406 | 0.979 |
| E4b.2 | 20260608_013501 | throw_020 | sim real decisions | ok | exit_gate_front_wall | 1.029 | 6.618 | 2.305 | 0.839 |

## First 0.04 s State Residual Audit

| case | throw | model | largest position MAE m | largest attitude MAE deg | largest velocity/rate MAE | largest surface MAE rad |
|---|---|---|---:|---:|---:|---:|
| E4b.0 | throw_004 | sim real decisions | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E4b.0 | throw_004 | sim own governor | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E4b.0 | throw_010 | sim real decisions | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E4b.0 | throw_010 | sim own governor | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E4b.1 | throw_008 | sim real decisions | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E4b.1 | throw_008 | sim own governor | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E4b.1 | throw_013 | sim real decisions | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E4b.1 | throw_013 | sim own governor | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E4b.2 | throw_001 | sim real decisions | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E4b.2 | throw_001 | sim own governor | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E4b.2 | throw_020 | sim real decisions | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E4b.2 | throw_020 | sim own governor | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Execution Timing Audit

| case | throw | launch speed m/s | first active lag ms | max decision ms | p95 decision ms | max commit lag ms | >20 ms lags | >50 ms lags | runtime late decisions |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E4b.0 | throw_004 | 5.761 | 26.47 | nan | nan | nan | 0 | 0 | 0 |
| E4b.0 | throw_010 | 6.350 | 0.92 | nan | nan | nan | 0 | 0 | 0 |
| E4b.1 | throw_013 | 6.220 | 3.18 | 4.71 | 3.84 | 69.46 | 2 | 1 | 0 |
| E4b.1 | throw_008 | 5.762 | 83.72 | 2.88 | 2.46 | 62.16 | 4 | 1 | 0 |
| E4b.2 | throw_001 | 4.900 | 2.30 | 4.83 | 3.67 | 78.13 | 3 | 1 | 0 |
| E4b.2 | throw_020 | 5.724 | 7.76 | 23.92 | 22.17 | 10.49 | 0 | 0 | 0 |
