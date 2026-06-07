# Real-Flight Simulation Replay Figures

- Figure run version: `real_flight_sim_replay_measured_fan_updraft_v2`
- Output root: `04_Flight_Test/A_figures/real_flight_sim_replay_E2_updraft_smoke`
- Library tier: `balanced_cluster`
- Replay dt (s): `0.0050`
- Real-decision timing: `logged`
- Replay environment: `04_Flight_Test/A_figures/real_flight_sim_replay_E2_updraft_smoke/metrics/replay_environment_summary.csv`
- Replay environment policy: Each throw builds a W2 annular-GP wind field from measured fan_positions.csv xy positions, using nominal fan power/width and active masks; dry-air is used only when no visible fan position is available.
- First-window state audit: `04_Flight_Test/A_figures/real_flight_sim_replay_E2_updraft_smoke/metrics/first_0p04_state_replay_error_summary.csv`
- Execution timing audit: `04_Flight_Test/A_figures/real_flight_sim_replay_E2_updraft_smoke/metrics/execution_timing_audit.csv`

| case | session | throw | model | status | termination | duration s | final x m | final y m | final z m |
|---|---|---|---|---|---|---:|---:|---:|---:|
| E2.1 | 20260607_165533 | throw_016 | reality | measured | exit_gate_front_wall | 1.284 | 6.603 | 1.775 | 0.705 |
| E2.1 | 20260607_165533 | throw_016 | sim own governor | ok | inside_operational_region | 1.284 | 5.998 | 2.179 | 1.266 |
| E2.1 | 20260607_165533 | throw_016 | sim real decisions | ok | inside_operational_region | 1.284 | 6.167 | 2.152 | 1.162 |

## First 0.04 s State Residual Audit

| case | throw | model | largest position MAE m | largest attitude MAE deg | largest velocity/rate MAE | largest surface MAE rad |
|---|---|---|---:|---:|---:|---:|
| E2.1 | throw_016 | sim real decisions | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E2.1 | throw_016 | sim own governor | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Execution Timing Audit

| case | throw | launch speed m/s | first active lag ms | max decision ms | p95 decision ms | max commit lag ms | >20 ms lags | >50 ms lags | runtime late decisions |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E2.1 | throw_016 | 5.947 | 3.15 | 1.81 | 1.76 | 50.27 | 2 | 1 | 0 |
