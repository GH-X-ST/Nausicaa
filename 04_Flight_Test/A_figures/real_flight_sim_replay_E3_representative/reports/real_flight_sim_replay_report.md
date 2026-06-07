# Real-Flight Simulation Replay Figures

- Figure run version: `real_flight_sim_replay_measured_fan_updraft_v2`
- Output root: `04_Flight_Test/A_figures/real_flight_sim_replay_E3_representative`
- Library tier: `balanced_cluster`
- Replay dt (s): `0.0050`
- Real-decision timing: `logged`
- Replay environment: `04_Flight_Test/A_figures/real_flight_sim_replay_E3_representative/metrics/replay_environment_summary.csv`
- Replay environment policy: Each throw builds a W2 annular-GP wind field from measured fan_positions.csv xy positions, using nominal fan power/width and active masks; dry-air is used only when no visible fan position is available.
- First-window state audit: `04_Flight_Test/A_figures/real_flight_sim_replay_E3_representative/metrics/first_0p04_state_replay_error_summary.csv`
- Execution timing audit: `04_Flight_Test/A_figures/real_flight_sim_replay_E3_representative/metrics/execution_timing_audit.csv`

| case | session | throw | model | status | termination | duration s | final x m | final y m | final z m |
|---|---|---|---|---|---|---:|---:|---:|---:|
| E3.0 | 20260607_202556 | throw_001 | reality | measured | exit_gate_floor | 1.214 | 5.442 | 2.059 | 0.399 |
| E3.0 | 20260607_202556 | throw_001 | sim own governor | ok | exit_gate_front_wall | 1.085 | 6.601 | 2.680 | 1.192 |
| E3.0 | 20260607_202556 | throw_001 | sim real decisions | ok | inside_operational_region | 1.214 | 6.545 | 2.951 | 1.249 |
| E3.0 | 20260607_202556 | throw_008 | reality | measured | exit_gate_floor | 1.291 | 5.715 | 2.565 | 0.383 |
| E3.0 | 20260607_202556 | throw_008 | sim own governor | ok | inside_operational_region | 1.291 | 6.557 | 2.189 | 1.441 |
| E3.0 | 20260607_202556 | throw_008 | sim real decisions | ok | inside_operational_region | 1.291 | 5.591 | 2.329 | 1.565 |
| E3.1 | 20260607_204604 | throw_009 | reality | measured | exit_gate_front_wall | 1.236 | 6.607 | 2.802 | 1.289 |
| E3.1 | 20260607_204604 | throw_009 | sim own governor | ok | exit_gate_front_wall | 1.210 | 6.601 | 2.115 | 1.716 |
| E3.1 | 20260607_204604 | throw_009 | sim real decisions | ok | exit_gate_front_wall | 1.212 | 6.605 | 2.119 | 1.716 |
| E3.1 | 20260607_204604 | throw_023 | reality | measured | exit_gate_floor | 1.525 | 5.730 | 2.516 | 0.384 |
| E3.1 | 20260607_204604 | throw_023 | sim own governor | ok | exit_gate_front_wall | 1.060 | 6.611 | 2.678 | 1.604 |
| E3.1 | 20260607_204604 | throw_023 | sim real decisions | ok | exit_gate_front_wall | 1.039 | 6.620 | 2.554 | 1.430 |
| E3.2 | 20260607_213312 | throw_013 | reality | measured | exit_gate_floor | 1.457 | 6.511 | 1.595 | 0.379 |
| E3.2 | 20260607_213312 | throw_013 | sim own governor | ok | exit_gate_front_wall | 1.210 | 6.602 | 2.134 | 1.419 |
| E3.2 | 20260607_213312 | throw_013 | sim real decisions | ok | exit_gate_front_wall | 1.199 | 6.602 | 2.143 | 1.474 |
| E3.2 | 20260607_214908 | throw_016 | reality | measured | exit_gate_front_wall | 1.138 | 6.602 | 2.702 | 1.165 |
| E3.2 | 20260607_214908 | throw_016 | sim own governor | ok | exit_gate_front_wall | 0.940 | 6.605 | 2.258 | 1.660 |
| E3.2 | 20260607_214908 | throw_016 | sim real decisions | ok | exit_gate_front_wall | 0.960 | 6.614 | 2.298 | 1.742 |

## First 0.04 s State Residual Audit

| case | throw | model | largest position MAE m | largest attitude MAE deg | largest velocity/rate MAE | largest surface MAE rad |
|---|---|---|---:|---:|---:|---:|
| E3.0 | throw_001 | sim real decisions | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E3.0 | throw_001 | sim own governor | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E3.0 | throw_008 | sim real decisions | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E3.0 | throw_008 | sim own governor | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E3.1 | throw_009 | sim real decisions | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E3.1 | throw_009 | sim own governor | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E3.1 | throw_023 | sim real decisions | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E3.1 | throw_023 | sim own governor | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E3.2 | throw_013 | sim real decisions | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E3.2 | throw_013 | sim own governor | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E3.2 | throw_016 | sim real decisions | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E3.2 | throw_016 | sim own governor | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Execution Timing Audit

| case | throw | launch speed m/s | first active lag ms | max decision ms | p95 decision ms | max commit lag ms | >20 ms lags | >50 ms lags | runtime late decisions |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E3.0 | throw_001 | 6.209 | 0.75 | nan | nan | nan | 0 | 0 | 0 |
| E3.0 | throw_008 | 6.335 | 0.64 | nan | nan | nan | 0 | 0 | 0 |
| E3.1 | throw_009 | 6.653 | 43.26 | 2.20 | 1.88 | 6.52 | 0 | 0 | 0 |
| E3.1 | throw_023 | 6.372 | 3.72 | 5.06 | 2.99 | 28.69 | 1 | 0 | 0 |
| E3.2 | throw_013 | 6.496 | 22.75 | 23.57 | 22.27 | 14.84 | 0 | 0 | 0 |
| E3.2 | throw_016 | 6.757 | 2.28 | 37.51 | 31.50 | 37.08 | 1 | 0 | 0 |
