# Real-Flight Simulation Replay Figures

- Figure run version: `real_flight_sim_replay_measured_fan_updraft_v2`
- Output root: `04_Flight_Test/A_figures/real_flight_sim_replay_E4a_representative`
- Library tier: `balanced_cluster`
- Replay dt (s): `0.0050`
- Real-decision timing: `logged`
- Replay environment: `04_Flight_Test/A_figures/real_flight_sim_replay_E4a_representative/metrics/replay_environment_summary.csv`
- Replay environment policy: Each throw builds a W2 annular-GP wind field from measured fan_positions.csv xy positions, using nominal fan power/width and active masks; dry-air is used only when no visible fan position is available.
- First-window state audit: `04_Flight_Test/A_figures/real_flight_sim_replay_E4a_representative/metrics/first_0p04_state_replay_error_summary.csv`
- Execution timing audit: `04_Flight_Test/A_figures/real_flight_sim_replay_E4a_representative/metrics/execution_timing_audit.csv`

| case | session | throw | model | status | termination | duration s | final x m | final y m | final z m |
|---|---|---|---|---|---|---:|---:|---:|---:|
| E4a.0 | 20260607_230250 | throw_004 | reality | measured | exit_gate_floor | 1.698 | 6.335 | 4.180 | 0.093 |
| E4a.0 | 20260607_230250 | throw_004 | sim own governor | ok | exit_gate_front_wall | 1.230 | 6.613 | 2.185 | 0.627 |
| E4a.0 | 20260607_230250 | throw_004 | sim real decisions | ok | exit_gate_floor | 1.335 | 6.219 | 2.335 | 0.395 |
| E4a.0 | 20260607_230250 | throw_006 | reality | measured | exit_gate_front_wall | 1.621 | 6.609 | 1.491 | 0.946 |
| E4a.0 | 20260607_230250 | throw_006 | sim own governor | ok | exit_gate_front_wall | 1.095 | 6.612 | 2.190 | 0.900 |
| E4a.0 | 20260607_230250 | throw_006 | sim real decisions | ok | exit_gate_front_wall | 1.105 | 6.617 | 2.399 | 0.930 |
| E4a.1 | 20260607_224440 | throw_004 | reality | measured | exit_gate_front_wall | 1.250 | 6.614 | 2.855 | 1.561 |
| E4a.1 | 20260607_224440 | throw_004 | sim own governor | ok | inside_operational_region | 1.250 | 6.417 | 2.354 | 0.658 |
| E4a.1 | 20260607_224440 | throw_004 | sim real decisions | ok | inside_operational_region | 1.250 | 6.412 | 2.349 | 0.657 |
| E4a.1 | 20260607_224440 | throw_010 | reality | measured | exit_gate_floor | 1.699 | 5.572 | 4.347 | 0.397 |
| E4a.1 | 20260607_224440 | throw_010 | sim own governor | ok | exit_gate_front_wall | 1.230 | 6.611 | 1.775 | 0.499 |
| E4a.1 | 20260607_224440 | throw_010 | sim real decisions | ok | exit_gate_front_wall | 1.235 | 6.618 | 1.773 | 0.489 |
| E4a.2 | 20260607_231704 | throw_024 | reality | measured | exit_gate_floor | 1.845 | 6.308 | 4.215 | 0.338 |
| E4a.2 | 20260607_231704 | throw_024 | sim own governor | ok | exit_gate_front_wall | 1.275 | 6.604 | 2.000 | 0.436 |
| E4a.2 | 20260607_231704 | throw_024 | sim real decisions | ok | exit_gate_front_wall | 1.273 | 6.606 | 2.063 | 0.421 |
| E4a.2 | 20260607_231704 | throw_027 | reality | measured | exit_gate_front_wall | 1.422 | 6.601 | 2.440 | 1.289 |
| E4a.2 | 20260607_231704 | throw_027 | sim own governor | ok | exit_gate_floor | 1.265 | 6.490 | 2.368 | 0.390 |
| E4a.2 | 20260607_231704 | throw_027 | sim real decisions | ok | exit_gate_floor | 1.289 | 6.332 | 2.377 | 0.394 |

## First 0.04 s State Residual Audit

| case | throw | model | largest position MAE m | largest attitude MAE deg | largest velocity/rate MAE | largest surface MAE rad |
|---|---|---|---:|---:|---:|---:|
| E4a.0 | throw_004 | sim real decisions | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E4a.0 | throw_004 | sim own governor | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E4a.0 | throw_006 | sim real decisions | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E4a.0 | throw_006 | sim own governor | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E4a.1 | throw_004 | sim real decisions | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E4a.1 | throw_004 | sim own governor | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E4a.1 | throw_010 | sim real decisions | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E4a.1 | throw_010 | sim own governor | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E4a.2 | throw_024 | sim real decisions | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E4a.2 | throw_024 | sim own governor | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E4a.2 | throw_027 | sim real decisions | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E4a.2 | throw_027 | sim own governor | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Execution Timing Audit

| case | throw | launch speed m/s | first active lag ms | max decision ms | p95 decision ms | max commit lag ms | >20 ms lags | >50 ms lags | runtime late decisions |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E4a.0 | throw_004 | 5.969 | 0.60 | nan | nan | nan | 0 | 0 | 0 |
| E4a.0 | throw_006 | 5.876 | 0.78 | nan | nan | nan | 0 | 0 | 0 |
| E4a.1 | throw_004 | 6.161 | 2.62 | 1.64 | 1.58 | 40.00 | 2 | 0 | 0 |
| E4a.1 | throw_010 | 6.030 | 1.60 | 3.13 | 2.69 | 14.66 | 0 | 0 | 0 |
| E4a.2 | throw_024 | 5.416 | 20.49 | 28.98 | 27.35 | 40.20 | 1 | 0 | 0 |
| E4a.2 | throw_027 | 5.584 | 28.41 | 30.38 | 28.26 | 30.86 | 1 | 0 | 0 |
