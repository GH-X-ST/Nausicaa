# Real-Flight Replay Comparison Figures

These figures compare measured real-flight launches against two dry-air replay models:

- uncalibrated theory replay: comparison-only pure theory/geometry baseline
- active calibrated replay: current neutral residual-calibrated model with conservative alpha-regime scheduled aileron/elevator/rudder effectiveness

- active calibration: `neutral_dry_air_residual_calibrated_replay_n30_joint_pareto_040_local_s5_yaw0p75_clr0p60_surface_schedule_v3p2_cons_nominal`
- surface-effectiveness model: `alpha_regime_scheduled_v1`
- representative selection source: `C:/Users/GH-X-ST/OneDrive - Imperial College London/Year 4/Final Year Project/01 - Github/Nausicaa/03_Control/A_figures/real_flight_replay_comparison/metrics/real_flight_replay_comparison_summary.csv`
- replay dt: `0.005` s
- workers: `8`

## Figures

- `neutral`: `C:/Users/GH-X-ST/OneDrive - Imperial College London/Year 4/Final Year Project/01 - Github/Nausicaa/03_Control/A_figures/real_flight_replay_comparison/figures/real_vs_replay_neutral.png` (command `0.0`, launch confidence `0.9731638867171145`)
- `neutral_random_good_01`: `C:/Users/GH-X-ST/OneDrive - Imperial College London/Year 4/Final Year Project/01 - Github/Nausicaa/03_Control/A_figures/real_flight_replay_comparison/figures/real_vs_replay_neutral_random_good_01.png` (command `0.0`, launch confidence `0.8527007392351884`)
- `neutral_random_good_02`: `C:/Users/GH-X-ST/OneDrive - Imperial College London/Year 4/Final Year Project/01 - Github/Nausicaa/03_Control/A_figures/real_flight_replay_comparison/figures/real_vs_replay_neutral_random_good_02.png` (command `0.0`, launch confidence `0.8697314088504883`)
- `neutral_random_good_03`: `C:/Users/GH-X-ST/OneDrive - Imperial College London/Year 4/Final Year Project/01 - Github/Nausicaa/03_Control/A_figures/real_flight_replay_comparison/figures/real_vs_replay_neutral_random_good_03.png` (command `0.0`, launch confidence `0.8744900618946082`)
- `max_elevator_neg`: `C:/Users/GH-X-ST/OneDrive - Imperial College London/Year 4/Final Year Project/01 - Github/Nausicaa/03_Control/A_figures/real_flight_replay_comparison/figures/real_vs_replay_max_elevator_neg.png` (command `-1.0`, launch confidence `0.9561763409`)
- `max_elevator_pos`: `C:/Users/GH-X-ST/OneDrive - Imperial College London/Year 4/Final Year Project/01 - Github/Nausicaa/03_Control/A_figures/real_flight_replay_comparison/figures/real_vs_replay_max_elevator_pos.png` (command `1.0`, launch confidence `0.8957321551`)
- `max_rudder_neg`: `C:/Users/GH-X-ST/OneDrive - Imperial College London/Year 4/Final Year Project/01 - Github/Nausicaa/03_Control/A_figures/real_flight_replay_comparison/figures/real_vs_replay_max_rudder_neg.png` (command `-1.0`, launch confidence `0.9614109862`)
- `max_rudder_pos`: `C:/Users/GH-X-ST/OneDrive - Imperial College London/Year 4/Final Year Project/01 - Github/Nausicaa/03_Control/A_figures/real_flight_replay_comparison/figures/real_vs_replay_max_rudder_pos.png` (command `1.0`, launch confidence `0.9486813838`)
- `max_aileron_neg`: `C:/Users/GH-X-ST/OneDrive - Imperial College London/Year 4/Final Year Project/01 - Github/Nausicaa/03_Control/A_figures/real_flight_replay_comparison/figures/real_vs_replay_max_aileron_neg.png` (command `-1.0`, launch confidence `0.8009126981`)
- `max_aileron_pos`: `C:/Users/GH-X-ST/OneDrive - Imperial College London/Year 4/Final Year Project/01 - Github/Nausicaa/03_Control/A_figures/real_flight_replay_comparison/figures/real_vs_replay_max_aileron_pos.png` (command `1.0`, launch confidence `0.8661229736`)
