# v5.20 Transition-Aware Dense Pipeline Report

- Status: `complete`
- Blocked reason: ``
- Project title version: `LQR-Stabilised Contextual Primitive v5.20`
- Docs hashes: `{"docs/Daily_Schedule.txt":"98306dc7c3bc09e046cc132a63c9a3fb3d7e9d4de0a47923d1b2a6eaa7129861","docs/Glider_Control_Project_Plan.md":"859465e5af068b878d3a819fa060397ecc04fd0ab886b28f3a17036806011b01","docs/MATLAB Coding.txt":"4c4ed2ae637a1475c08d58e565142be2dbb3c3d0e346000473bb590ab6545ed7","docs/PR.txt":"eed476ef516391eb06fc9d28f2051dcab057bb09e48f0583f27ae836ec3f745a","docs/Python Coding Instruction.txt":"dbeeaa50e6b973122d4d5be19068f95855c8cc895e16555bb9d15fd1b4179ca2","docs/Python Coding to CODEX.txt":"d796fd28f612aaded89fdf4a524b3f619436984eafd52688c4c10d164f3371da","docs/Skills.md":"b613b7c68bfdaf8116e68b8fdc6196e7ec19de76ab9dbd9142a116b76445322b","docs/housekeeping_and_naming_rules.md":"6d69adca8d2247874ff34bbc238b2ecce16292608cdee61f284f3893003b2984"}`
- R5 decision: `R5_TRANSITION_AWARE_DENSE_PASSED_FOR_REVIEW`
- R5 run root: `03_Control/05_Results/R5_dense/E01`
- R5 rows written: `102400` / `102400`
- Worker/chunk/storage settings: workers `8`, chunk count `128`, chunk size `800`, storage `csv_gz`.
- Active primitive IDs: `["glide", "recovery", "lift_entry", "lift_dwell_arc", "mild_turn_left", "mild_turn_right", "energy_retaining_bank", "safe_exit_or_recovery_handoff"]`
- R6 is archived as diagnostic-only and is not an active gate in this pipeline.
- Active thesis workflow: R5 -> R7 -> R8 -> R10 -> R11 -> Reality; R9 is internal preflight/ablation only.
- R7-R8-R10-R11 deliberately not run when `--stop-after-stage R5` is selected.
- Claim boundary: simulation evidence only; no hardware readiness, real-flight transfer, mission success, full autonomy, or memory-improvement claim is made here.

Stages:
- `R5`: `complete` root `03_Control/05_Results/R5_dense/E01`
- `R7`: `complete` root `03_Control/05_Results/R7_survival/E01`
- `R8`: `complete` root `03_Control/05_Results/R8_library_size_study/E01`
- `R10`: `not_run` root ``
- `R11`: `not_run` root ``
