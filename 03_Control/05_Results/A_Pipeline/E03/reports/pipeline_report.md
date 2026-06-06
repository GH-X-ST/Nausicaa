# v5.20 Transition-Aware Dense Pipeline Report

- Status: `blocked`
- Blocked reason: `failed_gate:pass_gate_true,r11_validation_gate_hard_failure_rate_within_stage_profile,r11_validation_gate_floor_or_ceiling_violation_rate_within_stage_profile,r11_validation_gate_no_viable_primitive_rate_within_stage_profile,r11_validation_gate_safe_success_rate_within_stage_profile,r11_validation_gate_terminal_or_lift_capture_within_stage_profile,r11_validation_gate_full_safe_success_rate_within_stage_profile`
- Project title version: `LQR-Stabilised Contextual Primitive v5.20`
- Docs hashes: `{"docs/Daily_Schedule.txt":"f10538e64a30084c12ea0e3c62b2aa1279582c5d36fc79b89ed8e318e5bb4498","docs/Glider_Control_Project_Plan.md":"47cd1471385d943f33d39eb008a25e7787ab64b90ddaeb243ba966f1620cfe70","docs/MATLAB Coding.txt":"70b13c5c22200f9326950f0dec481cf1bf765993a9f09f63a60e3f8a4f8872fa","docs/PR.txt":"8ad431782b2bcb0b175747f914680861781dd41e9910cae07aa8927652e4e611","docs/Python Coding Instruction.txt":"bb149ee07f00648a0b24eb683949ab0a8424886bd91f41d72fbc5b21cc8878de","docs/Python Coding to CODEX.txt":"ed165222f7ae52c0d0ac1d58ba3124082d3bf01eb4001593c76a11b81398e54f","docs/Skills.md":"2544a0d670495520c829d1862e1e3ed624f699a095b0c53d5f5dc81916d452c6","docs/housekeeping_and_naming_rules.md":"26240f9a516f1d3081568b86a098e53659ce617dc9ad08c80ed88bb389d9eb23"}`
- R5 decision: `R5_TRANSITION_AWARE_DENSE_PASSED_FOR_REVIEW`
- R5 run root: `03_Control/05_Results/R5_dense/E03`
- R5 rows written: `102400` / `102400`
- Worker/chunk/storage settings: workers `8`, chunk count `128`, chunk size `800`, storage `csv_gz`.
- Active primitive IDs: `["glide", "recovery", "lift_entry", "lift_dwell_arc", "mild_turn_left", "mild_turn_right", "energy_retaining_bank", "safe_exit_or_recovery_handoff"]`
- R6 is archived as diagnostic-only and is not an active gate in this pipeline.
- Active thesis workflow: R5 -> R7 -> R8 -> R10 -> R11 -> Reality; R9 is internal preflight/ablation only.
- R7-R8-R10-R11 deliberately not run when `--stop-after-stage R5` is selected.
- Claim boundary: simulation evidence only; no hardware readiness, real-flight transfer, mission success, full autonomy, or memory-improvement claim is made here.

Stages:
- `R5`: `complete` root `03_Control/05_Results/R5_dense/E03`
- `R7`: `complete` root `03_Control/05_Results/R7_survival/E03`
- `R8`: `complete` root `03_Control/05_Results/R8_library_size_study/E03`
- `R10`: `complete` root `03_Control/05_Results/R10_learn/E03`
- `R11`: `not_run` root ``
