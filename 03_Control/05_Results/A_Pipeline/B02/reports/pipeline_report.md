# v5.20 Transition-Aware Dense Pipeline Report

- Status: `blocked`
- Blocked reason: `failed_gate:pass_gate_true,r11_validation_gate_hard_failure_rate_within_stage_profile,r11_validation_gate_floor_or_ceiling_violation_rate_within_stage_profile,r11_validation_gate_no_viable_primitive_rate_within_stage_profile,r11_validation_gate_safe_success_rate_within_stage_profile,r11_validation_gate_terminal_or_lift_capture_within_stage_profile,r11_validation_gate_full_safe_success_rate_within_stage_profile`
- Project title version: `LQR-Stabilised Contextual Primitive v5.20`
- Docs hashes: `{"docs/Daily_Schedule.txt":"e8f930c1618d7d6001ae6cf483a7320d3f1a573ec984a28a09b00050a7da3b51","docs/Glider_Control_Project_Plan.md":"58f7d191efbd480767233f8e1e37f62e5bc6396c2b888dc74be6d92b010d4ff8","docs/MATLAB Coding.txt":"78933a973aa89a6ccda7023fedfd496d42617d4d59834def0bb3df0999901520","docs/PR.txt":"43fc6db4273ccb6e3c9e6675063f4b657f21db21d26c3819e4ca15cc7f33c461","docs/Python Coding Instruction.txt":"387f08601111cb0812aaebb28e42a4b1246eaedc365dc893468c2fbab9e7aedb","docs/Python Coding to CODEX.txt":"dd59b35a4a5e5d71e2e77f70ec5f2a82ff0eb1d41faeeecd312c516a95275c5d","docs/Skills.md":"2c0860da5a54057d317797929fde6428aa3fea450f22ced93255cc00c5cb0ecb","docs/housekeeping_and_naming_rules.md":"155fb97fc139e6e1f14941447f5896d19810d75b80454700440eee0833a380f6"}`
- R5 decision: `R5_TRANSITION_AWARE_DENSE_PASSED_FOR_REVIEW`
- R5 run root: `03_Control/05_Results/R5_dense/B02`
- R5 rows written: `102400` / `102400`
- Worker/chunk/storage settings: workers `8`, chunk count `128`, chunk size `800`, storage `csv_gz`.
- Active primitive IDs: `["glide", "recovery", "lift_entry", "lift_dwell_arc", "mild_turn_left", "mild_turn_right", "energy_retaining_bank", "safe_exit_or_recovery_handoff"]`
- R6 is archived as diagnostic-only and is not an active gate in this pipeline.
- Active thesis workflow: R5 -> R7 -> R8 -> R10 -> R11 -> Reality; R9 is internal preflight/ablation only.
- R7-R8-R10-R11 deliberately not run when `--stop-after-stage R5` is selected.
- Claim boundary: simulation evidence only; no hardware readiness, real-flight transfer, mission success, full autonomy, or memory-improvement claim is made here.

Stages:
- `R5`: `complete` root `03_Control/05_Results/R5_dense/B02`
- `R7`: `complete` root `03_Control/05_Results/R7_survival/B02`
- `R8`: `complete` root `03_Control/05_Results/R8_library_size_study/B02`
- `R10`: `complete` root `03_Control/05_Results/R10_learn/B04`
- `R11`: `not_run` root ``
