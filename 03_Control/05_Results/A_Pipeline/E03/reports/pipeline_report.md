# v5.20 Transition-Aware Dense Pipeline Report

- Status: `blocked_strict_gate_failed_physical_acceptance`
- Blocked reason: `failed_gate:pass_gate_true,r11_validation_gate_hard_failure_rate_within_stage_profile,r11_validation_gate_floor_or_ceiling_violation_rate_within_stage_profile,r11_validation_gate_no_viable_primitive_rate_within_stage_profile,r11_validation_gate_safe_success_rate_within_stage_profile,r11_validation_gate_terminal_or_lift_capture_within_stage_profile,r11_validation_gate_full_safe_success_rate_within_stage_profile`
- R11 acceptance override: strict R11 gate failed, but the E03.1 result is retained for reporting/freezing because behavior is physically explainable and closed-loop is not broadly worse than open-loop.
- Project title version: `LQR-Stabilised Contextual Primitive v5.20`
- Docs hashes: `{"docs/Daily_Schedule.txt":"71d30b3fe2e85305eb2f5fe99823a78aa0a52aa3f4978bac4314c11e3e94346c","docs/Glider_Control_Project_Plan.md":"54e8d21495e4eb89f7966f41d9a10ee3b5164986d9494f1e1b1267d015d608ed","docs/MATLAB Coding.txt":"30970f9cce25339f165db6af2e21c84439594904148ca017c38d03b16278e4ee","docs/PR.txt":"694150cb4199e5f58c4186f737e136002d3117e721fe51d830bf1e75c87feae0","docs/Python Coding Instruction.txt":"2f0fdccb61ec915e4c0fe0b2618e3f969b571713d286c98162477333df119b09","docs/Python Coding to CODEX.txt":"37dbe2bfe6256eec3413a8ec162999cda0e11a1b74d670b46c263e6a2cfab81e","docs/Skills.md":"554ea2696bfbaea51f6219ebdb3773ef38a7f7f8f4327ace2caa9f4938046ec9","docs/housekeeping_and_naming_rules.md":"9e5a7c5d6e1314f896779befc524bf5f95e528a0f2fafc234a74ce3b41ed1545"}`
- Active thesis workflow: R5 -> R7 -> R8 -> R10 -> R11 -> Reality; R9 is internal preflight/ablation only.
- Claim boundary: simulation evidence only; no hardware readiness, real-flight transfer, strict R11 pass, mission success, full autonomy, or memory-improvement claim is made here.

Stages:
- `R5`: `complete` root `03_Control/05_Results/R5_dense/E03`
- `R7`: `complete` root `03_Control/05_Results/R7_survival/E03`
- `R8`: `complete` root `03_Control/05_Results/R8_library_size_study/E03`
- `R10`: `complete` root `03_Control/05_Results/R10_learn/E03` pass_gate `True` final `1000` history `10750`
- `R11`: `blocked_strict_gate_failed_physical_acceptance` root `03_Control/05_Results/R11_validation/E03.1` pass_gate `False` final `4000` history `34400`
