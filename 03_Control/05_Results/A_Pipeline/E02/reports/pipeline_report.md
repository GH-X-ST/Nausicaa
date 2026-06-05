# v5.20 Transition-Aware Dense Pipeline Report

- Status: `complete`
- Blocked reason: ``
- Project title version: `LQR-Stabilised Contextual Primitive v5.20`
- Docs hashes: `{"docs/Daily_Schedule.txt":"be26abb6081c6869eccd259d6f098d78e83f8387a5383e97306e9b3252cc87b0","docs/Glider_Control_Project_Plan.md":"d5dc303456d27423e6a5d0d18357bd17cda19d7d468162883fcc282fcb6c14b0","docs/MATLAB Coding.txt":"453ba0929d0cee2d83a1a9ec8d0ca4b025d5c01751e477120794c829a9a98e8e","docs/PR.txt":"d64839c4859a2d3b5ba8034dd1fe9ab9c9db8c5ce45297562e4449ff4dfbb07e","docs/Python Coding Instruction.txt":"88155276be9dfb9c62b8d8bd84108bedf6710c19cff8f4dea3ad974c28e29c00","docs/Python Coding to CODEX.txt":"6c719c7460532ebaa0ce4680937f41306448965b22103d9e90f72808079634b1","docs/Skills.md":"05be416bf2003270b7a1529924d1a68929bdcba7a96c7d51ca5ca761b6b934ad","docs/housekeeping_and_naming_rules.md":"fcb56aebfd780b9a417e9ca8126cf7aaf6e7ddcf1ea8d7b8260bf03929d2bffa"}`
- R5 decision: `R5_TRANSITION_AWARE_DENSE_PASSED_FOR_REVIEW`
- R5 run root: `03_Control/05_Results/R5_dense/E02`
- R5 rows written: `102400` / `102400`
- Worker/chunk/storage settings: workers `8`, chunk count `128`, chunk size `800`, storage `csv_gz`.
- Active primitive IDs: `["glide", "recovery", "lift_entry", "lift_dwell_arc", "mild_turn_left", "mild_turn_right", "energy_retaining_bank", "safe_exit_or_recovery_handoff"]`
- R6 is archived as diagnostic-only and is not an active gate in this pipeline.
- Active thesis workflow: R5 -> R7 -> R8 -> R10 -> R11 -> Reality; R9 is internal preflight/ablation only.
- R7-R8-R10-R11 deliberately not run when `--stop-after-stage R5` is selected.
- Claim boundary: simulation evidence only; no hardware readiness, real-flight transfer, mission success, full autonomy, or memory-improvement claim is made here.

Stages:
- `R5`: `complete` root `03_Control/05_Results/R5_dense/E02`
- `R7`: `complete` root `03_Control/05_Results/R7_survival/E02`
- `R8`: `complete` root `03_Control/05_Results/R8_library_size_study/E02`
- `R10`: `complete` root `03_Control/05_Results/R10_learn/E02`
- `R11`: `not_run` root ``
