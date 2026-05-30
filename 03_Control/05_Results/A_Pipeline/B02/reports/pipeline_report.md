# v5.20 Transition-Aware Dense Pipeline Report

- Status: `blocked`
- Blocked reason: `docs_changed_reaudit_required:hash_changed:docs/Daily_Schedule.txt;byte_count_changed:docs/Daily_Schedule.txt;hash_changed:docs/Glider_Control_Project_Plan.md;byte_count_changed:docs/Glider_Control_Project_Plan.md;hash_changed:docs/MATLAB Coding.txt;byte_count_changed:docs/MATLAB Coding.txt;hash_changed:docs/PR.txt;byte_count_changed:docs/PR.txt;hash_changed:docs/Python Coding Instruction.txt;byte_count_changed:docs/Python Coding Instruction.txt;hash_changed:docs/Python Coding to CODEX.txt;byte_count_changed:docs/Python Coding to CODEX.txt;hash_changed:docs/Skills.md;byte_count_changed:docs/Skills.md;hash_changed:docs/housekeeping_and_naming_rules.md;byte_count_changed:docs/housekeeping_and_naming_rules.md`
- Project title version: `LQR-Stabilised Contextual Primitive v5.20`
- Docs hashes: `{"docs/Daily_Schedule.txt":"dc89d757201079c136747d881dd8a26e412b3301f55a161549fc9086da54d426","docs/Glider_Control_Project_Plan.md":"2bcd9269c72228fc7d4a0ee7b28f159ff6e5c3fed8a25e7d25987929c0883c70","docs/MATLAB Coding.txt":"bff4c54ea9c9fec5769e081d233bae75dc56fe715c4759949a691aca54c0df65","docs/PR.txt":"30d0c2ad84a0a1c2eacff712bd2f0681164141e867a6c589a0d6bbec811758bf","docs/Python Coding Instruction.txt":"6f89f1726a4d04a3de413d7391aeeed084b3e07e3d3ec494e3862e00d312d8fe","docs/Python Coding to CODEX.txt":"aec91de7b13631b6ada1d065087761cbae345ae0e86caf6a0762d100ed40d480","docs/Skills.md":"854aacc1d8d1ddf33bebf31a96f7417065f603c1bd4f5aabf73f9cffbb10a0c7","docs/housekeeping_and_naming_rules.md":"26c8b1cd763008557ab40b12e92977ed5204535f254428913d53ca7c28acda4c"}`
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
