# v5.20 Transition-Aware Dense Pipeline Report

- Status: `complete`
- Blocked reason: ``
- Project title version: `LQR-Stabilised Contextual Primitive v5.20`
- Docs hashes: `{"docs/Daily_Schedule.txt":"2045f81dbe411564b80b6251f192226d5476d2c0f04fea36d716d8ff264a893f","docs/Glider_Control_Project_Plan.md":"4fc39088f51fa89669a640221233aa19277a73b131387e13decbb31d5bddd0de","docs/MATLAB Coding.txt":"003448f2217a3f300366d6827fd3af54bf8bd95e8d2dba2f641a9ce2a1057642","docs/PR.txt":"22adbc30d0648825748df8db1ac192840babab291c6023cccec87639becfa6be","docs/Python Coding Instruction.txt":"735ab9c679f7fbd57844370ab592a723f352d9851bf8ccc70ed08c04b13d1bf8","docs/Python Coding to CODEX.txt":"578633517e31d972ec8dfec75fcea797c427fd3721338a7b2296e2c9bfcc8262","docs/Skills.md":"bae3d53d6cc52ff744ea7cb6bd93bc0321e6ae47646e2f4a273f408fcbf7b044","docs/housekeeping_and_naming_rules.md":"82012d4b874fe8a30428c9e851213cb7bc12da72777e6adaf594090a79cda4b1"}`
- R5 decision: `R5_TRANSITION_AWARE_DENSE_INCOMPLETE_RESUME_REQUIRED`
- R5 run root: `03_Control/05_Results/R5_dense/B01_probe`
- R5 rows written: `0` / `102400`
- Worker/chunk/storage settings: workers `8`, chunk count `128`, chunk size `800`, storage `csv_gz`.
- Active primitive IDs: `["glide", "recovery", "lift_entry", "lift_dwell_arc", "mild_turn_left", "mild_turn_right", "energy_retaining_bank", "safe_exit_or_recovery_handoff"]`
- R6 is archived as diagnostic-only and is not an active gate in this pipeline.
- Active thesis workflow: R5 -> R7 -> R8 -> R10 -> R11 -> Reality; R9 is internal preflight/ablation only.
- R7-R8-R10-R11 deliberately not run when `--stop-after-stage R5` is selected.
- Claim boundary: simulation evidence only; no hardware readiness, real-flight transfer, mission success, full autonomy, or memory-improvement claim is made here.

Stages:
- `R5`: `dry_run_schedule` root `03_Control/05_Results/R5_dense/B01_probe`
- `R7`: `not_run` root ``
- `R8`: `not_run` root ``
- `R10`: `not_run` root ``
- `R11`: `not_run` root ``
