# Phase 2 Overnight Latency/Recovery Report

Seed: `1`
Output root: `03_Control/05_Results/03_primitives/12_tight_turn_phase2_overnight/001`
Selected candidate variant: `baseline`
Selected candidate tag: `baseline_bank_yaw`
Selected feedback variant: `k_smooth3_r110`
phase2_status: `boundary_only`

## Gate Summary

- hard OCP 30 reproduced: `True`
- open-loop no latency: `True`
- closed-loop no latency: `True`
- open-loop nominal latency: `True`
- closed-loop nominal latency: `False`
- terminal recovery sensitivity: `False`
- active failure class: `latency_limited_high_alpha`
- all failure classes: `latency_limited_high_alpha;terminal_recovery_limited`
- latency mechanism diagnosis: `latency_limited_high_alpha`
- limitation: `angle of attack exceeded bound`

## Selected Metrics

- directed heading change deg: `28.293163149683984`
- max alpha deg: `4.8362953644871185`
- max beta deg: `25.143698311128077`
- terminal speed m/s: `6.08297857579008`
- wall margin m: `0.25`
- floor margin m: `2.5263011311490597`
- ceiling margin m: `0.2999999999999998`
- saturation fraction: `0.0`

## Metrics Paths

- `03_Control/05_Results/03_primitives/12_tight_turn_phase2_overnight/001/metrics/stage0_baseline_reproduction_s001.csv`
- `03_Control/05_Results/03_primitives/12_tight_turn_phase2_overnight/001/metrics/stage1_latency_ablation_s001.csv`
- `03_Control/05_Results/03_primitives/12_tight_turn_phase2_overnight/001/metrics/stage2_candidate_variants_s001.csv`
- `03_Control/05_Results/03_primitives/12_tight_turn_phase2_overnight/001/metrics/stage3_tvlqr_variants_s001.csv`
- `03_Control/05_Results/03_primitives/12_tight_turn_phase2_overnight/001/metrics/stage3_tvlqr_replay_s001.csv`
- `03_Control/05_Results/03_primitives/12_tight_turn_phase2_overnight/001/metrics/stage4_phase2_gate_s001.csv`

## Phase 3 Permission

Phase 3 continuation is not allowed because strict Phase 2 was not promoted.
