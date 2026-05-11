# Reproducibility Commands Executed by CODEX

Start time: 2026-05-10 UTC during CODEX task execution.
Repository commit: `046f739`.

## Environment

### discovery

Command:

```bash
Get-Location
```
Exit status: `0`
Summary: repository root confirmed

### discovery

Command:

```bash
git status --short
```
Exit status: `0`
Summary: initial status showed A .gitignore

### discovery

Command:

```bash
git rev-parse --short HEAD
```
Exit status: `0`
Summary: 046f739

### discovery

Command:

```bash
python -c "import sys, platform; print(sys.executable); print(sys.version); print(platform.platform())"
```
Exit status: `0`
Summary: Python 3.12.11 on Windows 11

### discovery

Command:

```bash
python -m pip --version
```
Exit status: `0`
Summary: pip 26.1.1

### setup

Command:

```bash
python -m pip install -e ".[test]"
```
Exit status: `1`
Summary: no setup.py or pyproject.toml present

### setup

Command:

```bash
python -m pip install -e .
```
Exit status: `1`
Summary: no setup.py or pyproject.toml present

### setup

Command:

```bash
python -m pip install -r requirements-dev.txt
```
Exit status: `1`
Summary: requirements-dev.txt missing

### setup

Command:

```bash
python -m pip install -r requirements.txt
```
Exit status: `1`
Summary: requirements.txt contains pyproject-style TOML, not pip requirements

### validation

Command:

```bash
python -m compileall 03_Control tests
```
Exit status: `0`
Summary: compiled control modules and tests

### validation

Command:

```bash
python -m pytest -q tests/test_latency_envelope.py tests/test_primitive_no_nan.py tests/test_primitive_full_duration.py tests/test_governor_rollout_selection.py tests/test_metrics_schema.py tests/test_updraft_randomisation.py
```
Exit status: `0`
Summary: 7 passed

### validation

Command:

```bash
python -m pytest -q
```
Exit status: `0`
Summary: 21 passed

### scenario

Command:

```bash
python 03_Control/04_Scenarios/run_one.py --scenario s4_full_nominal_glide_no_wind --seed 1 --output-root 03_Control/05_Results/s4_execution
```
Exit status: `0`
Summary: scenario command exited 0 and produced metrics

### scenario

Command:

```bash
python 03_Control/04_Scenarios/run_one.py --scenario s4_full_bank_reversal_left_no_wind --seed 1 --output-root 03_Control/05_Results/s4_execution
```
Exit status: `0`
Summary: scenario command exited 0 and produced metrics

### scenario

Command:

```bash
python 03_Control/04_Scenarios/run_one.py --scenario s4_full_recovery_no_wind --seed 1 --output-root 03_Control/05_Results/s4_execution
```
Exit status: `0`
Summary: scenario command exited 0 and produced metrics

### scenario

Command:

```bash
python 03_Control/04_Scenarios/run_one.py --scenario s4_latency_low_bank_reversal_left --seed 1 --output-root 03_Control/05_Results/s4_execution
```
Exit status: `0`
Summary: scenario command exited 0 and produced metrics

### scenario

Command:

```bash
python 03_Control/04_Scenarios/run_one.py --scenario s4_latency_nominal_bank_reversal_left --seed 1 --output-root 03_Control/05_Results/s4_execution
```
Exit status: `0`
Summary: scenario command exited 0 and produced metrics

### scenario

Command:

```bash
python 03_Control/04_Scenarios/run_one.py --scenario s4_latency_high_bank_reversal_left --seed 1 --output-root 03_Control/05_Results/s4_execution
```
Exit status: `0`
Summary: scenario command exited 0 and produced metrics

### scenario

Command:

```bash
python 03_Control/04_Scenarios/run_one.py --scenario s4_gaussian_single_panel --seed 1 --output-root 03_Control/05_Results/s4_execution
```
Exit status: `0`
Summary: scenario command exited 0 and produced metrics

### scenario

Command:

```bash
python 03_Control/04_Scenarios/run_one.py --scenario s4_annular_single_panel --seed 1 --output-root 03_Control/05_Results/s4_execution
```
Exit status: `0`
Summary: scenario command exited 0 and produced metrics

### scenario

Command:

```bash
python 03_Control/04_Scenarios/run_one.py --scenario s4_governor_selection --seed 1 --output-root 03_Control/05_Results/s4_execution
```
Exit status: `0`
Summary: scenario command exited 0 and produced metrics

### audit

Command:

```bash
python 03_Control/03_Primitives/run_primitive_audit.py --seed 1 --output-root 03_Control/05_Results/s4_execution/audit
```
Exit status: `0`
Summary: 15 S4 audit scenarios reported success True after rerun

### batch

Command:

```bash
python 03_Control/04_Scenarios/run_batch.py --seed 42
```
Exit status: `0`
Summary: batch_seed42.csv produced

### replay

Command:

```bash
run_scenario("s4_gaussian_single_panel_randomised", seed=8) twice and compare metrics
```
Exit status: `0`
Summary: deterministic replay passed

### cleanup

Command:

```bash
git restore -- <generated files>
```
Exit status: `1`
Summary: .git/index.lock could not be created due sandbox/Git ACL; restored file contents via git show instead

## Generated Files
- `03_Control/05_Results/s4_execution/audit/logs/s4_annular_four_panel_seed1.csv` (11915 bytes)
- `03_Control/05_Results/s4_execution/audit/logs/s4_annular_single_cg_seed1.csv` (15443 bytes)
- `03_Control/05_Results/s4_execution/audit/logs/s4_annular_single_panel_seed1.csv` (15488 bytes)
- `03_Control/05_Results/s4_execution/audit/logs/s4_full_bank_reversal_left_no_wind_seed1.csv` (33961 bytes)
- `03_Control/05_Results/s4_execution/audit/logs/s4_full_bank_reversal_right_no_wind_seed1.csv` (34376 bytes)
- `03_Control/05_Results/s4_execution/audit/logs/s4_full_nominal_glide_no_wind_seed1.csv` (33059 bytes)
- `03_Control/05_Results/s4_execution/audit/logs/s4_full_recovery_no_wind_seed1.csv` (33703 bytes)
- `03_Control/05_Results/s4_execution/audit/logs/s4_gaussian_four_panel_seed1.csv` (11366 bytes)
- `03_Control/05_Results/s4_execution/audit/logs/s4_gaussian_single_cg_seed1.csv` (14691 bytes)
- `03_Control/05_Results/s4_execution/audit/logs/s4_gaussian_single_panel_randomised_seed1.csv` (18884 bytes)
- `03_Control/05_Results/s4_execution/audit/logs/s4_gaussian_single_panel_seed1.csv` (14723 bytes)
- `03_Control/05_Results/s4_execution/audit/logs/s4_governor_selection_seed1.csv` (19912 bytes)
- `03_Control/05_Results/s4_execution/audit/logs/s4_latency_high_bank_reversal_left_seed1.csv` (33783 bytes)
- `03_Control/05_Results/s4_execution/audit/logs/s4_latency_low_bank_reversal_left_seed1.csv` (33843 bytes)
- `03_Control/05_Results/s4_execution/audit/logs/s4_latency_nominal_bank_reversal_left_seed1.csv` (34093 bytes)
- `03_Control/05_Results/s4_execution/audit/metrics/s4_annular_four_panel_seed1.csv` (1150 bytes)
- `03_Control/05_Results/s4_execution/audit/metrics/s4_annular_single_cg_seed1.csv` (1132 bytes)
- `03_Control/05_Results/s4_execution/audit/metrics/s4_annular_single_panel_seed1.csv` (1133 bytes)
- `03_Control/05_Results/s4_execution/audit/metrics/s4_full_bank_reversal_left_no_wind_seed1.csv` (1078 bytes)
- `03_Control/05_Results/s4_execution/audit/metrics/s4_full_bank_reversal_right_no_wind_seed1.csv` (1081 bytes)
- `03_Control/05_Results/s4_execution/audit/metrics/s4_full_nominal_glide_no_wind_seed1.csv` (1075 bytes)
- `03_Control/05_Results/s4_execution/audit/metrics/s4_full_recovery_no_wind_seed1.csv` (1023 bytes)
- `03_Control/05_Results/s4_execution/audit/metrics/s4_gaussian_four_panel_seed1.csv` (1117 bytes)
- `03_Control/05_Results/s4_execution/audit/metrics/s4_gaussian_single_cg_seed1.csv` (1093 bytes)
- `03_Control/05_Results/s4_execution/audit/metrics/s4_gaussian_single_panel_randomised_seed1.csv` (1366 bytes)
- `03_Control/05_Results/s4_execution/audit/metrics/s4_gaussian_single_panel_seed1.csv` (1108 bytes)
- `03_Control/05_Results/s4_execution/audit/metrics/s4_governor_selection_seed1.csv` (1197 bytes)
- `03_Control/05_Results/s4_execution/audit/metrics/s4_governor_selection_seed1_governor_candidates.csv` (420 bytes)
- `03_Control/05_Results/s4_execution/audit/metrics/s4_governor_selection_seed1_governor_rejections.csv` (333 bytes)
- `03_Control/05_Results/s4_execution/audit/metrics/s4_latency_high_bank_reversal_left_seed1.csv` (1075 bytes)
- `03_Control/05_Results/s4_execution/audit/metrics/s4_latency_low_bank_reversal_left_seed1.csv` (1103 bytes)
- `03_Control/05_Results/s4_execution/audit/metrics/s4_latency_nominal_bank_reversal_left_seed1.csv` (1090 bytes)
- `03_Control/05_Results/s4_execution/logs/s4_annular_single_panel_seed1.csv` (15488 bytes)
- `03_Control/05_Results/s4_execution/logs/s4_full_bank_reversal_left_no_wind_seed1.csv` (33961 bytes)
- `03_Control/05_Results/s4_execution/logs/s4_full_nominal_glide_no_wind_seed1.csv` (33059 bytes)
- `03_Control/05_Results/s4_execution/logs/s4_full_recovery_no_wind_seed1.csv` (33703 bytes)
- `03_Control/05_Results/s4_execution/logs/s4_gaussian_single_panel_seed1.csv` (14723 bytes)
- `03_Control/05_Results/s4_execution/logs/s4_governor_selection_seed1.csv` (19912 bytes)
- `03_Control/05_Results/s4_execution/logs/s4_latency_high_bank_reversal_left_seed1.csv` (33783 bytes)
- `03_Control/05_Results/s4_execution/logs/s4_latency_low_bank_reversal_left_seed1.csv` (33843 bytes)
- `03_Control/05_Results/s4_execution/logs/s4_latency_nominal_bank_reversal_left_seed1.csv` (34093 bytes)
- `03_Control/05_Results/s4_execution/metrics/s4_annular_single_panel_seed1.csv` (1121 bytes)
- `03_Control/05_Results/s4_execution/metrics/s4_full_bank_reversal_left_no_wind_seed1.csv` (1066 bytes)
- `03_Control/05_Results/s4_execution/metrics/s4_full_nominal_glide_no_wind_seed1.csv` (1063 bytes)
- `03_Control/05_Results/s4_execution/metrics/s4_full_recovery_no_wind_seed1.csv` (1011 bytes)
- `03_Control/05_Results/s4_execution/metrics/s4_gaussian_single_panel_seed1.csv` (1096 bytes)
- `03_Control/05_Results/s4_execution/metrics/s4_governor_selection_seed1.csv` (1179 bytes)
- `03_Control/05_Results/s4_execution/metrics/s4_governor_selection_seed1_governor_candidates.csv` (420 bytes)
- `03_Control/05_Results/s4_execution/metrics/s4_governor_selection_seed1_governor_rejections.csv` (333 bytes)
- `03_Control/05_Results/s4_execution/metrics/s4_latency_high_bank_reversal_left_seed1.csv` (1063 bytes)
- `03_Control/05_Results/s4_execution/metrics/s4_latency_low_bank_reversal_left_seed1.csv` (1091 bytes)
- `03_Control/05_Results/s4_execution/metrics/s4_latency_nominal_bank_reversal_left_seed1.csv` (1078 bytes)
- `03_Control/05_Results/s4_execution/replay_a/logs/s4_gaussian_single_panel_randomised_seed8.csv` (18978 bytes)
- `03_Control/05_Results/s4_execution/replay_a/metrics/s4_gaussian_single_panel_randomised_seed8.csv` (1373 bytes)
- `03_Control/05_Results/s4_execution/replay_b/logs/s4_gaussian_single_panel_randomised_seed8.csv` (18978 bytes)
- `03_Control/05_Results/s4_execution/replay_b/metrics/s4_gaussian_single_panel_randomised_seed8.csv` (1373 bytes)
- `03_Control/05_Results/metrics/batch_seed1.csv` (2250 bytes)
- `03_Control/05_Results/metrics/batch_seed42.csv` (9981 bytes)
- `03_Control/05_Results/metrics/s0_no_wind_seed1.csv` (495 bytes)
- `03_Control/05_Results/metrics/s0_no_wind_seed42.csv` (958 bytes)
- `03_Control/05_Results/metrics/s0_no_wind_seed7.csv` (493 bytes)
- `03_Control/05_Results/metrics/s11_governor_rejection_seed1.csv` (513 bytes)
- `03_Control/05_Results/metrics/s11_governor_rejection_seed1_governor_rejections.csv` (279 bytes)
- `03_Control/05_Results/metrics/s11_governor_rejection_seed3.csv` (511 bytes)
- `03_Control/05_Results/metrics/s11_governor_rejection_seed3_governor_rejections.csv` (279 bytes)
- `03_Control/05_Results/metrics/s11_governor_rejection_seed42.csv` (1013 bytes)
- `03_Control/05_Results/metrics/s11_governor_rejection_seed42_governor_rejections.csv` (279 bytes)
- `03_Control/05_Results/metrics/s1_latency_nominal_no_wind_seed1.csv` (523 bytes)
- `03_Control/05_Results/metrics/s1_latency_nominal_no_wind_seed42.csv` (1008 bytes)
- `03_Control/05_Results/metrics/s1_latency_robust_upper_no_wind_seed1.csv` (537 bytes)
- `03_Control/05_Results/metrics/s1_latency_robust_upper_no_wind_seed42.csv` (1032 bytes)
- `03_Control/05_Results/metrics/s4_annular_single_panel_seed42.csv` (1099 bytes)
- `03_Control/05_Results/metrics/s4_full_bank_reversal_left_no_wind_seed42.csv` (1044 bytes)
- `03_Control/05_Results/metrics/s4_full_nominal_glide_no_wind_seed42.csv` (1041 bytes)
- `03_Control/05_Results/metrics/s4_full_recovery_no_wind_seed42.csv` (989 bytes)
- `03_Control/05_Results/metrics/s4_gaussian_single_panel_randomised_seed42.csv` (1365 bytes)
- `03_Control/05_Results/metrics/s4_gaussian_single_panel_seed42.csv` (1074 bytes)
- `03_Control/05_Results/metrics/s4_governor_selection_seed42.csv` (1145 bytes)
- `03_Control/05_Results/metrics/s4_governor_selection_seed42_governor_candidates.csv` (420 bytes)
- `03_Control/05_Results/metrics/s4_governor_selection_seed42_governor_rejections.csv` (333 bytes)
- `03_Control/05_Results/metrics/s4_latency_high_bank_reversal_left_seed42.csv` (1041 bytes)
- `03_Control/05_Results/metrics/s4_latency_low_bank_reversal_left_seed42.csv` (1069 bytes)
- `03_Control/05_Results/metrics/s4_latency_nominal_bank_reversal_left_seed42.csv` (1056 bytes)
- `03_Control/05_Results/metrics/s6_four_gaussian_var_seed1.csv` (542 bytes)
- `03_Control/05_Results/metrics/s6_four_gaussian_var_seed42.csv` (1032 bytes)
- `03_Control/05_Results/metrics/s6_single_gaussian_var_seed1.csv` (549 bytes)
- `03_Control/05_Results/metrics/s6_single_gaussian_var_seed42.csv` (1052 bytes)
- `03_Control/05_Results/metrics/s7_four_annular_gp_seed1.csv` (541 bytes)
- `03_Control/05_Results/metrics/s7_four_annular_gp_seed42.csv` (1022 bytes)
- `03_Control/05_Results/metrics/s7_single_annular_gp_seed1.csv` (545 bytes)
- `03_Control/05_Results/metrics/s7_single_annular_gp_seed42.csv` (1036 bytes)
