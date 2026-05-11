# S4 Verification Report

Generated UTC: 2026-05-10T19:59:54.270383+00:00
Repository root: `C:\Users\GH-X-ST\OneDrive - Imperial College London\Year 4\Final Year Project\01 - Github\Nausicaa`
Commit before task: `046f739`
Dirty status before implementation: `A  .gitignore`
Dirty status after validation: Source/test/report files modified. Generated output under s4_execution and seed42 batch outputs are ignored. Three tracked legacy result files show status-only modifications because git index refresh is blocked by .git/index ACL, but git diff shows no content changes for them.

## Environment

- Python: `C:/ProgramData/miniforge3/python.exe`, 3.12.11 packaged by conda-forge.
- Platform: Windows-11-10.0.26200-SP0.
- pip: 26.1.1.
- Setup attempts failed because this repository has no `setup.py`/`pyproject.toml`; `requirements.txt` is pyproject-style TOML, not pip requirements.
- Hardware/Vicon/MATLAB/Arduino paths were not run; this S4 task is simulation-only Python validation.

## Commands Run

| Stage | Command | Exit | Summary |
|---|---|---:|---|
| discovery | `Get-Location` | 0 | repository root confirmed |
| discovery | `git status --short` | 0 | initial status showed A .gitignore |
| discovery | `git rev-parse --short HEAD` | 0 | 046f739 |
| discovery | `python -c "import sys, platform; print(sys.executable); print(sys.version); print(platform.platform())"` | 0 | Python 3.12.11 on Windows 11 |
| discovery | `python -m pip --version` | 0 | pip 26.1.1 |
| setup | `python -m pip install -e ".[test]"` | 1 | no setup.py or pyproject.toml present |
| setup | `python -m pip install -e .` | 1 | no setup.py or pyproject.toml present |
| setup | `python -m pip install -r requirements-dev.txt` | 1 | requirements-dev.txt missing |
| setup | `python -m pip install -r requirements.txt` | 1 | requirements.txt contains pyproject-style TOML, not pip requirements |
| validation | `python -m compileall 03_Control tests` | 0 | compiled control modules and tests |
| validation | `python -m pytest -q tests/test_latency_envelope.py tests/test_primitive_no_nan.py tests/test_primitive_full_duration.py tests/test_governor_rollout_selection.py tests/test_metrics_schema.py tests/test_updraft_randomisation.py` | 0 | 7 passed |
| validation | `python -m pytest -q` | 0 | 21 passed |
| scenario | `python 03_Control/04_Scenarios/run_one.py --scenario s4_full_nominal_glide_no_wind --seed 1 --output-root 03_Control/05_Results/s4_execution` | 0 | scenario command exited 0 and produced metrics |
| scenario | `python 03_Control/04_Scenarios/run_one.py --scenario s4_full_bank_reversal_left_no_wind --seed 1 --output-root 03_Control/05_Results/s4_execution` | 0 | scenario command exited 0 and produced metrics |
| scenario | `python 03_Control/04_Scenarios/run_one.py --scenario s4_full_recovery_no_wind --seed 1 --output-root 03_Control/05_Results/s4_execution` | 0 | scenario command exited 0 and produced metrics |
| scenario | `python 03_Control/04_Scenarios/run_one.py --scenario s4_latency_low_bank_reversal_left --seed 1 --output-root 03_Control/05_Results/s4_execution` | 0 | scenario command exited 0 and produced metrics |
| scenario | `python 03_Control/04_Scenarios/run_one.py --scenario s4_latency_nominal_bank_reversal_left --seed 1 --output-root 03_Control/05_Results/s4_execution` | 0 | scenario command exited 0 and produced metrics |
| scenario | `python 03_Control/04_Scenarios/run_one.py --scenario s4_latency_high_bank_reversal_left --seed 1 --output-root 03_Control/05_Results/s4_execution` | 0 | scenario command exited 0 and produced metrics |
| scenario | `python 03_Control/04_Scenarios/run_one.py --scenario s4_gaussian_single_panel --seed 1 --output-root 03_Control/05_Results/s4_execution` | 0 | scenario command exited 0 and produced metrics |
| scenario | `python 03_Control/04_Scenarios/run_one.py --scenario s4_annular_single_panel --seed 1 --output-root 03_Control/05_Results/s4_execution` | 0 | scenario command exited 0 and produced metrics |
| scenario | `python 03_Control/04_Scenarios/run_one.py --scenario s4_governor_selection --seed 1 --output-root 03_Control/05_Results/s4_execution` | 0 | scenario command exited 0 and produced metrics |
| audit | `python 03_Control/03_Primitives/run_primitive_audit.py --seed 1 --output-root 03_Control/05_Results/s4_execution/audit` | 0 | 15 S4 audit scenarios reported success True after rerun |
| batch | `python 03_Control/04_Scenarios/run_batch.py --seed 42` | 0 | batch_seed42.csv produced |
| replay | `run_scenario("s4_gaussian_single_panel_randomised", seed=8) twice and compare metrics` | 0 | deterministic replay passed |
| cleanup | `git restore -- <generated files>` | 1 | .git/index.lock could not be created due sandbox/Git ACL; restored file contents via git show instead |

## Required Scenario Metrics

| Scenario | Success | Termination | Duration s | Height m | Speed m/s | Min wall m | Max alpha deg | Saturation | Metrics |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| s4_full_nominal_glide_no_wind | True |  | 0.850 | -0.691 | 6.689 | 0.113 | 5.771 | 0.000 | `03_Control/05_Results/s4_execution/metrics/s4_full_nominal_glide_no_wind_seed1.csv` |
| s4_full_bank_reversal_left_no_wind | True |  | 0.850 | -0.626 | 6.381 | 0.126 | 12.781 | 0.000 | `03_Control/05_Results/s4_execution/metrics/s4_full_bank_reversal_left_no_wind_seed1.csv` |
| s4_full_recovery_no_wind | True |  | 0.850 | -0.417 | 6.305 | 0.200 | 5.854 | 0.000 | `03_Control/05_Results/s4_execution/metrics/s4_full_recovery_no_wind_seed1.csv` |
| s4_latency_low_bank_reversal_left | True |  | 0.850 | -0.620 | 6.254 | 0.135 | 14.918 | 0.008 | `03_Control/05_Results/s4_execution/metrics/s4_latency_low_bank_reversal_left_seed1.csv` |
| s4_latency_nominal_bank_reversal_left | True |  | 0.850 | -0.626 | 6.381 | 0.126 | 12.781 | 0.000 | `03_Control/05_Results/s4_execution/metrics/s4_latency_nominal_bank_reversal_left_seed1.csv` |
| s4_latency_high_bank_reversal_left | True |  | 0.850 | -0.630 | 6.420 | 0.126 | 12.458 | 0.000 | `03_Control/05_Results/s4_execution/metrics/s4_latency_high_bank_reversal_left_seed1.csv` |
| s4_gaussian_single_panel | True |  | 0.340 | -0.140 | 6.269 | 0.200 | 12.339 | 0.000 | `03_Control/05_Results/s4_execution/metrics/s4_gaussian_single_panel_seed1.csv` |
| s4_annular_single_panel | True |  | 0.340 | -0.175 | 6.407 | 0.200 | 9.382 | 0.000 | `03_Control/05_Results/s4_execution/metrics/s4_annular_single_panel_seed1.csv` |
| s4_governor_selection | True |  | 0.500 | -0.788 | 6.261 | 0.200 | 24.430 | 0.179 | `03_Control/05_Results/s4_execution/metrics/s4_governor_selection_seed1.csv` |

## Audit Metrics

| Scenario | Success | Duration s | Selected primitive | Metrics |
|---|---:|---:|---|---|
| s4_annular_four_panel | True | 0.260 | nominal_glide | `03_Control/05_Results/s4_execution/audit/metrics/s4_annular_four_panel_seed1.csv` |
| s4_annular_single_cg | True | 0.340 | nominal_glide | `03_Control/05_Results/s4_execution/audit/metrics/s4_annular_single_cg_seed1.csv` |
| s4_annular_single_panel | True | 0.340 | nominal_glide | `03_Control/05_Results/s4_execution/audit/metrics/s4_annular_single_panel_seed1.csv` |
| s4_full_bank_reversal_left_no_wind | True | 0.850 | bank_reversal | `03_Control/05_Results/s4_execution/audit/metrics/s4_full_bank_reversal_left_no_wind_seed1.csv` |
| s4_full_bank_reversal_right_no_wind | True | 0.850 | bank_reversal | `03_Control/05_Results/s4_execution/audit/metrics/s4_full_bank_reversal_right_no_wind_seed1.csv` |
| s4_full_nominal_glide_no_wind | True | 0.850 | nominal_glide | `03_Control/05_Results/s4_execution/audit/metrics/s4_full_nominal_glide_no_wind_seed1.csv` |
| s4_full_recovery_no_wind | True | 0.850 | recovery | `03_Control/05_Results/s4_execution/audit/metrics/s4_full_recovery_no_wind_seed1.csv` |
| s4_gaussian_four_panel | True | 0.260 | nominal_glide | `03_Control/05_Results/s4_execution/audit/metrics/s4_gaussian_four_panel_seed1.csv` |
| s4_gaussian_single_cg | True | 0.340 | nominal_glide | `03_Control/05_Results/s4_execution/audit/metrics/s4_gaussian_single_cg_seed1.csv` |
| s4_gaussian_single_panel_randomised | True | 0.340 | nominal_glide | `03_Control/05_Results/s4_execution/audit/metrics/s4_gaussian_single_panel_randomised_seed1.csv` |
| s4_gaussian_single_panel | True | 0.340 | nominal_glide | `03_Control/05_Results/s4_execution/audit/metrics/s4_gaussian_single_panel_seed1.csv` |
| s4_governor_selection | True | 0.500 | recovery | `03_Control/05_Results/s4_execution/audit/metrics/s4_governor_selection_seed1.csv` |
| s4_latency_high_bank_reversal_left | True | 0.850 | bank_reversal | `03_Control/05_Results/s4_execution/audit/metrics/s4_latency_high_bank_reversal_left_seed1.csv` |
| s4_latency_low_bank_reversal_left | True | 0.850 | bank_reversal | `03_Control/05_Results/s4_execution/audit/metrics/s4_latency_low_bank_reversal_left_seed1.csv` |
| s4_latency_nominal_bank_reversal_left | True | 0.850 | bank_reversal | `03_Control/05_Results/s4_execution/audit/metrics/s4_latency_nominal_bank_reversal_left_seed1.csv` |

## Result Generation

- Primary S4 result root: `03_Control/05_Results/s4_execution/`.
- Audit result root: `03_Control/05_Results/s4_execution/audit/`.
- Batch metrics: `03_Control/05_Results/metrics/batch_seed42.csv`.
- Governor candidate tables were written for `s4_governor_selection` in primary, audit, and batch outputs.
- Manifest: `03_Control/05_Results/manifests/results_manifest.json` and `docs/s4_results_manifest.json`.

## Reproducibility

- Deterministic replay command executed `s4_gaussian_single_panel_randomised` twice with seed 8.
- Compared `success`, `termination_reason`, `height_change_m`, `terminal_speed_m_s`, `max_alpha_deg`, `min_wall_distance_m`, and `wind_param_label`.
- Result: exact match / floating-point match within 1e-12.

## Missing or Failed Items

- Editable install failed: no Python project metadata file is present.
- `requirements-dev.txt` is absent.
- `requirements.txt` is not a pip requirements file.
- Git index refresh/restore through Git failed because `.git/index.lock` cannot be created in this sandbox ACL; generated tracked CSV contents were restored from HEAD via `git show`, but status may still show them until Git can refresh the index.
- No hardware, Vicon, MATLAB, Arduino, real-flight, thesis figure, RL, MPC, or outer-loop belief planning paths were executed.

## Thesis Implication

The repository now supports S4 simulation-only primitive analysis: full-duration indoor-feasible primitive cases, latency envelope sweeps, measured Gaussian/annular updraft stress cases, deterministic randomised updraft labels, rollout-based governor candidate selection, stable metrics, and regression tests. It is not evidence for S5/RV1 real validation because no hardware/Vicon/flight execution was run.
