# S4 Arena-Axis Verification Report

Generated UTC: 2026-05-11T23:44:20+00:00
Repository head at task start: `3e522ef`.

## Arena Contract

- Tracker-limit axes: `x_w=[0.0, 8.0] m`, `y_w=[0.0, 4.8] m`, `z_w=[0.0, 3.5] m`.
- True safety volume: `x_w=[1.2, 6.6] m`, `y_w=[0.0, 4.4] m`, `z_w=[0.0, 3.0] m`.
- Nominal hand-launch centre: `[1.2, 0.4, 1.5] m`.
- The larger `10.0 x 6.2 x 5.5 m` room box is not used for control, metrics, or plotting axes.

## Commands Run

| Stage | Command | Exit |
|---|---|---:|
| housekeeping | cleanup generated `flight_case_results`, `s4_execution`, `manifests`, logs/metrics CSVs, `_run`, `__pycache__`, `.pytest_cache` | 0 |
| validation | `python -m compileall 03_Control tests` | 0 |
| validation | `python -m pytest -q tests/test_arena_bounds.py tests/test_launch_state_contract.py tests/test_plotting_smoke.py tests/test_plotting_output_paths.py` | 0 |
| validation | `python -m pytest -q` | 0 |
| scenario | `python 03_Control/04_Scenarios/run_one.py --scenario s4_full_nominal_glide_no_wind --seed 1 --output-root 03_Control/05_Results/s4_execution` | 0 |
| plot | `python 03_Control/05_Results/plot_one.py --scenario s4_full_nominal_glide_no_wind --seed 1` | 0 |
| plot batch | `python 03_Control/05_Results/plot_batch.py --scenario-set audit --seed 1` | 0 |
| batch | `python 03_Control/04_Scenarios/run_batch.py --seed 42` | 0 |
| post-cleanup | remove `_run`, `__pycache__`, `.pytest_cache`, and non-policy test CSVs | 0 |

## Regenerated Flight-Case Results

| Folder | Scenario | Success | Duration s | Min wall m | Height change m | Max alpha deg | Metrics |
|---|---|---:|---:|---:|---:|---:|---|
| `governor_recovery_selection_high_bank_seed_001` | s4_governor_selection | True | 0.500 | 0.250 | -0.820 | 25.387 | `03_Control/05_Results/flight_case_results/governor_recovery_selection_high_bank_seed_001/analysis_data/actual_metrics.csv` |
| `latency_high_bank_reversal_left_seed_001` | s4_latency_high_bank_reversal_left | True | 0.760 | 0.225 | -0.586 | 5.843 | `03_Control/05_Results/flight_case_results/latency_high_bank_reversal_left_seed_001/analysis_data/actual_metrics.csv` |
| `latency_low_bank_reversal_left_seed_001` | s4_latency_low_bank_reversal_left | True | 0.760 | 0.228 | -0.591 | 6.578 | `03_Control/05_Results/flight_case_results/latency_low_bank_reversal_left_seed_001/analysis_data/actual_metrics.csv` |
| `latency_nominal_bank_reversal_left_seed_001` | s4_latency_nominal_bank_reversal_left | True | 0.760 | 0.226 | -0.589 | 6.227 | `03_Control/05_Results/flight_case_results/latency_nominal_bank_reversal_left_seed_001/analysis_data/actual_metrics.csv` |
| `launch_nominal_glide_no_wind_seed_001` | s4_launch_nominal_glide_no_wind | True | 0.550 | 0.000 | -0.395 | 3.054 | `03_Control/05_Results/flight_case_results/launch_nominal_glide_no_wind_seed_001/analysis_data/actual_metrics.csv` |
| `primitive_bank_reversal_left_no_wind_seed_001` | s4_full_bank_reversal_left_no_wind | True | 0.760 | 0.226 | -0.589 | 6.227 | `03_Control/05_Results/flight_case_results/primitive_bank_reversal_left_no_wind_seed_001/analysis_data/actual_metrics.csv` |
| `primitive_bank_reversal_right_no_wind_seed_001` | s4_full_bank_reversal_right_no_wind | True | 0.760 | 0.227 | -0.588 | 6.259 | `03_Control/05_Results/flight_case_results/primitive_bank_reversal_right_no_wind_seed_001/analysis_data/actual_metrics.csv` |
| `primitive_nominal_glide_no_wind_seed_001` | s4_full_nominal_glide_no_wind | True | 0.760 | 0.221 | -0.585 | 3.054 | `03_Control/05_Results/flight_case_results/primitive_nominal_glide_no_wind_seed_001/analysis_data/actual_metrics.csv` |
| `primitive_recovery_no_wind_seed_001` | s4_full_recovery_no_wind | True | 0.760 | 0.250 | -0.275 | 3.935 | `03_Control/05_Results/flight_case_results/primitive_recovery_no_wind_seed_001/analysis_data/actual_metrics.csv` |
| `updraft_four_fan_annular_gp_panel_wind_seed_001` | s4_annular_four_panel | True | 0.240 | 0.250 | -0.041 | 15.271 | `03_Control/05_Results/flight_case_results/updraft_four_fan_annular_gp_panel_wind_seed_001/analysis_data/actual_metrics.csv` |
| `updraft_four_fan_gaussian_panel_wind_seed_001` | s4_gaussian_four_panel | True | 0.240 | 0.250 | 0.034 | 18.866 | `03_Control/05_Results/flight_case_results/updraft_four_fan_gaussian_panel_wind_seed_001/analysis_data/actual_metrics.csv` |
| `updraft_single_fan_annular_gp_centre_wind_seed_001` | s4_annular_single_cg | True | 0.340 | 0.250 | -0.194 | 3.060 | `03_Control/05_Results/flight_case_results/updraft_single_fan_annular_gp_centre_wind_seed_001/analysis_data/actual_metrics.csv` |
| `updraft_single_fan_annular_gp_panel_wind_seed_001` | s4_annular_single_panel | True | 0.340 | 0.250 | -0.175 | 9.378 | `03_Control/05_Results/flight_case_results/updraft_single_fan_annular_gp_panel_wind_seed_001/analysis_data/actual_metrics.csv` |
| `updraft_single_fan_gaussian_centre_wind_seed_001` | s4_gaussian_single_cg | True | 0.340 | 0.250 | -0.171 | 3.950 | `03_Control/05_Results/flight_case_results/updraft_single_fan_gaussian_centre_wind_seed_001/analysis_data/actual_metrics.csv` |
| `updraft_single_fan_gaussian_panel_wind_seed_001` | s4_gaussian_single_panel | True | 0.340 | 0.250 | -0.140 | 12.482 | `03_Control/05_Results/flight_case_results/updraft_single_fan_gaussian_panel_wind_seed_001/analysis_data/actual_metrics.csv` |
| `updraft_single_fan_gaussian_randomised_seed_001` | s4_gaussian_single_panel_randomised | True | 0.340 | 0.250 | -0.143 | 11.728 | `03_Control/05_Results/flight_case_results/updraft_single_fan_gaussian_randomised_seed_001/analysis_data/actual_metrics.csv` |

## Required Scenario Output

| Metrics | Scenario | Success | Duration s | Min wall m |
|---|---|---:|---:|---:|
| `03_Control/05_Results/s4_execution/metrics/s4_full_nominal_glide_no_wind_seed1.csv` | s4_full_nominal_glide_no_wind | True | 0.760 | 0.221 |

## Batch Seed 42

- Batch metrics: `03_Control/05_Results/metrics/batch_seed42.csv`.
- Batch scenario rows: 19.
- Scenario failures after the boundary update:
  - `s11_governor_rejection`: success=False, termination=`governor_rejected`, failure_class=`governor`.
  - `s4_gaussian_single_panel_randomised`: success=False, termination=`angle of attack exceeded bound`, failure_class=`model`.

## Housekeeping

- Old `flight_case_results/`, `s4_execution/`, `manifests/`, logs CSVs, metrics CSVs, `_run`, `__pycache__`, and `.pytest_cache` were deleted before validation/regeneration.
- Historical generated `controller_execution_audit/` and `latency_validation/` folders were also deleted after inspection showed stale tracker-margin columns from the obsolete centred tracker bounds.
- Direct `git rm` was attempted first but could not create `.git/index.lock` in this sandbox, so the same explicit generated paths were removed through a workspace-contained cleanup script.
- Post-run checks found no `_run`, `__pycache__`, or `.pytest_cache` directories.
- Stale centred tracker-bound tuple checks over active scenario/plotting code and regenerated outputs returned no matches.
- `_run` and machine-local absolute-path checks over regenerated outputs returned no matches.

## Result Policy

Preferred policy used: old generated result folders were deleted, then a fresh audit `flight_case_results` set, required `s4_execution` scenario output, and seed-42 batch metrics/logs were regenerated. No old `flight_case_results` folder was carried over.
