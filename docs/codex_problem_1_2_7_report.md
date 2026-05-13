# Codex Problems 1, 2, and 7 Report

Date: 2026-05-13

## Scope

This update addresses only:

- Problem 1: agile confined reversal behavioural feasibility or boundary.
- Problem 2: complete manoeuvre horizon evidence.
- Problem 7: packaging and reproducibility cleanup.

Frozen project invariants were preserved: public world frame z-up, body frame x-forward/y-starboard/z-down, canonical 15-state order, command order `[delta_a_cmd, delta_e_cmd, delta_r_cmd]`, existing signs, wind modes, true safety volume use, calibrated asymmetric surface throws, full normalised command range `[-1.0, +1.0]`, and saturation logging.

## Packaging First Result

Packaging was repaired before treating the agile work as complete.

- `requirements.txt` is now valid pip requirements syntax.
- `pyproject.toml` provides minimal editable-install metadata without renaming numeric-prefix research folders.
- `nausicaa_build_backend.py` builds a minimal editable wheel with a `.pth` file exposing the existing research folders.
- Pytest path configuration is kept in `pyproject.toml`.

Validation:

- `python -m pip install -r requirements.txt`: passed.
- `python -m pip install -e .`: backend hooks and wheel build passed, install failed at external user site with `[WinError 5] Access denied` for `AppData/Roaming/Python/Python312/site-packages/nausicaa-0.1.1.dist-info`.
- `$env:PYTHONUSERBASE=03_Control/05_Results/codex_packaging_audit_rerun/python_userbase; python -m pip install -e . --no-deps`: passed, confirming the repository editable wheel installs when the target site is writable.
- `python -c "import numpy, scipy, pandas, matplotlib, casadi; print('dependency imports ok')"`: passed.
- `python -m compileall 03_Control tests`: passed.
- `python -m pytest -q`: passed, `73 passed in 109.91s`.

## Agile Search

Deterministic compact search was run before final feasibility:

```bash
python 03_Control/04_Scenarios/run_agile_template_search.py --seed 1 --output-root 03_Control/05_Results/codex_agile_search
```

Search configuration:

- Targets: `30, 60, 90, 120, 180 deg`.
- Families: `high_bank_roll_recovery`, `brake_roll_yaw_recovery`, `pitch_up_redirect_recovery`.
- `updraft_assisted` was not implemented because it was optional and not needed for the deterministic confined no-wind boundary.

Generated outputs:

- `03_Control/05_Results/codex_agile_search/metrics/agile_template_search_summary_seed1.csv`
- `03_Control/05_Results/codex_agile_search/metrics/agile_template_search_candidates_seed1.csv`
- `03_Control/05_Results/codex_agile_search/metrics/agile_template_search_best_by_target_seed1.csv`
- `03_Control/05_Results/codex_agile_search/manifests/agile_template_search_seed1.json`
- `03_Control/05_Results/codex_agile_search/logs/`

The search evaluated 15 candidates and selected `brake_roll_yaw_recovery` for every target. The best 30 deg candidate was `030_a`.

## Final Fixed-Start Feasibility

Final feasibility consumed the searched manifest and selected the best 30 deg candidate:

```bash
python 03_Control/04_Scenarios/run_agile_feasibility.py --seed 1 --output-root 03_Control/05_Results/codex_problem_1_2_7/agile_fixed
```

Result path:

- `03_Control/05_Results/codex_problem_1_2_7/agile_fixed/metrics/s9_agile_feasibility_seed1.csv`

30 deg fixed-start gate:

- Candidate: `brake_roll_yaw_recovery / 030_a`
- `abs(actual_heading_change_deg)`: `21.687866218439385`, below the required `24`.
- `success`: `True` in rollout terms.
- `exit_recoverable`: `False`.
- `min_wall_distance_m`: `0.25`.
- `feasibility_label`: `fixed_start_unrecoverable`.
- Gate result: fail, documented as feasibility boundary.

The random-entry sweep was not run because the post-search 30 deg fixed-start gate did not pass.

## Complete-Horizon Evidence

Every generated searched candidate stores complete phase metadata:

- `entry`
- `brake_or_pitch`
- `roll_yaw_redirect`
- `turn_hold_or_heading_capture`
- `recover`
- `exit_check`

Every generated searched trajectory stores finite:

- `times_s`
- `x_ref`
- `u_ff`
- `a_mats`
- `b_mats`
- `k_lqr`
- `s_mats`

Evidence:

- Candidate CSV field `finite_trajectory_arrays` is `True` for all searched candidates.
- Manifest stores the selected template for each target.
- `tests/test_agile_template_search.py` verifies the compact search grid, phase metadata, manifest loading, and finite trajectory arrays.

## Regression And Replay

Baseline regressions:

```bash
python 03_Control/04_Scenarios/run_one.py --scenario s4_full_nominal_glide_no_wind --seed 1 --output-root 03_Control/05_Results/codex_problem_1_2_7/baseline
python 03_Control/04_Scenarios/run_one.py --scenario s4_full_bank_reversal_left_no_wind --seed 1 --output-root 03_Control/05_Results/codex_problem_1_2_7/baseline
python 03_Control/04_Scenarios/run_one.py --scenario s4_full_recovery_no_wind --seed 1 --output-root 03_Control/05_Results/codex_problem_1_2_7/baseline
python 03_Control/04_Scenarios/run_one.py --scenario s4_governor_selection --seed 1 --output-root 03_Control/05_Results/codex_problem_1_2_7/baseline
```

All four baseline commands reported `success: True`.

Replay:

```bash
python 03_Control/04_Scenarios/run_agile_feasibility.py --seed 3 --output-root 03_Control/05_Results/codex_problem_1_2_7/replay_a
python 03_Control/04_Scenarios/run_agile_feasibility.py --seed 3 --output-root 03_Control/05_Results/codex_problem_1_2_7/replay_b
```

Replay summaries matched on target, family, candidate id, heading change, wall margin, height change, terminal speed, alpha, saturation, recoverability, and feasibility label.

## Thesis-Safe Claim

Under the frozen indoor safety and recoverability gates, the deterministic compact agile search did not validate a 30 deg fixed-start agile reversal. The best searched 30 deg candidate remained inside the wall margin but achieved only `21.69 deg` absolute heading change and did not exit recoverably. This supports a documented confined-agile-reversal feasibility boundary, not a successful 30 deg primitive claim.
