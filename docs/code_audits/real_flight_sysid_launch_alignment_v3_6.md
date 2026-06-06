# Real-Flight SysID Launch Alignment v3.6

Date: 2026-06-06

## Scope

This note records the documentation and workflow status after the 40 ms launch-aligned SysID work that followed `Real-Flight SysID Launch Alignment v3.5`.

The default neutral replay alignment remains `0.040 s`, matching the synchronized simulation/real-flight launch handoff boundary. Legacy `0.100 s` alignment is no longer the default fitting or acceptance path.

## Heavy Joint Pareto Diagnostic

The neutral aero residual SysID runner now supports a bounded heavy 40 ms joint Pareto profile. The profile keeps the same compact parameter family and increases search coverage only by evaluating more longitudinal bases, scaled lateral/cross-coupling single terms, pairs, and capped triples on the same held-out replay split.

The corrected heavy diagnostic run writes to:

`03_Control/05_Results/glider_model_calibration_prep/n30_joint_pareto_040_heavy_v1/`

Key run counts:

- Candidates: 210
- Accepted rows: 41
- Selected Pareto survivors: 6
- Workers: 8
- Alignment window: 0.040 s

The heavy selected rows localise the promising region around transition-blend longitudinal proposals plus yaw-beta and post-stall `Cl_r` corrections. These rows are diagnostic only. They should feed a narrower dense local Pareto refinement before any active calibration constants are promoted.

## Claim Boundary

No active simulation, controller, or real-flight model constants are changed by this diagnostic. The current promoted model remains the compact residual-calibrated replay model documented in the active workflow notes.

The historical small 40 ms audit path remains available as `n30_joint_pareto_040_audit`. The heavy path is an extension for trade-off diagnosis, not a broader aerodynamic model refit.

## Verification

Relevant checks completed during the v3.6 update:

- `python -m pytest 03_Control/tests/test_neutral_aero_residual_sysid_contract.py`
- Heavy 40 ms refit with `--workers 8`
- Generated-file size audit; all generated files are below GitHub limits
- `git diff --check`
