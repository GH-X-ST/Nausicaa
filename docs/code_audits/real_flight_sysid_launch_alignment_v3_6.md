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

The heavy selected rows localise the promising region around transition-blend longitudinal proposals plus yaw-beta and post-stall `Cl_r` corrections. This heavy run was diagnostic. It was followed by the audit-only local Pareto refinement `n30_joint_pareto_040_local_promising_v1`, which promoted the conservative `S5-transition yaw0.75+Cl_r0.60` row as the active replay-alignment update.

## Claim Boundary

No active simulation, controller, or real-flight model constants were changed by the heavy diagnostic itself. The later local audit promoted `neutral_dry_air_residual_calibrated_replay_n30_joint_pareto_040_local_s5_yaw0p75_clr0p60_elevator_rudder_effectiveness_v1`; see the active workflow notes for the current runtime model.

The historical small 40 ms audit path remains available as `n30_joint_pareto_040_audit`. The heavy path is an extension for trade-off diagnosis, not a broader aerodynamic model refit.

## Verification

Relevant checks completed during the v3.6 update:

- `python -m pytest 03_Control/tests/test_neutral_aero_residual_sysid_contract.py`
- Heavy 40 ms refit with `--workers 8`
- Generated-file size audit; all generated files are below GitHub limits
- `git diff --check`
