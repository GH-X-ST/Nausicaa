# Real-Flight SysID Launch Alignment v3.9

Date: 2026-06-06

## Scope

This note records the documentation and figure-evidence status after
`Real-Flight SysID Launch Alignment v3.8` and the current uncommitted model
updates.

The active dry-air replay correction remains:

`neutral_dry_air_residual_calibrated_replay_n30_joint_pareto_040_local_s5_yaw0p75_clr0p60_surface_schedule_v3p2_cons_nominal`

The neutral-glide residual calibration is unchanged from v3.7/v3.8. The
post-v3.8 changes audited here are limited to:

- W3 plant-side control-surface authority perturbation alignment;
- regenerated real-flight replay-comparison figures under the active model;
- documentation updates so the active workflow text matches the code and
  regenerated figure manifests.

## Plant Perturbation Alignment

R5/R7/R10/R11 robustness now treats aerodynamic surface authority uncertainty
as a multiplier around the active alpha-regime schedule, not as a pre-schedule
`control_mix` or old scalar-effectiveness perturbation.

The active W3 plant-side policy is:

`global_plus_axis_scheduled_surface_authority_multiplier_v4`

with:

- global scheduled-authority multiplier: `0.75--1.25`;
- aileron/elevator/rudder axis multipliers: `0.85--1.15`;
- `control_mix` retained as geometry/sign mapping;
- `surface_calibration_scale` retained nominal for W3 plant instances;
- implementation-side surface-effectiveness scaling retired to avoid
  duplicating plant-side authority uncertainty; command/actuator uncertainty
  still covers command timing, actuator lag, neutral bias, saturation-limit
  clipping, and left/right aileron asymmetry.

R5/R7/R10/R11 use the same active model path through
`build_nausicaa_glider()`, `plant_instance_for_layer()`, and
`apply_plant_instance_to_aircraft()`. R8 does not define a separate plant model;
it depends on regenerated upstream evidence.

## Replay-Comparison Figures

The real-flight replay comparison folder was regenerated against the active
model:

`03_Control/A_figures/real_flight_replay_comparison/`

The updated manifest records:

- active calibration ID:
  `neutral_dry_air_residual_calibrated_replay_n30_joint_pareto_040_local_s5_yaw0p75_clr0p60_surface_schedule_v3p2_cons_nominal`;
- active surface-effectiveness model: `alpha_regime_scheduled_v1`;
- active schedule:
  `aileron=0.85/0.55/0.45`,
  `elevator=0.75/0.55/0.45`,
  `rudder=0.85/0.55/0.40`;
- representative-selection source. If the old archived v3.0
  `control_surface_inventory.csv` is absent, the plotting script reuses the
  existing replay-comparison summary to preserve the same representative
  launches while recomputing theory and calibrated replay traces.

The figure update is a visual/model-consistency refresh only. It is not new
R5/R7/R8/R10/R11 evidence.

## Documentation Audit

The active `.txt` and `.md` workflow docs now describe:

- the active v3.8/v3.9 calibration ID and conservative scheduled surface model;
- the W3 global-plus-axis surface-authority perturbation ranges;
- the fact that perturbations multiply the active scheduled surface authority,
  not the geometry-basis `control_mix`;
- the regenerated `real_flight_replay_comparison` figures and manifest fields;
- the summary-based figure-selection fallback used when archived v3.0 surface
  inventory files are no longer present in the cleaned result set.

Historical v3.6/v3.7/v3.8 audit files remain historical records and retain the
older wording where they describe earlier evidence stages.

## Claim Boundary

This update aligns the active model, robustness assumptions, documentation, and
replay-comparison figures. It does not regenerate R5/R7/R8/R10/R11 evidence and
does not make hardware-readiness, real-flight transfer, mission-success, full
aerodynamic SysID, or autonomy claims.

The old pure theory/geometry baseline remains comparison-only:

`03_Control/02_Inner_Loop/A_model_parameters/neutral_dry_air_theory_baseline_comparison.json`

## Checks

- `python -m py_compile 03_Control/03_Primitives/plant_instance.py 03_Control/03_Primitives/prim_roll.py 03_Control/04_Scenarios/run_repeated_launch_learning_curve.py 03_Control/04_Scenarios/run_lqr_w01_dense_chunked.py 03_Control/04_Scenarios/run_w3_survival.py`
- `python -m pytest 03_Control/tests/test_implementation_plant_instances.py --basetemp .codex_pytest_tmp_impl -p no:cacheprovider`
- `python -m pytest 03_Control/tests/test_v53_algorithm_contract.py --basetemp .codex_pytest_tmp_v53 -p no:cacheprovider`
- `python -m pytest 03_Control/tests/test_model_foundation_smoke.py 03_Control/tests/test_prim_model.py --basetemp .codex_pytest_tmp_model -p no:cacheprovider`
- `python -m py_compile 03_Control/01_Plotting/run_real_flight_replay_comparison_figures.py`
- `python 03_Control/01_Plotting/run_real_flight_replay_comparison_figures.py --extra-neutral-samples 3`
- visual sanity check of one regenerated replay-comparison PNG
- `git diff --check`
