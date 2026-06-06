# Real-Flight SysID Launch Alignment v3.7

Date: 2026-06-06

## Scope

This note records the documentation and scoring-alignment status after
`Real-Flight SysID Launch Alignment v3.6`.

The active dry-air replay correction remains
`neutral_dry_air_residual_calibrated_replay_n30_joint_pareto_040_local_s5_yaw0p75_clr0p60_elevator_rudder_effectiveness_v1`,
promoted from `n30_joint_pareto_040_local_promising_v1`. The default neutral
replay alignment remains `0.040 s`, matching the synchronized simulation and
real-flight launch-handoff boundary. Legacy `0.100 s` alignment is not the
default fitting or acceptance path.

## Scoring Alignment

The governor calibrated-regime mismatch penalty now follows the active model
instead of a stale fixed boundary. Candidate alpha risk uses the active
`neutral_dry_air_calibration.py` residual-blend limits:

- transition starts at 14 deg
- full post-stall exposure starts at 18 deg
- full-risk score penalty remains `-0.12`, capped by the existing `0.18` limit

The penalty remains a bounded score term, not a hard anti-lift gate. High-AoA
candidates can still be selected when their mission value justifies the
active-model mismatch exposure. Memory shielding still rejects remembered-flow
switches that move into a higher calibrated-regime risk state.

## Logging Contract

Candidate, selector, and memory-opportunity logs now expose the active risk
source by recording:

- `calibrated_regime_source_calibration_id`
- `calibrated_regime_transition_start_alpha_deg`
- `calibrated_regime_post_stall_alpha_deg`

This prevents future audits from silently mixing old-fit risk boundaries with
the promoted active calibration.

## Claim Boundary

This update aligns scoring, diagnostics, and documentation with the promoted
active model. It does not promote a new aerodynamic parameter set and does not
claim full lateral SysID, hardware readiness, real-flight transfer, R10/R11
validation, mission success, or full autonomy.

## Verification

Relevant checks completed during the v3.7 update:

- direct import check of the active calibration ID and 14--18 deg risk boundary
- `python -m pytest 03_Control/tests/test_v53_algorithm_contract.py --basetemp .codex_pytest_tmp_v53 -p no:cacheprovider`
- `python -m pytest 03_Control/tests/test_neutral_aero_residual_sysid_contract.py 03_Control/tests/test_model_foundation_smoke.py 03_Control/tests/test_prim_model.py --basetemp .codex_pytest_tmp_model -p no:cacheprovider`
- `python -m pytest 03_Control/tests/test_control_surface_effectiveness_study.py --basetemp .codex_pytest_tmp_effectiveness -p no:cacheprovider`
- `git diff --check`
- `git diff --cached --check`
