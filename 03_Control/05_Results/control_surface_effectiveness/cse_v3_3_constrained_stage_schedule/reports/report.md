# Real-Flight Control Surface Effectiveness Constrained Stage-Schedule Audit v3.3

## 1. Purpose and Claim Boundary

This audit tests physically constrained alpha-regime surface-effectiveness schedules with joint combined replay scoring. It does not promote a schedule into the active model.

- active model: `neutral_dry_air_replay_040_local_s5_yaw0p75_clr0p60_surface_scale_v3p1_a0p65_e0p70_r0p45`
- selected candidate: `CST05_lower_bound_limited`
- selected accepted: `False`
- promotion decision: `not_promoted_constrained_stage_schedule_pending_post_analysis`

## 2. Candidate Summary

`candidate | selected | accepted | score | dx | dy | altitude | phi | theta | psi | primary | status`
`CST05_lower_bound_limited | True | False | 2.528686608 | 0.2201509362 | 0.5035889981 | 0.2069384672 | 14.37846584 | 9.171265073 | 15.13870948 | 0.3083935272 | rejected_final_phi_deg_and_primary_residual`
`CST07_deep_post_relief | False | False | 2.600075766 | 0.2189878466 | 0.5029267117 | 0.2054295429 | 14.46115033 | 9.297221711 | 15.25269545 | 0.3723627485 | rejected_final_phi_deg_and_primary_residual`
`CST08_balanced_mid_decay | False | False | 2.690861716 | 0.2305407483 | 0.50435582 | 0.2014636415 | 13.77542468 | 9.913761369 | 16.50755664 | 0.4146100832 | rejected_final_phi_deg_and_final_psi_deg_and_primary_residual`
`CST01_attached_prior_active_post_relief | False | False | 2.724608781 | 0.2265804445 | 0.5037059765 | 0.2051447754 | 14.67913029 | 10.05204826 | 16.36138723 | 0.4194253919 | rejected_final_phi_deg_and_primary_residual`
`CST03_mild_authority_decay | False | False | 2.840550215 | 0.2315791179 | 0.5043069822 | 0.2057285182 | 14.9531569 | 10.4824123 | 17.43773103 | 0.4698255891 | rejected_final_phi_deg_and_final_psi_deg_and_primary_residual`
`CST06_active_post_with_attached_prior | False | False | 2.841704841 | 0.235393268 | 0.5042337599 | 0.2054244006 | 14.9021662 | 10.83501957 | 17.28452367 | 0.4625964311 | rejected_final_phi_deg_and_final_psi_deg_and_primary_residual`
`CST02_attached_prior_scalar_transition | False | False | 2.898804447 | 0.2326410477 | 0.5035997628 | 0.2074957657 | 14.99625455 | 10.98749619 | 17.4401226 | 0.5076054261 | rejected_final_phi_deg_and_final_psi_deg_and_primary_residual`
`CST04_conservative_decay | False | False | 3.05637377 | 0.2397043935 | 0.505132957 | 0.2067052377 | 15.48171242 | 11.65723815 | 18.93314166 | 0.5690947736 | rejected_final_phi_deg_and_final_psi_deg_and_primary_residual`

## 3. Output Tables

- `metrics/constrained_cand.csv`: joint candidate score and deltas against active baseline.
- `metrics/constrained_replay.csv`: all candidate replay rows.
- `metrics/constrained_err.csv`: replay summaries for baseline and all constrained candidates.
- `metrics/regime_err.csv`: normal/transition/post-stall replay ladder.

## 4. Notes

- constrained candidates: `8`
- error rows: `108`
- regime ladder rows: `1215`
- Command conversion, measured surface angles, actuator lag, servo signs, and hardware packet mapping are unchanged.
