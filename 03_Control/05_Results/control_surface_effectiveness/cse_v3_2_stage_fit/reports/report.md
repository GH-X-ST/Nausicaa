# Real-Flight Control Surface Effectiveness Stage-Wise Refit v3.2

## 1. Purpose and Claim Boundary

This run fits alpha-regime surface-effectiveness schedules from all kept pulse-ladder rows. It does not use a held-out promotion gate and does not promote the schedule into the active model.

- active model: `neutral_dry_air_replay_040_local_s5_yaw0p75_clr0p60_surface_scale_v3p1_a0p65_e0p70_r0p45`
- fit policy: `all_kept_rows_no_heldout_gate_no_minimum_evidence_threshold`
- zero-support policy: `absolute_scale_1p0_prior_not_fitted`
- promotion decision: `not_promoted_stage_schedule_pending_post_analysis`

## 2. Selected Stage Schedule

`surface | regime | selected scale | n | pos | neg | primary | primary MAE | attitude MAE | dx | dy | altitude | status`
`aileron | normal | 1 | 4 | 4 | 0 | peak_p_rad_s | 0.2667959648 | 40.69874583 | 0.7096740405 | 0.5201464353 | 0.9176592196 | selected_all_available_data_stage_fit`
`aileron | transition | 0.4 | 24 | 14 | 10 | peak_p_rad_s | 0.5828834112 | 11.40567007 | 0.3787243757 | 0.3367082161 | 0.2401872875 | selected_all_available_data_stage_fit`
`aileron | post_stall | 0.4 | 31 | 12 | 19 | peak_p_rad_s | 0.7866637402 | 10.24804418 | 0.2968170653 | 0.6844293646 | 0.1897868511 | selected_all_available_data_stage_fit`
`elevator | normal | 0.75 | 8 | 0 | 8 | peak_q_rad_s | 0.5686521427 | 22.23200834 | 0.2276646977 | 0.1951182968 | 0.6424295639 | selected_all_available_data_stage_fit`
`elevator | transition | 0.55 | 18 | 0 | 18 | peak_q_rad_s | 0.7205289119 | 13.27941341 | 0.08526509549 | 0.4626089115 | 0.1309629986 | selected_all_available_data_stage_fit`
`elevator | post_stall | 0.4 | 31 | 28 | 3 | peak_q_rad_s | 0.8854079671 | 11.22235051 | 0.1480326 | 0.5478494597 | 0.07829993209 | selected_all_available_data_stage_fit`
`rudder | normal | 1 | 0 | 0 | 0 | peak_r_rad_s |  |  |  |  |  | zero_support_1p0_prior_not_fitted`
`rudder | transition | 0.4 | 38 | 19 | 19 | peak_r_rad_s | 0.4001007497 | 7.66920932 | 0.2565475769 | 0.4443301558 | 0.1705381043 | selected_all_available_data_stage_fit`
`rudder | post_stall | 0.4 | 20 | 10 | 10 | peak_r_rad_s | 0.4235430668 | 7.230413702 | 0.2057760497 | 0.5440503147 | 0.1254812222 | selected_all_available_data_stage_fit`

## 3. Combined Schedule Replay

`candidate | split | surface | n | dx | dy | altitude | phi | theta | psi | primary`
`C0_frozen_neutral | all | all | 174 | 0.2653702475 | 0.4934685156 | 0.1937571674 | 11.77315732 | 11.07162589 | 14.37485976 | 0.2496519649`
`C0_frozen_neutral | all | aileron | 59 | 0.3796977007 | 0.5312133122 | 0.2522596614 | 11.56016771 | 6.868867214 | 27.95078592 | 0.1428582285`
`C0_frozen_neutral | all | elevator | 57 | 0.1736952689 | 0.4683901787 | 0.1726748316 | 9.909081603 | 20.2309726 | 7.245512824 | 0.4441424708`
`C0_frozen_neutral | all | rudder | 58 | 0.239166007 | 0.4797188985 | 0.154964857 | 13.82175564 | 6.345418993 | 7.571293067 | 0.1619551953`
`STG_combined_all_data_stage_schedule | all | all | 174 | 0.2192917597 | 0.5033248575 | 0.2075994942 | 14.3619576 | 8.746267024 | 14.56653544 | 0.2885523874`
`STG_combined_all_data_stage_schedule | all | aileron | 59 | 0.3434029358 | 0.5200730232 | 0.298598557 | 19.75479904 | 7.048573344 | 20.84769958 | 0.1497365768`
`STG_combined_all_data_stage_schedule | all | elevator | 57 | 0.1411549783 | 0.4710991731 | 0.1740526022 | 9.654191666 | 12.98626766 | 7.228972364 | 0.2687543436`
`STG_combined_all_data_stage_schedule | all | rudder | 58 | 0.1698303313 | 0.5179579994 | 0.1479999792 | 13.50273368 | 6.306334104 | 15.38812872 | 0.447166242`

## 4. Output Tables

- `metrics/stage_fit.csv`: selected 9-cell schedule.
- `metrics/stage_cand.csv`: all 117 single-cell candidates and scores.
- `metrics/stage_replay.csv`: single-cell evidence replays plus combined schedule replay.
- `metrics/stage_err.csv`: frozen baseline and combined schedule replay summaries.
- `metrics/regime_err.csv`: normal/transition/post-stall replay ladder for baseline and combined schedule.

## 5. Notes

- stage candidates generated: `117`
- stage candidate score rows: `117`
- regime ladder rows: `270`
- Command conversion, measured surface angles, actuator lag, servo signs, and hardware packet mapping are unchanged.
