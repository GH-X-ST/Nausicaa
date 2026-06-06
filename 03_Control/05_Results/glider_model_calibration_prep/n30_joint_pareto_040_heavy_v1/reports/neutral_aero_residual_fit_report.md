# Neutral Aero Residual Regime Fit

This run uses only neutral open-loop real throws. It estimates 6-DoF force/moment residuals from Vicon state trajectories, fits a claim-bearing longitudinal candidate by default, reports lateral residuals without claiming accurate lateral SysID, and validates candidates by held-out dry-air replay. The default `cm_regime_staged` workflow fits attached Cm, transition Cm, post-stall Cm/Cmq, transition blend, and optional post-stall CL/CD cleanup in separate held-out-gated stages. The `compact_joint_sweep` workflow starts from active constants, keeps the same compact model family, and jointly sweeps longitudinal plus small lateral/coupling terms after sign/range discovery. Longitudinal fitting uses lateral-contamination confidence; lateral/coupling fitting uses excitation-aware confidence. Rich transition lateral deltas, post-stall lateral surfaces, and post-stall alpha-RBF longitudinal surfaces are diagnostic-only unless explicitly enabled.

## Rerun Recipe

- source session root: `C:\Users\GH-X-ST\OneDrive - Imperial College London\Year 4\Final Year Project\01 - Github\Nausicaa\04_Flight_Test\05_Results\cal\n30`
- alignment window: `0.040` s
- derivative window: `0.040` s
- replay dt: `0.0050` s
- ridge lambda: `0.001`
- min speed: `1.50` m/s
- workers: `8`
- fit workflow: `cm_regime_staged`
- group iterations: `3`
- group improvement tolerance: `0.001`
- aligned launch filter: `True`
- aligned launch filter bounds: `u=[3.00, 8.00]` m/s, `|v|<=1.50` m/s, `|w|<=0.90` m/s
- apply attached Cm bias: `False`
- fit transition Cm bias: `False`
- fit post-stall CL/CD cleanup: `True`
- fit post-stall Cm bias: `True`
- fit compact post-stall longitudinal residuals: `True`
- fit transition blender: `True`
- fit post-stall alpha-RBF surfaces: `False`
- fit post-stall damping: `True`
- fit attached lateral coupling: `False`
- fit transition lateral coupling: `False`
- fit lateral surfaces: `False`
- fit secondary lateral diagnostic: `True`

```powershell
python 03_Control/02_Inner_Loop/run_fit_neutral_aero_residual_calibration.py --session-root "C:\Users\GH-X-ST\OneDrive - Imperial College London\Year 4\Final Year Project\01 - Github\Nausicaa\04_Flight_Test\05_Results\cal\n30" --run-label n30_joint_pareto_040_heavy_v1 --heldout-count 14 --heldout-seed 606 --alignment-window-s 0.04 --derivative-window-s 0.04 --replay-dt-s 0.005 --ridge-lambda 0.001 --min-speed-m-s 1.5 --workers 8 --fit-workflow cm_regime_staged --group-iterations 3 --group-improvement-tol 0.001 --aligned-u-min-m-s 3 --aligned-u-max-m-s 8 --aligned-v-abs-max-m-s 1.5 --aligned-w-abs-max-m-s 0.9 --no-sensitivity-alignment --joint-pareto-audit --joint-pareto-audit-alignment-window-s 0.04 --joint-pareto-profile heavy --joint-pareto-top-longitudinal 6 --joint-pareto-top-lateral 10 --joint-pareto-max-lateral-order 3 --joint-pareto-top-triples 80 --joint-pareto-max-candidates 900 --joint-pareto-selected-limit 8 --filter-aligned-launch-state --no-apply-attached-cm-bias --no-fit-transition-pitch-moment --fit-post-stall-lift-drag --fit-post-stall-pitch-moment --fit-post-stall-longitudinal --fit-transition-blender --no-fit-post-stall-surfaces --fit-post-stall-damping --no-fit-attached-lateral-coupling --no-fit-transition-lateral-coupling --no-fit-lateral-surfaces --fit-secondary-lateral-diagnostic
```

## Aligned Launch Filter

- loaded logged-valid throws: `105`
- kept throws after replay-start filter: `96`
- filtered throws: `9`
- filter audit CSV: `metrics/neutral_aero_residual_filtered_throws.csv`
- launch-confidence weighting: enabled for residual coefficient fitting
- confidence reference: replay-aligned lateral contamination `phi0=psi0=v0=p0=r0=0`; minimum weight `0.25`
- kept-throw confidence weight min/mean/max: `0.4944487386`, `0.7521380952`, `0.9816120127`
- kept-throw lateral score min/mean/max: `0.1112329321`, `0.4275734515`, `0.6852307623`
- rejected logged-valid throws:
  - `20260604_185934/v006`: aligned_v_outside_launch_gate; u0 `5.613730419`, v0 `1.644833847`, w0 `0.3397646591` m/s
  - `20260604_185934/v008`: aligned_u_outside_replay_filter; u0 `2.816040728`, v0 `0.7072700209`, w0 `0.4446827236` m/s
  - `20260604_195702/v010`: aligned_u_outside_replay_filter;aligned_v_outside_launch_gate;aligned_w_outside_launch_gate; u0 `8.575142964`, v0 `2.127116648`, w0 `1.408641211` m/s
  - `20260604_203516/v014`: aligned_v_outside_launch_gate; u0 `6.281828813`, v0 `1.691445057`, w0 `0.6205414612` m/s
  - `20260604_203516/v017`: aligned_v_outside_launch_gate; u0 `5.432502452`, v0 `1.509652866`, w0 `0.6363192345` m/s
  - `20260604_210642/v017`: aligned_w_outside_launch_gate; u0 `5.279067686`, v0 `0.60925795`, w0 `0.965680656` m/s
  - `20260604_210642/v018`: aligned_u_outside_replay_filter; u0 `2.311449042`, v0 `0.5872059989`, w0 `0.3228716841` m/s
  - `20260604_210642/v022`: aligned_w_outside_launch_gate; u0 `6.603567235`, v0 `1.237843141`, w0 `0.9190388099` m/s
  - `20260604_210642/v027`: aligned_w_outside_launch_gate; u0 `5.803018087`, v0 `0.870276046`, w0 `0.9322510307` m/s

## Coefficient Fit

- fit status: `ok`
- sample count: `14600`
- used sample count: `14600`
- post-stall used sample count: `1746`
- post-stall fit profile: `compact_scalar_activation`
- fit MAE in Cm: `0.04347`
- attached Cm residual: `-0.0158487`
- transition Cm residual before post-stall: `-0.04314`
- transition Cm residual after post-stall: `-0.00579819`
- attached lateral coupling:
  - CY: bias `0`, beta `0`, p_hat `0`, r_hat `0`
  - Cl: bias `0`, beta `0`, p_hat `0`, r_hat `0`
  - Cn: bias `0`, beta `0`, p_hat `0`, r_hat `0`
- transition-window lateral coupling:
  - CY: bias `0`, beta `0`, p_hat `0`, r_hat `0`
  - Cl: bias `0`, beta `0`, p_hat `0`, r_hat `0`
  - Cn: bias `0`, beta `0`, p_hat `0`, r_hat `0`
- primary lateral-coupling interpretation: report-only unless `--fit-attached-lateral-coupling` or other lateral flags are explicitly enabled; default primary SysID does not claim accurate lateral identification
- post-stall surface centres: `20, 45, 70` deg (`diagnostic only unless fit_post_stall_surfaces=True`)
- post-stall surface width: `15` deg
- post-stall CL surface: 20 deg `0`, 45 deg `0`, 70 deg `0`
- post-stall CD surface: 20 deg `0`, 45 deg `0`, 70 deg `0`
- post-stall Cm surface: 20 deg `0`, 45 deg `0`, 70 deg `0`
- post-stall Cmq surface: 20 deg `0`, 45 deg `0`, 70 deg `0`
- post-stall CY surface: bias: 20 deg `0`, 45 deg `0`, 70 deg `0`; beta: 20 deg `0`, 45 deg `0`, 70 deg `0`; p_hat: 20 deg `0`, 45 deg `0`, 70 deg `0`; r_hat: 20 deg `0`, 45 deg `0`, 70 deg `0`
- post-stall Cl surface: bias: 20 deg `0`, 45 deg `0`, 70 deg `0`; beta: 20 deg `0`, 45 deg `0`, 70 deg `0`; p_hat: 20 deg `0`, 45 deg `0`, 70 deg `0`; r_hat: 20 deg `0`, 45 deg `0`, 70 deg `0`
- post-stall Cn surface: bias: 20 deg `0`, 45 deg `0`, 70 deg `0`; beta: 20 deg `0`, 45 deg `0`, 70 deg `0`; p_hat: 20 deg `0`, 45 deg `0`, 70 deg `0`; r_hat: 20 deg `0`, 45 deg `0`, 70 deg `0`
- post-stall Cmq residual: `0.507254`
- selected compact post-stall replay scale: `1.000`
- transition blender status: `ok`
- transition blender fit group: `transition_before_post_stall`
- transition blender start alpha: `12.000` deg
- transition blender full alpha: `20.000` deg

## Grouped Replay Refinement

- grouped history rows: `0`
- grouped history CSV: `metrics/neutral_aero_residual_group_iteration_history.csv`
- grouped replay refinement disabled

## Regime-Staged Cm Workflow

- staged history rows: `10`
- staged history CSV: `metrics/neutral_aero_residual_cm_stage_history.csv`
- staged decisions:
  - `stage0_baseline` `none`: accepted `True`, held-out pitch `4.683` -> `4.683` deg, altitude-loss `0.3323` -> `0.3323` m
  - `stage1_attached_cm` `attached_pitch_moment`: accepted `False`, held-out pitch `4.683` -> `6.575` deg, altitude-loss `0.3323` -> `0.3899` m
  - `stage2_transition_cm` `transition_pitch_moment`: accepted `False`, held-out pitch `4.683` -> `3.880` deg, altitude-loss `0.3323` -> `0.3461` m
  - `stage3_post_stall_cm` `post_stall_pitch_moment`: accepted `False`, held-out pitch `4.683` -> `4.656` deg, altitude-loss `0.3323` -> `0.3443` m
  - `stage4_post_stall_cmq` `post_stall_pitch_damping`: accepted `False`, held-out pitch `4.683` -> `4.665` deg, altitude-loss `0.3323` -> `0.3425` m
  - `stage5_transition_blend` `transition_blend_start_full`: accepted `False`, held-out pitch `4.683` -> `4.066` deg, altitude-loss `0.3323` -> `0.3265` m
  - `stage6_post_blend_transition_cm` `transition_pitch_moment`: accepted `False`, held-out pitch `4.683` -> `3.880` deg, altitude-loss `0.3323` -> `0.3461` m
  - `stage7_post_blend_post_stall_cm` `post_stall_pitch_moment`: accepted `False`, held-out pitch `4.683` -> `4.656` deg, altitude-loss `0.3323` -> `0.3443` m
  - `stage8_post_blend_post_stall_cmq` `post_stall_pitch_damping`: accepted `False`, held-out pitch `4.683` -> `4.665` deg, altitude-loss `0.3323` -> `0.3425` m
  - `stage9_post_blend_post_stall_lift_drag` `post_stall_lift_drag`: accepted `False`, held-out pitch `4.683` -> `4.603` deg, altitude-loss `0.3323` -> `0.3832` m

## Secondary Lateral Diagnostic

- enabled: `True`
- status: `ok`
- diagnostic coefficients CSV: `metrics/neutral_aero_residual_lateral_diagnostic_coefficients.csv`
- diagnostic policy: longitudinal parameters are frozen at the primary candidate; only `CY_beta`, `Cl_p`, and `Cn_r` may change; lateral-only fitting uses excitation-aware confidence weighting
- attached lateral coupling:
  - CY: bias `0`, beta `0.743097`, p_hat `0`, r_hat `0`
  - Cl: bias `0`, beta `0`, p_hat `0.416993`, r_hat `0`
  - Cn: bias `0`, beta `0`, p_hat `0`, r_hat `0.30171`
- transition-window lateral coupling:
  - CY: bias `0`, beta `0`, p_hat `0`, r_hat `0`
  - Cl: bias `0`, beta `0`, p_hat `0`, r_hat `0`
  - Cn: bias `0`, beta `0`, p_hat `0`, r_hat `0`
- lateral diagnostic held-out dy/roll/yaw MAE: `0.8711` m, `66.398` deg, `58.182` deg
- lateral diagnostic acceptance: `rejected_diagnostic_only`
- lateral diagnostic policy: secondary lateral diagnostic is reportable only if held-out dy, roll, and yaw improve versus the primary longitudinal candidate while dx, altitude loss, sink, and pitch do not degrade
  - dy_mae_m (lateral_improvement_required): primary `0.5929`, diagnostic `0.8711`, delta `0.2782`, pass `False`
  - final_phi_mae_deg (lateral_improvement_required): primary `15.6062`, diagnostic `66.3980`, delta `50.7918`, pass `False`
  - final_psi_mae_deg (lateral_improvement_required): primary `4.0268`, diagnostic `58.1815`, delta `54.1547`, pass `False`
  - dx_mae_m (longitudinal_preservation_required): primary `0.2533`, diagnostic `0.9944`, delta `0.7411`, pass `False`
  - altitude_loss_mae_m (longitudinal_preservation_required): primary `0.3323`, diagnostic `0.6500`, delta `0.3178`, pass `False`
  - sink_mae_m_s (longitudinal_preservation_required): primary `0.2867`, diagnostic `0.5501`, delta `0.2633`, pass `False`
  - final_theta_mae_deg (longitudinal_preservation_required): primary `4.6827`, diagnostic `40.7504`, delta `36.0677`, pass `False`

## Lateral One-Term Ablation

- ablation CSV: `metrics/neutral_aero_residual_lateral_ablation.csv`
- policy: longitudinal candidate is frozen; each diagnostic candidate fits one lateral term and one regime family at a time. If held-out `attached_CY_beta` is accepted, a second pass tests remaining cross-couplings against that side-force baseline.
- acceptance: held-out dy, roll, and yaw must all improve while dx, altitude loss, sink, and pitch stay within practical tolerance.
- held-out ablations tested: `27`
- accepted held-out lateral ablations:
  - `attached_CY_r` vs `primary_longitudinal` coeff `-1.41596` (negative); dy `0.593` -> `0.568` m, roll `15.61` -> `15.56` deg, yaw `4.03` -> `2.98` deg
  - `attached_Cn_beta` vs `primary_longitudinal` coeff `-0.0416012` (negative); dy `0.593` -> `0.590` m, roll `15.61` -> `12.03` deg, yaw `4.03` -> `2.80` deg
  - `transition_Cn_r` vs `primary_longitudinal` coeff `-0.0618688` (negative); dy `0.593` -> `0.590` m, roll `15.61` -> `15.27` deg, yaw `4.03` -> `3.77` deg
- best held-out dy reductions, even if rejected:
  - `post_stall_Cl_p` vs `primary_longitudinal` coeff `0.540468` delta dy `-0.027` m, delta roll `10.82` deg, delta yaw `8.78` deg, reason `rejected_lateral_metrics_not_all_improved`
  - `attached_CY_r` vs `primary_longitudinal` coeff `-1.41596` delta dy `-0.025` m, delta roll `-0.04` deg, delta yaw `-1.04` deg, reason `heldout_lateral_improved_with_longitudinal_tolerance`
  - `transition_Cl_p` vs `primary_longitudinal` coeff `0.406724` delta dy `-0.009` m, delta roll `0.84` deg, delta yaw `0.08` deg, reason `rejected_lateral_metrics_not_all_improved`
  - `transition_Cn_r` vs `primary_longitudinal` coeff `-0.0618688` delta dy `-0.003` m, delta roll `-0.34` deg, delta yaw `-0.26` deg, reason `heldout_lateral_improved_with_longitudinal_tolerance`
  - `attached_Cn_beta` vs `primary_longitudinal` coeff `-0.0416012` delta dy `-0.003` m, delta roll `-3.58` deg, delta yaw `-1.23` deg, reason `heldout_lateral_improved_with_longitudinal_tolerance`

## Compact Joint Sweep

- candidate CSV: `metrics/neutral_aero_residual_joint_sweep_candidates.csv`
- pareto CSV: `metrics/neutral_aero_residual_joint_sweep_pareto.csv`
- selected CSV: `metrics/neutral_aero_residual_joint_sweep_selected.csv`
- policy: from active constants only; signs/ranges are discovered from current residuals, then compact longitudinal/lateral terms are swept jointly with 8-worker replay.
- compact joint sweep not run for this workflow

## 40 ms Joint Pareto Audit

- candidate CSV: `metrics/neutral_aero_residual_joint_pareto_audit_candidates.csv`
- selected CSV: `metrics/neutral_aero_residual_joint_pareto_audit_selected.csv`
- policy: diagnostic held-out replay at the launch-handoff-aligned window; accepted rows must keep longitudinal metrics within balanced tolerance while improving dy, roll, and yaw.
- profile: `heavy`
- audit alignment window: `0.040` s
- candidate rows: `210`
- accepted rows: `41`
- selected accepted Pareto rows: `6`
- accepted Pareto candidates:
  - `jp040h_L05_proposal_stage_9_stage9__X026_ablation_attached_Cn_ablation_post_stall` from `proposal_stage_9_stage9_post_blend_post_stall_lift_dr` + `ablation_attached_Cn_beta__yaw_moment_beta_coeff__s1p25+ablation_post_stall_Cl_r__post_stall_roll_moment_r_hat_r__s0p75`: dx delta `0.0278` m, dy delta `-0.0042` m, pitch delta `-0.329` deg, roll delta `-8.044` deg, yaw delta `-0.266` deg
  - `jp040h_L05_proposal_stage_9_stage9__X029_ablation_post_stall_ablation_attached_Cn` from `proposal_stage_9_stage9_post_blend_post_stall_lift_dr` + `ablation_post_stall_Cl_r__post_stall_roll_moment_r_hat_r__s0p75+ablation_attached_Cn_beta__yaw_moment_beta_coeff__s1p0`: dx delta `0.0211` m, dy delta `-0.0084` m, pitch delta `-0.298` deg, roll delta `-8.065` deg, yaw delta `-0.064` deg
  - `jp040h_L00_proposal_stage_5_stage5__X029_ablation_post_stall_ablation_attached_Cn` from `proposal_stage_5_stage5_transition_blend` + `ablation_post_stall_Cl_r__post_stall_roll_moment_r_hat_r__s0p75+ablation_attached_Cn_beta__yaw_moment_beta_coeff__s1p0`: dx delta `0.0202` m, dy delta `-0.0262` m, pitch delta `-0.209` deg, roll delta `-8.431` deg, yaw delta `-0.306` deg
  - `jp040h_L00_proposal_stage_5_stage5__X031_ablation_attached_Cn_ablation_post_stall` from `proposal_stage_5_stage5_transition_blend` + `ablation_attached_Cn_beta__yaw_moment_beta_coeff__s1p0+ablation_post_stall_Cl_r__post_stall_roll_moment_r_hat_r__s0p5`: dx delta `0.0184` m, dy delta `-0.0246` m, pitch delta `-0.216` deg, roll delta `-6.866` deg, yaw delta `-1.192` deg
  - `jp040h_L00_proposal_stage_5_stage5__X033_ablation_attached_Cn_ablation_post_stall` from `proposal_stage_5_stage5_transition_blend` + `ablation_attached_Cn_beta__yaw_moment_beta_coeff__s0p75+ablation_post_stall_Cl_r__post_stall_roll_moment_r_hat_r__s0p5`: dx delta `0.0127` m, dy delta `-0.0263` m, pitch delta `-0.172` deg, roll delta `-6.573` deg, yaw delta `-0.575` deg
  - `jp040h_L00_proposal_stage_5_stage5__X008_ablation_attached_Cn` from `proposal_stage_5_stage5_transition_blend` + `ablation_attached_Cn_beta__yaw_moment_beta_coeff__s0p75`: dx delta `0.0130` m, dy delta `-0.0243` m, pitch delta `-0.146` deg, roll delta `-2.782` deg, yaw delta `-1.605` deg
- heavy candidate CSV: `metrics/neutral_aero_residual_joint_pareto_heavy_candidates.csv`
- heavy selected CSV: `metrics/neutral_aero_residual_joint_pareto_heavy_selected.csv`
- heavy stage replay CSV: `metrics/neutral_aero_residual_joint_pareto_heavy_stage_replay.csv`
- heavy selected row count: `6`
- heavy stage replay row count: `54`

## Lateral Launch-Correlation Audit

- correlation CSV: `metrics/neutral_aero_residual_lateral_launch_correlation.csv`
- interpretation: strong correlation means the remaining lateral replay error is largely launch-condition dependent, so bad lateral launches should be down-weighted before stronger lateral aerodynamics are promoted.
- strongest held-out correlations for the accepted longitudinal candidate:
  - `dy` residual vs `psi0_deg`: r `-0.908`, slope `-0.153`, n `14`
  - `roll` residual vs `phi0_deg`: r `-0.830`, slope `-1.597`, n `14`
  - `dy` residual vs `p0_rad_s`: r `-0.752`, slope `-1.531`, n `14`
  - `roll` residual vs `p0_rad_s`: r `-0.669`, slope `-14.486`, n `14`
  - `dy` residual vs `v0_m_s`: r `0.657`, slope `1.384`, n `14`
  - `dy` residual vs `phi0_deg`: r `-0.643`, slope `-0.116`, n `14`
  - `roll` residual vs `psi0_deg`: r `-0.383`, slope `-0.689`, n `14`
  - `dy` residual vs `r0_rad_s`: r `-0.318`, slope `-1.063`, n `14`
- interpretation: at least one held-out lateral residual has strong launch-condition correlation; down-weighting contaminated launches is likely safer than promoting stronger lateral aerodynamics.

## Replay Validation

- baseline train pitch MAE: `6.081` deg
- candidate train pitch MAE: `6.081` deg
- baseline held-out pitch MAE: `4.683` deg
- candidate held-out pitch MAE: `4.683` deg
- baseline held-out altitude-loss MAE: `0.3323` m
- candidate held-out altitude-loss MAE: `0.3323` m
- baseline held-out dx MAE: `0.2533` m
- candidate held-out dx MAE: `0.2533` m
- held-out acceptance: `accepted`
- acceptance policy: primary longitudinal candidate must improve or preserve held-out dx, altitude loss, sink, and pitch MAE versus baseline_active; lateral errors are reported but not claim-bearing
  - dx_mae_m: baseline `0.2533`, candidate `0.2533`, delta `0.0000`, pass `True`
  - altitude_loss_mae_m: baseline `0.3323`, candidate `0.3323`, delta `0.0000`, pass `True`
  - sink_mae_m_s: baseline `0.2867`, candidate `0.2867`, delta `0.0000`, pass `True`
  - final_theta_mae_deg: baseline `4.6827`, candidate `4.6827`, delta `0.0000`, pass `True`

## Alignment-Window Sensitivity Replay

- disabled

## Stage Replay Errors

These are sample-aligned replay residuals grouped by the measured Vicon alpha regime. Use them to see whether the candidate fixes attached, transition, or post-stall behaviour; keep the full-throw held-out replay gate as the acceptance criterion.

- train/attached: samples `3339`, throws `82`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.088` m, `0.133` m, `0.086` m, `0.190` m/s, `5.83` deg, `4.89` deg, `2.47` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.088` m, `0.133` m, `0.086` m, `0.190` m/s, `5.83` deg, `4.89` deg, `2.47` deg
- train/transition: samples `9407`, throws `82`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.145` m, `0.214` m, `0.230` m, `0.362` m/s, `9.63` deg, `5.97` deg, `4.13` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.145` m, `0.214` m, `0.230` m, `0.362` m/s, `9.63` deg, `5.97` deg, `4.13` deg
- train/post_stall: samples `1772`, throws `31`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.194` m, `0.362` m, `0.249` m, `0.337` m/s, `12.98` deg, `5.74` deg, `8.16` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.194` m, `0.362` m, `0.249` m, `0.337` m/s, `12.98` deg, `5.74` deg, `8.16` deg
- heldout/attached: samples `604`, throws `14`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.136` m, `0.248` m, `0.155` m, `0.249` m/s, `7.19` deg, `4.36` deg, `1.86` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.136` m, `0.248` m, `0.155` m, `0.249` m/s, `7.19` deg, `4.36` deg, `1.86` deg
- heldout/transition: samples `1630`, throws `14`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.145` m, `0.251` m, `0.277` m, `0.431` m/s, `8.52` deg, `6.55` deg, `3.26` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.145` m, `0.251` m, `0.277` m, `0.431` m/s, `8.52` deg, `6.55` deg, `3.26` deg
- heldout/post_stall: samples `164`, throws `5`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.273` m, `0.241` m, `0.339` m, `0.504` m/s, `13.52` deg, `4.84` deg, `3.93` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.273` m, `0.241` m, `0.339` m, `0.504` m/s, `13.52` deg, `4.84` deg, `3.93` deg

## Candidate Parameters

- legacy scalar post-stall CL residual: baseline `0`, candidate `0`
- legacy scalar post-stall CD residual: baseline `0`, candidate `0`
- legacy global Cm bias: baseline `0`, candidate `0`
- attached Cm bias: baseline `0.113098`, candidate `0.113098`
- transition Cm bias: baseline `0.0571156`, candidate `0.0571156`
- post-stall Cm bias: baseline `0.0758587`, candidate `0.0758587`
- post-stall CL surface: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`
- post-stall CD surface: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`
- post-stall Cm surface: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`
- post-stall Cmq surface: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`
- post-stall CY surface: bias: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`; beta: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`; p_hat: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`; r_hat: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`
- post-stall Cl surface: bias: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`; beta: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`; p_hat: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`; r_hat: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`
- post-stall Cn surface: bias: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`; beta: 20 deg baseline `-0.0187086` -> candidate `-0.0187086`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`; p_hat: 20 deg baseline `-0.0711992` -> candidate `-0.0711992`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`; r_hat: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`
- attached lateral coupling: unchanged
- transition lateral coupling: unchanged
- baseline post-stall Cmq: `4`
- candidate post-stall Cmq: `4`
- baseline residual blend start alpha: `12.000` deg
- candidate residual blend start alpha: `12.000` deg
- baseline residual blend full alpha: `22.000` deg
- candidate residual blend full alpha: `22.000` deg

## Regime Summary

- train/attached: count `3421`, Cm mean `-0.01894`, Cm MAE `0.06069`, CY mean `0.00638`, Cl mean `0.01279`, Cn mean `-0.00566`
- train/transition: count `9407`, Cm mean `-0.03883`, Cm MAE `0.04829`, CY mean `-0.11960`, Cl mean `-0.00739`, Cn mean `-0.00304`
- train/post_stall: count `1772`, Cm mean `0.00035`, Cm MAE `0.04201`, CY mean `-0.41203`, Cl mean `-0.02507`, Cn mean `-0.00563`
- heldout/attached: count `618`, Cm mean `-0.03540`, Cm MAE `0.06672`, CY mean `-0.11487`, Cl mean `0.00926`, Cn mean `-0.00379`
- heldout/transition: count `1630`, Cm mean `-0.04200`, Cm MAE `0.04873`, CY mean `-0.16187`, Cl mean `-0.00971`, Cn mean `-0.00353`
- heldout/post_stall: count `164`, Cm mean `-0.00772`, Cm MAE `0.03652`, CY mean `0.32229`, Cl mean `0.00312`, Cn mean `-0.01758`

## Independent Stage Fit Summary

Transition is split into samples before and after the first post-stall exposure in each throw. This separates real transition-model evidence from transition samples that may already be contaminated by earlier post-stall divergence.

- train/attached: count `3421`, Cm mean `-0.01894`, Cm MAE `0.06069`, Cm fit residual MAE `0.05739`, CY mean `0.00638`, Cl mean `0.01279`, Cn mean `-0.00566`
- train/transition_before_post_stall: count `8183`, Cm mean `-0.04311`, Cm MAE `0.04950`, Cm fit residual MAE `0.03839`, CY mean `-0.10179`, Cl mean `-0.00834`, Cn mean `-0.00319`
- train/transition_after_post_stall: count `1224`, Cm mean `-0.01017`, Cm MAE `0.04020`, Cm fit residual MAE `0.04097`, CY mean `-0.23866`, Cl mean `-0.00103`, Cn mean `-0.00205`
- train/post_stall: count `1772`, Cm mean `0.00035`, Cm MAE `0.04201`, Cm fit residual MAE `0.04171`, CY mean `-0.41203`, Cl mean `-0.02507`, Cn mean `-0.00563`
- heldout/attached: count `618`, Cm mean `-0.03540`, Cm MAE `0.06672`, Cm fit residual MAE `0.06099`, CY mean `-0.11487`, Cl mean `0.00926`, Cn mean `-0.00379`
- heldout/transition_before_post_stall: count `1380`, Cm mean `-0.04871`, Cm MAE `0.05313`, Cm fit residual MAE `0.04017`, CY mean `-0.19388`, Cl mean `-0.01132`, Cn mean `-0.00323`
- heldout/transition_after_post_stall: count `250`, Cm mean `-0.00496`, Cm MAE `0.02444`, Cm fit residual MAE `0.02760`, CY mean `0.01483`, Cl mean `-0.00084`, Cn mean `-0.00520`
- heldout/post_stall: count `164`, Cm mean `-0.00772`, Cm MAE `0.03652`, Cm fit residual MAE `0.03749`, CY mean `0.32229`, Cl mean `0.00312`, Cn mean `-0.01758`

## Interpretation

Accept the primary candidate only for the longitudinal claim-bearing model when held-out dx, altitude loss, sink, and pitch improve or preserve the active baseline. Treat dy, roll, and yaw as reported residual evidence unless the secondary lateral diagnostic improves held-out dy/roll/yaw without damaging those longitudinal metrics.
