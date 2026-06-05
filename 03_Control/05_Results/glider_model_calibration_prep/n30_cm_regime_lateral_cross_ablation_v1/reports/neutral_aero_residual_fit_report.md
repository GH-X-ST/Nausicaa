# Neutral Aero Residual Regime Fit

This run uses only neutral open-loop real throws. It estimates 6-DoF force/moment residuals from Vicon state trajectories, fits a claim-bearing longitudinal candidate by default, reports lateral residuals without claiming accurate lateral SysID, and validates the primary candidate by held-out dry-air replay. The default `cm_regime_staged` workflow fits attached Cm, transition Cm, post-stall Cm/Cmq, transition blend, and optional post-stall CL/CD cleanup in separate held-out-gated stages. The optional secondary lateral diagnostic freezes the longitudinal candidate and fits only `CY_beta`, `Cl_p`, and `Cn_r`; rich transition lateral deltas, post-stall lateral surfaces, and post-stall alpha-RBF longitudinal surfaces are diagnostic-only unless explicitly enabled.

## Rerun Recipe

- source session root: `04_Flight_Test\05_Results\cal\n30`
- alignment window: `0.100` s
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
python 03_Control/02_Inner_Loop/run_fit_neutral_aero_residual_calibration.py --session-root 04_Flight_Test\05_Results\cal\n30 --run-label n30_cm_regime_lateral_cross_ablation_v1 --heldout-count 11 --heldout-seed 606 --alignment-window-s 0.1 --derivative-window-s 0.04 --replay-dt-s 0.005 --ridge-lambda 0.001 --min-speed-m-s 1.5 --workers 8 --fit-workflow cm_regime_staged --group-iterations 3 --group-improvement-tol 0.001 --aligned-u-min-m-s 3 --aligned-u-max-m-s 8 --aligned-v-abs-max-m-s 1.5 --aligned-w-abs-max-m-s 0.9 --filter-aligned-launch-state --no-apply-attached-cm-bias --no-fit-transition-pitch-moment --fit-post-stall-lift-drag --fit-post-stall-pitch-moment --fit-post-stall-longitudinal --fit-transition-blender --no-fit-post-stall-surfaces --fit-post-stall-damping --no-fit-attached-lateral-coupling --no-fit-transition-lateral-coupling --no-fit-lateral-surfaces --fit-secondary-lateral-diagnostic
```

## Aligned Launch Filter

- loaded logged-valid throws: `105`
- kept throws after replay-start filter: `74`
- filtered throws: `31`
- filter audit CSV: `metrics/neutral_aero_residual_filtered_throws.csv`
- launch-confidence weighting: enabled for residual coefficient fitting
- confidence reference: replay-aligned lateral contamination `phi0=psi0=v0=p0=r0=0`; minimum weight `0.25`
- kept-throw confidence weight min/mean/max: `0.5270482418`, `0.7473873516`, `0.9604324303`
- kept-throw lateral score min/mean/max: `0.1640561848`, `0.4358949077`, `0.6534335948`
- rejected logged-valid throws:
  - `20260604_185934/v001`: aligned_w_outside_launch_gate; u0 `4.543059555`, v0 `0.3501702742`, w0 `1.008779882` m/s
  - `20260604_185934/v006`: aligned_v_outside_launch_gate; u0 `5.623978617`, v0 `1.917732218`, w0 `0.5411629165` m/s
  - `20260604_185934/v008`: aligned_u_outside_replay_filter; u0 `0.9721909615`, v0 `0.2445794314`, w0 `0.1511750445` m/s
  - `20260604_185934/v010`: aligned_w_outside_launch_gate; u0 `5.149037042`, v0 `1.238403229`, w0 `1.018297939` m/s
  - `20260604_190407/v002`: aligned_w_outside_launch_gate; u0 `4.97722127`, v0 `0.7757127111`, w0 `0.9282428727` m/s
  - `20260604_190407/v005`: aligned_w_outside_launch_gate; u0 `5.929172756`, v0 `0.609645104`, w0 `0.9622552935` m/s
  - `20260604_191406/v003`: aligned_w_outside_launch_gate; u0 `4.947658717`, v0 `1.138830854`, w0 `1.137633454` m/s
  - `20260604_191406/v008`: aligned_v_outside_launch_gate; u0 `5.288296354`, v0 `1.521964301`, w0 `0.4222859752` m/s
  - `20260604_191406/v009`: aligned_w_outside_launch_gate; u0 `5.790927828`, v0 `0.8098222082`, w0 `1.061122725` m/s
  - `20260604_195702/v001`: aligned_v_outside_launch_gate; u0 `4.468433221`, v0 `1.652145915`, w0 `0.3445464762` m/s
  - `20260604_195702/v006`: aligned_v_outside_launch_gate; u0 `5.305858468`, v0 `1.50640443`, w0 `0.5582058265` m/s
  - `20260604_203516/v004`: aligned_v_outside_launch_gate; u0 `5.300256981`, v0 `1.502123524`, w0 `0.3933348901` m/s
  - `20260604_203516/v014`: aligned_v_outside_launch_gate; u0 `5.407679143`, v0 `1.532947734`, w0 `0.7194391593` m/s
  - `20260604_203516/v019`: aligned_w_outside_launch_gate; u0 `5.076755044`, v0 `0.4722628657`, w0 `0.927494506` m/s
  - `20260604_203516/v028`: aligned_w_outside_launch_gate; u0 `5.437784542`, v0 `1.36808424`, w0 `0.974330837` m/s
  - `20260604_203516/v030`: aligned_w_outside_launch_gate; u0 `4.944508884`, v0 `0.8920286267`, w0 `0.9275182519` m/s
  - `20260604_210642/v004`: aligned_v_outside_launch_gate; u0 `5.115376591`, v0 `1.601200356`, w0 `0.3254876519` m/s
  - `20260604_210642/v005`: aligned_w_outside_launch_gate; u0 `5.07228044`, v0 `0.4521471936`, w0 `0.9458010087` m/s
  - `20260604_210642/v010`: aligned_w_outside_launch_gate; u0 `5.472734486`, v0 `1.065714728`, w0 `1.052461085` m/s
  - `20260604_210642/v011`: aligned_v_outside_launch_gate; u0 `5.404006321`, v0 `1.574461375`, w0 `0.6176215087` m/s
  - `20260604_210642/v012`: aligned_w_outside_launch_gate; u0 `5.232525824`, v0 `1.391064124`, w0 `0.907277047` m/s
  - `20260604_210642/v016`: aligned_w_outside_launch_gate; u0 `4.846783329`, v0 `1.252923758`, w0 `0.9696971187` m/s
  - `20260604_210642/v017`: aligned_w_outside_launch_gate; u0 `5.150762971`, v0 `0.6391571858`, w0 `1.095460686` m/s
  - `20260604_210642/v018`: aligned_w_outside_launch_gate; u0 `4.589658975`, v0 `1.207842316`, w0 `0.931583461` m/s
  - `20260604_210642/v022`: aligned_w_outside_launch_gate; u0 `5.431260865`, v0 `1.010952956`, w0 `0.9711440148` m/s
  - `20260604_210642/v025`: aligned_v_outside_launch_gate; u0 `5.527437353`, v0 `1.623930224`, w0 `0.7267628261` m/s
  - `20260604_210642/v027`: aligned_w_outside_launch_gate; u0 `5.934935939`, v0 `0.7867333274`, w0 `1.309058612` m/s
  - `20260604_210642/v028`: aligned_w_outside_launch_gate; u0 `4.749414251`, v0 `0.7903720488`, w0 `1.009358567` m/s
  - `20260605_000625/v001`: aligned_u_outside_replay_filter; u0 `2.052399304`, v0 `0.109984173`, w0 `0.4930914769` m/s
  - `20260605_000625/v002`: aligned_w_outside_launch_gate; u0 `3.92907431`, v0 `0.4403444132`, w0 `0.9397252277` m/s
  - `20260605_000625/v005`: aligned_w_outside_launch_gate; u0 `4.529400173`, v0 `0.3394290587`, w0 `1.024652453` m/s

## Coefficient Fit

- fit status: `ok`
- sample count: `10560`
- used sample count: `10560`
- post-stall used sample count: `876`
- post-stall fit profile: `compact_scalar_activation`
- fit MAE in Cm: `0.03738`
- attached Cm residual: `0.0753989`
- transition Cm residual before post-stall: `0.0373137`
- transition Cm residual after post-stall: `0.0505457`
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
- post-stall Cmq residual: `3.85874`
- selected compact post-stall replay scale: `1.000`
- transition blender status: `ok`
- transition blender fit group: `transition_before_post_stall`
- transition blender start alpha: `11.000` deg
- transition blender full alpha: `20.000` deg

## Grouped Replay Refinement

- grouped history rows: `0`
- grouped history CSV: `metrics/neutral_aero_residual_group_iteration_history.csv`
- grouped replay refinement disabled

## Regime-Staged Cm Workflow

- staged history rows: `10`
- staged history CSV: `metrics/neutral_aero_residual_cm_stage_history.csv`
- staged decisions:
  - `stage0_baseline` `none`: accepted `True`, held-out pitch `20.505` -> `20.505` deg, altitude-loss `1.1807` -> `1.1807` m
  - `stage1_attached_cm` `attached_pitch_moment`: accepted `True`, held-out pitch `20.505` -> `7.778` deg, altitude-loss `1.1807` -> `0.3842` m
  - `stage2_transition_cm` `transition_pitch_moment`: accepted `True`, held-out pitch `7.778` -> `7.668` deg, altitude-loss `0.3842` -> `0.3814` m
  - `stage3_post_stall_cm` `post_stall_pitch_moment`: accepted `False`, held-out pitch `7.668` -> `7.613` deg, altitude-loss `0.3814` -> `0.3868` m
  - `stage4_post_stall_cmq` `post_stall_pitch_damping`: accepted `False`, held-out pitch `7.668` -> `7.528` deg, altitude-loss `0.3814` -> `0.4400` m
  - `stage5_transition_blend` `transition_blend_start_full`: accepted `True`, held-out pitch `7.668` -> `5.437` deg, altitude-loss `0.3814` -> `0.3435` m
  - `stage6_post_blend_transition_cm` `transition_pitch_moment`: accepted `False`, held-out pitch `5.437` -> `5.362` deg, altitude-loss `0.3435` -> `0.3544` m
  - `stage7_post_blend_post_stall_cm` `post_stall_pitch_moment`: accepted `False`, held-out pitch `5.437` -> `5.132` deg, altitude-loss `0.3435` -> `0.3488` m
  - `stage8_post_blend_post_stall_cmq` `post_stall_pitch_damping`: accepted `False`, held-out pitch `5.437` -> `5.214` deg, altitude-loss `0.3435` -> `0.4011` m
  - `stage9_post_blend_post_stall_lift_drag` `post_stall_lift_drag`: accepted `False`, held-out pitch `5.437` -> `5.474` deg, altitude-loss `0.3435` -> `0.3662` m

## Secondary Lateral Diagnostic

- enabled: `True`
- status: `ok`
- diagnostic coefficients CSV: `metrics/neutral_aero_residual_lateral_diagnostic_coefficients.csv`
- diagnostic policy: longitudinal parameters are frozen at the primary candidate; only `CY_beta`, `Cl_p`, and `Cn_r` may change; launch-confidence weighting is ignored for this lateral-only fit
- attached lateral coupling:
  - CY: bias `0`, beta `-1.18595`, p_hat `0`, r_hat `0`
  - Cl: bias `0`, beta `0`, p_hat `0.25406`, r_hat `0`
  - Cn: bias `0`, beta `0`, p_hat `0`, r_hat `0.290767`
- transition-window lateral coupling:
  - CY: bias `0`, beta `0`, p_hat `0`, r_hat `0`
  - Cl: bias `0`, beta `0`, p_hat `0`, r_hat `0`
  - Cn: bias `0`, beta `0`, p_hat `0`, r_hat `0`
- lateral diagnostic held-out dy/roll/yaw MAE: `1.0581` m, `29.195` deg, `58.748` deg
- lateral diagnostic acceptance: `rejected_diagnostic_only`
- lateral diagnostic policy: secondary lateral diagnostic is reportable only if held-out dy, roll, and yaw improve versus the primary longitudinal candidate while dx, altitude loss, sink, and pitch do not degrade
  - dy_mae_m (lateral_improvement_required): primary `1.6122`, diagnostic `1.0581`, delta `-0.5542`, pass `True`
  - final_phi_mae_deg (lateral_improvement_required): primary `28.1574`, diagnostic `29.1952`, delta `1.0378`, pass `False`
  - final_psi_mae_deg (lateral_improvement_required): primary `27.0982`, diagnostic `58.7480`, delta `31.6497`, pass `False`
  - dx_mae_m (longitudinal_preservation_required): primary `0.4075`, diagnostic `0.6007`, delta `0.1931`, pass `False`
  - altitude_loss_mae_m (longitudinal_preservation_required): primary `0.3435`, diagnostic `0.5540`, delta `0.2106`, pass `False`
  - sink_mae_m_s (longitudinal_preservation_required): primary `0.3089`, diagnostic `0.4931`, delta `0.1842`, pass `False`
  - final_theta_mae_deg (longitudinal_preservation_required): primary `5.4371`, diagnostic `37.1170`, delta `31.6800`, pass `False`

## Lateral One-Term Ablation

- ablation CSV: `metrics/neutral_aero_residual_lateral_ablation.csv`
- policy: longitudinal candidate is frozen; each diagnostic candidate fits one lateral term and one regime family at a time. If held-out `attached_CY_beta` is accepted, a second pass tests remaining cross-couplings against that side-force baseline.
- acceptance: held-out dy, roll, and yaw must all improve while dx, altitude loss, sink, and pitch stay within practical tolerance.
- held-out ablations tested: `51`
- accepted held-out lateral ablations:
  - `attached_CY_beta` vs `primary_longitudinal` coeff `-1.18595` (negative); dy `1.612` -> `0.805` m, roll `28.16` -> `23.56` deg, yaw `27.10` -> `10.64` deg
  - `transition_CY_r` vs `primary_longitudinal` coeff `-3` (negative); dy `1.612` -> `1.308` m, roll `28.16` -> `27.98` deg, yaw `27.10` -> `17.21` deg
  - `post_stall_CY_r` vs `primary_longitudinal` coeff `-3.19933` (negative); dy `1.612` -> `1.340` m, roll `28.16` -> `28.14` deg, yaw `27.10` -> `18.84` deg
  - `attached_Cn_beta` vs `primary_longitudinal` coeff `-0.0443444` (negative); dy `1.612` -> `0.956` m, roll `28.16` -> `16.07` deg, yaw `27.10` -> `11.50` deg
  - `transition_Cn_beta` vs `primary_longitudinal` coeff `-0.0377679` (negative); dy `1.612` -> `1.544` m, roll `28.16` -> `23.95` deg, yaw `27.10` -> `21.55` deg
  - `post_stall_Cn_beta` vs `primary_longitudinal` coeff `-0.0460636` (negative); dy `1.612` -> `1.508` m, roll `28.16` -> `25.78` deg, yaw `27.10` -> `21.53` deg
  - `transition_Cn_p` vs `primary_longitudinal` coeff `-0.134659` (negative); dy `1.612` -> `1.595` m, roll `28.16` -> `26.49` deg, yaw `27.10` -> `24.40` deg
  - `after_CY_beta_transition_CY_r` vs `attached_CY_beta` coeff `-3` (negative); dy `0.805` -> `0.695` m, roll `23.56` -> `22.93` deg, yaw `10.64` -> `7.82` deg
  - `after_CY_beta_transition_Cn_p` vs `attached_CY_beta` coeff `-0.134659` (negative); dy `0.805` -> `0.786` m, roll `23.56` -> `22.35` deg, yaw `10.64` -> `9.27` deg
  - `after_CY_beta_post_stall_Cn_p` vs `attached_CY_beta` coeff `-0.0470512` (negative); dy `0.805` -> `0.802` m, roll `23.56` -> `23.33` deg, yaw `10.64` -> `10.28` deg
- best held-out dy reductions, even if rejected:
  - `attached_Cl_r` vs `primary_longitudinal` coeff `-0.586558` delta dy `-0.852` m, delta roll `-13.75` deg, delta yaw `-17.53` deg, reason `rejected_longitudinal_metrics_degraded`
  - `attached_CY_beta` vs `primary_longitudinal` coeff `-1.18595` delta dy `-0.807` m, delta roll `-4.60` deg, delta yaw `-16.46` deg, reason `heldout_lateral_improved_with_longitudinal_tolerance`
  - `attached_Cn_beta` vs `primary_longitudinal` coeff `-0.0443444` delta dy `-0.656` m, delta roll `-12.09` deg, delta yaw `-15.60` deg, reason `heldout_lateral_improved_with_longitudinal_tolerance`
  - `transition_Cl_r` vs `primary_longitudinal` coeff `-0.768761` delta dy `-0.469` m, delta roll `-18.62` deg, delta yaw `-11.09` deg, reason `rejected_longitudinal_metrics_degraded`
  - `post_stall_Cl_r` vs `primary_longitudinal` coeff `-0.758373` delta dy `-0.385` m, delta roll `-14.16` deg, delta yaw `-10.85` deg, reason `rejected_longitudinal_metrics_degraded`

## Lateral Launch-Correlation Audit

- correlation CSV: `metrics/neutral_aero_residual_lateral_launch_correlation.csv`
- interpretation: strong correlation means the remaining lateral replay error is largely launch-condition dependent, so bad lateral launches should be down-weighted before stronger lateral aerodynamics are promoted.
- strongest held-out correlations for the accepted longitudinal candidate:
  - `dy` residual vs `phi0_deg`: r `-0.830`, slope `-0.170`, n `11`
  - `roll` residual vs `p0_rad_s`: r `-0.707`, slope `-70.420`, n `11`
  - `roll` residual vs `psi0_deg`: r `-0.704`, slope `-4.078`, n `11`
  - `yaw` residual vs `p0_rad_s`: r `-0.620`, slope `-58.079`, n `11`
  - `yaw` residual vs `psi0_deg`: r `-0.518`, slope `-2.823`, n `11`
  - `dy` residual vs `psi0_deg`: r `-0.497`, slope `-0.125`, n `11`
  - `dy` residual vs `p0_rad_s`: r `-0.471`, slope `-2.031`, n `11`
  - `roll` residual vs `v0_m_s`: r `0.442`, slope `25.542`, n `11`
- interpretation: at least one held-out lateral residual has strong launch-condition correlation; down-weighting contaminated launches is likely safer than promoting stronger lateral aerodynamics.

## Replay Validation

- baseline train pitch MAE: `22.878` deg
- candidate train pitch MAE: `5.822` deg
- baseline held-out pitch MAE: `20.505` deg
- candidate held-out pitch MAE: `5.437` deg
- baseline held-out altitude-loss MAE: `1.1807` m
- candidate held-out altitude-loss MAE: `0.3435` m
- baseline held-out dx MAE: `0.9444` m
- candidate held-out dx MAE: `0.4075` m
- held-out acceptance: `accepted`
- acceptance policy: primary longitudinal candidate must improve or preserve held-out dx, altitude loss, sink, and pitch MAE versus baseline_active; lateral errors are reported but not claim-bearing
  - dx_mae_m: baseline `0.9444`, candidate `0.4075`, delta `-0.5368`, pass `True`
  - altitude_loss_mae_m: baseline `1.1807`, candidate `0.3435`, delta `-0.8372`, pass `True`
  - sink_mae_m_s: baseline `1.0739`, candidate `0.3089`, delta `-0.7650`, pass `True`
  - final_theta_mae_deg: baseline `20.5054`, candidate `5.4371`, delta `-15.0683`, pass `True`

## Stage Replay Errors

These are sample-aligned replay residuals grouped by the measured Vicon alpha regime. Use them to see whether the candidate fixes attached, transition, or post-stall behaviour; keep the full-throw held-out replay gate as the acceptance criterion.

- train/attached: samples `2248`, throws `63`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.251` m, `0.417` m, `0.365` m, `0.490` m/s, `6.36` deg, `10.99` deg, `5.00` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.148` m, `0.487` m, `0.144` m, `0.255` m/s, `9.70` deg, `4.63` deg, `8.31` deg
- train/transition: samples `7358`, throws `63`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.324` m, `0.497` m, `0.473` m, `0.696` m/s, `8.13` deg, `19.83` deg, `8.14` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.182` m, `0.561` m, `0.231` m, `0.382` m/s, `13.08` deg, `7.66` deg, `11.50` deg
- train/post_stall: samples `891`, throws `19`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.548` m, `0.809` m, `0.520` m, `0.669` m/s, `11.03` deg, `20.21` deg, `8.66` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.308` m, `0.913` m, `0.198` m, `0.290` m/s, `14.00` deg, `7.84` deg, `9.74` deg
- heldout/attached: samples `394`, throws `11`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.208` m, `0.321` m, `0.267` m, `0.447` m/s, `4.02` deg, `10.20` deg, `2.19` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.088` m, `0.402` m, `0.087` m, `0.243` m/s, `9.34` deg, `3.57` deg, `6.92` deg
- heldout/transition: samples `1260`, throws `11`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.408` m, `0.583` m, `0.506` m, `0.763` m/s, `9.09` deg, `19.92` deg, `8.33` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.172` m, `0.686` m, `0.227` m, `0.393` m/s, `17.77` deg, `5.78` deg, `13.84` deg
- heldout/post_stall: samples `251`, throws `5`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.632` m, `0.730` m, `0.672` m, `0.917` m/s, `9.60` deg, `19.43` deg, `7.23` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.362` m, `0.851` m, `0.357` m, `0.526` m/s, `22.68` deg, `3.94` deg, `11.64` deg

## Candidate Parameters

- legacy scalar post-stall CL residual: baseline `0`, candidate `0`
- legacy scalar post-stall CD residual: baseline `0`, candidate `0`
- legacy global Cm bias: baseline `0`, candidate `0`
- attached Cm bias: baseline `0`, candidate `0.0753989`
- transition Cm bias: baseline `0`, candidate `0.0039978`
- post-stall Cm bias: baseline `0`, candidate `0`
- post-stall CL surface: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`
- post-stall CD surface: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`
- post-stall Cm surface: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`
- post-stall Cmq surface: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`
- post-stall CY surface: bias: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`; beta: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`; p_hat: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`; r_hat: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`
- post-stall Cl surface: bias: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`; beta: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`; p_hat: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`; r_hat: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`
- post-stall Cn surface: bias: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`; beta: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`; p_hat: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`; r_hat: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`
- attached lateral coupling: unchanged
- transition lateral coupling: unchanged
- baseline post-stall Cmq: `0`
- candidate post-stall Cmq: `0`
- baseline residual blend start alpha: `12.000` deg
- candidate residual blend start alpha: `14.000` deg
- baseline residual blend full alpha: `20.000` deg
- candidate residual blend full alpha: `22.000` deg

## Regime Summary

- train/attached: count `2311`, Cm mean `0.07198`, Cm MAE `0.07753`, CY mean `-0.16803`, Cl mean `0.00905`, Cn mean `-0.00440`
- train/transition: count `7358`, Cm mean `0.03431`, Cm MAE `0.04557`, CY mean `-0.06124`, Cl mean `-0.00868`, Cn mean `0.00318`
- train/post_stall: count `891`, Cm mean `-0.00147`, Cm MAE `0.05516`, CY mean `-0.07397`, Cl mean `-0.02329`, Cn mean `0.00823`
- heldout/attached: count `405`, Cm mean `0.07171`, Cm MAE `0.07885`, CY mean `-0.23805`, Cl mean `0.01051`, Cn mean `-0.00242`
- heldout/transition: count `1260`, Cm mean `0.03244`, Cm MAE `0.04527`, CY mean `-0.05825`, Cl mean `-0.01061`, Cn mean `0.00573`
- heldout/post_stall: count `251`, Cm mean `-0.06016`, Cm MAE `0.07134`, CY mean `0.08834`, Cl mean `-0.03519`, Cn mean `0.00034`

## Independent Stage Fit Summary

Transition is split into samples before and after the first post-stall exposure in each throw. This separates real transition-model evidence from transition samples that may already be contaminated by earlier post-stall divergence.

- train/attached: count `2311`, Cm mean `0.07198`, Cm MAE `0.07753`, Cm fit residual MAE `0.04699`, CY mean `-0.16803`, Cl mean `0.00905`, Cn mean `-0.00440`
- train/transition_before_post_stall: count `6520`, Cm mean `0.03379`, Cm MAE `0.04399`, Cm fit residual MAE `0.03366`, CY mean `-0.07866`, Cl mean `-0.00916`, Cn mean `0.00330`
- train/transition_after_post_stall: count `838`, Cm mean `0.03835`, Cm MAE `0.05787`, Cm fit residual MAE `0.03846`, CY mean `0.07426`, Cl mean `-0.00494`, Cn mean `0.00221`
- train/post_stall: count `891`, Cm mean `-0.00147`, Cm MAE `0.05516`, Cm fit residual MAE `0.03876`, CY mean `-0.07397`, Cl mean `-0.02329`, Cn mean `0.00823`
- heldout/attached: count `405`, Cm mean `0.07171`, Cm MAE `0.07885`, Cm fit residual MAE `0.05323`, CY mean `-0.23805`, Cl mean `0.01051`, Cn mean `-0.00242`
- heldout/transition_before_post_stall: count `1077`, Cm mean `0.02897`, Cm MAE `0.04179`, Cm fit residual MAE `0.03535`, CY mean `-0.07988`, Cl mean `-0.01269`, Cn mean `0.00372`
- heldout/transition_after_post_stall: count `183`, Cm mean `0.05292`, Cm MAE `0.06572`, Cm fit residual MAE `0.05824`, CY mean `0.06906`, Cl mean `0.00158`, Cn mean `0.01756`
- heldout/post_stall: count `251`, Cm mean `-0.06016`, Cm MAE `0.07134`, Cm fit residual MAE `0.03680`, CY mean `0.08834`, Cl mean `-0.03519`, Cn mean `0.00034`

## Interpretation

Accept the primary candidate only for the longitudinal claim-bearing model when held-out dx, altitude loss, sink, and pitch improve or preserve the active baseline. Treat dy, roll, and yaw as reported residual evidence unless the secondary lateral diagnostic improves held-out dy/roll/yaw without damaging those longitudinal metrics.
