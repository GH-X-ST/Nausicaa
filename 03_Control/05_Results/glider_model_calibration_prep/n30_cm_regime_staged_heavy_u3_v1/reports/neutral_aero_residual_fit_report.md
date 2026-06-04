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
python 03_Control/02_Inner_Loop/run_fit_neutral_aero_residual_calibration.py --session-root 04_Flight_Test\05_Results\cal\n30 --run-label n30_cm_regime_staged_heavy_u3_v1 --heldout-count 11 --heldout-seed 606 --alignment-window-s 0.1 --derivative-window-s 0.04 --replay-dt-s 0.005 --ridge-lambda 0.001 --min-speed-m-s 1.5 --workers 8 --fit-workflow cm_regime_staged --group-iterations 3 --group-improvement-tol 0.001 --aligned-u-min-m-s 3 --aligned-u-max-m-s 8 --aligned-v-abs-max-m-s 1.5 --aligned-w-abs-max-m-s 0.9 --filter-aligned-launch-state --no-apply-attached-cm-bias --no-fit-transition-pitch-moment --fit-post-stall-lift-drag --fit-post-stall-pitch-moment --fit-post-stall-longitudinal --fit-transition-blender --no-fit-post-stall-surfaces --fit-post-stall-damping --no-fit-attached-lateral-coupling --no-fit-transition-lateral-coupling --no-fit-lateral-surfaces --fit-secondary-lateral-diagnostic
```

## Aligned Launch Filter

- loaded logged-valid throws: `100`
- kept throws after replay-start filter: `72`
- filtered throws: `28`
- filter audit CSV: `metrics/neutral_aero_residual_filtered_throws.csv`
- launch-confidence weighting: enabled for residual coefficient fitting
- confidence reference: replay-aligned lateral contamination `phi0=psi0=v0=p0=r0=0`; minimum weight `0.25`
- kept-throw confidence weight min/mean/max: `0.5270482418`, `0.742802716`, `0.931118568`
- kept-throw lateral score min/mean/max: `0.2181263461`, `0.4413962271`, `0.6534335948`
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

## Coefficient Fit

- fit status: `ok`
- sample count: `10215`
- used sample count: `10215`
- post-stall used sample count: `876`
- post-stall fit profile: `compact_scalar_activation`
- fit MAE in Cm: `0.03768`
- attached Cm residual: `0.0761363`
- transition Cm residual before post-stall: `0.0367993`
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

- staged history rows: `7`
- staged history CSV: `metrics/neutral_aero_residual_cm_stage_history.csv`
- staged decisions:
  - `stage0_baseline` `none`: accepted `True`, held-out pitch `17.424` -> `17.424` deg, altitude-loss `1.0472` -> `1.0472` m
  - `stage1_attached_cm` `attached_pitch_moment`: accepted `True`, held-out pitch `17.424` -> `7.794` deg, altitude-loss `1.0472` -> `0.4124` m
  - `stage2_transition_cm` `transition_pitch_moment`: accepted `False`, held-out pitch `7.794` -> `7.696` deg, altitude-loss `0.4124` -> `0.4095` m
  - `stage3_post_stall_cm` `post_stall_pitch_moment`: accepted `False`, held-out pitch `7.794` -> `7.733` deg, altitude-loss `0.4124` -> `0.4183` m
  - `stage4_post_stall_cmq` `post_stall_pitch_damping`: accepted `False`, held-out pitch `7.794` -> `7.492` deg, altitude-loss `0.4124` -> `0.4695` m
  - `stage5_transition_blend` `transition_blend_start_full`: accepted `True`, held-out pitch `7.794` -> `5.817` deg, altitude-loss `0.4124` -> `0.3648` m
  - `stage6_post_stall_lift_drag` `post_stall_lift_drag`: accepted `False`, held-out pitch `5.817` -> `6.336` deg, altitude-loss `0.3648` -> `0.3920` m

## Secondary Lateral Diagnostic

- enabled: `True`
- status: `ok`
- diagnostic coefficients CSV: `metrics/neutral_aero_residual_lateral_diagnostic_coefficients.csv`
- diagnostic policy: longitudinal parameters are frozen at the primary candidate; only `CY_beta`, `Cl_p`, and `Cn_r` may change; launch-confidence weighting is ignored for this lateral-only fit
- attached lateral coupling:
  - CY: bias `0`, beta `-1.19521`, p_hat `0`, r_hat `0`
  - Cl: bias `0`, beta `0`, p_hat `0.258558`, r_hat `0`
  - Cn: bias `0`, beta `0`, p_hat `0`, r_hat `0.291217`
- transition-window lateral coupling:
  - CY: bias `0`, beta `0`, p_hat `0`, r_hat `0`
  - Cl: bias `0`, beta `0`, p_hat `0`, r_hat `0`
  - Cn: bias `0`, beta `0`, p_hat `0`, r_hat `0`
- lateral diagnostic held-out dy/roll/yaw MAE: `1.0692` m, `23.580` deg, `55.666` deg
- lateral diagnostic acceptance: `rejected_diagnostic_only`
- lateral diagnostic policy: secondary lateral diagnostic is reportable only if held-out dy, roll, and yaw improve versus the primary longitudinal candidate while dx, altitude loss, sink, and pitch do not degrade
  - dy_mae_m (lateral_improvement_required): primary `1.7024`, diagnostic `1.0692`, delta `-0.6331`, pass `True`
  - final_phi_mae_deg (lateral_improvement_required): primary `28.7007`, diagnostic `23.5800`, delta `-5.1208`, pass `True`
  - final_psi_mae_deg (lateral_improvement_required): primary `29.3575`, diagnostic `55.6658`, delta `26.3083`, pass `False`
  - dx_mae_m (longitudinal_preservation_required): primary `0.4100`, diagnostic `0.6487`, delta `0.2387`, pass `False`
  - altitude_loss_mae_m (longitudinal_preservation_required): primary `0.3648`, diagnostic `0.6110`, delta `0.2462`, pass `False`
  - sink_mae_m_s (longitudinal_preservation_required): primary `0.3265`, diagnostic `0.5396`, delta `0.2131`, pass `False`
  - final_theta_mae_deg (longitudinal_preservation_required): primary `5.8167`, diagnostic `34.9429`, delta `29.1262`, pass `False`

## Replay Validation

- baseline train pitch MAE: `23.161` deg
- candidate train pitch MAE: `5.884` deg
- baseline held-out pitch MAE: `17.424` deg
- candidate held-out pitch MAE: `5.817` deg
- baseline held-out altitude-loss MAE: `1.0472` m
- candidate held-out altitude-loss MAE: `0.3648` m
- baseline held-out dx MAE: `0.9325` m
- candidate held-out dx MAE: `0.4100` m
- held-out acceptance: `accepted`
- acceptance policy: primary longitudinal candidate must improve or preserve held-out dx, altitude loss, sink, and pitch MAE versus baseline_active; lateral errors are reported but not claim-bearing
  - dx_mae_m: baseline `0.9325`, candidate `0.4100`, delta `-0.5225`, pass `True`
  - altitude_loss_mae_m: baseline `1.0472`, candidate `0.3648`, delta `-0.6824`, pass `True`
  - sink_mae_m_s: baseline `0.9252`, candidate `0.3265`, delta `-0.5987`, pass `True`
  - final_theta_mae_deg: baseline `17.4239`, candidate `5.8167`, delta `-11.6072`, pass `True`

## Stage Replay Errors

These are sample-aligned replay residuals grouped by the measured Vicon alpha regime. Use them to see whether the candidate fixes attached, transition, or post-stall behaviour; keep the full-throw held-out replay gate as the acceptance criterion.

- train/attached: samples `2211`, throws `61`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.254` m, `0.423` m, `0.370` m, `0.496` m/s, `6.43` deg, `11.12` deg, `5.03` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.149` m, `0.493` m, `0.145` m, `0.256` m/s, `9.72` deg, `4.64` deg, `8.36` deg
- train/transition: samples `7052`, throws `61`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.332` m, `0.504` m, `0.486` m, `0.715` m/s, `8.17` deg, `20.18` deg, `8.04` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.189` m, `0.569` m, `0.234` m, `0.386` m/s, `12.92` deg, `7.84` deg, `11.40` deg
- train/post_stall: samples `891`, throws `19`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.548` m, `0.809` m, `0.520` m, `0.669` m/s, `11.03` deg, `20.21` deg, `8.66` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.306` m, `0.913` m, `0.197` m, `0.289` m/s, `14.09` deg, `7.87` deg, `9.75` deg
- heldout/attached: samples `354`, throws `11`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.213` m, `0.332` m, `0.210` m, `0.372` m/s, `4.50` deg, `8.01` deg, `2.29` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.082` m, `0.416` m, `0.085` m, `0.246` m/s, `8.95` deg, `3.80` deg, `7.28` deg
- heldout/transition: samples `1363`, throws `11`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.384` m, `0.594` m, `0.455` m, `0.680` m/s, `9.54` deg, `17.99` deg, `9.21` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.160` m, `0.695` m, `0.231` m, `0.397` m/s, `18.53` deg, `5.64` deg, `14.60` deg
- heldout/post_stall: samples `251`, throws `5`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.632` m, `0.730` m, `0.672` m, `0.917` m/s, `9.60` deg, `19.43` deg, `7.23` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.360` m, `0.851` m, `0.356` m, `0.526` m/s, `22.82` deg, `3.87` deg, `11.63` deg

## Candidate Parameters

- legacy scalar post-stall CL residual: baseline `0`, candidate `0`
- legacy scalar post-stall CD residual: baseline `0`, candidate `0`
- legacy global Cm bias: baseline `0`, candidate `0`
- attached Cm bias: baseline `0`, candidate `0.0761363`
- transition Cm bias: baseline `0`, candidate `0`
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

- train/attached: count `2272`, Cm mean `0.07217`, Cm MAE `0.07782`, CY mean `-0.16914`, Cl mean `0.00908`, Cn mean `-0.00424`
- train/transition: count `7052`, Cm mean `0.03374`, Cm MAE `0.04518`, CY mean `-0.06331`, Cl mean `-0.00906`, Cn mean `0.00348`
- train/post_stall: count `891`, Cm mean `-0.00147`, Cm MAE `0.05516`, CY mean `-0.07397`, Cl mean `-0.02329`, Cn mean `0.00823`
- heldout/attached: count `365`, Cm mean `0.07264`, Cm MAE `0.08056`, CY mean `-0.25583`, Cl mean `0.01212`, Cn mean `-0.00405`
- heldout/transition: count `1363`, Cm mean `0.03371`, Cm MAE `0.04633`, CY mean `-0.05947`, Cl mean `-0.01001`, Cn mean `0.00477`
- heldout/post_stall: count `251`, Cm mean `-0.06016`, Cm MAE `0.07134`, CY mean `0.08834`, Cl mean `-0.03519`, Cn mean `0.00034`

## Independent Stage Fit Summary

Transition is split into samples before and after the first post-stall exposure in each throw. This separates real transition-model evidence from transition samples that may already be contaminated by earlier post-stall divergence.

- train/attached: count `2272`, Cm mean `0.07217`, Cm MAE `0.07782`, Cm fit residual MAE `0.04758`, CY mean `-0.16914`, Cl mean `0.00908`, Cn mean `-0.00424`
- train/transition_before_post_stall: count `6214`, Cm mean `0.03312`, Cm MAE `0.04346`, Cm fit residual MAE `0.03380`, CY mean `-0.08186`, Cl mean `-0.00961`, Cn mean `0.00365`
- train/transition_after_post_stall: count `838`, Cm mean `0.03835`, Cm MAE `0.05787`, Cm fit residual MAE `0.03842`, CY mean `0.07426`, Cl mean `-0.00494`, Cn mean `0.00221`
- train/post_stall: count `891`, Cm mean `-0.00147`, Cm MAE `0.05516`, Cm fit residual MAE `0.03876`, CY mean `-0.07397`, Cl mean `-0.02329`, Cn mean `0.00823`
- heldout/attached: count `365`, Cm mean `0.07264`, Cm MAE `0.08056`, Cm fit residual MAE `0.05477`, CY mean `-0.25583`, Cl mean `0.01212`, Cn mean `-0.00405`
- heldout/transition_before_post_stall: count `1180`, Cm mean `0.03073`, Cm MAE `0.04333`, Cm fit residual MAE `0.03362`, CY mean `-0.07940`, Cl mean `-0.01181`, Cn mean `0.00278`
- heldout/transition_after_post_stall: count `183`, Cm mean `0.05292`, Cm MAE `0.06572`, Cm fit residual MAE `0.05824`, CY mean `0.06906`, Cl mean `0.00158`, Cn mean `0.01756`
- heldout/post_stall: count `251`, Cm mean `-0.06016`, Cm MAE `0.07134`, Cm fit residual MAE `0.03680`, CY mean `0.08834`, Cl mean `-0.03519`, Cn mean `0.00034`

## Interpretation

Accept the primary candidate only for the longitudinal claim-bearing model when held-out dx, altitude loss, sink, and pitch improve or preserve the active baseline. Treat dy, roll, and yaw as reported residual evidence unless the secondary lateral diagnostic improves held-out dy/roll/yaw without damaging those longitudinal metrics.
